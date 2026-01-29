# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# import rlinf.algorithms.advantages_sac  # noqa: F401
from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.hybrid_engines.fsdp import (
    FSDP,
    FSDPModule,
)
from rlinf.scheduler import Channel
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_rollout_metrics,
)
from rlinf.utils.nested_dict_process import (
    concat_batch,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedFQLFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # offpolicy-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.base_alpha = None
        self.demo_buffer = None  # human demo, no use now
        self.alpha_optimizer = None
        self.update_step = 0

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_fql_components()
        # save memeory, algorithm-neutral.
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()


    def setup_model_and_optimizer(self, initialize_target=False):
        """Setup model, lr_scheduler, optimizer and grad_scaler.
        Not initializing target model here.
        """
        # get model and load checkpoint(if exist)
        module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.model.get("gradient_checkpointing", False):
            self._logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
        else:
            self._logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh,
        )

        self.build_optimizer(
            model=self.model, enable_critic_warmup=self.critic_warmup_steps > 0
        )

        self.build_lr_scheduler()

        self.grad_scaler = self.build_grad_scaler(
            self._cfg.fsdp_config.amp.use_grad_scaler
        )

    def build_optimizer(
        self,
        model: Union[nn.Module, FSDPModule, FSDP],
        enable_critic_warmup: bool = False,
    ):
        """Build FQL optimizer, abort entropy tuning in SACPolicy.
        Add support for VLM, based on fsdp_model_manager.
        Only support Adam optmizer.
        """
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        params_actor = []
        params_critic = []

        if enable_critic_warmup:
            raise NotImplementedError
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    #TODO: check target_q_head
                    if "q_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)
        assert len(params_critic) > 0
        # actor opt & critic opt.
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": params_actor,
                    "lr": self._cfg.optim.lr,
                    "betas": betas,
                }
            ]
        )
        self.qf_optimizer = torch.optim.Adam(
            [
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            ]
        )

    def build_lr_scheduler(self):
        lr_scheduler_type = self._cfg.optim.get("lr_scheduler_type", "constant")
        if lr_scheduler_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1
            )
            self.qf_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.qf_optimizer, factor=1
            )
        elif lr_scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps, eta_min=1e-6
            )
            self.qf_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.qf_optimizer, T_max=self.max_steps, eta_min=1e-6
            )
        else:
            raise NotImplementedError

    def setup_fql_components(self):
        """Initialize FQL-specific components
        """
        seed = self.cfg.actor.get("seed", 1234)
        self.replay_buffer = SACReplayBuffer(
            capacity=self.cfg.algorithm.replay_buffer_capacity,
            device=self.device,
            seed=seed,
        )

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

    # Rollout Setup
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def recv_rollout_batch(self, input_channel: Channel):
        super().recv_rollout_batch(input_channel)
        self.replay_buffer.add_rollout_batch(self.rollout_batch)

    async def recv_demo_data(self, input_channel: Channel):
        demo_data = await input_channel.get(async_op=True).async_wait()
        self.demo_buffer = SACReplayBuffer.create_from_buffer(
            demo_data, seed=self.cfg.actor.seed
        )
    # Forward function
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        assert not use_crossq, "Currently not support crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        rewards = batch["rewards"].to(self.torch_dtype)
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["transitions"]["obs"]
        next_obs = batch["transitions"]["next_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            #TODO
            next_state_actions = self.model("fql_onestep_forward", obs=next_obs, **kwargs)
            if not use_crossq:
                all_qf_next_target = self.model(
                    "fql_target_q_forward",
                    obs=next_obs,
                    actions=next_state_actions,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx,
                    )

                # agg_q multi q_head
                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=-1, keepdim=True,
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(
                        all_qf_next_target, dim=-1, keepdim=True,
                    )

                # bootstrap type
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards.sum(dim=-1, keepdim=True)
                        + self.cfg.algorithm.gamma * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards.sum(dim=-1, keepdim=True).unsqueeze(-1)
                        + (~(terminations.any(dim=-1, keepdim=True))).unsqueeze(-1)
                        * self.cfg.algorithm.gamma
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                "fql_q_forward",
                obs=curr_obs,
                actions=batch["action"]
                if "action" in batch
                else batch["action_tokens"],
            )
        else:
            raise NotImplementedError(f"crossq is not supported currently.")

        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )

        critic_info = {
            "critic_loss": critic_loss,
            "q_mean": all_data_q_values.mean(),
            "q_min": all_data_q_values.min(),
            "q_max": all_data_q_values.max(),
        }
        return critic_loss, critic_info


    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        curr_obs = batch["transitions"]["obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
             kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train

        # BC flow loss
        pi0_input = {
            "observation": curr_obs,
            "actions": batch["action"],
        }
        bc_flow_loss = self.model(
            "sft_forward",
            data=pi0_input,
        )

        # Distillation loss
        teacher_flow_actions = self.model(
            "fql_teacher_forward",
            obs=curr_obs,
        )
        student_onestep_actions = self.model(
            "fql_onestep_forward",
            obs=curr_obs,
        )
        distill_loss = F.mse_loss(
            teacher_flow_actions,
            student_onestep_actions,
            reduction="mean",
        )

        # Q loss
        qs = self.model(
            "fql_q_forward",
            obs=curr_obs,
            actions=student_onestep_actions,
        )
        q = torch.mean(
            qs, dim=1, keepdim=True
        )
        q_loss = -q.mean()
        # optional: normalize Q loss scale
        if self.cfg.algorithm.get("normalize_q_loss", False):
            lam = (1.0 / q.abs().mean()).detach()
            q_loss = lam * q_loss

        fql_alpha = self.cfg.algorithm.get("fql_alpha", 10.0)
        actor_loss = bc_flow_loss + fql_alpha * distill_loss + q_loss

        mse = F.mse_loss(student_onestep_actions, batch["action"])
        actor_info = {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q": q.mean(),
            "mse": mse,
        }
        return actor_loss, actor_info

    def update_one_epoch(self, train_actor):
        # 1. sample a global_batch from (2) buffer
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        if self.demo_buffer is not None:
            replay_batch = self.replay_buffer.sample(global_batch_size_per_rank // 2)
            demo_batch = self.demo_buffer.sample(global_batch_size_per_rank // 2)
            global_batch = concat_batch(replay_batch, demo_batch)
        else:
            # Sample batch from replay buffer
            global_batch = self.replay_buffer.sample(global_batch_size_per_rank)

        # 2. cut micro-batch for grad
        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # 3. update critic
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        for batch in train_micro_batch_list:
            batch = put_tensor_device(batch, device=self.device)
            critic_loss, critic_info = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "fql/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
        }
        for k, v in critic_info.items():
            metrics_data[f"critic/{k}"] = v

        # 4. update actor
        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            for batch in train_micro_batch_list:
                batch = put_tensor_device(batch, device=self.device)
                actor_loss, actor_info = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "fql/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                }
            )
            for k, v in actor_info.items():
                metrics_data[f"actor/{k}"] = v

        # 5. soft update target network
        if self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
            tau = self.cfg.algorithm.get("tau", 0.005)
            self.model.soft_update_target_q_head(tau)

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = (
            self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
        )
        train_actor_steps = (
            self.cfg.algorithm.get("train_actor_steps", 0) // self._world_size
        )
        train_actor_steps = max(min_buffer_size, train_actor_steps)

        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        # Just compute basic rollout metrics without advantages/returns
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        buffer_path = os.path.join(
            self.cfg.runner.logger.log_path, f"replay_buffer_{self._rank}.pkl"
        )
        self.replay_buffer.save(buffer_path)

