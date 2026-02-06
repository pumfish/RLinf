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

import math
import copy
import random
import pytest
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openpi.models.gemma as _gemma
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import (
        PI0Pytorch,
        sample_beta,
        make_att_2d_masks,
        create_sinusoidal_pos_embedding,
)

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.modules.q_head import QHead, MultiQHead


class OneStepStudent(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pi05 = self.config.pi05

        # Get config.
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        use_adarms = [False, True] if self.pi05 else [False, False]
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )
        precision=config.dtype

        # Get module
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.to(dtype=torch.bfloat16)
        # self.to_bfloat16_for_selected_params(precision)

        torch.set_float32_matmul_precision("high")

        # Initialize gradient checkpoint flag
        self.gradient_checkpointing_enabled = False

    # gemma_pytorch
    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for OneStepStudent.gemma_expert.")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for OneStepStudent.gemma_expert")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # Sample gaussian noise
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def action_expert_forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        assert inputs_embeds[0] is None, "action expert only use suffix_embs"
        if adarms_cond is None:
            adarms_cond = [None, None]
        suffix_output = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds[1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
        )
        suffix_output = suffix_output.last_hidden_state
        prefix_output = None
        prefix_past_key_values = None
        return [prefix_output, suffix_output], prefix_past_key_values

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise 'x_t' at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = \
                self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.action_expert_forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def forward(self, observation, inputs, noise=None):
        """Onestep policy forward just output action
        observation: _model.Observation
        """
        device = observation.state.device
        bsize = observation.state.shape[0]

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        noise = noise.requires_grad_(True)

        # We get prefix out from externel.
        # state, prefix_pad_masks, past_key_values = get_prefix_out_from_pi0(observation)
        (state, prefix_pad_masks, past_key_values) = inputs

        num_steps = 1
        dt_src = -1.0 / num_steps
        dt = torch.tensor(dt_src, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time
            )
            x_t = x_t + dt * v_t
            time += dt

        x_0 = x_t
        return x_0

    def load_pi0_weight_to_onestep(self, state_dict):
        """Load params from PI0 model.safetensors
        """
        print("[WARN] Loading teacher weights into OneStepStudent (init only!)")
        with torch.no_grad():
            # action_proj
            for name in ["action_in_proj", "action_out_proj"]:
                layer = getattr(self, name)
                layer.weight.data.copy_(
                    state_dict[f"{name}.weight"].to(
                        device=layer.weight.device,
                        dtype=layer.weight.dtype,
                    )
                )
                layer.bias.data.copy_(
                    state_dict[f"{name}.bias"].to(
                        device=layer.bias.device,
                        dtype=layer.bias.dtype,
                    )
                )

            # time mlp / action_time mlp
            if self.pi05:
                names = ["time_mlp_in", "time_mlp_out"]
            else:
                names = ["action_time_mlp_in", "action_time_mlp_out"]

            for name in names:
                layer = getattr(self, name)
                layer.weight.data.copy_(
                    state_dict[f"{name}.weight"].to(
                        device=layer.weight.device,
                        dtype=layer.weight.dtype,
                    )
                )
                layer.bias.data.copy_(
                    state_dict[f"{name}.bias"].to(
                        device=layer.bias.device,
                        dtype=layer.bias.dtype,
                    )
                )

            # Gemma Expert
            gemma_prefix = "paligemma_with_expert.gemma_expert."
            for k, v in state_dict.items():
                if not k.startswith(gemma_prefix):
                    continue

                new_k = k.replace(gemma_prefix, "gemma_expert.")
                pointer = self.gemma_expert
                path = new_k.split(".")[1:]  # drop "gemma_expert"

                for p in path[:-1]:
                    pointer = pointer[int(p)] if p.isdigit() else getattr(pointer, p)

                if not hasattr(pointer, path[-1]):
                    continue

                param = getattr(pointer, path[-1])
                param.data.copy_(
                    v.to(device=param.device, dtype=param.dtype)
                )

        print("OneStepStudent load PI0 weights successfully!")

