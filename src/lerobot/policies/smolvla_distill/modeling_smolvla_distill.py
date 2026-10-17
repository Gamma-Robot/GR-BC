# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)
from lerobot.policies.smolvla_distill.configuration_smolvla_distill import SmolVLADistillConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def flow_matching_sample_actions(
    flow: VLAFlowMatching,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state: Tensor,
    noise: Tensor,
    euler_steps: int,
    **kwargs: Unpack[ActionSelectKwargs],
) -> Tensor:
    """Run flow-matching sampling for `euler_steps` Euler steps (independent of `flow.config.num_steps`)."""
    if euler_steps < 1:
        raise ValueError(f"euler_steps must be >= 1, got {euler_steps}")
    if flow.config.rtc_config is not None and flow.config.rtc_config.enabled:
        raise NotImplementedError("RTC sampling is not supported in smolvla_distill distillation path.")

    bsize = state.shape[0]
    device = state.device

    prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, past_key_values = flow.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=flow.config.use_cache,
        fill_kv_cache=True,
    )
    dt = -1.0 / euler_steps
    x_t = noise
    for step in range(euler_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return flow.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
            )

        v_t = denoise_step_partial_call(x_t)
        x_t = x_t + dt * v_t

    return x_t


class SmolVLADistillPolicy(SmolVLAPolicy):
    """SmolVLA student trained with flow matching plus MSE to a frozen multi-step teacher."""

    config_class = SmolVLADistillConfig
    name = "smolvla_distill"

    def __init__(self, config: SmolVLADistillConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.teacher: VLAFlowMatching = copy.deepcopy(self.model)
        self._load_teacher_weights()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def _load_teacher_weights(self) -> None:
        path = self.config.teacher_pretrained_path
        if path is None or str(path).strip() == "":
            logging.info(
                "smolvla_distill: `teacher_pretrained_path` not set; teacher is an exact copy of the "
                "initialized student weights."
            )
            return
        model_file = Path(os.path.expanduser(str(path))) / SAFETENSORS_SINGLE_FILE
        if not model_file.is_file():
            logging.warning(
                "smolvla_distill: teacher weights not found at %s. "
                "Expected a LeRobot policy directory containing %s. "
                "Falling back to initialized student weights as teacher for this run.",
                model_file,
                SAFETENSORS_SINGLE_FILE,
            )
            return
        state = load_file(str(model_file))
        prefix = "model."
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        incompatible = self.teacher.load_state_dict(sub, strict=False)
        if incompatible.missing_keys:
            logging.warning(
                "smolvla_distill teacher load_state_dict missing_keys (showing up to 8): %s",
                incompatible.missing_keys[:8],
            )
        if incompatible.unexpected_keys:
            logging.warning(
                "smolvla_distill teacher load_state_dict unexpected_keys (showing up to 8): %s",
                incompatible.unexpected_keys[:8],
            )

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, float]]:
        if self._rtc_enabled():
            raise NotImplementedError("smolvla_distill does not support RTC.")

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict: dict[str, float] = {}

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]

        fm_loss = losses.mean()

        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        shared_noise = self.model.sample_noise(actions_shape, device)

        with torch.no_grad():
            teacher_actions = flow_matching_sample_actions(
                self.teacher,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                shared_noise,
                self.config.teacher_num_steps,
            )

        student_actions = flow_matching_sample_actions(
            self.model,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            shared_noise,
            self.config.student_num_steps,
        )

        original_action_dim = self.config.action_feature.shape[0]
        teacher_actions = teacher_actions[:, :, :original_action_dim]
        student_actions = student_actions[:, :, :original_action_dim]
        distill_loss = F.mse_loss(student_actions, teacher_actions)

        action_mse_teacher_vs_student = distill_loss.detach().item()
        loss_dict["fm_loss"] = fm_loss.detach().item()
        loss_dict["distill_loss"] = distill_loss.detach().item()
        loss_dict["action_mse_teacher_vs_student"] = action_mse_teacher_vs_student

        if reduction == "none":
            per_sample_fm = losses.mean(dim=(1, 2))
            distill_share = (self.config.distill_loss_weight * distill_loss) / max(per_sample_fm.shape[0], 1)
            per_sample_total = self.config.fm_loss_weight * per_sample_fm + distill_share
            loss_dict["loss"] = per_sample_total.mean().item()
            return per_sample_total, loss_dict

        loss = self.config.fm_loss_weight * fm_loss + self.config.distill_loss_weight * distill_loss
        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict
