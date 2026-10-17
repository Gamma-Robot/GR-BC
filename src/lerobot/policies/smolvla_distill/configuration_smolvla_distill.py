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

from dataclasses import dataclass

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@SmolVLAConfig.register_subclass("smolvla_distill")
@dataclass
class SmolVLADistillConfig(SmolVLAConfig):
    """SmolVLA with distillation from multi-step denoising (teacher) to few-step / one-step (student)."""

    # Student uses this many Euler steps at inference (1 = single-step denoising).
    num_steps: int = 1

    # Teacher uses this many denoising steps (typical SFT SmolVLA uses 10).
    teacher_num_steps: int = 10

    # Directory with `model.safetensors` from the frozen multi-step teacher (e.g. SFT checkpoint).
    # If None, the teacher is a deep copy of the student right after weight loading (same init as student).
    teacher_pretrained_path: str | None = None

    fm_loss_weight: float = 1.0
    distill_loss_weight: float = 1.0

    @property
    def student_num_steps(self) -> int:
        return self.num_steps
