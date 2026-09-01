# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Chunk-anchored relative EEF actions for flat xyz+rot6d action blocks.

Each pose in a predicted chunk is expressed against the observation pose from
which the chunk was predicted: ``delta_k = inverse(T_state) @ T_k``. Scalar
action dimensions are represented by an elementwise offset from state.

The rot6d convention is the first two rows of the rotation matrix, flattened.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry


def rot6d_to_matrix(r6: Tensor) -> Tensor:
    """Convert ``(..., 6)`` row-based rot6d values to rotation matrices."""
    rows = r6.reshape(*r6.shape[:-1], 2, 3)
    row_1 = F.normalize(rows[..., 0, :], dim=-1)
    row_2 = rows[..., 1, :] - (row_1 * rows[..., 1, :]).sum(dim=-1, keepdim=True) * row_1
    row_2 = F.normalize(row_2, dim=-1)
    row_3 = torch.linalg.cross(row_1, row_2)
    return torch.stack([row_1, row_2, row_3], dim=-2)


def pose9_to_hom(p9: Tensor) -> Tensor:
    """Convert ``(..., xyz+rot6d)`` poses to homogeneous transforms."""
    hom = torch.zeros(*p9.shape[:-1], 4, 4, dtype=p9.dtype, device=p9.device)
    hom[..., :3, :3] = rot6d_to_matrix(p9[..., 3:9])
    hom[..., :3, 3] = p9[..., :3]
    hom[..., 3, 3] = 1
    return hom


def hom_to_pose9(hom: Tensor) -> Tensor:
    """Convert homogeneous transforms to row-based ``xyz+rot6d`` poses."""
    return torch.cat([hom[..., :3, 3], hom[..., 0, :3], hom[..., 1, :3]], dim=-1)


def hom_inverse(hom: Tensor) -> Tensor:
    """Invert rigid homogeneous transforms using ``R^-1 = R^T``."""
    rotation_t = hom[..., :3, :3].transpose(-1, -2)
    inverse = torch.zeros_like(hom)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = (-rotation_t @ hom[..., :3, 3:4]).squeeze(-1)
    inverse[..., 3, 3] = 1
    return inverse


def _validate_inputs(actions: Tensor, state: Tensor, eef_starts: Sequence[int]) -> None:
    if actions.ndim not in (2, 3):
        raise ValueError(f"actions must have shape (B, D) or (B, T, D), got {tuple(actions.shape)}.")
    if state.ndim != 2:
        raise ValueError(f"state must have shape (B, Ds), got {tuple(state.shape)}.")
    if actions.shape[0] != state.shape[0]:
        raise ValueError(
            f"actions and state batch sizes must match, got {actions.shape[0]} and {state.shape[0]}."
        )

    action_dim = actions.shape[-1]
    if state.shape[-1] < action_dim:
        raise ValueError(
            "anchored relative actions require observation.state "
            f"({state.shape[-1]} dims) to carry the action layout as a prefix ({action_dim} dims)."
        )
    for start in eef_starts:
        if start < 0 or start + 9 > action_dim:
            raise ValueError(
                f"eef_start {start} does not fit a 9-dim xyz+rot6d block in {action_dim} action dims."
            )


def anchored_delta(actions: Tensor, state: Tensor, eef_starts: Sequence[int]) -> Tensor:
    """Convert absolute actions to chunk-anchored relative actions.

    ``actions`` has shape ``(B, T, D)`` or ``(B, D)`` and ``state`` has shape
    ``(B, Ds)``. Pose blocks use SE(3) composition while all other dimensions
    use elementwise subtraction. SE(3) calculations use float64 internally.
    """
    _validate_inputs(actions, state, eef_starts)
    squeeze_time = actions.ndim == 2
    actions_with_time = actions.unsqueeze(1) if squeeze_time else actions
    state = state.to(device=actions.device, dtype=actions.dtype)
    action_dim = actions.shape[-1]

    result = actions_with_time - state[:, None, :action_dim]
    for start in eef_starts:
        anchor_inverse = hom_inverse(pose9_to_hom(state[:, start : start + 9].double()))
        absolute = pose9_to_hom(actions_with_time[:, :, start : start + 9].double())
        result[:, :, start : start + 9] = hom_to_pose9(anchor_inverse[:, None] @ absolute).to(result.dtype)
    return result.squeeze(1) if squeeze_time else result


def anchored_compose(actions: Tensor, state: Tensor, eef_starts: Sequence[int]) -> Tensor:
    """Compose chunk-anchored relative actions onto an absolute state anchor."""
    _validate_inputs(actions, state, eef_starts)
    squeeze_time = actions.ndim == 2
    actions_with_time = actions.unsqueeze(1) if squeeze_time else actions
    state = state.to(device=actions.device, dtype=actions.dtype)
    action_dim = actions.shape[-1]

    result = actions_with_time + state[:, None, :action_dim]
    for start in eef_starts:
        anchor = pose9_to_hom(state[:, start : start + 9].double())
        relative = pose9_to_hom(actions_with_time[:, :, start : start + 9].double())
        result[:, :, start : start + 9] = hom_to_pose9(anchor[:, None] @ relative).to(result.dtype)
    return result.squeeze(1) if squeeze_time else result


@ProcessorStepRegistry.register("anchored_relative_eef_processor")
@dataclass
class AnchoredRelativeEEFStep(ProcessorStep):
    """Convert absolute chunks to anchored deltas and cache state for decoding."""

    enabled: bool = False
    eef_starts: list[int] = field(default_factory=list)
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return transition

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = anchored_delta(action, state, self.eef_starts)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "eef_starts": list(self.eef_starts)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("anchored_absolute_eef_processor")
@dataclass
class AnchoredAbsoluteEEFStep(ProcessorStep):
    """Decode anchored deltas, freezing the observation anchor for each ACT chunk."""

    enabled: bool = False
    eef_starts: list[int] = field(default_factory=list)
    n_action_steps: int = 1
    relative_step: AnchoredRelativeEEFStep | None = field(default=None, repr=False)
    _anchor: Tensor | None = field(default=None, init=False, repr=False)
    _ticks: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be at least 1, got {self.n_action_steps}.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if self.relative_step is None or self.relative_step._last_state is None:
            raise RuntimeError(
                "AnchoredAbsoluteEEFStep has no cached observation.state to compose onto. "
                "Ensure the paired AnchoredRelativeEEFStep ran before this step."
            )

        if action.ndim == 3:
            anchor = self.relative_step._last_state
        elif action.ndim == 2:
            if self._ticks % self.n_action_steps == 0:
                self._anchor = self.relative_step._last_state.clone()
            self._ticks += 1
            anchor = self._anchor
        else:
            raise ValueError(f"actions must have shape (B, D) or (B, T, D), got {tuple(action.shape)}.")

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = anchored_compose(action, anchor, self.eef_starts)
        return new_transition

    def reset(self) -> None:
        self._anchor = None
        self._ticks = 0

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "eef_starts": list(self.eef_starts),
            "n_action_steps": self.n_action_steps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
