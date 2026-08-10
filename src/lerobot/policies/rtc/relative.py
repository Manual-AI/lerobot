#!/usr/bin/env python

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

"""Relative-action helpers for Real-Time Chunking (RTC)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)


@runtime_checkable
class RelativeRTCPrefixEncoder(Protocol):
    """A preprocessor step that owns its policy's relative-action RTC prefix encoding.

    The generic :func:`reanchor_relative_rtc_prefix` path below assumes relative
    actions are a plain per-dimension ``action - state`` offset normalized with a
    single stats row. Policies whose relative actions do not fit that shape --
    e.g. GR00T N1.7, which composes SE(3) deltas for end-effector groups and
    normalizes each chunk step with its own stats row -- implement this protocol
    instead, so the RTC engine can hand them the absolute leftovers and let the
    step produce a model-space prefix.
    """

    def uses_relative_rtc_prefix(self) -> bool:
        """Whether this step must encode the RTC prefix itself.

        When ``True`` the RTC engine stops deriving the prefix from the queue's
        model-space leftovers and calls :meth:`encode_absolute_rtc_prefix`
        instead; a ``None`` result then means "run this chunk without prefix
        guidance" rather than "fall back to the raw leftovers".
        """
        ...

    def encode_absolute_rtc_prefix(self, prev_actions_absolute: torch.Tensor) -> torch.Tensor | None:
        """Re-anchor and normalize absolute leftover actions into model space.

        Args:
            prev_actions_absolute: ``(T, A)`` or ``(B, T, A)`` unexecuted tail of
                the previous chunk in absolute environment coordinates, where row
                ``i`` is the action the robot would execute ``i`` steps from now
                (i.e. aligned with step ``i`` of the chunk about to be predicted).

        Returns:
            The prefix in the policy's model action space, or ``None`` when it
            cannot be built for this step.
        """
        ...


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    """Convert absolute leftover actions into model-space for relative-action RTC policies.

    When using relative actions, the RTC prefix (previous chunk's unexecuted tail)
    is stored in absolute coordinates. Before feeding it back to the policy, this
    helper re-expresses those actions relative to the robot's current joint state
    and optionally normalizes them so the policy receives correctly scaled inputs.
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)
