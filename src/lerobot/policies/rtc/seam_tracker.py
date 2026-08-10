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

"""Chunk-boundary continuity measurement for Real-Time Chunking (RTC)."""

from collections import deque

import numpy as np
from torch import Tensor


class ChunkSeamTracker:
    """Measures commanded-action discontinuity at RTC chunk boundaries.

    Every time the RTC engine swaps a freshly predicted chunk into the action
    queue, the robot's next commanded action jumps from "what the old chunk said
    to do now" to "what the new chunk says to do now". That jump is the seam. It
    is the quantity RTC prefix guidance is supposed to shrink, and the only way
    to tell real seam conditioning from a rollout that merely *looks* smoother.

    Both actions are recorded in postprocessed (absolute, environment) units, so
    the reported gap is directly comparable across runs with and without prefix
    guidance -- as long as the same policy, task and fps are used.

    Args:
        maxlen (int): Sliding window of boundaries kept for percentile queries.
    """

    def __init__(self, maxlen: int = 1000):
        self._max_abs: deque[float] = deque(maxlen=maxlen)
        self._l2: deque[float] = deque(maxlen=maxlen)
        self._boundaries = 0

    def reset(self) -> None:
        """Clear all recorded boundaries."""
        self._max_abs.clear()
        self._l2.clear()
        self._boundaries = 0

    def record(self, previous_action: Tensor | None, next_action: Tensor | None) -> float | None:
        """Record one chunk boundary and return its max-abs discontinuity.

        Args:
            previous_action: The action the *old* chunk would have commanded next,
                i.e. row 0 of its unexecuted tail at swap time. ``None`` on the
                first chunk, which has no seam.
            next_action: The first action the *new* chunk actually commands, i.e.
                the row at the resolved inference delay.

        Returns:
            The max-abs gap between the two, or ``None`` when there is no seam to
            measure (first chunk, or mismatched action dimensions).
        """
        if previous_action is None or next_action is None:
            return None
        previous = previous_action.detach().float().reshape(-1).cpu()
        upcoming = next_action.detach().float().reshape(-1).cpu()
        if previous.numel() == 0 or previous.numel() != upcoming.numel():
            return None

        delta = (upcoming - previous).numpy()
        max_abs = float(np.abs(delta).max())
        self._max_abs.append(max_abs)
        self._l2.append(float(np.linalg.norm(delta)))
        self._boundaries += 1
        return max_abs

    def __len__(self) -> int:
        return len(self._max_abs)

    def summary(self) -> dict[str, float]:
        """Return aggregate seam statistics over the sliding window.

        ``boundaries`` counts every recorded seam; the remaining keys are 0.0 when
        nothing has been recorded yet.
        """
        if not self._max_abs:
            return {
                "boundaries": float(self._boundaries),
                "max_abs_mean": 0.0,
                "max_abs_p95": 0.0,
                "max_abs_max": 0.0,
                "l2_mean": 0.0,
            }
        max_abs = np.asarray(self._max_abs, dtype=np.float64)
        l2 = np.asarray(self._l2, dtype=np.float64)
        return {
            "boundaries": float(self._boundaries),
            "max_abs_mean": float(max_abs.mean()),
            "max_abs_p95": float(np.quantile(max_abs, 0.95)),
            "max_abs_max": float(max_abs.max()),
            "l2_mean": float(l2.mean()),
        }
