#!/usr/bin/env python

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

"""The relative-action stats window must start at ``action_delta_offset``.

Both the training normalizer and the decode step index the per-chunk-timestep relative stats by
chunk step (``min_t[:horizon]``), so stats row k is applied to chunk step k. Chunk step k holds
delta index ``action_delta_offset + k``. If the stats window were hardcoded to start at 0, row k
would be paired with a distribution one offset away from step k's actual content -- a silent
horizon misalignment baked into the saved checkpoint.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

from lerobot.policies.groot.processor_groot import (
    N1_7_NATIVE_ACTION_HORIZON,
    _make_relative_action_training_stats_from_dataset_meta,
)


def _capture_delta_timestamps(offset: int) -> list[float]:
    """Run the stats builder far enough to capture the delta_timestamps it asks the dataset for."""
    captured: dict = {}

    class _FakeDataset:
        def __init__(self, repo_id, **kwargs):
            captured["delta_timestamps"] = kwargs["delta_timestamps"]
            raise RuntimeError("stop after capture")

    config = SimpleNamespace(
        action_delta_offset=offset,
        relative_exclude_joints=[],
        relative_eef_groups=[],
    )
    dataset_meta = SimpleNamespace(repo_id="org/ds", root="/tmp/ds", fps=30, revision=None)

    # The fake dataset raises once it has captured the window; nothing past that matters here.
    with (
        patch("lerobot.policies.groot.processor_groot.LeRobotDataset", _FakeDataset),
        patch("lerobot.policies.groot.processor_groot.require_package", lambda *a, **k: None),
        contextlib.suppress(RuntimeError),
    ):
        _make_relative_action_training_stats_from_dataset_meta(config, dataset_meta)

    from lerobot.utils.constants import ACTION

    return captured["delta_timestamps"][ACTION]


def test_default_offset_window_starts_at_zero():
    """Backward compatible: offset 0 keeps the original range(HORIZON) window."""
    stamps = _capture_delta_timestamps(0)

    assert len(stamps) == N1_7_NATIVE_ACTION_HORIZON
    assert stamps[0] == 0.0
    assert stamps == [i / 30 for i in range(N1_7_NATIVE_ACTION_HORIZON)]


def test_offset_shifts_the_stats_window_to_match_training():
    """offset=1 must sample delta indices 1..HORIZON, so stats row k == chunk step k."""
    stamps = _capture_delta_timestamps(1)

    assert len(stamps) == N1_7_NATIVE_ACTION_HORIZON
    assert stamps[0] == 1 / 30, "row 0 must correspond to delta index 1, not 0"
    assert stamps == [(1 + i) / 30 for i in range(N1_7_NATIVE_ACTION_HORIZON)]


def test_offset_window_never_includes_the_degenerate_zero_delta():
    """With offset>=1 no stats row describes the identity delta.

    At offset 0 row 0 is delta index 0 -- the action at the anchor's own frame, which for a
    direct-recording dataset (action[t] == state[t]) is identically zero, giving that row a zero
    range and forcing chunk step 0's normalized target to exactly 0.
    """
    assert 0.0 not in _capture_delta_timestamps(1)
    assert 0.0 not in _capture_delta_timestamps(3)


def test_window_length_is_independent_of_offset():
    for offset in (0, 1, 5):
        assert len(_capture_delta_timestamps(offset)) == N1_7_NATIVE_ACTION_HORIZON
