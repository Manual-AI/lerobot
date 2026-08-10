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

"""Tests for the RTC chunk-boundary continuity tracker."""

import torch

from lerobot.policies.rtc.seam_tracker import ChunkSeamTracker


def test_records_max_abs_and_l2_gap():
    tracker = ChunkSeamTracker()

    gap = tracker.record(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.0, -1.0]))

    assert gap == 4.0
    summary = tracker.summary()
    assert summary["boundaries"] == 1.0
    assert summary["max_abs_mean"] == 4.0
    assert summary["l2_mean"] == torch.tensor([0.5, 0.0, -4.0]).norm().item()


def test_first_chunk_has_no_seam():
    tracker = ChunkSeamTracker()

    assert tracker.record(None, torch.zeros(3)) is None
    assert len(tracker) == 0
    assert tracker.summary()["boundaries"] == 0.0


def test_mismatched_action_dims_are_skipped():
    tracker = ChunkSeamTracker()

    assert tracker.record(torch.zeros(3), torch.zeros(4)) is None
    assert len(tracker) == 0


def test_summary_aggregates_across_boundaries():
    tracker = ChunkSeamTracker()

    for gap in (1.0, 2.0, 10.0):
        tracker.record(torch.zeros(2), torch.tensor([gap, 0.0]))

    summary = tracker.summary()
    assert summary["boundaries"] == 3.0
    assert summary["max_abs_mean"] == (1.0 + 2.0 + 10.0) / 3
    assert summary["max_abs_max"] == 10.0
    assert summary["max_abs_p95"] <= 10.0


def test_reset_clears_history():
    tracker = ChunkSeamTracker()
    tracker.record(torch.zeros(2), torch.ones(2))

    tracker.reset()

    assert len(tracker) == 0
    assert tracker.summary() == {
        "boundaries": 0.0,
        "max_abs_mean": 0.0,
        "max_abs_p95": 0.0,
        "max_abs_max": 0.0,
        "l2_mean": 0.0,
    }


def test_window_bounds_memory_but_keeps_boundary_count():
    tracker = ChunkSeamTracker(maxlen=2)

    for gap in (1.0, 2.0, 3.0):
        tracker.record(torch.zeros(2), torch.tensor([gap, 0.0]))

    assert len(tracker) == 2
    assert tracker.summary()["boundaries"] == 3.0
    assert tracker.summary()["max_abs_mean"] == 2.5
