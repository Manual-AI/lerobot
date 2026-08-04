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

"""``GrootConfig.action_delta_offset``: shifting the sampled action window."""

import pytest

from lerobot.policies.groot.configuration_groot import GrootConfig


def _config(**kwargs) -> GrootConfig:
    return GrootConfig(device="cpu", use_bf16=False, **kwargs)


def test_default_offset_is_backward_compatible():
    config = _config(chunk_size=16, n_action_steps=16)

    assert config.action_delta_indices == list(range(16))
    assert config.drop_n_last_frames == 15


def test_offset_shifts_the_window_without_changing_its_length():
    config = _config(chunk_size=16, n_action_steps=16, action_delta_offset=1)

    assert config.action_delta_indices == list(range(1, 17))
    assert len(config.action_delta_indices) == 16


def test_drop_n_last_frames_follows_the_largest_index():
    """Otherwise the tail frames get silently zero-padded instead of excluded."""
    config = _config(chunk_size=16, n_action_steps=16, action_delta_offset=1)

    assert config.drop_n_last_frames == 16
    assert config.drop_n_last_frames == max(config.action_delta_indices)


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_offset_never_shortens_the_chunk(offset):
    config = _config(chunk_size=8, n_action_steps=8, action_delta_offset=offset)

    assert len(config.action_delta_indices) == 8
    assert config.action_delta_indices[0] == offset


def test_negative_offset_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        _config(chunk_size=16, n_action_steps=16, action_delta_offset=-1)


def test_observation_delta_indices_are_unaffected():
    """The observation stays at t; only the action window moves."""
    config = _config(chunk_size=16, n_action_steps=16, action_delta_offset=1)

    assert config.observation_delta_indices is None
