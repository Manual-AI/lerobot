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

"""Training-time conversion of ``type: EEF`` relative action groups.

The decode direction has always understood EEF groups (it must, to run the pretrained
``*_relative_eef`` embodiments). These cover the forward direction, which previously raised.
"""

import numpy as np
import pytest
import torch

from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
from lerobot.policies.groot.utils import (
    absolute_eef_to_relative,
    homogeneous_to_xyz_rot6d,
    relative_eef_to_absolute,
)

EEF_DIM = 9
HAND_DIM = 3
HORIZON = 4


def _stats_entry(dim: int) -> dict[str, list[float]]:
    return {"min": [-1.0] * dim, "max": [1.0] * dim}


def _random_pose_xyz_rot6d(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = q
    transform[:3, 3] = rng.uniform(-1.0, 1.0, size=3)
    return homogeneous_to_xyz_rot6d(transform)


def _make_step(*, eef_format: str = "XYZ_ROT6D", hand_rep: str = "ABSOLUTE") -> GrootN17PackInputsStep:
    """A pack step whose layout is one EEF group followed by one non-EEF hand group."""
    raw_stats = {
        "state": {"arm_eef": _stats_entry(EEF_DIM), "hand": _stats_entry(HAND_DIM)},
        "action": {"arm_eef": _stats_entry(EEF_DIM), "hand": _stats_entry(HAND_DIM)},
        "relative_action": {"arm_eef": _stats_entry(EEF_DIM), "hand": _stats_entry(HAND_DIM)},
    }
    modality_config = {
        "state": {"modality_keys": ["arm_eef", "hand"]},
        "action": {
            "modality_keys": ["arm_eef", "hand"],
            "action_configs": [
                {
                    "rep": "RELATIVE",
                    "type": "EEF",
                    "format": eef_format,
                    "state_key": "arm_eef",
                },
                {"rep": hand_rep, "type": "NON_EEF", "format": "DEFAULT", "state_key": "hand"},
            ],
        },
    }
    return GrootN17PackInputsStep(
        raw_stats=raw_stats,
        modality_config=modality_config,
        normalize_min_max=False,
        training=True,
    )


def _make_batch(rng: np.random.Generator, batch: int = 2):
    state_eef = np.stack([_random_pose_xyz_rot6d(rng) for _ in range(batch)])
    state_hand = rng.uniform(-0.5, 0.5, size=(batch, HAND_DIM))
    state = torch.from_numpy(np.concatenate([state_eef, state_hand], axis=-1)).float()

    action_eef = np.stack([[_random_pose_xyz_rot6d(rng) for _ in range(HORIZON)] for _ in range(batch)])
    action_hand = rng.uniform(-0.5, 0.5, size=(batch, HORIZON, HAND_DIM))
    action = torch.from_numpy(np.concatenate([action_eef, action_hand], axis=-1)).float()
    return state, action


def test_eef_group_uses_se3_composition_not_subtraction():
    rng = np.random.default_rng(0)
    step = _make_step()
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)

    expected = absolute_eef_to_relative(action[..., :EEF_DIM].numpy(), state[:, :EEF_DIM].numpy())
    np.testing.assert_allclose(converted[..., :EEF_DIM].numpy(), expected, atol=1e-5)

    elementwise = (action[..., :EEF_DIM] - state[:, None, :EEF_DIM]).numpy()
    assert not np.allclose(converted[..., :EEF_DIM].numpy(), elementwise, atol=1e-3), (
        "EEF group was converted by elementwise subtraction"
    )


def test_absolute_non_eef_group_is_left_untouched():
    rng = np.random.default_rng(1)
    step = _make_step(hand_rep="ABSOLUTE")
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)

    np.testing.assert_allclose(converted[..., EEF_DIM:].numpy(), action[..., EEF_DIM:].numpy(), atol=1e-6)


def test_relative_non_eef_group_still_subtracts_elementwise():
    """The NON_EEF path must be unchanged — joint angles are a genuine vector space."""
    rng = np.random.default_rng(2)
    step = _make_step(hand_rep="RELATIVE")
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)

    expected = (action[..., EEF_DIM:] - state[:, None, EEF_DIM:]).numpy()
    np.testing.assert_allclose(converted[..., EEF_DIM:].numpy(), expected, atol=1e-6)


def test_forward_conversion_round_trips_through_the_decode_helper():
    """Training conversion and decode conversion are exact inverses."""
    rng = np.random.default_rng(3)
    step = _make_step()
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)
    recovered = relative_eef_to_absolute(converted[..., :EEF_DIM].numpy(), state[:, :EEF_DIM].numpy())

    np.testing.assert_allclose(recovered, action[..., :EEF_DIM].numpy(), atol=1e-5)


def test_eef_group_no_longer_raises():
    """Regression: this configuration previously raised 'Unsupported relative N1.7 action config'."""
    rng = np.random.default_rng(4)
    step = _make_step()
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)

    assert converted.shape == action.shape
    assert torch.isfinite(converted).all()


def test_unsupported_eef_format_still_raises():
    """Only xyz+rot6d is implemented; xyz+rotvec must fail loudly rather than fall through."""
    rng = np.random.default_rng(5)
    step = _make_step(eef_format="XYZ_ROTVEC")
    state, action = _make_batch(rng)

    with pytest.raises(ValueError, match="Unsupported relative N1.7 action config"):
        step._convert_relative_action_groups_for_training(action, state)


def test_missing_state_group_raises():
    rng = np.random.default_rng(6)
    step = _make_step()
    step.modality_config["action"]["action_configs"][0]["state_key"] = "nope"
    state, action = _make_batch(rng)

    with pytest.raises(KeyError, match="Missing raw state group 'nope'"):
        step._convert_relative_action_groups_for_training(action, state)


def test_input_action_is_not_mutated_in_place():
    rng = np.random.default_rng(7)
    step = _make_step()
    state, action = _make_batch(rng)
    original = action.clone()

    step._convert_relative_action_groups_for_training(action, state)

    np.testing.assert_allclose(action.numpy(), original.numpy(), atol=0)


def test_dtype_is_preserved():
    rng = np.random.default_rng(8)
    step = _make_step()
    state, action = _make_batch(rng)

    converted = step._convert_relative_action_groups_for_training(action, state)

    assert converted.dtype == action.dtype
