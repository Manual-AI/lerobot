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

"""Declaring end-effector pose groups via ``relative_eef_groups``."""

import numpy as np
import pytest
import torch

from lerobot.policies.groot.processor_groot import (
    _convert_action_batch_to_relative,
    _GrootN17ActionGroup,
    _infer_n1_7_action_groups,
)
from lerobot.policies.groot.utils import absolute_eef_to_relative, homogeneous_to_xyz_rot6d


def _pose_names(side: str) -> list[str]:
    return [f"{side}_ee_pos_{a}" for a in "xyz"] + [f"{side}_ee_rot_{i}" for i in range(6)]


def _finger_names(side: str, count: int = 4) -> list[str]:
    return [f"{side}_finger_{i}.pos" for i in range(count)]


BIMANUAL_NAMES = _pose_names("right") + _finger_names("right") + _pose_names("left") + _finger_names("left")


def test_no_eef_tokens_preserves_previous_grouping():
    """Empty default must behave exactly as before: one relative run, no EEF groups."""
    groups = _infer_n1_7_action_groups(BIMANUAL_NAMES, action_dim=len(BIMANUAL_NAMES), exclude_joints=[])

    assert [g.key for g in groups] == ["single_arm"]
    assert groups[0].indices == list(range(len(BIMANUAL_NAMES)))
    assert groups[0].relative is True
    assert groups[0].eef is False


def test_declared_eef_groups_split_out_in_layout_order():
    groups = _infer_n1_7_action_groups(
        BIMANUAL_NAMES,
        action_dim=len(BIMANUAL_NAMES),
        exclude_joints=[],
        eef_groups=["right_ee", "left_ee"],
    )

    assert [(g.key, g.eef, g.relative) for g in groups] == [
        ("right_ee", True, True),
        ("single_arm", False, True),
        ("left_ee", True, True),
        ("single_arm_3", False, True),
    ]
    # Layout order matters: every consumer walks groups with a running offset.
    flat = [i for g in groups for i in g.indices]
    assert flat == list(range(len(BIMANUAL_NAMES)))
    assert groups[0].indices == list(range(9))
    assert groups[2].indices == list(range(13, 22))


def test_eef_group_must_be_nine_dimensions():
    """A token matching the wrong number of dims is a config error, not a silent reshape."""
    with pytest.raises(ValueError, match="an XYZ_ROT6D"):
        _infer_n1_7_action_groups(
            BIMANUAL_NAMES,
            action_dim=len(BIMANUAL_NAMES),
            exclude_joints=[],
            eef_groups=["right_ee_pos"],  # matches only the 3 translation dims
        )


def test_one_token_matching_several_runs_gets_unique_keys():
    """Keys index the raw_stats/modality_keys dicts, so a collision would drop a group."""
    groups = _infer_n1_7_action_groups(
        BIMANUAL_NAMES,
        action_dim=len(BIMANUAL_NAMES),
        exclude_joints=[],
        eef_groups=["_ee_"],  # matches the right pose run and the left pose run
    )

    eef = [g for g in groups if g.eef]
    assert len(eef) == 2
    assert len({g.key for g in eef}) == 2, "two EEF groups collided on one key"
    assert len({g.key for g in groups}) == len(groups)
    assert [g.indices for g in eef] == [list(range(9)), list(range(13, 22))]


def test_eef_and_exclude_overlap_raises():
    """Contradictory config fails loudly rather than one silently winning."""
    with pytest.raises(ValueError, match="matches both relative_eef_groups"):
        _infer_n1_7_action_groups(
            BIMANUAL_NAMES,
            action_dim=len(BIMANUAL_NAMES),
            exclude_joints=["ee_rot"],
            eef_groups=["right_ee"],
        )


def test_exclude_joints_still_works_alongside_eef_groups():
    groups = _infer_n1_7_action_groups(
        BIMANUAL_NAMES,
        action_dim=len(BIMANUAL_NAMES),
        exclude_joints=["right_finger_0"],
        eef_groups=["right_ee", "left_ee"],
    )

    absolute = [g for g in groups if not g.relative]
    assert len(absolute) == 1
    assert absolute[0].indices == [9]
    flat = [i for g in groups for i in g.indices]
    assert flat == list(range(len(BIMANUAL_NAMES)))


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = q
    transform[:3, 3] = rng.uniform(-1.0, 1.0, size=3)
    return homogeneous_to_xyz_rot6d(transform)


def test_batch_converter_applies_se3_to_eef_and_subtraction_elsewhere():
    rng = np.random.default_rng(0)
    batch, horizon = 2, 3
    groups = [
        _GrootN17ActionGroup(key="arm", indices=list(range(9)), relative=True, eef=True),
        _GrootN17ActionGroup(key="hand", indices=[9, 10], relative=True),
        _GrootN17ActionGroup(key="aux", indices=[11], relative=False),
    ]

    state_pose = np.stack([_random_pose(rng) for _ in range(batch)])
    state = torch.from_numpy(
        np.concatenate([state_pose, rng.uniform(-1, 1, size=(batch, 3))], axis=-1)
    ).float()
    action_pose = np.stack([[_random_pose(rng) for _ in range(horizon)] for _ in range(batch)])
    action = torch.from_numpy(
        np.concatenate([action_pose, rng.uniform(-1, 1, size=(batch, horizon, 3))], axis=-1)
    ).float()

    converted = _convert_action_batch_to_relative(action, state, groups)

    np.testing.assert_allclose(
        converted[:, :, :9].numpy(),
        absolute_eef_to_relative(action[:, :, :9].numpy(), state[:, :9].numpy()),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        converted[:, :, 9:11].numpy(),
        (action[:, :, 9:11] - state[:, None, 9:11]).numpy(),
        atol=1e-6,
    )
    # Absolute group untouched.
    np.testing.assert_allclose(converted[:, :, 11].numpy(), action[:, :, 11].numpy(), atol=0)


def test_batch_converter_does_not_mutate_its_input():
    rng = np.random.default_rng(1)
    groups = [_GrootN17ActionGroup(key="arm", indices=list(range(9)), relative=True, eef=True)]
    state = torch.from_numpy(_random_pose(rng)[None, ...]).float()
    action = torch.from_numpy(_random_pose(rng)[None, None, ...]).float()
    original = action.clone()

    _convert_action_batch_to_relative(action, state, groups)

    np.testing.assert_allclose(action.numpy(), original.numpy(), atol=0)


def test_batch_converter_matches_pack_step_conversion():
    """The stats converter and the pack step must produce identical values.

    If these diverge, relative actions get normalized against the distribution of a different
    transform than the one training applies -- silently, and only visible as a bad policy.
    """
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep

    rng = np.random.default_rng(2)
    batch, horizon = 2, 4
    entry9 = {"min": [-1.0] * 9, "max": [1.0] * 9}
    entry2 = {"min": [-1.0] * 2, "max": [1.0] * 2}
    raw_stats = {
        "state": {"arm": entry9, "hand": entry2},
        "action": {"arm": entry9, "hand": entry2},
        "relative_action": {"arm": entry9, "hand": entry2},
    }
    modality_config = {
        "state": {"modality_keys": ["arm", "hand"]},
        "action": {
            "modality_keys": ["arm", "hand"],
            "action_configs": [
                {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "arm"},
                {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "hand"},
            ],
        },
    }
    step = GrootN17PackInputsStep(
        raw_stats=raw_stats, modality_config=modality_config, normalize_min_max=False, training=True
    )
    groups = [
        _GrootN17ActionGroup(key="arm", indices=list(range(9)), relative=True, eef=True),
        _GrootN17ActionGroup(key="hand", indices=[9, 10], relative=True),
    ]

    state_pose = np.stack([_random_pose(rng) for _ in range(batch)])
    state = torch.from_numpy(
        np.concatenate([state_pose, rng.uniform(-1, 1, size=(batch, 2))], axis=-1)
    ).float()
    action_pose = np.stack([[_random_pose(rng) for _ in range(horizon)] for _ in range(batch)])
    action = torch.from_numpy(
        np.concatenate([action_pose, rng.uniform(-1, 1, size=(batch, horizon, 2))], axis=-1)
    ).float()

    from_stats_path = _convert_action_batch_to_relative(action, state, groups)
    from_pack_step = step._convert_relative_action_groups_for_training(action, state)

    np.testing.assert_allclose(from_stats_path.numpy(), from_pack_step.numpy(), atol=1e-6)
