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

"""Geometry of ``type: EEF`` relative action groups (xyz + rot6d)."""

import numpy as np
import pytest

from lerobot.policies.groot.utils import (
    absolute_eef_to_relative,
    homogeneous_to_xyz_rot6d,
    invert_homogeneous,
    relative_eef_to_absolute,
    rot6d_to_matrix,
    xyz_rot6d_to_homogeneous,
)

IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation matrix, via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = _random_rotation(rng)
    transform[:3, 3] = rng.uniform(-2.0, 2.0, size=3)
    return transform


def _random_poses_xyz_rot6d(rng: np.random.Generator, count: int) -> np.ndarray:
    return np.stack([homogeneous_to_xyz_rot6d(_random_pose(rng)) for _ in range(count)]).astype(
        np.float32
    )


def test_absolute_to_relative_round_trips_for_arbitrary_rotation_pairs():
    """The forward and inverse conversions compose to the identity.

    Uses uniformly random rotations, not small perturbations: a delta encoding that is only
    correct near identity would pass a small-angle test and still be wrong in general.
    """
    rng = np.random.default_rng(0)
    horizon = 7
    for _ in range(64):
        reference = _random_poses_xyz_rot6d(rng, 1)
        absolute = _random_poses_xyz_rot6d(rng, horizon)[None, ...]

        relative = absolute_eef_to_relative(absolute, reference)
        recovered = relative_eef_to_absolute(relative, reference)

        assert relative.shape == absolute.shape
        np.testing.assert_allclose(recovered, absolute, atol=1e-5)


def test_relative_to_absolute_round_trips_the_other_way():
    """Starting from deltas rather than absolutes closes equally well."""
    rng = np.random.default_rng(1)
    reference = _random_poses_xyz_rot6d(rng, 1)
    relative = _random_poses_xyz_rot6d(rng, 5)[None, ...]

    absolute = relative_eef_to_absolute(relative, reference)
    recovered = absolute_eef_to_relative(absolute, reference)

    np.testing.assert_allclose(recovered, relative, atol=1e-5)


def test_delta_against_own_pose_is_identity():
    """An absolute pose expressed relative to itself is the identity transform."""
    rng = np.random.default_rng(2)
    reference = _random_poses_xyz_rot6d(rng, 1)

    relative = absolute_eef_to_relative(reference[None, ...], reference)

    np.testing.assert_allclose(relative[0, 0, :3], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(relative[0, 0, 3:], IDENTITY_ROT6D, atol=1e-6)


def test_translation_is_expressed_in_the_reference_frame():
    """A pure translation along world X reads as a delta along the reference's own axes.

    With the reference yawed 90 degrees, a target displaced by +1 in world X must come back as a
    displacement of -1 on the reference frame's Y axis. An elementwise subtraction would instead
    report +1 on X, which is the bug this conversion exists to avoid.
    """
    yaw90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference_transform = np.eye(4)
    reference_transform[:3, :3] = yaw90
    reference = homogeneous_to_xyz_rot6d(reference_transform)[None, ...].astype(np.float32)

    target_transform = reference_transform.copy()
    target_transform[:3, 3] = np.array([1.0, 0.0, 0.0])
    absolute = homogeneous_to_xyz_rot6d(target_transform)[None, None, ...].astype(np.float32)

    relative = absolute_eef_to_relative(absolute, reference)

    np.testing.assert_allclose(relative[0, 0, :3], np.array([0.0, -1.0, 0.0]), atol=1e-6)
    # Same orientation as the reference, so the rotation part is the identity.
    np.testing.assert_allclose(relative[0, 0, 3:], IDENTITY_ROT6D, atol=1e-6)


def test_rotation_delta_is_a_group_operation_not_a_subtraction():
    """``rot6d(delta)`` equals ``vec6(R_ref.T @ R_abs)``, which is not ``rot6d_abs - rot6d_ref``."""
    rng = np.random.default_rng(3)
    reference_transform = _random_pose(rng)
    absolute_transform = _random_pose(rng)

    reference = homogeneous_to_xyz_rot6d(reference_transform)[None, ...].astype(np.float32)
    absolute = homogeneous_to_xyz_rot6d(absolute_transform)[None, None, ...].astype(np.float32)

    relative = absolute_eef_to_relative(absolute, reference)

    expected = reference_transform[:3, :3].T @ absolute_transform[:3, :3]
    np.testing.assert_allclose(rot6d_to_matrix(relative[0, 0, 3:]), expected, atol=1e-5)

    elementwise = absolute[0, 0, 3:] - reference[0, 3:]
    assert not np.allclose(relative[0, 0, 3:], elementwise, atol=1e-3), (
        "group delta coincided with the elementwise difference; pick a less degenerate case"
    )


@pytest.mark.parametrize("horizon", [1, 4, 16])
def test_batch_and_horizon_shapes_are_preserved(horizon):
    rng = np.random.default_rng(4)
    batch = 3
    reference = _random_poses_xyz_rot6d(rng, batch)
    absolute = np.stack([_random_poses_xyz_rot6d(rng, horizon) for _ in range(batch)])

    relative = absolute_eef_to_relative(absolute, reference)

    assert relative.shape == (batch, horizon, 9)
    assert relative.dtype == np.float32
    np.testing.assert_allclose(
        relative_eef_to_absolute(relative, reference), absolute, atol=1e-5
    )


def test_each_batch_element_uses_its_own_reference():
    """Batch elements must not share a reference pose."""
    rng = np.random.default_rng(5)
    reference = _random_poses_xyz_rot6d(rng, 2)
    absolute = np.stack([_random_poses_xyz_rot6d(rng, 3) for _ in range(2)])

    relative = absolute_eef_to_relative(absolute, reference)

    first_only = absolute_eef_to_relative(absolute[:1], reference[:1])
    np.testing.assert_allclose(relative[0], first_only[0], atol=1e-6)
    assert not np.allclose(relative[1], first_only[0], atol=1e-3)


def test_invert_homogeneous_matches_a_general_inverse():
    rng = np.random.default_rng(6)
    for _ in range(16):
        transform = _random_pose(rng)
        np.testing.assert_allclose(
            invert_homogeneous(transform), np.linalg.inv(transform), atol=1e-9
        )


def test_conversion_tolerates_non_orthonormal_rot6d_input():
    """Predicted rot6d is not exactly orthonormal; the conversion must still yield a rotation.

    ``rot6d_to_matrix`` re-orthonormalises, so ``invert_homogeneous``'s transpose-as-inverse
    shortcut stays valid even when the incoming 6 values came from a network rather than from a
    real rotation matrix.
    """
    reference = np.array([[0.0, 0.0, 0.0, 0.9, 0.05, 0.0, 0.1, 1.2, 0.0]], dtype=np.float32)
    absolute = np.array([[[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=np.float32)

    relative = absolute_eef_to_relative(absolute, reference)
    rotation = rot6d_to_matrix(relative[0, 0, 3:])

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        relative_eef_to_absolute(relative, reference)[0, 0, :3], absolute[0, 0, :3], atol=1e-5
    )
    # Sanity: the reference really was non-orthonormal to begin with.
    assert not np.allclose(
        xyz_rot6d_to_homogeneous(reference[0])[:2, :3].reshape(-1), reference[0, 3:], atol=1e-3
    )
