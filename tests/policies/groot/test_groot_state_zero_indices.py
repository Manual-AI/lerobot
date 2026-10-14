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

"""`GrootN17PackInputsStep.state_zero_indices`: a recorded, always-on state mask.

The listed state dims are zeroed after min-max normalization and padding on every call, in
training and eval, with or without grad. The raw-state cache that relative actions anchor on
keeps the real values, and the field survives serialization so a reloaded checkpoint applies it.
"""

import json

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
from lerobot.policies.groot.utils import homogeneous_to_xyz_rot6d
from lerobot.processor import DataProcessorPipeline
from lerobot.processor.converters import identity_transition
from lerobot.utils.constants import ACTION, OBS_STATE

EEF_DIM = 9
HAND_DIM = 3
STATE_DIM = EEF_DIM + HAND_DIM
HORIZON = 4
MAX_DIM = 16
EEF_MASK = list(range(EEF_DIM))
HAND = list(range(EEF_DIM, STATE_DIM))


def _stats_entry(dim: int) -> dict[str, list[float]]:
    return {"min": [-1.0] * dim, "max": [1.0] * dim}


def _make_step(**kwargs) -> GrootN17PackInputsStep:
    """A pack step with one relative EEF group and one absolute hand group, min-max normalized."""
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
                {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "arm_eef"},
                {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "hand"},
            ],
        },
    }
    params = {
        "max_state_dim": MAX_DIM,
        "max_action_dim": MAX_DIM,
        "action_horizon": HORIZON,
        "valid_action_horizon": HORIZON,
        "raw_stats": raw_stats,
        "modality_config": modality_config,
        "stats": {
            OBS_STATE: {"min": torch.full((STATE_DIM,), -2.0), "max": torch.full((STATE_DIM,), 2.0)},
            ACTION: {"min": torch.full((STATE_DIM,), -2.0), "max": torch.full((STATE_DIM,), 2.0)},
        },
        "normalize_min_max": True,
    }
    params.update(kwargs)
    return GrootN17PackInputsStep(**params)


def _pose(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = q
    transform[:3, 3] = rng.uniform(-1.0, 1.0, size=3)
    return homogeneous_to_xyz_rot6d(transform)


def _make_transition(seed: int = 0, batch: int = 2):
    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [np.stack([_pose(rng) for _ in range(batch)]), rng.uniform(-0.5, 0.5, size=(batch, HAND_DIM))],
        axis=-1,
    )
    action = np.concatenate(
        [
            np.stack([[_pose(rng) for _ in range(HORIZON)] for _ in range(batch)]),
            rng.uniform(-0.5, 0.5, size=(batch, HORIZON, HAND_DIM)),
        ],
        axis=-1,
    )
    return {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.from_numpy(state).float()},
        TransitionKey.ACTION: torch.from_numpy(action).float(),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move"] * batch},
    }


def _run(step: GrootN17PackInputsStep, seed: int = 0):
    return step(_make_transition(seed))


@pytest.mark.parametrize("training", [False, True])
def test_partial_mask_zeroes_listed_dims_and_keeps_the_rest(training):
    masked_step = _make_step(state_zero_indices=EEF_MASK, training=training)
    plain_step = _make_step()

    masked = _run(masked_step)
    plain = _run(plain_step)

    state = masked[TransitionKey.OBSERVATION]["state"]
    reference = plain[TransitionKey.OBSERVATION]["state"]
    assert state.shape == (2, 1, MAX_DIM)
    assert torch.equal(state[..., EEF_MASK], torch.zeros(2, 1, EEF_DIM))
    assert torch.equal(state[..., HAND], reference[..., HAND])
    assert not torch.equal(reference[..., HAND], torch.zeros_like(reference[..., HAND]))
    assert torch.equal(state[..., STATE_DIM:], torch.zeros(2, 1, MAX_DIM - STATE_DIM))


def test_mask_applies_under_no_grad_in_training_mode():
    step = _make_step(state_zero_indices=EEF_MASK, training=True)
    with torch.no_grad():
        out = _run(step)
    assert torch.equal(out[TransitionKey.OBSERVATION]["state"][..., EEF_MASK], torch.zeros(2, 1, EEF_DIM))


def test_raw_state_anchor_and_relative_targets_are_unchanged():
    masked_step = _make_step(state_zero_indices=EEF_MASK)
    plain_step = _make_step()
    transition = _make_transition()
    raw_state = transition[TransitionKey.OBSERVATION][OBS_STATE].clone()

    masked = masked_step(_make_transition())
    plain = plain_step(_make_transition())

    torch.testing.assert_close(masked_step._last_raw_state_tensor, raw_state, rtol=0, atol=0)
    assert masked_step._last_raw_state.keys() == plain_step._last_raw_state.keys()
    for key, value in plain_step._last_raw_state.items():
        np.testing.assert_array_equal(masked_step._last_raw_state[key], value)
    np.testing.assert_array_equal(masked_step._last_raw_state["arm_eef"], raw_state[:, :EEF_DIM].numpy())
    # Relative targets are encoded against the raw state, so they match the unmasked step's.
    assert torch.equal(masked[TransitionKey.ACTION], plain[TransitionKey.ACTION])
    assert torch.equal(
        masked[TransitionKey.COMPLEMENTARY_DATA]["action_mask"],
        plain[TransitionKey.COMPLEMENTARY_DATA]["action_mask"],
    )


def test_mask_does_not_mutate_the_input_state_when_no_padding_is_needed():
    """With `dim == max_state_dim` there is no padding copy between the input and the mask."""
    step = _make_step(max_state_dim=STATE_DIM, state_zero_indices=EEF_MASK)
    transition = _make_transition()
    state = transition[TransitionKey.OBSERVATION][OBS_STATE]
    original = state.clone()

    out = step(transition)

    assert torch.equal(state, original)
    assert torch.equal(step._last_raw_state_tensor, original)
    assert torch.equal(out[TransitionKey.OBSERVATION]["state"][..., EEF_MASK], torch.zeros(2, 1, EEF_DIM))


@pytest.mark.parametrize("training", [False, True])
def test_all_indices_mask_matches_old_full_state_dropout(training):
    """The full mask feeds the model exactly what `state_dropout_prob=1.0` did in a training pass."""
    old = _make_step(training=True, state_dropout_prob=1.0)
    new = _make_step(training=training, state_zero_indices=list(range(STATE_DIM)))

    with torch.enable_grad():
        old_out = _run(old)
        new_out = _run(new)

    assert torch.equal(
        new_out[TransitionKey.OBSERVATION]["state"], old_out[TransitionKey.OBSERVATION]["state"]
    )
    assert torch.equal(new_out[TransitionKey.OBSERVATION]["state"], torch.zeros(2, 1, MAX_DIM))
    assert torch.equal(new_out[TransitionKey.ACTION], old_out[TransitionKey.ACTION])


def test_old_full_state_dropout_is_inert_outside_a_training_grad_pass():
    """Why the field exists: the dropout it replaces did not act at inference or under no_grad."""
    eval_step = _make_step(training=False, state_dropout_prob=1.0)
    train_step = _make_step(training=True, state_dropout_prob=1.0)
    with torch.no_grad():
        no_grad_out = _run(train_step)
    eval_out = _run(eval_step)
    for out in (eval_out, no_grad_out):
        assert not torch.equal(out[TransitionKey.OBSERVATION]["state"], torch.zeros(2, 1, MAX_DIM))


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ([0, 0], "repeated"),
        ([-1], "negative"),
        ([MAX_DIM], "out of range"),
        ([1.0], "integers"),
    ],
)
def test_invalid_indices_are_refused_at_construction(indices, match):
    with pytest.raises(ValueError, match=match):
        _make_step(state_zero_indices=indices)


def test_indices_beyond_the_real_state_dim_are_refused_at_call_time():
    step = _make_step(state_zero_indices=[STATE_DIM])
    with pytest.raises(ValueError, match="out of range for a dimension of size 12"):
        _run(step)


def test_mask_requires_min_max_normalization():
    with pytest.raises(ValueError, match="normalize_min_max"):
        _make_step(state_zero_indices=EEF_MASK, normalize_min_max=False)


def test_mask_requires_state_stats():
    with pytest.raises(ValueError, match="min/max stats"):
        _make_step(state_zero_indices=EEF_MASK, stats={ACTION: {"min": [0.0], "max": [1.0]}})
    step = _make_step(state_zero_indices=EEF_MASK, stats=None)
    with pytest.raises(ValueError, match="min/max stats"):
        _run(step)


def test_mask_assigned_after_construction_is_validated_on_call():
    step = _make_step(normalize_min_max=False)
    step.state_zero_indices = EEF_MASK
    with pytest.raises(ValueError, match="normalize_min_max"):
        _run(step)


def test_get_config_emits_the_field_only_when_set():
    assert "state_zero_indices" not in _make_step().get_config()
    config = _make_step(state_zero_indices=EEF_MASK).get_config()
    assert config["state_zero_indices"] == EEF_MASK
    restored = GrootN17PackInputsStep(**config)
    assert restored.state_zero_indices == EEF_MASK


def test_mask_survives_save_and_reload(tmp_path):
    pipeline = DataProcessorPipeline(
        [_make_step(state_zero_indices=EEF_MASK)],
        name="groot_masked",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    expected = pipeline(_make_transition())

    pipeline.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "groot_masked.json").read_text())
    assert saved["steps"][0]["config"]["state_zero_indices"] == EEF_MASK

    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="groot_masked.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    loaded_step = loaded.steps[0]
    assert loaded_step.state_zero_indices == EEF_MASK
    assert loaded_step.training is False

    out = loaded(_make_transition())
    assert torch.equal(out[TransitionKey.OBSERVATION]["state"], expected[TransitionKey.OBSERVATION]["state"])
    assert torch.equal(out[TransitionKey.OBSERVATION]["state"][..., EEF_MASK], torch.zeros(2, 1, EEF_DIM))


def test_pipeline_saved_without_the_field_loads_unmasked(tmp_path):
    pipeline = DataProcessorPipeline(
        [_make_step()], name="groot_plain", to_transition=identity_transition, to_output=identity_transition
    )
    pipeline.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "groot_plain.json").read_text())
    assert "state_zero_indices" not in saved["steps"][0]["config"]

    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="groot_plain.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    assert loaded.steps[0].state_zero_indices == []
