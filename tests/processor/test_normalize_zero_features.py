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

"""`NormalizerProcessorStep.zero_features`: observation dims zeroed in normalized space.

Zero is the mean under MEAN_STD and the range midpoint under MIN_MAX / QUANTILES. The mask is
applied on every call, out of place, so steps that cached the raw observation (the relative-action
anchor) keep the real values; keys that the normalizer would hand back raw are refused.
"""

import json

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DataProcessorPipeline,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.utils.constants import ACTION, OBS_STATE

STATE_DIM = 4
ACTION_DIM = 4
MASK = [1, 3]
KEPT = [0, 2]

FEATURES = {
    OBS_STATE: PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
    ACTION: PolicyFeature(FeatureType.ACTION, (ACTION_DIM,)),
}

STATS = {
    OBS_STATE: {
        "mean": torch.tensor([1.0, -2.0, 3.0, 0.5]),
        "std": torch.tensor([2.0, 0.5, 1.0, 4.0]),
        "min": torch.tensor([-1.0, -4.0, 0.0, -2.0]),
        "max": torch.tensor([3.0, 0.0, 6.0, 2.0]),
        "q01": torch.tensor([-0.5, -3.5, 0.5, -1.5]),
        "q99": torch.tensor([2.5, -0.5, 5.5, 1.5]),
    },
    ACTION: {
        "mean": torch.zeros(ACTION_DIM),
        "std": torch.ones(ACTION_DIM),
        "min": -torch.ones(ACTION_DIM),
        "max": torch.ones(ACTION_DIM),
        "q01": -torch.ones(ACTION_DIM),
        "q99": torch.ones(ACTION_DIM),
    },
}


def _norm_map(state_mode: NormalizationMode) -> dict[FeatureType, NormalizationMode]:
    return {FeatureType.STATE: state_mode, FeatureType.ACTION: NormalizationMode.MEAN_STD}


def _normalizer(state_mode=NormalizationMode.MEAN_STD, zero_features=None, **kwargs):
    return NormalizerProcessorStep(
        features=FEATURES,
        norm_map=_norm_map(state_mode),
        stats=STATS,
        zero_features={OBS_STATE: MASK} if zero_features is None else zero_features,
        **kwargs,
    )


def _state(shape=(2, STATE_DIM)) -> torch.Tensor:
    return torch.randn(*shape) * 3 + 1


@pytest.mark.parametrize(
    "state_mode",
    [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES],
)
@pytest.mark.parametrize("shape", [(2, STATE_DIM), (2, 3, STATE_DIM)])
def test_zero_features_zero_masked_dims_and_keep_the_rest(state_mode, shape):
    state = _state(shape)
    action = torch.randn(2, ACTION_DIM)
    masked = _normalizer(state_mode)
    reference = _normalizer(state_mode, zero_features={})

    out = masked(create_transition(observation={OBS_STATE: state}, action=action))
    ref = reference(create_transition(observation={OBS_STATE: state}, action=action))

    out_state = out[TransitionKey.OBSERVATION][OBS_STATE]
    ref_state = ref[TransitionKey.OBSERVATION][OBS_STATE]
    assert out_state.shape == state.shape
    assert torch.equal(out_state[..., MASK], torch.zeros_like(out_state[..., MASK]))
    assert torch.equal(out_state[..., KEPT], ref_state[..., KEPT])
    assert torch.equal(out[TransitionKey.ACTION], ref[TransitionKey.ACTION])


@pytest.mark.parametrize(
    ("state_mode", "center"),
    [
        (NormalizationMode.MEAN_STD, STATS[OBS_STATE]["mean"]),
        (NormalizationMode.MIN_MAX, (STATS[OBS_STATE]["min"] + STATS[OBS_STATE]["max"]) / 2),
        (NormalizationMode.QUANTILES, (STATS[OBS_STATE]["q01"] + STATS[OBS_STATE]["q99"]) / 2),
    ],
)
def test_zero_is_the_normalized_value_of_the_distribution_center(state_mode, center):
    """The mask feeds the policy what an observation sitting at the mean / midpoint would."""
    unmasked = _normalizer(state_mode, zero_features={})
    out = unmasked(create_transition(observation={OBS_STATE: center.unsqueeze(0)}))
    torch.testing.assert_close(out[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros(1, STATE_DIM))


def test_all_indices_mask_zeroes_the_whole_feature():
    step = _normalizer(zero_features={OBS_STATE: list(range(STATE_DIM))})
    out = step(create_transition(observation={OBS_STATE: _state((2, 3, STATE_DIM))}))
    assert torch.equal(out[TransitionKey.OBSERVATION][OBS_STATE], torch.zeros(2, 3, STATE_DIM))


def test_zero_features_is_out_of_place():
    state = _state()
    original = state.clone()
    step = _normalizer()
    out = step(create_transition(observation={OBS_STATE: state}))
    assert torch.equal(state, original)
    assert out[TransitionKey.OBSERVATION][OBS_STATE] is not state


def test_zero_features_applies_under_no_grad():
    step = _normalizer()
    with torch.no_grad():
        out = step(create_transition(observation={OBS_STATE: _state()}))
    assert torch.equal(out[TransitionKey.OBSERVATION][OBS_STATE][:, MASK], torch.zeros(2, len(MASK)))


def test_relative_actions_anchor_stays_raw():
    """The relative step caches a reference to the raw state; the mask must not reach it."""
    state = _state()
    original = state.clone()
    relative = RelativeActionsProcessorStep(enabled=True)
    pipeline = DataProcessorPipeline(
        [relative, _normalizer()], to_transition=identity_transition, to_output=identity_transition
    )

    out = pipeline(create_transition(observation={OBS_STATE: state}, action=torch.randn(2, ACTION_DIM)))

    assert relative._last_state is state
    assert torch.equal(relative._last_state, original)
    assert torch.equal(out[TransitionKey.OBSERVATION][OBS_STATE][:, MASK], torch.zeros(2, len(MASK)))


@pytest.mark.parametrize(
    ("zero_features", "match"),
    [
        ({ACTION: [0]}, "action feature"),
        ({"observation.missing": [0]}, "not a feature"),
        ({OBS_STATE: []}, "non-empty"),
        ({OBS_STATE: [0, 0]}, "repeated"),
        ({OBS_STATE: [-1]}, "negative"),
        ({OBS_STATE: [STATE_DIM]}, "out of range"),
        ({OBS_STATE: [1.0]}, "integers"),
        ({OBS_STATE: [True]}, "integers"),
    ],
)
def test_zero_features_refuses_invalid_entries(zero_features, match):
    with pytest.raises(ValueError, match=match):
        _normalizer(zero_features=zero_features)


def test_zero_features_refuses_identity_mode():
    with pytest.raises(ValueError, match="IDENTITY"):
        _normalizer(NormalizationMode.IDENTITY)


def test_zero_features_refuses_a_key_without_norm_map_entry():
    with pytest.raises(ValueError, match="IDENTITY"):
        NormalizerProcessorStep(
            features=FEATURES,
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats=STATS,
            zero_features={OBS_STATE: MASK},
        )


def test_zero_features_refuses_a_key_without_stats():
    with pytest.raises(ValueError, match="no normalization stats"):
        NormalizerProcessorStep(
            features=FEATURES,
            norm_map=_norm_map(NormalizationMode.MEAN_STD),
            stats={ACTION: STATS[ACTION]},
            zero_features={OBS_STATE: MASK},
        )


def test_zero_features_refuses_a_key_excluded_from_normalization():
    with pytest.raises(ValueError, match="normalize_observation_keys"):
        _normalizer(normalize_observation_keys={"observation.other"})


def test_zero_features_without_stats_refuses_at_call_time():
    """No stats at construction (the `from_pretrained` order) defers the stats check to the call."""
    step = NormalizerProcessorStep(
        features=FEATURES,
        norm_map=_norm_map(NormalizationMode.MEAN_STD),
        zero_features={OBS_STATE: MASK},
    )
    with pytest.raises(ValueError, match="no normalization stats"):
        step(create_transition(observation={OBS_STATE: _state()}))


def test_zero_features_assigned_after_construction_is_validated_on_call():
    step = _normalizer(zero_features={})
    step.zero_features = {ACTION: [0]}
    with pytest.raises(ValueError, match="action feature"):
        step(create_transition(observation={OBS_STATE: _state()}))


def test_zero_features_refuses_indices_beyond_the_runtime_dim():
    step = NormalizerProcessorStep(
        features={OBS_STATE: PolicyFeature(FeatureType.STATE, ())},
        norm_map=_norm_map(NormalizationMode.MEAN_STD),
        stats={OBS_STATE: {"mean": torch.zeros(2), "std": torch.ones(2)}},
        zero_features={OBS_STATE: [3]},
    )
    with pytest.raises(ValueError, match="out of range"):
        step(create_transition(observation={OBS_STATE: torch.randn(1, 2)}))


def test_zero_features_get_config_only_when_set():
    assert "zero_features" not in _normalizer(zero_features={}).get_config()
    assert _normalizer().get_config()["zero_features"] == {OBS_STATE: MASK}


def test_zero_features_round_trip_through_save_and_load(tmp_path):
    state = _state()
    pipeline = DataProcessorPipeline(
        [_normalizer(NormalizationMode.MIN_MAX)],
        name="masked",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    expected = pipeline(create_transition(observation={OBS_STATE: state}))

    pipeline.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "masked.json").read_text())
    assert saved["steps"][0]["config"]["zero_features"] == {OBS_STATE: MASK}

    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="masked.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    assert loaded.steps[0].zero_features == {OBS_STATE: MASK}
    out = loaded(create_transition(observation={OBS_STATE: state}))
    assert torch.equal(
        out[TransitionKey.OBSERVATION][OBS_STATE], expected[TransitionKey.OBSERVATION][OBS_STATE]
    )


def test_config_saved_without_zero_features_loads_unchanged(tmp_path):
    """Pipelines saved before the field existed have no `zero_features` key."""
    state = _state()
    pipeline = DataProcessorPipeline(
        [_normalizer(zero_features={})],
        name="legacy",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    expected = pipeline(create_transition(observation={OBS_STATE: state}))
    pipeline.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "legacy.json").read_text())
    assert "zero_features" not in saved["steps"][0]["config"]

    loaded = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="legacy.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    assert loaded.steps[0].zero_features == {}
    out = loaded(create_transition(observation={OBS_STATE: state}))
    assert torch.equal(
        out[TransitionKey.OBSERVATION][OBS_STATE], expected[TransitionKey.OBSERVATION][OBS_STATE]
    )


def test_pi05_prompt_state_bins_come_from_the_masked_state():
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    state = _state()
    pipeline = DataProcessorPipeline(
        [
            RelativeActionsProcessorStep(enabled=True),
            _normalizer(NormalizationMode.QUANTILES),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=STATE_DIM),
        ],
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    out = pipeline(
        create_transition(
            observation={OBS_STATE: state},
            action=torch.randn(2, ACTION_DIM),
            complementary_data={"task": ["pick", "pick"]},
        )
    )

    masked_state = out[TransitionKey.OBSERVATION][OBS_STATE].numpy()
    bin_edges = np.linspace(-1, 1, 256 + 1)[:-1]
    zero_bin = int(np.digitize(0.0, bins=bin_edges)) - 1
    for row, prompt in enumerate(out[TransitionKey.COMPLEMENTARY_DATA]["task"]):
        bins = [int(b) for b in prompt.split("State: ")[1].split(";")[0].split()]
        assert [bins[i] for i in MASK] == [zero_bin] * len(MASK)
        expected_kept = np.digitize(masked_state[row, KEPT], bins=bin_edges) - 1
        assert [bins[i] for i in KEPT] == expected_kept.tolist()
