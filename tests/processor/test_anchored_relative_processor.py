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

import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    AnchoredAbsoluteEEFStep,
    AnchoredRelativeEEFStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.anchored_relative_processor import (
    anchored_compose,
    anchored_delta,
    hom_inverse,
    hom_to_pose9,
    pose9_to_hom,
    rot6d_to_matrix,
)
from lerobot.utils.constants import OBS_STATE

IDENTITY_POSE9 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def random_pose9(*batch_shape: int, seed: int) -> torch.Tensor:
    """Create valid xyz+row-rot6d poses from random proper rotations."""
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(*batch_shape, 3, 3, generator=generator, dtype=torch.float64))
    determinant = torch.linalg.det(rotation)
    rotation[..., :, 0] = rotation[..., :, 0] * determinant.unsqueeze(-1)
    translation = torch.randn(*batch_shape, 3, generator=generator, dtype=torch.float64)
    return torch.cat([translation, rotation[..., 0, :], rotation[..., 1, :]], dim=-1).float()


def _transition(state: torch.Tensor | None = None, action: torch.Tensor | None = None):
    transition = {}
    if state is not None:
        transition[TransitionKey.OBSERVATION] = {OBS_STATE: state}
    if action is not None:
        transition[TransitionKey.ACTION] = action
    return transition


def test_rot6d_matrix_round_trip():
    poses = random_pose9(4, seed=0)
    rotation = rot6d_to_matrix(poses[..., 3:9])
    recovered = torch.cat([rotation[..., 0, :], rotation[..., 1, :]], dim=-1)

    torch.testing.assert_close(recovered, poses[..., 3:9], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2),
        torch.eye(3).expand(4, 3, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(torch.linalg.det(rotation) > 0.99)


def test_hom_inverse():
    hom = pose9_to_hom(random_pose9(4, seed=1))
    torch.testing.assert_close(hom_inverse(hom) @ hom, torch.eye(4).expand(4, 4, 4), atol=1e-5, rtol=1e-5)


def test_anchored_delta_of_state_is_identity():
    state = torch.cat([random_pose9(2, seed=2), torch.randn(2, 8)], dim=-1)
    actions = state.unsqueeze(1).clone()

    delta = anchored_delta(actions, state, eef_starts=[0])

    torch.testing.assert_close(delta[:, 0, :9], IDENTITY_POSE9.expand(2, 9), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(delta[:, 0, 9:], torch.zeros(2, 8), atol=1e-6, rtol=0)


@pytest.mark.parametrize("with_time", [False, True])
def test_delta_compose_round_trip(with_time: bool):
    state = torch.cat([random_pose9(3, seed=3), torch.randn(3, 8)], dim=-1)
    if with_time:
        actions = torch.cat([random_pose9(3, 5, seed=4), torch.randn(3, 5, 8)], dim=-1)
    else:
        actions = torch.cat([random_pose9(3, seed=4), torch.randn(3, 8)], dim=-1)

    recovered = anchored_compose(anchored_delta(actions, state, [0]), state, [0])

    assert recovered.shape == actions.shape
    torch.testing.assert_close(recovered, actions, atol=1e-4, rtol=1e-4)


def test_world_frame_invariance():
    state = random_pose9(2, seed=5)
    actions = random_pose9(2, 4, seed=6)
    global_transform = pose9_to_hom(random_pose9(seed=7))
    moved_state = hom_to_pose9(global_transform @ pose9_to_hom(state))
    moved_actions = hom_to_pose9(global_transform @ pose9_to_hom(actions))

    expected = anchored_delta(actions, state, eef_starts=[0])
    actual = anchored_delta(moved_actions, moved_state, eef_starts=[0])

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_two_eef_blocks_round_trip():
    state = torch.cat(
        [
            random_pose9(2, seed=20),
            torch.randn(2, 8),
            random_pose9(2, seed=21),
            torch.randn(2, 8),
        ],
        dim=-1,
    )
    actions = torch.cat(
        [
            random_pose9(2, 4, seed=22),
            torch.randn(2, 4, 8),
            random_pose9(2, 4, seed=23),
            torch.randn(2, 4, 8),
        ],
        dim=-1,
    )

    recovered = anchored_compose(anchored_delta(actions, state, [0, 17]), state, [0, 17])

    torch.testing.assert_close(recovered, actions, atol=1e-4, rtol=1e-4)


def test_invalid_layout_raises():
    with pytest.raises(ValueError, match="carry the action layout as a prefix"):
        anchored_delta(torch.zeros(2, 17), torch.zeros(2, 16), [0])
    with pytest.raises(ValueError, match="does not fit"):
        anchored_delta(torch.zeros(2, 17), torch.zeros(2, 17), [9])


def test_pre_step_converts_training_chunk():
    state = torch.cat([random_pose9(2, seed=10), torch.randn(2, 8)], dim=-1)
    actions = torch.cat([random_pose9(2, 5, seed=11), torch.randn(2, 5, 8)], dim=-1)
    step = AnchoredRelativeEEFStep(enabled=True, eef_starts=[0])

    output = step(_transition(state=state, action=actions))

    torch.testing.assert_close(output[TransitionKey.ACTION], anchored_delta(actions, state, [0]))
    assert step._last_state is state


def test_pre_step_disabled_still_caches_state():
    state = torch.cat([random_pose9(1, seed=12), torch.randn(1, 8)], dim=-1)
    actions = torch.cat([random_pose9(1, 3, seed=13), torch.randn(1, 3, 8)], dim=-1)
    step = AnchoredRelativeEEFStep(enabled=False, eef_starts=[0])

    output = step(_transition(state=state, action=actions))

    assert output[TransitionKey.ACTION] is actions
    assert step._last_state is state


def test_post_step_freezes_anchor_per_chunk():
    pre_step = AnchoredRelativeEEFStep(enabled=True, eef_starts=[0])
    post_step = AnchoredAbsoluteEEFStep(
        enabled=True, eef_starts=[0], n_action_steps=3, relative_step=pre_step
    )
    anchor_a = torch.cat([random_pose9(1, seed=14), torch.randn(1, 8)], dim=-1)
    anchor_b = torch.cat([random_pose9(1, seed=15), torch.randn(1, 8)], dim=-1)
    delta = torch.cat([random_pose9(1, seed=16), torch.randn(1, 8)], dim=-1)

    pre_step(_transition(state=anchor_a))
    first = post_step(_transition(action=delta))[TransitionKey.ACTION]
    pre_step(_transition(state=anchor_b))
    second = post_step(_transition(action=delta))[TransitionKey.ACTION]
    pre_step(_transition(state=anchor_b))
    third = post_step(_transition(action=delta))[TransitionKey.ACTION]
    pre_step(_transition(state=anchor_b))
    fourth = post_step(_transition(action=delta))[TransitionKey.ACTION]

    torch.testing.assert_close(first, anchored_compose(delta, anchor_a, [0]))
    torch.testing.assert_close(second, first)
    torch.testing.assert_close(third, first)
    torch.testing.assert_close(fourth, anchored_compose(delta, anchor_b, [0]))

    post_step.reset()
    pre_step(_transition(state=anchor_a))
    reset_output = post_step(_transition(action=delta))[TransitionKey.ACTION]
    torch.testing.assert_close(reset_output, anchored_compose(delta, anchor_a, [0]))


def test_post_step_chunk_path_uses_latest_state():
    pre_step = AnchoredRelativeEEFStep(enabled=True, eef_starts=[0])
    post_step = AnchoredAbsoluteEEFStep(
        enabled=True, eef_starts=[0], n_action_steps=3, relative_step=pre_step
    )
    state = torch.cat([random_pose9(2, seed=17), torch.randn(2, 8)], dim=-1)
    chunk = torch.cat([random_pose9(2, 4, seed=18), torch.randn(2, 4, 8)], dim=-1)

    pre_step(_transition(state=state))
    output = post_step(_transition(action=chunk))[TransitionKey.ACTION]

    torch.testing.assert_close(output, anchored_compose(chunk, state, [0]))


def test_post_step_without_state_raises():
    post_step = AnchoredAbsoluteEEFStep(
        enabled=True,
        eef_starts=[0],
        n_action_steps=3,
        relative_step=AnchoredRelativeEEFStep(enabled=True, eef_starts=[0]),
    )

    with pytest.raises(RuntimeError, match="no cached observation.state"):
        post_step(_transition(action=torch.zeros(1, 17)))


def test_act_pipeline_wiring():
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    config = ACTConfig(chunk_size=5, n_action_steps=5, relative_actions=True, relative_eef_starts=[0])

    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=None)

    relative_step = next(step for step in preprocessor.steps if isinstance(step, AnchoredRelativeEEFStep))
    absolute_step = next(step for step in postprocessor.steps if isinstance(step, AnchoredAbsoluteEEFStep))
    assert relative_step.enabled
    assert relative_step.eef_starts == [0]
    assert absolute_step.enabled
    assert absolute_step.n_action_steps == 5
    assert absolute_step.relative_step is relative_step
    assert preprocessor.steps.index(relative_step) < next(
        index for index, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)
    )


def test_act_default_pipeline_is_unchanged():
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    preprocessor, postprocessor = make_act_pre_post_processors(ACTConfig(), dataset_stats=None)

    assert not any(isinstance(step, AnchoredRelativeEEFStep) for step in preprocessor.steps)
    assert not any(isinstance(step, AnchoredAbsoluteEEFStep) for step in postprocessor.steps)


def test_act_config_rejects_relative_with_temporal_ensembling():
    from lerobot.policies.act.configuration_act import ACTConfig

    with pytest.raises(ValueError, match="incompatible with temporal ensembling"):
        ACTConfig(
            relative_actions=True,
            relative_eef_starts=[0],
            temporal_ensemble_coeff=0.01,
            n_action_steps=1,
        )


def test_reconnect_after_save_load(tmp_path):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.policies.factory import _reconnect_relative_absolute_steps

    config = ACTConfig(chunk_size=5, n_action_steps=5, relative_actions=True, relative_eef_starts=[0])
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=None)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    loaded_preprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path, config_filename=f"{preprocessor.name}.json"
    )
    loaded_postprocessor = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{postprocessor.name}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    absolute_step = next(
        step for step in loaded_postprocessor.steps if isinstance(step, AnchoredAbsoluteEEFStep)
    )
    assert absolute_step.relative_step is None

    _reconnect_relative_absolute_steps(loaded_preprocessor, loaded_postprocessor)

    relative_step = next(
        step for step in loaded_preprocessor.steps if isinstance(step, AnchoredRelativeEEFStep)
    )
    assert absolute_step.relative_step is relative_step
