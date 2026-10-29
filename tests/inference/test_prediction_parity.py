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

"""Local RTC and the remote runner share post-preprocessing action semantics."""

import pytest
import torch

pytest.importorskip("datasets", reason="local rollout imports the dataset integration")

from lerobot.inference import ExecutionMode, predict_chunk
from lerobot.inference.prediction import find_relative_rtc_prefix_encoder
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import ProcessorStep
from tests.inference.policy_helpers import (
    ConformingPolicy,
    local_engine_for,
    observation,
    processors,
    runner_for,
    tiny_config,
)

pytestmark = pytest.mark.dataset


class ObservedPolicy(ConformingPolicy):
    name = "observed_prediction"

    def __init__(self, config):
        super().__init__(config)
        self.context = None
        self.output = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        self.context = (torch.is_inference_mode_enabled(), torch.is_grad_enabled())
        return self.output.clone() if isinstance(self.output, torch.Tensor) else self.output


def local_prediction(
    monkeypatch, policy, *, relative=False, previous=None, canonical_previous=None, extra_steps=()
):
    """Run one real local prediction, capturing the chunk before its queue merge."""
    engine = local_engine_for(policy, relative=relative, queue_threshold=8, extra_steps=extra_steps)
    engine.resume()
    if previous is not None and len(previous):
        engine.action_queue.merge(previous, canonical_previous, real_delay=0)
    engine._runtime.turnarounds.append(1 / 30)
    engine.notify_observation(
        {"a.pos": 20.0, "b.pos": 20.0, "c.pos": 20.0, "env_a": 0.0, "env_b": 0.0, "env_c": 0.0}
    )
    result = []

    def capture(request, chunk, *, task_version):
        result.append(chunk)
        engine._shutdown_event.set()
        return True

    monkeypatch.setattr(engine._runtime, "accept", capture)
    engine._rtc_loop()
    return engine, result


@pytest.mark.parametrize(
    "mode,relative,prefix_steps",
    [
        (ExecutionMode.CHUNK, False, 0),
        (ExecutionMode.RTC_GUIDED, True, 2),
        (ExecutionMode.RTC_TRAINED, True, 6),
    ],
)
def test_local_and_remote_prefix_context_and_execution_slice_match(monkeypatch, mode, relative, prefix_steps):
    config = tiny_config()
    config.rtc_training_max_delay = 3
    config.rtc_config = RTCConfig(
        enabled=mode is not ExecutionMode.CHUNK,
        mode="trained" if mode is ExecutionMode.RTC_TRAINED else "guided",
        execution_horizon=4,
    )
    previous = torch.arange(prefix_steps * 3, dtype=torch.float32).reshape(prefix_steps, 3)
    canonical_previous = previous * 2 + torch.tensor([1.0, 2.0, 3.0]) + (10 if relative else 0)
    local_policy = ObservedPolicy(config)
    engine, chunks = local_prediction(
        monkeypatch,
        local_policy,
        relative=relative,
        previous=previous,
        canonical_previous=canonical_previous,
    )
    assert not engine.failed, engine.failure_traceback
    assert len(chunks) == 1
    remote_policy = ObservedPolicy(config)
    remote = runner_for(remote_policy, relative=relative, modes=(mode,)).predict(
        observation(20),
        mode=mode,
        inference_delay=1 if prefix_steps else 0,
        model_continuation=previous,
        canonical_continuation=canonical_previous,
    )
    assert local_policy.context == remote_policy.context == (mode is not ExecutionMode.RTC_GUIDED, False)
    torch.testing.assert_close(chunks[0].canonical_actions, remote.canonical_actions)
    assert chunks[0].execution_steps == remote.execution_steps == (3 if mode is ExecutionMode.CHUNK else 8)
    if mode is ExecutionMode.CHUNK:
        assert local_policy.last_kwargs == remote_policy.last_kwargs == {}
        assert remote.model_actions is None
        return
    torch.testing.assert_close(chunks[0].model_actions, remote.model_actions)
    assert local_policy.last_kwargs["inference_delay"] == remote_policy.last_kwargs["inference_delay"]
    local_prefix = local_policy.last_kwargs["prev_chunk_left_over"]
    remote_prefix = remote_policy.last_kwargs["prev_chunk_left_over"]
    expected = previous[:4] - (5 if relative else 0)
    if len(expected) < 4:
        expected = torch.cat([expected, expected[-1:].expand(4 - len(expected), -1)])
    torch.testing.assert_close(local_prefix, expected)
    torch.testing.assert_close(remote_prefix, expected)


@pytest.mark.parametrize("output", [torch.zeros(1, 7, 3), torch.full((1, 8, 3), float("nan"))])
def test_runner_rejects_invalid_policy_outputs(output):
    policy = ObservedPolicy(tiny_config())
    policy.output = output
    with pytest.raises(ValueError):
        runner_for(policy).predict(observation())


def test_shared_prediction_preserves_model_coordinates_before_inplace_postprocessing():
    policy = ObservedPolicy(tiny_config())
    prediction = predict_chunk(
        policy,
        lambda actions: actions.add_(10),
        {},
        spec=policy.chunk_inference_spec(),
        mode=ExecutionMode.CHUNK,
        canonical_action_dim=3,
        model_action_dim=None,
        device="cpu",
    )
    expected = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    torch.testing.assert_close(prediction.model_actions, expected)
    torch.testing.assert_close(prediction.canonical_actions, expected + 10)
    assert not prediction.model_actions.requires_grad
    assert not prediction.canonical_actions.requires_grad


class PrefixEncoderStep(ProcessorStep):
    """Pass-through preprocessor step implementing ``RelativeRTCPrefixEncoder``.

    Encodes an absolute tail as ``absolute - 100`` with a batch dimension, so a test can
    tell its output apart from both the model-space continuation and the generic
    ``RelativeActionsProcessorStep`` re-anchoring.
    """

    def __init__(self, *, owns_prefix=True, declines=False):
        self.owns_prefix = owns_prefix
        self.declines = declines
        self.calls = []

    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features

    def uses_relative_rtc_prefix(self):
        return self.owns_prefix

    def encode_absolute_rtc_prefix(self, prev_actions_absolute):
        self.calls.append(prev_actions_absolute.clone())
        return None if self.declines else (prev_actions_absolute - 100).unsqueeze(0)


def encoded_predictions(monkeypatch, config, *, relative=False, declines=False):
    """Predict one successor chunk locally and remotely with a policy-owned prefix encoder."""
    previous = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    canonical_previous = previous + 50
    results = []
    local_policy, local_encoder = ObservedPolicy(config), PrefixEncoderStep(declines=declines)
    engine, chunks = local_prediction(
        monkeypatch,
        local_policy,
        relative=relative,
        previous=previous,
        canonical_previous=canonical_previous,
        extra_steps=(local_encoder,),
    )
    assert not engine.failed, engine.failure_traceback
    results.append((local_policy, local_encoder, chunks[0]))
    remote_policy, remote_encoder = ObservedPolicy(config), PrefixEncoderStep(declines=declines)
    remote = runner_for(
        remote_policy, relative=relative, modes=(ExecutionMode.RTC_GUIDED,), extra_steps=(remote_encoder,)
    ).predict(
        observation(20),
        mode=ExecutionMode.RTC_GUIDED,
        inference_delay=1,
        model_continuation=previous,
        canonical_continuation=canonical_previous,
    )
    results.append((remote_policy, remote_encoder, remote))
    return canonical_previous, results


def test_prefix_encoder_is_found_only_when_it_owns_the_prefix():
    pre, _ = processors(tiny_config(), extra_steps=(PrefixEncoderStep(owns_prefix=False),))
    assert find_relative_rtc_prefix_encoder(pre) is None
    owner = PrefixEncoderStep()
    pre.steps.append(owner)
    assert find_relative_rtc_prefix_encoder(pre) is owner


@pytest.mark.parametrize("relative", [False, True])
def test_local_and_remote_condition_on_the_policy_encoded_prefix(monkeypatch, relative):
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    canonical_previous, results = encoded_predictions(monkeypatch, config, relative=relative)
    # The encoder sees only the real canonical rows (2 of the RTC horizon's 4, no hold
    # padding), and its output replaces both the model-space rows and generic re-anchoring.
    for policy, encoder, chunk in results:
        assert len(encoder.calls) == 1
        torch.testing.assert_close(encoder.calls[0], canonical_previous)
        torch.testing.assert_close(policy.last_kwargs["prev_chunk_left_over"], canonical_previous - 100)
        assert policy.last_kwargs["prev_chunk_left_over_reanchored"] is True
        assert policy.last_kwargs["inference_delay"] == 1
        assert chunk.execution_steps == 8


def test_declined_prefix_encoding_predicts_unguided_without_a_delay(monkeypatch):
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    _, results = encoded_predictions(monkeypatch, config, declines=True)
    for policy, encoder, chunk in results:
        assert len(encoder.calls) == 1
        # A delay refers to prefix rows; with the prefix refused there are none to freeze.
        assert policy.last_kwargs == {"inference_delay": 0, "prev_chunk_left_over": None}
        assert chunk.execution_steps == 8


def test_prefix_guidance_ablation_keeps_rtc_chunks_but_drops_conditioning(monkeypatch):
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4, prefix_guidance=False)
    _, results = encoded_predictions(monkeypatch, config)
    for policy, encoder, chunk in results:
        assert encoder.calls == []
        assert policy.last_kwargs == {"inference_delay": 0, "prev_chunk_left_over": None}
        # Still an RTC continuation: the full prediction horizon replaces the queue.
        assert chunk.execution_steps == 8
        assert chunk.model_actions is not None


def test_trained_rtc_rejects_disabling_prefix_guidance():
    with pytest.raises(ValueError, match="requires prefix_guidance=True"):
        RTCConfig(mode="trained", prefix_guidance=False)
    policy = ObservedPolicy(tiny_config())
    with pytest.raises(ValueError, match="requires prefix guidance"):
        predict_chunk(
            policy,
            lambda actions: actions,
            {},
            spec=policy.chunk_inference_spec(),
            mode=ExecutionMode.RTC_TRAINED,
            canonical_action_dim=3,
            model_action_dim=None,
            device="cpu",
            rtc_horizon=4,
            prefix_guidance=False,
        )
