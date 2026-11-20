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

"""Local RTC warm-up and the executed-delay history that sizes each prefix."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from tests.inference.rtc_helpers import ControlledPolicy, Pipeline, make_engine, observe, stop
from tests.utils import wait_for

_STEPS = 30


class CountingPipeline(Pipeline):
    def __init__(self, width: int | None = None):
        super().__init__(width)
        self.resets = 0

    def reset(self):
        self.resets += 1


class ExecutingPolicy(ControlledPolicy):
    """An RTC policy whose loop inferences overlap a scripted number of executed actions.

    Loop inference ``i`` plays ``executed[i]`` control ticks (dispatch an action, then publish
    an observation) while it runs; ticking from inside the inference keeps the counts
    deterministic. The ``warm_calls`` first calls tick nothing, and calls past the script wait
    for the test's release, so the loop parks there.
    """

    def __init__(self, executed: list[int], *, warm_calls: int = 0, training_max_delay: int = 4):
        super().__init__(steps=_STEPS)
        self.config.rtc_training_max_delay = training_max_delay
        self.engine = None
        self.executed = list(executed)
        self.warm_calls = warm_calls
        self.threads: list[str] = []
        self.resets = 0

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append((prev_chunk_left_over, inference_delay))
        self.threads.append(threading.current_thread().name)
        loop_call = len(self.calls) - self.warm_calls
        if loop_call > len(self.executed):
            assert self.release.acquire(timeout=3), "test must release the model call"
        elif loop_call >= 1:
            for _ in range(self.executed[loop_call - 1]):
                assert self.engine.get_action(None) is not None
                observe(self.engine)
        start = 1000.0 * len(self.calls)
        return (start + torch.arange(_STEPS * 2, dtype=torch.float32)).view(1, _STEPS, 2)

    def reset(self):
        self.resets += 1


def _engine(policy: ExecutingPolicy, rtc_config: RTCConfig | None = None):
    engine = make_engine(
        policy=policy,
        postprocessor=CountingPipeline(width=2),
        rtc_config=rtc_config or RTCConfig(enabled=True, execution_horizon=8),
        rtc_queue_threshold=_STEPS,  # re-plan back to back
    )
    engine._preprocessor = CountingPipeline()
    # A frozen clock: every turnaround is zero, so only counted delays size the prefix.
    now = time.monotonic()
    engine._runtime.clock = lambda: now
    engine._robot.observation_time = now
    policy.engine = engine
    return engine


def test_rtc_warmup_runs_both_paths_on_the_rtc_thread_then_resets():
    policy = ExecutingPolicy(executed=[], warm_calls=2)
    engine = _engine(policy)
    engine.start()
    try:
        observed = []
        engine.warmup(lambda: observed.append(1) or {"j1.pos": 0.0, "j2.pos": 0.0})
    finally:
        stop(engine)

    assert observed == [1]
    assert policy.threads == ["RTCInference", "RTCInference"]
    # Unguided first, then conditioned on the first chunk's head with a nonzero delay.
    (first_prefix, first_delay), (second_prefix, second_delay) = policy.calls
    assert first_prefix is None and first_delay == 0
    assert second_delay == 1
    torch.testing.assert_close(second_prefix, 1000.0 + torch.arange(8 * 2, dtype=torch.float32).view(8, 2))
    assert policy.resets == 1
    assert engine._preprocessor.resets == 1 and engine._postprocessor.resets == 1
    # The queue never saw the warm-up chunks; the warm call seeded the delay history.
    assert engine.action_queue.qsize() == 0
    assert len(engine._runtime.delays) == 1 and engine._runtime.delays[0] >= 1


def test_rtc_warmup_failure_is_raised_to_the_caller_and_still_resets():
    policy = ExecutingPolicy(executed=[])

    def fail(*args, **kwargs):
        raise RuntimeError("no GPU")

    policy.predict_action_chunk = fail
    engine = _engine(policy)
    engine.start()
    try:
        with pytest.raises(RuntimeError, match="no GPU"):
            engine.warmup(lambda: {"j1.pos": 0.0, "j2.pos": 0.0})
        assert not engine.failed  # the RTC thread is still up
    finally:
        stop(engine)

    assert policy.resets == 1
    assert list(engine._runtime.delays) == []


def test_rtc_warmup_requires_the_inference_thread():
    engine = _engine(ExecutingPolicy(executed=[]))
    with pytest.raises(RuntimeError, match="call start"):
        engine.warmup(lambda: {"j1.pos": 0.0, "j2.pos": 0.0})


def test_rtc_loop_conditions_on_the_largest_recently_executed_delay():
    """Each merge skips, and the history records, exactly the actions executed meanwhile."""
    # Inference 1 starts from an empty queue (nothing executes); 2-4 overlap 4, 7 and 2 actions.
    policy = ExecutingPolicy(executed=[0, 4, 7, 2])
    engine = _engine(policy)
    engine.start()
    try:
        engine.resume()
        observe(engine)
        assert wait_for(lambda: len(policy.calls) == 5)  # the fifth waits for release

        assert list(engine._runtime.delays) == [4, 7, 2]
        # Conditioned on max(history so far): nothing, nothing yet, then 4, then 7 twice.
        assert [delay for _, delay in policy.calls] == [0, 0, 4, 7, 7]
        # The queue resumed at row 2 of chunk 4: the action 2 steps after its request.
        torch.testing.assert_close(engine.action_queue.get(), torch.tensor([4004.0, 4005.0]))
    finally:
        stop(engine)


def test_trained_rtc_loop_retries_a_chunk_whose_prefix_was_outrun():
    """Conditioned on 4+1 steps but 9 executed: the chunk is discarded and the retry conditions
    on the 9 it overlapped."""
    policy = ExecutingPolicy(executed=[0, 4, 9, 3], training_max_delay=10)
    engine = _engine(policy, RTCConfig(enabled=True, mode="trained", execution_horizon=10))
    engine.start()
    try:
        engine.resume()
        observe(engine)
        assert wait_for(lambda: len(policy.calls) == 5)

        # 1: unconditioned first chunk. 2: bootstraps on the checkpoint's capacity (10). 3: the
        # 4 executed during 2, plus a step of margin, outrun by 9. 4: the retry, 9 + 1.
        assert [delay for _, delay in policy.calls] == [0, 10, 5, 10, 10]
        assert list(engine._runtime.delays) == [4, 9, 3]
        # The discarded chunk never reached the robot: the queue holds chunk 4, 3 rows in.
        torch.testing.assert_close(engine.action_queue.get(), torch.tensor([4006.0, 4007.0]))
        assert not engine.failed
    finally:
        stop(engine)
