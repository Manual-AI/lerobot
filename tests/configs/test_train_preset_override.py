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
"""Explicit optimizer / scheduler settings override the policy presets individually."""

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import WSDSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig


def make_cfg(tmp_path, **overrides) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/dummy"),
        policy=ACTConfig(push_to_hub=False),
        output_dir=tmp_path / "out",
        **overrides,
    )


def test_presets_fill_both_when_nothing_is_set(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.validate()
    preset = ACTConfig(push_to_hub=False)
    assert cfg.optimizer == preset.get_optimizer_preset()
    assert cfg.scheduler == preset.get_scheduler_preset()


def test_explicit_scheduler_keeps_the_preset_optimizer(tmp_path):
    scheduler = WSDSchedulerConfig(num_warmup_steps=10)
    cfg = make_cfg(tmp_path, scheduler=scheduler)
    cfg.validate()
    assert cfg.scheduler is scheduler
    assert cfg.optimizer == ACTConfig(push_to_hub=False).get_optimizer_preset()


def test_explicit_optimizer_keeps_the_preset_scheduler(tmp_path):
    optimizer = AdamWConfig(lr=3e-4)
    cfg = make_cfg(tmp_path, optimizer=optimizer)
    cfg.validate()
    assert cfg.optimizer is optimizer
    assert cfg.scheduler == ACTConfig(push_to_hub=False).get_scheduler_preset()


def test_disabling_presets_still_requires_both(tmp_path):
    cfg = make_cfg(tmp_path, use_policy_training_preset=False, scheduler=WSDSchedulerConfig())
    with pytest.raises(ValueError, match="Optimizer and Scheduler must be set"):
        cfg.validate()
