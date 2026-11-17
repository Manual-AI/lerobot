# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
    WSDSchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import is_package_available


@pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
def test_diffuser_scheduler(optimizer):
    config = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()  # so that we don't get torch warning
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0002],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_vqbet_scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_cosine_decay_with_warmup_scheduler(optimizer):
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001818181818181819],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()


def _wsd_lrs(optimizer, steps, **kwargs):
    scheduler = WSDSchedulerConfig(**kwargs).build(optimizer, num_training_steps=steps)
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_wsd_scheduler_phases(optimizer):
    peak = optimizer.param_groups[0]["lr"]
    lrs = _wsd_lrs(optimizer, 100, num_warmup_steps=10, decay_fraction=0.2)
    assert lrs[0] == 0.0 and lrs[10] == pytest.approx(peak)  # warmup reaches the peak
    assert all(lr == pytest.approx(peak) for lr in lrs[10:80])  # stable phase
    assert lrs[80] == pytest.approx(peak) and lrs[99] < peak * 0.1  # decay to ~0
    assert all(a >= b for a, b in zip(lrs[80:], lrs[81:], strict=False))


def test_wsd_scheduler_cosine_and_floor(optimizer):
    peak = optimizer.param_groups[0]["lr"]
    lrs = _wsd_lrs(
        optimizer, 100, num_warmup_steps=0, decay_fraction=0.5, decay_type="cosine", final_lr_ratio=0.1
    )
    assert lrs[49] == pytest.approx(peak)
    assert lrs[75] == pytest.approx(peak * 0.55, rel=0.05)  # halfway: floor + (1-floor)/2
    assert min(lrs) >= peak * 0.1 - 1e-12


def test_wsd_scheduler_explicit_decay_steps_and_scaling(optimizer):
    peak = optimizer.param_groups[0]["lr"]
    lrs = _wsd_lrs(optimizer, 50, num_warmup_steps=10, num_decay_steps=10)
    assert lrs[39] == pytest.approx(peak) and lrs[49] < peak * 0.2
    # warmup + decay longer than the run: scaled down, still warms up and decays
    lrs = _wsd_lrs(optimizer, 20, num_warmup_steps=100, num_decay_steps=100)
    assert max(lrs) == pytest.approx(peak)
    assert lrs[-1] < max(lrs)


def test_wsd_scheduler_registered_and_validated():
    assert WSDSchedulerConfig().type == "wsd"
    with pytest.raises(ValueError, match="decay_type"):
        WSDSchedulerConfig(decay_type="exp")
    with pytest.raises(ValueError, match="decay_fraction"):
        WSDSchedulerConfig(decay_fraction=1.5)
