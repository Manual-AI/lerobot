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

"""Static (AST) checks that ``cfg.pin_memory`` overrides both DataLoader constructors in lerobot_train.

Limitation: this inspects source only; it does not execute ``make_dataloader``/training.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "lerobot"
TRAIN_SCRIPT = ROOT / "scripts" / "lerobot_train.py"
TRAIN_CFG = ROOT / "configs" / "train.py"


def _pin_memory_exprs() -> list[str]:
    tree = ast.parse(TRAIN_SCRIPT.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.utils.data.DataLoader":
            for kw in node.keywords:
                if kw.arg == "pin_memory":
                    out.append(ast.unparse(kw.value))
    return out


def test_both_dataloaders_honor_override_else_cuda():
    exprs = _pin_memory_exprs()
    assert len(exprs) == 2, exprs  # train + eval constructors
    for e in exprs:
        assert e == "cfg.pin_memory if cfg.pin_memory is not None else device_type == 'cuda'"

    # Evaluate the expression semantically for all inputs.
    for override, device, expected in [
        (None, "cuda", True),
        (None, "cpu", False),
        (True, "cpu", True),
        (False, "cuda", False),
    ]:
        cfg = type("C", (), {"pin_memory": override})()
        assert eval(exprs[0], {"cfg": cfg, "device_type": device}) is expected  # noqa: S307


def test_train_config_pin_memory_defaults_to_none():
    tree = ast.parse(TRAIN_CFG.read_text())
    fields = {
        n.target.id: n
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "TrainPipelineConfig"
        for n in c.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    f = fields["pin_memory"]
    assert ast.unparse(f.annotation) == "bool | None"
    assert ast.literal_eval(f.value) is None


def test_real_config_default_if_importable():
    train = pytest.importorskip("lerobot.configs.train")
    names = {f.name: f for f in dataclasses.fields(train.TrainPipelineConfig)}
    assert names["pin_memory"].default is None
