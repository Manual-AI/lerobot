import math
from types import SimpleNamespace

import pytest

pytest.importorskip("accelerate")
import torch

from lerobot.scripts.lerobot_train import _action_error_terms, resolve_eval_steps
from lerobot.utils.constants import ACTION


def _cfg(eval_steps=None):
    return SimpleNamespace(eval_steps=eval_steps)


def test_resolve_eval_steps():
    assert resolve_eval_steps(_cfg(), 1000, 120) == 120  # one epoch
    assert resolve_eval_steps(_cfg(), 10**6, 40_000) == 5000  # capped
    assert resolve_eval_steps(_cfg(), 1000, None) == 5000  # no epoch length
    assert resolve_eval_steps(_cfg(), 4, 1) == 0  # too few samples
    assert resolve_eval_steps(_cfg(0), 1000, 120) == 0
    assert resolve_eval_steps(_cfg(77), 1000, 120) == 77


def test_action_error_ignores_padded_dims_and_padded_steps():
    gt = torch.zeros(2, 3, 4)
    pred = torch.zeros(2, 3, 32)  # model output wider than the real action
    pred[..., :4] = 2.0
    pred[..., 4:] = 100.0  # must not count
    batch = {ACTION: gt, f"{ACTION}_is_pad": torch.tensor([[False, False, True], [False, True, True]])}
    sq, n = _action_error_terms(pred, batch)
    assert n.item() == 3 * 4  # 3 valid steps x 4 dims
    assert math.sqrt(sq.item() / n.item()) == pytest.approx(2.0)
