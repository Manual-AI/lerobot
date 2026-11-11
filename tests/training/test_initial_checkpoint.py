"""CPU tests of real save orchestration extracted without optional trainer imports.

Policy/processor serializers and Accelerator are explicit fakes with real CPU
safetensors. No ACT forward, backward, dataset loading or GPU qualification.
"""

import ast
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2] / "src/lerobot"
FORMATS = SimpleNamespace(SAFETENSORS="safetensors", SAFETENSORS_DCP="both", DCP="dcp")


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def api(monkeypatch):
    # Never query/init GPU or MPS in this CPU suite.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    ns = {
        "torch": torch,
        "np": np,
        "random": random,
        "Path": Path,
        "os": os,
        "hashlib": hashlib,
        "json": json,
        "CheckpointFormat": FORMATS,
        "is_sharded_module": lambda _: False,
        "is_main_process": lambda: True,
        "CHECKPOINTS_DIR": "checkpoints",
        "PRETRAINED_MODEL_DIR": "pretrained_model",
        "TRAINING_STATE_DIR": "training_state",
        "TRAINING_STEP": "training_step.json",
        "RNG_STATE": "rng_state.safetensors",
        "save_file": save_file,
        "OPTIMIZER_STATE": "optimizer_state.safetensors",
        "OPTIMIZER_PARAM_GROUPS": "optimizer_param_groups.json",
        "write_json": lambda d, p: p.write_text(json.dumps(d)),
    }
    extract(ROOT / "utils/utils.py", ["flatten_dict"], ns)
    extract(ROOT / "optim/optimizers.py", ["save_optimizer_state", "_save_single_optimizer_state"], ns)
    extract(
        ROOT / "utils/random_utils.py",
        [
            "get_rng_state",
            "set_rng_state",
            "save_rng_state",
            "serialize_rng_state",
            "serialize_python_rng_state",
            "serialize_numpy_rng_state",
            "serialize_torch_rng_state",
        ],
        ns,
    )
    extract(
        ROOT / "common/train_utils.py",
        [
            "save_initial_checkpoint",
            "save_checkpoint",
            "save_training_state",
            "save_training_metadata",
            "get_step_checkpoint_dir",
            "get_step_identifier",
        ],
        ns,
    )
    extract(ROOT / "configs/train.py", ["validate_initial_checkpoint"], ns)
    return ns


def config(tmp_path, api):
    c = SimpleNamespace(
        save_initial_checkpoint=True,
        resume=False,
        reward_model=None,
        policy=SimpleNamespace(
            type="act",
            pretrained_path=None,
            pretrained_backbone_weights=None,
            use_peft=False,
            push_to_hub=False,
        ),
        peft=None,
        ema=SimpleNamespace(enable=False),
        save_checkpoint=True,
        save_checkpoint_to_hub=False,
        seed=7,
        steps=100,
        batch_size=2,
        save_freq=25,
        checkpoint_format="safetensors",
        output_dir=tmp_path,
        accelerator=SimpleNamespace(
            mixed_precision="no",
            gradient_accumulation=SimpleNamespace(steps=1),
            compile=SimpleNamespace(enabled=False),
            activation_checkpointing=SimpleNamespace(mode="none"),
        ),
        parallelism=SimpleNamespace(
            dp_replicate=1,
            dp_shard=1,
            cp_size=1,
            cfg_parallel=1,
            dp_world_size=1,
            context_parallel=SimpleNamespace(ring_degree=1, ulysses_degree=1),
        ),
    )
    c.validate_initial_checkpoint = lambda: api["validate_initial_checkpoint"](c)
    c.save_pretrained = lambda p: (p / "train_config.json").write_text('{"save_initial_checkpoint":true}')
    return c


class TinyLivePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def save_pretrained(self, p):
        p.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), p / "model.safetensors")
        (p / "config.json").write_text('{"type":"act","test_fake":true}')
        # Deliberately consume RNG to verify export restores the entry state.
        random.random()
        np.random.rand()
        torch.rand(1)


class Processor:
    def __init__(self, name):
        self.name = name
        self.mean = torch.tensor([3.0, 5.0])

    def save_pretrained(self, p):
        (p / f"policy_{self.name}.json").write_text('{"test_fake":true}')
        save_file({"mean": self.mean}, p / f"policy_{self.name}.safetensors")


def objects():
    policy = TinyLivePolicy()
    optimizer = torch.optim.AdamW(policy.parameters())  # no step/backward
    pre, post = Processor("preprocessor"), Processor("postprocessor")
    acc = SimpleNamespace(
        num_processes=1,
        distributed_type="NO",
        mixed_precision="no",
        gradient_accumulation_steps=1,
        device=torch.device("cpu"),
        unwrap_model=lambda p: p,
        scaler=None,
        wait_for_everyone=lambda: None,
    )
    return policy, optimizer, None, pre, post, acc, 0


def assert_rng_equal(a, b):
    assert a["random_state"] == b["random_state"]
    assert np.array_equal(a["numpy_random_state"][1], b["numpy_random_state"][1])
    assert a["numpy_random_state"][2:] == b["numpy_random_state"][2:]
    assert torch.equal(a["torch_random_state"], b["torch_random_state"])


def test_live_checkpoint_and_entry_rng_exact(api, tmp_path):
    cfg, args = config(tmp_path, api), objects()
    before = api["get_rng_state"]()
    expected_rng = api["serialize_rng_state"]()
    original = {k: v.clone() for k, v in args[0].state_dict().items()}
    dest = api["save_initial_checkpoint"](cfg, *args)
    assert_rng_equal(before, api["get_rng_state"]())
    loaded = load_file(dest / "pretrained_model/model.safetensors")
    assert loaded.keys() == original.keys()
    assert all(
        torch.equal(loaded[k], v) and torch.equal(args[0].state_dict()[k], v) for k, v in original.items()
    )
    for processor in args[3:5]:
        assert torch.equal(
            load_file(dest / f"pretrained_model/policy_{processor.name}.safetensors")["mean"], processor.mean
        )
    saved_rng = load_file(dest / "training_state/rng_state.safetensors")
    assert saved_rng.keys() == expected_rng.keys()
    assert all(torch.equal(saved_rng[k], v) for k, v in expected_rng.items())
    assert json.loads((dest / "training_state/training_step.json").read_text())["step"] == 0
    assert load_file(dest / "training_state/optimizer_state.safetensors") == {}
    assert (
        json.loads((dest / "training_state/optimizer_param_groups.json").read_text())[0]["lr"]
        == args[1].param_groups[0]["lr"]
    )
    manifest = json.loads((dest / "initial_checkpoint.json").read_text())
    assert manifest["optimizer_updates"] == 0
    assert all(hashlib.sha256((dest / k).read_bytes()).hexdigest() == h for k, h in manifest["files"].items())
    assert not (tmp_path / "checkpoints/last").exists()
    assert all(p.grad is None for p in args[0].parameters())


def test_failed_export_restores_rng_and_cannot_retry(api, tmp_path):
    cfg, args = config(tmp_path, api), objects()
    original = api["save_checkpoint"]
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        original(**kwargs)
        raise OSError("disk failure")

    api["save_checkpoint"] = fail
    before = api["get_rng_state"]()
    with pytest.raises(OSError, match="disk failure"):
        api["save_initial_checkpoint"](cfg, *args)
    assert_rng_equal(before, api["get_rng_state"]())
    assert len(calls) == 1 and calls[0]["policy"] is args[0] and calls[0]["preprocessor"] is args[3]
    assert not (tmp_path / "checkpoints/000000/initial_checkpoint.json").exists()
    with pytest.raises(FileExistsError):
        api["save_initial_checkpoint"](cfg, *args)
    assert len(calls) == 1


def test_default_noop_and_existing_path_refusal(api, tmp_path):
    cfg, args = config(tmp_path, api), objects()
    cfg.save_initial_checkpoint = False
    assert api["save_initial_checkpoint"](cfg, *args) is None
    assert not list(tmp_path.iterdir())
    cfg.save_initial_checkpoint = True
    target = tmp_path / "checkpoints/000000"
    target.parent.mkdir()
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        api["save_initial_checkpoint"](cfg, *args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resume", True),
        ("seed", None),
        ("steps", 0),
        ("batch_size", 0),
        ("save_freq", 0),
        ("save_checkpoint", False),
        ("peft", object()),
        ("ema.enable", True),
        ("policy.type", "diffusion"),
        ("policy.pretrained_path", "old"),
        ("policy.pretrained_backbone_weights", "DEFAULT"),
        ("policy.use_peft", True),
        ("policy.push_to_hub", True),
        ("checkpoint_format", "dcp"),
        ("accelerator.mixed_precision", "bf16"),
        ("accelerator.gradient_accumulation.steps", 2),
        ("accelerator.compile.enabled", True),
        ("accelerator.activation_checkpointing.mode", "full"),
        ("parallelism.dp_shard", 2),
        ("parallelism.dp_replicate", 2),
    ],
)
def test_unsupported_config_rejected(api, tmp_path, field, value):
    cfg = config(tmp_path, api)
    obj = cfg
    parts = field.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *objects())
    assert not list(tmp_path.iterdir())


def test_runtime_and_world_size_rejection(api, tmp_path, monkeypatch):
    cfg, args = config(tmp_path, api), objects()
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    args[-2].num_processes = 2
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *args)
    args[-2].num_processes = 1
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *args[:-1], 1)


def test_trainer_calls_once_after_prepare_before_iterator_or_update():
    tree = ast.parse((ROOT / "scripts/lerobot_train.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)]

    def lines(name):
        return [n.lineno for n in calls if ast.unparse(n.func) == name]

    (hook,) = lines("save_initial_checkpoint")
    assert max(lines("accelerator.prepare") + lines("resume_after_prepare")) < hook
    assert hook < min(lines("cycle") + lines("update_policy"))
    # The call is an unconditional top-level expression: export errors propagate.
    assert any(
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.lineno == hook
        for n in train.body
    )
    cfg_tree = ast.parse((ROOT / "configs/train.py").read_text())
    field = next(
        n
        for n in ast.walk(cfg_tree)
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "save_initial_checkpoint"
    )
    assert ast.literal_eval(field.value) is False


def test_all_initialized_cuda_rng_states_restored_without_cuda(api, tmp_path, monkeypatch):
    """Exercise the all-device branch using CPU stand-ins, never a CUDA query."""
    states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [s.clone() for s in states])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: restored.append(values))
    api["save_initial_checkpoint"](config(tmp_path, api), *objects())
    assert len(restored) == 2
    assert all(torch.equal(a, b) for restore in restored for a, b in zip(restore, states, strict=True))


def test_runtime_distributed_and_compiled_rejected(api, tmp_path):
    cfg, args = config(tmp_path, api), objects()
    args[-2].distributed_type = "DEEPSPEED"
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *args)
    args[-2].distributed_type = "NO"
    args[0]._orig_mod = object()
    with pytest.raises(ValueError):
        api["save_initial_checkpoint"](cfg, *args)
    assert not list(tmp_path.iterdir())
