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

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

pytest.importorskip("datasets")
pytest.importorskip("av")

from datasets import Dataset, Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.dataset_reader import DatasetReader  # noqa: E402
from lerobot.datasets.io_utils import hf_transform_to_torch  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.relative_stats import (  # noqa: E402
    compute_act_relative_action_stats,
    resolve_act_training_processor_inputs,
)
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.processor import NormalizerProcessorStep  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def _pose(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return [x, y, 0, c, -s, 0, s, c, 0]


def _homogeneous(pose):
    matrix = np.eye(4)
    matrix[:2, :3] = np.asarray(pose[3:]).reshape(2, 3)
    matrix[2, :3] = np.cross(matrix[0, :3], matrix[1, :3])
    matrix[:3, 3] = pose[:3]
    return matrix


class _Dataset:
    def __init__(self, reader):
        self.reader = reader
        self.meta = reader._meta
        self.episodes = reader.episodes
        self.absolute_to_relative_idx = reader.absolute_to_relative_idx

    def __len__(self):
        return self.reader.num_frames

    def _ensure_reader(self):
        return self.reader


def _dataset(tmp_path, config, *, renamed=False):
    action_key = "recorded.action" if renamed else ACTION
    state_key = "observation.pose" if renamed else OBS_STATE
    # Middle episode is held out; its extreme targets must not affect statistics.
    # Selected episodes are non-contiguous and one is shorter than chunk_size.
    episodes = Dataset.from_dict({"dataset_from_index": [0, 3, 5], "dataset_to_index": [3, 5, 6]})
    actions, states = [], []
    for i in range(6):
        offset = 10000 if i in (3, 4) else 0
        states.append(_pose(10 + i, 20, np.pi / 2) + _pose(-i, 2, 0) + [i, 2 * i, 99])
        actions.append(
            _pose(11 + i + offset, 23 + 2 * i, np.pi / 2 + i * np.pi / 4)
            + _pose(2 + i, 4, -i * np.pi / 4)
            + [3 + i, 7 - i]
        )
    meta = SimpleNamespace(
        episodes=episodes,
        total_episodes=3,
        total_frames=6,
        fps=10,
        features={action_key: {}, state_key: {}, "observation.images.bad": {"dtype": "image"}},
        depth_keys=[],
        image_keys=["observation.images.bad"],
        camera_keys=["observation.images.bad"],
        video_keys=[],
        stats={
            action_key: {"mean": torch.full((20,), 100.0), "std": torch.full((20,), 50.0)},
            state_key: {"mean": torch.arange(21).float(), "std": torch.full((21,), 2.0)},
            "observation.images.bad": {"mean": torch.ones(3, 1, 1), "std": torch.ones(3, 1, 1)},
        },
    )
    reader = DatasetReader(
        meta=meta,
        root=tmp_path,
        episodes=[0, 2],
        tolerance_s=1e-4,
        video_backend="pyav",
        delta_timestamps={action_key: [i / 10 for i in config.action_delta_indices]},
        image_transforms=None,
    )
    table = Dataset.from_dict(
        {
            "index": list(range(6)),
            "episode_index": [0, 0, 0, 1, 1, 2],
            action_key: actions,
            state_key: states,
            # Any accidental full-row/image access fails during decoding.
            "observation.images.bad": [{"bytes": b"invalid image", "path": None}] * 6,
        }
    ).cast_column("observation.images.bad", Image())
    reader.hf_dataset = table.select([0, 1, 2, 5]).flatten_indices().with_transform(hf_transform_to_torch)
    reader._build_index_mapping()
    return _Dataset(reader), np.asarray(actions), np.asarray(states)


def _config(**kwargs):
    return ACTConfig(
        chunk_size=3,
        n_action_steps=3,
        relative_actions=True,
        relative_eef_starts=[0, 9],
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (21,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (20,))},
        **kwargs,
    )


def _expected_targets(config, actions, states):
    # Independent SE(3) oracle, using a general matrix inverse rather than the
    # production rot6d/anchored_delta helpers. Never include padded or held-out rows.
    targets = []
    # ACTConfig.drop_n_last_frames excludes anchors whose target window would run past
    # the episode end, so the sampler never yields them and they are not expected here.
    drop = getattr(config, "drop_n_last_frames", 0)
    for episode in ([0, 1, 2], [5]):
        for position, frame in enumerate(episode[: len(episode) - drop]):
            anchor = actions[frame] if config.relative_anchor == "first_action" else states[frame, :20]
            for target in episode[position : position + config.chunk_size]:
                result = actions[target].copy()
                if config.relative_scalars:
                    result -= anchor
                for start in (0, 9):
                    delta = np.linalg.inv(_homogeneous(anchor[start : start + 9])) @ _homogeneous(
                        actions[target, start : start + 9]
                    )
                    result[start : start + 9] = np.concatenate([delta[:3, 3], delta[:2, :3].reshape(-1)])
                targets.append(result)
    return np.stack(targets)


@pytest.mark.parametrize("relative_anchor", ["state", "first_action"])
@pytest.mark.parametrize("relative_scalars", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_stats_match_independent_se3_oracle_and_selected_unpadded_training_targets(
    tmp_path, relative_anchor, relative_scalars, batch_size
):
    config = _config(relative_anchor=relative_anchor, relative_scalars=relative_scalars)
    dataset, actions, states = _dataset(tmp_path, config)
    stats = compute_act_relative_action_stats(config, dataset, batch_size=batch_size)
    targets = _expected_targets(config, actions, states)

    assert stats["count"].item() == targets.shape[0]
    for name, reducer in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        assert stats[name].shape == (20,)  # same stats across all chunk positions
        np.testing.assert_allclose(stats[name], reducer(targets, axis=0), atol=2e-6, rtol=1e-6)
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_renamed_columns_and_metadata_preservation(tmp_path):
    config = _config(relative_scalars=False)
    dataset, actions, states = _dataset(tmp_path, config, renamed=True)
    before = deepcopy(dataset.meta.stats)
    path, stats = resolve_act_training_processor_inputs(
        config,
        dataset,
        resume=False,
        rename_map={"recorded.action": ACTION, "observation.pose": OBS_STATE},
    )
    assert path is None
    np.testing.assert_allclose(
        stats[ACTION]["mean"], _expected_targets(config, actions, states).mean(0), atol=1e-6
    )
    torch.testing.assert_close(stats[OBS_STATE]["mean"], before["observation.pose"]["mean"])
    torch.testing.assert_close(
        stats["observation.images.bad"]["mean"], before["observation.images.bad"]["mean"]
    )
    for key, values in before.items():
        for name, value in values.items():
            torch.testing.assert_close(dataset.meta.stats[key][name], value)


@pytest.mark.parametrize("resume,relative_actions", [(True, True), (True, False), (False, False)])
def test_resume_and_absolute_act_do_not_scan_dataset(tmp_path, resume, relative_actions):
    config = _config()
    config.relative_actions = relative_actions
    config.pretrained_path = tmp_path / "checkpoint"
    # Deliberately no reader or length: these paths must not inspect training data.
    dataset = SimpleNamespace(meta=SimpleNamespace(stats={ACTION: {"mean": torch.tensor([123.0])}}))
    path, stats = resolve_act_training_processor_inputs(config, dataset, resume=resume, rename_map={})
    assert path == config.pretrained_path
    torch.testing.assert_close(stats[ACTION]["mean"], torch.tensor([123.0]))


@pytest.mark.parametrize("relative_anchor", ["state", "first_action"])
def test_finetuning_rebuilds_stats_and_checkpoint_restores_normalization(tmp_path, relative_anchor):
    config = _config(relative_anchor=relative_anchor, relative_scalars=False)
    dataset, actions, states = _dataset(tmp_path, config)
    config.pretrained_path = tmp_path / "old-checkpoint"
    path, stats = resolve_act_training_processor_inputs(config, dataset, resume=False, rename_map={})
    assert path is None  # do not load the old representation or absolute dataset statistics
    pre, post = make_pre_post_processors(config, pretrained_path=path, dataset_stats=stats)

    batch = {
        OBS_STATE: torch.tensor(states[0:1], dtype=torch.float32),
        ACTION: torch.tensor(actions[None, :3], dtype=torch.float32),
    }
    encoded = pre(batch)[ACTION]
    expected = torch.tensor(_expected_targets(config, actions, states)[:3], dtype=torch.float32)
    normalizer = next(step for step in pre.steps if isinstance(step, NormalizerProcessorStep))
    torch.testing.assert_close(
        encoded[0], (expected - stats[ACTION]["mean"]) / (stats[ACTION]["std"] + normalizer.eps)
    )
    decoded = post(encoded)

    saved = tmp_path / "saved"
    pre.save_pretrained(saved)
    post.save_pretrained(saved)
    # Both JSON files must point at serialized action statistics, with the exact
    # training values in the actual tensor artifacts (not just in memory).
    for name in ("policy_preprocessor", "policy_postprocessor"):
        manifest = json.loads((saved / f"{name}.json").read_text())
        state_file = next(step["state_file"] for step in manifest["steps"] if "state_file" in step)
        tensors = load_file(saved / state_file)
        torch.testing.assert_close(tensors["action.mean"], stats[ACTION]["mean"])
        torch.testing.assert_close(tensors["action.std"], stats[ACTION]["std"])

    # Runtime dataset statistics are intentionally different. Checkpoint loading
    # must use saved processor state, as it does during inference and resume.
    loaded_pre, loaded_post = make_pre_post_processors(
        config, pretrained_path=saved, dataset_stats=dataset.meta.stats
    )
    torch.testing.assert_close(loaded_pre(batch)[ACTION], encoded)
    loaded_pre({OBS_STATE: batch[OBS_STATE]})  # inference has no action targets or dataset
    torch.testing.assert_close(loaded_post(encoded), decoded)
    if relative_anchor == "state":
        torch.testing.assert_close(decoded, batch[ACTION], atol=1e-5, rtol=1e-5)
    else:
        # first_action deliberately reconstructs onto measured state, not action[0].
        torch.testing.assert_close(decoded[0, 0, :18], batch[OBS_STATE][0, :18], atol=1e-5, rtol=1e-5)


def test_new_run_recomputes_after_representation_changes(tmp_path):
    config = _config()
    dataset, _, _ = _dataset(tmp_path, config)
    _, first = resolve_act_training_processor_inputs(config, dataset, resume=False, rename_map={})
    config.relative_anchor = "first_action"
    config.relative_scalars = False
    _, second = resolve_act_training_processor_inputs(config, dataset, resume=False, rename_map={})
    assert not torch.allclose(first[ACTION]["mean"], second[ACTION]["mean"])


def test_streaming_requires_saved_stats_on_resume(tmp_path):
    class Stream(IterableDataset):
        meta = SimpleNamespace(stats={})

        def __iter__(self):
            raise AssertionError("Statistics must not consume the stream")

    config = _config()
    with pytest.raises(ValueError, match="disable streaming"):
        resolve_act_training_processor_inputs(config, Stream(), resume=False, rename_map={})
    config.pretrained_path = tmp_path / "saved"
    path, _ = resolve_act_training_processor_inputs(config, Stream(), resume=True, rename_map={})
    assert path == config.pretrained_path


def test_generic_reader_excludes_padding_and_rejects_nonfinite_targets(tmp_path):
    config = ACTConfig(chunk_size=3, n_action_steps=3, relative_actions=True, device="cpu")
    meta = SimpleNamespace(
        # Long enough to keep one anchor after drop_n_last_frames (chunk_size - 1 = 2);
        # the stub reader returns the same padded item for whichever anchor is drawn.
        episodes={"dataset_from_index": [0], "dataset_to_index": [3]},
        features={ACTION: {}, OBS_STATE: {}},
    )
    item = {
        ACTION: torch.tensor([[11.0], [13.0], [float("nan")]]),
        OBS_STATE: torch.tensor([10.0]),
        "action_is_pad": torch.tensor([False, False, True]),
    }
    reader = SimpleNamespace(
        _meta=meta,
        episodes=None,
        absolute_to_relative_idx=None,
        num_frames=3,
        get_items=lambda indices: [item for _ in indices],
    )
    dataset = _Dataset(reader)
    stats = compute_act_relative_action_stats(config, dataset)
    torch.testing.assert_close(stats["mean"], torch.tensor([2.0]))
    torch.testing.assert_close(stats["std"], torch.tensor([1.0]))
    item["action_is_pad"][-1] = False
    with pytest.raises(ValueError, match="non-finite"):
        compute_act_relative_action_stats(config, dataset)
    reader.num_frames = 0
    with pytest.raises(ValueError, match="empty training dataset"):
        compute_act_relative_action_stats(config, dataset)


@pytest.mark.parametrize("resume", [False, True])
def test_training_entrypoint_rebuilds_for_finetuning_but_preserves_resume_stats(
    tmp_path, monkeypatch, resume
):
    import lerobot.scripts.lerobot_train as train_module

    config = _config()
    dataset, actions, states = _dataset(tmp_path, config)
    source_config = deepcopy(config)
    source_config.relative_actions = resume  # fresh fine-tuning may start from absolute ACT
    saved = tmp_path / "old-checkpoint"
    old_pre, old_post = make_pre_post_processors(source_config, dataset_stats=dataset.meta.stats)
    old_pre.save_pretrained(saved)
    old_post.save_pretrained(saved)
    saved_mean = dataset.meta.stats[ACTION]["mean"].clone()
    config.pretrained_path = saved
    dataset.meta.stats[ACTION]["mean"] = saved_mean + 1000
    if resume:
        monkeypatch.setattr(dataset, "_ensure_reader", lambda: pytest.fail("Resume rescanned the dataset"))

    # Exercise the real training entrypoint through processor construction. Replace
    # distributed/model setup only; stop before optimizer/model training is needed.
    cfg = SimpleNamespace(
        job=SimpleNamespace(is_remote=False),
        validate=lambda: None,
        to_dict=lambda: {},
        parallelism=None,
        wandb=SimpleNamespace(enable=False),
        seed=None,
        cudnn_deterministic=False,
        resume=resume,
        checkpoint_format=SimpleNamespace(wants_dcp=False),
        is_reward_model_training=False,
        policy=config,
        trainable_config=config,
        peft=None,
        rename_map={},
    )
    accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), wait_for_everyone=lambda: None)
    monkeypatch.setattr(train_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "make_accelerator", lambda cfg: accelerator)
    monkeypatch.setattr(train_module, "ParallelDims", SimpleNamespace(from_config=lambda *args: None))
    monkeypatch.setattr(train_module, "init_logging", lambda **kwargs: None)
    monkeypatch.setattr(train_module, "is_main_process", lambda: True)
    monkeypatch.setattr(train_module, "make_train_eval_datasets", lambda cfg: (dataset, None))
    monkeypatch.setattr(train_module, "make_policy", lambda **kwargs: SimpleNamespace(config=config))
    for obj, name in [
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cuda.matmul, "allow_tf32"),
    ]:
        monkeypatch.setattr(obj, name, getattr(obj, name))
    captured = []

    def capture_processors(**kwargs):
        processors = make_pre_post_processors(**kwargs)
        captured.append(processors)
        return processors

    class ProcessorsReadyError(Exception):
        pass

    def stop_before_optimizer(*args):
        raise ProcessorsReadyError

    monkeypatch.setattr(train_module, "make_pre_post_processors", capture_processors)
    monkeypatch.setattr(train_module, "make_optimizer_and_scheduler", stop_before_optimizer)
    with pytest.raises(ProcessorsReadyError):
        train_module.train.__wrapped__(cfg)

    pre, post = captured[0]
    expected_mean = (
        saved_mean
        if resume
        else torch.tensor(_expected_targets(config, actions, states).mean(0), dtype=torch.float32)
    )
    for pipeline in (pre, post):
        state = next(step.state_dict() for step in pipeline.steps if "action.mean" in step.state_dict())
        torch.testing.assert_close(state["action.mean"], expected_mean, atol=2e-6, rtol=1e-6)
    # The fresh path must rebuild relative steps even though the source checkpoint
    # had an absolute-action pipeline.
    assert any(type(step).__name__ == "AnchoredRelativeEEFStep" for step in pre.steps)


@pytest.mark.parametrize("change", ["dataset", "chunk_size", "eef_layout"])
def test_new_runs_do_not_reuse_stats_after_data_or_sampling_changes(tmp_path, change):
    config = _config()
    dataset, _, _ = _dataset(tmp_path, config)
    _, before = resolve_act_training_processor_inputs(config, dataset, resume=False, rename_map={})
    if change == "chunk_size":
        config.chunk_size = config.n_action_steps = 2
        dataset.reader.delta_indices[ACTION] = [0, 1]
    elif change == "eef_layout":
        config.relative_eef_starts = [0]
    else:
        table = dataset.reader.hf_dataset
        actions = torch.stack(table.select_columns(ACTION)[:][ACTION])
        actions[:, 18] += 17
        dataset.reader.hf_dataset = table.remove_columns(ACTION).add_column(ACTION, actions.tolist())
    _, after = resolve_act_training_processor_inputs(config, dataset, resume=False, rename_map={})
    assert not torch.allclose(before[ACTION]["mean"], after[ACTION]["mean"])


def test_statistics_reject_a_different_dataset_chunk_length(tmp_path):
    config = _config()
    dataset, _, _ = _dataset(tmp_path, config)
    dataset.reader.delta_indices[ACTION] = [0, 1]
    with pytest.raises(ValueError, match="same action chunk_size"):
        compute_act_relative_action_stats(config, dataset)
