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

"""ACT future-action window construction and episode-tail filtering."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.scripts.lerobot_train import make_dataloaders  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


def _config(*, offset: int = 0, chunk_size: int = 4) -> ACTConfig:
    return ACTConfig(
        device="cpu",
        push_to_hub=False,
        action_delta_offset=offset,
        chunk_size=chunk_size,
        n_action_steps=1,
    )


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_action_delta_offset_selects_a_full_chunk(offset):
    config = _config(offset=offset, chunk_size=4)

    assert config.action_delta_indices == list(range(offset, offset + 4))
    assert config.drop_n_last_frames == offset + 3


def test_single_step_offset_zero_keeps_the_last_episode_row():
    config = _config(offset=0, chunk_size=1)

    assert config.action_delta_indices == [0]
    assert config.drop_n_last_frames == 0
    assert list(EpisodeAwareSampler([0], [1], drop_n_last_frames=config.drop_n_last_frames)) == [0]


@pytest.mark.parametrize("offset", [-1, True, 1.5, "1"])
def test_action_delta_offset_rejects_negative_and_non_integer_values(offset):
    with pytest.raises(ValueError, match="action_delta_offset"):
        _config(offset=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [0, 1, 3])
@pytest.mark.parametrize("extra_rows", [-1, 0, 1])
def test_sampler_admits_exactly_complete_future_windows(offset, extra_rows):
    """The final admitted anchor ends at the final same-episode action row."""
    chunk_size = 4
    episode_length = offset + chunk_size + extra_rows
    config = _config(offset=offset, chunk_size=chunk_size)
    expected_count = max(0, episode_length - offset - chunk_size + 1)

    if expected_count == 0:
        with pytest.raises(ValueError, match="No valid frames remain"):
            EpisodeAwareSampler([0], [episode_length], drop_n_last_frames=config.drop_n_last_frames)
        return

    sampler = EpisodeAwareSampler([0], [episode_length], drop_n_last_frames=config.drop_n_last_frames)
    anchors = list(sampler)
    assert anchors == list(range(expected_count))
    assert anchors[0] + config.action_delta_indices[0] == offset
    assert anchors[-1] + config.action_delta_indices[-1] == episode_length - 1
    assert all(anchor + index < episode_length for anchor in anchors for index in config.action_delta_indices)


def test_sampler_never_crosses_episode_boundaries_or_admits_padded_targets():
    """Sentinel episode ranges prove all requested target rows stay in their anchor episode."""
    config = _config(offset=3, chunk_size=4)
    starts, ends = [0, 8], [8, 16]
    sampler = EpisodeAwareSampler(starts, ends, drop_n_last_frames=config.drop_n_last_frames)

    anchors = sampler.indices
    assert anchors == [0, 1, 8, 9]
    for anchor in anchors:
        episode_end = 8 if anchor < 8 else 16
        targets = [anchor + index for index in config.action_delta_indices]
        assert targets[-1] < episode_end
        assert all(target // 8 == anchor // 8 for target in targets)


def test_dataset_delta_timestamps_and_dataloader_sampler_share_the_same_geometry():
    """Exercise the factory and training path, not just the config properties."""
    config = _config(offset=1, chunk_size=3)
    meta = SimpleNamespace(fps=20, features={ACTION: object()})
    assert resolve_delta_timestamps(config, meta) == {ACTION: [0.05, 0.1, 0.15]}

    class Dataset(torch.utils.data.Dataset):
        meta = SimpleNamespace(
            episodes={"dataset_from_index": [0, 5], "dataset_to_index": [5, 11]},
            has_language_columns=False,
        )
        episodes = None
        absolute_to_relative_idx = None

        def __len__(self):
            return 11

        def __getitem__(self, index):
            return {ACTION: torch.tensor(index)}

    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/act-action-window"),
        policy=config,
        batch_size=1,
        num_workers=0,
    )
    dataloader, _ = make_dataloaders(
        train_cfg,
        Dataset(),
        eval_dataset=None,
        step=0,
        parallel_dims=SimpleNamespace(device_type="cpu", dp_world_size=1),
    )
    sampler = dataloader.sampler
    assert isinstance(sampler, EpisodeAwareSampler)
    assert sampler.indices == [0, 1, 5, 6, 7]
    for anchor in sampler.indices:
        episode_end = 5 if anchor < 5 else 11
        assert anchor + config.action_delta_indices[-1] < episode_end


def test_real_dataset_offset_windows_are_complete_and_never_cross_episodes(
    tmp_path, empty_lerobot_dataset_factory
):
    """Exercise ACT's offsets through LeRobotDataset's real query and padding path."""
    config = _config(offset=1, chunk_size=3)
    source = empty_lerobot_dataset_factory(
        root=tmp_path / "act-offset-windows",
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
            ACTION: {"dtype": "float32", "shape": (1,), "names": ["action"]},
        },
        use_videos=False,
        fps=10,
    )
    episode_length = 7
    for episode_index in range(2):
        for frame_index in range(episode_length):
            sentinel = episode_index * 100 + frame_index
            source.add_frame(
                {
                    "observation.state": torch.tensor([sentinel], dtype=torch.float32),
                    ACTION: torch.tensor([sentinel], dtype=torch.float32),
                    "task": f"episode-{episode_index}",
                }
            )
        source.save_episode()
    source.finalize()

    dataset = LeRobotDataset(
        source.repo_id,
        root=source.root,
        delta_timestamps=resolve_delta_timestamps(config, source.meta),
        tolerance_s=0.04,
    )
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_last_frames=config.drop_n_last_frames,
    )

    # Each seven-row episode has anchors 0..3: the last target uses its own row 6.
    assert sampler.indices == [0, 1, 2, 3, 7, 8, 9, 10]
    for anchor in sampler.indices:
        episode_index = anchor // episode_length
        frame_index = anchor % episode_length
        sample = dataset[anchor]
        expected_actions = [episode_index * 100 + frame_index + delta for delta in (1, 2, 3)]

        assert sample[ACTION].squeeze(-1).tolist() == expected_actions
        assert sample[f"{ACTION}_is_pad"].tolist() == [False, False, False]
        assert sample["episode_index"].item() == episode_index

    # Explicit first/last checks make the target-row contract and terminal inclusion clear.
    assert dataset[sampler.indices[0]][ACTION].squeeze(-1).tolist() == [1.0, 2.0, 3.0]
    assert dataset[sampler.indices[3]][ACTION].squeeze(-1).tolist() == [4.0, 5.0, 6.0]
    assert dataset[sampler.indices[-1]][ACTION].squeeze(-1).tolist() == [104.0, 105.0, 106.0]


def test_action_delta_offset_serializes_and_loads(tmp_path):
    config = _config(offset=3, chunk_size=7)
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, ACTConfig)
    assert loaded.action_delta_offset == 3
    assert loaded.action_delta_indices == list(range(3, 10))
    assert loaded.drop_n_last_frames == 9


@pytest.mark.parametrize(
    ("offset", "chunk_size", "should_fail"),
    [(0, 1, False), (1, 1, True), (0, 2, True), (3, 4, True)],
)
def test_streaming_act_rejects_windows_that_need_episode_tail_filtering(
    tmp_path, monkeypatch, offset, chunk_size, should_fail
):
    config = _config(offset=offset, chunk_size=chunk_size)
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/streaming-act", streaming=True),
        policy=config,
        output_dir=tmp_path / f"output-{offset}-{chunk_size}",
    )
    monkeypatch.setattr(train_cfg, "_resolve_pretrained_from_cli", lambda: None)

    if should_fail:
        with pytest.raises(
            ValueError, match="Streaming datasets cannot exclude incomplete future-action windows"
        ):
            train_cfg.validate()
    else:
        train_cfg.validate()
