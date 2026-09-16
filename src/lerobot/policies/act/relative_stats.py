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

"""Training-only statistics for ACT's chunk-anchored action representation."""

import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.processor.anchored_relative_processor import anchored_delta
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_act import ACTConfig


def _stack(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.stack([torch.as_tensor(value) for value in values])


def _iter_training_chunks(
    config: ACTConfig, dataset: Any, rename_map: dict[str, str], batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Use the same frame selection as the training loader, including episode subsets
    # and any dropped tails. Never reconstruct a dataset from repo_id: doing so would
    # lose the held-out split, local edits, revision, or selected episodes.
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(config, "drop_n_last_frames", 0),
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    keys = {rename_map.get(key, key): key for key in dataset.meta.features}
    action_key, state_key = keys[ACTION], keys[OBS_STATE]
    reader = dataset._ensure_reader()
    # The parquet reader can project low-dimensional columns without decoding any
    # images/videos. Other storage formats use their normal chunk-reading contract.
    frame_view = None
    if hasattr(reader, "hf_dataset"):
        if reader.hf_dataset is None:
            reader.load_and_activate()
        frame_view = reader.hf_dataset.select_columns(["index", "episode_index", state_key])

    indices = iter(sampler)
    while batch_indices := list(islice(indices, batch_size)):
        if frame_view is not None:
            frames = frame_view[batch_indices]
            query_actions, masks = [], []
            for abs_idx, ep_idx in zip(frames["index"], frames["episode_index"], strict=True):
                query, padding = reader._get_query_indices(int(abs_idx), int(ep_idx))
                if len(query[action_key]) != config.chunk_size:
                    raise ValueError("ACT statistics require the same action chunk_size as training.")
                query_actions.extend(query[action_key])
                masks.append(padding[f"{action_key}_is_pad"])
            actions = reader._query_hf_dataset({action_key: query_actions})[action_key]
            actions = actions.reshape(len(batch_indices), config.chunk_size, -1)
            states = _stack(frames[state_key])
            is_pad = _stack(masks)
        else:
            items = reader.get_items(batch_indices)
            actions = _stack([item[action_key] for item in items])
            states = _stack([item[state_key] for item in items])
            is_pad = _stack([item[f"{action_key}_is_pad"] for item in items])
        yield actions, states, is_pad


def compute_act_relative_action_stats(
    config: ACTConfig,
    dataset: Any,
    *,
    rename_map: dict[str, str] | None = None,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Pool per-dimension statistics over valid targets in the training split.

    Each observation contributes its training chunk, including valid rows of partial
    episode-tail chunks. Padded rows are excluded, and overlapping chunks count as
    separate targets because their anchors differ. Statistics have shape (D,), shared
    across chunk positions. Only bounded batches of transformed targets are materialized.
    """
    if isinstance(dataset, IterableDataset):
        raise ValueError("Relative ACT statistics require a finite map-style dataset; disable streaming.")
    if batch_size < 1:
        raise ValueError("Statistics batch_size must be positive.")
    if len(dataset) == 0:
        raise ValueError("Cannot compute relative ACT statistics from an empty training dataset.")

    logging.info("Computing ACT relative action statistics from the selected training episodes")
    running = RunningQuantileStats()
    for actions, states, is_pad in _iter_training_chunks(config, dataset, rename_map or {}, batch_size):
        actions, states = actions.float(), states.float()
        if actions.ndim != 3 or actions.shape[1] != config.chunk_size:
            raise ValueError("ACT statistics require the same action chunk_size as training.")
        if is_pad.shape != actions.shape[:2]:
            raise ValueError("ACT action padding mask must match the batch and chunk dimensions.")
        targets = anchored_delta(
            actions, states, config.relative_eef_starts, config.relative_scalars, config.relative_anchor
        )
        valid = targets[~is_pad.bool()]
        if valid.numel() == 0:
            continue
        if not torch.isfinite(valid).all():
            raise ValueError("Relative ACT training targets contain non-finite values.")
        # Float64 accumulation avoids cancellation in nearly constant dimensions.
        running.update(valid.double().cpu().numpy())

    stats = running.get_statistics()
    logging.info("Computed ACT relative statistics from %d unpadded targets", int(stats["count"][0]))
    return {
        name: torch.as_tensor(value, dtype=torch.int64 if name == "count" else torch.float32)
        for name, value in stats.items()
    }


def resolve_act_training_processor_inputs(
    config: ACTConfig,
    dataset: Any,
    *,
    resume: bool,
    rename_map: dict[str, str],
) -> tuple[str | Path | None, dict[str, dict[str, Any]]]:
    """Choose processor source and stats for training without altering dataset metadata.

    New relative runs (including fine-tuning) rebuild processors from the active
    representation and training split. Resume keeps the checkpoint's processor source;
    the training loader must not override its saved statistics. No cross-run statistics
    cache is used, so dataset/anchor/layout changes cannot reuse stale values.
    """
    stats = rename_stats(dataset.meta.stats, rename_map)
    if resume or not config.relative_actions:
        return config.pretrained_path, stats
    stats[ACTION] = compute_act_relative_action_stats(config, dataset, rename_map=rename_map)
    return None, stats
