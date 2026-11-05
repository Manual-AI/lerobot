"""Fixed-factor action targets without changing recorded observations or anchor sampling.

The supported reservoir is a pre-delay state trace plus retained compensated action
samples at its tail. Every episode's overlap is checked exactly before use. Queries
are in source-frame units; group delay is fixed in *output* time, before division
by the stretch factor. This module is also the video-free offline/statistics API.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from lerobot.configs.online_action_resampling import OnlineActionResamplingConfig
from lerobot.utils.import_utils import _scipy_available, require_package

if TYPE_CHECKING or _scipy_available:
    from scipy.spatial.transform import Rotation


def _resolve_groups(
    action_names: list[str], state_names: list[str], config: OnlineActionResamplingConfig
) -> list[tuple[str, int, list[int], list[int], bool]]:
    if len(set(action_names)) != len(action_names) or len(set(state_names)) != len(state_names):
        raise ValueError("online action resampling requires unique action/state feature names")
    if not set(action_names).issubset(state_names):
        raise ValueError("every action channel must have a same-named pre-delay state channel")
    used: set[str] = set()
    groups = []
    for name, delay in config.group_delays.items():
        if name.endswith("_ee_pose"):
            prefix = name.removesuffix("_pose")
            names = [f"{prefix}_pos_{axis}.pos" for axis in "xyz"] + [
                f"{prefix}_rot_{i}.pos" for i in range(6)
            ]
            if not set(names).issubset(action_names):
                raise ValueError(f"pose group {name!r} requires named xyz and six row-rotation channels")
            rotation = True
        else:
            names = [name]
            if name not in action_names or "_ee_rot_" in name or "_ee_pos_" in name:
                raise ValueError(f"unsupported group {name!r}; pose components must share one group delay")
            rotation = False
        if used.intersection(names):
            raise ValueError("action resampling groups overlap")
        used.update(names)
        groups.append(
            (
                name,
                delay,
                [action_names.index(n) for n in names],
                [state_names.index(n) for n in names],
                rotation,
            )
        )
    if used != set(action_names):
        raise ValueError(
            f"group_delays must cover every action channel; missing {sorted(set(action_names) - used)}"
        )
    return groups


def _rotation_matrices(values: np.ndarray) -> np.ndarray:
    rows = values.reshape(-1, 2, 3).astype(np.float64)
    first = rows[:, 0]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate row_rot6d first row")
    first = first / norm
    second = rows[:, 1] - np.sum(first * rows[:, 1], axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("degenerate row_rot6d second row")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


class EpisodeActionResampler:
    """Validated numeric episode, reusable across arbitrary anchors and horizons."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        action_names: list[str],
        state_names: list[str],
        config: OnlineActionResamplingConfig,
    ):
        self.config = config.validate()
        self.states = np.asarray(states)
        self.actions = np.asarray(actions)
        if (
            self.states.ndim != 2
            or self.actions.ndim != 2
            or len(self.states) != len(self.actions)
            or not len(self.actions)
        ):
            raise ValueError("states/actions must be nonempty aligned episode matrices")
        if self.states.shape[1] != len(state_names) or self.actions.shape[1] != len(action_names):
            raise ValueError("state/action names must match their feature dimensions")
        groups = _resolve_groups(action_names, state_names, config)
        state_channels = [state_names.index(name) for name in action_names]
        if not np.isfinite(self.states[:, state_channels]).all() or not np.isfinite(self.actions).all():
            raise ValueError("state/action reservoir contains non-finite values")
        self.groups = []
        for name, delay, ai, si, rotation in groups:
            if delay >= len(self.actions):
                raise ValueError(f"episode too short to verify pre-delay overlap for {name}")
            if not np.array_equal(self.actions[: len(self.actions) - delay, ai], self.states[delay:, si]):
                raise ValueError(f"pre-delay state/action overlap mismatch for {name}")
            reservoir = (
                np.concatenate((self.states[:, si], self.actions[-delay:, ai]), axis=0)
                if delay
                else self.states[:, si].copy()
            )
            rotations, rotvecs = None, None
            if rotation:
                require_package("scipy", extra="scipy-dep")
                rotations = Rotation.from_matrix(_rotation_matrices(reservoir[:, 3:]))
                rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
            self.groups.append((delay, ai, reservoir, rotations, rotvecs))

    def targets(
        self, anchor_indices: list[int] | np.ndarray, chunk_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        anchors = np.asarray(anchor_indices)
        if (
            anchors.ndim != 1
            or not np.issubdtype(anchors.dtype, np.integer)
            or np.any(anchors < 0)
            or np.any(anchors >= len(self.actions))
        ):
            raise ValueError("anchor_indices must be episode-local integer row indices")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        shape = (len(anchors), chunk_size)
        targets = np.empty((*shape, self.actions.shape[1]), dtype=self.actions.dtype)
        is_pad = np.zeros(shape, dtype=bool)
        cfg = self.config
        for delay, ai, reservoir, rotations, rotvecs in self.groups:
            # Delay d/fs seconds remains fixed in robot/output time as s changes.
            query = (
                anchors[:, None]
                + (np.arange(chunk_size)[None, :] * cfg.source_fps / cfg.output_fps + delay)
                / cfg.stretch_factor
            )
            is_pad |= (query < 0) | (query > len(reservoir) - 1)
            query = np.clip(query, 0, len(reservoir) - 1)
            lower = np.floor(query).astype(np.int64)
            upper = np.ceil(query).astype(np.int64)
            fraction = query - lower
            values = reservoir[lower].copy()
            fractional = fraction != 0
            if np.any(fractional):
                alpha = fraction[fractional, None]
                values[fractional] = (1 - alpha) * reservoir[lower[fractional]] + alpha * reservoir[
                    upper[fractional]
                ]
                if rotations is not None:
                    idx = lower[fractional]
                    interpolated = rotations[idx] * Rotation.from_rotvec(rotvecs[idx] * alpha)
                    values[fractional, 3:] = interpolated.as_matrix()[:, :2].reshape(-1, 6)
            # Exact integer samples never take a rotation round trip.
            targets[..., ai] = values
        return targets, is_pad


def build_absolute_action_targets(
    states: np.ndarray,
    actions: np.ndarray,
    action_names: list[str],
    state_names: list[str],
    anchor_indices: list[int] | np.ndarray,
    chunk_size: int,
    config: OnlineActionResamplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (B,T,D) absolute targets and (B,T) padding, without videos or torch."""
    return EpisodeActionResampler(states, actions, action_names, state_names, config).targets(
        anchor_indices, chunk_size
    )


class DatasetActionResampler:
    """Video-free numeric index for the selected map-style episodes.

    Construction validates *every* loaded episode, not only episodes sampled for
    previews or statistics. Arrays are small compared with video and can be copied
    safely to DataLoader workers. No action chunk cache changes the anchor sampler.
    """

    def __init__(self, dataset, config: OnlineActionResamplingConfig):
        config.validate()
        if dataset.meta.storage_format != "lerobot":
            raise ValueError("online action resampling currently requires the default parquet dataset")
        if dataset.revision != config.source_revision:
            raise ValueError("dataset revision does not match online action resampling source_revision")
        if dataset.meta.fps != config.source_fps:
            raise ValueError("dataset FPS does not match online action resampling source_fps")
        stamps = (dataset.delta_timestamps or {}).get("action")
        if not stamps or not np.allclose(
            stamps, np.arange(len(stamps)) / dataset.meta.fps, rtol=0, atol=1e-9
        ):
            raise ValueError(
                "online action resampling requires a contiguous action horizon starting at offset zero"
            )
        for key, values in (dataset.delta_timestamps or {}).items():
            if key != "action" and values != [0.0]:
                raise ValueError("online action resampling only supports a single current observation")
        self.chunk_size = len(stamps)
        reader = dataset._ensure_reader()
        if not hasattr(reader, "hf_dataset"):
            raise ValueError("online action resampling requires the default parquet reader")
        if reader.hf_dataset is None:
            reader.load_and_activate()
        numeric = reader.hf_dataset.select_columns(
            ["action", "observation.state", "episode_index", "frame_index"]
        ).with_format("numpy")[:]
        states = np.asarray(numeric["observation.state"])
        actions = np.asarray(numeric["action"])
        self.episode_indices = np.asarray(numeric["episode_index"]).reshape(-1)
        self.frame_indices = np.asarray(numeric["frame_index"]).reshape(-1)
        selected = sorted(int(v) for v in np.unique(self.episode_indices))
        if config.selected_episodes is not None and not set(selected).issubset(config.selected_episodes):
            raise ValueError("loaded episodes are outside the resolved online action resampling selection")
        self.config = replace(
            config,
            selected_episodes=None
            if config.selected_episodes is None and selected == list(range(dataset.meta.total_episodes))
            else selected,
        )
        features = dataset.meta.features
        action_names = features["action"].get("names")
        state_names = features["observation.state"].get("names")
        if not isinstance(action_names, list) or not isinstance(state_names, list):
            raise ValueError("online action resampling requires named action and state feature schemas")
        self.episodes = {}
        for episode in selected:
            rows = np.flatnonzero(self.episode_indices == episode)
            if not np.array_equal(self.frame_indices[rows], np.arange(len(rows))):
                raise ValueError(f"episode {episode} must contain every contiguous source frame")
            try:
                self.episodes[episode] = EpisodeActionResampler(
                    states[rows], actions[rows], action_names, state_names, self.config
                )
            except ValueError as exc:
                raise ValueError(f"online action resampling episode {episode}: {exc}") from exc

    def sample(self, index: int, chunk_size: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode = self.episodes[int(self.episode_indices[index])]
        anchor = int(self.frame_indices[index])
        action, pad = episode.targets([anchor], self.chunk_size if chunk_size is None else chunk_size)
        return action[0], episode.states[anchor], pad[0]

    def apply(self, items: list[dict], indices: list[int]) -> list[dict]:
        import torch

        # Grouping lets batched data loading share SO(3) operations per episode.
        positions: dict[int, list[int]] = {}
        for pos, index in enumerate(indices):
            positions.setdefault(int(self.episode_indices[index]), []).append(pos)
        for episode_id, batch_positions in positions.items():
            anchors = [int(self.frame_indices[indices[pos]]) for pos in batch_positions]
            actions, pads = self.episodes[episode_id].targets(anchors, self.chunk_size)
            for row, pos in enumerate(batch_positions):
                items[pos]["action"] = torch.from_numpy(actions[row])
                items[pos]["action_is_pad"] = torch.from_numpy(pads[row])
        return items
