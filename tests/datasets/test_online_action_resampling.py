"""Observable geometry, boundary, dataset and statistics agreement for online targets."""

from dataclasses import replace
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.online_action_resampling import (
    EpisodeActionResampler,
    OnlineActionResamplingConfig,
    build_absolute_action_targets,
)

pytestmark = pytest.mark.dataset
NAMES = (
    [f"right_ee_pos_{a}.pos" for a in "xyz"]
    + [f"right_ee_rot_{i}.pos" for i in range(6)]
    + ["right_thumb_joint.pos", "right_middle_joint.pos"]
)
STATE_NAMES = ["unused"] + NAMES  # Builder must resolve named channels, not assume first D.


def config(factor=1.0, **kwargs):
    return OnlineActionResamplingConfig(
        stretch_factor=factor,
        source_revision="a" * 40,
        group_delays={"right_ee_pose": 3, "right_thumb_joint.pos": 3, "right_middle_joint.pos": 2},
        source_contract={
            "reservoir": "state_plus_recorded_action_tail_v1",
            "recorded_action_semantics": "per_group_delay_compensated",
            "state_semantics": "pre_delay_same_frame",
            "rotation_encoding": "row_rot6d",
            "time_stretch_applied": False,
            "idm_applied": False,
            "calibration_sha256": "b" * 64,
            "pre_delay_source_revision": "c" * 40,
            "delay_source_revision": "d" * 40,
        },
        **kwargs,
    )


def recording(length=60, shift=0):
    t = np.arange(length + 3, dtype=np.float64)
    xyz = np.stack((t + shift, 2 * t, -t), axis=-1)
    rot = (
        Rotation.from_rotvec(np.column_stack((np.zeros_like(t), np.zeros_like(t), t * 0.025)))
        .as_matrix()[:, :2]
        .reshape(-1, 6)
    )
    raw = np.concatenate((xyz, rot, t[:, None] * 4, t[:, None] * 5), axis=1).astype(np.float32)
    states = np.column_stack((np.full(length, -123, dtype=np.float32), raw[:length]))
    actions = raw[3 : length + 3].copy()
    actions[:, -1] = raw[2 : length + 2, -1]
    return states, actions


def test_identity_exact_all_anchors_masks_and_tails():
    states, actions = recording()
    anchors = np.arange(len(actions))
    result, pad = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, anchors, 30, config())
    index = anchors[:, None] + np.arange(30)
    np.testing.assert_array_equal(result, actions[np.minimum(index, len(actions) - 1)])
    np.testing.assert_array_equal(pad, index >= len(actions))


def test_fractional_geometry_and_output_time_delay():
    states, actions = recording()
    result, pad = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, [4], 7, config(2.5))
    q_pose = 4 + (np.arange(7) + 3) / 2.5
    q_middle = 4 + (np.arange(7) + 2) / 2.5
    np.testing.assert_allclose(result[0, :, 0], q_pose, atol=1e-6)
    np.testing.assert_allclose(result[0, :, -2], q_pose * 4, atol=1e-6)
    np.testing.assert_allclose(result[0, :, -1], q_middle * 5, atol=1e-6)
    expected = (
        Rotation.from_rotvec(np.column_stack((np.zeros_like(q_pose), np.zeros_like(q_pose), q_pose * 0.025)))
        .as_matrix()[:, :2]
        .reshape(-1, 6)
    )
    np.testing.assert_allclose(result[0, :, 3:9], expected, atol=1e-6)
    assert not pad.any()
    # Non-integer source span (6/2.5), and no extra source-index delay after warp.
    assert result[0, -1, 0] == pytest.approx(7.6)


def test_distinct_rates_and_unequal_group_support():
    states, actions = recording(8)
    cfg = config(2.5, source_fps=30, output_fps=20)
    out, pad = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, [7], 8, cfg)
    queries = 7 + (np.arange(8) * 1.5 + 2) / 2.5
    np.testing.assert_array_equal(pad[0], queries > 9)
    np.testing.assert_allclose(out[0, :, -1], np.minimum(queries, 9) * 5, atol=1e-5)


def test_short_corrupt_and_missing_provenance_rejected():
    states, actions = recording(3)
    with pytest.raises(ValueError, match="too short"):
        EpisodeActionResampler(states, actions, NAMES, STATE_NAMES, config())
    states, actions = recording()
    actions[-5, 1] += 1
    with pytest.raises(ValueError, match="overlap mismatch"):
        EpisodeActionResampler(states, actions, NAMES, STATE_NAMES, config())
    with pytest.raises(ValueError, match="provenance"):
        replace(config(), source_contract={}).validate()
    with pytest.raises(ValueError, match="offset"):
        replace(config(), action_delta_offset=1).validate()


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_factors(factor):
    with pytest.raises(ValueError, match="positive"):
        config(factor).validate()


def test_identity_and_typed_serialization():
    cfg = config(2.5)
    dataset_config = DatasetConfig(repo_id="test/dataset", online_action_resampling=cfg)
    decoded = draccus.decode(DatasetConfig, draccus.encode(dataset_config))
    assert decoded.online_action_resampling.fingerprint() == cfg.fingerprint()
    for change in (
        {"stretch_factor": 1.0},
        {"source_revision": "f" * 40},
        {"selected_episodes": [2]},
        {"output_fps": 20.0},
        {"group_delays": {**cfg.group_delays, "right_ee_pose": 2}},
    ):
        assert replace(cfg, **change).fingerprint() != cfg.fingerprint()


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("resampling") / "dataset"
    # The reference policy expects action channels aligned in its state vector;
    # the independent builder test above exercises arbitrary schema positions.
    features = {
        "action": {"dtype": "float32", "shape": (len(NAMES),), "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": (len(NAMES),), "names": NAMES},
    }
    ds = LeRobotDataset.create("test/online", root=root, fps=30, features=features, use_videos=False)
    for ep in range(3):
        states, actions = recording(55, shift=ep * 100)
        for state, action in zip(states[:, 1:], actions, strict=True):
            ds.add_frame({"observation.state": state, "action": action, "task": "test"})
        ds.save_episode()
    ds.finalize()
    return root


def load(root, factor=None, episodes=None, horizon=30):
    return LeRobotDataset(
        "test/online",
        root=root,
        revision="a" * 40,
        episodes=episodes,
        delta_timestamps={"action": [i / 30 for i in range(horizon)]},
        online_action_resampling=None if factor is None else config(factor),
        download_videos=False,
    )


def test_real_dataset_scalar_batch_and_offline(dataset_root):
    legacy = load(dataset_root, episodes=[2])
    identity = load(dataset_root, 1.0, episodes=[2])
    slow = load(dataset_root, 2.5, episodes=[2])
    assert identity.online_action_resampling.selected_episodes == [2]
    assert len(slow) == len(legacy) == 55
    indices = [0, 20, 54]
    for i, batch in zip(indices, slow.__getitems__(indices), strict=True):
        baseline, one, scalar = legacy[i], identity[i], slow[i]
        torch.testing.assert_close(one["action"], baseline["action"], rtol=0, atol=0)
        torch.testing.assert_close(one["action_is_pad"], baseline["action_is_pad"])
        for key in baseline.keys() - {"action", "action_is_pad"}:
            if isinstance(baseline[key], torch.Tensor):
                torch.testing.assert_close(scalar[key], baseline[key], rtol=0, atol=0)
            else:
                assert scalar[key] == baseline[key]
        np.testing.assert_array_equal(scalar["action"], batch["action"])
        offline, state, pad = slow.get_online_action_sample(i)
        np.testing.assert_array_equal(offline, scalar["action"])
        np.testing.assert_array_equal(state, scalar["observation.state"])
        np.testing.assert_array_equal(pad, scalar["action_is_pad"])


def test_statistics_both_routes_and_selected_population(dataset_root):
    from lerobot.policies.groot.processor_groot import (
        _infer_n1_7_action_groups,
        _make_relative_action_training_stats,
        _make_relative_action_training_stats_from_dataset_meta,
    )

    groups = _infer_n1_7_action_groups(
        NAMES, action_dim=len(NAMES), exclude_joints=["joint"], eef_groups=["right_ee"]
    )
    policy = SimpleNamespace(
        action_delta_offset=0, relative_exclude_joints=["joint"], relative_eef_groups=["right_ee"]
    )
    legacy = load(dataset_root, episodes=[2], horizon=40)
    one = load(dataset_root, 1.0, episodes=[2], horizon=40)
    slow = load(dataset_root, 2.5, episodes=[2], horizon=40)

    def stats(ds):
        return _make_relative_action_training_stats(
            ds, exclude_joints=["joint"], action_names=NAMES, groups=groups
        )["action"]

    for a, b in (
        (stats(legacy), stats(one)),
        (stats(slow), _make_relative_action_training_stats_from_dataset_meta(policy, slow.meta)["action"]),
    ):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert stats(one)["count"].tolist() == [16] * 40  # whole padded chunks excluded
    assert np.all(stats(slow)["count"] > stats(one)["count"])


def training_config(root, *, eval_split=0.0):
    policy = SimpleNamespace(
        type="groot",
        n_obs_steps=1,
        action_delta_offset=0,
        use_relative_actions=True,
        relative_eef_groups=["right_ee"],
        relative_exclude_joints=["joint"],
        action_delta_indices=list(range(30)),
        observation_delta_indices=[0],
        reward_delta_indices=None,
        chunk_size=30,
    )
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="test/online",
            root=str(root),
            revision="a" * 40,
            online_action_resampling=config(2.5),
            eval_split=eval_split,
        ),
        trainable_config=policy,
        rename_map={},
        tolerance_s=1e-4,
    )


def test_factory_and_split_keep_resolved_episode_population(dataset_root):
    from lerobot.datasets.factory import make_dataset, make_train_eval_datasets

    cfg = training_config(dataset_root)
    full = make_dataset(cfg)
    assert full.online_action_resampling.selected_episodes is None
    assert cfg.dataset.online_action_resampling.selected_episodes is None
    cfg = training_config(dataset_root, eval_split=0.3)
    train, evaluation = make_train_eval_datasets(cfg)
    assert train.online_action_resampling.selected_episodes == [0, 1]
    assert evaluation.online_action_resampling.selected_episodes == [2]
    assert cfg.dataset.online_action_resampling.selected_episodes is None
    # A saved config must reconstruct the full population before making the same split.
    cfg.dataset = draccus.decode(DatasetConfig, draccus.encode(cfg.dataset))
    reloaded_train, reloaded_eval = make_train_eval_datasets(cfg)
    assert reloaded_train.episodes == train.episodes
    assert reloaded_eval.episodes == evaluation.episodes
    from lerobot.policies.groot.processor_groot import _make_relative_action_training_stats_from_dataset_meta

    train_stats = _make_relative_action_training_stats_from_dataset_meta(cfg.trainable_config, train.meta)
    eval_stats = _make_relative_action_training_stats_from_dataset_meta(cfg.trainable_config, evaluation.meta)
    # Two training episodes contribute; held-out episode 2 contributes no vectors.
    torch.testing.assert_close(train_stats["action"]["count"], 2 * eval_stats["action"]["count"])
    assert (
        train.meta.online_action_resampling.fingerprint()
        != evaluation.meta.online_action_resampling.fingerprint()
    )
    assert len(train) == 110 and len(evaluation) == 55


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "act"),
        ("n_obs_steps", 2),
        ("action_delta_offset", 1),
        ("use_relative_actions", False),
        ("relative_exclude_joints", ["unrelated"]),
    ],
)
def test_factory_rejects_unsupported_policy(dataset_root, field, value):
    from lerobot.datasets.factory import make_dataset

    cfg = training_config(dataset_root)
    setattr(cfg.trainable_config, field, value)
    with pytest.raises(ValueError, match="online action resampling"):
        make_dataset(cfg)


def test_every_selected_episode_is_validated(dataset_root, tmp_path):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "corrupt"
    shutil.copytree(dataset_root, root)
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    actions = table["action"].to_pylist()
    actions[-5][0] += 1
    field = table.schema.field("action")
    table = table.set_column(
        table.schema.get_field_index("action"), field, pa.array(actions, type=field.type)
    )
    pq.write_table(table, path)
    # Even never-fetched late episodes are checked at construction.
    with pytest.raises(ValueError, match="episode 2.*overlap mismatch"):
        load(root, 2.5)
    assert len(load(root, 2.5, episodes=[0])) == 55


def test_saved_algorithm_and_statistics_semantics_are_versioned():
    for change in ({"algorithm_version": "future-version"}, {"statistics_support": "include_partial_chunks"}):
        with pytest.raises(ValueError, match="unsupported saved"):
            replace(config(), **change).validate()


def test_unused_state_diagnostics_do_not_affect_targets():
    states, actions = recording()
    states[:, 0] = np.nan
    target, _ = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, [0], 3, config())
    np.testing.assert_array_equal(target[0], actions[:3])


def test_batches_keep_episode_boundaries_order_and_duplicate_anchors(dataset_root):
    """DataLoader requests may be shuffled, duplicated and span adjacent episodes."""
    dataset = load(dataset_root, 2.5)
    indices = [109, 55, 54, 110, 0, 54, 164]
    for index, batch in zip(indices, dataset.__getitems__(indices), strict=True):
        scalar = dataset[index]
        for name in ("action", "action_is_pad", "observation.state", "episode_index"):
            torch.testing.assert_close(batch[name], scalar[name], rtol=0, atol=0)
        episode = index // 55
        assert torch.all(batch["action"][:, 0] >= episode * 100)
        assert torch.all(batch["action"][:, 0] < episode * 100 + 58)


def test_so3_crosses_pi_on_short_arc_in_rotated_coordinate_frame():
    """A 170 to -170 degree crossing must pass through 180, not zero."""
    length = 8
    states, actions = recording(length)
    t = np.arange(length + 3)
    base = Rotation.from_euler("xyz", [23, -41, 17], degrees=True)
    angles = 170 + t * 20
    rotations = base * Rotation.from_rotvec(
        np.column_stack((np.zeros_like(t), np.zeros_like(t), np.deg2rad(angles)))
    )
    raw = rotations.as_matrix()[:, :2].reshape(-1, 6).astype(np.float32)
    states[:, 4:10] = raw[:length]
    actions[:, 3:9] = raw[3:]
    target, pad = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, [0], 5, config(6.0))
    q = (np.arange(5) + 3) / 6
    expected = (
        base
        * Rotation.from_rotvec(
            np.column_stack((np.zeros_like(q), np.zeros_like(q), np.deg2rad(170 + q * 20)))
        )
    ).as_matrix()[:, :2]
    np.testing.assert_allclose(target[0, :, 3:9].reshape(-1, 2, 3), expected, atol=1e-6)
    assert not pad.any()


def test_zero_delay_single_frame_has_one_real_target():
    """No delay needs no discarded tail; a one-frame recording remains usable."""
    states, _ = recording(1)
    actions = states[:, 1:].copy()
    cfg = replace(config(2.5), group_delays=dict.fromkeys(config().group_delays, 0))
    targets, pad = build_absolute_action_targets(states, actions, NAMES, STATE_NAMES, [0], 3, cfg)
    np.testing.assert_array_equal(targets[0], np.repeat(actions, 3, axis=0))
    np.testing.assert_array_equal(pad, [[False, True, True]])


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_stretch_applied", 0),
        ("idm_applied", None),
        ("state_semantics", "delay_compensated"),
        ("rotation_encoding", "column_rot6d"),
        ("calibration_sha256", "b" * 63),
        ("pre_delay_source_revision", "main"),
        ("delay_source_revision", "d" * 39),
    ],
)
def test_malformed_provenance_is_not_accepted_as_an_equivalent_contract(field, value):
    cfg = config()
    cfg.source_contract[field] = value
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_native_cli_parses_typed_online_dataset_config():
    import json
    from dataclasses import asdict

    parsed = draccus.parse(
        DatasetConfig,
        args=[
            "--repo_id=test/online",
            "--online_action_resampling=" + json.dumps(asdict(config(2.5))),
        ],
    )
    assert isinstance(parsed.online_action_resampling, OnlineActionResamplingConfig)
    assert parsed.online_action_resampling.fingerprint() == config(2.5).fingerprint()


def test_selected_episode_statistics_match_absolute_finger_population(dataset_root, tmp_path):
    """A different episode has a distinct finger distribution, even after EEF anchoring."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.policies.groot.processor_groot import (
        _infer_n1_7_action_groups,
        _make_relative_action_training_stats,
        _make_relative_action_training_stats_from_dataset_meta,
    )

    root = tmp_path / "distinct-fingers"
    shutil.copytree(dataset_root, root)
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        episode_ids = table["episode_index"].to_pylist()
        for column in ("action", "observation.state"):
            values = table[column].to_pylist()
            for row, episode in zip(values, episode_ids, strict=True):
                row[-2] += 1000 * episode
                row[-1] += 1000 * episode
            field = table.schema.field(column)
            table = table.set_column(
                table.schema.get_field_index(column), field, pa.array(values, type=field.type)
            )
        pq.write_table(table, path)
    selected = load(root, 2.5, episodes=[2], horizon=30)
    groups = _infer_n1_7_action_groups(
        NAMES, action_dim=len(NAMES), exclude_joints=["joint"], eef_groups=["right_ee"]
    )
    policy = SimpleNamespace(
        action_delta_offset=0, relative_exclude_joints=["joint"], relative_eef_groups=["right_ee"]
    )
    direct = _make_relative_action_training_stats(
        selected, exclude_joints=["joint"], action_names=NAMES, groups=groups
    )["action"]
    bare = _make_relative_action_training_stats_from_dataset_meta(policy, selected.meta)["action"]
    assert len(selected) == 55
    # The bare GR00T stats path intentionally uses its native 40-step horizon.
    # For each route, full-chunk eligibility is limited by middle-finger support.
    for stats, horizon in ((direct, 30), (bare, 40)):
        count = int(np.floor(56 - (horizon - 1 + 2) / 2.5)) + 1
        anchors = np.arange(count, dtype=np.float32)
        expected = 2000 + 5 * (anchors[:, None] + (np.arange(horizon, dtype=np.float32)[None] + 2) / 2.5)
        np.testing.assert_array_equal(stats["count"], np.full(horizon, count))
        np.testing.assert_allclose(stats["mean"][:, -1], expected.mean(axis=0), atol=2e-3, rtol=0)
        np.testing.assert_allclose(stats["std"][:, -1], expected.std(axis=0), atol=2e-3, rtol=0)
        np.testing.assert_allclose(stats["min"][:, -1], expected.min(axis=0), atol=2e-3, rtol=0)
        np.testing.assert_allclose(stats["max"][:, -1], expected.max(axis=0), atol=2e-3, rtol=0)


def groot_config():
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig

    return GrootConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (len(NAMES),))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (len(NAMES),))},
        device="cpu",
        use_bf16=False,
        use_relative_actions=True,
        relative_eef_groups=["right_ee"],
        relative_exclude_joints=["joint"],
        chunk_size=30,
        n_action_steps=30,
    )


def raw_checkpoint(root, *, baked_stats):
    """Minimal real raw sidecars, with intentionally wrong preexisting normalization."""
    import json

    from lerobot.policies.groot.processor_groot import _infer_n1_7_action_groups

    root.mkdir()
    (root / "config.json").write_text(json.dumps({"architectures": ["Gr00tN1d7"]}))
    groups = _infer_n1_7_action_groups(
        NAMES, action_dim=len(NAMES), exclude_joints=["joint"], eef_groups=["right_ee"]
    )
    modality = {
        "state": {"modality_keys": [g.key for g in groups]},
        "action": {"modality_keys": [g.key for g in groups], "delta_indices": list(range(40))},
    }
    (root / "processor_config.json").write_text(
        json.dumps(
            {"processor_kwargs": {"modality_configs": {"new_embodiment": modality} if baked_stats else {}}}
        )
    )
    stats = {
        kind: {
            g.key: {
                "min": [-999.0] * len(g.indices),
                "max": [999.0] * len(g.indices),
                "mean": [0.0] * len(g.indices),
                "std": [999.0] * len(g.indices),
            }
            for g in groups
        }
        for kind in ("state", "action")
    }
    (root / "statistics.json").write_text(json.dumps({"new_embodiment": stats} if baked_stats else {}))
    return root


@pytest.mark.parametrize("baked_stats", [False, True])
def test_raw_online_processors_recompute_final_stats_and_oracle_round_trip(
    dataset_root, tmp_path, baked_stats
):
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.groot.processor_groot import (
        GrootN17ActionDecodeStep,
        GrootN17PackInputsStep,
        _make_relative_action_training_stats_from_dataset_meta,
        make_groot_pre_post_processors_from_pretrained,
    )

    ds = load(dataset_root, 2.5, episodes=[2])
    policy = groot_config()
    source = raw_checkpoint(tmp_path / "raw", baked_stats=baked_stats)
    # Caller 2D stats must not bypass the online target identity either.
    poison = {
        "action": {"min": torch.full((40, len(NAMES)), -999.0), "max": torch.full((40, len(NAMES)), 999.0)}
    }
    pre, post = make_groot_pre_post_processors_from_pretrained(
        policy, str(source), dataset_meta=ds.meta, dataset_stats=poison
    )
    pack = next(s for s in pre.steps if isinstance(s, GrootN17PackInputsStep))
    decode = next(s for s in post.steps if isinstance(s, GrootN17ActionDecodeStep))
    expected = _make_relative_action_training_stats_from_dataset_meta(policy, ds.meta)["action"]
    for key in ("min", "max", "mean", "std", "q01", "q99"):
        np.testing.assert_array_equal(
            np.asarray(pack.raw_stats["action"]["right_middle_joint"][key])[:, 0], expected[key][:, -1]
        )
    np.testing.assert_array_equal(
        np.asarray(pack.raw_stats["action"]["right_middle_joint"]["min"])[:3, 0], [4.0, 6.0, 8.0]
    )
    assert pack.raw_stats == decode.raw_stats
    assert (
        pack.online_action_statistics_identity["target_fingerprint"]
        == ds.online_action_resampling.fingerprint()
    )
    # Exercise the final pack/decode steps, including first finger targets below source minima.
    # Min/max (rather than percentile clipping) gives an exact normalization oracle.
    pack.use_percentiles = decode.use_percentiles = False
    for index in (0, 20):
        sample = ds[index]
        target = sample["action"].unsqueeze(0)
        packed = pack(
            {
                TransitionKey.OBSERVATION: {"observation.state": sample["observation.state"].unsqueeze(0)},
                TransitionKey.ACTION: target,
            }
        )
        recovered = decode({TransitionKey.ACTION: packed[TransitionKey.ACTION]})[TransitionKey.ACTION]
        torch.testing.assert_close(recovered[:, : target.shape[1]], target, atol=3e-5, rtol=1e-5)


def test_selected_shifted_finger_final_assets(dataset_root, tmp_path):
    """Metadata source stats remain unchanged when an episode's numeric targets shift."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.policies.groot.processor_groot import (
        GrootN17PackInputsStep,
        make_groot_pre_post_processors,
    )

    root = tmp_path / "shifted"
    shutil.copytree(dataset_root, root)
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        for column in ("action", "observation.state"):
            values = table[column].to_pylist()
            for row, episode in zip(values, table["episode_index"].to_pylist(), strict=True):
                row[-1] += 1000 * episode
            field = table.schema.field(column)
            table = table.set_column(
                table.schema.get_field_index(column), field, pa.array(values, type=field.type)
            )
        pq.write_table(table, path)
    ds = load(root, 2.5, episodes=[2])
    policy = groot_config()
    policy.base_model_path = str(raw_checkpoint(tmp_path / "raw", baked_stats=True))
    pre, _ = make_groot_pre_post_processors(policy, dataset_meta=ds.meta)
    pack = next(s for s in pre.steps if isinstance(s, GrootN17PackInputsStep))
    np.testing.assert_array_equal(
        np.asarray(pack.raw_stats["action"]["right_middle_joint"]["min"])[:3, 0], [2004.0, 2006.0, 2008.0]
    )


def test_online_serialized_statistics_identity_guards_resume_and_allows_inference(dataset_root, tmp_path):
    import json
    from copy import deepcopy

    from lerobot.policies.groot.processor_groot import (
        GrootN17PackInputsStep,
        make_groot_pre_post_processors,
        make_groot_pre_post_processors_from_pretrained,
    )

    ds = load(dataset_root, 2.5, episodes=[0, 1])
    policy = groot_config()
    policy.base_model_path = str(raw_checkpoint(tmp_path / "raw", baked_stats=False))
    pre, post = make_groot_pre_post_processors(policy, dataset_meta=ds.meta)
    saved = tmp_path / "saved"
    pre.save_pretrained(saved)
    post.save_pretrained(saved)
    pack = next(s for s in pre.steps if isinstance(s, GrootN17PackInputsStep))
    loaded_pre, _ = make_groot_pre_post_processors_from_pretrained(policy, str(saved), dataset_meta=ds.meta)
    loaded_pack = next(s for s in loaded_pre.steps if isinstance(s, GrootN17PackInputsStep))
    assert loaded_pack.online_action_statistics_identity == pack.online_action_statistics_identity
    assert loaded_pack.raw_stats == pack.raw_stats
    assert loaded_pack.training is True
    policy._runtime_dataset_meta = ds.meta
    inferred_pre, _ = make_groot_pre_post_processors_from_pretrained(policy, str(saved))
    assert next(s for s in inferred_pre.steps if isinstance(s, GrootN17PackInputsStep)).training is True
    del policy._runtime_dataset_meta
    # Disabling the feature still changes the target distribution; metadata-free
    # inference remains valid because it continues to use the saved representation.
    disabled_meta = load(dataset_root, episodes=[0, 1]).meta
    with pytest.raises(ValueError, match="identity.*raw N1.7 checkpoint"):
        make_groot_pre_post_processors_from_pretrained(policy, str(saved), dataset_meta=disabled_meta)
    # Inferring disabled metadata must still enforce the saved online identity.
    policy._runtime_dataset_meta = disabled_meta
    with pytest.raises(ValueError, match="identity.*raw N1.7 checkpoint"):
        make_groot_pre_post_processors_from_pretrained(policy, str(saved))
    del policy._runtime_dataset_meta
    # Dataset-free inference is valid even for older serialized processors.
    inference_pre, _ = make_groot_pre_post_processors_from_pretrained(policy, str(saved))
    assert next(s for s in inference_pre.steps if isinstance(s, GrootN17PackInputsStep)).training is False
    for change in (
        {"stretch_factor": 1.0},
        {"selected_episodes": [2]},
        {"source_revision": "f" * 40},
        {"output_fps": 20.0},
        {"group_delays": {**config().group_delays, "right_ee_pose": 2}},
    ):
        meta = deepcopy(ds.meta)
        meta.online_action_resampling = replace(ds.online_action_resampling, **change)
        with pytest.raises(ValueError, match="identity.*raw N1.7 checkpoint"):
            make_groot_pre_post_processors_from_pretrained(policy, str(saved), dataset_meta=meta)
    for policy_change in (replace(policy, relative_eef_groups=["right_ee_"]), replace(policy, chunk_size=35)):
        with pytest.raises(ValueError, match="identity.*raw N1.7 checkpoint"):
            make_groot_pre_post_processors_from_pretrained(policy_change, str(saved), dataset_meta=ds.meta)
    config_path = saved / "policy_preprocessor.json"
    original = json.loads(config_path.read_text())
    for bad_identity in (None, {**pack.online_action_statistics_identity, "horizon": 39}):
        changed = deepcopy(original)
        saved_pack_config = next(
            s["config"] for s in changed["steps"] if s["registry_name"] == "groot_n1_7_pack_inputs_v1"
        )
        if bad_identity is None:
            del saved_pack_config["online_action_statistics_identity"]
        else:
            saved_pack_config["online_action_statistics_identity"] = bad_identity
        config_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="identity.*raw N1.7 checkpoint"):
            make_groot_pre_post_processors_from_pretrained(policy, str(saved), dataset_meta=ds.meta)
        make_groot_pre_post_processors_from_pretrained(policy, str(saved))
        if bad_identity is None:
            # Ordinary serialized processors without an online identity keep their
            # previous disabled-dataset loading behavior.
            make_groot_pre_post_processors_from_pretrained(policy, str(saved), dataset_meta=disabled_meta)


@pytest.mark.parametrize("change", ["reordered_pose", "relative_fingers", "absolute_pose"])
def test_online_processor_entrypoint_rejects_unsupported_geometry(dataset_root, change):
    from copy import deepcopy

    from lerobot.policies.groot.processor_groot import _online_action_statistics_identity

    ds = load(dataset_root, 2.5)
    policy = groot_config()
    meta = deepcopy(ds.meta)
    if change == "reordered_pose":
        names = list(NAMES)
        names[0], names[1] = names[1], names[0]
        for key in ("action", "observation.state"):
            meta.features[key]["names"] = names
    elif change == "relative_fingers":
        policy.relative_exclude_joints = []
    else:
        policy.use_relative_actions = False
    with pytest.raises(ValueError, match="online action resampling"):
        _online_action_statistics_identity(policy, meta)


def test_disabled_serialized_resume_preserves_explicit_metadata_training_mode(dataset_root, tmp_path):
    from lerobot.policies.groot.processor_groot import (
        GrootN17PackInputsStep,
        make_groot_pre_post_processors,
        make_groot_pre_post_processors_from_pretrained,
    )

    disabled_meta = load(dataset_root).meta
    policy = groot_config()
    source = raw_checkpoint(tmp_path / "raw", baked_stats=False)
    policy.base_model_path = str(source)
    policy._runtime_dataset_meta = disabled_meta
    fresh_pre, fresh_post = make_groot_pre_post_processors(policy)
    assert next(s for s in fresh_pre.steps if isinstance(s, GrootN17PackInputsStep)).training is True
    saved = tmp_path / "saved"
    fresh_pre.save_pretrained(saved)
    fresh_post.save_pretrained(saved)

    # Ordinary serialized loading historically requires explicitly passed metadata
    # to enable training mode, even when config carries runtime dataset metadata.
    resumed_pre, _ = make_groot_pre_post_processors_from_pretrained(policy, str(saved))
    resumed_pack = next(s for s in resumed_pre.steps if isinstance(s, GrootN17PackInputsStep))
    assert resumed_pack.online_action_statistics_identity is None
    assert resumed_pack.training is False
    explicit_pre, _ = make_groot_pre_post_processors_from_pretrained(
        policy, str(saved), dataset_meta=disabled_meta
    )
    assert next(s for s in explicit_pre.steps if isinstance(s, GrootN17PackInputsStep)).training is True
    # Raw checkpoints keep their existing runtime metadata fallback.
    raw_pre, _ = make_groot_pre_post_processors_from_pretrained(policy, str(source))
    assert next(s for s in raw_pre.steps if isinstance(s, GrootN17PackInputsStep)).training is True
