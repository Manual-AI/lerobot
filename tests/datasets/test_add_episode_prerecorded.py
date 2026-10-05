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

"""Tests for saving episodes whose videos were encoded ahead of time (VideoFileEncoder +
LeRobotDataset.add_episode)."""

import json

import numpy as np
import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av  # noqa: E402

from lerobot.configs import RGBEncoderConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.video_utils import (  # noqa: E402
    StreamingVideoEncoder,
    VideoFileEncoder,
)

CAM = "observation.images.cam"
FEATURES = {
    CAM: {"dtype": "video", "shape": (64, 96, 3), "names": ["height", "width", "channels"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}


def _encoder():
    return RGBEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p", g=2, crf=30, preset=13)


def _episode(seed, n=12):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (n, 64, 96, 3), dtype=np.uint8)
    action = rng.standard_normal((n, 2)).astype(np.float32)
    return frames, action


def _decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def _create(root, streaming):
    return LeRobotDataset.create(
        repo_id="test/prerecorded",
        fps=30,
        features=FEATURES,
        root=root,
        use_videos=True,
        streaming_encoding=streaming,
        rgb_encoder=_encoder(),
    )


def _add_streamed(ds, frames, action):
    for img, a in zip(frames, action, strict=True):
        ds.add_frame({CAM: img, "action": a, "task": "t"})
    ds.save_episode()


def _add_prerecorded(ds, frames, action, tmp_path, name):
    enc = VideoFileEncoder(tmp_path / name / "cam.mp4", 30, _encoder())
    for img in frames:
        enc.add(img)
    stats = enc.close()
    ds.add_episode({"action": action}, ["t"] * len(frames), {CAM: (enc.video_path, stats)})
    return enc.video_path


def test_video_file_encoder_matches_the_streaming_thread(tmp_path):
    frames, _ = _episode(0)
    streaming = StreamingVideoEncoder(fps=30, rgb_encoder=_encoder())
    (tmp_path / "s").mkdir()
    streaming.start_episode([CAM], tmp_path / "s")
    for img in frames:
        streaming.feed_frame(CAM, img)
    s_path, s_stats = streaming.finish_episode()[CAM]
    streaming.close()

    enc = VideoFileEncoder(tmp_path / "f" / "cam.mp4", 30, _encoder())
    for img in frames:
        enc.add(img)
    f_stats = enc.close()

    np.testing.assert_array_equal(np.stack(_decode(s_path)), np.stack(_decode(enc.video_path)))
    assert s_stats.keys() == f_stats.keys()
    for k in s_stats:
        np.testing.assert_array_equal(s_stats[k], f_stats[k])


def test_add_episode_equals_streaming_the_same_frames(tmp_path):
    eps = [_episode(1), _episode(2, n=9)]
    streamed = _create(tmp_path / "streamed", streaming=True)
    pre = _create(tmp_path / "pre", streaming=False)
    for i, (frames, action) in enumerate(eps):
        _add_streamed(streamed, frames, action)
        _add_prerecorded(pre, frames, action, tmp_path, f"ep{i}")
    streamed.finalize()
    pre.finalize()

    a = LeRobotDataset("test/prerecorded", root=tmp_path / "streamed", video_backend="pyav")
    b = LeRobotDataset("test/prerecorded", root=tmp_path / "pre", video_backend="pyav")
    assert len(a) == len(b) == sum(len(f) for f, _ in eps)
    for i in range(len(a)):
        x, y = a[i], b[i]
        for key in (CAM, "action", "timestamp", "frame_index", "episode_index", "index"):
            np.testing.assert_array_equal(np.asarray(x[key]), np.asarray(y[key]), err_msg=f"{i} {key}")
    stats_a = json.loads((tmp_path / "streamed/meta/stats.json").read_text())
    stats_b = json.loads((tmp_path / "pre/meta/stats.json").read_text())
    assert stats_a == stats_b


def test_mixed_streamed_and_prerecorded_episodes_load(tmp_path):
    eps = [_episode(3), _episode(4, n=7), _episode(5, n=10)]
    ds = _create(tmp_path / "mixed", streaming=True)
    _add_streamed(ds, *eps[0])
    src = _add_prerecorded(ds, *eps[1], tmp_path, "ep1")
    _add_streamed(ds, *eps[2])
    ds.finalize()

    assert src.exists(), "the caller's file must be copied, not moved"
    loaded = LeRobotDataset("test/prerecorded", root=tmp_path / "mixed", video_backend="pyav")
    assert loaded.meta.total_episodes == 3
    i = 0
    for ep, (frames, action) in enumerate(eps):
        for j in range(len(frames)):
            item = loaded[i]
            assert int(item["episode_index"]) == ep and int(item["frame_index"]) == j
            np.testing.assert_array_equal(np.asarray(item["action"]), action[j])
            i += 1


def test_add_episode_rejects_bad_input(tmp_path):
    frames, action = _episode(6)
    ds = _create(tmp_path / "bad", streaming=False)
    with pytest.raises(ValueError, match="video features"):
        ds.add_episode({"action": action}, ["t"] * len(frames), {})
    with pytest.raises(ValueError, match="'action' has shape"):
        ds.add_episode({"action": action[:, :1]}, ["t"] * len(frames), {CAM: (tmp_path / "x.mp4", None)})
    with pytest.raises(ValueError, match="missing"):
        ds.add_episode({}, ["t"] * len(frames), {CAM: (tmp_path / "x.mp4", None)})
