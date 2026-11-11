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

"""CPU tests for ``VideoDecoderCache(decoder_factory=...)`` using fake decoders (no torchcodec)."""

import importlib.util

import pytest

pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("pyarrow")

from lerobot.datasets.video_utils import VideoDecoderCache  # noqa: E402

pytestmark = pytest.mark.dataset


class _FakeDecoder:
    def __init__(self, handle):
        self.handle = handle


def _make_files(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"clip_{i}.mp4"
        p.write_bytes(b"not-a-real-video")
        paths.append(p)
    return paths


def test_factory_receives_open_binary_handle_without_torchcodec(tmp_path, monkeypatch):
    # Pretend torchcodec is absent: an injected factory must not need it.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "torchcodec" else real_find_spec(name, *a, **k),
    )
    seen = []

    def factory(handle):
        assert not handle.closed
        seen.append(handle.read(4))
        return _FakeDecoder(handle)

    cache = VideoDecoderCache(max_size=2, decoder_factory=factory)
    (path,) = _make_files(tmp_path, 1)
    dec = cache.get_decoder(path)
    assert isinstance(dec, _FakeDecoder)
    assert seen == [b"not-"]
    assert cache.get_decoder(path) is dec  # cached, factory not called again
    assert len(seen) == 1


def test_default_without_torchcodec_still_raises(tmp_path, monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "torchcodec" else real_find_spec(name, *a, **k),
    )
    (path,) = _make_files(tmp_path, 1)
    with pytest.raises(ImportError, match="torchcodec"):
        VideoDecoderCache(max_size=2).get_decoder(path)


def test_lru_eviction_closes_handles_with_factory(tmp_path):
    paths = _make_files(tmp_path, 3)
    cache = VideoDecoderCache(max_size=2, decoder_factory=_FakeDecoder)
    d0 = cache.get_decoder(paths[0])
    cache.get_decoder(paths[1])
    cache.get_decoder(paths[0])  # promote 0; 1 is now LRU
    cache.get_decoder(paths[2])  # evicts 1
    assert cache.size() == 2
    assert paths[0] in cache and paths[2] in cache and paths[1] not in cache
    assert not d0.handle.closed
    cache.clear()
    assert d0.handle.closed
    assert cache.size() == 0


def test_evicted_handle_is_closed(tmp_path):
    paths = _make_files(tmp_path, 2)
    cache = VideoDecoderCache(max_size=1, decoder_factory=_FakeDecoder)
    d0 = cache.get_decoder(paths[0])
    cache.get_decoder(paths[1])
    assert d0.handle.closed


def test_factory_failure_closes_handle_and_does_not_cache(tmp_path):
    handles = []

    def bad_factory(handle):
        handles.append(handle)
        raise RuntimeError("boom")

    cache = VideoDecoderCache(max_size=2, decoder_factory=bad_factory)
    (path,) = _make_files(tmp_path, 1)
    with pytest.raises(RuntimeError, match="boom"):
        cache.get_decoder(path)
    assert handles[0].closed
    assert cache.size() == 0
