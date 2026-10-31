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

"""How `ImageTransformConfig.type` names resolve: the `v2.` prefix, ambiguous names, and registered types."""

import importlib
import logging
import pickle
import re
import sys
import textwrap
from typing import Any

import pytest
import torch
import torchvision
from packaging import version
from torch import Tensor
from torchvision.transforms import v2

import lerobot.transforms
from lerobot.transforms import (
    BatchedGaussianNoise,
    BatchedTransform,
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
    PerSampleTransform,
    make_batched_transform_from_config,
    register_image_transform,
    transforms as tf,
)

TRANSFORMS_LOGGER = "lerobot.transforms.transforms"

# The resolver reads these module-level objects by name, so the tests mutate and restore them in place.
BATCHED_REGISTRY = tf._BATCHED_TRANSFORMS  # noqa: SLF001
PER_SAMPLE_REGISTRY = tf._PER_SAMPLE_TRANSFORMS  # noqa: SLF001
WARNED_TYPES = tf._warned_ambiguous_types  # noqa: SLF001
TORCHVISION_EQUIVALENTS = tf._TORCHVISION_EQUIVALENTS  # noqa: SLF001

# torchvision 0.24 is the first release whose `v2.GaussianNoise` accepts uint8 images (pytorch/vision#9169).
REQUIRES_UINT8_GAUSSIAN_NOISE = pytest.mark.skipif(
    version.parse(torchvision.__version__.split("+")[0]) < version.parse("0.24"),
    reason="torchvision.transforms.v2.GaussianNoise supports uint8 images from torchvision 0.24",
)

EQUIVALENT_KWARGS: dict[str, dict[str, Any]] = {
    "Identity": {},
    "ColorJitter": {"brightness": (0.8, 1.2)},
    "RandomAffine": {"degrees": (-5.0, 5.0)},
    "RandomRotation": {"degrees": (-5.0, 5.0)},
}


class BrightnessOffset(BatchedTransform):
    """Add a constant to every pixel."""

    def __init__(self, offset: float = 0.2) -> None:
        super().__init__()
        self.offset = offset

    def transform(self, frames: Tensor, params: dict[str, Any]) -> Tensor:
        return (frames + self.offset).clamp(0.0, 1.0)


class AlwaysInvert(v2.RandomApply):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__([v2.RandomInvert(p=1.0)], p=p)


@pytest.fixture(autouse=True)
def restore_registries():
    """Start every test with no ambiguity warnings issued, and undo its registrations afterwards."""
    batched, per_sample, warned = dict(BATCHED_REGISTRY), dict(PER_SAMPLE_REGISTRY), set(WARNED_TYPES)
    WARNED_TYPES.clear()
    yield
    BATCHED_REGISTRY.clear()
    BATCHED_REGISTRY.update(batched)
    PER_SAMPLE_REGISTRY.clear()
    PER_SAMPLE_REGISTRY.update(per_sample)
    WARNED_TYPES.clear()
    WARNED_TYPES.update(warned)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _frame(dtype: torch.dtype) -> Tensor:
    if dtype == torch.uint8:
        return torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8, generator=_generator(0))
    return torch.rand(3, 32, 32, generator=_generator(0), dtype=dtype)


def _single(type_name: str, kwargs: dict[str, Any] | None = None) -> ImageTransforms:
    """`ImageTransforms` that always applies exactly the one configured transform."""
    tf_cfg = ImageTransformConfig(type=type_name, kwargs=kwargs or {})
    return ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=1, tfs={"tf": tf_cfg}))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == TRANSFORMS_LOGGER and r.levelno >= logging.WARNING]


# --- The `v2.` prefix ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_name", "kwargs", "torchvision_cls"),
    [
        ("v2.GaussianNoise", {"sigma": 0.1}, v2.GaussianNoise),
        # LeRobot registers its batched copy under the bare name; the prefix still builds torchvision's.
        ("v2.ColorJitter", {"brightness": (0.8, 1.2)}, v2.ColorJitter),
    ],
)
def test_v2_prefix_builds_the_torchvision_class(type_name, kwargs, torchvision_cls):
    built = make_batched_transform_from_config(ImageTransformConfig(type=type_name, kwargs=kwargs))
    assert isinstance(built, PerSampleTransform)
    assert type(built.per_sample) is torchvision_cls


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, pytest.param(torch.uint8, marks=REQUIRES_UINT8_GAUSSIAN_NOISE)],
    ids=["float32", "uint8"],
)
def test_v2_gaussian_noise_transforms_frames(dtype):
    frame = _frame(dtype)
    out = _single("v2.GaussianNoise", {"sigma": 0.1})(frame, generator=_generator(1))
    assert out.dtype == dtype
    assert out.shape == frame.shape
    assert not torch.equal(out, frame)


@pytest.mark.parametrize(
    "type_name",
    ["v2.Nope", "v2.functional", "torchvision.transforms.v2.GaussianNoise"],
    ids=["unknown", "module", "fully-qualified"],
)
def test_unknown_v2_type_raises(type_name):
    with pytest.raises(ValueError, match=re.escape(f"'{type_name}'")):
        make_batched_transform_from_config(ImageTransformConfig(type=type_name))


# --- Names LeRobot and torchvision share ---------------------------------------------------------


def test_bare_gaussian_noise_builds_lerobots_and_warns_once(caplog):
    cfg = ImageTransformConfig(type="GaussianNoise")
    with caplog.at_level(logging.WARNING, logger=TRANSFORMS_LOGGER):
        built = [make_batched_transform_from_config(cfg) for _ in range(2)]
    assert all(isinstance(b, BatchedGaussianNoise) for b in built)
    records = _warnings(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "BatchedGaussianNoise" in message
    assert "'v2.GaussianNoise'" in message
    # The kwargs torchvision's class takes instead, and the release it needs for uint8 frames.
    assert "`sigma`" in message
    assert "0.24" in message


def test_unknown_bare_type_error_points_to_registration():
    with pytest.raises(ValueError, match="not valid") as excinfo:
        make_batched_transform_from_config(ImageTransformConfig(type="NotRegisteredAnywhere"))
    assert "register_image_transform" in str(excinfo.value)
    assert "--dataset.image_transforms.discover_packages_path" in str(excinfo.value)


@pytest.mark.parametrize("type_name", sorted(TORCHVISION_EQUIVALENTS))
def test_torchvision_equivalents_do_not_warn(caplog, type_name):
    cfg = ImageTransformConfig(type=type_name, kwargs=EQUIVALENT_KWARGS[type_name])
    with caplog.at_level(logging.WARNING, logger=TRANSFORMS_LOGGER):
        make_batched_transform_from_config(cfg)
    assert not _warnings(caplog)


def test_default_config_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=TRANSFORMS_LOGGER):
        ImageTransforms(ImageTransformsConfig(enable=True))
    assert not _warnings(caplog)


def test_every_name_shared_with_torchvision_warns(caplog):
    clashes = sorted(
        name
        for name in {*BATCHED_REGISTRY, *PER_SAMPLE_REGISTRY}
        if name not in TORCHVISION_EQUIVALENTS and tf._v2_transform_class(name) is not None  # noqa: SLF001
    )
    # A newer torchvision may add more clashes; each must warn like `GaussianNoise` does.
    assert set(clashes) >= {"GaussianNoise"}
    with caplog.at_level(logging.WARNING, logger=TRANSFORMS_LOGGER):
        for name in clashes:
            make_batched_transform_from_config(ImageTransformConfig(type=name))
    messages = [r.getMessage() for r in _warnings(caplog)]
    assert len(messages) == len(clashes)
    for name in clashes:
        assert sum(f"'v2.{name}'" in m for m in messages) == 1, name


def test_torchvision_equivalents_are_registered_torchvision_names():
    for name in TORCHVISION_EQUIVALENTS:
        assert name in BATCHED_REGISTRY
        assert tf._v2_transform_class(name) is not None, name  # noqa: SLF001


# --- `register_image_transform` ------------------------------------------------------------------


def test_registered_types_land_in_their_registry():
    register_image_transform("BrightnessOffset", BrightnessOffset)
    register_image_transform("AlwaysInvert", AlwaysInvert)
    assert BATCHED_REGISTRY["BrightnessOffset"] is BrightnessOffset
    assert PER_SAMPLE_REGISTRY["AlwaysInvert"] is AlwaysInvert
    assert "BrightnessOffset" not in PER_SAMPLE_REGISTRY
    assert "AlwaysInvert" not in BATCHED_REGISTRY

    batched = make_batched_transform_from_config(ImageTransformConfig(type="BrightnessOffset"))
    assert isinstance(batched, BrightnessOffset)
    per_sample = make_batched_transform_from_config(ImageTransformConfig(type="AlwaysInvert"))
    assert isinstance(per_sample, PerSampleTransform)
    assert isinstance(per_sample.per_sample, AlwaysInvert)


@pytest.mark.parametrize(
    ("type_name", "cls", "kwargs"),
    [("BrightnessOffset", BrightnessOffset, {"offset": 0.5}), ("AlwaysInvert", AlwaysInvert, {"p": 1.0})],
    ids=["batched", "v2"],
)
def test_registered_type_transforms_uint8_frames(type_name, cls, kwargs):
    register_image_transform(type_name, cls)
    frame = _frame(torch.uint8)
    out = _single(type_name, kwargs)(frame, generator=_generator(1))
    assert out.dtype == torch.uint8
    assert out.shape == frame.shape
    assert not torch.equal(out, frame)


def test_registered_types_survive_pickling():
    register_image_transform("BrightnessOffset", BrightnessOffset)
    register_image_transform("AlwaysInvert", AlwaysInvert)
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=2,
        tfs={
            "offset": ImageTransformConfig(type="BrightnessOffset"),
            "invert": ImageTransformConfig(type="AlwaysInvert"),
        },
    )
    transforms = ImageTransforms(cfg)
    restored = pickle.loads(pickle.dumps(transforms))
    frame = _frame(torch.uint8)
    torch.testing.assert_close(
        restored(frame, generator=_generator(1)), transforms(frame, generator=_generator(1))
    )


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("GaussianBlur", "torchvision.transforms.v2 class"),
        ("MotionBlur", "already registered"),
        ("a.b", "not valid"),
        ("", "not valid"),
    ],
    ids=["torchvision-name", "taken", "dotted", "empty"],
)
def test_register_rejects_name(name, match):
    with pytest.raises(ValueError, match=match):
        register_image_transform(name, BrightnessOffset)


def test_register_same_class_twice_is_a_no_op():
    register_image_transform("BrightnessOffset", BrightnessOffset)
    batched, per_sample = dict(BATCHED_REGISTRY), dict(PER_SAMPLE_REGISTRY)
    register_image_transform("BrightnessOffset", BrightnessOffset)
    assert batched == BATCHED_REGISTRY
    assert per_sample == PER_SAMPLE_REGISTRY


def _same_named_class(cls: type, base: type) -> type:
    """A new class object with `cls`'s `__module__` and `__qualname__`, as importing its module again makes."""
    return type(cls.__name__, (base,), {"__module__": cls.__module__, "__qualname__": cls.__qualname__})


def test_reloading_the_registering_module_points_the_name_at_the_new_class(tmp_path, monkeypatch):
    module_name = "lerobot_test_reloaded_image_transform"
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from lerobot.transforms import BatchedTransform, register_image_transform


            class ReloadedOffset(BatchedTransform):
                def transform(self, frames, params):
                    return frames


            register_image_transform("ReloadedOffset", ReloadedOffset)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module(module_name)
        first = module.ReloadedOffset
        module = importlib.reload(module)
    finally:
        sys.modules.pop(module_name, None)
    assert module.ReloadedOffset is not first
    assert BATCHED_REGISTRY["ReloadedOffset"] is module.ReloadedOffset
    built = make_batched_transform_from_config(ImageTransformConfig(type="ReloadedOffset"))
    assert type(built) is module.ReloadedOffset


def test_reregistering_a_same_named_class_can_change_its_kind():
    register_image_transform("Offset", BrightnessOffset)
    per_sample_twin = _same_named_class(BrightnessOffset, AlwaysInvert)
    register_image_transform("Offset", per_sample_twin)
    assert "Offset" not in BATCHED_REGISTRY
    assert PER_SAMPLE_REGISTRY["Offset"] is per_sample_twin


def test_register_different_class_under_taken_name_names_both_classes():
    register_image_transform("BrightnessOffset", BrightnessOffset)
    # Same `__name__` and `__qualname__`, different module: not the same class reloaded.
    lookalike = type("BrightnessOffset", (BatchedTransform,), {"__module__": "elsewhere"})
    with pytest.raises(ValueError, match="already registered") as excinfo:
        register_image_transform("BrightnessOffset", lookalike)
    assert f"{BrightnessOffset.__module__}.BrightnessOffset" in str(excinfo.value)
    assert "elsewhere.BrightnessOffset" in str(excinfo.value)
    assert BATCHED_REGISTRY["BrightnessOffset"] is BrightnessOffset


@pytest.mark.parametrize("name", [123, None, b"BrightnessOffset"], ids=["int", "None", "bytes"])
def test_register_rejects_non_str_name(name):
    with pytest.raises(TypeError, match="str"):
        register_image_transform(name, BrightnessOffset)


@pytest.mark.parametrize("cls", [int, torch.nn.Module], ids=["int", "nn.Module"])
def test_register_rejects_non_transform_class(cls):
    with pytest.raises(TypeError, match="BatchedTransform"):
        register_image_transform("NotATransform", cls)


def test_register_image_transform_is_public():
    assert register_image_transform is tf.register_image_transform
    assert "register_image_transform" in lerobot.transforms.__all__
