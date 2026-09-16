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

import pytest
import torch

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

LEFT_IMAGE = "observation.images.left"
RIGHT_IMAGE = "observation.images.right"


def make_config(*, separate_encoders: bool) -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            LEFT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            RIGHT_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 80, 96)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=2,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        n_groups=8,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        num_train_timesteps=2,
        num_inference_steps=1,
        use_separate_rgb_encoder_per_camera=separate_encoders,
        device="cpu",
    )


def test_separate_encoders_support_mixed_camera_resolutions():
    policy = DiffusionPolicy(make_config(separate_encoders=True))
    batch = {
        OBS_STATE: torch.randn(1, 2, 6),
        LEFT_IMAGE: torch.randn(1, 2, 3, 64, 64),
        RIGHT_IMAGE: torch.randn(1, 2, 3, 80, 96),
        ACTION: torch.randn(1, 8, 4),
        "action_is_pad": torch.zeros(1, 8, dtype=torch.bool),
    }

    loss, _ = policy(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    observations = {key: value for key, value in batch.items() if key not in {ACTION, "action_is_pad"}}
    actions = policy.predict_action_chunk(observations, noise=torch.randn(1, 8, 4))
    assert actions.shape == (1, 2, 4)


def test_shared_encoder_rejects_mixed_camera_resolutions():
    config = make_config(separate_encoders=False)

    with pytest.raises(ValueError, match="does not match"):
        config.validate_features()
