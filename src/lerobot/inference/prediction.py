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

"""Shared post-preprocessing chunk prediction for local and remote policy owners."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from lerobot.policies import PreTrainedPolicy, RelativeRTCPrefixEncoder, reanchor_relative_rtc_prefix
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, RelativeActionsProcessorStep

from .contracts import ChunkPolicySpec, ExecutionMode


class _RTCPredictActionChunk(Protocol):
    # Only policies whose preprocessor encodes its own relative prefix
    # (``RelativeRTCPrefixEncoder``) receive ``prev_chunk_left_over_reanchored``.
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
        prev_chunk_left_over_reanchored: bool = ...,
    ) -> torch.Tensor: ...


def find_relative_rtc_prefix_encoder(
    preprocessor: PolicyProcessorPipeline,
) -> RelativeRTCPrefixEncoder | None:
    """Return the preprocessor step that encodes this policy's relative RTC prefix, if any.

    Such policies (GR00T N1.7's per-horizon stats rows and SE(3) end-effector deltas)
    re-anchor the canonical continuation themselves instead of using the generic
    ``RelativeActionsProcessorStep`` path.
    """
    return next(
        (
            step
            for step in preprocessor.steps
            if isinstance(step, RelativeRTCPrefixEncoder) and step.uses_relative_rtc_prefix()
        ),
        None,
    )


@contextmanager
def chunk_inference_context(mode: ExecutionMode) -> Iterator[None]:
    """Disable gradients while allowing guided RTC to enable its correction graph.

    Policy owners also use this context for preprocessing, so guided RTC receives
    ordinary tensors rather than tensors created under inference mode.
    """
    with torch.inference_mode(mode is not ExecutionMode.RTC_GUIDED), torch.no_grad():
        yield


@dataclass(frozen=True)
class ChunkPrediction:
    """Validated unbatched actions on the policy device, before transport packing.

    Model coordinates are detached and cloned before canonical postprocessing can
    mutate them. ``predicted_at`` separates policy and postprocessing diagnostics.
    """

    model_actions: torch.Tensor
    canonical_actions: torch.Tensor
    predicted_at: float


def _encode_relative_continuation(
    canonical: torch.Tensor | None,
    *,
    steps: int,
    model_action_dim: int,
    horizon: int,
    device: str | torch.device,
    prefix_encoder: RelativeRTCPrefixEncoder,
) -> torch.Tensor | None:
    """Re-anchor the canonical continuation through the policy's own prefix encoder.

    Must run after preprocessing, so the encoder re-anchors against the state of the
    observation this chunk is predicted from. Returns ``None`` when the encoder declines,
    in which case the chunk is predicted without prefix guidance.

    Only the real continuation rows are encoded, without padding a short tail up to
    ``horizon``: policies that own their prefix (GR00T N1.7's native action overlap)
    treat every row as a hard constraint, so padded hold rows would pin the new chunk
    in place, while a shorter prefix only shortens the overlap.
    """
    if (
        canonical is None
        or canonical.ndim != 2
        or len(canonical) != steps
        or not canonical.is_floating_point()
        or not torch.isfinite(canonical).all()
    ):
        raise ValueError("Relative RTC requires a row-aligned finite canonical continuation.")
    encoded = prefix_encoder.encode_absolute_rtc_prefix(canonical[:horizon])
    if encoded is None:
        return None
    if encoded.ndim == 3 and encoded.shape[0] == 1:
        encoded = encoded[0]
    if (
        encoded.ndim != 2
        or not 0 < len(encoded) <= horizon
        or encoded.shape[1] != model_action_dim
        or not encoded.is_floating_point()
        or not torch.isfinite(encoded).all()
    ):
        raise ValueError("Relative RTC prefix encoder returned an invalid model-space prefix.")
    return encoded.to(device)


def _prepare_continuation(
    model: torch.Tensor | None,
    canonical: torch.Tensor | None,
    *,
    model_action_dim: int,
    horizon: int,
    device: str | torch.device,
    relative_step: RelativeActionsProcessorStep | None,
    normalizer_step: NormalizerProcessorStep | None,
    prefix_encoder: RelativeRTCPrefixEncoder | None = None,
) -> torch.Tensor | None:
    if model is None or model.numel() == 0:
        return None
    if (
        model.ndim != 2
        or model.shape[1] != model_action_dim
        or not model.is_floating_point()
        or not torch.isfinite(model).all()
    ):
        raise ValueError("Invalid model-space continuation.")
    if prefix_encoder is not None:
        # The model-space rows are deltas against the previous observation; only the
        # canonical (absolute) tail can be re-anchored to this one.
        return _encode_relative_continuation(
            canonical,
            steps=len(model),
            model_action_dim=model_action_dim,
            horizon=horizon,
            device=device,
            prefix_encoder=prefix_encoder,
        )
    if relative_step is not None:
        if canonical is None or canonical.shape != model.shape or not torch.isfinite(canonical).all():
            raise ValueError("Relative RTC requires matching finite canonical continuation.")
        state = relative_step.get_cached_state()
        if state is None:
            raise ValueError("Relative RTC preprocessor did not cache the current raw state.")
        model = reanchor_relative_rtc_prefix(canonical, state, relative_step, normalizer_step, device)
    model = model.to(device)
    if len(model) < horizon:
        # Zero in normalized coordinates decodes to the dataset mean, not hold.
        model = torch.cat([model, model[-1:].expand(horizon - len(model), -1)])
    return model[:horizon]


def predict_chunk(
    policy: PreTrainedPolicy,
    postprocessor: PolicyProcessorPipeline,
    prepared: dict[str, Any],
    *,
    spec: ChunkPolicySpec,
    mode: ExecutionMode,
    canonical_action_dim: int,
    model_action_dim: int | None,
    device: str | torch.device,
    rtc_horizon: int = 0,
    inference_delay: int = 0,
    model_continuation: torch.Tensor | None = None,
    canonical_continuation: torch.Tensor | None = None,
    relative_step: RelativeActionsProcessorStep | None = None,
    normalizer_step: NormalizerProcessorStep | None = None,
    prefix_encoder: RelativeRTCPrefixEncoder | None = None,
    prefix_guidance: bool = True,
) -> ChunkPrediction:
    """Predict and validate under exclusive policy/processor ownership.

    Retain the first model width across calls. Canonical postprocessing may crop
    padding; relative RTC requires matching widths unless ``prefix_encoder`` encodes
    the prefix. ``prefix_guidance=False`` predicts RTC chunks without conditioning on
    the continuation. This function owns no scheduling.
    """
    with chunk_inference_context(mode):
        if mode is ExecutionMode.CHUNK:
            predicted = policy.predict_action_chunk(prepared)
        else:
            if not prefix_guidance:
                if mode is ExecutionMode.RTC_TRAINED:
                    raise ValueError("Trained RTC requires prefix guidance.")
                # Ablation: keep asynchronous chunk replacement, drop prefix conditioning.
                model_continuation = canonical_continuation = None
                inference_delay = 0
            prefix = _prepare_continuation(
                model_continuation,
                canonical_continuation,
                model_action_dim=model_action_dim or canonical_action_dim,
                horizon=rtc_horizon,
                device=device,
                relative_step=relative_step,
                normalizer_step=normalizer_step,
                prefix_encoder=prefix_encoder,
            )
            if inference_delay < 0:
                raise ValueError("RTC delay must be nonnegative.")
            refused = (
                prefix is None
                and prefix_encoder is not None
                and model_continuation is not None
                and model_continuation.numel() > 0
            )
            if refused:
                # The encoder declined this continuation: predict unguided, so neither the
                # delay nor trained-mode limits may refer to the refused prefix.
                model_continuation = None
                inference_delay = 0
            if prefix is None and inference_delay:
                raise ValueError("RTC delay requires a real continuation prefix.")
            if mode is ExecutionMode.RTC_TRAINED:
                available = 0 if model_continuation is None else len(model_continuation)
                if inference_delay > min(spec.training_max_delay, available, rtc_horizon):
                    raise ValueError("Trained RTC delay exceeds checkpoint or available continuation limits.")
            predict = cast(_RTCPredictActionChunk, policy.predict_action_chunk)
            if prefix is not None and prefix_encoder is not None:
                predicted = predict(
                    prepared,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prefix,
                    prev_chunk_left_over_reanchored=True,
                )
            else:
                predicted = predict(prepared, inference_delay=inference_delay, prev_chunk_left_over=prefix)
        predicted_at = time.perf_counter()
        if (
            not isinstance(predicted, torch.Tensor)
            or predicted.ndim != 3
            or predicted.shape[:2] != (1, spec.prediction_steps)
            or predicted.shape[2] <= 0
            or not predicted.is_floating_point()
            or not torch.isfinite(predicted).all()
        ):
            raise ValueError(
                "Policy must return finite floating actions with its declared batch and horizon."
            )
        if model_action_dim is not None and predicted.shape[2] != model_action_dim:
            raise ValueError("Policy model action width changed after initial prediction.")
        # Older local policies can omit action feature metadata. Keep their
        # existing model-width fallback; remote admission requires explicit width.
        canonical_width = canonical_action_dim or predicted.shape[2]
        if relative_step is not None and rtc_horizon and predicted.shape[2] != canonical_width:
            raise ValueError("Relative RTC with different model/canonical widths requires a custom adapter.")
        # Preserve coordinates before a postprocessor may modify its argument.
        original = predicted.detach().clone()
        canonical = postprocessor(predicted)
        expected = (1, spec.prediction_steps, canonical_width)
        if not isinstance(canonical, torch.Tensor) or tuple(canonical.shape) != expected:
            raise ValueError(f"Policy/processor must return action shape {expected}.")
        if not canonical.is_floating_point() or not torch.isfinite(canonical).all():
            raise ValueError("Policy/processor returned non-finite or non-floating actions.")
        steps = spec.execution_steps if mode is ExecutionMode.CHUNK else spec.prediction_steps
        return ChunkPrediction(original[0, :steps], canonical[0, :steps].detach(), predicted_at)
