"""Serializable provenance and timing contract for online action targets."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

ALGORITHM_VERSION = "state-tail-so3-output-delay-v1"


@dataclass
class OnlineActionResamplingConfig:
    stretch_factor: float = 1.0
    source_revision: str = ""
    source_fps: float = 30.0
    output_fps: float = 30.0
    action_delta_offset: int = 0
    group_delays: dict[str, int] = field(default_factory=dict)
    source_contract: dict[str, Any] = field(default_factory=dict)
    selected_episodes: list[int] | None = None
    algorithm_version: str = ALGORITHM_VERSION
    statistics_support: str = "exclude_whole_padded_chunk"

    def validate(self) -> "OnlineActionResamplingConfig":
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError("unsupported saved online action resampling algorithm_version")
        if self.statistics_support != "exclude_whole_padded_chunk":
            raise ValueError("unsupported saved online action resampling statistics_support")
        for name in ("stretch_factor", "source_fps", "output_fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"online action resampling {name} must be finite and positive")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("online action resampling requires an immutable source_revision SHA")
        if self.action_delta_offset != 0:
            raise ValueError("online action resampling currently requires action_delta_offset=0")
        if not self.group_delays or any(type(v) is not int or v < 0 for v in self.group_delays.values()):
            raise ValueError("group_delays must specify non-negative integer source-frame delays")
        expected = {
            "reservoir": "state_plus_recorded_action_tail_v1",
            "recorded_action_semantics": "per_group_delay_compensated",
            "state_semantics": "pre_delay_same_frame",
            "rotation_encoding": "row_rot6d",
            "time_stretch_applied": False,
            "idm_applied": False,
        }
        for key, value in expected.items():
            actual = self.source_contract.get(key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"source_contract requires {key}={value!r}; missing pre-delay provenance")
        for key, size in (
            ("calibration_sha256", 64),
            ("pre_delay_source_revision", 40),
            ("delay_source_revision", 40),
        ):
            if not re.fullmatch(rf"[0-9a-f]{{{size}}}", str(self.source_contract.get(key, ""))):
                raise ValueError(f"source_contract requires immutable {key}")
        if self.selected_episodes is not None:
            if not self.selected_episodes or any(type(v) is not int or v < 0 for v in self.selected_episodes):
                raise ValueError("selected_episodes must be a nonempty list of non-negative episode IDs")
            if len(set(self.selected_episodes)) != len(self.selected_episodes):
                raise ValueError("selected_episodes contains duplicate episode IDs")
        json.dumps(self.source_contract, allow_nan=False)
        return self

    def identity(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["delay_units"] = "source_frames_applied_in_output_time"
        if value["selected_episodes"] is not None:
            value["selected_episodes"] = sorted(value["selected_episodes"])
        return value

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
