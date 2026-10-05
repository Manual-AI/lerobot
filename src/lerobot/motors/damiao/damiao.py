# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

# Portions of this file are derived from DM_Control_Python by cmjang.
# Licensed under the MIT License; see `LICENSE` for the full text:
# https://github.com/cmjang/DM_Control_Python

import logging
import math
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypedDict

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING or _can_available:
    import can
else:

    class can:  # noqa: N801
        Message = object
        interface = None


import numpy as np

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_CMD_SET_ZERO,
    CAN_PARAM_ID,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)

logger = logging.getLogger(__name__)


LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01
SHORT_TIMEOUT_SEC = 0.001
PRECISE_TIMEOUT_SEC = 0.0001

# Default bounded wait for this tick's refresh replies, after the queue has been drained (settable
# per bus as `state_wait_s`). The replies to N refreshes need about 2 * N * 130 us on classic 1 Mbps
# CAN, so 2 ms is tight for 7-8 motors there; a reply that misses it is reported with its true age.
STATE_WAIT_S = 0.002
# Max frames consumed by one drain phase, so a flooded bus can't stall a tick.
DRAIN_CAP = 4096
# Max wall time of the non-blocking pre-send drain (GIL contention can make each recv slow).
DRAIN_MAX_S = 0.005
# A reply's rx timestamp may precede the host's send timestamp by this much (clock granularity).
FRESH_EPS_S = 0.0005
# An rx timestamp further than this from `time.time()` is not epoch time; host receipt time is used.
PLAUSIBLE_TS_SKEW_S = 10.0
# Minimum interval between repeats of one kind of warning on one bus.
STALE_WARN_INTERVAL_S = 1.0


class MotorState(TypedDict):
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float
    # Kernel rx time of the frame this state was decoded from, in epoch seconds
    # (`can.Message.timestamp`); 0.0 until the motor has replied once.
    timestamp: float


class DamiaoMotorsBus(MotorsBusBase):
    """
    The Damiao implementation for a MotorsBus using CAN bus communication.

    This class uses python-can for CAN bus communication with Damiao motors.
    For more info, see:
    - python-can documentation: https://python-can.readthedocs.io/en/stable/
    - Seedstudio documentation: https://wiki.seeedstudio.com/damiao_series/
    - DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    """

    # CAN-specific settings
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "auto",
        use_can_fd: bool = True,
        bitrate: int = 1000000,
        data_bitrate: int | None = 5000000,
    ):
        """
        Initialize the Damiao motors bus.

        Args:
            port: CAN interface name (e.g., "can0" for Linux, "/dev/cu.usbmodem*" for macOS)
            motors: Dictionary mapping motor names to Motor objects
            calibration: Optional calibration data
            can_interface: CAN interface type - "auto" (default), "socketcan" (Linux), or "slcan" (macOS/serial)
            use_can_fd: Whether to use CAN FD mode (default: True for OpenArms)
            bitrate: Nominal bitrate in bps (default: 1000000 = 1 Mbps)
            data_bitrate: Data bitrate for CAN FD in bps (default: 5000000 = 5 Mbps), ignored if use_can_fd is False
        """
        require_package("python-can", extra="damiao", import_name="can")
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        self.canbus: can.interface.Bus | None = None
        self._is_connected = False

        # Map motor names to CAN IDs
        self._motor_can_ids: dict[str, int] = {}
        self._recv_id_to_motor: dict[int, str] = {}
        self._motor_types: dict[str, MotorType] = {}

        for name, motor in self.motors.items():
            if motor.motor_type_str is None:
                raise ValueError(f"Motor '{name}' is missing required 'motor_type'")
            self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper().replace("-", "_"))

            # Map recv_id to motor name for filtering responses
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name

        # State cache for handling packet drops safely
        self._last_known_states: dict[str, MotorState] = {
            name: {
                "position": 0.0,
                "velocity": 0.0,
                "torque": 0.0,
                "temp_mos": 0.0,
                "temp_rotor": 0.0,
                "timestamp": 0.0,
            }
            for name in self.motors
        }

        # Age (s) above which a reading is reported as stale; owners may set it to e.g. 2 / fps.
        self.stale_warn_s: float = 2 / 30
        # Bounded wait (s) for refresh replies in `sync_read_all_states` / `sync_read` / `read`.
        self.state_wait_s: float = STATE_WAIT_S
        # Per motor: refreshes whose reply missed the wait window (state fell back to an older reading).
        self.refresh_miss_count: dict[str, int] = dict.fromkeys(self.motors, 0)
        # Per motor: seconds from sending the last refresh to its reply's rx timestamp (the first
        # fresh frame), or NaN if that refresh was missed. Measures what `state_wait_s` must cover.
        self.last_refresh_latency_s: dict[str, float] = dict.fromkeys(self.motors, math.nan)
        # Rx timestamp of the first fresh frame per recv ID in the last `_drain_newest` with a send.
        self._first_fresh_ts: dict[int, float] = {}
        self._last_send_wall = 0.0
        self._last_warn: dict[str, float] = {}
        self._warned_host_ts = False

        # Dynamic gains storage
        # Defaults: Kp=10.0 (Stiffness), Kd=0.5 (Damping)
        self._gains: dict[str, dict[str, float]] = {name: {"kp": 10.0, "kd": 0.5} for name in self.motors}

    @property
    def is_connected(self) -> bool:
        """Check if the CAN bus is connected."""
        return self._is_connected and self.canbus is not None

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """
        Open the CAN bus and initialize communication.

        Args:
            handshake: If True, ping all motors to verify they're present
        """

        try:
            # Auto-detect interface type based on port name
            if self.can_interface == "auto":
                if self.port.startswith("/dev/"):
                    self.can_interface = "slcan"
                    logger.info(f"Auto-detected slcan interface for port {self.port}")
                else:
                    self.can_interface = "socketcan"
                    logger.info(f"Auto-detected socketcan interface for port {self.port}")

            # Connect to CAN bus
            kwargs = {
                "channel": self.port,
                "bitrate": self.bitrate,
                "interface": self.can_interface,
            }

            if self.can_interface == "socketcan" and self.use_can_fd and self.data_bitrate is not None:
                kwargs.update({"data_bitrate": self.data_bitrate, "fd": True})
                logger.info(
                    f"Connected to {self.port} with CAN FD (bitrate={self.bitrate}, data_bitrate={self.data_bitrate})"
                )
            else:
                logger.info(f"Connected to {self.port} with {self.can_interface} (bitrate={self.bitrate})")

            self.canbus = can.interface.Bus(**kwargs)
            self._is_connected = True

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            if self.canbus is not None:
                try:
                    self.canbus.shutdown()
                except Exception as shutdown_error:
                    logger.debug(f"Error closing CAN bus after failed connect: {shutdown_error}")
                self.canbus = None
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """
        Verify all motors are present and populate initial state cache.
        Raises ConnectionError if any motor fails to respond.
        """
        logger.info("Starting handshake with motors...")

        bus = self.canbus
        if bus is None:
            raise RuntimeError("CAN bus is not initialized.")

        missing_motors = []
        for motor_name in self.motors:
            motor_id = self._get_motor_id(motor_name)
            recv_id = self._get_motor_recv_id(motor_name)

            # Send enable command and wait (longer timeout) for a reply sent after it
            data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, CAN_CMD_ENABLE]
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            _, fresh = self._drain_newest([recv_id], send=lambda m=msg: bus.send(m), wait_s=LONG_TIMEOUT_SEC)

            if recv_id in fresh:
                self._process_response(motor_name, fresh[recv_id])
            else:
                missing_motors.append(motor_name)
            time.sleep(MEDIUM_TIMEOUT_SEC)

        if missing_motors:
            raise ConnectionError(
                f"Handshake failed. The following motors did not respond: {missing_motors}. "
                "Check power (24V) and CAN wiring."
            )
        logger.info("Handshake successful. All motors ready.")

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        """
        Close the CAN bus connection.

        Args:
            disable_torque: If True, disable torque on all motors before disconnecting
        """

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        if self.canbus:
            self.canbus.shutdown()
            self.canbus = None
        self._is_connected = False
        logger.debug(f"{self.__class__.__name__} disconnected.")

    def configure_motors(self) -> None:
        """Configure all motors with default settings."""
        # Damiao motors don't require much configuration in MIT mode
        # Just ensure they're enabled
        for motor in self.motors:
            self._send_simple_command(motor, CAN_CMD_ENABLE)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _send_simple_command(self, motor: NameOrID, command_byte: int) -> None:
        """Helper to send simple 8-byte commands (Enable, Disable, Zero)."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [command_byte]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        bus = self.canbus
        if bus is None:
            raise RuntimeError("CAN bus is not initialized.")

        _, fresh = self._drain_newest([recv_id], send=lambda: bus.send(msg), wait_s=SHORT_TIMEOUT_SEC)
        if reply := fresh.get(recv_id):
            self._process_response(motor_name, reply)
        else:
            logger.debug(f"No response from {motor_name} after command 0x{command_byte:02X}")

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_ENABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            for _ in range(num_retry + 1):
                try:
                    self._send_simple_command(motor, CAN_CMD_DISABLE)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        """
        Context manager that guarantees torque is re-enabled.

        This helper is useful to temporarily disable torque when configuring motors.
        """
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """Set current position as zero for selected motors."""
        target_motors = self._get_motors_list(motors)
        for motor in target_motors:
            self._send_simple_command(motor, CAN_CMD_SET_ZERO)
            time.sleep(MEDIUM_TIMEOUT_SEC)

    def _refresh_motor(self, motor: NameOrID) -> can.Message | None:
        """Refresh one motor and return a fresh reply (see `_drain_newest`) within `state_wait_s`, else None.

        Older frames still queued from that motor are drained and discarded, so a dead motor is never
        masked by a stale frame.
        """
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
        msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        bus = self.canbus
        if bus is None:
            raise RuntimeError("CAN bus is not initialized.")

        _, fresh = self._drain_newest([recv_id], send=lambda: bus.send(msg), wait_s=self.state_wait_s)
        return fresh.get(recv_id)

    def _drain_newest(
        self,
        expected_ids: Iterable[int],
        send: Callable[[], None] | None = None,
        wait_s: float = 0.0,
    ) -> tuple[dict[int, can.Message], dict[int, can.Message]]:
        """
        The one receive path for every Damiao read and write: drain the rx queue newest-wins,
        optionally send a request, then wait (bounded) for replies to it.

        1. Non-blocking drain: `recv(timeout=0)` until the queue is empty, keeping the newest frame
           per expected ID. Older duplicates and unexpected IDs are consumed and dropped, so a backlog
           can't build up across calls. The drain stops early after `DRAIN_CAP` frames or
           `DRAIN_MAX_S` of wall time (rate-limited warning); the rest is consumed on later calls.
        2. If `send` is given it is called once, outside any error handling, so TX errors propagate.
        3. Wait until `wait_s` has elapsed or every expected ID has a fresh frame, consuming at most
           `DRAIN_CAP` more frames. If the wait ran out first, drain once more without blocking:
           a reply that reached the kernel in time but was not read because this thread woke late
           (GIL contention under load) is still taken, so a miss means no reply had arrived. A frame is *fresh* if it was dequeued after `send` and its rx
           timestamp is not older than the send time (minus `FRESH_EPS_S`), so a backlog left over by a
           capped drain is never mistaken for a reply. Freshness is by ID: a late MIT reply from the
           same motor that lands after the send counts too (both reply types share the recv ID and
           carry the same state layout, each with its own honest rx timestamp).

        Every kept frame's `timestamp` is normalized: interfaces without usable rx timestamps (0, or
        more than `PLAUSIBLE_TS_SKEW_S` away from `time.time()`, e.g. a device-relative clock) get the
        host receipt time instead, with a one-time warning. slcan stamps frames when they are parsed,
        so there the age hides time spent queued in the host.

        Not thread-safe: one bus must only be used from one thread at a time.

        Args:
            expected_ids: CAN recv IDs whose frames to keep.
            send: Optional callable that transmits the request(s).
            wait_s: Maximum time to wait for fresh frames. 0 means drain only.

        Returns:
            `(newest, fresh)`: the newest frame per expected ID seen in any phase, and the subset of
            those that are fresh (empty if `send` is None).
        """
        bus = self.canbus
        if bus is None:
            raise RuntimeError("CAN bus is not initialized.")

        expected = set(expected_ids)
        newest: dict[int, can.Message] = {}
        fresh: dict[int, can.Message] = {}

        def keep(msg: can.Message) -> None:
            self._normalize_timestamp(msg)
            newest[msg.arbitration_id] = msg

        drain_deadline = time.perf_counter() + DRAIN_MAX_S
        consumed = 0
        while True:
            msg = self._recv(bus, 0)
            if msg is None:
                break
            consumed += 1
            if msg.arbitration_id in expected:
                keep(msg)
            if consumed >= DRAIN_CAP or time.perf_counter() >= drain_deadline:
                self._rate_limited_warning(
                    "flood",
                    f"CAN rx flood on {self.port}: drained {consumed} frames in "
                    f"{DRAIN_MAX_S * 1e3:.0f} ms without emptying the queue.",
                )
                break

        if send is None:
            return newest, fresh
        send_wall = time.time()
        self._last_send_wall = send_wall
        self._first_fresh_ts = {}
        send()

        def take(msg: can.Message) -> None:
            if msg.arbitration_id not in expected:
                return
            keep(msg)
            if msg.timestamp >= send_wall - FRESH_EPS_S:
                fresh[msg.arbitration_id] = msg
                self._first_fresh_ts.setdefault(msg.arbitration_id, msg.timestamp)

        deadline = time.perf_counter() + wait_s
        consumed = 0
        while consumed < DRAIN_CAP and not expected.issubset(fresh):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            msg = self._recv(bus, remaining)
            if msg is None:
                break
            consumed += 1
            take(msg)

        # The wait ran out: take whatever is already queued, without blocking.
        while consumed < DRAIN_CAP and not expected.issubset(fresh):
            msg = self._recv(bus, 0)
            if msg is None:
                break
            consumed += 1
            take(msg)

        return newest, fresh

    def _recv(self, bus: "can.BusABC", timeout: float) -> can.Message | None:
        """`bus.recv`, turning receive errors into a rate-limited warning and `None`."""
        try:
            return bus.recv(timeout=timeout)
        except Exception as e:
            self._rate_limited_warning("recv", f"CAN receive error on {self.port}: {e}")
            return None

    def _normalize_timestamp(self, msg: can.Message) -> None:
        """Replace a missing or implausible rx timestamp with the host receipt time (warns once)."""
        now = time.time()
        if msg.timestamp and abs(now - msg.timestamp) <= PLAUSIBLE_TS_SKEW_S:
            return
        if not self._warned_host_ts:
            self._warned_host_ts = True
            logger.warning(
                f"CAN interface on {self.port} gives no usable rx timestamps (got {msg.timestamp!r}); "
                "state ages use host receipt time and hide time spent queued in the driver."
            )
        msg.timestamp = now

    def _rate_limited_warning(self, key: str, text: str) -> None:
        """Log `text` at most once per `STALE_WARN_INTERVAL_S` per `key` on this bus."""
        now = time.monotonic()
        if now - self._last_warn.get(key, -math.inf) < STALE_WARN_INTERVAL_S:
            return
        self._last_warn[key] = now
        logger.warning(text)

    def _encode_mit_packet(
        self,
        motor_type: MotorType,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> list[int]:
        """Helper to encode control parameters into 8 bytes for MIT mode."""
        # Convert degrees to radians
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Encode parameters
        kp_uint = self._float_to_uint(kp, *MIT_KP_RANGE, 12)
        kd_uint = self._float_to_uint(kd, *MIT_KD_RANGE, 12)
        q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
        dq_uint = self._float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
        tau_uint = self._float_to_uint(torque, -tmax, tmax, 12)

        # Pack data
        data = [0] * 8
        data[0] = (q_uint >> 8) & 0xFF
        data[1] = q_uint & 0xFF
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        data[4] = kp_uint & 0xFF
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        data[7] = tau_uint & 0xFF
        return data

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> None:
        """Send MIT control command to a motor."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types[motor_name]

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
        self.canbus.send(msg)

        # MIT replies are only drained (no wait) so they can't queue up; the cache is updated by
        # refreshes. A late MIT reply can still be picked up by the next refresh (see `_drain_newest`).
        self._drain_newest(self._recv_id_to_motor)

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> None:
        """
        Send MIT control commands to multiple motors in batch.
        Sends all commands, then drains (without waiting) any queued replies so they can't
        accumulate. This call never updates the state cache; that comes from refreshes in
        `sync_read_all_states` / `sync_read`. An MIT reply that lands after this drain shares its
        recv ID with refresh replies, so the next refresh may use it (as a fresh or fallback frame,
        with its own true rx timestamp).

        Args:
            commands: Dict mapping motor name/ID to (kp, kd, position_deg, velocity_deg/s, torque)
                     Example: {'joint_1': (10.0, 0.5, 45.0, 0.0, 0.0), ...}
        """
        if not commands:
            return

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        # Step 1: Send all MIT control commands
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_id = self._get_motor_id(motor)
            motor_name = self._get_motor_name(motor)
            motor_type = self._motor_types[motor_name]

            data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            self.canbus.send(msg)

        # Step 2: Drain queued replies (newest-wins, no wait); they are not the observation source
        self._drain_newest(self._recv_id_to_motor)

    def _float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float to unsigned integer for CAN transmission."""
        x = max(x_min, min(x_max, x))  # Clamp to range
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def _uint_to_float(self, x: int, x_min: float, x_max: float, bits: int) -> float:
        """Convert unsigned integer from CAN to float."""
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    def _decode_motor_state(
        self, data: bytearray | bytes, motor_type: MotorType
    ) -> tuple[float, float, float, int, int]:
        """
        Decode motor state from CAN data.
        Returns: (position_deg, velocity_deg_s, torque, temp_mos, temp_rotor)
        """
        if len(data) < 8:
            raise ValueError("Invalid motor state data")

        # Extract encoded values
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = data[6]
        t_rotor = data[7]

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Decode to physical values
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        return np.degrees(position_rad), np.degrees(velocity_rad_per_sec), torque, t_mos, t_rotor

    def _process_response(self, motor: str, msg: can.Message) -> None:
        """Decode a message and update the motor state cache, keeping the frame's rx timestamp."""
        try:
            motor_type = self._motor_types[motor]
            pos, vel, torque, t_mos, t_rotor = self._decode_motor_state(msg.data, motor_type)

            self._last_known_states[motor] = {
                "position": pos,
                "velocity": vel,
                "torque": torque,
                "temp_mos": float(t_mos),
                "temp_rotor": float(t_rotor),
                # Normalized to epoch seconds by `_drain_newest` (host receipt time if unusable).
                "timestamp": float(msg.timestamp),
            }
        except Exception as e:
            logger.warning(f"Failed to decode response from {motor}: {e}")

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        """Read a value from a single motor. Positions are always in degrees."""

        # Refresh motor to get latest state
        msg = self._refresh_motor(motor)
        if msg is None:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            raise ConnectionError(
                f"No response from motor '{motor}' (send ID: 0x{motor_id:02X}, recv ID: 0x{recv_id:02X}). "
                f"Check that: 1) Motor is powered (24V), 2) CAN wiring is correct, "
                f"3) Motor IDs are configured correctly using Damiao Debugging Tools"
            )

        self._process_response(motor, msg)
        return self._get_cached_value(motor, data_name)

    def _get_cached_value(self, motor: str, data_name: str) -> Value:
        """Retrieve a specific value from the cache."""
        state = self._last_known_states[motor]
        mapping: dict[str, Any] = {
            "Present_Position": state["position"],
            "Present_Velocity": state["velocity"],
            "Present_Torque": state["torque"],
            "Temperature_MOS": state["temp_mos"],
            "Temperature_Rotor": state["temp_rotor"],
        }
        if data_name not in mapping:
            raise ValueError(f"Unknown data_name: {data_name}")
        return mapping[data_name]

    @check_if_not_connected
    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
    ) -> None:
        """
        Write a value to a single motor. Positions are always in degrees.
        Can write 'Goal_Position', 'Kp', or 'Kd'.
        """

        if data_name in ("Kp", "Kd"):
            self._gains[motor][data_name.lower()] = float(value)
        elif data_name == "Goal_Position":
            kp = self._gains[motor]["kp"]
            kd = self._gains[motor]["kd"]
            self._mit_control(motor, kp, kd, float(value), 0.0, 0.0)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        """
        Read the same value from multiple motors simultaneously.
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._get_cached_value(motor, data_name)
        return result

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, MotorState]:
        """
        Read ALL motor states (position, velocity, torque) from multiple motors in ONE refresh cycle.

        A motor whose reply misses the `state_wait_s` window keeps its newest older reading; its
        `timestamp` (and `state_age_s()`) then reports how old that reading really is.

        Returns:
            Dictionary mapping motor names to `MotorState` dicts with keys 'position', 'velocity',
            'torque', 'temp_mos', 'temp_rotor' and 'timestamp' (kernel rx time, epoch seconds).
            Example: {'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5, ...}, ...}
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._last_known_states[motor].copy()
        return result

    def _batch_refresh(self, motors: list[str]) -> None:
        """
        Refresh a list of motors and update the cache.

        Drains the rx queue (newest-wins), sends one refresh per motor, then waits at most
        `state_wait_s` for the replies. A motor that misses the window falls back to the newest older
        frame drained from the queue (which may be a late MIT reply), or else its last-known state,
        each with its true rx timestamp, and its `refresh_miss_count` is incremented. Each motor's
        `last_refresh_latency_s` is set to its reply's delay after the send (NaN on a miss). A motor that has
        never replied keeps zeros with timestamp 0.0 (age `math.inf`) and gets its own warning.
        """

        bus = self.canbus
        if bus is None:
            raise RuntimeError("CAN bus is not initialized.")

        refreshes = []
        for motor in motors:
            motor_id = self._get_motor_id(motor)
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            refreshes.append(
                can.Message(
                    arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd
                )
            )

        def send_refreshes() -> None:
            for msg in refreshes:
                bus.send(msg)

        expected_recv_ids = [self._get_motor_recv_id(m) for m in motors]
        newest, fresh = self._drain_newest(expected_recv_ids, send=send_refreshes, wait_s=self.state_wait_s)

        # Update cache
        missed = []
        for motor, recv_id in zip(motors, expected_recv_ids, strict=True):
            if recv_id not in fresh:
                missed.append(motor)
                self.refresh_miss_count[motor] += 1
                self.last_refresh_latency_s[motor] = math.nan
            else:
                self.last_refresh_latency_s[motor] = max(
                    0.0, self._first_fresh_ts[recv_id] - self._last_send_wall
                )
            # Never step back to a reading older than the cached one.
            msg = newest.get(recv_id)
            if msg is not None and msg.timestamp >= self._last_known_states[motor]["timestamp"]:
                self._process_response(motor, msg)
        if missed:
            logger.debug(
                f"No refresh reply within {self.state_wait_s * 1e3:.1f} ms from {missed} on {self.port}."
            )
        self._warn_if_stale(motors)

    def state_age_s(self) -> dict[str, float]:
        """
        Age in seconds of each motor's last reading: `time.time()` minus the rx timestamp of the frame
        it was decoded from (kernel rx time on socketcan, host receipt time on interfaces without
        usable timestamps), clamped to >= 0.

        Returns:
            `{motor_name: age_s}` for every motor on the bus. A motor that has never replied maps to
            `math.inf` (its cached values are placeholders, not a reading).
        """
        now = time.time()
        return {
            motor: max(0.0, now - state["timestamp"]) if state["timestamp"] > 0 else math.inf
            for motor, state in self._last_known_states.items()
        }

    def _warn_if_stale(self, motors: list[str]) -> None:
        """Warn (at most once per `STALE_WARN_INTERVAL_S` per bus and kind) about stale or missing state."""
        ages = self.state_age_s()
        never = [m for m in motors if math.isinf(ages[m])]
        if never:
            self._rate_limited_warning(
                "never",
                f"Damiao motors on {self.port} never replied: {never}. Their state is a placeholder.",
            )
        stale = {m: ages[m] for m in motors if self.stale_warn_s < ages[m] < math.inf}
        if stale:
            detail = ", ".join(f"{m}={age * 1e3:.0f}ms" for m, age in stale.items())
            self._rate_limited_warning(
                "stale",
                f"Stale Damiao state on {self.port} (> {self.stale_warn_s * 1e3:.0f} ms): {detail}. "
                "Using the newest older reading.",
            )

    @check_if_not_connected
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """
        Write values to multiple motors simultaneously. Positions are always in degrees.
        """

        if data_name in ("Kp", "Kd"):
            key = data_name.lower()
            for motor, val in values.items():
                self._gains[motor][key] = float(val)

        elif data_name == "Goal_Position":
            # Step 1: Send all MIT control commands
            if self.canbus is None:
                raise RuntimeError("CAN bus is not initialized.")
            for motor, value_degrees in values.items():
                motor_id = self._get_motor_id(motor)
                motor_name = self._get_motor_name(motor)
                motor_type = self._motor_types[motor_name]

                kp = self._gains[motor]["kp"]
                kd = self._gains[motor]["kd"]

                data = self._encode_mit_packet(motor_type, kp, kd, float(value_degrees), 0.0, 0.0)
                msg = can.Message(
                    arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd
                )
                self.canbus.send(msg)
                precise_sleep(PRECISE_TIMEOUT_SEC)

            # Step 2: Drain queued MIT replies (no wait); observed state comes from refreshes
            self._drain_newest(self._recv_id_to_motor)
        else:
            # Fall back to individual writes
            for motor, value in values.items():
                self.write(data_name, motor, value)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration data from motors."""
        # Damiao motors don't store calibration internally
        # Return existing calibration or empty dict
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration data to motors."""
        # Damiao motors don't store calibration internally
        # Just cache it in memory
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self,
        motors: str | list[str] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """
        Interactively record the min/max values of each motor in degrees.

        Move the joints by hand (with torque disabled) while the method streams live positions.
        Press Enter to finish.
        """
        target_motors = self._get_motors_list(motors)

        self.disable_torque(target_motors)
        time.sleep(LONG_TIMEOUT_SEC)

        start_positions = self.sync_read("Present_Position", target_motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", target_motors)

            for motor in target_motors:
                if motor in positions:
                    mins[motor] = min(positions[motor], mins.get(motor, positions[motor]))
                    maxes[motor] = max(positions[motor], maxes.get(motor, positions[motor]))

            if display_values:
                print("\n" + "=" * 50)
                print(f"{'MOTOR':<20} | {'MIN (deg)':>12} | {'POS (deg)':>12} | {'MAX (deg)':>12}")
                print("-" * 50)
                for motor in target_motors:
                    if motor in positions:
                        print(
                            f"{motor:<20} | {mins[motor]:>12.1f} | {positions[motor]:>12.1f} | {maxes[motor]:>12.1f}"
                        )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(target_motors) + 4)

            time.sleep(LONG_TIMEOUT_SEC)

        self.enable_torque(target_motors)

        for motor in target_motors:
            if (motor in mins) and (motor in maxes) and (int(abs(maxes[motor] - mins[motor])) < 5):
                raise ValueError(f"Motor {motor} has insufficient range of motion (< 5 degrees)")

        return mins, maxes

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """Convert motor specification to list of motor names."""
        if motors is None:
            return list(self.motors.keys())
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors
        else:
            raise TypeError(f"Invalid motors type: {type(motors)}")

    def _get_motor_id(self, motor: NameOrID) -> int:
        """Get CAN ID for a motor."""
        if isinstance(motor, str):
            if motor in self.motors:
                return self.motors[motor].id
            else:
                raise ValueError(f"Unknown motor: {motor}")
        else:
            return motor

    def _get_motor_name(self, motor: NameOrID) -> str:
        """Get motor name from name or ID."""
        if isinstance(motor, str):
            return motor
        else:
            for name, m in self.motors.items():
                if m.id == motor:
                    return name
            raise ValueError(f"Unknown motor ID: {motor}")

    def _get_motor_recv_id(self, motor: NameOrID) -> int:
        """Get motor recv_id from name or ID."""
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)
        if motor_obj and motor_obj.recv_id is not None:
            return motor_obj.recv_id
        else:
            raise ValueError(f"Motor {motor_obj} doesn't have a valid recv_id (None).")

    @property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        return bool(self.calibration)
