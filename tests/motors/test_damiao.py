"""Tests for the Damiao CAN motor bus.

Most tests run the real driver against python-can's in-process `virtual` interface: a second
virtual bus on the same channel plays the motors. Replies are injected synchronously from a
wrapper around the driver's `canbus.send`, so every test is deterministic (no responder threads,
no sleeps). `test_damiao_motor` needs a physical motor and is skipped.
"""

import copy
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest

can = pytest.importorskip("can")

from lerobot.motors import Motor  # noqa: E402
from lerobot.motors.damiao import DamiaoMotorsBus, damiao  # noqa: E402
from lerobot.motors.damiao.tables import (  # noqa: E402
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_PARAM_ID,
    MotorType,
)

LOGGER_NAME = "lerobot.motors.damiao.damiao"
SEND_IDS = {"m1": 0x01, "m2": 0x02}
RECV_IDS = {"m1": 0x11, "m2": 0x12}
UNEXPECTED_ID = 0x55


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus._mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


# ---------------------------------------------------------------------------------------------
# Virtual-bus fixtures and helpers
# ---------------------------------------------------------------------------------------------


def _motors() -> dict[str, Motor]:
    return {
        name: Motor(
            id=SEND_IDS[name],
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=RECV_IDS[name],
        )
        for name in SEND_IDS
    }


def _state_data(seq: int) -> bytes:
    """State frame payload whose 16-bit position field carries `seq`, so the winning frame is identifiable."""
    return bytes([0x11, (seq >> 8) & 0xFF, seq & 0xFF, 0x80, 0x08, 0x00, 25, 30])


def _frame(arb_id: int, seq: int, ts: float) -> "can.Message":
    return can.Message(arbitration_id=arb_id, data=_state_data(seq), is_extended_id=False, timestamp=ts)


def _position(seq: int) -> float:
    """Position (deg) the driver decodes from a frame built with `_state_data(seq)`."""
    probe = DamiaoMotorsBus.__new__(DamiaoMotorsBus)
    return float(probe._decode_motor_state(_state_data(seq), MotorType.DM4310)[0])


@dataclass
class FakeMotors:
    """The motor side of a virtual CAN channel.

    `attach(bus)` wraps the driver's `canbus.send` so each request the driver transmits triggers an
    immediate reply (queued before `send` returns) unless that motor is in `silent`. Replies carry a
    rising sequence number in their position field and `timestamp=time.time()`.
    """

    canbus: "can.BusABC"
    silent: set[str] = field(default_factory=set)
    reply_to_mit: bool = True
    seq: int = 1000
    replies: list[tuple[str, int, float]] = field(default_factory=list)

    def inject(self, motor_or_id: str | int, seq: int, ts: float) -> None:
        arb_id = RECV_IDS[motor_or_id] if isinstance(motor_or_id, str) else motor_or_id
        self.canbus.send(_frame(arb_id, seq, ts))

    def reply(self, motor: str) -> None:
        self.seq += 1
        ts = time.time()
        self.replies.append((motor, self.seq, ts))
        self.inject(motor, self.seq, ts)

    def last_reply(self, motor: str) -> tuple[int, float]:
        _, seq, ts = next(r for r in reversed(self.replies) if r[0] == motor)
        return seq, ts

    def _on_driver_send(self, msg: "can.Message") -> None:
        by_send_id = {v: k for k, v in SEND_IDS.items()}
        if msg.arbitration_id == CAN_PARAM_ID and msg.data[2] == CAN_CMD_REFRESH:
            motor = by_send_id[msg.data[0] | (msg.data[1] << 8)]
        elif msg.arbitration_id in by_send_id:
            motor = by_send_id[msg.arbitration_id]
            is_mit = list(msg.data[:7]) != [0xFF] * 7
            if is_mit and not self.reply_to_mit:
                return
        else:
            return
        if motor not in self.silent:
            self.reply(motor)

    def attach(self, canbus: "can.BusABC") -> None:
        original_send = canbus.send

        def send(msg: "can.Message", timeout: float | None = None) -> None:
            original_send(msg, timeout)
            self._on_driver_send(msg)

        canbus.send = send


def _queue_empty(bus: DamiaoMotorsBus) -> bool:
    return bus.canbus.recv(timeout=0) is None


@pytest.fixture
def channel(request) -> str:
    return f"damiao-test-{request.node.nodeid}"


@pytest.fixture
def fake(channel) -> Iterator[FakeMotors]:
    motor_side = can.interface.Bus(interface="virtual", channel=channel, preserve_timestamps=True)
    yield FakeMotors(motor_side)
    motor_side.shutdown()


@pytest.fixture
def make_bus(channel) -> Iterator[Callable[[], DamiaoMotorsBus]]:
    buses: list[DamiaoMotorsBus] = []

    def make() -> DamiaoMotorsBus:
        bus = DamiaoMotorsBus(port=channel, motors=_motors(), can_interface="virtual", use_can_fd=False)
        buses.append(bus)
        return bus

    yield make
    for bus in buses:
        if bus.canbus is not None:
            bus.canbus.shutdown()
            bus.canbus = None
        bus._is_connected = False


@pytest.fixture
def bus(make_bus, fake) -> DamiaoMotorsBus:
    bus = make_bus()
    bus.connect(handshake=False)
    fake.attach(bus.canbus)
    return bus


@pytest.fixture
def warnings_log(caplog) -> Callable[[], list[logging.LogRecord]]:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    def records() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]

    return records


def _stale_warnings(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if "Stale Damiao state" in r.getMessage()]


# ---------------------------------------------------------------------------------------------
# Refresh request format
# ---------------------------------------------------------------------------------------------


def test_sync_read_sends_one_refresh_per_motor(bus, fake):
    bus.sync_read_all_states()

    sent = []
    while (msg := fake.canbus.recv(timeout=0)) is not None:
        sent.append(msg)
    assert [m.arbitration_id for m in sent] == [CAN_PARAM_ID, CAN_PARAM_ID]
    assert [m.data[0] for m in sent] == [SEND_IDS["m1"], SEND_IDS["m2"]]
    assert all(m.data[2] == CAN_CMD_REFRESH for m in sent)


# ---------------------------------------------------------------------------------------------
# 1. Backlog drain
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("replies", [True, False], ids=["fresh-reply", "reply-missed"])
def test_backlog_is_drained_newest_wins(bus, fake, replies):
    backlog = 50
    t0 = time.time() - 1.0
    for i in range(backlog):
        for motor in SEND_IDS:
            fake.inject(motor, i, t0 + i * 1e-3)
    if not replies:
        fake.silent = set(SEND_IDS)

    states = bus.sync_read_all_states()

    for motor in SEND_IDS:
        if replies:
            seq, ts = fake.last_reply(motor)
        else:
            seq, ts = backlog - 1, t0 + (backlog - 1) * 1e-3
        assert states[motor]["position"] == pytest.approx(_position(seq))
        assert states[motor]["timestamp"] == ts
    assert _queue_empty(bus)


# ---------------------------------------------------------------------------------------------
# 2. Duplicates and unexpected IDs
# ---------------------------------------------------------------------------------------------


def test_duplicates_and_unexpected_ids_are_consumed(bus, fake):
    t = time.time() - 0.5
    for _ in range(10):
        fake.inject("m1", 7, t)  # exact duplicates
        fake.inject(UNEXPECTED_ID, 999, t + 0.1)
        fake.inject(SEND_IDS["m1"], 998, t + 0.1)  # a send id is not a reply id
    fake.inject("m2", 8, t)
    fake.inject(UNEXPECTED_ID, 997, t + 0.2)  # queued last, still must not win
    fake.silent = set(SEND_IDS)

    states = bus.sync_read_all_states()

    assert states["m1"]["position"] == pytest.approx(_position(7))
    assert states["m2"]["position"] == pytest.approx(_position(8))
    assert states["m1"]["timestamp"] == t
    assert _queue_empty(bus)


def test_frames_of_motors_not_read_are_consumed_without_updating_them(bus, fake):
    t = time.time() - 0.5
    fake.inject("m2", 5, t)
    fake.inject("m1", 6, t)

    bus.sync_read_all_states(["m1"])

    assert bus._last_known_states["m2"]["timestamp"] == 0.0
    assert bus._last_known_states["m1"]["position"] == pytest.approx(_position(fake.last_reply("m1")[0]))
    assert _queue_empty(bus)


# ---------------------------------------------------------------------------------------------
# 3. Missed reply -> newest older frame / last-known, with true timestamps and rate-limited warning
# ---------------------------------------------------------------------------------------------


def test_missed_reply_falls_back_to_newest_older_frame(bus, fake, warnings_log):
    bus.stale_warn_s = 0.1
    fake.silent = {"m1"}
    misses_before = dict(bus.refresh_miss_count)
    t_old = time.time() - 0.5
    fake.inject("m1", 3, t_old - 0.1)
    fake.inject("m1", 4, t_old)

    states = bus.sync_read_all_states()

    assert states["m1"]["position"] == pytest.approx(_position(4))
    assert states["m1"]["timestamp"] == t_old
    assert states["m2"]["timestamp"] == fake.last_reply("m2")[1]
    ages = bus.state_age_s()
    assert ages["m1"] == pytest.approx(0.5, abs=0.05)
    assert ages["m2"] < 0.05
    assert bus.refresh_miss_count == {"m1": misses_before["m1"] + 1, "m2": misses_before["m2"]}

    # Several more misses within STALE_WARN_INTERVAL_S: one warning in total.
    for i in range(5):
        fake.inject("m1", 10 + i, time.time() - 0.3)
        bus.sync_read_all_states()
    stale = _stale_warnings(warnings_log())
    assert len(stale) == 1
    assert "m1=" in stale[0].getMessage() and "m2=" not in stale[0].getMessage()
    assert _queue_empty(bus)


def test_reply_queued_in_time_but_read_after_the_wait_is_not_a_miss(bus, fake):
    """The thread wakes after the wait ran out (GIL contention) while the replies already sit in the
    kernel queue: the final non-blocking drain still takes them, so nothing counts as a miss."""
    bus.state_wait_s = 0.0  # the wait is over before the first blocking recv
    misses_before = dict(bus.refresh_miss_count)

    states = bus.sync_read_all_states()

    assert bus.refresh_miss_count == misses_before
    for motor in ("m1", "m2"):
        seq, ts = fake.last_reply(motor)
        assert states[motor]["position"] == pytest.approx(_position(seq))
        assert states[motor]["timestamp"] == ts
        assert math.isfinite(bus.last_refresh_latency_s[motor])
    assert _queue_empty(bus)


def test_refresh_latency_is_the_reply_delay_after_send_and_nan_on_a_miss(bus, fake, monkeypatch):
    delay = {"m1": 0.0012, "m2": 0.0031}
    reply = fake.reply

    def delayed_reply(motor: str) -> None:
        fake.seq += 1
        ts = time.time() + delay[motor]  # rx stamp as if the reply took `delay` on the wire
        fake.replies.append((motor, fake.seq, ts))
        fake.inject(motor, fake.seq, ts)

    monkeypatch.setattr(fake, "reply", delayed_reply)
    bus.sync_read_all_states()
    assert bus.last_refresh_latency_s["m1"] == pytest.approx(0.0012, abs=0.001)
    assert bus.last_refresh_latency_s["m2"] == pytest.approx(0.0031, abs=0.001)

    monkeypatch.setattr(fake, "reply", reply)
    fake.silent = {"m2"}
    bus.sync_read_all_states()
    assert math.isnan(bus.last_refresh_latency_s["m2"])
    assert math.isfinite(bus.last_refresh_latency_s["m1"])


def test_late_reply_is_used_on_the_next_read(bus, fake):
    fake.silent = {"m1"}
    bus.sync_read_all_states()
    assert bus._last_known_states["m1"]["timestamp"] == 0.0

    t_late = time.time()
    fake.inject("m1", 42, t_late)  # the reply that missed the wait window
    states = bus.sync_read_all_states()

    assert states["m1"]["position"] == pytest.approx(_position(42))
    assert states["m1"]["timestamp"] == t_late


def test_missed_reply_without_older_frame_keeps_last_known(bus, fake):
    first = bus.sync_read_all_states()
    seq, ts = fake.last_reply("m1")
    assert first["m1"]["timestamp"] == ts

    fake.silent = {"m1"}
    for _ in range(3):
        states = bus.sync_read_all_states()
        assert states["m1"] == first["m1"]
        assert states["m1"]["position"] == pytest.approx(_position(seq))
        assert states["m2"]["timestamp"] > first["m2"]["timestamp"]


def test_stale_warning_respects_threshold_and_interval(bus, fake, warnings_log, monkeypatch):
    bus.sync_read_all_states()
    assert _stale_warnings(warnings_log()) == []  # fresh: below default 2/30 s

    fake.silent = set(SEND_IDS)
    bus.stale_warn_s = 0.0
    bus.sync_read_all_states()
    bus.sync_read_all_states()
    assert len(_stale_warnings(warnings_log())) == 1

    # Once the interval has passed, the next stale read warns again.
    monkeypatch.setattr(damiao, "STALE_WARN_INTERVAL_S", 0.0)
    bus.sync_read_all_states()
    assert len(_stale_warnings(warnings_log())) == 2


def test_never_replied_motor_gets_its_own_rate_limited_warning(bus, fake, warnings_log):
    fake.silent = {"m2"}
    for _ in range(3):
        bus.sync_read_all_states()
    assert bus.state_age_s()["m2"] == math.inf
    never = [r for r in warnings_log() if "never replied" in r.getMessage()]
    assert len(never) == 1 and "'m2'" in never[0].getMessage()
    assert _stale_warnings(warnings_log()) == []


# ---------------------------------------------------------------------------------------------
# 4. No ratchet over many ticks with a backlog injected every tick
# ---------------------------------------------------------------------------------------------


def test_age_stays_bounded_over_1000_ticks_with_backlog(bus, fake, warnings_log):
    period = 1 / 30
    max_age = 0.0
    for tick in range(1000):
        now = time.time()
        # Late/duplicate traffic that arrived since the last tick, at most one period old.
        for motor in SEND_IDS:
            for k in range(3):
                fake.inject(motor, k, now - period + k * 1e-4)
            fake.inject(motor, 2, now - period + 2e-4)  # duplicate
        fake.inject(UNEXPECTED_ID, 0, now)

        state = bus.sync_read_all_states()
        bus._mit_control_batch({m: (10.0, 0.5, state[m]["position"], 0.0, 0.0) for m in SEND_IDS})
        assert _queue_empty(bus), f"queue not empty after MIT batch at tick {tick}"

        # Some ticks the refresh reply misses the window; the fallback must be the backlog frame.
        fake.silent = set(SEND_IDS) if tick % 10 == 9 else set()
        for motor in SEND_IDS:
            fake.inject(motor, 3, time.time() - period)
        bus.sync_read_all_states()
        max_age = max(max_age, *bus.state_age_s().values())
        assert _queue_empty(bus), f"queue not empty after read at tick {tick}"

    assert max_age < 2 * period
    assert _stale_warnings(warnings_log()) == []


# ---------------------------------------------------------------------------------------------
# 5. MIT writes drain replies without touching observed state
# ---------------------------------------------------------------------------------------------


def _mit_batch(bus):
    bus._mit_control_batch(dict.fromkeys(SEND_IDS, (10.0, 0.5, 1.0, 0.0, 0.0)))


def _mit_single(bus):
    for m in SEND_IDS:
        bus._mit_control(m, 10.0, 0.5, 1.0, 0.0, 0.0)


def _sync_write(bus):
    bus.sync_write("Goal_Position", dict.fromkeys(SEND_IDS, 1.0))


@pytest.mark.parametrize("mit_write", [_mit_batch, _mit_single, _sync_write], ids=lambda f: f.__name__)
def test_mit_writes_drain_replies_without_updating_state(bus, fake, mit_write):
    before = copy.deepcopy(bus.sync_read_all_states())
    fake.inject("m1", 77, time.time())  # queued before the write
    n_replies = len(fake.replies)

    mit_write(bus)

    assert [motor for motor, _, _ in fake.replies[n_replies:]] == ["m1", "m2"]  # the motors replied
    assert bus._last_known_states == before
    assert _queue_empty(bus)

    # The drained MIT replies are not a fallback for a later missed refresh either.
    fake.silent = set(SEND_IDS)
    assert bus.sync_read_all_states() == before


# ---------------------------------------------------------------------------------------------
# 6. state_age_s
# ---------------------------------------------------------------------------------------------


def test_state_age_s(bus, fake, monkeypatch):
    ages = bus.state_age_s()
    assert set(ages) == set(SEND_IDS)
    assert all(age == math.inf for age in ages.values())

    fake.silent = set(SEND_IDS)
    t_m1, t_m2 = time.time() - 1.25, time.time() + 5.0  # m2 from the future (within plausible skew)
    fake.inject("m1", 1, t_m1)
    fake.inject("m2", 2, t_m2)
    bus.sync_read_all_states()

    now = t_m1 + 1.5
    monkeypatch.setattr(damiao.time, "time", lambda: now)
    ages = bus.state_age_s()
    assert ages == {"m1": now - t_m1, "m2": 0.0}


@pytest.mark.parametrize("rx_ts", [0.0, 100.0], ids=["missing", "not-epoch"])
def test_unusable_rx_timestamp_is_replaced_by_host_time(bus, fake, warnings_log, rx_ts):
    fake.silent = set(SEND_IDS)
    before = time.time()
    fake.inject("m1", 1, rx_ts)
    bus.sync_read_all_states()
    fake.inject("m1", 2, rx_ts)
    bus.sync_read_all_states()

    assert before <= bus._last_known_states["m1"]["timestamp"] <= time.time()
    assert bus._last_known_states["m1"]["position"] == pytest.approx(_position(2))
    assert len([r for r in warnings_log() if "no usable rx timestamps" in r.getMessage()]) == 1


# ---------------------------------------------------------------------------------------------
# 7. Drain cap
# ---------------------------------------------------------------------------------------------


def test_drain_stops_at_cap_and_warns(bus, fake, warnings_log, monkeypatch):
    monkeypatch.setattr(damiao, "DRAIN_CAP", 10)
    t = time.time() - 1.0
    for i in range(25):
        fake.inject("m1", i, t + i * 1e-3)

    newest, fresh = bus._drain_newest(RECV_IDS.values())

    assert newest[RECV_IDS["m1"]].data == _state_data(9)
    assert fresh == {}
    flood = [r for r in warnings_log() if "CAN rx flood" in r.getMessage()]
    assert len(flood) == 1
    remaining = 0
    while bus.canbus.recv(timeout=0) is not None:
        remaining += 1
    assert remaining == 15


def test_drain_stopped_by_its_time_limit_is_not_called_a_flood(bus, fake, warnings_log, monkeypatch):
    monkeypatch.setattr(damiao, "DRAIN_MAX_S", 0.0)  # every recv "took too long"
    for i in range(3):
        fake.inject("m1", i, time.time())

    bus._drain_newest(RECV_IDS.values())

    messages = [r.getMessage() for r in warnings_log()]
    assert not any("CAN rx flood" in m for m in messages)
    assert sum("hit its 0 ms limit after 1 frames" in m for m in messages) == 1


def test_backlog_beyond_cap_is_not_reported_as_fresh(bus, fake, monkeypatch):
    monkeypatch.setattr(damiao, "DRAIN_CAP", 10)
    for i in range(25):
        fake.inject("m1", i, time.time() - 1.0)

    _, fresh = bus._drain_newest(RECV_IDS.values(), send=lambda: None, wait_s=0.0)

    assert fresh == {}  # leftover backlog dequeued after send() predates it: not a reply


def test_post_send_phase_is_bounded_under_sustained_flood(bus, monkeypatch):
    monkeypatch.setattr(damiao, "DRAIN_CAP", 10)
    calls = 0
    flood_frame = _frame(UNEXPECTED_ID, 0, time.time())

    def flooding_recv(timeout=None):
        nonlocal calls
        calls += 1
        if calls > 100_000:
            raise RuntimeError("safety stop")  # the driver swallows this and returns
        return flood_frame

    monkeypatch.setattr(bus.canbus, "recv", flooding_recv)
    bus._drain_newest(RECV_IDS.values(), send=lambda: None, wait_s=damiao.STATE_WAIT_S)

    assert calls <= 2 * damiao.DRAIN_CAP + 1


# ---------------------------------------------------------------------------------------------
# read() / enable_torque() go through the same receive path
# ---------------------------------------------------------------------------------------------


def test_read_uses_fresh_reply_and_never_a_stale_frame(bus, fake):
    fake.inject("m1", 5, time.time() - 1.0)
    assert bus.read("Present_Position", "m1") == pytest.approx(_position(fake.last_reply("m1")[0]))
    assert bus._last_known_states["m1"]["timestamp"] == fake.last_reply("m1")[1]
    assert _queue_empty(bus)

    # A silent motor is not masked by an older queued frame: read() raises and drains it.
    fake.silent = {"m1"}
    fake.inject("m1", 6, time.time() - 1.0)
    with pytest.raises(ConnectionError, match="No response from motor 'm1'"):
        bus.read("Present_Position", "m1")
    assert _queue_empty(bus)


def test_enable_torque_processes_reply_and_drains(bus, fake):
    fake.inject(UNEXPECTED_ID, 0, time.time())
    bus.enable_torque()
    for motor in SEND_IDS:
        seq, ts = fake.last_reply(motor)
        assert bus._last_known_states[motor]["position"] == pytest.approx(_position(seq))
        assert bus._last_known_states[motor]["timestamp"] == ts
    assert _queue_empty(bus)


# ---------------------------------------------------------------------------------------------
# 8. Handshake
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def attach_on_connect(fake, monkeypatch) -> list["can.Message"]:
    """Make `connect()` attach the fake motors to the bus it opens, before the handshake runs.

    Frames appended to the returned list are queued on the driver's bus as soon as it opens.
    """
    real_bus = can.interface.Bus
    queued_on_open: list[can.Message] = []

    def bus_factory(**kwargs):
        canbus = real_bus(**kwargs)
        fake.attach(canbus)
        for msg in queued_on_open:
            fake.canbus.send(msg)
        return canbus

    monkeypatch.setattr(damiao.can.interface, "Bus", bus_factory)
    monkeypatch.setattr(damiao, "LONG_TIMEOUT_SEC", 0.01)
    return queued_on_open


def test_handshake_with_responder_populates_state(make_bus, fake, attach_on_connect):
    bus = make_bus()
    bus.connect(handshake=True)

    assert bus.is_connected
    for motor in SEND_IDS:
        seq, ts = fake.last_reply(motor)
        assert ts > 0
        assert bus._last_known_states[motor]["timestamp"] == ts
        assert bus._last_known_states[motor]["position"] == pytest.approx(_position(seq))
    enable_frames = []
    while (msg := fake.canbus.recv(timeout=0)) is not None:
        enable_frames.append(msg)
    assert [(m.arbitration_id, m.data[7]) for m in enable_frames] == [
        (SEND_IDS["m1"], CAN_CMD_ENABLE),
        (SEND_IDS["m2"], CAN_CMD_ENABLE),
    ]


def test_handshake_with_silent_motor_raises(make_bus, fake, attach_on_connect):
    fake.silent = {"m2"}
    attach_on_connect.append(_frame(RECV_IDS["m2"], 1, time.time() - 1.0))  # old frame, not a reply
    bus = make_bus()

    with pytest.raises(ConnectionError, match=r"did not respond: \['m2'\]"):
        bus.connect(handshake=True)
    assert not bus.is_connected


if __name__ == "__main__":
    test_damiao_motor()


def test_failed_connect_closes_the_bus(make_bus, fake, attach_on_connect):
    fake.silent = {"m2"}
    bus = make_bus()
    with pytest.raises(ConnectionError):
        bus.connect()
    assert bus.canbus is None
    assert not bus.is_connected


# ---------------------------------------------------------------------------------------------
# Error propagation and cache monotonicity
# ---------------------------------------------------------------------------------------------


def test_send_errors_propagate(bus, monkeypatch):
    def failing_send(msg, timeout=None):
        raise can.CanOperationError("tx failed")

    monkeypatch.setattr(bus.canbus, "send", failing_send)
    with pytest.raises(can.CanOperationError):
        bus.sync_read_all_states()
    with pytest.raises(can.CanOperationError):
        bus.enable_torque()


def test_older_frame_never_replaces_newer_cached_state(bus, fake):
    cached = copy.deepcopy(bus.sync_read_all_states()["m1"])
    fake.silent = {"m1"}
    fake.inject("m1", 7, cached["timestamp"] - 1.0)

    assert bus.sync_read_all_states()["m1"] == cached
    assert _queue_empty(bus)
