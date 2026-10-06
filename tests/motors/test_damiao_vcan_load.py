"""Soak test: Damiao state freshness under CPU/GIL load on a real SocketCAN `vcan0`.

Runs the driver at 30 Hz (`_mit_control_batch` + `sync_read_all_states`) while CPU-burner threads
fight it for the GIL, against fake motors served from a separate process (so the "hardware"
answers promptly, as real motors do, regardless of the GIL in the test process). Asserts that the
reported state age stays under two control periods and that the rx backlog does not ratchet.

Skipped unless `vcan0` exists. One-time setup (needs root, run it yourself):

    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
    sudo ip link set up vcan0

Run:

    uv run --no-sync pytest tests/motors/test_damiao_vcan_load.py -v -s

Env overrides: `DAMIAO_VCAN_SOAK_S` (duration, default 60), `DAMIAO_VCAN_BURNERS` (pure-Python
burner threads, default 4).
"""

import multiprocessing as mp
import os
import threading
import time

import pytest

can = pytest.importorskip("can")

if not os.path.exists("/sys/class/net/vcan0"):
    pytest.skip("vcan0 not present (see module docstring for setup)", allow_module_level=True)

import numpy as np  # noqa: E402

from lerobot.motors import Motor  # noqa: E402
from lerobot.motors.damiao import DamiaoMotorsBus  # noqa: E402
from lerobot.motors.damiao.tables import CAN_CMD_REFRESH, CAN_PARAM_ID  # noqa: E402

CHANNEL = "vcan0"
FPS = 30
PERIOD = 1 / FPS
WARMUP_S = 2.0
SOAK_S = float(os.environ.get("DAMIAO_VCAN_SOAK_S", "60"))
N_BURNERS = int(os.environ.get("DAMIAO_VCAN_BURNERS", "4"))
# send id -> recv id, 8 motors like one arm
MOTOR_IDS = {i: 0x10 + i for i in range(1, 9)}


def _fake_motors(stop: "mp.synchronize.Event", ready: "mp.synchronize.Event") -> None:
    """Reply to every refresh and MIT command with a state frame, like a Damiao motor does."""
    bus = can.interface.Bus(interface="socketcan", channel=CHANNEL)
    counter = 0
    ready.set()
    try:
        while not stop.is_set():
            msg = bus.recv(timeout=0.05)
            if msg is None:
                continue
            if msg.arbitration_id == CAN_PARAM_ID and msg.data[2] == CAN_CMD_REFRESH:
                send_id = msg.data[0] | (msg.data[1] << 8)
            elif msg.arbitration_id in MOTOR_IDS:
                send_id = msg.arbitration_id
            else:
                continue
            if (recv_id := MOTOR_IDS.get(send_id)) is None:
                continue
            counter = (counter + 1) & 0xFFFF
            data = [send_id, counter >> 8, counter & 0xFF, 0x80, 0x08, 0x00, 25, 30]
            bus.send(can.Message(arbitration_id=recv_id, data=data, is_extended_id=False))
    finally:
        bus.shutdown()


def _burn_python(stop: threading.Event) -> None:
    x = 0
    while not stop.is_set():
        for i in range(10_000):
            x = (x * 31 + i) % 1_000_003


def _burn_numpy(stop: threading.Event) -> None:
    a = np.random.default_rng(0).standard_normal((256, 256))
    while not stop.is_set():
        a = np.tanh(a @ a.T / 256)


def test_state_age_bounded_under_cpu_load():
    motors = {
        f"joint_{i}": Motor(id=i, model="damiao", norm_mode="degrees", motor_type_str="dm4310", recv_id=r)
        for i, r in MOTOR_IDS.items()
    }
    # fork before any burner thread starts; the child only uses python-can.
    ctx = mp.get_context("fork")
    stop_motors, motors_ready = ctx.Event(), ctx.Event()
    responder = ctx.Process(target=_fake_motors, args=(stop_motors, motors_ready), daemon=True)
    stop_burners = threading.Event()
    burners = [
        threading.Thread(target=_burn_python, args=(stop_burners,), daemon=True) for _ in range(N_BURNERS)
    ]
    burners.append(threading.Thread(target=_burn_numpy, args=(stop_burners,), daemon=True))

    bus = DamiaoMotorsBus(port=CHANNEL, motors=motors, can_interface="socketcan", use_can_fd=False)
    responder.start()
    try:
        assert motors_ready.wait(10), "fake motor process did not start"
        bus.connect(handshake=True)

        # Count frames consumed per tick to detect a growing backlog.
        frames_this_tick = 0
        real_recv = bus.canbus.recv

        def counting_recv(timeout=None):
            nonlocal frames_this_tick
            msg = real_recv(timeout=timeout)
            if msg is not None:
                frames_this_tick += 1
            return msg

        bus.canbus.recv = counting_recv

        for t in burners:
            t.start()

        ages, frames, overruns = [], [], 0
        start = time.perf_counter()
        next_tick = start
        state = bus.sync_read_all_states()
        while (now := time.perf_counter()) - start < WARMUP_S + SOAK_S:
            frames_this_tick = 0
            bus._mit_control_batch({m: (10.0, 0.5, state[m]["position"], 0.0, 0.0) for m in motors})
            state = bus.sync_read_all_states()
            if now - start >= WARMUP_S:
                ages.append(max(bus.state_age_s().values()))
                frames.append(frames_this_tick)
            next_tick += PERIOD
            if (sleep_s := next_tick - time.perf_counter()) > 0:
                time.sleep(sleep_s)
            else:
                overruns += 1
                next_tick = time.perf_counter()

        stop_burners.set()
        n = len(ages)
        ages_ms = np.array(ages) * 1e3
        print(
            f"\n{n} ticks, overruns={overruns}, age ms: p50={np.median(ages_ms):.1f} "
            f"p99={np.percentile(ages_ms, 99):.1f} max={ages_ms.max():.1f}; "
            f"frames/tick: mean={np.mean(frames):.1f} max={max(frames)}"
        )

        assert n > 0.5 * SOAK_S * FPS, f"loop too slow: {n} ticks in {SOAK_S} s"
        assert ages_ms.max() < 2 * PERIOD * 1e3
        # 2 replies per motor per tick (MIT + refresh); allow a late batch from the previous tick.
        assert max(frames) <= 4 * len(motors)
        half = n // 2
        assert np.mean(frames[half:]) <= np.mean(frames[:half]) + 1, "rx backlog grew over the run"
    finally:
        stop_burners.set()
        for t in burners:
            if t.is_alive():
                t.join(timeout=5)
        if bus.canbus is not None:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
            else:
                bus.canbus.shutdown()
        stop_motors.set()
        responder.join(timeout=5)
        if responder.is_alive():
            responder.terminate()
