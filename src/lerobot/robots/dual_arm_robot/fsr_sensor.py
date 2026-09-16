#!/usr/bin/env python3
"""
FSR Sensor Module for LeRobot Integration (Dual Sensor Version)

Reads two comma-separated values from a single serial port in a background thread.
Normalization is [0, 1] where 0 is no pressure and 1 is max pressure.
Save this file as: fsr_sensor.py
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None

logger = logging.getLogger(__name__)


@dataclass
class FSRSensorConfig:
    """Configuration for FSR sensor"""

    port: str = "/dev/ttyACM4"
    baudrate: int = 115200
    timeout: float = 0.1
    buffer_size: int = 10
    max_adc_value: int = 3400


class FSRSensor:
    """
    Dual FSR sensor interface for LeRobot integration.
    Provides non-blocking FSR data reading for two sensors via a background thread.
    """

    def __init__(self, config: FSRSensorConfig):
        self.config = config

        # Thread-safe data storage for two sensors
        self._lock = threading.Lock()
        self._latest_raw = {"right": 0, "left": 0}
        self._latest_normalized = {"right": 0.0, "left": 0.0}
        self._last_update_time = 0.0
        self._data_buffer = deque(maxlen=config.buffer_size)

        # Connection management
        self._serial = None
        self._is_connected = False
        self._should_stop = False
        self._reader_thread = None

        # Statistics
        self._read_count = 0
        self._error_count = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Connect to FSR sensor and start background reading"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"FSR sensor on {self.config.port} already connected")

        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        assert serial is not None

        try:
            self._serial = serial.Serial(self.config.port, self.config.baudrate, timeout=self.config.timeout)
            self._serial.reset_input_buffer()
            self._is_connected = True
            self._should_stop = False

            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            logger.info(f"FSR sensor connected on {self.config.port}")

        except serial.SerialException as e:
            logger.error(f"Failed to connect FSR sensor on {self.config.port}: {e}")
            self._is_connected = False
            raise DeviceNotConnectedError(f"Cannot connect to FSR sensor: {e}") from e

    def disconnect(self) -> None:
        """Stop reading and disconnect from FSR sensor"""
        if not self._is_connected:
            return

        self._should_stop = True

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        if self._serial:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning(f"Error closing FSR serial connection: {e}")
            finally:
                self._serial = None

        self._is_connected = False
        logger.info("FSR sensor disconnected")

    def read(self) -> dict[str, float]:
        """
        Read current FSR values for both sensors (thread-safe, non-blocking).
        """
        if not self._is_connected:
            # Return default value of 0.0 for "no pressure"
            return {
                "right_arm.gripper.fsr": -1.0,
                "left_arm.gripper.fsr": -1.0,
            }

        with self._lock:
            return {
                "right_arm.gripper.fsr": self._latest_normalized["right"],
                "left_arm.gripper.fsr": self._latest_normalized["left"],
            }

    def _reader_loop(self) -> None:
        """Background thread that continuously reads and parses dual FSR data"""
        logger.info("FSR reader thread started")

        while not self._should_stop and self._is_connected:
            try:
                if self._serial and self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    parts = line.split(",")

                    if len(parts) == 2:
                        raw_right = int(parts[0])
                        raw_left = int(parts[1])

                        # Normalize right sensor to [0, 1] range
                        norm_right = raw_right / self.config.max_adc_value
                        norm_right = max(0.0, min(1.0, norm_right))  # Clamp between 0 and 1

                        # Normalize left sensor to [0, 1] range
                        norm_left = raw_left / self.config.max_adc_value
                        norm_left = max(0.0, min(1.0, norm_left))  # Clamp between 0 and 1

                        current_time = time.time()

                        with self._lock:
                            self._latest_raw["right"] = raw_right
                            self._latest_raw["left"] = raw_left
                            self._latest_normalized["right"] = norm_right
                            self._latest_normalized["left"] = norm_left
                            self._last_update_time = current_time
                            self._data_buffer.append((current_time, raw_right, raw_left))
                            self._read_count += 1

                time.sleep(0.001)

            except Exception as e:
                with self._lock:
                    self._error_count += 1
                if self._error_count % 100 == 1:
                    logger.warning(f"FSR read error: {e} on line: '{line}'")
                time.sleep(0.01)

        logger.info("FSR reader thread stopped")
