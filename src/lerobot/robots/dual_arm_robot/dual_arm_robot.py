#!/usr/bin/env python3
"""
LeRobot Integration for Dual-Arm Leader-Follower System (Stable & Optimized)

This version uses the reliable CalibrationManager for data conversion and
an optimized get_observation method that reads all motor data with a single
hardware call per arm for high performance.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# LeRobot imports
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

from .fsr_sensor import FSRSensor, FSRSensorConfig

# Import your refactored motor bus
try:
    from .mixed_feetech_bus import MixedFeetechMotorsBus, MotorConfig

    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import mixed_feetech_bus: {e}")
    HARDWARE_AVAILABLE = False

logger = logging.getLogger(__name__)
DEFAULT_CALIBRATION_FILE = str(Path(__file__).with_name("calibration.json"))

# =============================================================================
# CONFIGURATIONS (WITH ENABLE FLAGS)
# =============================================================================


@TeleoperatorConfig.register_subclass("dual_arm_leader")
@dataclass
class DualArmLeaderConfig(TeleoperatorConfig):
    right_leader_port: str = "/dev/ttyACM0"
    left_leader_port: str = "/dev/ttyACM2"
    calibration_file: str = DEFAULT_CALIBRATION_FILE
    enable_left_arm: bool = True
    enable_right_arm: bool = True


@RobotConfig.register_subclass("dual_arm_follower")
@dataclass
class DualArmFollowerConfig(RobotConfig):
    right_follower_port: str = "/dev/ttyACM1"
    left_follower_port: str = "/dev/ttyACM3"
    calibration_file: str = DEFAULT_CALIBRATION_FILE
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "right": OpenCVCameraConfig(index_or_path="/dev/video0", fps=30, width=320, height=240),
            "left": OpenCVCameraConfig(index_or_path="/dev/video2", fps=30, width=320, height=240),
            "front": OpenCVCameraConfig(index_or_path="/dev/video4", fps=30, width=640, height=480),
        }
    )
    # FSR ADDITIONS - these 2 lines are new:
    fsr_enabled: bool = True
    fsr_sensor: FSRSensorConfig = field(default_factory=FSRSensorConfig)
    control_frequency: int | None = None
    enable_left_arm: bool = True
    enable_right_arm: bool = True
    reduced_observation_state: bool = False


# =============================================================================
# CALIBRATION MANAGER (Restored)
# =============================================================================


class CalibrationManager:
    """Manages calibration data loading and position normalization"""

    def __init__(self, calibration_file: str):
        self.calibration_file = calibration_file
        self.data = self._load_calibration()

    def _load_calibration(self) -> dict[str, Any]:
        try:
            with open(self.calibration_file) as f:
                data = json.load(f)
                logger.info(f"Loaded calibration data from {self.calibration_file}")
                return data
        except FileNotFoundError:
            logger.error(f"Calibration file {self.calibration_file} not found!")
            raise

    def normalize_position(self, raw_pos: float, servo_id: int, arm_type: str) -> float:
        """Convert raw servo position to [-1, 1] range using calibration data"""
        config_section = (
            "leader_configs"
            if "leader" in arm_type
            else ("right_follower_configs" if "right" in arm_type else "left_follower_configs")
        )
        config = self.data[config_section].get(str(servo_id))
        if not config:
            return 0.0

        raw_min, raw_max = config["raw_min"], config["raw_max"]

        if raw_max < raw_min:
            normalized = 2.0 * (raw_min - raw_pos) / (raw_min - raw_max) - 1.0
        else:
            normalized = 2.0 * (raw_pos - raw_min) / (raw_max - raw_min) - 1.0

        return max(-1.0, min(1.0, normalized))

    def denormalize_position(self, norm_pos: float, servo_id: int, arm_type: str) -> float:
        """Convert [-1, 1] range to raw servo position using calibration data"""
        config_section = (
            "leader_configs"
            if "leader" in arm_type
            else ("right_follower_configs" if "right" in arm_type else "left_follower_configs")
        )
        config = self.data[config_section].get(str(servo_id))
        if not config:
            return 2048

        raw_min, raw_max = config["raw_min"], config["raw_max"]
        norm_pos = max(-1.0, min(1.0, norm_pos))

        if raw_max < raw_min:
            raw_pos = raw_min - (norm_pos + 1.0) * (raw_min - raw_max) / 2.0
        else:
            raw_pos = raw_min + (norm_pos + 1.0) * (raw_max - raw_min) / 2.0

        return int(raw_pos)


# =============================================================================
# HELPER FUNCTION TO CREATE MOTORS
# =============================================================================


def create_motors_and_configs(arm_type: str) -> tuple[dict[str, Motor], dict[int, MotorConfig]]:
    motor_id_mappings = {
        "right_leader": [1, 2, 3, 7, 4, 5, 6],
        "right_follower": [11, 12, 13, 14, 16, 15, 1, 17],
        "left_leader": [8, 9, 10, 14, 11, 12, 13],
        "left_follower": [11, 12, 13, 14, 16, 15, 1, 17],
    }
    motor_type_mappings = {
        "right_leader": ["STS"] * 7,
        "right_follower": ["HLS"] * 5 + ["STS"] * 3,
        "left_leader": ["STS"] * 7,
        "left_follower": ["HLS"] * 5 + ["STS"] * 3,
    }

    motor_ids = motor_id_mappings[arm_type]
    motor_types = motor_type_mappings[arm_type]

    motors = {}
    motor_configs = {}

    for i, (motor_id, motor_type) in enumerate(zip(motor_ids, motor_types, strict=True)):
        if "follower" in arm_type:
            joint_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: -1}
            joint_index = joint_map[i]
            motor_name = f"joint_{joint_index}" if joint_index != -1 else "gripper"
            if joint_index == 1:
                # FIX: Correctly name the two servos for joint 1
                servo_suffix = "_servo_1" if i == 1 else "_servo_2"
                motor_name += servo_suffix
        else:
            joint_index = i if i < 6 else -1
            motor_name = f"joint_{joint_index}" if joint_index != -1 else "gripper"

        motors[motor_name] = Motor(
            id=motor_id, model=f"{motor_type.lower()}_{motor_id}", norm_mode=MotorNormMode.RANGE_M100_100
        )
        motor_configs[motor_id] = MotorConfig(
            motor_id=motor_id,
            motor_type=motor_type,
            joint_index=joint_index,
            model=f"{motor_type.lower()}_{motor_id}",
            inverted=(joint_index == 1 and "servo_2" in motor_name),
        )

    return motors, motor_configs


# =============================================================================
# DUAL ARM LEADER (TELEOPERATOR) - WITH ENABLE FLAGS SUPPORT
# =============================================================================


class DualArmLeader(Teleoperator):
    config_class = DualArmLeaderConfig
    name = "dual_arm_leader"

    def __init__(self, config: DualArmLeaderConfig):
        super().__init__(config)
        self.config = config
        self.calibration = CalibrationManager(config.calibration_file)

        if HARDWARE_AVAILABLE:
            if self.config.enable_right_arm:
                right_motors, right_configs = create_motors_and_configs("right_leader")
                self.right_leader_bus = MixedFeetechMotorsBus(
                    config.right_leader_port, right_motors, right_configs
                )

            if self.config.enable_left_arm:
                left_motors, left_configs = create_motors_and_configs("left_leader")
                self.left_leader_bus = MixedFeetechMotorsBus(
                    config.left_leader_port, left_motors, left_configs
                )

        self._action_dict = dict.fromkeys(self.action_features, 0.0)

    @property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self.config.enable_right_arm:
            for part in [f"joint_{i}" for i in range(6)] + ["gripper"]:
                features[f"right_arm.{part}.pos"] = float
        if self.config.enable_left_arm:
            for part in [f"joint_{i}" for i in range(6)] + ["gripper"]:
                features[f"left_arm.{part}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        if not HARDWARE_AVAILABLE:
            return False
        right_ok = not self.config.enable_right_arm or (
            hasattr(self, "right_leader_bus") and self.right_leader_bus.is_connected
        )
        left_ok = not self.config.enable_left_arm or (
            hasattr(self, "left_leader_bus") and self.left_leader_bus.is_connected
        )
        return right_ok and left_ok

    def connect(self, **kwargs) -> None:
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Hardware not available")

        if self.config.enable_right_arm and hasattr(self, "right_leader_bus"):
            self.right_leader_bus.connect()
            self.right_leader_bus.disable_torque()

        if self.config.enable_left_arm and hasattr(self, "left_leader_bus"):
            self.left_leader_bus.connect()
            self.left_leader_bus.disable_torque()

        logger.info(f"{self} connected successfully.")

    def disconnect(self) -> None:
        if (
            self.config.enable_right_arm
            and hasattr(self, "right_leader_bus")
            and self.right_leader_bus.is_connected
        ):
            self.right_leader_bus.disconnect()
        if (
            self.config.enable_left_arm
            and hasattr(self, "left_leader_bus")
            and self.left_leader_bus.is_connected
        ):
            self.left_leader_bus.disconnect()

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        if self.config.enable_right_arm and hasattr(self, "right_leader_bus"):
            raw_right = self.right_leader_bus.sync_read_positions_only()["Present_Position"]
            for name, raw_pos in raw_right.items():
                motor_id = self.right_leader_bus.motors[name].id
                norm_pos = self.calibration.normalize_position(raw_pos, motor_id, "right_leader")
                action_key = f"right_arm.{name}.pos"
                if action_key in self._action_dict:
                    self._action_dict[action_key] = norm_pos

        if self.config.enable_left_arm and hasattr(self, "left_leader_bus"):
            raw_left = self.left_leader_bus.sync_read_positions_only()["Present_Position"]
            for name, raw_pos in raw_left.items():
                motor_id = self.left_leader_bus.motors[name].id
                norm_pos = self.calibration.normalize_position(raw_pos, motor_id, "left_leader")
                action_key = f"left_arm.{name}.pos"
                if action_key in self._action_dict:
                    self._action_dict[action_key] = norm_pos

        return self._action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True


# =============================================================================
# DUAL ARM FOLLOWER (ROBOT) - WITH ENABLE FLAGS SUPPORT
# =============================================================================


class DualArmFollower(Robot):
    config_class = DualArmFollowerConfig
    name = "dual_arm_follower"

    def __init__(self, config: DualArmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.calibration = CalibrationManager(config.calibration_file)

        if HARDWARE_AVAILABLE:
            if self.config.enable_right_arm:
                right_motors, right_configs = create_motors_and_configs("right_follower")
                self.right_follower_bus = MixedFeetechMotorsBus(
                    config.right_follower_port, right_motors, right_configs
                )

            if self.config.enable_left_arm:
                left_motors, left_configs = create_motors_and_configs("left_follower")
                self.left_follower_bus = MixedFeetechMotorsBus(
                    config.left_follower_port, left_motors, left_configs
                )

        # Only create cameras for enabled arms
        filtered_cameras = {}
        for cam_name, cam_config in config.cameras.items():
            if cam_name == "left" and not self.config.enable_left_arm:
                continue
            if cam_name == "right" and not self.config.enable_right_arm:
                continue
            filtered_cameras[cam_name] = cam_config

        self.cameras = make_cameras_from_configs(filtered_cameras)

        # FSR ADDITIONS:
        self.fsr_sensor = None
        if config.fsr_enabled:
            self.fsr_sensor = FSRSensor(config.fsr_sensor)

        # FREQUENCY CONTROL ADDITIONS:
        if self.config.control_frequency is not None:
            self.control_period = 1.0 / self.config.control_frequency
            self.last_action_timestamp = time.perf_counter()
        else:
            self.control_period = None

        # FOR FREQUENCY LOGGING:
        self._action_count_for_log = 0
        self._last_log_time = time.perf_counter()

        self._observation_dict = {
            key: 0.0 for key in self.observation_features if isinstance(self.observation_features[key], type)
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features = {}

        # Only add features for enabled arms
        arms_to_process = []
        if self.config.enable_right_arm:
            arms_to_process.append("right_arm")
        if self.config.enable_left_arm:
            arms_to_process.append("left_arm")

        for arm in arms_to_process:
            for part in [f"joint_{i}" for i in range(6)] + ["gripper"]:
                if self.config.reduced_observation_state:
                    # Reduced mode: only positions for joints, pos+load for gripper
                    if "joint_" in part:
                        features[f"{arm}.{part}.pos"] = float
                    else:  # gripper
                        features[f"{arm}.{part}.pos"] = float
                        features[f"{arm}.{part}.load"] = float
                else:
                    # Full mode: all observation types
                    for obs_type in ["pos", "speed", "load"]:
                        features[f"{arm}.{part}.{obs_type}"] = float

        # Only add cameras for enabled arms
        for cam_name, cam_config in self.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)

        # FSR ADDITION - only for enabled arms and not in reduced mode
        if self.fsr_sensor and not self.config.reduced_observation_state:
            if self.config.enable_right_arm:
                features["right_arm.gripper.fsr"] = float
            if self.config.enable_left_arm:
                features["left_arm.gripper.fsr"] = float

        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {
            key: val
            for key, val in self.observation_features.items()
            if "pos" in key and isinstance(val, type)
        }

    @property
    def is_connected(self) -> bool:
        if not HARDWARE_AVAILABLE:
            return False

        right_ok = not self.config.enable_right_arm or (
            hasattr(self, "right_follower_bus") and self.right_follower_bus.is_connected
        )
        left_ok = not self.config.enable_left_arm or (
            hasattr(self, "left_follower_bus") and self.left_follower_bus.is_connected
        )
        cameras_ok = not self.cameras or any(camera.is_connected for camera in self.cameras.values())

        return right_ok and left_ok and cameras_ok

    def connect(self) -> None:
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Hardware not available")

        # Connect buses based on enabled arms
        if self.config.enable_right_arm and hasattr(self, "right_follower_bus"):
            self.right_follower_bus.connect()
            self.right_follower_bus.enable_torque()

        if self.config.enable_left_arm and hasattr(self, "left_follower_bus"):
            self.left_follower_bus.connect()
            self.left_follower_bus.enable_torque()

        # Connect cameras (already filtered in __init__)
        for camera in self.cameras.values():
            camera.connect()

        # Warm up cameras
        logger.info("Warming up cameras...")
        for cam_name, camera in self.cameras.items():
            for attempt in range(3):  # Try up to 3 times
                try:
                    camera.async_read(timeout_ms=1500)  # Longer timeout for warm-up
                    logger.info(f"Camera {cam_name} warmed up successfully")
                    break
                except Exception as e:
                    logger.warning(f"Camera {cam_name} warm-up attempt {attempt + 1}: {e}")
                    if attempt == 2:
                        raise RuntimeError(f"Camera {cam_name} failed to warm up after 3 attempts") from e

        # FSR sensor connection (only if at least one arm is enabled)
        if self.fsr_sensor and (self.config.enable_left_arm or self.config.enable_right_arm):
            try:
                self.fsr_sensor.connect()
                logger.info("FSR sensor connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect FSR sensor: {e}. Continuing without FSR.")
                self.fsr_sensor = None

        logger.info(f"{self} connected successfully.")

    def disconnect(self) -> None:
        # FSR disconnection
        if self.fsr_sensor:
            self.fsr_sensor.disconnect()

        # Disconnect buses for enabled arms
        if (
            self.config.enable_right_arm
            and hasattr(self, "right_follower_bus")
            and self.right_follower_bus.is_connected
        ):
            self.right_follower_bus.disable_torque()
            self.right_follower_bus.disconnect()

        if (
            self.config.enable_left_arm
            and hasattr(self, "left_follower_bus")
            and self.left_follower_bus.is_connected
        ):
            self.left_follower_bus.disable_torque()
            self.left_follower_bus.disconnect()

        # Disconnect cameras
        for camera in self.cameras.values():
            camera.disconnect()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        raw_right_data = {}
        raw_left_data = {}

        def get_all_raw_data(bus, result_dict):
            result_dict.update(bus.sync_read_all_data())

        # Only read from enabled arms
        threads = []
        if self.config.enable_right_arm and hasattr(self, "right_follower_bus"):
            right_thread = threading.Thread(
                target=get_all_raw_data, args=(self.right_follower_bus, raw_right_data)
            )
            threads.append(right_thread)
            right_thread.start()

        if self.config.enable_left_arm and hasattr(self, "left_follower_bus"):
            left_thread = threading.Thread(
                target=get_all_raw_data, args=(self.left_follower_bus, raw_left_data)
            )
            threads.append(left_thread)
            left_thread.start()

        for thread in threads:
            thread.join()

        # Process raw data using the reliable CalibrationManager
        def process_raw_data(bus, raw_data, arm_name):
            for name, raw_pos in raw_data.get("Present_Position", {}).items():
                motor_id = bus.motors[name].id
                norm_pos = self.calibration.normalize_position(raw_pos, motor_id, arm_name + "_follower")
                self._observation_dict[f"{arm_name}.{name}.pos"] = norm_pos

            # For speed and load, we just use the raw values for now (only if not in reduced mode)
            if not self.config.reduced_observation_state:
                for name, raw_speed in raw_data.get("Present_Speed", {}).items():
                    self._observation_dict[f"{arm_name}.{name}.speed"] = raw_speed
                for name, raw_load in raw_data.get("Present_Load", {}).items():
                    self._observation_dict[f"{arm_name}.{name}.load"] = raw_load
            else:
                # In reduced mode, only keep gripper load
                for name, raw_load in raw_data.get("Present_Load", {}).items():
                    if "gripper" in name:
                        self._observation_dict[f"{arm_name}.{name}.load"] = raw_load

        if raw_right_data and self.config.enable_right_arm:
            process_raw_data(self.right_follower_bus, raw_right_data, "right_arm")
            self._aggregate_joint_1("right_arm")

        if raw_left_data and self.config.enable_left_arm:
            process_raw_data(self.left_follower_bus, raw_left_data, "left_arm")
            self._aggregate_joint_1("left_arm")

        # Read cameras
        for cam_name, camera in self.cameras.items():
            frame = camera.async_read(1000)
            if frame is not None:
                self._observation_dict[cam_name] = frame

        # FSR readings - only for enabled arms
        if self.fsr_sensor and self.fsr_sensor.is_connected:
            try:
                fsr_data = self.fsr_sensor.read()
                self._observation_dict.update(fsr_data)
            except Exception as e:
                logger.warning(f"Failed to read FSR sensor: {e}")
                if self.config.enable_right_arm:
                    self._observation_dict["right_arm.gripper.fsr"] = -1.0
                if self.config.enable_left_arm:
                    self._observation_dict["left_arm.gripper.fsr"] = -1.0
        else:
            # Set defaults for enabled arms
            if self.config.enable_right_arm:
                self._observation_dict["right_arm.gripper.fsr"] = -1.0
            if self.config.enable_left_arm:
                self._observation_dict["left_arm.gripper.fsr"] = -1.0

        return self._observation_dict

    def _aggregate_joint_1(self, arm_name: str):
        s1_pos = self._observation_dict.pop(f"{arm_name}.joint_1_servo_1.pos", 0.0)
        s2_pos = self._observation_dict.pop(f"{arm_name}.joint_1_servo_2.pos", 0.0)

        # Only aggregate speed and load if not in reduced mode
        if not self.config.reduced_observation_state:
            s1_speed = self._observation_dict.pop(f"{arm_name}.joint_1_servo_1.speed", 0.0)
            s2_speed = self._observation_dict.pop(f"{arm_name}.joint_1_servo_2.speed", 0.0)
            s1_load = self._observation_dict.pop(f"{arm_name}.joint_1_servo_1.load", 0.0)
            s2_load = self._observation_dict.pop(f"{arm_name}.joint_1_servo_2.load", 0.0)

            self._observation_dict[f"{arm_name}.joint_1.speed"] = (s1_speed - s2_speed) / 2
            self._observation_dict[f"{arm_name}.joint_1.load"] = s1_load - s2_load

        self._observation_dict[f"{arm_name}.joint_1.pos"] = (s1_pos + s2_pos) / 2

    def apply_action(self, action: dict[str, float]) -> None:
        # FREQUENCY CONTROL ADDITION
        if self.control_period:
            # Calculate the time elapsed since the last action
            elapsed_time = time.perf_counter() - self.last_action_timestamp
            # Calculate the necessary wait time to meet the control period
            wait_time = self.control_period - elapsed_time
            if wait_time > 0:
                precise_sleep(wait_time)
            # Update the timestamp for the next iteration
            self.last_action_timestamp = time.perf_counter()

        # FOR FREQUENCY LOGGING
        self._action_count_for_log += 1
        current_time = time.perf_counter()
        log_interval = current_time - self._last_log_time

        # Log frequency every 1.0 second
        if log_interval >= 1.0:
            frequency = self._action_count_for_log / log_interval
            logger.info(f"Control Loop Frequency: {frequency:.2f} Hz")
            # Reset counters for the next interval
            self._action_count_for_log = 0
            self._last_log_time = current_time

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        right_commands = {}
        left_commands = {}

        for key, value in action.items():
            parts = key.split(".")
            arm_name, logical_joint_name = parts[0], parts[1]

            # Skip actions for disabled arms
            if arm_name == "right_arm" and not self.config.enable_right_arm:
                continue
            if arm_name == "left_arm" and not self.config.enable_left_arm:
                continue

            arm_suffix = "_follower"
            bus = self.right_follower_bus if arm_name == "right_arm" else self.left_follower_bus
            commands = right_commands if arm_name == "right_arm" else left_commands

            # Skip if bus doesn't exist for this arm
            if not hasattr(self, "right_follower_bus") and arm_name == "right_arm":
                continue
            if not hasattr(self, "left_follower_bus") and arm_name == "left_arm":
                continue

            motor_names_to_command = []
            if logical_joint_name == "joint_1":
                motor_names_to_command.extend(["joint_1_servo_1", "joint_1_servo_2"])
            else:
                motor_names_to_command.append(logical_joint_name)

            for motor_name in motor_names_to_command:
                if motor_name in bus.motors:
                    motor_id = bus.motors[motor_name].id
                    raw_pos = self.calibration.denormalize_position(value, motor_id, arm_name + arm_suffix)
                    commands[motor_id] = raw_pos

        def write_to_bus(bus, commands):
            if commands:
                bus.batch_write_all_positions(commands)

        # Only write to enabled arms
        threads = []
        if right_commands and self.config.enable_right_arm and hasattr(self, "right_follower_bus"):
            right_thread = threading.Thread(
                target=write_to_bus, args=(self.right_follower_bus, right_commands)
            )
            threads.append(right_thread)
            right_thread.start()

        if left_commands and self.config.enable_left_arm and hasattr(self, "left_follower_bus"):
            left_thread = threading.Thread(target=write_to_bus, args=(self.left_follower_bus, left_commands))
            threads.append(left_thread)
            left_thread.start()

        for thread in threads:
            thread.join()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.apply_action(action)
        return action

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True
