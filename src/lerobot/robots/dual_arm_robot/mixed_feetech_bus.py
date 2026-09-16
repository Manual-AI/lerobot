#!/usr/bin/env python3
"""
High-Performance Mixed Feetech Motor Bus (Stable Version)

This version focuses on providing efficient, low-level hardware access
for reading and writing raw motor data. All normalization and calibration
logic is handled at the robot level.
"""

import logging
from dataclasses import dataclass

from lerobot.motors.motors_bus import Motor, MotorsBus

# LeRobot imports
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# Feetech SDK imports
try:
    from .FTServo_Python import scservo_sdk as servo_sdk
except (ImportError, ModuleNotFoundError):
    try:
        import scservo_sdk as servo_sdk
    except (ImportError, ModuleNotFoundError):
        servo_sdk = None

HARDWARE_AVAILABLE = servo_sdk is not None

logger = logging.getLogger(__name__)


@dataclass
class MotorConfig:
    """Configuration for a single motor"""

    motor_id: int
    motor_type: str
    joint_index: int
    model: str
    inverted: bool = False


class MixedFeetechMotorsBus(MotorsBus):
    """
    High-performance bus that provides raw data access. It is not responsible
    for normalization, which is handled by the Robot class.
    """

    # FIX: Restore the control and resolution tables required by the base class constructor.
    model_ctrl_table = {
        **{f"sts_{i}": {"Torque_Enable": (40, 1)} for i in range(1, 18)},
        **{f"hls_{i}": {"Torque_Enable": (40, 1)} for i in range(11, 17)},
    }
    model_resolution_table = {
        **{f"sts_{i}": 4096 for i in range(1, 18)},
        **{f"hls_{i}": 4096 for i in range(11, 17)},
    }

    # Other base class attributes can remain empty as they are not used by the constructor.
    normalized_data = []
    model_encoding_table = {}
    model_baudrate_table = {}
    model_number_table = {}
    apply_drive_mode = False

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        motor_configs: dict[int, MotorConfig],
        baudrate: int = 1000000,
    ):
        super().__init__(port, motors, calibration={})

        self.motor_configs = motor_configs
        self.baudrate = baudrate
        self.port_handler = None
        self.sts_handler = None
        self.hls_handler = None

        self.sts_motors = {
            mid: mcfg for mid, mcfg in motor_configs.items() if mcfg.motor_type.upper() == "STS"
        }
        self.hls_motors = {
            mid: mcfg for mid, mcfg in motor_configs.items() if mcfg.motor_type.upper() == "HLS"
        }

        self.sts_sync_read_all = None
        self.hls_sync_read_all = None
        self.sts_sync_read_positions = None

        logger.info(f"Initialized MixedFeetechMotorsBus on {port}")

    @property
    def is_connected(self) -> bool:
        return self.port_handler is not None and self.port_handler.is_open

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"MotorsBus on port '{self.port}' is already connected")

        if not HARDWARE_AVAILABLE:
            raise ConnectionError("FTServo_Python SDK not found. Cannot connect.")

        try:
            self.port_handler = servo_sdk.PortHandler(self.port)
            if not self.port_handler.openPort():
                raise ConnectionError(f"Failed to open port {self.port}")
            if not self.port_handler.setBaudRate(self.baudrate):
                raise ConnectionError(f"Failed to set baudrate {self.baudrate}")

            if self.sts_motors:
                self.sts_handler = servo_sdk.sms_sts(self.port_handler)
                self._setup_sts_batch_operations()
                self._setup_sts_position_only_read()

            if self.hls_motors:
                self.hls_handler = servo_sdk.hls(self.port_handler)
                self._setup_hls_batch_operations()

            if handshake:
                self._handshake()

            logger.info(f"Successfully connected to {self.port}")

        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect to motor bus: {e}") from e

    def _setup_sts_batch_operations(self):
        if not self.sts_motors:
            return
        self.sts_sync_read_all = servo_sdk.GroupSyncRead(
            self.sts_handler, servo_sdk.SMS_STS_PRESENT_POSITION_L, 6
        )
        for motor_id in self.sts_motors:
            self.sts_sync_read_all.addParam(motor_id)

    def _setup_sts_position_only_read(self):
        if not self.sts_motors:
            return
        self.sts_sync_read_positions = servo_sdk.GroupSyncRead(
            self.sts_handler, servo_sdk.SMS_STS_PRESENT_POSITION_L, 2
        )
        for motor_id in self.sts_motors:
            self.sts_sync_read_positions.addParam(motor_id)

    def _setup_hls_batch_operations(self):
        if not self.hls_motors:
            return
        self.hls_sync_read_all = servo_sdk.GroupSyncRead(
            self.hls_handler, servo_sdk.HLS_PRESENT_POSITION_L, 6
        )
        for motor_id in self.hls_motors:
            self.hls_sync_read_all.addParam(motor_id)

    def sync_read_all_data(self) -> dict[str, dict[str, int]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Motor bus not connected on {self.port}")

        all_data = {"Present_Position": {}, "Present_Speed": {}, "Present_Load": {}}

        def robust_single_read(group_sync_read_obj, motor_ids, start_address):
            """
            Robust sync read that handles both HLS and STS servos with proper data conversion.
            """
            # Execute the sync read command
            comm_result = group_sync_read_obj.txRxPacket()
            if comm_result != servo_sdk.COMM_SUCCESS:
                logger.error(f"Sync read txRxPacket failed on port {self.port}")
                return

            # Process each motor's data
            for mid in motor_ids:
                # Check if data is available for this motor (6 bytes: pos + speed + load)
                if not group_sync_read_obj.isAvailable(mid, start_address, 6):
                    continue

                try:
                    # Read raw 2-byte values
                    raw_position = group_sync_read_obj.getData(mid, start_address, 2)
                    raw_speed = group_sync_read_obj.getData(mid, start_address + 2, 2)
                    raw_load = group_sync_read_obj.getData(mid, start_address + 4, 2)

                    # Position: No conversion needed (already in correct format)
                    all_data["Present_Position"][mid] = raw_position

                    # Speed: Convert using 15-bit signed format (BIT15 = direction)
                    if mid in self.sts_motors:
                        converted_speed = self.sts_handler.scs_tohost(raw_speed, 15)
                        converted_speed = (
                            converted_speed * 0.0146 / 0.732
                        )  # Convert 0.0146RPM to 0.732RPM units
                    else:  # HLS motor
                        converted_speed = self.hls_handler.scs_tohost(raw_speed, 15)

                    # Load: Convert using 10-bit signed format (BIT10 = direction)
                    if mid in self.sts_motors:
                        converted_load = self.sts_handler.scs_tohost(raw_load, 10)
                    else:  # HLS motor
                        converted_load = self.hls_handler.scs_tohost(raw_load, 10)

                    # Store converted values
                    all_data["Present_Speed"][mid] = converted_speed
                    all_data["Present_Load"][mid] = converted_load

                except (TypeError, KeyError) as e:
                    logger.warning(f"SDK error while getting data for motor {mid}: {e}. Skipping this cycle.")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing motor {mid}: {e}. Skipping this cycle.")
                    continue

        # Execute reads for each motor type
        if self.sts_motors:
            robust_single_read(self.sts_sync_read_all, self.sts_motors, servo_sdk.SMS_STS_PRESENT_POSITION_L)
        if self.hls_motors:
            robust_single_read(self.hls_sync_read_all, self.hls_motors, servo_sdk.HLS_PRESENT_POSITION_L)

        # Convert motor IDs to motor names
        named_data = {"Present_Position": {}, "Present_Speed": {}, "Present_Load": {}}
        id_to_name_map = {motor.id: name for name, motor in self.motors.items()}

        for data_name, values in all_data.items():
            for motor_id, raw_value in values.items():
                if motor_id in id_to_name_map:
                    named_data[data_name][id_to_name_map[motor_id]] = raw_value

        return named_data

    def sync_read_positions_only(self) -> dict[str, dict[str, int]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Motor bus not connected on {self.port}")

        position_data = {"Present_Position": {}}

        def robust_position_read(group_sync_read_obj, motor_ids, start_address):
            comm_result = group_sync_read_obj.txRxPacket()
            if comm_result != servo_sdk.COMM_SUCCESS:
                return
            for mid in motor_ids:
                if group_sync_read_obj.isAvailable(mid, start_address, 2):
                    try:
                        position_data["Present_Position"][mid] = group_sync_read_obj.getData(
                            mid, start_address, 2
                        )
                    except (TypeError, KeyError):
                        logger.warning(f"SDK error while getting position for motor {mid}. Skipping.")
                        continue

        if self.sts_motors:
            robust_position_read(
                self.sts_sync_read_positions, self.sts_motors, servo_sdk.SMS_STS_PRESENT_POSITION_L
            )

        named_data = {"Present_Position": {}}
        id_to_name_map = {motor.id: name for name, motor in self.motors.items()}
        for motor_id, position in position_data["Present_Position"].items():
            if motor_id in id_to_name_map:
                named_data["Present_Position"][id_to_name_map[motor_id]] = position

        return named_data

    def batch_write_all_positions(self, positions: dict[int, int]):
        hls_servo_ids = {m.motor_id for m in self.motor_configs.values() if m.motor_type == "HLS"}

        has_sts_commands = False
        has_hls_commands = False

        for motor_id, position in positions.items():
            if motor_id in hls_servo_ids:
                self.hls_handler.SyncWritePosEx(motor_id, int(position), 110, 255, 1000)
                has_hls_commands = True
            else:
                self.sts_handler.SyncWritePosEx(motor_id, int(position), 0, 255)
                has_sts_commands = True

        if has_hls_commands:
            self.hls_handler.groupSyncWrite.txPacket()
            self.hls_handler.groupSyncWrite.clearParam()
        if has_sts_commands:
            self.sts_handler.groupSyncWrite.txPacket()
            self.sts_handler.groupSyncWrite.clearParam()

    def disconnect(self) -> None:
        try:
            if self.port_handler and self.port_handler.is_open:
                self.port_handler.closePort()
            logger.info(f"Disconnected from {self.port}")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

    def _handshake(self) -> None:
        # A simple ping is sufficient for a handshake
        all_motor_ids = list(self.sts_motors.keys()) + list(self.hls_motors.keys())
        for motor_id in all_motor_ids:
            handler = self.sts_handler if motor_id in self.sts_motors else self.hls_handler
            _result, comm, _error = handler.ping(motor_id)
            if comm != servo_sdk.COMM_SUCCESS:
                logger.warning(f"Motor {motor_id} failed handshake.")

    def enable_torque(self, motors: list[str] | None = None, **kwargs) -> None:
        ids = [self.motors[motor].id for motor in (motors or self.motors.keys()) if motor in self.motors]
        for motor_id in ids:
            handler = self.sts_handler if motor_id in self.sts_motors else self.hls_handler
            addr = (
                servo_sdk.SMS_STS_TORQUE_ENABLE
                if motor_id in self.sts_motors
                else servo_sdk.HLS_TORQUE_ENABLE
            )
            handler.write1ByteTxRx(motor_id, addr, 1)

    def disable_torque(self, motors: list[str] | None = None, **kwargs) -> None:
        ids = [self.motors[motor].id for motor in (motors or self.motors.keys()) if motor in self.motors]
        for motor_id in ids:
            handler = self.sts_handler if motor_id in self.sts_motors else self.hls_handler
            addr = (
                servo_sdk.SMS_STS_TORQUE_ENABLE
                if motor_id in self.sts_motors
                else servo_sdk.HLS_TORQUE_ENABLE
            )
            handler.write1ByteTxRx(motor_id, addr, 0)

    # All abstract methods from the base class must be implemented.
    @property
    def is_calibrated(self) -> bool:
        # Since calibration is handled externally, we can report True.
        return True

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        pass

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _split_into_byte_chunks(self, value: int, length: int) -> list:
        pass

    def _find_single_motor(self, motor: str, initial_baudrate: int = None) -> tuple:
        pass

    def _get_half_turn_homings(self, positions) -> dict:
        pass

    def broadcast_ping(self, **kwargs) -> dict:
        pass

    def configure_motors(self) -> None:
        pass

    def read_calibration(self) -> dict:
        return {}

    def write_calibration(self, calibration: dict, **kwargs) -> None:
        pass

    def _encode_sign(self, data_name: str, ids_values: dict) -> dict:
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict) -> dict:
        return ids_values
