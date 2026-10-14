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

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.lerobot_types import RobotAction
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_smart_servo_available, require_package

from ..teleoperator import Teleoperator
from .config_rebot_102_leader import RebotArm102LeaderTeleopConfig
from .readonly_transport import Readonly102Transport

if TYPE_CHECKING or _motorbridge_smart_servo_available:
    from motorbridge_smart_servo import FashionStarServo, ServoMonitor
else:
    FashionStarServo = None
    ServoMonitor = None

logger = logging.getLogger(__name__)

_SETTLE_SEC = 0.01


@dataclass(frozen=True)
class LeaderJointDiagnostic:
    """SDK-visible value/status; unreliable angles are diagnostic only."""

    name: str
    sdk_reliable: bool | None
    sdk_angle_deg: float | None
    status: str
    receive_timestamp_ns: int | None = None
    receive_sequence: int | None = None
    motor_status: int | None = None


class LeaderReadError(RuntimeError):
    """Reject a whole action while preserving actual returned SDK diagnostics."""

    def __init__(self, message, *, started_ns, completed_ns, diagnostics):
        super().__init__(message)
        self.request_started_ns = started_ns
        self.response_completed_ns = completed_ns
        self.joint_diagnostics = diagnostics


@dataclass(frozen=True)
class LeaderActionSnapshot:
    """Fresh SDK response and the existing absolute degree mapping.

    ``raw_action_deg`` contains the SDK's reliable angle before LeRobot's
    direction/range mapping, not unfiltered wire bytes. The SDK exposes no
    individual receive timestamps: completion is host availability, while
    request start supplies a conservative lower bound for freshness checks.
    """

    raw_action_deg: tuple[tuple[str, float], ...]
    action_deg: tuple[tuple[str, float], ...]
    request_started_ns: int
    response_completed_ns: int
    sequence: int
    timestamp_source: str = "host_sync_monitor_return"
    joint_diagnostics: tuple[LeaderJointDiagnostic, ...] = ()


class RebotArm102Leader(Teleoperator):
    """Seeed Studio StarArm102 / reBot Arm 102 leader arm.

    A 7-joint (incl. gripper) leader built on FashionStar UART smart servos. Servo
    communication is handled by the ``motorbridge-smart-servo`` package; this class
    only reads joint angles, so it produces actions but accepts no feedback.
    """

    config_class = RebotArm102LeaderTeleopConfig
    name = "rebot_102_leader"

    def __init__(self, config: RebotArm102LeaderTeleopConfig):
        if config.readonly_transport != "protocol_v1":
            require_package("motorbridge-smart-servo", extra="rebot", import_name="motorbridge_smart_servo")
        super().__init__(config)
        self.config = config
        self.bus: FashionStarServo | Readonly102Transport | None = None
        self.motor_names = list(config.joint_ids.keys())
        self._last_raw_positions: dict[str, float] = {}
        self._readonly_connection = False
        self._read_sequence = 0

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    @property
    def has_resources(self) -> bool:
        """A close failure retains the bus until its reader owns cleanup."""
        return self.bus is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        require_package("motorbridge-smart-servo", extra="rebot", import_name="motorbridge_smart_servo")
        logger.info(f"Connecting {self} on {self.config.port}...")
        bus = FashionStarServo(self.config.port, baudrate=self.config.baudrate)
        try:
            for motor_name, motor_id in self.config.joint_ids.items():
                if not bus.ping(motor_id):
                    raise RuntimeError(f"Servo not found for {motor_name} (id={motor_id}).")
                self._last_raw_positions[motor_name] = 0.0
            self.bus = bus

            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()
        except Exception:
            bus.close()
            self.bus = None
            raise

        logger.info(f"{self} connected.")

    @check_if_already_connected
    def connect_readonly(self) -> None:
        """Open only the UART; never unlock, calibrate, or reset multi-turn.

        This deliberately does not call ``connect(calibrate=False)``, which
        still configures servos. The audited motorbridge-smart-servo 0.0.4
        constructor only opens the serial transport and creates host state.
        """
        if self.config.readonly_transport == "protocol_v1":
            bus = Readonly102Transport(
                self.config.port, self.config.baudrate, tuple(self.config.joint_ids.values())
            )
            self.bus = bus
            try:
                bus.open()
            except Exception:
                if not bus.has_resources:
                    self.bus = None
                raise
        elif self.config.readonly_transport == "sdk":
            self.bus = FashionStarServo(self.config.port, baudrate=self.config.baudrate)
        else:
            raise ValueError("readonly_transport must be 'sdk' or 'protocol_v1'")
        self._readonly_connection = True
        self._read_sequence = 0

    @check_if_not_connected
    def read_action_snapshot(self, *, max_read_duration_s: float = 0.15) -> LeaderActionSnapshot:
        """Reject held/missing values and late responses without reusing cache.

        The response budget is checked after SDK return, not an interruptible
        I/O timeout. Version 0.0.4 exposes no Python timeout setter; its native
        read loop defaults to 100 ms overall and 10 ms per serial read and
        releases the GIL. Call this only from the independent leader worker.
        """
        if not math.isfinite(max_read_duration_s) or max_read_duration_s <= 0:
            raise ValueError("max_read_duration_s must be finite and positive")
        started_ns = time.perf_counter_ns()
        is_protocol = isinstance(self.bus, Readonly102Transport)
        if is_protocol:
            result = self.bus.read_monitors(max_read_duration_s=max_read_duration_s)
        else:
            result = self.bus.sync_monitor(list(self.config.joint_ids.values()))
        completed_ns = time.perf_counter_ns()
        if completed_ns - started_ns > max_read_duration_s * 1e9:
            raise TimeoutError("102 response exceeded max_read_duration_s")
        if set(result) != set(self.config.joint_ids.values()):
            raise RuntimeError("102 response joint IDs do not match the configured joints")
        raw_positions = {}
        diagnostics = []
        for name, motor_id in self.config.joint_ids.items():
            monitor = result[motor_id]
            reliable = True if is_protocol else getattr(monitor, "reliable", None)
            value = float(monitor.angle_deg) if monitor is not None else None
            if value is not None and not math.isfinite(value):
                status = "non-finite angle"
                value = None
            elif monitor is None:
                status = "missing response"
            elif reliable is not True:
                status = "no fresh reliable response (SDK filter hold or missing packet)"
            else:
                status = "reliable"
                raw_positions[name] = value
            diagnostics.append(
                LeaderJointDiagnostic(
                    name,
                    None if is_protocol else reliable if type(reliable) is bool else None,
                    value,
                    "packet_validated_no_sdk_angle_filter" if is_protocol else status,
                    receive_timestamp_ns=monitor.receive_timestamp_ns if is_protocol else None,
                    receive_sequence=monitor.receive_sequence if is_protocol else None,
                    motor_status=monitor.motor_status if is_protocol else None,
                )
            )
        if len(raw_positions) != len(self.config.joint_ids):
            failed = next(d for d in diagnostics if d.status != "reliable")
            raise LeaderReadError(
                f"102 {failed.name} has {failed.status}",
                started_ns=started_ns,
                completed_ns=completed_ns,
                diagnostics=tuple(diagnostics),
            )
        action = self._map_raw_positions(raw_positions)
        self._read_sequence += 1
        return LeaderActionSnapshot(
            raw_action_deg=tuple((f"{name}.pos", value) for name, value in raw_positions.items()),
            action_deg=tuple(action.items()),
            request_started_ns=started_ns,
            response_completed_ns=completed_ns,
            sequence=self._read_sequence,
            joint_diagnostics=tuple(diagnostics),
            timestamp_source="host_validated_monitor_packet_read_return"
            if is_protocol
            else "host_sync_monitor_return",
        )

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and set(self.calibration) == set(self.motor_names)

    def calibrate(self) -> None:
        if self._readonly_connection:
            raise RuntimeError("Calibration is forbidden on a read-only 102 connection")
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        input(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot Arm 102 to its zero pose and close the gripper.\n"
            "Press ENTER when ready..."
        )

        self.calibration = {}
        for motor_name, motor_id in self.config.joint_ids.items():
            self.bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
            self.bus.set_origin_point(motor_id)
            range_min, range_max = self.config.joint_ranges[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=motor_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        if self._readonly_connection:
            raise RuntimeError("Configuration is forbidden on a read-only 102 connection")
        for motor_id in self.config.joint_ids.values():
            self.bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
        # Reset the multi-turn counter of each servo individually.
        for motor_id in self.config.joint_ids.values():
            self.bus.reset_multi_turn(motor_id)

    def _read_raw_positions(self) -> dict[str, float]:
        result: dict[int, ServoMonitor | None] = self.bus.sync_monitor(list(self.config.joint_ids.values()))
        id_to_name = {v: k for k, v in self.config.joint_ids.items()}
        raw_positions: dict[str, float] = {}
        for motor_id, monitor in result.items():
            motor_name = id_to_name[motor_id]
            if monitor is None:
                raise RuntimeError(f"Servo {motor_name} (id={motor_id}) has never responded.")
            raw_positions[motor_name] = monitor.angle_deg
        return raw_positions

    @staticmethod
    def _round_to_valid_range(value: float, min_value: float, max_value: float) -> tuple[float, int]:
        """Unwrap a multi-turn angle into the ±180° window centred on (min+max)/2.

        The servo may report an angle that has accumulated extra full rotations
        (value = true_angle + N*360). Subtract the nearest whole number of turns
        to bring it back into [center-180, center+180]. Returns the unwrapped
        angle and the number of turns removed.
        """
        center = (min_value + max_value) / 2.0
        turns = round((value - center) / 360.0)
        return value - turns * 360.0, abs(turns)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        try:
            raw_positions = self._read_raw_positions()
            self._last_raw_positions = raw_positions
        except Exception as e:
            logger.error(f"Failed to read raw positions: {e}")
            logger.warning("[EMERGENCY STOP] Hold the follower arm and cut off the main power to the arms.")
            logger.warning(
                "[EMERGENCY STOP] Break the teleoperation session and check the leader USB connection or power."
            )
            raw_positions = self._last_raw_positions

        action_dict = self._map_raw_positions(raw_positions)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict

    def _map_raw_positions(self, raw_positions: dict[str, float]) -> dict[str, float]:
        """Apply the unchanged native absolute direction/unwrap/clamp mapping."""
        action_dict: dict[str, float] = {}
        for motor_name in self.motor_names:
            range_min, range_max = self.config.joint_ranges[motor_name]
            direction = self.config.joint_directions[motor_name]
            sign = 1.0 if direction >= 0 else -1.0
            unwrapped, k = self._round_to_valid_range(
                raw_positions[motor_name], range_min * sign, range_max * sign
            )
            position = unwrapped * direction
            if k > 0:
                logger.debug(
                    f"Servo {motor_name} (id={self.config.joint_ids[motor_name]}) wrapped {k} * 360°. "
                    f"Unwrapped pos: {unwrapped:.1f}° (raw: {raw_positions[motor_name]:.1f}°)"
                )
            action_dict[f"{motor_name}.pos"] = max(float(range_min), min(float(range_max), position))

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the reBot Arm 102 leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.close()
        self.bus = None
        self._readonly_connection = False
        logger.info(f"{self} disconnected.")
