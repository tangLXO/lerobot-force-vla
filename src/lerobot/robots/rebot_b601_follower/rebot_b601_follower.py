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
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.force_control.hardware_types import (
    BoundedJointFeedbackSource,
    JointFeedback,
    JointSendResult,
    JointSnapshot,
    SendReceipt,
)
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig
from .control_io import B601BusLease, audit_motorbridge_capabilities

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# Joint controlled in FORCE_POS mode; every other joint runs in POS_VEL mode.
GRIPPER_MOTOR = "gripper"
# Per-joint Damiao motor models for the B601-DM (passed to motorbridge).
MOTOR_MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601-DM follower arm (6-DOF + gripper, Damiao CAN motors).

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or a SocketCAN adapter.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras = make_cameras_from_configs(config.cameras)
        self._io_owner_id: int | None = None
        self._io_owner_pid: int | None = None
        self._io_owner_thread: threading.Thread | None = None
        self._io_owner_lock = threading.Lock()
        self._observe_only = False
        self._control_io_fault: str | None = None
        self._joint_feedback_source: BoundedJointFeedbackSource | None = None
        self._joint_feedback_max_age_ns: int | None = None
        self._last_joint_snapshot: JointSnapshot | None = None
        self._joint_receive_sequences: dict[str, int] = {}
        self._bus_lease: B601BusLease | None = None

    def preflight_control(self) -> None:
        """Refuse unsupported Teacher hardware before ownership, transport or motion.

        The reusable runtime calls this even when constructed programmatically,
        so bypassing the command-line entrypoint cannot reach legacy connect's
        calibration/configuration. A mock feedback source does not authorize a
        real motorbridge transport or establish a validated physical fault stop.
        """
        audit_motorbridge_capabilities().require_teacher_ready()

    def bind_io_owner(self) -> None:
        """Bind every application-side B601 I/O call to the current thread.

        Binding persists through disconnect. A driver object cannot be passed to
        another owner while old references may still issue commands.
        """
        # Reject a forked copy before touching an inherited threading lock.
        if self._io_owner_pid is not None and self._io_owner_pid != os.getpid():
            raise RuntimeError("B601 I/O is owned by a different process")
        with self._io_owner_lock:
            current = threading.get_ident()
            if self._io_owner_id is None:
                self._io_owner_id = current
                self._io_owner_pid = os.getpid()
                self._io_owner_thread = threading.current_thread()
            elif self._io_owner_id != current or self._io_owner_thread is not threading.current_thread():
                raise RuntimeError("B601 I/O is owned by a different thread")

    def _assert_io_owner(self, *, motion: bool = False) -> None:
        self.bind_io_owner()
        if motion and self._observe_only:
            raise RuntimeError("B601 observe_only transport forbids motion/configuration")
        if motion and self._control_io_fault is not None:
            raise RuntimeError(f"B601 control I/O fault is latched: {self._control_io_fault}")

    def install_joint_feedback_source(self, source: BoundedJointFeedbackSource, *, max_age_ns: int) -> None:
        """Install an explicitly reviewed adapter before connection, or a test double.

        This is not SDK feature detection. motorbridge 0.3.9 has no implementation
        of this protocol; the hardware runner must refuse its motion preflight.
        """
        self._assert_io_owner()
        if self.bus is not None:
            raise RuntimeError("Install feedback adapter before opening the B601 transport")
        if type(max_age_ns) is not int or max_age_ns <= 0:
            raise ValueError("max_age_ns must be a positive integer")
        self._joint_feedback_source = source
        self._joint_feedback_max_age_ns = max_age_ns

    def _open_transport(self) -> None:
        self._assert_io_owner()
        if self.bus is not None or self._bus_lease is not None:
            raise RuntimeError("B601 transport or its ownership lease is already open")
        lease = B601BusLease(self.config.can_adapter, self.config.port)
        lease.acquire()
        self._bus_lease = lease
        try:
            if self.config.can_adapter == "damiao":
                self.bus = MotorBridgeController.from_dm_serial(
                    serial_port=self.config.port, baud=self.config.dm_serial_baud
                )
            elif self.config.can_adapter == "socketcan":
                self.bus = MotorBridgeController(channel=self.config.port)
            for name, (send_id, recv_id) in self.config.motor_can_ids.items():
                self.motors[name] = self.bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[name])
        except BaseException:
            for motor in self.motors.values():
                motor.close()
            if self.bus is not None:
                self.bus.close()
            # A close exception deliberately retains the lease and handles.
            self.bus = None
            self.motors.clear()
            self._release_bus_lease()
            raise

    def _release_bus_lease(self) -> None:
        self._assert_io_owner()
        if self.bus is not None:
            raise RuntimeError("Cannot release B601 ownership before confirmed transport closure")
        if self._bus_lease is not None:
            self._bus_lease.release()
            self._bus_lease = None

    def connect_observe_only(self) -> None:
        """Open only transport/handles: no enable, mode, zero, calibration or camera.

        This does not make cached SDK feedback trustworthy. Without a reviewed
        bounded feedback adapter, read_joint_snapshot reports invalid evidence.
        motorbridge 0.3.9 starts an internal reader when adding motors; this
        legacy handle-only entry is not the independent single-owner probe.
        Closing this transport never disables motors or clears motor errors.
        """
        self._assert_io_owner()
        if self.bus is not None:
            raise RuntimeError("B601 transport is already open")
        self._observe_only = True
        self._open_transport()

    def close_observe_only(self) -> None:
        self._assert_io_owner()
        if not self._observe_only:
            raise RuntimeError("close_observe_only requires an observe-only connection")
        for motor in self.motors.values():
            motor.close()
        if self.bus is not None:
            self.bus.close()
        self.motors.clear()
        self.bus = None
        self._release_bus_lease()
        self._last_joint_snapshot = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus is not None and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._assert_io_owner(motion=True)
        logger.info(f"Connecting {self} on {self.config.port} (adapter={self.config.can_adapter})...")
        self._open_transport()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        self._assert_io_owner(motion=True)
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_all()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot B601 to its ZERO POSITION and close the gripper.\n"
            "See the B601 manual for the zero pose (the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(_ZERO_SETTLE_SEC)
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self._assert_io_owner(motion=True)
        if self.config.control_mode not in ("pos_vel", "mit"):
            raise ValueError(
                f"Unsupported control_mode '{self.config.control_mode}'. Use 'pos_vel' or 'mit'."
            )
        if self.config.gripper_control_mode not in ("force_pos", "mit"):
            raise ValueError(
                f"Unsupported gripper_control_mode '{self.config.gripper_control_mode}'. "
                "Use 'force_pos' or 'mit'."
            )
        use_mit = self.config.control_mode == "mit"
        gripper_use_mit = self.config.gripper_control_mode == "mit"
        self.bus.enable_all()
        for motor_name, motor in self.motors.items():
            if motor_name == GRIPPER_MOTOR:
                target_mode = MotorBridgeMode.MIT if gripper_use_mit else MotorBridgeMode.FORCE_POS
            elif use_mit:
                target_mode = MotorBridgeMode.MIT
            else:
                target_mode = MotorBridgeMode.POS_VEL
            for attempt in range(_ENSURE_MODE_RETRIES + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception:
                    if attempt == _ENSURE_MODE_RETRIES:
                        raise
                    time.sleep(_SETTLE_SEC)
            logger.debug(f"{motor_name} mode set to {target_mode}")

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        self._assert_io_owner(motion=True)
        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def _present_pos(self) -> dict[str, float]:
        """Read present joint positions in degrees."""
        self._assert_io_owner()
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("CAN bus poll feedback failed.")

        present_pos = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            present_pos[motor_name] = math.degrees(state.pos) if state is not None else 0.0
        return present_pos

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._assert_io_owner()
        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": pos for motor, pos in self._present_pos().items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    def read_joint_snapshot(self, names: Sequence[str], *, deadline_ns: int) -> JointSnapshot:
        """Return evidence for exactly these joints, with no cameras or cache restamping.

        Missing audited capability returns invalid data without calling the SDK's
        unbounded poll. Installing a protocol implementation is an explicit adapter
        review step, not a promise that the installed motorbridge has RX metadata.
        """
        self._assert_io_owner()
        requested = tuple(names)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("names must be nonempty and unique")
        if set(requested) - set(self.motor_names):
            raise ValueError("Unknown B601 joint name (use bare names, with positions in degrees)")
        if type(deadline_ns) is not int or deadline_ns <= 0:
            raise ValueError("deadline_ns must be a positive integer")
        error = None
        snapshot = None
        if self.bus is None:
            error = "transport_disconnected"
        elif self._joint_feedback_source is None:
            error = "unavailable_bounded_receive_evidence"
        elif time.perf_counter_ns() >= deadline_ns:
            error = "deadline_expired"
        else:
            try:
                snapshot = self._joint_feedback_source.read_snapshot(requested, deadline_ns=deadline_ns)
            except Exception as exc:
                error = f"feedback_error: {type(exc).__name__}: {exc}"
        completed = time.perf_counter_ns()
        if completed > deadline_ns:
            error = error or "feedback_deadline_exceeded"
        joints = []
        if snapshot is not None:
            if len({joint.name for joint in snapshot.joints}) != len(snapshot.joints):
                error = "duplicate_feedback_joint"
            error = error or snapshot.error
        for name in requested:
            incoming = (
                next((joint for joint in snapshot.joints if joint.name == name), None) if snapshot else None
            )
            if incoming is None:
                joints.append(
                    JointFeedback(name, None, False, None, None, None, error=error or "missing_state")
                )
                continue
            receive = incoming.receive_timestamp_ns
            sequence = incoming.receive_sequence
            joint_error = error or incoming.error
            age = completed - receive if type(receive) is int else None
            if not incoming.is_valid:
                joint_error = joint_error or "invalid_feedback"
            if (
                incoming.position_deg is None
                or isinstance(incoming.position_deg, bool)
                or not isinstance(incoming.position_deg, (float, int))
                or not math.isfinite(incoming.position_deg)
            ):
                joint_error = joint_error or "nonfinite_or_missing_position"
            if type(receive) is not int or receive <= 0 or receive > completed:
                joint_error = joint_error or "invalid_receive_timestamp"
            if type(sequence) is not int or sequence < 0:
                joint_error = joint_error or "invalid_receive_sequence"
            elif sequence <= self._joint_receive_sequences.get(name, -1):
                joint_error = joint_error or "receive_sequence_not_advanced"
            if incoming.time_source != "host_receive_perf_counter_ns":
                joint_error = joint_error or "unverified_clock_domain"
            if age is None or age < 0 or age > self._joint_feedback_max_age_ns:
                joint_error = joint_error or "stale_feedback"
            if joint_error is None:
                self._joint_receive_sequences[name] = sequence
            joints.append(replace(incoming, is_valid=joint_error is None, age_ns=age, error=joint_error))
        result = JointSnapshot(tuple(joints), completed, deadline_ns, error)
        self._last_joint_snapshot = result
        return result

    @staticmethod
    def _finite_number(value: float, name: str, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric value")
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive")
        return float(value)

    def _validated_joint_goal(self, name: str, candidate: float, feedback: JointFeedback) -> float:
        candidate = self._finite_number(candidate, f"{name} position_deg")
        if name not in self.config.joint_limits:
            raise ValueError(f"Explicit degree limits are required for {name}")
        low, high = self.config.joint_limits[name]
        low = self._finite_number(low, f"{name} lower limit")
        high = self._finite_number(high, f"{name} upper limit")
        if low >= high:
            raise ValueError(f"Invalid joint limits for {name}")
        relative = self.config.max_relative_target
        if relative is not None:
            if isinstance(relative, dict):
                if name not in relative:
                    raise ValueError(f"Missing max_relative_target for {name}")
                relative = relative[name]
            relative = self._finite_number(relative, f"{name} max_relative_target", positive=True)
            low = max(low, feedback.position_deg - relative)
            high = min(high, feedback.position_deg + relative)
            if low > high:
                raise ValueError(f"Measured-relative and soft limits have no intersection for {name}")
        return max(low, min(high, candidate))

    def _validate_joint_command_config(self, name: str) -> None:
        index = self.motor_names.index(name)
        mode = self.config.gripper_control_mode if name == GRIPPER_MOTOR else self.config.control_mode
        if mode not in (("force_pos", "mit") if name == GRIPPER_MOTOR else ("pos_vel", "mit")):
            raise ValueError(f"Invalid control mode for {name}")
        if mode == "mit":
            if name == GRIPPER_MOTOR:
                gains = self.config.gripper_mit_kp, self.config.gripper_mit_kd
            else:
                gains = tuple(
                    values[index] if isinstance(values, list) else values
                    for values in (self.config.mit_kp, self.config.mit_kd)
                )
            if any(self._finite_number(value, f"{name} MIT gain") < 0 for value in gains):
                raise ValueError("MIT gains must be nonnegative")
        else:
            velocity = self.config.pos_vel_velocity
            velocity = velocity[index] if isinstance(velocity, list) else velocity
            self._finite_number(velocity, f"{name} velocity deg/s", positive=True)
        if mode == "force_pos":
            ratio = self._finite_number(self.config.gripper_torque_ratio, "gripper_torque_ratio")
            if not 0 <= ratio <= 1:
                raise ValueError("gripper_torque_ratio must be in [0, 1]")

    def _send_joint_position(self, name: str, position_deg: float) -> None:
        """One SDK submission, used only after caller validation; never polls."""
        self._assert_io_owner(motion=True)
        motor = self.motors[name]
        index = self.motor_names.index(name)
        position = math.radians(position_deg)
        if name == GRIPPER_MOTOR and self.config.gripper_control_mode == "mit":
            motor.send_mit(position, 0.0, self.config.gripper_mit_kp, self.config.gripper_mit_kd, 0.0)
        elif name != GRIPPER_MOTOR and self.config.control_mode == "mit":
            kp = self.config.mit_kp[index] if isinstance(self.config.mit_kp, list) else self.config.mit_kp
            kd = self.config.mit_kd[index] if isinstance(self.config.mit_kd, list) else self.config.mit_kd
            motor.send_mit(position, 0.0, kp, kd, 0.0)
        else:
            velocity = self.config.pos_vel_velocity
            velocity = velocity[index] if isinstance(velocity, list) else velocity
            if name == GRIPPER_MOTOR:
                motor.send_force_pos(position, math.radians(velocity), self.config.gripper_torque_ratio)
            else:
                motor.send_pos_vel(position, math.radians(velocity))

    def send_joint_positions_partial(
        self, goals_deg: Mapping[str, float], *, feedback_snapshot: JointSnapshot
    ) -> SendReceipt:
        """Submit only named joints; never synthesizes wrist_yaw or reads a bus.

        Validate the entire request before the first write. On the first SDK
        exception, mark that command unknown, do not attempt later joints, and
        latch normal motion. No command-level hardware acknowledgement is claimed.
        """
        self._assert_io_owner(motion=True)
        if self.bus is None:
            raise RuntimeError("B601 transport is disconnected")
        if not goals_deg or set(goals_deg) - set(self.motors):
            raise ValueError("Partial targets require known bare joint names and degree values")
        if feedback_snapshot is not self._last_joint_snapshot:
            raise ValueError("Use the latest snapshot returned by this B601 adapter")
        now = time.perf_counter_ns()
        clipped = {}
        for name, candidate in goals_deg.items():
            feedback = feedback_snapshot.get(name)
            if (
                not feedback_snapshot.is_valid
                or not feedback.is_valid
                or feedback.receive_timestamp_ns is None
            ):
                raise ValueError(f"Fresh valid joint feedback is required for {name}")
            if now - feedback.receive_timestamp_ns > self._joint_feedback_max_age_ns:
                raise ValueError(f"Joint feedback expired before submission for {name}")
            clipped[name] = self._validated_joint_goal(name, candidate, feedback)
            self._validate_joint_command_config(name)
        started = time.perf_counter_ns()
        results = []
        failed = False
        for name, candidate in goals_deg.items():
            if failed:
                results.append(
                    JointSendResult(
                        name,
                        candidate,
                        clipped[name],
                        None,
                        "not_attempted",
                        None,
                        None,
                        error="prior_joint_submission_unknown",
                    )
                )
                continue
            begin = time.perf_counter_ns()
            try:
                self._send_joint_position(name, clipped[name])
            except Exception as exc:
                self._control_io_fault = f"{name}: {type(exc).__name__}: {exc}"
                results.append(
                    JointSendResult(
                        name,
                        candidate,
                        clipped[name],
                        None,
                        "unknown",
                        begin,
                        time.perf_counter_ns(),
                        error=self._control_io_fault,
                    )
                )
                failed = True
            else:
                results.append(
                    JointSendResult(
                        name,
                        candidate,
                        clipped[name],
                        clipped[name],
                        "success",
                        begin,
                        time.perf_counter_ns(),
                    )
                )
        return SendReceipt(tuple(results), started, time.perf_counter_ns())

    def send_gripper_position(self, position_deg: float, *, feedback_snapshot: JointSnapshot) -> SendReceipt:
        return self.send_joint_positions_partial(
            {GRIPPER_MOTOR: position_deg}, feedback_snapshot=feedback_snapshot
        )

    def fault_stop_gripper(self, *, reason: str) -> None:
        """Explicit restricted fault channel; current SDK has no validated stop.

        Refusal must be recorded by the runtime as stop-unconfirmed. Clearing a
        sensor latch or issuing an old position target is never a fallback.
        """
        self._assert_io_owner()
        self._control_io_fault = reason
        raise RuntimeError("B601 gripper fault stop is unverified; hardware stop remains unconfirmed")

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to a target joint configuration.

        Positions are expressed in degrees. The relative action magnitude may be
        clipped depending on `max_relative_target`, so the action actually sent is
        always returned.
        """
        self._assert_io_owner(motion=True)
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clip against soft joint limits.
        for motor_name in list(goal_pos):
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, goal_pos[motor_name]))
                if clipped != goal_pos[motor_name]:
                    logger.debug(f"Clipped {motor_name} from {goal_pos[motor_name]:.2f} to {clipped:.2f}")
                goal_pos[motor_name] = clipped

        # Tolerate 6-DOF leaders that have no wrist_yaw joint by holding it at zero.
        # This is intentional: it lets a 6-DOF leader such as the SO-100 / SO-101
        # (so100_leader / so101_leader) teleoperate this 7-DOF follower — the missing
        # wrist_yaw command is simply treated as 0.0 instead of raising.
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = self._present_pos()
            goal_present_pos = {key: (g, present_pos.get(key, g)) for key, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        use_mit = self.config.control_mode == "mit"
        for motor_name, position_deg in goal_pos.items():
            motor = self.motors.get(motor_name)
            if motor is None:
                continue
            idx = self.motor_names.index(motor_name)
            pos_rad = math.radians(position_deg)
            if motor_name == GRIPPER_MOTOR:
                if self.config.gripper_control_mode == "mit":
                    motor.send_mit(pos_rad, 0.0, self.config.gripper_mit_kp, self.config.gripper_mit_kd, 0.0)
                else:
                    vel_deg_s = (
                        self.config.pos_vel_velocity[idx]
                        if isinstance(self.config.pos_vel_velocity, list)
                        else self.config.pos_vel_velocity
                    )
                    motor.send_force_pos(pos_rad, math.radians(vel_deg_s), self.config.gripper_torque_ratio)
            elif use_mit:
                kp = self.config.mit_kp[idx] if isinstance(self.config.mit_kp, list) else self.config.mit_kp
                kd = self.config.mit_kd[idx] if isinstance(self.config.mit_kd, list) else self.config.mit_kd
                motor.send_mit(pos_rad, 0.0, kp, kd, 0.0)
            else:
                vel_deg_s = (
                    self.config.pos_vel_velocity[idx]
                    if isinstance(self.config.pos_vel_velocity, list)
                    else self.config.pos_vel_velocity
                )
                motor.send_pos_vel(pos_rad, math.radians(vel_deg_s))

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self._assert_io_owner()
        if self._observe_only:
            self.close_observe_only()
            return
        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()

        self.bus.close()
        self.bus = None
        self._release_bus_lease()
        self.motors = {}

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
