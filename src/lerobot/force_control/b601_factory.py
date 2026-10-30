"""Disconnected B601 session construction shared by the bench and P3 recorder.

The calibrated ID-7 FORCE_POS path supports explicit supervised starts. Other
scopes retain the independent hardware acceptance gate even with complete profiles.
"""

from dataclasses import dataclass

from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlBackend, B601ControlConfig
from lerobot.robots.rebot_b601_follower.full_arm_validation import FullArmValidationConfig
from lerobot.robots.rebot_b601_follower.operator_teleoperation import OperatorTeleoperationConfig
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, B601ReadonlyJointSpec
from lerobot.sensors import SensorConfig
from lerobot.sensors.x518 import X518SensorConfig

from .config import SafetyConfig
from .runtime import RuntimeConfig


@dataclass(frozen=True)
class B601HardwareConfig:
    """Explicit hardware session; missing calibration and lifecycle choices stay unresolved."""

    port: str | None = None
    baudrate: int = 921600
    joint_specs: tuple[B601ReadonlyJointSpec, ...] = ()
    control: B601ControlConfig | None = None
    startup_mode: str | None = None
    validation: FullArmValidationConfig | None = None
    operator_teleoperation: OperatorTeleoperationConfig | None = None

    def validate(self, safety: SafetyConfig, runtime: RuntimeConfig, joint_names: tuple[str, ...]) -> None:
        """Validate scope and shared limits without opening or constructing a serial handle."""
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("b601.port must be explicitly configured")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("b601.baudrate must be a positive integer")
        if self.control is None:
            raise ValueError("b601.control requires explicit command profiles and execution budgets")
        self.control.validate()
        names = tuple(spec.name for spec in self.joint_specs)
        if len(set(names)) != len(names) or set(names) != set(joint_names):
            raise ValueError("B601 selected joints must exactly match the runtime scope")
        if {p.name for p in self.control.profiles} != set(names):
            raise ValueError("B601 command profiles must cover exactly the selected joints")
        for field in ("motor_id", "feedback_id"):
            if len({getattr(spec, field) for spec in self.joint_specs}) != len(names):
                raise ValueError(f"B601 {field} entries must be unique")
        for spec in self.joint_specs:
            if type(spec.motor_id) is not int or not 1 <= spec.motor_id <= 15:
                raise ValueError("B601 motor IDs must be in 1..15")
            if type(spec.feedback_id) is not int or not 1 <= spec.feedback_id <= 0x7FF:
                raise ValueError("B601 feedback IDs must be standard nonzero CAN IDs")
            if spec.model not in ("4310", "4340P"):
                raise ValueError("Only reviewed B601 motor model labels are supported")
        gripper_only = (
            names == ("gripper",)
            and self.joint_specs[0].motor_id == 7
            and self.joint_specs[0].feedback_id == 0x17
            and self.control.profiles[0].mode == "force_pos"
            and self.control.fault_action == "disable_gripper"
        )
        if self.validation is not None and self.operator_teleoperation is not None:
            raise ValueError("Choose either operator teleoperation or the separate diagnostic validation")
        if gripper_only and (self.validation is not None or self.operator_teleoperation is not None):
            raise ValueError("A full-arm validation capability requires all seven joints")
        required_startup = (
            "prepare_disabled_gripper"
            if gripper_only
            else (
                "prepare_disabled_operator_teleoperation"
                if self.operator_teleoperation is not None
                else (
                    "prepare_disabled_full_arm_validation"
                    if self.validation is not None
                    else "verify_pre_enabled"
                )
            )
        )
        if self.startup_mode != required_startup:
            raise ValueError(f"b601.startup_mode must explicitly select {required_startup}")
        if gripper_only:
            if safety.stall_position_tolerance_deg is None or safety.stall_timeout_s is None:
                raise ValueError("Physical gripper control requires explicit stall tolerance and timeout")
            # Constructing and validating the disconnected backend performs no I/O.
            self.make_device().preflight_control()
        else:
            self.validate_arm_profile()
            if self.validation is not None or self.operator_teleoperation is not None:
                # Evidence admission is read-only. The backend still requires fresh
                # disabled feedback and leader alignment before its explicit start.
                self.make_device().preflight_control()
        if self.control.fault_action is None:
            raise ValueError("B601 physical fault disposition is unresolved")
        if safety.fault_action != "stop":
            raise ValueError(
                "The B601 disable candidates require safety.fault_action=stop; measured hold is unimplemented"
            )
        if self.control.feedback_max_age_s * 1000 > safety.joint_feedback_max_age_ms:
            raise ValueError("B601 feedback age must not exceed the runtime safety envelope")
        if (
            self.control.command_feedback_mode != "cached"
            and (runtime.feedback_budget_ms + self.control.send_budget_s * 1000) >= runtime.tick_deadline_ms
        ):
            raise ValueError(
                "B601 read and send budgets must leave computation time inside the tick deadline"
            )
        gripper = next(p for p in self.control.profiles if p.name == "gripper")
        if not safety.q_min_deg <= gripper.minimum_deg < gripper.maximum_deg <= safety.q_max_deg:
            raise ValueError("B601 gripper limits must fit inside the runtime safety envelope")
        if gripper.max_relative_deg > safety.max_relative_target_deg:
            raise ValueError("B601 gripper relative limit exceeds the runtime safety envelope")

    def validate_arm_profile(self) -> B601ArmCalibration:
        """Inspect complete explicit arm configuration without lifting the hardware gate."""
        if self.control is None:
            raise ValueError("Full-arm preflight requires explicit command profiles")
        self.control.validate()
        expected = {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
        if {spec.name: spec.motor_id for spec in self.joint_specs} != expected:
            raise ValueError("Full-arm preflight requires the seven B601 joint names and motor IDs")
        required_startup = (
            "prepare_disabled_operator_teleoperation"
            if self.operator_teleoperation is not None
            else (
                "prepare_disabled_full_arm_validation"
                if self.validation is not None
                else "verify_pre_enabled"
            )
        )
        if self.startup_mode != required_startup:
            raise ValueError(f"Full-arm startup requires explicit {required_startup}")
        if self.control.fault_action != "disable_selected":
            raise ValueError("Full-arm fault policy requires explicit disable_selected covering all joints")
        if self.control.calibration_file is None:
            raise ValueError("Full-arm preflight requires an existing calibration_file")
        calibration = B601ArmCalibration.load(self.control.calibration_file, expected)
        profiles = {profile.name: profile for profile in self.control.profiles}
        if set(profiles) != set(expected):
            raise ValueError("Full-arm profiles must cover exactly seven selected joints")
        for name, profile in profiles.items():
            required_mode = (
                "force_pos"
                if name == "gripper"
                else "mit"
                if self.operator_teleoperation is not None
                else "pos_vel"
            )
            if profile.mode != required_mode:
                raise ValueError(f"Full-arm profile for {name} requires explicit {required_mode}")
            calibration.require_bounds(name, profile.minimum_deg, profile.maximum_deg)
        return calibration

    def preflight_report(self, safety: SafetyConfig, runtime: RuntimeConfig) -> dict:
        """Report file/configuration readiness only; never open a serial port or authorize motion."""
        report = {
            "configuration_valid": False,
            "motion_authorized": False,
            "hardware_ack": "unknown",
            "joint_names": list(runtime.joint_names),
            "hardware_acceptance_required": tuple(runtime.joint_names) != ("gripper",),
            "errors": [],
        }
        try:
            safety.validate()
            runtime.validate()
            self.validate(safety, runtime, runtime.joint_names)
            if report["hardware_acceptance_required"]:
                calibration = self.validate_arm_profile()
                report["calibration"] = {
                    "path": calibration.path,
                    "sha256": calibration.sha256,
                    "mtime_ns": calibration.mtime_ns,
                }
                report["acceptance_status"] = (
                    "operator_teleoperation_config_only"
                    if self.operator_teleoperation is not None
                    else (
                        "bounded_validation_only"
                        if self.validation is not None
                        else "hardware_acceptance_incomplete"
                    )
                )
                if self.operator_teleoperation is not None:
                    report["operator_teleoperation"] = {
                        "session_id": self.operator_teleoperation.session_id,
                        "reference_envelope_deg": self.operator_teleoperation.reference_envelope_deg,
                        "alignment_tolerance_deg": self.operator_teleoperation.alignment_tolerance_deg,
                        "preload_tolerance_deg": self.operator_teleoperation.preload_tolerance_deg,
                        "micro_history_required": False,
                        "fixed_duration_limit_s": None,
                        "fixed_force_target_N": None,
                        "hardware_acceptance_confirmed": False,
                    }
                    report["micro_history_required"] = False
                    report["physical_validation_confirmed"] = False
                    report["next_operator_step"] = "user_operated_unloaded_joint_teleoperation"
                if self.validation is not None:
                    report["bounded_validation"] = {
                        "session_id": self.validation.session_id,
                        "maximum_duration_s": self.validation.max_duration_s,
                        "alignment_tolerance_deg": self.validation.alignment_tolerance_deg,
                        "target_force_N": self.validation.target_force_N,
                        "production_arm_authorized": False,
                    }
                if self.operator_teleoperation is None:
                    report["required_physical_checks"] = [
                        "per_joint_direction_limits_and_startup",
                        "dynamic_command_rx_ordering",
                        "joint_feedback_freshness_under_load",
                        "combined_leader_teacher_control",
                        "selected_fault_disposition_and_cleanup",
                    ]
            else:
                report["acceptance_status"] = "gripper_config_only"
            report["configuration_valid"] = True
        except (ValueError, RuntimeError) as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
        return report

    def make_device(self) -> B601ControlBackend:
        """Construct only; the bus owner alone will connect and perform I/O."""
        if self.control is None or self.port is None:
            raise ValueError("Validate the explicit B601 hardware profile first")
        if self.operator_teleoperation is not None:
            return B601ControlBackend(
                self.port,
                self.joint_specs,
                self.control,
                baudrate=self.baudrate,
                operator_teleoperation=self.operator_teleoperation,
            )
        if self.validation is None:
            return B601ControlBackend(self.port, self.joint_specs, self.control, baudrate=self.baudrate)
        return B601ControlBackend(
            self.port, self.joint_specs, self.control, baudrate=self.baudrate, validation=self.validation
        )


def validate_b601_sensor(sensors: dict[str, SensorConfig], runtime: RuntimeConfig) -> None:
    """Forbid silently pairing a physical follower with simulated or absent force data."""
    if set(sensors) != {runtime.force_instance}:
        raise ValueError("B601 requires one explicitly configured X518 force stream")
    config = sensors[runtime.force_instance]
    if not isinstance(config, X518SensorConfig) or not config.required:
        raise ValueError("B601 requires a required X518 sensor configuration")
    if set(config.channels) != {runtime.left_feature, runtime.right_feature}:
        raise ValueError("B601 force channels must match the runtime's two semantic feature names")
    if config.frame_features not in (None, [runtime.left_feature, runtime.right_feature]):
        raise ValueError("B601 force stream must expose the ordered two-force current view")
