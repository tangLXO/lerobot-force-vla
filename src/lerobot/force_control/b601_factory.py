"""Disconnected B601 session construction shared by the bench and P3 recorder.

The calibrated ID-7 FORCE_POS path supports explicit supervised starts. Other
scopes retain the independent hardware acceptance gate even with complete profiles.
"""

from dataclasses import dataclass

from lerobot.robots.rebot_b601_follower.control_backend import B601ControlBackend, B601ControlConfig
from lerobot.robots.rebot_b601_follower.readonly_transport import B601ReadonlyJointSpec
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
        required_startup = "prepare_disabled_gripper" if gripper_only else "verify_pre_enabled"
        if self.startup_mode != required_startup:
            raise ValueError(f"b601.startup_mode must explicitly select {required_startup}")
        if gripper_only:
            if safety.stall_position_tolerance_deg is None or safety.stall_timeout_s is None:
                raise ValueError("Physical gripper control requires explicit stall tolerance and timeout")
            # Constructing and validating the disconnected backend performs no I/O.
            self.make_device().preflight_control()
        if self.control.fault_action is None:
            raise ValueError("B601 physical fault disposition is unresolved")
        if safety.fault_action != "stop":
            raise ValueError(
                "The B601 disable candidates require safety.fault_action=stop; measured hold is unimplemented"
            )
        if self.control.feedback_max_age_s * 1000 > safety.joint_feedback_max_age_ms:
            raise ValueError("B601 feedback age must not exceed the runtime safety envelope")
        if (runtime.feedback_budget_ms + self.control.send_budget_s * 1000) >= runtime.tick_deadline_ms:
            raise ValueError(
                "B601 read and send budgets must leave computation time inside the tick deadline"
            )
        gripper = next(p for p in self.control.profiles if p.name == "gripper")
        if not safety.q_min_deg <= gripper.minimum_deg < gripper.maximum_deg <= safety.q_max_deg:
            raise ValueError("B601 gripper limits must fit inside the runtime safety envelope")
        if gripper.max_relative_deg > safety.max_relative_target_deg:
            raise ValueError("B601 gripper relative limit exceeds the runtime safety envelope")

    def make_device(self) -> B601ControlBackend:
        """Construct only; the bus owner alone will connect and perform I/O."""
        if self.control is None or self.port is None:
            raise ValueError("Validate the explicit B601 hardware profile first")
        return B601ControlBackend(self.port, self.joint_specs, self.control, baudrate=self.baudrate)


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
