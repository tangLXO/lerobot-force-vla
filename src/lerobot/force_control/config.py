"""Explicit I-P gains and protection limits in degrees, newtons and seconds."""

# SI symbols are part of the public configuration contract.
# ruff: noqa: N803, N815

import math
from dataclasses import dataclass, fields


def finite_number(name: str, value: object, *, minimum: float | None = None) -> float:
    """Reject wrong types and nonfinite values before any resource is opened."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in the documented units")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def positive(name: str, value: object) -> float:
    """Validate a strictly positive finite scalar."""
    result = finite_number(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def validate_close_sign(close_sign: object) -> None:
    """Require an explicitly measured closing direction."""
    if isinstance(close_sign, bool) or not isinstance(close_sign, int) or close_sign not in (-1, 1):
        raise ValueError("close_sign must be an explicitly calibrated integer +1 or -1")


@dataclass(frozen=True)
class TeacherConfig:
    """One incremental I-P law; missing hardware gains never acquire defaults."""

    controller_type: str = "ip"
    force_target_N: float | None = None
    target_min_N: float | None = None
    target_max_N: float | None = None
    kp_deg_per_N: float | None = None
    ki_deg_per_N_s: float | None = None
    integral_deadband_N: float = 0.0
    approach_speed_deg_s: float | None = None
    approach_force_threshold_N: float | None = None
    release_speed_deg_s: float | None = None
    release_travel_deg: float | None = None
    max_release_time_s: float | None = None
    controller_id: str = "force_teacher_ip"
    profile_id: str = "unconfigured"

    def validate(self) -> None:
        """Reject legacy gains rather than silently changing their dimensions."""
        if self.controller_type != "ip":
            raise ValueError("controller_type must be ip; legacy P/PI/PID gains cannot be reused")
        for name in ("force_target_N", "target_min_N", "target_max_N"):
            finite_number(name, getattr(self, name), minimum=0)
        for name in ("kp_deg_per_N", "ki_deg_per_N_s", "integral_deadband_N"):
            finite_number(name, getattr(self, name), minimum=0)
        if self.target_min_N > self.target_max_N:
            raise ValueError("target range must be ordered and nonnegative")
        self.validate_target(self.force_target_N)
        if self.approach_speed_deg_s is not None or self.approach_force_threshold_N is not None:
            positive("approach_speed_deg_s", self.approach_speed_deg_s)
            positive("approach_force_threshold_N", self.approach_force_threshold_N)
        for name in ("release_speed_deg_s", "release_travel_deg", "max_release_time_s"):
            positive(name, getattr(self, name))
        for name in ("controller_id", "profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")

    def validate_target(self, target_N: float) -> None:
        """Reject an out-of-domain target instead of clipping it."""
        target = finite_number("force_target_N", target_N)
        if not self.target_min_N <= target <= self.target_max_N:
            raise ValueError("requested target is outside the configured target range; it is not clipped")

    @classmethod
    def fake(cls, **overrides) -> "TeacherConfig":
        """Synthetic test gains, never a B601 calibration."""
        values = {
            "force_target_N": 1.0,
            "target_min_N": 0.0,
            "target_max_N": 3.0,
            "kp_deg_per_N": 0.1,
            "ki_deg_per_N_s": 4.0,
            "release_speed_deg_s": 2.0,
            "release_travel_deg": 2.0,
            "max_release_time_s": 3.0,
            "profile_id": "fake_example_only",
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class SafetyConfig:
    """Hard protection plus optional trajectory limits; None explicitly disables an option."""

    q_min_deg: float | None = None
    q_max_deg: float | None = None
    max_relative_target_deg: float | None = None
    max_command_tracking_error_deg: float | None = None
    max_close_speed_deg_s: float | None = None
    max_open_speed_deg_s: float | None = None
    acceleration_limit_deg_s2: float | None = None
    hard_force_limit_left_N: float | None = None
    hard_force_limit_right_N: float | None = None
    force_max_age_ms: float | None = None
    joint_feedback_max_age_ms: float | None = None
    decision_max_age_ms: float | None = None
    max_dt_s: float | None = None
    fault_action: str | None = None
    supported_object_bench: bool = False
    soft_q_min_deg: float | None = None
    soft_q_max_deg: float | None = None
    max_total_travel_deg: float | None = None
    max_run_time_s: float | None = None
    stall_position_tolerance_deg: float | None = None
    stall_timeout_s: float | None = None

    def validate(self) -> None:
        """Validate each displayed limit without implicit tuning-mode bypasses."""
        for name in ("q_min_deg", "q_max_deg"):
            finite_number(name, getattr(self, name))
        if self.q_min_deg >= self.q_max_deg:
            raise ValueError("joint hard limits require q_min_deg < q_max_deg")
        optional = {
            "acceleration_limit_deg_s2",
            "max_total_travel_deg",
            "max_run_time_s",
            "stall_position_tolerance_deg",
            "stall_timeout_s",
        }
        excluded = {
            "q_min_deg",
            "q_max_deg",
            "soft_q_min_deg",
            "soft_q_max_deg",
            "fault_action",
            "supported_object_bench",
        }
        for item in fields(self):
            if item.name in excluded:
                continue
            value = getattr(self, item.name)
            if value is not None or item.name not in optional:
                positive(item.name, value)
        if (self.soft_q_min_deg is None) != (self.soft_q_max_deg is None):
            raise ValueError("soft_q_min_deg and soft_q_max_deg must both be set or both be None")
        if self.soft_q_min_deg is not None:
            finite_number("soft_q_min_deg", self.soft_q_min_deg)
            finite_number("soft_q_max_deg", self.soft_q_max_deg)
            if not self.q_min_deg <= self.soft_q_min_deg < self.soft_q_max_deg <= self.q_max_deg:
                raise ValueError("soft limits must be ordered inside the hard calibrated limits")
        if (self.stall_position_tolerance_deg is None) != (self.stall_timeout_s is None):
            raise ValueError("stall tolerance and timeout must both be set or both be None")
        if type(self.supported_object_bench) is not bool:
            raise ValueError("supported_object_bench must be boolean")
        if self.fault_action not in ("hold_measured", "stop"):
            raise ValueError("fault_action must explicitly select hold_measured or stop")

    @classmethod
    def fake(cls, **overrides) -> "SafetyConfig":
        """Synthetic model limits, separate from any physical acceptance."""
        values = {
            "q_min_deg": -10.0,
            "q_max_deg": 10.0,
            "max_relative_target_deg": 1.0,
            "max_command_tracking_error_deg": 2.0,
            "max_close_speed_deg_s": 5.0,
            "max_open_speed_deg_s": 5.0,
            "acceleration_limit_deg_s2": 50.0,
            "hard_force_limit_left_N": 5.0,
            "hard_force_limit_right_N": 5.0,
            "force_max_age_ms": 100.0,
            "joint_feedback_max_age_ms": 100.0,
            "decision_max_age_ms": 50.0,
            "max_dt_s": 0.1,
            "fault_action": "hold_measured",
            "supported_object_bench": True,
            "stall_position_tolerance_deg": 0.01,
            "stall_timeout_s": 0.5,
        }
        values.update(overrides)
        return cls(**values)


def validate_profile(teacher: TeacherConfig, safety: SafetyConfig, close_sign: int) -> None:
    """Validate the unique controller and its shared force/action envelope."""
    teacher.validate()
    safety.validate()
    validate_close_sign(close_sign)
    if teacher.target_max_N >= min(safety.hard_force_limit_left_N, safety.hard_force_limit_right_N):
        raise ValueError("target domain must be strictly below both hard per-side force limits")
    if teacher.release_speed_deg_s > safety.max_open_speed_deg_s:
        raise ValueError("release speed must not exceed the configured opening speed limit")
    if teacher.approach_speed_deg_s is not None:
        if teacher.approach_speed_deg_s > safety.max_close_speed_deg_s:
            raise ValueError("approach speed must not exceed the configured closing speed limit")
        if teacher.approach_force_threshold_N >= min(
            safety.hard_force_limit_left_N, safety.hard_force_limit_right_N
        ):
            raise ValueError("approach force threshold must be strictly below both hard per-side limits")


def validate_motion_request(
    teacher: TeacherConfig, safety: SafetyConfig, mode: str, position_deg=None, position_speed_deg_s=None
):
    """Validate operator motion intent before enabling hardware or changing phase."""
    if mode not in ("force", "approach", "position"):
        raise ValueError("motion mode must be force, approach or position")
    if mode == "approach" and teacher.approach_speed_deg_s is None:
        raise ValueError("请填写恒速闭合速度和切换平均力阈值")
    if mode == "position":
        goal = finite_number("指定绝对位置 (°)", position_deg)
        if position_speed_deg_s is not None:
            positive("定位速度 (°/s)", position_speed_deg_s)
        lower = safety.soft_q_min_deg if safety.soft_q_min_deg is not None else safety.q_min_deg
        upper = safety.soft_q_max_deg if safety.soft_q_max_deg is not None else safety.q_max_deg
        if not lower <= goal <= upper:
            raise ValueError(f"指定位置必须在校准 / 软件范围 [{lower:g}, {upper:g}]° 内")
    elif position_deg is not None or position_speed_deg_s is not None:
        raise ValueError("position and speed require position mode")
