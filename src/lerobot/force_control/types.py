"""Immutable, hardware-independent inputs and I-P position candidates."""

# Unit symbols identify control diagnostics, not Sensor feature keys.
# ruff: noqa: N802, N815

from dataclasses import dataclass
from enum import StrEnum


class ControlPhase(StrEnum):
    """Operator motion and the explicitly started hybrid grasp lifecycle."""

    DISABLED = "DISABLED"
    ARMED = "ARMED"
    APPROACH = "APPROACH"
    FOLLOW = "FOLLOW"
    FORCE_TRACK = "FORCE_TRACK"
    FOLLOW_WAIT_CLEAR = "FOLLOW_WAIT_CLEAR"
    POSITION_HOLD = "POSITION_HOLD"
    RELEASE = "RELEASE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class ForceObservation:
    """One causal semantic sample, without a second calibration/filter."""

    left_N: float
    right_N: float
    timestamp_ns: int
    arrival_timestamp_ns: int
    sequence: int
    is_valid: bool = True
    status: str = "ok"
    unit: str = "N"
    calibration_id: str = "driver_semantic"
    filter_id: str = "none"

    @property
    def mean_N(self) -> float:
        """Return the signed mean without overflowing the sum."""
        return self.left_N / 2 + self.right_N / 2


@dataclass(frozen=True)
class ControlInputs:
    """Inputs actually visible at one host-monotonic decision anchor."""

    now_ns: int
    dt_control_s: float
    q_measured_deg: float
    q_prev_sent_deg: float
    force: ForceObservation
    joint_received_ns: int | None
    joint_sequence: int | None
    joint_valid: bool = True
    leader_gripper_deg: float | None = None
    leader_raw_gripper_deg: float | None = None
    leader_sequence: int | None = None
    leader_captured_ns: int | None = None


@dataclass(frozen=True)
class ContactReference:
    """Contact evidence in follower coordinates, committed only after submission."""

    timestamp_ns: int
    leader_gripper_deg: float
    leader_raw_gripper_deg: float
    leader_sequence: int
    leader_captured_ns: int
    follower_gripper_deg: float
    q_prev_sent_deg: float


@dataclass(frozen=True)
class TeacherProposal:
    """Uncommitted closing-coordinate increments and an absolute physical target."""

    revision: int
    phase: ControlPhase
    next_phase: ControlPhase
    inputs: ControlInputs
    command_enabled: bool
    target_N: float
    target_version: int
    error_N: float
    delta_i_deg: float
    delta_p_deg: float
    delta_raw_deg: float
    u_raw_deg_s: float
    q_candidate_deg: float
    tracking_tag: str = "inactive"
    transition: str | None = None
    failure_reason: str | None = None
    delta_contact_deg: float = 0.0
    contact_reference: ContactReference | None = None


@dataclass(frozen=True)
class SafetyDecision:
    """Final restriction and explicit integral-blocking diagnostics."""

    q_after_safety_deg: float | None
    u_limited_deg_s: float
    clip_flags: tuple[str, ...] = ()
    safety_override: bool = False
    fault_reason: str | None = None
    i_blocked: bool = False
    blocked_direction: int = 0

    @property
    def needs_fault_disposition(self) -> bool:
        """Withholding a command cannot stop a previously submitted target."""
        return self.fault_reason is not None
