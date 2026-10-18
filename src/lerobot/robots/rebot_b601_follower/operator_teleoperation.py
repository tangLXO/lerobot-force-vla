"""Explicit operator-selected whole-arm settings, independent of diagnostic micro trials."""

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .arm_calibration import B601ArmCalibration
from .control_protocol import encode_timeout_write
from .full_arm_control_types import (
    JOINT_NAMES,
    EvidenceFileIdentity,
    FullArmLeaderSample,
    require_sha256,
)
from .readonly_transport import DEFAULT_B601_JOINTS

if TYPE_CHECKING:
    from .control_backend import B601ControlConfig

# The source cache contract is identical; its name does not authorize diagnostic motion.
OperatorLeaderSample = FullArmLeaderSample


@dataclass(frozen=True)
class OperatorTeleoperationAdmission:
    """Pinned native calibration; no historical motion receipt is used as an admission gate."""

    calibration: B601ArmCalibration
    file_identities: tuple[EvidenceFileIdentity, ...]
    production_arm_authorized: bool = False

    def require_unchanged(self):
        """Keep the pinned calibration identity stable before another write."""
        for identity in self.file_identities:
            identity.require_unchanged()


@dataclass(frozen=True)
class OperatorTeleoperationConfig:
    """Native arm defaults with optional operator constraints, separate from diagnostics.

    Arm MIT gains and optional per-frame target limits live in the command profiles.
    None leaves the additional alignment, preload and command-gap thresholds disabled.
    There is no fixed force target, session duration or cumulative-travel restriction.
    The six arm joints use the selected fresh startup reference envelope,
    intersected with their profiles and native calibration. The FORCE_POS gripper
    uses its own working target limits and calibrated hard feedback limits.
    """

    session_id: str
    port: str
    calibration_file: str
    calibration_sha256: str
    leader_config_sha256: str
    reference_envelope_deg: float
    leader_max_age_s: float
    watchdog_ticks: int
    supported_arm: bool
    stop_policy: str
    alignment_tolerance_deg: float | None = None
    preload_tolerance_deg: float | None = None
    max_command_gap_s: float | None = None

    def validate(self):
        """Reject missing choices before constructing/opening any transport."""
        if str(uuid.UUID(self.session_id)) != self.session_id:
            raise ValueError("Operator teleoperation requires a canonical session ID")
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("Operator teleoperation requires an explicit follower port")
        if not isinstance(self.calibration_file, str) or not self.calibration_file.strip():
            raise ValueError("Operator teleoperation requires an explicit native calibration")
        require_sha256(self.calibration_sha256, "Operator calibration")
        require_sha256(self.leader_config_sha256, "Operator leader configuration")
        if self.supported_arm is not True or self.stop_policy != "disable_all_supported_arm":
            raise ValueError("Operator teleoperation requires explicit support and all-seven disable")
        for name in (
            "reference_envelope_deg",
            "leader_max_age_s",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Operator teleoperation requires an explicit positive {name}")
        for name in ("alignment_tolerance_deg", "preload_tolerance_deg", "max_command_gap_s"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Optional operator {name} must be positive when selected")
        encode_timeout_write(7, self.watchdog_ticks)

    def validate_binding(self, joint_specs, control: "B601ControlConfig", *, port: str):
        """Bind explicit profiles/calibration/stop budgets, without using micro-trial evidence."""
        self.validate()
        control.validate()
        if tuple(joint_specs) != DEFAULT_B601_JOINTS or self.port.casefold() != port.casefold():
            raise ValueError("Operator installation/port differs from the explicit configuration")
        if len(control.profiles) != 7 or {profile.name for profile in control.profiles} != set(JOINT_NAMES):
            raise ValueError("Operator teleoperation requires seven explicit command profiles")
        if control.fault_action != "disable_selected" or control.fault_joint_names != JOINT_NAMES:
            raise ValueError("Operator teleoperation requires the exact all-seven fault scope")
        if control.watchdog_ticks != self.watchdog_ticks:
            raise ValueError("Operator TIMEOUT configuration differs from the selected readback value")
        if control.fault_budget_s < 7 * control.send_budget_s + 0.006 + control.feedback_max_age_s:
            raise ValueError("Operator fault budget must cover seven bounded writes and fresh feedback")
        if (
            control.calibration_file is None
            or Path(control.calibration_file).resolve() != Path(self.calibration_file).resolve()
        ):
            raise ValueError("Operator native calibration source differs from its binding")
        calibration = B601ArmCalibration.load(
            self.calibration_file, {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
        )
        pin, _ = EvidenceFileIdentity.read(self.calibration_file, self.calibration_sha256)
        if calibration.sha256 != pin.sha256:
            raise ValueError("Operator native calibration changed while loading")
        for profile in control.profiles:
            calibration.require_bounds(profile.name, profile.minimum_deg, profile.maximum_deg)
            if profile.mode != ("force_pos" if profile.name == "gripper" else "mit"):
                raise ValueError("Operator teleoperation requires six MIT profiles and FORCE_POS gripper")
            if (
                self.preload_tolerance_deg is not None
                and profile.max_relative_deg is not None
                and self.preload_tolerance_deg > profile.max_relative_deg
            ):
                raise ValueError("Explicit preload tolerance must fit every per-frame target limit")
        pin.require_unchanged()
        return OperatorTeleoperationAdmission(calibration, (pin,))
