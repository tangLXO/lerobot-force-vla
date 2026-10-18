"""One-shot, supported single-arm-joint acceptance; never production authorization.

The request pins a calibration and exactly one POS_VEL profile. Every actual
write is still guarded by B601ControlBackend. This module opens no hardware at
import/construction time and supplies no MIT gains or implicit drive settings.
"""

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .arm_calibration import B601ArmCalibration
from .readonly_transport import DEFAULT_B601_JOINTS

if TYPE_CHECKING:
    from .control_backend import B601ControlBackend, B601ControlConfig
    from .control_protocol import B601JointCommandProfile


def command_profile_digest(profile: "B601JointCommandProfile") -> str:
    """Bind the visible limits, mode and velocity without changing their values."""
    profile.validate()
    values = asdict(profile)
    for name in (
        "minimum_deg",
        "maximum_deg",
        "max_relative_deg",
        "velocity_deg_s",
        "torque_ratio",
        "kp",
        "kd",
    ):
        if values[name] is not None:
            numeric = float(values[name])
            values[name] = 0.0 if numeric == 0 else numeric
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SingleJointAcceptanceConfig:
    """Explicit authorization for one small out-and-return supported-arm test."""

    session_id: str
    authorization_id: str
    joint_name: str
    calibration_file: str
    calibration_sha256: str
    profile_sha256: str
    supported_arm: bool
    stop_policy: str
    relative_bound_deg: float
    displacement_deg: float
    velocity_deg_s: float
    max_total_travel_deg: float
    max_duration_s: float
    watchdog_ticks: int
    allow_disabled_watchdog_prepare: bool

    def validate(self) -> None:
        """Reject anything beyond the explicitly bounded acceptance interface."""
        if str(uuid.UUID(self.session_id)) != self.session_id:
            raise ValueError("Acceptance session_id must be a canonical UUID")
        if not isinstance(self.authorization_id, str) or not self.authorization_id.strip():
            raise ValueError("Acceptance requires an explicit authorization record")
        if self.joint_name not in {spec.name for spec in DEFAULT_B601_JOINTS[:-1]}:
            raise ValueError("Acceptance selects exactly one of the six arm joints")
        if self.supported_arm is not True or self.stop_policy != "disable_selected_supported_arm":
            raise ValueError("Acceptance requires confirmed arm support and explicit selected-joint disable")
        for name in ("calibration_sha256", "profile_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Acceptance {name} must pin a SHA256 digest")
        bounds = {
            "relative_bound_deg": (0, 100),
            "velocity_deg_s": (0, 1),
            "max_total_travel_deg": (0, 2),
            "max_duration_s": (0, 5),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"Acceptance {name} exceeds its bounded scope")
        if type(self.displacement_deg) not in (int, float) or not math.isfinite(self.displacement_deg):
            raise ValueError("Acceptance displacement must be finite")
        if not 0 < abs(self.displacement_deg) <= min(0.5, self.relative_bound_deg):
            raise ValueError("Acceptance displacement must be nonzero and at most 0.5 degrees")
        if self.max_total_travel_deg < 2 * abs(self.displacement_deg):
            raise ValueError("Acceptance total-travel budget must include the return")
        if type(self.watchdog_ticks) is not int or not 1 <= self.watchdog_ticks <= 2000:
            raise ValueError("Acceptance watchdog requires an explicit nonzero uint32 value at most 2000")
        if not isinstance(self.allow_disabled_watchdog_prepare, bool):
            raise ValueError("Watchdog preparation authorization must be explicit boolean")

    def validate_binding(self, joint_specs, control: "B601ControlConfig") -> B601ArmCalibration:
        """Require all seven observed IDs but precisely one authorized command profile."""
        self.validate()
        if tuple(joint_specs) != DEFAULT_B601_JOINTS:
            raise ValueError("Acceptance must observe the exact seven-joint B601 installation")
        if len(control.profiles) != 1 or control.profiles[0].name != self.joint_name:
            raise ValueError("Acceptance permits precisely one selected-joint command profile")
        profile = control.profiles[0]
        profile.validate()
        if profile.mode != "pos_vel" or profile.velocity_deg_s != self.velocity_deg_s:
            raise ValueError("Acceptance requires the pinned explicit POS_VEL speed")
        if profile.max_relative_deg > 0.5 or command_profile_digest(profile) != self.profile_sha256:
            raise ValueError("Acceptance command profile differs from the authorized profile")
        if control.fault_action != "disable_selected" or control.fault_joint_names != (self.joint_name,):
            raise ValueError("Acceptance fault disposition must disable only the selected joint")
        if control.watchdog_ticks != self.watchdog_ticks:
            raise ValueError("Acceptance watchdog value differs from the explicit command configuration")
        if (
            control.calibration_file is None
            or Path(control.calibration_file).resolve() != Path(self.calibration_file).resolve()
        ):
            raise ValueError("Acceptance calibration source differs from the pinned source")
        calibration = B601ArmCalibration.load(
            self.calibration_file, {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
        )
        if calibration.sha256 != self.calibration_sha256:
            raise ValueError("Acceptance calibration digest changed; acquire a fresh reference")
        calibration.require_bounds(self.joint_name, profile.minimum_deg, profile.maximum_deg)
        return calibration


def execute_single_joint_acceptance(
    device: "B601ControlBackend",
    *,
    stop_event,
    event_sink,
    input_guard,
    control_hz: float = 20.0,
    manual_stop_established: bool = False,
) -> dict:
    """Execute one authorized microstep and return on the existing single I/O owner.

    Call only after reviewing the visible request. All steps, raw I/O audits,
    cleanup receipts and failures are passed to event_sink. Return values retain
    hardware_ack=unknown; this does not unlock ordinary whole-arm operation.
    """
    request = device.acceptance_request
    if request is None:
        raise ValueError("A dedicated single-joint acceptance backend is required")
    request.validate()
    if manual_stop_established is not True:
        raise ValueError("Establish the live Escape/Ctrl-C/manual stop channel before acceptance execution")
    if type(control_hz) not in (int, float) or not math.isfinite(control_hz) or not 1 <= control_hz <= 50:
        raise ValueError("Acceptance control_hz must be explicitly within 1..50")
    if not callable(event_sink) or not callable(input_guard) or not hasattr(stop_event, "is_set"):
        raise ValueError("Acceptance requires an evidence sink, live input guard and priority stop event")
    events = []
    error = None
    reference = None
    returned = False

    def emit(event):
        events.append(event)
        event_sink(event)

    def guard():
        if stop_event.is_set():
            raise RuntimeError("Acceptance operator stop requested")
        input_guard()

    try:
        guard()
        device.connect()
        emit({"stage": "connected_disabled", "state": asdict(device.acceptance_observation())})
        reference = device.start_acceptance(guard)
        emit({"stage": "startup", "receipt": device.startup_receipt, "io": device.take_io_evidence()})
        previous = reference
        deadline = time.perf_counter()
        for phase, endpoint in (("outbound", reference + request.displacement_deg), ("return", reference)):
            while True:
                guard()
                magnitude = min(abs(endpoint - previous), request.velocity_deg_s / control_hz)
                target = previous + math.copysign(magnitude, endpoint - previous)
                receipt = device.acceptance_step(target, guard)
                previous = receipt.get(request.joint_name).submitted_deg
                emit(
                    {
                        "stage": phase,
                        "receipt": asdict(receipt),
                        "observation": asdict(device.acceptance_observation()),
                        "io": device.take_io_evidence(),
                    }
                )
                observed = next(
                    item.position_deg
                    for item in device.acceptance_observation().feedback
                    if item.name == request.joint_name
                )
                if abs(endpoint - previous) <= 1e-4 and abs(endpoint - observed) <= 0.05:
                    break
                deadline += 1 / control_hz
                now = time.perf_counter()
                if now >= deadline:
                    # Skip overdue slots instead of replaying target increments.
                    deadline += (math.floor((now - deadline) * control_hz) + 1) / control_hz
                while time.perf_counter() < deadline:
                    guard()
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(0.005, remaining))
        returned = True
    except BaseException as exc:
        error = exc
    finally:
        cleanup = None
        try:
            if device.has_resources:
                cleanup = device.emergency_stop("acceptance_complete" if error is None else str(error))
                emit({"stage": "disable", "receipt": cleanup, "io": device.take_io_evidence()})
        except BaseException as exc:
            error = (
                exc if error is None else BaseExceptionGroup("Acceptance and disable failed", [error, exc])
            )
        try:
            device.disconnect()
        except BaseException as exc:
            error = exc if error is None else BaseExceptionGroup("Acceptance and close failed", [error, exc])
        result = {
            "schema": "b601_single_joint_acceptance_v1",
            "request": asdict(request),
            "reference_deg": reference,
            "returned_command": returned,
            "hardware_ack": "unknown",
            "watchdog_semantics_verified": False,
            "production_arm_authorized": False,
            "endpoint_position_tolerance_deg": 0.05,
            "complete": error is None
            and cleanup is not None
            and cleanup.get("all_joints_disabled_observed") is True,
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "cleanup": cleanup,
            "transport_closed": not device.has_resources,
            "events": events,
        }
        event_sink({"stage": "result", **{key: value for key, value in result.items() if key != "events"}})
    if error is not None:
        raise error
    return result


run_single_joint_acceptance = execute_single_joint_acceptance
