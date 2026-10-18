"""Typed, evidence-bound whole-arm validation capability; never production acceptance."""

# SI force unit remains explicit in the public validation target.
# ruff: noqa: N815

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .arm_acceptance import SingleJointAcceptanceConfig, command_profile_digest
from .arm_calibration import B601ArmCalibration
from .control_protocol import B601JointCommandProfile
from .full_arm_control_types import (
    ARM_NAMES,
    JOINT_NAMES,
    EvidenceFileIdentity,
    ValidationLeaderSample as ValidationLeaderSample,
    leader_configuration_digest as leader_configuration_digest,
    require_sha256 as _sha,
)
from .readonly_transport import DEFAULT_B601_JOINTS

if TYPE_CHECKING:
    from .control_backend import B601ControlConfig

REQUIRED_SOURCE_PATHS = (
    "examples/force_teacher/single_joint_arm_acceptance.py",
    *(
        f"src/lerobot/robots/rebot_b601_follower/{name}.py"
        for name in (
            "arm_acceptance",
            "control_backend",
            "control_protocol",
            "control_state",
            "readonly_transport",
            "control_io",
            "arm_calibration",
        )
    ),
    *(
        f"src/lerobot/sensors/{name}.py"
        for name in ("__init__", "sensor", "buffer", "synchronization", "configs", "utils")
    ),
    *(
        f"src/lerobot/sensors/x518/{name}.py"
        for name in (
            "__init__",
            "configuration_x518",
            "sensor_x518",
            "acquisition",
            "process_transport",
            "protocol",
            "diagnostics",
        )
    ),
    "src/lerobot/utils/keyboard_input.py",
    "src/lerobot/utils/import_utils.py",
)


@dataclass(frozen=True)
class MicroTrialBinding:
    """Explicit local prerequisite files for exactly one successful single-joint micro trial."""

    joint_name: str
    result_path: str
    result_sha256: str
    reviewed_config_path: str
    reviewed_config_sha256: str
    original_config_path: str | None = None


@dataclass(frozen=True)
class MicroTrialLimit:
    """Verified command-profile scope, separate from observed movement/production acceptance."""

    joint_name: str
    max_relative_deg: float
    velocity_deg_s: float
    profile_sha256: str


@dataclass(frozen=True)
class FullArmValidationAdmission:
    """Pinned admission result; this authorizes bounded validation only."""

    calibration: B601ArmCalibration
    micro_limits: tuple[MicroTrialLimit, ...]
    file_identities: tuple[EvidenceFileIdentity, ...]
    production_arm_authorized: bool = False

    def require_unchanged(self):
        """Check all admitted artifact identities before another physical write."""
        for identity in self.file_identities:
            identity.require_unchanged()


def _contained(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("Evidence closure requires contained relative file paths")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Evidence closure escapes its pinned root")
    return target


def _typed_micro_control(values):
    from .control_backend import B601ControlConfig

    arguments = dict(values)
    arguments["profiles"] = tuple(B601JointCommandProfile(**row) for row in arguments["profiles"])
    arguments["fault_joint_names"] = tuple(arguments.get("fault_joint_names", ()))
    return B601ControlConfig(**arguments)


def _verify_micro_movements(events, backend, request):
    stages = [row.get("stage") for row in events]
    required = ("startup", "outbound", "return", "disable")
    if any(stage not in stages for stage in required) or not (
        stages.index("startup") < stages.index("outbound") < stages.index("return") < stages.index("disable")
    ):
        raise ValueError("Micro event log lacks ordered startup/outbound/return/disable evidence")
    startup = events[stages.index("startup")]["receipt"]
    if (
        startup.get("kind") != "bounded_single_joint_acceptance"
        or startup.get("complete") is not True
        or startup.get("request_session_id") != request.session_id
    ):
        raise ValueError("Micro startup receipt differs from the bounded request")
    command_stages = {row.get("stage") for row in startup.get("commands", [])}
    if not {"disabled_target_preload", "enable_selected_only", "enabled_reference_target"} <= command_stages:
        raise ValueError("Micro startup does not contain its actual preload/enable/reference sequence")
    if any(row.get("status") != "submitted" for row in startup["commands"]):
        raise ValueError("Micro startup contains an unknown command submission")
    reference = backend.get("reference_deg")
    tolerance = backend.get("endpoint_position_tolerance_deg")
    if (
        type(reference) not in (int, float)
        or not math.isfinite(reference)
        or type(tolerance) not in (int, float)
        or not 0 < tolerance <= 0.05
    ):
        raise ValueError("Micro event log has no bounded endpoint/reference criterion")
    for phase, endpoint in (("outbound", reference + request.displacement_deg), ("return", reference)):
        selected = [event for event in events if event.get("stage") == phase]
        for event in selected:
            results = event.get("receipt", {}).get("results", [])
            if (
                len(results) != 1
                or results[0].get("name") != request.joint_name
                or results[0].get("status") != "success"
            ):
                raise ValueError("Micro movement lacks an actual selected-joint successful submission")
            item = results[0]
            if any(
                type(item.get(key)) not in (int, float) or not math.isfinite(item[key])
                for key in ("candidate_deg", "clipped_deg", "submitted_deg")
            ):
                raise ValueError("Micro movement contains nonfinite command evidence")
            if (
                not min(reference, reference + request.displacement_deg) - 1e-4
                <= item["submitted_deg"]
                <= max(reference, reference + request.displacement_deg) + 1e-4
            ):
                raise ValueError("Micro movement expands its bounded target path")
            observed = event.get("observation", {}).get("feedback", [])
            if len(observed) != 7 or {row.get("name") for row in observed} != set(JOINT_NAMES):
                raise ValueError("Micro movement lacks actual seven-joint feedback")
            if any(
                type(row.get("status_code")) is not int
                or row["status_code"] != (1 if row["name"] == request.joint_name else 0)
                for row in observed
            ):
                raise ValueError("Micro movement feedback actuator states differ from its scope")
            if any(
                type(row.get("position_deg")) not in (int, float) or not math.isfinite(row["position_deg"])
                for row in observed
            ):
                raise ValueError("Micro movement has no finite observed positions")
        last = selected[-1]
        position = next(
            row["position_deg"]
            for row in last["observation"]["feedback"]
            if row["name"] == request.joint_name
        )
        if (
            abs(last["receipt"]["results"][0]["submitted_deg"] - endpoint) > 1e-4
            or abs(position - endpoint) > tolerance
        ):
            raise ValueError(f"Micro movement did not observe its {phase} endpoint")
    if events[stages.index("disable")].get("receipt") != backend.get("cleanup"):
        raise ValueError("Micro disable event differs from its final cleanup receipt")


def _micro_trial(binding: MicroTrialBinding, calibration: B601ArmCalibration, port: str):
    if binding.joint_name not in ARM_NAMES:
        raise ValueError("Micro prerequisite must name one arm joint")
    _sha(binding.result_sha256, "Micro result")
    _sha(binding.reviewed_config_sha256, "Micro reviewed configuration")
    result_pin, raw = EvidenceFileIdentity.read(binding.result_path, binding.result_sha256)
    config_pin, config_raw = EvidenceFileIdentity.read(
        binding.reviewed_config_path, binding.reviewed_config_sha256
    )
    result, reviewed = json.loads(raw), json.loads(config_raw)
    if result.get("schema") != "b601_single_joint_acceptance_cli_v1":
        raise ValueError("A real single-joint CLI receipt is required as a micro prerequisite")
    for field in (
        "executed",
        "complete",
        "transport_closed",
        "sensor_closed",
        "sensor_lease_released",
        "manual_stop_ready",
        "escape_hook_closed",
        "evidence_closed",
    ):
        if result.get(field) is not True:
            raise ValueError(f"Micro prerequisite is incomplete/unclosed: {field}")
    if (
        result.get("production_arm_authorized") is not False
        or result.get("cleanup_pending") is not False
        or result.get("errors") != []
    ):
        raise ValueError("Micro prerequisite cannot claim production acceptance or retain errors/resources")
    if result.get("source_snapshot_status") != "captured":
        raise ValueError("Micro prerequisite has no captured source snapshot")
    backend = result.get("backend_result", {})
    if backend.get("schema") != "b601_single_joint_acceptance_v1" or any(
        backend.get(key) is not True for key in ("complete", "returned_command", "transport_closed")
    ):
        raise ValueError("Micro backend receipt did not complete the bounded out-and-return trial")
    if backend.get("production_arm_authorized") is not False or backend.get("error") is not None:
        raise ValueError("Micro backend receipt is not a successful validation-only result")
    request = SingleJointAcceptanceConfig(**backend["request"])
    request.validate()
    if SingleJointAcceptanceConfig(**reviewed.get("acceptance", {})) != request:
        raise ValueError("Micro reviewed acceptance differs from its actual backend request")
    if request.joint_name != binding.joint_name or request.calibration_sha256 != calibration.sha256:
        raise ValueError("Micro prerequisite joint/calibration differs from its binding")
    if Path(request.calibration_file).resolve() != Path(calibration.path):
        raise ValueError("Micro prerequisite calibration source differs from the current source")
    if result.get("calibration", {}).get("sha256") != calibration.sha256:
        raise ValueError("Micro CLI calibration differs from its backend request")
    if str(reviewed.get("port", "")).casefold() != port.casefold():
        raise ValueError("Micro prerequisite belongs to another follower port")
    micro_control = _typed_micro_control(reviewed.get("control", {}))
    request.validate_binding(DEFAULT_B601_JOINTS, micro_control)
    profiles = reviewed.get("control", {}).get("profiles", [])
    if len(profiles) != 1:
        raise ValueError("A micro prerequisite must contain exactly one command profile")
    profile = B601JointCommandProfile(**profiles[0])
    if (
        profile.name != binding.joint_name
        or profile.mode != "pos_vel"
        or command_profile_digest(profile) != request.profile_sha256
    ):
        raise ValueError("Micro prerequisite reviewed profile differs from its actual request")
    if profile.velocity_deg_s != request.velocity_deg_s:
        raise ValueError("Micro prerequisite speed differs from its request")
    calibration.require_bounds(profile.name, profile.minimum_deg, profile.maximum_deg)
    cleanup = backend.get("cleanup", {})
    if (
        cleanup.get("status") != "submitted"
        or cleanup.get("error") is not None
        or any(
            cleanup.get(key) is not True
            for key in ("all_joints_disabled_observed", "outstanding_query_accounted")
        )
    ):
        raise ValueError("Micro prerequisite has no complete all-seven disabled cleanup")
    receipts = cleanup.get("per_joint_results", [])
    if (
        len(receipts) != 1
        or receipts[0].get("name") != binding.joint_name
        or receipts[0].get("status") != "submitted"
    ):
        raise ValueError("Micro cleanup actuator scope differs from its selected joint")
    feedback = cleanup.get("post_disable_feedback", {}).get("feedback", [])
    if (
        len(feedback) != 7
        or {row.get("name") for row in feedback} != set(JOINT_NAMES)
        or any(row.get("status_code") != 0 or type(row.get("status_code")) is not int for row in feedback)
    ):
        raise ValueError("Micro cleanup lacks seven actual DISABLED feedback entries")
    root = Path(binding.result_path).resolve().parent
    event_pin, event_raw = EvidenceFileIdentity.read(root / "events.jsonl")
    events = [json.loads(line) for line in event_raw.splitlines()]
    if (
        type(result.get("evidence_rows")) is not int
        or len(events) != result["evidence_rows"]
        or not events
        or events[-1].get("stage") != "result"
    ):
        raise ValueError("Micro event log is incomplete or differs from its receipt")
    for field in (
        "schema",
        "request",
        "reference_deg",
        "returned_command",
        "complete",
        "cleanup",
        "transport_closed",
        "error",
    ):
        if events[-1].get(field) != backend.get(field):
            raise ValueError(f"Micro event log and final backend receipt differ: {field}")
    relative_manifest = result.get("source_snapshot_manifest")
    if relative_manifest != "source_snapshot/manifest.json":
        raise ValueError("Micro source snapshot path is not canonical")
    source_pin, source_raw = EvidenceFileIdentity.read(_contained(root, relative_manifest))
    source = json.loads(source_raw)
    if (
        source.get("schema") != "b601_acceptance_source_snapshot_v1"
        or source.get("complete") is not True
        or not source.get("files")
    ):
        raise ValueError("Micro source snapshot is incomplete")
    _verify_micro_movements(events, backend, request)
    _sha(result.get("config_sha256"), "Original micro configuration")
    pins = [result_pin, config_pin, event_pin, source_pin]
    if result["config_sha256"] != config_pin.sha256:
        if binding.original_config_path is None:
            raise ValueError("Original micro configuration is required when typed review bytes differ")
        original_pin, original_raw = EvidenceFileIdentity.read(
            binding.original_config_path, result["config_sha256"]
        )
        original = json.loads(original_raw)
        if (
            str(original.get("port", "")).casefold() != port.casefold()
            or SingleJointAcceptanceConfig(**original.get("acceptance", {})) != request
            or _typed_micro_control(original.get("control", {})) != micro_control
        ):
            raise ValueError("Original micro configuration differs from its typed review/request")
        pins.append(original_pin)
    seen = set()
    for entry in source["files"]:
        if entry["path"] in seen:
            raise ValueError("Micro source snapshot has duplicate file references")
        seen.add(entry["path"])
        pin, _ = EvidenceFileIdentity.read(
            _contained(root / "source_snapshot", entry["path"]), entry["sha256"]
        )
        if pin.size_bytes != entry.get("size_bytes"):
            raise ValueError("Micro source snapshot file length differs from its manifest")
        pins.append(pin)
    if not set(REQUIRED_SOURCE_PATHS) <= seen:
        raise ValueError("Micro source snapshot omits critical executed source files")
    return MicroTrialLimit(
        profile.name, profile.max_relative_deg, profile.velocity_deg_s, request.profile_sha256
    ), tuple(pins)


@dataclass(frozen=True)
class FullArmValidationConfig:
    """Explicit thirty-second validation envelope bound to six real local micro artifacts."""

    session_id: str
    authorization_id: str
    port: str
    calibration_file: str
    calibration_sha256: str
    profile_sha256: tuple[tuple[str, str], ...]
    micro_trials: tuple[MicroTrialBinding, ...]
    supported_arm: bool
    stop_policy: str
    reference_envelope_deg: float
    alignment_tolerance_deg: float
    max_duration_s: float
    watchdog_ticks: int
    target_force_N: float
    leader_config_sha256: str
    leader_max_age_s: float = 0.15
    max_command_gap_s: float = 0.05

    def validate(self):
        """Validate only the explicit bounded scope; no transport may open here."""
        if (
            str(uuid.UUID(self.session_id)) != self.session_id
            or not isinstance(self.authorization_id, str)
            or not self.authorization_id.strip()
        ):
            raise ValueError("Full-arm validation requires a canonical session and explicit authorization")
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("Full-arm validation requires an explicit follower port")
        _sha(self.calibration_sha256, "Full-arm calibration")
        _sha(self.leader_config_sha256, "Full-arm leader configuration")
        if len(self.profile_sha256) != 7 or set(dict(self.profile_sha256)) != set(JOINT_NAMES):
            raise ValueError("Full-arm validation requires exactly seven pinned command profiles")
        for _, digest in self.profile_sha256:
            _sha(digest, "Full-arm command profile")
        if len(self.micro_trials) != 6 or {binding.joint_name for binding in self.micro_trials} != set(
            ARM_NAMES
        ):
            raise ValueError("Full-arm validation requires six distinct real arm-joint micro bindings")
        if self.supported_arm is not True or self.stop_policy != "disable_all_supported_arm":
            raise ValueError("Full-arm validation requires confirmed support and safe all-seven disable")
        for name, upper in (
            ("reference_envelope_deg", 100),
            ("alignment_tolerance_deg", 5),
            ("max_duration_s", 30),
            ("leader_max_age_s", 0.15),
            ("max_command_gap_s", 0.05),
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f"Full-arm validation {name} exceeds its bounded scope")
        if type(self.watchdog_ticks) is not int or self.watchdog_ticks != 2000:
            raise ValueError("Full-arm validation requires explicit TIMEOUT=2000 readback")
        if type(self.target_force_N) not in (int, float) or self.target_force_N != 1:
            raise ValueError("Full-arm validation pins the explicit target force to 1 N")

    def validate_binding(self, joint_specs, control: "B601ControlConfig", *, port: str):
        """Verify local micro evidence/calibration/profile closure once before admission."""
        self.validate()
        control.validate()
        if tuple(joint_specs) != DEFAULT_B601_JOINTS or self.port.casefold() != port.casefold():
            raise ValueError("Full-arm validation installation/port differs from its admitted scope")
        if len(control.profiles) != 7 or {p.name for p in control.profiles} != set(JOINT_NAMES):
            raise ValueError("Full-arm validation requires exactly seven explicit command profiles")
        if control.fault_action != "disable_selected" or control.fault_joint_names != JOINT_NAMES:
            raise ValueError("Full-arm validation requires the exact all-seven fault scope")
        if control.watchdog_ticks != 2000 or control.feedback_max_age_s > 0.05:
            raise ValueError("Full-arm validation requires TIMEOUT=2000 and feedback age at most 50 ms")
        if control.fault_budget_s < 7 * control.send_budget_s + 0.006 + control.feedback_max_age_s:
            raise ValueError(
                "Full-arm fault budget must cover seven bounded writes, 1 ms pacing and fresh feedback"
            )
        if (
            control.calibration_file is None
            or Path(control.calibration_file).resolve() != Path(self.calibration_file).resolve()
        ):
            raise ValueError("Full-arm validation calibration source differs from its binding")
        calibration = B601ArmCalibration.load(
            self.calibration_file, {s.name: s.motor_id for s in DEFAULT_B601_JOINTS}
        )
        if calibration.sha256 != self.calibration_sha256:
            raise ValueError("Full-arm validation calibration digest changed")
        calibration_pin, _ = EvidenceFileIdentity.read(self.calibration_file, self.calibration_sha256)
        limits, pins = [], [calibration_pin]
        for binding in self.micro_trials:
            limit, identities = _micro_trial(binding, calibration, self.port)
            limits.append(limit)
            pins.extend(identities)
        by_name = {limit.joint_name: limit for limit in limits}
        expected = dict(self.profile_sha256)
        for profile in control.profiles:
            if command_profile_digest(profile) != expected[profile.name]:
                raise ValueError("Full-arm validation command profile digest changed")
            calibration.require_bounds(profile.name, profile.minimum_deg, profile.maximum_deg)
            if profile.name == "gripper":
                if profile.mode != "force_pos":
                    raise ValueError("Full-arm validation gripper must retain FORCE_POS")
            else:
                limit = by_name[profile.name]
                if (
                    profile.mode != "pos_vel"
                    or not 0 < profile.velocity_deg_s <= min(1, limit.velocity_deg_s)
                    or profile.max_relative_deg > limit.max_relative_deg
                ):
                    raise ValueError(f"Full-arm profile expands the verified micro scope: {profile.name}")
        for pin in pins:
            pin.require_unchanged()
        return FullArmValidationAdmission(calibration, tuple(limits), tuple(pins))
