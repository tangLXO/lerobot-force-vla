"""One bus owner with bounded mailboxes, causal input selection and command evidence.

The owner submits configured arm joints and the supported gripper. Cameras, leader reads, Dataset
transactions and trace serialization are deliberately outside this execution context.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import asdict, dataclass, replace

from lerobot.robots.rebot_b601_follower.full_arm_control_types import FullArmLeaderSample

from .capture_annotations import ANNOTATION_SCHEMA, annotate_capture
from .config import (
    canonicalize_teacher_fields,
    finite_number,
    positive,
    validate_motion_request,
    validate_profile,
)
from .types import ControlInputs, ControlPhase, ForceObservation

DEFAULT_FEEDBACK_BUDGET_MS = 15.0
DEFAULT_TICK_DEADLINE_MS = 40.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Scheduling and semantic identity, separate from the calibrated profile."""

    control_hz: float = 50.0
    feedback_budget_ms: float = DEFAULT_FEEDBACK_BUDGET_MS
    tick_deadline_ms: float = DEFAULT_TICK_DEADLINE_MS
    command_ttl_ms: float = 200.0
    force_instance: str = "gripper_force"
    left_feature: str = "left.normal_force"
    right_feature: str = "right.normal_force"
    force_metric: str = "mean_normal"
    software_calibration_id: str = "driver_semantic_N"
    object_id: str = "unspecified"
    contact_condition_id: str = "unspecified"
    joint_names: tuple[str, ...] = ("gripper",)
    arm_intent_required: bool = False
    handover_tolerance_deg: float | None = None
    scheduler_sleep_slice_ms: float = 5.0

    def validate(self):
        """Reject malformed budgets and unsupported force semantics before connection."""
        for name in (
            "control_hz",
            "scheduler_sleep_slice_ms",
            "feedback_budget_ms",
            "tick_deadline_ms",
            "command_ttl_ms",
        ):
            positive(name, getattr(self, name))
        if self.feedback_budget_ms >= self.tick_deadline_ms:
            raise ValueError("feedback budget must be less than the tick deadline")
        if self.force_metric != "mean_normal":
            raise ValueError("Only mean_normal force is supported")
        for name in (
            "force_instance",
            "left_feature",
            "right_feature",
            "software_calibration_id",
            "object_id",
            "contact_condition_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.left_feature == self.right_feature:
            raise ValueError("Two distinct force channels are required")
        if (
            not self.joint_names
            or "gripper" not in self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or any(not isinstance(name, str) or not name or "." in name for name in self.joint_names)
        ):
            raise ValueError("joint_names must contain unique bare names including gripper")
        if not isinstance(self.arm_intent_required, bool):
            raise ValueError("arm_intent_required must be boolean")
        if self.handover_tolerance_deg is not None:
            positive("handover_tolerance_deg", self.handover_tolerance_deg)


@dataclass(frozen=True)
class OwnerCommand:
    """Latest intent with a lease in time and in the current ownership epoch."""

    kind: str
    version: int
    epoch: int
    created_ns: int
    ttl_ns: int
    value: object = None

    def __post_init__(self):
        """Reject malformed TTL/version evidence at the mailbox boundary."""
        for name in ("version", "epoch", "created_ns", "ttl_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "ttl_ns" else 0):
                raise ValueError(f"{name} must be an integer with a valid monotonic bound")


class LatestMailbox:
    """Store at most one intent; fault invalidation cannot revive older commands."""

    def __init__(self):
        """Allocate two fixed slots: mode transitions and ordinary latest intent."""
        self._lock = threading.Lock()
        self._value = None
        self._mode = None
        self._version = -1
        self._epoch = 0

    def put(self, command: OwnerCommand) -> bool:
        """Reject prior epochs and reordered submissions without an action backlog."""
        with self._lock:
            if command.epoch != self._epoch or command.version <= self._version:
                return False
            if command.kind in (
                "arm",
                "start",
                "release",
                "reset",
                "disable",
                "end_episode",
                "discard",
                "teleop",
                "takeover",
                "freeze",
                "resume",
                "step",
                "approach",
                "position",
            ):
                self._mode = command
                self._value = None
            else:
                self._value = command
            self._version = command.version
            return True

    def take(self):
        """Consume at most one latest command without waiting."""
        with self._lock:
            if self._mode is not None:
                value, self._mode = self._mode, None
            else:
                value, self._value = self._value, None
            return value

    def invalidate(self, epoch: int):
        """Clear queued intent when safety changes ownership."""
        with self._lock:
            self._epoch = epoch
            self._value = None
            self._mode = None
            self._version = -1


@dataclass(frozen=True)
class ArmIntent:
    """Immutable leader intent using capture time, never mailbox arrival as freshness."""

    goals: tuple[tuple[str, float], ...]
    version: int
    epoch: int
    captured_ns: int
    ttl_ns: int
    metadata_json: str = "{}"

    def __post_init__(self):
        """Reject malformed evidence before it can enter the owner context."""
        OwnerCommand("arm_intent", self.version, self.epoch, self.captured_ns, self.ttl_ns)
        if not self.goals or len(dict(self.goals)) != len(self.goals):
            raise ValueError("Arm intent must contain unique position keys")
        for name, value in self.goals:
            if not isinstance(name, str) or not name.endswith(".pos"):
                raise ValueError("Arm intent keys must use joint.pos in degrees")
            finite_number(name, value)


class ArmMailbox:
    """One retained latest intent, expired or invalidated without a replay queue."""

    def __init__(self):
        """Allocate a fixed single slot and ownership epoch."""
        self._lock = threading.Lock()
        self._value = None
        self._epoch = 0
        self._version = -1

    def put(self, intent: ArmIntent) -> bool:
        """Accept monotonically newer versions in the current epoch only."""
        with self._lock:
            if intent.epoch != self._epoch or intent.version <= self._version:
                return False
            self._value, self._version = intent, intent.version
            return True

    def latest(self):
        """Read immutable intent without consuming another reader's state."""
        with self._lock:
            return self._value

    def invalidate(self, epoch):
        """Forget all previous hand positions on fault or handover."""
        with self._lock:
            self._epoch, self._version, self._value = epoch, -1, None


class ForceRuntime:
    """Run any compatible proposal source behind the same owner and safety boundary."""

    def __init__(
        self,
        device,
        sensor,
        controller,
        safety,
        trace,
        config=None,
        *,
        clock=time.perf_counter_ns,
        stop_event=None,
    ):
        """Validate disconnected interfaces and retain no hardware access in the controller."""
        self.device, self.sensor, self.controller, self.safety, self.trace = (
            device,
            sensor,
            controller,
            safety,
            trace,
        )
        self.config = config or RuntimeConfig()
        self.config.validate()
        if hasattr(trace, "metadata"):
            trace.metadata["capture_annotation_schema"] = ANNOTATION_SCHEMA
        validate_profile(controller.config, safety.config, controller.close_sign)
        for key in (self.config.left_feature, self.config.right_feature):
            feature = sensor.features.get(key)
            if feature is None or feature.unit != "N" or feature.shape != ():
                raise ValueError(f"{key} must be a calibrated scalar semantic force in N")
        self.clock = clock
        self.owner = None
        self.mailbox = LatestMailbox()
        self.arm_mailbox = ArmMailbox()
        self._metadata_intent = None
        self._metadata_json = None
        self._metadata_value = None
        self._leader_input_cache = None
        self._leader_sample_cache = None
        self._stop = threading.Event() if stop_event is None else stop_event
        self.epoch = 0
        self.segment_id = 0
        self.tick_id = 0
        self.last_sent = None
        self.last_success_tick = None
        self.last_send_start = None
        self.last_tick_start = None
        self._last_feedback_sequence = None
        self._last_feedback_receive_ns = None
        self._joint_evidence = {}
        self.latest_joint_snapshot = None
        self.last_submitted = {}
        self.mode = "TEACHER"
        self._handover_after_ns = None
        self._fault_disposition = None
        self.latest_row = None
        self.event = None
        self.command_version = None
        self._started = False
        self.shutdown_evidence = None
        self._parameter_update = None
        self._startup_performed = False
        self._hybrid_leader_required = False
        self._validation_parameter_pin = (
            (asdict(controller.config), asdict(safety.config))
            if getattr(device, "full_arm_validation_config", None) is not None
            else None
        )
        self._check_validation_target()

    def _check_validation_target(self, *, teacher=None, safety=None, target=None):
        """Keep short validation bound to its reviewed force, gains and protections."""
        validation = getattr(self.device, "full_arm_validation_config", None)
        if validation is not None:
            requested = (teacher or self.controller.config).force_target_N if target is None else target
            if requested != validation.target_force_N:
                raise RuntimeError("full_arm_validation_force_target_changed")
            if asdict(teacher or self.controller.config) != self._validation_parameter_pin[0]:
                raise RuntimeError("full_arm_validation_teacher_config_changed")
            if asdict(safety or self.safety.config) != self._validation_parameter_pin[1]:
                raise RuntimeError("full_arm_validation_safety_config_changed")

    def _validation_leader_sample(self):
        """Expose fresh complete leader evidence without reading its serial port here."""
        intent = self._arm_intent(self.clock(), None)
        if intent is None:
            raise RuntimeError("full_arm_validation_requires_leader")
        cached = self._leader_sample_cache
        if cached is not None and cached[0] is intent and cached[1] == intent.metadata_json:
            return cached[2]
        metadata = self._intent_metadata(intent)
        sample = FullArmLeaderSample(
            positions_deg=tuple((key.removesuffix(".pos"), value) for key, value in intent.goals),
            captured_ns=intent.captured_ns,
            response_completed_ns=metadata.get("leader_response_completed_ns"),
            sequence=metadata.get("leader_sequence"),
            source_config_sha256=metadata.get("leader_config_sha256"),
            epoch=intent.epoch,
        )
        self._leader_sample_cache = (intent, intent.metadata_json, sample)
        return sample

    def open(self):
        """Claim and connect the bus only on the thread that will execute all ticks."""
        if self._started:
            raise RuntimeError("Owner already opened")
        if self._stop.is_set():
            raise RuntimeError("user_stop_before_connect")
        # Programmatic callers pass the same no-I/O capability gate as the CLI.
        # Never accidentally use legacy B601.connect(calibrate=True).
        self.device.preflight_control()
        self.owner = threading.get_ident()
        self.device.bind_io_owner()
        try:
            self.device.connect()
        except BaseException as connection_error:
            errors = [connection_error]
            try:
                self._fault("owner_connection_failed")
            except BaseException as fault_error:
                errors.append(fault_error)
            try:
                self.device.disconnect()
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
            self._capture_shutdown_evidence()
            if len(errors) > 1:
                raise BaseExceptionGroup("Owner connection and cleanup failed", errors) from None
            raise
        self._started = True

    def _assert_owner(self):
        if not self._started or self.owner != threading.get_ident():
            raise RuntimeError("Runtime operation must run in the active bus owner")

    def request_stop(self):
        """Latch a priority user stop, independently of the ordinary mailbox."""
        self._stop.set()

    def post(self, kind, *, version, epoch, value=None):
        """Submit explicit arm/start/release/target/reset/segment intent from another context."""
        return self.mailbox.put(
            OwnerCommand(kind, version, epoch, self.clock(), int(self.config.command_ttl_ms * 1e6), value)
        )

    def post_arm_intent(self, intent: ArmIntent):
        """Accept only configured joints, preserving the leader gripper as trace intent."""
        if set(dict(intent.goals)) - {f"{name}.pos" for name in self.config.joint_names}:
            raise ValueError("Unknown joint in arm intent")
        return self.arm_mailbox.put(intent)

    def _invalidate_reference(self, *, preserve_last_sent=False):
        self.epoch += 1
        self.segment_id += 1
        self.mailbox.invalidate(self.epoch)
        self.arm_mailbox.invalidate(self.epoch)
        if not preserve_last_sent:
            self.last_sent = None
        self.last_success_tick = self.last_send_start = None
        self._last_feedback_sequence = self._last_feedback_receive_ns = None
        self._joint_evidence.clear()

    def _fault(self, reason):
        if self._fault_disposition is None:
            self.controller.fault(reason)
            self.safety.latch_fault(reason)
            self._invalidate_reference(preserve_last_sent=True)
            try:
                self._fault_disposition = self.device.emergency_stop(reason)
            except Exception as exc:
                self._fault_disposition = {
                    "action": self.safety.config.fault_action,
                    "status": "unknown",
                    "hardware_ack": "unknown",
                    "error": repr(exc),
                    "reason": reason,
                }

    def _startup_guard(self):
        """Recheck force/stop throughout startup and active guarded transport I/O."""
        if self._stop.is_set():
            raise RuntimeError(
                "user_stop_during_startup" if self.controller.phase == ControlPhase.ARMED else "user_stop"
            )
        self._check_validation_target()
        if self._hybrid_leader_required:
            self._arm_intent(self.clock(), None)
        self.trace.check_health()
        self.sensor._check_recorder_fault()
        sample = self.sensor.read_latest_before(
            self.clock(),
            max_age_ms=self.safety.config.force_max_age_ms,
            minimum_sequence=getattr(self.trace, "start_sequence", 0),
        )
        if not sample.is_valid or sample.sequence < getattr(self.trace, "start_sequence", 0):
            raise RuntimeError("invalid_startup_force")
        left = finite_number("left_N", sample.values[self.config.left_feature])
        right = finite_number("right_N", sample.values[self.config.right_feature])
        if left >= self.safety.config.hard_force_limit_left_N:
            raise RuntimeError("left_force_hard_limit")
        if right >= self.safety.config.hard_force_limit_right_N:
            raise RuntimeError("right_force_hard_limit")
        return sample

    def _apply_event(self, now, measured, force=None):
        command = self.mailbox.take()
        if command is None:
            return False
        if command.epoch != self.epoch:
            return False
        if command.ttl_ns <= 0 or not 0 <= now - command.created_ns <= command.ttl_ns:
            self._fault("command_ttl_expired")
            return True
        if self.controller.phase == ControlPhase.FAULT and command.kind != "reset":
            return False
        self.command_version = command.version
        self.event = command.kind
        if command.kind == "reset":
            # An explicit reset starts DISABLED, never resumes a cached closing command.
            self.controller.reset_fault()
            self.safety.reset_fault()
            self._stop.clear()
            self._fault_disposition = None
            self._invalidate_reference()
            self.mode = "TEACHER"
        elif command.kind == "teleop":
            if self.config.handover_tolerance_deg is None:
                raise ValueError("TELEOP handover requires an explicit synchronization tolerance")
            if self.controller.phase != ControlPhase.DISABLED:
                raise RuntimeError("TELEOP return requires completed explicit RELEASE first")
            self._invalidate_reference()
            self.safety.reset_motion_reference()
            self.mode = "HANDOVER_PENDING"
            self._handover_after_ns = now
        elif command.kind == "takeover":
            if self.mode not in ("TELEOP", "HANDOVER_PENDING"):
                raise RuntimeError("takeover requires TELEOP or HANDOVER_PENDING")
            self.controller.disable()
            self.controller.arm()
            self.safety.reset_motion_reference()
            self._invalidate_reference()
            self.mode = "TEACHER"
        elif command.kind == "arm":
            if self.mode in ("TELEOP", "HANDOVER_PENDING"):
                raise RuntimeError("Explicit takeover is required before Teacher arming")
            self.controller.arm()
        elif command.kind == "start":
            if self.controller.phase != ControlPhase.ARMED:
                raise RuntimeError("start requires explicit arm")
            request = {"mode": "force", "position_deg": None} if command.value is None else command.value
            if not isinstance(request, dict) or set(request) not in (
                {"mode", "position_deg"},
                {"mode", "position_deg", "position_speed_deg_s"},
            ):
                raise ValueError("start requires mode, position_deg and optional position_speed_deg_s")
            validate_motion_request(self.controller.config, self.safety.config, **request)
            if request["mode"] == "hybrid":
                if self._hybrid_leader_required:
                    raise RuntimeError("hybrid_restart_requires_new_session")
                # Validate the actual leader sample before any startup enable/write.
                self._hybrid_leader_required = True
                self._arm_intent(now, {})
            self._startup_performed = tuple(self.config.joint_names) == ("gripper",) and hasattr(
                self.device, "start_gripper"
            )
            if self._startup_performed:
                measured = self.device.start_gripper(self.latest_joint_snapshot, self._startup_guard)
            elif getattr(self.device, "full_arm_validation_config", None) is not None:
                if request["mode"] != "hybrid":
                    raise RuntimeError("full_arm_validation_requires_hybrid")
                self._check_validation_target()
                self.latest_joint_snapshot = self.device.start_full_arm_validation(
                    guard=self._startup_guard, leader_sample=self._validation_leader_sample
                )
                measured = self.latest_joint_snapshot.get("gripper").position_deg
                self._startup_performed = True
            elif getattr(self.device, "operator_teleoperation_config", None) is not None:
                if request["mode"] != "hybrid":
                    raise RuntimeError("operator_teleoperation_requires_hybrid")
                self.latest_joint_snapshot = self.device.start_operator_teleoperation(
                    guard=self._startup_guard, leader_sample=self._validation_leader_sample
                )
                measured = self.latest_joint_snapshot.get("gripper").position_deg
                self._startup_performed = True
            if self._startup_performed:
                sample = self._startup_guard()
                force = ForceObservation(
                    sample.values[self.config.left_feature],
                    sample.values[self.config.right_feature],
                    sample.timestamp_ns,
                    sample.arrival_timestamp_ns,
                    sample.sequence,
                    sample.is_valid,
                    sample.status,
                    calibration_id=self.config.software_calibration_id,
                )
            self.controller.start(
                now_ns=self.clock(), q_measured_deg=measured, force_N=force.mean_N, **request
            )
            self.safety.reset_motion_reference()
            self.last_sent = measured
            self.last_success_tick = self.last_send_start = None
            self.last_tick_start = self.clock()
        elif command.kind == "freeze":
            self.controller.freeze(now, measured)
            self.safety.reset_motion_reference(preserve_budget=True)
            self.last_success_tick = None
            self.last_tick_start = self.clock()
        elif command.kind == "resume":
            self.controller.resume(now, measured, force_N=force.mean_N)
            self.safety.reset_motion_reference(preserve_budget=True)
            self.last_sent = measured
            self.last_success_tick = None
            self.last_tick_start = self.clock()
        elif command.kind == "step":
            tolerance = self.safety.config.stall_position_tolerance_deg
            goal = self.controller.hold_target_deg
            if (
                goal is None
                or tolerance is None
                or self.last_sent is None
                or abs(measured - goal) > tolerance
                or abs(self.last_sent - goal) > tolerance
            ):
                raise RuntimeError("manual_step_requires_settled_hold")
            self.controller.step(finite_number("step_deg", command.value))
            self.last_success_tick = None
        elif command.kind in ("approach", "position"):
            if command.kind == "approach":
                request = canonicalize_teacher_fields(command.value)
                if set(request) != {
                    "gripper_follow_speed_deg_s",
                    "contact_force_threshold_N",
                }:
                    raise ValueError("approach requires explicit speed and force threshold")
                teacher = replace(self.controller.config, **request)
                self._check_validation_target(teacher=teacher)
                validate_profile(teacher, self.safety.config, self.controller.close_sign)
                validate_motion_request(teacher, self.safety.config, "approach")
                self.controller.update_config(teacher)
                self.controller.approach(now, measured)
                self._parameter_update = {
                    "timestamp_ns": now,
                    "teacher": asdict(teacher),
                    "safety": asdict(self.safety.config),
                }
            else:
                request = command.value
                if isinstance(request, dict):
                    if set(request) != {"position_deg", "position_speed_deg_s"}:
                        raise ValueError("position requires one position/speed bundle")
                    positive("position_speed_deg_s", request["position_speed_deg_s"])
                else:
                    request = {"position_deg": request}  # Legacy callers retain the global rate limits.
                validate_motion_request(self.controller.config, self.safety.config, "position", **request)
                self.controller.move_to(now, measured, **request)
            self.safety.reset_motion_reference(preserve_budget=True)
            # Retain the last successful wire position so a mode change cannot bypass slew limits.
            self.last_success_tick = None
            self.last_tick_start = self.clock()
        elif command.kind == "parameters":
            if not isinstance(command.value, dict) or set(command.value) != {"teacher", "safety"}:
                raise ValueError("parameters requires a complete teacher/safety bundle")
            teacher, safety = command.value["teacher"], command.value["safety"]
            self._check_validation_target(teacher=teacher, safety=safety)
            validate_profile(teacher, safety, self.controller.close_sign)
            live_teacher = {
                "force_target_N",
                "kp_deg_per_N",
                "ki_deg_per_N_s",
                "integral_deadband_N",
                "gripper_follow_speed_deg_s",
                "contact_force_threshold_N",
            }
            live_safety = {"max_close_speed_deg_s", "max_open_speed_deg_s", "acceleration_limit_deg_s2"}
            if any(
                asdict(teacher)[key] != value
                for key, value in asdict(self.controller.config).items()
                if key not in live_teacher
            ) or any(
                asdict(safety)[key] != value
                for key, value in asdict(self.safety.config).items()
                if key not in live_safety
            ):
                raise ValueError("device and protection settings require a stopped session")
            self.controller.update_config(teacher)
            self.safety.config = safety
            self._parameter_update = {
                "timestamp_ns": now,
                "teacher": asdict(teacher),
                "safety": asdict(safety),
            }
        elif command.kind == "release":
            self.controller.release(now_ns=now, q_measured_deg=measured)
            self.last_success_tick = None
        elif command.kind in ("disable", "end_episode", "discard"):
            self._fault(command.kind)
        elif command.kind == "target":
            self._check_validation_target(target=command.value)
            self.controller.set_target(command.value)
            self.last_success_tick = None
        elif command.kind == "segment":
            if command.value not in (
                "stable_hold",
                "low_force_recovery",
                "high_force_recovery",
                "contact_change_recovery",
            ):
                raise ValueError("Unknown supported bench segment")
            self.segment_id += 1
            self.last_success_tick = None
            self.event = str(command.value)
        else:
            raise ValueError("Unsupported owner event")
        return True

    def _intent_metadata(self, intent):
        """Decode immutable mailbox metadata once; all live safety checks still run."""
        if self._metadata_intent is not intent or self._metadata_json != intent.metadata_json:
            value = json.loads(intent.metadata_json)
            self._metadata_intent, self._metadata_json, self._metadata_value = (
                intent,
                intent.metadata_json,
                value,
            )
        return self._metadata_value

    def _arm_intent(self, anchor, row):
        intent = self.arm_mailbox.latest()
        active = self.controller.phase not in (ControlPhase.DISABLED, ControlPhase.ARMED)
        required = self._hybrid_leader_required or (
            self.config.arm_intent_required and (active or self.mode == "TELEOP")
        )
        if intent is None:
            if required:
                raise RuntimeError("arm_intent_missing")
            return None
        if intent.epoch != self.epoch or not 0 <= anchor - intent.captured_ns <= intent.ttl_ns:
            raise RuntimeError("arm_intent_ttl_expired")
        if self._hybrid_leader_required:
            self._leader_inputs(intent)
        if row is not None:
            row.update(
                arm_command_version=intent.version,
                arm_command_timestamp_ns=intent.captured_ns,
                arm_command_ttl_ns=intent.ttl_ns,
                arm_intent=dict(intent.goals),
                arm_intent_metadata=copy.deepcopy(self._intent_metadata(intent)),
                arm_filtered_keys=[] if self.mode == "TELEOP" else ["gripper.pos"],
            )
        return intent

    def _leader_inputs(self, intent):
        """Extract the sampled leader evidence; mailbox versions are not sample sequences."""
        if intent is None or not self._hybrid_leader_required:
            return {}
        cached = self._leader_input_cache
        if (
            cached is not None
            and cached[0] is intent
            and cached[1] == intent.metadata_json
            and cached[2] == self.config.joint_names
        ):
            return dict(cached[3])
        metadata = self._intent_metadata(intent)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("raw_action_deg"), dict):
            raise RuntimeError("hybrid_requires_leader_metadata")
        expected = {f"{name}.pos" for name in self.config.joint_names}
        if set(dict(intent.goals)) != expected or set(metadata["raw_action_deg"]) != expected:
            raise RuntimeError("hybrid_requires_complete_joint_intent")
        for name in expected:
            finite_number(f"leader_raw_{name}", metadata["raw_action_deg"][name])
        values = {
            "leader_gripper_deg": dict(intent.goals).get("gripper.pos"),
            "leader_raw_gripper_deg": metadata.get("raw_action_deg", {}).get("gripper.pos"),
            "leader_sequence": metadata.get("leader_sequence"),
            "leader_captured_ns": intent.captured_ns,
        }
        finite_number("leader_gripper_deg", values["leader_gripper_deg"])
        finite_number("leader_raw_gripper_deg", values["leader_raw_gripper_deg"])
        if type(values["leader_sequence"]) is not int or values["leader_sequence"] < 0:
            raise RuntimeError("hybrid_requires_leader_sample_sequence")
        if metadata.get("leader_request_started_ns") != intent.captured_ns:
            raise RuntimeError("hybrid_leader_capture_mismatch")
        self._leader_input_cache = (intent, intent.metadata_json, self.config.joint_names, dict(values))
        return values

    def tick(self):
        """Read fresh joints, select force once, propose, limit, submit, then publish evidence."""
        self._assert_owner()
        self.event = None
        self._parameter_update = None
        start = self.clock()
        dt = (
            1 / self.config.control_hz
            if self.last_tick_start is None
            else (start - self.last_tick_start) / 1e9
        )
        self.last_tick_start = start
        row = {
            "tick_id": self.tick_id,
            "run_uid": getattr(self.trace, "run_uid", "test"),
            "segment_id": self.segment_id,
            "epoch": self.epoch,
            "mode": self.mode,
            "phase_at_tick_start": str(self.controller.phase),
            "phase_before": str(self.controller.phase),
            "controller_id": self.controller.config.controller_id,
            "profile_id": self.controller.config.profile_id,
            "object_id": self.config.object_id,
            "contact_condition_id": self.config.contact_condition_id,
            "decision_anchor_ns": start,
            "dt_control_s": dt,
            "tick_start_ns": start,
            "deadline_miss": False,
            "feedback_valid": False,
            "teacher_active": False,
            "label_valid": False,
            "hybrid_transition": None,
            "contact_reference_candidate": None,
            "safety_override": False,
            "close_sign": self.controller.close_sign,
            "q_prev_sent_deg": self.last_sent,
            "q_measured_deg": None,
            "q_candidate_deg": None,
            "q_after_safety_deg": None,
            "q_sent_deg": None,
            "send_status": "none",
            "hardware_ack": "unknown",
            "send_start_ns": None,
            "send_end_ns": None,
            "dt_send_s": None,
            "prev_successful_tick_id": self.last_success_tick,
            "u_label_deg_s": None,
            "force_sequence": None,
            "force_hardware_sequence": None,
            "force_status": "unavailable",
            "force_valid": False,
            "force_max_age_ms": self.safety.config.force_max_age_ms,
            "joint_feedback_max_age_ms": self.safety.config.joint_feedback_max_age_ms,
            "force_timestamp_ns": None,
            "force_arrival_timestamp_ns": None,
            "force_left_N": None,
            "force_right_N": None,
            "force_mean_N": None,
            "force_target_requested_N": self.controller.config.force_target_N,
            "force_target_effective_N": None,
            "failure_reason": None,
            "clip_flags": [],
            "calibration_id": self.config.software_calibration_id,
            "filter_id": "none",
            "force_metric": "mean_normal",
            "target_source": "explicit_profile_or_event",
            "controller_algorithm": "ip",
            "main_frame_id": None,
            "arm_command_version": None,
            "mode_transition": False,
            "candidate_source": self.controller.config.controller_id,
            "submitted_action": {},
            "arm_intent": {},
            "arm_intent_metadata": {},
            "arm_filtered_keys": [],
            "arm_command_timestamp_ns": None,
            "arm_command_ttl_ns": None,
        }
        try:
            reset_applied = False
            if self._stop.is_set():
                self._fault("user_stop")
            # Reset is an explicit owner event, accepted even while normal reads are blocked.
            # It only enters DISABLED; the following tick must reacquire valid inputs.
            if self.controller.phase == ControlPhase.FAULT:
                reset_applied = self._apply_event(start, 0.0)
                if reset_applied and self.controller.phase == ControlPhase.DISABLED:
                    row.update(
                        mode_transition=True,
                        epoch=self.epoch,
                        segment_id=self.segment_id,
                        prev_successful_tick_id=None,
                        q_prev_sent_deg=None,
                    )
            self.trace.check_health()
            self._check_validation_target()
            self.sensor._check_recorder_fault()
            if not self.sensor.is_connected:
                raise RuntimeError("force_sensor_disconnected")
            if self.controller.phase == ControlPhase.FAULT:
                row["safety_override"] = True
            elif reset_applied:
                # Reset is a masked ownership boundary. Fresh feedback is required next tick;
                # no old reference, long operator pause or second mailbox event can command here.
                pass
            else:
                request_deadline = start + int(self.config.feedback_budget_ms * 1e6)
                snapshot = self.device.read_joint_snapshot(
                    tuple(self.config.joint_names), deadline_ns=request_deadline
                )
                self.latest_joint_snapshot = snapshot
                anchor = self.clock()
                row["decision_anchor_ns"] = anchor
                feedback = snapshot.get("gripper")
                row.update(
                    q_measured_deg=feedback.position_deg,
                    feedback_valid=snapshot.is_valid,
                    joint_receive_timestamp_ns=feedback.receive_timestamp_ns,
                    joint_receive_sequence=feedback.receive_sequence,
                    feedback_sequence=feedback.receive_sequence,
                    feedback_received_ns=feedback.receive_timestamp_ns,
                    feedback_age_ns=None
                    if feedback.receive_timestamp_ns is None
                    else anchor - feedback.receive_timestamp_ns,
                    feedback_time_source=feedback.time_source,
                    joint_age_ns=None
                    if feedback.receive_timestamp_ns is None
                    else anchor - feedback.receive_timestamp_ns,
                    joint_time_source=feedback.time_source,
                    joint_error=feedback.error,
                    joint_snapshot=asdict(snapshot),
                )
                if not snapshot.is_valid or feedback.position_deg is None:
                    raise RuntimeError("invalid_joint_feedback")
                if {item.name for item in snapshot.joints} != set(self.config.joint_names) or len(
                    snapshot.joints
                ) != len(self.config.joint_names):
                    raise RuntimeError("joint_feedback_names_mismatch")
                if (
                    snapshot.deadline_ns != request_deadline
                    or snapshot.completed_ns > request_deadline
                    or anchor > request_deadline
                ):
                    raise RuntimeError("joint_feedback_deadline")
                # A reused receive sequence is not new evidence even if a cache accessor reports now.
                if self._last_feedback_sequence is not None and (
                    feedback.receive_sequence is None
                    or feedback.receive_sequence <= self._last_feedback_sequence
                    or feedback.receive_timestamp_ns < self._last_feedback_receive_ns
                ):
                    raise RuntimeError("joint_feedback_not_updated")
                self._last_feedback_sequence = feedback.receive_sequence
                self._last_feedback_receive_ns = feedback.receive_timestamp_ns
                for joint in snapshot.joints:
                    finite_number("position_deg", joint.position_deg)
                    if (
                        type(joint.receive_timestamp_ns) is not int
                        or type(joint.receive_sequence) is not int
                        or joint.receive_sequence < 0
                        or not 0
                        <= anchor - joint.receive_timestamp_ns
                        <= self.safety.config.joint_feedback_max_age_ms * 1e6
                    ):
                        raise RuntimeError(f"invalid_joint_receive_evidence:{joint.name}")
                    previous = self._joint_evidence.get(joint.name)
                    if previous and (
                        joint.receive_sequence <= previous[0] or joint.receive_timestamp_ns < previous[1]
                    ):
                        raise RuntimeError(f"joint_feedback_not_updated:{joint.name}")
                    self._joint_evidence[joint.name] = (joint.receive_sequence, joint.receive_timestamp_ns)
                sample = self.sensor.read_latest_before(
                    anchor,
                    max_age_ms=self.safety.config.force_max_age_ms,
                    minimum_sequence=getattr(self.trace, "start_sequence", 0),
                )
                if sample.sequence < getattr(self.trace, "start_sequence", 0):
                    raise RuntimeError("force_sample_precedes_recording_subscription")
                force = ForceObservation(
                    sample.values[self.config.left_feature],
                    sample.values[self.config.right_feature],
                    sample.timestamp_ns,
                    sample.arrival_timestamp_ns,
                    sample.sequence,
                    sample.is_valid,
                    sample.status,
                    calibration_id=self.config.software_calibration_id,
                )
                row.update(
                    force_sequence=sample.sequence,
                    force_hardware_sequence=sample.hardware_sequence,
                    force_timestamp_ns=sample.timestamp_ns,
                    force_arrival_timestamp_ns=sample.arrival_timestamp_ns,
                    force_age_ns=anchor - sample.timestamp_ns,
                    force_left_N=force.left_N,
                    force_right_N=force.right_N,
                    force_mean_N=force.mean_N,
                    force_valid=sample.is_valid,
                    force_status=sample.status,
                )
                changed = self._apply_event(anchor, feedback.position_deg, force)
                row["phase_before"] = str(self.controller.phase)
                if changed and self.event == "start" and self._startup_performed:
                    # Startup may outlast a leader publication. Its later cache entry
                    # must be tested against completion time, never the old anchor.
                    anchor = self.clock()
                    # Keep the original receive evidence; its derived ages must
                    # use this same final anchor rather than the pre-start time.
                    row.update(
                        decision_anchor_ns=anchor,
                        feedback_age_ns=anchor - row["feedback_received_ns"],
                        joint_age_ns=anchor - row["joint_receive_timestamp_ns"],
                        force_age_ns=anchor - row["force_timestamp_ns"],
                    )
                intent = self._arm_intent(anchor, row)
                if self.mode == "HANDOVER_PENDING" and intent is not None:
                    leader_gripper = dict(intent.goals).get("gripper.pos")
                    if (
                        intent.captured_ns >= self._handover_after_ns
                        and leader_gripper is not None
                        and abs(leader_gripper - feedback.position_deg) <= self.config.handover_tolerance_deg
                    ):
                        self.mode = "TELEOP"
                        self.last_sent = self.last_success_tick = self.last_send_start = None
                        self.safety.reset_motion_reference()
                        row["arm_filtered_keys"] = []
                        self.event = "teleop_synchronized"
                        changed = True
                row["mode_transition"] = changed
                row.update(epoch=self.epoch, segment_id=self.segment_id, mode=self.mode)
                row["prev_successful_tick_id"] = self.last_success_tick
                if changed and self.event == "start" and self._startup_performed:
                    row.update(
                        q_prev_sent_deg=self.last_sent,
                        force_target_effective_N=self.controller.config.force_target_N,
                        startup_evidence=getattr(self.device, "startup_receipt", None),
                    )
                    return self._finish_tick(row, start, dt, startup=True)
                q_prev = self.last_sent if self.last_sent is not None else feedback.position_deg
                row["q_prev_sent_deg"] = q_prev
                inputs = ControlInputs(
                    anchor,
                    dt,
                    feedback.position_deg,
                    q_prev,
                    force,
                    feedback.receive_timestamp_ns,
                    feedback.receive_sequence,
                    snapshot.is_valid,
                    **self._leader_inputs(intent),
                )
                proposal = self.controller.propose(inputs)
                if self.mode == "TELEOP":
                    if intent is None or "gripper.pos" not in dict(intent.goals):
                        raise RuntimeError("teleop_requires_fresh_gripper_intent")
                    q_teleop = dict(intent.goals)["gripper.pos"]
                    proposal = replace(
                        proposal,
                        command_enabled=True,
                        q_candidate_deg=q_teleop,
                        u_raw_deg_s=self.controller.close_sign * (q_teleop - q_prev) / dt,
                        delta_raw_deg=self.controller.close_sign * (q_teleop - q_prev),
                        delta_i_deg=0.0,
                        delta_p_deg=0.0,
                        delta_contact_deg=0.0,
                        tracking_tag="teleop",
                    )
                row.update(
                    q_candidate_deg=proposal.q_candidate_deg,
                    force_target_requested_N=proposal.target_N,
                    force_target_effective_N=proposal.target_N,
                    target_version=proposal.target_version,
                    teacher_active=proposal.command_enabled
                    and self.mode != "TELEOP"
                    and proposal.next_phase not in (ControlPhase.FOLLOW, ControlPhase.FOLLOW_WAIT_CLEAR),
                    error_N=proposal.error_N,
                    delta_i_deg=proposal.delta_i_deg,
                    delta_p_deg=proposal.delta_p_deg,
                    delta_contact_deg=proposal.delta_contact_deg,
                    delta_raw_deg=proposal.delta_raw_deg,
                    u_raw_deg_s=proposal.u_raw_deg_s,
                    tracking_tag=proposal.tracking_tag,
                    transition=proposal.transition,
                    phase_before=str(proposal.phase),
                    contact_reference_candidate=None
                    if proposal.contact_reference is None
                    else asdict(proposal.contact_reference),
                )
                if self.controller.hybrid_active:
                    row["arm_filtered_keys"] = (
                        []
                        if proposal.next_phase in (ControlPhase.FOLLOW, ControlPhase.FOLLOW_WAIT_CLEAR)
                        else ["gripper.pos"]
                    )
                decision = self.safety.apply(inputs, proposal, now_ns=self.clock())
                row.update(
                    q_after_safety_deg=decision.q_after_safety_deg,
                    u_limited_deg_s=decision.u_limited_deg_s,
                    clip_flags=list(decision.clip_flags),
                    safety_override=decision.safety_override,
                    i_blocked=decision.i_blocked,
                    blocked_direction=decision.blocked_direction,
                )
                if decision.fault_reason or proposal.failure_reason:
                    self._fault(decision.fault_reason or proposal.failure_reason)
                elif proposal.command_enabled:
                    if self._stop.is_set():
                        self._fault("user_stop_before_send")
                    elif self.clock() - start >= self.config.tick_deadline_ms * 1e6:
                        self._fault("tick_deadline_before_send")
                    else:
                        self.trace.check_health()
                        self._check_validation_target()
                        # Check visibility again immediately before submitting the command.
                        age_check = self.safety.check_inputs(inputs, now_ns=self.clock())
                        if age_check is not None:
                            self._fault(age_check.fault_reason)
                        elif (
                            self._stop.is_set() or self.clock() - start >= self.config.tick_deadline_ms * 1e6
                        ):
                            self._fault("stop_or_tick_deadline_at_submission")
                        else:
                            # An exception after entering an adapter may have followed a real send.
                            # Until its receipt returns, execution is unknown and last-sent cannot advance.
                            # Every scheduled arm submission uses this tick's fresh snapshot.
                            # No unspecified joint is filled with zero or silently commanded.
                            goals = (
                                {}
                                if intent is None
                                else {
                                    name.removesuffix(".pos"): value
                                    for name, value in intent.goals
                                    if name != "gripper.pos"
                                }
                            )
                            goals["gripper"] = decision.q_after_safety_deg
                            if (
                                intent is not None
                                and not 0 <= self.clock() - intent.captured_ns <= intent.ttl_ns
                            ):
                                raise RuntimeError("arm_intent_expired_before_send")
                            submission_ns = self.clock()
                            if (
                                not 0
                                <= submission_ns - inputs.now_ns
                                <= self.safety.config.decision_max_age_ms * 1e6
                            ):
                                raise RuntimeError("decision_expired")
                            if (
                                not 0
                                <= submission_ns - inputs.force.timestamp_ns
                                <= self.safety.config.force_max_age_ms * 1e6
                            ):
                                raise RuntimeError("stale_force")
                            if any(
                                not 0
                                <= submission_ns - item.receive_timestamp_ns
                                <= self.safety.config.joint_feedback_max_age_ms * 1e6
                                for item in snapshot.joints
                            ):
                                raise RuntimeError("joint_feedback_expired_before_send")
                            row.update(send_status="unknown", send_start_ns=self.clock())
                            if tuple(self.config.joint_names) == ("gripper",):
                                receipt = self.device.send_gripper_position(
                                    decision.q_after_safety_deg, feedback_snapshot=snapshot
                                )
                            else:
                                receipt = self.device.send_joint_positions_partial(
                                    goals, feedback_snapshot=snapshot
                                )
                            row.update(
                                send_start_ns=receipt.send_start_ns,
                                send_end_ns=receipt.send_end_ns,
                                per_joint_results=[asdict(item) for item in receipt.results],
                            )
                            if {item.name for item in receipt.results} != set(goals) or len(
                                receipt.results
                            ) != len(goals):
                                raise RuntimeError("partial_receipt_names_mismatch")
                            for item in receipt.results:
                                finite_number("candidate_deg", item.candidate_deg)
                                finite_number("clipped_deg", item.clipped_deg)
                                if item.candidate_deg != goals[item.name]:
                                    raise RuntimeError("partial_receipt_candidate_mismatch")
                                if item.status == "success":
                                    finite_number("submitted_deg", item.submitted_deg)
                                    row["submitted_action"][f"{item.name}.pos"] = item.submitted_deg
                                    self.last_submitted[item.name] = item.submitted_deg
                                    if item.name != "gripper" and item.clipped_deg != item.candidate_deg:
                                        row["clip_flags"].append(f"driver_clip:{item.name}")
                            result = receipt.get("gripper")
                            success = receipt.all_submitted and result.submitted_deg is not None
                            if success:
                                finite_number("submitted_deg", result.submitted_deg)
                            row.update(
                                send_start_ns=receipt.send_start_ns,
                                send_end_ns=receipt.send_end_ns,
                                send_status="submitted" if success else "unknown",
                                per_joint_results=[asdict(item) for item in receipt.results],
                                q_sent_deg=result.submitted_deg if success else None,
                                delta_final_deg=None
                                if not success
                                else self.controller.close_sign * (result.submitted_deg - q_prev),
                                command_lead_deg=None
                                if not success
                                else self.controller.close_sign
                                * (result.submitted_deg - feedback.position_deg),
                                dt_send_s=None
                                if self.last_send_start is None
                                else (receipt.send_start_ns - self.last_send_start) / 1e9,
                            )
                            if success:
                                row["u_label_deg_s"] = (
                                    self.controller.close_sign * (result.submitted_deg - q_prev) / dt
                                )
                                row["label_valid"] = (
                                    self.last_success_tick is not None
                                    and not changed
                                    and not decision.safety_override
                                    and proposal.phase == ControlPhase.FORCE_TRACK
                                    and proposal.next_phase == ControlPhase.FORCE_TRACK
                                    and proposal.delta_contact_deg == 0
                                    and self.mode != "TELEOP"
                                )
                                if result.clipped_deg != decision.q_after_safety_deg:
                                    row["clip_flags"].append("driver_clip")
                                self.last_sent, self.last_success_tick = result.submitted_deg, self.tick_id
                                self.last_send_start = receipt.send_start_ns
                            if self.mode != "TELEOP" and success:
                                self.controller.commit(
                                    proposal,
                                    q_sent_deg=result.submitted_deg,
                                    software_success=success,
                                    safety_override=decision.safety_override,
                                    q_accepted_deg=result.clipped_deg,
                                    i_blocked=decision.i_blocked,
                                )
                                if (
                                    success
                                    and self.controller.hybrid_active
                                    and proposal.transition is not None
                                ):
                                    row["mode_transition"] = True
                                    row["label_valid"] = False
                                    row["hybrid_transition"] = proposal.transition
                            self.safety.commit(q_prev, result.submitted_deg, dt, software_success=success)
                            if not success:
                                reason = getattr(self.device, "failure", None) or "send_failed_or_unknown"
                                row["failure_reason"] = reason
                                self._fault(reason)
                else:
                    self.controller.commit(proposal, software_success=False)
        except Exception as exc:
            if row["send_status"] == "unknown" and row["send_end_ns"] is None:
                row["send_end_ns"] = self.clock()
            row["failure_reason"] = repr(exc)
            self._fault(str(exc))
        return self._finish_tick(row, start, dt)

    def _finish_tick(self, row, start, dt, *, startup=False):
        """Publish every transition and retain failed/disabled evidence."""
        elapsed = (self.clock() - start) / 1e9
        row["loop_elapsed_s"] = elapsed
        row["deadline_miss"] = not startup and (
            elapsed * 1000 > self.config.tick_deadline_ms or dt > self.safety.config.max_dt_s
        )
        if row["deadline_miss"] and (
            self.controller.phase not in (ControlPhase.DISABLED, ControlPhase.FAULT) or self.mode == "TELEOP"
        ):
            self._fault("tick_deadline_after_send")
        if self.controller.phase == ControlPhase.FAULT:
            row["label_valid"] = False
            row["safety_override"] = True
            row["failure_reason"] = row["failure_reason"] or self.controller.fault_reason
        row.update(
            phase=str(self.controller.phase),
            mode=self.mode,
            event=self.event,
            command_version=self.command_version,
            fault_disposition=self._fault_disposition,
            parameter_update=self._parameter_update,
            hold_target_deg=self.controller.hold_target_deg,
            hold_speed_deg_s=self.controller.hold_speed_deg_s,
            contact_alignment_pending=self.controller.contact_alignment_pending,
            hybrid_state=str(self.controller.phase) if self.controller.hybrid_active else None,
            contact_reference=None
            if self.controller.contact_reference is None
            else asdict(self.controller.contact_reference),
            gripper_follow_speed_deg_s=self.controller.config.gripper_follow_speed_deg_s,
            contact_force_threshold_N=self.controller.config.contact_force_threshold_N,
            control_reference_reset_ns=self.last_tick_start
            if self.event in ("start", "freeze", "resume", "approach", "position")
            else None,
            event_timestamp_ns=row["decision_anchor_ns"] if self.event else None,
            event_kind="reference_change"
            if self.event == "target"
            else (
                "segment_annotation"
                if self.event
                in ("stable_hold", "low_force_recovery", "high_force_recovery", "contact_change_recovery")
                else self.event
            ),
        )
        if hasattr(self.device, "take_io_evidence"):
            row["device_io_evidence"] = self.device.take_io_evidence()
            if not row["device_io_evidence"]["complete"]:
                self._fault("device_io_evidence_incomplete")
                row.update(
                    phase=str(ControlPhase.FAULT),
                    label_valid=False,
                    safety_override=True,
                    failure_reason="device_io_evidence_incomplete",
                    fault_disposition=self._fault_disposition,
                )
        row.update(annotate_capture(row))
        if not self.trace.submit(row):
            self._fault("control_trace_incomplete")
            # Consumers must never bind a frame to an unpersistable successful-looking tick.
            row.update(
                phase=str(ControlPhase.FAULT),
                label_valid=False,
                safety_override=True,
                failure_reason="control_trace_incomplete",
                fault_disposition=self._fault_disposition,
            )
            row.update(annotate_capture(row))
        self.latest_row = row
        self.tick_id += 1
        return row

    def run_for(self, duration_s, *, stop_on_disabled=False):
        """Use absolute deadlines and skip missed slots; never replay catch-up commands."""
        self._assert_owner()
        positive("duration_s", duration_s)
        period_ns = round(1e9 / self.config.control_hz)
        until = self.clock() + int(duration_s * 1e9)
        deadline = self.clock()
        while self.clock() < until:
            self.tick()
            if self.controller.phase == ControlPhase.FAULT:
                break
            if stop_on_disabled and self.controller.phase == ControlPhase.DISABLED:
                break
            deadline += period_ns
            now = self.clock()
            if now >= deadline:
                deadline += ((now - deadline) // period_ns + 1) * period_ns
            self._stop.wait(max(0.0, (deadline - self.clock()) / 1e9))
        return self.latest_row

    def close(self):
        """Stop ordinary motion before transport teardown, always from the same owner."""
        self._assert_owner()
        errors = []
        try:
            if (
                self.controller.phase not in (ControlPhase.DISABLED, ControlPhase.FAULT)
                or self.mode == "TELEOP"
            ):
                self._fault("owner_teardown_while_active")
        except BaseException as exc:
            errors.append(exc)
        try:
            self.device.disconnect()
        except BaseException as exc:
            errors.append(exc)
        finally:
            self._started = False
            self._capture_shutdown_evidence()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Owner fault disposition and cleanup failed", errors)

    def _capture_shutdown_evidence(self):
        """Retain transport and disposition facts without performing more bus I/O."""
        self.shutdown_evidence = {
            "transport_closed": not bool(getattr(self.device, "has_resources", False)),
            "fault_disposition": getattr(self.device, "fault_receipt", self._fault_disposition),
            "device_io_evidence": self.device.take_io_evidence()
            if hasattr(self.device, "take_io_evidence")
            else None,
            "completed_ns": self.clock(),
        }
