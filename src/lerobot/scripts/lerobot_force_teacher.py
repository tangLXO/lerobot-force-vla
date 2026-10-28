#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Run the shared I-P owner with bounded asynchronous control/raw evidence."""

# SI unit N remains explicit in configuration dimensions.
# ruff: noqa: N815

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import draccus

from lerobot.force_control.b601_factory import B601HardwareConfig, validate_b601_sensor
from lerobot.force_control.config import (
    SafetyConfig,
    TeacherConfig,
    finite_number,
    positive,
    validate_motion_request,
    validate_profile,
)
from lerobot.force_control.demonstration import DemonstrationController, TeacherRecordConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES, FakeForceSensor, FakeGripper, FakePlant
from lerobot.force_control.leader import FakeReadonlyLeader, LeaderSamplerWorker
from lerobot.force_control.owner import OwnerService
from lerobot.force_control.runtime import ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.trace import BenchTrace, validate_trace
from lerobot.force_control.types import ControlPhase
from lerobot.robots.rebot_b601_follower import control_backend
from lerobot.robots.rebot_b601_follower.full_arm_control_types import leader_configuration_digest
from lerobot.sensors import SensorConfig, make_sensors_from_configs
from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig

_UNSTOPPED_BENCHES: list[tuple] = []


def _remember_exit_errors(trace, errors):
    """Retain earlier failures while resource cleanup needs another observation."""
    recorded = trace.metadata.setdefault("cleanup_errors", [])
    for error in errors:
        message = f"{type(error).__name__}: {error}"
        if message not in recorded:
            recorded.append(message)
    return recorded


def _close_trace_then_sensor(sensor, trace, *, terminal_state, complete, started, errors):
    """Drain recording and release its lease before acquisition is disconnected."""
    if started or trace.has_resources:
        try:
            trace.close(terminal_state=terminal_state, complete=False)
        except BaseException as exc:
            errors.append(exc)
    # A timeout retains the writer and lease, so its Sensor must stay alive.
    if not trace.has_resources and sensor.has_resources:
        try:
            sensor.disconnect()
        except BaseException as exc:
            errors.append(exc)
    recorded = _remember_exit_errors(trace, errors)
    if trace.closed:
        try:
            trace.finish_cleanup(
                complete=complete and not errors and not sensor.has_resources, errors=recorded
            )
        except BaseException as exc:
            errors.append(exc)
            _remember_exit_errors(trace, [exc])


def finalize_retained_benches() -> list[dict]:
    """Finish deferred log/Sensor cleanup only after the motor owner relinquishes resources.

    This never sends a motor command, retries a fault disposition or treats an
    earlier timeout as successful execution. Completed traces remain incomplete.
    """
    completed = []
    for session in tuple(_UNSTOPPED_BENCHES):
        service, sensor, trace = session[:3]
        leader = session[3] if len(session) >= 4 else None
        capture = session[4] if len(session) >= 5 else None
        if service.has_resources or (leader is not None and leader.has_resources):
            continue
        if capture is not None:
            try:
                capture.close(timeout_s=0.1)
            except Exception as exc:
                _remember_exit_errors(trace, [exc])
                continue
        if leader is not None:
            trace.metadata["leader_shutdown"] = {
                "resources_retained": False,
                "status": leader.latest().error,
            }
        trace.metadata["shutdown_evidence"] = service.runtime.shutdown_evidence
        errors = []
        _close_trace_then_sensor(
            sensor,
            trace,
            terminal_state=str(service.runtime.controller.phase),
            complete=False,
            started=False,
            errors=errors,
        )
        if sensor.has_resources or trace.has_resources:
            continue
        _UNSTOPPED_BENCHES.remove(session)
        completed.append(
            {
                "trace_root": str(trace.root.resolve()),
                "shutdown_evidence": service.runtime.shutdown_evidence,
                "cleanup_errors": list(trace.metadata.get("cleanup_errors", [])),
                "valid": False,
            }
        )
    return completed


@dataclass
class ForceTeacherRunConfig:
    """Explicit experiment settings; hardware critical values have no implicit defaults."""

    backend: str = "fake"
    dry_run: bool = True
    observe_only: bool = False
    allow_motion: bool = False
    explicit_arm: bool = False
    initial_mode: str = "force"
    initial_position_deg: float | None = None
    initial_position_speed_deg_s: float | None = None
    leader_backend: str | None = None
    leader_port: str | None = None
    leader_readonly_transport: str = "protocol_v1"
    leader_baudrate: int = 1_000_000
    leader_frequency_hz: float = 50.0
    leader_max_age_s: float = 0.15
    leader_read_budget_s: float = 0.1
    leader_joint_ids: dict[str, int] | None = None
    leader_joint_directions: dict[str, int] | None = None
    leader_joint_ranges: dict[str, list[float]] | None = None
    close_sign: int | None = None
    b601: B601HardwareConfig | None = None
    sensors: dict[str, SensorConfig] = field(default_factory=dict)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    trace_root: Path = field(default_factory=lambda: Path("outputs/force_teacher") / uuid.uuid4().hex)
    queue_capacity: int = 2048
    duration_s: float | None = 3.0
    startup_timeout_s: float = 2.0
    shutdown_timeout_s: float = 2.0
    raw_queue_capacity: int = 4096
    trace_join_timeout_s: float = 2.0
    recording: TeacherRecordConfig | None = None
    fake_sample_rate_hz: float = 200.0
    fake_response_time_s: float = 0.03
    fake_relaxation_N_s: float = 0.0

    def validate(self, *, sensors=None, joint_names=None):
        """Validate intent and profile before any transport may be opened."""
        if self.recording is not None:
            self.recording.validate()
            if self.runtime.control_hz != 50:
                raise ValueError("Teacher capture requires 50 Hz control and recording")
            if tuple(self.runtime.joint_names) != FAKE_B601_JOINT_NAMES:
                raise ValueError("Demonstration recording requires the complete seven-joint owner")
        if self.backend not in ("fake", "b601"):
            raise ValueError("backend must be fake or b601")
        for name in ("dry_run", "observe_only", "allow_motion", "explicit_arm"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.dry_run and self.backend != "fake":
            raise ValueError("dry_run requires backend=fake and never opens real hardware")
        if self.observe_only and (self.allow_motion or self.explicit_arm):
            raise ValueError("observe_only cannot authorize motion or arm")
        if self.initial_mode == "hybrid":
            scope = tuple(joint_names or self.runtime.joint_names)
            if scope != ("gripper",) and set(scope) != set(FAKE_B601_JOINT_NAMES):
                raise ValueError("Hybrid scope must be calibrated ID-7 or all seven B601 joints")
            if not self.explicit_arm:
                raise ValueError("Hybrid sessions require explicit_arm")
            if self.leader_backend not in ("fake", "rebot_102"):
                raise ValueError("Hybrid sessions require an explicit leader_backend")
            if self.dry_run and self.leader_backend != "fake":
                raise ValueError("dry_run cannot connect a physical leader")
            if self.backend == "b601" and self.leader_backend != "rebot_102":
                raise ValueError("Physical gripper hybrid control requires a real 102 leader")
            if self.leader_backend == "rebot_102":
                if not isinstance(self.leader_port, str) or not self.leader_port.strip():
                    raise ValueError("leader_port must be explicitly configured")
                if (
                    self.b601 is not None
                    and isinstance(self.b601.port, str)
                    and self.leader_port.strip().casefold() == self.b601.port.strip().casefold()
                ):
                    raise ValueError("Leader and follower cannot use the same serial port")
            elif self.leader_port is not None:
                raise ValueError("Fake leader cannot use a physical leader_port")
            if self.leader_readonly_transport not in ("protocol_v1", "sdk"):
                raise ValueError("leader_readonly_transport must be protocol_v1 or sdk")
            if type(self.leader_baudrate) is not int or self.leader_baudrate <= 0:
                raise ValueError("leader_baudrate must be a positive integer")
            for name in ("leader_frequency_hz", "leader_max_age_s", "leader_read_budget_s"):
                positive(name, getattr(self, name))
            if self.leader_read_budget_s >= self.leader_max_age_s:
                raise ValueError("Leader read budget must leave freshness margin")
            mapping = (self.leader_joint_ids, self.leader_joint_directions, self.leader_joint_ranges)
            if any(item is not None for item in mapping):
                if any(item is None or set(item) != set(scope) for item in mapping):
                    raise ValueError("Explicit leader mapping must cover exactly the selected joints")
                ids = tuple(self.leader_joint_ids.values())
                if any(type(value) is not int or not 0 <= value <= 253 for value in ids) or len(
                    set(ids)
                ) != len(ids):
                    raise ValueError("Leader joint IDs must be unique integers in 0..253")
                if any(
                    type(value) is not int or value == 0 for value in self.leader_joint_directions.values()
                ):
                    raise ValueError("Leader directions must be explicit nonzero integer scales")
                for name, bounds in self.leader_joint_ranges.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        raise ValueError(f"Leader range for {name} requires two bounds")
                    finite_number(f"{name}.minimum", bounds[0])
                    finite_number(f"{name}.maximum", bounds[1])
                    if bounds[0] >= bounds[1]:
                        raise ValueError(f"Leader range for {name} must increase")
            elif scope != ("gripper",) and self.leader_backend == "rebot_102":
                raise ValueError("Physical full-arm hybrid requires explicit leader joint mapping")
        elif any(
            item is not None
            for item in (
                self.leader_backend,
                self.leader_port,
                self.leader_joint_ids,
                self.leader_joint_directions,
                self.leader_joint_ranges,
            )
        ):
            raise ValueError("Leader configuration requires initial_mode=hybrid")
        if self.backend == "b601":
            if tuple(joint_names or self.runtime.joint_names) != ("gripper",):
                operator = None if self.b601 is None else self.b601.operator_teleoperation
                if operator is not None:
                    operator.validate()
                    if self.initial_mode != "hybrid":
                        raise ValueError("Operator full-arm teleoperation requires hybrid leader control")
                    if leader_configuration_digest(leader_configuration_projection(self)) != (
                        operator.leader_config_sha256
                    ):
                        raise ValueError("Operator teleoperation leader configuration changed")
                    if self.leader_max_age_s != operator.leader_max_age_s:
                        raise ValueError("Use the same explicit leader age in the run and operator profile")
                elif self.b601 is None or self.b601.validation is None:
                    control_backend.require_b601_control_acceptance()
                else:
                    capability = self.b601.validation
                    capability.validate()
                    if self.initial_mode != "hybrid":
                        raise ValueError("Bounded full-arm validation requires hybrid leader control")
                    if self.teacher.force_target_N != capability.target_force_N:
                        raise ValueError("Bounded full-arm validation requires its reviewed explicit force")
                    if leader_configuration_digest(leader_configuration_projection(self)) != (
                        capability.leader_config_sha256
                    ):
                        raise ValueError("Bounded full-arm validation leader configuration changed")
                    if self.duration_s is None or not 0 < self.duration_s <= capability.max_duration_s:
                        raise ValueError("Bounded full-arm validation requires an explicit bounded duration")
                    if self.b601.control is None:
                        raise ValueError("Bounded full-arm validation requires explicit control budgets")
                    self.b601.control.validate()
                    positive("shutdown_timeout_s", self.shutdown_timeout_s)
                    positive("control_hz", self.runtime.control_hz)
                    margin = (
                        self.b601.control.startup_budget_s
                        + self.shutdown_timeout_s
                        + 1 / self.runtime.control_hz
                    )
                    if self.duration_s + margin >= capability.max_duration_s:
                        raise ValueError("Bounded full-arm duration must leave startup and shutdown margin")
            if self.observe_only or not (self.allow_motion and self.explicit_arm):
                raise ValueError("Hardware motion requires allow_motion and explicit_arm")
        self.runtime.validate()
        validate_profile(self.teacher, self.safety, self.close_sign)
        validate_motion_request(
            self.teacher,
            self.safety,
            self.initial_mode,
            self.initial_position_deg,
            self.initial_position_speed_deg_s,
        )
        resolved_sensors = self.sensors if sensors is None else sensors
        if self.backend == "b601":
            if self.b601 is None:
                raise ValueError("An explicit b601 hardware session is required")
            self.b601.validate(self.safety, self.runtime, joint_names or self.runtime.joint_names)
            validate_b601_sensor(resolved_sensors, self.runtime)
        elif self.b601 is not None:
            raise ValueError("A B601 session cannot be silently ignored by backend=fake")
        if self.dry_run and resolved_sensors:
            raise ValueError("dry_run cannot open configured physical sensors")
        if self.duration_s is not None:
            positive("duration_s", self.duration_s)
        for name in ("startup_timeout_s", "shutdown_timeout_s", "trace_join_timeout_s"):
            positive(name, getattr(self, name))
        positive("fake_sample_rate_hz", self.fake_sample_rate_hz)
        positive("fake_response_time_s", self.fake_response_time_s)
        finite_number("fake_relaxation_N_s", self.fake_relaxation_N_s, minimum=0)
        for name in ("queue_capacity", "raw_queue_capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def leader_configuration_projection(config: ForceTeacherRunConfig) -> dict:
    """Describe the native absolute leader mapping; no cached offset file is applied."""
    return {
        "backend": config.leader_backend,
        "port": config.leader_port,
        "baudrate": config.leader_baudrate,
        "readonly_transport": config.leader_readonly_transport,
        "joint_ids": config.leader_joint_ids,
        "joint_directions": config.leader_joint_directions,
        "joint_ranges": config.leader_joint_ranges,
        "action_space": "rebot_b601_absolute_degrees_v1",
        "calibration_source": "native_absolute_mapping_no_offset_file",
    }


def _leader_worker(config: ForceTeacherRunConfig) -> LeaderSamplerWorker | None:
    """Construct a read-only leader only for an explicitly configured hybrid session."""
    if config.initial_mode != "hybrid":
        return None
    if config.leader_backend == "fake":
        source = FakeReadonlyLeader(goals_deg={f"{name}.pos": 0.0 for name in config.runtime.joint_names})
    else:
        source = RebotArm102Leader(
            RebotArm102LeaderTeleopConfig(
                port=config.leader_port,
                baudrate=config.leader_baudrate,
                readonly_transport=config.leader_readonly_transport,
                joint_ids=config.leader_joint_ids or {"gripper": 6},
                joint_directions=config.leader_joint_directions or {"gripper": -6},
                joint_ranges=config.leader_joint_ranges or {"gripper": [-270, 0]},
            )
        )
    guard, digest = None, None
    capability = (
        None
        if config.b601 is None
        else (getattr(config.b601, "operator_teleoperation", None) or config.b601.validation)
    )
    if capability is not None:
        digest = leader_configuration_digest(leader_configuration_projection(config))
        if digest != capability.leader_config_sha256:
            raise ValueError("Full-arm leader configuration changed")
        source_pin = asdict(source.config)

        def guard():
            # Both the reviewed run mapping and actual producer must remain fixed.
            if (
                leader_configuration_digest(leader_configuration_projection(config)) != digest
                or asdict(source.config) != source_pin
            ):
                raise RuntimeError("full_arm_validation_leader_configuration_changed")

    return LeaderSamplerWorker(
        source,
        frequency_hz=config.leader_frequency_hz,
        max_read_duration_s=config.leader_read_budget_s,
        configuration_guard=guard,
        source_config_sha256=digest,
    )


def _publish_leader(config, worker, service, previous):
    """Forward each measured leader attempt once per owner epoch, retaining its actual age."""
    if worker is None:
        return previous
    intent = worker.latest(max_age_s=config.leader_max_age_s)
    if not intent.is_valid:
        raise RuntimeError(f"leader_input_invalid: {intent.error}")
    if type(intent.source_sequence) is not int or intent.source_sequence <= 0:
        raise RuntimeError("leader_input_invalid: missing source sequence")
    mapped, raw = dict(intent.action_deg), dict(intent.raw_action_deg)
    expected = {f"{name}.pos" for name in config.runtime.joint_names}
    if set(mapped) != expected or set(raw) != expected:
        raise RuntimeError("leader_input_invalid: selected joint coverage mismatch")
    for name in expected:
        finite_number(f"leader_mapped_{name}", mapped[name])
        finite_number(f"leader_raw_{name}", raw[name])
    key = (intent.sequence, service.runtime.epoch)
    if key == previous:
        return previous
    service.submit_arm_intent(
        mapped,
        captured_ns=intent.request_started_ns,
        ttl_ms=min(config.runtime.command_ttl_ms, config.leader_max_age_s * 1000),
        metadata={
            "raw_action_deg": raw,
            "leader_sequence": intent.source_sequence,
            "leader_request_started_ns": intent.request_started_ns,
            "leader_response_completed_ns": intent.response_completed_ns,
            "leader_timestamp_source": intent.timestamp_source,
            "leader_joint_diagnostics": [asdict(item) for item in intent.joint_diagnostics],
            "leader_config_sha256": getattr(worker, "source_config_sha256", None),
        },
    )
    return key


def run(config: ForceTeacherRunConfig, *, live=None) -> dict:
    """Use one owner for CLI, GUI and recording; never run an alternate bench loop."""
    config.validate()
    if _UNSTOPPED_BENCHES:
        raise RuntimeError("Earlier owner shutdown was not confirmed; resources remain retained")
    if config.initial_mode == "hybrid":
        config = replace(config, runtime=replace(config.runtime, arm_intent_required=True))
    plant = (
        None
        if config.backend == "b601"
        else FakePlant(
            close_sign=config.close_sign,
            contact_position_deg=float(config.close_sign),
            response_time_s=config.fake_response_time_s,
            relaxation_N_s=config.fake_relaxation_N_s,
        )
    )
    sensor = (
        make_sensors_from_configs(config.sensors)[config.runtime.force_instance]
        if config.sensors
        else FakeForceSensor(plant, config.fake_sample_rate_hz)
    )
    device = config.b601.make_device() if config.backend == "b601" else FakeGripper(plant)
    controller = ForceTeacher(config.teacher, config.close_sign)
    trace = BenchTrace(
        config.trace_root,
        sensor,
        {
            "config": asdict(config),
            "controller_algorithm": "ip",
            "software_status": "software_tested",
            "hardware_status": "hardware_not_tested",
            "backend": config.backend,
            "force_instance": config.runtime.force_instance,
            "left_feature": config.runtime.left_feature,
            "right_feature": config.runtime.right_feature,
            "force_max_age_ms": config.safety.force_max_age_ms,
            "joint_feedback_max_age_ms": config.safety.joint_feedback_max_age_ms,
        },
        capacity=config.queue_capacity,
        raw_capacity=config.raw_queue_capacity,
        join_timeout_s=config.trace_join_timeout_s,
    )
    runtime = ForceRuntime(
        device,
        sensor,
        controller,
        SafetySupervisor(config.safety, config.close_sign),
        trace,
        config.runtime,
        stop_event=None if live is None else live.stop,
    )
    service = OwnerService(runtime)
    leader = _leader_worker(config)
    leader_version = None
    capture = None
    preview_at = 0.0

    def publish_leader():
        nonlocal leader_version
        leader_version = _publish_leader(config, leader, service, leader_version)

    def wait_command(version):
        """Keep leader input fresh while startup/parameter commands are acknowledged."""
        if leader is None:
            return service.wait_for_command(version, timeout_s=config.startup_timeout_s)
        deadline = time.monotonic() + config.startup_timeout_s
        while True:
            if live is not None and live.stop.is_set():
                runtime.request_stop()
                raise RuntimeError("user_stop_during_command")
            publish_leader()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for owner evidence")
            try:
                return service.wait_for_command(
                    version, timeout_s=min(remaining, 1 / config.runtime.control_hz)
                )
            except TimeoutError as exc:
                if str(exc) != "Timed out waiting for owner evidence":
                    raise

    errors, started, complete = [], False, False
    owner_start_attempted = False
    try:
        if live is not None and live.stop.is_set():
            raise RuntimeError("user_stop_before_connect")
        if config.recording is not None:
            if live is not None:
                live.recording = {"state": "preparing", "active": False, "file_complete": False}
            capture = DemonstrationController(
                service,
                sensor,
                force_instance=config.runtime.force_instance,
                provenance={"run_config": asdict(config), "bench_trace_root": str(config.trace_root)},
            )
            # First camera/codec initialization can also starve the read-only
            # leader's serial deadline. Prepare before either sampling worker.
            capture.prepare_before_control(config.recording)
            if live is not None:
                live.recording = capture.status()
        if live is not None and live.stop.is_set():
            raise RuntimeError("user_stop_before_owner_start")
        if leader is not None:
            leader.start()
            deadline = time.monotonic() + config.startup_timeout_s
            while not leader.latest(max_age_s=config.leader_max_age_s).is_valid:
                if live is not None and live.stop.is_set():
                    raise RuntimeError("user_stop_before_owner_start")
                if time.monotonic() >= deadline or not leader.has_resources:
                    raise TimeoutError(f"102 has no valid snapshot: {leader.latest().error}")
                time.sleep(min(0.005, max(0, deadline - time.monotonic())))
        sensor.connect()
        trace.start()
        started = True
        ready_by = time.monotonic() + config.startup_timeout_s
        while True:
            if live is not None and live.stop.is_set():
                raise RuntimeError("user_stop_before_owner_start")
            try:
                sample = sensor.read_latest_before(
                    time.perf_counter_ns(), max_age_ms=config.safety.force_max_age_ms
                )
            except (RuntimeError, TimeoutError):
                sample = None
            if sample is not None and sample.sequence >= trace.start_sequence:
                break
            if time.monotonic() >= ready_by:
                raise TimeoutError("No fresh post-subscription force sample")
            time.sleep(min(1 / config.runtime.control_hz, max(0, ready_by - time.monotonic())))
        owner_start_attempted = True
        service.start(timeout_s=config.startup_timeout_s)
        if config.explicit_arm:
            publish_leader()
            wait_command(service.post("arm"))
            publish_leader()
            wait_command(
                service.post(
                    "start",
                    value={
                        "mode": config.initial_mode,
                        "position_deg": config.initial_position_deg,
                        "position_speed_deg_s": config.initial_position_speed_deg_s,
                    },
                ),
            )
            if live is not None:
                live.startup = getattr(device, "startup_receipt", None)
        until = None if config.duration_s is None else time.monotonic() + config.duration_s
        finish_deadline = None
        capture_finish_requested = False
        release_quiesced = False
        while finish_deadline is not None or until is None or time.monotonic() < until:
            if live is not None and live.stop.is_set():
                runtime.request_stop()
                break
            if not release_quiesced:
                publish_leader()
            snapshot = service.snapshot()
            if live is not None:
                live.latest = snapshot.row
                if capture is not None:
                    live.recording = capture.status()
                    if time.monotonic() >= preview_at:
                        live.preview = capture.preview()
                        preview_at = time.monotonic() + 0.25
                if snapshot.row.get("parameter_update") is not None:
                    live.applied = snapshot.row["parameter_update"]
                if live.stop.is_set():
                    runtime.request_stop()
                    break
                pending = live.take() if finish_deadline is None else None
                if pending is not None:
                    kind, value = pending
                    if kind.startswith("record_"):
                        if capture is None:
                            live.recording = {
                                "state": "error",
                                "active": False,
                                "error": "Prepare recording settings and a seven-joint session before recording",
                            }
                        else:
                            if kind == "record_start" and not value:
                                value = {"config": asdict(config.recording)}
                            capture.submit(kind, value)
                        continue
                    if kind == "release_and_finish":
                        if value is not None:
                            raise ValueError("release_and_finish uses the configured gripper release")
                        if snapshot.row["phase"] not in (
                            "FORCE_TRACK",
                            "POSITION_HOLD",
                            "APPROACH",
                            "FOLLOW",
                            "FOLLOW_WAIT_CLEAR",
                        ):
                            raise RuntimeError("release_and_finish requires an active non-faulted controller")
                        requested_ns = time.perf_counter_ns()
                        version = service.post("release")
                        trace.metadata["finish_request"] = {
                            "kind": kind,
                            "requested_ns": requested_ns,
                            "command_version": version,
                        }
                    else:
                        version = service.post(kind, value=value)
                    applied = wait_command(version)
                    snapshot = applied
                    if kind == "parameters":
                        live.applied = applied.row.get("parameter_update")
                    elif kind == "release_and_finish":
                        # Keep polling and forwarding fresh leader observations while the
                        # existing bounded release runs. Stop always remains a priority event.
                        finish_deadline = (
                            time.monotonic() + config.teacher.max_release_time_s + config.shutdown_timeout_s
                        )
            if snapshot.row["phase"] == "FAULT":
                raise RuntimeError(snapshot.row.get("failure_reason") or "owner fault")
            if finish_deadline is not None:
                if snapshot.row["phase"] == "DISABLED":
                    trace.metadata["release_and_finish_completed"] = True
                    if not release_quiesced:
                        # Stop the physical session before encoding/saving can
                        # consume its watchdog or bounded validation deadline.
                        # Controller RELEASE has completed normally, so this
                        # close does not turn the episode into a user-stop fault.
                        service.quiesce_after_release(timeout_s=config.shutdown_timeout_s)
                        release_quiesced = True
                        trace.metadata["release_stopped_before_save"] = True
                    if capture is not None:
                        recording_status = capture.status()
                        if not capture_finish_requested and recording_status.get("active"):
                            capture.submit(
                                "record_finish",
                                {"task_result": recording_status.get("task_result", "unknown")},
                            )
                            capture_finish_requested = True
                            finish_deadline = time.monotonic() + 30
                        if capture_finish_requested:
                            if recording_status.get("state") == "error":
                                raise RuntimeError(
                                    "Released demonstration could not be saved: "
                                    + str(recording_status.get("error"))
                                )
                            if recording_status.get("state") != "saved":
                                if time.monotonic() >= finish_deadline:
                                    raise TimeoutError(
                                        "Released demonstration did not finish saving within 30 seconds"
                                    )
                                live.stop.wait(1 / config.runtime.control_hz)
                                continue
                    break
                if time.monotonic() >= finish_deadline:
                    raise TimeoutError("release_and_finish did not reach DISABLED within its release budget")
            # Wake on user stop; control cadence remains solely in OwnerService.
            if live is not None:
                live.stop.wait(1 / config.runtime.control_hz)
            else:
                time.sleep(1 / config.runtime.control_hz)
        if (
            live is None
            and leader is None
            and controller.phase
            in (
                ControlPhase.FORCE_TRACK,
                ControlPhase.POSITION_HOLD,
                ControlPhase.APPROACH,
            )
        ):
            service.wait_for_command(service.post("release"), timeout_s=config.startup_timeout_s)
            service.wait_for_phase(
                ControlPhase.DISABLED, timeout_s=config.teacher.max_release_time_s + config.shutdown_timeout_s
            )
        complete = True
    except BaseException as exc:
        origin = service.fault_origin
        if origin is not None:
            row = origin.row
            trace.metadata["fault_origin"] = {
                **{
                    key: row.get(key)
                    for key in (
                        "tick_id",
                        "failure_reason",
                        "joint_error",
                        "joint_snapshot",
                        "tick_start_ns",
                        "decision_anchor_ns",
                        "loop_elapsed_s",
                    )
                },
                "epoch": origin.epoch,
            }
            if row.get("feedback_valid") is False and row.get("joint_error"):
                note = f"joint_error: {row['joint_error']}"
                if note not in getattr(exc, "__notes__", ()):
                    exc.add_note(note)
        errors.append(exc)
        runtime.request_stop()
    finally:
        try:
            service.stop(timeout_s=config.shutdown_timeout_s)
        except BaseException as exc:
            if not any(exc is existing for existing in errors):
                errors.append(exc)
            complete = False
        if capture is not None:
            try:
                capture.close(timeout_s=config.shutdown_timeout_s)
            except BaseException as exc:
                errors.append(exc)
                complete = False
            if live is not None:
                live.recording = capture.status()
        if leader is not None:
            try:
                leader.stop(timeout_s=config.shutdown_timeout_s)
            except BaseException as exc:
                errors.append(exc)
                complete = False
            trace.metadata["leader_shutdown"] = {
                "resources_retained": leader.has_resources,
                "status": leader.latest().error,
            }
        retained = (
            (service, sensor, trace, leader, capture)
            if capture is not None
            else (service, sensor, trace)
            if leader is None
            else (service, sensor, trace, leader)
        )
        if (
            service.has_resources
            or (leader is not None and leader.has_resources)
            or (capture is not None and capture.has_resources)
        ):
            if live is not None:
                live.latest = {
                    **(runtime.latest_row or {}),
                    "cleanup_pending": True,
                    "cleanup_error": "; ".join(str(error) for error in errors),
                }
            _UNSTOPPED_BENCHES.append(retained)
            errors.append(RuntimeError("Owner or leader shutdown unconfirmed; dependent resources retained"))
            _remember_exit_errors(trace, errors)
        else:
            trace.metadata["shutdown_evidence"] = runtime.shutdown_evidence
            _close_trace_then_sensor(
                sensor,
                trace,
                terminal_state=str(controller.phase),
                complete=complete,
                started=started,
                errors=errors,
            )
            if sensor.has_resources or trace.has_resources:
                _UNSTOPPED_BENCHES.append(retained)
        if live is not None:
            # The supervisor must retain closure facts even if preparation failed
            # before trace.start() could create a manifest. This is no motor-off claim.
            live.shutdown = runtime.shutdown_evidence
            if not owner_start_attempted and not service.has_resources:
                live.shutdown = {
                    "transport_closed": True,
                    "owner_start_attempted": False,
                    "fault_disposition": None,
                    "completed_ns": time.perf_counter_ns(),
                }
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("I-P session failed, including cleanup evidence", errors)
    report = validate_trace(config.trace_root)
    report.update(
        hardware_status="hardware_not_tested",
        trace_root=str(config.trace_root.resolve()),
        shutdown_evidence=runtime.shutdown_evidence,
    )
    if capture is not None:
        report["recording"] = capture.status()
    return report


def main():
    """Parse the repository's standard dataclass/draccus CLI and report actual evidence."""
    config = draccus.parse(ForceTeacherRunConfig)
    report = run(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
