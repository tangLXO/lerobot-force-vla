#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Independent bench evidence, not a LeRobot Sensor Sidecar or training dataset.

The bus owner only submits detached rows to a bounded queue. A separate worker
serializes both control rows and an existing Sensor subscription. There is no
sensor acquisition, command submission, automatic recovery, or hardware access
here. A missing/incomplete manifest is never accepted as complete evidence.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .capture_annotations import ANNOTATION_SCHEMA, annotate_capture
from .types import ControlPhase

if TYPE_CHECKING:
    from lerobot.sensors import Sensor
    from lerobot.sensors.sensor import SensorRecorderLease, SensorSubscription

SCHEMA = "teacher_bench_schema_v1"
CONTROL_FILE = "control.jsonl"
RAW_FILE = "raw_force.jsonl"
MANIFEST_FILE = "manifest.json"

# Null values retain failed/disabled ticks; they must not become valid labels.
REQUIRED_FIELDS = frozenset(
    {
        "tick_id",
        "segment_id",
        "epoch",
        "decision_anchor_ns",
        "dt_control_s",
        "phase",
        "mode",
        "teacher_active",
        "label_valid",
        "safety_override",
        "deadline_miss",
        "feedback_valid",
        "q_prev_sent_deg",
        "q_measured_deg",
        "q_candidate_deg",
        "q_after_safety_deg",
        "q_sent_deg",
        "send_status",
        "hardware_ack",
        "send_start_ns",
        "send_end_ns",
        "dt_send_s",
        "prev_successful_tick_id",
        "close_sign",
        "u_label_deg_s",
        "force_sequence",
        "force_timestamp_ns",
        "force_arrival_timestamp_ns",
        "force_left_N",
        "force_right_N",
        "force_mean_N",
        "force_target_effective_N",
        "force_target_requested_N",
    }
)

CAPTURE_INPUT_FIELDS = frozenset(
    {
        "phase_at_tick_start",
        "phase_before",
        "force_valid",
        "force_max_age_ms",
        "joint_feedback_max_age_ms",
        "contact_force_threshold_N",
    }
)
CAPTURE_ANNOTATION_FIELDS = frozenset(
    {
        "annotation_schema",
        "phase_after",
        "phase_changed",
        "capture_stage",
        "contact_estimate",
        "contact_estimate_valid",
        "contact_estimate_rule",
        "action_label_valid",
    }
)


class TraceError(RuntimeError):
    """A trace can no longer be regarded as complete evidence."""


def _json_default(value: Any) -> Any:
    # Sensor scalars may be NumPy values; avoid a NumPy/hardware dependency here.
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


def _encode(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, default=_json_default)


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class BenchTrace:
    """Bounded asynchronous control/raw recorder for one newly created run.

    ``start`` and ``close`` belong outside the high-frequency loop. Call
    ``check_health`` before normal commands on every tick. ``submit`` returning
    false requires the owner to latch a fault; it never silently drops labels.
    A close timeout retains the Sensor lease and worker, preventing sequence
    reset/reuse; call close again after the worker actually terminates.
    """

    schema = SCHEMA

    def __init__(
        self,
        root: str | Path,
        sensor: Sensor,
        metadata: dict[str, Any],
        *,
        capacity: int = 2048,
        raw_capacity: int | None = None,
        join_timeout_s: float = 5.0,
    ) -> None:
        """Prepare bounded queues without starting workers or opening files."""
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        raw_capacity = capacity if raw_capacity is None else raw_capacity
        if type(raw_capacity) is not int or raw_capacity <= 0:
            raise ValueError("raw_capacity must be a positive integer")
        if isinstance(join_timeout_s, bool) or not math.isfinite(join_timeout_s) or join_timeout_s <= 0:
            raise ValueError("join_timeout_s must be finite and positive")
        self.root = Path(root)
        self.sensor = sensor
        self.metadata = copy.deepcopy(metadata)
        self.run_uid = str(uuid.uuid4())
        self.capacity = capacity
        self.raw_capacity = raw_capacity
        self.join_timeout_s = join_timeout_s
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(capacity)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease: SensorRecorderLease | None = None
        self._subscription: SensorSubscription | None = None
        self._error: str | None = None
        self._error_lock = threading.Lock()
        self._started = False
        self._accepting = False
        self._closed = False
        self._counts = {CONTROL_FILE: 0, RAW_FILE: 0}
        self._manifest: dict[str, Any] = {}

    @property
    def has_resources(self) -> bool:
        """Keep ownership while the writer or its Sensor recorder lease remains."""
        return self._lease is not None or (self._thread is not None and self._thread.is_alive())

    @property
    def closed(self) -> bool:
        """Whether the files were sealed and recorder ownership was released."""
        return self._closed

    @property
    def start_sequence(self) -> int:
        """First recorded Sensor sequence; older history is outside this run."""
        if self._subscription is None:
            raise TraceError("Trace has not subscribed")
        return self._subscription.start_sequence

    @property
    def failed(self) -> bool:
        """Include subscriber/lease failures even before the writer notices."""
        self._check_sensor_errors()
        return self._error is not None

    @property
    def failure_reason(self) -> str | None:
        """Return the first latched failure, including Sensor ownership errors."""
        self._check_sensor_errors()
        return self._error

    def _fail(self, reason: str) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = reason

    def _check_sensor_errors(self) -> None:
        if self._subscription is not None and self._subscription.overflowed:
            self._fail("Sensor subscription overflow; raw evidence incomplete")
        if self._lease is not None and self._lease.error is not None:
            self._fail(f"Sensor recorder lease failed: {self._lease.error}")

    def check_health(self) -> None:
        """Non-I/O check suitable for the bus owner before applying a command."""
        if self.failed:
            raise TraceError(self._error)
        if not self._started or self._closed:
            raise TraceError("Trace is not active")

    def _write_manifest(self) -> None:
        temp = self.root / "manifest.json.tmp"
        temp.write_text(_encode(self._manifest) + "\n", encoding="utf-8")
        temp.replace(self.root / MANIFEST_FILE)

    def start(self) -> None:
        """Acquire existing Sensor recording ownership, then start disk worker."""
        if self._started:
            raise TraceError("A BenchTrace instance cannot be restarted")
        # Exclusive new directory is the run writer ownership boundary.
        self.root.mkdir(parents=True, exist_ok=False)
        self._started = True
        try:
            self._lease = self.sensor.acquire_recorder()
            self._subscription = self.sensor.subscribe(self.raw_capacity)
            self._manifest = {
                "schema": self.schema,
                "run_uid": self.run_uid,
                "complete": False,
                "terminal_state": "RUNNING",
                "clock": "host.perf_counter_ns",
                "created_ns": time.perf_counter_ns(),
                "metadata": self.metadata,
                "sensor_provenance": self.sensor.provenance,
                "sensor_features": {key: asdict(value) for key, value in self.sensor.features.items()},
                "native_features": {key: asdict(value) for key, value in self.sensor.native_features.items()},
                "start_sequence": self.start_sequence,
                "end_sequence": None,
                "capacity": self.capacity,
                "raw_capacity": self.raw_capacity,
                "files": {},
                "failure_reason": None,
                "durability": "incomplete runs are rejected; no power-loss durability guarantee",
            }
            self._write_manifest()
            self._thread = threading.Thread(target=self._worker, name="force-bench-trace", daemon=True)
            self._accepting = True
            self._thread.start()
        except BaseException as exc:
            self._fail(f"Trace startup failed: {type(exc).__name__}: {exc}")
            self._accepting = False
            if self._subscription is not None:
                self.sensor.unsubscribe(self._subscription)
            if self._lease is not None:
                self.sensor.release_recorder(self._lease)
                self._lease = None
            raise

    def submit(self, row: dict[str, Any]) -> bool:
        """Detach and enqueue one bounded row; never serialize or write on owner."""
        if not self._accepting or self.failed:
            return False
        try:
            detached = copy.deepcopy(row)
            detached["run_uid"] = self.run_uid
            self._queue.put_nowait(detached)
        except queue.Full:
            self._fail("Control trace queue overflow; run incomplete")
            return False
        except Exception as exc:
            self._fail(f"Trace row snapshot failed: {type(exc).__name__}: {exc}")
            return False
        self._wake.set()
        return True

    def _write_row(self, handle: Any, row: dict[str, Any]) -> None:
        handle.write(_encode(row) + "\n")

    def _worker(self) -> None:
        try:
            with (
                (self.root / CONTROL_FILE).open("x", encoding="utf-8", newline="\n") as control,
                (self.root / RAW_FILE).open("x", encoding="utf-8", newline="\n") as raw,
            ):
                while True:
                    self._wake.clear()
                    self._check_sensor_errors()
                    for _ in range(64):
                        try:
                            row = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        self._write_row(control, row)
                        self._counts[CONTROL_FILE] += 1
                    for _ in range(256):
                        try:
                            sample = self._subscription.get_nowait()
                        except queue.Empty:
                            break
                        row = asdict(sample)
                        payload = row.pop("native_payload")
                        row["native_payload_base64"] = (
                            base64.b64encode(payload).decode("ascii") if payload is not None else None
                        )
                        row["run_uid"] = self.run_uid
                        self._write_row(raw, row)
                        self._counts[RAW_FILE] += 1
                    empty = self._queue.empty() and self._subscription.queue.empty()
                    if self._stop.is_set() and empty:
                        break
                    if empty:
                        self._wake.wait(0.005)
        except BaseException as exc:
            self._fail(f"Trace writer failed: {type(exc).__name__}: {exc}")

    def close(self, terminal_state: str, complete: bool) -> dict[str, Any]:
        """Unsubscribe/drain outside the owner, seal files, and release the lease."""
        if not self._started:
            raise TraceError("Trace was not started")
        if self._closed:
            return copy.deepcopy(self._manifest)
        self._accepting = False
        try:
            if self._subscription is not None and not self._subscription.closed:
                self.sensor.unsubscribe(self._subscription)
        except Exception as exc:
            self._fail(f"Sensor unsubscribe failed: {type(exc).__name__}: {exc}")
            raise TraceError(self._error) from exc  # retain resources and lease
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(self.join_timeout_s)
            if self._thread.is_alive():
                self._fail("Trace writer shutdown timeout; worker and recorder lease retained")
                raise TraceError(self._error)
        self._check_sensor_errors()
        self._manifest.update(
            terminal_state=terminal_state,
            complete=bool(complete and not self.failed),
            closed_ns=time.perf_counter_ns(),
            end_sequence=self._subscription.end_sequence if self._subscription is not None else None,
            failure_reason=self._error,
            files={},
        )
        try:
            for name, count in self._counts.items():
                path = self.root / name
                if path.is_file():
                    self._manifest["files"][name] = {
                        "rows": count,
                        "size_bytes": path.stat().st_size,
                        "sha256": _digest(path),
                    }
            self._write_manifest()
        except Exception as exc:
            self._fail(f"Trace sealing failed: {type(exc).__name__}: {exc}")
            self._manifest["complete"] = False
            self._manifest["failure_reason"] = self._error
            raise TraceError(self._error) from exc
        finally:
            if self._lease is not None:
                self.sensor.release_recorder(self._lease)
                self._lease = None
            self._closed = True
        return copy.deepcopy(self._manifest)

    def finish_cleanup(self, *, complete: bool, errors: list[str]) -> dict[str, Any]:
        """Record dependent-resource cleanup after ``close(..., complete=False)``.

        The caller disconnects the Sensor only after recorder ownership is
        released, then makes the final completeness decision here. A failed
        trace or earlier cleanup error cannot be upgraded on a later retry.
        """
        if (
            type(complete) is not bool
            or not isinstance(errors, list)
            or any(not isinstance(error, str) or not error for error in errors)
        ):
            raise ValueError("Cleanup requires a boolean completion decision and error strings")
        if not self.closed or self.has_resources:
            raise TraceError("Trace must release its writer and recorder lease before cleanup finishes")
        recorded = list(self.metadata.get("cleanup_errors", []))
        for error in errors:
            if error not in recorded:
                recorded.append(error)
        self.metadata["cleanup_errors"] = recorded
        if recorded:
            self._fail(f"Session cleanup failed: {'; '.join(recorded)}")
        self._manifest.update(
            metadata=self.metadata,
            complete=bool(complete and not self.failed),
            failure_reason=self._error,
            cleanup_finished_ns=time.perf_counter_ns(),
        )
        try:
            self._write_manifest()
        except Exception as exc:
            self._fail(f"Trace cleanup sealing failed: {type(exc).__name__}: {exc}")
            self._manifest["complete"] = False
            self._manifest["failure_reason"] = self._error
            raise TraceError(self._error) from exc
        return copy.deepcopy(self._manifest)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    values = sorted(values)

    def percentile(p: float) -> float:
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[-1],
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant {value}")

    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line, parse_constant=reject_constant) for line in handle]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Non-object row in {path.name}")
    return rows


def validate_trace(
    root: str | Path,
    *,
    require_complete: bool = True,
    band_n: float | None = None,
    recovery_dwell_s: float | None = None,
) -> dict[str, Any]:
    """Read-only full integrity, causal reference, predecessor and label checks.

    This verifier never repairs a run, creates locks, rewrites manifests, or
    claims that software submission is motor execution/arrival confirmation.
    ``require_complete=False`` permits forensic examination only; errors still
    include any file/row corruption. Training must use the default strict mode.
    """
    for name, value in (("band_n", band_n), ("recovery_dwell_s", recovery_dwell_s)):
        if value is not None and (not _finite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")
    if recovery_dwell_s is not None and band_n is None:
        raise ValueError("recovery_dwell_s requires an explicitly chosen band_n")
    root = Path(root)
    errors: list[str] = []
    report: dict[str, Any] = {"schema": SCHEMA, "valid": False, "errors": errors, "statistics": {}}
    try:
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            errors.append("Unsupported bench schema")
        if manifest.get("clock") != "host.perf_counter_ns":
            errors.append("Unsupported/missing clock domain")
        if require_complete and manifest.get("complete") is not True:
            errors.append("Run is incomplete")
        if manifest.get("terminal_state") not in {"DISABLED", "STOPPED", "FAULT", "COMPLETED"}:
            errors.append("Missing recognized terminal state")
        if not isinstance(manifest.get("closed_ns"), int):
            errors.append("Missing close evidence")
        if manifest.get("complete") and manifest.get("failure_reason"):
            errors.append("Complete run contains latched trace failure")
        parsed: dict[str, list[dict[str, Any]]] = {}
        for name in (CONTROL_FILE, RAW_FILE):
            path = root / name
            evidence = manifest.get("files", {}).get(name, {})
            if not evidence:
                errors.append(f"Missing manifest evidence for {name}")
            if evidence.get("size_bytes") != path.stat().st_size or evidence.get("sha256") != _digest(path):
                errors.append(f"File integrity mismatch: {name}")
            parsed[name] = _rows(path)
            if evidence.get("rows") != len(parsed[name]):
                errors.append(f"Row count mismatch: {name}")
        control, raw = parsed[CONTROL_FILE], parsed[RAW_FILE]
        if not control:
            errors.append("No control ticks")
        elif (
            manifest.get("complete")
            and manifest.get("terminal_state") in {"DISABLED", "FAULT"}
            and control[-1].get("phase") != manifest["terminal_state"]
        ):
            errors.append("Terminal manifest disagrees with last control phase")
        _validate_rows(manifest, control, raw, errors)
        _validate_capture_annotations(manifest, control, raw, errors)
        report["statistics"] = _statistics(control, raw, band_n=band_n, recovery_dwell_s=recovery_dwell_s)
        report["capture_summary"] = _capture_summary(manifest, control)
        report["run_uid"] = manifest.get("run_uid")
        report["complete"] = manifest.get("complete")
        report["terminal_state"] = manifest.get("terminal_state")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"Malformed trace: {type(exc).__name__}: {exc}")
    report["valid"] = not errors
    return report


def _validate_capture_annotations(
    manifest: dict[str, Any], control: list[dict[str, Any]], raw: list[dict[str, Any]], errors: list[str]
) -> None:
    """Verify declared annotations without adding them to historical evidence."""
    metadata = manifest.get("metadata", {})
    declared = metadata.get("capture_annotation_schema")
    row_declared = any(CAPTURE_ANNOTATION_FIELDS.intersection(row) for row in control)
    if "capture_annotation_schema" not in metadata and not row_declared:
        return
    if declared != ANNOTATION_SCHEMA:
        errors.append("Unsupported/missing capture annotation schema in metadata")
    phases = {str(phase) for phase in ControlPhase}
    raw_by_sequence = {sample.get("sequence"): sample for sample in raw}
    required = CAPTURE_INPUT_FIELDS | CAPTURE_ANNOTATION_FIELDS
    for row in control:
        prefix = f"Tick {row.get('tick_id')}: "
        missing = required - row.keys()
        if missing:
            errors.append(prefix + f"missing capture annotation fields: {sorted(missing)}")
            continue
        for field in ("phase_at_tick_start", "phase_before", "phase_after"):
            if not isinstance(row[field], str) or row[field] not in phases:
                errors.append(prefix + f"unrecognized capture phase: {field}")
        for field in ("force_valid", "phase_changed", "contact_estimate_valid", "action_label_valid"):
            if type(row[field]) is not bool:
                errors.append(prefix + f"{field} must be a boolean")
        for field in ("force_max_age_ms", "joint_feedback_max_age_ms"):
            if not _finite(row[field]) or row[field] <= 0:
                errors.append(prefix + f"capture annotations require positive finite {field}")
        threshold = row["contact_force_threshold_N"]
        if threshold is not None and (not _finite(threshold) or threshold <= 0):
            errors.append(prefix + "contact_force_threshold_N must be null or positive finite")
        sample = raw_by_sequence.get(row.get("force_sequence"))
        if row["force_valid"] is not (sample is not None and sample.get("is_valid") is True):
            errors.append(prefix + "force_valid disagrees with selected raw evidence")
        expected = annotate_capture(row)
        for field, value in expected.items():
            if type(row[field]) is not type(value) or row[field] != value:
                errors.append(prefix + f"capture annotation mismatch: {field}")


def _capture_summary(manifest: dict[str, Any], control: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe recorded coverage, never physical completion or training acceptance."""
    declared = manifest.get("metadata", {}).get("capture_annotation_schema")
    annotated = [row for row in control if row.get("annotation_schema") == ANNOTATION_SCHEMA]
    stage_counts = Counter(row.get("capture_stage", "unknown") for row in annotated)
    transitions: Counter[str] = Counter()
    release_epochs: set[int] = set()
    completed_releases = 0
    for row in annotated:
        before, after = row.get("phase_before"), row.get("phase_after")
        phases = (row.get("phase_at_tick_start"), before, after)
        for left, right in zip(phases, phases[1:], strict=False):
            if isinstance(left, str) and isinstance(right, str) and left != right:
                transitions[f"{left}->{right}"] += 1
        epoch = row.get("epoch")
        normal = (
            type(epoch) is int
            and row.get("failure_reason") is None
            and row.get("safety_override") is False
            and row.get("deadline_miss") is False
        )
        if normal and before == after == "RELEASE" and row.get("send_status") == "submitted":
            release_epochs.add(epoch)
        if (
            normal
            and epoch in release_epochs
            and before == "RELEASE"
            and after == "DISABLED"
            and row.get("transition") == "RELEASE->DISABLED"
            and row.get("send_status") == "none"
        ):
            completed_releases += 1
            release_epochs.discard(epoch)
        elif not normal or after != "RELEASE":
            release_epochs.discard(epoch)
    observed = {
        stage: stage_counts[stage] > 0
        for stage in ("approach", "contact_transition", "force_control", "release")
    }

    def labels(field: str) -> dict[str, int]:
        return {
            "valid": sum(row.get(field) is True for row in control),
            "invalid": sum(row.get(field) is False for row in control),
            "unavailable": sum(type(row.get(field)) is not bool for row in control),
        }

    return {
        "available": declared == ANNOTATION_SCHEMA,
        "annotation_schema": declared,
        "file_complete": manifest.get("complete") is True,
        "control_tick_count": len(control),
        "annotated_tick_count": len(annotated),
        "phase_counts": dict(Counter(row.get("phase", "unknown") for row in control)),
        "phase_changed_tick_count": sum(row.get("phase_changed") is True for row in annotated),
        "phase_transition_count": sum(transitions.values()),
        "phase_transition_counts": dict(transitions),
        "capture_stage_counts": dict(stage_counts),
        "contact_estimate_counts": dict(Counter(row.get("contact_estimate", "unknown") for row in annotated)),
        "force_label_counts": labels("label_valid"),
        "action_label_counts": labels("action_label_valid"),
        "observed_stages": observed,
        "all_stages_observed": all(observed.values()),
        "normal_release_completed": completed_releases > 0,
        "normal_release_completed_count": completed_releases,
        "software_evidence_only": True,
    }


def _validate_b601_io(metadata: dict, control: list[dict], errors: list[str]) -> None:
    """Require contiguous host I/O audit and truthful cleanup for the unified backend."""
    if metadata.get("backend", metadata.get("follower_backend")) != "b601":
        return
    shutdown = metadata.get("shutdown_evidence")
    if not isinstance(shutdown, dict) or shutdown.get("transport_closed") is not True:
        errors.append("B601 shutdown lacks confirmed transport closure")
        shutdown = shutdown if isinstance(shutdown, dict) else {}
    disposition = shutdown.get("fault_disposition")
    if (
        not isinstance(disposition, dict)
        or disposition.get("status") != "submitted"
        or disposition.get("hardware_ack") != "unknown"
        or disposition.get("hardware_stop_confirmed") is not False
    ):
        errors.append("B601 shutdown lacks truthful software disposition evidence")
    streams = []
    for row in control:
        startup = row.get("startup_evidence")
        if isinstance(startup, dict):
            stages = startup.get("io_stages", [])
            if not isinstance(stages, list):
                errors.append("B601 startup I/O stages are malformed")
            else:
                # Startup drains its bounded audit between disabled preloads
                # and enables. These are canonical events between this row's
                # preceding tick and its ordinary end-of-tick audit.
                streams.extend(stages)
        streams.append(row.get("device_io_evidence"))
    streams.append(shutdown.get("device_io_evidence"))
    sequence, previous_time = 1, -1
    io_errors = set()

    def io_error(message):
        # A single omitted range must not create one repeated error per later
        # event. Keep strict rejection while bounding the diagnostic output.
        if message not in io_errors:
            io_errors.add(message)
            errors.append(message)

    for stream in streams:
        if (
            not isinstance(stream, dict)
            or stream.get("complete") is not True
            or stream.get("correlation") != "host_rx_only_no_request_token"
            or stream.get("hardware_ack") != "unknown"
            or not isinstance(stream.get("events"), list)
        ):
            io_error("B601 I/O audit is missing, incomplete or misrepresents correlation")
            continue
        for event in stream["events"]:
            if not isinstance(event, dict):
                io_error("B601 I/O event is malformed")
                continue
            timestamp = event.get("timestamp_ns")
            kind = event.get("kind")
            if event.get("sequence") != sequence or type(timestamp) is not int or timestamp < previous_time:
                io_error("B601 I/O event sequence/time is not contiguous and monotonic")
            sequence += 1
            if type(timestamp) is int:
                previous_time = timestamp
            try:
                data = bytes.fromhex(event.get("data_hex", ""))
            except (ValueError, TypeError):
                data = b""
            if not data or kind not in {
                "query_attempt",
                "query_rx",
                "command_attempt",
                "command_rx",
                "pipelined_control_rx",
                "cached_control_rx",
                "pre_query_unattributed_rx",
                "parameter_write_unattributed_rx",
                "full_validation_stop_rx",
                "post_disable_unexpected_rx",
            }:
                io_error("B601 I/O event has malformed kind or bytes")
            if kind in {"query_attempt", "command_attempt"} and (
                len(data) != 30 or data[:4] != b"\x55\xaa\x1e\x03"
            ):
                io_error("B601 I/O TX event is not a reviewed bridge packet")
    if sequence == 1:
        errors.append("B601 I/O audit contains no transport evidence")


def _validate_rows(
    manifest: dict[str, Any], control: list[dict[str, Any]], raw: list[dict[str, Any]], errors: list[str]
) -> None:
    _validate_b601_io(manifest.get("metadata", {}), control, errors)
    raw_by_sequence: dict[int, dict[str, Any]] = {}
    previous_arrival = -1
    expected_sequence = manifest.get("start_sequence")
    for row in raw:
        sequence = row.get("sequence")
        if sequence != expected_sequence:
            errors.append(f"Raw sequence gap/duplicate at {sequence}")
        expected_sequence = sequence + 1
        if row.get("run_uid") != manifest.get("run_uid"):
            errors.append(f"Raw run identity mismatch at {sequence}")
        if row["arrival_timestamp_ns"] < previous_arrival:
            errors.append(f"Raw arrival reversed at {sequence}")
        previous_arrival = row["arrival_timestamp_ns"]
        raw_by_sequence[sequence] = row
    if expected_sequence != manifest.get("end_sequence"):
        errors.append("Raw subscription end sequence mismatch")
    ticks: dict[int, dict[str, Any]] = {}
    last_anchor = -1
    last_success: dict[tuple[Any, Any], dict[str, Any]] = {}
    previous_boundary = None
    previous_row = None
    fault_epoch = None
    metadata = manifest.get("metadata", {})
    for feature in (
        metadata.get("left_feature", "left.normal_force"),
        metadata.get("right_feature", "right.normal_force"),
    ):
        if manifest.get("sensor_features", {}).get(feature, {}).get("unit") != "N":
            errors.append(f"Raw semantic feature must declare N units: {feature}")
    for row in control:
        tick = row.get("tick_id")
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"Tick {tick} missing fields: {sorted(missing)}")
            continue
        if type(tick) is not int or tick in ticks or (ticks and tick <= max(ticks)):
            errors.append(f"Nonunique/nonincreasing tick ID {tick}")
        if tick != len(ticks):
            errors.append(f"Control tick sequence gap at {tick}")
        ticks[tick] = row
        prefix = f"Tick {tick}: "
        for field in ("label_valid", "feedback_valid", "teacher_active", "safety_override", "deadline_miss"):
            if type(row[field]) is not bool:
                errors.append(prefix + f"{field} must be a boolean")
        if type(row["epoch"]) is not int or row["epoch"] < 0:
            errors.append(prefix + "epoch must be a nonnegative integer")
        if previous_row is not None and row["epoch"] < previous_row["epoch"]:
            errors.append(prefix + "epoch must never decrease")
        if type(row["close_sign"]) is not int or row["close_sign"] not in (-1, 1):
            errors.append(prefix + "close_sign must be a calibrated integer +1/-1")
        if row.get("run_uid") != manifest.get("run_uid"):
            errors.append(prefix + "run identity mismatch")
        anchor = row["decision_anchor_ns"]
        if type(anchor) is not int or anchor <= last_anchor:
            errors.append(prefix + "nonincreasing decision anchor")
        last_anchor = anchor
        communication_only = row.get("communication_only", False)
        if communication_only and (
            row.get("send_status") != "none"
            or row.get("submitted_action")
            or row.get("per_joint_results")
            or row.get("feedback_valid")
            or row.get("label_valid")
            or row.get("action_label_valid")
        ):
            errors.append(prefix + "communication-only evidence cannot submit actions or control labels")
        if (
            not communication_only
            and previous_row is not None
            and "tick_start_ns" in row
            and "tick_start_ns" in previous_row
        ):
            base = previous_row.get("control_reference_reset_ns") or previous_row["tick_start_ns"]
            actual_dt = (row["tick_start_ns"] - base) / 1e9
            if not _finite(row["dt_control_s"]) or not math.isclose(
                row["dt_control_s"], actual_dt, rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append(prefix + "dt_control_s differs from actual tick-start interval")
        if row["feedback_valid"]:
            received = row.get("feedback_received_ns")
            if (
                type(received) is not int
                or received > anchor
                or type(row.get("feedback_sequence")) is not int
                or not _finite(row["q_measured_deg"])
                or row.get("feedback_age_ns") != anchor - received
                or row.get("feedback_time_source") in (None, "unavailable", "")
            ):
                errors.append(prefix + "valid feedback requires causal receive time/sequence/age/source")
            if row.get("joint_snapshot"):
                snapshot = row["joint_snapshot"]
                joint = next((item for item in snapshot["joints"] if item.get("name") == "gripper"), None)
                if joint is None or any(
                    (
                        joint.get("position_deg") != row["q_measured_deg"],
                        joint.get("receive_sequence") != row.get("feedback_sequence"),
                        joint.get("receive_timestamp_ns") != received,
                        joint.get("is_valid") is not True,
                        snapshot.get("completed_ns", anchor + 1) > anchor,
                    )
                ):
                    errors.append(prefix + "feedback view disagrees with visible joint snapshot")
        sample = raw_by_sequence.get(row["force_sequence"])
        causal = False
        if row["force_sequence"] is not None:
            if sample is None:
                errors.append(prefix + "missing raw force reference")
            else:
                causal = bool(
                    sample["is_valid"]
                    and sample["timestamp_ns"] <= anchor
                    and sample["arrival_timestamp_ns"] <= anchor
                )
                if not causal:
                    errors.append(prefix + "invalid/noncausal force reference")
                for field, raw_field in (
                    ("force_timestamp_ns", "timestamp_ns"),
                    ("force_arrival_timestamp_ns", "arrival_timestamp_ns"),
                ):
                    if row[field] != sample[raw_field]:
                        errors.append(prefix + f"force reference mismatch: {field}")
                for field, feature in (
                    ("force_left_N", metadata.get("left_feature", "left.normal_force")),
                    ("force_right_N", metadata.get("right_feature", "right.normal_force")),
                ):
                    if row[field] != sample["values"].get(feature):
                        errors.append(prefix + f"raw semantic value mismatch: {field}")
                if _finite(row["force_mean_N"]):
                    mean = (row["force_left_N"] + row["force_right_N"]) / 2
                    if not math.isclose(row["force_mean_N"], mean, abs_tol=1e-8):
                        errors.append(prefix + "force mean mismatch")
        if row["hardware_ack"] != "unknown":
            errors.append(prefix + "bench schema has no command-level hardware acknowledgement evidence")
        submitted = row["send_status"] == "submitted"
        if row["send_status"] not in {"submitted", "none", "unknown", "not_attempted"}:
            errors.append(prefix + "unknown software submission status")
        if submitted:
            if not _finite(row["q_sent_deg"]):
                errors.append(prefix + "submitted command lacks finite final target")
            if not (
                type(row["send_start_ns"]) is int
                and type(row["send_end_ns"]) is int
                and anchor <= row["send_start_ns"] <= row["send_end_ns"]
            ):
                errors.append(prefix + "invalid submission timing")
        elif row["q_sent_deg"] is not None:
            errors.append(prefix + "unconfirmed submission cannot have q_sent_deg")
        if row.get("per_joint_results"):
            results = row["per_joint_results"]
            gripper = next((item for item in results if item.get("name") == "gripper"), None)
            # Overall submission covers the complete action. A stop on an arm
            # joint can leave the gripper not attempted and the frame unknown.
            all_submitted = all(item.get("status") == "success" for item in results)
            valid_results = all(
                item.get("status") in {"success", "unknown", "not_attempted"}
                and (
                    _finite(item.get("submitted_deg"))
                    if item.get("status") == "success"
                    else item.get("submitted_deg") is None
                )
                for item in results
            )
            if gripper is None or (
                not valid_results
                or all_submitted != submitted
                or (submitted and gripper.get("submitted_deg") != row["q_sent_deg"])
                or (not submitted and row["send_status"] != "unknown")
            ):
                errors.append(prefix + "final target disagrees with per-joint software receipt")
        if fault_epoch is not None:
            if (
                row.get("event") == "reset"
                and row["epoch"] > fault_epoch
                and row["phase"] == "DISABLED"
                and not submitted
            ):
                fault_epoch = None
            elif submitted and not row["safety_override"]:
                errors.append(prefix + "normal submission after fault requires explicit disabled reset")
        boundary = (row["epoch"], row["segment_id"])
        if previous_boundary != boundary:
            last_success.clear()
        predecessor = last_success.get(boundary)
        ref = row["prev_successful_tick_id"]
        if ref is not None:
            if predecessor is None or ref != predecessor["tick_id"]:
                errors.append(prefix + "predecessor crosses boundary or is not last successful tick")
            elif row["q_prev_sent_deg"] != predecessor["q_sent_deg"]:
                errors.append(prefix + "last-sent reference mismatch")
        if row["label_valid"]:
            if (
                ref is None
                or predecessor is None
                or not submitted
                or not causal
                or not row["feedback_valid"]
                or row["safety_override"]
                or not row["teacher_active"]
                or row["deadline_miss"]
                or row.get("action_override", False)
                or row.get("mode_transition", False)
                or row.get("hybrid_transition") is not None
                or row.get("delta_contact_deg", 0) != 0
                or row["phase"]
                in ("FAULT", "DISABLED", "RELEASE", "ARMED", "APPROACH", "FOLLOW", "FOLLOW_WAIT_CLEAR")
                or (previous_row is not None and row["mode"] != previous_row["mode"])
            ):
                errors.append(prefix + "unsafe/initial/failed/discontinuous tick must be masked")
            # A timestamp may be causal yet too old for an admissible input.
            check_time = row["send_start_ns"] if submitted else anchor
            for field, limit in (
                ("force_timestamp_ns", metadata.get("force_max_age_ms")),
                ("feedback_received_ns", metadata.get("joint_feedback_max_age_ms")),
            ):
                source_time = row.get(field)
                if not _finite(limit) or limit <= 0:
                    errors.append(prefix + f"valid label needs a recorded positive age budget for {field}")
                if (
                    _finite(limit)
                    and type(check_time) is int
                    and type(source_time) is int
                    and check_time - source_time > limit * 1e6
                ):
                    errors.append(prefix + f"stale {field} must be masked")
            dt = row["dt_control_s"]
            if not _finite(dt) or dt <= 0 or row["close_sign"] not in (-1, 1):
                errors.append(prefix + "invalid label integration dt or close_sign")
            elif all(_finite(row[field]) for field in ("q_sent_deg", "q_prev_sent_deg", "u_label_deg_s")):
                label = row["close_sign"] * (row["q_sent_deg"] - row["q_prev_sent_deg"]) / dt
                if not math.isclose(row["u_label_deg_s"], label, rel_tol=1e-8, abs_tol=1e-8):
                    errors.append(prefix + "label must use final target and dt_control_s")
            else:
                errors.append(prefix + "valid label needs finite reference/target/rate")
        if submitted:
            last_success[boundary] = row
        previous_boundary = boundary
        if not communication_only:
            previous_row = row
        if row["phase"] == "FAULT" or row["send_status"] in {"unknown", "not_attempted"}:
            fault_epoch = row["epoch"]


def _statistics(
    control: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    *,
    band_n: float | None = None,
    recovery_dwell_s: float | None = None,
) -> dict[str, Any]:
    scheduled = [row for row in control if not row.get("communication_only", False)]
    anchors = [row["decision_anchor_ns"] for row in scheduled if type(row.get("decision_anchor_ns")) is int]
    intervals = [(right - left) / 1e9 for left, right in zip(anchors, anchors[1:], strict=False)]
    dt = [row["dt_control_s"] for row in scheduled if _finite(row.get("dt_control_s"))]
    sends = [
        (row["send_end_ns"] - row["send_start_ns"]) / 1e6
        for row in control
        if type(row.get("send_start_ns")) is int and type(row.get("send_end_ns")) is int
    ]
    ages = [
        (row["decision_anchor_ns"] - row["force_timestamp_ns"]) / 1e6
        for row in control
        if type(row.get("decision_anchor_ns")) is int and type(row.get("force_timestamp_ns")) is int
    ]
    errors = [
        row["force_mean_N"] - row["force_target_effective_N"]
        for row in control
        if row.get("phase") == "FORCE_TRACK"
        and _finite(row.get("force_mean_N"))
        and _finite(row.get("force_target_effective_N"))
    ]
    selected = [row.get("force_sequence") for row in control if row.get("force_sequence") is not None]
    feedback = [row.get("feedback_sequence") for row in control if row.get("feedback_sequence") is not None]
    receives = sorted(
        {
            row["feedback_received_ns"]
            for row in control
            if row.get("feedback_valid") and type(row.get("feedback_received_ns")) is int
        }
    )
    raw_times = sorted(row["timestamp_ns"] for row in raw if row.get("is_valid"))
    return {
        "control_tick_count": len(scheduled),
        "communication_row_count": len(control) - len(scheduled),
        "raw_attempt_count": len(raw),
        "label_valid_count": sum(row.get("label_valid") is True for row in control),
        "measured_control_hz": (len(anchors) - 1) * 1e9 / (anchors[-1] - anchors[0])
        if len(anchors) > 1 and anchors[-1] > anchors[0]
        else None,
        "decision_interval_s": _distribution(intervals),
        "dt_control_s": _distribution(dt),
        "send_duration_ms": _distribution(sends),
        "selected_force_age_ms": _distribution(ages),
        "deadline_miss_count": sum(row.get("deadline_miss") is True for row in control),
        "force_sample_updates": len(set(selected)),
        "feedback_updates": len(set(feedback)),
        "measured_feedback_update_hz": (len(receives) - 1) * 1e9 / (receives[-1] - receives[0])
        if len(receives) > 1 and receives[-1] > receives[0]
        else None,
        "selected_force_reuse_count": sum(a == b for a, b in zip(selected, selected[1:], strict=False)),
        "raw_valid_response_hz": (len(raw_times) - 1) * 1e9 / (raw_times[-1] - raw_times[0])
        if len(raw_times) > 1 and raw_times[-1] > raw_times[0]
        else None,
        "force_track_mae_N": sum(abs(value) for value in errors) / len(errors) if errors else None,
        "force_track_rmse_N": math.sqrt(sum(value * value for value in errors) / len(errors))
        if errors
        else None,
        "force_track_max_overshoot_N": max(0.0, max(errors)) if errors else None,
        "saturation_tick_count": sum(bool(row.get("clip_flags")) for row in control),
        "safety_override_tick_count": sum(row.get("safety_override") is True for row in control),
        "submission_failure_tick_count": sum(
            row.get("send_status") not in ("submitted", "none") for row in control
        ),
        "band_N": band_n,
        "force_track_in_band_fraction": sum(abs(value) <= band_n for value in errors) / len(errors)
        if errors and band_n is not None
        else None,
        "recovery_dwell_s": recovery_dwell_s,
        "segment_recovery": _recovery_statistics(control, band_n, recovery_dwell_s),
        "software_submission_only": True,
    }


def _recovery_statistics(
    control: list[dict[str, Any]], band_n: float | None, dwell_s: float | None
) -> list[dict[str, Any]]:
    """Time to observed in-band dwell after each explicitly marked segment.

    This uses sampled evidence only; it cannot prove behavior between ticks.
    Unrecovered/censored segments retain a null recovery time. No band/dwell is
    inferred from the fake model or applied as a hardware acceptance threshold.
    """
    if band_n is None or dwell_s is None:
        return []
    reports = []
    current = None
    stable_start = None
    for row in control:
        identity = (row.get("epoch"), row.get("segment_id"))
        anchor = row.get("decision_anchor_ns")
        if type(anchor) is not int:
            continue
        if current is None or current["identity"] != identity:
            current = {
                "identity": identity,
                "event": row.get("event"),
                "start_ns": anchor,
                "recovery_time_s": None,
                "dwell_confirmed_ns": None,
            }
            reports.append(current)
            stable_start = None
        mean, target = row.get("force_mean_N"), row.get("force_target_effective_N")
        eligible = (
            row.get("phase") == "FORCE_TRACK"
            and row.get("feedback_valid")
            and not row.get("safety_override")
            and _finite(mean)
            and _finite(target)
            and abs(mean - target) <= band_n
        )
        if not eligible:
            stable_start = None
        elif current["recovery_time_s"] is None:
            if stable_start is None:
                stable_start = anchor
            if anchor - stable_start >= dwell_s * 1e9:
                current["recovery_time_s"] = (stable_start - current["start_ns"]) / 1e9
                current["dwell_confirmed_ns"] = anchor
    return reports
