#!/usr/bin/env python
"""Review or execute one explicitly supported B601 arm microstep; production gate stays closed."""

# SI unit suffixes match the established force configuration interface.
# ruff: noqa: N815

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus

from lerobot.robots.rebot_b601_follower.arm_acceptance import (
    SingleJointAcceptanceConfig,
    execute_single_joint_acceptance,
)
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlBackend, B601ControlConfig
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.sensors import SensorConfig, make_sensors_from_configs
from lerobot.sensors.synchronization import SensorDataUnavailableError
from lerobot.sensors.x518 import X518SensorConfig

FORCE_FEATURES = ("left.normal_force", "right.normal_force")
SOURCE_FILES = (
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


def git_source_state(repository_root):
    """Read Git identity without refreshing the index or including private file contents."""
    result = {"head": None, "dirty": None}
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise FileNotFoundError("Git executable is unavailable on PATH")
        head = subprocess.run(
            [git_executable, "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            [git_executable, "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode == 0:
            result["head"] = head.stdout.strip()
        if status.returncode == 0:
            result["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def snapshot_sources(destination, *, repository_root=None, source_files=None):
    """Copy a source-only allowlist and reject changes throughout the complete copy pass."""
    repository_root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    git = git_source_state(repository_root)
    captured = []

    def stamp(path):
        item = path.stat()
        return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns

    for relative in SOURCE_FILES if source_files is None else source_files:
        relative = Path(relative)
        source = (repository_root / relative).resolve()
        if not source.is_relative_to(repository_root) or relative.is_absolute() or source.suffix != ".py":
            raise ValueError("Source snapshot only permits allowlisted repository Python files")
        before = stamp(source)
        data = source.read_bytes()
        if stamp(source) != before:
            raise ValueError(f"Source changed while snapshotting: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise OSError(f"Source snapshot copy differs: {relative}")
        captured.append(
            (
                source,
                before,
                {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": len(data),
                    "mtime_ns": before[3],
                },
            )
        )
    # Detect an earlier source changing while a later source was being copied,
    # including same-size changes that restore the original timestamp.
    for source, before, entry in captured:
        if stamp(source) != before or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source changed while snapshotting: {entry['path']}")
        if stamp(source) != before:
            raise ValueError(f"Source changed while snapshotting: {entry['path']}")
    manifest = {
        "schema": "b601_acceptance_source_snapshot_v1",
        "complete": True,
        "capture_started_ns": started,
        "capture_completed_ns": time.perf_counter_ns(),
        "files": [entry for _, _, entry in captured],
        "git": git,
        "scope": "allowlisted Python source only; no calibration or local parameter files",
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class AcceptanceCLIConfig:
    """Visible, typed authorization and genuine force-input configuration."""

    port: str
    baudrate: int
    control: B601ControlConfig
    acceptance: SingleJointAcceptanceConfig
    sensors: dict[str, SensorConfig]
    force_max_age_ms: float
    hard_force_limit_left_N: float
    hard_force_limit_right_N: float
    authorization: dict
    control_hz: float = 20.0

    def validate(self):
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("port must be explicit")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        self.control.validate()
        calibration = self.acceptance.validate_binding(DEFAULT_B601_JOINTS, self.control)
        for name in ("force_max_age_ms", "hard_force_limit_left_N", "hard_force_limit_right_N", "control_hz"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be explicit, finite and positive")
        if self.force_max_age_ms > 50 or not 1 <= self.control_hz <= 50:
            raise ValueError("Acceptance force age must be <=50 ms and control_hz within 1..50")
        if set(self.sensors) != {"gripper_force"}:
            raise ValueError("Exactly one required gripper_force X518 stream must be configured")
        sensor = self.sensors["gripper_force"]
        if (
            not isinstance(sensor, X518SensorConfig)
            or sensor.required is not True
            or tuple(sensor.channels) != FORCE_FEATURES
            or sensor.frame_features not in (None, list(FORCE_FEATURES))
        ):
            raise ValueError("Acceptance force must be required real X518 with ordered left/right features")
        if not isinstance(self.authorization, dict) or not self.authorization:
            raise ValueError("Human authorization evidence must be preserved explicitly")
        json.dumps(self.authorization, allow_nan=False)
        return calibration


class EscapeStopHook:
    """Listen for Escape only; other keys are neither recorded nor interpreted."""

    def __init__(self, stop_event):
        self.stop_event, self.listener = stop_event, None

    def start(self):
        from lerobot.utils.keyboard_input import (
            keyboard,
            pynput_can_capture,
            pynput_listener_is_trusted,
        )

        if keyboard is None or not pynput_can_capture():
            raise RuntimeError("A global Escape hook is required; install/enable the pynput backend")

        def on_press(key):
            if key == keyboard.Key.esc:
                self.stop_event.set()
                return False
            return None

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()
        ready = threading.Event()

        def wait_ready():
            self.listener.wait()
            ready.set()

        threading.Thread(target=wait_ready, name="acceptance-escape-ready", daemon=True).start()
        if not ready.wait(2) or not pynput_listener_is_trusted(self.listener) or not self.listener.is_alive():
            self.close()
            raise RuntimeError("Global Escape hook did not become ready; refusing hardware startup")

    def check(self):
        if self.listener is None or not self.listener.is_alive():
            raise RuntimeError("Global Escape hook is unavailable")

    def close(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener.join(timeout=2)
            if self.listener.is_alive():
                raise RuntimeError("Global Escape hook shutdown is unconfirmed")


class FreshForceUnavailableError(RuntimeError):
    """Only the initial post-lease sample wait may retry missing force input."""


class EvidenceWriter:
    """Disk serialization never executes on the motor owner; failures latch closed."""

    def __init__(self, path, capacity=1024):
        self.path = path
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.overflowed = False
        self.rows = 0
        self.force_guards = deque(maxlen=1024)
        self.thread = threading.Thread(target=self._run, name="acceptance-evidence", daemon=False)

    def start(self):
        self.thread.start()
        if not self.ready.wait(2):
            self.stop.set()
            raise TimeoutError("Acceptance evidence writer did not become ready")
        self.check_health()

    @property
    def has_resources(self):
        return self.thread.is_alive()

    def _run(self):
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                self.ready.set()
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        row = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self._link_force_guards(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    self.rows += 1
        except BaseException as exc:
            self.error = exc
        finally:
            self.ready.set()

    def _link_force_guards(self, row):
        """Associate by actual write time; a post-write guard cannot label an earlier write."""
        if row.get("stage") == "force_guard":
            self.force_guards.append(row["force"])
        for event in (row.get("io") or {}).get("events", []):
            if event.get("kind") != "command_attempt":
                continue
            timestamp = event.get("timestamp_ns")
            selected = next(
                (guard for guard in reversed(self.force_guards) if guard["checked_ns"] <= timestamp),
                None,
            )
            event["force_guard_before_write"] = dict(selected) if selected is not None else None
            if selected is not None:
                event["force_guard_before_write"]["force_age_at_write_ms"] = (
                    timestamp - selected["timestamp_ns"]
                ) / 1e6

    def check_health(self):
        if self.error is not None:
            raise RuntimeError(f"Acceptance evidence writer failed: {self.error}") from self.error
        if self.overflowed:
            raise RuntimeError("Acceptance evidence queue overflowed")
        if not self.thread.is_alive() and not self.stop.is_set():
            raise RuntimeError("Acceptance evidence worker exited unexpectedly")

    def submit(self, row):
        self.check_health()
        try:
            self.queue.put_nowait(copy.deepcopy(row))
        except queue.Full as exc:
            self.overflowed = True
            raise RuntimeError("Acceptance evidence queue overflowed") from exc

    def close(self, timeout_s=0.1):
        self.stop.set()
        self.thread.join(timeout=timeout_s)
        if self.thread.is_alive():
            raise TimeoutError("Acceptance evidence writer shutdown unconfirmed")
        self.check_health()


class ForceGuard:
    """Pin one causal force sample before motion; never block emergency disposition."""

    def __init__(self, config, sensor, lease, stop_event, hook, sink, evidence_health, stop_file=None):
        self.config, self.sensor, self.lease = config, sensor, lease
        self.stop_event, self.hook, self.sink = stop_event, hook, sink
        self.evidence_health = evidence_health
        self.stop_file = stop_file
        self.latest = None
        self.count = 0
        self.logged_sequence = None
        self.logged_count = 0

    def __call__(self):
        self.count += 1
        if self.stop_file is not None and self.stop_file.exists():
            self.stop_event.set()
        if self.stop_event.is_set():
            raise RuntimeError("Acceptance operator stop requested")
        self.hook.check()
        self.evidence_health()
        if self.lease.error is not None:
            raise RuntimeError(f"Required force Sensor lease failed: {self.lease.error}")
        anchor = time.perf_counter_ns()
        try:
            sample = self.sensor.read_latest_before(anchor, max_age_ms=self.config.force_max_age_ms)
        except (SensorDataUnavailableError, TimeoutError) as exc:
            raise FreshForceUnavailableError(str(exc)) from exc
        except RuntimeError as exc:
            if "has not produced a sample yet" in str(exc):
                raise FreshForceUnavailableError(str(exc)) from exc
            raise
        if (
            sample.is_valid is not True
            or sample.error is not None
            or type(sample.sequence) is not int
            or sample.sequence < self.lease.start_sequence
            or type(sample.timestamp_ns) is not int
            or type(sample.arrival_timestamp_ns) is not int
            or not 0 < sample.timestamp_ns <= sample.arrival_timestamp_ns <= anchor
            or anchor - sample.timestamp_ns > self.config.force_max_age_ms * 1e6
        ):
            raise RuntimeError("Force guard rejected stale, noncausal or pre-lease input")
        values = [float(sample.values[key]) for key in FORCE_FEATURES]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("Force guard rejected nonfinite force input")
        self.latest = {
            "guard_id": self.count,
            "checked_ns": anchor,
            "sequence": sample.sequence,
            "timestamp_ns": sample.timestamp_ns,
            "arrival_timestamp_ns": sample.arrival_timestamp_ns,
            "force_left_N": values[0],
            "force_right_N": values[1],
            "status": sample.status,
            "hardware_sequence": sample.hardware_sequence,
        }
        # Every callback still selects/checks force and priority stop. Repeated
        # reads of one immutable SensorSample need only one detached evidence row.
        if sample.sequence != self.logged_sequence:
            self.sink({"stage": "force_guard", "force": self.latest})
            self.logged_sequence = sample.sequence
            self.logged_count += 1
        for side, value, limit in (
            ("left", values[0], self.config.hard_force_limit_left_N),
            ("right", values[1], self.config.hard_force_limit_right_N),
        ):
            if value >= limit:
                raise RuntimeError(f"{side}_force_hard_limit: {value:g} >= {limit:g} N")

    def wait_ready(self, timeout_s):
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                self()
                return
            except FreshForceUnavailableError:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("No fresh post-lease force sample before startup") from None
                time.sleep(0.005)


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(config_file, output, *, execute=False, stop_file=None):
    """Validate offline by default; execute exactly once after the manual stop channel exists."""
    config_file, output = Path(config_file).resolve(), Path(output).resolve()
    raw = config_file.read_bytes()
    config = draccus.decode(AcceptanceCLIConfig, json.loads(raw))
    calibration = config.validate()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "b601_single_joint_acceptance_cli_v1",
        "executed": False,
        "complete": False,
        "hardware_ack": "unknown",
        "production_arm_authorized": False,
        "transport_closed": True,
        "sensor_closed": True,
        "sensor_lease_released": True,
        "manual_stop_ready": False,
        "escape_hook_closed": True,
        "source_snapshot_status": "unavailable",
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration": asdict(calibration),
        "errors": [],
    }
    _write_json(output / "reviewed_config.json", draccus.encode(config))
    _write_json(output / "result.json", result)
    if not execute:
        result.update(preflight_valid=True, motion_attempted=False)
        _write_json(output / "result.json", result)
        return result
    try:
        snapshot = snapshot_sources(output / "source_snapshot")
        result.update(
            source_snapshot_status="captured",
            source_snapshot_manifest="source_snapshot/manifest.json",
            source_git=snapshot["git"],
        )
        _write_json(output / "result.json", result)
    except BaseException as exc:
        result["errors"].append(f"Source snapshot: {type(exc).__name__}: {exc}")
        _write_json(output / "result.json", result)
        return result
    stop_event = threading.Event()
    hook = EscapeStopHook(stop_event)
    sensor = device = lease = guard = None
    writer = EvidenceWriter(output / "events.jsonl")

    def sink(event):
        row = {"recorded_ns": time.perf_counter_ns(), **event}
        if guard is not None and guard.latest is not None and event.get("stage") != "force_guard":
            row["last_force_guard_at_event"] = dict(guard.latest)
        if event.get("stage") == "result":
            result["backend_result"] = {key: value for key, value in event.items() if key != "stage"}
        writer.submit(row)

    def persist_result():
        try:
            _write_json(output / "result.json", result)
        except BaseException as exc:
            message = f"Result write: {type(exc).__name__}: {exc}"
            if message not in result["errors"]:
                result["errors"].append(message)

    def pause_cleanup():
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            stop_event.set()

    try:
        writer.start()
        result["escape_hook_closed"] = False
        hook.start()
        result["manual_stop_ready"] = True
        sink(
            {
                "stage": "manual_stop_ready",
                "escape_only": True,
                "stop_file": str(stop_file) if stop_file else None,
            }
        )
        sensor = make_sensors_from_configs(config.sensors)["gripper_force"]
        sensor.connect()
        result["sensor_closed"] = False
        lease = sensor.acquire_recorder()
        result["sensor_lease_released"] = False
        guard = ForceGuard(
            config,
            sensor,
            lease,
            stop_event,
            hook,
            sink,
            writer.check_health,
            Path(stop_file) if stop_file else None,
        )
        guard.wait_ready(min(5.0, config.sensors["gripper_force"].startup_timeout_s))
        sink(
            {
                "stage": "force_ready",
                "sensor_provenance": sensor.provenance,
                "lease_start_sequence": lease.start_sequence,
            }
        )
        device = B601ControlBackend(
            config.port,
            DEFAULT_B601_JOINTS,
            config.control,
            baudrate=config.baudrate,
            acceptance=config.acceptance,
        )
        result["executed"] = True
        result["backend_result"] = execute_single_joint_acceptance(
            device,
            stop_event=stop_event,
            event_sink=sink,
            input_guard=guard,
            control_hz=config.control_hz,
            manual_stop_established=True,
        )
    except BaseException as exc:
        stop_event.set()
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # The backend already attempted its one fault/disable disposition. Any
        # retained cleanup below closes transport only, never resubmits a command.
        cleanup_stage = "transport"
        while True:
            try:
                cleanup_stage = "transport"
                if device is not None and device.has_resources:
                    result.update(cleanup_pending=True, transport_closed=False)
                    ReadonlyB601Transport.close(device)
                    continue
                result["transport_closed"] = True
                cleanup_stage = "Sensor lease"
                if lease is not None:
                    sensor.release_recorder(lease)
                    lease = None
                    result["sensor_lease_released"] = True
                    continue
                cleanup_stage = "Sensor"
                if sensor is not None and sensor.has_resources:
                    sensor.disconnect()
                    continue
                result["sensor_closed"] = True
                break
            except BaseException as exc:
                message = f"{cleanup_stage} cleanup: {type(exc).__name__}: {exc}"
                if message not in result["errors"]:
                    result["errors"].append(message)
                result["cleanup_pending"] = True
                persist_result()
                pause_cleanup()
        try:
            hook.close()
            result["escape_hook_closed"] = True
        except BaseException as exc:
            result["errors"].append(f"Escape cleanup: {type(exc).__name__}: {exc}")
        while True:
            try:
                writer.close()
                break
            except BaseException as exc:
                message = f"Evidence cleanup: {type(exc).__name__}: {exc}"
                if message not in result["errors"]:
                    result["errors"].append(message)
                if not writer.has_resources:
                    break
                result["cleanup_pending"] = True
                persist_result()
                pause_cleanup()
        result["evidence_rows"] = writer.rows
        result["evidence_closed"] = not writer.has_resources
        result["force_guard_checks"] = guard.count if guard is not None else 0
        result["force_guard_samples_logged"] = guard.logged_count if guard is not None else 0
        result["force_guard_logging"] = (
            "one row per selected Sensor sequence; all callbacks still check inputs"
        )
        result["cleanup_pending"] = False
        result["complete"] = bool(
            result.get("backend_result", {}).get("complete")
            and not result["errors"]
            and result["transport_closed"]
            and result["sensor_closed"]
            and result["sensor_lease_released"]
            and result["escape_hook_closed"]
            and result["evidence_closed"]
        )
        persist_result()
        if result["errors"]:
            result["complete"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args(argv)
    result = run(args.config, args.output, execute=args.execute, stop_file=args.stop_file)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("preflight_valid") or result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
