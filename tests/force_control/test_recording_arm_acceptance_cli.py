"""Acceptance CLI lifecycle tests with synthetic configuration and mocked hardware only."""

import hashlib
import importlib
import json
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.robots.rebot_b601_follower.arm_acceptance import (
    SingleJointAcceptanceConfig,
    command_profile_digest,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlConfig
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from lerobot.sensors import SensorSample
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("single_joint_arm_acceptance")


@pytest.fixture
def configuration(cli, tmp_path):
    calibration_path = tmp_path / "synthetic_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                joint.name: {
                    "id": joint.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -30,
                    "range_max": 30,
                }
                for joint in DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    calibration = B601ArmCalibration.load(
        calibration_path, {joint.name: joint.motor_id for joint in DEFAULT_B601_JOINTS}
    )
    profile = B601JointCommandProfile("shoulder_pan", "pos_vel", -20.0, 20.0, 0.5, velocity_deg_s=1.0)
    acceptance = SingleJointAcceptanceConfig(
        str(uuid.uuid4()),
        "synthetic_unit_test_only",
        "shoulder_pan",
        str(calibration_path),
        calibration.sha256,
        command_profile_digest(profile),
        True,
        "disable_selected_supported_arm",
        100,
        0.25,
        1,
        1,
        5,
        2000,
        True,
    )
    control = B601ControlConfig(
        (profile,),
        1,
        0.05,
        0.1,
        0.1,
        "disable_selected",
        ("shoulder_pan",),
        calibration_file=str(calibration_path),
        watchdog_ticks=2000,
    )
    sensor = X518SensorConfig(
        host="192.0.2.1",
        channels={
            "left.normal_force": X518ChannelConfig(channel=2),
            "right.normal_force": X518ChannelConfig(channel=1),
        },
        sample_rate_hz=400,
        max_age_ms=50,
        frame_features=["left.normal_force", "right.normal_force"],
    )
    config = cli.AcceptanceCLIConfig(
        "FAKE_ACCEPTANCE_CLI",
        921600,
        control,
        acceptance,
        {"gripper_force": sensor},
        50,
        14.709975,
        14.709975,
        {"human_relative100": True, "support": True, "cameras": {"front": 1, "wrist": 0}},
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(draccus.encode(config)), encoding="utf-8")
    return config, path


@pytest.fixture
def runtime(cli, monkeypatch, tmp_path):
    calls = []
    state = SimpleNamespace(device=None, sensor=None)
    source_root = tmp_path / "test_source"
    source_root.mkdir()
    (source_root / "source.py").write_text("SYNTHETIC = True\n", encoding="utf-8")
    snapshot = cli.snapshot_sources

    def snapshot_local(destination):
        calls.append("source_snapshot")
        return snapshot(destination, repository_root=source_root, source_files=("source.py",))

    monkeypatch.setattr(cli, "snapshot_sources", snapshot_local)

    class Hook:
        def __init__(self, stop):
            calls.append("hook_constructed")
            self.stop = stop

        def start(self):
            calls.append("escape_ready")

        def check(self):
            pass

        def close(self):
            calls.append("escape_closed")

    class Sensor:
        provenance = {"driver": "synthetic_test_only"}

        def __init__(self):
            self.has_resources = False
            self.lease = SimpleNamespace(error=None, start_sequence=10)
            self.sequence = 10
            self.values = {"left.normal_force": 0.2, "right.normal_force": 0.3}

        def connect(self):
            calls.append("sensor_connected")
            self.has_resources = True

        def acquire_recorder(self):
            calls.append("lease_acquired")
            return self.lease

        def read_latest_before(self, anchor, max_age_ms):
            assert "lease_acquired" in calls
            calls.append("force_read")
            return SensorSample(
                timestamp_ns=anchor - 1000,
                arrival_timestamp_ns=anchor - 500,
                sequence=self.sequence,
                values=self.values,
            )

        def disconnect(self):
            assert state.device is None or not state.device.has_resources
            calls.append("sensor_closed")
            self.has_resources = False

        def release_recorder(self, lease):
            assert state.device is None or not state.device.has_resources
            calls.append("lease_released")

    class Device:
        def __init__(self, port, specs, control, **kwargs):
            assert "force_read" in calls
            calls.append("device_constructed")
            self.has_resources = False
            self.acceptance_request = kwargs["acceptance"]
            state.device = self

    def sensors(configs):
        calls.append("sensor_constructed")
        state.sensor = Sensor()
        return {"gripper_force": state.sensor}

    def executor(device, *, stop_event, event_sink, input_guard, manual_stop_established, control_hz):
        assert manual_stop_established is True
        calls.append("execute")
        input_guard()
        device.has_resources = True
        event_sink({"stage": "startup", "transport_opened": True})
        input_guard()
        calls.append("transport_closed")
        device.has_resources = False
        result = {"complete": True, "hardware_ack": "unknown", "transport_closed": True}
        event_sink({"stage": "result", **result})
        return result

    monkeypatch.setattr(cli, "EscapeStopHook", Hook)
    monkeypatch.setattr(cli, "make_sensors_from_configs", sensors)
    monkeypatch.setattr(cli, "B601ControlBackend", Device)
    monkeypatch.setattr(cli, "execute_single_joint_acceptance", executor)
    return SimpleNamespace(calls=calls, state=state, Hook=Hook, Sensor=Sensor, Device=Device)


def test_review_does_not_construct_or_open_hardware_or_keyboard(cli, configuration, monkeypatch, tmp_path):
    _, path = configuration
    for name in ("make_sensors_from_configs", "B601ControlBackend", "EscapeStopHook"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: pytest.fail("offline review opened resources"))
    result = cli.run(path, tmp_path / "review")
    assert result["preflight_valid"] and not result["executed"] and not result["motion_attempted"]
    assert result["production_arm_authorized"] is False and result["hardware_ack"] == "unknown"
    assert not (tmp_path / "review/events.jsonl").exists()


def test_source_snapshot_copies_only_allowed_source_with_digest_mtime_and_git_state(
    cli, monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "first.py").write_text("FIRST = 1\n", encoding="utf-8")
    (source / "second.py").write_text("SECOND = 2\n", encoding="utf-8")
    (source / "private_settings.json").write_text('{"not_for_copy":true}', encoding="utf-8")
    monkeypatch.setattr(cli, "git_source_state", lambda root: {"head": "a" * 40, "dirty": True})
    output = tmp_path / "snapshot"
    manifest = cli.snapshot_sources(output, repository_root=source, source_files=("first.py", "second.py"))
    assert manifest["complete"] and manifest["git"] == {"head": "a" * 40, "dirty": True}
    assert not (output / "private_settings.json").exists()
    for entry in manifest["files"]:
        original = source / entry["path"]
        assert (output / entry["path"]).read_bytes() == original.read_bytes()
        assert entry["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert entry["mtime_ns"] == original.stat().st_mtime_ns
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["complete"]


def test_source_snapshot_rejects_same_size_change_even_if_mtime_is_restored(cli, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    watched = source / "first.py"
    watched.write_bytes(b"VALUE=1\n")
    initial = watched.stat()
    original_read = Path.read_bytes
    changed = []

    def racing_read(path):
        result = original_read(path)
        if path == watched and not changed:
            changed.append(True)
            watched.write_bytes(b"VALUE=2\n")
            os.utime(watched, ns=(initial.st_atime_ns, initial.st_mtime_ns))
        return result

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    with pytest.raises(ValueError, match="Source changed"):
        cli.snapshot_sources(tmp_path / "raced", repository_root=source, source_files=("first.py",))
    assert not (tmp_path / "raced/manifest.json").exists()


def test_source_snapshot_failure_blocks_hook_sensor_and_motor_construction(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    def fail(destination):
        raise OSError("source changed during copy")

    monkeypatch.setattr(cli, "snapshot_sources", fail)
    result = cli.run(configuration[1], tmp_path / "bad_source", execute=True)
    assert not result["executed"] and result["source_snapshot_status"] == "unavailable"
    assert "hook_constructed" not in runtime.calls and "sensor_constructed" not in runtime.calls
    assert "device_constructed" not in runtime.calls
    assert any("source changed during copy" in error for error in result["errors"])


def test_execution_orders_escape_lease_force_then_owner_and_preserves_force_receipts(
    cli, configuration, runtime, tmp_path
):
    _, path = configuration
    output = tmp_path / "execute"
    result = cli.run(path, output, execute=True)
    assert result["complete"] and result["executed"]
    assert runtime.calls.index("source_snapshot") < runtime.calls.index("hook_constructed")
    assert result["source_snapshot_status"] == "captured"
    assert (output / "source_snapshot/manifest.json").is_file()
    assert runtime.calls.index("escape_ready") < runtime.calls.index("sensor_connected")
    assert runtime.calls.index("lease_acquired") < runtime.calls.index("force_read")
    assert runtime.calls.index("force_read") < runtime.calls.index("device_constructed")
    assert runtime.calls.index("transport_closed") < runtime.calls.index("sensor_closed")
    assert runtime.calls.index("lease_released") < runtime.calls.index("sensor_closed")
    rows = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    startup = next(row for row in rows if row["stage"] == "startup")
    force = startup["last_force_guard_at_event"]
    assert force["sequence"] == 10 and force["force_left_N"] == 0.2
    assert force["timestamp_ns"] <= force["arrival_timestamp_ns"] <= force["checked_ns"]
    assert result["evidence_closed"] and result["sensor_lease_released"] and not result["cleanup_pending"]


def test_escape_hook_failure_refuses_sensor_or_motor_start(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    def unavailable(self):
        raise RuntimeError("escape unavailable")

    monkeypatch.setattr(runtime.Hook, "start", unavailable)
    result = cli.run(configuration[1], tmp_path / "no_escape", execute=True)
    assert not result["complete"] and not result["executed"]
    assert "sensor_constructed" not in runtime.calls and "device_constructed" not in runtime.calls
    assert any("escape unavailable" in error for error in result["errors"])


@pytest.mark.parametrize("failure", ["pre_lease", "overforce", "lease_fault", "future", "stale"])
def test_force_failures_prevent_motor_construction(
    cli, configuration, runtime, monkeypatch, tmp_path, failure
):
    original = runtime.Sensor.read_latest_before

    def invalid(self, anchor, max_age_ms):
        sample = original(self, anchor, max_age_ms)
        if failure == "pre_lease":
            return replace(sample, sequence=9)
        if failure == "overforce":
            return replace(sample, values={"left.normal_force": 15, "right.normal_force": 0})
        if failure == "future":
            return replace(sample, arrival_timestamp_ns=anchor + 1)
        if failure == "stale":
            return replace(sample, timestamp_ns=anchor - 51_000_000)
        self.lease.error = "required stream reconnected"
        return sample

    if failure == "lease_fault":
        original_lease = runtime.Sensor.acquire_recorder

        def bad_lease(self):
            lease = original_lease(self)
            lease.error = "required stream reconnected"
            return lease

        monkeypatch.setattr(runtime.Sensor, "acquire_recorder", bad_lease)
    else:
        monkeypatch.setattr(runtime.Sensor, "read_latest_before", invalid)
    result = cli.run(configuration[1], tmp_path / failure, execute=True)
    assert not result["complete"] and "device_constructed" not in runtime.calls
    assert "sensor_closed" in runtime.calls and "lease_released" in runtime.calls


def test_stop_file_prevents_any_motor_open(cli, configuration, runtime, tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    result = cli.run(configuration[1], tmp_path / "stopped", execute=True, stop_file=stop)
    assert not result["complete"] and "device_constructed" not in runtime.calls
    assert any("operator stop" in error for error in result["errors"])


def test_unknown_fields_and_missing_calibration_reject_before_resource_creation(
    cli, configuration, runtime, tmp_path
):
    config, path = configuration
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["bypass_force"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(Exception, match="bypass_force"):
        cli.run(path, tmp_path / "unknown", execute=True)
    assert runtime.calls == []
    raw.pop("bypass_force")
    path.write_text(json.dumps(raw), encoding="utf-8")
    Path(config.acceptance.calibration_file).unlink()
    with pytest.raises(ValueError, match="calibration"):
        cli.run(path, tmp_path / "missing", execute=True)
    assert runtime.calls == []


def test_retained_transport_close_keeps_sensor_lease_and_never_reexecutes(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    executions, closes = [], []

    def fails(device, **kwargs):
        device.has_resources = True
        executions.append("one_attempt")
        raise RuntimeError("disable attempted but transport close failed")

    def close_only(device):
        assert runtime.state.sensor.has_resources
        assert "lease_released" not in runtime.calls
        closes.append("close_only")
        if len(closes) == 1:
            raise OSError("close delayed")
        device.has_resources = False

    monkeypatch.setattr(cli, "execute_single_joint_acceptance", fails)
    monkeypatch.setattr(cli.ReadonlyB601Transport, "close", close_only)
    result = cli.run(configuration[1], tmp_path / "retained", execute=True)
    assert executions == ["one_attempt"] and closes == ["close_only", "close_only"]
    assert not result["complete"] and result["transport_closed"] and result["sensor_closed"]
    assert any("disable attempted" in error for error in result["errors"])


def test_sensor_cleanup_failure_retries_only_sensor_and_preserves_failure(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    attempts = []
    original = runtime.Sensor.disconnect

    def delayed(self):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("sensor close delayed")
        original(self)

    monkeypatch.setattr(runtime.Sensor, "disconnect", delayed)
    result = cli.run(configuration[1], tmp_path / "sensor_delayed", execute=True)
    assert len(attempts) == 2 and runtime.calls.count("execute") == 1
    assert result["sensor_closed"] and result["sensor_lease_released"] and not result["complete"]


def test_waits_for_first_post_lease_sample_without_opening_motor_early(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    original = runtime.Sensor.read_latest_before
    attempts = []

    def not_ready(self, anchor, max_age_ms):
        attempts.append(1)
        if len(attempts) <= 2:
            assert "device_constructed" not in runtime.calls
            raise RuntimeError("X518Sensor has not produced a sample yet.")
        return original(self, anchor, max_age_ms)

    monkeypatch.setattr(runtime.Sensor, "read_latest_before", not_ready)
    result = cli.run(configuration[1], tmp_path / "wait_fresh", execute=True)
    assert result["complete"] and len(attempts) >= 3


def test_writer_health_is_checked_before_force_read_or_motion(cli, configuration, runtime):
    def failed():
        raise RuntimeError("latched evidence error")

    stop = threading.Event()
    sensor = runtime.Sensor()
    guard = cli.ForceGuard(
        configuration[0], sensor, sensor.lease, stop, runtime.Hook(stop), lambda row: None, failed
    )
    with pytest.raises(RuntimeError, match="latched evidence error"):
        guard()
    assert "force_read" not in runtime.calls and "device_constructed" not in runtime.calls


def test_result_disk_failure_cannot_skip_retained_transport_or_sensor_cleanup(
    cli, configuration, runtime, monkeypatch, tmp_path
):
    attempts = []
    original_write = cli._write_json

    def write(path, result):
        if result.get("cleanup_pending"):
            raise OSError("result disk unavailable")
        original_write(path, result)

    def executor(device, **kwargs):
        device.has_resources = True
        raise RuntimeError("fault already attempted")

    def close(device):
        assert runtime.state.sensor.has_resources and "lease_released" not in runtime.calls
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("transport close pending")
        device.has_resources = False

    monkeypatch.setattr(cli, "_write_json", write)
    monkeypatch.setattr(cli, "execute_single_joint_acceptance", executor)
    monkeypatch.setattr(cli.ReadonlyB601Transport, "close", close)
    result = cli.run(configuration[1], tmp_path / "disk_cleanup", execute=True)
    assert len(attempts) == 2 and result["transport_closed"] and result["sensor_closed"]
    assert result["sensor_lease_released"] and not result["complete"]
    assert any("result disk unavailable" in error for error in result["errors"])


def test_evidence_handoff_is_bounded_nonblocking_and_overflow_latches(cli, tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, value):
            assert threading.current_thread().name == "acceptance-evidence"
            entered.set()
            release.wait(3)

        def flush(self):
            pass

    writer = cli.EvidenceWriter(SimpleNamespace(open=lambda *args, **kwargs: Handle()), capacity=1)
    writer.start()
    try:
        writer.submit({"stage": "first"})
        assert entered.wait(1)
        writer.submit({"stage": "second"})
        before = time.perf_counter()
        with pytest.raises(RuntimeError, match="overflow"):
            writer.submit({"stage": "third"})
        assert time.perf_counter() - before < 0.2
        with pytest.raises(RuntimeError, match="overflow"):
            writer.check_health()
    finally:
        release.set()
        writer.stop.set()
        writer.thread.join(2)
    assert not writer.has_resources


def test_write_force_link_uses_last_guard_before_actual_write_not_later_rx_guard(cli, tmp_path):
    writer = cli.EvidenceWriter(tmp_path / "unused.jsonl")
    writer._link_force_guards(
        {
            "stage": "force_guard",
            "force": {"guard_id": 1, "checked_ns": 1000, "timestamp_ns": 900},
        }
    )
    writer._link_force_guards(
        {
            "stage": "force_guard",
            "force": {"guard_id": 2, "checked_ns": 3000, "timestamp_ns": 2900},
        }
    )
    row = {
        "stage": "startup",
        "io": {
            "events": [
                {"kind": "command_attempt", "timestamp_ns": 2000},
                {"kind": "command_attempt", "timestamp_ns": 800},
                {"kind": "rx", "timestamp_ns": 3500},
            ]
        },
    }
    writer._link_force_guards(row)
    attempts = row["io"]["events"]
    assert attempts[0]["force_guard_before_write"]["guard_id"] == 1
    assert attempts[0]["force_guard_before_write"]["force_age_at_write_ms"] == 0.0011
    assert attempts[1]["force_guard_before_write"] is None
    assert "force_guard_before_write" not in attempts[2]


def test_guard_deduplicates_only_evidence_and_checks_force_stop_and_writer_each_call(
    cli, configuration, runtime
):
    selected, checked, rows = [], [], []
    now = time.perf_counter_ns()
    sample = SensorSample(
        timestamp_ns=now - 1000,
        arrival_timestamp_ns=now - 500,
        sequence=10,
        values={"left.normal_force": 0.2, "right.normal_force": 0.3},
    )

    def read(anchor, max_age_ms):
        selected.append(1)
        return sample

    sensor = SimpleNamespace(read_latest_before=read)
    lease = SimpleNamespace(start_sequence=10, error=None)
    stop = threading.Event()
    guard = cli.ForceGuard(
        configuration[0], sensor, lease, stop, runtime.Hook(stop), rows.append, lambda: checked.append(1)
    )
    for _ in range(5):
        guard()
    assert len(selected) == len(checked) == guard.count == 5
    assert guard.logged_count == len(rows) == 1
    stop.set()
    with pytest.raises(RuntimeError, match="operator stop"):
        guard()
    assert guard.count == 6 and len(rows) == 1


def test_escape_listener_only_interprets_escape(cli, monkeypatch):
    from lerobot.utils import keyboard_input

    callbacks = []

    class Listener:
        def __init__(self, *, on_press):
            callbacks.append(on_press)
            self.alive = False

        def start(self):
            self.alive = True

        def wait(self):
            pass

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

        def join(self, timeout):
            pass

    monkeypatch.setattr(
        keyboard_input, "keyboard", SimpleNamespace(Listener=Listener, Key=SimpleNamespace(esc="ESC"))
    )
    monkeypatch.setattr(keyboard_input, "pynput_can_capture", lambda: True)
    monkeypatch.setattr(keyboard_input, "pynput_listener_is_trusted", lambda listener: True)
    stop = threading.Event()
    hook = cli.EscapeStopHook(stop)
    hook.start()
    assert callbacks[0]("A") is None and not stop.is_set()
    assert callbacks[0]("ESC") is False and stop.is_set()
    hook.close()
