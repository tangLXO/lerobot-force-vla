"""A normal connection resolves prior full-arm uncertainty through query-only proof."""

import importlib
import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import draccus
import pytest

from tests.force_control import test_default_mit_entry as entry, test_tuner_fault_recovery as recovery
from tests.force_control.hardware_helpers import bridge_packet

default_tuner = entry.default_tuner
tk_root = entry.tk_root
readonly_scope = recovery.readonly_scope
recovery_module = recovery.recovery_module


def wait_poll(app, predicate, *, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        app.poll()
        if predicate():
            return
        time.sleep(0.002)
    pytest.fail(f"GUI transition did not complete: {app.status.get()}")


@pytest.fixture
def connection(default_tuner, readonly_scope, panel, monkeypatch, tmp_path):
    app, _, _, settings_path = default_tuner
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])
    app.confirmed.set(True)
    config = app.default_full_arm_config(tmp_path / "historical-stream")
    history = {
        "errors": ["Full-arm disable has an ambiguous same-ID zero-prefix query response"],
        "config": draccus.encode(config),
        "calibration": asdict(app.calibration),
        "startup_evidence": {"complete": True},
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {
                "status": "unknown",
                "all_joints_disabled_observed": False,
                "error": "ambiguous same-ID zero-prefix query response",
                "per_joint_results": [{"name": joint.name} for joint in panel.DEFAULT_B601_JOINTS],
            },
        },
    }
    path = tmp_path / "historical-result.json"
    path.write_text(json.dumps(history), encoding="utf-8")
    before_history, before_settings = path.read_bytes(), settings_path.read_bytes()
    app.messages.put(("result", (history, path)))
    app.messages.put(("done", None))
    app.poll()
    assert not app.exit_confirmed and app.unconfirmed_full_arm
    assert app.unconfirmed_transport_closed
    monkeypatch.setattr(panel, "ReadonlyB601Transport", readonly_scope.factory)
    monkeypatch.setattr(app, "launch", panel.Tuner.launch.__get__(app, panel.Tuner))
    monitor_started = threading.Event()
    owner_started = threading.Event()
    counts = {"monitor": 0, "owner": 0}

    def monitor(stop, publish, values, calibration, cleanup_publish):
        counts["monitor"] += 1
        now = readonly_scope.clock.perf_counter_ns()
        publish(
            {
                "read_only": True,
                "joint_valid": True,
                "force_valid": True,
                "phase": "DISABLED",
                "joint_status_code": 0,
                "q_measured_deg": 0.0,
                "force_left_N": 0.1,
                "force_right_N": 0.1,
                "joint_received_ns": now,
                "force_timestamp_ns": now,
            }
        )
        monitor_started.set()
        assert stop.wait(5), "Test preview was not stopped"
        cleanup_publish({"read_only": True, "transport_closed": True, "sensor_closed": True})

    monkeypatch.setattr(panel, "monitor", monitor)

    class Live:
        def __init__(self, **kwargs):
            self.stop = threading.Event()
            self.latest = self.startup = self.applied = self.recording = self.preview = None
            self.closed = False

        def receive(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(panel, "ProcessLive", Live)

    def owner(output, *, reviewed_config, live):
        _draft, chosen = importlib.import_module("b601_supervised_bench").load_reviewed(reviewed_config)
        assert len(chosen.runtime.joint_names) == 7
        assert chosen.b601.port == config.b601.port
        counts["owner"] += 1
        owner_started.set()
        try:
            assert live.stop.wait(5), "Test owner stub was not stopped"
            return {"errors": ["synthetic owner stopped before device I/O"], "config": draccus.encode(chosen)}
        finally:
            live.close()

    monkeypatch.setattr(panel, "run_in_process", owner)
    state = SimpleNamespace(
        app=app,
        readonly=readonly_scope,
        monitor_started=monitor_started,
        owner_started=owner_started,
        counts=counts,
        config=config,
    )
    try:
        yield state
        assert path.read_bytes() == before_history
        assert settings_path.read_bytes() == before_settings
        assert all(
            int.from_bytes(packet[13:17], "little") == 0x7FF and packet[23] in (0x33, 0xCC)
            for packet in readonly_scope.serial.writes
        )
    finally:
        app.stop()
        worker = app.worker
        if worker is not None:
            worker.join(5)
            assert not worker.is_alive()
        app.poll()


def test_single_normal_connection_recovers_then_keeps_preview_ready_for_manual_start(connection):
    state = connection
    app = state.app
    app.read_button.invoke()
    wait_poll(app, lambda: app.monitoring and state.monitor_started.is_set())
    assert app.exit_confirmed and not app.unconfirmed_full_arm
    assert state.counts == {"monitor": 1, "owner": 0}
    assert app.recovery_evidence["read_only"] and not app.recovery_evidence["motion_attempted"]
    assert len(app.recovery_evidence["feedback"]["feedback"]) == 7
    assert not app.confirmed.get()
    app.confirm_check.invoke()
    app.hybrid_button.invoke()
    wait_poll(app, state.owner_started.is_set)
    assert state.counts["owner"] == 1 and app.running
    assert app.config.teacher == state.config.teacher
    assert app.config.safety == state.config.safety
    assert app.config.b601.control == state.config.b601.control


@pytest.mark.parametrize("fault", ["enabled", "missing", "stop", "port", "calibration"])
def test_incomplete_or_changed_connection_proof_never_continues_monitoring(connection, fault):
    state = connection
    app, scope = state.app, state.readonly
    entered, release = threading.Event(), threading.Event()
    if fault == "enabled":
        scope.serial.feedback_transform = lambda packet, motor_id: bridge_packet(
            motor_id, status=1 if motor_id == 3 else 0
        )
    elif fault == "missing":
        scope.serial.feedback_transform = lambda packet, motor_id: b"" if motor_id == 6 else packet
    elif fault == "port":
        app.fields["port"].set("DIFFERENT_FAKE_PORT")
    else:

        def last_mapping(packet, motor_id, register_id):
            if motor_id == 7 and register_id == 23:
                entered.set()
                assert release.wait(5)
            return packet

        scope.serial.register_transform = last_mapping
    app.read_button.invoke()
    if fault in ("stop", "calibration"):
        assert entered.wait(5)
        if fault == "stop":
            app.stop_button.invoke()
        else:
            path = app.calibration_path
            path.write_bytes(path.read_bytes() + b"\n")
        release.set()
    wait_poll(app, lambda: app.worker is None and not app.monitoring)
    assert not app.exit_confirmed
    assert state.counts == {"monitor": 0, "owner": 0}
    assert app.next_run is None and app.pending_cleanup is None
    if fault == "port":
        assert not scope.devices and not scope.serial.writes
