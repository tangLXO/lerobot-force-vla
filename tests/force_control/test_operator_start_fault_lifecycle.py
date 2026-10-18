"""Real panel callbacks, fake serial control, IPC failure and seven-axis recovery."""

import copy
import importlib
import json
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import draccus
import pytest

from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from tests.force_control import (
    operator_teleoperation_helpers,
    test_default_mit_entry as entry,
    test_operator_timeout_latency as latency,
    test_tuner_fault_recovery as recovery,
)
from tests.force_control.hardware_helpers import bridge_packet

default_tuner = entry.default_tuner
tk_root = entry.tk_root
readonly_scope = recovery.readonly_scope
recovery_module = recovery.recovery_module


@pytest.mark.parametrize("unknown_cleanup", [False, True])
def test_panel_to_runtime_fault_ipc_and_readonly_recovery_without_hardware(
    default_tuner, readonly_scope, panel, monkeypatch, tmp_path, unknown_cleanup
):
    app, _, _, settings_path = default_tuner
    original_settings = settings_path.read_bytes()
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])
    live = recovery.make_ipc(panel, None)
    live.parameters.close_error = False
    live._latest = None
    monkeypatch.setattr(panel, "ProcessLive", lambda **kwargs: live)
    launched = []

    def launch(function):
        launched.append(function)
        app.worker = SimpleNamespace(is_alive=lambda: True)

    monkeypatch.setattr(app, "launch", launch)
    app.monitor_session = "synthetic-monitor"
    app.monitoring = True
    app.monitor_port = app.fields["port"].get()
    app.preview_latest = {
        "joint_received_ns": panel.time.perf_counter_ns(),
        "phase": "DISABLED",
    }
    app.confirmed.set(True)
    app.hybrid_button.invoke()
    assert app.running and callable(app.next_run), app.status.get()
    selected = draccus.encode(app.config)
    app.monitoring = False
    app.messages.put(
        (
            "preview_cleanup",
            {"monitor_session": app.monitor_session, "transport_closed": True, "sensor_closed": True},
        )
    )
    app.messages.put(("done", None))
    app.poll()
    assert len(launched) == 1 and app.next_run is None
    captured = {}

    def run_owner(output, *, reviewed_config, live):
        _draft, config = importlib.import_module("b601_supervised_bench").load_reviewed(reviewed_config)
        assert draccus.encode(config) == selected
        files = SimpleNamespace(
            operator=config.b601.operator_teleoperation,
            configuration=config.b601.operator_teleoperation,
            control=config.b601.control,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(latency, "make_operator_files", lambda path: files)
            patch.setattr(operator_teleoperation_helpers, "Clock", latency.ProfiledWorkClock)
            patch.setattr(
                latency, "ForceTeacher", lambda *_args: ForceTeacher(config.teacher, config.close_sign)
            )
            patch.setattr(
                latency, "SafetySupervisor", lambda *_args: SafetySupervisor(config.safety, config.close_sign)
            )
            patch.setattr(latency, "RuntimeConfig", lambda **kwargs: config.runtime)
            with latency.latency_scope(tmp_path / "fake-owner", patch, fast_timeout=True) as rig:
                assert rig.scope.device.config == config.b601.control
                session = rig.scope.device._full_validation_session
                guard, binding = session.wait_guard, session.require_binding

                def charged_guard():
                    rig.scope.clock.now += 50_000
                    return guard()

                def charged_binding(*args, **kwargs):
                    rig.scope.clock.now += 200_000
                    return binding(*args, **kwargs)

                patch.setattr(session, "wait_guard", charged_guard)
                patch.setattr(session, "require_binding", charged_binding)
                rig.runtime.trace.rows = deque(maxlen=8)
                assert rig.tick()["phase"] == "DISABLED"
                assert rig.tick("arm")["phase"] == "ARMED"
                assert rig.tick("start")["phase"] == "FOLLOW"
                for _ in range(100):
                    before = len(rig.serial.writes)
                    row = rig.tick()
                    assert row["phase"] == "FOLLOW" and row["send_status"] == "submitted"
                    assert len(rig.serial.writes[before:]) == 7
                captured["successful_frames"] = 100
                if unknown_cleanup:
                    rig.serial.failed_disables.add(3)

                def disconnected():
                    raise OSError("synthetic force sensor I/O disconnected")

                patch.setattr(rig.runtime.sensor, "_check_recorder_fault", disconnected)
                fault = rig.tick()
                assert fault["phase"] == "FAULT"
                captured["fault"] = copy.deepcopy(rig.scope.device.fault_receipt)
                live.latest = fault
                live.startup = rig.scope.device.startup_receipt
            readonly_scope.clock.now = rig.scope.clock.now + 1_000_000
            result = {
                "errors": [fault["failure_reason"]],
                "config": selected,
                "calibration": asdict(B601GripperCalibration.load(config.b601.control.calibration_file)),
                "startup_evidence": copy.deepcopy(rig.scope.device.startup_receipt),
                "shutdown_evidence": copy.deepcopy(rig.runtime.shutdown_evidence),
                "hardware_ack": "unknown",
            }
        assert len(captured["fault"]["per_joint_results"]) == 7
        assert captured["fault"]["all_joints_disabled_observed"] is not unknown_cleanup
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        captured["result"] = result
        captured["result_path"] = output / "result.json"
        # Exercise the real IPC EOF and close paths before the GUI receives result/done.
        live.telemetry.error = EOFError("synthetic telemetry EOF after owner cleanup")
        live.receive()
        assert live.stop.is_set() and live.ipc_errors
        live.close()
        return result

    monkeypatch.setattr(panel, "run_in_process", run_owner)
    launched.pop(0)()
    app.messages.put(("done", None))
    app.poll()
    assert captured["successful_frames"] == 100
    assert not app.running and app.worker is None and app.live is None
    assert app.exit_confirmed is not unknown_cleanup
    assert not app.ui_errors
    assert draccus.encode(app.config) == selected
    assert settings_path.read_bytes() == original_settings
    result_bytes = captured["result_path"].read_bytes()
    assert str(app.stop_button.cget("state")) == "normal"
    assert str(app.read_button.cget("state")) == "normal"
    app.stop_button.invoke()
    assert app.monitor_stop.is_set()

    if unknown_cleanup:
        assert app.full_arm_recovery_ready()
        assert str(app.full_arm_recovery_button.cget("state")) == "normal"
        monkeypatch.setattr(panel, "ReadonlyB601Transport", readonly_scope.factory)
        readonly_scope.serial.feedback_transform = lambda packet, motor_id: bridge_packet(
            motor_id, status=1 if motor_id == 3 else 0
        )
        app.hybrid_button.invoke()
        assert not app.running and not app.exit_confirmed
        assert len(launched) == 1
        launched.pop(0)()
        app.messages.put(("done", None))
        app.poll()
        assert not app.exit_confirmed and app.full_arm_recovery_ready()
        # The fixture now represents a later external disabled observation.
        readonly_scope.serial.feedback_transform = lambda packet, motor_id: bridge_packet(motor_id, status=0)
        app.full_arm_recovery_button.invoke()
        launched.pop(0)()
        app.messages.put(("done", None))
        app.poll()
        assert app.exit_confirmed and not app.confirmed.get()
        assert not app.running and app.next_run is None
        assert app.recovery_evidence["read_only"] and not app.recovery_evidence["motion_attempted"]
        assert len(app.recovery_evidence["feedback"]["feedback"]) == 7
        assert all(
            int.from_bytes(packet[13:17], "little") == 0x7FF for packet in readonly_scope.serial.writes
        )
        assert captured["result_path"].read_bytes() == result_bytes

    app.read_button.invoke()
    assert app.monitoring and launched
    app.stop_button.invoke()
    assert app.monitor_stop.is_set()
    # The captured preview has not run and therefore owns no device resources.
    app.monitoring = False
    app.worker = None
    launched.clear()
    app.root.tk.call(app.root.protocol("WM_DELETE_WINDOW"))
    assert app.closing and app.monitor_stop.is_set()
    assert settings_path.read_bytes() == original_settings
