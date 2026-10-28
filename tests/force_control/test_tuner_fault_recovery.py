"""UI fault recovery and seven-axis read-only verification use fake transports only."""

import importlib
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.robots.rebot_b601_follower import readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from tests.force_control import test_default_mit_entry as default_entry
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet

default_tuner = default_entry.default_tuner
tk_root = default_entry.tk_root


@pytest.fixture(autouse=True)
def isolated_preview_ownership(panel, monkeypatch):
    """Each fake UI starts with no retained transport from another test."""
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])


@pytest.fixture
def recovery_module(panel):
    return importlib.import_module("b601_full_arm_recovery")


@pytest.fixture
def readonly_scope(panel, recovery_module, monkeypatch, tmp_path):
    path = tmp_path / "all-joint-calibration.json"
    path.write_text(
        json.dumps(
            {
                spec.name: {
                    "id": spec.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -40,
                    "range_max": 40,
                }
                for spec in readonly.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    calibration = B601ArmCalibration.load(
        path, {spec.name: spec.motor_id for spec in readonly.DEFAULT_B601_JOINTS}
    )
    clock = Clock()
    serial = FakeSerial(clock)
    monkeypatch.setattr(readonly.time, "perf_counter_ns", clock.perf_counter_ns)
    monkeypatch.setattr(readonly.serial, "Serial", lambda **kwargs: serial)
    devices = []

    def factory(port, specs, **kwargs):
        assert tuple(specs) == readonly.DEFAULT_B601_JOINTS
        device = readonly.ReadonlyB601Transport(port, specs, **kwargs, lock_root=tmp_path / "locks")
        devices.append(device)
        return device

    scope = SimpleNamespace(
        calibration=calibration,
        clock=clock,
        serial=serial,
        devices=devices,
        factory=factory,
        stop=threading.Event(),
        retained=[],
    )
    yield scope
    serial.close_fails = False
    for device in devices:
        if device.has_resources:
            device.close()


def inspect(scope, module, calibration=None):
    return module.query_disabled_arm(
        "FAKE_FOLLOWER",
        calibration or scope.calibration,
        baudrate=921600,
        mapping_budget_s=1,
        feedback_budget_s=0.05,
        max_age_s=0.15,
        stop=scope.stop,
        transport_factory=scope.factory,
        retain=scope.retained.append,
    )


def assert_query_only(serial):
    assert serial.writes
    for packet in serial.writes:
        assert len(packet) == 30
        assert int.from_bytes(packet[13:17], "little") == 0x7FF
        assert packet[23] in (0x33, 0xCC)


def test_seven_axis_recovery_uses_only_queries_and_closes_owner(readonly_scope, recovery_module):
    scope = readonly_scope
    receipt = inspect(scope, recovery_module)
    assert receipt["complete"] and receipt["all_joints_disabled_observed"]
    assert receipt["transport_closed"] and not scope.devices[0].has_resources
    assert receipt["hardware_ack"] == "unknown" and not receipt["hardware_stop_confirmed"]
    assert not receipt["motion_attempted"] and not receipt["parameter_writes"]
    assert len(receipt["feedback"]["feedback"]) == 7
    assert len(scope.serial.writes) == 28  # 21 mapping reads and seven feedback queries.
    assert_query_only(scope.serial)


def test_recovery_cannot_query_while_another_owner_holds_the_bus(readonly_scope, recovery_module):
    scope = readonly_scope
    owner = scope.factory("FAKE_FOLLOWER", readonly.DEFAULT_B601_JOINTS, baudrate=921600)
    owner.open()
    receipt = inspect(scope, recovery_module)
    assert not receipt["complete"] and not receipt["all_joints_disabled_observed"]
    assert "already has an owner" in "; ".join(receipt["errors"])
    assert scope.serial.writes == []
    assert owner.has_resources  # Recovery cannot close or steal the other owner's port.


@pytest.mark.parametrize("fault", ["enabled", "bounds", "missing", "stale", "calibration", "stop", "close"])
def test_failed_seven_axis_observation_never_unlocks(readonly_scope, recovery_module, fault):
    scope = readonly_scope
    if fault == "enabled":
        scope.serial.feedback_transform = lambda packet, motor_id: bridge_packet(
            motor_id, status=1 if motor_id == 3 else 0
        )
    elif fault == "bounds":
        scope.serial.feedback_transform = lambda packet, motor_id: bridge_packet(
            motor_id, position_uint=50000
        )
    elif fault == "missing":
        scope.serial.feedback_transform = lambda packet, motor_id: b"" if motor_id == 6 else packet
    elif fault == "stale":
        original = scope.factory

        def delayed_factory(*args, **kwargs):
            device = original(*args, **kwargs)
            read = device.read_feedback

            def delayed_feedback(**options):
                batch = read(**options)
                scope.clock.now += 1_000_000_000
                return batch

            device.read_feedback = delayed_feedback
            return device

        scope.factory = delayed_factory
    elif fault in ("calibration", "stop"):

        def change_during_mapping(packet, motor_id, rid):
            if motor_id == 7 and rid == 23:
                if fault == "stop":
                    scope.stop.set()
                else:
                    Path(scope.calibration.path).write_text("{}", encoding="utf-8")
            return packet

        scope.serial.register_transform = change_during_mapping
    else:
        scope.serial.close_fails = True
    receipt = inspect(scope, recovery_module)
    assert not receipt["complete"] and receipt["errors"]
    assert receipt["hardware_ack"] == "unknown"
    assert receipt["transport_closed"] is (fault != "close")
    assert bool(scope.retained) is (fault == "close")
    assert_query_only(scope.serial)


class FakeChannel:
    def __init__(self, error=None, close_error=False):
        self.error = error
        self.close_error = close_error
        self.get_calls = self.close_calls = self.cancel_calls = 0
        self.put_calls = 0

    def get_nowait(self):
        self.get_calls += 1
        if self.error is not None:
            raise self.error
        raise queue.Empty

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise OSError("injected channel close failure")

    def cancel_join_thread(self):
        self.cancel_calls += 1

    def put_nowait(self, value):
        self.put_calls += 1
        if self.error is not None:
            raise self.error

    def get(self, **kwargs):
        return self.get_nowait()


def make_ipc(panel, error):
    module = importlib.import_module("b601_tuner_process")
    live = module.ProcessLive.__new__(module.ProcessLive)
    live.closed = False
    live.finishing = False
    live.poll_s = 0.01
    live._receive_failed = False
    live.ipc_errors = []
    live.stop = threading.Event()
    live._latest = {"phase": "old_snapshot"}
    live._applied = live._startup = live._recording = live._preview = None
    live.telemetry = FakeChannel(error)
    live.parameters, live.events, live.recording_events = (
        FakeChannel(close_error=True),
        FakeChannel(),
        FakeChannel(),
    )
    return live


@pytest.mark.parametrize(
    "error", [EOFError("pipe ended"), BrokenPipeError("pipe broken"), OSError("bad handle")]
)
def test_ipc_pipe_failure_requests_stop_and_close_still_visits_every_queue(panel, error):
    live = make_ipc(panel, error)
    live.receive()
    assert live.stop.is_set() and live._latest == {"phase": "old_snapshot"}
    assert live.ipc_errors and type(error).__name__ in live.ipc_errors[0]
    live.receive()
    assert live.telemetry.get_calls == 1
    live.close()
    assert live.closed
    for channel in (live.parameters, live.events, live.recording_events, live.telemetry):
        assert channel.close_calls == channel.cancel_calls == 1
    live.close()
    assert live.parameters.close_calls == 1
    assert any("parameters.close" in message for message in live.ipc_errors)


@pytest.mark.parametrize("operation", ["publish", "submit", "take"])
@pytest.mark.parametrize(
    "error",
    [
        EOFError("pipe ended"),
        BrokenPipeError("pipe broken"),
        OSError("bad handle"),
        ValueError("closed queue"),
    ],
)
def test_broken_ipc_operations_request_stop_and_raise_visible_runtime_error(panel, operation, error):
    live = make_ipc(panel, None)
    if operation == "publish":
        live.telemetry.error = error
        action = live._publish
    else:
        live.events.error = error
        action = (lambda: live.submit("freeze")) if operation == "submit" else live.take
    with pytest.raises(RuntimeError, match="控制进程通信失败"):
        action()
    assert live.stop.is_set()
    assert any(type(error).__name__ in message for message in live.ipc_errors)


def test_full_empty_and_user_validation_errors_keep_existing_mailbox_semantics(panel):
    live = make_ipc(panel, None)
    live.telemetry.error = queue.Full()
    live._publish()
    assert not live.stop.is_set()
    assert live.take() is None
    assert not live.stop.is_set()
    live.events.error = queue.Full()
    with pytest.raises(queue.Full):
        live.submit("freeze")
    assert not live.stop.is_set()
    with pytest.raises(ValueError, match="目标力"):
        live.submit("target", -1)
    assert not live.stop.is_set()


def test_broken_cleanup_telemetry_cannot_abort_retained_owner_finalization(panel, monkeypatch, tmp_path):
    bridge = importlib.import_module("b601_tuner_process")
    retained = [object()]
    monkeypatch.setattr(bridge.common_runner, "_UNSTOPPED_BENCHES", retained)
    finalized = []

    def finalize():
        finalized.append(True)
        if len(finalized) == 1:
            return []
        retained.clear()
        return [{"shutdown_evidence": {"transport_closed": True}, "cleanup_errors": []}]

    monkeypatch.setattr(bridge.common_runner, "finalize_retained_benches", finalize)
    monkeypatch.setattr(bridge, "run", lambda *args, **kwargs: {"errors": ["original cleanup failure"]})
    monkeypatch.setattr(bridge.time, "sleep", lambda seconds: None)
    live = make_ipc(panel, BrokenPipeError("cleanup telemetry broken"))
    results = queue.Queue()
    bridge.control_worker(tmp_path, None, live, results)
    success, result = results.get_nowait()
    assert success and result["shutdown_evidence"]["transport_closed"]
    assert len(finalized) == 2 and not retained
    assert live.stop.is_set()
    assert any("cleanup_publish" in message for message in live.ipc_errors)


@pytest.mark.parametrize("eof", [False, True])
def test_result_pipe_and_final_queue_close_errors_do_not_skip_owner_join(panel, monkeypatch, tmp_path, eof):
    bridge = importlib.import_module("b601_tuner_process")
    received = {"shutdown_evidence": {"transport_closed": True}, "errors": []}
    operations = []

    class ResultQueue:
        def get(self, **kwargs):
            if eof:
                raise EOFError("result pipe ended")
            return True, received

        def close(self):
            operations.append("result.close")
            raise OSError("result queue close failed")

        def cancel_join_thread(self):
            operations.append("result.cancel")
            raise ValueError("result queue already closed")

    class Process:
        alive = False
        exitcode = 0

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, **kwargs):
            operations.append("join")
            self.alive = False

    process, result_queue = Process(), ResultQueue()
    context = SimpleNamespace(Queue=lambda **kwargs: result_queue, Process=lambda **kwargs: process)
    monkeypatch.setattr(bridge.mp, "get_context", lambda mode: context)
    live = make_ipc(panel, None)
    if eof:
        with pytest.raises(RuntimeError, match="result_receive"):
            bridge.run_in_process(tmp_path, reviewed_config=None, live=live)
    else:
        assert bridge.run_in_process(tmp_path, reviewed_config=None, live=live) is received
    assert not process.alive and "join" in operations
    assert operations[-2:] == ["result.close", "result.cancel"]
    assert live.closed and live.stop.is_set()
    assert any("result_queue.close" in message for message in live.ipc_errors)


def test_poll_survives_pipe_error_drains_done_and_keeps_unknown_exit_gate(
    default_tuner, panel, monkeypatch, tmp_path
):
    app, _, _, _ = default_tuner
    app.confirmed.set(True)
    app.config = app.default_full_arm_config(tmp_path / "trace")
    app.running = True
    live = SimpleNamespace(
        receive=lambda: (_ for _ in ()).throw(EOFError("injected broken telemetry")),
        stop=threading.Event(),
        closed=True,
        latest=None,
        startup=None,
        applied=None,
    )
    app.live = live
    result = {
        "errors": ["active feedback timeout"],
        "config": draccus.encode(app.config),
        "calibration": asdict(app.calibration),
        "startup_evidence": {"complete": True},
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {
                "status": "unknown",
                "error": "ambiguous query",
                "all_joints_disabled_observed": False,
            },
        },
    }
    app.messages.put(("result", (result, tmp_path / "failed-result.json")))
    app.messages.put(("done", None))
    app.poll()
    assert live.stop.is_set()
    assert not app.running and app.worker is None and app.live is None
    assert not app.exit_confirmed and app.unconfirmed_full_arm
    assert str(app.read_button.cget("state")) == str(app.stop_button.cget("state")) == "normal"
    assert str(app.full_arm_recovery_button.cget("state")) == "normal"
    assert app.root.tk.call("after", "info")
    app.status.set("保留本次按钮反馈")
    assert not app.recover_readonly_exit({"phase": "DISABLED"})
    assert app.status.get() == "保留本次按钮反馈"
    closed = []
    monkeypatch.setattr(app.root, "destroy", lambda: closed.append(True))
    app.close()
    app.poll()
    assert closed and app.closing


def test_render_failure_does_not_stop_polling_or_completion_handling(default_tuner, monkeypatch):
    app, _, _, _ = default_tuner
    monkeypatch.setattr(app, "draw", lambda: (_ for _ in ()).throw(ValueError("injected plot failure")))
    app.messages.put(("done", None))
    app.poll()
    assert app.worker is None and not app.running
    assert app.root.tk.call("after", "info")
    assert "injected plot failure" in app.status.get()
    assert str(app.stop_button.cget("state")) == "normal"


def prepare_unknown_full_arm(app):
    app.exit_confirmed = False
    app.unconfirmed_full_arm = True
    app.unconfirmed_transport_closed = True
    app.unconfirmed_port = app.fields["port"].get()
    app.unconfirmed_calibration = asdict(app.calibration)


@pytest.mark.parametrize("previous_closed", [True, False])
def test_explicit_full_arm_recovery_saves_new_receipt_then_unlocks(
    default_tuner, readonly_scope, panel, monkeypatch, tmp_path, previous_closed
):
    app, _, _, _ = default_tuner
    scope = readonly_scope
    tasks = []
    monkeypatch.setattr(app, "launch", tasks.append)
    monkeypatch.setattr(panel, "ReadonlyB601Transport", scope.factory)
    prepare_unknown_full_arm(app)
    app.unconfirmed_transport_closed = previous_closed
    previous = tmp_path / "original-result.json"
    previous.write_text('{"motion_exit": "unknown"}', encoding="utf-8")
    original = previous.read_bytes(), previous.stat().st_mtime_ns
    app.unconfirmed_result_path = previous
    app.poll()
    assert not scope.devices and str(app.full_arm_recovery_button.cget("state")) == "normal"
    app.full_arm_recovery_button.invoke()
    assert len(tasks) == 1 and app.recovering_full_arm and not app.exit_confirmed
    tasks[0]()
    app.messages.put(("done", None))
    app.poll()
    assert app.exit_confirmed and not app.unconfirmed_full_arm and not app.confirmed.get()
    assert not app.running and app.next_run is None and app.preview_latest is None
    assert app.recovery_evidence["all_joints_disabled_observed"]
    assert app.recovery_evidence["transport_closed"] and app.recovery_evidence["hardware_ack"] == "unknown"
    assert len(list(panel.OUTPUTS.glob("readonly_recovery/*-full-arm.json"))) == 1
    assert (previous.read_bytes(), previous.stat().st_mtime_ns) == original
    assert_query_only(scope.serial)


@pytest.mark.parametrize("cancel", ["stop", "source"])
def test_automatic_connection_waits_for_done_and_can_cancel_after_receipt(
    default_tuner, readonly_scope, panel, monkeypatch, cancel
):
    app, _, _, _ = default_tuner
    tasks = []
    monkeypatch.setattr(app, "launch", tasks.append)
    monkeypatch.setattr(panel, "ReadonlyB601Transport", readonly_scope.factory)
    prepare_unknown_full_arm(app)
    app.read()
    assert len(tasks) == 1 and app.connection_recovery is not None
    tasks[0]()
    app._drain_messages()  # A saved receipt alone does not retire the worker.
    assert app.exit_confirmed and not app.connection_recovery["worker_done"]
    app.dispatch_connection_recovery()
    assert len(tasks) == 1 and not app.monitoring
    if cancel == "stop":
        app.stop()
    else:
        app.fields["port"].set("CHANGED_FAKE_PORT")
    app.messages.put(("done", None))
    app.poll()
    assert app.connection_recovery is None and app.monitor_stop.is_set()
    assert len(tasks) == 1 and not app.monitoring and not app.running


def test_unknown_full_arm_start_only_queues_readonly_recovery(default_tuner, monkeypatch):
    app, _, _, _ = default_tuner
    tasks = []
    monkeypatch.setattr(app, "launch", tasks.append)
    prepare_unknown_full_arm(app)
    app.hybrid_button.invoke()
    assert len(tasks) == 1 and app.connection_recovery is not None
    assert app.recovering_full_arm and not app.running and app.live is None
    assert app.next_run is None and not app.confirmed.get()
    app.stop()
    assert app.connection_recovery is None


@pytest.mark.parametrize("problem", ["worker", "running", "port", "calibration", "retained"])
def test_full_arm_recovery_cannot_open_before_idle_owner_and_identity_match(
    default_tuner, panel, monkeypatch, problem
):
    app, _, _, _ = default_tuner
    prepare_unknown_full_arm(app)
    monkeypatch.setattr(app, "launch", lambda function: pytest.fail("unadmitted recovery opened an owner"))
    if problem == "worker":
        app.worker = SimpleNamespace(is_alive=lambda: True)
    elif problem == "running":
        app.running = True
    elif problem == "port":
        app.unconfirmed_port = "ANOTHER_PORT"
    elif problem == "retained":
        monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [object()])
    else:
        app.unconfirmed_calibration = {"sha256": "wrong"}
    app.recover_full_arm_readonly()
    assert not app.exit_confirmed and not app.recovering_full_arm


def test_pending_ipc_cancel_error_does_not_leave_running_or_pending_state(default_tuner):
    app, _, _, _ = default_tuner
    app.running = True
    app.monitor_session = "closed-preview"
    app.preview_cleanup = {
        "monitor_session": app.monitor_session,
        "transport_closed": True,
        "sensor_closed": True,
    }
    app.pending_cleanup = lambda: (_ for _ in ()).throw(OSError("injected cancel close failure"))
    app.next_run = None
    app.dispatch_pending_start()
    assert not app.running and app.pending_cleanup is None and app.next_run is None
    assert "取消通信清理失败" in app.status.get()


@pytest.mark.parametrize("scaling", [1.25, 1.5])
def test_explicit_full_arm_recovery_button_remains_visible_at_minimum_size(default_tuner, scaling):
    app, _, _, _ = default_tuner
    root = app.root
    original_scaling = root.tk.call("tk", "scaling")
    prepare_unknown_full_arm(app)
    try:
        root.tk.call("tk", "scaling", scaling)
        root.geometry("1000x740+10000+10000")
        root.deiconify()
        root.update()
        app.poll()
        root.update_idletasks()
        button = app.full_arm_recovery_button
        assert "50 Hz 采集版" in root.title()
        assert button.winfo_ismapped() and str(button.cget("state")) == "normal"
        for parent in (root, button.master):
            x = button.winfo_rootx() - parent.winfo_rootx()
            y = button.winfo_rooty() - parent.winfo_rooty()
            assert 0 <= x < x + button.winfo_width() <= parent.winfo_width()
            assert 0 <= y < y + button.winfo_height() <= parent.winfo_height()
        assert button.winfo_height() >= button.winfo_reqheight()
    finally:
        root.withdraw()
        root.tk.call("tk", "scaling", original_scaling)
