"""CPU-only runner lifecycle checks, including incomplete evidence on interrupted cleanup."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lerobot.force_control.fake_hardware import FakeForceSensor, FakeGripper
from lerobot.force_control.trace import BenchTrace, TraceError, validate_trace
from lerobot.scripts import lerobot_force_teacher as runner
from tests.force_control.helpers import run_config


def manifest(root):
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def rows(root):
    return [json.loads(line) for line in (root / "control.jsonl").read_text(encoding="utf-8").splitlines()]


def instrument_device(monkeypatch, **operations):
    instances = []

    class ObservedGripper(FakeGripper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    for name, function in operations.items():
        setattr(ObservedGripper, name, function)
    monkeypatch.setattr(runner, "FakeGripper", ObservedGripper)
    return instances


def instrument_required_sensor(monkeypatch):
    """Model Required X518 disconnect invalidating every still-held recorder lease."""
    instances = []

    class LeaseSensitiveSensor(FakeForceSensor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config.required = True
            self.cleanup_events = []
            self.disconnect_with_lease = False
            instances.append(self)

        def release_recorder(self, lease):
            self.cleanup_events.append("release_recorder")
            super().release_recorder(lease)

        def disconnect(self):
            self.cleanup_events.append("disconnect")
            if self._recorder_leases:
                self.disconnect_with_lease = True
                self._latch_recorder_fault("Required X518 disconnected with an active recorder")
            super().disconnect()

    monkeypatch.setattr(runner, "FakeForceSensor", LeaseSensitiveSensor)
    return instances


@pytest.mark.parametrize("mode", ["approach", "position"])
def test_new_start_modes_flow_through_runner_trace_and_cleanup(tmp_path, mode):
    config = run_config(tmp_path, armed=True)
    config.initial_mode = mode
    config.initial_position_deg = -0.1 if mode == "position" else None
    config.initial_position_speed_deg_s = 0.5 if mode == "position" else None
    config.teacher = replace(config.teacher, approach_speed_deg_s=5, approach_force_threshold_N=0.05)
    config.duration_s = 0.6
    report = runner.run(config)
    assert report["valid"], report["errors"]
    history = rows(config.trace_root)
    active = [row for row in history if row["q_sent_deg"] is not None]
    assert active[0]["phase"] == ("APPROACH" if mode == "approach" else "POSITION_HOLD")
    if mode == "approach":
        assert any(row["transition"] == "APPROACH->FORCE_TRACK" for row in history)
    else:
        assert not any(row["phase"] == "FORCE_TRACK" for row in history)
        positioning = [row for row in active if row["phase"] == "POSITION_HOLD"]
        assert positioning and all(row["hold_speed_deg_s"] == 0.5 for row in positioning)
        assert all(
            abs(row["q_sent_deg"] - row["q_prev_sent_deg"]) <= 0.5 * row["dt_control_s"] + 1e-9
            for row in positioning
        )
        assert any(abs(row["q_sent_deg"] + 0.1) < 0.001 for row in active)
    assert history[-1]["phase"] == "DISABLED"


def test_default_config_fails_before_any_backend_construction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Backend creation before required calibration")

    monkeypatch.setattr(runner, "FakeGripper", forbidden)
    with pytest.raises(ValueError):
        runner.run(runner.ForceTeacherRunConfig(trace_root=tmp_path / "default"))
    assert not (tmp_path / "default").exists()


def test_even_a_ready_capability_audit_cannot_enable_unimplemented_hardware_runner(tmp_path, monkeypatch):
    from lerobot.robots.rebot_b601_follower import control_io

    monkeypatch.setattr(
        control_io,
        "audit_motorbridge_capabilities",
        lambda: SimpleNamespace(require_teacher_ready=lambda: None),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A hardware template must never construct a motion backend")

    monkeypatch.setattr(runner, "FakeGripper", forbidden)
    config = run_config(tmp_path)
    config.backend = "b601"
    config.dry_run = False
    config.observe_only = True
    with pytest.raises(ValueError, match="Hardware motion"):
        runner.run(config)
    assert not config.trace_root.exists()


def test_unarmed_fake_runs_read_only_and_seals_disabled_trace(tmp_path, monkeypatch):
    devices = instrument_device(monkeypatch)
    config = run_config(tmp_path)
    report = runner.run(config)
    assert report["valid"], report["errors"]
    assert report["terminal_state"] == "DISABLED"
    assert report["hardware_status"] == "hardware_not_tested"
    assert not devices[0].connected
    assert not any(op in ("send_gripper", "emergency_stop") for op, _ in devices[0].calls)
    assert {thread_id for _, thread_id in devices[0].calls} == {devices[0].owner}
    assert all(not row["label_valid"] for row in rows(config.trace_root))


def test_armed_fake_uses_real_trace_subscription_and_explicit_release(tmp_path, monkeypatch):
    devices = instrument_device(monkeypatch)
    config = run_config(tmp_path, armed=True)
    report = runner.run(config)
    assert report["valid"], report["errors"]
    assert report["complete"] and report["terminal_state"] == "DISABLED"
    history = rows(config.trace_root)
    assert any(row["phase"] == "FORCE_TRACK" for row in history)
    assert any(row["label_valid"] for row in history)
    assert any(row["phase"] == "RELEASE" for row in history)
    assert history[-1]["phase"] == "DISABLED"
    assert sum(row["event"] == "start" for row in history) == 1
    assert sum(row["event"] == "release" for row in history) == 1
    assert all(row["hardware_ack"] == "unknown" for row in history)
    assert {thread_id for _, thread_id in devices[0].calls} == {devices[0].owner}
    assert not devices[0].connected


@pytest.mark.parametrize("armed", [False, True])
def test_required_sensor_normal_shutdown_releases_recording_before_disconnect(tmp_path, monkeypatch, armed):
    sensors = instrument_required_sensor(monkeypatch)
    config = run_config(tmp_path, armed=armed)
    report = runner.run(config)
    assert report["valid"], report["errors"]
    assert manifest(config.trace_root)["complete"] is True
    assert sensors[0].cleanup_events == ["release_recorder", "disconnect"]
    assert not sensors[0].disconnect_with_lease and not sensors[0].has_resources


@pytest.mark.parametrize("failure", [RuntimeError("injected run failure"), KeyboardInterrupt()])
def test_run_exception_and_keyboard_interrupt_stop_and_seal_incomplete(tmp_path, monkeypatch, failure):
    devices = instrument_device(monkeypatch)
    original = runner.OwnerService.snapshot
    calls = 0

    def fail_while_active(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(runner.OwnerService, "snapshot", fail_while_active)
    config = run_config(tmp_path, armed=True)
    with pytest.raises(type(failure)):
        runner.run(config)
    result = manifest(config.trace_root)
    assert result["complete"] is False
    assert result["terminal_state"] == "FAULT"
    assert rows(config.trace_root)[-1]["phase"] == "FAULT"
    assert any(op == "emergency_stop" for op, _ in devices[0].calls)
    assert not devices[0].connected
    assert not validate_trace(config.trace_root)["valid"]


def test_partial_connect_failure_is_disposed_and_disconnected(tmp_path, monkeypatch):
    def partial_connect(self):
        FakeGripper.connect(self)
        raise RuntimeError("injected partial connect")

    devices = instrument_device(monkeypatch, connect=partial_connect)
    config = run_config(tmp_path)
    with pytest.raises(RuntimeError, match="partial connect"):
        runner.run(config)
    assert not devices[0].connected
    assert any(op == "disconnect" for op, _ in devices[0].calls)
    assert manifest(config.trace_root)["complete"] is False


def test_disconnect_exception_never_seals_a_complete_manifest(tmp_path, monkeypatch):
    def broken_disconnect(self):
        FakeGripper.disconnect(self)
        raise RuntimeError("injected disconnect failure")

    instrument_device(monkeypatch, disconnect=broken_disconnect)
    config = run_config(tmp_path)
    with pytest.raises(RuntimeError, match="disconnect failure"):
        runner.run(config)
    assert manifest(config.trace_root)["complete"] is False
    assert not validate_trace(config.trace_root)["valid"]


def test_sensor_shutdown_exception_never_seals_a_complete_manifest(tmp_path, monkeypatch):
    class BrokenShutdownSensor(FakeForceSensor):
        def disconnect(self):
            super().disconnect()
            raise RuntimeError("injected sensor shutdown failure")

    monkeypatch.setattr(runner, "FakeForceSensor", BrokenShutdownSensor)
    config = run_config(tmp_path)
    with pytest.raises(RuntimeError, match="sensor shutdown failure"):
        runner.run(config)
    assert manifest(config.trace_root)["complete"] is False
    assert "sensor shutdown failure" in manifest(config.trace_root)["failure_reason"]
    assert not validate_trace(config.trace_root)["valid"]


def test_current_invalid_joint_feedback_error_is_attached_without_changing_fault_classification(
    tmp_path, monkeypatch
):
    def invalid_feedback(self, names, *, deadline_ns):
        self.snapshot_error = "TimeoutError: injected missing feedback frame"
        return FakeGripper.read_joint_snapshot(self, names, deadline_ns=deadline_ns)

    instrument_device(monkeypatch, read_joint_snapshot=invalid_feedback)
    config = run_config(tmp_path)
    with pytest.raises(RuntimeError, match="invalid_joint_feedback") as caught:
        runner.run(config)
    assert any("TimeoutError: injected missing feedback frame" in note for note in caught.value.__notes__)
    assert any("invalid_joint_feedback" in (row["failure_reason"] or "") for row in rows(config.trace_root))
    assert not manifest(config.trace_root)["complete"]


def test_delayed_fault_observer_retains_original_feedback_note_and_manifest_diagnosis(tmp_path, monkeypatch):
    masked_fault_published = threading.Event()
    faults = []
    original_publish = runner.OwnerService._publish
    original_snapshot = runner.OwnerService.snapshot
    read_count = 0

    def invalid_after_start(self, names, *, deadline_ns):
        nonlocal read_count
        read_count += 1
        if read_count == 3:
            self.snapshot_error = "TimeoutError: triggering feedback response incomplete"
        return FakeGripper.read_joint_snapshot(self, names, deadline_ns=deadline_ns)

    def observe_fault_ticks(self, row):
        original_publish(self, row)
        if row["phase"] == "FAULT":
            faults.append(row.copy())
            if len(faults) >= 2:
                masked_fault_published.set()

    def delayed_snapshot(self):
        assert masked_fault_published.wait(2)
        return original_snapshot(self)

    instrument_device(monkeypatch, read_joint_snapshot=invalid_after_start)
    monkeypatch.setattr(runner.OwnerService, "_publish", observe_fault_ticks)
    monkeypatch.setattr(runner.OwnerService, "snapshot", delayed_snapshot)
    config = run_config(tmp_path)
    with pytest.raises(RuntimeError, match="invalid_joint_feedback") as caught:
        runner.run(config)
    assert faults[0]["joint_error"] == "TimeoutError: triggering feedback response incomplete"
    assert faults[1].get("joint_error") is None
    assert any("triggering feedback response incomplete" in note for note in caught.value.__notes__)
    origin = manifest(config.trace_root)["metadata"]["fault_origin"]
    assert origin["tick_id"] == faults[0]["tick_id"]
    assert origin["joint_error"] == faults[0]["joint_error"]
    assert origin["joint_snapshot"] == json.loads(json.dumps(faults[0]["joint_snapshot"]))
    assert origin["tick_start_ns"] == faults[0]["tick_start_ns"]
    assert origin["decision_anchor_ns"] == faults[0]["decision_anchor_ns"]


def test_shared_live_stop_interrupts_backend_startup_before_enable(tmp_path, monkeypatch):
    stop = threading.Event()
    stages = []

    def guarded_start(self, snapshot, guard):
        stages.append("prepare")
        stop.set()
        guard()
        stages.append("enable")
        return snapshot.get("gripper").position_deg

    devices = instrument_device(monkeypatch, start_gripper=guarded_start)
    live = SimpleNamespace(stop=stop, latest=None, applied=None, take=lambda: None)
    config = run_config(tmp_path, armed=True)
    with pytest.raises(RuntimeError, match="user_stop_during_startup"):
        runner.run(config, live=live)
    assert stages == ["prepare"]
    assert not devices[0].connected
    assert sum(op == "emergency_stop" for op, _ in devices[0].calls) == 1
    assert not any(op == "send_gripper" for op, _ in devices[0].calls)
    assert manifest(config.trace_root)["complete"] is False


def test_stop_before_run_opens_no_transport(tmp_path, monkeypatch):
    stop = threading.Event()
    stop.set()
    devices = instrument_device(monkeypatch)
    live = SimpleNamespace(stop=stop, latest=None, applied=None, take=lambda: None)
    config = run_config(tmp_path, armed=True)
    with pytest.raises(RuntimeError, match="user_stop_before_connect"):
        runner.run(config, live=live)
    assert devices[0].calls == []
    assert not config.trace_root.exists()


@pytest.mark.parametrize("armed", [False, True])
def test_deferred_cleanup_waits_for_owner_then_seals_incomplete_trace(tmp_path, monkeypatch, armed):
    monkeypatch.setattr(runner, "_UNSTOPPED_BENCHES", [])
    sensors = instrument_required_sensor(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    count = 0

    def delayed_feedback(self, names, *, deadline_ns):
        nonlocal count
        count += 1
        active = any(op == "send_gripper" for op, _ in self.calls)
        if not entered.is_set() and (active if armed else count == 2):
            entered.set()
            release.wait()
        return FakeGripper.read_joint_snapshot(self, names, deadline_ns=deadline_ns)

    devices = instrument_device(monkeypatch, read_joint_snapshot=delayed_feedback)
    config = run_config(tmp_path, armed=armed)
    config.shutdown_timeout_s = 0.01
    config.duration_s = 0.08
    try:
        with pytest.raises(BaseExceptionGroup, match="cleanup evidence"):
            runner.run(config)
        assert entered.is_set() and len(runner._UNSTOPPED_BENCHES) == 1
        service, sensor, trace = runner._UNSTOPPED_BENCHES[0]
        assert service.has_resources and trace.has_resources and sensor.has_resources
        assert runner.finalize_retained_benches() == []
        assert devices[0].connected
    finally:
        release.set()
        if runner._UNSTOPPED_BENCHES:
            runner._UNSTOPPED_BENCHES[0][0].stop(timeout_s=2)
            completed = runner.finalize_retained_benches()
    assert completed[0]["shutdown_evidence"]["transport_closed"]
    assert completed[0]["valid"] is False
    assert not runner._UNSTOPPED_BENCHES
    assert not sensor.has_resources and not trace.has_resources
    assert sensors[0].cleanup_events == ["release_recorder", "disconnect"]
    assert not sensors[0].disconnect_with_lease
    assert manifest(config.trace_root)["complete"] is False
    assert manifest(config.trace_root)["metadata"]["shutdown_evidence"]["transport_closed"]
    disable_count = sum(op == "emergency_stop" for op, _ in devices[0].calls)
    if armed:
        assert disable_count == 1
    else:
        assert not any(op == "send_gripper" for op, _ in devices[0].calls)
        # A late read may itself fault; an otherwise disabled owner needs only transport cleanup.
        assert disable_count <= 1


def test_trace_cleanup_failure_retains_sensor_until_recorder_can_release(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_UNSTOPPED_BENCHES", [])
    sensors = instrument_required_sensor(monkeypatch)
    can_close = threading.Event()

    class DelayedTrace(BenchTrace):
        def close(self, terminal_state, complete):
            if not can_close.is_set():
                raise TraceError("injected trace drain pending")
            return super().close(terminal_state, complete)

    monkeypatch.setattr(runner, "BenchTrace", DelayedTrace)
    config = run_config(tmp_path)
    try:
        with pytest.raises(TraceError, match="trace drain pending"):
            runner.run(config)
        service, sensor, trace = runner._UNSTOPPED_BENCHES[0]
        assert not service.has_resources and trace.has_resources and sensor.has_resources
        assert sensors[0].cleanup_events == []
        assert runner.finalize_retained_benches() == []
        assert sensors[0].cleanup_events == [] and sensors[0].is_connected
    finally:
        can_close.set()
        completed = runner.finalize_retained_benches()
    assert not runner._UNSTOPPED_BENCHES
    assert sensors[0].cleanup_events == ["release_recorder", "disconnect"]
    assert not sensors[0].disconnect_with_lease
    assert completed[0]["valid"] is False
    assert "trace drain pending" in "; ".join(completed[0]["cleanup_errors"])
    assert not manifest(config.trace_root)["complete"]


def test_sensor_cleanup_retry_preserves_prior_error_after_resources_release(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_UNSTOPPED_BENCHES", [])

    class RetrySensor(FakeForceSensor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_disconnect = True

        def disconnect(self):
            assert not self._recorder_leases
            if self.first_disconnect:
                self.first_disconnect = False
                raise RuntimeError("injected sensor acquisition still alive")
            super().disconnect()

    monkeypatch.setattr(runner, "FakeForceSensor", RetrySensor)
    config = run_config(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="acquisition still alive"):
            runner.run(config)
        assert len(runner._UNSTOPPED_BENCHES) == 1
        service, sensor, trace = runner._UNSTOPPED_BENCHES[0]
        assert not service.has_resources and not trace.has_resources and sensor.has_resources
        assert not manifest(config.trace_root)["complete"]
    finally:
        completed = runner.finalize_retained_benches()
    assert not runner._UNSTOPPED_BENCHES and not sensor.has_resources
    assert "acquisition still alive" in "; ".join(completed[0]["cleanup_errors"])
    final = manifest(config.trace_root)
    assert final["complete"] is False and "acquisition still alive" in final["failure_reason"]
