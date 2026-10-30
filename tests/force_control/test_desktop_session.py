"""Headless desktop lifecycle tests; every transport and owner is injected/fake."""

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.robots.rebot_b601_follower.readonly_transport import (
    DEFAULT_B601_JOINTS,
    B601ReadonlyBatch,
    B601ReceivedFeedback,
    B601ReceivedRegister,
)


@pytest.fixture
def session_module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    module = importlib.import_module("b601_session")
    monkeypatch.setattr(module, "BINDINGS_FILE", tmp_path / "camera-bindings.json")
    return module


@pytest.fixture
def calibration_path(tmp_path):
    path = tmp_path / "all-joints.json"
    path.write_text(
        json.dumps(
            {
                spec.name: {
                    "id": spec.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -300,
                    "range_max": 0,
                }
                for spec in DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    return path


def wait_for(predicate, *, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("Headless session did not reach the expected state")


def command(controller, action, payload=None):
    request_id = controller.request(action, payload)
    return wait_for(
        lambda: next(
            (
                result
                for result in controller.snapshot().operation_results
                if result["request_id"] == request_id
            ),
            None,
        )
    )


@pytest.fixture
def make_controller(session_module, calibration_path, tmp_path):
    controllers = []

    def make(*, demo=False, services=None, start=True, **kwargs):
        controller = session_module.SessionController(
            demo=demo,
            services=services,
            settings_path=tmp_path / "missing-settings.json",
            calibration_path=calibration_path,
            output_root=tmp_path / f"run-{len(controllers)}",
            **kwargs,
        )
        controllers.append(controller)
        if start:
            controller.start()
            wait_for(lambda: controller.snapshot().state != "loading")
        return controller

    yield make
    for controller in controllers:
        controller._retained.clear()
        controller._camera_session = None
        controller.close()
        if controller._thread is not None:
            controller._thread.join(4)
            assert not controller._thread.is_alive()


def test_session_import_has_no_qt_tk_or_device_initialization():
    folder = Path(__file__).resolve().parents[2] / "examples/force_teacher"
    script = (
        f"import sys; sys.path.insert(0, {str(folder)!r}); import b601_session; "
        "assert not any(k.startswith(('tkinter', 'PySide6', 'pyqtgraph')) for k in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_snapshot_keeps_all_original_result_errors_until_cleared(make_controller, tmp_path):
    controller = make_controller(start=False)
    errors = [f"failure {index} " + "x" * 80 for index in range(40)]
    controller._accept_result(
        {"config": {}, "errors": errors, "shutdown_evidence": {}}, tmp_path / "result.json"
    )
    assert controller.snapshot().result_errors == tuple(errors)
    assert errors[-1] not in controller.snapshot().error  # The existing short report keeps only 32.
    controller._changed(error="")
    assert controller.snapshot().result_errors == ()


def test_default_capture_profile_does_not_load_legacy_tuning_values(
    session_module, calibration_path, tmp_path, monkeypatch
):
    enabled = {key: default for key, (_, default) in session_module.OPTIONAL_GROUPS.items()}
    for name, target in (("b601_capture_parameters.json", "2"), ("b601_ip_parameters.json", "9")):
        values = {**session_module.default_values(), "target_N": target}
        session_module.save_settings(tmp_path / name, values, enabled, calibration_path)
    monkeypatch.setattr(session_module, "DEFAULT_SETTINGS", tmp_path / session_module.DEFAULT_SETTINGS.name)
    before = {path: path.read_bytes() for path in tmp_path.glob("b601_*.json")}
    for explicit, target in ((None, "2"), (tmp_path / "b601_ip_parameters.json", "9")):
        controller = session_module.SessionController(
            settings_path=explicit, calibration_path=calibration_path, output_root=tmp_path / "runs"
        )
        try:
            controller.start()
            wait_for(lambda controller=controller: controller.snapshot().state != "loading")
            assert controller.snapshot().state == "disconnected"
            assert controller.snapshot().values["target_N"] == target
        finally:
            controller.close()
            controller._thread.join(4)
            assert not controller._thread.is_alive()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("mode", [None, "sequential", "cached", "pipelined"])
def test_current_communication_and_recording_config(make_controller, complete_values, mode):
    controller = make_controller(command_feedback_mode=mode, command_tx_gap_us=1000)
    controller._values.update(complete_values)
    config = controller._build_config("hybrid", confirmed=True)
    assert config.b601.control.command_feedback_mode == (mode or "sequential")
    if mode:
        assert config.b601.control.command_tx_gap_us == 1000
    controller._recording_values.update(task="test", object_id="test", contact_condition_id="test")
    controller._cameras = {
        role: {"index_or_path": i, "fps": 50, "width": 640, "height": 480}
        for i, role in enumerate(("front", "wrist"))
    }
    recording = controller._recording_config()
    assert recording["h264_encoder"] == "nvenc"
    assert "rate_profile" not in recording


class FakeVerificationWorker:
    def __init__(self, project, root):
        self.root = root
        self.events = []
        self.finished = False
        self.process = SimpleNamespace(returncode=None)
        self.output = []
        self.cancelled = False

    def poll(self):
        events, self.events = self.events, []
        return events

    def complete(self, report):
        self.events.append({"event": "result", "report": report})
        self.process.returncode = 0 if report["valid"] else 1
        self.finished = True

    def cancel(self):
        self.cancelled = self.finished = True
        self.process.returncode = 1

    def close(self):
        assert self.finished


def test_verification_blocks_new_capture_and_writes_only_external_report(make_controller, tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    controller = make_controller(demo=True, services={"verification_factory": FakeVerificationWorker})
    assert command(controller, "update_form", {"recording_values": {"root": str(root)}})["success"]
    assert command(controller, "verify_dataset")["success"]
    worker = controller._verification_worker
    worker.events.extend(
        [
            {"event": "episode", "index": 1, "total": 2, "episode_uid": "first"},
            {"event": "video", "video": "videos/front.mp4"},
        ]
    )
    progress = wait_for(lambda: controller.snapshot().recording["full_verification"].get("video"))
    assert progress == "videos/front.mp4"
    assert controller.snapshot().capabilities["connect"]
    assert not command(controller, "connect")["success"]
    worker.complete({"valid": True, "errors": [], "episodes": [{"episode_uid": "first"}]})
    result = wait_for(
        lambda: (
            controller.snapshot().recording["full_verification"]
            if controller.snapshot().recording["full_verification"]["state"] == "passed"
            else None
        )
    )
    report_path = Path(result["report_path"])
    assert report_path.parent == root.parent / "dataset_verification_reports"
    assert json.loads(report_path.read_text(encoding="utf-8"))["result"]["valid"]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    episodes = root / "meta/teacher_episodes"
    episodes.mkdir()
    (episodes / "new.json").write_text("{}", encoding="utf-8")
    wait_for(lambda: controller.snapshot().recording["full_verification"]["state"] == "idle")
    assert command(controller, "update_form", {"recording_values": {"root": str(tmp_path / "other")}})[
        "success"
    ]
    assert controller.snapshot().recording["full_verification"]["state"] == "idle"


def test_verification_cancel_and_close_reap_worker(make_controller, tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    controller = make_controller(demo=True, services={"verification_factory": FakeVerificationWorker})
    assert command(controller, "update_form", {"recording_values": {"root": str(root)}})["success"]
    assert command(controller, "verify_dataset")["success"]
    worker = controller._verification_worker
    assert command(controller, "cancel_verification")["success"]
    wait_for(lambda: controller.snapshot().recording["full_verification"]["state"] == "cancelled")
    assert worker.cancelled and controller._verification_worker is None
    assert command(controller, "verify_dataset")["success"]
    worker = controller._verification_worker
    controller.close()
    wait_for(lambda: controller.closed)
    assert worker.cancelled and controller._verification_worker is None


def test_verification_refuses_live_writer(make_controller, tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    (root / ".sensor-writer.lock").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    controller = make_controller(demo=True, services={"verification_factory": FakeVerificationWorker})
    assert command(controller, "update_form", {"recording_values": {"root": str(root)}})["success"]
    result = command(controller, "verify_dataset")
    assert not result["success"] and controller._verification_worker is None
    assert "live writer" in controller.snapshot().error


def test_verification_requires_stopped_session_and_clean_resources(make_controller, tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    controller = make_controller(demo=True, services={"verification_factory": FakeVerificationWorker})
    assert command(controller, "update_form", {"recording_values": {"root": str(root)}})["success"]
    assert command(controller, "connect")["success"]
    assert command(controller, "start", {"confirmed": True})["success"]
    assert controller.snapshot().capabilities["verify_dataset"]
    assert not command(controller, "verify_dataset")["success"]
    controller.stop()
    wait_for(lambda: controller.snapshot().state == "disconnected")
    controller._retained.append(object())
    controller._changed()
    assert controller.snapshot().capabilities["verify_dataset"]


def test_prefilled_verify_video_option_stays_boolean(make_controller, tmp_path):
    config = {
        "root": str(tmp_path / "dataset"),
        "task": "test",
        "object_id": "object",
        "contact_condition_id": "condition",
        "verify_video_on_save": True,
        "camera_configs": {"front": {"index_or_path": 0, "fps": 50, "width": 16, "height": 16}},
    }
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    controller = make_controller(start=False)
    controller.prefill_recording(path)
    assert controller.snapshot().recording_values["verify_video_on_save"] is True
    assert controller._recording_config()["verify_video_on_save"] is True


def test_snapshot_is_recursive_immutable_and_detached(make_controller):
    controller = make_controller(demo=True)
    snapshot = controller.snapshot()
    with pytest.raises(TypeError):
        snapshot.values["target_N"] = "99"
    with pytest.raises(TypeError):
        snapshot.camera_configs["front"]["fps"] = 1
    assert command(controller, "update_form", {"values": {"target_N": "6"}})["success"]
    assert snapshot.values["target_N"] == "5"
    assert controller.snapshot().values["target_N"] == "6"


def test_demo_repeated_start_finish_and_old_commands_cannot_duplicate_work(make_controller):
    controller = make_controller(demo=True)
    assert command(controller, "connect")["success"]
    assert command(controller, "start", {"confirmed": True})["success"]
    assert not command(controller, "start", {"confirmed": True})["success"]
    assert command(controller, "record_start")["success"]
    assert not command(controller, "record_start")["success"]
    assert command(controller, "record_finish", {"task_result": "unknown"})["success"]
    assert not command(controller, "record_finish")["success"]
    assert controller.snapshot().recording["completed_episode_count"] == 1
    assert controller.snapshot().state == "running"
    assert not controller.snapshot().recording["file_complete"]
    assert not controller.snapshot().recording["training_eligible"]


def test_queued_start_is_canceled_before_supervisor_starts(make_controller):
    controller = make_controller(demo=True, start=False)
    request_id = controller.request("start", {"confirmed": True})
    controller.stop()
    controller.start()
    result = wait_for(
        lambda: next(
            (row for row in controller.snapshot().operation_results if row["request_id"] == request_id), None
        )
    )
    assert not result["success"] and controller.snapshot().state != "running"


def test_stale_completion_cannot_replace_a_new_session(make_controller):
    controller = make_controller(demo=True, start=False)
    original = controller._session_id
    controller._completions.put(("owner_error", "older-session", "old failure"))
    controller._drain_completions()
    assert controller._session_id == original and not controller._error


def test_monitoring_allows_metadata_but_locks_device_settings(make_controller):
    controller = make_controller(demo=True)
    assert command(controller, "connect")["success"]
    assert command(controller, "update_form", {"recording_values": {"task": "pick"}})["success"]
    assert not command(controller, "update_form", {"values": {"port": "NEW_PORT"}})["success"]
    assert command(controller, "start", {"confirmed": True})["success"]
    assert not command(controller, "update_form", {"recording_values": {"task": "changed"}})["success"]
    assert controller.snapshot().recording_values["task"] == "pick"


def test_monitor_exit_without_cleanup_evidence_requires_recovery(make_controller):
    def monitor(stop, publish, values, calibration, cleanup, *, retained):
        publish({"phase": "DISABLED", "joint_received_ns": time.perf_counter_ns()})

    controller = make_controller(services={"monitor": monitor})
    assert command(controller, "connect")["success"]
    wait_for(lambda: controller._resources_stopped())
    wait_for(lambda: controller.snapshot().state != "monitoring")
    assert controller.snapshot().state == "recovery_required"
    assert not controller.snapshot().exit_confirmed


def test_close_does_not_confirm_exit_with_retained_resources(make_controller):
    controller = make_controller(start=False)
    controller._retained.append(object())
    controller.start()
    wait_for(lambda: controller.snapshot().state != "loading")
    controller.close()
    time.sleep(0.06)
    assert not controller.closed
    controller._retained.clear()
    wait_for(lambda: controller.closed)


def recovery_receipt(module, port, calibration, *, session_id, fault=None):
    started = time.perf_counter_ns()
    observed, closed = started + 100, started + 200
    rows = [
        {
            "name": spec.name,
            "motor_id": spec.motor_id,
            "feedback_id": spec.feedback_id,
            "status_code": 0,
            "receive_timestamp_ns": started + 50,
            "receive_sequence": 1,
            "position_deg": -1,
        }
        for spec in DEFAULT_B601_JOINTS
    ]
    receipt = {
        "complete": True,
        "read_only": True,
        "motion_attempted": False,
        "parameter_writes": False,
        "all_joints_disabled_observed": True,
        "transport_closed": True,
        "errors": [],
        "session_id": session_id,
        "port": port,
        "calibration": asdict(calibration),
        "started_ns": started,
        "observed_ns": observed,
        "closed_ns": closed,
        "feedback": {"is_valid": True, "feedback": rows},
    }
    if fault == "enabled":
        rows[0]["status_code"] = 1
    elif fault == "stale":
        rows[0]["receive_timestamp_ns"] = started - 1
    elif fault == "missing":
        rows.pop()
    elif fault == "close":
        receipt["transport_closed"] = False
    elif fault == "invalid_feedback":
        receipt["feedback"]["is_valid"] = False
    elif fault == "invalid_sequence":
        rows[0]["receive_sequence"] = 0
    elif fault == "nonfinite_position":
        rows[0]["position_deg"] = float("nan")
    return receipt


@pytest.mark.parametrize(
    "fault",
    [
        "enabled",
        "stale",
        "missing",
        "close",
        "invalid_feedback",
        "invalid_sequence",
        "nonfinite_position",
        "retained",
    ],
)
def test_recovery_rejects_incomplete_or_invalid_seven_joint_proof(make_controller, session_module, fault):
    def query(port, calibration, **kwargs):
        if fault == "retained":
            kwargs["retain"](object())
        return recovery_receipt(
            session_module, port, calibration, session_id=kwargs["session_id"], fault=fault
        )

    controller = make_controller(services={"query_disabled_arm": query})
    controller._exit_confirmed = False
    controller._unconfirmed_port = controller._values["port"]
    controller._unconfirmed_calibration = asdict(controller._calibration)
    controller._changed(state="recovery_required")
    result = command(controller, "recover")
    assert not result["success"] and not controller.snapshot().exit_confirmed


def test_recovery_saves_fresh_proof_and_requires_new_explicit_connection(make_controller, session_module):
    def query(port, calibration, **kwargs):
        return recovery_receipt(session_module, port, calibration, session_id=kwargs["session_id"])

    controller = make_controller(services={"query_disabled_arm": query})
    controller._exit_confirmed = False
    controller._unconfirmed_port = controller._values["port"]
    controller._unconfirmed_calibration = asdict(controller._calibration)
    controller._changed(state="recovery_required")
    assert command(controller, "recover")["success"]
    assert controller.snapshot().exit_confirmed and controller.snapshot().state == "disconnected"
    assert controller._telemetry == {}
    assert len(list((controller._output_root / "readonly_recovery").glob("*.json"))) == 1


@pytest.mark.parametrize("status", [0, 1, 13])
def test_real_recovery_helper_serializes_validity_for_desktop_without_bypassing_status(
    make_controller, status
):
    calls = []

    class ReadonlyTransport:
        def __init__(self, *_args, **_kwargs):
            self.has_resources = False

        def open(self):
            self.has_resources = True
            calls.append("open")

        def read_mapping(self, **_kwargs):
            now = time.perf_counter_ns()
            calls.append("mapping_query")
            registers = tuple(
                B601ReceivedRegister(
                    spec.name, spec.motor_id, spec.feedback_id, spec.model, rid, 12.5, now, 1, "", ""
                )
                for spec in DEFAULT_B601_JOINTS
                for rid in (21, 22, 23)
            )
            return B601ReadonlyBatch("mapping", now, now, now + 1_000_000, registers=registers)

        def read_feedback(self, **_kwargs):
            now = time.perf_counter_ns()
            calls.append("feedback_query")
            rows = tuple(
                B601ReceivedFeedback(
                    spec.name,
                    spec.motor_id,
                    spec.feedback_id,
                    spec.model,
                    status,
                    {0: "DISABLED", 1: "ENABLED", 13: "LOST_COMM"}[status],
                    -0.01,
                    -1.0,
                    0,
                    0,
                    30,
                    30,
                    now,
                    1,
                    "",
                    "",
                    (12.5, 10, 28),
                )
                for spec in DEFAULT_B601_JOINTS
            )
            return B601ReadonlyBatch(
                "feedback",
                now,
                now,
                now + 1_000_000,
                feedback=rows,
                error="drive fault" if status == 13 else None,
            )

        def close(self):
            calls.append("close")
            self.has_resources = False

    controller = make_controller(services={"readonly_factory": ReadonlyTransport})
    controller._exit_confirmed = False
    controller._unconfirmed_port = controller._values["port"]
    controller._unconfirmed_calibration = asdict(controller._calibration)
    controller._changed(state="recovery_required")
    result = command(controller, "recover")
    assert result["success"] is (status == 0)
    assert controller.snapshot().exit_confirmed is (status == 0)
    assert calls == ["open", "mapping_query", "feedback_query", "close"]
    if status == 0:
        receipt = controller._recovery_evidence
        assert receipt["feedback"].get("error") is None
        assert receipt["all_joints_disabled_observed"] is True
        assert not receipt["motion_attempted"] and not receipt["parameter_writes"]
        assert controller.snapshot().state == "disconnected"


def test_canceled_recovery_cannot_apply_a_late_valid_receipt(make_controller, session_module):
    entered, release = threading.Event(), threading.Event()

    def query(port, calibration, **kwargs):
        entered.set()
        assert release.wait(2)
        return recovery_receipt(session_module, port, calibration, session_id=kwargs["session_id"])

    controller = make_controller(services={"query_disabled_arm": query})
    controller._exit_confirmed = False
    controller._unconfirmed_port = controller._values["port"]
    controller._unconfirmed_calibration = asdict(controller._calibration)
    controller._changed(state="recovery_required")
    try:
        request_id = controller.request("recover")
        assert entered.wait(2)
        controller.stop()
        release.set()
        result = wait_for(
            lambda: next(
                (row for row in controller.snapshot().operation_results if row["request_id"] == request_id),
                None,
            )
        )
        assert not result["success"] and not controller.snapshot().exit_confirmed
        assert not (controller._output_root / "readonly_recovery").exists()
    finally:
        release.set()


@dataclass
class ReviewedConfig:
    b601: object
    trace_root: object = None
    runtime: object = field(
        default_factory=lambda: SimpleNamespace(joint_names=tuple(spec.name for spec in DEFAULT_B601_JOINTS))
    )


def test_owner_start_waits_for_monitor_cleanup_and_cancel_prevents_handoff(
    make_controller, session_module, monkeypatch
):
    monitor_stopped, allow_close, owner_started = threading.Event(), threading.Event(), threading.Event()

    def monitor(stop, publish, values, calibration, cleanup, *, retained):
        while not stop.wait(0.005):
            publish({"phase": "DISABLED", "joint_received_ns": time.perf_counter_ns()})
        monitor_stopped.set()
        assert allow_close.wait(3)
        cleanup({"transport_closed": True, "sensor_closed": True})

    def owner(*args, **kwargs):
        owner_started.set()
        raise AssertionError("Canceled handoff opened an owner")

    controller = make_controller(
        services={"monitor": monitor, "review": lambda *a, **k: None, "run_in_process": owner}
    )
    controller._build_config = lambda *a, **k: ReviewedConfig(
        SimpleNamespace(port=controller._values["port"])
    )
    monkeypatch.setattr(session_module.draccus, "encode", lambda config: {"offline": True})
    try:
        assert command(controller, "connect")["success"]
        wait_for(lambda: controller.snapshot().state == "monitoring")
        request_id = controller.request("start", {"confirmed": True})
        assert monitor_stopped.wait(3), (
            controller.snapshot().state,
            controller.snapshot().message,
            controller.snapshot().error,
            controller.snapshot().operation_results,
        )
        assert not owner_started.is_set()
        controller.stop()
        allow_close.set()
        result = wait_for(
            lambda: next(
                (row for row in controller.snapshot().operation_results if row["request_id"] == request_id),
                None,
            )
        )
        assert not result["success"] and not owner_started.is_set()
        wait_for(lambda: not controller.snapshot().capabilities["connect"])
    finally:
        allow_close.set()


def test_failed_camera_close_retains_session_and_blocks_connect(make_controller):
    class RetainedCamera:
        def close(self):
            raise TimeoutError("camera reader still active")

    controller = make_controller(start=False)
    controller._camera_session = RetainedCamera()
    with pytest.raises(TimeoutError):
        controller._close_camera_preview()
    assert controller._camera_session is not None
    controller._changed(state="disconnected")
    assert controller.snapshot().capabilities["connect"]


@pytest.mark.parametrize("fault", ["review", "live_factory"])
def test_partial_start_failure_preserves_safe_monitor_or_confirmed_disconnect(
    make_controller, session_module, monkeypatch, fault
):
    def monitor(stop, publish, values, calibration, cleanup, *, retained):
        while not stop.wait(0.005):
            publish({"phase": "DISABLED", "joint_received_ns": time.perf_counter_ns()})
        cleanup({"transport_closed": True, "sensor_closed": True})

    def fail(*args, **kwargs):
        raise OSError("injected preparation failure")

    controller = make_controller(
        services={
            "monitor": monitor,
            "review": fail if fault == "review" else lambda *a, **k: None,
            "live_factory": fail,
        }
    )
    controller._build_config = lambda *a, **k: ReviewedConfig(
        SimpleNamespace(port=controller._values["port"])
    )
    monkeypatch.setattr(session_module.draccus, "encode", lambda config: {"offline": True})
    assert command(controller, "connect")["success"]
    wait_for(lambda: controller.snapshot().state == "monitoring")
    result = command(controller, "start", {"confirmed": True})
    assert not result["success"] and controller.snapshot().exit_confirmed
    assert controller._owner_thread is None
    assert controller.snapshot().state == ("monitoring" if fault == "review" else "disconnected")


@pytest.fixture
def complete_values(session_module):
    values = session_module.default_values()
    values.update(
        port="FAKE_FOLLOWER",
        leader_port="FAKE_LEADER",
        target_N="5",
        kp_deg_per_N="0.04",
        ki_deg_per_N_s="0.08",
        close_sign="-1",
        target_max_N="9.80665",
        max_relative_target_deg="0.4",
        max_command_tracking_error_deg="0.8",
        stall_position_tolerance_deg="0.01",
        stall_timeout_s="0.3",
        gripper_follow_speed_deg_s="0.17",
        contact_force_threshold_N="0.43",
        motor_velocity_code="1000",
        drive_percent="25",
    )
    return values


def test_direct_position_config_uses_requested_target_and_preserves_limits(make_controller, complete_values):
    controller = make_controller(start=False)
    controller._values.update(complete_values)
    config = controller._build_config(
        "position", confirmed=True, payload={"position_deg": -10.0, "position_speed_deg_s": 0.7}
    )
    assert config.initial_mode == "position"
    assert config.initial_position_deg == -10.0 and config.initial_position_speed_deg_s == 0.7
    assert list(config.runtime.joint_names) == ["gripper"]
    with pytest.raises(ValueError, match="指定位置"):
        controller._build_config(
            "position", confirmed=True, payload={"position_deg": 10.0, "position_speed_deg_s": 0.7}
        )
    with pytest.raises(ValueError, match="定位速度"):
        controller._build_config(
            "position", confirmed=True, payload={"position_deg": -10.0, "position_speed_deg_s": 0.0}
        )


def fake_monitor_with_fresh_reference(stop, publish, values, calibration, cleanup, *, retained):
    while not stop.wait(0.005):
        now = time.perf_counter_ns()
        publish(
            {
                "phase": "DISABLED",
                "joint_received_ns": now,
                "force_timestamp_ns": now,
                "read_only": True,
                "joint_valid": True,
                "force_valid": True,
                "joint_status_code": 0,
            }
        )
    cleanup({"transport_closed": True, "sensor_closed": True})


class HeadlessLive:
    def __init__(self, *, poll_s):
        self.poll_s, self.closed = poll_s, False
        self.stop = threading.Event()
        self.latest = {"phase": "FOLLOW", "tick_id": 1}
        self.startup = {"complete": True}
        self.recording = self.applied = None
        self.preview, self.ipc_errors = {}, []
        self.submitted = []

    def receive(self):
        pass

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled

    def close(self):
        self.closed = True

    def submit(self, kind, value):
        self.submitted.append((kind, value))


def test_direct_position_start_checks_before_fake_owner(make_controller, session_module, complete_values):
    reviewed = []
    owner_started = threading.Event()

    def review(output, *, reviewed_config):
        document = json.loads(Path(reviewed_config).read_text(encoding="utf-8"))
        assert document["operator_values"]["position_target_deg"] == "-10.0"
        assert document["operator_values"]["position_speed_deg_s"] == "0.7"
        config = session_module.draccus.decode(session_module.ForceTeacherRunConfig, document["config"])
        config.validate()
        reviewed.append(config)

    def owner(output, *, reviewed_config, live):
        document = json.loads(Path(reviewed_config).read_text(encoding="utf-8"))
        owner_started.set()
        assert live.stop.wait(3)
        live.close()
        return {
            "config": document["config"],
            "calibration": document["calibration"],
            "errors": [],
            "shutdown_evidence": {
                "transport_closed": True,
                "fault_disposition": {
                    "all_joints_disabled_observed": True,
                    "disabled_feedback_observed": True,
                    "status": "submitted",
                    "error": None,
                },
            },
            "hardware_ack": "unknown",
            "simulated": True,
        }

    controller = make_controller(
        services={
            "monitor": fake_monitor_with_fresh_reference,
            "review": review,
            "live_factory": HeadlessLive,
            "run_in_process": owner,
        }
    )
    assert command(controller, "update_form", {"values": complete_values})["success"]
    assert command(controller, "connect")["success"]
    wait_for(lambda: controller.snapshot().state == "monitoring")
    payload = {"mode": "position", "confirmed": True, "position_deg": -10.0, "position_speed_deg_s": 0.7}
    for invalid in (
        {**payload, "confirmed": False},
        {**payload, "position_deg": 10.0},
        {**payload, "position_speed_deg_s": 0.0},
        {**payload, "position_deg": True},
        {**payload, "position_speed_deg_s": True},
        {**payload, "position_deg": float("nan")},
    ):
        result = command(controller, "start", invalid)
        assert not result["success"] and controller.snapshot().state == "monitoring"
        assert not reviewed and not owner_started.is_set()
    assert command(controller, "update_form", {"recording_values": {"enabled": True}})["success"]
    result = command(controller, "start", payload)
    assert not result["success"] and "关闭录制准备" in result["message"]
    assert not reviewed and not owner_started.is_set()
    assert command(controller, "update_form", {"recording_values": {"enabled": False}})["success"]
    result = command(controller, "start", payload)
    assert result["success"], result["message"]
    assert owner_started.wait(2)
    assert len(reviewed) == 1 and list(reviewed[0].runtime.joint_names) == ["gripper"]
    assert reviewed[0].initial_mode == "position" and reviewed[0].initial_position_deg == -10.0
    assert reviewed[0].initial_position_speed_deg_s == 0.7
    assert controller.snapshot().values["position_target_deg"] == ""
    controller.stop()
    wait_for(lambda: controller.snapshot().state == "disconnected")
    assert controller.snapshot().exit_confirmed


def test_real_configuration_build_review_and_fake_owner_start_successfully(
    make_controller, session_module, complete_values
):
    reviewed, owner_started = [], threading.Event()

    def review(output, *, reviewed_config):
        document = json.loads(Path(reviewed_config).read_text(encoding="utf-8"))
        config = session_module.draccus.decode(session_module.ForceTeacherRunConfig, document["config"])
        config.validate()
        reviewed.append(config)

    def owner(output, *, reviewed_config, live):
        assert reviewed and len(reviewed[0].runtime.joint_names) == 7
        document = json.loads(Path(reviewed_config).read_text(encoding="utf-8"))
        owner_started.set()
        assert live.stop.wait(3)
        result = {
            "config": document["config"],
            "calibration": document["calibration"],
            "errors": [],
            "shutdown_evidence": {
                "transport_closed": True,
                "fault_disposition": {
                    "all_joints_disabled_observed": True,
                    "status": "submitted",
                    "error": None,
                },
            },
            "hardware_ack": "unknown",
            "simulated": True,
        }
        result_path = Path(output) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        live.close()
        return result

    controller = make_controller(
        services={
            "monitor": fake_monitor_with_fresh_reference,
            "review": review,
            "live_factory": HeadlessLive,
            "run_in_process": owner,
        },
    )
    assert command(controller, "update_form", {"values": complete_values})["success"]
    assert command(controller, "connect")["success"]
    wait_for(lambda: controller.snapshot().state == "monitoring")
    result = command(controller, "start", {"confirmed": True})
    assert result["success"], result["message"]
    assert owner_started.wait(2)
    wait_for(lambda: controller.snapshot().state == "running")
    assert controller._monitor_cleanup["transport_closed"] and not controller._monitor_thread.is_alive()
    assert reviewed[0].teacher.kp_deg_per_N == pytest.approx(0.04)
    assert reviewed[0].b601.control.profiles[-1].mode == "force_pos"
    assert reviewed[0].b601.control.command_feedback_mode == "sequential"
    controller.stop()
    wait_for(lambda: controller.snapshot().state == "disconnected")
    assert controller.snapshot().exit_confirmed


def test_atomic_start_form_failure_does_not_start_with_old_values(make_controller):
    controller = make_controller(demo=True)
    assert command(controller, "connect")["success"]
    original_port = controller.snapshot().values["port"]
    result = command(
        controller,
        "start",
        {"confirmed": True, "form": {"values": {"port": "MUST_NOT_REPLACE_CONNECTED_PORT", "target_N": "7"}}},
    )
    assert not result["success"]
    assert controller.snapshot().state == "monitoring"
    assert controller.snapshot().values["port"] == original_port
    assert controller.snapshot().values["target_N"] == "5"
    assert controller._owner_thread is None
    assert command(controller, "start", {"confirmed": True, "form": {"values": {"target_N": "6"}}})["success"]
    assert controller.snapshot().state == "running" and controller.snapshot().values["target_N"] == "6"


def test_full_arm_preflight_requires_explicit_confirmation_and_then_validates_without_motion(
    make_controller, complete_values
):
    controller = make_controller()
    assert command(controller, "update_form", {"values": complete_values})["success"]
    rejected = command(controller, "preflight", {"confirmed": False})
    assert not rejected["success"] and "确认" in rejected["message"]
    accepted = command(controller, "preflight", {"confirmed": True})
    assert accepted["success"], accepted["message"]
    assert controller.snapshot().state == "disconnected"
    assert controller._monitor_thread is controller._owner_thread is controller._live is None


@pytest.mark.parametrize(
    "fault", [None, "missing_force", "out_of_range", "stale", "cancel", "close", "retained"]
)
def test_standalone_recovery_requires_fresh_force_disabled_and_complete_close(
    make_controller, calibration_path, fault
):
    content = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_path.write_text(json.dumps({"gripper": content["gripper"]}), encoding="utf-8")
    controller = None

    def monitor(stop, publish, values, calibration, cleanup, *, retained):
        now = time.perf_counter_ns()
        row = {
            "phase": "DISABLED",
            "joint_received_ns": now,
            "force_timestamp_ns": now,
            "read_only": True,
            "joint_valid": True,
            "force_valid": True,
            "joint_status_code": 0,
            "q_measured_deg": -1,
            "force_left_N": 0.01,
            "force_right_N": 0.02,
        }
        if fault == "missing_force":
            row.pop("force_timestamp_ns")
        elif fault == "out_of_range":
            row["q_measured_deg"] = 999
        elif fault == "stale":
            row["force_timestamp_ns"] = now - 1_000_000_000
        elif fault == "retained":
            retained.append(object())
        try:
            publish(row)
            if fault == "cancel":
                controller.stop()
        finally:
            cleanup({"transport_closed": fault != "close", "sensor_closed": True})

    controller = make_controller(services={"monitor": monitor}, gripper_only=True)
    controller._last_result = {"config": {"runtime": {"joint_names": ["gripper"]}}}
    controller._exit_confirmed = False
    controller._unconfirmed_port = controller._values["port"]
    controller._unconfirmed_calibration = asdict(controller._calibration)
    controller._changed(state="recovery_required")
    result = command(controller, "recover")
    assert result["success"] is (fault is None), result["message"]
    assert controller.snapshot().exit_confirmed is (fault is None)
    proofs = list((controller._output_root / "readonly_recovery").glob("*-gripper.json"))
    assert len(proofs) == (1 if fault is None else 0)
    if fault is None:
        assert controller.snapshot().state == "disconnected" and not controller._telemetry
        evidence = json.loads(proofs[0].read_text(encoding="utf-8"))
        assert evidence["read_only"] and not evidence["motion_attempted"]
        assert evidence["hardware_ack"] == "unknown"


@pytest.mark.parametrize("history", ["unknown_exit", "malformed"])
def test_invalid_settings_do_not_erase_historical_unknown_exit_gate(make_controller, history):
    controller = make_controller(start=False)
    controller._settings_path.write_text("{malformed settings", encoding="utf-8")
    result_path = controller._output_root / "previous/run/result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        "{malformed history"
        if history == "malformed"
        else json.dumps(
            {
                "config": {"runtime": {"joint_names": ["gripper"]}, "b601": {"port": "COM22"}},
                "errors": ["unknown exit"],
                "shutdown_evidence": {"transport_closed": False},
            }
        ),
        encoding="utf-8",
    )
    controller.start()
    wait_for(lambda: controller.snapshot().state != "loading")
    assert not controller.snapshot().exit_confirmed
    assert controller.snapshot().state == "recovery_required"
    assert controller.snapshot().capabilities["start"]


@pytest.mark.parametrize("phase", ["FAULT", "RELEASE"])
def test_fault_and_release_phases_disable_new_motion_but_keep_stop_available(make_controller, phase):
    controller = make_controller(demo=True, start=False)
    controller._telemetry = {"phase": phase}
    controller._changed(state="running")
    capabilities = controller.snapshot().capabilities
    for action in ("resume", "approach", "position", "release", "apply", "target"):
        assert capabilities[action], action
    assert "freeze" not in capabilities and "step" not in capabilities
    assert not capabilities["stop"]


def test_removed_freeze_and_step_commands_do_not_change_form_or_session(make_controller):
    controller = make_controller(demo=True)
    assert command(controller, "connect")["success"]
    assert command(controller, "start", {"confirmed": True})["success"]
    target_before = controller.snapshot().values["target_N"]
    for action in ("freeze", "step"):
        result = command(controller, action, {"form": {"values": {"target_N": "999"}}, "value": 1})
        assert not result["success"] and "未知界面操作" in result["message"]
        assert controller.snapshot().values["target_N"] == target_before
        assert controller.snapshot().state == "running"


def test_frame_counters_do_not_acknowledge_pending_save_or_allow_duplicate_finish(make_controller):
    controller = make_controller(start=False)
    live = HeadlessLive(poll_s=0.01)
    controller._live = live
    controller._telemetry = dict(live.latest)
    controller._recording_values["enabled"] = True
    controller._recording = {
        "state": "recording",
        "active": True,
        "busy": False,
        "episode_uid": "episode-a",
        "frame_count": 1,
    }
    controller._changed(state="running")
    controller._record_command("record_finish", {"task_result": "unknown"})
    assert len(live.submitted) == 1
    live.recording = {**controller._recording, "frame_count": 2}
    controller._poll()
    assert controller._pending_record is not None
    assert controller.snapshot().capabilities["record_finish"]
    live.recording = {
        **controller._recording,
        "state": "saved",
        "active": False,
        "busy": False,
        "file_complete": True,
    }
    controller._poll()
    assert controller._pending_record is None
    assert len(live.submitted) == 1


def test_recording_prefill_preserves_resume_and_encoder_without_io(make_controller, tmp_path):
    controller = make_controller(start=False)
    document = {
        "root": str(tmp_path / "dataset"),
        "task": "test",
        "resume": True,
        "h264_encoder": "software",
        "camera_configs": {"front": {"index_or_path": 0, "fps": 50, "width": 640, "height": 480}},
    }
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    controller.prefill_recording(path)
    config = controller._recording_config()
    assert config["resume"] is True and config["h264_encoder"] == "software"
    assert config["camera_configs"] == document["camera_configs"]
    assert path.read_bytes() == original and not Path(document["root"]).exists()
