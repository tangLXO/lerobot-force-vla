"""Recording dispatch and supervised configuration stay outside motor commands."""

import importlib
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import draccus
import pytest

from lerobot.force_control.demonstration import TeacherRecordConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES
from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.scripts import lerobot_force_teacher as runner
from tests.force_control.helpers import run_config


def recording_config(tmp_path):
    return TeacherRecordConfig(
        root=str(tmp_path / "demonstrations"),
        task="拿起并放下",
        camera_configs={"front": {"index_or_path": 0, "fps": 50, "width": 64, "height": 64}},
    )


def hybrid_config(tmp_path):
    config = run_config(tmp_path, armed=True)
    config.initial_mode, config.leader_backend = "hybrid", "fake"
    config.runtime = replace(config.runtime, joint_names=FAKE_B601_JOINT_NAMES, control_hz=50)
    config.teacher = replace(config.teacher, gripper_follow_speed_deg_s=5, contact_force_threshold_N=0.1)
    config.recording = recording_config(tmp_path)
    return config


class Requests:
    def __init__(self, events):
        self.stop = threading.Event()
        self.events = iter(events)
        self.latest = self.applied = self.startup = self.recording = self.preview = None

    def take(self):
        try:
            return next(self.events)
        except StopIteration:
            self.stop.set()
            return None


class Capture:
    instances = []

    def __init__(self, owner, sensor, **kwargs):
        self.owner, self.sensor = owner, sensor
        self.events = []
        self.owner_open_during_events = []
        self.closed = False
        self.closed_after_owner = False
        type(self).instances.append(self)

    @property
    def has_resources(self):
        return not self.closed

    def status(self):
        return {
            "state": "closed" if self.closed else "ready",
            "active": False,
            "received_events": len(self.events),
            "file_complete": False,
        }

    def preview(self):
        return {}

    def prepare_before_control(self, config):
        assert not self.owner.has_resources and not self.owner.is_running
        self.prepared = True

    def submit(self, kind, value):
        self.events.append((kind, value))
        self.owner_open_during_events.append(self.owner.has_resources)

    def close(self, timeout_s):
        self.closed_after_owner = not self.owner.has_resources
        self.closed = True


def observe_commands(monkeypatch):
    events = []
    original = runner.OwnerService.post

    def post(self, kind, value=None):
        events.append((kind, value))
        assert not kind.startswith("record_"), "record event escaped into hardware owner's command mailbox"
        return original(self, kind, value)

    monkeypatch.setattr(runner.OwnerService, "post", post)
    return events


def test_recording_events_dispatch_to_capture_and_stop_preserves_final_status(tmp_path, monkeypatch):
    Capture.instances = []
    monkeypatch.setattr(runner, "DemonstrationController", Capture)
    motor_events = observe_commands(monkeypatch)
    events = [
        ("record_start", None),
        ("record_finish", {"task_result": "success"}),
        ("record_discard", {"rerecord": True}),
        ("record_outcome", {"task_result": "failure"}),
    ]
    config = hybrid_config(tmp_path)
    live = Requests(events)
    result = runner.run(config, live=live)
    capture = Capture.instances[0]
    assert [kind for kind, _ in capture.events] == [kind for kind, _ in events]
    assert capture.events[0][1] == {"config": asdict(config.recording)}
    assert all(capture.owner_open_during_events)
    assert [kind for kind, _ in motor_events] == ["arm", "start"]
    assert capture.closed and capture.closed_after_owner
    assert capture.prepared
    assert result["recording"] == live.recording == capture.status()
    assert result["recording"]["received_events"] == 4


def test_unconfigured_recording_event_cannot_reach_motor_owner(tmp_path, monkeypatch):
    motor_events = observe_commands(monkeypatch)
    live = Requests([("record_start", None)])
    config = run_config(tmp_path, armed=True)
    runner.run(config, live=live)
    assert live.recording["state"] == "error"
    assert "seven-joint" in live.recording["error"]
    assert [kind for kind, _ in motor_events] == ["arm", "start"]


@pytest.mark.parametrize("outcome", ["failure", "stop"])
def test_capture_preparation_failure_or_stop_never_opens_motor_owner(tmp_path, monkeypatch, outcome):
    live = Requests([])

    class FailedPreparation(Capture):
        instances = []

        def prepare_before_control(self, config):
            super().prepare_before_control(config)
            if outcome == "failure":
                raise RuntimeError("injected camera initialization failure")
            live.stop.set()

    monkeypatch.setattr(runner, "DemonstrationController", FailedPreparation)
    motor_events = observe_commands(monkeypatch)
    with pytest.raises(RuntimeError, match="initialization failure|user_stop_before_owner_start"):
        runner.run(hybrid_config(tmp_path), live=live)
    capture = FailedPreparation.instances[0]
    assert capture.owner._thread is None and motor_events == []
    assert capture.closed and not capture.sensor.has_resources


def test_full_arm_explicit_release_saves_capture_only_after_disabled(tmp_path, monkeypatch):
    class FinishingCapture(Capture):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.saved = False

        def status(self):
            return {
                "state": "saved" if self.saved else "recording",
                "active": not self.saved,
                "file_complete": self.saved,
                "task_result": "unknown",
            }

        def submit(self, kind, value):
            super().submit(kind, value)
            assert kind == "record_finish"
            assert str(self.owner.runtime.controller.phase) == "DISABLED"
            assert not self.owner.has_resources, "Saving must not leave released drives waiting for encoding"
            self.saved = True

    monkeypatch.setattr(runner, "DemonstrationController", FinishingCapture)
    motor_events = observe_commands(monkeypatch)
    config = hybrid_config(tmp_path)
    live = Requests([("release_and_finish", None)])
    report = runner.run(config, live=live)
    capture = FinishingCapture.instances[0]
    assert capture.events == [("record_finish", {"task_result": "unknown"})]
    assert report["recording"]["file_complete"]
    assert [kind for kind, _ in motor_events] == ["arm", "start", "release"]
    assert capture.closed_after_owner


def test_unconfirmed_capture_shutdown_retains_sensor_until_retry(tmp_path, monkeypatch):
    class RetainedCapture(Capture):
        instances = []
        release_cleanup = False

        def close(self, timeout_s):
            if not self.release_cleanup:
                raise TimeoutError("injected encoder cleanup timeout")
            super().close(timeout_s)

    monkeypatch.setattr(runner, "DemonstrationController", RetainedCapture)
    monkeypatch.setattr(runner, "_UNSTOPPED_BENCHES", [])
    config = hybrid_config(tmp_path)
    live = Requests([])
    try:
        with pytest.raises(BaseExceptionGroup, match="cleanup evidence"):
            runner.run(config, live=live)
        capture = RetainedCapture.instances[0]
        assert len(runner._UNSTOPPED_BENCHES) == 1
        assert not capture.owner.has_resources and capture.sensor.has_resources
        assert runner.finalize_retained_benches() == []
        capture.release_cleanup = True
        recovered = runner.finalize_retained_benches()
        assert len(recovered) == 1 and not recovered[0]["valid"]
        assert not capture.sensor.has_resources and not runner._UNSTOPPED_BENCHES
    finally:
        for capture in RetainedCapture.instances:
            capture.release_cleanup = True
        runner.finalize_retained_benches()


@pytest.fixture
def supervised(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("b601_supervised_bench")


def test_supervisor_decodes_draft_recording_before_runtime_validation(supervised, calibration_file, tmp_path):
    calibration = B601GripperCalibration.load(calibration_file)
    config = hybrid_config(tmp_path)
    config.recording = None
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {
                "calibration": asdict(calibration),
                "config": draccus.encode(config),
                "recording": asdict(recording_config(tmp_path)),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _, decoded = supervised.load_reviewed(path)
    assert isinstance(decoded.recording, TeacherRecordConfig)
    assert decoded.recording.task == "拿起并放下" and decoded.recording.fps == 50
    assert decoded.runtime.joint_names == FAKE_B601_JOINT_NAMES


def test_supervisor_rejects_recording_schema_before_runtime(supervised, calibration_file, tmp_path):
    calibration = B601GripperCalibration.load(calibration_file)
    config = hybrid_config(tmp_path)
    config.recording = None
    path = tmp_path / "invalid_recording.json"
    invalid = asdict(recording_config(tmp_path))
    invalid["fps"] = 0
    path.write_text(
        json.dumps(
            {
                "calibration": asdict(calibration),
                "config": draccus.encode(config),
                "recording": invalid,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fps"):
        supervised.load_reviewed(path)


def test_supervisor_saved_result_includes_final_recording_status(
    supervised, panel, calibration, settings, tmp_path, monkeypatch
):
    config = panel.build_config(*settings, calibration, tmp_path / "trace")
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {
                "config": draccus.encode(config),
                "calibration": asdict(calibration),
                "executable": True,
                "consumed": False,
            }
        ),
        encoding="utf-8",
    )
    live = Requests([])
    live.recording = {"state": "closed", "file_complete": False, "error": "interrupted"}
    monkeypatch.setattr(supervised, "run_runtime", lambda *args, **kwargs: {"valid": False, "errors": []})
    result = supervised.run(tmp_path / "run", execute=True, reviewed_config=path, live=live)
    saved = json.loads((tmp_path / "run/result.json").read_text(encoding="utf-8"))
    assert saved["recording"] == result["recording"] == live.recording
