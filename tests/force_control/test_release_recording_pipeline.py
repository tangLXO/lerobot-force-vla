"""Real Dataset/video/Sidecar completion with fake robot, force and RGB acquisition only."""

import importlib
import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

pytest.importorskip("av")
pytest.importorskip("datasets")
pytest.importorskip("pyarrow")

from lerobot.force_control.demonstration import DemonstrationController, TeacherRecordConfig
from lerobot.force_control.demonstration_validation import validate_demonstration
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES
from lerobot.scripts import lerobot_force_teacher as runner
from tests.force_control.helpers import run_config


@pytest.mark.parametrize("scenario", ["release", "release_leader_loss", "manual_finish", "stop_during_save"])
def test_real_files_finish_after_owner_quiesce_and_respect_user_stop(monkeypatch, tmp_path, scenario):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    synthetic = importlib.import_module("demonstration_smoke_test")
    audit = {"owner_closed": False, "owner_closed_at_finish": None, "leader_loss_injected": False}
    finishing = threading.Event()
    config = run_config(tmp_path, armed=True)
    config.initial_mode, config.leader_backend = "hybrid", "fake"
    config.runtime = replace(config.runtime, joint_names=FAKE_B601_JOINT_NAMES, control_hz=50)
    config.teacher = replace(config.teacher, gripper_follow_speed_deg_s=5, contact_force_threshold_N=0.1)
    config.duration_s, config.shutdown_timeout_s = None, 10
    config.recording = TeacherRecordConfig(
        root=str(tmp_path / "dataset"),
        task="Synthetic asynchronous save lifecycle",
        fps=50,
        h264_encoder="software",
        camera_configs={"front": {"index_or_path": 0, "fps": 60, "width": 64, "height": 48}},
        queue_capacity=512,
    )
    original_stop, original_leader = runner.OwnerService.stop, runner._leader_worker

    def controller_factory(owner, sensor, **kwargs):
        controller = DemonstrationController(
            owner, sensor, camera_factory=synthetic.SyntheticCameras, **kwargs
        )
        original_finish = controller._finish

        def delayed_finish(task_result):
            audit["owner_closed_at_finish"] = not owner.has_resources and not owner.is_running
            if audit["owner_closed_at_finish"]:
                audit["owner_closed"] = True
            finishing.set()
            # Real asynchronous work must remain valid after the runner next polls.
            time.sleep(0.2)
            return original_finish(task_result)

        controller._finish = delayed_finish
        return controller

    def stop(owner, *args, **kwargs):
        result = original_stop(owner, *args, **kwargs)
        if str(owner.runtime.controller.phase) == "DISABLED" and not owner.has_resources:
            audit["owner_closed"] = True
        return result

    def leader_factory(configuration):
        worker = original_leader(configuration)
        original_latest = worker.latest

        def latest(*args, **kwargs):
            intent = original_latest(*args, **kwargs)
            if scenario == "release_leader_loss" and audit["owner_closed"]:
                audit["leader_loss_injected"] = True
                return replace(intent, is_valid=False, error="injected_leader_loss_after_owner_close")
            return intent

        worker.latest = latest
        return worker

    monkeypatch.setattr(runner, "DemonstrationController", controller_factory)
    monkeypatch.setattr(runner.OwnerService, "stop", stop)
    monkeypatch.setattr(runner, "_leader_worker", leader_factory)

    class Live:
        def __init__(self):
            self.stop = threading.Event()
            self.latest = self.applied = self.startup = self.recording = self.preview = None
            self.started = self.finished = False
            self.deadline = time.monotonic() + 60

        def take(self):
            if time.monotonic() > self.deadline:
                self.stop.set()
                raise TimeoutError("Synthetic pipeline did not finish")
            if self.recording and self.recording.get("state") == "error":
                self.stop.set()
                raise RuntimeError(self.recording.get("error"))
            if not self.started:
                self.started = True
                return "record_start", {"config": asdict(config.recording)}
            if (
                self.recording
                and self.recording.get("state") == "recording"
                and self.recording.get("frame_count", 0) >= 12
                and not self.finished
            ):
                self.finished = True
                if scenario == "manual_finish":
                    return "record_finish", {"task_result": "unknown"}
                return "release_and_finish", None
            if scenario == "manual_finish" and self.recording and self.recording.get("state") == "saved":
                self.stop.set()
            return None

    live = Live()
    stopper = None
    if scenario == "stop_during_save":

        def interrupt():
            if finishing.wait(30):
                live.stop.set()

        stopper = threading.Thread(target=interrupt, name="synthetic-user-stop", daemon=True)
        stopper.start()
    result = runner.run(config, live=live)
    if stopper is not None:
        stopper.join(1)
    report = validate_demonstration(tmp_path / "dataset")
    if scenario == "stop_during_save":
        assert audit["owner_closed_at_finish"] is True
        assert report["valid"] is False and not result["recording"]["file_complete"]
        return
    assert report["valid"], {"verification": report, "recording": result["recording"]}
    assert result["recording"]["file_complete"] is True and len(report["episodes"]) == 1
    episode = report["episodes"][0]
    assert episode["video_decoded"] is True and episode["files_complete"] is True
    manifest = next((tmp_path / "dataset/meta/teacher_episodes").glob("*.json"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["complete"] is True and metadata["status"] == "complete"
    assert audit["owner_closed_at_finish"] is (scenario != "manual_finish")
    assert metadata["capture_summary"]["normal_release_completed"] is (scenario != "manual_finish")
    if scenario == "release_leader_loss":
        assert audit["leader_loss_injected"] is True
