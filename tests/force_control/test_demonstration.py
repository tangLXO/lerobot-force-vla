"""Unified demonstration evidence, bounded handoff and recording lifecycle regressions."""

import copy
import json
import queue
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.force_control import demonstration as demo
from lerobot.force_control.fake_hardware import FakePlant
from lerobot.force_control.hardware_types import JointFeedback, JointSnapshot
from lerobot.force_control.owner import OwnerService, OwnerSnapshot, OwnerSubscription
from lerobot.utils.import_utils import _av_available, _datasets_available
from tests.force_control.hardware_helpers import ManualSensor

_RECORDING_AVAILABLE = _av_available and _datasets_available
requires_recording = pytest.mark.skipif(
    not _RECORDING_AVAILABLE, reason="Asynchronous recording requires lerobot[dataset]"
)


def record_config(tmp_path, **changes):
    return demo.TeacherRecordConfig(
        root=str(tmp_path / "dataset"),
        task="抓取并放下",
        fps=30,
        camera_configs={"front": {"index_or_path": 0, "fps": 30, "width": 16, "height": 16}},
        **changes,
    )


def snapshot(tick=0, *, anchor=1_000_000_000, sequence=1):
    joints = JointSnapshot(
        tuple(
            JointFeedback(name, index + 0.25, True, anchor - 1000, tick + 1, 1000, "fake")
            for index, name in enumerate(demo.JOINT_NAMES)
        ),
        anchor - 100,
        anchor,
    )
    receipts = [
        {
            "name": name,
            "candidate_deg": 100 + index,
            "clipped_deg": index + 0.5,
            "submitted_deg": index + 0.5,
            "status": "success",
            "send_start_ns": anchor + 10,
            "send_end_ns": anchor + 20,
            "hardware_ack": "unknown",
            "error": None,
        }
        for index, name in enumerate(demo.JOINT_NAMES)
    ]
    row = {
        "tick_id": tick,
        "tick_start_ns": anchor - 2000,
        "decision_anchor_ns": anchor,
        "epoch": 0,
        "phase": "FOLLOW",
        "phase_at_tick_start": "FOLLOW",
        "send_status": "submitted",
        "feedback_valid": True,
        "force_valid": True,
        "force_sequence": sequence,
        "force_timestamp_ns": anchor - 500,
        "force_arrival_timestamp_ns": anchor - 200,
        "force_max_age_ms": 100,
        "joint_feedback_max_age_ms": 100,
        "force_left_N": 0.5,
        "force_right_N": 0.75,
        "send_start_ns": anchor + 10,
        "send_end_ns": anchor + 20,
        "per_joint_results": receipts,
        "joint_snapshot": asdict(joints),
        "arm_intent": {f"{name}.pos": 100 + index for index, name in enumerate(demo.JOINT_NAMES)},
        "submitted_action": {f"{item['name']}.pos": item["submitted_deg"] for item in receipts},
        "label_valid": False,
        "action_label_valid": True,
        "capture_stage": "approach",
        "failure_reason": None,
    }
    return OwnerSnapshot(
        row, joints, tuple((item["name"], item["submitted_deg"]) for item in receipts), 0, "TEACHER", tick
    )


def camera_buffer(anchor=1_000_000_000):
    buffer = demo.CausalCameraBuffer(4)
    buffer.publish(np.full((16, 16, 3), 37, dtype=np.uint8), anchor - 100, anchor - 50)
    return buffer


def test_frame_uses_submitted_clipped_action_and_same_tick_force(tmp_path):
    completed = snapshot()
    completed.row.update(force_hardware_sequence=47, force_status="driver_calibrated")
    frame, capture, evidence = demo.completed_tick_frame(
        completed, {"front": camera_buffer()}, record_config(tmp_path), frame_index=4
    )
    np.testing.assert_array_equal(frame["action"], np.arange(7, dtype=np.float32) + 0.5)
    np.testing.assert_array_equal(frame["observation.state"], np.arange(7, dtype=np.float32) + 0.25)
    np.testing.assert_array_equal(frame["observation.tactile"], [0.5, 0.75])
    assert "sensor_windows" not in frame
    assert evidence["frame_index"] == 4 and evidence["control_tick_id"] == completed.tick_id
    assert capture["sensors"]["gripper_force"]["sequence"] == completed.row["force_sequence"]
    assert capture["sensors"]["gripper_force"]["hardware_sequence"] == 47
    assert capture["sensors"]["gripper_force"]["status"] == "driver_calibrated"
    assert evidence["action_deg"] != list(completed.row["arm_intent"].values())


@pytest.mark.parametrize(
    "failure", ["future_force", "late_force", "stale_force", "stale_joint", "missing_joint", "unknown_send"]
)
def test_frame_rejects_noncausal_stale_or_unknown_evidence(tmp_path, failure):
    completed = snapshot()
    if failure == "future_force":
        completed.row["force_timestamp_ns"] = completed.row["decision_anchor_ns"] + 1
    elif failure == "late_force":
        completed.row["force_arrival_timestamp_ns"] = completed.row["decision_anchor_ns"] + 1
    elif failure == "stale_force":
        completed.row["force_timestamp_ns"] -= 101_000_000
    elif failure == "stale_joint":
        first, *rest = completed.joint_snapshot.joints
        completed = replace(
            completed,
            joint_snapshot=replace(
                completed.joint_snapshot, joints=(replace(first, receive_timestamp_ns=1), *rest)
            ),
        )
    elif failure == "missing_joint":
        completed = replace(
            completed,
            joint_snapshot=replace(completed.joint_snapshot, joints=completed.joint_snapshot.joints[:-1]),
        )
    else:
        completed.row["per_joint_results"][2]["status"] = "unknown"
    with pytest.raises(ValueError):
        demo.completed_tick_frame(completed, {"front": camera_buffer()}, record_config(tmp_path))


def test_camera_history_is_bounded_and_rejects_late_future_and_stale_frames():
    buffer = demo.CausalCameraBuffer(2)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    buffer.publish(image, 100, 120)
    buffer.publish(image + 1, 150, 250)
    buffer.publish(image + 2, 300, 310)
    assert len(buffer._frames) == 2
    with pytest.raises(TimeoutError):
        buffer.select(200, 1)
    assert buffer.select(260, 1).timestamp_ns == 150
    with pytest.raises(TimeoutError):
        buffer.select(2_000_000, 1)


@pytest.mark.parametrize("duplicate", ["joint", "receipt"])
def test_frame_rejects_duplicate_joint_or_submission_entries(tmp_path, duplicate):
    completed = snapshot()
    if duplicate == "joint":
        completed = replace(
            completed,
            joint_snapshot=replace(
                completed.joint_snapshot,
                joints=(*completed.joint_snapshot.joints, completed.joint_snapshot.joints[0]),
            ),
        )
    else:
        completed.row["per_joint_results"].append(copy.deepcopy(completed.row["per_joint_results"][0]))
    with pytest.raises(ValueError):
        demo.completed_tick_frame(completed, {"front": camera_buffer()}, record_config(tmp_path))


@pytest.mark.parametrize("cameras", [{}, {"unexpected": camera_buffer()}])
def test_frame_requires_every_configured_camera(tmp_path, cameras):
    with pytest.raises(ValueError, match="camera"):
        demo.completed_tick_frame(snapshot(), cameras, record_config(tmp_path))


def test_owner_subscription_overflow_does_not_block_or_corrupt_owner():
    completed = snapshot()
    runtime = SimpleNamespace(
        latest_joint_snapshot=completed.joint_snapshot,
        last_submitted=dict(completed.last_submitted),
        epoch=0,
        mode="TEACHER",
    )
    owner = OwnerService(runtime)
    first, second = owner.subscribe(1), owner.subscribe(2)
    owner._publish(completed.row)
    first.get().row["per_joint_results"][0]["submitted_deg"] = -999
    assert owner.snapshot().row["per_joint_results"][0]["submitted_deg"] == 0.5
    assert second.get().row["per_joint_results"][0]["submitted_deg"] == 0.5
    owner._publish(completed.row)
    owner._publish(completed.row)
    with pytest.raises(RuntimeError, match="overflow"):
        first.check_health()
    assert owner.snapshot().tick_id == completed.tick_id
    owner.unsubscribe(first)
    assert first.closed


class StubOwner:
    """A cache/subscription source with trap motor operations."""

    def __init__(self):
        self.runtime = SimpleNamespace(config=SimpleNamespace(joint_names=demo.JOINT_NAMES, control_hz=100))
        self.is_running = True
        self.subscriptions = []
        self.post = Mock(side_effect=AssertionError("recording must not issue motion commands"))
        self.stop = Mock(side_effect=AssertionError("recording must not stop motor ownership"))

    def subscribe(self, capacity):
        item = OwnerSubscription(capacity)
        self.subscriptions.append(item)
        return item

    def unsubscribe(self, item):
        self.subscriptions.remove(item)
        item.closed = True

    def emit(self, item):
        for subscription in self.subscriptions:
            try:
                subscription.queue.put_nowait(copy.deepcopy(item))
            except queue.Full:
                subscription.error = "Completed control-tick queue overflow; demonstration is incomplete"


class StubCameras:
    def __init__(self, config):
        self.buffers = {name: demo.CausalCameraBuffer() for name in config.camera_configs}
        self.closed = False

    def start(self):
        pass

    def check_health(self):
        pass

    def close(self):
        self.closed = True


class StubDataset:
    @classmethod
    def create(cls, **kwargs):
        instance = cls()
        instance.root = Path(kwargs["root"])
        instance.root.mkdir(parents=True)
        instance.features, instance.fps = kwargs["features"], kwargs["fps"]
        instance.meta = SimpleNamespace(total_episodes=0)
        instance.frames = []
        instance.finalized = False
        return instance

    def prepare_episode_recording(self):
        pass

    def add_frame(self, frame):
        self.frames.append(frame)

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True


class StubRecorder:
    def __init__(self, root, sensors):
        self.active = False
        self._transaction = None
        self._subscriptions = {name: SimpleNamespace(start_sequence=0) for name in sensors}
        self.closed = False

    @property
    def is_active(self):
        return self.active

    def start_episode(self, *args, **kwargs):
        self.active = True

    def wait_until_ready(self):
        pass

    def check_health(self):
        pass

    def record_sync(self, *args):
        pass

    def prepare_episode(self, **kwargs):
        self.active = False

    def commit_prepared(self, dataset):
        dataset.meta.total_episodes += 1
        dataset.clear_episode_buffer()

    def abort_episode(self, reason):
        self.active = False

    def abort_prepared(self, reason):
        self._transaction = None

    def close(self):
        self.closed = True


def wait_status(controller, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = controller.status()
        if predicate(status):
            return status
        if status["state"] == "error":
            pytest.fail(f"Recorder failed: {status}")
        time.sleep(0.005)
    pytest.fail(f"Recorder did not reach expected state: {controller.status()}")


def publish_frame(controller, owner, tick=0, *, anchor=1_000_000_000, sample=None):
    completed = snapshot(tick, anchor=anchor, sequence=tick + 1)
    if sample is not None:
        completed.row.update(
            force_sequence=sample.sequence,
            force_timestamp_ns=sample.timestamp_ns,
            force_arrival_timestamp_ns=sample.arrival_timestamp_ns,
            force_left_N=sample.values["left.normal_force"],
            force_right_N=sample.values["right.normal_force"],
        )
    for buffer in controller._cameras.buffers.values():
        buffer.publish(np.full((16, 16, 3), 40 + tick, np.uint8), anchor - 100, anchor - 50)
    owner.emit(completed)
    return completed


@pytest.fixture
def stub_recording(tmp_path, monkeypatch):
    if not _RECORDING_AVAILABLE:
        pytest.skip("Asynchronous recording requires lerobot[dataset]")
    from lerobot.force_control import demonstration_validation

    monkeypatch.setattr(
        demonstration_validation, "validate_demonstration", lambda *a, **k: {"valid": True, "errors": []}
    )
    owner = StubOwner()
    controller = demo.DemonstrationController(
        owner,
        object(),
        camera_factory=StubCameras,
        dataset_factory=StubDataset,
        recorder_factory=StubRecorder,
    )
    config = record_config(tmp_path)
    yield controller, owner, config
    controller.close()


def begin(stub):
    controller, _, config = stub
    controller.submit("record_start", {"config": asdict(config)})
    return wait_status(controller, lambda status: status["state"] == "recording")


def test_manual_finish_saves_without_release_or_other_motor_commands(stub_recording):
    controller, owner, config = stub_recording
    begin(stub_recording)
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    controller.submit("record_finish", {"task_result": "success"})
    status = wait_status(controller, lambda status: status["state"] == "saved")
    assert status["file_complete"] and status["successful_episode_count"] == 1
    assert not status["normal_release_completed"]
    owner.post.assert_not_called()
    owner.stop.assert_not_called()
    metadata = json.loads(
        next((Path(config.root) / "meta/teacher_episodes").glob("*.json")).read_text(encoding="utf-8")
    )
    assert metadata["complete"] and metadata["task_result"] == "success"


def test_discard_and_rerecord_uses_new_uid_without_committing_discard(stub_recording):
    controller, owner, config = stub_recording
    initial = begin(stub_recording)["episode_uid"]
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    controller.submit("record_discard", {"rerecord": True})
    current = wait_status(
        controller, lambda status: status["state"] == "recording" and status["episode_uid"] != initial
    )
    assert current["episode_count"] == 0 and current["frame_count"] == 0
    metadata = json.loads(
        (Path(config.root) / "meta/teacher_episodes" / f"{initial}.json").read_text(encoding="utf-8")
    )
    assert not metadata["complete"] and metadata["status"] == "discarded"
    publish_frame(controller, owner, tick=3, anchor=1_040_000_000)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    controller.submit("record_finish")
    wait_status(controller, lambda status: status["state"] == "saved")
    assert controller._dataset.meta.total_episodes == 1


def test_stop_marks_active_demonstration_interrupted(stub_recording):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    dataset = controller._dataset
    controller.close()
    metadata = json.loads(
        (Path(config.root) / "meta/teacher_episodes" / f"{uid}.json").read_text(encoding="utf-8")
    )
    assert not metadata["complete"] and metadata["status"] == "interrupted"
    assert not owner.subscriptions and dataset.meta.total_episodes == 0


def test_writer_failure_preserves_incomplete_evidence(stub_recording, monkeypatch):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    monkeypatch.setattr(controller._dataset, "add_frame", Mock(side_effect=OSError("disk full")))
    publish_frame(controller, owner)
    status = wait_status(controller, lambda status: status["state"] == "error")
    assert "disk full" in status["error"] and not status["file_complete"]
    metadata = json.loads(
        (Path(config.root) / "meta/teacher_episodes" / f"{uid}.json").read_text(encoding="utf-8")
    )
    assert not metadata["complete"] and metadata["status"] == "error"
    assert metadata["files"]["control"]["rows"] == 1
    assert not owner.subscriptions


def test_resume_refuses_main_episodes_without_teacher_attachments(stub_recording, tmp_path):
    controller = stub_recording[0]
    root = tmp_path / "orphaned"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="lack complete Teacher"):
        controller._check_resume(root)


def test_owner_fault_row_is_retained_and_recording_cannot_be_complete(stub_recording):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    failed = snapshot(1, anchor=1_010_000_000)
    failed.row.update(phase="FAULT", send_status="unknown", failure_reason="stale_force")
    owner.emit(failed)
    wait_status(controller, lambda status: status["state"] == "error")
    root = Path(config.root)
    rows = [
        json.loads(line)
        for line in (root / "teacher" / uid / "control.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2 and rows[-1]["failure_reason"] == "stale_force"
    metadata = json.loads((root / "meta/teacher_episodes" / f"{uid}.json").read_text(encoding="utf-8"))
    assert not metadata["complete"] and metadata["frame_count"] == 1


def test_capture_queue_fault_invalidates_episode_and_keeps_queued_rows(stub_recording):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    subscription = owner.subscriptions[0]
    subscription.queue.put_nowait(snapshot())
    subscription.error = "Completed control-tick queue overflow; demonstration is incomplete"
    status = wait_status(controller, lambda status: status["state"] == "error")
    assert "overflow" in status["error"] and not status["file_complete"]
    metadata = json.loads(
        (Path(config.root) / "meta/teacher_episodes" / f"{uid}.json").read_text(encoding="utf-8")
    )
    assert not metadata["complete"] and metadata["files"]["control"]["rows"] == 1
    assert not owner.subscriptions


@requires_recording
def test_abort_timeout_retains_active_dataset_and_sensor_dependencies(tmp_path, monkeypatch):
    class RetainedRecorder(StubRecorder):
        allow_cleanup = False

        def abort_episode(self, reason):
            if not self.allow_cleanup:
                raise TimeoutError("injected raw writer drain timeout")
            super().abort_episode(reason)

        def close(self):
            if not self.allow_cleanup:
                raise TimeoutError("injected raw writer remains alive")
            self.abort_episode("close")
            super().close()

    owner = StubOwner()
    controller = demo.DemonstrationController(
        owner,
        object(),
        camera_factory=StubCameras,
        dataset_factory=StubDataset,
        recorder_factory=RetainedRecorder,
    )
    config = record_config(tmp_path)
    try:
        controller.submit("record_start", {"config": asdict(config)})
        wait_status(controller, lambda status: status["state"] == "recording")
        publish_frame(controller, owner)
        wait_status(controller, lambda status: status["frame_count"] == 1)
        dataset, recorder, uid = controller._dataset, controller._recorder, controller._uid
        controller.submit("record_discard", {"rerecord": False})
        wait_status(controller, lambda status: status["state"] == "error")
        assert len(dataset.frames) == 1 and not dataset.finalized
        assert controller._uid == uid and controller._cleanup_pending
        with pytest.raises(RuntimeError, match="resources remain retained"):
            controller.close(timeout_s=1)
        assert controller.has_resources and controller._dataset is dataset
        assert len(dataset.frames) == 1 and not dataset.finalized
        recorder.allow_cleanup = True
        controller.close(timeout_s=1)
        assert dataset.finalized and not controller.has_resources
    finally:
        if controller._recorder is not None:
            controller._recorder.allow_cleanup = True
        controller.close(timeout_s=1)


@pytest.mark.parametrize("kind", ["record_start", "record_outcome"])
def test_invalid_request_does_not_abort_an_active_demonstration(stub_recording, kind):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    value = {"config": asdict(config)} if kind == "record_start" else {"task_result": "invalid"}
    controller.submit(kind, value)
    status = wait_status(controller, lambda status: bool(status.get("request_error")))
    assert status["state"] == "recording" and status["active"] and controller._uid == uid
    assert owner.subscriptions and controller._recorder.is_active
    publish_frame(controller, owner, tick=1, anchor=1_040_000_000)
    wait_status(controller, lambda status: status["frame_count"] == 2)
    controller.submit("record_finish", {"task_result": "success"})
    wait_status(controller, lambda status: status["state"] == "saved")


def test_duplicate_finish_preserves_saved_completion_and_result(stub_recording):
    controller, owner, config = stub_recording
    uid = begin(stub_recording)["episode_uid"]
    publish_frame(controller, owner)
    wait_status(controller, lambda status: status["frame_count"] == 1)
    controller.submit("record_finish", {"task_result": "success"})
    wait_status(controller, lambda status: status["state"] == "saved")
    path = Path(config.root) / "meta/teacher_episodes" / f"{uid}.json"
    original = path.read_bytes()
    controller.submit("record_finish", {"task_result": "failure"})
    status = wait_status(controller, lambda status: bool(status.get("request_error")))
    assert status["state"] == "saved" and status["file_complete"]
    assert status["task_result"] == "success" and status["successful_episode_count"] == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("dataset_cleanup_fails", [False, True])
@requires_recording
def test_prepare_failure_retains_each_resource_with_unconfirmed_cleanup(tmp_path, dataset_cleanup_fails):
    class Cameras(StubCameras):
        instances = []

        def __init__(self, config):
            super().__init__(config)
            self.allow_cleanup = False
            self.instances.append(self)

        def close(self):
            if not self.allow_cleanup:
                raise RuntimeError("camera disconnect unconfirmed")
            super().close()

    class Dataset(StubDataset):
        instances = []

        @classmethod
        def create(cls, **kwargs):
            result = super().create(**kwargs)
            result.allow_cleanup = not dataset_cleanup_fails
            cls.instances.append(result)
            return result

        def finalize(self):
            if not self.allow_cleanup:
                raise RuntimeError("dataset finalize unconfirmed")
            super().finalize()

    def failing_recorder(*args, **kwargs):
        raise OSError("cannot create Sensor writer")

    controller = demo.DemonstrationController(
        StubOwner(),
        object(),
        camera_factory=Cameras,
        dataset_factory=Dataset,
        recorder_factory=failing_recorder,
    )
    try:
        with pytest.raises(ExceptionGroup, match="preparation and cleanup failed") as captured:
            controller._prepare(record_config(tmp_path))
        assert isinstance(captured.value.exceptions[0], OSError)
        assert "Sensor writer" in str(captured.value.exceptions[0])
        assert any("camera disconnect" in str(error) for error in captured.value.exceptions[1:])
        dataset, cameras = Dataset.instances[0], Cameras.instances[0]
        assert controller._cleanup_pending and controller._cameras is cameras
        assert dataset.finalized is (not dataset_cleanup_fails)
        if dataset_cleanup_fails:
            assert controller._dataset is dataset
        with pytest.raises(RuntimeError, match="resources remain retained"):
            controller.close(timeout_s=1)
        assert controller.has_resources
        dataset.allow_cleanup = cameras.allow_cleanup = True
        controller.close(timeout_s=1)
        assert dataset.finalized and cameras.closed and not controller.has_resources
    finally:
        for instance in Dataset.instances:
            instance.allow_cleanup = True
        for instance in Cameras.instances:
            instance.allow_cleanup = True
        controller.close(timeout_s=1)


@requires_recording
def test_real_dataset_video_and_sidecar_roundtrip(tmp_path):
    from lerobot.datasets.sensor_window import SensorStreamReader
    from lerobot.force_control.demonstration_validation import validate_demonstration

    owner = StubOwner()
    sensor = ManualSensor(FakePlant())

    controller = demo.DemonstrationController(owner, sensor, camera_factory=StubCameras)
    config = record_config(tmp_path)
    try:
        controller.submit("record_start", {"config": asdict(config)})
        # Required Sensor startup waits for its first post-subscription sample.
        deadline = time.monotonic() + 10
        while not controller.status()["active"]:
            sensor.publish_once()
            if controller.status()["state"] == "error":
                pytest.fail(str(controller.status()))
            assert time.monotonic() < deadline, controller.status()
            time.sleep(0.005)
        uid = controller.status()["episode_uid"]
        for tick in range(4):
            sample = sensor.publish_once()
            anchor = time.perf_counter_ns()
            publish_frame(controller, owner, tick, anchor=anchor, sample=sample)
            wait_status(controller, lambda status, expected=tick + 1: status["frame_count"] == expected)
            time.sleep(0.035)
        controller.submit("record_finish", {"task_result": "success"})
        status = wait_status(controller, lambda status: status["state"] == "saved", timeout=20)
        assert status["file_complete"] and status["episode_count"] == 1
    finally:
        controller.close()
    report = validate_demonstration(config.root, uid)
    assert report["valid"], report["errors"]
    assert report["episodes"][0]["video_decoded"]
    reader = SensorStreamReader(config.root, instance="gripper_force", episode_uid=uid, verify="full")
    raw, sync = reader.read_raw(), reader.read_sync()
    assert len(sync) == 4 and len(raw) >= 4
    sequences = {row["sequence"] for row in raw}
    assert all(row["sensors"]["gripper_force"]["sequence"] in sequences for row in sync)
    assert not (Path(config.root) / ".sensor-writer.lock").exists()
