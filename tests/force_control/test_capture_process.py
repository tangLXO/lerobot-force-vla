"""Windows-spawn capture boundaries and failure isolation, without hardware."""

import json
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.force_teacher.demonstration_smoke_test import SyntheticCameras, SyntheticOwner
from lerobot.datasets.sensor_raw_input import RawStreamDescriptor, RawStreamInput
from lerobot.force_control.capture_process import ProcessDemonstrationController, _CompletedTicks
from lerobot.force_control.demonstration import JOINT_NAMES, TeacherRecordConfig
from lerobot.force_control.fake_hardware import FakeForceSensor, FakePlant
from lerobot.force_control.owner import OwnerService
from lerobot.sensors import SensorSample


def wait_for(capture, state, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = capture.status()
        if status["state"] == state:
            return status
        assert status["state"] != "error", status
        time.sleep(0.01)
    pytest.fail(f"Capture never reached {state}: {capture.status()}")


def config(root, **changes):
    return TeacherRecordConfig(
        root=str(root),
        task="Synthetic IPC test",
        h264_encoder="software",
        camera_configs={
            name: {"index_or_path": i, "fps": 50, "width": 64, "height": 64}
            for i, name in enumerate(("front", "wrist"))
        },
        **changes,
    )


class LargePreviewCameras(SyntheticCameras):
    def start(self):
        super().start()
        self._stop.set()
        self._thread.join()
        rng = np.random.default_rng(3)
        for name, spec in self.config.camera_objects().items():
            image = rng.integers(0, 256, size=(spec.height, spec.width, 3), dtype=np.uint8)
            self.buffers[name].publish(image, time.perf_counter_ns(), time.perf_counter_ns())


class DelayedPreparationCameras(SyntheticCameras):
    def start(self):
        time.sleep(0.3)
        super().start()


def test_preparation_does_not_fill_ipc_before_child_receives(tmp_path):
    sensor = FakeForceSensor(FakePlant())
    owner = OwnerService(
        SimpleNamespace(
            config=SimpleNamespace(joint_names=JOINT_NAMES, control_hz=50),
            device=SimpleNamespace(has_resources=False),
        )
    )
    capture = ProcessDemonstrationController(owner, sensor, camera_factory=DelayedPreparationCameras)
    # Two telemetry messages would fill this queue in 100 ms, well before the
    # child can finish the deliberately delayed preparation (or Windows spawn).
    capture._incoming.close()
    capture._incoming = capture._context.Queue(maxsize=2)
    try:
        capture.prepare_before_control(config(tmp_path / "dataset"))
        assert capture.status()["state"] == "ready"
        assert not owner.is_running and not sensor.has_resources
    finally:
        capture.close(timeout_s=5)
    assert not capture.has_resources


def test_large_preview_pipe_cannot_hold_child_open(tmp_path):
    sensor = FakeForceSensor(FakePlant())
    owner = OwnerService(
        SimpleNamespace(
            config=SimpleNamespace(joint_names=JOINT_NAMES, control_hz=50),
            device=SimpleNamespace(has_resources=False),
        )
    )
    with pytest.raises(RuntimeError, match="No owner tick"):
        owner.snapshot()
    capture = ProcessDemonstrationController(owner, sensor, camera_factory=LargePreviewCameras)
    settings = replace(
        config(tmp_path / "dataset"),
        camera_configs={
            name: {"index_or_path": i, "fps": 50, "width": 320, "height": 240}
            for i, name in enumerate(("front", "wrist"))
        },
    )
    try:
        capture.prepare_before_control(settings)
        time.sleep(0.3)
        assert sum(len(item["jpeg"]) for item in capture.preview().values()) > 50_000
        assert capture.status()["state"] == "ready"
        assert owner.snapshot(allow_empty=True) is None
    finally:
        capture.close(timeout_s=5)
    assert not capture.has_resources


def test_optional_empty_snapshot_does_not_hide_owner_failure():
    owner = OwnerService(SimpleNamespace())
    owner._error = RuntimeError("owner transport failed")
    with pytest.raises(RuntimeError, match="owner transport failed"):
        owner.snapshot(allow_empty=True)


def test_raw_input_preserves_original_identity_and_refuses_missing_tail():
    sensor = FakeForceSensor(FakePlant())
    source = RawStreamInput(RawStreamDescriptor.from_sensor(sensor), 42)
    sample = SensorSample(
        100,
        42,
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        arrival_timestamp_ns=110,
        hardware_sequence=12,
    )
    source.accept(sample)
    assert source.subscription.get_nowait() is sample
    with pytest.raises(RuntimeError, match="tail incomplete"):
        source.seal(44)
    source.seal(43)
    assert source.subscription.end_sequence == 43
    with pytest.raises(RuntimeError, match="sealed"):
        source.accept(replace(sample, sequence=43))
    assert not sensor.subscriber_count and not sensor._recorder_leases


def test_raw_input_gap_and_completed_tick_overflow_fail_closed():
    source = RawStreamInput(RawStreamDescriptor.from_sensor(FakeForceSensor(FakePlant())), 0)
    with pytest.raises(RuntimeError, match="sequence gap"):
        source.accept(SensorSample(100, 1))
    with pytest.raises(RuntimeError, match="sequence gap"):
        source.check_health()
    owner = _CompletedTicks(JOINT_NAMES, 50)
    subscription = owner.subscribe(1)
    owner.accept(None)
    with pytest.raises(RuntimeError, match="overflow"):
        owner.accept(None)
    with pytest.raises(RuntimeError, match="overflow"):
        subscription.check_health()


def test_spawn_three_commits_resume_and_interrupted_episode(tmp_path):
    from lerobot.force_control.demonstration_validation import validate_demonstration

    root = tmp_path / "dataset"
    sensor = FakeForceSensor(FakePlant(position_deg=1.2, contact_position_deg=1), sample_rate_hz=400)
    for resumed in (False, True):
        owner = SyntheticOwner(sensor)
        capture = ProcessDemonstrationController(
            owner, sensor, camera_factory=SyntheticCameras, provenance={"backend": "synthetic"}
        )
        settings = config(root, resume=resumed)
        try:
            capture.prepare_before_control(settings)
            assert not sensor.has_resources and not owner.is_running
            sensor.connect()
            sensor.publish_once()
            owner.start()
            for _ in range(1 if resumed else 3):
                capture.submit("record_start", {"config": asdict(settings)})
                wait_for(capture, "recording")
                time.sleep(0.3)
                before = owner.tick_count
                capture.submit("record_finish", {"task_result": "unknown"})
                saved = wait_for(capture, "saved")
                assert saved["file_complete"] and owner.tick_count > before
                deadline = time.monotonic() + 2
                while capture._lease is not None and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert capture._lease is None
            if resumed:
                capture.submit("record_start", {"config": asdict(settings)})
                wait_for(capture, "recording")
                time.sleep(0.1)
        finally:
            capture.close()
            owner.close()
            sensor.disconnect()
        assert not capture.has_resources and not sensor._recorder_leases
    report = validate_demonstration(root)
    assert report["valid"], report
    assert len(report["episodes"]) == 4
    assert len(report["excluded_episodes"]) == 1
    incomplete = [
        json.loads(p.read_text())
        for p in (root / "meta/teacher_episodes").glob("*.json")
        if json.loads(p.read_text())["complete"] is False
    ]
    assert len(incomplete) == 1


@pytest.mark.parametrize("failure", ["stop_during_save", "child_exit"])
def test_failed_child_never_publishes_teacher_complete(tmp_path, failure):
    sensor = FakeForceSensor(FakePlant(position_deg=1.2, contact_position_deg=1), sample_rate_hz=400)
    owner = SyntheticOwner(sensor)
    capture = ProcessDemonstrationController(owner, sensor, camera_factory=SyntheticCameras)
    settings = config(tmp_path / "dataset")
    try:
        capture.prepare_before_control(settings)
        sensor.connect()
        sensor.publish_once()
        owner.start()
        capture.submit("record_start", {"config": asdict(settings)})
        wait_for(capture, "recording")
        time.sleep(0.3)
        if failure == "child_exit":
            capture._process.terminate()  # Fault injection into a synthetic writer only.
            deadline = time.monotonic() + 3
            while capture.status()["state"] != "error" and time.monotonic() < deadline:
                time.sleep(0.01)
            assert capture.status()["state"] == "error"
        else:
            capture.submit("record_finish", {"task_result": "unknown"})
            wait_for(capture, "saving")
            capture._cancel.set()
    finally:
        capture.close()
        owner.close()
        sensor.disconnect()
    manifests = list((tmp_path / "dataset/meta/teacher_episodes").glob("*.json"))
    assert manifests and all(json.loads(p.read_text())["complete"] is False for p in manifests)
    assert not sensor._recorder_leases and not sensor.subscriber_count
