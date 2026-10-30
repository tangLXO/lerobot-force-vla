"""Windows-spawn capture boundaries and failure isolation, without hardware."""

import json
import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
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
        assert status["state"] != "error", json.dumps(status, ensure_ascii=False, default=str)
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


@pytest.mark.parametrize("claim_before_end", [False, True])
@pytest.mark.parametrize("fail_start", [False, True])
def test_child_retains_ticks_during_delayed_recorder_start(
    monkeypatch, tmp_path, claim_before_end, fail_start
):
    """Exercise the actual IPC worker while its recorder startup is delayed."""
    from lerobot.force_control import capture_process

    received = []
    controllers = []

    class Recorder:
        def close(self):
            pass

    class Controller:
        def __init__(self, owner, *args, **kwargs):
            self.owner = owner
            self._dataset = SimpleNamespace(root=tmp_path)
            self._recorder = Recorder()
            self._uid = None
            self._cleanup_pending = self.start_processed = False
            self.subscription = None
            controllers.append(self)

        def prepare_before_control(self, config):
            pass

        def claim(self):
            self.subscription = self.owner.subscribe(8)
            self.start_processed = True

        def submit(self, kind, value):
            if kind == "record_start":
                if fail_start:
                    raise RuntimeError("Injected recorder startup failure")
                self._uid = "episode"
            elif kind == "record_finish":
                if self.subscription is None:
                    self.claim()
                received.append([self.subscription.queue.get_nowait().tick_id for _ in range(3)])
                self.owner.unsubscribe(self.subscription)
                self.subscription = None
                self._uid = None

        def status(self):
            return {"state": "ready", "active": False}

        def preview(self):
            return {}

        def close(self):
            pass

    class Incoming:
        def __init__(self):
            settings = asdict(replace(config(tmp_path / "dataset"), queue_capacity=8))
            self.messages = iter(
                [
                    ("owner", (True, None)),
                    *[
                        (kind, payload)
                        for token in ("first", "second")
                        for kind, payload in [
                            ("start", (token, descriptor, 0, {"config": settings})),
                            (
                                "data",
                                (token, [], [SimpleNamespace(tick_id=101), SimpleNamespace(tick_id=102)]),
                            ),
                            ("data", (token, [], [SimpleNamespace(tick_id=103)])),
                            ("end", (token, 3, 0, "record_finish", {})),
                        ]
                    ],
                    ("close", None),
                ]
            )

        def get(self, timeout):
            kind, payload = next(self.messages)
            if kind == "end" and claim_before_end:
                controllers[0].claim()
            return kind, payload

    class Outgoing(queue.Queue):
        def cancel_join_thread(self):
            pass

    descriptor = RawStreamDescriptor.from_sensor(FakeForceSensor(FakePlant()))
    outgoing = Outgoing(maxsize=32)
    monkeypatch.setattr(capture_process, "_ChildController", Controller)
    monkeypatch.setattr(
        capture_process, "SensorStreamRecorder", SimpleNamespace(from_raw_inputs=lambda *a: Recorder())
    )
    monkeypatch.setattr(capture_process.multiprocessing, "parent_process", lambda: None)
    capture_process._capture_worker(
        None,
        descriptor,
        JOINT_NAMES,
        50,
        "gripper_force",
        {},
        Incoming(),
        outgoing,
        threading.Event(),
        threading.Event(),
        None,
        str(Path(capture_process.__file__).resolve()),
    )
    events = list(outgoing.queue)
    if fail_start:
        assert any("Injected recorder startup failure" in event.get("error", "") for event in events)
        assert not controllers[0].owner._subscriptions
        assert controllers[0].owner._prepared is None
        assert not received
        return
    assert not [event for event in events if event.get("error")]
    assert received == [[101, 102, 103], [101, 102, 103]]


def test_unclaimed_startup_queue_overflow_and_failed_start_cleanup():
    owner = _CompletedTicks(JOINT_NAMES, 50)
    owner.prepare_episode(1)
    owner.accept(SimpleNamespace(tick_id=1))
    with pytest.raises(RuntimeError, match="overflow"):
        owner.accept(SimpleNamespace(tick_id=2))
    with pytest.raises(RuntimeError, match="overflow"):
        owner.subscribe(1).check_health()
    owner.clear_episode()
    owner.prepare_episode(1)
    owner.accept(SimpleNamespace(tick_id=3))
    assert owner.subscribe(1).get().tick_id == 3
    owner.clear_episode()
    assert not owner._subscriptions and owner._prepared is None


@pytest.mark.parametrize("close_delivery_failed", [False, True])
def test_cancel_waits_for_ordered_close_unless_delivery_failed(monkeypatch, close_delivery_failed):
    from lerobot.force_control import capture_process

    class Incoming:
        calls = 0

        def get(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise queue.Empty
            assert self.calls == 2
            return "close", None

    class Outgoing(queue.Queue):
        def cancel_join_thread(self):
            pass

    class Controller:
        closed = False

        def prepare_before_control(self, config):
            pass

        def status(self):
            return {"state": "ready", "active": False}

        def preview(self):
            return {}

        def close(self):
            self.closed = True

    incoming, controller = Incoming(), Controller()
    cancel, close_failed = threading.Event(), threading.Event()
    cancel.set()
    if close_delivery_failed:
        close_failed.set()
    monkeypatch.setattr(capture_process, "_ChildController", lambda *args, **kwargs: controller)
    monkeypatch.setattr(capture_process.multiprocessing, "parent_process", lambda: None)
    capture_process._capture_worker(
        None,
        RawStreamDescriptor.from_sensor(FakeForceSensor(FakePlant())),
        JOINT_NAMES,
        50,
        "gripper_force",
        {},
        incoming,
        Outgoing(),
        cancel,
        close_failed,
        None,
        str(Path(capture_process.__file__).resolve()),
    )
    assert incoming.calls == (1 if close_delivery_failed else 2)
    assert controller.closed


@pytest.mark.parametrize("raw_error_races_status", [False, True])
def test_primary_recorder_error_survives_cleanup_and_replaceable_status(monkeypatch, raw_error_races_status):
    from lerobot.force_control import capture_process

    class Outgoing(queue.Queue):
        def cancel_join_thread(self):
            pass

    class Incoming:
        def get(self, timeout):
            if raw_error_races_status:
                raise RuntimeError("Raw producer tail was not sealed")
            raise queue.Empty

    class Controller:
        def prepare_before_control(self, config):
            pass

        def status(self):
            return {"state": "error", "active": False, "error": "Original camera freshness failure"}

        def preview(self):
            return {}

        def close(self):
            raise RuntimeError("Raw producer tail was not sealed")

    outgoing = Outgoing(maxsize=1)
    cancel = threading.Event()
    monkeypatch.setattr(capture_process, "_ChildController", lambda *a, **kw: Controller())
    monkeypatch.setattr(capture_process.multiprocessing, "parent_process", lambda: None)
    capture_process._capture_worker(
        None,
        RawStreamDescriptor.from_sensor(FakeForceSensor(FakePlant())),
        JOINT_NAMES,
        50,
        "gripper_force",
        {},
        Incoming(),
        outgoing,
        cancel,
        threading.Event(),
        None,
        str(Path(capture_process.__file__).resolve()),
    )
    final = outgoing.get_nowait()
    assert final["closed"] and cancel.is_set()
    assert "Original camera freshness failure" in final["error"]
    assert "Raw producer tail was not sealed" in final["error"]


@pytest.mark.parametrize("cached", [False, True])
def test_spawn_three_commits_resume_and_interrupted_episode(tmp_path, cached):
    from lerobot.force_control.demonstration_validation import validate_demonstration

    root = tmp_path / "dataset"
    sensor = FakeForceSensor(FakePlant(position_deg=1.2, contact_position_deg=1), sample_rate_hz=400)
    for resumed in (False, True):
        owner = SyntheticOwner(sensor)
        if cached:
            # This exercises cached recording policy through a real spawned writer,
            # not physical control or transport timing.
            owner.runtime.device = SimpleNamespace(
                config=SimpleNamespace(
                    command_feedback_mode="cached", feedback_max_age_s=0.05, command_tx_gap_us=1000
                )
            )
            owner.runtime.config.command_ttl_ms = 200
            owner.runtime.safety = SimpleNamespace(
                config=SimpleNamespace(
                    joint_feedback_max_age_ms=50, force_max_age_ms=100, decision_max_age_ms=40
                )
            )
            original_snapshot = owner._snapshot
            previous = [None]

            def reused_snapshot(tick, original_snapshot=original_snapshot, previous=previous):
                snap = original_snapshot(tick)
                prior = previous[0]
                row = snap.joint_snapshot.get("gripper")
                reused = (
                    prior is not None
                    and tick % 2 == 1
                    and row.receive_timestamp_ns - prior.receive_timestamp_ns < 50_000_000
                )
                if reused:
                    joints = replace(
                        snap.joint_snapshot,
                        joints=(
                            *snap.joint_snapshot.joints[:-1],
                            replace(prior, age_ns=row.receive_timestamp_ns - prior.receive_timestamp_ns),
                        ),
                    )
                    snap = replace(snap, joint_snapshot=joints)
                    snap.row["joint_snapshot"] = asdict(joints)
                    snap.row["feedback_received_ns"] = prior.receive_timestamp_ns
                else:
                    previous[0] = row
                snap.row.update(
                    command_feedback_mode="cached",
                    soft_deadline_miss=True,
                    soft_deadline_streak=1,
                    feedback_reused={name: reused and name == "gripper" for name in JOINT_NAMES},
                )
                return snap

            owner._snapshot = reused_snapshot
        capture = ProcessDemonstrationController(
            owner, sensor, camera_factory=SyntheticCameras, provenance={"backend": "synthetic"}
        )
        # Exercise spawn/save/reopen semantics, not host scheduling performance.
        # Causality is still enforced; production's 100 ms and the separate
        # stale-camera rejection tests remain unchanged.
        settings = replace(config(root, resume=resumed), camera_max_age_ms=500)
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
                assert saved["save_timings"]["total_seconds"] >= 0
                deadline = time.monotonic() + 2
                while capture._lease is not None and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert capture._lease is None
                capture.submit("record_outcome", {"task_result": "success"})
                deadline = time.monotonic() + 2
                while capture.status().get("task_result") != "success" and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert capture.status()["task_result"] == "success"
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

    import hashlib

    import pyarrow.parquet as pq

    assert (
        json.loads((root / "meta/teacher_capture.json").read_text())["schema"]
        == "teacher_capture_contract_v2"
    )
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        assert "teacher.action_pre_safety" in table.column_names
        for before, after in zip(
            table["teacher.action_pre_safety"].to_pylist(), table["action"].to_pylist(), strict=True
        ):
            np.testing.assert_array_equal(before, after)
    manifest_path = next(
        p for p in (root / "meta/teacher_episodes").glob("*.json") if json.loads(p.read_text())["complete"]
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    uid = manifest["episode_uid"]
    frames_path = root / manifest["files"]["frames"]["path"]
    frames_bytes = frames_path.read_bytes()
    frames = [json.loads(line) for line in frames_bytes.splitlines()]
    frames[0]["action_pre_safety_deg"][0] += 1
    frames_path.write_text("".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8")
    manifest["files"]["frames"]["sha256"] = hashlib.sha256(frames_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invalid = validate_demonstration(root, uid, verify_video=False)
    assert not invalid["valid"] and "pre-safety" in str(invalid["errors"])
    control_path = root / manifest["files"]["control"]["path"]
    control_bytes = control_path.read_bytes()
    controls = [json.loads(line) for line in control_bytes.splitlines()]
    control = next(row for row in controls if row["tick_id"] == frames[0]["control_tick_id"])
    next(item for item in control["per_joint_results"] if item["name"] == JOINT_NAMES[0])[
        "candidate_deg"
    ] += 1
    control_path.write_text("".join(json.dumps(row) + "\n" for row in controls), encoding="utf-8")
    manifest["files"]["control"]["sha256"] = hashlib.sha256(control_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invalid = validate_demonstration(root, uid, verify_video=False)
    assert not invalid["valid"] and "Main pre-safety action" in str(invalid["errors"])
    control_path.write_bytes(control_bytes)
    frames_path.write_bytes(frames_bytes)
    manifest_path.write_bytes(manifest_bytes)
    incomplete = [
        json.loads(p.read_text())
        for p in (root / "meta/teacher_episodes").glob("*.json")
        if json.loads(p.read_text())["complete"] is False
    ]
    assert len(incomplete) == 1
    if cached:
        policy = json.loads((root / "meta/teacher_capture.json").read_text())["communication_policy"]
        assert policy["mode"] == "cached" and policy["hard_deadline_ns"] == 40_000_000


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
