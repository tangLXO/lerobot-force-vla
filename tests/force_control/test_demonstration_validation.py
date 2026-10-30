"""Complete demonstration validation against real local Dataset/Sidecar files."""

# Optional dependency checks must run before Dataset imports during base-tier collection.
# ruff: noqa: E402

import hashlib
import importlib
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets", reason="Demonstration integration requires lerobot[dataset]")
pytest.importorskip("pyarrow", reason="Demonstration integration requires lerobot[dataset]")
av = pytest.importorskip("av", reason="Demonstration video integration requires lerobot[dataset]")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.force_control.demonstration_validation import validate_demonstration
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES
from lerobot.sensors import Sensor, SensorConfig, SensorFeature


class ManualSensor(Sensor):
    def __init__(self):
        super().__init__(SensorConfig(expected_sample_rate_hz=400, max_age_ms=100))

    @property
    def features(self):
        return {name: SensorFeature("float32", "N") for name in ("left.normal_force", "right.normal_force")}

    @property
    def is_connected(self):
        return True

    def connect(self):
        pass

    def disconnect(self):
        pass


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {
        "path": path.as_posix(),
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def demonstration(tmp_path, monkeypatch):
    root, names = tmp_path / "dataset", list(FAKE_B601_JOINT_NAMES)
    sources = [f"sensor.gripper_force.{name}.normal_force" for name in ("left", "right")]
    dataset = LeRobotDataset.create(
        repo_id="test/unified-demonstration",
        root=root,
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"{name}.pos" for name in names],
            },
            "action": {"dtype": "float32", "shape": (7,), "names": [f"{name}.pos" for name in names]},
            "observation.tactile": {"dtype": "float32", "shape": (2,), "names": sources},
            "observation.images.front": {
                "dtype": "video",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            },
        },
    )

    def encode(_key, episode_index):
        path = root / f"encoded-{episode_index}" / "front.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for i in range(3):
                image = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(image):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    monkeypatch.setattr(dataset.writer, "_encode_temporary_episode_video", encode)
    sensor = ManualSensor()
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    uid = recorder.start_episode(0, dataset=dataset)
    controls, frames = [], []
    for i in range(3):
        timestamp = 1_000_000_000 + i * 40_000_000
        anchor, send_start, send_end = timestamp + 20, timestamp + 30, timestamp + 40
        state, action, tactile = [float(i)] * 7, [float(20 + i)] * 7, [float(i + 1), float(i + 2)]
        sample = sensor._publish_sample(
            dict(zip(("left.normal_force", "right.normal_force"), tactile, strict=True)),
            timestamp_ns=timestamp,
            arrival_timestamp_ns=timestamp + 1,
        )
        row = {
            "tick_id": 10 + i,
            "decision_anchor_ns": anchor,
            "epoch": 3,
            "capture_stage": "release" if i == 2 else "force_control",
            "label_valid": i == 1,
            "action_label_valid": i > 0,
            "send_status": "submitted",
            "send_start_ns": send_start,
            "send_end_ns": send_end,
            "force_sequence": sample.sequence,
            "force_timestamp_ns": timestamp,
            "force_arrival_timestamp_ns": timestamp + 1,
            "force_left_N": tactile[0],
            "force_right_N": tactile[1],
            "per_joint_results": [
                {"name": name, "status": "success", "submitted_deg": value}
                for name, value in zip(names, action, strict=True)
            ],
            "joint_snapshot": {
                "joints": [
                    {"name": name, "is_valid": True, "position_deg": value}
                    for name, value in zip(names, state, strict=True)
                ]
            },
        }
        controls.append(row)
        frames.append(
            {
                "frame_index": i,
                "control_tick_id": row["tick_id"],
                "decision_anchor_ns": anchor,
                "epoch": 3,
                "state_deg": state,
                "action_deg": action,
                "tactile_N": tactile,
                "force_sequence": sample.sequence,
                "force_timestamp_ns": timestamp,
                "force_arrival_timestamp_ns": timestamp + 1,
                "cameras": {"front": {"timestamp_ns": timestamp - 10, "arrival_timestamp_ns": timestamp}},
                "send_start_ns": send_start,
                "send_end_ns": send_end,
            }
        )
        dataset.add_frame(
            {
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "observation.tactile": np.asarray(tactile, dtype=np.float32),
                "observation.images.front": np.zeros((16, 16, 3), dtype=np.uint8),
                "task": "carry object",
            }
        )
        recorder.record_sync(
            i,
            {
                "observation_start_ns": anchor - 1,
                "frame_anchor_ns": anchor,
                "observation_complete_ns": anchor + 1,
                "sensors": {
                    "gripper_force": {
                        "timestamp_ns": timestamp,
                        "arrival_timestamp_ns": timestamp + 1,
                        "sequence": sample.sequence,
                        "hardware_sequence": None,
                        "age_ns": anchor - timestamp,
                        "status": sample.status,
                    }
                },
            },
        )
    recorder.save_episode(dataset, task_info=["carry object"])
    recorder.close()
    dataset.finalize()
    files = {}
    for name, rows in (("control", controls), ("frames", frames)):
        record = _write_rows(root / f"teacher/{uid}/{name}.jsonl", rows)
        record["path"] = f"teacher/{uid}/{name}.jsonl"
        files[name] = record
    manifest = {
        "schema": "teacher_demo_episode_v1",
        "episode_uid": uid,
        "episode_index": 0,
        "frame_count": 3,
        "complete": True,
        "status": "saved",
        "task_result": "unknown",
        "joint_names": names,
        "files": files,
        "capture_summary": {
            "stage_counts": dict(Counter(row["capture_stage"] for row in controls)),
            "force_label_count": 1,
            "action_label_count": 2,
            "normal_release_completed": False,
        },
        "config": {},
        "provenance": {},
    }
    _write_json(root / f"meta/teacher_episodes/{uid}.json", manifest)
    return root, uid, manifest, controls, frames


def _rewrite(demo, *, control=False, frame=False):
    root, uid, manifest, controls, frames = demo
    for name, rows, changed in (("control", controls, control), ("frames", frames, frame)):
        if changed:
            record = _write_rows(root / f"teacher/{uid}/{name}.jsonl", rows)
            record["path"] = f"teacher/{uid}/{name}.jsonl"
            manifest["files"][name] = record
    _write_json(root / f"meta/teacher_episodes/{uid}.json", manifest)


def _tree(root):
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_round_trip_is_readonly_and_manual_release_does_not_require_auto_release(demonstration):
    root, uid, *_ = demonstration
    before = _tree(root)
    report = validate_demonstration(root, uid)
    assert report["valid"], report["errors"]
    assert _tree(root) == before
    episode = report["episodes"][0]
    assert episode["files_complete"] and episode["video_decoded"]
    assert episode["task_result"] == "unknown" and not episode["successful_demo"]
    assert episode["capture_summary"]["normal_release_completed"] is False


def test_fast_validation_keeps_other_checks_and_skips_video_decode(demonstration, monkeypatch):
    root, uid, *_ = demonstration
    before = _tree(root)

    def unexpected_decode(*_args, **_kwargs):
        pytest.fail("Fast validation must not decode video")

    monkeypatch.setattr("lerobot.force_control.demonstration_validation._decode_video", unexpected_decode)
    progress, timings = [], {}
    report = validate_demonstration(root, uid, verify_video=False, progress=progress.append, timings=timings)
    assert report["valid"], report["errors"]
    assert report["episodes"][0]["video_decoded"] is False
    assert progress == [{"event": "episode", "episode_uid": uid, "index": 1, "total": 1}]
    assert timings.get("video_decode_seconds", 0) == 0
    assert _tree(root) == before


def test_full_validation_reports_video_progress_and_duration(demonstration):
    root, uid, *_ = demonstration
    progress, timings = [], {}
    report = validate_demonstration(root, uid, progress=progress.append, timings=timings)
    assert report["valid"], report["errors"]
    assert [event["event"] for event in progress] == ["episode", "video"]
    assert progress[1]["episode_uid"] == uid and "front" in progress[1]["video"]
    assert timings["video_decode_seconds"] > 0


@pytest.mark.parametrize("corrupt", [False, True])
def test_verification_child_checks_real_files_without_mutating_dataset(demonstration, monkeypatch, corrupt):
    root, uid, *_ = demonstration
    project = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(project / "examples/force_teacher"))
    worker_type = importlib.import_module("b601_verification").VerificationWorker
    if corrupt:
        video = next((root / "videos").rglob("*.mp4"))
        video.write_bytes(b"corrupt video")
    before = _tree(root)
    worker = worker_type(project, root)
    events = []
    try:
        deadline = time.monotonic() + 30
        while not worker.finished:
            events.extend(worker.poll())
            assert time.monotonic() < deadline
            time.sleep(0.01)
        events.extend(worker.poll())
        result = next(event["report"] for event in events if event["event"] == "result")
        assert result["valid"] is not corrupt, result
        assert worker.process.returncode == int(corrupt)
        assert any(event["event"] == "episode" and event["episode_uid"] == uid for event in events)
        assert _tree(root) == before
    finally:
        worker.close()


def test_verification_child_cancel_reaps_process_without_dataset_writes(demonstration, monkeypatch):
    root, *_ = demonstration
    project = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(project / "examples/force_teacher"))
    worker_type = importlib.import_module("b601_verification").VerificationWorker
    before = _tree(root)
    worker = worker_type(project, root)
    worker.cancel()
    assert worker.finished and worker.process.poll() is not None and not worker.thread.is_alive()
    assert _tree(root) == before


def test_cancel_after_windows_venv_child_starts_reaps_the_process_tree(monkeypatch, tmp_path):
    psutil = pytest.importorskip("psutil")
    project = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(project / "examples/force_teacher"))
    worker_type = importlib.import_module("b601_verification").VerificationWorker
    fake_project = tmp_path / "project"
    script = fake_project / "examples/force_teacher/validate_demonstration.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        'import time\nprint(\'{"event":"video","video":"front.mp4"}\', flush=True)\ntime.sleep(120)\n',
        encoding="utf-8",
    )
    root = tmp_path / "dataset"
    root.mkdir()
    worker = worker_type(fake_project, root)
    children = []
    try:
        deadline = time.monotonic() + 20
        while not any(event.get("event") == "video" for event in worker.poll()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        children = psutil.Process(worker.process.pid).children(recursive=True)
        worker.cancel()
        _, alive = psutil.wait_procs(children, timeout=3)
        assert worker.finished and not alive
    finally:
        for process in children:
            if process.is_running():
                process.kill()
        if worker.process.poll() is None:
            worker.process.kill()
            worker.process.wait(timeout=2)


def test_desktop_session_runs_real_verification_child_and_persists_report(
    demonstration, monkeypatch, tmp_path
):
    root, *_ = demonstration
    project = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(project / "examples/force_teacher"))
    session_type = importlib.import_module("b601_session").SessionController
    controller = session_type(demo=True, output_root=tmp_path / "logs")
    before = _tree(root)

    def wait_for(predicate):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        pytest.fail("Desktop verification did not finish")

    def request(action, payload=None):
        request_id = controller.request(action, payload)
        return wait_for(
            lambda: next(
                (row for row in controller.snapshot().operation_results if row["request_id"] == request_id),
                None,
            )
        )

    controller.start()
    try:
        wait_for(lambda: controller.snapshot().state == "disconnected")
        assert request("update_form", {"recording_values": {"root": str(root)}})["success"]
        assert request("verify_dataset")["success"]
        result = wait_for(
            lambda: (
                controller.snapshot().recording["full_verification"]
                if controller.snapshot().recording["full_verification"]["state"] in {"passed", "failed"}
                else None
            )
        )
        assert result["state"] == "passed", dict(result)
        report_path = Path(result["report_path"])
        assert report_path.parent == root.parent / f"{root.name}_verification_reports"
        assert json.loads(report_path.read_text(encoding="utf-8"))["result"]["valid"]
        assert _tree(root) == before
    finally:
        controller.close()
        controller._thread.join(5)
        assert not controller._thread.is_alive()


@pytest.mark.parametrize("task_result", ["success", "failure", "unknown"])
def test_task_result_is_independent_of_file_validity(demonstration, task_result):
    root, uid, manifest, *_ = demonstration
    manifest["task_result"] = task_result
    _rewrite(demonstration)
    report = validate_demonstration(root, uid, verify_video=False)
    assert report["valid"], report["errors"]
    assert report["episodes"][0]["successful_demo"] == (task_result == "success")


@pytest.mark.parametrize(
    "change,expected",
    [
        ("receipt", "action differs"),
        ("state", "state differs"),
        ("camera_future", "camera is noncausal"),
        ("force_future", "force is noncausal"),
        ("force_sequence", "Sensor Sync reference differ"),
        ("control_gap", "missing or unordered ticks"),
        ("summary", "capture summary differs"),
    ],
)
def test_inconsistent_evidence_is_rejected_even_with_updated_digests(demonstration, change, expected):
    root, uid, manifest, controls, frames = demonstration
    if change == "receipt":
        controls[1]["per_joint_results"][0]["submitted_deg"] += 1
    elif change == "state":
        controls[1]["joint_snapshot"]["joints"][0]["position_deg"] += 1
    elif change == "camera_future":
        frames[1]["cameras"]["front"]["arrival_timestamp_ns"] = frames[1]["decision_anchor_ns"] + 1
    elif change == "force_future":
        frames[1]["force_timestamp_ns"] = controls[1]["force_timestamp_ns"] = (
            frames[1]["decision_anchor_ns"] + 1
        )
    elif change == "force_sequence":
        frames[1]["force_sequence"] = controls[1]["force_sequence"] = 999
    elif change == "control_gap":
        controls[1]["tick_id"] += 4
    else:
        manifest["capture_summary"]["force_label_count"] = 8
    _rewrite(demonstration, control=True, frame=True)
    report = validate_demonstration(root, uid, verify_video=False)
    assert not report["valid"]
    assert any(expected in error for error in report["errors"]), report["errors"]


def test_incomplete_hash_damage_and_live_writer_are_rejected_without_repair(demonstration):
    root, uid, manifest, *_ = demonstration
    manifest["complete"] = False
    _rewrite(demonstration)
    assert "incomplete" in " ".join(validate_demonstration(root, uid)["errors"])
    manifest["complete"] = True
    _rewrite(demonstration)
    with (root / f"teacher/{uid}/frames.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    assert "digest mismatch" in " ".join(validate_demonstration(root, uid)["errors"])
    _write_json(root / ".sensor-writer.lock", {"pid": os.getpid()})
    before = _tree(root)
    assert "live writer" in " ".join(validate_demonstration(root, uid)["errors"])
    assert _tree(root) == before


def test_teacher_path_cannot_escape_root(demonstration):
    root, uid, manifest, *_ = demonstration
    manifest["files"]["control"]["path"] = "../outside.jsonl"
    _rewrite(demonstration)
    assert "path must be" in " ".join(validate_demonstration(root, uid)["errors"])


def test_candidate_integrity_check_does_not_publish_complete_marker(demonstration):
    root, uid, manifest, *_ = demonstration
    manifest.update(complete=False, status="verifying", task_result="success")
    _rewrite(demonstration)
    before = _tree(root)
    assert not validate_demonstration(root, uid)["valid"]
    report = validate_demonstration(root, uid, require_complete=False)
    assert report["valid"], report["errors"]
    assert not report["episodes"][0]["complete"]
    assert not report["episodes"][0]["successful_demo"]
    assert _tree(root) == before


@pytest.mark.parametrize("status", ["discarded", "interrupted", "error"])
def test_whole_root_excludes_known_unsaved_attempts_but_explicit_uid_remains_strict(demonstration, status):
    root, _uid, manifest, *_ = demonstration
    discarded_uid = str(uuid.uuid4())
    discarded = dict(manifest, episode_uid=discarded_uid, complete=False, status=status)
    _write_json(root / f"meta/teacher_episodes/{discarded_uid}.json", discarded)
    before = _tree(root)
    report = validate_demonstration(root)
    assert report["valid"], report["errors"]
    assert len(report["episodes"]) == len(report["excluded_episodes"]) == 1
    assert report["excluded_episodes"][0]["episode_uid"] == discarded_uid
    assert not validate_demonstration(root, discarded_uid)["valid"]
    assert _tree(root) == before


@pytest.mark.parametrize("status", ["recording", "interrupted"])
def test_committed_main_cannot_hide_behind_an_incomplete_marker(demonstration, status):
    root, _uid, manifest, *_ = demonstration
    manifest.update(complete=False, status=status)
    _rewrite(demonstration)
    report = validate_demonstration(root)
    assert not report["valid"] and not report["excluded_episodes"]


@pytest.mark.parametrize("registered", [None, "files", "main_artifacts", "main_evidence", "missing_field"])
def test_quarantined_attempt_excluded_only_without_registered_save_evidence(demonstration, registered):
    root, _uid, manifest, *_ = demonstration
    failed_uid = str(uuid.uuid4())
    failed = dict(manifest, episode_uid=failed_uid, complete=False, status="error")
    _write_json(root / f"meta/teacher_episodes/{failed_uid}.json", failed)
    journal = {
        "journal_version": 1,
        "episode_uid": failed_uid,
        "state": "QUARANTINED",
        "files": [],
        "main_artifacts": [],
        "main_evidence": None,
    }
    if registered in ("files", "main_artifacts"):
        journal[registered] = [{"path": "registered-artifact"}]
    elif registered == "main_evidence":
        journal[registered] = {"registered": True}
    elif registered == "missing_field":
        del journal["files"]
    _write_json(root / f"meta/sensor_transactions/{failed_uid}.json", journal)
    before = _tree(root)
    report = validate_demonstration(root)
    if registered is None:
        assert report["valid"], report["errors"]
        assert report["excluded_episodes"][0]["transaction_state"] == "QUARANTINED"
    else:
        assert not report["valid"] and not report["excluded_episodes"]
    assert not validate_demonstration(root, failed_uid)["valid"]
    assert _tree(root) == before


def test_whole_root_rejects_orphan_main_episode(demonstration):
    root, *_ = demonstration
    path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["total_episodes"] += 1
    _write_json(path, info)
    report = validate_demonstration(root)
    assert not report["valid"]
    assert any("episode closure" in error for error in report["errors"])


def test_video_decode_failure_is_not_a_complete_teacher_episode(demonstration, monkeypatch):
    root, uid, *_ = demonstration

    def fail_decode(*_args, **_kwargs):
        raise ValueError("injected decode error")

    monkeypatch.setattr("lerobot.force_control.demonstration_validation._decode_video", fail_decode)
    assert "injected decode error" in " ".join(validate_demonstration(root, uid)["errors"])
