#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import os
import shutil
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import (
    SIDECAR_SCHEMA_VERSION,
    SensorQueueOverflowError,
    SensorRecorderError,
    SensorStreamRecorder,
    preflight_sensor_stream_writer,
)
from lerobot.datasets.sensor_transaction import TransactionState
from lerobot.datasets.sensor_window import SensorStreamReader
from lerobot.robots.sensorized_robot import SensorizedRobot
from lerobot.sensors import Sensor, SensorConfig, SensorFeature
from lerobot.utils.constants import OBS_STATE, OBS_TACTILE
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features


class FakeSensor(Sensor):
    def __init__(self):
        super().__init__(SensorConfig(expected_sample_rate_hz=100, max_age_ms=100))
        self.connected = True
        self.device_id = "fake-a"

    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float32", "N"),
            "right.normal_force": SensorFeature("float32", "N"),
        }

    @property
    def native_features(self):
        return {"channel_1.register": SensorFeature("int32", "device_count")}

    @property
    def provenance(self):
        return {**super().provenance, "device_id": self.device_id}

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False


class IncompatibleFakeSensor(FakeSensor):
    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float64", "N"),
            "right.normal_force": SensorFeature("float32", "N"),
        }


class WrongUnitFakeSensor(FakeSensor):
    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float32", "kg"),
            "right.normal_force": SensorFeature("float32", "N"),
        }


def initialize_main(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 0,
                "total_frames": 0,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        ),
        encoding="utf-8",
    )


def write_main_episode(root, frame_count=2):
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{"index": index, "episode_index": 0} for index in range(frame_count)]),
        data_path,
    )
    episode_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "length": frame_count,
                    "dataset_from_index": 0,
                    "dataset_to_index": frame_count,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "tasks": ["test"],
                }
            ]
        ),
        episode_path,
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": frame_count,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        ),
        encoding="utf-8",
    )


class FakeDataset:
    def __init__(self, root):
        self.root = root

    def save_episode(self):
        transaction = getattr(self, "_sensor_transaction", None)
        if transaction is not None:
            transaction.register_artifact(self.root / "data/chunk-000/file-000.parquet", "data")
            transaction.register_artifact(
                self.root / "meta/episodes/chunk-000/file-000.parquet", "episode_metadata"
            )
            transaction.register_artifact(self.root / "meta/info.json", "info")
        write_main_episode(self.root)


def capture(anchor, timestamp, sequence):
    return {
        "observation_start_ns": anchor - 20,
        "frame_anchor_ns": anchor,
        "observation_complete_ns": anchor + 20,
        "hardware_observation_timestamps": None,
        "sensors": {
            "gripper_force": {
                "timestamp_ns": timestamp,
                "arrival_timestamp_ns": timestamp + 1,
                "sequence": sequence,
                "hardware_sequence": None,
                "age_ns": anchor - timestamp,
                "status": "ok",
            }
        },
    }


def test_native_rate_raw_and_sync_commit_and_read(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    first = time.perf_counter_ns()
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        first,
        arrival_timestamp_ns=first + 1,
        native_values={"channel_1.register": 123},
        native_payload=b"disabled-by-default",
    )
    second = first + 10_000_000
    sensor._publish_sample(
        {"left.normal_force": 3.0, "right.normal_force": 4.0},
        second,
        arrival_timestamp_ns=second + 1,
        native_values={"channel_1.register": 456},
    )
    recorder.record_sync(0, capture(second + 2, second, 1))
    recorder.record_sync(1, capture(second + 5_000_000, second, 1))
    threads = recorder.worker_threads
    recorder.save_episode(FakeDataset(tmp_path), task_info=["test"])
    recorder.close()

    assert sensor.subscriber_count == 0
    assert all(not thread.is_alive() for thread in threads)
    journal = json.loads((tmp_path / "meta" / "sensor_transactions" / f"{uid}.json").read_text())
    assert journal["state"] == TransactionState.COMMITTED
    assert journal["journal_version"] == 1
    assert journal["main_evidence"]["version"] == 1
    episode_metadata = json.loads(
        (tmp_path / "meta" / "sensor_episodes" / f"{uid}.json").read_text(encoding="utf-8")
    )
    assert episode_metadata["sidecar_schema_version"] == 2

    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    rows = reader.read_raw()
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[1]["values"] == {"left.normal_force": 3.0, "right.normal_force": 4.0}
    assert rows[1]["native_values"] == {"channel_1.register": 456}
    assert rows[0]["native_payload"] is None
    assert len(reader.read_sync()) == 2


def test_transaction_commits_against_real_lerobot_dataset(tmp_path) -> None:
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/sensor-transaction",
        fps=50,
        root=root,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint.pos"],
            }
        },
    )
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    recorder.start_episode(0)
    timestamp = time.perf_counter_ns()
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0}, timestamp, arrival_timestamp_ns=timestamp + 1
    )
    for frame_index in range(2):
        dataset.add_frame({"observation.state": np.asarray([frame_index], dtype=np.float32), "task": "test"})
        recorder.record_sync(frame_index, capture(timestamp + frame_index + 1, timestamp, 0))

    recorder.save_episode(dataset, task_info=["test"])
    sensor.device_id = "fake-b"
    recorder.start_episode(1)
    second_timestamp = timestamp + 1_000_000
    sensor._publish_sample(
        {"left.normal_force": 3.0, "right.normal_force": 4.0},
        second_timestamp,
        arrival_timestamp_ns=second_timestamp + 1,
    )
    for frame_index in range(2):
        dataset.add_frame(
            {"observation.state": np.asarray([frame_index + 2], dtype=np.float32), "task": "test"}
        )
        recorder.record_sync(
            frame_index,
            capture(second_timestamp + frame_index + 1, second_timestamp, 1),
        )
    recorder.save_episode(dataset, task_info=["test"])
    recorder.close()

    reader = SensorStreamReader(root)
    assert reader.committed_episode_indices == (0, 1)
    assert len(reader.read_raw(instance="gripper_force", episode_index=0)) == 1
    assert len(reader.read_raw(instance="gripper_force", episode_index=1)) == 1
    metadata_template = reader.manifest["storage_layout"]["episode_metadata_path_template"]
    metadata = [
        json.loads((root / metadata_template.format(episode_uid=uid)).read_text(encoding="utf-8"))
        for _index, uid in sorted(reader._index_to_uid.items())
    ]
    assert [item["streams"]["gripper_force"]["provenance"]["device_id"] for item in metadata] == [
        "fake-a",
        "fake-b",
    ]


def test_persisted_tactile_and_sync_reuse_selection_before_concurrent_publish(tmp_path) -> None:
    class ProprioRobot:
        name = "fake"
        robot_type = "fake"
        id = "robot"
        cameras = {}
        observation_features = {"joint.pos": float}
        action_features = {}
        is_connected = True
        is_calibrated = True

        def get_observation(self):
            return {"joint.pos": 0.5}

    root = tmp_path / "causal-frame"
    sensor = FakeSensor()
    robot = SensorizedRobot(ProprioRobot(), {"gripper_force": sensor})
    robot.validate_tactile_v2_profile()
    features = robot.route_observation_dataset_features(
        hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
    )
    dataset = LeRobotDataset.create(
        repo_id="test/sensor-causal-frame",
        fps=30,
        root=root,
        features=features,
    )
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    recorder.start_episode(0, dataset=dataset)

    timestamp_a = time.perf_counter_ns()
    sample_a = sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        timestamp_a,
        arrival_timestamp_ns=timestamp_a,
    )
    raw_a = robot.get_observation()
    capture_a = {
        **robot.last_capture_metadata,
        "sensors": {
            name: dict(reference) for name, reference in robot.last_capture_metadata["sensors"].items()
        },
    }

    # B arrives after A was selected but before frame packing and Sync writing. Its
    # measurement time is the old anchor, so arrival causality alone excludes it from frame 0.
    timestamp_b = capture_a["frame_anchor_ns"]
    arrival_b = time.perf_counter_ns()
    assert arrival_b > capture_a["frame_anchor_ns"]
    sample_b = sensor._publish_sample(
        {"left.normal_force": 10.0, "right.normal_force": 20.0},
        timestamp_b,
        arrival_timestamp_ns=arrival_b,
    )

    robot.assert_current_frame_values(raw_a, dict(raw_a))
    frame_a = build_dataset_frame(dataset.features, raw_a, "observation")
    dataset.add_frame({**frame_a, "task": "test"})
    recorder.record_sync(0, capture_a)

    raw_b = robot.get_observation()
    capture_b = {
        **robot.last_capture_metadata,
        "sensors": {
            name: dict(reference) for name, reference in robot.last_capture_metadata["sensors"].items()
        },
    }
    robot.assert_current_frame_values(raw_b, dict(raw_b))
    frame_b = build_dataset_frame(dataset.features, raw_b, "observation")
    dataset.add_frame({**frame_b, "task": "test"})
    recorder.record_sync(1, capture_b)

    recorder.save_episode(dataset, task_info=["test"])
    recorder.close()
    dataset.finalize()

    reopened = LeRobotDataset("test/sensor-causal-frame", root=root)
    reader = SensorStreamReader(root, instance="gripper_force", episode_index=0, verify="full")
    raw_rows = reader.read_raw()
    sync_rows = reader.read_sync()
    raw_by_sequence = {row["sequence"]: row for row in raw_rows}

    assert reopened.features[OBS_STATE]["names"] == ["joint.pos"]
    assert reopened[0][OBS_TACTILE].tolist() == pytest.approx([1.0, 2.0])
    assert reopened[1][OBS_TACTILE].tolist() == pytest.approx([10.0, 20.0])
    assert [row["sequence"] for row in raw_rows] == [sample_a.sequence, sample_b.sequence]
    assert sync_rows[0]["sensors"]["gripper_force"]["sequence"] == sample_a.sequence
    assert sync_rows[1]["sensors"]["gripper_force"]["sequence"] == sample_b.sequence
    assert raw_by_sequence[sample_b.sequence]["arrival_timestamp_ns"] > sync_rows[0]["frame_anchor_ns"]

    for frame_index, expected in enumerate(([1.0, 2.0], [10.0, 20.0])):
        sequence = sync_rows[frame_index]["sensors"]["gripper_force"]["sequence"]
        selected_values = raw_by_sequence[sequence]["values"]
        assert reopened[frame_index][OBS_TACTILE].tolist() == pytest.approx(expected)
        assert expected == pytest.approx(
            [selected_values["left.normal_force"], selected_values["right.normal_force"]]
        )


def test_manifest_is_stable_but_episode_provenance_is_dynamic(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    manifest = json.loads((tmp_path / "meta" / "sensor_streams.json").read_text())
    recorder.close()

    manifest_text = json.dumps(manifest)
    assert "FakeSensor" not in manifest_text
    assert "device_id" not in manifest_text
    assert manifest["sidecar_schema_version"] == SIDECAR_SCHEMA_VERSION == 2
    assert manifest["storage_layout"]["version"] == 1
    assert manifest["raw_schema"] == "typed_semantic_and_native_structs_v1"
    assert manifest["sync_schema"] == "one_row_per_main_dataset_frame_v1"
    assert manifest["streams"]["gripper_force"]["frame_features"] == [
        "left.normal_force",
        "right.normal_force",
    ]
    assert manifest["frame_view"] == {
        "dataset_key": "observation.tactile",
        "dtype": "float32",
        "shape": [2],
        "alignment": "frame_anchor_ns",
        "sources": [
            {
                "instance": "gripper_force",
                "feature": "left.normal_force",
                "qualified_name": "sensor.gripper_force.left.normal_force",
            },
            {
                "instance": "gripper_force",
                "feature": "right.normal_force",
                "qualified_name": "sensor.gripper_force.right.normal_force",
            },
        ],
    }

    with pytest.raises(ValueError, match="differs from sensor_streams.json"):
        SensorStreamRecorder(tmp_path, {"gripper_force": IncompatibleFakeSensor()})
    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_frame_view_does_not_drop_raw_semantic_or_opt_in_payload(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.record_native_payload = True
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    timestamp = time.perf_counter_ns()
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        timestamp,
        arrival_timestamp_ns=timestamp + 1,
        native_values={"channel_1.register": 7},
        native_payload=b"native",
    )
    recorder.record_sync(0, capture(timestamp + 1, timestamp, 0))
    recorder.record_sync(1, capture(timestamp + 2, timestamp, 0))
    recorder.save_episode(FakeDataset(tmp_path))
    recorder.close()

    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    row = reader.read_raw()[0]
    assert row["values"] == {"left.normal_force": 1.0, "right.normal_force": 2.0}
    assert row["native_payload"] == b"native"
    assert reader.manifest["streams"]["gripper_force"]["frame_features"] == [
        "left.normal_force",
        "right.normal_force",
    ]


def test_v2_frame_view_supports_ordered_sources_across_instances(tmp_path) -> None:
    initialize_main(tmp_path)
    left, right = FakeSensor(), FakeSensor()
    left.config.frame_features = ["left.normal_force"]
    right.config.frame_features = ["right.normal_force"]

    recorder = SensorStreamRecorder(tmp_path, {"left_force": left, "right_force": right})
    manifest = recorder.manifest
    recorder.close()

    assert manifest["frame_view"]["sources"] == [
        {
            "instance": "left_force",
            "feature": "left.normal_force",
            "qualified_name": "sensor.left_force.left.normal_force",
        },
        {
            "instance": "right_force",
            "feature": "right.normal_force",
            "qualified_name": "sensor.right_force.right.normal_force",
        },
    ]


def test_v2_raw_only_manifest_omits_frame_view(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.frame_features = []

    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    manifest = recorder.manifest
    recorder.close()

    assert manifest["streams"]["gripper_force"]["frame_features"] == []
    assert "frame_view" not in manifest


@pytest.mark.parametrize(
    "frame_features",
    [["left.normal_force"], ["right.normal_force", "left.normal_force"]],
)
def test_v2_profile_rejects_partial_or_reordered_frame_features(tmp_path, frame_features) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.frame_features = frame_features

    with pytest.raises(ValueError, match="Sidecar v2 tactile profile"):
        SensorStreamRecorder(tmp_path, {"gripper_force": sensor})

    assert not (tmp_path / ".sensor-writer.lock").exists()
    assert not (tmp_path / "meta/sensor_streams.json").exists()


def test_v2_profile_requires_newtons(tmp_path) -> None:
    initialize_main(tmp_path)

    with pytest.raises(ValueError, match="semantic unit 'N'"):
        SensorStreamRecorder(tmp_path, {"gripper_force": WrongUnitFakeSensor()})

    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_v2_writer_rejects_v1_before_lock_or_recovery(tmp_path, monkeypatch) -> None:
    initialize_main(tmp_path)
    (tmp_path / "meta/sensor_streams.json").write_text(
        json.dumps({"sidecar_schema_version": 1}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "lerobot.datasets.sensor_stream.TransactionRecoveryManager.recover",
        lambda *_args, **_kwargs: pytest.fail("v1 writer must fail before recovery"),
    )

    with pytest.raises(ValueError, match="new Dataset root"):
        SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})

    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_writer_preflight_is_read_only_when_sidecar_is_absent(tmp_path) -> None:
    root = tmp_path / "new-root"

    preflight_sensor_stream_writer(root)

    assert not root.exists()


@pytest.mark.parametrize("manifest", [{}, {"sidecar_schema_version": 3}])
def test_writer_rejects_missing_or_unknown_schema_before_lock(tmp_path, manifest) -> None:
    initialize_main(tmp_path)
    (tmp_path / "meta/sensor_streams.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar_schema_version|Unsupported Sensor Sidecar"):
        SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})

    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_writer_lock_refuses_a_live_writer(tmp_path) -> None:
    initialize_main(tmp_path)
    first = SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})
    try:
        with pytest.raises(SensorRecorderError, match="live writer"):
            SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})
        with pytest.raises(SensorRecorderError, match="live writer"):
            SensorStreamReader(tmp_path)
    finally:
        first.close()


def test_writer_lock_recovers_a_dead_process(monkeypatch, tmp_path) -> None:
    initialize_main(tmp_path)
    lock_path = tmp_path / ".sensor-writer.lock"
    lock_path.write_text(json.dumps({"pid": os.getpid() + 1, "created_ns": 0}), encoding="utf-8")
    monkeypatch.setattr("lerobot.datasets.sensor_stream._process_is_alive", lambda _pid: False)

    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})
    recorder.close()

    assert not lock_path.exists()


def test_recorder_capacity_is_validated_before_dataset_mutation(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config = SensorConfig(required=False, frame_features=[])

    with pytest.raises(ValueError, match="recorder_queue_capacity"):
        SensorStreamRecorder(tmp_path, {"raw_only": sensor})

    assert not (tmp_path / ".sensor-writer.lock").exists()
    assert not (tmp_path / "meta" / "sensor_streams.json").exists()


def test_recorder_rejects_non_uuid_episode_uid_without_starting(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})

    with pytest.raises(ValueError, match="canonical UUID4"):
        recorder.start_episode(0, episode_uid="../outside")

    assert not recorder.is_active
    assert sensor.subscriber_count == 0
    recorder.close()


def test_recorder_queue_overflow_aborts_without_leaking_subscriber(monkeypatch, tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.recorder_queue_capacity = 1
    release_worker = threading.Event()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    monkeypatch.setattr(recorder, "_drain", lambda _instance, _subscription: release_worker.wait())
    recorder.start_episode(0)
    timestamp = time.perf_counter_ns()
    sensor._publish_sample({"left.normal_force": 1.0, "right.normal_force": 2.0}, timestamp)
    sensor._publish_sample({"left.normal_force": 3.0, "right.normal_force": 4.0}, timestamp + 1)

    with pytest.raises(SensorQueueOverflowError):
        recorder.check_health()
    release_worker.set()
    with pytest.raises(SensorQueueOverflowError):
        recorder.close()

    assert sensor.subscriber_count == 0
    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_rerecord_and_worker_failure_cleanup_subscribers_and_threads(monkeypatch, tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    first_threads = recorder.worker_threads
    recorder.abort_episode("explicit rerecord")
    assert sensor.subscriber_count == 0
    assert all(not thread.is_alive() for thread in first_threads)

    recorder.start_episode(0)
    second_threads = recorder.worker_threads
    monkeypatch.setattr(recorder, "_raw_row", lambda *_args: (_ for _ in ()).throw(RuntimeError("write")))
    timestamp = time.perf_counter_ns()
    sensor._publish_sample({"left.normal_force": 1.0, "right.normal_force": 2.0}, timestamp)
    deadline = time.perf_counter() + 1
    while not recorder._worker_errors and time.perf_counter() < deadline:
        time.sleep(0.001)
    with pytest.raises(SensorRecorderError, match="worker"):
        recorder.close()
    assert sensor.subscriber_count == 0
    assert all(not thread.is_alive() for thread in second_threads)
    assert not (tmp_path / ".sensor-writer.lock").exists()


def test_reader_uses_manifest_layout_templates(tmp_path) -> None:
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    timestamp = time.perf_counter_ns()
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0}, timestamp, arrival_timestamp_ns=timestamp + 1
    )
    recorder.record_sync(0, capture(timestamp + 1, timestamp, 0))
    recorder.record_sync(1, capture(timestamp + 2, timestamp, 0))
    recorder.save_episode(FakeDataset(tmp_path))
    recorder.close()

    manifest_path = tmp_path / "meta" / "sensor_streams.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layout = manifest["storage_layout"]
    custom_templates = {
        "raw_path_template": "custom/raw/{instance}/{episode_uid}.parquet",
        "sync_path_template": "custom/sync/{episode_uid}.parquet",
        "episode_metadata_path_template": "custom/meta/{episode_uid}.json",
    }
    for key, template in custom_templates.items():
        source = tmp_path / layout[key].format(instance="gripper_force", episode_uid=uid)
        destination = tmp_path / template.format(instance="gripper_force", episode_uid=uid)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        layout[key] = template
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    assert len(reader.read_raw()) == 1
    assert len(reader.read_sync()) == 2


@pytest.mark.parametrize("prepared", [False, True])
def test_sequence_reset_quarantines_episode_even_after_unsubscribe(tmp_path, prepared):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    if prepared:
        recorder.prepare_episode()
        assert sensor.subscriber_count == 0
    with pytest.raises(RuntimeError, match="recorder ownership"):
        sensor._reset_framework_state()
    with pytest.raises(SensorRecorderError, match="episode fault"):
        if prepared:
            recorder.commit_prepared(FakeDataset(tmp_path))
        else:
            recorder.prepare_episode()
    recorder.close()
    journal = json.loads((tmp_path / "meta/sensor_transactions" / f"{uid}.json").read_text())
    assert journal["state"] == TransactionState.QUARANTINED
    sensor._reset_framework_state()


def test_join_timeout_retains_worker_and_sequence_ownership(tmp_path, monkeypatch):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    release = threading.Event()
    monkeypatch.setattr(recorder, "_drain", lambda *_args: release.wait())
    recorder.start_episode(0)
    threads = recorder.worker_threads
    thread = recorder._threads["gripper_force"]
    original_join = thread.join
    monkeypatch.setattr(thread, "join", lambda timeout: original_join(timeout=0))
    try:
        with pytest.raises(SensorRecorderError, match="did not stop"):
            recorder.close()
        assert recorder.worker_threads == threads
        assert (tmp_path / ".sensor-writer.lock").exists()
        with pytest.raises(RuntimeError, match="recorder ownership"):
            sensor._reset_framework_state()
        with pytest.raises(SensorRecorderError, match="already active"):
            recorder.start_episode(0)
    finally:
        release.set()
        original_join(1)
        with pytest.raises(SensorRecorderError):
            recorder.close()
    assert not (tmp_path / ".sensor-writer.lock").exists()
