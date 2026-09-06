#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_sensor_stream import FakeDataset, FakeSensor, capture, initialize_main

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorDatasetWriterLock, SensorStreamRecorder
from lerobot.datasets.sensor_transaction import SensorTransaction, SensorTransactionError, TransactionState
from lerobot.datasets.sensor_transaction_v2 import (
    SensorTransactionV2,
    TransactionRecoveryManager,
    capture_logical_evidence,
)

CRASH_EXPECTATIONS = {
    "intent:before": None,
    "intent:after": "ABORTED",
    "recording:before": "ABORTED",
    "recording:after": "ABORTED",
    "pointer:before": "ABORTED",
    "pointer:after": "ABORTED",
    "prepared:before": "ABORTED",
    "prepared:after": "ABORTED",
    "pointer_replace:before": "ABORTED",
    "pointer_replace:after": "ABORTED",
    "main_data:before": "ABORTED",
    "main_data:after": "QUARANTINED",
    "main_metadata:before": "QUARANTINED",
    "main_metadata:after": "QUARANTINED",
    "main_info:before": "QUARANTINED",
    "main_info:after": "COMMITTED",
    **{
        f"{phase}:{edge}": "COMMITTED"
        for phase in ("main_saved", "promote", "sidecar_promoted", "committed", "cleanup", "pointer_clear")
        for edge in ("before", "after")
    },
}


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (p.stat().st_mtime_ns, p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != ".sensor-writer.lock"
    }


@pytest.mark.parametrize("phase,expected", CRASH_EXPECTATIONS.items())
def test_real_process_crash_and_three_idempotent_recoveries(tmp_path, phase, expected):
    worker = Path(__file__).with_name("sensor_crash_worker.py")
    result = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), phase], capture_output=True, timeout=30
    )
    assert result.returncode == 73, (
        result.stdout.decode(errors="replace"),
        result.stderr.decode(errors="replace"),
    )
    manager = TransactionRecoveryManager(tmp_path)
    with SensorDatasetWriterLock(tmp_path) as lock:
        manager.recover(writer_lock=lock)
        journals = list((tmp_path / "meta/sensor_transactions").glob("*.json"))
        if expected is None:
            assert not journals
        else:
            assert len(journals) == 1
            transaction = SensorTransaction.load(journals[0])
            assert transaction.state == expected
        state = snapshot(tmp_path)
        for _ in range(3):
            manager.recover(writer_lock=lock)
            if expected is not None:
                assert SensorTransaction.load(journals[0]).replay() == expected
            assert snapshot(tmp_path) == state
    assert not (tmp_path / ".sensor-staging/active_transaction.json").exists()


def prepared_recorder(root):
    initialize_main(root)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0}, 100, arrival_timestamp_ns=101
    )
    for i in range(2):
        recorder.record_sync(i, capture(102, 100, 0))
    transaction = recorder.prepare_episode(task_info=["test"])
    return recorder, transaction, uid


def test_recording_is_discoverable_before_first_subscription(tmp_path, monkeypatch):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    original = sensor.subscribe

    def subscribe(capacity):
        pointer = json.loads((tmp_path / ".sensor-staging/active_transaction.json").read_text())
        journal = SensorTransaction.load(
            tmp_path / "meta/sensor_transactions" / f"{pointer['episode_uid']}.json"
        )
        assert journal.state == "RECORDING"
        return original(capacity)

    monkeypatch.setattr(sensor, "subscribe", subscribe)
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    recorder.abort_episode("test")
    recorder.close()


def test_recovery_requires_writer_lock(tmp_path):
    initialize_main(tmp_path)
    with pytest.raises(SensorTransactionError, match="writer lock"):
        SensorTransactionV2.begin(tmp_path, str(uuid.uuid4()), 0)


@pytest.mark.parametrize("pointer_state", ["missing", "malformed", "stale", "stale_identity"])
def test_pointer_reconstructed_only_from_uncleaned_staging(tmp_path, pointer_state):
    recorder, transaction, uid = prepared_recorder(tmp_path)
    pointer = tmp_path / ".sensor-staging/active_transaction.json"
    if pointer_state == "missing":
        pointer.unlink()
    elif pointer_state == "stale_identity":
        pointer.write_text(json.dumps({"episode_uid": uid, "intent_id": "old generation"}))
    else:
        pointer.write_text(
            "{"
            if pointer_state == "malformed"
            else json.dumps({"episode_uid": str(uuid.uuid4()), "intent_id": "stale"})
        )
    recovered = TransactionRecoveryManager(tmp_path).recover(writer_lock=recorder._writer_lock)
    assert recovered.episode_uid == uid and recovered.state == "ABORTED"
    recorder._reset_episode()
    recorder.close()


def test_multiple_active_candidates_refused_without_mutation(tmp_path):
    recorder, transaction, uid = prepared_recorder(tmp_path)
    other = str(uuid.uuid4())
    staging = tmp_path / ".sensor-staging" / other
    staging.mkdir()
    intent = {
        "episode_uid": other,
        "intent_id": str(uuid.uuid4()),
        "episode_index": 0,
        "precondition": transaction.journal["main_precondition"],
    }
    (staging / "transaction_intent.json").write_text(json.dumps(intent))
    before = snapshot(tmp_path)
    with pytest.raises(SensorTransactionError, match="Multiple conflicting"):
        TransactionRecoveryManager(tmp_path).recover(writer_lock=recorder._writer_lock)
    assert snapshot(tmp_path) == before
    recorder.abort_prepared("test")
    recorder.close()


def test_staging_identity_corruption_is_quarantined(tmp_path):
    recorder, transaction, uid = prepared_recorder(tmp_path)
    path = tmp_path / ".sensor-staging" / uid / "transaction_intent.json"
    intent = json.loads(path.read_text())
    intent["intent_id"] = str(uuid.uuid4())
    path.write_text(json.dumps(intent))
    assert transaction.replay() == "QUARANTINED"
    recorder._reset_episode()
    recorder.close()


def test_existing_mutable_artifact_stamp_is_not_proof_of_no_write(tmp_path):
    recorder, transaction, _uid = prepared_recorder(tmp_path)
    path = tmp_path / "data/shared.parquet"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"existing artifact")
    transaction.register_artifact(path, "data")
    assert not transaction._definitely_unsaved()
    assert transaction.replay() == "QUARANTINED"
    recorder._reset_episode()
    recorder.close()


def test_append_shared_main_artifacts_keeps_old_logical_evidence_valid(tmp_path):
    recorder, transaction, _uid = prepared_recorder(tmp_path)
    recorder.commit_prepared(FakeDataset(tmp_path))
    expected = transaction.journal["main_evidence"]
    data = tmp_path / expected["data_path"]
    metadata = tmp_path / expected["metadata_path"]
    data_rows = pq.read_table(data).to_pylist() + [{"index": 2, "episode_index": 1}]
    meta_rows = pq.read_table(metadata).to_pylist()
    meta_rows.append(
        {**meta_rows[0], "episode_index": 1, "length": 1, "dataset_from_index": 2, "dataset_to_index": 3}
    )
    pq.write_table(pa.Table.from_pylist(data_rows), data, row_group_size=1)
    pq.write_table(pa.Table.from_pylist(meta_rows), metadata, row_group_size=1)
    info = json.loads((tmp_path / "meta/info.json").read_text())
    info.update(total_episodes=2, total_frames=3)
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    assert capture_logical_evidence(transaction) == expected
    assert transaction.replay() == "COMMITTED"
    recorder.close()


def test_save_exception_after_commit_returns_success(tmp_path, monkeypatch):
    recorder, transaction, _uid = prepared_recorder(tmp_path)
    original = transaction.commit

    def fail_once():
        original()
        monkeypatch.setattr(transaction, "commit", original)
        raise OSError("after durable commit")

    monkeypatch.setattr(transaction, "commit", fail_once)
    recorder.commit_prepared(FakeDataset(tmp_path))
    assert transaction.state == TransactionState.COMMITTED
    recorder.close()


def test_real_writer_recovers_success_and_records_next_episode_without_history_load(tmp_path, monkeypatch):
    dataset = LeRobotDataset.create(
        repo_id="test/v2",
        root=tmp_path / "dataset",
        fps=30,
        features={"observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}},
    )
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(dataset.root, {"gripper_force": sensor})
    monkeypatch.setattr(
        "lerobot.datasets.dataset_metadata.load_episodes",
        lambda *_args: pytest.fail("historical metadata load"),
    )
    for episode in range(2):
        recorder.start_episode(episode, dataset=dataset)
        sensor._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 2.0},
            100 + episode,
            arrival_timestamp_ns=101 + episode,
        )
        for i in range(2):
            dataset.add_frame({"observation.state": np.array([i], dtype=np.float32), "task": "test"})
            recorder.record_sync(i, capture(103, 100 + episode, episode))
        transaction = recorder.prepare_episode(task_info=["test"])
        if episode == 0:
            original = transaction.commit

            def fail_once(original=original, transaction=transaction):
                original()
                monkeypatch.setattr(transaction, "commit", original)
                raise OSError("crash after complete seal")

            monkeypatch.setattr(transaction, "commit", fail_once)
        recorder.commit_prepared(dataset)
        assert dataset.meta.total_episodes == episode + 1
        assert dataset.writer.episode_buffer["size"] == 0
    assert len(list((dataset.root / "data").rglob("*.parquet"))) == 2
    assert len(list((dataset.root / "meta/episodes").rglob("*.parquet"))) == 2
    recorder.close()
    dataset.finalize()


def test_v2_open_read_and_hash_work_is_independent_of_unrelated_history(tmp_path, monkeypatch):
    import lerobot.datasets.sensor_stream as stream
    import lerobot.datasets.sensor_transaction as legacy

    original_open, original_hash = Path.open, legacy.sha256_file
    counts = {}

    class MeteredFile:
        def __init__(self, handle):
            self.handle = handle

        def read(self, *args):
            value = self.handle.read(*args)
            counts["bytes"] += len(value.encode() if isinstance(value, str) else value)
            return value

        def readinto(self, buffer):
            count = self.handle.readinto(buffer)
            counts["bytes"] += count or 0
            return count

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

    def metered_open(path, *args, **kwargs):
        assert "unrelated" not in str(path), f"Read unrelated history: {path}"
        counts["opens"] += 1
        return MeteredFile(original_open(path, *args, **kwargs))

    def metered_hash(path):
        counts["hashes"] += 1
        assert "data/chunk" not in path.as_posix() and "meta/episodes" not in path.as_posix()
        return original_hash(path)

    results = []
    for history in (0, 100):
        root = tmp_path / str(history)
        initialize_main(root)
        for i in range(history):
            for directory in ("data/unrelated", "meta/episodes/unrelated"):
                path = root / directory / f"{i}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a valid historical file")
            journal = root / "meta/sensor_transactions" / f"unrelated-{i}.json"
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text("must never be opened by active recovery")
        counts = {"opens": 0, "bytes": 0, "hashes": 0}
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", metered_open)
            patch.setattr(legacy, "sha256_file", metered_hash)
            patch.setattr(stream, "sha256_file", metered_hash)
            patch.setattr(legacy, "capture_main_dataset_state", lambda *_args: pytest.fail("legacy snapshot"))
            sensor = FakeSensor()
            recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
            recorder.start_episode(0)
            sensor._publish_sample(
                {"left.normal_force": 1.0, "right.normal_force": 2.0}, 100, arrival_timestamp_ns=101
            )
            for i in range(2):
                recorder.record_sync(i, capture(102, 100, 0))
            recorder.save_episode(FakeDataset(root), task_info=["test"])
            recorder.close()
        results.append(counts)
    assert results[0]["opens"] == results[1]["opens"]
    assert results[0]["hashes"] == results[1]["hashes"] > 0
    # Decimal lengths of measured runtime values can vary; history must not add reads.
    assert abs(results[0]["bytes"] - results[1]["bytes"]) < 1024


@pytest.mark.parametrize(
    "initial,terminal", [("main_data:before", "ABORTED"), ("main_data:after", "QUARANTINED")]
)
@pytest.mark.parametrize(
    "edge",
    [
        "state:before",
        "state:after",
        "terminal_move:before",
        "terminal_move:after",
        "pointer_clear:before",
        "pointer_clear:after",
    ],
)
def test_crash_during_terminal_recovery_is_idempotent(tmp_path, initial, terminal, edge):
    worker = Path(__file__).with_name("sensor_crash_worker.py")
    for phase in (initial, "recover/" + edge.replace("state", terminal.lower())):
        result = subprocess.run(
            [sys.executable, str(worker), str(tmp_path), phase], capture_output=True, timeout=30
        )
        assert result.returncode == 73, result.stderr.decode(errors="replace")
    manager = TransactionRecoveryManager(tmp_path)
    with SensorDatasetWriterLock(tmp_path) as lock:
        manager.recover(writer_lock=lock)
        path = next((tmp_path / "meta/sensor_transactions").glob("*.json"))
        assert SensorTransaction.load(path).state == terminal
        before = snapshot(tmp_path)
        for _ in range(3):
            manager.recover(writer_lock=lock)
            assert SensorTransaction.load(path).replay() == terminal
            assert snapshot(tmp_path) == before
