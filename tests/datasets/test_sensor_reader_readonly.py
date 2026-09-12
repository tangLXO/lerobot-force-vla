#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_sensor_stream import FakeDataset, FakeSensor, capture, initialize_main

from lerobot.configs.default import SensorWindowConfig
from lerobot.datasets.sensor_stream import SensorDatasetWriterLock, SensorRecorderError, SensorStreamRecorder
from lerobot.datasets.sensor_transaction import JOURNAL_VERSION, SensorTransactionError, sha256_file
from lerobot.datasets.sensor_window import SensorStreamReader, SensorWindowDataset


def create_episode(root):
    initialize_main(root)
    sensor = FakeSensor()
    sensor.config.max_age_ms = 1000
    sensor.config.recorder_flush_rows = 2
    sensor.config.recorder_flush_interval_s = 100
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    timestamps = [900, 899, 100, 99, 800, 799, 200, 199, 700, 699, 300, 299, 600, 599]
    for i, timestamp in enumerate(timestamps):
        sensor._publish_sample(
            {"left.normal_force": float(i), "right.normal_force": -float(i)},
            timestamp * 1_000_000,
            arrival_timestamp_ns=i + 1,
        )
    metadata = capture(1_000_000_000, 900_000_000, 0)
    metadata["sensors"]["gripper_force"]["arrival_timestamp_ns"] = 1
    recorder.record_sync(0, metadata)
    recorder.record_sync(1, metadata)
    recorder.save_episode(FakeDataset(root), task_info=["test"])
    recorder.close()
    return uid


def tree(root):
    return {
        p.relative_to(root).as_posix(): (
            p.is_dir(),
            p.stat().st_mtime_ns,
            None if p.is_dir() else p.read_bytes(),
        )
        for p in [root, *root.rglob("*")]
    }


def journal_path(root, uid):
    return root / "meta/sensor_transactions" / f"{uid}.json"


def downgrade_manifest_to_v1(root):
    path = root / "meta/sensor_streams.json"
    payload = json.loads(path.read_text())
    payload["sidecar_schema_version"] = 1
    payload.pop("frame_view", None)
    for stream in payload["streams"].values():
        stream["state_features"] = stream.pop("frame_features")
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("verify", ["fast", "full"])
def test_read_operations_preserve_tree_mtime_count_and_journals(tmp_path, verify, monkeypatch):
    uid = create_episode(tmp_path)
    monkeypatch.setattr(
        "lerobot.datasets.sensor_transaction.SensorTransaction.replay",
        lambda *_a, **_k: pytest.fail("reader repair"),
    )
    before = tree(tmp_path)
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0, verify=verify)
    reader.read_raw()
    reader.read_sync()
    reader.frame_anchor(0, 0)
    result = reader.get_window(350_000_000, 100, 10, 60)
    assert result.sequence.tolist() == [10]
    assert reader.episode_uid == uid
    assert tree(tmp_path) == before


def test_v1_reader_remains_read_only_and_does_not_synthesize_tactile(tmp_path):
    create_episode(tmp_path)
    downgrade_manifest_to_v1(tmp_path)

    class Base:
        root = tmp_path
        episodes = [0]

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {"episode_index": 0, "frame_index": 0, "observation.state": [1.0, 2.0, 3.0]}

    before = tree(tmp_path)
    wrapper = SensorWindowDataset(Base(), {"gripper_force": SensorWindowConfig(100, 10, max_age_ms=60)})
    item = wrapper[0]

    assert wrapper.reader.manifest["sidecar_schema_version"] == 1
    assert "observation.tactile" not in item
    assert "gripper_force" in item["sensor_windows"]
    assert tree(tmp_path) == before


@pytest.mark.parametrize("version", [None, 3])
def test_reader_rejects_missing_or_unknown_sidecar_schema_version(tmp_path, version):
    create_episode(tmp_path)
    path = tmp_path / "meta/sensor_streams.json"
    payload = json.loads(path.read_text())
    if version is None:
        del payload["sidecar_schema_version"]
    else:
        payload["sidecar_schema_version"] = version
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = tree(tmp_path)

    with pytest.raises(ValueError, match="sidecar_schema_version|Unsupported Sensor Sidecar"):
        SensorStreamReader(tmp_path)

    assert tree(tmp_path) == before


@pytest.mark.parametrize("verify", ["fast", "full"])
def test_dead_writer_lock_is_neither_cleared_nor_replaced(tmp_path, verify, monkeypatch):
    create_episode(tmp_path)
    (tmp_path / ".sensor-writer.lock").write_text(json.dumps({"pid": 99999999, "owner_token": "dead"}))
    monkeypatch.setattr("lerobot.datasets.sensor_verification._process_is_alive", lambda _pid: False)
    before = tree(tmp_path)
    SensorStreamReader(tmp_path, verify=verify).read_raw(instance="gripper_force", episode_index=0)
    assert tree(tmp_path) == before


def test_each_read_entry_refuses_a_new_live_writer(tmp_path):
    create_episode(tmp_path)
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    with SensorDatasetWriterLock(tmp_path):
        before = tree(tmp_path)
        for operation in (
            reader.read_raw,
            reader.read_sync,
            lambda: reader.frame_anchor(0, 0),
            lambda: reader.get_window(350_000_000, 100, 10, 60),
        ):
            with pytest.raises(SensorRecorderError, match="live writer"):
                operation()
        assert tree(tmp_path) == before


def add_unfinished(root, *, shared=False):
    uid = str(uuid.uuid4())
    payload = {
        "journal_version": JOURNAL_VERSION,
        "episode_uid": uid,
        "state": "QUARANTINED",
        "expected_main": {"episode_index": 1},
        "main_artifacts": [],
    }
    if shared:
        payload["main_artifacts"] = [{"path": "data/chunk-000/file-000.parquet", "role": "data"}]
    journal_path(root, uid).write_text(json.dumps(payload))


@pytest.mark.parametrize("verify", ["fast", "full"])
@pytest.mark.parametrize("shared", [False, True])
def test_subset_accepts_unselected_failure_only_when_selected_evidence_intact(tmp_path, verify, shared):
    create_episode(tmp_path)
    add_unfinished(tmp_path, shared=shared)
    before = tree(tmp_path)
    with pytest.raises(SensorTransactionError, match="QUARANTINED"):
        SensorStreamReader(tmp_path, verify=verify)
    reader = SensorStreamReader(tmp_path, episode_index=0, verify=verify)
    assert reader.committed_episode_indices == (0,)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("verify", ["fast", "full"])
def test_subset_refuses_corrupt_shared_logical_range_without_recovery(tmp_path, verify):
    create_episode(tmp_path)
    add_unfinished(tmp_path, shared=True)
    data = tmp_path / "data/chunk-000/file-000.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"index": 0, "episode_index": 0}, {"index": 99, "episode_index": 0}]), data
    )
    before = tree(tmp_path)
    with pytest.raises(SensorTransactionError):
        SensorStreamReader(tmp_path, episode_index=0, verify=verify)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("state", ["RECORDING", "PREPARED", "MAIN_SAVED", "SIDECAR_PROMOTED", "QUARANTINED"])
def test_selected_unfinished_state_is_readonly_error(tmp_path, state):
    uid = create_episode(tmp_path)
    path = journal_path(tmp_path, uid)
    payload = json.loads(path.read_text())
    payload["state"] = state
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    with pytest.raises(SensorTransactionError, match=state):
        SensorStreamReader(tmp_path, episode_index=0)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("statistics", [True, False])
def test_range_prunes_safely_with_unordered_measurements_or_missing_statistics(
    tmp_path, statistics, monkeypatch
):
    uid = create_episode(tmp_path)
    raw = tmp_path / "raw/sensors/gripper_force" / f"{uid}.parquet"
    if not statistics:
        table = pq.read_table(raw)
        pq.write_table(table, raw, row_group_size=2, write_statistics=False)
        path = journal_path(tmp_path, uid)
        payload = json.loads(path.read_text())
        for record in payload["files"]:
            if record["final_path"] == raw.relative_to(tmp_path).as_posix():
                record.update(size=raw.stat().st_size, sha256=sha256_file(raw))
        path.write_text(json.dumps(payload))
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    original = pq.ParquetFile
    seen = []

    class Spy(original):
        def iter_batches(self, *args, **kwargs):
            seen.append(kwargs.get("row_groups"))
            yield from super().iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    monkeypatch.setattr(pq, "read_table", lambda *_a, **_k: pytest.fail("whole Raw read"))
    window = reader.get_window(350_000_000, 100, 10, 60)
    assert window.sequence.tolist() == [10]
    assert [group for groups in seen for group in groups] == ([5] if statistics else list(range(7)))


def test_sequence_boundary_is_always_applied(tmp_path):
    uid = create_episode(tmp_path)
    path = journal_path(tmp_path, uid)
    payload = json.loads(path.read_text())
    payload["episode_start_sequence"] = {"gripper_force": 11}
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    assert reader.get_window(350_000_000, 100, 10, 60).sequence.tolist() == [11]
    assert tree(tmp_path) == before


def test_missing_sequence_boundary_is_rejected(tmp_path):
    uid = create_episode(tmp_path)
    path = journal_path(tmp_path, uid)
    payload = json.loads(path.read_text())
    del payload["episode_start_sequence"]
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    with pytest.raises(SensorTransactionError, match="Missing episode sequence boundary"):
        reader.get_window(350_000_000, 100, 10, 60)
    assert tree(tmp_path) == before


def test_dense_initialization_rejects_unresolved_age_before_raw_read(tmp_path):
    uid = create_episode(tmp_path)
    metadata = tmp_path / "meta/sensor_episodes" / f"{uid}.json"
    payload = json.loads(metadata.read_text())
    payload["streams"]["gripper_force"]["resolved_max_age_ms"] = None
    metadata.write_text(json.dumps(payload))
    path = journal_path(tmp_path, uid)
    journal = json.loads(path.read_text())
    for record in journal["files"]:
        if record["kind"] == "json":
            record.update(size=metadata.stat().st_size, sha256=sha256_file(metadata))
    path.write_text(json.dumps(journal))

    class Base:
        root = tmp_path
        episodes = [0]

    with pytest.raises(ValueError, match="max_age_ms"):
        SensorWindowDataset(Base(), {"gripper_force": SensorWindowConfig(100, 10)})
    SensorWindowDataset(Base(), {"gripper_force": SensorWindowConfig(100, 10, max_age_ms=0)})


def test_existing_reader_rejects_changed_file_instead_of_serving_cached_anchor(tmp_path):
    uid = create_episode(tmp_path)
    reader = SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0)
    reader.frame_anchor(0, 0)
    path = journal_path(tmp_path, uid)
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    with pytest.raises(SensorTransactionError, match="changed since"):
        reader.frame_anchor(0, 0)


@pytest.mark.parametrize("verify", ["fast", "full"])
def test_unverifiable_shared_global_artifact_requires_explicit_recovery(tmp_path, verify):
    create_episode(tmp_path)
    add_unfinished(tmp_path, shared=True)
    uid = str(uuid.uuid4())
    journal_path(tmp_path, uid).write_text(
        json.dumps(
            {
                "journal_version": JOURNAL_VERSION,
                "episode_uid": uid,
                "state": "QUARANTINED",
                "expected_main": {"episode_index": 2},
                "main_artifacts": [{"path": "meta/info.json", "role": "info"}],
            }
        )
    )
    before = tree(tmp_path)
    with pytest.raises(SensorTransactionError, match="Cannot prove shared"):
        SensorStreamReader(tmp_path, episodes=[0], verify=verify)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("layout", [{"type": "remote"}, {"type": "per_episode_parquet", "version": 2}])
def test_unsupported_storage_rejected_at_initialization(tmp_path, layout):
    create_episode(tmp_path)
    path = tmp_path / "meta/sensor_streams.json"
    payload = json.loads(path.read_text())
    payload["storage_layout"] = layout
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    with pytest.raises(ValueError, match="Unsupported sensor storage"):
        SensorStreamReader(tmp_path)
    assert tree(tmp_path) == before


def test_fast_uses_footer_and_full_checks_raw_digest(tmp_path):
    uid = create_episode(tmp_path)
    path = journal_path(tmp_path, uid)
    payload = json.loads(path.read_text())
    record = next(r for r in payload["files"] if "/sensors/gripper_force/" in r["final_path"])
    record["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    SensorStreamReader(tmp_path, verify="fast")
    with pytest.raises(SensorTransactionError, match="digest mismatch"):
        SensorStreamReader(tmp_path, verify="full")
    assert tree(tmp_path) == before


@pytest.mark.parametrize("verify", ["fast", "full"])
def test_sidecar_footer_mismatch_is_readonly_error_in_both_modes(tmp_path, verify):
    uid = create_episode(tmp_path)
    path = journal_path(tmp_path, uid)
    payload = json.loads(path.read_text())
    record = next(r for r in payload["files"] if "/sensors/gripper_force/" in r["final_path"])
    record["row_count"] += 1
    path.write_text(json.dumps(payload))
    before = tree(tmp_path)
    with pytest.raises(SensorTransactionError, match="schema/row count"):
        SensorStreamReader(tmp_path, verify=verify)
    assert tree(tmp_path) == before


def test_anchor_cache_reuses_decoded_sync(tmp_path, monkeypatch):
    create_episode(tmp_path)
    reader = SensorStreamReader(tmp_path)
    first = reader.frame_anchor(0, 0)
    monkeypatch.setattr(pq, "ParquetFile", lambda *_a, **_k: pytest.fail("reopening cached Sync"))
    assert reader.frame_anchor(0, 1) == first
