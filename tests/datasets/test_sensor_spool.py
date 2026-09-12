#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import queue
import sqlite3
import threading
import time
from contextlib import closing

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_sensor_stream import FakeSensor, capture, initialize_main

from lerobot.datasets.sensor_spool import SensorParquetSpoolWriter
from lerobot.datasets.sensor_stream import SensorRecorderError, SensorStreamRecorder
from lerobot.datasets.sensor_transaction import JOURNAL_VERSION
from lerobot.sensors import SensorConfig


def publish(sensor, timestamp, arrival=None):
    return sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        timestamp,
        arrival_timestamp_ns=timestamp + 1 if arrival is None else arrival,
    )


def wait_for(predicate):
    deadline = time.perf_counter() + 3
    while not predicate():
        assert time.perf_counter() < deadline
        time.sleep(0.001)


def test_diagnostic_footer_failure_does_not_reinsert_committed_spool_rows(tmp_path, monkeypatch):
    schema = pa.schema(
        [
            ("sequence", pa.int64()),
            ("timestamp_ns", pa.int64()),
            ("arrival_timestamp_ns", pa.int64()),
            ("is_valid", pa.bool_()),
        ]
    )
    spool = SensorParquetSpoolWriter(tmp_path / "spool", schema, 1, 1.0)

    def fail_footer(*args, **kwargs):
        assert spool.fragment_count == 0
        raise OSError("diagnostic footer read failed")

    monkeypatch.setattr(pq, "read_metadata", fail_footer)
    with pytest.raises(OSError, match="diagnostic footer"):
        spool.append({"sequence": 0, "timestamp_ns": 10, "arrival_timestamp_ns": 11, "is_valid": True})
    spool.close()
    with closing(sqlite3.connect(spool.index_path)) as db:
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 1


@pytest.mark.parametrize("count", [256, 8192])
def test_raw_and_sync_buffers_are_bounded_independent_of_episode_length(tmp_path, count):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.recorder_flush_rows = 32
    sensor.config.recorder_flush_interval_s = 100
    sensor.config.recorder_queue_capacity = 256
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    for i in range(count):
        publish(sensor, i * 1000)
        recorder.record_sync(i, capture(i * 1000 + 2, i * 1000, i))
        if i % 64 == 63:
            wait_for(
                lambda: (
                    recorder._subscriptions["gripper_force"].queue.empty() and recorder._sync_queue.empty()
                )
            )
    transaction = recorder.prepare_episode()
    raw_spool = recorder._spools["gripper_force"]
    assert raw_spool.peak_buffer_rows <= 32
    assert recorder._sync_spool.peak_buffer_rows <= 4096
    assert raw_spool.fragment_count == count // 32
    assert recorder._sync_spool.fragment_count >= 1
    assert not hasattr(recorder, "_raw_rows") and not hasattr(recorder, "_sync_rows")
    with closing(sqlite3.connect(recorder._sync_spool.index_path)) as db:
        assert db.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == count
    raw = tmp_path / ".sensor-staging" / uid / "raw/sensors/gripper_force" / f"{uid}.parquet"
    sync = tmp_path / ".sensor-staging" / uid / "raw/sync" / f"{uid}.parquet"
    assert pq.read_metadata(raw).num_rows == count
    assert pq.read_metadata(sync).num_rows == count
    assert transaction.journal["journal_version"] == JOURNAL_VERSION == 1
    recorder.abort_prepared("test finished")
    recorder.close()


def test_time_flush_closes_low_rate_fragment_before_episode_stop(tmp_path):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.recorder_flush_interval_s = 0.01
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    publish(sensor, 1)
    wait_for(lambda: recorder._spools["gripper_force"].fragment_count == 1)
    assert recorder.is_active
    recorder.abort_episode("test finished")
    recorder.close()


@pytest.mark.parametrize("required", [True, False])
def test_startup_barrier_includes_required_raw_only_and_ignores_old_history(tmp_path, required):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.required = required
    sensor.config.frame_features = []
    sensor.config.startup_timeout_s = 0.02
    publish(sensor, time.perf_counter_ns())
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    if required:
        with pytest.raises(SensorRecorderError, match="startup timeout"):
            recorder.wait_until_ready()
        publish(sensor, time.perf_counter_ns())
    recorder.wait_until_ready()
    recorder.abort_episode("test finished")
    recorder.close()


def test_episode_boundary_applies_to_causal_selection_even_for_older_measurement(tmp_path):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    old = publish(sensor, 100, 101)
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    current = publish(sensor, 90, 102)
    assert sensor.read_latest_before(103, 100).sequence == current.sequence
    assert current.sequence > old.sequence
    recorder.abort_episode("test finished")
    recorder.close()


@pytest.mark.parametrize("known_window", [True, False])
def test_safe_trim_retains_unknown_windows_and_filters_late_old_rows(tmp_path, known_window):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    publish(sensor, 10, 100)
    recorder.trim_before(100, max_age_ms=0.00005 if known_window else None)
    publish(sensor, 20, 101)
    publish(sensor, 50, 102)
    recorder.record_sync(
        0,
        capture(103, 50, 2)
        | {
            "sensors": {
                "gripper_force": dict(
                    capture(103, 50, 2)["sensors"]["gripper_force"], arrival_timestamp_ns=102
                )
            }
        },
    )
    recorder.prepare_episode()
    raw = tmp_path / ".sensor-staging" / uid / "raw/sensors/gripper_force" / f"{uid}.parquet"
    assert pq.read_table(raw)["sequence"].to_pylist() == ([2] if known_window else [0, 1, 2])
    recorder.abort_prepared("test finished")
    recorder.close()


@pytest.mark.parametrize("corruption", ["missing", "timestamp", "arrival", "future", "invalid", "trimmed"])
def test_bad_required_sync_is_quarantined_before_prepared(tmp_path, corruption):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    if corruption == "trimmed":
        recorder.trim_before(200, max_age_ms=0)
    if corruption == "invalid":
        sensor._publish_sample(
            None, 100, arrival_timestamp_ns=101, is_valid=False, status="error", error="test"
        )
    else:
        publish(sensor, 100)
    metadata = capture(102, 100, 0)
    ref = metadata["sensors"]["gripper_force"]
    if corruption == "missing":
        ref["sequence"] = 1
    elif corruption in ("timestamp", "arrival"):
        ref[f"{corruption}_timestamp_ns" if corruption == "arrival" else "timestamp_ns"] += 1
    elif corruption == "future":
        metadata = capture(99, 100, 0)
    recorder.record_sync(0, metadata)
    with pytest.raises(SensorRecorderError, match="Required Sync reference"):
        recorder.prepare_episode()
    recorder.close()
    journal = json.loads((tmp_path / "meta/sensor_transactions" / f"{uid}.json").read_text())
    assert journal["state"] == "QUARANTINED"
    assert not (tmp_path / ".sensor-staging" / uid).exists()


def test_sync_overflow_is_fatal_and_stop_does_not_accept_more_frames(tmp_path, monkeypatch):
    initialize_main(tmp_path)
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})
    release = threading.Event()
    monkeypatch.setattr(recorder, "_drain_sync", lambda: release.wait())
    recorder.start_episode(0)
    recorder._sync_queue = queue.Queue(maxsize=1)
    recorder.record_sync(0, capture(2, 1, 0))
    with pytest.raises(RuntimeError, match="Sync queue overflowed"):
        recorder.record_sync(1, capture(2, 1, 0))
    release.set()
    with pytest.raises(SensorRecorderError):
        recorder.close()
    with pytest.raises(SensorRecorderError):
        recorder.record_sync(1, capture(2, 1, 0))


def test_fragment_corruption_closes_handles_and_quarantines_on_windows(tmp_path):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    sensor.config.recorder_flush_rows = 1
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    publish(sensor, 100)
    wait_for(lambda: recorder._spools["gripper_force"].fragment_count == 1)
    fragment = recorder._spools["gripper_force"].root / "fragment-00000000.parquet"
    fragment.write_bytes(b"corrupt")
    with pytest.raises(pa.ArrowInvalid):
        recorder.prepare_episode()
    recorder.close()
    assert (tmp_path / ".sensor-quarantine" / uid / "spool/gripper_force" / fragment.name).exists()


def test_spool_uses_arrow_batches_for_final_merge(tmp_path, monkeypatch):
    schema = pa.schema(
        [
            pa.field("sequence", pa.int64()),
            pa.field("timestamp_ns", pa.int64()),
            pa.field("arrival_timestamp_ns", pa.int64()),
            pa.field("is_valid", pa.bool_()),
        ]
    )
    spool = SensorParquetSpoolWriter(tmp_path / "spool", schema, 2, 100)
    for i in range(7):
        spool.append({"sequence": i, "timestamp_ns": i, "arrival_timestamp_ns": i, "is_valid": True})
    spool.close()
    monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: pytest.fail("unbounded table read"))
    assert spool.merge(tmp_path / "merged.parquet") == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recorder_flush_rows": 0},
        {"recorder_flush_rows": True},
        {"recorder_flush_interval_s": 0},
        {"recorder_flush_interval_s": float("inf")},
    ],
)
def test_invalid_flush_config(kwargs):
    with pytest.raises(ValueError, match="recorder_flush"):
        SensorConfig(**kwargs)


def test_fragment_instance_identity_is_verified_without_changing_final_schema(tmp_path):
    schema = pa.schema(
        [
            pa.field("sequence", pa.int64()),
            pa.field("timestamp_ns", pa.int64()),
            pa.field("arrival_timestamp_ns", pa.int64()),
            pa.field("is_valid", pa.bool_()),
        ]
    )
    first = SensorParquetSpoolWriter(tmp_path / "left", schema, 1, 1, episode_uid="episode", instance="left")
    second = SensorParquetSpoolWriter(
        tmp_path / "right", schema, 1, 1, episode_uid="episode", instance="right"
    )
    for spool in (first, second):
        spool.append({"sequence": 0, "timestamp_ns": 1, "arrival_timestamp_ns": 2, "is_valid": True})
        spool.close()
    first.merge(tmp_path / "final.parquet")
    assert pq.read_schema(tmp_path / "final.parquet").equals(schema, check_metadata=True)
    (first.root / "fragment-00000000.parquet").write_bytes(
        (second.root / "fragment-00000000.parquet").read_bytes()
    )
    with pytest.raises(ValueError, match="stream identity"):
        first.merge(tmp_path / "bad.parquet")


def test_raw_disk_index_is_cross_checked_against_fragment_and_handles_close(tmp_path):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    uid = recorder.start_episode(0)
    publish(sensor, 100)
    recorder._stop_workers()
    with closing(sqlite3.connect(recorder._spools["gripper_force"].index_path)) as db:
        db.execute("UPDATE samples SET arrival_timestamp_ns=999")
        db.commit()
    with pytest.raises(SensorRecorderError, match="disk reference index mismatch"):
        recorder.prepare_episode()
    recorder.close()
    assert (tmp_path / ".sensor-quarantine" / uid / "spool/gripper_force/index.sqlite").exists()
