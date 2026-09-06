#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import logging
import queue

import pyarrow as pa
import pytest
from test_sensor_stream import FakeDataset, FakeSensor, capture, initialize_main

from lerobot.datasets.sensor_spool import SensorParquetSpoolWriter
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.sensors.diagnostics import SensorQueueDiagnostics


def test_first_above_75_percent_warns_and_subsequent_warnings_are_rate_limited(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr("lerobot.sensors.diagnostics.time.monotonic", lambda: clock[0])
    diagnostics = SensorQueueDiagnostics("test Raw")
    bounded = queue.Queue(maxsize=4)
    for index in range(3):
        bounded.put_nowait(index)
        diagnostics.observe(bounded)
    assert not caplog.records
    bounded.put_nowait(3)
    diagnostics.observe(bounded)
    assert len(caplog.records) == 1
    assert "above 75%" in caplog.text
    for _ in range(100):
        diagnostics.observe(bounded)
    assert len(caplog.records) == 1
    clock[0] = 30
    diagnostics.observe(bounded)
    assert len(caplog.records) == 2
    assert diagnostics.snapshot(bounded) == {
        "queue_size": 4,
        "queue_capacity": 4,
        "queue_utilization": 1,
        "peak_queue_size": 4,
        "overflow_count": 0,
    }


def test_raw_publication_exposes_queue_utilization_and_fatal_overflow(caplog):
    sensor = FakeSensor()
    subscription = sensor.subscribe(4)
    for sequence in range(5):
        sensor._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 2.0}, sequence, arrival_timestamp_ns=sequence
        )
    assert subscription.overflowed and subscription.overflow_count == 1
    metrics = subscription.diagnostics.snapshot(
        subscription.queue, overflow_count=subscription.overflow_count
    )
    assert metrics["peak_queue_size"] == metrics["queue_size"] == 4
    assert metrics["overflow_count"] == 1
    assert len(caplog.records) == 1
    sensor.unsubscribe(subscription)


def test_spool_metrics_cover_lag_sequence_gaps_flush_bytes_and_compression(tmp_path, monkeypatch):
    schema = pa.schema(
        [
            ("sequence", pa.int64()),
            ("timestamp_ns", pa.int64()),
            ("arrival_timestamp_ns", pa.int64()),
            ("is_valid", pa.bool_()),
        ]
    )
    spool = SensorParquetSpoolWriter(tmp_path / "spool", schema, 2, 100)
    monkeypatch.setattr("lerobot.datasets.sensor_spool.time.perf_counter_ns", lambda: 1000)
    for sequence, arrival, valid in [(0, 100, True), (2, 200, False), (3, 300, True)]:
        spool.append(
            {
                "sequence": sequence,
                "timestamp_ns": arrival,
                "arrival_timestamp_ns": arrival,
                "is_valid": valid,
            }
        )
    spool.close()
    final = tmp_path / "final.parquet"
    spool.merge(final)
    stats = spool.diagnostics()
    assert stats["rows"] == 3 and stats["invalid_samples"] == 1 and stats["sequence_gaps"] == 1
    assert stats["flush_count"] == stats["fragments"] == stats["row_groups"] == 2
    assert stats["lag_ns"] == 700 and stats["max_lag_ns"] == 900 and stats["mean_lag_ns"] == 800
    assert stats["buffer_rows"] == 0 and stats["peak_buffer_rows"] == 2
    assert stats["spool_bytes"] == sum(path.stat().st_size for path in spool.root.glob("*.parquet"))
    assert stats["final_bytes"] == final.stat().st_size and stats["final_rows"] == 3
    assert stats["final_row_groups"] == 2
    assert stats["compression_ratio"] == stats["uncompressed_bytes"] / stats["compressed_bytes"]
    assert stats["flush_seconds"] >= stats["max_flush_seconds"] > 0


def test_committed_diagnostics_include_raw_and_sync_survive_cleanup_and_do_not_change_manifest(
    tmp_path, caplog
):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    before = (tmp_path / "meta/sensor_streams.json").read_bytes()
    uid = recorder.start_episode(0)
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0}, 100, arrival_timestamp_ns=101
    )
    recorder.record_sync(0, capture(102, 100, 0))
    recorder.record_sync(1, capture(103, 100, 0))
    with caplog.at_level(logging.INFO):
        recorder.save_episode(FakeDataset(tmp_path), task_info=["test"])
    recorder.close()
    stats = recorder.diagnostics
    assert stats["state"] == "COMMITTED" and stats["episode_uid"] == uid
    assert stats["streams"]["gripper_force"]["rows"] == 1
    assert stats["sync"]["rows"] == 2
    assert stats["sync"]["queue_size"] == 0
    assert stats["sync"]["flush_count"] == 1
    assert stats["streams"]["gripper_force"]["queue_utilization"] == 0
    assert stats["worker_errors"] == {}
    assert "Sensor COMMITTED" in caplog.text
    assert (tmp_path / "meta/sensor_streams.json").read_bytes() == before
    assert "diagnostics" not in json.loads(before)
    stats["streams"].clear()
    assert recorder.diagnostics["streams"]


def test_sync_enqueue_updates_high_watermark_and_overflow_is_fatal(tmp_path, monkeypatch, caplog):
    from lerobot.datasets.sensor_stream import SensorQueueOverflowError

    initialize_main(tmp_path)
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": FakeSensor()})
    # Exercise the real producer without starting a worker: bounded capacity is deterministic.
    recorder._active = True
    recorder._accept_sync = True
    recorder._sync_queue = queue.Queue(maxsize=4)
    try:
        for frame in range(4):
            recorder.record_sync(frame, capture(102, 100, 0))
        with pytest.raises(SensorQueueOverflowError):
            recorder.record_sync(4, capture(102, 100, 0))
        stats = recorder.diagnostics
        assert stats["sync"]["queue_utilization"] == 1
        assert stats["sync"]["peak_queue_size"] == 4
        assert stats["sync"]["overflow_count"] == 1
        assert "__sync__" in stats["worker_errors"]
        assert "Sensor Sync queue above 75%" in caplog.text
    finally:
        recorder._active = False
        recorder.close()
