#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Actual abrupt exits at sensor transaction boundaries (invoked by pytest)."""

import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import lerobot.datasets.sensor_transaction as transaction_module
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.sensors import Sensor, SensorConfig, SensorFeature


def main():
    root, phase = Path(sys.argv[1]), sys.argv[2]
    recovering = phase.startswith("recover/")
    if recovering:
        phase = phase.removeprefix("recover/")
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    template = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    if not recovering:
        (root / "meta/info.json").write_text(
            json.dumps({"total_episodes": 0, "total_frames": 0, "data_path": template})
        )

    def checkpoint(name):
        if phase == name:
            os._exit(73)

    original_atomic = transaction_module._atomic_write_json

    def atomic(path, payload):
        label = (
            "intent"
            if path.name == "transaction_intent.json"
            else "pointer"
            if path.name == "active_transaction.json"
            else str(payload.get("state", "other")).lower()
        )
        checkpoint(label + ":before")
        original_atomic(path, payload)
        checkpoint(label + ":after")

    transaction_module._atomic_write_json = atomic
    original_replace = os.replace

    def replace(source, destination):
        terminal_move = ".sensor-quarantine" in str(destination)
        if terminal_move:
            checkpoint("terminal_move:before")
        is_pointer = Path(destination).name == "active_transaction.json"
        if is_pointer:
            checkpoint("pointer_replace:before")
        is_promote = ".sensor-staging" in str(source) and ".sensor-staging" not in str(destination)
        if is_promote:
            checkpoint("promote:before")
        original_replace(source, destination)
        if terminal_move:
            checkpoint("terminal_move:after")
        if is_pointer:
            checkpoint("pointer_replace:after")
        if is_promote:
            checkpoint("promote:after")

    os.replace = replace
    original_cleanup = transaction_module.SensorTransaction._cleanup_staging

    def cleanup(transaction):
        checkpoint("cleanup:before")
        original_cleanup(transaction)
        checkpoint("cleanup:after")

    transaction_module.SensorTransaction._cleanup_staging = cleanup
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name == "active_transaction.json":
            checkpoint("pointer_clear:before")
        original_unlink(path, *args, **kwargs)
        if path.name == "active_transaction.json":
            checkpoint("pointer_clear:after")

    Path.unlink = unlink
    if recovering:
        transaction_module.TransactionRecoveryManager(root).recover()
        raise AssertionError(f"Recovery crash point was not reached: {phase}")

    class Force(Sensor):
        @property
        def features(self):
            return {
                "left.normal_force": SensorFeature("float32", "N"),
                "right.normal_force": SensorFeature("float32", "N"),
            }

        @property
        def is_connected(self):
            return True

        def connect(self):
            pass

        def disconnect(self):
            pass

    class MainDataset:
        def save_episode(self):
            paths = {
                "data": root / "data/chunk-000/file-000.parquet",
                "episode_metadata": root / "meta/episodes/chunk-000/file-000.parquet",
                "info": root / "meta/info.json",
            }
            for role, path in paths.items():
                self._sensor_transaction.register_artifact(path, role)
                path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint("main_data:before")
            pq.write_table(
                pa.Table.from_pylist([{"index": i, "episode_index": 0, "value": float(i)} for i in range(2)]),
                paths["data"],
            )
            checkpoint("main_data:after")
            checkpoint("main_metadata:before")
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 0,
                            "length": 2,
                            "dataset_from_index": 0,
                            "dataset_to_index": 2,
                            "data/chunk_index": 0,
                            "data/file_index": 0,
                            "tasks": ["test"],
                        }
                    ]
                ),
                paths["episode_metadata"],
            )
            checkpoint("main_metadata:after")
            checkpoint("main_info:before")
            paths["info"].write_text(
                json.dumps({"total_episodes": 1, "total_frames": 2, "data_path": template})
            )
            checkpoint("main_info:after")

    sensor = Force(SensorConfig(sample_rate_hz=100, max_age_ms=100))
    recorder = SensorStreamRecorder(root, {"force": sensor})
    recorder.start_episode(0)
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0}, 100, arrival_timestamp_ns=101
    )
    for i in range(2):
        recorder.record_sync(
            i,
            {
                "observation_start_ns": 99,
                "frame_anchor_ns": 102,
                "observation_complete_ns": 103,
                "sensors": {
                    "force": {
                        "timestamp_ns": 100,
                        "arrival_timestamp_ns": 101,
                        "sequence": 0,
                        "hardware_sequence": None,
                        "age_ns": 2,
                        "status": "ok",
                    }
                },
            },
        )
    recorder.save_episode(MainDataset(), task_info=["test"])
    raise AssertionError(f"Crash point was not reached: {phase}")


if __name__ == "__main__":
    main()
