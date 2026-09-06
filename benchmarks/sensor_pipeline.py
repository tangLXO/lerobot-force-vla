#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Native Sensor capture and window benchmark; no hardware or Hub access required.

Default workload: two 1000 Hz streams, 30 minutes, 30 Hz main frames and 500 ms /
200 Hz windows. Synthetic acquisition is accelerated, with producer-side throttling
to let disk workers drain; production publication still uses nonblocking queues.
Wall-clock speed ratios are descriptive and never CI pass/fail thresholds.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from lerobot.configs.default import SensorWindowConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.datasets.sensor_window import SensorWindowDataset
from lerobot.sensors import Sensor, SensorConfig, SensorFeature


class SyntheticSensor(Sensor):
    def __init__(self):
        super().__init__(SensorConfig(sample_rate_hz=1000, max_age_ms=5, recorder_flush_rows=4096))
        self.connected = False

    @property
    def features(self):
        return {"contact.normal_force": SensorFeature("float32", "N")}

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self._reset_framework_state()
        self.connected = True

    def disconnect(self):
        self.connected = False


def generate(root, duration_s):
    frames = int(duration_s * 30)
    if frames < 1:
        raise ValueError("duration_s must span at least one 30 Hz frame.")
    dataset = LeRobotDataset.create(
        "benchmark/sensors",
        root=root,
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["sensor.left.contact.normal_force", "sensor.right.contact.normal_force"],
            }
        },
    )
    sensors = {name: SyntheticSensor() for name in ("left", "right")}
    for sensor in sensors.values():
        sensor.connect()
    started = time.perf_counter()
    recorder = SensorStreamRecorder(root, sensors)
    try:
        recorder.start_episode(0, dataset=dataset)
        # These timestamps are a synthetic monotonic domain. Queue lag in this
        # accelerated workload is not a measurement of physical device latency.
        base, sample_index = 1_000_000_000_000, 0
        total_samples = frames * 1000 // 30
        for frame in range(frames):
            anchor = base + (frame + 1) * 1_000_000_000 // 30
            while sample_index < total_samples and base + sample_index * 1_000_000 < anchor:
                while any(
                    subscription.queue.qsize() >= subscription.queue.maxsize // 2
                    for subscription in recorder._subscriptions.values()
                ):
                    recorder.check_health()
                    time.sleep(0.0001)
                timestamp = base + sample_index * 1_000_000
                for offset, sensor in enumerate(sensors.values()):
                    sensor._publish_sample(
                        {"contact.normal_force": math.sin(sample_index / 1000 + offset)},
                        timestamp,
                        arrival_timestamp_ns=timestamp + 1,
                    )
                sample_index += 1
            selected = {
                name: sensor.read_latest_before(anchor, max_age_ms=5) for name, sensor in sensors.items()
            }
            dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        [sample.values["contact.normal_force"] for sample in selected.values()],
                        dtype=np.float32,
                    ),
                    "task": "synthetic force benchmark",
                }
            )
            recorder.record_sync(
                frame,
                {
                    "observation_start_ns": anchor - 1,
                    "frame_anchor_ns": anchor,
                    "observation_complete_ns": anchor + 1,
                    "sensors": {
                        name: {
                            "timestamp_ns": sample.timestamp_ns,
                            "arrival_timestamp_ns": sample.arrival_timestamp_ns,
                            "sequence": sample.sequence,
                            "hardware_sequence": None,
                            "age_ns": anchor - sample.timestamp_ns,
                            "status": sample.status,
                        }
                        for name, sample in selected.items()
                    },
                },
            )
        recorder.save_episode(dataset, task_info=["synthetic force benchmark"])
    finally:
        recorder.close()
        dataset.finalize()
        for sensor in sensors.values():
            sensor.disconnect()
    return {
        "seconds": time.perf_counter() - started,
        "logical_seconds": frames / 30,
        "frames": frames,
        "raw_rows_per_stream": total_samples,
        "diagnostics": recorder.diagnostics,
    }


def measure(dataset, indices, *, batch_size, cache_mb):
    wrapper = SensorWindowDataset(
        dataset, {name: SensorWindowConfig(500, 200, 5) for name in ("left", "right")}, cache_mb=cache_mb
    )
    started = time.perf_counter()
    valid_points = 0
    for begin in range(0, len(indices), batch_size):
        items = wrapper.__getitems__(indices[begin : begin + batch_size])
        valid_points += sum(
            int(window["valid_mask"].sum()) for item in items for window in item["sensor_windows"].values()
        )
    elapsed = time.perf_counter() - started
    cache = wrapper.reader._row_group_cache
    return {
        "seconds": elapsed,
        "frames_per_second": len(indices) / elapsed,
        "frames": len(indices),
        "valid_points": valid_points,
        "batch_size": batch_size,
        "row_groups_decoded": cache.misses,
        "row_group_cache_hits": cache.hits,
        "resident_decoded_bytes": cache.decoded_bytes,
        "cache_limit_bytes": cache.limit_bytes,
    }


def run(root, *, duration_s=1800, read_frames=None, batch_size=32, cache_mb=64):
    if not math.isfinite(duration_s) or duration_s * 30 < 1:
        raise ValueError("duration_s must be finite and span at least one frame.")
    if (read_frames is not None and read_frames <= 0) or batch_size <= 0:
        raise ValueError("read_frames and batch_size must be positive.")
    if not math.isfinite(cache_mb) or cache_mb < 0:
        raise ValueError("cache_mb must be finite and non-negative.")
    if root.exists():
        raise ValueError(f"Benchmark root must be new: {root}.")
    captured = generate(root, duration_s)
    dataset = LeRobotDataset("benchmark/sensors", root=root)
    count = len(dataset) if read_frames is None else min(read_frames, len(dataset))
    if count <= 0 or batch_size <= 0:
        raise ValueError("read_frames and batch_size must be positive.")
    sequential = list(range(count))
    random = np.random.default_rng(1729).choice(len(dataset), count, replace=False).tolist()
    report = {
        "workload": {
            "streams": 2,
            "sensor_hz": 1000,
            "frame_hz": 30,
            "duration_s": duration_s,
            "window_duration_ms": 500,
            "window_hz": 200,
        },
        "capture": captured,
    }
    report_path = root / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for mode, indices, size in (
        ("sequential", sequential, 1),
        ("random", random, 1),
        ("batch", random, batch_size),
    ):
        logging.info("Measuring %s: %d frames, batch size %d", mode, len(indices), size)
        report[mode] = measure(dataset, indices, batch_size=size, cache_mb=cache_mb)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info("Completed %s in %.2f seconds", mode, report[mode]["seconds"])
    report["batch_speedup_vs_random_single"] = report["random"]["seconds"] / report["batch"]["seconds"]
    report["non_ci_speedup_target"] = 5
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmarks/sensor_pipeline"))
    parser.add_argument("--duration-s", type=float, default=1800)
    parser.add_argument("--read-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-mb", type=float, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run(
        args.root,
        duration_s=args.duration_s,
        read_frames=args.read_frames,
        batch_size=args.batch_size,
        cache_mb=args.cache_mb,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
