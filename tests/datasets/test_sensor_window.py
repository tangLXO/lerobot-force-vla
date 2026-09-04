#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import numpy as np
import pytest
from test_sensor_stream import FakeDataset, FakeSensor, capture, initialize_main

from lerobot.configs import DatasetConfig, SensorWindowConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.datasets.sensor_window import SensorStreamReader
from lerobot.policies.act.configuration_act import ACTConfig


def committed_reader(tmp_path, samples):
    initialize_main(tmp_path)
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(tmp_path, {"gripper_force": sensor})
    recorder.start_episode(0)
    for timestamp, arrival, valid, values in samples:
        sensor._publish_sample(
            values,
            timestamp,
            arrival_timestamp_ns=arrival,
            is_valid=valid,
            status="ok" if valid else "invalid",
            error=None if valid else "simulated failure",
        )
    end = max(item[0] for item in samples) + 1_000_000
    recorder.record_sync(0, capture(end, samples[-1][0], len(samples) - 1))
    recorder.record_sync(1, capture(end, samples[-1][0], len(samples) - 1))
    recorder.save_episode(FakeDataset(tmp_path))
    recorder.close()
    return SensorStreamReader(tmp_path, instance="gripper_force", episode_index=0), end


def test_dense_window_uses_same_timestamp_and_arrival_causality(tmp_path) -> None:
    base = 10_000_000_000
    reader, _end = committed_reader(
        tmp_path,
        [
            (base + 15_000_000, base + 15_000_001, True, {"left.normal_force": 1, "right.normal_force": 2}),
            (base + 19_000_000, base + 19_000_001, True, {"left.normal_force": 3, "right.normal_force": 4}),
            (base + 20_000_000, base + 21_000_000, True, {"left.normal_force": 5, "right.normal_force": 6}),
            (base + 21_000_000, base + 21_000_001, True, {"left.normal_force": 7, "right.normal_force": 8}),
        ],
    )
    window = reader.get_window(
        base + 20_000_000,
        duration_ms=20,
        target_hz=100,
        max_age_ms=20,
    )

    assert window.values.shape == (2, 2)
    assert window.valid_mask.tolist() == [False, True]
    assert window.source_timestamp_ns.tolist() == [-1, base + 19_000_000]
    assert window.sequence.tolist() == [-1, 1]
    assert window.age_ns.tolist() == [-1, 1_000_000]


def test_raw_window_retains_invalid_rows_with_false_mask(tmp_path) -> None:
    base = 1_000_000_000
    reader, end = committed_reader(
        tmp_path,
        [
            (base, base, True, {"left.normal_force": 1, "right.normal_force": 2}),
            (base + 1_000_000, base + 1_000_001, False, {}),
        ],
    )
    window = reader.get_window(end, duration_ms=10)
    assert window.valid_mask.tolist() == [True, False]
    np.testing.assert_array_equal(window.values[1], np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("source_hz", "target_hz"),
    [(200, 30), (500, 50), (1000, 100)],
)
def test_rate_alignment_never_uses_future_or_late_samples(tmp_path, source_hz: int, target_hz: int) -> None:
    base = 20_000_000_000
    duration_ms = 100
    period_ns = 1_000_000_000 // source_hz
    samples = [
        (
            base + offset,
            base + offset + period_ns // 4,
            True,
            {"left.normal_force": offset / 1e9, "right.normal_force": -offset / 1e9},
        )
        for offset in range(0, duration_ms * 1_000_000 + 1, period_ns)
    ]
    reader, _ = committed_reader(tmp_path, samples)
    window = reader.get_window(
        base + duration_ms * 1_000_000,
        duration_ms=duration_ms,
        target_hz=target_hz,
        max_age_ms=20,
    )
    arrival_by_timestamp = {timestamp: arrival for timestamp, arrival, *_ in samples}

    assert len(window.values) == int(np.ceil(duration_ms * target_hz / 1000))
    for target, source, valid in zip(
        window.target_timestamp_ns,
        window.source_timestamp_ns,
        window.valid_mask,
        strict=True,
    ):
        if valid:
            assert source <= target
            assert arrival_by_timestamp[int(source)] <= target


def test_training_rejects_windows_for_phase_one_policies_before_dataset_io() -> None:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="test/not-read",
            sensor_windows={"gripper_force": SensorWindowConfig(duration_ms=100, target_hz=50)},
        ),
        policy=ACTConfig(),
    )

    with pytest.raises(ValueError, match="does not declare temporal sensor-window support"):
        make_dataset(cfg)
