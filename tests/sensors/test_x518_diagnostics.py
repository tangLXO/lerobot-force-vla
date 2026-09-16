# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Diagnostics must not stall publication while calculating percentiles."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lerobot.sensors.x518 import (
    X518ChannelConfig,
    X518Sensor,
    X518SensorConfig,
    diagnostics as diagnostics_module,
)


def test_percentiles_allow_concurrent_publication_and_use_detached_snapshot(monkeypatch):
    sensor = X518Sensor(
        X518SensorConfig(
            host="127.0.0.1",
            expected_sample_rate_hz=400,
            channels={
                "left.normal_force": X518ChannelConfig(channel=1),
                "right.normal_force": X518ChannelConfig(channel=2),
            },
        )
    )
    entered, release = threading.Event(), threading.Event()
    original = diagnostics_module.distribution

    def blocked_distribution(values):
        entered.set()
        assert release.wait(5)
        return original(values)

    def publish(sequence):
        stamp = sequence * 2_500_000
        sensor._publish_acquired(
            {
                "values": {"left.normal_force": 1.0, "right.normal_force": 2.0},
                "timestamp_ns": stamp,
                "arrival_timestamp_ns": stamp + 100_000,
            },
            missed=0,
        )

    for index in range(1, 4):
        publish(index)
    monkeypatch.setattr(diagnostics_module, "distribution", blocked_distribution)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(lambda: sensor.diagnostics)
        try:
            assert entered.wait(5)
            # This must finish even though the diagnostic percentile calculation is blocked.
            pool.submit(publish, 4).result(timeout=2)
        finally:
            release.set()
        before = reading.result(timeout=5)
    after = sensor.diagnostics
    assert before["published_attempts"] == 3
    assert after["published_attempts"] == 4
    assert before["publication_latency_ms"]["p99"] == pytest.approx(0.1)


def test_detached_copy_preserves_bounds_and_isolation():
    stats = diagnostics_module.AcquisitionDiagnostics()
    for window in (stats.intervals, stats.latencies, stats.ages):
        window.extend(range(5000))
    copied = stats.detached_copy()
    for original, detached in zip(
        (stats.intervals, stats.latencies, stats.ages),
        (copied.intervals, copied.latencies, copied.ages),
        strict=True,
    ):
        assert detached.maxlen == 4096
        assert len(detached) == 4096
        original.clear()
        assert len(detached) == 4096
