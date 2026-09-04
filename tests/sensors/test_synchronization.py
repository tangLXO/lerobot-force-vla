#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from lerobot.sensors import SensorSample
from lerobot.sensors.synchronization import latest_causal_sample


def sample(timestamp_ms: float, sequence: int, *, arrival_ms: float | None = None, valid=True):
    timestamp_ns = int(timestamp_ms * 1_000_000)
    return SensorSample(
        timestamp_ns=timestamp_ns,
        arrival_timestamp_ns=int((arrival_ms if arrival_ms is not None else timestamp_ms) * 1_000_000),
        sequence=sequence,
        values={"force": float(sequence)},
        is_valid=valid,
        status="ok" if valid else "read_error",
        error=None if valid else "simulated failure",
    )


def test_latest_before_uses_measurement_and_arrival_causality() -> None:
    samples = [sample(10_015, 0), sample(10_019, 1), sample(10_021, 2)]
    selected = latest_causal_sample(samples, 10_020_000_000, max_age_ms=10)
    assert selected is samples[1]

    equal_but_late = sample(10_020, 3, arrival_ms=10_021)
    selected = latest_causal_sample(samples + [equal_but_late], 10_020_000_000, max_age_ms=10)
    assert selected is samples[1]

    equal_and_available = sample(10_020, 4, arrival_ms=10_020)
    selected = latest_causal_sample(samples + [equal_and_available], 10_020_000_000, max_age_ms=10)
    assert selected is equal_and_available


def test_invalid_latest_falls_back_and_stale_or_no_prior_returns_none() -> None:
    older = sample(100, 0)
    invalid = sample(105, 1, valid=False)
    assert latest_causal_sample([older, invalid], 106_000_000, 10) is older
    assert latest_causal_sample([older], 120_000_000, 10) is None
    assert latest_causal_sample([sample(101, 2)], 100_000_000, 10) is None
