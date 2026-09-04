#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Causal selection shared by current-value and temporal-window readers."""

from __future__ import annotations

from collections.abc import Iterable

from .sensor import SensorSample


class SensorDataUnavailableError(RuntimeError):
    """Raised when a required sensor has no valid causal sample."""


def latest_causal_sample(
    samples: Iterable[SensorSample], target_timestamp_ns: int, max_age_ms: float | None
) -> SensorSample | None:
    """Return the newest valid sample known by ``target_timestamp_ns``."""
    max_age_ns = None if max_age_ms is None else int(max_age_ms * 1_000_000)
    best: SensorSample | None = None
    for sample in samples:
        if not sample.is_valid:
            continue
        if sample.timestamp_ns > target_timestamp_ns:
            continue
        if sample.arrival_timestamp_ns > target_timestamp_ns:
            continue
        age_ns = target_timestamp_ns - sample.timestamp_ns
        if max_age_ns is not None and age_ns > max_age_ns:
            continue
        if best is None or (sample.timestamp_ns, sample.sequence) > (best.timestamp_ns, best.sequence):
            best = sample
    return best


def causal_samples_in_interval(
    samples: Iterable[SensorSample], start_timestamp_ns: int, end_timestamp_ns: int
) -> tuple[SensorSample, ...]:
    """Return raw records in ``(start, end]`` that had arrived by ``end``."""
    return tuple(
        sample
        for sample in samples
        if start_timestamp_ns < sample.timestamp_ns <= end_timestamp_ns
        and sample.arrival_timestamp_ns <= end_timestamp_ns
    )
