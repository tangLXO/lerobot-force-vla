#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Thread-safe, bounded-duration sensor sample history."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sensor import SensorSample


class HistoryBuffer:
    """Short-lived history used only for online causal selection.

    Recorder consumers never read this buffer. They receive every published
    sample through independent subscriber queues, so history eviction cannot
    create gaps in native-rate recordings.
    """

    def __init__(self, duration_s: float) -> None:
        """Create a history retaining approximately ``duration_s`` seconds."""
        if duration_s <= 0:
            raise ValueError("History duration_s must be greater than zero.")
        self.duration_ns = int(duration_s * 1_000_000_000)
        self._samples: deque[SensorSample] = deque()
        self._lock = RLock()
        self._last_arrival_timestamp_ns: int | None = None

    def append(self, sample: SensorSample) -> None:
        """Append one sample and evict entries outside the arrival horizon."""
        with self._lock:
            if (
                self._last_arrival_timestamp_ns is not None
                and sample.arrival_timestamp_ns < self._last_arrival_timestamp_ns
            ):
                raise ValueError("Sensor arrival_timestamp_ns must be monotonically non-decreasing.")
            self._last_arrival_timestamp_ns = sample.arrival_timestamp_ns
            self._samples.append(sample)
            cutoff_ns = sample.arrival_timestamp_ns - self.duration_ns
            while self._samples and self._samples[0].arrival_timestamp_ns < cutoff_ns:
                self._samples.popleft()

    def snapshot(self) -> tuple[SensorSample, ...]:
        """Return an immutable point-in-time copy."""
        with self._lock:
            return tuple(self._samples)

    def clear(self) -> None:
        """Remove all history entries."""
        with self._lock:
            self._samples.clear()
            self._last_arrival_timestamp_ns = None

    def __len__(self) -> int:
        """Return the retained sample count."""
        with self._lock:
            return len(self._samples)

    def extend(self, samples: Iterable[SensorSample]) -> None:
        """Append samples in iterable order."""
        for sample in samples:
            self.append(sample)
