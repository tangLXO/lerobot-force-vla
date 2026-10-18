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
        self._max_valid_sample: SensorSample | None = None

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
            if sample.is_valid and (
                self._max_valid_sample is None
                or (sample.timestamp_ns, sample.sequence)
                > (self._max_valid_sample.timestamp_ns, self._max_valid_sample.sequence)
            ):
                self._max_valid_sample = sample
            cutoff_ns = sample.arrival_timestamp_ns - self.duration_ns
            rebuild_latest = False
            while self._samples and self._samples[0].arrival_timestamp_ns < cutoff_ns:
                if self._samples.popleft() is self._max_valid_sample:
                    rebuild_latest = True
            if rebuild_latest:
                self._max_valid_sample = max(
                    (item for item in self._samples if item.is_valid),
                    key=lambda item: (item.timestamp_ns, item.sequence),
                    default=None,
                )

    def _latest_valid(self) -> SensorSample | None:
        """Return the retained maximum by measurement time and sequence."""
        with self._lock:
            return self._max_valid_sample

    def snapshot(self) -> tuple[SensorSample, ...]:
        """Return an immutable point-in-time copy."""
        with self._lock:
            return tuple(self._samples)

    def clear(self) -> None:
        """Remove all history entries."""
        with self._lock:
            self._samples.clear()
            self._last_arrival_timestamp_ns = None
            self._max_valid_sample = None

    def __len__(self) -> int:
        """Return the retained sample count."""
        with self._lock:
            return len(self._samples)

    def extend(self, samples: Iterable[SensorSample]) -> None:
        """Append samples in iterable order."""
        for sample in samples:
            self.append(sample)
