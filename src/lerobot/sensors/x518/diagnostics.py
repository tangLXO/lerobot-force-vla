# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Bounded X518 runtime statistics, separate from persistent schemas."""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def distribution(values):
    """Summarize a bounded recent window in milliseconds."""
    if not values:
        return dict.fromkeys(("p50", "p95", "p99", "max"))
    return dict(
        zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist(), strict=True)
    )


class AcquisitionDiagnostics:
    """Keep lifetime counters and bounded recent timing samples."""

    def __init__(self):
        """Start bounded percentile windows and lifetime counters."""
        self.count = self.valid = 0
        self.first_response = self.last_response = None
        self.first_arrival = self.last_arrival = None
        self.intervals = deque(maxlen=4096)
        self.latencies = deque(maxlen=4096)
        self.ages = deque(maxlen=4096)
        self.long_intervals = 0
        self.window_start = None
        self.window_count = 0
        self.last_warning = float("-inf")

    def observe(self, sample, rate):
        """Observe published attempts using response time for the rate warning."""
        self.count += 1
        arrival = sample.arrival_timestamp_ns
        if self.first_arrival is None:
            self.first_arrival = arrival
        self.last_arrival = arrival
        if not sample.is_valid:
            return
        self.valid += 1
        stamp = sample.timestamp_ns
        if self.first_response is None:
            self.first_response = stamp
        if self.last_response is not None:
            interval = (stamp - self.last_response) / 1e6
            self.intervals.append(interval)
            self.long_intervals += interval > 4.5
        self.last_response = stamp
        self.latencies.append((arrival - stamp) / 1e6)
        if self.window_start is None:
            self.window_start = stamp
            return
        self.window_count += 1
        elapsed = (stamp - self.window_start) / 1e9
        if elapsed >= 10:
            measured = self.window_count / elapsed
            if rate and measured < rate * 0.99 and stamp / 1e9 - self.last_warning >= 30:
                logger.warning(
                    "X518 response rate %.2f Hz is below 99%% of %.2f Hz; recording retained.", measured, rate
                )
                self.last_warning = stamp / 1e9
            self.window_start, self.window_count = stamp, 0

    def snapshot(self):
        """Return a detached snapshot; percentiles cover at most 4096 recent samples."""

        def rate(count, first, last):
            return (count - 1) * 1e9 / (last - first) if count > 1 and last > first else None

        return {
            "published_attempts": self.count,
            "invalid_attempts": self.count - self.valid,
            "response_rate_hz": rate(self.valid, self.first_response, self.last_response),
            "publication_rate_hz": rate(self.count, self.first_arrival, self.last_arrival),
            "interval_ms": distribution(self.intervals),
            "publication_latency_ms": distribution(self.latencies),
            "selected_age_ms": distribution(self.ages),
            "intervals_above_4_5_ms": self.long_intervals,
            "percentile_window_capacity": 4096,
        }


def summarize_capture(rows, *, target_hz=400, start_ns=None, end_ns=None):
    """Measure actual responses over an explicit capture interval, without resampling."""
    selected = [
        row
        for row in rows
        if (start_ns is None or row["timestamp_ns"] >= start_ns)
        and (end_ns is None or row["timestamp_ns"] <= end_ns)
    ]
    valid = [row for row in selected if row["is_valid"]]
    stamps = np.asarray([row["timestamp_ns"] for row in valid], dtype=np.int64)
    intervals = np.diff(stamps) / 1e6
    latency = [(row["arrival_timestamp_ns"] - row["timestamp_ns"]) / 1e6 for row in valid]
    rate = (
        (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
        if len(stamps) > 1 and stamps[-1] > stamps[0]
        else None
    )
    gaps = sum(b["sequence"] != a["sequence"] + 1 for a, b in zip(selected, selected[1:], strict=False))
    interval_stats = distribution(intervals.tolist())
    latency_stats = distribution(latency)
    checks = {
        "response_rate": bool(rate is not None and target_hz * 0.99 <= rate <= target_hz * 1.01),
        "interval_p99": interval_stats["p99"] is not None and interval_stats["p99"] <= 4.5,
        "publication_latency_p99": latency_stats["p99"] is not None and latency_stats["p99"] <= 10,
        "valid_samples": len(selected) == len(valid) and len(valid) > 1,
        "sequence_continuity": gaps == 0,
    }
    return {
        "sample_count": len(selected),
        "invalid_count": len(selected) - len(valid),
        "response_rate_hz": rate,
        "interval_ms": interval_stats,
        "publication_latency_ms": latency_stats,
        "intervals_above_4_5_ms": int((intervals > 4.5).sum()),
        "sequence_gaps": gaps,
        "checks": checks,
        "passed": all(checks.values()),
    }
