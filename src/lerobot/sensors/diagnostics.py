#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Constant-space queue telemetry; never part of the stable Sensor manifest."""

import logging
import time

logger = logging.getLogger(__name__)


class SensorQueueDiagnostics:
    """Track a queue high-water mark and rate-limit capacity warnings."""

    def __init__(self, label="Sensor subscriber"):
        """Start runtime counters for one subscriber or Sync queue."""
        self.label = label
        self.peak_size = 0
        self._last_warning_s = float("-inf")

    def observe(self, queue):
        """Observe a successful enqueue; warn first above 75%, then at most every 30 s."""
        size = queue.qsize()
        self.peak_size = max(self.peak_size, size)
        now = time.monotonic()
        if queue.maxsize and size / queue.maxsize > 0.75 and now - self._last_warning_s >= 30:
            self._last_warning_s = now
            logger.warning(
                "%s queue above 75%%: %s/%s entries; recorder is falling behind.",
                self.label,
                size,
                queue.maxsize,
            )

    def snapshot(self, queue, *, overflow_count=0):
        """Read current utilization without changing queue or ownership state."""
        size = queue.qsize()
        return {
            "queue_size": size,
            "queue_capacity": queue.maxsize,
            "queue_utilization": size / queue.maxsize if queue.maxsize else 0,
            "peak_queue_size": self.peak_size,
            "overflow_count": overflow_count,
        }
