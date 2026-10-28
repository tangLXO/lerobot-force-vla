"""Recorder-only Raw inputs that preserve control-process Sensor identities.

These objects are not Sensors: they cannot connect, select history, publish, or
own a hardware lease. The control process retains those responsibilities.
"""

from __future__ import annotations

import copy
import queue
import uuid
from dataclasses import dataclass, field
from typing import Any

from lerobot.sensors import SensorFeature, SensorSample, SensorSubscription
from lerobot.sensors.configs import SensorConfig


@dataclass(frozen=True)
class RawStreamDescriptor:
    """Disconnected-safe stream schema and episode provenance for a writer."""

    config: SensorConfig
    features: dict[str, SensorFeature]
    native_features: dict[str, SensorFeature] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sensor(cls, sensor) -> RawStreamDescriptor:
        """Copy metadata only; never retain or pickle the live Sensor."""
        return cls(
            copy.deepcopy(sensor.config),
            copy.deepcopy(sensor.features),
            copy.deepcopy(sensor.native_features),
            copy.deepcopy(sensor.provenance),
            copy.deepcopy(sensor.diagnostics),
        )


class RawStreamInput:
    """Bounded exact-sequence receiver consumed by SensorStreamRecorder."""

    def __init__(self, descriptor: RawStreamDescriptor, start_sequence: int):
        if type(start_sequence) is not int or start_sequence < 0:
            raise ValueError("Raw start_sequence must be a nonnegative integer")
        self.descriptor = descriptor
        self.subscription = SensorSubscription(
            id=str(uuid.uuid4()),
            queue=queue.Queue(maxsize=descriptor.config.resolve_recorder_queue_capacity()),
            start_sequence=start_sequence,
        )
        self.next_sequence = start_sequence
        self.error: str | None = None
        self.valid_received = False

    def accept(self, sample: SensorSample) -> None:
        """Enqueue original fields without Sensor publication or renumbering."""
        self.check_health()
        if self.subscription.closed:
            raise RuntimeError("Raw input is already sealed")
        if sample.sequence != self.next_sequence:
            self.error = f"Raw sequence gap: expected {self.next_sequence}, received {sample.sequence}"
            raise RuntimeError(self.error)
        try:
            self.subscription.queue.put_nowait(sample)
        except queue.Full as exc:
            self.subscription.overflowed = True
            self.subscription.overflow_count += 1
            self.error = "Recorder Raw input overflow"
            raise RuntimeError(self.error) from exc
        self.next_sequence += 1
        self.valid_received |= sample.is_valid
        self.subscription.diagnostics.observe(self.subscription.queue)

    def seal(self, end_sequence: int, *, error: str | None = None) -> None:
        """Accept an exclusive boundary only after every original sample arrived."""
        if self.subscription.closed:
            if self.subscription.end_sequence != end_sequence or self.error != error:
                raise RuntimeError("Conflicting Raw seal")
            return
        if type(end_sequence) is not int or end_sequence != self.next_sequence:
            raise RuntimeError(
                f"Raw tail incomplete: received through {self.next_sequence}, sealed at {end_sequence}"
            )
        self.error = error or self.error
        self.subscription.end_sequence = end_sequence
        self.subscription.closed = True
        self.check_health()

    def abort(self, reason: str) -> None:
        """Close a failed receiver for cleanup without certifying its tail."""
        self.error = self.error or reason
        self.subscription.end_sequence = self.next_sequence
        self.subscription.closed = True

    def check_health(self) -> None:
        if self.error:
            raise RuntimeError(self.error)
