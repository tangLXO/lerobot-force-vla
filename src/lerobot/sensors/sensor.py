#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Core contracts and publication machinery for non-visual sensors."""

from __future__ import annotations

import abc
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceNotConnectedError

from .buffer import HistoryBuffer
from .configs import SensorConfig


@dataclass(frozen=True)
class SensorFeature:
    """Schema for one scalar semantic or native sensor value."""

    dtype: str | np.dtype
    unit: str
    shape: tuple[()] = ()

    def __post_init__(self) -> None:
        """Normalize and validate the phase-one scalar schema."""
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biuf":
            raise ValueError(f"SensorFeature dtype must be a numeric NumPy dtype, got {dtype}.")
        if self.shape != ():
            raise ValueError("Phase-one SensorFeature values must be scalar (shape=()).")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("SensorFeature unit must be a non-empty string.")
        object.__setattr__(self, "dtype", dtype.name)


@dataclass(frozen=True)
class SensorSample:
    """One acquisition attempt in the canonical host-monotonic clock domain."""

    timestamp_ns: int
    sequence: int
    values: dict[str, Any] = field(default_factory=dict)
    arrival_timestamp_ns: int | None = None
    native_values: dict[str, Any] | None = None
    native_payload: bytes | None = None
    hardware_timestamp_ns: int | None = None
    hardware_sequence: int | None = None
    is_valid: bool = True
    status: str = "ok"
    error: str | None = None

    def __post_init__(self) -> None:
        """Normalize arrival time and enforce auditable invalid-sample status."""
        if self.arrival_timestamp_ns is None:
            object.__setattr__(self, "arrival_timestamp_ns", self.timestamp_ns)
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("SensorSample status must be a non-empty string.")
        if self.is_valid and self.error is not None:
            raise ValueError("A valid SensorSample cannot contain an error.")
        if not self.is_valid and (self.status == "ok" or not isinstance(self.error, str) or not self.error):
            raise ValueError("An invalid SensorSample must retain a non-ok status and error message.")


@dataclass
class SensorSubscription:
    """Independent bounded queue for one native-rate sample consumer."""

    id: str
    queue: queue.Queue[SensorSample]
    overflowed: bool = False
    overflow_count: int = 0
    closed: bool = False
    start_sequence: int = 0
    end_sequence: int | None = None

    def get(self, timeout: float | None = None) -> SensorSample:
        """Get the next queued native-rate sample."""
        return self.queue.get(timeout=timeout)

    def get_nowait(self) -> SensorSample:
        """Get the next sample without blocking."""
        return self.queue.get_nowait()


@dataclass
class SensorRecorderLease:
    """Recorder ownership retained through draining and transaction cleanup."""

    id: str
    error: str | None = None
    start_sequence: int = 0


class Sensor(abc.ABC):
    """Base sensor with causal history and decoupled recorder subscriptions."""

    def __init__(self, config: SensorConfig):
        """Initialize framework state without touching sensor hardware."""
        self.config = config
        self.sample_rate_hz = config.sample_rate_hz
        self._history = HistoryBuffer(config.history_duration_s)
        self._publication_condition = threading.Condition(threading.RLock())
        self._next_sequence = 0
        self._last_consumed_sequence = -1
        self._subscribers: dict[str, SensorSubscription] = {}
        self._recorder_leases: dict[str, SensorRecorderLease] = {}

    def __enter__(self):
        """Connect the sensor for context-manager use."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Disconnect the sensor on context-manager exit."""
        self.disconnect()

    def __del__(self) -> None:
        """Best-effort cleanup when explicit disconnect was omitted."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def features(self) -> dict[str, SensorFeature]:
        """Disconnected-safe relative semantic feature schema."""

    @property
    def native_features(self) -> dict[str, SensorFeature]:
        """Return the optional device-native value schema."""
        return {}

    @property
    def provenance(self) -> dict[str, Any]:
        """Return episode-level driver provenance."""
        return {"driver": f"{type(self).__module__}.{type(self).__name__}"}

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Whether hardware acquisition is active."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect hardware and start its background acquisition reader."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Stop acquisition and release hardware resources."""

    def _publish_sample(
        self,
        values: dict[str, Any] | None,
        timestamp_ns: int,
        *,
        arrival_timestamp_ns: int | None = None,
        native_values: dict[str, Any] | None = None,
        native_payload: bytes | None = None,
        hardware_timestamp_ns: int | None = None,
        hardware_sequence: int | None = None,
        is_valid: bool = True,
        status: str = "ok",
        error: str | None = None,
    ) -> SensorSample:
        """Validate, sequence, retain, and fan out one acquisition attempt."""
        semantic_values = {} if values is None else dict(values)
        if is_valid:
            expected = set(self.features)
            actual = set(semantic_values)
            if actual != expected:
                raise ValueError(
                    f"Valid sample values must exactly match semantic schema; missing={sorted(expected - actual)}, "
                    f"unexpected={sorted(actual - expected)}."
                )
            self._validate_values(semantic_values, self.features, "semantic")
            if error is not None:
                raise ValueError("A valid SensorSample cannot contain an error.")
        else:
            unexpected = set(semantic_values) - set(self.features)
            if unexpected:
                raise ValueError(f"Unknown semantic sensor values: {sorted(unexpected)}.")
            self._validate_values(semantic_values, self.features, "semantic")
        if native_values is not None:
            unexpected_native = set(native_values) - set(self.native_features)
            if unexpected_native:
                raise ValueError(f"Unknown native sensor values: {sorted(unexpected_native)}.")
            self._validate_values(native_values, self.native_features, "native")
        if native_payload is not None and not isinstance(native_payload, bytes):
            raise TypeError("native_payload must be bytes or None.")

        with self._publication_condition:
            if arrival_timestamp_ns is None:
                arrival_timestamp_ns = time.perf_counter_ns()
            sample = SensorSample(
                timestamp_ns=int(timestamp_ns),
                arrival_timestamp_ns=int(arrival_timestamp_ns),
                sequence=self._next_sequence,
                hardware_timestamp_ns=hardware_timestamp_ns,
                hardware_sequence=hardware_sequence,
                values=semantic_values,
                native_values=dict(native_values) if native_values is not None else None,
                native_payload=native_payload,
                is_valid=is_valid,
                status=status,
                error=error,
            )
            self._history.append(sample)
            self._next_sequence += 1
            for subscription in tuple(self._subscribers.values()):
                if subscription.closed:
                    continue
                try:
                    subscription.queue.put_nowait(sample)
                except queue.Full:
                    subscription.overflowed = True
                    subscription.overflow_count += 1
            self._publication_condition.notify_all()
        return sample

    @staticmethod
    def _validate_values(values: dict[str, Any], schema: dict[str, SensorFeature], label: str) -> None:
        for key, value in values.items():
            if isinstance(value, np.ndarray) and value.shape != ():
                raise ValueError(f"{label} value {key!r} must be scalar.")
            try:
                converted = np.asarray(value, dtype=np.dtype(schema[key].dtype))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{label} value {key!r} is incompatible with dtype {schema[key].dtype}."
                ) from exc
            if converted.shape != ():
                raise ValueError(f"{label} value {key!r} must be scalar.")

    def subscribe(self, capacity: int) -> SensorSubscription:
        """Create an independent bounded native-rate subscriber queue."""
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Sensor subscription capacity must be a positive integer.")
        subscription = SensorSubscription(id=str(uuid.uuid4()), queue=queue.Queue(maxsize=capacity))
        with self._publication_condition:
            subscription.start_sequence = self._next_sequence
            self._subscribers[subscription.id] = subscription
        return subscription

    def acquire_recorder(self) -> SensorRecorderLease:
        """Reserve sequence identity until the recorder releases this lease."""
        with self._publication_condition:
            self._check_recorder_fault()
            lease = SensorRecorderLease(str(uuid.uuid4()), start_sequence=self._next_sequence)
            self._recorder_leases[lease.id] = lease
            return lease

    def release_recorder(self, lease: SensorRecorderLease) -> None:
        """Release ownership only after all recorder workers and cleanup finish."""
        with self._publication_condition:
            self._recorder_leases.pop(lease.id, None)

    def _reset_framework_state(self) -> None:
        """Reinitialize a disconnected driver only outside recorder ownership."""
        with self._publication_condition:
            if self._recorder_leases or self._subscribers:
                reason = "Cannot reset Sensor framework sequence during recorder ownership."
                self._latch_recorder_fault(reason)
                raise RuntimeError(reason)
            self._history.clear()
            self._next_sequence = 0
            self._last_consumed_sequence = -1

    def _latch_recorder_fault(self, reason: str) -> None:
        with self._publication_condition:
            for lease in self._recorder_leases.values():
                if lease.error is None:
                    lease.error = reason
            self._publication_condition.notify_all()

    def _notify_reconnect_required(self, error: Exception) -> bool:
        """Latch required-stream failure; return whether recovery must stop."""
        with self._publication_condition:
            if self.config.required and self._recorder_leases:
                self._latch_recorder_fault(f"Required Sensor needs reconnect: {error}")
                return True
            return False

    def _check_recorder_fault(self) -> None:
        with self._publication_condition:
            for lease in self._recorder_leases.values():
                if lease.error is not None:
                    raise RuntimeError(lease.error)

    def unsubscribe(self, subscription: SensorSubscription) -> None:
        """Remove a subscriber reference and mark it closed."""
        with self._publication_condition:
            removed = self._subscribers.pop(subscription.id, None)
            if removed is not None:
                removed.closed = True
                removed.end_sequence = self._next_sequence

    @property
    def subscriber_count(self) -> int:
        """Return the number of active subscriber queues."""
        with self._publication_condition:
            return len(self._subscribers)

    @property
    def has_subscriber_overflow(self) -> bool:
        """Whether any active subscriber has irrecoverably lost a sample."""
        with self._publication_condition:
            return any(subscription.overflowed for subscription in self._subscribers.values())

    def read(self) -> SensorSample:
        """Return the newest unconsumed valid sample, waiting indefinitely."""
        with self._publication_condition:
            while True:
                self._require_active()
                sample = self._consume_latest_valid()
                if sample is not None:
                    return sample
                self._publication_condition.wait()

    def _consume_latest_valid(self) -> SensorSample | None:
        """Consume the full unseen interval under the publication lock."""
        samples = self._history.snapshot()
        baseline = self._last_consumed_sequence
        if not samples or samples[-1].sequence <= baseline:
            return None
        self._last_consumed_sequence = samples[-1].sequence
        return next(
            (sample for sample in reversed(samples) if sample.sequence > baseline and sample.is_valid), None
        )

    def async_read(self, timeout_ms: float = 200) -> SensorSample:
        """Return the newest unconsumed valid sample within a timeout."""
        self._validate_finite_number("timeout_ms", timeout_ms, allow_zero=True)
        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._publication_condition:
            self._require_active()
            while True:
                sample = self._consume_latest_valid()
                if sample is not None:
                    return sample
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    label = type(self).__name__.removesuffix("Sensor")
                    raise TimeoutError(f"Timed out waiting for a new {label} sample.")
                self._publication_condition.wait(timeout=remaining_s)
                self._require_active()

    def read_latest(self, max_age_ms: float | None = 500) -> SensorSample:
        """Return the current valid sample using the caller's current time."""
        from .synchronization import SensorDataUnavailableError

        label = type(self).__name__.removesuffix("Sensor")
        if max_age_ms is not None:
            if (
                isinstance(max_age_ms, bool)
                or not isinstance(max_age_ms, int | float)
                or not math.isfinite(max_age_ms)
            ):
                raise ValueError("max_age_ms must be a finite number or None.")
            if max_age_ms < 0:
                raise TimeoutError(f"Latest {label} sample exceeds max_age_ms={max_age_ms:g}.")
        try:
            return self.read_latest_before(time.perf_counter_ns(), max_age_ms=max_age_ms)
        except SensorDataUnavailableError as exc:
            raise TimeoutError(f"Latest {label} sample is unavailable or stale.") from exc

    def read_latest_before(self, target_timestamp_ns: int, max_age_ms: float | None = None) -> SensorSample:
        """Select the newest causal valid sample at a fixed target time."""
        from .synchronization import SensorDataUnavailableError, latest_causal_sample

        self._require_active()
        with self._publication_condition:
            boundary = max((lease.start_sequence for lease in self._recorder_leases.values()), default=0)
            samples = tuple(sample for sample in self._history.snapshot() if sample.sequence >= boundary)
        if not samples:
            raise RuntimeError(f"{type(self).__name__} has not produced a sample yet.")
        if max_age_ms is None:
            selected = self.features if self.config.state_features is None else self.config.state_features
            max_age_ms = self.config.resolve_max_age_ms(state_features_present=bool(selected))
        else:
            self._validate_finite_number("max_age_ms", max_age_ms, allow_zero=True)
        sample = latest_causal_sample(samples, target_timestamp_ns, max_age_ms)
        if sample is None:
            raise SensorDataUnavailableError(
                f"{type(self).__name__} has no valid causal sample at {target_timestamp_ns}."
            )
        return sample

    def read_window(self, start_timestamp_ns: int, end_timestamp_ns: int) -> tuple[SensorSample, ...]:
        """Return raw causal records in ``(start, end]`` from short history."""
        from .synchronization import causal_samples_in_interval

        self._require_active()
        if start_timestamp_ns > end_timestamp_ns:
            raise ValueError("Window start_timestamp_ns must not exceed end_timestamp_ns.")
        return causal_samples_in_interval(self._history.snapshot(), start_timestamp_ns, end_timestamp_ns)

    def _require_active(self) -> None:
        self._check_recorder_fault()
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{type(self).__name__} is not connected.")

    @staticmethod
    def _validate_finite_number(name: str, value: float, *, allow_zero: bool) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}.")
