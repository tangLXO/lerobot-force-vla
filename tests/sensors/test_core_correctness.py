#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import threading
import time
from dataclasses import fields

import pytest

from lerobot.configs.default import SensorWindowConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.sensors import SensorConfig
from tests.sensors.test_buffer import FakeSensor


def publish(sensor, timestamp, arrival, *, valid=True):
    return sensor._publish_sample(
        {"left.normal_force": 1.0} if valid else None,
        timestamp,
        arrival_timestamp_ns=arrival,
        is_valid=valid,
        status="ok" if valid else "read_error",
        error=None if valid else "invalid attempt",
    )


def test_arrival_is_monotonic_while_measurement_can_go_backwards():
    sensor = FakeSensor(SensorConfig())
    publish(sensor, 20, 30)
    publish(sensor, 10, 30)
    with pytest.raises(ValueError, match="monotonically"):
        publish(sensor, 25, 29)
    assert publish(sensor, 15, 31).sequence == 2


def test_history_clear_does_not_reset_framework_sequence_or_cursor():
    sensor = FakeSensor(SensorConfig())
    publish(sensor, 20, 30)
    assert sensor.async_read(0).sequence == 0
    sensor._history.clear()
    assert sensor._last_consumed_sequence == 0
    assert publish(sensor, 1, 2).sequence == 1
    assert sensor.async_read(0).sequence == 1


@pytest.mark.parametrize("method", ["read", "async_read"])
def test_read_selects_latest_valid_in_entire_unconsumed_interval(method):
    sensor = FakeSensor(SensorConfig())
    publish(sensor, 30, 40)
    expected = publish(sensor, 10, 41)
    publish(sensor, 35, 42, valid=False)
    assert getattr(sensor, method)() is expected
    assert sensor._last_consumed_sequence == 2
    with pytest.raises(TimeoutError):
        sensor.async_read(0)


def test_continuous_invalid_waits_until_valid_without_reusing_consumed_sample():
    sensor = FakeSensor(SensorConfig())
    publish(sensor, 1, 1)
    sensor.async_read(0)
    publish(sensor, 2, 2, valid=False)
    with pytest.raises(TimeoutError):
        sensor.async_read(0)
    result = []
    reader = threading.Thread(target=lambda: result.append(sensor.read()), daemon=True)
    reader.start()
    try:
        publish(sensor, 3, 3, valid=False)
        assert not result
        expected = publish(sensor, 4, 4)
        reader.join(1)
        assert result == [expected]
    finally:
        sensor.connected = False
        with sensor._publication_condition:
            sensor._publication_condition.notify_all()
        reader.join(1)


def test_default_arrival_is_generated_under_publication_lock(monkeypatch):
    sensor = FakeSensor(SensorConfig())
    original_clock = time.perf_counter_ns

    def clock():
        assert sensor._publication_condition._is_owned()
        return original_clock()

    monkeypatch.setattr("lerobot.sensors.sensor.time.perf_counter_ns", clock)
    sensor._publish_sample({"left.normal_force": 1.0}, 1)


@pytest.mark.parametrize("required", [True, False])
def test_sequence_reset_fault_is_latched_until_recorder_releases(required):
    sensor = FakeSensor(SensorConfig(required=required, state_features=[]))
    lease = sensor.acquire_recorder()
    subscription = sensor.subscribe(10)
    publish(sensor, 1, 1)
    sensor.unsubscribe(subscription)
    with pytest.raises(RuntimeError, match="recorder ownership"):
        sensor._reset_framework_state()
    assert lease.error is not None
    assert sensor._next_sequence == 1
    sensor.release_recorder(lease)
    sensor._reset_framework_state()
    assert publish(sensor, 2, 2).sequence == 0


@pytest.mark.parametrize("required", [True, False])
def test_only_required_recorded_stream_latches_reconnect_failure(required):
    sensor = FakeSensor(SensorConfig(required=required, state_features=[]))
    lease = sensor.acquire_recorder()
    assert sensor._notify_reconnect_required(ConnectionError("lost")) is required
    assert (lease.error is not None) is required
    sensor.release_recorder(lease)
    assert not sensor._notify_reconnect_required(ConnectionError("outside episode"))


def test_window_age_zero_and_policy_capability_are_explicit():
    assert SensorWindowConfig(duration_ms=100, target_hz=50, max_age_ms=0).max_age_ms == 0
    for age in [-1, float("inf"), float("nan"), True]:
        with pytest.raises(ValueError, match="max_age_ms"):
            SensorWindowConfig(duration_ms=100, target_hz=50, max_age_ms=age)
    assert PreTrainedConfig.supports_sensor_windows is False
    assert "supports_sensor_windows" not in {field.name for field in fields(PreTrainedConfig)}
