#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from lerobot.sensors import Sensor, SensorConfig, SensorFeature


class FakeSensor(Sensor):
    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.connected = True

    @property
    def features(self):
        return {"left.normal_force": SensorFeature("float32", "N")}

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False


def test_history_eviction_does_not_remove_recorder_samples() -> None:
    sensor = FakeSensor(SensorConfig(history_duration_s=0.001, sample_rate_hz=1000))
    subscriber = sensor.subscribe(10)
    sensor._publish_sample({"left.normal_force": 1.0}, 1_000_000, arrival_timestamp_ns=1_000_000)
    sensor._publish_sample({"left.normal_force": 2.0}, 3_000_000, arrival_timestamp_ns=3_000_000)

    assert len(sensor._history) == 1
    assert [subscriber.get_nowait().sequence, subscriber.get_nowait().sequence] == [0, 1]


def test_subscriber_overflow_is_latched_without_blocking_publish() -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    subscriber = sensor.subscribe(1)
    sensor._publish_sample({"left.normal_force": 1.0}, 1)
    second = sensor._publish_sample({"left.normal_force": 2.0}, 2)

    assert second.sequence == 1
    assert subscriber.overflowed
    assert subscriber.overflow_count == 1
    sensor.unsubscribe(subscriber)
    assert sensor.subscriber_count == 0


def test_static_capacity_and_max_age_resolution() -> None:
    cfg = SensorConfig(expected_sample_rate_hz=200, recorder_queue_duration_s=1.25)
    assert cfg.resolve_max_age_ms(state_features_present=True) == 15
    assert cfg.resolve_recorder_queue_capacity() == 250

    overridden = SensorConfig(expected_sample_rate_hz=200, max_age_ms=7, recorder_queue_capacity=3)
    assert overridden.resolve_max_age_ms(state_features_present=True) == 7
    assert overridden.resolve_recorder_queue_capacity() == 3
