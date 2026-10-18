"""Constant-time current selection retains the complete causal history contract."""

import random

import pytest

from lerobot.sensors import Sensor, SensorConfig, SensorDataUnavailableError, SensorFeature
from lerobot.sensors.synchronization import latest_causal_sample
from lerobot.utils.errors import DeviceNotConnectedError


class ManualSensor(Sensor):
    def __init__(self, *, history_duration_s=2.0):
        super().__init__(SensorConfig(max_age_ms=50, history_duration_s=history_duration_s))
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

    def publish(self, stamp, arrival, *, valid=True):
        return self._publish_sample(
            {"left.normal_force": 0.1} if valid else None,
            stamp,
            arrival_timestamp_ns=arrival,
            is_valid=valid,
            status="ok" if valid else "timeout",
            error=None if valid else "synthetic invalid acquisition",
        )


def test_current_selection_with_800_samples_does_not_snapshot_or_scan(monkeypatch):
    sensor = ManualSensor()
    for index in range(800):
        latest = sensor.publish(1_000_000_000 + index * 2_500_000, 1_000_000_000 + index * 2_500_000)

    def forbidden(*args, **kwargs):
        pytest.fail("Current eligible maximum must not copy or scan sensor history")

    monkeypatch.setattr(sensor._history, "snapshot", forbidden)
    monkeypatch.setattr("lerobot.sensors.synchronization.latest_causal_sample", forbidden)
    assert sensor.read_latest_before(latest.timestamp_ns + 1_000_000, minimum_sequence=0) is latest


@pytest.mark.parametrize("which", ["future_measurement", "future_arrival", "newer_sequence_boundary"])
def test_ineligible_global_maximum_falls_back_without_changing_selection(monkeypatch, which):
    sensor = ManualSensor()
    first = sensor.publish(100_000_000, 101_000_000)
    if which == "future_measurement":
        sensor.publish(110_000_000, 102_000_000)
        target, boundary, expected = 105_000_000, 0, first
    elif which == "future_arrival":
        sensor.publish(103_000_000, 110_000_000)
        target, boundary, expected = 105_000_000, 0, first
    else:
        expected = sensor.publish(90_000_000, 102_000_000)
        target, boundary = 105_000_000, expected.sequence
    snapshot = sensor._history.snapshot
    calls = []

    def observed_snapshot():
        calls.append(True)
        return snapshot()

    monkeypatch.setattr(sensor._history, "snapshot", observed_snapshot)
    assert sensor.read_latest_before(target, minimum_sequence=boundary) is expected
    assert calls


def test_invalid_trailing_attempt_cannot_replace_valid_maximum(monkeypatch):
    sensor = ManualSensor()
    expected = sensor.publish(100, 100)
    sensor.publish(110, 110, valid=False)
    monkeypatch.setattr(
        sensor._history, "snapshot", lambda: pytest.fail("Valid maximum should remain direct")
    )
    assert sensor.read_latest_before(120, max_age_ms=1) is expected


def test_eviction_rebuilds_maximum_and_clear_resets_the_reference():
    sensor = ManualSensor(history_duration_s=0.001)
    former = sensor.publish(9_000_000, 1_000_000)
    retained = sensor.publish(1_500_000, 1_500_000)
    assert sensor._history._latest_valid() is former
    sensor.publish(2_000_000, 2_100_000, valid=False)
    assert sensor._history._latest_valid() is retained
    assert sensor.read_latest_before(2_200_000) is retained
    sensor._history.clear()
    assert sensor._history._latest_valid() is None
    with pytest.raises(RuntimeError, match="has not produced"):
        sensor.read_latest_before(3_000_000)
    replacement = sensor.publish(3_100_000, 3_100_000)
    assert sensor._history._latest_valid() is replacement


@pytest.mark.parametrize("max_age_ms", [True, -1, float("nan"), float("inf")])
def test_direct_candidate_still_validates_age_argument(max_age_ms):
    sensor = ManualSensor()
    sensor.publish(100, 100)
    with pytest.raises(ValueError, match="max_age_ms"):
        sensor.read_latest_before(100, max_age_ms=max_age_ms)


def test_direct_candidate_still_checks_active_state_and_latched_recorder_fault():
    sensor = ManualSensor()
    sensor.publish(100, 100)
    sensor.connected = False
    with pytest.raises(DeviceNotConnectedError, match="not connected"):
        sensor.read_latest_before(100)
    sensor.connected = True
    lease = sensor.acquire_recorder()
    with pytest.raises(RuntimeError, match="Cannot reset"):
        sensor._reset_framework_state()
    with pytest.raises(RuntimeError):
        sensor.read_latest_before(100, minimum_sequence=0)
    sensor.release_recorder(lease)


def test_random_publications_and_queries_match_the_full_history_oracle():
    rng = random.Random(719)
    sensor = ManualSensor(history_duration_s=0.01)
    arrival = 1_000_000_000
    for _ in range(500):
        arrival += rng.randrange(50_000, 500_000)
        measurement = arrival + rng.randrange(-20_000_000, 5_000_000)
        sensor.publish(measurement, arrival, valid=rng.random() > 0.2)
        retained = sensor._history.snapshot()
        maximum = max(
            (sample for sample in retained if sample.is_valid),
            key=lambda sample: (sample.timestamp_ns, sample.sequence),
            default=None,
        )
        assert sensor._history._latest_valid() is maximum
        for _query in range(3):
            boundary = rng.randrange(sensor._next_sequence + 2)
            target = arrival + rng.randrange(-5_000_000, 5_000_000)
            age = rng.choice((None, 0, 1, 5, 50))
            eligible = tuple(sample for sample in retained if sample.sequence >= boundary)
            if not eligible:
                with pytest.raises(RuntimeError, match="has not produced"):
                    sensor.read_latest_before(target, age, minimum_sequence=boundary)
                continue
            expected = latest_causal_sample(eligible, target, 50 if age is None else age)
            if expected is None:
                with pytest.raises(SensorDataUnavailableError):
                    sensor.read_latest_before(target, age, minimum_sequence=boundary)
            else:
                assert sensor.read_latest_before(target, age, minimum_sequence=boundary) is expected
