"""An independent recorder startup must not invalidate a live control boundary."""

import pytest

from lerobot.sensors import Sensor, SensorConfig, SensorDataUnavailableError, SensorFeature


class ManualSensor(Sensor):
    def __init__(self):
        super().__init__(SensorConfig(expected_sample_rate_hz=400, max_age_ms=100))

    @property
    def features(self):
        return {"left.normal_force": SensorFeature("float32", "N")}

    @property
    def is_connected(self):
        return True

    def connect(self):
        pass

    def disconnect(self):
        pass


def test_independent_episode_boundary_preserves_explicit_owner_selection():
    sensor = ManualSensor()
    owner = sensor.acquire_recorder()
    first = sensor._publish_sample({"left.normal_force": 1.0}, 100, arrival_timestamp_ns=101)
    episode = sensor.acquire_recorder()
    assert episode.start_sequence == 1
    with pytest.raises(RuntimeError, match="not produced"):
        sensor.read_latest_before(102)
    assert sensor.read_latest_before(102, minimum_sequence=owner.start_sequence) is first
    second = sensor._publish_sample({"left.normal_force": 2.0}, 105, arrival_timestamp_ns=106)
    assert sensor.read_latest_before(107) is second
    assert sensor.read_latest_before(107, minimum_sequence=owner.start_sequence) is second
    sensor.release_recorder(episode)
    sensor.release_recorder(owner)


def test_explicit_sequence_boundary_preserves_future_arrival_and_stale_checks():
    sensor = ManualSensor()
    first = sensor._publish_sample({"left.normal_force": 1.0}, 100, arrival_timestamp_ns=101)
    sensor._publish_sample({"left.normal_force": 3.0}, 103, arrival_timestamp_ns=104)
    assert sensor.read_latest_before(102, minimum_sequence=0) is first
    with pytest.raises(SensorDataUnavailableError):
        sensor.read_latest_before(102, max_age_ms=0, minimum_sequence=0)
    with pytest.raises(SensorDataUnavailableError):
        sensor.read_latest_before(102, minimum_sequence=1)


@pytest.mark.parametrize("value", [-1, True, 1.0, "0"])
def test_minimum_sequence_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="minimum_sequence"):
        ManualSensor().read_latest_before(100, minimum_sequence=value)


def test_explicit_boundary_cannot_bypass_recorder_fault():
    sensor = ManualSensor()
    lease = sensor.acquire_recorder()
    sensor._publish_sample({"left.normal_force": 1.0}, 100, arrival_timestamp_ns=101)
    with pytest.raises(RuntimeError, match="Cannot reset"):
        sensor._reset_framework_state()
    with pytest.raises(RuntimeError):
        sensor.read_latest_before(102, minimum_sequence=0)
    sensor.release_recorder(lease)
