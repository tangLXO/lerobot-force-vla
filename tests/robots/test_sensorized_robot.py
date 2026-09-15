#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import numpy as np
import pytest

from lerobot.configs import FeatureType
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.robots.sensorized_robot import SensorizedRobot, attach_sensors
from lerobot.sensors import Sensor, SensorConfig, SensorDataUnavailableError, SensorFeature
from lerobot.utils.constants import OBS_STATE, OBS_TACTILE
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    dataset_to_policy_features,
    hw_to_dataset_features,
)


class FakeRobot:
    name = "fake"
    robot_type = "fake"
    id = "robot"
    cameras = {}

    def __init__(self):
        self.connected = True
        self.sent = []

    @property
    def observation_features(self):
        return {"joint.pos": float}

    @property
    def action_features(self):
        return {"joint.pos": float}

    @property
    def is_connected(self):
        return self.connected

    @property
    def is_calibrated(self):
        return True

    def get_observation(self):
        return {"joint.pos": 0.5}

    def send_action(self, action):
        self.sent.append(action)
        return action

    def connect(self, calibrate=True):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass


class FakeSensor(Sensor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.connected = True

    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float32", "N"),
            "right.normal_force": SensorFeature("float32", "N"),
        }

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False


class InvalidFeatureSensor(FakeSensor):
    @property
    def features(self):
        return {"tactile.left.normal_force": SensorFeature("float32", "N")}


class WrongUnitSensor(FakeSensor):
    @property
    def features(self):
        return {
            "left.normal_force": SensorFeature("float32", "kg"),
            "right.normal_force": SensorFeature("float32", "N"),
        }


class ConflictingRobot(FakeRobot):
    @property
    def observation_features(self):
        return {
            "joint.pos": float,
            "sensor.gripper_force.left.normal_force": float,
        }


def test_no_sensor_attachment_preserves_robot_identity() -> None:
    robot = FakeRobot()
    assert attach_sensors(robot, {}) is robot


def test_selected_frame_features_capture_in_stable_order(monkeypatch) -> None:
    cfg = SensorConfig(
        sample_rate_hz=100,
        frame_features=["right.normal_force", "left.normal_force"],
    )
    sensor = FakeSensor(cfg)
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    times = iter([100, 200, 300])
    monkeypatch.setattr("lerobot.robots.sensorized_robot.time.perf_counter_ns", lambda: next(times))
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        150,
        arrival_timestamp_ns=160,
    )

    observation = robot.get_observation()
    assert list(robot.observation_features) == [
        "joint.pos",
        "sensor.gripper_force.right.normal_force",
        "sensor.gripper_force.left.normal_force",
    ]
    assert observation == {
        "joint.pos": 0.5,
        "sensor.gripper_force.right.normal_force": 2.0,
        "sensor.gripper_force.left.normal_force": 1.0,
    }
    assert robot.last_capture_metadata["frame_anchor_ns"] == 200


def test_missing_required_data_latches_fault_and_blocks_actions() -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    with pytest.raises((RuntimeError, SensorDataUnavailableError)):
        robot.get_observation()
    with pytest.raises(SensorDataUnavailableError):
        robot.send_action({"joint.pos": 1.0})
    assert robot.inner.sent == []


def test_required_sensor_connect_fails_fast_and_cleans_up() -> None:
    inner = FakeRobot()
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100, startup_timeout_s=0.001))
    robot = SensorizedRobot(inner, {"gripper_force": sensor})

    with pytest.raises(SensorDataUnavailableError, match="did not become ready"):
        robot.connect()

    assert not inner.is_connected
    assert not sensor.is_connected


@pytest.mark.parametrize("required", [True, False])
def test_connect_cleanup_uses_owned_resources_after_failed_start(required):
    class FailedStartSensor(FakeSensor):
        owned = False

        @property
        def has_resources(self):
            return self.owned

        def connect(self):
            self.connected = False
            self.owned = True
            raise RuntimeError("partial startup")

        def disconnect(self):
            self.owned = False

    sensor = FailedStartSensor(SensorConfig(required=required, frame_features=[]))
    robot = SensorizedRobot(FakeRobot(), {"ambient_force": sensor})
    if required:
        with pytest.raises(RuntimeError, match="partial startup"):
            robot.connect()
        assert not robot.inner.is_connected
    else:
        robot.connect()
        assert robot.inner.is_connected
        robot.disconnect()
    assert not sensor.has_resources


def test_optional_raw_only_sensor_records_unavailable_sync_without_frame_view() -> None:
    sensor = FakeSensor(
        SensorConfig(
            required=False,
            frame_features=[],
            recorder_queue_capacity=1,
        )
    )
    sensor.connected = False
    robot = SensorizedRobot(FakeRobot(), {"ambient_force": sensor})

    observation = robot.get_observation()

    assert observation == {"joint.pos": 0.5}
    assert robot.last_capture_metadata["sensors"]["ambient_force"]["status"] == "unavailable"


def test_namespace_unknown_duplicate_and_conflict_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid relative feature paths"):
        SensorizedRobot(FakeRobot(), {"gripper_force": InvalidFeatureSensor(SensorConfig())})

    with pytest.raises(ValueError, match="unknown paths"):
        SensorizedRobot(
            FakeRobot(),
            {
                "gripper_force": FakeSensor(
                    SensorConfig(sample_rate_hz=100, frame_features=["missing.normal_force"])
                )
            },
        )

    with pytest.raises(ValueError, match="duplicates"):
        SensorConfig(
            sample_rate_hz=100,
            frame_features=["left.normal_force", "left.normal_force"],
        )

    with pytest.raises(ValueError, match="conflicts with an existing feature"):
        SensorizedRobot(
            ConflictingRobot(),
            {
                "gripper_force": FakeSensor(
                    SensorConfig(sample_rate_hz=100, frame_features=["left.normal_force"])
                )
            },
        )


def test_sensor_current_view_uses_separate_tactile_dataset_contract() -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    timestamp = 100
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        timestamp,
        arrival_timestamp_ns=timestamp,
    )
    dataset_features = robot.route_observation_dataset_features(
        hw_to_dataset_features(robot.observation_features, "observation")
    )

    assert dataset_features[OBS_STATE]["names"] == ["joint.pos"]
    assert dataset_features[OBS_STATE]["shape"] == (1,)
    assert dataset_features[OBS_TACTILE]["names"] == [
        "sensor.gripper_force.left.normal_force",
        "sensor.gripper_force.right.normal_force",
    ]
    assert dataset_features[OBS_TACTILE]["shape"] == (2,)
    assert set(dataset_features[OBS_STATE]["names"]).isdisjoint(dataset_features[OBS_TACTILE]["names"])
    policy_features = dataset_to_policy_features(dataset_features)
    assert policy_features[OBS_STATE].type == FeatureType.STATE
    assert policy_features[OBS_STATE].shape == (1,)
    assert policy_features[OBS_TACTILE].type == FeatureType.STATE
    assert policy_features[OBS_TACTILE].shape == (2,)
    frame = build_dataset_frame(
        dataset_features,
        {
            "joint.pos": 0.5,
            "sensor.gripper_force.left.normal_force": 1.0,
            "sensor.gripper_force.right.normal_force": 2.0,
        },
        "observation",
    )
    assert frame[OBS_STATE].tolist() == [0.5]
    assert frame[OBS_TACTILE].tolist() == [1.0, 2.0]
    stats = compute_episode_stats(
        {
            OBS_STATE: np.asarray([[0.5], [1.5]], dtype=np.float32),
            OBS_TACTILE: np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
        dataset_features,
    )
    assert stats[OBS_STATE]["mean"].shape == (1,)
    assert stats[OBS_TACTILE]["mean"].shape == (2,)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"], [1.0])
    np.testing.assert_allclose(stats[OBS_TACTILE]["mean"], [2.0, 3.0])


def test_generic_routing_supports_arbitrary_width_but_v2_profile_is_narrow() -> None:
    one_force = SensorizedRobot(
        FakeRobot(),
        {"gripper_force": FakeSensor(SensorConfig(sample_rate_hz=100, frame_features=["left.normal_force"]))},
    )
    routed = one_force.route_observation_dataset_features(
        hw_to_dataset_features(one_force.observation_features, "observation")
    )
    assert routed[OBS_STATE]["shape"] == (1,)
    assert routed[OBS_TACTILE]["shape"] == (1,)
    with pytest.raises(ValueError, match="exactly two ordered"):
        one_force.validate_tactile_v2_profile()

    raw_only = SensorizedRobot(
        FakeRobot(),
        {"ambient": FakeSensor(SensorConfig(required=False, frame_features=[], recorder_queue_capacity=1))},
    )
    raw_only.validate_tactile_v2_profile()
    raw_only_features = raw_only.route_observation_dataset_features(
        hw_to_dataset_features(raw_only.observation_features, "observation")
    )
    assert OBS_TACTILE not in raw_only_features

    two_force = SensorizedRobot(
        FakeRobot(), {"arbitrary_instance": FakeSensor(SensorConfig(sample_rate_hz=100))}
    )
    two_force.validate_tactile_v2_profile()


def test_v2_profile_rejects_wrong_order() -> None:
    robot = SensorizedRobot(
        FakeRobot(),
        {
            "gripper_force": FakeSensor(
                SensorConfig(
                    sample_rate_hz=100,
                    frame_features=["right.normal_force", "left.normal_force"],
                )
            )
        },
    )
    with pytest.raises(ValueError, match="exactly two ordered"):
        robot.validate_tactile_v2_profile()

    wrong_unit = SensorizedRobot(
        FakeRobot(), {"gripper_force": WrongUnitSensor(SensorConfig(sample_rate_hz=100))}
    )
    with pytest.raises(ValueError, match="semantic unit 'N'"):
        wrong_unit.validate_tactile_v2_profile()


def test_state_features_is_not_a_compatibility_alias() -> None:
    with pytest.raises(TypeError, match="state_features"):
        SensorConfig(sample_rate_hz=100, state_features=[])


def test_routing_fails_on_missing_duplicate_or_existing_tactile() -> None:
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": FakeSensor(SensorConfig(sample_rate_hz=100))})
    source = robot.current_frame_feature_names[0]

    with pytest.raises(ValueError, match="missing="):
        robot.route_observation_dataset_features(
            hw_to_dataset_features(FakeRobot().observation_features, "observation")
        )

    with pytest.raises(ValueError, match="duplicated="):
        robot.route_observation_dataset_features(
            {
                OBS_STATE: {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": ["joint.pos", source, source, robot.current_frame_feature_names[1]],
                }
            }
        )

    features = hw_to_dataset_features(robot.observation_features, "observation")
    features[OBS_TACTILE] = {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}
    with pytest.raises(ValueError, match="only owner"):
        robot.route_observation_dataset_features(features)


def test_current_frame_values_cannot_be_dropped_renamed_or_transformed(monkeypatch) -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    times = iter([100, 200, 300])
    monkeypatch.setattr("lerobot.robots.sensorized_robot.time.perf_counter_ns", lambda: next(times))
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        150,
        arrival_timestamp_ns=160,
    )
    raw = robot.get_observation()
    robot.assert_current_frame_values(raw, dict(raw))

    dropped = dict(raw)
    dropped.pop("sensor.gripper_force.left.normal_force")
    with pytest.raises(ValueError, match="dropped or renamed"):
        robot.assert_current_frame_values(raw, dropped)

    transformed = dict(raw)
    transformed["sensor.gripper_force.left.normal_force"] = 1.5
    with pytest.raises(ValueError, match="numerically transformed"):
        robot.assert_current_frame_values(raw, transformed)

    mutated_raw = dict(raw)
    mutated_raw["sensor.gripper_force.left.normal_force"] = 9.0
    with pytest.raises(ValueError, match="mutated after causal selection"):
        robot.assert_current_frame_values(mutated_raw, dict(mutated_raw))


def test_frame_values_and_sync_metadata_share_one_causal_selection(monkeypatch) -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    times = iter([100, 200, 300, 350, 400, 450])
    monkeypatch.setattr("lerobot.robots.sensorized_robot.time.perf_counter_ns", lambda: next(times))
    original_read = sensor.read_latest_before
    read_count = 0

    def counted_read(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(sensor, "read_latest_before", counted_read)
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        150,
        arrival_timestamp_ns=160,
    )

    raw = robot.get_observation()
    selected_metadata = dict(robot.last_capture_metadata["sensors"]["gripper_force"])
    sensor._publish_sample(
        {"left.normal_force": 10.0, "right.normal_force": 20.0},
        190,
        arrival_timestamp_ns=310,
    )
    features = robot.route_observation_dataset_features(
        hw_to_dataset_features(robot.observation_features, "observation")
    )
    frame = build_dataset_frame(features, raw, "observation")

    assert read_count == 1
    assert frame[OBS_TACTILE].tolist() == [1.0, 2.0]
    assert selected_metadata["sequence"] == 0
    assert selected_metadata["timestamp_ns"] == 150

    next_raw = robot.get_observation()
    next_frame = build_dataset_frame(features, next_raw, "observation")
    assert read_count == 2
    assert next_frame[OBS_TACTILE].tolist() == [10.0, 20.0]
    assert robot.last_capture_metadata["sensors"]["gripper_force"]["sequence"] == 1
