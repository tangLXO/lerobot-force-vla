#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import numpy as np
import pytest

from lerobot.configs import FeatureType
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.robots.sensorized_robot import SensorizedRobot, attach_sensors
from lerobot.sensors import Sensor, SensorConfig, SensorDataUnavailableError, SensorFeature
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


def test_selected_features_merge_in_stable_order(monkeypatch) -> None:
    cfg = SensorConfig(
        sample_rate_hz=100,
        state_features=["right.normal_force", "left.normal_force"],
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


def test_optional_raw_only_sensor_records_unavailable_sync_without_state() -> None:
    sensor = FakeSensor(
        SensorConfig(
            required=False,
            state_features=[],
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
                    SensorConfig(sample_rate_hz=100, state_features=["missing.normal_force"])
                )
            },
        )

    with pytest.raises(ValueError, match="duplicates"):
        SensorConfig(
            sample_rate_hz=100,
            state_features=["left.normal_force", "left.normal_force"],
        )

    with pytest.raises(ValueError, match="conflicts with an existing feature"):
        SensorizedRobot(
            ConflictingRobot(),
            {
                "gripper_force": FakeSensor(
                    SensorConfig(sample_rate_hz=100, state_features=["left.normal_force"])
                )
            },
        )


def test_sensor_state_uses_standard_policy_dataset_contract() -> None:
    sensor = FakeSensor(SensorConfig(sample_rate_hz=100))
    robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
    timestamp = 100
    sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 2.0},
        timestamp,
        arrival_timestamp_ns=timestamp,
    )
    dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    assert dataset_features["observation.state"]["names"] == [
        "joint.pos",
        "sensor.gripper_force.left.normal_force",
        "sensor.gripper_force.right.normal_force",
    ]
    assert dataset_features["observation.state"]["shape"] == (3,)
    policy_features = dataset_to_policy_features(dataset_features)
    assert policy_features["observation.state"].type == FeatureType.STATE
    assert policy_features["observation.state"].shape == (3,)
    frame = build_dataset_frame(
        dataset_features,
        {
            "joint.pos": 0.5,
            "sensor.gripper_force.left.normal_force": 1.0,
            "sensor.gripper_force.right.normal_force": 2.0,
        },
        "observation",
    )
    assert frame["observation.state"].tolist() == [0.5, 1.0, 2.0]
    stats = compute_episode_stats(
        {"observation.state": np.asarray([[0.5, 1.0, 2.0], [1.5, 3.0, 4.0]], dtype=np.float32)},
        dataset_features,
    )
    assert stats["observation.state"]["mean"].shape == (3,)
    np.testing.assert_allclose(stats["observation.state"]["mean"], [1.0, 2.0, 3.0])
