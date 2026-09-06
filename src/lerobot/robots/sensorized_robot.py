#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Robot decorator that causally merges selected sensor values into state."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.sensors import (
    Sensor,
    SensorConfig,
    SensorDataUnavailableError,
    SensorFeature,
    make_sensors_from_configs,
)
from lerobot.utils.errors import DeviceNotConnectedError

from .robot import Robot

logger = logging.getLogger(__name__)
_INSTANCE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_FEATURE_PATH_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


class SensorizedRobot:
    """Delegate robot I/O while adding selected scalar sensor state features."""

    def __init__(self, robot: Robot, sensors: dict[str, Sensor]) -> None:
        """Validate the stable state schema without touching hardware."""
        self._robot = robot
        self.sensors = sensors
        self._selected_features: dict[str, tuple[str, ...]] = {}
        self._resolved_max_age_ms: dict[str, float | None] = {}
        self._fatal_error: Exception | None = None
        self.last_capture_metadata: dict[str, Any] | None = None

        qualified = set(robot.observation_features)
        for instance, sensor in sensors.items():
            if _INSTANCE_PATTERN.fullmatch(instance) is None:
                raise ValueError(
                    f"Sensor instance {instance!r} must be a lowercase snake_case namespace token."
                )
            semantic_features = sensor.features
            invalid_paths = [
                path
                for path in semantic_features
                if _FEATURE_PATH_PATTERN.fullmatch(path) is None or path.startswith(("sensor.", "tactile."))
            ]
            if invalid_paths:
                raise ValueError(
                    f"Sensor {instance!r} exposes invalid relative feature paths: {invalid_paths}."
                )
            invalid_schemas = [
                path for path, feature in semantic_features.items() if not isinstance(feature, SensorFeature)
            ]
            if invalid_schemas:
                raise TypeError(
                    f"Sensor {instance!r} features must use SensorFeature schemas: {invalid_schemas}."
                )
            configured = sensor.config.state_features
            selected = tuple(semantic_features) if configured is None else tuple(configured)
            unknown = set(selected) - set(semantic_features)
            if unknown:
                raise ValueError(
                    f"Sensor {instance!r} state_features contains unknown paths: {sorted(unknown)}."
                )
            if selected and not sensor.config.required:
                raise ValueError(f"Sensor {instance!r} contributes state and must be required.")
            self._resolved_max_age_ms[instance] = sensor.config.resolve_max_age_ms(
                state_features_present=bool(selected)
            )
            for feature_path in selected:
                name = self.qualify_feature(instance, feature_path)
                if name in qualified:
                    raise ValueError(f"Sensor state feature {name!r} conflicts with an existing feature.")
                qualified.add(name)
            self._selected_features[instance] = selected

    @staticmethod
    def qualify_feature(instance: str, feature_path: str) -> str:
        """Build the locked Dataset/state namespace."""
        return f"sensor.{instance}.{feature_path}"

    @property
    def observation_features(self) -> dict:
        """Return base features followed by selected sensor scalar features."""
        features = dict(self._robot.observation_features)
        for instance in self.sensors:
            for feature_path in self._selected_features[instance]:
                features[self.qualify_feature(instance, feature_path)] = float
        return features

    @property
    def action_features(self) -> dict:
        """Delegate the action schema unchanged."""
        return self._robot.action_features

    @property
    def name(self) -> str:
        """Delegate the robot name unchanged."""
        return self._robot.name

    @property
    def robot_type(self) -> str:
        """Delegate the robot type unchanged."""
        return self._robot.robot_type

    @property
    def id(self) -> str:
        """Delegate the robot installation id unchanged."""
        return self._robot.id

    @property
    def cameras(self):
        """Delegate the camera mapping unchanged."""
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        """Require the inner robot and every required sensor to be connected."""
        return self._robot.is_connected and all(
            sensor.is_connected for sensor in self.sensors.values() if sensor.config.required
        )

    @property
    def is_calibrated(self) -> bool:
        """Delegate robot calibration state."""
        return self._robot.is_calibrated

    @property
    def inner(self) -> Robot:
        """Expose the unwrapped robot for existing teardown helpers."""
        return self._robot

    def connect(self, calibrate: bool = True) -> None:
        """Connect the robot and sensors, then await required sensor readiness."""
        self._fatal_error = None
        self.last_capture_metadata = None
        self._robot.connect(calibrate=calibrate)
        connected: list[Sensor] = []
        try:
            for instance, sensor in self.sensors.items():
                try:
                    sensor.connect()
                    connected.append(sensor)
                    if sensor.config.required:
                        self._wait_until_ready(instance, sensor)
                except Exception:
                    if sensor.config.required:
                        raise
                    logger.warning("Optional sensor %s failed to connect", instance, exc_info=True)
        except Exception:
            for sensor in reversed(connected):
                if sensor.is_connected:
                    sensor.disconnect()
            if self._robot.is_connected:
                self._robot.disconnect()
            raise

    def _wait_until_ready(self, instance: str, sensor: Sensor) -> None:
        deadline = time.perf_counter() + sensor.config.startup_timeout_s
        max_age_ms = self._resolved_max_age_ms[instance]
        while True:
            target_ns = time.perf_counter_ns()
            try:
                sensor.read_latest_before(target_ns, max_age_ms)
                return
            except (RuntimeError, SensorDataUnavailableError, DeviceNotConnectedError):
                if time.perf_counter() >= deadline:
                    raise SensorDataUnavailableError(
                        f"Required sensor {instance!r} did not become ready within "
                        f"{sensor.config.startup_timeout_s:g}s."
                    ) from None
                time.sleep(min(0.01, max(0.0, deadline - time.perf_counter())))

    def get_observation(self) -> RobotObservation:
        """Apply the locked post-robot observation anchor and causal selection."""
        self.check_health()
        observation_start_ns = time.perf_counter_ns()
        observation = self._robot.get_observation()
        frame_anchor_ns = time.perf_counter_ns()
        selections: dict[str, dict[str, Any]] = {}

        try:
            for instance, sensor in self.sensors.items():
                selected_features = self._selected_features[instance]
                try:
                    sample = sensor.read_latest_before(frame_anchor_ns, self._resolved_max_age_ms[instance])
                except (RuntimeError, SensorDataUnavailableError, DeviceNotConnectedError):
                    if sensor.config.required:
                        raise
                    selections[instance] = {
                        "timestamp_ns": None,
                        "arrival_timestamp_ns": None,
                        "sequence": None,
                        "hardware_sequence": None,
                        "age_ns": None,
                        "status": "unavailable",
                    }
                    continue
                for feature_path in selected_features:
                    if feature_path not in sample.values:
                        raise SensorDataUnavailableError(
                            f"Required sensor {instance!r} sample is missing {feature_path!r}."
                        )
                    observation[self.qualify_feature(instance, feature_path)] = float(
                        np.float32(sample.values[feature_path])
                    )
                selections[instance] = {
                    "timestamp_ns": sample.timestamp_ns,
                    "arrival_timestamp_ns": sample.arrival_timestamp_ns,
                    "sequence": sample.sequence,
                    "hardware_sequence": sample.hardware_sequence,
                    "age_ns": frame_anchor_ns - sample.timestamp_ns,
                    "status": sample.status,
                }
        except Exception as exc:
            self._fatal_error = exc
            raise

        observation_complete_ns = time.perf_counter_ns()
        self.last_capture_metadata = {
            "observation_start_ns": observation_start_ns,
            "frame_anchor_ns": frame_anchor_ns,
            "observation_complete_ns": observation_complete_ns,
            "hardware_observation_timestamps": None,
            "sensors": selections,
        }
        return observation

    def check_health(self) -> None:
        """Fail before observation/action after any latched required-sensor fault."""
        if self._fatal_error is not None:
            raise SensorDataUnavailableError(
                "SensorizedRobot is in a fatal sensor state."
            ) from self._fatal_error
        for instance, sensor in self.sensors.items():
            try:
                sensor._check_recorder_fault()
            except RuntimeError as exc:
                self._fatal_error = exc
                raise SensorDataUnavailableError(
                    f"Sensor {instance!r} has a recorder episode fault."
                ) from exc
            if sensor.config.required and not sensor.is_connected:
                error = SensorDataUnavailableError(f"Required sensor {instance!r} is disconnected.")
                self._fatal_error = error
                raise error
            if sensor.has_subscriber_overflow:
                error = SensorDataUnavailableError(f"Sensor recorder subscriber overflowed for {instance!r}.")
                self._fatal_error = error
                raise error

    def latch_fatal_error(self, exc: Exception) -> None:
        """Prevent every subsequent action after a recorder or sensor failure."""
        self._fatal_error = exc

    @property
    def has_fatal_error(self) -> bool:
        """Whether a sensor failure has been latched."""
        return self._fatal_error is not None

    def send_action(self, action: RobotAction) -> RobotAction:
        """Check sensor health immediately before delegating a motion command."""
        self.check_health()
        return self._robot.send_action(action)

    def calibrate(self) -> None:
        """Delegate calibration unchanged."""
        self._robot.calibrate()

    def configure(self) -> None:
        """Delegate one-time robot configuration unchanged."""
        self._robot.configure()

    def disconnect(self) -> None:
        """Disconnect sensors before releasing the inner robot."""
        first_error: Exception | None = None
        for sensor in reversed(tuple(self.sensors.values())):
            if sensor.is_connected:
                try:
                    sensor.disconnect()
                except Exception as exc:  # pragma: no cover - hardware cleanup path
                    first_error = first_error or exc
        if self._robot.is_connected:
            try:
                self._robot.disconnect()
            except Exception as exc:  # pragma: no cover - hardware cleanup path
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def __getattr__(self, name: str) -> Any:
        """Delegate robot-specific compatibility helpers."""
        return getattr(self._robot, name)


def attach_sensors(robot: Robot, sensor_configs: dict[str, SensorConfig]) -> Robot | SensorizedRobot:
    """Return the original robot unchanged or a validated sensor decorator."""
    if not sensor_configs:
        return robot
    return SensorizedRobot(robot, make_sensors_from_configs(sensor_configs))
