#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration primitives for non-visual physical sensors.

本模块只定义所有非视觉传感器共享的配置入口。具体硬件应继承
``SensorConfig`` 并通过 Draccus 注册自己的配置类型。
"""

import abc
import math
from dataclasses import dataclass

import draccus  # type: ignore  # TODO: add type stubs for draccus


@dataclass(kw_only=True)
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base configuration for non-visual physical sensors.

    中文说明：这是具体传感器配置的统一父类。使用 ``kw_only=True`` 后，
    公共可选字段不会妨碍子类继续声明串口、设备地址等必填字段。
    ``ChoiceRegistry`` 使命令行和配置文件能够通过 ``type`` 选择具体配置类。
    """

    sample_rate_hz: float | None = None
    expected_sample_rate_hz: float | None = None
    max_age_ms: float | None = None
    history_duration_s: float = 2.0
    startup_timeout_s: float = 5.0
    required: bool = True
    state_features: list[str] | None = None
    recorder_queue_duration_s: float = 2.0
    recorder_queue_capacity: int | None = None
    record_native_values: bool = True
    record_native_payload: bool = False

    def __post_init__(self) -> None:
        """Validate static timing, queue, and state-selection settings."""
        for name in ("sample_rate_hz", "expected_sample_rate_hz"):
            value = getattr(self, name)
            if value is not None:
                self._positive_number(name, value)
        if self.max_age_ms is not None:
            self._positive_number("max_age_ms", self.max_age_ms)
        self._positive_number("history_duration_s", self.history_duration_s)
        self._positive_number("startup_timeout_s", self.startup_timeout_s)
        self._positive_number("recorder_queue_duration_s", self.recorder_queue_duration_s)
        if self.recorder_queue_capacity is not None and (
            type(self.recorder_queue_capacity) is not int or self.recorder_queue_capacity <= 0
        ):
            raise ValueError("recorder_queue_capacity must be a positive integer or None.")
        if self.state_features is not None:
            if any(not isinstance(name, str) or not name for name in self.state_features):
                raise ValueError("state_features entries must be non-empty relative feature paths.")
            if len(self.state_features) != len(set(self.state_features)):
                raise ValueError("state_features must not contain duplicates.")
            if self.state_features and not self.required:
                raise ValueError("A sensor contributing state_features must be required.")
        elif not self.required:
            raise ValueError("state_features=None selects all features, so the sensor must be required.")

    def static_sample_rate_hz(self) -> float | None:
        """Return the pre-connection rate usable for static resolution."""
        return self.expected_sample_rate_hz or self.sample_rate_hz

    def resolve_max_age_ms(self, *, state_features_present: bool) -> float | None:
        """Resolve the immutable current/window staleness threshold."""
        if self.max_age_ms is not None:
            return self.max_age_ms
        rate_hz = self.static_sample_rate_hz()
        if rate_hz is not None:
            return float(math.ceil(3000.0 / rate_hz))
        if state_features_present:
            raise ValueError(
                "A sensor contributing observation.state requires max_age_ms or a static sample rate."
            )
        return None

    def resolve_recorder_queue_capacity(self) -> int:
        """Resolve a bounded recorder queue before hardware connection."""
        if self.recorder_queue_capacity is not None:
            return self.recorder_queue_capacity
        rate_hz = self.static_sample_rate_hz()
        if rate_hz is None:
            raise ValueError(
                "Sensor recording requires recorder_queue_capacity or a static expected/sample rate."
            )
        return max(1, math.ceil(self.recorder_queue_duration_s * rate_hz))

    @staticmethod
    def _positive_number(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite number greater than zero.")

    @property
    def type(self) -> str:
        """Return the registered choice name for this sensor configuration.

        中文说明：返回子类注册时使用的名称。例如未来
        ``@SensorConfig.register_subclass("x518")`` 注册后，这里返回
        ``"x518"``，供 Draccus 解析配置以及 factory 判断设备类型。
        """
        # 沿用 CameraConfig 的 ChoiceRegistry 机制，不为 Sensor 另建注册表。
        return str(self.get_choice_name(self.__class__))
