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

"""Configuration for the X518 dual-channel force acquisition device.

中文说明：定义 X518 网络参数、采样参数，以及设备通道到语义特征名的映射规则。
"""

import math
import re
from dataclasses import dataclass

from ..configs import SensorConfig

_FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_HARDWARE_FEATURE_TOKEN_PATTERN = re.compile(r"(?:x518|ch_?\d+|channel_?\d+)")
_UNIT_BEARING_QUANTITIES = frozenset({"force_n", "torque_nm", "distance_mm"})
_QUALIFIED_FEATURE_PREFIXES = frozenset({"sensor", "tactile"})
_SUPPORTED_UNITS = frozenset({"t", "kg", "g", "kN", "N", "lb"})


@dataclass(frozen=True, kw_only=True)
class X518ChannelConfig:
    """Map one semantic output feature to an X518 hardware channel.

    Channel identifiers use the device's native one-based numbering.

    中文说明：描述一个语义特征对应的 X518 原生通道编号；设备通道从 1 开始编号。
    """

    channel: int

    def __post_init__(self) -> None:
        """Validate the device-native channel identifier.

        中文说明：数据类构造完成后立即检查通道号，只接受设备实际支持的 1 或 2。
        """
        if type(self.channel) is not int or self.channel not in (1, 2):
            raise ValueError(f"X518 channel must be 1 or 2, got {self.channel!r}.")


@SensorConfig.register_subclass("x518")
@dataclass(kw_only=True)
class X518SensorConfig(SensorConfig):
    """Network, acquisition, and semantic-channel configuration for an X518.

    中文说明：集中保存连接、重试、单位、采样率和语义通道映射；``channels`` 必须显式提供。
    """

    host: str = "192.168.1.100"
    port: int = 502
    unit_id: int = 1
    request_timeout_s: float = 0.05
    reconnect_delay_s: float = 0.05
    connect_retries: int = 5
    connect_backoff_s: float = 0.2
    expected_unit: str | None = "kg"
    channels: dict[str, X518ChannelConfig]

    def __post_init__(self) -> None:
        """Validate network, acquisition, and feature mapping settings.

        中文说明：在联网前一次性检查全部配置，避免后台线程启动后才暴露参数错误。
        """
        super().__post_init__()
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("X518 host must be a non-empty string.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError(f"X518 port must be in 1..65535, got {self.port!r}.")
        if type(self.unit_id) is not int or not 0 <= self.unit_id <= 255:
            raise ValueError(f"X518 unit_id must be in 0..255, got {self.unit_id!r}.")

        self._validate_positive_finite("request_timeout_s", self.request_timeout_s)
        self._validate_non_negative_finite("reconnect_delay_s", self.reconnect_delay_s)
        self._validate_non_negative_finite("connect_backoff_s", self.connect_backoff_s)
        if self.sample_rate_hz is not None:
            self._validate_positive_finite("sample_rate_hz", self.sample_rate_hz)

        if type(self.connect_retries) is not int or self.connect_retries < 0:
            raise ValueError(
                f"X518 connect_retries must be a non-negative integer, got {self.connect_retries!r}."
            )

        if self.expected_unit is not None and self.expected_unit not in _SUPPORTED_UNITS:
            raise ValueError(
                f"Unsupported X518 expected_unit {self.expected_unit!r}; "
                f"expected one of {sorted(_SUPPORTED_UNITS)} or None."
            )

        if not self.channels:
            raise ValueError("X518 channels must contain at least one semantic feature mapping.")
        if len(self.channels) > 2:
            raise ValueError("X518 exposes at most two hardware channels.")

        channel_ids: list[int] = []
        for feature_name, channel_config in self.channels.items():
            tokens = feature_name.split(".") if isinstance(feature_name, str) else []
            is_semantic_name = (
                isinstance(feature_name, str)
                and _FEATURE_NAME_PATTERN.fullmatch(feature_name) is not None
                and not any(_HARDWARE_FEATURE_TOKEN_PATTERN.fullmatch(token) for token in tokens)
                and tokens[0] not in _QUALIFIED_FEATURE_PREFIXES
                and tokens[-1] not in _UNIT_BEARING_QUANTITIES
            )
            if not is_semantic_name:
                raise ValueError(
                    "X518 feature names must follow relative semantic paths with at least two "
                    f"lowercase snake_case components; got {feature_name!r}."
                )
            if not isinstance(channel_config, X518ChannelConfig):
                raise TypeError(
                    f"X518 channel mapping for {feature_name!r} must be X518ChannelConfig, "
                    f"got {type(channel_config).__name__}."
                )
            channel_ids.append(channel_config.channel)

        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("Each X518 hardware channel may be mapped to at most one semantic feature.")

    @staticmethod
    def _validate_positive_finite(name: str, value: float) -> None:
        """Validate a finite numeric parameter that must be greater than zero.

        中文说明：校验必须大于零的有限数值参数，例如请求超时和采样率。
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"X518 {name} must be a finite number greater than zero, got {value!r}.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"X518 {name} must be a finite number greater than zero, got {value!r}.")

    @staticmethod
    def _validate_non_negative_finite(name: str, value: float) -> None:
        """Validate a finite numeric parameter that may be zero.

        中文说明：校验允许为零但不能为负数或无穷值的参数，例如重连等待时间。
        """
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"X518 {name} must be a finite non-negative number, got {value!r}.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"X518 {name} must be a finite non-negative number, got {value!r}.")
