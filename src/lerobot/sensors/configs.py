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
from dataclasses import dataclass

import draccus  # type: ignore  # TODO: add type stubs for draccus


@dataclass(kw_only=True)
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base configuration for non-visual physical sensors.

    中文说明：这是具体传感器配置的统一父类。使用 ``kw_only=True`` 后，
    公共可选字段不会妨碍子类继续声明串口、设备地址等必填字段。
    ``ChoiceRegistry`` 使命令行和配置文件能够通过 ``type`` 选择具体配置类。
    """

    # 传感器的目标采样频率；事件驱动或无需固定频率的设备可保持为 None。
    sample_rate_hz: float | None = None

    @property
    def type(self) -> str:
        """Return the registered choice name for this sensor configuration.

        中文说明：返回子类注册时使用的名称。例如未来
        ``@SensorConfig.register_subclass("x518")`` 注册后，这里返回
        ``"x518"``，供 Draccus 解析配置以及 factory 判断设备类型。
        """
        # 沿用 CameraConfig 的 ChoiceRegistry 机制，不为 Sensor 另建注册表。
        return str(self.get_choice_name(self.__class__))
