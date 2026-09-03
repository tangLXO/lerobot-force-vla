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

"""Factory helpers for non-visual physical sensors.

本模块负责把配置字典转换为传感器实例字典，不负责连接或读取设备。
"""

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .configs import SensorConfig
from .sensor import Sensor


def make_sensors_from_configs(sensor_configs: dict[str, SensorConfig]) -> dict[str, Sensor]:
    """Instantiate sensors while preserving their configured instance names.

    中文说明：输入键表示稳定的安装位置或功能，例如 ``gripper_fingers``；
    输入值是已经由 Draccus 解析出的具体 ``SensorConfig``。函数逐项复用
    LeRobot 通用设备 factory 创建实例，并用相同的键返回，因此不会丢失
    配置顺序或安装名称。这里仅构造对象，不调用 ``connect()``。

    如果某个配置无法找到对应类或构造失败，异常会被包装为 ``ValueError``，
    同时保留传感器实例键、配置内容和原始异常，便于定位是哪台设备失败。
    """
    sensors: dict[str, Sensor] = {}

    for key, config in sensor_configs.items():
        try:
            # 复用 LeRobot 现有设备工厂约定：FooSensorConfig 对应 FooSensor。
            sensors[key] = cast(Sensor, make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"Error creating sensor {key} with config {config}: {e}") from e

    return sensors
