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

"""Core contracts for non-visual physical sensors.

本模块定义“传感器硬件到标准化样本”的边界，不包含机器人控制、策略、
数据集写入或任何具体硬件协议。
"""

import abc
from dataclasses import dataclass
from typing import Any

from .configs import SensorConfig


@dataclass(frozen=True)
class SensorSample:
    """A timestamped, ordered sample produced by a sensor.

    The keys in ``values`` must match the keys exposed by the sensor's
    :attr:`Sensor.features` property.

    中文说明：一个 ``SensorSample`` 表示传感器的一次完整采样。它与
    ``RobotObservation`` 不同：前者保留传感器自己的采样时间和顺序，后者
    是机器人主循环在某个时刻组装出的整帧观测。冻结 dataclass 可以避免
    样本进入缓冲区后被意外替换时间戳、序号或有效性标志。
    """

    # 时间戳和序号属于单个传感器的采样域，供后续缓冲与时间对齐使用。
    timestamp_ns: int
    sequence: int
    # values 使用硬件无关的语义键；硬件型号和通道号只保留在配置或溯源信息中。
    values: dict[str, Any]
    is_valid: bool = True


class Sensor(abc.ABC):
    """Base class for non-visual physical sensor implementations.

    Concrete sensors own hardware communication and any background acquisition
    resources. They start those resources in :meth:`connect` and stop them in
    :meth:`disconnect`.

    中文说明：这是所有非视觉传感器必须实现的最小接口。基类只规定行为，
    不创建串口、线程或缓冲区。具体驱动负责连接硬件、采样、单位转换，
    并把结果包装成 ``SensorSample``。
    """

    def __init__(self, config: SensorConfig):
        """Initialize common sensor settings from ``config``.

        中文说明：只保存所有传感器共有的目标采样频率。串口、设备地址、
        校准参数等硬件专属配置应由具体传感器子类自行保存和处理。
        """
        self.sample_rate_hz = config.sample_rate_hz

    def __enter__(self):
        """Connect the sensor when entering a context manager.

        中文说明：支持 ``with sensor:`` 用法。进入代码块时自动调用
        ``connect()``，并返回当前传感器实例。
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Disconnect the sensor when leaving a context manager.

        中文说明：离开 ``with`` 代码块时自动断开设备，即使代码块内部发生
        异常也会执行资源清理；该方法不吞掉原始异常。
        """
        self.disconnect()

    def __del__(self) -> None:
        """Attempt to release connected hardware during garbage collection.

        中文说明：这是忘记显式断开设备时的最后一道安全网。析构时不能保证
        其他对象仍然可用，所以这里忽略清理异常；正常代码仍应主动调用
        ``disconnect()`` 或使用上下文管理器。
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def features(self) -> dict[str, type]:
        """Describe the semantic scalar values produced by this sensor.

        Keys follow ``<modality>.<location_path>.<quantity>`` and must not
        contain hardware model, transport, or unit names. Physical values use
        SI units. This property must be available while disconnected.

        中文说明：返回“语义名称到 Python 标量类型”的映射，例如
        ``{"tactile.gripper.left_finger.normal_force": float}``。这里描述的是
        传感器将产生什么数据，而不是读取数据，因此断连时也必须可用。
        """
        # feature 描述不能依赖连接状态，以便录制流程在连接硬件前构建数据结构。
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return whether the sensor is connected and ready to acquire data.

        中文说明：具体驱动应根据真实硬件资源判断连接状态，而不是仅记录
        用户是否调用过 ``connect()``。读取方法可据此抛出标准的未连接异常。
        """
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to the sensor and start any background acquisition resources.

        中文说明：负责打开硬件连接、应用必要配置，并启动具体驱动所需的
        后台采样线程。连接过程部分失败时，驱动也应清理已经创建的资源。
        """
        # 如具体驱动需要后台采样线程，由 connect() 创建并由 disconnect() 回收。
        pass

    @abc.abstractmethod
    def read(self) -> SensorSample:
        """Wait for and return a fresh sensor sample.

        中文说明：同步等待一个新样本，适合调用方希望采样节奏跟随传感器的
        场景。返回完整 ``SensorSample``，而不是只返回数值字典。
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = 200) -> SensorSample:
        """Return the newest unconsumed sample, waiting up to ``timeout_ms``.

        中文说明：名称与 Camera API 保持一致；它是普通阻塞函数，不是
        ``async def`` 协程。方法从后台采样结果中取得最新的“未消费”样本；
        在超时时间内没有新样本时应抛出 ``TimeoutError``。
        """
        # 该接口等待“新样本”，与下面非消费式读取当前缓存的 read_latest() 不同。
        pass

    @abc.abstractmethod
    def read_latest(self, max_age_ms: int = 500) -> SensorSample:
        """Return the latest sample without consuming it.

        Implementations raise :class:`TimeoutError` when the latest sample is
        older than ``max_age_ms``.

        中文说明：立即查看缓存中最新的样本，不等待新数据，也不把样本标记
        为已消费，适合传感器频率高于机器人主循环的场景。尚无样本时应抛出
        ``RuntimeError``；样本超过允许年龄时应抛出 ``TimeoutError``。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Stop acquisition and release sensor resources.

        中文说明：先通知后台采样停止并等待其退出，再关闭串口或其他硬件
        句柄，使 ``is_connected`` 恢复为 ``False``。
        """
        pass
