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

import time
from dataclasses import FrozenInstanceError, dataclass

import pytest

import lerobot.sensors.utils as sensor_utils
from lerobot.sensors import Sensor, SensorConfig, SensorFeature, SensorSample, make_sensors_from_configs
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

FEATURE_NAME = "left_finger.normal_force"


@SensorConfig.register_subclass("dummy")
@dataclass
class DummySensorConfig(SensorConfig):
    """Test-only sensor configuration."""

    value: float = 1.5


class DummySensor(Sensor):
    """Test-only sensor implementation.

    中文说明：这个类不接触真实硬件，只用内存状态模拟连接、采样和缓存，
    用来验证 Sensor Core 的公共契约。
    """

    def __init__(self, config: DummySensorConfig):
        """保存测试配置并初始化断连状态、序号和最新样本缓存。"""
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._latest_sample: SensorSample | None = None

    @property
    def features(self) -> dict[str, SensorFeature]:
        """声明 Dummy 会输出一个以牛顿为 SI 单位的左指法向力标量。"""
        return {FEATURE_NAME: SensorFeature("float32", "N")}

    @property
    def is_connected(self) -> bool:
        """返回内存中的模拟连接状态。"""
        return self._is_connected

    @check_if_already_connected
    def connect(self) -> None:
        """把 Dummy 切换为已连接；装饰器负责拦截重复连接。"""
        self._is_connected = True

    def _new_sample(self) -> SensorSample:
        """生成新样本、递增序号，并把该样本保存为当前最新值。"""
        timestamp_ns = time.perf_counter_ns()
        sample = self._publish_sample(
            {FEATURE_NAME: self.config.value},
            timestamp_ns,
            arrival_timestamp_ns=timestamp_ns,
        )
        self._latest_sample = sample
        return sample

    @check_if_not_connected
    def read(self) -> SensorSample:
        """模拟同步读取一个新样本。"""
        return self._new_sample()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> SensorSample:
        """模拟后台读取接口；Dummy 无需真正等待，因此立即产生新样本。"""
        return self._new_sample()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> SensorSample:
        """返回缓存样本，并模拟“尚无样本”和“样本过期”两类错误。"""
        if self._latest_sample is None:
            raise RuntimeError("No sensor sample is available yet.")

        age_ms = (time.perf_counter_ns() - self._latest_sample.timestamp_ns) / 1e6
        if age_ms > max_age_ms:
            raise TimeoutError(f"Latest sensor sample is {age_ms:.1f}ms old.")

        return self._latest_sample

    @check_if_not_connected
    def disconnect(self) -> None:
        """把 Dummy 切换为断连状态；装饰器负责拦截重复断开。"""
        self._is_connected = False


def test_sensor_is_abstract() -> None:
    """验证未实现抽象方法时不能直接创建 Sensor。"""
    with pytest.raises(TypeError, match="abstract"):
        Sensor(DummySensorConfig())


def test_sensor_config_registration() -> None:
    """验证配置注册名称、采样率字段和 ChoiceRegistry 查询结果。"""
    config = DummySensorConfig(sample_rate_hz=200, value=2.0)

    assert config.type == "dummy"
    assert config.sample_rate_hz == 200
    assert SensorConfig.get_choice_class("dummy") is DummySensorConfig


def test_sensor_sample_value_object() -> None:
    """验证样本字段能够保存数据，并且冻结字段不能被重新赋值。"""
    sample = SensorSample(
        timestamp_ns=123,
        sequence=4,
        values={FEATURE_NAME: 2.5},
        is_valid=False,
        status="read_error",
        error="simulated failure",
    )

    assert sample.timestamp_ns == 123
    assert sample.sequence == 4
    assert sample.values == {FEATURE_NAME: 2.5}
    assert not sample.is_valid
    with pytest.raises(FrozenInstanceError):
        sample.sequence = 5  # type: ignore[misc]

    with pytest.raises(ValueError, match="invalid SensorSample"):
        SensorSample(timestamp_ns=123, sequence=5, is_valid=False)


def test_invalid_publication_is_auditable_and_consumes_framework_sequence() -> None:
    sensor = DummySensor(DummySensorConfig())
    sensor.connect()
    invalid = sensor._publish_sample(
        None,
        100,
        arrival_timestamp_ns=101,
        is_valid=False,
        status="read_error",
        error="simulated read failure",
    )
    valid = sensor._publish_sample(
        {FEATURE_NAME: 1.0},
        102,
        arrival_timestamp_ns=103,
    )

    assert invalid.sequence == 0
    assert invalid.values == {}
    assert invalid.error == "simulated read failure"
    assert valid.sequence == 1


def test_feature_and_read_contract() -> None:
    """验证 feature、连接状态以及三种读取接口之间的基本契约。"""
    sensor = DummySensor(DummySensorConfig(sample_rate_hz=200))

    assert sensor.features == {FEATURE_NAME: SensorFeature("float32", "N")}
    assert not sensor.is_connected
    with pytest.raises(DeviceNotConnectedError):
        sensor.read()

    sensor.connect()
    with pytest.raises(DeviceAlreadyConnectedError):
        sensor.connect()

    first = sensor.read()
    second = sensor.async_read()
    assert first.sequence == 0
    assert second.sequence == 1
    assert set(first.values) == set(sensor.features)
    assert sensor.read_latest() is second

    sensor.disconnect()
    assert not sensor.is_connected
    with pytest.raises(DeviceNotConnectedError):
        sensor.disconnect()


def test_read_latest_rejects_missing_and_stale_samples() -> None:
    """验证 read_latest 对空缓存和过期缓存给出明确异常。"""
    sensor = DummySensor(DummySensorConfig())
    sensor.connect()

    with pytest.raises(RuntimeError, match="No sensor sample"):
        sensor.read_latest()

    sensor.read()
    with pytest.raises(TimeoutError, match="Latest sensor sample"):
        sensor.read_latest(max_age_ms=-1)

    sensor.disconnect()


def test_context_manager_connects_and_disconnects() -> None:
    """验证 with 代码块进入时连接、退出时断开。"""
    sensor = DummySensor(DummySensorConfig())

    with sensor as connected_sensor:
        assert connected_sensor is sensor
        assert sensor.is_connected

    assert not sensor.is_connected


def test_factory_creates_named_sensors_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 factory 保留实例名称、输入顺序和对应配置对象。"""
    configs = {
        "left_finger": DummySensorConfig(value=1.0),
        "right_finger": DummySensorConfig(value=2.0),
    }

    monkeypatch.setattr(sensor_utils, "make_device_from_device_class", DummySensor)

    sensors = make_sensors_from_configs(configs)

    assert list(sensors) == ["left_finger", "right_finger"]
    assert all(isinstance(sensor, DummySensor) for sensor in sensors.values())
    assert sensors["left_finger"].config is configs["left_finger"]
    assert sensors["right_finger"].config is configs["right_finger"]


def test_factory_adds_instance_context_to_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证构造失败时的错误消息包含出错的传感器实例名称。"""

    def fail_to_create(_config: SensorConfig) -> Sensor:
        """模拟底层设备构造失败。"""
        raise RuntimeError("construction failed")

    monkeypatch.setattr(sensor_utils, "make_device_from_device_class", fail_to_create)

    with pytest.raises(ValueError, match="Error creating sensor gripper_fingers.*construction failed"):
        make_sensors_from_configs({"gripper_fingers": DummySensorConfig()})
