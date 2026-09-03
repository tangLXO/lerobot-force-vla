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

import struct
import threading
import time
from collections.abc import Callable

import draccus
import pytest
from draccus.utils import DecodingError

from lerobot.sensors import SensorConfig, make_sensors_from_configs
from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig
from lerobot.sensors.x518.protocol import _ModbusTCPClient, _X518DeviceSettings
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

_LEFT_FINGER_FORCE = "tactile.gripper.left_finger.normal_force"
_RIGHT_FINGER_FORCE = "tactile.gripper.right_finger.normal_force"


def _gripper_channels(
    *, left_channel: int = 1, right_channel: int = 2
) -> dict[str, X518ChannelConfig]:
    return {
        _LEFT_FINGER_FORCE: X518ChannelConfig(channel=left_channel),
        _RIGHT_FINGER_FORCE: X518ChannelConfig(channel=right_channel),
    }


def _x518_config(**kwargs) -> X518SensorConfig:
    return X518SensorConfig(channels=_gripper_channels(), **kwargs)


class _FakeSocket:
    def __init__(self, response: bytes):
        self.response = bytearray(response)
        self.sent = b""
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def recv(self, size: int) -> bytes:
        if not self.response:
            return b""
        chunk = bytes(self.response[:size])
        del self.response[:size]
        return chunk

    def shutdown(self, _how: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _response(
    body: bytes,
    *,
    transaction_id: int = 1,
    protocol_id: int = 0,
    unit_id: int = 1,
    length: int | None = None,
) -> bytes:
    response_length = len(body) + 1 if length is None else length
    return struct.pack(">HHHB", transaction_id, protocol_id, response_length, unit_id) + body


def _settings(
    *,
    unit: str = "kg",
    decimal: int = 0,
    sample_rate_hz: float = 200.0,
    word_swap: bool = False,
    ethernet_protocol: int = 1,
) -> _X518DeviceSettings:
    unit_codes = {"t": 1, "kg": 2, "g": 3, "kN": 4, "N": 5, "lb": 6}
    rate_codes = {6.25: 0, 12.5: 1, 25.0: 2, 50.0: 3, 100.0: 4, 200.0: 5, 400.0: 6, 800.0: 7, 1600.0: 8}
    return _X518DeviceSettings(
        unit_code=unit_codes[unit],
        unit=unit,
        decimal=decimal,
        scale=10.0 ** (-decimal),
        sample_rate_code=rate_codes[sample_rate_hz],
        sample_rate_hz=sample_rate_hz,
        channel_select=0,
        data_format=10 if word_swap else 0,
        word_swap=word_swap,
        ethernet_protocol=ethernet_protocol,
    )


class _FakeClient:
    def __init__(
        self,
        settings: list[_X518DeviceSettings] | None = None,
        raw_reader: Callable[[bool], tuple[int, int]] | None = None,
        connect_error: Exception | None = None,
    ):
        self._settings = settings or [_settings()]
        self._raw_reader = raw_reader
        self._connect_error = connect_error
        self.socket: object | None = None
        self.connect_calls = 0
        self.settings_calls = 0
        self.raw_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        if self._connect_error is not None:
            raise self._connect_error
        self.socket = object()

    def close(self) -> None:
        self.socket = None

    def read_device_settings(self) -> _X518DeviceSettings:
        self.settings_calls += 1
        index = min(self.settings_calls - 1, len(self._settings) - 1)
        return self._settings[index]

    def read_raw_channels(self, *, word_swap: bool) -> tuple[int, int]:
        self.raw_calls += 1
        if self._raw_reader is not None:
            return self._raw_reader(word_swap)
        return self.raw_calls, -self.raw_calls


def _activate_without_hardware(sensor: X518Sensor) -> None:
    sensor._stop_event.clear()
    sensor._lifecycle_active = True
    sensor._thread = threading.Thread(target=sensor._stop_event.wait, daemon=True)
    sensor._thread.start()


def test_x518_config_registration_and_explicit_mapping() -> None:
    config = _x518_config()

    assert config.type == "x518"
    assert SensorConfig.get_choice_class("x518") is X518SensorConfig
    assert config.channels == _gripper_channels()


def test_x518_config_requires_explicit_mapping() -> None:
    with pytest.raises(TypeError, match="channels"):
        X518SensorConfig()  # type: ignore[call-arg]

    with pytest.raises(DecodingError, match="Missing required field.*channels"):
        draccus.decode(SensorConfig, {"type": "x518"})


def test_x518_config_decodes_from_registered_choice_payload() -> None:
    config = draccus.decode(
        SensorConfig,
        {
            "type": "x518",
            "host": "10.0.0.8",
            "channels": {_LEFT_FINGER_FORCE: {"channel": 2}},
        },
    )

    assert isinstance(config, X518SensorConfig)
    assert config.host == "10.0.0.8"
    assert config.channels == {_LEFT_FINGER_FORCE: X518ChannelConfig(channel=2)}


def test_x518_channel_mapping_can_be_swapped() -> None:
    config = X518SensorConfig(
        channels=_gripper_channels(left_channel=2, right_channel=1)
    )
    sensor = X518Sensor(config)
    sensor._device_settings = _settings(unit="N")
    sensor._client = _FakeClient(raw_reader=lambda _word_swap: (3, 7))

    values, _timestamp_ns = sensor._read_values()

    assert values == {_LEFT_FINGER_FORCE: 7.0, _RIGHT_FINGER_FORCE: 3.0}


@pytest.mark.parametrize("channel", [0, 3, -1, True])
def test_x518_channel_rejects_invalid_device_identifier(channel) -> None:
    with pytest.raises(ValueError, match="channel must be 1 or 2"):
        X518ChannelConfig(channel=channel)


def test_x518_config_rejects_duplicate_channel_mapping() -> None:
    with pytest.raises(ValueError, match="at most one semantic feature"):
        X518SensorConfig(
            channels=_gripper_channels(left_channel=1, right_channel=1)
        )


@pytest.mark.parametrize(
    "feature_name",
    ["force", "Tactile.gripper.force", "tactile.gripper.force-N", "x518.gripper.ch1"],
)
def test_x518_config_rejects_invalid_semantic_feature_name(feature_name: str) -> None:
    with pytest.raises(ValueError, match="feature names must follow"):
        X518SensorConfig(channels={feature_name: X518ChannelConfig(channel=1)})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": "  "}, "host"),
        ({"port": 0}, "port"),
        ({"unit_id": 256}, "unit_id"),
        ({"request_timeout_s": 0}, "request_timeout_s"),
        ({"reconnect_delay_s": -1}, "reconnect_delay_s"),
        ({"connect_backoff_s": float("inf")}, "connect_backoff_s"),
        ({"connect_retries": -1}, "connect_retries"),
        ({"sample_rate_hz": float("nan")}, "sample_rate_hz"),
        ({"expected_unit": "oz"}, "expected_unit"),
        ({"channels": {}}, "channels"),
    ],
)
def test_x518_config_rejects_invalid_parameters(kwargs: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        X518SensorConfig(**({"channels": _gripper_channels()} | kwargs))


def test_modbus_read_holding_registers_validates_and_decodes_response() -> None:
    body = b"\x03\x04" + struct.pack(">HH", 0x1234, 0xABCD)
    fake_socket = _FakeSocket(_response(body))
    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client._socket = fake_socket

    registers = client.read_holding_registers(0x0A00, 2)

    assert registers == (0x1234, 0xABCD)
    assert fake_socket.sent == struct.pack(">HHHBBHH", 1, 0, 6, 1, 3, 0x0A00, 2)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(b"\x03\x02\x00\x01", transaction_id=2), "transaction ID mismatch"),
        (_response(b"\x03\x02\x00\x01", protocol_id=1), "protocol ID"),
        (_response(b"\x03\x02\x00\x01", unit_id=2), "unit ID mismatch"),
        (_response(b"", length=1), "response length"),
        (_response(b"\x83\x02"), "exception response"),
        (_response(b"\x04\x02\x00\x01"), "function code"),
        (_response(b"\x03\x04\x00\x01"), "byte count mismatch"),
    ],
)
def test_modbus_rejects_invalid_responses(response: bytes, message: str) -> None:
    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client._socket = _FakeSocket(response)

    with pytest.raises(RuntimeError, match=message):
        client.read_holding_registers(0x0A00, 1)


def test_modbus_rejects_truncated_response() -> None:
    body = b"\x03\x04\x00\x01"
    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client._socket = _FakeSocket(_response(body, length=7))

    with pytest.raises(ConnectionError, match="closed"):
        client.read_holding_registers(0x0A00, 2)


def test_modbus_decodes_signed_channels_and_word_swap() -> None:
    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client.read_holding_registers = lambda _address, _count: (0xFFFF, 0xFFFE, 0x0001, 0x0002)
    assert client.read_raw_channels(word_swap=False) == (-2, 0x00010002)

    client.read_holding_registers = lambda _address, _count: (0xFFFE, 0xFFFF, 0x0002, 0x0001)
    assert client.read_raw_channels(word_swap=True) == (-2, 0x00010002)


def test_read_device_settings_decodes_required_registers() -> None:
    measurement = [0] * 10
    measurement[1] = 2  # kg
    measurement[3] = 3  # three decimal places
    measurement[7] = 5  # 200 Hz
    measurement[9] = 3  # channel-select provenance value
    transport = [0] * 14
    transport[1] = 10  # swapped realtime words
    transport[13] = 1  # Modbus-TCP
    blocks = iter((tuple(measurement), tuple(transport)))

    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client.read_holding_registers = lambda _address, _count: next(blocks)

    settings = client.read_device_settings()

    assert settings.unit == "kg"
    assert settings.scale == pytest.approx(0.001)
    assert settings.sample_rate_hz == 200
    assert settings.channel_select == 3
    assert settings.word_swap
    assert settings.ethernet_protocol == 1


@pytest.mark.parametrize(
    ("unit_code", "decimal", "sample_rate_code", "message"),
    [
        (99, 0, 5, "unit code"),
        (2, 6, 5, "decimal setting"),
        (2, 0, 99, "sample-rate code"),
    ],
)
def test_read_device_settings_rejects_unknown_configuration(
    unit_code: int, decimal: int, sample_rate_code: int, message: str
) -> None:
    measurement = [0] * 10
    measurement[1] = unit_code
    measurement[3] = decimal
    measurement[7] = sample_rate_code
    transport = [0] * 14
    transport[13] = 1
    blocks = iter((tuple(measurement), tuple(transport)))

    client = _ModbusTCPClient("127.0.0.1", 502, 1, 0.05)
    client.read_holding_registers = lambda _address, _count: next(blocks)

    with pytest.raises(RuntimeError, match=message):
        client.read_device_settings()


@pytest.mark.parametrize(
    ("unit", "newtons"),
    [
        ("t", 9806.65),
        ("kg", 9.80665),
        ("g", 0.00980665),
        ("kN", 1000.0),
        ("N", 1.0),
        ("lb", 4.4482216152605),
    ],
)
def test_sensor_converts_supported_device_units_to_newtons(unit: str, newtons: float) -> None:
    sensor = X518Sensor(_x518_config(expected_unit=None))
    sensor._device_settings = _settings(unit=unit)
    sensor._client = _FakeClient(raw_reader=lambda _word_swap: (1, -1))

    values, _timestamp_ns = sensor._read_values()

    assert values[_LEFT_FINGER_FORCE] == pytest.approx(newtons)
    assert values[_RIGHT_FINGER_FORCE] == pytest.approx(-newtons)


def test_sensor_applies_device_decimal_before_newton_conversion() -> None:
    sensor = X518Sensor(_x518_config())
    sensor._device_settings = _settings(unit="kg", decimal=2)
    sensor._client = _FakeClient(raw_reader=lambda _word_swap: (25, -25))

    values, _timestamp_ns = sensor._read_values()

    assert values[_LEFT_FINGER_FORCE] == pytest.approx(0.25 * 9.80665)
    assert values[_RIGHT_FINGER_FORCE] == pytest.approx(-0.25 * 9.80665)


def test_sensor_timestamps_received_response_with_perf_counter_ns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def read_raw(_word_swap: bool) -> tuple[int, int]:
        events.append("response_received")
        return 1, 2

    def timestamp_response() -> int:
        assert events == ["response_received"]
        return 123_456_789

    sensor = X518Sensor(_x518_config(expected_unit=None))
    sensor._device_settings = _settings(unit="N")
    sensor._client = _FakeClient(raw_reader=read_raw)
    monkeypatch.setattr(time, "perf_counter_ns", timestamp_response)

    _values, timestamp_ns = sensor._read_values()

    assert timestamp_ns == 123_456_789


def test_sensor_read_interfaces_distinguish_fresh_consumed_and_latest_samples() -> None:
    sensor = X518Sensor(_x518_config())
    _activate_without_hardware(sensor)
    try:
        with pytest.raises(RuntimeError, match="has not produced"):
            sensor.read_latest()

        sensor._publish_sample(
            {_LEFT_FINGER_FORCE: 1.0, _RIGHT_FINGER_FORCE: 2.0}, time.perf_counter_ns()
        )
        first = sensor.async_read(timeout_ms=0)
        assert first.sequence == 0
        assert sensor.read_latest() is first

        with pytest.raises(TimeoutError, match="new X518 sample"):
            sensor.async_read(timeout_ms=0)
        with pytest.raises(TimeoutError, match="Latest X518 sample"):
            sensor.read_latest(max_age_ms=-1)

        publisher = threading.Thread(
            target=lambda: (
                time.sleep(0.01),
                sensor._publish_sample(
                    {_LEFT_FINGER_FORCE: 3.0, _RIGHT_FINGER_FORCE: 4.0}, time.perf_counter_ns()
                ),
            )
        )
        publisher.start()
        fresh = sensor.read()
        publisher.join()

        assert fresh.sequence == 1
        assert fresh.values[_LEFT_FINGER_FORCE] == 3.0
        assert fresh.is_valid
    finally:
        sensor.disconnect()


def test_disconnect_wakes_a_blocked_read() -> None:
    sensor = X518Sensor(_x518_config())
    _activate_without_hardware(sensor)
    errors: list[Exception] = []
    reader_started = threading.Event()

    def blocking_read() -> None:
        reader_started.set()
        try:
            sensor.read()
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=blocking_read)
    reader.start()
    assert reader_started.wait(timeout=0.5)
    time.sleep(0.01)

    sensor.disconnect()
    reader.join(timeout=0.5)

    assert not reader.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DeviceNotConnectedError)


def test_sensor_connection_starts_sampling_and_disconnect_stops_it() -> None:
    sensor = X518Sensor(_x518_config(sample_rate_hz=100, expected_unit="kg"))
    fake_client = _FakeClient(settings=[_settings(unit="kg", sample_rate_hz=200)])
    sensor._client = fake_client

    sensor.connect()
    sample = sensor.async_read(timeout_ms=500)

    assert sensor.is_connected
    assert sensor.sample_rate_hz == 100
    assert sample.sequence == 0
    assert sample.values[_LEFT_FINGER_FORCE] == pytest.approx(9.80665)
    assert sample.values[_RIGHT_FINGER_FORCE] == pytest.approx(-9.80665)
    with pytest.raises(DeviceAlreadyConnectedError):
        sensor.connect()

    sensor.disconnect()
    assert not sensor.is_connected
    assert fake_client.socket is None
    with pytest.raises(DeviceNotConnectedError):
        sensor.read_latest()
    with pytest.raises(DeviceNotConnectedError):
        sensor.disconnect()


def test_sensor_uses_reported_rate_when_no_poll_rate_is_configured() -> None:
    sensor = X518Sensor(_x518_config(sample_rate_hz=None, expected_unit=None))
    sensor._client = _FakeClient(settings=[_settings(unit="N", sample_rate_hz=400)])

    try:
        sensor.connect()
        sensor.async_read(timeout_ms=500)
        assert sensor.sample_rate_hz == 400
    finally:
        sensor.disconnect()


def test_sensor_reconnects_and_refreshes_device_settings() -> None:
    attempts = 0

    def fail_once(_word_swap: bool) -> tuple[int, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary failure")
        return 1, 2

    sensor = X518Sensor(_x518_config(expected_unit=None, reconnect_delay_s=0, sample_rate_hz=100))
    fake_client = _FakeClient(
        settings=[_settings(unit="kg"), _settings(unit="N")],
        raw_reader=fail_once,
    )
    sensor._client = fake_client

    try:
        sensor.connect()
        sample = sensor.async_read(timeout_ms=500)

        assert fake_client.connect_calls >= 2
        assert fake_client.settings_calls >= 2
        assert sample.values[_LEFT_FINGER_FORCE] == 1.0
        assert sample.values[_RIGHT_FINGER_FORCE] == 2.0
    finally:
        sensor.disconnect()


def test_sensor_rejects_unit_mismatch_and_excessive_poll_rate() -> None:
    wrong_unit = X518Sensor(_x518_config(expected_unit="kg", connect_retries=0))
    wrong_unit._client = _FakeClient(settings=[_settings(unit="N")])
    with pytest.raises(RuntimeError, match="expected 'kg'"):
        wrong_unit.connect()
    assert not wrong_unit.is_connected
    assert wrong_unit._client.socket is None

    excessive_rate = X518Sensor(
        _x518_config(sample_rate_hz=200, expected_unit=None, connect_retries=0)
    )
    excessive_rate._client = _FakeClient(settings=[_settings(sample_rate_hz=100)])
    with pytest.raises(ValueError, match="exceeds the device rate"):
        excessive_rate.connect()
    assert not excessive_rate.is_connected
    assert excessive_rate._client.socket is None


def test_sensor_rejects_non_modbus_ethernet_mode() -> None:
    sensor = X518Sensor(_x518_config(expected_unit=None, connect_retries=0))
    sensor._client = _FakeClient(settings=[_settings(ethernet_protocol=2)])

    with pytest.raises(RuntimeError, match="must be Modbus-TCP"):
        sensor.connect()

    assert not sensor.is_connected
    assert sensor._client.socket is None


def test_sensor_cleans_up_after_initial_connection_failure() -> None:
    sensor = X518Sensor(
        _x518_config(connect_retries=1, connect_backoff_s=0, expected_unit=None)
    )
    sensor._client = _FakeClient(connect_error=OSError("unreachable"))

    with pytest.raises(OSError, match="unreachable"):
        sensor.connect()

    assert not sensor.is_connected
    assert sensor._thread is None
    assert sensor._client.socket is None


def test_disconnect_closes_socket_created_during_reconnect() -> None:
    reconnect_started = threading.Event()

    class _DelayedReconnectClient(_FakeClient):
        def connect(self) -> None:
            self.connect_calls += 1
            if self.connect_calls > 1:
                reconnect_started.set()
                time.sleep(0.05)
            self.socket = object()

        def read_raw_channels(self, *, word_swap: bool) -> tuple[int, int]:
            raise ConnectionError("force reconnect")

    sensor = X518Sensor(_x518_config(expected_unit=None, reconnect_delay_s=0, sample_rate_hz=100))
    sensor._client = _DelayedReconnectClient()
    sensor.connect()

    assert reconnect_started.wait(timeout=0.5)
    sensor.disconnect()

    assert not sensor.is_connected
    assert sensor._client.socket is None


def test_factory_constructs_x518_without_connecting() -> None:
    config = _x518_config()

    sensors = make_sensors_from_configs({"gripper_fingers": config})

    sensor = sensors["gripper_fingers"]
    assert isinstance(sensor, X518Sensor)
    assert not sensor.is_connected
    assert sensor._client.socket is None
