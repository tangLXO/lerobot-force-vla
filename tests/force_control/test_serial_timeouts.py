"""Pure Win32 timeout API mocks; no serial handle or hardware is opened."""

import ctypes
from types import SimpleNamespace

import pytest

from lerobot.robots.rebot_b601_follower import serial_timeouts


class Timeouts(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", ctypes.c_uint32),
        ("ReadTotalTimeoutMultiplier", ctypes.c_uint32),
        ("ReadTotalTimeoutConstant", ctypes.c_uint32),
        ("WriteTotalTimeoutMultiplier", ctypes.c_uint32),
        ("WriteTotalTimeoutConstant", ctypes.c_uint32),
    ]


class NativeSerial:
    """Mocked strict Win32 serial type, with observable public setter calls."""

    def __init__(self):
        self._timeout = 0.002
        self._write_timeout = 0.02
        self._port_handle = 123
        self.is_open = True
        self.public_updates = []
        self.whole_port_reconfigurations = 0

    def _reconfigure_port(self):
        self.whole_port_reconfigurations += 1

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self.public_updates.append(("timeout", value))
        self._timeout = value
        if self.is_open:
            self._reconfigure_port()

    @property
    def write_timeout(self):
        return self._write_timeout

    @write_timeout.setter
    def write_timeout(self, value):
        self.public_updates.append(("write_timeout", value))
        self._write_timeout = value
        if self.is_open:
            self._reconfigure_port()


@pytest.fixture
def native(monkeypatch):
    serial = NativeSerial()
    original = {
        "ReadIntervalTimeout": 17,
        "ReadTotalTimeoutMultiplier": 19,
        "ReadTotalTimeoutConstant": 2,
        "WriteTotalTimeoutMultiplier": 23,
        "WriteTotalTimeoutConstant": 20,
    }
    state = SimpleNamespace(
        serial=serial,
        values=original.copy(),
        original=original,
        calls=[],
        get_ok=True,
        set_ok=True,
        cache_at_set=[],
    )

    def get_timeouts(handle, pointer):
        assert handle == serial._port_handle
        state.calls.append("GetCommTimeouts")
        value = ctypes.cast(pointer, ctypes.POINTER(Timeouts)).contents
        for field, data in state.values.items():
            setattr(value, field, data)
        return state.get_ok

    def set_timeouts(handle, pointer):
        assert handle == serial._port_handle
        state.calls.append("SetCommTimeouts")
        state.cache_at_set.append((serial.timeout, serial.write_timeout))
        if state.set_ok:
            value = ctypes.cast(pointer, ctypes.POINTER(Timeouts)).contents
            state.values = {field: getattr(value, field) for field, _ in Timeouts._fields_}
        return state.set_ok

    monkeypatch.setattr(serial_timeouts, "Win32Serial", NativeSerial)
    monkeypatch.setattr(
        serial_timeouts,
        "win32",
        SimpleNamespace(
            COMMTIMEOUTS=Timeouts,
            GetCommTimeouts=get_timeouts,
            SetCommTimeouts=set_timeouts,
            INVALID_HANDLE_VALUE=-1,
        ),
    )
    return state


@pytest.mark.parametrize(
    "name,value,field,encoded",
    [
        ("timeout", 0.001, "ReadTotalTimeoutConstant", 1),
        ("write_timeout", 0.013, "WriteTotalTimeoutConstant", 13),
        ("write_timeout", 0.0037, "WriteTotalTimeoutConstant", 3),
        ("timeout", 0.0004, "ReadTotalTimeoutConstant", 1),
    ],
)
def test_native_update_changes_only_the_selected_constant_after_api_success(
    native, name, value, field, encoded
):
    serial_timeouts.set_serial_timeout(native.serial, name, value)
    expected = {**native.original, field: encoded}
    assert native.values == expected
    assert native.calls == ["GetCommTimeouts", "SetCommTimeouts"]
    assert native.cache_at_set == [(0.002, 0.02)]
    assert native.serial.public_updates == []
    assert native.serial.whole_port_reconfigurations == 0
    assert getattr(native.serial, name) == value
    assert getattr(native.serial, "write_timeout" if name == "timeout" else "timeout") == (
        0.02 if name == "timeout" else 0.002
    )


@pytest.mark.parametrize("operation", ["get", "set"])
@pytest.mark.parametrize("name", ["timeout", "write_timeout"])
def test_native_api_failure_raises_without_cache_mutation_or_setter_fallback(native, operation, name):
    setattr(native, f"{operation}_ok", False)
    with pytest.raises(OSError, match="GetCommTimeouts" if operation == "get" else "SetCommTimeouts"):
        serial_timeouts.set_serial_timeout(native.serial, name, 0.001)
    assert native.serial.timeout == 0.002 and native.serial.write_timeout == 0.02
    assert native.serial.public_updates == []
    assert native.serial.whole_port_reconfigurations == 0
    assert native.values == native.original
    assert native.calls == (
        ["GetCommTimeouts"] if operation == "get" else ["GetCommTimeouts", "SetCommTimeouts"]
    )


@pytest.mark.parametrize(
    "attribute,value", [("is_open", False), ("_port_handle", None), ("_port_handle", 0), ("_port_handle", -1)]
)
def test_closed_or_missing_native_handle_retains_the_public_setter(native, attribute, value):
    setattr(native.serial, attribute, value)
    serial_timeouts.set_serial_timeout(native.serial, "timeout", 0.001)
    assert native.serial.public_updates == [("timeout", 0.001)]
    assert native.calls == []


@pytest.mark.parametrize("current,target", [(None, 0.001), (0, 0.001), (0.002, None), (0.002, 0)])
def test_blocking_and_nonblocking_transitions_retain_pyserial_semantics(native, current, target):
    native.serial._timeout = current
    serial_timeouts.set_serial_timeout(native.serial, "timeout", target)
    assert native.serial.public_updates == [("timeout", target)]
    assert native.calls == []


def test_non_native_or_subclass_objects_cannot_enter_the_native_path(native):
    class WrapperSerial(NativeSerial):
        pass

    wrapped = WrapperSerial()
    serial_timeouts.set_serial_timeout(wrapped, "write_timeout", 0.01)
    assert wrapped.public_updates == [("write_timeout", 0.01)]
    assert native.calls == []


def test_non_windows_or_missing_optional_dependency_uses_the_ordinary_setter(native, monkeypatch):
    monkeypatch.setattr(serial_timeouts, "Win32Serial", None)
    serial_timeouts.set_serial_timeout(native.serial, "write_timeout", 0.01)
    assert native.serial.public_updates == [("write_timeout", 0.01)]
    assert native.calls == []


def test_helper_rejects_non_timeout_fields_without_mutation(native):
    with pytest.raises(ValueError, match="Only serial"):
        serial_timeouts.set_serial_timeout(native.serial, "baudrate", 123)
    assert native.serial.public_updates == [] and native.calls == []
