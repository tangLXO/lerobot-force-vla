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


def test_nonblocking_read_changes_only_read_timeouts_without_modem_reconfiguration(native):
    with serial_timeouts.nonblocking_serial_read(native.serial):
        assert native.serial.timeout == 0
        assert native.values == {
            **native.original,
            "ReadIntervalTimeout": 0xFFFFFFFF,
            "ReadTotalTimeoutMultiplier": 0,
            "ReadTotalTimeoutConstant": 0,
        }
        native.values["WriteTotalTimeoutConstant"] = 7
    assert native.serial.timeout == 0.002
    assert native.values == {**native.original, "WriteTotalTimeoutConstant": 7}
    assert native.serial.public_updates == []
    assert native.serial.whole_port_reconfigurations == 0


def test_nonblocking_read_restores_original_mode_after_read_error(native):
    with (
        pytest.raises(OSError, match="injected read failure"),
        serial_timeouts.nonblocking_serial_read(native.serial),
    ):
        raise OSError("injected read failure")
    assert native.values == native.original and native.serial.timeout == 0.002


@pytest.mark.parametrize("operation", ["get", "set"])
def test_nonblocking_read_setup_failure_never_enters_body_or_changes_cache(native, operation):
    setattr(native, f"{operation}_ok", False)
    with (
        pytest.raises(OSError, match="before nonblocking"),
        serial_timeouts.nonblocking_serial_read(native.serial),
    ):
        pytest.fail("read must not start after timeout setup failure")
    assert native.serial.timeout == 0.002
    assert native.serial.public_updates == []


def test_nonblocking_restore_failure_does_not_claim_original_mode_was_restored(native):
    with (
        pytest.raises(OSError, match="restoring serial read"),
        serial_timeouts.nonblocking_serial_read(native.serial),
    ):
        native.set_ok = False
    assert native.serial.timeout == 0
    assert native.values["ReadIntervalTimeout"] == 0xFFFFFFFF


def test_nonblocking_read_and_restore_errors_are_both_preserved(native):
    with pytest.raises(BaseExceptionGroup) as failure, serial_timeouts.nonblocking_serial_read(native.serial):
        native.set_ok = False
        raise OSError("read failed too")
    assert len(failure.value.exceptions) == 2
    assert "read failed too" in str(failure.value.exceptions[0])
    assert "restoring serial read" in str(failure.value.exceptions[1])


def test_non_native_nonblocking_read_restores_public_timeout():
    port = SimpleNamespace(timeout=0.002, write_timeout=0.02)
    with serial_timeouts.nonblocking_serial_read(port):
        assert port.timeout == 0 and port.write_timeout == 0.02
    assert port.timeout == 0.002


def test_first_byte_mode_matches_official_windows_read_fields_and_preserves_write_limits(native):
    serial_timeouts.configure_native_first_byte_read(native.serial)
    assert native.values == {
        **native.original,
        "ReadIntervalTimeout": 0xFFFFFFFF,
        "ReadTotalTimeoutMultiplier": 0xFFFFFFFF,
        "ReadTotalTimeoutConstant": 2,
    }
    assert serial_timeouts.native_first_byte_read_ready(native.serial)
    assert native.calls == ["GetCommTimeouts", "SetCommTimeouts"]
    assert native.serial.timeout == 0.002 and native.serial.write_timeout == 0.02
    assert native.serial.public_updates == [] and native.serial.whole_port_reconfigurations == 0
    # Checking the owner-managed token itself adds no native administration call.
    assert serial_timeouts.native_first_byte_read_ready(native.serial)
    assert len(native.calls) == 2


@pytest.mark.parametrize("operation", ["get", "set"])
def test_first_byte_setup_failure_does_not_claim_mode_is_ready(native, operation):
    setattr(native, f"{operation}_ok", False)
    with pytest.raises(OSError, match="configuring first-byte"):
        serial_timeouts.configure_native_first_byte_read(native.serial)
    assert not serial_timeouts.native_first_byte_read_ready(native.serial)
    assert native.serial.public_updates == [] and native.serial.timeout == 0.002


def test_first_byte_mode_survives_bounded_positive_read_and_write_updates(native):
    serial_timeouts.configure_native_first_byte_read(native.serial)
    serial_timeouts.set_serial_timeout(native.serial, "write_timeout", 0.003)
    serial_timeouts.set_serial_timeout(native.serial, "timeout", 0.001)
    assert native.values["ReadIntervalTimeout"] == native.values["ReadTotalTimeoutMultiplier"] == 0xFFFFFFFF
    assert native.values["ReadTotalTimeoutConstant"] == 1
    assert native.values["WriteTotalTimeoutConstant"] == 3
    assert serial_timeouts.native_first_byte_read_ready(native.serial)


@pytest.mark.parametrize("name", ["timeout", "write_timeout"])
def test_whole_port_timeout_transition_invalidates_first_byte_token(native, name):
    serial_timeouts.configure_native_first_byte_read(native.serial)
    serial_timeouts.set_serial_timeout(native.serial, name, 0)
    assert not serial_timeouts.native_first_byte_read_ready(native.serial)


def test_first_byte_token_cannot_be_reused_for_another_handle_or_closed_port(native):
    serial_timeouts.configure_native_first_byte_read(native.serial)
    native.serial._port_handle = 124
    assert not serial_timeouts.native_first_byte_read_ready(native.serial)
    native.serial._port_handle = 123
    native.serial.is_open = False
    assert not serial_timeouts.native_first_byte_read_ready(native.serial)


def test_first_byte_mode_does_not_change_non_native_serial_semantics():
    port = SimpleNamespace(timeout=0.002)
    serial_timeouts.configure_native_first_byte_read(port)
    assert not serial_timeouts.native_first_byte_read_ready(port)
    assert vars(port) == {"timeout": 0.002}


@pytest.mark.parametrize("timeout", [None, 0, -1, True, float("inf"), float("nan")])
def test_first_byte_mode_requires_a_positive_bounded_native_read_timeout(native, timeout):
    native.serial._timeout = timeout
    with pytest.raises(ValueError, match="positive bounded"):
        serial_timeouts.configure_native_first_byte_read(native.serial)
    assert native.calls == []
