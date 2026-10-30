"""Update positive Win32 serial timeouts without reconfiguring the whole port."""

import ctypes
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _serial_available

if TYPE_CHECKING or (_serial_available and sys.platform == "win32"):
    from serial import win32
    from serial.serialwin32 import Serial as Win32Serial
else:
    Win32Serial = None
    win32 = None


def _native_handle(serial_port: Any):
    """Use native APIs only for this owner's exact open PySerial Win32 port."""
    handle = getattr(serial_port, "_port_handle", None)
    if (
        Win32Serial is not None
        and win32 is not None
        and type(serial_port) is Win32Serial
        and getattr(serial_port, "is_open", False)
        and handle
        and handle != win32.INVALID_HANDLE_VALUE
    ):
        return handle
    return None


def configure_native_first_byte_read(serial_port: Any) -> None:
    """Match serialport-rs Windows reads once, preserving positive read/write limits.

    MAXDWORD/MAXDWORD/positive constant returns queued bytes immediately, or
    waits only for the first byte. Query loops retain their bounded positive
    waits; async reads request at most bytes already observed by the sole owner.
    """
    handle = _native_handle(serial_port)
    timeout = serial_port.timeout
    if handle is None:
        return
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("First-byte serial mode requires a positive bounded timeout")
    timeouts = win32.COMMTIMEOUTS()
    if not win32.GetCommTimeouts(handle, ctypes.byref(timeouts)):
        raise OSError("GetCommTimeouts failed configuring first-byte read")
    timeouts.ReadIntervalTimeout = timeouts.ReadTotalTimeoutMultiplier = 0xFFFFFFFF
    timeouts.ReadTotalTimeoutConstant = max(int(timeout * 1000), 1)
    if not win32.SetCommTimeouts(handle, ctypes.byref(timeouts)):
        raise OSError("SetCommTimeouts failed configuring first-byte read")
    serial_port._b601_first_byte_read_handle = handle


def native_first_byte_read_ready(serial_port: Any) -> bool:
    """Check the owner-managed mode token without another serial administration call."""
    handle = _native_handle(serial_port)
    timeout = serial_port.timeout
    return (
        handle is not None
        and getattr(serial_port, "_b601_first_byte_read_handle", None) == handle
        and type(timeout) in (int, float)
        and math.isfinite(timeout)
        and timeout > 0
    )


def set_serial_timeout(serial_port: object, name: str, timeout_s: float) -> None:
    """Keep pyserial's positive timeout values while updating only COMMTIMEOUTS.

    Pyserial's public setters reapply DCB and modem-line configuration too.
    For its open native Win32 transport, a positive-to-positive timeout change
    only needs the corresponding total-timeout constant. Other transports and
    transitions involving blocking/nonblocking modes retain pyserial's setter.
    """
    if name not in ("timeout", "write_timeout"):
        raise ValueError("Only serial read and write timeouts can be updated")
    current = getattr(serial_port, name)
    positive = all(
        type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (current, timeout_s)
    )
    handle = getattr(serial_port, "_port_handle", None)
    if (
        Win32Serial is None
        or win32 is None
        or type(serial_port) is not Win32Serial
        or not getattr(serial_port, "is_open", False)
        or not handle
        or handle == win32.INVALID_HANDLE_VALUE
        or not positive
    ):
        # Public setters can reapply all read fields, including on a write-only
        # timeout change. Do not reuse our mode token after such a transition.
        if hasattr(serial_port, "_b601_first_byte_read_handle"):
            delattr(serial_port, "_b601_first_byte_read_handle")
        setattr(serial_port, name, timeout_s)
        return
    timeouts = win32.COMMTIMEOUTS()
    if not win32.GetCommTimeouts(handle, ctypes.byref(timeouts)):
        raise OSError("GetCommTimeouts failed during serial timeout update")
    field = "ReadTotalTimeoutConstant" if name == "timeout" else "WriteTotalTimeoutConstant"
    setattr(timeouts, field, max(int(timeout_s * 1000), 1))
    if not win32.SetCommTimeouts(handle, ctypes.byref(timeouts)):
        raise OSError("SetCommTimeouts failed during serial timeout update")
    # Synchronize the cache only after the native API has accepted the update.
    setattr(serial_port, f"_{name}", timeout_s)


@contextmanager
def nonblocking_serial_read(serial_port: Any) -> Iterator[None]:
    """Read queued bytes immediately, preserving DCB/modem lines and write limits.

    Win32 MAXDWORD/0/0 is the documented immediate-return read mode. Stage and
    audit received bytes inside this context: restoration can independently fail.
    No deadline is extended and no bytes are discarded on entry or exit.
    """
    previous = serial_port.timeout
    if previous == 0:
        yield
        return
    handle = getattr(serial_port, "_port_handle", None)
    native = (
        Win32Serial is not None
        and win32 is not None
        and type(serial_port) is Win32Serial
        and getattr(serial_port, "is_open", False)
        and handle
        and handle != win32.INVALID_HANDLE_VALUE
    )
    read_fields = ("ReadIntervalTimeout", "ReadTotalTimeoutMultiplier", "ReadTotalTimeoutConstant")
    saved = None
    if native:
        timeouts = win32.COMMTIMEOUTS()
        if not win32.GetCommTimeouts(handle, ctypes.byref(timeouts)):
            raise OSError("GetCommTimeouts failed before nonblocking serial read")
        saved = tuple(getattr(timeouts, name) for name in read_fields)
        timeouts.ReadIntervalTimeout = 0xFFFFFFFF
        timeouts.ReadTotalTimeoutMultiplier = timeouts.ReadTotalTimeoutConstant = 0
        if not win32.SetCommTimeouts(handle, ctypes.byref(timeouts)):
            raise OSError("SetCommTimeouts failed before nonblocking serial read")
        serial_port._timeout = 0
    else:
        serial_port.timeout = 0
    read_error = None
    try:
        yield
    except BaseException as exc:
        read_error = exc
        raise
    finally:
        try:
            if native:
                assert saved is not None
                current = win32.COMMTIMEOUTS()
                if not win32.GetCommTimeouts(handle, ctypes.byref(current)):
                    raise OSError("GetCommTimeouts failed restoring serial read timeout")
                for name, value in zip(read_fields, saved, strict=True):
                    setattr(current, name, value)
                if not win32.SetCommTimeouts(handle, ctypes.byref(current)):
                    raise OSError("SetCommTimeouts failed restoring serial read timeout")
                serial_port._timeout = previous
            else:
                serial_port.timeout = previous
        except BaseException as restore_error:
            if read_error is not None:
                raise BaseExceptionGroup(
                    "Serial read and timeout restoration failed", [read_error, restore_error]
                ) from None
            raise
