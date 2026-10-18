"""Update positive Win32 serial timeouts without reconfiguring the whole port."""

import ctypes
import math
import sys
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _serial_available

if TYPE_CHECKING or (_serial_available and sys.platform == "win32"):
    from serial import win32
    from serial.serialwin32 import Serial as Win32Serial
else:
    Win32Serial = None
    win32 = None


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
