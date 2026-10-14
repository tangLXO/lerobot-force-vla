"""Query-only FashionStar code-25 transport for fresh 102 packet evidence.

Wire layout follows motorbridge-smart-servo 0.0.4's
``smart_servo_vendors/fashionstar/src/protocol.rs``. The only packet encoder
in this module emits query 25/subcommand 22. There are no actuator commands.

This separates packet freshness from the SDK's near-zero filter, which holds
29 of every 30 readings within 0.2 degrees. Packet validity does not assert
physical angle accuracy, calibration, or immunity to power-cycle transients.
The wire protocol has no request token or device measurement timestamp.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None


@dataclass(frozen=True)
class ReceivedMonitor:
    """Checksum-validated packet received during the current host transaction."""

    motor_id: int
    angle_deg: float
    receive_timestamp_ns: int
    receive_sequence: int
    motor_status: int


def encode_monitor_query(joint_ids: tuple[int, ...]) -> bytes:
    """Encode only code 25 / subcommand 22, using the vendor additive checksum."""
    if not joint_ids or len(joint_ids) > 252 or len(set(joint_ids)) != len(joint_ids):
        raise ValueError("Monitor query requires 1..252 unique servo IDs")
    if any(type(motor_id) is not int or not 0 <= motor_id <= 253 for motor_id in joint_ids):
        raise ValueError("Servo IDs must be integers in 0..253")
    params = bytes((22, 1, len(joint_ids), *joint_ids))
    prefix = bytes((0x12, 0x4C, 25, len(params))) + params
    return prefix + bytes((sum(prefix) & 0xFF,))


class Readonly102Transport:
    """One serial owner with bounded reads and a latched failure after bad I/O.

    Missing/invalid/duplicate packets latch the transport until explicitly
    closed and reopened. No automatic retry can attribute a delayed packet
    from a failed query to a subsequent query. A successful query still has
    host receive evidence, not an echoed transaction ID or device sample time.
    """

    def __init__(self, port: str, baudrate: int, joint_ids: tuple[int, ...]):
        """Validate IDs before opening a device; do not transmit anything."""
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        self.port = port
        self.baudrate = baudrate
        self.joint_ids = joint_ids
        self.query_packet = encode_monitor_query(joint_ids)
        self._serial = None
        self._owner_ident = None
        self._failure: str | None = None
        self._receive_sequences = dict.fromkeys(joint_ids, 0)

    @property
    def has_resources(self) -> bool:
        """Keep a failed-close handle visible to the owning worker."""
        return self._serial is not None

    def open(self) -> None:
        """Open 8N1 UART without issuing configuration or servo commands."""
        if self._serial is not None:
            raise RuntimeError("102 read-only transport already open")
        self._owner_ident = threading.get_ident()
        bus = serial.Serial(port=None, baudrate=self.baudrate, timeout=0.005, write_timeout=0.05)
        self._serial = bus
        try:
            bus.dtr = False
            bus.rts = False
            bus.port = self.port
            bus.open()
        except Exception:
            bus.close()
            self._serial = None
            raise
        self._failure = None

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner_ident:
            raise RuntimeError("102 transport I/O must run on its single owner thread")
        if self._serial is None:
            raise RuntimeError("102 read-only transport is closed")

    def read_monitors(self, *, max_read_duration_s: float) -> dict[int, ReceivedMonitor]:
        """Receive every requested ID once before the deadline; never reuse cache."""
        self._check_owner()
        if not math.isfinite(max_read_duration_s) or max_read_duration_s <= 0:
            raise ValueError("max_read_duration_s must be finite and positive")
        if self._failure is not None:
            raise RuntimeError(f"102 protocol transport fault latched: {self._failure}")
        try:
            return self._read_transaction(max_read_duration_s)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise

    def _read_transaction(self, budget_s: float) -> dict[int, ReceivedMonitor]:
        started_ns = time.perf_counter_ns()
        deadline_ns = started_ns + int(budget_s * 1e9)
        self._serial.reset_input_buffer()
        self._serial.write_timeout = min(budget_s, 0.05)
        if self._serial.write(self.query_packet) != len(self.query_packet):
            raise OSError("102 query write was partial")
        # No serial.flush(): on some transports it can wait without a deadline.
        pending = bytearray()
        result = {}
        max_bytes = len(self.joint_ids) * 21
        byte_count = 0
        while len(result) < len(self.joint_ids):
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                raise TimeoutError(
                    f"102 monitor deadline; missing IDs={sorted(set(self.joint_ids) - set(result))}"
                )
            self._serial.timeout = min(0.005, remaining_ns / 1e9)
            data = self._serial.read(max(1, min(max_bytes - byte_count + 1, self._serial.in_waiting)))
            received_ns = time.perf_counter_ns()
            if received_ns > deadline_ns:
                raise TimeoutError("102 monitor response arrived after deadline")
            byte_count += len(data)
            if byte_count > max_bytes:
                raise RuntimeError("102 response exceeds bounded monitor packet size")
            pending.extend(data)
            while len(pending) >= 4:
                if pending[:2] != b"\x05\x1c":
                    raise RuntimeError("102 invalid response header")
                code, size = pending[2:4]
                if code != 22 or size not in (14, 16):
                    raise RuntimeError(f"102 unexpected response code/length: {code}/{size}")
                packet_size = 5 + size
                if len(pending) < packet_size:
                    break
                packet = bytes(pending[:packet_size])
                del pending[:packet_size]
                if (sum(packet[:-1]) & 0xFF) != packet[-1]:
                    raise RuntimeError("102 monitor checksum mismatch")
                params = packet[4:-1]
                motor_id = params[0]
                if motor_id not in self._receive_sequences:
                    raise RuntimeError(f"102 unexpected servo ID {motor_id}")
                if motor_id in result:
                    raise RuntimeError(f"102 duplicate servo ID {motor_id}")
                self._receive_sequences[motor_id] += 1
                result[motor_id] = ReceivedMonitor(
                    motor_id=motor_id,
                    angle_deg=int.from_bytes(params[10:14], byteorder="little", signed=True) / 10.0,
                    receive_timestamp_ns=received_ns,
                    receive_sequence=self._receive_sequences[motor_id],
                    motor_status=params[9],
                )
        if pending:
            raise RuntimeError("102 trailing partial packet after complete monitor response")
        return result

    def close(self) -> None:
        """Close the UART on its owner; do not change servo power or configuration."""
        self._check_owner()
        self._serial.close()
        self._serial = None
