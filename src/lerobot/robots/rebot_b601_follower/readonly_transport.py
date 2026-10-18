"""Query-only B601 Damiao bridge transport with packet receive evidence.

Wire layouts were reviewed at motorbridge v0.3.9, commit
23c22998db072bc0f2ee841874220ac36844a550: motor_core/src/dm_serial.rs and
motor_vendors/damiao/src/{protocol,motor,registers}.rs. Only feedback query
0xCC and mapping-register reads 0x33/21..23 can be encoded here. There are no
motor commands, SDK background readers, parameter writes, or automatic retries.

The bridge's RX framing has no host-visible checksum or transaction token.
Timestamps describe host read completion, not device measurement time or ACK.
Register readback establishes decoding scales, not mechanical calibration.
This diagnostic adapter does not authorize the Force Teacher motion backend.
"""

import math
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _serial_available, require_package

from .control_io import B601BusLease
from .serial_timeouts import set_serial_timeout

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None


MAPPING_REGISTERS = (21, 22, 23)
STATUS_NAMES = {
    0: "DISABLED",
    1: "ENABLED",
    8: "OVER_VOLTAGE",
    9: "UNDER_VOLTAGE",
    10: "OVER_CURRENT",
    11: "MOS_OVER_TEMP",
    12: "ROTOR_OVER_TEMP",
    13: "LOST_COMM",
    14: "OVERLOAD",
}


@dataclass(frozen=True)
class B601ReadonlyJointSpec:
    """Explicit expected motor and feedback IDs; model is an audit label."""

    name: str
    motor_id: int
    feedback_id: int
    model: str


DEFAULT_B601_JOINTS = tuple(
    B601ReadonlyJointSpec(name, index, 0x10 + index, "4340P" if index <= 3 else "4310")
    for index, name in enumerate(
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"),
        start=1,
    )
)


@dataclass(frozen=True)
class B601ReceivedRegister:
    """One validated mapping-register reply, retaining its entire wire packet."""

    name: str
    motor_id: int
    feedback_id: int
    model: str
    register_id: int
    value: float
    receive_timestamp_ns: int
    receive_sequence: int
    packet_hex: str
    payload_hex: str


@dataclass(frozen=True)
class B601ReceivedFeedback:
    """Fresh host packet decoded using this session's actual mapping readback."""

    name: str
    motor_id: int
    feedback_id: int
    model: str
    status_code: int
    status_name: str
    position_rad: float
    position_deg: float
    velocity_rad_s: float
    torque: float
    mos_temperature: int
    rotor_temperature: int
    receive_timestamp_ns: int
    receive_sequence: int
    packet_hex: str
    payload_hex: str
    mapping_values: tuple[float, float, float]
    mapping_source: str = "session_register_readback_21_22_23"
    mapping_verified_on_device: bool = True
    timestamp_source: str = "host_serial_read_return_perf_counter_ns"


@dataclass(frozen=True)
class B601ReadonlyBatch:
    """One attempt including partial evidence; an error always invalidates it."""

    kind: str
    started_ns: int
    completed_ns: int
    deadline_ns: int
    feedback: tuple[B601ReceivedFeedback, ...] = ()
    registers: tuple[B601ReceivedRegister, ...] = ()
    query_packets_hex: tuple[str, ...] = ()
    receive_chunks: tuple[tuple[int, str], ...] = ()
    missing: tuple[str, ...] = ()
    error: str | None = None
    packet_integrity: str = "bridge_framing_and_ids_only_no_host_checksum"
    hardware_measurement_timestamp: None = None
    hardware_acknowledgement: str = "unavailable"

    @property
    def is_valid(self) -> bool:
        """Require a complete attempt with no I/O, protocol, or device error."""
        return (
            self.error is None
            and self.started_ns <= self.completed_ns <= self.deadline_ns
            and not self.missing
            and bool(self.feedback or self.registers)
        )


def encode_readonly_query(motor_id: int, *, register_id: int | None = None) -> bytes:
    """Encode only standard CAN 0x7FF feedback or allowlisted mapping reads."""
    if type(motor_id) is not int or not 1 <= motor_id <= 15:
        raise ValueError("B601 feedback motor_id must be an integer in 1..15")
    if register_id is not None and (type(register_id) is not int or register_id not in MAPPING_REGISTERS):
        raise ValueError("Only mapping-register reads 21, 22, 23 are allowed")
    payload = bytes((motor_id, 0, 0xCC if register_id is None else 0x33, register_id or 0, 0, 0, 0, 0))
    raw = bytearray(30)
    raw[:4] = bytes((0x55, 0xAA, 0x1E, 0x03))
    raw[4:8] = struct.pack("<I", 1)
    raw[8:12] = struct.pack("<I", 10)
    raw[13:17] = struct.pack("<I", 0x7FF)
    raw[18] = 8
    raw[21:29] = payload
    return bytes(raw)


class ReadonlyB601Transport:
    """One owner for open/query/read/close, with lease and latched failed I/O.

    Start with ``read_mapping``. Feedback never uses SDK default limits or
    cached joint positions. A failed attempt preserves received bytes but
    latches subsequent queries. Create a new instance for a deliberate retry.
    Read/write timeouts and post-call deadline checks bound useful attempts;
    OS scheduling and serial administration/close are not hard real-time calls.
    Late responses are always invalid.
    """

    def __init__(
        self,
        port: str,
        joint_specs: tuple[B601ReadonlyJointSpec, ...] = DEFAULT_B601_JOINTS,
        *,
        baudrate: int = 921600,
        lock_root: Path | None = None,
    ):
        """Validate the diagnostic configuration without opening any hardware."""
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        if not isinstance(port, str) or not port.strip():
            raise ValueError("A nonempty B601 serial port is required")
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if not joint_specs or len(joint_specs) > 7:
            raise ValueError("Specify 1..7 B601 joints")
        for field in ("name", "motor_id", "feedback_id"):
            if len({getattr(spec, field) for spec in joint_specs}) != len(joint_specs):
                raise ValueError(f"B601 joint {field} entries must be unique")
        for spec in joint_specs:
            encode_readonly_query(spec.motor_id)
            if not isinstance(spec.name, str) or not spec.name:
                raise ValueError("Every B601 joint needs a name")
            if type(spec.feedback_id) is not int or not 1 <= spec.feedback_id <= 0x7FF:
                raise ValueError("Expected feedback IDs must be nonzero standard CAN IDs")
            if spec.model not in ("4310", "4340P"):
                raise ValueError("This B601 diagnostic supports model labels 4310 and 4340P only")
        self.port = port
        self.baudrate = baudrate
        self.joint_specs = tuple(joint_specs)
        self._lease = B601BusLease("damiao", port, lock_root=lock_root)
        self._serial = None
        self._owner_thread = None
        self._owner_pid = None
        self._opened_once = False
        self._failure: str | None = None
        self._sequences = dict.fromkeys((spec.name for spec in self.joint_specs), 0)
        self._mapping: dict[tuple[str, int], B601ReceivedRegister] = {}
        self.last_batch: B601ReadonlyBatch | None = None
        # One fixed-size wire frame survives interrupted reads on this owner.
        # It is never transferred to a new transport or promoted to a later query.
        self._rx_pending = bytearray()
        self._rx_context: dict | None = None
        self._rx_operation = 0
        self._rx_poison: str | None = None

    @property
    def has_resources(self) -> bool:
        """Retain transport and ownership on an unconfirmed close failure."""
        return self._serial is not None

    @property
    def failure(self) -> str | None:
        """The first latched failure; no method automatically clears it."""
        return self._failure

    @property
    def mapping_ready(self) -> bool:
        """Require every scale register from this live, nonfaulted session."""
        return (
            self._serial is not None
            and self._failure is None
            and len(self._mapping) == len(self.joint_specs) * len(MAPPING_REGISTERS)
        )

    def open(self) -> None:
        """Acquire the shared bus lease, then open 8N1 without sending packets."""
        if self._opened_once:
            raise RuntimeError("B601 read-only transport is single-use")
        self._opened_once = True
        self._owner_thread = threading.current_thread()
        self._owner_pid = os.getpid()
        self._lease.acquire()
        bus = None
        try:
            bus = serial.Serial(
                port=None,
                baudrate=self.baudrate,
                timeout=0.002,
                write_timeout=0.02,
                bytesize=8,
                parity="N",
                stopbits=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            self._serial = bus
            bus.dtr = False
            bus.rts = False
            bus.port = self.port
            bus.open()
        except BaseException as open_error:
            # If close raises, leave the handle and lease in place for review.
            if bus is not None:
                try:
                    bus.close()
                except BaseException as close_error:
                    raise BaseExceptionGroup(
                        "B601 open and close failed", [open_error, close_error]
                    ) from None
            self._serial = None
            self._lease.release()
            raise

    def _check_owner(self) -> None:
        if self._owner_pid != os.getpid() or self._owner_thread is not threading.current_thread():
            raise RuntimeError("B601 read-only I/O must run on its single owner thread and process")
        if self._serial is None:
            raise RuntimeError("B601 read-only transport is closed")

    def _start_attempt(self, budget_s: float) -> tuple[int, int]:
        self._check_owner()
        if isinstance(budget_s, bool) or not isinstance(budget_s, (float, int)):
            raise ValueError("Read duration must be finite and positive")
        if not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError("Read duration must be finite and positive")
        started = time.perf_counter_ns()
        return started, started + int(budget_s * 1e9)

    @staticmethod
    def _remaining(deadline_ns: int) -> float:
        remaining = (deadline_ns - time.perf_counter_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError("B601 read-only deadline expired")
        return remaining

    def _bounded_serial_timeout(self, name: str, cap_s: float, deadline_ns: int) -> None:
        """Reuse short positive waits and bound new Windows settings to whole milliseconds."""
        operation = "write" if name == "write_timeout" else "read"
        timeout = math.floor(min(cap_s, self._remaining(deadline_ns)) * 1000) / 1000
        if timeout < 0.001:
            raise TimeoutError(f"B601 {operation} has less than one millisecond remaining")
        current = getattr(self._serial, name)
        if current is not None and 0 < current <= timeout:
            timeout = current
        if current != timeout:
            # Keep a bounded OS-millisecond wait. Native Windows updates only
            # COMMTIMEOUTS, preserving the port's DCB and modem-line state.
            set_serial_timeout(self._serial, name, timeout)
        # Serial configuration itself consumes time; an expired attempt must
        # never progress to a query write or read after a slow setter returns.
        if self._remaining(deadline_ns) < 0.001:
            raise TimeoutError(f"B601 {operation} has less than one millisecond remaining")

    def _rx_session_signature(self) -> tuple:
        """Bind retained bytes to this owner and the exact live decoding scales."""
        return (
            self.port,
            self._owner_pid,
            self._owner_thread.ident,
            tuple((spec.name, spec.motor_id, spec.feedback_id, spec.model) for spec in self.joint_specs),
            tuple((name, rid, item.value) for (name, rid), item in sorted(self._mapping.items())),
        )

    def _new_rx_context(self, source: str, deadline_ns: int, packet_hex: str | None = None) -> dict:
        self._check_owner()
        self._rx_operation += 1
        return {
            "operation": self._rx_operation,
            "source": source,
            "packet_hex": packet_hex,
            "deadline_ns": deadline_ns,
            "session_signature": self._rx_session_signature(),
            "first_receive_timestamp_ns": None,
            "last_receive_timestamp_ns": None,
            "first_audit_sequence": None,
            "last_audit_sequence": None,
        }

    def _read_rx_chunk(self, context: dict, deadline_ns: int, receive) -> None:
        """Stage at most one frame before audit/deadline failures can discard bytes."""
        self._check_owner()
        if self._rx_poison is not None:
            raise RuntimeError(self._rx_poison)
        room = 16 - len(self._rx_pending)
        if room <= 0:
            raise RuntimeError("B601 retained RX frame must be validated before another read")
        self._bounded_serial_timeout("timeout", 0.002, deadline_ns)
        data = self._serial.read(max(1, min(room, self._serial.in_waiting)))
        received_ns = time.perf_counter_ns()
        if data:
            if len(data) > room:
                # A driver violating read(size) destroys our known frame boundary.
                # Audit the actual chunk, but never scan or silently truncate it.
                self._rx_poison = "B601 RX exceeds its fixed 16-byte frame slot"
                receive(data, received_ns)
                raise RuntimeError(self._rx_poison)
            if not self._rx_pending:
                self._rx_context = dict(context)
                self._rx_context["first_receive_timestamp_ns"] = received_ns
            elif self._rx_context["operation"] != context["operation"]:
                self._rx_context.setdefault("inherited_prefix_hex", self._rx_pending.hex())
            self._rx_pending.extend(data)
            self._rx_context["last_receive_timestamp_ns"] = received_ns
            audit_sequence = receive(data, received_ns)
            if audit_sequence is not None:
                if self._rx_context["first_audit_sequence"] is None:
                    self._rx_context["first_audit_sequence"] = audit_sequence
                self._rx_context["last_audit_sequence"] = audit_sequence
        if received_ns > deadline_ns:
            if data:
                raise TimeoutError("B601 response arrived after deadline")
            if not self._rx_pending:
                raise TimeoutError("B601 no response before deadline")
            raise TimeoutError(
                f"B601 response incomplete at deadline ({len(self._rx_pending)}/16 bytes received)"
            )
        self._remaining(deadline_ns)

    def _peek_rx_frame(self, context: dict, deadline_ns: int, *, allow_inherited: bool = False) -> tuple:
        """Return a complete frame without clearing it until its consumer validates it."""
        self._check_owner()
        self._remaining(deadline_ns)
        if self._rx_poison is not None:
            raise RuntimeError(self._rx_poison)
        if len(self._rx_pending) != 16 or self._rx_context is None:
            raise RuntimeError("B601 RX frame is incomplete")
        if self._rx_context["session_signature"] != self._rx_session_signature():
            self._rx_poison = "B601 retained RX decoding context changed"
            raise RuntimeError(self._rx_poison)
        inherited = self._rx_context["operation"] != context["operation"]
        if inherited and not allow_inherited:
            raise RuntimeError("B601 inherited RX cannot become current feedback")
        evidence = dict(self._rx_context)
        evidence["inherited"] = inherited
        if inherited:
            evidence.setdefault("inherited_prefix_hex", self._rx_pending.hex())
        return bytes(self._rx_pending), evidence

    def _finish_rx_frame(self) -> None:
        """Release only a validated full frame; partial/error bytes remain retained."""
        if len(self._rx_pending) != 16:
            raise RuntimeError("B601 cannot finish an incomplete RX frame")
        self._rx_pending.clear()
        self._rx_context = None

    def _exchange(self, requests, deadline_ns: int, queries: list, chunks: list, consume) -> None:
        """Bound RX and retain failures; a consumer returns an undo for its one frame."""
        if len(requests) > 1:
            # The bridge did not reliably process tightly burst or concatenated
            # frames in hardware tests. Complete each query/reply on this owner
            # before submitting the next, under the same whole-batch deadline.
            for request in requests:
                self._exchange((request,), deadline_ns, queries, chunks, consume)
            return
        self._remaining(deadline_ns)
        self._prepare_query(deadline_ns)
        self._bounded_serial_timeout("write_timeout", 0.02, deadline_ns)
        # Retain attempted-query evidence before the bounded write.
        queries.extend(request.hex() for request in requests)
        payload = b"".join(requests)
        self._query_attempt(payload)
        if self._serial.write(payload) != len(payload):
            raise OSError("B601 read-only query write was partial")
        self._query_submitted(payload, deadline_ns)
        self._remaining(deadline_ns)
        # No flush(): its wait is not deadline bounded on all platforms.
        context = self._new_rx_context("query_rx", deadline_ns, payload.hex())

        def receive(data, received_ns):
            chunks.append((received_ns, data.hex()))
            return self._receive_chunk(data, received_ns)

        while len(self._rx_pending) < 16:
            self._read_rx_chunk(context, deadline_ns, receive)
        raw, evidence = self._peek_rx_frame(context, deadline_ns)
        if self._serial.in_waiting:
            raise RuntimeError("B601 response exceeds the bounded query packet count")
        if raw[:2] != b"\xaa\x11" or raw[15] != 0x55:
            raise RuntimeError("B601 invalid bridge response framing")
        if raw[2] != 8:
            raise RuntimeError("B601 response must be a standard, non-RTR DLC-8 frame")
        arbitration_id = int.from_bytes(raw[3:7], "little")
        if arbitration_id > 0x7FF:
            raise RuntimeError("B601 nonstandard arbitration ID")
        frame_context = dict(self._rx_context)
        rollback = consume(raw, arbitration_id, evidence["last_receive_timestamp_ns"])
        try:
            self._finish_rx_frame()
            self._remaining(deadline_ns)
        except BaseException as exc:
            self._rx_pending[:] = raw
            self._rx_context = frame_context
            if rollback is not None:
                try:
                    rollback()
                except BaseException as rollback_error:
                    self._rx_poison = "B601 RX consumer rollback failed; retained frame is not usable"
                    raise BaseExceptionGroup(
                        f"B601 query failed ({type(exc).__name__}: {exc}); "
                        f"rollback failed ({type(rollback_error).__name__}: {rollback_error})",
                        [exc, rollback_error],
                    ) from None
            raise
        self._query_completed(payload)

    def _prepare_query(self, deadline_ns: int) -> None:
        """Discard pre-query bytes only in the original query-only diagnostic path."""
        if self._rx_pending or self._rx_poison is not None:
            raise RuntimeError("B601 prior RX remains pending; no new query is allowed")
        self._serial.reset_input_buffer()

    def _query_attempt(self, payload: bytes) -> None:
        """Allow the unified execution adapter to retain transport audit evidence."""

    def _query_submitted(self, payload: bytes, deadline_ns: int) -> None:
        """Allow the unified owner to track a complete query before any RX/deadline check."""

    def _query_completed(self, payload: bytes) -> None:
        """Allow the unified owner to retire only a validated, timely query response."""

    def _receive_chunk(self, data: bytes, received_ns: int) -> int | None:
        """Allow the unified adapter to audit received bytes before decoding them."""

    def read_mapping(self, *, max_read_duration_s: float = 1.0) -> B601ReadonlyBatch:
        """Read only PMAX/VMAX/TMAX; preserve partial replies and fail closed."""
        started, deadline = self._start_attempt(max_read_duration_s)
        if self._mapping and self._failure is None:
            raise RuntimeError("Mapping has already been read for this single-use session")
        queries, chunks, received = [], [], []
        error = f"fault_latched: {self._failure}" if self._failure else None
        interrupted = None
        try:
            if error is None:
                for spec in self.joint_specs:
                    for rid in MAPPING_REGISTERS:

                        def consume(raw, arbitration_id, received_ns, spec=spec, rid=rid):
                            data = raw[7:15]
                            if arbitration_id != spec.feedback_id:
                                raise RuntimeError("B601 mapping reply has an unexpected feedback ID")
                            if data[:4] != bytes((spec.motor_id, 0, 0x33, rid)):
                                raise RuntimeError(
                                    "B601 mapping reply has an unexpected motor/opcode/register"
                                )
                            value = struct.unpack("<f", data[4:])[0]
                            if not math.isfinite(value) or value <= 0:
                                raise RuntimeError("B601 mapping reply is not finite and positive")
                            previous_sequence = self._sequences[spec.name]
                            item = B601ReceivedRegister(
                                spec.name,
                                spec.motor_id,
                                arbitration_id,
                                spec.model,
                                rid,
                                value,
                                received_ns,
                                previous_sequence + 1,
                                raw.hex(),
                                data.hex(),
                            )
                            key = (spec.name, rid)
                            previous_mapping = self._mapping.get(key)
                            received.append(item)
                            self._mapping[key] = item
                            self._sequences[spec.name] = previous_sequence + 1

                            def rollback():
                                received.pop()
                                self._sequences[spec.name] = previous_sequence
                                if previous_mapping is None:
                                    self._mapping.pop(key)
                                else:
                                    self._mapping[key] = previous_mapping

                            return rollback

                        self._exchange(
                            (encode_readonly_query(spec.motor_id, register_id=rid),),
                            deadline,
                            queries,
                            chunks,
                            consume,
                        )
        except BaseException as exc:
            self._failure = error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupted = exc
        missing = tuple(
            f"{spec.name}:{rid}"
            for spec in self.joint_specs
            for rid in MAPPING_REGISTERS
            if (spec.name, rid) not in {(item.name, item.register_id) for item in received}
        )
        completed_ns = time.perf_counter_ns()
        if error is None and completed_ns > deadline:
            # Earlier frames completed their own transaction in time. Their RX
            # sequences remain diagnostic evidence, while this batch is invalid.
            self._failure = error = "TimeoutError: B601 read-only batch completed after deadline"
        batch = B601ReadonlyBatch(
            "mapping",
            started,
            completed_ns,
            deadline,
            registers=tuple(received),
            query_packets_hex=tuple(queries),
            receive_chunks=tuple(chunks),
            missing=missing,
            error=error,
        )
        self.last_batch = batch
        if interrupted is not None:
            raise interrupted
        return batch

    def read_feedback(
        self,
        *,
        max_read_duration_s: float = 0.05,
        names: tuple[str, ...] | None = None,
        deadline_ns: int | None = None,
    ) -> B601ReadonlyBatch:
        """Query exactly the requested joints within both duration and caller deadline.

        Omitted names retain the diagnostic's original all-joint behavior. An
        absolute deadline can only shorten the budget; it cannot restamp late RX.
        """
        requested = tuple(spec.name for spec in self.joint_specs) if names is None else tuple(names)
        by_name = {spec.name: spec for spec in self.joint_specs}
        if not requested or len(set(requested)) != len(requested) or set(requested) - by_name.keys():
            raise ValueError("Feedback names must be nonempty, unique configured joints")
        if deadline_ns is not None and (type(deadline_ns) is not int or deadline_ns <= 0):
            raise ValueError("deadline_ns must be a positive integer")
        specs = tuple(by_name[name] for name in requested)
        started, deadline = self._start_attempt(max_read_duration_s)
        if deadline_ns is not None:
            deadline = min(deadline, deadline_ns)
        queries, chunks, received = [], [], {}
        error = f"fault_latched: {self._failure}" if self._failure else None
        interrupted = None
        if error is None and not self.mapping_ready:
            raise RuntimeError("Read and validate device mapping registers before feedback")
        try:
            if error is None:

                def consume(raw, arbitration_id, received_ns):
                    # Bind the reply to the one outstanding query, not merely
                    # any joint in the batch. There is no wire transaction ID.
                    if arbitration_id != spec.feedback_id:
                        raise RuntimeError("B601 unexpected feedback ID")
                    data = raw[7:15]
                    if data[0] & 15 != spec.motor_id:
                        raise RuntimeError("B601 feedback motor ID does not match arbitration ID")
                    if spec.name in received:
                        raise RuntimeError("B601 duplicate joint feedback")
                    if data[0:3] == bytes((spec.motor_id, 0, 0x33)):
                        raise RuntimeError("B601 register-shaped packet during feedback query")
                    limits = tuple(self._mapping[(spec.name, rid)].value for rid in MAPPING_REGISTERS)
                    pos = int.from_bytes(data[1:3], "big") * (2 * limits[0]) / 65535 - limits[0]
                    vel = ((data[3] << 4) | (data[4] >> 4)) * (2 * limits[1]) / 4095 - limits[1]
                    torque = (((data[4] & 15) << 8) | data[5]) * (2 * limits[2]) / 4095 - limits[2]
                    previous_sequence = self._sequences[spec.name]
                    status = data[0] >> 4
                    received[spec.name] = B601ReceivedFeedback(
                        spec.name,
                        spec.motor_id,
                        arbitration_id,
                        spec.model,
                        status,
                        STATUS_NAMES.get(status, "UNKNOWN"),
                        pos,
                        math.degrees(pos),
                        vel,
                        torque,
                        data[6],
                        data[7],
                        received_ns,
                        previous_sequence + 1,
                        raw.hex(),
                        data.hex(),
                        limits,
                    )
                    self._sequences[spec.name] = previous_sequence + 1

                    def rollback():
                        received.pop(spec.name)
                        self._sequences[spec.name] = previous_sequence

                    return rollback

                for spec in specs:
                    self._exchange(
                        (encode_readonly_query(spec.motor_id),), deadline, queries, chunks, consume
                    )
                if any(item.status_code not in (0, 1) for item in received.values()):
                    raise RuntimeError("B601 feedback reports a fault or unknown device status")
        except BaseException as exc:
            self._failure = error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupted = exc
        completed_ns = time.perf_counter_ns()
        if error is None and completed_ns > deadline:
            # RX sequence counts completed frame transactions, never a control
            # state commit. Late batch assembly still invalidates all use of it.
            self._failure = error = "TimeoutError: B601 read-only batch completed after deadline"
        batch = B601ReadonlyBatch(
            "feedback",
            started,
            completed_ns,
            deadline,
            feedback=tuple(received[spec.name] for spec in specs if spec.name in received),
            query_packets_hex=tuple(queries),
            receive_chunks=tuple(chunks),
            missing=tuple(spec.name for spec in specs if spec.name not in received),
            error=error,
        )
        self.last_batch = batch
        if interrupted is not None:
            raise interrupted
        return batch

    def close(self) -> None:
        """Close on the owner without sending disable, clear-error, or motion."""
        self._check_owner()
        self._serial.close()
        self._serial = None
        self._lease.release()
        self._rx_pending.clear()
        self._rx_context = None
