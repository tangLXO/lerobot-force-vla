"""Query-only CTRL_MODE/TIMEOUT inspection on an existing B601 I/O owner.

This API does not widen the mapping-register allowlist, open a second handle,
write parameters, enable drives, or establish a physical watchdog guarantee.
"""

import struct
import time
from dataclasses import dataclass

from .control_protocol import encode_mode_query, encode_timeout_query
from .readonly_transport import ReadonlyB601Transport


@dataclass(frozen=True)
class B601ControlRegister:
    """One actual uint32 register response, with raw receive provenance."""

    name: str
    motor_id: int
    feedback_id: int
    register_id: int
    value: int
    receive_timestamp_ns: int
    receive_sequence: int
    packet_hex: str


@dataclass(frozen=True)
class B601ControlState:
    """A complete bounded query attempt; register readback is not motion acceptance."""

    started_ns: int
    completed_ns: int
    deadline_ns: int
    registers: tuple[B601ControlRegister, ...]
    query_packets_hex: tuple[str, ...]
    receive_chunks: tuple[tuple[int, str], ...]
    missing: tuple[str, ...]
    error: str | None
    hardware_ack: str = "unknown"
    watchdog_semantics_verified: bool = False

    @property
    def is_valid(self) -> bool:
        """Require a complete, error-free response set within its whole-batch deadline."""
        return (
            bool(self.registers)
            and not self.missing
            and self.error is None
            and self.started_ns <= self.completed_ns <= self.deadline_ns
        )


def read_control_state(
    transport: ReadonlyB601Transport,
    *,
    names: tuple[str, ...] | None = None,
    max_read_duration_s: float = 1.0,
) -> B601ControlState:
    """Read registers 10 and 9 through the existing transport and owner only.

    The caller opens/closes its transport normally. This function never performs
    a parameter write and can run before mapping acquisition; failed RX remains
    latched on that transport, including partial bytes and attempted queries.
    """
    if not isinstance(transport, ReadonlyB601Transport):
        raise TypeError("Control-state queries require an existing B601 transport")
    by_name = {spec.name: spec for spec in transport.joint_specs}
    selected = tuple(by_name) if names is None else tuple(names)
    if not selected or len(set(selected)) != len(selected) or set(selected) - by_name.keys():
        raise ValueError("Control-state names must be unique configured joints")
    started, deadline = transport._start_attempt(max_read_duration_s)
    queries, chunks, received = [], [], []
    error = f"fault_latched: {transport.failure}" if transport.failure else None
    interrupted = None
    try:
        if error is None:
            for name in selected:
                spec = by_name[name]
                for rid in (10, 9):

                    def consume(raw, arbitration_id, received_ns, spec=spec, rid=rid):
                        if arbitration_id != spec.feedback_id or raw[7:11] != bytes(
                            (spec.motor_id, 0, 0x33, rid)
                        ):
                            raise RuntimeError("B601 control-state response ID/opcode/register mismatch")
                        previous = transport._sequences[spec.name]
                        received.append(
                            B601ControlRegister(
                                spec.name,
                                spec.motor_id,
                                arbitration_id,
                                rid,
                                struct.unpack("<I", raw[11:15])[0],
                                received_ns,
                                previous + 1,
                                raw.hex(),
                            )
                        )
                        transport._sequences[spec.name] = previous + 1

                        def rollback():
                            received.pop()
                            transport._sequences[spec.name] = previous

                        return rollback

                    packet = (
                        encode_mode_query(spec.motor_id) if rid == 10 else encode_timeout_query(spec.motor_id)
                    )
                    transport._exchange((packet,), deadline, queries, chunks, consume)
    except BaseException as exc:
        transport._failure = error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            interrupted = exc
    completed = time.perf_counter_ns()
    if error is None and completed > deadline:
        transport._failure = error = "TimeoutError: B601 control-state batch completed after deadline"
    present = {(item.name, item.register_id) for item in received}
    missing = tuple(f"{name}:{rid}" for name in selected for rid in (10, 9) if (name, rid) not in present)
    result = B601ControlState(
        started, completed, deadline, tuple(received), tuple(queries), tuple(chunks), missing, error
    )
    if interrupted is not None:
        raise interrupted
    return result
