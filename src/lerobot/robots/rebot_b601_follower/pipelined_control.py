"""Single-owner, bounded seven-axis submission/RX; no wire ACK or transaction ID."""

import copy
import math
import time
from dataclasses import replace

from lerobot.force_control.hardware_types import JointFeedback, JointSnapshot

from .full_arm_control_types import JOINT_NAMES


class PipelinedControl:
    """Keep one group until all real replies have been validated and consumed once."""

    def __init__(self, session):
        self.session = session
        self.device = session.device
        self.cycle = None
        self.generation = 0
        self.context = None
        self.snapshot = None
        self.consumed_evidence = None
        self.cached_rows = {}
        self.service_deadline_ns = None

    @property
    def pending(self) -> bool:
        if self.device.config.command_feedback_mode == "cached":
            return False
        return self.cycle is not None and self.cycle["completed_ns"] is None

    def evidence(self) -> dict | None:
        """Export one decoding signature per group without changing live validation."""
        if self.cycle is None:
            return None
        signature = self.context["session_signature"]

        def reference_signature(record):
            # Preserve a differing signature verbatim; never conceal a context change.
            if record.get("session_signature") != signature:
                return record
            return {
                **{key: value for key, value in record.items() if key != "session_signature"},
                "session_signature_ref": "group.session_signature",
            }

        exported = {
            **self.cycle,
            "session_signature": signature,
            "commands": {
                name: reference_signature(command) for name, command in self.cycle["commands"].items()
            },
            "feedback": {
                name: {**feedback, "rx_context": reference_signature(feedback["rx_context"])}
                for name, feedback in self.cycle["feedback"].items()
            },
        }
        return copy.deepcopy(exported)

    def begin(self, started_ns: int, deadline_ns: int) -> None:
        if self.device.config.command_feedback_mode == "cached":
            self.service_deadline_ns = deadline_ns
            if not self.cached_rows and self.device._latest_snapshot is not None:
                self.cached_rows = {row.name: row for row in self.device._latest_snapshot.joints}
            self.poll()
            self.cycle = None
        if self.cycle is not None:
            self.poll()
            if not self.cycle["consumed"]:
                raise RuntimeError("Previous seven-axis feedback group has not been consumed")
        if self.device.config.command_feedback_mode != "cached":
            self.device._prepare_query(deadline_ns)
        elif (
            self.device._rx_poison is not None
            or self.device._outstanding_query is not None
            or self.device._query_write_inflight is not None
        ):
            raise RuntimeError("Cached control cannot inherit an unresolved query or poisoned RX")
        self.generation += 1
        self.cycle = {
            "generation": self.generation,
            "send_start_ns": started_ns,
            "deadline_ns": deadline_ns,
            "completed_ns": None,
            "consumed": False,
            "hardware_ack": "unknown",
            "correlation": "host_rx_only_no_request_token",
            "commands": {},
            "feedback": {},
            "actual_write_intervals_ns": [],
            "rx_service": {
                "poll_count": 0,
                "empty_poll_count": 0,
                "last_poll_started_ns": None,
                "last_poll_completed_ns": None,
                "max_service_gap_ns": 0,
                "last_driver_bytes": None,
                "last_retained_bytes": 0,
            },
        }
        self.snapshot = None
        self.context = self.device._new_rx_context("pipelined_control", deadline_ns)

    def submit(self, name: str, packet: bytes) -> None:
        cycle, device, session = self.cycle, self.device, self.session
        # Service arrived bytes while pacing, never wait for a particular motor.
        previous = list(cycle["commands"].values())
        if previous:
            due = previous[-1]["write_entered_ns"] + round(device.config.command_tx_gap_us * 1000)
            while time.perf_counter_ns() < due:
                # Keep servicing RX and all live guards during this short gap.
                # A sub-ms sleep also releases the GIL; reacquisition can exceed
                # the gap on Windows. Never add that wait to the group deadline.
                self.poll()
        self.poll()
        command = {
            "name": name,
            "packet_hex": packet.hex(),
            "expected_status": 1,
            "submission": "planned",
            "session_signature": device._rx_session_signature(),
        }
        cycle["commands"][name] = session.pending_response = command
        session.allowed_packet = packet
        try:
            device._write_once(packet, cycle["deadline_ns"])
            session.last_command_ns[name] = time.perf_counter_ns()
            if device.config.command_feedback_mode == "cached":
                # Include the final axis: a stop/fault during write must prevent commit.
                self.poll()
        finally:
            session.allowed_packet = session.pending_response = None
            if previous and "write_entered_ns" in command:
                cycle["actual_write_intervals_ns"].append(
                    command["write_entered_ns"] - previous[-1]["write_entered_ns"]
                )

    def poll(self) -> None:
        """Drain only available bytes, bounded by one group and the original deadline."""
        stats = self.cycle["rx_service"] if self.pending else None
        if stats is not None:
            started = time.perf_counter_ns()
            previous = stats["last_poll_completed_ns"]
            if previous is not None:
                stats["max_service_gap_ns"] = max(stats["max_service_gap_ns"], started - previous)
            stats["last_poll_started_ns"] = started
            stats["poll_count"] += 1
        try:
            self._poll_available()
        finally:
            if stats is not None:
                stats["last_poll_completed_ns"] = time.perf_counter_ns()

    def _poll_available(self) -> None:
        """Keep the official available-byte RX approach without waiting during TX."""
        if self.device.config.command_feedback_mode == "cached":
            self._poll_cached()
            return
        device, session, cycle = self.device, self.session, self.cycle
        device._check_owner()
        session.wait_guard()
        if cycle is None:
            return
        deadline = cycle["deadline_ns"]
        if self.pending and time.perf_counter_ns() >= deadline:
            raise TimeoutError("Seven-axis command feedback deadline expired")
        for _ in range(device.config.prequery_byte_limit + 1):
            session.wait_guard()
            available = device._serial.in_waiting
            cycle["rx_service"]["last_driver_bytes"] = available
            cycle["rx_service"]["last_retained_bytes"] = len(device._rx_pending)
            cycle["rx_service"]["last_driver_bytes_observed_ns"] = time.perf_counter_ns()
            if available + len(device._rx_pending) > device.config.prequery_byte_limit:
                raise RuntimeError("Seven-axis feedback receive buffer overflow")
            if not available and len(device._rx_pending) < 16:
                cycle["rx_service"]["empty_poll_count"] += 1
                return
            if not self.pending:
                raise RuntimeError("Unexpected or duplicate feedback after complete seven-axis group")
            if len(device._rx_pending) < 16:
                device._read_rx_chunk(
                    self.context,
                    deadline,
                    lambda data, received: device._audit_io("pipelined_control_rx", data, received),
                    available_only=True,
                    available_bytes=available,
                )
                if len(device._rx_pending) < 16:
                    continue
            raw, evidence = device._peek_rx_frame(self.context, deadline)
            if raw[:3] != b"\xaa\x11\x08" or raw[15] != 0x55:
                raise RuntimeError("Seven-axis feedback framing failure")
            spec = next(
                (s for s in session.pinned_specs if s.feedback_id == int.from_bytes(raw[3:7], "little")), None
            )
            if spec is None or raw[7] & 15 != spec.motor_id:
                raise RuntimeError("Seven-axis feedback unknown ID/motor")
            command = cycle["commands"].get(spec.name)
            if command is None or command["submission"] != "complete":
                raise RuntimeError("Feedback without a complete host submission")
            if spec.name in cycle["feedback"]:
                raise RuntimeError("Duplicate seven-axis feedback")
            if raw[7] >> 4 != 1:
                raise RuntimeError("Seven-axis feedback unexpected drive status")
            if evidence["first_receive_timestamp_ns"] < command["write_entered_ns"]:
                raise RuntimeError("Seven-axis feedback predates host submission")
            pmax = device._mapping_values(spec.name)[0]
            position = math.degrees(int.from_bytes(raw[8:10], "big") * 2 * pmax / 65535 - pmax)
            bounds = next(item for item in session.admission.calibration.joints if item[0] == spec.name)
            if not bounds[2] <= position <= bounds[3] or session.reference_envelope_exceeded(
                spec.name, position
            ):
                raise RuntimeError("Seven-axis feedback outside calibration/reference bounds")
            device._sequences[spec.name] += 1
            cycle["feedback"][spec.name] = {
                "position_deg": position,
                "receive_timestamp_ns": evidence["last_receive_timestamp_ns"],
                "receive_sequence": device._sequences[spec.name],
                "rx_context": evidence,
                "packet_hex": raw.hex(),
            }
            session.observed_statuses[spec.name] = 1
            device._finish_rx_frame()
            if len(cycle["feedback"]) == 7:
                completed = time.perf_counter_ns()
                if completed > deadline:
                    raise TimeoutError("Seven-axis feedback completed after its original deadline")
                cycle["completed_ns"] = completed
                self.snapshot = JointSnapshot(
                    tuple(
                        JointFeedback(
                            name,
                            row["position_deg"],
                            True,
                            row["receive_timestamp_ns"],
                            row["receive_sequence"],
                            completed - row["receive_timestamp_ns"],
                            "host_receive_no_device_request_token",
                        )
                        for name in JOINT_NAMES
                        for row in (cycle["feedback"][name],)
                    ),
                    completed,
                    deadline,
                )
        raise RuntimeError("Seven-axis feedback service exceeded its byte bound")

    def _poll_cached(self) -> None:
        """Drain available replies without requiring a same-cycle seven-axis group."""
        device, session = self.device, self.session
        device._check_owner()
        session.wait_guard()
        if device._rx_poison is not None:
            raise RuntimeError(device._rx_poison)
        deadline = self.service_deadline_ns
        if deadline is None:
            # Between ticks there is no live decision. Bound service by the oldest
            # cached input's expiry, never extend an active decision's deadline.
            if not self.cached_rows:
                return
            deadline = min(row.receive_timestamp_ns for row in self.cached_rows.values()) + int(
                device.config.feedback_max_age_s * 1e9
            )
        context = device._new_rx_context("cached_control", deadline)
        received_bytes = 0
        for _ in range(device.config.prequery_byte_limit + 1):
            session.wait_guard()
            device._remaining(deadline)
            available = device._serial.in_waiting
            if available + len(device._rx_pending) > device.config.prequery_byte_limit:
                raise RuntimeError("Cached feedback receive buffer overflow")
            if not available and len(device._rx_pending) < 16:
                return
            if len(device._rx_pending) < 16:
                device._read_rx_chunk(
                    context,
                    deadline,
                    lambda data, received: device._audit_io("cached_control_rx", data, received),
                    available_only=True,
                    available_bytes=available,
                )
                if len(device._rx_pending) < 16:
                    continue
            if device._rx_context["source"] != "cached_control":
                raise RuntimeError("Cached feedback cannot inherit lifecycle/query bytes")
            raw, evidence = device._peek_rx_frame(context, deadline, allow_inherited=True)
            if raw[:3] != b"\xaa\x11\x08" or raw[15] != 0x55:
                raise RuntimeError("Cached feedback framing failure")
            spec = next(
                (s for s in session.pinned_specs if s.feedback_id == int.from_bytes(raw[3:7], "little")), None
            )
            if spec is None or raw[7] & 15 != spec.motor_id:
                raise RuntimeError("Cached feedback unknown ID/motor")
            if raw[7] >> 4 != 1:
                raise RuntimeError("Cached feedback unexpected drive status")
            pmax = device._mapping_values(spec.name)[0]
            position = math.degrees(int.from_bytes(raw[8:10], "big") * 2 * pmax / 65535 - pmax)
            bounds = next(item for item in session.admission.calibration.joints if item[0] == spec.name)
            if not bounds[2] <= position <= bounds[3] or session.reference_envelope_exceeded(
                spec.name, position
            ):
                raise RuntimeError("Cached feedback outside calibration/reference bounds")
            device._sequences[spec.name] += 1
            row = JointFeedback(
                spec.name,
                position,
                True,
                evidence["last_receive_timestamp_ns"],
                device._sequences[spec.name],
                0,
                "host_receive_no_device_request_token",
            )
            self.cached_rows[spec.name] = row
            session.observed_statuses[spec.name] = 1
            device._finish_rx_frame()
            received_bytes += 16
            if received_bytes >= device.config.prequery_byte_limit:
                if device._serial.in_waiting or device._rx_pending:
                    raise RuntimeError("Cached feedback service exceeded its byte bound")
                return
        raise RuntimeError("Cached feedback service exceeded its byte bound")

    def consume_cached(self, names, deadline_ns: int) -> JointSnapshot:
        if len(names) != 7 or set(names) != set(JOINT_NAMES):
            raise ValueError("Cached control requires all seven joints")
        if not self.cached_rows and self.device._latest_snapshot is not None:
            self.cached_rows = {row.name: row for row in self.device._latest_snapshot.joints}
        self.service_deadline_ns = deadline_ns
        try:
            self.poll()
        finally:
            self.service_deadline_ns = None
        now = time.perf_counter_ns()
        if now > deadline_ns or set(self.cached_rows) != set(JOINT_NAMES):
            raise RuntimeError("Cached seven-axis feedback unavailable")
        rows = []
        for name in names:
            row = self.cached_rows[name]
            if not row.is_valid or row.position_deg is None or self.session.observed_statuses.get(name) != 1:
                raise RuntimeError(f"Cached feedback invalid: {name}")
            if not 0 <= now - row.receive_timestamp_ns <= self.device.config.feedback_max_age_s * 1e9:
                raise RuntimeError(f"Cached feedback expired: {name}")
            rows.append(replace(row, age_ns=now - row.receive_timestamp_ns))
        result = JointSnapshot(tuple(rows), now, deadline_ns)
        self.device._latest_snapshot = result
        return result

    def consume(self, names, deadline_ns: int) -> JointSnapshot:
        if set(names) != set(JOINT_NAMES) or len(names) != 7:
            raise ValueError("Pipelined control requires all seven joints")
        if self.cycle["consumed"]:
            raise RuntimeError("Seven-axis feedback already consumed; old feedback reuse forbidden")
        self.poll()
        while self.pending:
            if time.perf_counter_ns() >= deadline_ns:
                raise TimeoutError("Control feedback read deadline expired")
            time.sleep(0.00012)
            self.poll()
        now = time.perf_counter_ns()
        if now > deadline_ns or any(
            not 0 <= now - row.receive_timestamp_ns <= self.device.config.feedback_max_age_s * 1e9
            or row.receive_sequence <= self.session.returned_sequences.get(row.name, 0)
            for row in self.snapshot.joints
        ):
            raise RuntimeError("Seven-axis feedback is stale or already returned")
        result = JointSnapshot(
            tuple(
                replace(self.snapshot.get(name), age_ns=now - self.snapshot.get(name).receive_timestamp_ns)
                for name in names
            ),
            self.snapshot.completed_ns,
            deadline_ns,
        )
        self.cycle["consumed"] = True
        if self.consumed_evidence is not None:
            raise RuntimeError("Unrecorded seven-axis feedback evidence would be overwritten")
        self.consumed_evidence = self.evidence()
        self.device._latest_snapshot = result
        self.session.note_returned_feedback(result)
        return result

    def take_consumed_evidence(self) -> dict | None:
        result, self.consumed_evidence = self.consumed_evidence, None
        return result

    def retire(self) -> dict | None:
        evidence = self.evidence()
        self.cycle = self.context = self.snapshot = None
        self.cached_rows.clear()
        self.service_deadline_ns = None
        return evidence
