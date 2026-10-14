"""One transport for calibrated FORCE_POS gripper control and gated arm recording."""

import copy
import math
import os
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lerobot.force_control.hardware_types import JointFeedback, JointSendResult, JointSnapshot, SendReceipt

from .calibration import B601GripperCalibration
from .control_protocol import (
    B601JointCommandProfile,
    encode_disable,
    encode_enable,
    encode_mode_query,
    encode_mode_write,
    encode_position,
    encode_timeout_query,
    encode_timeout_write,
    force_pos_wire_codes,
)
from .readonly_transport import B601ReadonlyJointSpec, ReadonlyB601Transport


def require_b601_control_acceptance() -> None:
    """Refuse real control until dynamic RX, lifecycle and fault behavior are reviewed."""
    raise RuntimeError(
        "A reviewed B601 unified backend is required; hardware acceptance is incomplete: dynamic command/RX ordering, "
        "enable/mode lifecycle, calibrated profiles and physical fault disposition remain unverified"
    )


@dataclass(frozen=True)
class B601ControlConfig:
    """Explicit execution budgets and policy; no copied fake or legacy motor defaults."""

    profiles: tuple[B601JointCommandProfile, ...]
    startup_budget_s: float
    send_budget_s: float
    fault_budget_s: float
    feedback_max_age_s: float
    fault_action: str | None = None
    fault_joint_names: tuple[str, ...] = ()
    io_event_capacity: int = 256
    prequery_byte_limit: int = 256
    calibration_file: str | None = None
    watchdog_ticks: int = 2000
    parameter_write_settle_s: float = 0.01

    def validate(self) -> None:
        """Validate all targets before opening a bus or submitting any joint."""
        if not self.profiles or len({p.name for p in self.profiles}) != len(self.profiles):
            raise ValueError("Unique explicit joint command profiles are required")
        for profile in self.profiles:
            profile.validate()
        if self.calibration_file is not None and (
            not isinstance(self.calibration_file, str) or not self.calibration_file.strip()
        ):
            raise ValueError("calibration_file must name an existing B601 calibration")
        encode_timeout_write(7, self.watchdog_ticks)
        if (
            isinstance(self.parameter_write_settle_s, bool)
            or not isinstance(self.parameter_write_settle_s, (int, float))
            or not math.isfinite(self.parameter_write_settle_s)
            or self.parameter_write_settle_s < 0
        ):
            raise ValueError("parameter_write_settle_s must be finite and nonnegative")
        for name in ("io_event_capacity", "prequery_byte_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("startup_budget_s", "send_budget_s", "fault_budget_s", "feedback_max_age_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.fault_action not in (None, "disable_gripper", "disable_selected"):
            raise ValueError(
                "Only an unresolved policy or explicit unvalidated disable candidates are supported"
            )
        if self.fault_action == "disable_gripper" and tuple(p.name for p in self.profiles) != ("gripper",):
            raise ValueError("A gripper-only fault policy cannot authorize an arm control session")
        if self.fault_action == "disable_selected":
            if len(set(self.fault_joint_names)) != len(self.fault_joint_names) or set(
                self.fault_joint_names
            ) != {p.name for p in self.profiles}:
                raise ValueError(
                    "disable_selected requires an explicit fault joint list covering the whole session"
                )
        elif self.fault_joint_names:
            raise ValueError("fault_joint_names only applies to disable_selected")


class B601ControlBackend(ReadonlyB601Transport):
    """A disabled connection becomes active only through explicit gripper startup.

    Arm scopes keep the separate acceptance gate. The runtime owns force
    selection and supplies a fresh-data guard before every startup write.
    """

    def __init__(
        self,
        port: str,
        joint_specs: tuple[B601ReadonlyJointSpec, ...],
        config: B601ControlConfig,
        *,
        baudrate: int = 921600,
        lock_root: Path | None = None,
    ):
        """Validate disconnected state and allocate no SDK or serial resources."""
        config.validate()
        if {p.name for p in config.profiles} != {s.name for s in joint_specs}:
            raise ValueError("Command profiles must exactly cover this session's selected joints")
        super().__init__(port, joint_specs, baudrate=baudrate, lock_root=lock_root)
        self.config = config
        self._profiles = {p.name: p for p in config.profiles}
        self._specs = {s.name: s for s in joint_specs}
        self._modes: dict[str, int] = {}
        self.mode_evidence: tuple[dict, ...] = ()
        self.mode_queries: tuple[str, ...] = ()
        self.mode_receive_chunks: tuple[tuple[int, str], ...] = ()
        self._latest_snapshot: JointSnapshot | None = None
        self._motion_session = False
        self._observe_session = False
        self._control_session = False
        self._gripper_start_attempted = False
        self._gripper_starting = False
        self._may_be_active = False
        self.calibration: B601GripperCalibration | None = None
        self.startup_receipt: dict | None = None
        self._control_owner_lock = threading.Lock()
        self.last_receipt: SendReceipt | None = None
        self._last_command_feedback: dict | None = None
        self._fault_receipt: dict | None = None
        self._io_events: list[dict] = []
        self._io_event_sequence = 0
        self._io_overflow = False

    @property
    def fault_receipt(self) -> dict | None:
        """Expose detached disposition evidence, including unsuccessful cleanup."""
        return copy.deepcopy(self._fault_receipt)

    @property
    def command_settings(self) -> dict:
        profile = self._profiles["gripper"]
        velocity, drive = force_pos_wire_codes(
            profile.velocity_deg_s, profile.torque_ratio, velocity_wire_code=profile.velocity_wire_code
        )
        return {
            "requested": {
                "motor_velocity_deg_s": profile.velocity_limit_deg_s,
                "velocity_wire_code": profile.velocity_wire_code,
                "drive_limit_ratio": profile.torque_ratio,
            },
            "applied": {
                "motor_velocity_deg_s": math.degrees(velocity / 100),
                "velocity_wire_code": velocity,
                "drive_limit_ratio": drive / 10000,
            },
            "encoded": {"velocity_wire_code": velocity, "drive_wire_code": drive},
            "device_mapping": list(self._mapping_values()) if len(self._mapping) >= 3 else None,
        }

    def _audit_io(self, kind: str, data: bytes, timestamp_ns: int) -> int:
        """Bound unconsumed transport evidence; overflow latches instead of evicting."""
        if len(self._io_events) >= self.config.io_event_capacity:
            self._io_overflow = True
            self._failure = self._failure or "B601 I/O audit overflow"
            raise RuntimeError(self._failure)
        self._io_event_sequence += 1
        self._io_events.append(
            {
                "sequence": self._io_event_sequence,
                "kind": kind,
                "timestamp_ns": timestamp_ns,
                "data_hex": data.hex(),
            }
        )
        return self._io_event_sequence

    def take_io_evidence(self) -> dict:
        """Transfer detached audit events to the owner trace without I/O or serialization."""
        self.bind_io_owner()
        events, self._io_events = self._io_events, []
        return {
            "events": events,
            "complete": not self._io_overflow,
            "correlation": "host_rx_only_no_request_token",
            "hardware_ack": "unknown",
        }

    def _query_attempt(self, payload: bytes) -> None:
        self._audit_io("query_attempt", payload, time.perf_counter_ns())

    def _receive_chunk(self, data: bytes, received_ns: int) -> int:
        return self._audit_io("query_rx", data, received_ns)

    def _rx_session_signature(self) -> tuple:
        calibration = self.calibration
        return super()._rx_session_signature() + (
            None
            if calibration is None
            else (
                calibration.path,
                calibration.sha256,
                calibration.mtime_ns,
                calibration.minimum_deg,
                calibration.maximum_deg,
            ),
            tuple(
                (name, item.mode, item.minimum_deg, item.maximum_deg)
                for name, item in sorted(self._profiles.items())
            ),
        )

    def _prepare_query(self, deadline_ns: int) -> None:
        """Audit queued RX without making it current feedback or silently purging it.

        Complete ordinary feedback can precede a new query after a command.
        Retain it as unattributed historical traffic. Fragments, faults, unknown
        IDs and continuous traffic fail closed. Same-ID late replies after a
        new query remain indistinguishable without a device transaction token.
        """
        if self._rx_pending or self._rx_poison is not None:
            raise RuntimeError("B601 prior RX remains pending; no new query is allowed")
        context = self._new_rx_context("pre_query_unattributed_rx", deadline_ns)
        received_bytes = 0
        by_id = {spec.feedback_id: spec for spec in self.joint_specs}
        while self._serial.in_waiting:
            self._remaining(deadline_ns)
            if received_bytes + 16 > self.config.prequery_byte_limit:
                raise RuntimeError("B601 pre-query RX exceeds its bounded byte limit")
            self._read_rx_chunk(
                context,
                deadline_ns,
                lambda data, received_ns: self._audit_io("pre_query_unattributed_rx", data, received_ns),
            )
            if len(self._rx_pending) < 16:
                continue
            raw, _ = self._peek_rx_frame(context, deadline_ns)
            spec = by_id.get(int.from_bytes(raw[3:7], "little"))
            if raw[:3] != b"\xaa\x11\x08" or raw[15] != 0x55 or spec is None or raw[7] & 15 != spec.motor_id:
                raise RuntimeError("B601 pre-query RX has invalid framing or joint IDs")
            if raw[7:10] in (bytes((spec.motor_id, 0, 0x33)), bytes((spec.motor_id, 0, 0x55))) or raw[
                7
            ] >> 4 not in (0, 1):
                raise RuntimeError("B601 pre-query RX reports a fault or ambiguous register reply")
            self._finish_rx_frame()
            received_bytes += 16
        if self._rx_pending:
            raise RuntimeError("B601 pre-query RX contains an incomplete frame")

    def bind_io_owner(self) -> None:
        """Bind once, including before connection; never transfer a driver object."""
        # Reject forked copies before touching a possibly inherited locked mutex.
        if self._owner_pid is not None and self._owner_pid != os.getpid():
            raise RuntimeError("B601 control belongs to another owner thread or process")
        with self._control_owner_lock:
            if self._owner_thread is None:
                self._owner_thread = threading.current_thread()
                self._owner_pid = os.getpid()
            if self._owner_thread is not threading.current_thread():
                raise RuntimeError("B601 control belongs to another owner thread or process")

    @property
    def gripper_only(self) -> bool:
        """Select only the exact addressed FORCE_POS installation."""
        return (
            len(self.joint_specs) == 1
            and self.joint_specs[0].name == "gripper"
            and self.joint_specs[0].motor_id == 7
            and self.joint_specs[0].feedback_id == 0x17
            and tuple(p.name for p in self.config.profiles) == ("gripper",)
            and self.config.profiles[0].mode == "force_pos"
            and self.config.fault_action == "disable_gripper"
        )

    def _require_scope(self) -> None:
        self.config.validate()
        if {p.name: p for p in self.config.profiles} != self._profiles or {
            s.name: s for s in self.joint_specs
        } != self._specs:
            raise ValueError("B601 command scope changed; create a fresh backend")
        if self.config.fault_action is None:
            raise RuntimeError("B601 fault policy is unresolved")
        if not self.gripper_only:
            require_b601_control_acceptance()
            return
        if self.calibration is None:
            if self.config.calibration_file is None:
                raise ValueError("B601 gripper control requires calibration_file")
            self.calibration = B601GripperCalibration.load(self.config.calibration_file)
        profile = self.config.profiles[0]
        if (
            not self.calibration.minimum_deg
            <= profile.minimum_deg
            < profile.maximum_deg
            <= self.calibration.maximum_deg
        ):
            raise ValueError("Gripper command bounds must remain inside the calibration file")
        # Relative input is valid only when it resolves to the pinned source.
        if (
            self.config.calibration_file is None
            or str(Path(self.config.calibration_file).expanduser().resolve()) != self.calibration.path
        ):
            raise ValueError("Calibration source changed; create a fresh backend")

    def preflight_control(self) -> None:
        """Validate calibrated gripper scope; retain the arm acceptance gate."""
        self.config.validate()
        self._require_scope()

    def open(self) -> None:
        """Protect even direct transport open against bypassing control preflight."""
        self.preflight_control()
        if self.calibration is not None:
            self.calibration.require_unchanged()
        self.bind_io_owner()
        super().open()
        self._control_session = True
        self._motion_session = not self.gripper_only
        self._may_be_active = self._motion_session

    def connect_observe_only(self) -> None:
        """Open the same transport for explicit queries only, with motion latched off."""
        self.bind_io_owner()
        self._observe_session = True
        super().open()

    def connect(self) -> None:
        """Read session mapping/modes on one owner; never run legacy configuration."""
        self.open()
        deadline = time.perf_counter_ns() + int(self.config.startup_budget_s * 1e9)
        mapping = self.read_mapping(max_read_duration_s=self._remaining(deadline))
        if not mapping.is_valid:
            raise RuntimeError(f"B601 mapping initialization failed: {mapping.error}")
        if not self.gripper_only:
            self._read_modes(deadline)
        snapshot = self.read_joint_snapshot(tuple(self._specs), deadline_ns=deadline)
        if not snapshot.is_valid:
            raise RuntimeError(f"B601 initial feedback invalid: {snapshot.error}")
        if self.gripper_only:
            self._require_disabled(snapshot)

    def _read_modes(self, deadline_ns: int) -> None:
        """Verify CTRL_MODE via the same transport; retain partial and raw evidence."""
        queries, chunks, evidence = [], [], []
        modes = {"mit": 1, "pos_vel": 2, "force_pos": 4}
        try:
            for spec in self.joint_specs:

                def consume(raw, arbitration_id, received_ns, spec=spec):
                    payload = raw[7:15]
                    if arbitration_id != spec.feedback_id or payload[:4] != bytes(
                        (spec.motor_id, 0, 0x33, 10)
                    ):
                        raise RuntimeError("B601 control mode reply ID/register mismatch")
                    value = struct.unpack("<I", payload[4:])[0]
                    evidence.append(
                        {
                            "name": spec.name,
                            "mode": value,
                            "receive_timestamp_ns": received_ns,
                            "packet_hex": raw.hex(),
                        }
                    )
                    if value != modes[self._profiles[spec.name].mode]:
                        raise RuntimeError(f"B601 mode mismatch for {spec.name}; no automatic mode write")
                    previous_mode = self._modes.get(spec.name)
                    self._modes[spec.name] = value

                    def rollback():
                        evidence.pop()
                        if previous_mode is None:
                            self._modes.pop(spec.name)
                        else:
                            self._modes[spec.name] = previous_mode

                    return rollback

                self._exchange((encode_mode_query(spec.motor_id),), deadline_ns, queries, chunks, consume)
        except BaseException as exc:
            self._failure = self._failure or f"mode_read_failed: {type(exc).__name__}: {exc}"
            raise
        finally:
            self.mode_evidence = tuple(evidence)
            self.mode_queries = tuple(queries)
            self.mode_receive_chunks = tuple(chunks)

    def _mapping_values(self) -> tuple[float, float, float]:
        return tuple(self._mapping[("gripper", rid)].value for rid in (21, 22, 23))

    def _check_gripper_position(self, position_deg: float) -> None:
        if self.calibration is None:
            raise ValueError("Gripper calibration is missing")
        self.calibration.check_travel(position_deg, 0)
        profile = self._profiles["gripper"]
        if not profile.minimum_deg <= position_deg <= profile.maximum_deg:
            raise RuntimeError("Measured gripper position is outside the configured limits")

    def _require_disabled(self, snapshot: JointSnapshot) -> None:
        if snapshot is not self._latest_snapshot or not snapshot.is_valid:
            raise ValueError("Use the latest valid gripper snapshot from this session")
        feedback = snapshot.get("gripper")
        now = time.perf_counter_ns()
        if not 0 <= now - feedback.receive_timestamp_ns <= self.config.feedback_max_age_s * 1e9:
            raise ValueError("Fresh disabled gripper feedback is required")
        self._check_gripper_position(feedback.position_deg)
        if self.last_batch is None or self.last_batch.feedback[0].status_code != 0:
            raise RuntimeError("Gripper startup requires DISABLED feedback")

    def _disabled_snapshot(self, deadline_ns: int) -> JointSnapshot:
        snapshot = self.read_joint_snapshot(("gripper",), deadline_ns=deadline_ns)
        self._require_disabled(snapshot)
        self.startup_receipt["poses"].append(
            {
                "position_deg": snapshot.get("gripper").position_deg,
                "receive_timestamp_ns": snapshot.get("gripper").receive_timestamp_ns,
                "status_code": 0,
            }
        )
        return snapshot

    def _read_gripper_register(self, register_id: int, deadline_ns: int) -> dict:
        if register_id not in (9, 10):
            raise ValueError("Only TIMEOUT and CTRL_MODE startup registers are permitted")
        rows = []

        def consume(raw, arbitration_id, received_ns):
            if arbitration_id != 0x17 or raw[7:11] != bytes((7, 0, 0x33, register_id)):
                raise RuntimeError("Gripper startup register response ID/register mismatch")
            rows.append(
                {
                    "register_id": register_id,
                    "value": struct.unpack("<I", raw[11:15])[0],
                    "receive_timestamp_ns": received_ns,
                    "packet_hex": raw.hex(),
                }
            )
            return rows.pop

        query = encode_mode_query(7) if register_id == 10 else encode_timeout_query(7)
        self._exchange((query,), deadline_ns, [], [], consume)
        return rows[0]

    def _capture_parameter_rx(self, register_id: int, deadline_ns: int) -> None:
        """Audit optional configuration echoes separately from explicit readback."""
        end = min(deadline_ns, time.perf_counter_ns() + int(self.config.parameter_write_settle_s * 1e9))
        if self._rx_pending or self._rx_poison is not None:
            raise RuntimeError("B601 prior RX remains pending before parameter echo")
        context = self._new_rx_context("parameter_write_unattributed_rx", deadline_ns)
        received_bytes = 0
        while time.perf_counter_ns() < end or self._serial.in_waiting:
            self._remaining(deadline_ns)
            if received_bytes + 16 > self.config.prequery_byte_limit:
                raise RuntimeError("Parameter-write RX exceeds the configured byte limit")
            self._read_rx_chunk(
                context,
                deadline_ns,
                lambda data, received_ns: self._audit_io(
                    "parameter_write_unattributed_rx", data, received_ns
                ),
            )
            if len(self._rx_pending) < 16:
                continue
            raw, _ = self._peek_rx_frame(context, deadline_ns)
            if raw[:7] != b"\xaa\x11\x08\x17\x00\x00\x00" or raw[15] != 0x55:
                raise RuntimeError("Parameter-write RX has invalid framing or ID")
            matching_echo = raw[7:11] in (bytes((7, 0, 0x33, register_id)), bytes((7, 0, 0x55, register_id)))
            if not matching_echo and (raw[7] != 7 or raw[9] in (0x33, 0x55)):
                raise RuntimeError("Parameter-write RX is not disabled feedback or a matching echo")
            self._finish_rx_frame()
            received_bytes += 16
        if self._rx_pending:
            raise RuntimeError("Parameter-write RX contains a partial frame")

    def _receive_gripper_feedback(
        self,
        deadline_ns: int,
        expected_status: int,
        *,
        fault_attempt: bool = False,
        intermediate_feedback: list[dict] | None = None,
        packet_hex: str | None = None,
    ) -> dict:
        """Observe addressed feedback within one deadline; it is never a command ACK.

        A fault disable may be followed first by valid ENABLED feedback from a
        previous query. Keep those bounded observations and continue waiting
        for DISABLED without retransmitting. All other command/status checks
        remain strict, including ordinary position submissions.
        """
        self._last_command_feedback = None
        intermediate = [] if intermediate_feedback is None else intermediate_feedback
        skipped_bytes = 0
        context = self._new_rx_context(
            "fault_disable_feedback" if fault_attempt else "command_feedback", deadline_ns, packet_hex
        )

        def receive(data, received_ns):
            try:
                return self._audit_io("command_rx", data, received_ns)
            except RuntimeError:
                if not fault_attempt or not self._io_overflow:
                    raise
                return None

        while True:
            while len(self._rx_pending) < 16:
                self._read_rx_chunk(context, deadline_ns, receive)
            raw, evidence = self._peek_rx_frame(context, deadline_ns, allow_inherited=fault_attempt)
            if raw[:7] != b"\xaa\x11\x08\x17\x00\x00\x00" or raw[15] != 0x55:
                raise RuntimeError("Gripper command response framing/ID mismatch")
            payload = raw[7:15]
            if payload[0] & 15 != 7 or payload[:3] in (b"\x07\x00\x33", b"\x07\x00\x55"):
                raise RuntimeError("Gripper command response is not feedback")
            pmax, vmax, _ = self._mapping_values()
            row = {
                "position_deg": math.degrees(int.from_bytes(payload[1:3], "big") * 2 * pmax / 65535 - pmax),
                "velocity_rad_s": ((payload[3] << 4) | (payload[4] >> 4)) * 2 * vmax / 4095 - vmax,
                "status_code": payload[0] >> 4,
                "receive_timestamp_ns": evidence["last_receive_timestamp_ns"],
                "packet_hex": raw.hex(),
                "time_source": "host_receive_no_device_request_token",
                "rx_context": evidence,
            }
            if evidence["inherited"]:
                # Its first bytes came from an earlier (possibly timed-out)
                # operation. Even DISABLED is historical evidence, never this
                # disable's confirmation or a new protocol sequence.
                if row["status_code"] not in (0, 1):
                    raise RuntimeError("Unexpected gripper drive status after command")
                self._check_gripper_position(row["position_deg"])
                self._remaining(deadline_ns)
                skipped_bytes += 16
                if skipped_bytes > self.config.prequery_byte_limit:
                    raise RuntimeError("Fault intermediate feedback exceeds its bounded byte limit")
                row["historical_only"] = True
                intermediate.append(row)
                self._finish_rx_frame()
                continue
            self._last_command_feedback = row
            if row["status_code"] == expected_status:
                self._check_gripper_position(row["position_deg"])
                self._remaining(deadline_ns)
                self._finish_rx_frame()
                return row
            if fault_attempt and expected_status == 0 and row["status_code"] == 1:
                self._check_gripper_position(row["position_deg"])
                self._remaining(deadline_ns)
                skipped_bytes += 16
                if skipped_bytes > self.config.prequery_byte_limit:
                    raise RuntimeError("Fault intermediate feedback exceeds its bounded byte limit")
                intermediate.append(row)
                self._finish_rx_frame()
                continue
            raise RuntimeError("Unexpected gripper drive status after command")

    def _startup_command(
        self, stage: str, packet: bytes, expected_status: int, deadline_ns: int, guard: Callable[[], None]
    ) -> None:
        self._prepare_query(deadline_ns)
        self._guard_startup(guard)
        item = {"stage": stage, "packet_hex": packet.hex(), "status": "unknown", "hardware_ack": "unknown"}
        self.startup_receipt["commands"].append(item)
        item["send_start_ns"] = time.perf_counter_ns()
        try:
            self._write_once(
                packet, min(deadline_ns, item["send_start_ns"] + int(self.config.send_budget_s * 1e9))
            )
            item["status"] = "submitted"
        finally:
            item["send_end_ns"] = time.perf_counter_ns()
        item["feedback"] = self._receive_gripper_feedback(
            deadline_ns, expected_status, packet_hex=packet.hex()
        )

    def _guard_startup(self, guard: Callable[[], None]) -> None:
        self.calibration.require_unchanged()
        guard()

    def start_gripper(self, reference_snapshot: JointSnapshot, guard: Callable[[], None]) -> float:
        """Prepare and enable only ID 7, with a fresh guard before every write."""
        self.bind_io_owner()
        self._require_scope()
        if (
            not self.gripper_only
            or self._observe_session
            or not self._control_session
            or self._gripper_start_attempted
        ):
            raise RuntimeError("Gripper startup requires a fresh disabled control session")
        if not callable(guard):
            raise TypeError("Gripper startup requires a force/stop/calibration guard")
        self._require_disabled(reference_snapshot)
        self.calibration.require_unchanged()
        self._gripper_start_attempted = self._gripper_starting = True
        receipt = self.startup_receipt = {
            "original_registers": [],
            "writes": [],
            "final_registers": [],
            "poses": [],
            "commands": [],
            "watchdog_ticks": self.config.watchdog_ticks,
            "parameter_write_settle_s": self.config.parameter_write_settle_s,
            "complete": False,
            "hardware_ack": "unknown",
            "error": None,
            "command_settings": self.command_settings,
        }
        deadline = time.perf_counter_ns() + int(self.config.startup_budget_s * 1e9)
        try:
            self._guard_startup(guard)
            self._disabled_snapshot(deadline)
            # Validate fresh device mapping and wire settings before any configuration write.
            position = reference_snapshot.get("gripper").position_deg
            encode_position(7, position, position, self._profiles["gripper"], self._mapping_values())
            original = {}
            for rid in (10, 9):
                row = self._read_gripper_register(rid, deadline)
                receipt["original_registers"].append(row)
                original[rid] = row["value"]
            # damiao.tables.ControlMode defines VEL=3. A disabled motor may
            # arrive from that mode; startup still selects only FORCE_POS=4.
            if original[10] not in (1, 2, 3, 4):
                raise RuntimeError(f"Unsupported initial CTRL_MODE={original[10]}")
            for rid, value, packet in (
                (10, 4, encode_mode_write(7, "force_pos")),
                (9, self.config.watchdog_ticks, encode_timeout_write(7, self.config.watchdog_ticks)),
            ):
                current = (
                    self._read_gripper_register(rid, deadline) if rid == 9 and receipt["writes"] else None
                )
                if (original[rid] if current is None else current["value"]) == value:
                    continue
                self._disabled_snapshot(deadline)
                self._prepare_query(deadline)
                self._guard_startup(guard)
                item = {
                    "register_id": rid,
                    "requested_value": value,
                    "packet_hex": packet.hex(),
                    "status": "unknown",
                }
                receipt["writes"].append(item)
                item["send_start_ns"] = time.perf_counter_ns()
                try:
                    self._write_once(
                        packet, min(deadline, item["send_start_ns"] + int(self.config.send_budget_s * 1e9))
                    )
                    item["status"] = "submitted"
                finally:
                    item["send_end_ns"] = time.perf_counter_ns()
                self._capture_parameter_rx(rid, deadline)
                item["readback"] = self._read_gripper_register(rid, deadline)
                if item["readback"]["value"] != value:
                    raise RuntimeError("Gripper parameter readback differs from requested value")
                self._disabled_snapshot(deadline)
            for rid, expected in ((10, 4), (9, self.config.watchdog_ticks)):
                row = self._read_gripper_register(rid, deadline)
                receipt["final_registers"].append(row)
                if row["value"] != expected:
                    raise RuntimeError("Gripper final startup register check failed")
            self._modes["gripper"] = 4
            self._startup_command("disabled_exit_probe", encode_disable(7), 0, deadline, guard)
            fresh = self._disabled_snapshot(deadline).get("gripper").position_deg
            target = encode_position(7, fresh, fresh, self._profiles["gripper"], self._mapping_values())
            self._startup_command("disabled_target_preload", target.packet, 0, deadline, guard)
            self._startup_command("enable_only", encode_enable(7), 1, deadline, guard)
            self._startup_command("enabled_initial_target", target.packet, 1, deadline, guard)
            self._motion_session = True
            receipt["wire_position_deg"] = target.wire_position_deg
            receipt["complete"] = True
            return target.wire_position_deg
        except BaseException as exc:
            self._failure = self._failure or f"gripper_start_failed: {type(exc).__name__}: {exc}"
            receipt["error"] = self._failure
            self.emergency_stop(self._failure)
            raise
        finally:
            self._gripper_starting = False

    def read_joint_snapshot(self, names: Sequence[str], *, deadline_ns: int) -> JointSnapshot:
        """Adapt actual host RX to the runtime without caching or restamping positions."""
        batch = self.read_feedback(
            names=tuple(names),
            deadline_ns=deadline_ns,
            max_read_duration_s=self.config.feedback_max_age_s,
        )
        incoming = {item.name: item for item in batch.feedback}
        joints = []
        error = batch.error
        for name in names:
            item = incoming.get(name)
            joint_error = error
            if item is None:
                joints.append(
                    JointFeedback(name, None, False, None, None, None, error=error or "missing_feedback")
                )
                continue
            age = batch.completed_ns - item.receive_timestamp_ns
            if age < 0 or age > self.config.feedback_max_age_s * 1e9:
                joint_error = joint_error or "feedback_age_invalid"
            if self._motion_session and item.status_code != 1:
                joint_error = joint_error or "motor_not_enabled"
            error = error or joint_error
            joints.append(
                JointFeedback(
                    name,
                    item.position_deg,
                    joint_error is None,
                    item.receive_timestamp_ns,
                    item.receive_sequence,
                    age,
                    item.timestamp_source,
                    joint_error,
                )
            )
        if error is not None:
            self._failure = self._failure or error
        # Runtime compares the caller's exact deadline. The diagnostic batch
        # separately retains the possibly tighter transport/age budget.
        result = JointSnapshot(tuple(joints), batch.completed_ns, deadline_ns, error)
        self._latest_snapshot = result
        return result

    def _require_motion(self) -> None:
        """Apply ownership, scope, activation and fault gates at every write."""
        self._check_owner()
        self._require_scope()
        if self._observe_session or not self._motion_session:
            raise RuntimeError("Observe-only B601 session forbids control writes")
        if self._failure is not None:
            raise RuntimeError(f"B601 control fault is latched: {self._failure}")
        if not self.mapping_ready or len(self._modes) != len(self.joint_specs):
            raise RuntimeError("B601 session mapping/mode verification is incomplete")

    def _write_once(self, packet: bytes, deadline_ns: int, *, fault_attempt: bool = False) -> None:
        """Bound one write and reject late/partial returns; never flush or retry."""
        if len(packet) != 30 or packet[:4] != b"\x55\xaa\x1e\x03":
            raise ValueError("B601 bridge writes require exactly one 30-byte frame")
        self._check_owner()
        identifier = int.from_bytes(packet[13:17], "little")
        payload = packet[21:29]
        query = identifier == 0x7FF and payload[2] in (0x33, 0xCC)
        motor_id = payload[0] if identifier == 0x7FF else identifier & 15
        if motor_id not in {spec.motor_id for spec in self.joint_specs}:
            raise ValueError("Command ID is outside the configured B601 session")
        if not query:
            self._require_scope()
            if not fault_attempt and self.calibration is not None:
                self.calibration.require_unchanged()
            if self._observe_session or not self._control_session:
                raise RuntimeError("Observe-only B601 session forbids control writes")
            if not fault_attempt and not (self._motion_session or self._gripper_starting):
                raise RuntimeError("Gripper must complete explicit startup before motion")
            if self._failure is not None and not fault_attempt:
                raise RuntimeError(f"B601 control fault is latched: {self._failure}")
            if packet != encode_disable(motor_id):
                self._may_be_active = True
        # Windows serial setters reconfigure the port. Reuse an already bounded
        # timeout; floor new values to OS milliseconds rather than changing a
        # slightly different float on every packet. Never extend the deadline.
        remaining = self._remaining(deadline_ns)
        timeout = math.floor(remaining * 1000) / 1000
        if timeout < 0.001:
            raise TimeoutError("B601 write has less than one millisecond remaining")
        current = self._serial.write_timeout
        if current is not None and 0 < current <= timeout:
            timeout = current
        if self._serial.write_timeout != timeout:
            self._serial.write_timeout = timeout
        self._remaining(deadline_ns)
        try:
            self._audit_io("command_attempt", packet, time.perf_counter_ns())
        except RuntimeError:
            # A full audit queue must not prevent the restricted fault attempt.
            # Its packet/status remain in the separately retained fault receipt;
            # the run is already incomplete and cannot become training evidence.
            if not fault_attempt or not self._io_overflow:
                raise
        if self._serial.write(packet) != len(packet):
            raise OSError("B601 command write was partial; execution unknown")
        self._remaining(deadline_ns)

    def send_joint_positions_partial(
        self, goals_deg: Mapping[str, float], *, feedback_snapshot: JointSnapshot
    ) -> SendReceipt:
        """Prevalidate every target, then report sequential, non-atomic submissions."""
        self._require_motion()
        goals = dict(goals_deg)
        if not goals or set(goals) - self._specs.keys():
            raise ValueError("Partial commands require known bare joint names")
        if feedback_snapshot is not self._latest_snapshot or not feedback_snapshot.is_valid:
            raise ValueError("Use the latest valid snapshot from this same session")
        encoded = {}
        now = time.perf_counter_ns()
        for name, candidate in goals.items():
            feedback = feedback_snapshot.get(name)
            if (
                not feedback.is_valid
                or not 0 <= now - feedback.receive_timestamp_ns <= self.config.feedback_max_age_s * 1e9
            ):
                raise ValueError(f"Fresh feedback is required for {name}")
            mapping = tuple(self._mapping[(name, rid)].value for rid in (21, 22, 23))
            encoded[name] = encode_position(
                self._specs[name].motor_id, candidate, feedback.position_deg, self._profiles[name], mapping
            )
        started = time.perf_counter_ns()
        deadline = started + int(self.config.send_budget_s * 1e9)
        results, failed, interrupted = [], False, None
        for name, candidate in goals.items():
            target = encoded[name]
            begin = end = None
            status, error = "not_attempted", "prior_submission_failed" if failed else None
            if not failed:
                try:
                    # Earlier sequential writes may consume this joint's freshness.
                    now = time.perf_counter_ns()
                    if (
                        now >= deadline
                        or not 0
                        <= now - feedback_snapshot.get(name).receive_timestamp_ns
                        <= self.config.feedback_max_age_s * 1e9
                    ):
                        raise TimeoutError("B601 deadline/feedback expired before joint submission")
                    begin = time.perf_counter_ns()
                    status = "unknown"
                    self._write_once(target.packet, deadline)
                    if self.gripper_only:
                        self._receive_gripper_feedback(
                            min(deadline, time.perf_counter_ns() + int(self.config.feedback_max_age_s * 1e9)),
                            1,
                            packet_hex=target.packet.hex(),
                        )
                    status = "success"
                except BaseException as exc:
                    failed = True
                    self._failure = error = f"{name}: {type(exc).__name__}: {exc}"
                    if not isinstance(exc, Exception):
                        interrupted = exc
                finally:
                    end = time.perf_counter_ns() if begin is not None else None
            results.append(
                JointSendResult(
                    name,
                    candidate,
                    target.clipped_deg,
                    target.wire_position_deg if status == "success" else None,
                    status,
                    begin,
                    end,
                    error=error,
                )
            )
        receipt = SendReceipt(tuple(results), started, time.perf_counter_ns())
        self.last_receipt = receipt
        if interrupted is not None:
            raise interrupted
        return receipt

    def send_gripper_position(self, position_deg: float, *, feedback_snapshot: JointSnapshot) -> SendReceipt:
        """Submit only gripper; never synthesize wrist_yaw or other arm targets."""
        return self.send_joint_positions_partial(
            {"gripper": position_deg}, feedback_snapshot=feedback_snapshot
        )

    def emergency_stop(self, reason: str) -> dict:
        """Latch normal control and record one bounded fault attempt, without claiming a stop.

        Disable candidates require explicit session coverage, behind the fixed
        hardware gate. An unresolved policy sends no packet.
        No stale-position hold, arm disable, error clear or retry is inferred.
        """
        self.bind_io_owner()
        if self._fault_receipt is not None:
            return copy.deepcopy(self._fault_receipt)
        self._failure = self._failure or reason
        receipt = {
            "action": self.config.fault_action or "unresolved",
            "status": "not_attempted",
            "reason": reason,
            "hardware_ack": "unknown",
            "hardware_stop_confirmed": False,
            "send_start_ns": None,
            "send_end_ns": None,
            "error": None,
            "per_joint_results": [],
        }
        interrupted = None
        try:
            self._require_scope()
            if (
                not (self._motion_session or self._may_be_active)
                or self._observe_session
                or not self.has_resources
            ):
                raise RuntimeError("No active motion session for fault submission")
            if self.config.fault_action is None:
                raise RuntimeError("Fault policy unresolved for these selected joints")
            self.config.validate()
            self._check_owner()
            receipt["send_start_ns"] = time.perf_counter_ns()
            deadline = receipt["send_start_ns"] + int(self.config.fault_budget_s * 1e9)
            names = (
                ("gripper",)
                if self.config.fault_action == "disable_gripper"
                else self.config.fault_joint_names
            )
            failed = False
            for name in names:
                packet = encode_disable(self._specs[name].motor_id)
                item = {
                    "name": name,
                    "packet_hex": packet.hex(),
                    "status": "not_attempted",
                    "send_start_ns": None,
                    "send_end_ns": None,
                    "hardware_ack": "unknown",
                    "error": None,
                }
                receipt["per_joint_results"].append(item)
                if failed:
                    item["error"] = "prior_fault_submission_failed"
                    continue
                try:
                    self._remaining(deadline)
                    item["send_start_ns"] = time.perf_counter_ns()
                    item["status"] = "unknown"
                    self._last_command_feedback = None
                    self._write_once(packet, deadline, fault_attempt=True)
                    item["status"] = "submitted"
                    if self.gripper_only:
                        item["intermediate_feedback"] = []
                        item["intermediate_feedback_byte_limit"] = self.config.prequery_byte_limit
                        item["receive_budget_s"] = self.config.feedback_max_age_s
                        item["receive_deadline_ns"] = time.perf_counter_ns() + int(
                            self.config.feedback_max_age_s * 1e9
                        )
                        item["feedback"] = self._receive_gripper_feedback(
                            item["receive_deadline_ns"],
                            0,
                            fault_attempt=True,
                            intermediate_feedback=item["intermediate_feedback"],
                            packet_hex=packet.hex(),
                        )
                        receipt["disabled_feedback_observed"] = True
                except BaseException as exc:
                    failed = True
                    item["error"] = receipt["error"] = f"{type(exc).__name__}: {exc}"
                    if self.gripper_only and self._last_command_feedback is not None:
                        item["feedback"] = copy.deepcopy(self._last_command_feedback)
                        receipt["disabled_feedback_observed"] = item["feedback"]["status_code"] == 0
                    if not isinstance(exc, Exception):
                        interrupted = exc
                finally:
                    if item["send_start_ns"] is not None:
                        item["send_end_ns"] = time.perf_counter_ns()
            receipt["status"] = (
                "submitted"
                if not failed
                else "unknown"
                if any(item["status"] != "not_attempted" for item in receipt["per_joint_results"])
                else "not_attempted"
            )
            if len(names) == 1:
                receipt["packet_hex"] = receipt["per_joint_results"][0]["packet_hex"]
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            if receipt["send_start_ns"] is not None:
                receipt["send_end_ns"] = time.perf_counter_ns()
            self._fault_receipt = receipt
        if interrupted is not None:
            raise interrupted
        return copy.deepcopy(receipt)

    def fault_stop_gripper(self, *, reason: str) -> dict:
        """Expose the same restricted fault channel to narrow-interface wrappers."""
        return self.emergency_stop(reason)

    def disconnect(self) -> None:
        """Attempt the selected disposition once, close, and retain any unconfirmed result."""
        self.bind_io_owner()
        if not self.has_resources:
            return
        receipt = None
        errors = []
        try:
            if self._motion_session or self._may_be_active:
                receipt = self.emergency_stop("owner_disconnect")
        except BaseException as exc:
            errors.append(exc)
        if receipt is not None and (receipt["status"] != "submitted" or receipt["error"] is not None):
            errors.append(
                RuntimeError(f"B601 fault disposition unconfirmed at disconnect: {receipt['error']}")
            )
        # A failed close leaves the serial handle, lease and motion-session
        # state intact. A second cleanup attempts close, never another send.
        try:
            super().close()
            self._motion_session = False
            self._control_session = False
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("B601 fault disposition and close failed", errors)

    def close(self) -> None:
        """Keep direct close and runtime disconnect on the same cleanup path."""
        self.disconnect()
