"""Internal bounded full-arm lifecycle on one existing B601ControlBackend owner."""

import copy
import math
import os
import time
from dataclasses import asdict, replace

from lerobot.force_control.hardware_types import JointFeedback, JointSendResult, JointSnapshot, SendReceipt

from .calibration_reader import _CalibrationReader
from .control_protocol import (
    encode_disable,
    encode_enable,
    encode_mode_write,
    encode_position,
    encode_timeout_write,
    mit_gain_wire_codes,
)
from .control_state import read_control_state
from .full_arm_control_types import JOINT_NAMES, FullArmLeaderSample
from .readonly_transport import ReadonlyB601Transport


class FullArmValidationSession:
    """One guarded all-joint lifecycle; diagnostic and operator admission remain separate."""

    def __init__(self, device, configuration, admission, *, operator=False):
        # Equality with these validated deep copies rejects later configuration drift.
        device.config.validate()
        configuration.validate()
        self.device, self.configuration, self.admission = device, configuration, admission
        self.operator = operator
        self.kind = "operator_teleoperation" if operator else "bounded_full_arm_validation"
        self.pinned_control = copy.deepcopy(device.config)
        self.pinned_specs = tuple(device.joint_specs)
        self.pinned_by_name = {spec.name: spec for spec in self.pinned_specs}
        self.cleaning = False
        self.configuration_snapshot = asdict(configuration)
        self.pinned_configuration = copy.deepcopy(configuration)
        self.calibration_identity = next(
            pin for pin in admission.file_identities if pin.path == admission.calibration.path
        )
        self.enabled = set()
        self.activation_risk = False
        self.start_attempted = self.starting = self.ready = False
        self.started_ns = None
        self.references = self.hold_targets = None
        self.guard = self.leader_provider = None
        self.leader_epoch = self.leader_sequence = self.leader_capture = None
        self.last_command_ns = {}
        self.allowed_packet = self.pending_response = None
        self.pending_parameter_write = None
        self.watchdogs = {}
        self.observed_statuses = {}
        self.returned_sequences = {}
        self.last_observation = None
        self._calibration_reader = None

    @property
    def has_binding_resources(self) -> bool:
        """Keep the calibration handle in the owner's cleanup accounting."""
        return self._calibration_reader is not None and self._calibration_reader.has_resources

    def open_binding(self) -> None:
        """Prepare a verified read-only handle before transport startup."""
        if self._calibration_reader is not None:
            raise RuntimeError("Calibration binding is single-use; create a fresh session")
        self._calibration_reader = _CalibrationReader(self.calibration_identity)
        self._calibration_reader.open()

    def close_binding(self) -> None:
        """Close the file independently of motor fault-disposition success."""
        if self._calibration_reader is not None:
            self._calibration_reader.close()

    def require_binding(self, *, cleanup=False):
        """Keep cleanup independent of changed files while every normal write revalidates them."""
        if cleanup:
            return
        capability = self.device._operator_teleoperation if self.operator else self.device._validation
        if capability is not self.configuration:
            raise RuntimeError("Full-arm validation capability was replaced during its session")
        if self.device.config != self.pinned_control:
            raise RuntimeError("Full-arm validation command configuration/budgets changed during its session")
        if self.configuration != self.pinned_configuration:
            raise RuntimeError("Full-arm validation capability changed during its session")
        # Content reads still check path identity before and after every read.
        # Reuse only the OS handle; never cache calibration bytes or their hash.
        for pin in self.admission.file_identities:
            if pin is not self.calibration_identity:
                pin.require_unchanged()
        if self._calibration_reader is None:
            self.calibration_identity.require_contents_unchanged()
        else:
            self._calibration_reader.require_contents_unchanged()

    def leader(self):
        """Read a cache only and refuse mapping/epoch/sequence changes."""
        sample = self.leader_provider()
        if not isinstance(sample, FullArmLeaderSample):
            raise TypeError("Full-arm validation requires a typed immutable leader cache sample")
        sample.validate(self.configuration)
        if self.leader_epoch is not None and sample.epoch != self.leader_epoch:
            raise RuntimeError("Full-arm validation leader epoch changed")
        if self.leader_sequence is not None and (
            sample.sequence < self.leader_sequence or sample.captured_ns < self.leader_capture
        ):
            raise RuntimeError("Full-arm validation leader source moved backwards")
        self.leader_epoch, self.leader_sequence, self.leader_capture = (
            sample.epoch,
            sample.sequence,
            sample.captured_ns,
        )
        return sample

    def wait_guard(self):
        """Check live stop/force/leader, calibration identity and bounded activity on every RX chunk."""
        if self.guard is None:
            return
        if os.stat(self.calibration_identity.path).st_mtime_ns != self.admission.calibration.mtime_ns:
            raise ValueError("Full-arm validation calibration changed")
        self.guard()
        self.leader()
        now = time.perf_counter_ns()
        if (
            not self.operator
            and self.started_ns is not None
            and now - self.started_ns > self.configuration.max_duration_s * 1e9
        ):
            raise TimeoutError("Full-arm validation session duration elapsed")
        maximum_gap = self.configuration.max_command_gap_s
        if maximum_gap is not None:
            for name in self.enabled:
                if now - self.last_command_ns[name] > maximum_gap * 1e9:
                    raise TimeoutError(f"Full-arm validation command-refresh gap exceeded: {name}")
        if self.device.failure is not None:
            raise RuntimeError(f"Full-arm validation fault latched: {self.device.failure}")

    def connect(self, deadline_ns):
        """Inspect modes/timeouts and disabled state; never configure or enable drives."""
        state = read_control_state(self.device, max_read_duration_s=self.device._remaining(deadline_ns))
        if not state.is_valid:
            raise RuntimeError(f"Full-arm mode/TIMEOUT query failed: {state.error}")
        values = {(row.name, row.register_id): row.value for row in state.registers}
        for name in JOINT_NAMES:
            expected = 4 if name == "gripper" else 1 if self.operator else 2
            watchdog = self.configuration.watchdog_ticks
            mode = values.get((name, 10))
            # Operator Start selects arm MIT mode after fresh DISABLED feedback,
            # as the native follower does. Connecting still issues queries only.
            mode_matches = mode in (1, 2, 3, 4) if self.operator else mode == expected
            if not mode_matches or (not self.operator and values.get((name, 9)) != watchdog):
                raise RuntimeError(
                    f"Full-arm control requires existing mode={expected} and TIMEOUT={watchdog}: {name}"
                )
            self.device._modes[name] = mode
            self.watchdogs[name] = values[(name, 9)]
        self.read_all(deadline_ns, active=set())
        self.connection_state = state

    def check_batch(self, batch, active, *, cleanup=False):
        """Validate actual states/bounds without replacing unknown or missing feedback."""
        if (
            not batch.is_valid
            or len(batch.feedback) != 7
            or {row.name for row in batch.feedback} != set(JOINT_NAMES)
        ):
            raise RuntimeError(f"Full-arm validation all-joint feedback failed: {batch.error}")
        bounds = {name: (lower, upper) for name, _, lower, upper in self.admission.calibration.joints}
        self.position_limit_violations = []
        now = time.perf_counter_ns()
        for row in batch.feedback:
            if row.status_code != (1 if row.name in active else 0):
                raise RuntimeError(
                    f"Full-arm validation unexpected drive state: {row.name}={row.status_name}"
                )
            if not 0 <= now - row.receive_timestamp_ns <= self.pinned_control.feedback_max_age_s * 1e9:
                raise RuntimeError(f"Full-arm validation stale joint feedback: {row.name}")
            if not bounds[row.name][0] <= row.position_deg <= bounds[row.name][1] and not cleanup:
                raise RuntimeError(f"Full-arm validation measured calibration overtravel: {row.name}")
            if cleanup and (
                not bounds[row.name][0] <= row.position_deg <= bounds[row.name][1]
                or self.reference_envelope_exceeded(row.name, row.position_deg)
            ):
                self.position_limit_violations.append(
                    {
                        "name": row.name,
                        "position_deg": row.position_deg,
                        "calibration_bounds_deg": list(bounds[row.name]),
                    }
                )
            if not cleanup and self.reference_envelope_exceeded(row.name, row.position_deg):
                raise RuntimeError(f"Full-arm validation measured reference-envelope overtravel: {row.name}")
        self.observed_statuses = {row.name: row.status_code for row in batch.feedback}

    def note_returned_feedback(self, snapshot):
        """Track only observations explicitly returned by read_joint_snapshot."""
        if self.operator and snapshot.is_valid:
            self.returned_sequences.update((row.name, row.receive_sequence) for row in snapshot.joints)

    def take_operator_feedback(self, names, deadline_ns):
        """Reuse unconsumed command RX once, retaining every real receive timestamp."""
        if (
            not self.operator
            or not self.ready
            or self.cleaning
            or not self.device._motion_session
            or self.device.failure is not None
            or len(names) != 7
            or set(names) != set(JOINT_NAMES)
            or self.device._outstanding_query is not None
            or self.device._query_write_inflight is not None
            or self.pending_response is not None
            or self.pending_parameter_write is not None
            or self.device._rx_pending
            or self.device._rx_poison is not None
            or self.device._serial.in_waiting
        ):
            return None
        now = time.perf_counter_ns()
        if type(deadline_ns) is not int or deadline_ns <= now:
            return None
        snapshot = self.device._latest_snapshot
        if (
            snapshot is None
            or not snapshot.is_valid
            or len(snapshot.joints) != 7
            or {row.name for row in snapshot.joints} != set(JOINT_NAMES)
            or snapshot.completed_ns > now
        ):
            return None
        rows = []
        for name in names:
            row = snapshot.get(name)
            if (
                row.position_deg is None
                or not math.isfinite(row.position_deg)
                or type(row.receive_timestamp_ns) is not int
                or type(row.receive_sequence) is not int
                or row.receive_timestamp_ns > snapshot.completed_ns
                or row.receive_sequence <= self.returned_sequences.get(name, 0)
                or not 0 <= now - row.receive_timestamp_ns <= self.pinned_control.feedback_max_age_s * 1e9
                or self.observed_statuses.get(name) != 1
            ):
                return None
            rows.append(replace(row, age_ns=now - row.receive_timestamp_ns))
        # This is a view of actual completed RX, not a new receive operation.
        result = JointSnapshot(tuple(rows), snapshot.completed_ns, deadline_ns)
        self.device._latest_snapshot = result
        self.note_returned_feedback(result)
        return result

    def read_all(self, deadline_ns, *, active, cleanup=False):
        """Read all seven on the existing owner, preserving latched failure during restricted cleanup."""
        prior_failure, prior_strict = self.device._failure, self.device._acceptance_cleanup_strict_rx
        if cleanup:
            self.device._failure = None
            self.device._acceptance_cleanup_strict_rx = True
        try:
            batch = ReadonlyB601Transport.read_feedback(
                self.device,
                names=JOINT_NAMES,
                deadline_ns=deadline_ns,
                max_read_duration_s=self.pinned_control.feedback_max_age_s,
            )
        finally:
            if cleanup:
                self.device._failure = prior_failure or self.device._failure
                self.device._acceptance_cleanup_strict_rx = prior_strict
        self.check_batch(batch, active, cleanup=cleanup)
        self.last_observation = batch
        self.device._latest_snapshot = JointSnapshot(
            tuple(
                JointFeedback(
                    row.name,
                    row.position_deg,
                    True,
                    row.receive_timestamp_ns,
                    row.receive_sequence,
                    batch.completed_ns - row.receive_timestamp_ns,
                    row.timestamp_source,
                )
                for row in batch.feedback
            ),
            batch.completed_ns,
            deadline_ns,
        )
        return self.device._latest_snapshot

    def check_alignment(self, snapshot):
        """Record every leader/follower difference before refusing any enable."""
        now = time.perf_counter_ns()
        if (
            snapshot is None
            or not snapshot.is_valid
            or any(
                not 0 <= now - row.receive_timestamp_ns <= self.pinned_control.feedback_max_age_s * 1e9
                for row in snapshot.joints
            )
        ):
            raise RuntimeError("Full-arm validation alignment requires fresh all-seven follower feedback")
        sample = self.leader()
        goals = dict(sample.positions_deg)
        tolerance = self.configuration.alignment_tolerance_deg
        rows = [
            {
                "name": name,
                "leader_deg": goals[name],
                "follower_deg": snapshot.get(name).position_deg,
                "difference_deg": goals[name] - snapshot.get(name).position_deg,
                "tolerance_deg": tolerance,
                "aligned": None
                if tolerance is None
                else abs(goals[name] - snapshot.get(name).position_deg) <= tolerance,
            }
            for name in JOINT_NAMES
        ]
        self.device.startup_receipt["alignment_checks"].append(
            {
                "checked_ns": time.perf_counter_ns(),
                "leader_sample": asdict(sample),
                "alignment_gate_enabled": tolerance is not None,
                "joints": rows,
            }
        )
        if tolerance is not None and not all(row["aligned"] for row in rows):
            raise RuntimeError("Full-arm validation requires manual all-seven leader/follower alignment")
        return snapshot

    def aligned(self, deadline_ns):
        """Obtain a new bounded mixed-state snapshot before checking all leader angles."""
        return self.check_alignment(self.read_all(deadline_ns, active=self.enabled))

    def before_write(self, packet):
        """Recheck the latest cached leader immediately before every actual enable."""
        self.wait_guard()
        if any(packet == encode_enable(spec.motor_id) for spec in self.pinned_specs):
            self.check_alignment(self.device._latest_snapshot)
            name = next(spec.name for spec in self.pinned_specs if packet == encode_enable(spec.motor_id))
            self.check_preloaded_target(name, self.device._latest_snapshot)

    def check_preloaded_target(self, name, snapshot):
        """Never enable toward a target made stale by a known physical pose change."""
        tolerance = (
            self.configuration.preload_tolerance_deg
            if self.operator
            else min(self.profile(name).max_relative_deg, 0.05)
        )
        if tolerance is None:
            return
        measured = snapshot.get(name).position_deg
        if abs(self.hold_targets[name] - measured) > tolerance:
            self.device.startup_receipt.setdefault("preload_refusals", []).append(
                {
                    "name": name,
                    "preloaded_wire_deg": self.hold_targets[name],
                    "fresh_measured_deg": measured,
                    "tolerance_deg": tolerance,
                }
            )
            raise RuntimeError(
                f"Full-arm validation disabled preload no longer matches fresh position: {name}"
            )

    def record_startup_io(self, stage):
        """Keep bounded startup I/O segments in the auditable startup receipt."""
        evidence = self.device.take_io_evidence()
        self.device.startup_receipt["io_stages"].append({"stage": stage, **evidence})
        if not evidence["complete"]:
            raise RuntimeError("Full-arm validation startup I/O audit is incomplete")

    def command_write_entered(self, packet):
        """Distinguish pre-write guard refusal from a native write with unknown outcome."""
        pending = (
            self.pending_response if self.pending_parameter_write is None else self.pending_parameter_write
        )
        if pending is None or pending["packet_hex"] != packet.hex():
            raise RuntimeError("Full-arm validation command response ownership is ambiguous")
        pending["submission"] = "entered_unknown"

    def command_write_submitted(self, packet):
        """Pin a full host submission even if the subsequent deadline check fails."""
        pending = (
            self.pending_response if self.pending_parameter_write is None else self.pending_parameter_write
        )
        if pending is None or pending["packet_hex"] != packet.hex():
            raise RuntimeError("Full-arm validation submitted command differs from its pending owner")
        pending["submission"] = "complete"

    def uses_reference_envelope(self, name: str) -> bool:
        """Keep operator gripper travel in its own working/calibrated bounds.

        The startup envelope limits the six operator arm joints. Diagnostic
        sessions retain their original all-seven reference envelope.
        """
        return not (self.operator and name == "gripper")

    def reference_envelope_exceeded(self, name: str, position_deg: float) -> bool:
        """Apply the same pinned envelope policy to batch and command feedback."""
        return (
            self.uses_reference_envelope(name)
            and self.references is not None
            and abs(position_deg - self.references[name])
            > self.configuration_snapshot["reference_envelope_deg"]
        )

    def profile(self, name):
        profile = self.device._profiles[name]
        if self.starting and profile.mode == "mit":
            # Active references must not preload motion/torque bias into the
            # sequential disabled-to-enabled startup alignment.
            profile = replace(profile, mit_velocity_deg_s=0.0, mit_torque_Nm=0.0)
        if self.references is None or not self.uses_reference_envelope(name):
            return profile
        reference = self.references[name]
        return replace(
            profile,
            minimum_deg=max(profile.minimum_deg, reference - self.configuration.reference_envelope_deg),
            maximum_deg=min(profile.maximum_deg, reference + self.configuration.reference_envelope_deg),
        )

    def exchange(self, stage, name, packet, expected_status, deadline_ns):
        """Submit only a staged packet and consume exactly its addressed feedback."""
        self.device._prepare_query(deadline_ns)
        self.wait_guard()
        item = {
            "stage": stage,
            "name": name,
            "packet_hex": packet.hex(),
            "status": "unknown",
            "send_start_ns": time.perf_counter_ns(),
            "hardware_ack": "unknown",
        }
        self.allowed_packet = packet
        self.pending_response = {
            "name": name,
            "packet_hex": packet.hex(),
            "expected_status": expected_status,
            "submission": "planned",
            "session_signature": self.device._rx_session_signature(),
        }
        try:
            self.device._write_once(
                packet, min(deadline_ns, item["send_start_ns"] + int(self.device.config.send_budget_s * 1e9))
            )
            self.pending_response["submission"] = "complete"
            item["status"] = "submitted"
            self.last_command_ns[name] = time.perf_counter_ns()
        except BaseException:
            if self.pending_response is not None and self.pending_response["submission"] == "planned":
                self.pending_response = None
                item["status"] = "not_attempted"
            raise
        finally:
            self.allowed_packet = None
            item["send_end_ns"] = time.perf_counter_ns()
            if self.starting:
                self.device.startup_receipt["commands"].append(item)
        item["feedback"] = self.device._receive_gripper_feedback(
            min(deadline_ns, time.perf_counter_ns() + int(self.device.config.feedback_max_age_s * 1e9)),
            expected_status,
            packet_hex=packet.hex(),
            _joint_name=name,
            _guard=self.wait_guard,
        )
        self.pending_response = None
        self.observed_statuses[name] = item["feedback"]["status_code"]
        # A command reply is newer measured evidence too. Preserve it rather
        # than enabling against an older all-joint cache after a preload moved.
        current = self.device._latest_snapshot
        if current is not None:
            row = item["feedback"]
            self.device._sequences[name] += 1
            updated = JointFeedback(
                name,
                row["position_deg"],
                True,
                row["receive_timestamp_ns"],
                self.device._sequences[name],
                0,
                row["time_source"],
            )
            self.device._latest_snapshot = JointSnapshot(
                tuple(updated if joint.name == name else joint for joint in current.joints),
                max(current.completed_ns, row["receive_timestamp_ns"]),
                deadline_ns,
            )
        self.wait_guard()
        return item

    def hold_enabled(self, deadline_ns):
        """Refresh confirmed enabled joints at current reference targets during sequential startup."""
        for name in JOINT_NAMES:
            if name in self.enabled:
                measured = self.device._latest_snapshot.get(name).position_deg
                target = encode_position(
                    self.device._specs[name].motor_id,
                    self.hold_targets[name],
                    measured,
                    self.profile(name),
                    self.device._mapping_values(name),
                )
                self.exchange("startup_reference_refresh", name, target.packet, 1, deadline_ns)

    def write_operator_register(self, name, register_id, value, deadline_ns):
        """Prepare a selected register on a fresh disabled drive, without storing it."""
        if register_id not in (9, 10):
            raise ValueError("Operator startup only prepares TIMEOUT and CTRL_MODE")
        self.read_all(deadline_ns, active=set())
        self.device._prepare_query(deadline_ns)
        self.wait_guard()
        motor_id = self.device._specs[name].motor_id
        profile = self.device._profiles[name]
        packet = (
            encode_mode_write(motor_id, profile.mode)
            if register_id == 10
            else encode_timeout_write(motor_id, value)
        )
        category = "mode_writes" if register_id == 10 else "watchdog_writes"
        item = {
            "stage": (
                "operator_gripper_mode_selection" if name == "gripper" else "operator_arm_mode_selection"
            )
            if register_id == 10
            else "operator_watchdog_preparation",
            "name": name,
            "register_id": register_id,
            "requested_value": value,
            "packet_hex": packet.hex(),
            "status": "unknown",
            "hardware_ack": "unknown",
            "send_start_ns": time.perf_counter_ns(),
        }
        self.device.startup_receipt[category].append(item)
        self.allowed_packet = packet
        self.pending_parameter_write = {
            "name": name,
            "packet_hex": packet.hex(),
            "register_id": register_id,
            "requested_value": value,
            "submission": "planned",
            "session_signature": self.device._rx_session_signature(),
        }
        try:
            self.device._write_once(
                packet,
                min(deadline_ns, item["send_start_ns"] + int(self.device.config.send_budget_s * 1e9)),
            )
            item["status"] = "submitted"
            self.device._capture_parameter_rx(register_id, deadline_ns, _joint_name=name)
            self.pending_parameter_write = None
        except BaseException:
            if (
                self.pending_parameter_write is not None
                and self.pending_parameter_write["submission"] == "planned"
            ):
                self.pending_parameter_write = None
                item["status"] = "not_attempted"
            elif (
                self.pending_parameter_write is not None
                and self.pending_parameter_write["submission"] == "complete"
            ):
                item["status"] = "submitted"
            raise
        finally:
            self.allowed_packet = None
            item["send_end_ns"] = time.perf_counter_ns()
        state = read_control_state(
            self.device, names=(name,), max_read_duration_s=self.device._remaining(deadline_ns)
        )
        item["readback"] = asdict(state)
        expected = {10: self.device._modes[name], 9: self.watchdogs[name]}
        expected[register_id] = value
        if not state.is_valid or {row.register_id: row.value for row in state.registers} != expected:
            label = (
                ("FORCE_POS mode" if name == "gripper" else "MIT mode") if register_id == 10 else "TIMEOUT"
            )
            raise RuntimeError(f"Operator {label} readback failed: {name}")
        self.device._modes[name], self.watchdogs[name] = expected[10], expected[9]
        self.read_all(deadline_ns, active=set())
        self.record_startup_io(f"operator_register:{name}:{register_id}")

    def prepare_operator_registers(self, deadline_ns):
        """Prepare the selected watchdog and each profile's mode only on explicit Start."""
        if not self.operator:
            return
        state = read_control_state(self.device, max_read_duration_s=self.device._remaining(deadline_ns))
        self.device.startup_receipt["original_control_state"] = asdict(state)
        if not state.is_valid:
            raise RuntimeError(f"Operator initial mode/TIMEOUT query failed: {state.error}")
        values = {(row.name, row.register_id): row.value for row in state.registers}
        for name in JOINT_NAMES:
            mode = values[(name, 10)]
            if mode not in (1, 2, 3, 4):
                raise RuntimeError(f"Operator unsupported initial mode={mode}: {name}")
            self.device._modes[name], self.watchdogs[name] = mode, values[(name, 9)]
        for name in JOINT_NAMES:
            if self.watchdogs[name] != self.configuration.watchdog_ticks:
                self.write_operator_register(name, 9, self.configuration.watchdog_ticks, deadline_ns)
        for name in JOINT_NAMES:
            mode = 4 if name == "gripper" else 1
            if self.device._modes[name] != mode:
                self.write_operator_register(name, 10, mode, deadline_ns)
        final = read_control_state(self.device, max_read_duration_s=self.device._remaining(deadline_ns))
        self.device.startup_receipt["prepared_control_state"] = asdict(final)
        expected = {
            (name, register_id): (4 if name == "gripper" else 1)
            if register_id == 10
            else self.configuration.watchdog_ticks
            for name in JOINT_NAMES
            for register_id in (10, 9)
        }
        if (
            not final.is_valid
            or {(row.name, row.register_id): row.value for row in final.registers} != expected
        ):
            raise RuntimeError("Operator prepared mode/TIMEOUT readback failed")
        self.record_startup_io("operator_register_preparation_complete")

    def start(self, guard, leader_sample):
        """Preload disabled targets, then activate each joint through stage-correct fresh feedback."""
        if self.start_attempted or not self.device._control_session or self.device._observe_session:
            raise RuntimeError("Full-arm validation startup requires a fresh disabled control session")
        if not callable(guard) or not callable(leader_sample):
            raise TypeError("Full-arm validation requires live force/stop and immutable leader callbacks")
        self.start_attempted = self.starting = True
        self.guard, self.leader_provider = guard, leader_sample
        self.started_ns = time.perf_counter_ns()
        startup_budget_s = self.device.config.startup_budget_s
        if not self.operator:
            startup_budget_s = min(startup_budget_s, self.configuration.max_duration_s)
        deadline = self.started_ns + int(startup_budget_s * 1e9)
        receipt = self.device.startup_receipt = {
            "kind": self.kind,
            "session_id": self.configuration.session_id,
            "production_arm_authorized": False,
            "watchdog_semantics_verified": False,
            "complete": False,
            "commands": [],
            "alignment_checks": [],
            "io_stages": [],
            "mode_writes": [],
            "watchdog_writes": [],
            "error": None,
        }
        if not self.operator:
            receipt["target_force_N"] = self.configuration.target_force_N
        else:
            receipt["connection_control_state"] = asdict(self.connection_state)
            receipt["arm_mit_gains"] = {}
            for name in JOINT_NAMES[:-1]:
                profile = self.device._profiles[name]
                kp_code, kd_code = mit_gain_wire_codes(profile.kp, profile.kd)
                receipt["arm_mit_gains"][name] = {
                    "requested": {"kp": profile.kp, "kd": profile.kd},
                    "wire_effective": {"kp": kp_code * 500 / 4095, "kd": kd_code * 5 / 4095},
                    "encoded": {"kp": kp_code, "kd": kd_code},
                    "velocity_reference_rad_s": 0.0,
                    "feedforward_torque_Nm": 0.0,
                    "active_velocity_reference_deg_s": profile.mit_velocity_deg_s,
                    "active_feedforward_torque_Nm": profile.mit_torque_Nm,
                }
        try:
            self.wait_guard()
            initial = self.aligned(deadline)
            self.references = {name: initial.get(name).position_deg for name in JOINT_NAMES}
            self.hold_targets = dict(self.references)
            for name in JOINT_NAMES:
                encode_position(
                    self.device._specs[name].motor_id,
                    self.references[name],
                    self.references[name],
                    self.device._profiles[name],
                    self.device._mapping_values(name),
                )
            self.record_startup_io("initial_disabled_alignment")
            self.prepare_operator_registers(deadline)
            receipt["reference_deg"] = dict(self.references)
            receipt["reference_bounds_deg"] = {
                name: [self.profile(name).minimum_deg, self.profile(name).maximum_deg] for name in JOINT_NAMES
            }
            for name in JOINT_NAMES:
                fresh = self.read_all(deadline, active=set())
                measured = fresh.get(name).position_deg
                if not self.profile(name).minimum_deg <= measured <= self.profile(name).maximum_deg:
                    raise RuntimeError(
                        f"Full-arm validation initial target is outside its explicit profile: {name}"
                    )
                target = encode_position(
                    self.device._specs[name].motor_id,
                    measured,
                    measured,
                    self.profile(name),
                    self.device._mapping_values(name),
                )
                self.hold_targets[name] = target.wire_position_deg
                self.exchange("all_disabled_target_preload", name, target.packet, 0, deadline)
                self.record_startup_io(f"disabled_preload:{name}")
            for name in JOINT_NAMES:
                self.hold_enabled(deadline)
                fresh = self.aligned(deadline)
                measured = fresh.get(name).position_deg
                target = encode_position(
                    self.device._specs[name].motor_id,
                    measured,
                    measured,
                    self.profile(name),
                    self.device._mapping_values(name),
                )
                self.hold_targets[name] = target.wire_position_deg
                self.exchange("selected_disabled_fresh_preload", name, target.packet, 0, deadline)
                self.hold_enabled(deadline)
                newest = self.aligned(deadline)
                self.check_preloaded_target(name, newest)
                self.activation_risk = self.device._may_be_active = True
                self.exchange(
                    "enable_only", name, encode_enable(self.device._specs[name].motor_id), 1, deadline
                )
                self.enabled.add(name)
                self.exchange("enabled_fresh_reference", name, target.packet, 1, deadline)
                self.hold_enabled(deadline)
                self.read_all(deadline, active=self.enabled)
                self.record_startup_io(f"enabled_reference:{name}")
            self.hold_enabled(deadline)
            final = self.read_all(deadline, active=set(JOINT_NAMES))
            self.record_startup_io("all_enabled_fresh_reference")
            self.ready = self.device._motion_session = True
            receipt.update(complete=True, final_snapshot=asdict(final), enabled_joint_names=list(JOINT_NAMES))
            return final
        except BaseException as exc:
            self.device._failure = (
                self.device._failure or f"full_arm_validation_start_failed: {type(exc).__name__}: {exc}"
            )
            receipt["error"] = self.device._failure
            self.device.emergency_stop(self.device._failure)
            raise
        finally:
            self.starting = False

    def send(self, goals, snapshot):
        """Preserve seven sequential receipts and clipped wire targets behind the bounded capability."""
        if not self.ready or not self.device._motion_session:
            raise RuntimeError("Full-arm validation has not completed explicit startup")
        if set(goals) != set(JOINT_NAMES) or len(goals) != 7:
            raise ValueError("Full-arm validation requires a complete seven-joint target frame")
        if snapshot is not self.device._latest_snapshot or not snapshot.is_valid:
            raise ValueError("Full-arm validation requires this owner's latest complete snapshot")
        self.wait_guard()
        now = time.perf_counter_ns()
        encoded = {}
        for name in JOINT_NAMES:
            feedback = snapshot.get(name)
            if not 0 <= now - feedback.receive_timestamp_ns <= self.device.config.feedback_max_age_s * 1e9:
                raise RuntimeError("Full-arm validation target has stale feedback")
            encoded[name] = encode_position(
                self.device._specs[name].motor_id,
                goals[name],
                feedback.position_deg,
                self.profile(name),
                self.device._mapping_values(name),
            )
        started = time.perf_counter_ns()
        deadline = started + int(self.device.config.send_budget_s * 1e9)
        results, failure = [], None
        for name in JOINT_NAMES:
            target = encoded[name]
            begin = end = None
            status = "not_attempted" if failure is not None else "unknown"
            error = "prior_submission_failed" if failure is not None else None
            if failure is None:
                try:
                    if (
                        time.perf_counter_ns() - snapshot.get(name).receive_timestamp_ns
                        > self.device.config.feedback_max_age_s * 1e9
                    ):
                        raise RuntimeError("Full-arm validation feedback expired before submission")
                    begin = time.perf_counter_ns()
                    self.exchange(
                        "operator_control" if self.operator else "validation_control",
                        name,
                        target.packet,
                        1,
                        deadline,
                    )
                    status = "success"
                except BaseException as exc:
                    failure, error = exc, f"{type(exc).__name__}: {exc}"
                finally:
                    end = time.perf_counter_ns()
            results.append(
                JointSendResult(
                    name,
                    goals[name],
                    target.clipped_deg,
                    target.wire_position_deg if status == "success" else None,
                    status,
                    begin,
                    end,
                    error=error,
                )
            )
        receipt = SendReceipt(tuple(results), started, time.perf_counter_ns())
        self.device.last_receipt = receipt
        if failure is not None:
            self.device._failure = self.device._failure or str(failure)
            self.device.emergency_stop(str(failure))
            if not isinstance(failure, Exception):
                raise failure
        return receipt

    def stop(self, reason):
        """Attempt all seven disables before reading any response; never skip later motors."""
        self.guard = None
        self.cleaning = True
        self.device._failure = self.device._failure or reason
        receipt = {
            "action": "disable_all_operator_teleoperation" if self.operator else "disable_all_validation",
            "status": "unknown",
            "reason": reason,
            "error": None,
            "hardware_ack": "unknown",
            "hardware_stop_confirmed": False,
            "all_joints_disabled_observed": False,
            "production_arm_authorized": False,
            "per_joint_results": [],
            "send_start_ns": time.perf_counter_ns(),
            "send_end_ns": None,
            "outstanding_query_before_disable": copy.deepcopy(self.device._outstanding_query),
            "pending_command_before_disable": copy.deepcopy(self.pending_response),
            "pending_parameter_write_before_disable": copy.deepcopy(self.pending_parameter_write),
            "historical_responses": [],
            "unresolved_transactions": [],
            "transactions_retired_for_cleanup": False,
            "fresh_disabled_verification_complete": False,
        }
        deadline = receipt["send_start_ns"] + int(self.pinned_control.fault_budget_s * 1e9)
        receipt["disable_inter_packet_s"] = 0.001
        failures = []
        for name in JOINT_NAMES:
            item = {
                "name": name,
                "status": "unknown",
                "hardware_ack": "unknown",
                "error": None,
                "send_start_ns": time.perf_counter_ns(),
                "send_end_ns": None,
                "packet_hex": encode_disable(self.pinned_by_name[name].motor_id).hex(),
            }
            receipt["per_joint_results"].append(item)
            try:
                self.device._write_once(
                    bytes.fromhex(item["packet_hex"]),
                    min(deadline, time.perf_counter_ns() + int(self.pinned_control.send_budget_s * 1e9)),
                    fault_attempt=True,
                )
                item["status"] = "submitted"
            except BaseException as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(exc)
            finally:
                item["send_end_ns"] = time.perf_counter_ns()
            if name != JOINT_NAMES[-1]:
                try:
                    time.sleep(0.001)
                except BaseException as exc:
                    failures.append(exc)
        try:
            response_deadline = min(
                time.perf_counter_ns() + int(self.pinned_control.feedback_max_age_s * 1e9),
                deadline - int(self.pinned_control.feedback_max_age_s * 1e9),
            )
            receipt["response_deadline_ns"] = response_deadline
            if self.operator:
                self.quarantine_operator_responses(response_deadline, receipt)
            else:
                self.collect_disabled(response_deadline, receipt)
            final = self.read_all(deadline, active=set(), cleanup=True)
            receipt.update(
                all_joints_disabled_observed=True,
                fresh_disabled_verification_complete=True,
                post_disable_snapshot=asdict(final),
                post_disable_feedback=asdict(self.last_observation),
                position_limit_violations=list(self.position_limit_violations),
            )
            self.enabled.clear()
            self.ready = self.device._motion_session = self.device._may_be_active = False
        except BaseException as exc:
            failures.append(exc)
        receipt["send_end_ns"] = time.perf_counter_ns()
        receipt["outstanding_query_accounted"] = (
            self.device._outstanding_query is None
            and self.device._query_write_inflight is None
            and self.pending_response is None
            and self.pending_parameter_write is None
            and not receipt["unresolved_transactions"]
        )
        if failures:
            receipt["error"] = "; ".join(f"{type(error).__name__}: {error}" for error in failures)
        else:
            receipt["status"] = "submitted"
        return receipt

    def quarantine_operator_responses(self, deadline_ns, receipt):
        """Retire fault-only RX debt without crediting it as a fresh drive-state probe."""
        context = self.device._new_rx_context("full_validation_disable_feedback", deadline_ns)
        by_id = {spec.feedback_id: spec for spec in self.pinned_specs}
        maximum_frames = min(16, self.pinned_control.prequery_byte_limit // 16)
        drained = receipt["drained_responses"] = []
        while True:
            while len(self.device._rx_pending) < 16:
                previous = len(self.device._rx_pending)
                self.device._read_rx_chunk(
                    context,
                    deadline_ns,
                    lambda data, received_ns: self.device._audit_io(
                        "full_validation_stop_rx", data, received_ns
                    ),
                )
                if previous == 0 and not self.device._rx_pending and not self.device._serial.in_waiting:
                    break
            if not self.device._rx_pending:
                break
            raw, evidence = self.device._peek_rx_frame(context, deadline_ns, allow_inherited=True)
            if len(drained) >= maximum_frames or raw[:3] != b"\xaa\x11\x08" or raw[15] != 0x55:
                raise RuntimeError("Operator fault drain response bound/framing failure")
            spec = by_id.get(int.from_bytes(raw[3:7], "little"))
            if spec is None or raw[7] & 15 != spec.motor_id:
                raise RuntimeError("Operator fault drain received an unknown ID/motor")
            outstanding = self.device._outstanding_query
            command = self.pending_response
            parameter = self.pending_parameter_write
            entry = {
                "name": spec.name,
                "packet_hex": raw.hex(),
                "rx_context": evidence,
                "historical_only": True,
                "attribution": "unattributed_fault_drain",
                "hardware_ack": "unknown",
            }
            drained.append(entry)
            is_register = raw[7:10] in (
                bytes((spec.motor_id, 0, 0x33)),
                bytes((spec.motor_id, 0, 0x55)),
            )
            matching_parameter = (
                parameter is not None
                and parameter["name"] == spec.name
                and raw[10] == parameter["register_id"]
                and int.from_bytes(raw[11:15], "little") == parameter["requested_value"]
            )
            matching_register_query = (
                outstanding is not None
                and outstanding["feedback_id"] == spec.feedback_id
                and outstanding["opcode"] == 0x33
                and raw[7:11] == bytes((spec.motor_id, 0, 0x33, outstanding["register_id"]))
            )
            if is_register and not (matching_parameter or matching_register_query):
                raise RuntimeError("Operator fault drain received an unaccounted register packet")
            proven_query = (
                outstanding is not None
                and outstanding["feedback_id"] == spec.feedback_id
                and evidence["inherited"]
                and evidence["source"] == "query_rx"
                and evidence["packet_hex"] == outstanding["packet_hex"]
                and outstanding["submission"] == "complete"
            )
            if proven_query:
                historical = self.device._account_outstanding_query_frame(raw, evidence, deadline_ns)
                receipt["historical_responses"].append(historical)
                continue
            proven_command = (
                command is not None
                and command["name"] == spec.name
                and command["submission"] == "complete"
                and evidence["inherited"]
                and evidence["packet_hex"] == command["packet_hex"]
            )
            if proven_command:
                entry["pending_command"] = copy.deepcopy(command)
                self.pending_response = None
                receipt["historical_responses"].append(entry)
            elif (
                parameter is not None
                and parameter["name"] == spec.name
                and (matching_parameter or evidence["inherited"])
            ):
                entry["pending_parameter_write"] = copy.deepcopy(parameter)
                receipt["historical_responses"].append(entry)
            elif (
                outstanding is not None and outstanding["feedback_id"] == spec.feedback_id and not is_register
            ):
                entry["outstanding_query"] = copy.deepcopy(outstanding)
                receipt["historical_responses"].append(entry)
            elif command is not None and command["name"] == spec.name and raw[7] >> 4 == 1:
                entry["pending_command"] = copy.deepcopy(command)
                receipt["historical_responses"].append(entry)
            self.device._finish_rx_frame()
        unresolved = receipt["unresolved_transactions"]
        for kind, transaction in (
            ("query", self.device._outstanding_query),
            ("query_write", self.device._query_write_inflight),
            ("command", self.pending_response),
            ("parameter_write", self.pending_parameter_write),
        ):
            if transaction is not None and not (
                kind == "parameter_write" and transaction["submission"] == "complete"
            ):
                unresolved.append(
                    {
                        "kind": kind,
                        "evidence": copy.deepcopy(transaction),
                        "outcome": "unresolved_retired_after_fault_rx_drain",
                        "hardware_ack": "unknown",
                    }
                )
        # Active ownership pointers must not prevent independent fresh state
        # probes. Their unresolved provenance remains in this fault-only archive.
        self.device._outstanding_query = self.device._query_write_inflight = None
        self.pending_response = self.pending_parameter_write = None
        receipt["transactions_retired_for_cleanup"] = True

    def collect_disabled(self, deadline_ns, receipt):
        """Strictly account the known old exchange and all seven addressed disable observations."""
        context = self.device._new_rx_context("full_validation_disable_feedback", deadline_ns)
        disabled = {}
        maximum_frames = min(16, self.pinned_control.prequery_byte_limit // 16)
        frames = 0
        by_id = {spec.feedback_id: spec for spec in self.pinned_specs}
        parameter = self.pending_parameter_write
        parameter_disabled_extra = False
        if parameter is not None and (
            parameter["submission"] != "complete"
            or parameter["session_signature"] != self.device._rx_session_signature()
        ):
            raise RuntimeError("Full-arm disable has an unknown parameter-write submission")
        while (
            len(disabled) < 7
            or self.device._outstanding_query is not None
            or self.pending_response is not None
            or (parameter is not None and (self.device._rx_pending or self.device._serial.in_waiting))
        ):
            while len(self.device._rx_pending) < 16:
                self.device._read_rx_chunk(
                    context,
                    deadline_ns,
                    lambda data, received_ns: self.device._audit_io(
                        "full_validation_stop_rx", data, received_ns
                    ),
                )
            raw, evidence = self.device._peek_rx_frame(context, deadline_ns, allow_inherited=True)
            frames += 1
            if frames > maximum_frames or raw[:3] != b"\xaa\x11\x08" or raw[15] != 0x55:
                raise RuntimeError("Full-arm disable response bound/framing failure")
            spec = by_id.get(int.from_bytes(raw[3:7], "little"))
            if spec is None or raw[7] & 15 != spec.motor_id:
                raise RuntimeError("Full-arm disable received an unknown ID/motor")
            if parameter is not None and spec.name == parameter["name"]:
                matching_echo = (
                    raw[7:11]
                    in (
                        bytes((spec.motor_id, 0, 0x33, parameter["register_id"])),
                        bytes((spec.motor_id, 0, 0x55, parameter["register_id"])),
                    )
                    and int.from_bytes(raw[11:15], "little") == parameter["requested_value"]
                )
                inherited_disabled = (
                    evidence["inherited"]
                    and evidence["source"] == "parameter_write_unattributed_rx"
                    and raw[7] == spec.motor_id
                    and raw[7:10]
                    not in (
                        bytes((spec.motor_id, 0, 0x33)),
                        bytes((spec.motor_id, 0, 0x55)),
                    )
                )
                if matching_echo or inherited_disabled:
                    receipt["historical_responses"].append(
                        {
                            "packet_hex": raw.hex(),
                            "rx_context": evidence,
                            "historical_only": True,
                            "pending_parameter_write": copy.deepcopy(parameter),
                        }
                    )
                    self.device._finish_rx_frame()
                    continue
            outstanding = self.device._outstanding_query
            if raw[7:10] in (bytes((spec.motor_id, 0, 0x33)), bytes((spec.motor_id, 0, 0x55))) and not (
                outstanding is not None
                and outstanding["feedback_id"] == spec.feedback_id
                and outstanding["opcode"] == 0x33
            ):
                raise RuntimeError("Full-arm disable received an unaccounted register packet")
            if outstanding is not None and outstanding["feedback_id"] == int.from_bytes(raw[3:7], "little"):
                proven = (
                    evidence["inherited"]
                    and evidence["source"] == "query_rx"
                    and evidence["packet_hex"] == outstanding["packet_hex"]
                )
                if outstanding["opcode"] == 0xCC and not proven:
                    raise RuntimeError("Full-arm disable has an ambiguous same-ID zero-prefix query response")
                historical = self.device._account_outstanding_query_frame(raw, evidence, deadline_ns)
                receipt["historical_responses"].append(historical)
                continue
            if (
                self.pending_response is not None
                and spec is not None
                and self.pending_response["name"] == spec.name
            ):
                proven = (
                    evidence["inherited"] and evidence["packet_hex"] == self.pending_response["packet_hex"]
                )
                if not proven or self.pending_response["submission"] != "complete":
                    raise RuntimeError("Full-arm disable has an ambiguous/unknown old command response")
                if raw[7] >> 4 not in (0, 1):
                    raise RuntimeError("Full-arm old command response reports a drive fault")
                receipt["historical_responses"].append(
                    {
                        "packet_hex": raw.hex(),
                        "rx_context": evidence,
                        "historical_only": True,
                        "pending_command": copy.deepcopy(self.pending_response),
                    }
                )
                self.pending_response = None
                self.device._finish_rx_frame()
                continue
            if (
                parameter is not None
                and spec.name == parameter["name"]
                and spec.name in disabled
                and not parameter_disabled_extra
                and not evidence["inherited"]
                and raw[7] == spec.motor_id
            ):
                # A parameter write may also produce ordinary DISABLED feedback.
                # If Stop arrived before RX began, it is indistinguishable from
                # this motor's disable response. Retain the one extra observation
                # as unattributed history; the subsequent fresh query establishes
                # the state, and neither packet is treated as a command ACK.
                receipt["historical_responses"].append(
                    {
                        "packet_hex": raw.hex(),
                        "rx_context": evidence,
                        "historical_only": True,
                        "attribution": "unattributed_disabled_during_parameter_cleanup",
                        "pending_parameter_write": copy.deepcopy(parameter),
                    }
                )
                parameter_disabled_extra = True
                self.device._finish_rx_frame()
                continue
            if spec is None or evidence["inherited"] or raw[7] >> 4 != 0 or spec.name in disabled:
                raise RuntimeError(
                    "Full-arm disable received unexpected, duplicated or non-DISABLED feedback"
                )
            disabled[spec.name] = {
                "packet_hex": raw.hex(),
                "receive_timestamp_ns": evidence["last_receive_timestamp_ns"],
                "status_code": 0,
                "hardware_ack": "unknown",
            }
            self.device._finish_rx_frame()
        if self.device._serial.in_waiting:
            raise RuntimeError("Full-arm disable received excess unaccounted responses")
        # Register writes have optional echoes, unlike position/enable commands.
        # A complete host write with no remaining echo is accounted separately;
        # the fresh post-disable query below still establishes DISABLED state.
        self.pending_parameter_write = None
        receipt["disabled_feedback"] = disabled
