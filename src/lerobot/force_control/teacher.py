"""Pure incremental I-P candidates; hardware submission belongs to the owner."""

# SI force symbols match the public control schema.
# ruff: noqa: N803

import math
from dataclasses import replace

from .config import TeacherConfig, finite_number, positive, validate_close_sign
from .types import ContactReference, ControlInputs, ControlPhase, TeacherProposal

_ACTIVE_PHASES = (
    ControlPhase.FORCE_TRACK,
    ControlPhase.POSITION_HOLD,
    ControlPhase.APPROACH,
    ControlPhase.FOLLOW,
    ControlPhase.FOLLOW_WAIT_CLEAR,
)


class ForceTeacher:
    """Accumulate position once, using only the owner's final submitted reference.

    There is no separate error integral or private force-control position target.
    ``propose`` is pure; only a successful ``commit`` advances force history.
    """

    def __init__(self, config: TeacherConfig, close_sign: int):
        """Validate explicit gains and initialize an inert controller."""
        config.validate()
        validate_close_sign(close_sign)
        self.config = config
        self.close_sign = close_sign
        self.phase = ControlPhase.DISABLED
        self.target_version = 0
        self.fault_reason: str | None = None
        self._revision = 0
        self._last_force_N: float | None = None
        self._first_force_submission = True
        self._release_endpoint_encoded = False
        self._hold_target_deg: float | None = None
        self._hold_speed_deg_s: float | None = None
        self._contact_alignment_pending = False
        self._hybrid_active = False
        self._contact_reference: ContactReference | None = None
        self._last_leader_inputs: ControlInputs | None = None
        self._phase_start_ns: int | None = None
        self._phase_start_q: float | None = None

    @property
    def hold_target_deg(self) -> float | None:
        """Expose the single explicit frozen/manual position goal."""
        return self._hold_target_deg

    @property
    def hold_speed_deg_s(self) -> float | None:
        """Requested speed for the current absolute-position move, if specified."""
        return self._hold_speed_deg_s

    @property
    def contact_alignment_pending(self) -> bool:
        """Whether slew limits left old closing lead to remove after contact."""
        return self._contact_alignment_pending

    @property
    def hybrid_active(self) -> bool:
        """Whether this explicit start selected contact-triggered human following."""
        return self._hybrid_active

    @property
    def contact_reference(self) -> ContactReference | None:
        """Return only the contact reference accepted by a successful submission."""
        return self._contact_reference

    def _reset(self) -> None:
        self._last_force_N = None
        self._first_force_submission = True
        self._release_endpoint_encoded = False
        self._hold_target_deg = None
        self._hold_speed_deg_s = None
        self._contact_alignment_pending = False
        self._hybrid_active = False
        self._contact_reference = None
        self._last_leader_inputs = None
        self._phase_start_ns = self._phase_start_q = None
        self._revision += 1

    @staticmethod
    def _validate_reference(now_ns: int, q_measured_deg: float, force_N: float | None = None) -> None:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative monotonic timestamp")
        finite_number("q_measured_deg", q_measured_deg)
        if force_N is not None:
            finite_number("force_N", force_N)

    def arm(self) -> None:
        """Arming never produces a motion candidate."""
        if self.phase != ControlPhase.DISABLED:
            raise RuntimeError("arm requires DISABLED; a fault needs explicit reset first")
        self._reset()
        self.phase = ControlPhase.ARMED

    def start(
        self,
        now_ns: int,
        q_measured_deg: float,
        *,
        force_N: float | None = None,
        mode: str = "force",
        position_deg: float | None = None,
        position_speed_deg_s: float | None = None,
    ) -> None:
        """Enter the explicitly requested trajectory from the owner's fresh reference."""
        if self.phase != ControlPhase.ARMED:
            raise RuntimeError("start requires explicit arm")
        if mode not in ("force", "approach", "position", "hybrid"):
            raise ValueError("unknown start mode")
        if mode in ("approach", "hybrid") and self.config.gripper_follow_speed_deg_s is None:
            raise ValueError("approach/hybrid requires explicit speed and force threshold")
        if mode == "position":
            finite_number("position_deg", position_deg)
        if position_speed_deg_s is not None:
            if mode != "position":
                raise ValueError("position speed requires position mode")
            positive("position_speed_deg_s", position_speed_deg_s)
        self._validate_reference(now_ns, q_measured_deg, force_N)
        self._reset()
        self.phase = ControlPhase.FORCE_TRACK
        self._last_force_N = force_N
        if mode == "approach":
            self.phase = ControlPhase.APPROACH
        elif mode == "hybrid":
            self.phase = ControlPhase.FOLLOW
            self._hybrid_active = True
        elif mode == "position":
            self.phase = ControlPhase.POSITION_HOLD
            self._hold_target_deg = float(position_deg)
            self._hold_speed_deg_s = position_speed_deg_s

    def approach(self, now_ns: int, q_measured_deg: float) -> None:
        """Explicitly re-enter constant-speed closure; force loss never re-enters it."""
        if self.phase not in _ACTIVE_PHASES:
            raise RuntimeError("approach requires an active non-faulted controller")
        if self.config.gripper_follow_speed_deg_s is None:
            raise ValueError("approach requires explicit speed and force threshold")
        self._validate_reference(now_ns, q_measured_deg)
        self._reset()
        self.phase = ControlPhase.APPROACH

    def move_to(
        self,
        now_ns: int,
        q_measured_deg: float,
        position_deg: float,
        position_speed_deg_s: float | None = None,
    ) -> None:
        """Set one absolute hold goal; the owner checks calibration and Safety limits it."""
        finite_number("position_deg", position_deg)
        if position_speed_deg_s is not None:
            positive("position_speed_deg_s", position_speed_deg_s)
        if self.phase not in _ACTIVE_PHASES:
            raise RuntimeError("position requires an active non-faulted controller")
        self._validate_reference(now_ns, q_measured_deg)
        self._reset()
        self.phase = ControlPhase.POSITION_HOLD
        self._hold_target_deg = float(position_deg)
        self._hold_speed_deg_s = position_speed_deg_s

    def resume(self, now_ns: int, q_measured_deg: float, *, force_N: float | None = None) -> None:
        """Resume from fresh inputs, with no integration during the position hold."""
        if self.phase != ControlPhase.POSITION_HOLD:
            raise RuntimeError("resume requires POSITION_HOLD")
        self._validate_reference(now_ns, q_measured_deg, force_N)
        self._reset()
        self.phase = ControlPhase.FORCE_TRACK
        self._last_force_N = force_N

    def release(self, now_ns: int, q_measured_deg: float) -> None:
        """Start an explicit opening trajectory bounded by configured travel and time."""
        if self.phase not in _ACTIVE_PHASES:
            raise RuntimeError("release requires an active non-faulted controller")
        self._validate_reference(now_ns, q_measured_deg)
        self._reset()
        self.phase = ControlPhase.RELEASE
        self._phase_start_ns = now_ns
        self._phase_start_q = q_measured_deg

    def disable(self) -> None:
        """Disable candidates; the owner remains responsible for physical disposition."""
        if self.phase == ControlPhase.FAULT:
            raise RuntimeError("disable cannot clear a latched fault; explicit reset is required")
        self._reset()
        self.phase = ControlPhase.DISABLED

    def fault(self, reason: str) -> None:
        """Latch the first fault and invalidate outstanding candidates."""
        self._reset()
        self.phase = ControlPhase.FAULT
        self.fault_reason = self.fault_reason or reason

    def reset_fault(self) -> None:
        """Reset enters DISABLED and never re-arms hardware."""
        if self.phase != ControlPhase.FAULT:
            raise RuntimeError("reset_fault requires FAULT")
        self._reset()
        self.phase = ControlPhase.DISABLED
        self.fault_reason = None

    def set_target(self, target_N: float) -> None:
        """A reference change preserves measurement history and causes no P kick."""
        self.config.validate_target(target_N)
        if self.phase == ControlPhase.FAULT:
            raise RuntimeError("a target update cannot clear a fault")
        self.config = replace(self.config, force_target_N=float(target_N))
        self.target_version += 1
        self._revision += 1

    def update_config(self, config: TeacherConfig) -> None:
        """Apply one validated bundle; gain changes align the next P increment to zero."""
        config.validate()
        if self.phase == ControlPhase.FAULT:
            raise RuntimeError("a parameter update cannot clear a fault")
        if (
            self.phase == ControlPhase.APPROACH or self._hybrid_active
        ) and config.gripper_follow_speed_deg_s is None:
            raise ValueError("cannot remove approach parameters during closure")
        gains_changed = (config.kp_deg_per_N, config.ki_deg_per_N_s) != (
            self.config.kp_deg_per_N,
            self.config.ki_deg_per_N_s,
        )
        target_changed = config.force_target_N != self.config.force_target_N
        self.config = config
        if gains_changed:
            self._last_force_N = None
        if target_changed:
            self.target_version += 1
        self._revision += 1

    def _invalid_reason(self, inputs: ControlInputs) -> str | None:
        try:
            for name in ("dt_control_s", "q_measured_deg", "q_prev_sent_deg"):
                finite_number(name, getattr(inputs, name))
            finite_number("left_N", inputs.force.left_N)
            finite_number("right_N", inputs.force.right_N)
        except ValueError:
            return "invalid_numeric_input"
        if inputs.dt_control_s <= 0:
            return "dt_out_of_budget"
        if isinstance(inputs.now_ns, bool) or not isinstance(inputs.now_ns, int) or inputs.now_ns < 0:
            return "invalid_timestamp"
        if not inputs.joint_valid or not inputs.force.is_valid or inputs.force.unit != "N":
            return "invalid_feedback"
        if self._hybrid_active:
            try:
                finite_number("leader_gripper_deg", inputs.leader_gripper_deg)
                finite_number("leader_raw_gripper_deg", inputs.leader_raw_gripper_deg)
            except ValueError:
                return "invalid_hybrid_leader"
            if (
                type(inputs.leader_sequence) is not int
                or inputs.leader_sequence < 0
                or type(inputs.leader_captured_ns) is not int
                or not 0 <= inputs.leader_captured_ns <= inputs.now_ns
            ):
                return "invalid_hybrid_leader_evidence"
            previous = self._last_leader_inputs
            if previous is not None and (
                inputs.leader_sequence < previous.leader_sequence
                or inputs.leader_captured_ns < previous.leader_captured_ns
            ):
                return "nonmonotonic_hybrid_leader"
        return None

    def _new_leader_sample(self, inputs: ControlInputs, reference: ControlInputs | ContactReference) -> bool:
        return (
            inputs.leader_sequence > reference.leader_sequence
            and inputs.leader_captured_ns > reference.leader_captured_ns
        )

    def _opening_past_contact(self, inputs: ControlInputs) -> bool:
        reference = self._contact_reference
        previous = self._last_leader_inputs
        if (
            reference is None
            or previous is None
            or not self._new_leader_sample(inputs, reference)
            or not self._new_leader_sample(inputs, previous)
            or self.close_sign * (inputs.leader_gripper_deg - previous.leader_gripper_deg) >= -1e-8
        ):
            return False
        leader = self.close_sign * inputs.leader_gripper_deg
        return (
            leader
            < min(
                self.close_sign * reference.leader_gripper_deg,
                self.close_sign * reference.follower_gripper_deg,
                self.close_sign * inputs.q_measured_deg,
            )
            - 1e-8
        )

    def propose(self, inputs: ControlInputs) -> TeacherProposal:
        """Compute an immutable candidate without advancing any controller state."""
        proposal = TeacherProposal(
            revision=self._revision,
            phase=self.phase,
            next_phase=self.phase,
            inputs=inputs,
            command_enabled=False,
            target_N=self.config.force_target_N,
            target_version=self.target_version,
            error_N=0.0,
            delta_i_deg=0.0,
            delta_p_deg=0.0,
            delta_raw_deg=0.0,
            u_raw_deg_s=0.0,
            q_candidate_deg=inputs.q_prev_sent_deg,
            failure_reason=self.fault_reason,
            contact_reference=self._contact_reference,
        )
        if self.phase in (ControlPhase.DISABLED, ControlPhase.FAULT):
            return proposal
        reason = self._invalid_reason(inputs)
        if reason:
            return replace(proposal, next_phase=ControlPhase.FAULT, failure_reason=reason)
        if self.phase == ControlPhase.ARMED:
            return proposal
        cfg = self.config
        error = cfg.force_target_N - inputs.force.mean_N
        delta_i = delta_p = delta_contact = delta = 0.0
        next_phase = self.phase
        contact_reference = self._contact_reference
        tag = "inactive"
        if self.phase == ControlPhase.APPROACH and inputs.force.mean_N >= cfg.contact_force_threshold_N:
            next_phase = ControlPhase.FORCE_TRACK
        elif self.phase == ControlPhase.FOLLOW and inputs.force.mean_N >= cfg.contact_force_threshold_N:
            next_phase = ControlPhase.FORCE_TRACK
            contact_reference = ContactReference(
                timestamp_ns=inputs.now_ns,
                leader_gripper_deg=inputs.leader_gripper_deg,
                leader_raw_gripper_deg=inputs.leader_raw_gripper_deg,
                leader_sequence=inputs.leader_sequence,
                leader_captured_ns=inputs.leader_captured_ns,
                follower_gripper_deg=inputs.q_measured_deg,
                q_prev_sent_deg=inputs.q_prev_sent_deg,
            )
        elif (
            self._hybrid_active
            and self.phase == ControlPhase.FORCE_TRACK
            and self._opening_past_contact(inputs)
        ):
            next_phase = ControlPhase.FOLLOW_WAIT_CLEAR
        elif self.phase == ControlPhase.FOLLOW_WAIT_CLEAR:
            previous = self._last_leader_inputs
            if (
                inputs.force.mean_N < cfg.contact_force_threshold_N
                and previous is not None
                and self._new_leader_sample(inputs, previous)
                and self.close_sign * (inputs.leader_gripper_deg - previous.leader_gripper_deg) > 1e-8
            ):
                next_phase = ControlPhase.FOLLOW
                contact_reference = None
        # Keep measurement feedback continuous through contact; only the forward drive changes.
        if (
            next_phase in (ControlPhase.APPROACH, ControlPhase.FORCE_TRACK)
            and self._last_force_N is not None
            and not self._first_force_submission
        ):
            delta_p = -cfg.kp_deg_per_N * (inputs.force.mean_N - self._last_force_N)
        if next_phase == ControlPhase.FORCE_TRACK:
            if abs(error) > cfg.integral_deadband_N:
                delta_i = cfg.ki_deg_per_N_s * inputs.dt_control_s * error
            if self.phase in (ControlPhase.APPROACH, ControlPhase.FOLLOW) or self._contact_alignment_pending:
                # Remove closing lead from the candidate, never falsify the submitted reference.
                # An existing opening lead is retained; contact must not pull it back closed.
                delta_contact = -max(0.0, self.close_sign * (inputs.q_prev_sent_deg - inputs.q_measured_deg))
            delta = delta_i + delta_p + delta_contact
            tag = (
                "deadband"
                if abs(error) <= cfg.integral_deadband_N
                else ("low_force_recovery" if error > 0 else "high_force_recovery")
            )
        elif next_phase in (ControlPhase.FOLLOW, ControlPhase.FOLLOW_WAIT_CLEAR):
            delta = self.close_sign * (inputs.leader_gripper_deg - inputs.q_prev_sent_deg)
            step = cfg.gripper_follow_speed_deg_s * inputs.dt_control_s
            delta = max(-step, min(step, delta))
            tag = "follow" if next_phase == ControlPhase.FOLLOW else "follow_wait_clear"
        elif self.phase == ControlPhase.APPROACH:
            advance = cfg.gripper_follow_speed_deg_s * inputs.dt_control_s
            # Falling force must not accelerate recontact beyond the requested approach speed.
            # Negative increments remain available, with opening limits enforced by Safety.
            delta = min(advance, advance + delta_p)
            tag = "force_feedback_approach"
        elif self.phase == ControlPhase.POSITION_HOLD:
            delta = self.close_sign * (self._hold_target_deg - inputs.q_prev_sent_deg)
            if self._hold_speed_deg_s is not None:
                step = self._hold_speed_deg_s * inputs.dt_control_s
                delta = max(-step, min(step, delta))
            tag = "position_hold"
        elif self.phase == ControlPhase.RELEASE:
            elapsed = (inputs.now_ns - self._phase_start_ns) / 1e9
            # Compare to one representable endpoint. Subtracting two positions
            # to recompute total travel can leave a sub-ULP remainder forever.
            endpoint = self._phase_start_q - self.close_sign * cfg.release_travel_deg
            remaining = self.close_sign * (inputs.q_prev_sent_deg - endpoint)
            if elapsed < 0:
                reason = "nonmonotonic_phase_time"
            elif self._release_endpoint_encoded or remaining <= 0:
                next_phase = ControlPhase.DISABLED
            elif elapsed >= cfg.max_release_time_s:
                reason = "release_budget_exhausted"
            else:
                delta = -min(cfg.release_speed_deg_s * inputs.dt_control_s, remaining)
            tag = "release"
        candidate = inputs.q_prev_sent_deg + self.close_sign * delta
        rate = delta / inputs.dt_control_s
        if not all(
            math.isfinite(value) for value in (error, delta_i, delta_p, delta_contact, delta, candidate, rate)
        ):
            reason = "nonfinite_control_arithmetic"
        if reason:
            next_phase = ControlPhase.FAULT
            error = delta_i = delta_p = delta_contact = delta = rate = 0.0
            candidate = inputs.q_prev_sent_deg
        return replace(
            proposal,
            next_phase=next_phase,
            command_enabled=next_phase not in (ControlPhase.DISABLED, ControlPhase.FAULT),
            error_N=error,
            delta_i_deg=delta_i,
            delta_p_deg=delta_p,
            delta_contact_deg=delta_contact,
            delta_raw_deg=delta,
            u_raw_deg_s=rate,
            q_candidate_deg=candidate,
            tracking_tag=tag,
            failure_reason=reason,
            transition=None if next_phase == self.phase else f"{self.phase}->{next_phase}",
            contact_reference=contact_reference,
        )

    def commit(
        self,
        proposal: TeacherProposal,
        *,
        q_sent_deg: float | None = None,
        q_accepted_deg: float | None = None,
        software_success: bool = False,
        safety_override: bool = False,
        i_blocked: bool = False,
    ) -> None:
        """Advance history only after the owner reports a successful final wire submission."""
        if proposal.revision != self._revision or proposal.phase != self.phase:
            raise ValueError("stale or already committed Teacher proposal")
        if proposal.next_phase == ControlPhase.FAULT:
            self.fault(proposal.failure_reason or "controller_fault")
            return
        if safety_override:
            self.fault("safety_override")
            return
        if proposal.command_enabled:
            if not software_success or q_sent_deg is None:
                self.fault("send_failed_or_unknown")
                return
            finite_number("q_sent_deg", q_sent_deg)
            if q_accepted_deg is not None:
                finite_number("q_accepted_deg", q_accepted_deg)
            if (
                proposal.phase in (ControlPhase.APPROACH, ControlPhase.FOLLOW)
                and proposal.next_phase == ControlPhase.FORCE_TRACK
            ) or self._contact_alignment_pending:
                # Slew/driver clipping must not discard the remaining alignment. Allow only
                # new force-control advance and the actual wire-encoding rounding difference.
                force_advance = max(0.0, proposal.delta_p_deg + (0.0 if i_blocked else proposal.delta_i_deg))
                wire_rounding = 0.0 if q_accepted_deg is None else abs(q_sent_deg - q_accepted_deg)
                lead = self.close_sign * (q_sent_deg - proposal.inputs.q_measured_deg)
                self._contact_alignment_pending = (
                    proposal.delta_contact_deg < 0 and lead > force_advance + wire_rounding + 1e-8
                )
            if proposal.next_phase != ControlPhase.FORCE_TRACK:
                self._contact_alignment_pending = False
            if self._hybrid_active:
                self._contact_reference = proposal.contact_reference
                self._last_leader_inputs = proposal.inputs
            if proposal.phase == ControlPhase.RELEASE:
                endpoint = self._phase_start_q - self.close_sign * self.config.release_travel_deg
                # The accepted endpoint must survive both safety and driver
                # clipping. Only then does a repeated identical wire target
                # establish that this endpoint has no finer encoded step.
                self._release_endpoint_encoded = (
                    proposal.q_candidate_deg == endpoint
                    and q_accepted_deg == endpoint
                    and q_sent_deg == proposal.inputs.q_prev_sent_deg
                    and q_sent_deg != proposal.q_candidate_deg
                )
            self._last_force_N = proposal.inputs.force.mean_N
            self._first_force_submission = False
        self.phase = proposal.next_phase
        if self.phase == ControlPhase.DISABLED:
            self._reset()
        else:
            self._revision += 1
