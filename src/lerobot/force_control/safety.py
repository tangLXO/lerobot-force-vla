"""Pure target restrictions with raw-force checks and directional I blocking."""

import math

from .config import SafetyConfig, finite_number, validate_close_sign
from .types import ControlInputs, ControlPhase, SafetyDecision, TeacherProposal


class SafetySupervisor:
    """Restrict one candidate; the owner executes faults and commits actual wire targets."""

    def __init__(self, config: SafetyConfig, close_sign: int):
        """Validate explicit restrictions without claiming any hardware resource."""
        config.validate()
        validate_close_sign(close_sign)
        self.config = config
        self.close_sign = close_sign
        self.fault_reason: str | None = None
        self.epoch = 0
        self._clear_motion_state()

    def _clear_motion_state(self) -> None:
        self._last_u_deg_s = 0.0
        self._run_start_ns: int | None = None
        self._total_travel_deg = 0.0
        self._stall_since_ns: int | None = None
        self._stall_anchor_deg: float | None = None
        self._stall_direction = 0
        self._pending_state = None

    def latch_fault(self, reason: str) -> SafetyDecision:
        """Keep the first reason and invalidate old owner-event epochs."""
        if self.fault_reason is None:
            self.fault_reason = reason
            self.epoch += 1
        self._last_u_deg_s = 0.0
        self._pending_state = None
        return SafetyDecision(None, 0.0, safety_override=True, fault_reason=self.fault_reason)

    def reset_fault(self) -> None:
        """Reset protection state only after an explicit operator event."""
        if self.fault_reason is None:
            raise RuntimeError("reset_fault requires a latched fault")
        self.fault_reason = None
        self.epoch += 1
        self._clear_motion_state()

    def reset_motion_reference(
        self, q_measured_deg: float | None = None, now_ns: int | None = None, *, preserve_budget: bool = False
    ) -> None:
        """Reset only at an explicit fresh start/resume/handover, never to clear a fault."""
        if self.fault_reason is not None:
            raise RuntimeError("a motion reference cannot clear a latched fault")
        if q_measured_deg is not None:
            finite_number("q_measured_deg", q_measured_deg)
        if now_ns is not None and (type(now_ns) is not int or now_ns < 0):
            raise ValueError("now_ns must be a nonnegative monotonic timestamp")
        if type(preserve_budget) is not bool:
            raise ValueError("preserve_budget must be boolean")
        run_start, travel = self._run_start_ns, self._total_travel_deg
        self._clear_motion_state()
        self._run_start_ns = run_start if preserve_budget else now_ns
        if preserve_budget:
            self._total_travel_deg = travel
        self._stall_anchor_deg = q_measured_deg

    def _input_failure(self, inputs: ControlInputs, now_ns: int) -> str | None:
        cfg = self.config
        try:
            for name in ("dt_control_s", "q_measured_deg", "q_prev_sent_deg"):
                finite_number(name, getattr(inputs, name))
            finite_number("left_N", inputs.force.left_N)
            finite_number("right_N", inputs.force.right_N)
        except ValueError:
            return "invalid_numeric_input"
        if not 0 < inputs.dt_control_s <= cfg.max_dt_s:
            return "dt_out_of_budget"
        stamps = (
            inputs.now_ns,
            now_ns,
            inputs.force.timestamp_ns,
            inputs.force.arrival_timestamp_ns,
            inputs.joint_received_ns,
        )
        if any(type(stamp) is not int or stamp < 0 for stamp in stamps):
            return "missing_or_invalid_timestamp"
        if any(type(seq) is not int or seq < 0 for seq in (inputs.force.sequence, inputs.joint_sequence)):
            return "missing_or_invalid_sequence"
        if not inputs.force.is_valid or inputs.force.unit != "N":
            return "invalid_force"
        if not inputs.joint_valid:
            return "invalid_joint_feedback"
        if inputs.force.timestamp_ns > inputs.now_ns or inputs.force.arrival_timestamp_ns > inputs.now_ns:
            return "noncausal_force"
        if inputs.joint_received_ns > inputs.now_ns:
            return "noncausal_joint_feedback"
        if now_ns < inputs.now_ns or now_ns - inputs.now_ns > cfg.decision_max_age_ms * 1e6:
            return "decision_expired"
        if now_ns - inputs.force.timestamp_ns > cfg.force_max_age_ms * 1e6:
            return "stale_force"
        if now_ns - inputs.joint_received_ns > cfg.joint_feedback_max_age_ms * 1e6:
            return "stale_joint_feedback"
        if inputs.force.left_N >= cfg.hard_force_limit_left_N:
            return "left_force_hard_limit"
        if inputs.force.right_N >= cfg.hard_force_limit_right_N:
            return "right_force_hard_limit"
        if not cfg.q_min_deg <= inputs.q_measured_deg <= cfg.q_max_deg:
            return "measured_position_outside_limits"
        if not cfg.q_min_deg <= inputs.q_prev_sent_deg <= cfg.q_max_deg:
            return "reference_outside_limits"
        if abs(inputs.q_prev_sent_deg - inputs.q_measured_deg) > cfg.max_command_tracking_error_deg:
            return "command_tracking_error"
        if self._run_start_ns is not None:
            elapsed = (now_ns - self._run_start_ns) / 1e9
            if elapsed < 0:
                return "nonmonotonic_run_time"
            if cfg.max_run_time_s is not None and elapsed >= cfg.max_run_time_s:
                return "run_time_exhausted"
        return None

    def check_inputs(self, inputs: ControlInputs, *, now_ns: int | None = None) -> SafetyDecision | None:
        """Recheck raw feedback age and hard limits immediately before submission."""
        if self.fault_reason is not None:
            return self.latch_fault(self.fault_reason)
        reason = self._input_failure(inputs, inputs.now_ns if now_ns is None else now_ns)
        return self.latch_fault(reason) if reason else None

    def _stall_candidate(self, inputs: ControlInputs) -> tuple[int | None, float, int, int]:
        """Stage progress timers without committing movement or elapsed-time evidence."""
        cfg = self.config
        anchor = inputs.q_measured_deg if self._stall_anchor_deg is None else self._stall_anchor_deg
        if cfg.stall_timeout_s is None:
            return None, anchor, 0, 0
        tolerance = cfg.stall_position_tolerance_deg
        lead = self.close_sign * (inputs.q_prev_sent_deg - inputs.q_measured_deg)
        direction = (1 if lead > 0 else -1) if abs(lead) > tolerance else 0
        since = self._stall_since_ns
        if not direction:
            return None, inputs.q_measured_deg, 0, 0
        if (
            direction != self._stall_direction
            or since is None
            or direction * self.close_sign * (inputs.q_measured_deg - anchor) >= tolerance
        ):
            since, anchor = inputs.now_ns, inputs.q_measured_deg
        blocked = direction if (inputs.now_ns - since) / 1e9 >= cfg.stall_timeout_s else 0
        return since, anchor, direction, blocked

    def apply(
        self, inputs: ControlInputs, proposal: TeacherProposal, *, now_ns: int | None = None
    ) -> SafetyDecision:
        """Intersect restrictions; speed clipping never disables the sole I-driven closure."""
        failure = self.check_inputs(inputs, now_ns=now_ns)
        if failure is not None:
            return failure
        if proposal.inputs != inputs:
            return self.latch_fault("proposal_input_mismatch")
        if proposal.failure_reason is not None or proposal.next_phase == ControlPhase.FAULT:
            return self.latch_fault(proposal.failure_reason or "controller_fault")
        if not proposal.command_enabled:
            return SafetyDecision(None, 0.0)
        try:
            for name in (
                "q_candidate_deg",
                "u_raw_deg_s",
                "delta_raw_deg",
                "delta_i_deg",
                "delta_p_deg",
                "delta_contact_deg",
            ):
                finite_number(name, getattr(proposal, name))
        except ValueError:
            return self.latch_fault("invalid_candidate")
        cfg, dt = self.config, inputs.dt_control_s
        raw_delta = self.close_sign * (proposal.q_candidate_deg - inputs.q_prev_sent_deg)
        if not math.isfinite(raw_delta) or not math.isclose(
            raw_delta / dt, proposal.u_raw_deg_s, abs_tol=1e-8
        ):
            return self.latch_fault("candidate_rate_reference_mismatch")
        if not math.isclose(raw_delta, proposal.delta_raw_deg, abs_tol=1e-8):
            return self.latch_fault("candidate_increment_reference_mismatch")
        if proposal.delta_contact_deg != 0 and (
            proposal.next_phase != ControlPhase.FORCE_TRACK
            or proposal.phase not in (ControlPhase.APPROACH, ControlPhase.FOLLOW, ControlPhase.FORCE_TRACK)
            or proposal.delta_contact_deg > 0
            or not math.isclose(
                proposal.delta_contact_deg,
                -max(0.0, self.close_sign * (inputs.q_prev_sent_deg - inputs.q_measured_deg)),
                abs_tol=1e-8,
            )
        ):
            return self.latch_fault("invalid_contact_alignment")
        if proposal.next_phase == ControlPhase.FORCE_TRACK and not math.isclose(
            proposal.delta_raw_deg,
            proposal.delta_i_deg + proposal.delta_p_deg + proposal.delta_contact_deg,
            abs_tol=1e-8,
        ):
            return self.latch_fault("candidate_ip_increment_mismatch")
        lower, upper = cfg.q_min_deg, cfg.q_max_deg
        if cfg.soft_q_min_deg is not None:
            lower, upper = max(lower, cfg.soft_q_min_deg), min(upper, cfg.soft_q_max_deg)
        work_lower, work_upper = lower, upper
        lower = max(lower, inputs.q_measured_deg - cfg.max_relative_target_deg)
        upper = min(upper, inputs.q_measured_deg + cfg.max_relative_target_deg)
        flags = []
        since, anchor, stall_direction, stalled = self._stall_candidate(inputs)
        blocked_directions = set()
        if inputs.q_prev_sent_deg <= lower or math.isclose(inputs.q_prev_sent_deg, lower, abs_tol=1e-10):
            blocked_directions.add(-self.close_sign)
        if inputs.q_prev_sent_deg >= upper or math.isclose(inputs.q_prev_sent_deg, upper, abs_tol=1e-10):
            blocked_directions.add(self.close_sign)
        if stalled:
            blocked_directions.add(stalled)
            flags.append("stall")
        i_direction = (1 if proposal.delta_i_deg > 0 else -1) if proposal.delta_i_deg else 0
        if proposal.next_phase == ControlPhase.APPROACH:
            if stalled == 1:
                return self.latch_fault("approach_stalled")
            closing_limit = (
                (cfg.soft_q_max_deg if cfg.soft_q_max_deg is not None else cfg.q_max_deg)
                if self.close_sign == 1
                else (cfg.soft_q_min_deg if cfg.soft_q_min_deg is not None else cfg.q_min_deg)
            )
            if self.close_sign * (closing_limit - inputs.q_prev_sent_deg) <= 1e-10:
                return self.latch_fault("approach_position_limit")
        i_blocked = proposal.next_phase == ControlPhase.FORCE_TRACK and i_direction in blocked_directions
        delta = proposal.delta_p_deg + proposal.delta_contact_deg if i_blocked else raw_delta
        if i_blocked:
            flags.append("integral_blocked")
        q_candidate = inputs.q_prev_sent_deg + self.close_sign * delta
        raw_u = delta / dt
        if cfg.max_total_travel_deg is not None:
            remaining = max(0.0, cfg.max_total_travel_deg - self._total_travel_deg)
            if remaining <= 1e-10 and abs(delta) > 1e-10:
                return self.latch_fault("travel_budget_exhausted")
            lower = max(lower, inputs.q_prev_sent_deg - remaining)
            upper = min(upper, inputs.q_prev_sent_deg + remaining)
            if abs(delta) > remaining:
                flags.append("total_travel")
        if lower > upper:
            return self.latch_fault("incompatible_command_limits")
        # Form rate intent from the allowed position, not a request beyond an endpoint.
        # This keeps a valid submitted endpoint stationary even if measured inertia
        # carries the joint into the calibrated margin outside the working range.
        bounded_q = min(upper, max(lower, q_candidate))
        bounded_u = self.close_sign * (bounded_q - inputs.q_prev_sent_deg) / dt
        if cfg.acceleration_limit_deg_s2 is not None and proposal.next_phase in (
            ControlPhase.FOLLOW,
            ControlPhase.FOLLOW_WAIT_CLEAR,
            ControlPhase.POSITION_HOLD,
        ):
            boundary = work_upper if self.close_sign * bounded_u >= 0 else work_lower
            distance = abs(boundary - inputs.q_prev_sent_deg)
            acceleration = cfg.acceleration_limit_deg_s2
            # Reserve this tick's v*dt travel as well as the v^2/(2*a) stopping distance.
            # The equivalent rationalized root avoids cancellation near the endpoint.
            braking_speed = 2 * distance / (math.hypot(dt, math.sqrt(2 * distance / acceleration)) + dt)
            if abs(bounded_u) > braking_speed:
                bounded_u = math.copysign(braking_speed, bounded_u)
                flags.append("endpoint_braking")
        intent_low = max(-cfg.max_open_speed_deg_s, min(0.0, bounded_u))
        intent_high = min(cfg.max_close_speed_deg_s, max(0.0, bounded_u))
        rate_low, rate_high = intent_low, intent_high
        if cfg.acceleration_limit_deg_s2 is not None:
            rate_low = max(rate_low, self._last_u_deg_s - cfg.acceleration_limit_deg_s2 * dt)
            rate_high = min(rate_high, self._last_u_deg_s + cfg.acceleration_limit_deg_s2 * dt)
            if rate_low > rate_high:
                brake_rate = min(intent_high, max(intent_low, self._last_u_deg_s))
                rate_low = rate_high = brake_rate
                flags.append("deceleration_override")
            if abs(raw_u - self._last_u_deg_s) > cfg.acceleration_limit_deg_s2 * dt + 1e-10:
                flags.append("acceleration")
        rate_q = sorted(
            (
                inputs.q_prev_sent_deg + self.close_sign * rate_low * dt,
                inputs.q_prev_sent_deg + self.close_sign * rate_high * dt,
            )
        )
        lower, upper = max(lower, rate_q[0]), min(upper, rate_q[1])
        if lower > upper:
            return self.latch_fault("incompatible_command_limits")
        final_q = min(upper, max(lower, q_candidate))
        if raw_u < -cfg.max_open_speed_deg_s or raw_u > cfg.max_close_speed_deg_s:
            flags.append("speed")
        if proposal.q_candidate_deg < cfg.q_min_deg or proposal.q_candidate_deg > cfg.q_max_deg:
            flags.append("position")
        if (
            cfg.soft_q_min_deg is not None
            and not cfg.soft_q_min_deg <= proposal.q_candidate_deg <= cfg.soft_q_max_deg
        ):
            flags.append("soft_position")
        if abs(proposal.q_candidate_deg - inputs.q_measured_deg) > cfg.max_relative_target_deg:
            flags.append("relative_to_measured")
        self._pending_state = (
            inputs,
            self._run_start_ns if self._run_start_ns is not None else inputs.now_ns,
            since,
            anchor,
            stall_direction,
        )
        return SafetyDecision(
            final_q,
            self.close_sign * (final_q - inputs.q_prev_sent_deg) / dt,
            tuple(flags),
            i_blocked=i_blocked,
            blocked_direction=i_direction if i_blocked else stalled,
        )

    def commit(
        self, q_prev_sent_deg: float, q_sent_deg: float | None, dt_control_s: float, software_success: bool
    ) -> None:
        """Commit rate, travel and progress only after a successful final wire submission."""
        if self.fault_reason is not None:
            return
        if not software_success or q_sent_deg is None:
            self.latch_fault("send_failed_or_unknown")
            return
        try:
            finite_number("q_prev_sent_deg", q_prev_sent_deg)
            finite_number("q_sent_deg", q_sent_deg)
            finite_number("dt_control_s", dt_control_s)
        except ValueError:
            self.latch_fault("invalid_send_result")
            return
        if not 0 < dt_control_s <= self.config.max_dt_s:
            self.latch_fault("dt_out_of_budget")
            return
        self._last_u_deg_s = self.close_sign * (q_sent_deg - q_prev_sent_deg) / dt_control_s
        self._total_travel_deg += abs(q_sent_deg - q_prev_sent_deg)
        if self._pending_state is not None:
            _, self._run_start_ns, self._stall_since_ns, self._stall_anchor_deg, self._stall_direction = (
                self._pending_state
            )
        self._pending_state = None
