"""Raw protection, bounded increments and directional anti-windup without hardware."""

from dataclasses import replace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from tests.force_control.helpers import inputs


def make_case(*, sign=1, **config):
    inp = inputs(0, now=1_000_000_000)
    teacher = ForceTeacher(TeacherConfig.fake(kp_deg_per_N=0), sign)
    teacher.arm()
    teacher.start(inp.now_ns, 0)
    return SafetySupervisor(SafetyConfig.fake(**config), sign), inp, teacher.propose(inp)


def with_delta(proposal, inp, delta, sign=1, *, delta_i=None, delta_p=0):
    return replace(
        proposal,
        inputs=inp,
        delta_raw_deg=delta,
        delta_i_deg=delta - delta_p if delta_i is None else delta_i,
        delta_p_deg=delta_p,
        q_candidate_deg=inp.q_prev_sent_deg + sign * delta,
        u_raw_deg_s=delta / inp.dt_control_s,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_speed_acceleration_position_and_lead_constraints_intersect(sign):
    safety, inp, prop = make_case(sign=sign, acceleration_limit_deg_s2=10)
    prop = with_delta(prop, inp, 100, sign)
    decision = safety.apply(inp, prop)
    assert decision.q_after_safety_deg == pytest.approx(sign * 0.004)
    assert set(decision.clip_flags) == {"speed", "acceleration", "position", "relative_to_measured"}
    assert not decision.i_blocked and not decision.safety_override
    safety.commit(0, decision.q_after_safety_deg, inp.dt_control_s, True)
    assert safety.apply(inp, prop).u_limited_deg_s == pytest.approx(0.4)


@pytest.mark.parametrize(
    "left,right,reason", [(5, 0, "left_force_hard_limit"), (0, 5, "right_force_hard_limit")]
)
def test_raw_per_side_hard_force_is_not_hidden_by_mean(left, right, reason):
    safety, inp, prop = make_case()
    inp = replace(inp, force=replace(inp.force, left_N=left, right_N=right))
    decision = safety.apply(inp, replace(prop, inputs=inp))
    assert decision.fault_reason == reason and decision.q_after_safety_deg is None


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"joint_received_ns": None}, "missing_or_invalid_timestamp"),
        ({"joint_sequence": None}, "missing_or_invalid_sequence"),
        ({"joint_valid": False}, "invalid_joint_feedback"),
        ({"joint_received_ns": 800_000_000}, "stale_joint_feedback"),
        ({"joint_received_ns": 1_100_000_000}, "noncausal_joint_feedback"),
        ({"q_measured_deg": 11}, "measured_position_outside_limits"),
        ({"q_prev_sent_deg": 11}, "reference_outside_limits"),
        ({"q_measured_deg": -3}, "command_tracking_error"),
        ({"q_measured_deg": float("nan")}, "invalid_numeric_input"),
        ({"dt_control_s": 0.2}, "dt_out_of_budget"),
    ],
)
def test_mandatory_input_protection_cannot_be_bypassed(updates, reason):
    safety, inp, prop = make_case()
    inp = replace(inp, **updates)
    assert safety.apply(inp, replace(prop, inputs=inp)).fault_reason == reason


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"left_N": float("inf")}, "invalid_numeric_input"),
        ({"timestamp_ns": 800_000_000}, "stale_force"),
        ({"timestamp_ns": 1_000_000_001}, "noncausal_force"),
        ({"arrival_timestamp_ns": 1_000_000_001}, "noncausal_force"),
        ({"is_valid": False}, "invalid_force"),
        ({"unit": "kg"}, "invalid_force"),
    ],
)
def test_raw_force_validity_causality_age_and_units(updates, reason):
    safety, inp, prop = make_case()
    inp = replace(inp, force=replace(inp.force, **updates))
    assert safety.apply(inp, replace(prop, inputs=inp)).fault_reason == reason


def test_pre_send_freshness_latches_and_explicit_reset_changes_epoch():
    safety, inp, prop = make_case()
    assert safety.apply(inp, prop).fault_reason is None
    assert safety.check_inputs(inp, now_ns=1_051_000_000).fault_reason == "decision_expired"
    assert safety.apply(inp, prop).fault_reason == "decision_expired"
    assert safety.epoch == 1
    safety.reset_fault()
    assert safety.epoch == 2 and safety.apply(inp, prop).fault_reason is None


def test_safety_does_not_compute_feedback_from_target_or_error():
    safety, inp, prop = make_case()
    assert safety.apply(inp, replace(prop, target_N=1e6, error_N=-1e9)) == safety.apply(inp, prop)


def test_incompatible_constraints_fault_instead_of_jumping_reference():
    safety, inp, prop = make_case(max_relative_target_deg=0.1)
    inp = replace(inp, q_measured_deg=-1)
    assert safety.apply(inp, replace(prop, inputs=inp)).fault_reason == "incompatible_command_limits"


def test_failed_send_does_not_commit_travel_or_stall_timers():
    safety, inp, prop = make_case()
    safety.apply(inp, prop)
    assert safety._run_start_ns is None and safety._total_travel_deg == 0
    safety.commit(0, None, 0.02, False)
    assert safety._run_start_ns is None and safety._total_travel_deg == 0
    assert safety.apply(inp, prop).fault_reason == "send_failed_or_unknown"


def test_candidate_rate_and_position_share_one_reference():
    safety, inp, prop = make_case()
    assert (
        safety.apply(inp, replace(prop, q_candidate_deg=1)).fault_reason
        == "candidate_rate_reference_mismatch"
    )


@pytest.mark.parametrize("desired_u,maximum_u", [(0, 0), (-2, 0), (0.1, 0.1)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_acceleration_never_extends_closure_or_reverses_requested_opening(desired_u, maximum_u, sign):
    safety, inp, prop = make_case(sign=sign, acceleration_limit_deg_s2=1)
    safety.commit(0, sign * 0.04, 0.02, True)
    prop = with_delta(prop, inp, desired_u * 0.02, sign)
    decision = safety.apply(inp, prop)
    assert decision.u_limited_deg_s <= maximum_u + 1e-10
    assert "deceleration_override" in decision.clip_flags and decision.fault_reason is None


def test_i_only_speed_saturation_still_closes_instead_of_freezing_i():
    safety, inp, prop = make_case(acceleration_limit_deg_s2=None)
    decision = safety.apply(inp, with_delta(prop, inp, 2))
    assert decision.q_after_safety_deg == pytest.approx(0.1)
    assert "speed" in decision.clip_flags and not decision.i_blocked


@pytest.mark.parametrize("sign", [-1, 1])
def test_first_lead_collision_preserves_remaining_increment_then_blocks_only_outward_i(sign):
    safety, inp, prop = make_case(sign=sign, acceleration_limit_deg_s2=None, max_close_speed_deg_s=50)
    inp = replace(inp, q_prev_sent_deg=sign * 0.95)
    decision = safety.apply(inp, with_delta(prop, inp, 0.2, sign))
    assert decision.q_after_safety_deg == pytest.approx(sign)
    assert not decision.i_blocked
    safety.commit(inp.q_prev_sent_deg, decision.q_after_safety_deg, inp.dt_control_s, True)
    inp = replace(inp, q_prev_sent_deg=sign)
    decision = safety.apply(inp, with_delta(prop, inp, 0.1, sign))
    assert decision.q_after_safety_deg == pytest.approx(sign)
    assert decision.i_blocked and decision.blocked_direction == 1
    reverse = safety.apply(inp, with_delta(prop, inp, -0.05, sign))
    assert reverse.q_after_safety_deg == pytest.approx(sign * 0.95) and not reverse.i_blocked


def test_blocked_i_keeps_and_relimits_proportional_measurement_increment():
    safety, inp, prop = make_case(acceleration_limit_deg_s2=None)
    inp = replace(inp, q_prev_sent_deg=1)
    decision = safety.apply(inp, with_delta(prop, inp, 0.05, delta_i=0.1, delta_p=-0.05))
    assert decision.i_blocked and decision.q_after_safety_deg == pytest.approx(0.95)
    decision = safety.apply(inp, with_delta(prop, inp, 0.15, delta_i=0.1, delta_p=0.05))
    assert decision.i_blocked and decision.q_after_safety_deg == pytest.approx(1)


def test_confirmed_stall_blocks_same_direction_but_allows_reverse():
    safety, inp, prop = make_case(acceleration_limit_deg_s2=None, stall_timeout_s=0.1)
    inp = replace(inp, q_prev_sent_deg=0.5)
    first = safety.apply(inp, with_delta(prop, inp, 0.01))
    assert not first.i_blocked
    safety.commit(0.5, first.q_after_safety_deg, 0.02, True)
    later = inputs(0, now=1_100_000_000, q=first.q_after_safety_deg, measured=0)
    blocked = safety.apply(later, with_delta(prop, later, 0.01))
    assert blocked.i_blocked and "stall" in blocked.clip_flags
    assert blocked.q_after_safety_deg == later.q_prev_sent_deg
    reverse = safety.apply(later, with_delta(prop, later, -0.01))
    assert not reverse.i_blocked and reverse.q_after_safety_deg < later.q_prev_sent_deg


def test_actual_progress_prevents_false_stall():
    safety, inp, prop = make_case(acceleration_limit_deg_s2=None, stall_timeout_s=0.1)
    inp = replace(inp, q_prev_sent_deg=0.5)
    first = safety.apply(inp, with_delta(prop, inp, 0.01))
    safety.commit(0.5, first.q_after_safety_deg, 0.02, True)
    progressed = inputs(0, now=1_100_000_000, q=first.q_after_safety_deg, measured=0.02)
    assert not safety.apply(progressed, with_delta(prop, progressed, 0.01)).i_blocked


def test_optional_soft_limits_do_not_expand_calibration_and_preserve_reverse_exit():
    safety, inp, prop = make_case(soft_q_min_deg=-0.1, soft_q_max_deg=0.05, acceleration_limit_deg_s2=None)
    decision = safety.apply(inp, with_delta(prop, inp, 0.1))
    assert decision.q_after_safety_deg == pytest.approx(0.05)
    assert "soft_position" in decision.clip_flags


def test_total_commanded_travel_clips_remainder_then_faults_and_preserves_budget_across_freeze():
    safety, inp, prop = make_case(max_total_travel_deg=0.15, acceleration_limit_deg_s2=None)
    first = safety.apply(inp, with_delta(prop, inp, 0.1))
    safety.commit(0, first.q_after_safety_deg, 0.02, True)
    safety.reset_motion_reference(0.1, 1_010_000_000, preserve_budget=True)
    inp = inputs(0, q=0.1, now=1_020_000_000)
    second = safety.apply(inp, with_delta(prop, inp, 0.1))
    assert second.q_after_safety_deg == pytest.approx(0.15)
    safety.commit(0.1, second.q_after_safety_deg, 0.02, True)
    inp = inputs(0, q=0.15, now=1_040_000_000)
    assert safety.apply(inp, with_delta(prop, inp, 0.1)).fault_reason == "travel_budget_exhausted"


def test_runtime_budget_survives_freeze_and_resume_reference_resets():
    safety, inp, prop = make_case(max_run_time_s=0.2, acceleration_limit_deg_s2=None)
    decision = safety.apply(inp, prop)
    safety.commit(0, decision.q_after_safety_deg, 0.02, True)
    safety.reset_motion_reference(0.01, 1_100_000_000, preserve_budget=True)
    safety.reset_motion_reference(0.01, 1_150_000_000, preserve_budget=True)
    later = inputs(0, now=1_200_000_000)
    assert safety.apply(later, replace(prop, inputs=later)).fault_reason == "run_time_exhausted"
