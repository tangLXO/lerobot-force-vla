"""Motion phases and safety gates with synthetic forces; no physical hardware."""

from dataclasses import replace

import pytest

from lerobot.force_control.config import (
    SafetyConfig,
    TeacherConfig,
    validate_motion_request,
    validate_profile,
)
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlPhase
from tests.force_control.helpers import inputs, submit


def approach_teacher(sign=1):
    teacher = ForceTeacher(
        TeacherConfig.fake(gripper_follow_speed_deg_s=2, contact_force_threshold_N=0.5), sign
    )
    teacher.arm()
    teacher.start(0, 0, mode="approach")
    return teacher


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.037])
def test_closure_uses_requested_speed_actual_dt_and_only_submitted_reference(sign, dt):
    teacher = approach_teacher(sign)
    p = teacher.propose(inputs(0.1, dt=dt, q=sign * 0.8, measured=sign * 0.1))
    assert p.q_candidate_deg == pytest.approx(sign * (0.8 + 2 * dt))
    assert p.delta_i_deg == p.delta_p_deg == 0
    assert teacher.propose(p.inputs) == p
    submit(teacher, p, q=sign * 0.81)
    p = teacher.propose(inputs(0.1, dt=dt, q=sign * 0.81))
    assert p.q_candidate_deg == pytest.approx(sign * (0.81 + 2 * dt))


def test_mean_force_switch_is_one_way_and_preserves_contact_p_increment():
    teacher = approach_teacher()
    submit(teacher, teacher.propose(inputs(0.1)))
    p = teacher.propose(inputs(0.2, right=0.8))  # Mean exactly equals threshold.
    assert teacher.phase == ControlPhase.APPROACH  # Proposal is pure.
    assert p.next_phase == ControlPhase.FORCE_TRACK
    assert p.delta_p_deg == pytest.approx(-0.1 * (0.5 - 0.1))
    assert p.delta_i_deg == pytest.approx(4 * 0.02 * 0.5)
    assert p.delta_raw_deg == pytest.approx(0)
    submit(teacher, p)
    p = teacher.propose(inputs(0.1, q=p.q_candidate_deg))
    assert p.phase == p.next_phase == ControlPhase.FORCE_TRACK
    assert p.delta_i_deg == pytest.approx(4 * 0.02 * 0.9)
    assert p.delta_p_deg == pytest.approx(-0.1 * (0.1 - 0.5))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.037])
def test_approach_p_slows_before_threshold_and_caps_rebound_closure(sign, dt):
    teacher = approach_teacher(sign)
    first = teacher.propose(inputs(0.1, dt=dt))
    submit(teacher, first)
    rising = teacher.propose(inputs(0.3, dt=dt, q=first.q_candidate_deg))
    assert rising.next_phase == ControlPhase.APPROACH
    assert rising.delta_i_deg == 0
    assert rising.delta_p_deg == pytest.approx(-0.02)
    assert rising.delta_raw_deg == pytest.approx(2 * dt - 0.02)
    assert sign * (rising.q_candidate_deg - first.q_candidate_deg) == pytest.approx(2 * dt - 0.02)
    submit(teacher, rising)
    falling = teacher.propose(inputs(0.1, dt=dt, q=rising.q_candidate_deg))
    assert falling.delta_p_deg == pytest.approx(0.02)
    assert falling.delta_raw_deg == pytest.approx(2 * dt)
    assert falling.u_raw_deg_s == pytest.approx(2)
    submit(teacher, falling)
    steady = teacher.propose(inputs(0.1, dt=dt, q=falling.q_candidate_deg))
    assert steady.delta_p_deg == 0
    assert steady.u_raw_deg_s == pytest.approx(2)


@pytest.mark.parametrize("sign", [-1, 1])
def test_approach_p_can_open_but_existing_opening_limit_remains(sign):
    teacher = approach_teacher(sign)
    teacher.update_config(replace(teacher.config, kp_deg_per_N=1))
    first = teacher.propose(inputs(0.1))
    submit(teacher, first)
    inp = inputs(0.3, q=first.q_candidate_deg)
    proposal = teacher.propose(inp)
    assert proposal.next_phase == ControlPhase.APPROACH
    assert proposal.delta_raw_deg == pytest.approx(-0.16)
    safety = SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), sign)
    decision = safety.apply(inp, proposal)
    assert decision.fault_reason is None and "speed" in decision.clip_flags
    assert decision.u_limited_deg_s == pytest.approx(-5)
    assert sign * (decision.q_after_safety_deg - inp.q_prev_sent_deg) == pytest.approx(-0.1)


def test_proposals_do_not_consume_approach_p_history_or_accumulate_unsent_positions():
    teacher = approach_teacher()
    submit(teacher, teacher.propose(inputs(0.1)), q=0.03)
    unsubmitted = teacher.propose(inputs(0.2, q=0.03))
    assert unsubmitted.delta_p_deg == pytest.approx(-0.01)
    newer = teacher.propose(inputs(0.4, q=0.03))
    assert newer.delta_p_deg == pytest.approx(-0.03)
    assert newer.q_candidate_deg == pytest.approx(0.04)
    submit(teacher, newer, q=0.035)  # Owner reports its final limited/encoded position.
    repeated = teacher.propose(inputs(0.4, q=0.035))
    assert repeated.delta_p_deg == 0
    assert repeated.q_candidate_deg == pytest.approx(0.075)


def test_position_hold_and_release_ignore_force_history_after_approach():
    teacher = approach_teacher()
    submit(teacher, teacher.propose(inputs(0.1)))
    teacher.move_to(1, 0.04, -1)
    submit(teacher, teacher.propose(inputs(0.3, q=0.04)))
    held = teacher.propose(inputs(0.1, q=-1))
    assert held.q_candidate_deg == -1 and held.delta_p_deg == 0
    teacher.release(1_000_000_000, -1)
    released = teacher.propose(inputs(0.4, q=-1, now=1_020_000_000))
    assert released.delta_p_deg == 0
    assert released.delta_raw_deg == pytest.approx(-teacher.config.release_speed_deg_s * 0.02)


def test_already_contacting_starts_ip_without_one_constant_speed_step():
    teacher = approach_teacher()
    p = teacher.propose(inputs(1.0))
    assert p.next_phase == ControlPhase.FORCE_TRACK
    assert p.q_candidate_deg == 0


def test_failed_transition_submission_faults_and_never_enters_force_tracking():
    teacher = approach_teacher()
    teacher.commit(teacher.propose(inputs(0.5)), software_success=False)
    assert teacher.phase == ControlPhase.FAULT
    with pytest.raises(RuntimeError):
        teacher.approach(1, 0)
    with pytest.raises(RuntimeError):
        teacher.move_to(1, 0, -1)


def test_live_threshold_change_uses_current_sample_and_preserves_one_way_transition():
    teacher = approach_teacher()
    submit(teacher, teacher.propose(inputs(0.2)))
    teacher.update_config(replace(teacher.config, contact_force_threshold_N=0.2))
    proposal = teacher.propose(inputs(0.2))
    assert proposal.next_phase == ControlPhase.FORCE_TRACK and proposal.delta_p_deg == 0
    submit(teacher, proposal)
    teacher.update_config(replace(teacher.config, contact_force_threshold_N=0.8))
    assert teacher.propose(inputs(0.2)).next_phase == ControlPhase.FORCE_TRACK


def test_stale_force_cannot_trigger_transition_or_motion():
    teacher = approach_teacher()
    safety = SafetySupervisor(SafetyConfig.fake(), 1)
    inp = inputs(0.5)
    inp = replace(inp, force=replace(inp.force, timestamp_ns=inp.now_ns - 101_000_000))
    decision = safety.apply(inp, teacher.propose(inp))
    assert decision.fault_reason is not None and decision.q_after_safety_deg is None


def test_position_preempts_approach_and_holds_one_absolute_goal():
    teacher = approach_teacher(-1)
    teacher.move_to(1, 0, 1.5)
    p = teacher.propose(inputs(0.1))
    assert p.phase == ControlPhase.POSITION_HOLD
    assert p.q_candidate_deg == 1.5
    submit(teacher, p, q=0.02)
    assert teacher.propose(inputs(0.9, q=0.02)).q_candidate_deg == 1.5
    teacher.approach(2, 0.02)
    assert teacher.phase == ControlPhase.APPROACH
    assert teacher.hold_target_deg is None


@pytest.mark.parametrize(
    "override",
    [
        {"gripper_follow_speed_deg_s": 0},
        {"gripper_follow_speed_deg_s": float("nan")},
        {"gripper_follow_speed_deg_s": 6},
        {"contact_force_threshold_N": None},
        {"contact_force_threshold_N": 0},
        {"contact_force_threshold_N": 5},
    ],
)
def test_approach_configuration_rejects_invalid_or_unsafe_values(override):
    cfg = replace(approach_teacher().config, **override)
    with pytest.raises(ValueError):
        validate_profile(cfg, SafetyConfig.fake(), 1)


@pytest.mark.parametrize("position", [-11, 11, float("inf"), True])
def test_position_rejected_outside_calibration_and_for_invalid_numbers(position):
    with pytest.raises(ValueError):
        validate_motion_request(TeacherConfig.fake(), SafetyConfig.fake(), "position", position)


def test_position_soft_bounds_and_unconfigured_approach_are_rejected():
    with pytest.raises(ValueError):
        validate_motion_request(TeacherConfig.fake(), SafetyConfig.fake(), "approach")
    with pytest.raises(ValueError):
        validate_motion_request(
            TeacherConfig.fake(), SafetyConfig.fake(soft_q_min_deg=-1, soft_q_max_deg=1), "position", 2
        )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("target", [-1.0, 1.0])
@pytest.mark.parametrize("dt", [0.01, 0.037])
def test_position_speed_uses_actual_dt_and_absolute_goal_without_overshoot(sign, target, dt):
    teacher = ForceTeacher(TeacherConfig.fake(), sign)
    teacher.arm()
    teacher.start(0, 0, mode="position", position_deg=target, position_speed_deg_s=2)
    first = teacher.propose(inputs(0.1, dt=dt))
    assert first.q_candidate_deg == pytest.approx(target * 2 * dt)
    assert first.delta_p_deg == first.delta_i_deg == 0
    submit(teacher, first, q=target * 0.99)
    final = teacher.propose(inputs(0.3, dt=dt, q=target * 0.99))
    assert final.q_candidate_deg == pytest.approx(target)
    submit(teacher, final)
    assert teacher.propose(inputs(0.2, q=target)).delta_raw_deg == 0
    teacher.freeze(2_000_000_000, target)
    assert teacher.hold_speed_deg_s is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_position_speed_still_obeys_directional_safety_rate(sign, direction):
    teacher = approach_teacher(sign)
    teacher.move_to(0, 0, sign * direction, position_speed_deg_s=20)
    safety = SafetySupervisor(
        SafetyConfig.fake(max_close_speed_deg_s=2, max_open_speed_deg_s=3, acceleration_limit_deg_s2=None),
        sign,
    )
    inp = inputs(0.1, dt=0.02)
    proposal = teacher.propose(inp)
    decision = safety.apply(inp, proposal)
    assert decision.fault_reason is None and "speed" in decision.clip_flags
    assert decision.u_limited_deg_s == pytest.approx(2 if direction == 1 else -3)
    submit(teacher, proposal, q=decision.q_after_safety_deg)
    teacher.approach(2_000_000_000, decision.q_after_safety_deg)
    assert teacher.hold_speed_deg_s is None


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf"), True])
def test_invalid_position_speed_is_rejected_before_changing_phase(speed):
    teacher = approach_teacher()
    with pytest.raises(ValueError):
        teacher.move_to(0, 0, -1, position_speed_deg_s=speed)
    assert teacher.phase == ControlPhase.APPROACH and teacher.hold_target_deg is None
    with pytest.raises(ValueError):
        validate_motion_request(teacher.config, SafetyConfig.fake(), "position", -1, speed)


@pytest.mark.parametrize("force,reason", [(5, "left_force_hard_limit"), (0.1, "approach_position_limit")])
def test_approach_hard_force_and_end_of_travel_still_stop(force, reason):
    teacher = approach_teacher()
    safety = SafetySupervisor(SafetyConfig.fake(), 1)
    inp = inputs(force, q=10)
    assert safety.apply(inp, teacher.propose(inp)).fault_reason == reason


def test_approach_stall_cannot_keep_pushing_forever():
    teacher = approach_teacher()
    safety = SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), 1)
    for tick in range(40):
        inp = inputs(0.1, q=min(tick * 0.04, 1), measured=0, now=1_000_000_000 + tick * 20_000_000)
        decision = safety.apply(inp, teacher.propose(inp))
        if decision.fault_reason:
            break
        safety.commit(inp.q_prev_sent_deg, decision.q_after_safety_deg, inp.dt_control_s, True)
    assert decision.fault_reason == "approach_stalled"
