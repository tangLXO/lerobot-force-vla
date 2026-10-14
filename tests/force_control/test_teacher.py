"""Pure I-P lifecycle and final-submission reference regressions."""

import math
import struct
from dataclasses import replace

import pytest

from lerobot.force_control.config import TeacherConfig
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlPhase
from tests.force_control.helpers import inputs, submit, tracking_teacher


def test_disabled_and_armed_never_generate_motion_and_start_tracks_without_contact():
    teacher = ForceTeacher(TeacherConfig.fake(), 1)
    assert not teacher.propose(inputs(0)).command_enabled
    teacher.arm()
    assert not teacher.propose(inputs(0)).command_enabled
    teacher.start(0, 0)
    proposal = teacher.propose(inputs(0))
    assert proposal.phase == proposal.next_phase == ControlPhase.FORCE_TRACK
    assert proposal.delta_i_deg > 0 and proposal.delta_p_deg == 0


@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_bilateral_mean_and_closing_direction(sign):
    teacher = tracking_teacher(sign, kp_deg_per_N=0)
    proposal = teacher.propose(inputs(0.4, right=0.8))
    assert proposal.error_N == pytest.approx(0.4)
    assert proposal.q_candidate_deg == pytest.approx(sign * 0.032)


def test_proposal_is_repeatable_and_history_commits_only_success():
    teacher = tracking_teacher()
    proposal = teacher.propose(inputs(0.5))
    assert teacher.propose(proposal.inputs) == proposal
    assert teacher._last_force_N is None
    submit(teacher, proposal)
    assert teacher._last_force_N == 0.5
    with pytest.raises(ValueError, match="stale"):
        submit(teacher, proposal)


def test_next_tick_uses_final_wire_reference_without_private_position_windup():
    teacher = tracking_teacher(kp_deg_per_N=0, ki_deg_per_N_s=10)
    first = teacher.propose(inputs(0, q=2))
    assert first.q_candidate_deg == pytest.approx(2.2)
    submit(teacher, first, q=2.01)
    next_proposal = teacher.propose(inputs(0, q=2.01))
    assert next_proposal.q_candidate_deg == pytest.approx(2.21)
    assert not hasattr(teacher, "integral_N_s")


def test_target_change_preserves_measurement_history_and_has_no_proportional_kick():
    teacher = tracking_teacher(kp_deg_per_N=2)
    submit(teacher, teacher.propose(inputs(0.5)))
    pending = teacher.propose(inputs(0.5))
    teacher.set_target(2)
    proposal = teacher.propose(inputs(0.5))
    assert proposal.target_N == 2 and proposal.target_version == 1
    assert proposal.delta_p_deg == 0
    assert proposal.delta_i_deg == pytest.approx(4 * 0.02 * 1.5)
    with pytest.raises(ValueError, match="stale"):
        submit(teacher, pending)
    with pytest.raises(ValueError, match="outside"):
        teacher.set_target(20)


def test_gain_bundle_validation_is_atomic_and_first_updated_p_is_zero():
    teacher = tracking_teacher()
    submit(teacher, teacher.propose(inputs(0.5)))
    before = (teacher.config, teacher._last_force_N)
    with pytest.raises(ValueError):
        teacher.update_config(replace(teacher.config, kp_deg_per_N=float("nan")))
    assert (teacher.config, teacher._last_force_N) == before
    teacher.update_config(replace(teacher.config, kp_deg_per_N=2, ki_deg_per_N_s=3))
    proposal = teacher.propose(inputs(0.9))
    assert proposal.delta_p_deg == 0 and proposal.delta_i_deg == pytest.approx(0.006)


def test_freeze_manual_goal_does_not_accumulate_force_error_and_resume_realigns():
    teacher = tracking_teacher()
    submit(teacher, teacher.propose(inputs(0.5)))
    teacher.freeze(1_000_000_000, 0.3)
    assert teacher.hold_target_deg == 0.3
    teacher.step(0.5)
    assert teacher.hold_target_deg == 0.8
    first = teacher.propose(inputs(0, q=0.3))
    assert first.q_candidate_deg == pytest.approx(0.8)
    assert first.delta_i_deg == first.delta_p_deg == 0
    submit(teacher, first, q=0.4)
    assert teacher.propose(inputs(0, q=0.4)).q_candidate_deg == pytest.approx(0.8)
    teacher.resume(5_000_000_000, 0.45, force_N=0.2)
    proposal = teacher.propose(inputs(0.2, now=5_020_000_000, q=0.45))
    assert teacher.hold_target_deg is None
    assert proposal.delta_p_deg == 0
    assert proposal.delta_i_deg == pytest.approx(4 * 0.02 * 0.8)


@pytest.mark.parametrize("sign", [-1, 1])
def test_manual_step_is_in_closing_coordinates(sign):
    teacher = tracking_teacher(sign)
    teacher.freeze(0, 1)
    teacher.step(0.2)
    assert teacher.hold_target_deg == pytest.approx(1 + sign * 0.2)
    teacher.resume(0, 1)
    with pytest.raises(RuntimeError, match="POSITION_HOLD"):
        teacher.step(0.1)


@pytest.mark.parametrize("success,override", [(False, False), (True, True)])
def test_failed_submission_or_safety_override_faults_without_accepting_history(success, override):
    teacher = tracking_teacher()
    proposal = teacher.propose(inputs())
    teacher.commit(
        proposal, q_sent_deg=proposal.q_candidate_deg, software_success=success, safety_override=override
    )
    assert teacher.phase == ControlPhase.FAULT and teacher._last_force_N is None
    with pytest.raises(RuntimeError):
        teacher.arm()
    with pytest.raises(RuntimeError):
        teacher.disable()
    teacher.reset_fault()
    assert teacher.phase == ControlPhase.DISABLED


def test_zero_target_is_not_release_and_explicit_release_has_bounded_final_step():
    teacher = tracking_teacher()
    teacher.set_target(0)
    assert teacher.propose(inputs()).next_phase == ControlPhase.FORCE_TRACK
    teacher.release(1_000_000_000, 0)
    proposal = teacher.propose(inputs(q=-1.99))
    assert proposal.q_candidate_deg == pytest.approx(-2)
    submit(teacher, proposal)
    proposal = teacher.propose(inputs(q=-2, now=1_020_000_000))
    assert proposal.next_phase == ControlPhase.DISABLED and not proposal.command_enabled
    teacher.commit(proposal)
    assert teacher.phase == ControlPhase.DISABLED


def test_release_timeout_latches_fault_instead_of_continuing_to_open():
    teacher = tracking_teacher(max_release_time_s=0.1)
    teacher.release(0, 0)
    proposal = teacher.propose(inputs(now=100_000_000))
    assert proposal.failure_reason == "release_budget_exhausted"


@pytest.mark.parametrize("sign", [-1, 1])
def test_release_reaches_representable_endpoint_despite_travel_subtraction_roundoff(sign):
    teacher = tracking_teacher(sign, release_travel_deg=0.03, max_release_time_s=0.8)
    initial = sign * 0.282
    endpoint = initial - sign * teacher.config.release_travel_deg
    # This used to demand another sub-ULP step until the release timed out.
    assert -sign * (endpoint - initial) < teacher.config.release_travel_deg
    teacher.release(0, initial)
    proposal = teacher.propose(inputs(q=initial, now=0))
    assert proposal.q_candidate_deg == endpoint
    submit(teacher, proposal, q=endpoint)
    complete = teacher.propose(inputs(q=endpoint, now=800_000_000))
    assert complete.next_phase == ControlPhase.DISABLED
    assert complete.failure_reason is None and not complete.command_enabled


@pytest.mark.parametrize("event", ["start", "resume"])
def test_first_force_submission_masks_p_even_when_force_changes_after_reference(event):
    teacher = ForceTeacher(TeacherConfig.fake(kp_deg_per_N=2), 1)
    teacher.arm()
    teacher.start(0, 0, force_N=0.2)
    if event == "resume":
        submit(teacher, teacher.propose(inputs(0.5)))
        teacher.freeze(1_000_000_000, 0)
        teacher.resume(1_010_000_000, 0, force_N=0.2)
    first = teacher.propose(inputs(0.8))
    assert first.delta_p_deg == 0
    submit(teacher, first)
    assert teacher.propose(inputs(0.9)).delta_p_deg == pytest.approx(-0.2)


@pytest.mark.parametrize("sign", [-1, 1])
def test_release_credits_accepted_float32_endpoint_only_after_wire_stops_progressing(sign):
    teacher = tracking_teacher(sign, release_travel_deg=0.03, max_release_time_s=0.8)
    initial = sign * 0.1
    endpoint = initial - sign * teacher.config.release_travel_deg
    wire = math.degrees(struct.unpack("<f", struct.pack("<f", math.radians(endpoint)))[0])
    assert sign * (wire - endpoint) > 0  # Encoded endpoint is on the closing side.
    teacher.release(0, initial)
    first = teacher.propose(inputs(q=initial, now=0))
    teacher.commit(first, q_sent_deg=wire, q_accepted_deg=endpoint, software_success=True)
    assert not teacher._release_endpoint_encoded  # First wire command made progress.
    repeated = teacher.propose(inputs(q=wire, now=20_000_000))
    assert repeated.q_candidate_deg == endpoint
    teacher.commit(repeated, q_sent_deg=wire, q_accepted_deg=endpoint, software_success=True)
    assert teacher._release_endpoint_encoded
    complete = teacher.propose(inputs(q=wire, now=800_000_000))
    assert complete.next_phase == ControlPhase.DISABLED and complete.failure_reason is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("accepted", [None, 0.08])
def test_release_does_not_credit_missing_acceptance_or_safety_driver_clipping(sign, accepted):
    teacher = tracking_teacher(sign, release_travel_deg=0.03, max_release_time_s=0.8)
    teacher.release(0, sign * 0.1)
    # An earlier safety or driver boundary accepted only .08, outside the .07 endpoint.
    blocked_wire = sign * 0.08
    proposal = teacher.propose(inputs(q=blocked_wire, now=20_000_000))
    assert proposal.q_candidate_deg == sign * 0.07
    teacher.commit(
        proposal,
        q_sent_deg=blocked_wire,
        q_accepted_deg=None if accepted is None else sign * accepted,
        software_success=True,
    )
    assert not teacher._release_endpoint_encoded
    incomplete = teacher.propose(inputs(q=blocked_wire, now=800_000_000))
    assert incomplete.failure_reason == "release_budget_exhausted"


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), True])
def test_invalid_dt_never_generates_motion(dt):
    proposal = tracking_teacher().propose(inputs(dt=dt))
    assert proposal.next_phase == ControlPhase.FAULT and not proposal.command_enabled


def test_large_dt_is_safety_responsibility_without_duplicate_teacher_budget():
    proposal = tracking_teacher().propose(inputs(dt=1))
    assert proposal.command_enabled


def test_arithmetic_overflow_is_a_faulted_candidate():
    teacher = tracking_teacher(ki_deg_per_N_s=1e308)
    proposal = teacher.propose(inputs(-100))
    assert proposal.failure_reason == "nonfinite_control_arithmetic"
    assert not proposal.command_enabled
