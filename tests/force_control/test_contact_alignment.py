"""Contact lead removal uses fresh feedback and successful submissions, without hardware."""

from dataclasses import replace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlPhase
from tests.force_control.helpers import inputs, submit


def approaching(sign=1, **config):
    teacher = ForceTeacher(
        TeacherConfig.fake(
            ki_deg_per_N_s=0, approach_speed_deg_s=2, approach_force_threshold_N=0.5, **config
        ),
        sign,
    )
    teacher.arm()
    teacher.start(0, 0, mode="approach")
    submit(teacher, teacher.propose(inputs(0.1)))
    return teacher


@pytest.mark.parametrize("sign", [-1, 1])
def test_contact_removes_closing_lead_without_clearing_p_or_rewriting_submitted_reference(sign):
    teacher = approaching(sign)
    inp = inputs(0.6, q=sign * 0.8, measured=sign * 0.1)
    proposal = teacher.propose(inp)
    assert proposal.inputs.q_prev_sent_deg == sign * 0.8
    assert proposal.delta_contact_deg == pytest.approx(-0.7)
    assert proposal.delta_p_deg == pytest.approx(-0.05)
    assert proposal.delta_raw_deg == pytest.approx(-0.75)
    assert proposal.q_candidate_deg == pytest.approx(sign * 0.05)
    assert teacher.phase == ControlPhase.APPROACH and not teacher.contact_alignment_pending
    assert teacher.propose(inp) == proposal  # Unsubmitted proposals do not change alignment or force history.
    submit(teacher, proposal)
    assert not teacher.contact_alignment_pending
    next_proposal = teacher.propose(inputs(0.7, q=sign * 0.05, measured=0))
    assert next_proposal.delta_contact_deg == 0  # Ordinary force control does not continually rebase.
    assert next_proposal.delta_p_deg == pytest.approx(-0.01)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("q", [0.05, 0.1])
def test_contact_never_removes_opening_lead_or_adds_closing_alignment(sign, q):
    proposal = approaching(sign).propose(inputs(0.6, q=sign * q, measured=sign * 0.1))
    assert proposal.delta_contact_deg == 0
    assert proposal.q_candidate_deg == pytest.approx(sign * (q - 0.05))


@pytest.mark.parametrize("sign", [-1, 1])
def test_speed_limited_alignment_continues_until_old_lead_is_removed(sign):
    teacher = approaching(sign)
    safety = SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), sign)
    q = sign * 0.8
    for tick in range(7):
        inp = inputs(0.6, q=q, measured=sign * 0.1, now=1_000_000_000 + tick * 20_000_000)
        proposal = teacher.propose(inp)
        assert proposal.delta_contact_deg == pytest.approx(-sign * (q - sign * 0.1))
        assert proposal.delta_p_deg == pytest.approx(-0.05 if tick == 0 else 0)
        decision = safety.apply(inp, proposal)
        assert decision.fault_reason is None
        assert sign * (decision.q_after_safety_deg - q) == pytest.approx(-0.1)
        teacher.commit(proposal, q_sent_deg=decision.q_after_safety_deg, software_success=True)
        safety.commit(q, decision.q_after_safety_deg, inp.dt_control_s, True)
        q = decision.q_after_safety_deg
        assert teacher.contact_alignment_pending == (tick < 6)
    assert q == pytest.approx(sign * 0.1)
    assert teacher.propose(inputs(0.6, q=q, measured=sign * 0.1)).delta_contact_deg == 0


def test_driver_clipping_keeps_alignment_pending_and_wire_rounding_can_complete_it():
    teacher = approaching()
    proposal = teacher.propose(inputs(0.6, q=0.8, measured=0.1))
    teacher.commit(proposal, q_sent_deg=0.7, q_accepted_deg=0.7, software_success=True)
    assert teacher.contact_alignment_pending
    proposal = teacher.propose(inputs(0.6, q=0.7, measured=0.1))
    assert proposal.delta_contact_deg == pytest.approx(-0.6)
    teacher.commit(proposal, q_sent_deg=0.10001, q_accepted_deg=0.1, software_success=True)
    assert not teacher.contact_alignment_pending


def test_failed_alignment_submission_faults_without_accepting_the_new_reference():
    teacher = approaching()
    proposal = teacher.propose(inputs(0.6, q=0.8, measured=0.1))
    teacher.commit(proposal, software_success=False)
    assert teacher.phase == ControlPhase.FAULT and not teacher.contact_alignment_pending


def test_integral_blocking_retains_contact_correction_and_does_not_retire_it_early():
    teacher = approaching()
    teacher.update_config(replace(teacher.config, ki_deg_per_N_s=1000))
    inp = inputs(0.6, q=0.8, measured=0.1)
    proposal = teacher.propose(inp)
    safety = SafetySupervisor(
        SafetyConfig.fake(max_relative_target_deg=0.7, acceleration_limit_deg_s2=None), 1
    )
    decision = safety.apply(inp, proposal)
    assert decision.i_blocked and decision.fault_reason is None
    assert decision.q_after_safety_deg == pytest.approx(0.7)
    teacher.commit(proposal, q_sent_deg=decision.q_after_safety_deg, software_success=True, i_blocked=True)
    assert teacher.contact_alignment_pending
    teacher.freeze(2_000_000_000, 0.1)
    assert not teacher.contact_alignment_pending


def test_new_integral_advance_is_retained_after_old_lead_is_removed():
    teacher = approaching()
    teacher.update_config(replace(teacher.config, ki_deg_per_N_s=4))
    proposal = teacher.propose(inputs(0.6, q=0.8, measured=0.1))
    assert proposal.delta_i_deg == pytest.approx(0.032)
    assert proposal.q_candidate_deg == pytest.approx(0.132)
    submit(teacher, proposal)
    assert not teacher.contact_alignment_pending


@pytest.mark.parametrize("correction", [0.1, -0.8, float("nan")])
def test_safety_rejects_contact_correction_that_is_not_the_actual_closing_lead(correction):
    teacher = approaching()
    inp = inputs(0.6, q=0.8, measured=0.1)
    proposal = replace(teacher.propose(inp), delta_contact_deg=correction)
    decision = SafetySupervisor(SafetyConfig.fake(), 1).apply(inp, proposal)
    assert decision.fault_reason is not None and decision.q_after_safety_deg is None
