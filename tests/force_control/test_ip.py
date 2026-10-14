"""Numeric I-P tests distinguish measurement feedback from derivative action."""

import pytest

from tests.force_control.helpers import inputs, submit, tracking_teacher


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.037])
@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_ip_increment_uses_actual_dt_once(dt, sign):
    teacher = tracking_teacher(sign, kp_deg_per_N=2, ki_deg_per_N_s=0.4)
    first = teacher.propose(inputs(0.5))
    submit(teacher, first)
    proposal = teacher.propose(inputs(0.7, dt=dt, q=sign * 1.2))
    assert proposal.delta_i_deg == pytest.approx(0.4 * dt * 0.3)
    assert proposal.delta_p_deg == pytest.approx(-2 * 0.2)
    assert proposal.delta_raw_deg == pytest.approx(0.4 * dt * 0.3 - 2 * 0.2)
    assert proposal.q_candidate_deg == pytest.approx(sign * (1.2 + proposal.delta_raw_deg))
    assert proposal.u_raw_deg_s == pytest.approx(proposal.delta_raw_deg / dt)


def test_measurement_proportional_increment_is_not_multiplied_by_dt_or_integrated_twice():
    teacher = tracking_teacher(kp_deg_per_N=2, ki_deg_per_N_s=0)
    submit(teacher, teacher.propose(inputs(0.5)))
    fast = teacher.propose(inputs(0.7, dt=0.01))
    slow = teacher.propose(inputs(0.7, dt=0.05))
    assert fast.q_candidate_deg == slow.q_candidate_deg == pytest.approx(-0.4)


def test_i_only_constant_error_has_linear_position_growth_without_extra_integral():
    teacher = tracking_teacher(kp_deg_per_N=0, ki_deg_per_N_s=2)
    q = 0
    for index in range(10):
        proposal = teacher.propose(inputs(0.5, q=q, now=1_000_000_000 + index * 20_000_000))
        submit(teacher, proposal)
        q = proposal.q_candidate_deg
    assert q == pytest.approx(2 * 0.5 * 10 * 0.02)


def test_deadband_blocks_only_i_and_keeps_measurement_feedback():
    teacher = tracking_teacher(kp_deg_per_N=2, ki_deg_per_N_s=3, integral_deadband_N=0.1)
    submit(teacher, teacher.propose(inputs(0.9)))
    proposal = teacher.propose(inputs(0.95))
    assert proposal.delta_i_deg == 0
    assert proposal.delta_p_deg == pytest.approx(-0.1)
    assert proposal.error_N == pytest.approx(0.05)


def test_raw_signed_force_is_neither_clipped_nor_filtered():
    teacher = tracking_teacher(kp_deg_per_N=1, ki_deg_per_N_s=1)
    submit(teacher, teacher.propose(inputs(-0.2)))
    proposal = teacher.propose(inputs(0.8, right=1.2))
    assert proposal.error_N == 0
    assert proposal.delta_i_deg == 0
    assert proposal.delta_p_deg == pytest.approx(-1.2)
