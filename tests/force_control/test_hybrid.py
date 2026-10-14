"""Hybrid grasp transitions use fresh leader evidence and successful submissions only."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig, validate_motion_request
from lerobot.force_control.fake_hardware import FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlPhase
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace
from tests.force_control.helpers import inputs, submit


def hybrid(sign=1, **overrides):
    config = TeacherConfig.fake(
        gripper_follow_speed_deg_s=2,
        contact_force_threshold_N=0.5,
        ki_deg_per_N_s=0,
        **overrides,
    )
    controller = ForceTeacher(config, sign)
    controller.arm()
    controller.start(0, 0, mode="hybrid")
    return controller


def observation(force, leader, *, sequence=1, q=0, measured=None, sign=1):
    now = 1_000_000_000 + sequence * 20_000_000
    return replace(
        inputs(force, now=now, q=sign * q, measured=sign * (q if measured is None else measured)),
        leader_gripper_deg=sign * leader,
        leader_raw_gripper_deg=leader + 10,
        leader_sequence=sequence,
        leader_captured_ns=now - 1_000_000,
    )


def contact(controller):
    sign = controller.close_sign
    submit(controller, controller.propose(observation(0.1, 0.8, sign=sign)))
    proposal = controller.propose(observation(0.6, 0.8, sequence=2, q=0.8, measured=0.1, sign=sign))
    submit(controller, proposal)
    return proposal


@pytest.mark.parametrize("sign", [-1, 1])
def test_follow_is_rate_limited_and_contact_reference_commits_with_continuous_p(sign):
    controller = hybrid(sign)
    first = controller.propose(observation(0.1, 0.8, sign=sign))
    assert first.next_phase == ControlPhase.FOLLOW
    assert first.delta_raw_deg == pytest.approx(0.04)
    assert first.delta_i_deg == first.delta_p_deg == 0
    submit(controller, first)
    inp = observation(0.6, 0.8, sequence=2, q=0.8, measured=0.1, sign=sign)
    proposal = controller.propose(inp)
    assert proposal.next_phase == ControlPhase.FORCE_TRACK
    assert proposal.delta_contact_deg == pytest.approx(-0.7)
    assert proposal.delta_p_deg == pytest.approx(-0.05)
    assert proposal.inputs.q_prev_sent_deg == sign * 0.8
    assert proposal.contact_reference.follower_gripper_deg == sign * 0.1
    assert proposal.contact_reference.leader_raw_gripper_deg == 10.8
    assert controller.phase == ControlPhase.FOLLOW and controller.contact_reference is None
    assert controller.propose(inp) == proposal
    decision = SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), sign).apply(inp, proposal)
    assert decision.fault_reason is None
    controller.commit(proposal, q_sent_deg=decision.q_after_safety_deg, software_success=True)
    assert controller.contact_reference == proposal.contact_reference
    assert controller.phase == ControlPhase.FORCE_TRACK and controller.contact_alignment_pending


@pytest.mark.parametrize("sign", [-1, 1])
def test_force_is_latched_until_new_opening_crosses_contact_and_current_follower(sign):
    controller = hybrid(sign)
    contact(controller)
    for sequence, leader, measured in [(3, 0.9, 0.1), (4, 0.2, 0.1), (5, 0.0, -0.2)]:
        proposal = controller.propose(
            observation(0.1, leader, sequence=sequence, measured=measured, sign=sign)
        )
        assert proposal.next_phase == ControlPhase.FORCE_TRACK
        submit(controller, proposal)
    opened = controller.propose(observation(0.7, -0.3, sequence=6, measured=-0.2, sign=sign))
    assert opened.next_phase == ControlPhase.FOLLOW_WAIT_CLEAR
    assert opened.delta_i_deg == opened.delta_p_deg == opened.delta_contact_deg == 0
    assert controller.phase == ControlPhase.FORCE_TRACK
    submit(controller, opened)
    assert controller.phase == ControlPhase.FOLLOW_WAIT_CLEAR
    assert not controller.contact_alignment_pending


def test_duplicate_leader_capture_cannot_release_or_rearm():
    controller = hybrid()
    entry = contact(controller)
    duplicate = replace(entry.inputs, leader_gripper_deg=-0.1, force=replace(entry.inputs.force, left_N=0.8))
    assert controller.propose(duplicate).next_phase == ControlPhase.FORCE_TRACK
    opening = observation(0.8, -0.2, sequence=3)
    submit(controller, controller.propose(opening))
    assert controller.phase == ControlPhase.FOLLOW_WAIT_CLEAR
    duplicate = replace(
        opening, leader_gripper_deg=-0.1, force=replace(opening.force, left_N=0.1, right_N=0.1)
    )
    assert controller.propose(duplicate).next_phase == ControlPhase.FOLLOW_WAIT_CLEAR


@pytest.mark.parametrize("sign", [-1, 1])
def test_release_waits_for_low_force_and_new_closing_intent_before_next_contact(sign):
    controller = hybrid(sign)
    contact(controller)
    submit(controller, controller.propose(observation(0.8, -0.2, sequence=3, sign=sign)))
    reference = controller.contact_reference
    for sequence, force, leader in [(4, 0.8, -0.1), (5, 0.1, -0.1), (6, 0.1, -0.3)]:
        proposal = controller.propose(observation(force, leader, sequence=sequence, sign=sign))
        assert proposal.next_phase == ControlPhase.FOLLOW_WAIT_CLEAR
        submit(controller, proposal)
        assert controller.contact_reference == reference
    closing = controller.propose(observation(0.1, -0.25, sequence=7, sign=sign))
    assert closing.next_phase == ControlPhase.FOLLOW
    assert controller.contact_reference == reference
    submit(controller, closing)
    assert controller.contact_reference is None
    recontact = controller.propose(observation(0.6, 0.1, sequence=8, measured=0.02, sign=sign))
    assert recontact.next_phase == ControlPhase.FORCE_TRACK
    assert recontact.contact_reference.leader_sequence == 8


def test_already_open_leader_must_move_after_contact_before_releasing():
    controller = hybrid()
    inp = observation(0.6, -0.2, measured=0.1)
    submit(controller, controller.propose(inp))
    unchanged = observation(0.6, -0.2, sequence=2, measured=0.1)
    assert controller.propose(unchanged).next_phase == ControlPhase.FORCE_TRACK
    opening = observation(0.6, -0.3, sequence=2, measured=0.1)
    assert controller.propose(opening).next_phase == ControlPhase.FOLLOW_WAIT_CLEAR


@pytest.mark.parametrize("sign", [-1, 1])
def test_follower_recovery_cannot_hand_back_to_a_stationary_or_closing_leader(sign):
    controller = hybrid(sign)
    contact(controller)
    early_opening = observation(0.6, 0.0, sequence=3, measured=-0.2, sign=sign)
    proposal = controller.propose(early_opening)
    assert proposal.next_phase == ControlPhase.FORCE_TRACK
    submit(controller, proposal)
    for sequence, leader in [(4, 0.05), (5, 0.05)]:
        proposal = controller.propose(observation(0.6, leader, sequence=sequence, measured=0.1, sign=sign))
        assert proposal.next_phase == ControlPhase.FORCE_TRACK
        submit(controller, proposal)
    opening = controller.propose(observation(0.6, 0.04, sequence=6, measured=0.1, sign=sign))
    assert opening.next_phase == ControlPhase.FOLLOW_WAIT_CLEAR


def test_failed_submission_never_accepts_contact_or_human_release_reference():
    controller = hybrid()
    proposal = controller.propose(observation(0.6, 0.8, q=0.8, measured=0.1))
    controller.commit(proposal, software_success=False)
    assert controller.phase == ControlPhase.FAULT and controller.contact_reference is None
    controller = hybrid()
    contact(controller)
    proposal = controller.propose(observation(0.8, -0.2, sequence=3))
    controller.commit(proposal, software_success=False)
    assert controller.phase == ControlPhase.FAULT and not controller.hybrid_active


@pytest.mark.parametrize("event", ["freeze", "approach", "release"])
def test_manual_motion_exits_hybrid_and_invalidates_contact_reference(event):
    controller = hybrid()
    contact(controller)
    getattr(controller, event)(2_000_000_000, 0.1)
    assert not controller.hybrid_active and controller.contact_reference is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("leader_gripper_deg", None),
        ("leader_raw_gripper_deg", float("nan")),
        ("leader_sequence", None),
        ("leader_captured_ns", None),
    ],
)
def test_hybrid_requires_valid_leader_evidence(field, value):
    proposal = hybrid().propose(replace(observation(0.1, 0.5), **{field: value}))
    assert proposal.next_phase == ControlPhase.FAULT and not proposal.command_enabled


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now


@pytest.fixture
def rig():
    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    controller = hybrid()
    controller.disable()
    runtime = ForceRuntime(
        device,
        sensor,
        controller,
        SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), 1),
        MemoryTrace(),
        RuntimeConfig(),
        clock=clock,
    )
    runtime.open()
    yield runtime, device, sensor, clock
    runtime.close()


def tick(rig, force=0.1, leader=None, *, metadata=True, dt=0.02):
    runtime, _, sensor, clock = rig
    clock.now += round(dt * 1e9)
    sensor._publish_sample(
        {"left.normal_force": force, "right.normal_force": force},
        clock(),
        arrival_timestamp_ns=clock(),
    )
    if leader is not None:
        extra = (
            {}
            if not metadata
            else {
                "leader_sequence": clock(),
                "leader_request_started_ns": clock(),
                "raw_action_deg": {"gripper.pos": leader + 10},
            }
        )
        runtime.post_arm_intent(
            ArmIntent(
                (("gripper.pos", leader),),
                clock(),
                runtime.epoch,
                clock(),
                200_000_000,
                json.dumps(extra),
            )
        )
    return runtime.tick()


def activate(rig):
    runtime = rig[0]
    runtime.post("arm", version=0, epoch=runtime.epoch)
    tick(rig)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": "hybrid", "position_deg": None})
    return tick(rig, leader=0.8)


def test_runtime_hybrid_preserves_wire_history_and_masks_transitions(rig, monkeypatch):
    runtime, device, *_ = rig
    monkeypatch.setattr(device.plant, "advance", lambda: None)
    row = activate(rig)
    assert row["hybrid_state"] == "FOLLOW" and not row["teacher_active"]
    assert not row["label_valid"] and row["arm_filtered_keys"] == []
    for _ in range(5):
        row = tick(rig, leader=0.8)
    previous = row
    row = tick(rig, force=0.6, leader=0.8)
    assert row["hybrid_state"] == "FORCE_TRACK"
    assert row["hybrid_transition"] == "FOLLOW->FORCE_TRACK"
    assert row["q_prev_sent_deg"] == previous["q_sent_deg"]
    assert row["contact_reference"]["follower_gripper_deg"] == 0
    assert not row["label_valid"] and row["contact_alignment_pending"]
    for _ in range(5):
        row = tick(rig, force=0.6, leader=0.8)
    assert row["label_valid"] and row["hybrid_transition"] is None
    row = tick(rig, force=0.8, leader=-0.2)
    assert row["hybrid_state"] == "FOLLOW_WAIT_CLEAR"
    assert row["hybrid_transition"] == "FORCE_TRACK->FOLLOW_WAIT_CLEAR"
    assert not row["teacher_active"] and not row["label_valid"]
    row = tick(rig, force=0.1, leader=-0.3)
    assert row["hybrid_state"] == "FOLLOW_WAIT_CLEAR"
    row = tick(rig, force=0.1, leader=-0.25)
    assert row["hybrid_state"] == "FOLLOW" and row["contact_reference"] is None


@pytest.mark.parametrize("metadata", [False, True])
def test_runtime_rejects_missing_leader_before_hardware_start(rig, metadata):
    runtime, device, *_ = rig
    startup = Mock()
    device.start_gripper = startup
    runtime.post("arm", version=0, epoch=runtime.epoch)
    tick(rig)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": "hybrid", "position_deg": None})
    row = tick(rig, leader=0.8 if not metadata else None, metadata=metadata)
    startup.assert_not_called()
    assert row["phase"] == "FAULT" and row["submitted_action"] == {}


def test_runtime_requires_fresh_leader_after_explicit_freeze(rig):
    runtime = rig[0]
    activate(rig)
    runtime.post("freeze", version=2, epoch=runtime.epoch)
    row = tick(rig, leader=0.8)
    assert row["phase"] == "POSITION_HOLD" and row["hybrid_state"] is None
    for _ in range(11):
        row = tick(rig)
    assert row["phase"] == "FAULT" and "arm_intent_ttl_expired" in row["failure_reason"]


def test_hybrid_start_uses_shared_configuration_and_rejects_manual_position_arguments():
    controller = hybrid()
    validate_motion_request(controller.config, SafetyConfig.fake(), "hybrid")
    with pytest.raises(ValueError):
        validate_motion_request(TeacherConfig.fake(), SafetyConfig.fake(), "hybrid")
    with pytest.raises(ValueError):
        validate_motion_request(controller.config, SafetyConfig.fake(), "hybrid", position_deg=1)


def test_live_parameters_preserve_contact_reference_and_do_not_rearm_from_low_force():
    controller = hybrid()
    contact(controller)
    reference = controller.contact_reference
    controller.update_config(
        replace(
            controller.config,
            force_target_N=0.9,
            kp_deg_per_N=0.2,
            ki_deg_per_N_s=2,
            gripper_follow_speed_deg_s=1,
            contact_force_threshold_N=0.7,
        )
    )
    assert controller.contact_reference == reference and controller.hybrid_active
    proposal = controller.propose(observation(0.1, 0.8, sequence=3, q=0.1, measured=0.1))
    assert proposal.next_phase == ControlPhase.FORCE_TRACK
    assert proposal.contact_reference == reference and proposal.delta_p_deg == 0
    submit(controller, proposal)
    assert controller.contact_reference == reference


@pytest.mark.parametrize("phase", ["FOLLOW", "FORCE_TRACK", "FOLLOW_WAIT_CLEAR"])
@pytest.mark.parametrize("failure", ["overforce", "stale_leader", "unknown_send"])
def test_hybrid_states_keep_hard_guards_and_never_advance_an_unknown_send(rig, phase, failure):
    runtime, device, _, clock = rig
    activate(rig)
    if phase != "FOLLOW":
        tick(rig, force=0.6, leader=0.8)
    if phase == "FOLLOW_WAIT_CLEAR":
        tick(rig, force=0.6, leader=-0.8)
    assert str(runtime.controller.phase) == phase
    previous = runtime.last_sent
    if failure == "unknown_send":
        device.send_failure = True
    if failure == "stale_leader":
        previous_intent = runtime.arm_mailbox.latest()
        runtime.post_arm_intent(
            replace(
                previous_intent,
                version=clock() + 1,
                ttl_ns=1,
            )
        )
    row = tick(
        rig, force=6 if failure == "overforce" else 0.6, leader=None if failure == "stale_leader" else 0.8
    )
    assert row["phase"] == "FAULT" and not row["label_valid"]
    assert row["hybrid_transition"] is None
    assert row["contact_reference"] is None
    assert runtime.last_sent == previous
    if failure != "unknown_send":
        assert row["submitted_action"] == {}


def test_runtime_hot_parameter_bundle_keeps_committed_contact(rig):
    runtime = rig[0]
    activate(rig)
    row = tick(rig, force=0.6, leader=0.8)
    reference = row["contact_reference"]
    runtime.post(
        "parameters",
        version=2,
        epoch=runtime.epoch,
        value={
            "teacher": replace(runtime.controller.config, force_target_N=0.8, contact_force_threshold_N=0.7),
            "safety": runtime.safety.config,
        },
    )
    row = tick(rig, force=0.1, leader=0.8)
    assert row["phase"] == "FORCE_TRACK" and row["contact_reference"] == reference
    assert row["parameter_update"]["teacher"]["contact_force_threshold_N"] == 0.7
