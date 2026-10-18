"""Independent lifecycle regressions against fake serial; no physical commands."""

# The fake serial boundary still needs the optional serial module to construct the adapter.
# ruff: noqa: E402

from dataclasses import replace

import pytest

pytest.importorskip("serial")

from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable, encode_enable


@pytest.fixture
def full_scope(tmp_path, monkeypatch):
    scope = make_full_scope(tmp_path, monkeypatch)
    yield tuple(scope)
    close_full_scope(scope)


from tests.force_control.full_arm_backend_helpers import close_full_scope, make_full_scope


def disable_ids(packets):
    known = {encode_disable(index): index for index in range(1, 8)}
    return [known[packet] for packet in packets if packet in known]


def test_leader_moving_during_disabled_preload_must_refuse_before_first_enable(full_scope):
    device, serial, _, leader, positions, _ = full_scope

    def move_after_initial_alignment(motor_id):
        if motor_id == 1 and serial.position_counts[1] == 2:
            positions["shoulder_pan"] = 10.0

    serial.on_position = move_after_initial_alignment
    with pytest.raises(RuntimeError, match="align"):
        device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    assert not any(packet == encode_enable(index) for index in range(1, 8) for packet in serial.writes)


def test_newer_preload_feedback_cannot_be_ignored_in_favor_of_an_older_aligned_snapshot(full_scope):
    device, serial, _, leader, _, _ = full_scope

    def follower_moves_while_disabled(motor_id):
        if motor_id == 1 and serial.position_counts[1] == 2:
            serial.positions[1] = 6.0

    serial.on_position = follower_moves_while_disabled
    with pytest.raises(RuntimeError, match="align"):
        device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    assert not any(packet == encode_enable(index) for index in range(1, 8) for packet in serial.writes)


def test_preload_target_must_fit_latest_known_position_even_inside_alignment_tolerance(full_scope):
    device, serial, _, leader, _, _ = full_scope

    def follower_moves_inside_alignment_but_outside_command_delta(motor_id):
        if motor_id == 1 and serial.position_counts[1] == 2:
            serial.positions[1] = 1.0

    serial.on_position = follower_moves_inside_alignment_but_outside_command_delta
    with pytest.raises(RuntimeError, match="reference|preload|target|align"):
        device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    assert not any(packet == encode_enable(index) for index in range(1, 8) for packet in serial.writes)


def test_diagnostic_session_retains_gripper_startup_reference_envelope(full_scope):
    device, _, _, leader, _, _ = full_scope
    device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    session = device._full_validation_session
    reference = session.references["gripper"]
    envelope = session.configuration.reference_envelope_deg
    assert session.uses_reference_envelope("gripper")
    assert not session.reference_envelope_exceeded("gripper", reference)
    assert session.reference_envelope_exceeded("gripper", reference + envelope + 0.1)


@pytest.mark.parametrize(
    "corruption",
    [
        "invalid_control_profile",
        "invalid_diagnostics_capacity",
        "replaced_capability",
        "calibration",
        "evidence",
        "failed_force_guard",
    ],
)
def test_fault_disable_uses_pinned_cleanup_even_when_normal_validation_is_corrupted(full_scope, corruption):
    device, serial, files, leader, _, _ = full_scope
    device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    session = device._full_validation_session
    if corruption == "invalid_control_profile":
        profile = replace(device.config.profiles[0], max_relative_deg=-1)
        device.config = replace(device.config, profiles=(profile, *device.config.profiles[1:]))
    elif corruption == "invalid_diagnostics_capacity":
        device.config = replace(device.config, io_event_capacity=None)
    elif corruption == "replaced_capability":
        session.configuration = replace(session.configuration, reference_envelope_deg=2.0)
    elif corruption in ("calibration", "evidence"):
        from pathlib import Path

        path = Path(
            files.validation.calibration_file
            if corruption == "calibration"
            else files.validation.micro_trials[0].result_path
        )
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        session.guard = lambda: (_ for _ in ()).throw(RuntimeError("Injected stale/overforce input"))
    before = len(serial.writes)
    before_calls = len(serial.calls)
    receipt = device.emergency_stop("independent_corruption_test")
    assert disable_ids(serial.writes[before:]) == list(range(1, 8))
    io_calls = [name for name, _ in serial.calls[before_calls:] if name in {"write", "read"}]
    assert io_calls[:7] == ["write"] * 7
    assert all(status == 0 for status in serial.statuses.values())
    assert receipt["all_joints_disabled_observed"] is True
    count = len(serial.writes)
    device.emergency_stop("repeat cleanup must not resend")
    assert len(serial.writes) == count


def test_failed_first_disable_does_not_skip_the_six_later_actuator_attempts(full_scope):
    device, serial, _, leader, _, _ = full_scope
    device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    serial.failed_disables.add(1)
    before = len(serial.writes)
    receipt = device.emergency_stop("first_disable_failed")
    assert disable_ids(serial.writes[before:]) == list(range(1, 8))
    assert all(serial.statuses[index] == 0 for index in range(2, 8))
    assert receipt["status"] == "unknown" and receipt["error"]


def test_first_disable_cannot_take_the_whole_fault_budget_from_later_six_writes(full_scope):
    device, serial, files, leader, _, _ = full_scope
    device.start_full_arm_validation(guard=lambda: None, leader_sample=leader)
    serial.write_timeout = None
    serial.block_first_disable = True
    before = len(serial.writes)
    receipt = device.emergency_stop("blocking_first_disable")
    assert disable_ids(serial.writes[before:]) == list(range(1, 8))
    assert len(serial.disable_write_timeouts) == 7
    assert all(0 < value <= files.control.send_budget_s for value in serial.disable_write_timeouts)
    assert receipt["status"] == "unknown" and receipt["error"]
