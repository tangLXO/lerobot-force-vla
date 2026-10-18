"""Replay gripper travel across the old startup envelope using a fake serial bus."""

import hashlib
import json
from dataclasses import replace

import pytest

from tests.force_control.operator_teleoperation_helpers import (
    JOINT_NAMES,
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.fixture
def travel_scope(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)
    calibration = json.loads(files.calibration_path.read_text(encoding="utf-8"))
    calibration["gripper"].update(range_min=-300.0, range_max=0.0)
    files.calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    files.operator = files.configuration = replace(
        files.operator,
        calibration_sha256=hashlib.sha256(files.calibration_path.read_bytes()).hexdigest(),
        alignment_tolerance_deg=None,
    )
    files.control = replace(
        files.control,
        profiles=files.control.profiles[:-1]
        + (replace(files.control.profiles[-1], minimum_deg=-270.0, maximum_deg=0.0),),
    )
    scope = make_operator_scope(tmp_path, monkeypatch, files=files, connect=False)
    yield scope
    close_operator_scope(scope)


def start_at(scope, position):
    scope.serial.positions[7] = scope.serial.targets[7] = position
    scope.positions["gripper"] = position
    scope.device.connect()
    return scope.device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)


@pytest.mark.parametrize("initial,endpoint", [(-85.2967, -270.0), (-185.3359, -1.0)])
def test_operator_gripper_can_cross_100_degrees_and_reach_working_endpoint(travel_scope, initial, endpoint):
    scope = travel_scope
    snapshot = start_at(scope, initial)
    direction = 1 if endpoint > initial else -1
    for step in range(1, 39):
        target = initial + direction * min(step * 5.0, abs(endpoint - initial))
        goals = dict.fromkeys(JOINT_NAMES, 0.0)
        goals["gripper"] = target
        receipt = scope.device.send_joint_positions_partial(goals, feedback_snapshot=snapshot)
        assert receipt.all_submitted, [(item.name, item.error) for item in receipt.results]
        scope.clock.now += 1_000_000
        snapshot = scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 100_000_000)
        assert snapshot.is_valid, snapshot.error
        # Exercise all-joint queries as well as the command-reply cache path.
        snapshot = scope.device._full_validation_session.read_all(
            scope.clock.now + 100_000_000, active=set(JOINT_NAMES)
        )
        assert scope.device.take_io_evidence()["complete"]
    assert snapshot.get("gripper").position_deg == pytest.approx(endpoint, abs=0.02)
    assert scope.device.startup_receipt["reference_bounds_deg"]["gripper"] == [-270.0, 0.0]
    assert scope.device.failure is None
    cleanup = scope.device.emergency_stop("test_complete")
    assert cleanup["all_joints_disabled_observed"]
    assert cleanup["position_limit_violations"] == []
    assert cleanup["hardware_ack"] == "unknown"


@pytest.mark.parametrize(
    "joint,position,reason", [("shoulder_pan", 110.0, "reference"), ("gripper", -301.0, "calibration")]
)
@pytest.mark.parametrize("path", ["command_reply", "all_joint_query"])
def test_operator_keeps_arm_envelope_and_gripper_hard_calibration(
    travel_scope, joint, position, reason, path
):
    scope = travel_scope
    snapshot = start_at(scope, -85.2967)
    motor_id = JOINT_NAMES.index(joint) + 1
    if path == "command_reply":

        def overtravel(index):
            if index == motor_id:
                scope.serial.positions[index] = position

        scope.serial.on_position = overtravel
        goals = dict.fromkeys(JOINT_NAMES, 0.0)
        goals["gripper"] = -86.0
        receipt = scope.device.send_joint_positions_partial(goals, feedback_snapshot=snapshot)
        assert not receipt.all_submitted
        assert any(reason in (item.error or "") for item in receipt.results)
    else:
        scope.serial.positions[motor_id] = position
        with pytest.raises(RuntimeError, match=reason):
            scope.device._full_validation_session.read_all(
                scope.clock.now + 100_000_000, active=set(JOINT_NAMES)
            )
        scope.device.emergency_stop("test_overtravel")
    assert scope.device.fault_receipt["all_joints_disabled_observed"]
    assert scope.device.fault_receipt["position_limit_violations"]
