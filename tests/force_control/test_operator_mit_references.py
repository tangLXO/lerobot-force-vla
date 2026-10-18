"""Explicit active MIT references remain neutral throughout sequential startup."""

from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import encode_enable
from tests.force_control.operator_teleoperation_helpers import (
    JOINT_NAMES,
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.mark.parametrize("velocity,torque", [(12.0, 0.4), (-12.0, -0.4)])
def test_saved_references_only_apply_to_active_frames_after_neutral_startup(
    tmp_path, monkeypatch, velocity, torque
):
    files = make_operator_files(tmp_path)
    profile = replace(files.control.profiles[0], mit_velocity_deg_s=velocity, mit_torque_Nm=torque)
    files.control = replace(files.control, profiles=(profile, *files.control.profiles[1:]))
    scope = make_operator_scope(tmp_path, monkeypatch, files=files)
    try:
        snapshot = scope.device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)
        commands = scope.device.startup_receipt["commands"]
        for item in commands:
            if item["name"] == "shoulder_pan":
                packet = bytes.fromhex(item["packet_hex"])
                payload = packet[21:29]
                if payload[:7] != b"\xff" * 7:
                    assert ((payload[2] << 4) | (payload[3] >> 4)) == 2047
                    assert (((payload[6] & 15) << 8) | payload[7]) == 2047
        before = len(scope.serial.writes)
        result = scope.device.send_joint_positions_partial(
            dict.fromkeys(JOINT_NAMES, 0.0), feedback_snapshot=snapshot
        )
        assert result.all_submitted
        packet = scope.serial.writes[before]
        payload = packet[21:29]
        v = (payload[2] << 4) | (payload[3] >> 4)
        t = ((payload[6] & 15) << 8) | payload[7]
        assert (v > 2047 and t > 2047) if velocity > 0 else (v < 2047 and t < 2047)
    finally:
        close_operator_scope(scope)


@pytest.mark.parametrize("field,value", [("mit_velocity_deg_s", 6000), ("mit_torque_Nm", 1000)])
def test_unrepresentable_active_reference_refuses_before_any_enable(tmp_path, monkeypatch, field, value):
    files = make_operator_files(tmp_path)
    profile = replace(files.control.profiles[0], **{field: value})
    files.control = replace(files.control, profiles=(profile, *files.control.profiles[1:]))
    scope = make_operator_scope(tmp_path, monkeypatch, files=files)
    try:
        with pytest.raises(ValueError, match="verified device mapping"):
            scope.device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)
        assert not any(
            packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8)
        )
        assert scope.device.fault_receipt["all_joints_disabled_observed"]
    finally:
        close_operator_scope(scope)
