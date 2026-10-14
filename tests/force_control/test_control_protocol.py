"""Pure FORCE_POS protocol tests; integer speed input never touches hardware."""

import math
import struct
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import (
    B601JointCommandProfile,
    encode_position,
    force_pos_wire_codes,
)
from tests.force_control.hardware_helpers import profile


def test_every_integer_velocity_code_reaches_the_packet_unchanged():
    for code in range(1, 10001):
        configured = replace(profile(), velocity_deg_s=None, velocity_wire_code=code)
        encoded = encode_position(7, 0, 0, configured, (8, 100, 9))
        assert struct.unpack("<H", encoded.packet[25:27])[0] == code
        assert configured.velocity_limit_deg_s == math.degrees(code / 100)
        assert force_pos_wire_codes(torque_ratio=0.07, velocity_wire_code=code)[0] == code


@pytest.mark.parametrize("code", [53, 59, 105])
def test_integer_input_avoids_float_round_trip_without_changing_legacy_encoding(code):
    velocity_deg_s = math.degrees(code / 100)
    assert force_pos_wire_codes(velocity_deg_s, 0.07)[0] == code - 1
    assert force_pos_wire_codes(torque_ratio=0.07, velocity_wire_code=code)[0] == code


@pytest.mark.parametrize("code", [0, -1, 10001, True, False, 1.0, "1", float("nan"), float("inf")])
def test_velocity_code_rejects_noninteger_boolean_and_out_of_range_input(code):
    with pytest.raises(ValueError, match="integer in 1..10000"):
        force_pos_wire_codes(torque_ratio=0.07, velocity_wire_code=code)
    with pytest.raises(ValueError, match="integer in 1..10000"):
        replace(profile(), velocity_deg_s=None, velocity_wire_code=code).validate()


def test_velocity_input_is_required_and_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        force_pos_wire_codes(10, 0.07, velocity_wire_code=53)
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(profile(), velocity_wire_code=53).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ = replace(profile(), velocity_wire_code=53).velocity_limit_deg_s
    with pytest.raises(ValueError, match="velocity_deg_s"):
        force_pos_wire_codes(torque_ratio=0.07)


@pytest.mark.parametrize("mode", ["mit", "pos_vel"])
def test_velocity_wire_code_is_exclusive_to_force_pos(mode):
    with pytest.raises(ValueError):
        replace(profile("shoulder_pan", mode), velocity_wire_code=53).validate()


def test_force_pos_mapping_checks_the_actual_integer_speed_at_the_boundary():
    configured = replace(profile(), velocity_deg_s=None, velocity_wire_code=53)
    assert struct.unpack("<H", encode_position(7, 0, 0, configured, (8, 0.53, 9)).packet[25:27])[0] == 53
    with pytest.raises(ValueError, match="actual device mapping"):
        encode_position(7, 0, 0, configured, (8, math.nextafter(0.53, 0), 9))
    # The legacy degrees/s request still truncates to code 52, so its effective
    # speed fits a 0.52 rad/s mapping even though its requested speed is higher.
    legacy = replace(profile(), velocity_deg_s=math.degrees(0.53))
    assert struct.unpack("<H", encode_position(7, 0, 0, legacy, (8, 0.52, 9)).packet[25:27])[0] == 52


def test_degrees_input_and_positional_profile_remain_compatible():
    configured = B601JointCommandProfile("gripper", "force_pos", -20, 20, 2, 10, 0.07, None, None)
    assert configured.velocity_wire_code is None
    assert configured.velocity_limit_deg_s == configured.velocity_deg_s == 10
    assert force_pos_wire_codes(10, 0.07) == (17, 700)
