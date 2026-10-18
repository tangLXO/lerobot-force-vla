"""Pure FORCE_POS protocol tests; integer speed input never touches hardware."""

import math
import struct
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import (
    B601JointCommandProfile,
    encode_position,
    force_pos_wire_codes,
    mit_gain_wire_codes,
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


@pytest.mark.parametrize("velocity,torque", [(0, 0), (30, 1.5), (-20, -2)])
def test_mit_references_encode_explicit_signed_values(velocity, torque):
    configured = replace(profile("shoulder_pan", "mit"), mit_velocity_deg_s=velocity, mit_torque_Nm=torque)
    encoded = encode_position(1, 0, 0, configured, (12.5, 10, 28))
    payload = encoded.packet[21:29]
    v = (payload[2] << 4) | (payload[3] >> 4)
    t = ((payload[6] & 15) << 8) | payload[7]
    assert v == int((math.radians(velocity) + 10) * 4095 / 20)
    assert t == int((torque + 28) * 4095 / 56)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mit_velocity_deg_s", 600),
        ("mit_torque_Nm", 29),
        ("mit_velocity_deg_s", float("nan")),
        ("mit_torque_Nm", True),
    ],
)
def test_mit_reference_outside_verified_mapping_or_invalid_data_is_rejected(field, value):
    with pytest.raises(ValueError):
        encode_position(1, 0, 0, replace(profile("shoulder_pan", "mit"), **{field: value}), (12.5, 10, 28))


def _position_profile(mode, lower, upper, *, relative=1000):
    return B601JointCommandProfile(
        "gripper" if mode == "force_pos" else "shoulder_pan",
        mode,
        lower,
        upper,
        relative,
        velocity_deg_s=10,
        torque_ratio=0.07 if mode == "force_pos" else None,
    )


def _wire_degrees(encoded):
    return math.degrees(struct.unpack("<f", encoded.packet[21:25])[0])


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
@pytest.mark.parametrize("target", [-270.0, 0.0])
def test_calibrated_gripper_endpoints_encode_inside_unchanged_limits(mode, target):
    configured = _position_profile(mode, -270, 0)
    encoded = encode_position(7, target, -135, configured, (12.5, 30, 10))
    assert encoded.clipped_deg == target
    assert -270 <= encoded.wire_position_deg <= 0
    assert _wire_degrees(encoded) == encoded.wire_position_deg
    if target == -270:
        # Nearest float32 radians would be -270.00000068324533 degrees.
        assert encoded.wire_position_deg == -269.9999733624887
    else:
        assert encoded.wire_position_deg == 0


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
@pytest.mark.parametrize("target", [0.1, 30.0])
def test_positive_mechanical_endpoints_round_inward_by_one_float32_step(mode, target):
    configured = _position_profile(mode, 0.1, 30)
    encoded = encode_position(7, target, 15, configured, (12.5, 30, 10))
    nearest = struct.unpack("<I", struct.pack("<f", math.radians(target)))[0]
    expected_bits = nearest + (1 if target == 0.1 else -1)
    assert struct.unpack("<I", encoded.packet[21:25])[0] == expected_bits
    assert encoded.clipped_deg == target
    assert 0.1 <= encoded.wire_position_deg <= 30
    assert _wire_degrees(encoded) == encoded.wire_position_deg


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
@pytest.mark.parametrize("target", [-100.0, 100.0])
def test_dynamic_measured_relative_bounds_also_round_inward(mode, target):
    configured = _position_profile(mode, -100, 100, relative=30)
    encoded = encode_position(7, target, 0, configured, (12.5, 30, 10))
    assert encoded.clipped_deg == math.copysign(30, target)
    assert -30 <= encoded.wire_position_deg <= 30
    assert abs(encoded.wire_position_deg - encoded.clipped_deg) < 0.00001
    assert _wire_degrees(encoded) == encoded.wire_position_deg


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("pmax", [0.1, struct.unpack("<f", struct.pack("<f", 0.1))[0]])
def test_device_mapping_endpoints_cannot_be_crossed_by_float32_rounding(mode, direction, pmax):
    configured = _position_profile(mode, -100, 100)
    encoded = encode_position(7, direction * 100, 0, configured, (pmax, 30, 10))
    limit_deg = math.degrees(pmax)
    assert encoded.clipped_deg == direction * limit_deg
    assert -limit_deg <= encoded.wire_position_deg <= limit_deg
    assert _wire_degrees(encoded) == encoded.wire_position_deg
    assert abs(struct.unpack("<f", encoded.packet[21:25])[0]) <= pmax


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
def test_interval_with_no_float32_position_remains_rejected(mode):
    # Both endpoints lie strictly between 1.0 rad and its next float32 value.
    lower = math.degrees(1 + 2**-25)
    upper = math.degrees(1 + 3 * 2**-25)
    configured = _position_profile(mode, lower, upper)
    with pytest.raises(ValueError, match="quantization"):
        encode_position(7, (lower + upper) / 2, (lower + upper) / 2, configured, (12.5, 30, 10))


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
def test_single_representable_position_in_narrow_interval_is_accepted(mode):
    lower, upper = math.degrees(1 - 2**-26), math.degrees(1 + 2**-26)
    configured = _position_profile(mode, lower, upper)
    for candidate in (lower, upper):
        encoded = encode_position(7, candidate, math.degrees(1), configured, (12.5, 30, 10))
        assert encoded.wire_position_deg == math.degrees(1)
        assert encoded.packet[21:25] == bytes.fromhex("0000803f")


@pytest.mark.parametrize("mode", ["force_pos", "pos_vel"])
def test_zero_width_intersection_accepts_its_exactly_representable_position(mode):
    boundary = math.degrees(1)
    measured = boundary + 1
    assert measured - 1 == boundary
    configured = _position_profile(mode, 0, boundary, relative=1)
    encoded = encode_position(7, boundary, measured, configured, (12.5, 30, 10))
    assert encoded.clipped_deg == encoded.wire_position_deg == boundary
    assert encoded.packet[21:25] == bytes.fromhex("0000803f")


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("force_pos", "55aa1e03010000000a000000000703000000080000a692abbc1100bc0200"),
        ("pos_vel", "55aa1e03010000000a000000000101000000080000a692abbcc2b8323e00"),
    ],
)
def test_interior_golden_packets_remain_byte_identical(mode, expected):
    configured = profile("gripper" if mode == "force_pos" else "shoulder_pan", mode)
    encoded = encode_position(7 if mode == "force_pos" else 1, -1.2, 0, configured, (8, 14, 9))
    assert encoded.packet.hex() == expected
    assert encoded.clipped_deg == -1.2
    assert _wire_degrees(encoded) == encoded.wire_position_deg


@pytest.mark.parametrize("kp,kd,expected", [(45, 12, (368, 4095)), (8, 1, (65, 819)), (9, 1, (73, 819))])
def test_native_b601_mit_gains_keep_requested_values_and_match_sdk_wire_codes(kp, kd, expected):
    # motorbridge 0.3.9 protocol.rs float_to_uint clamps to kp=[0,500],
    # kd=[0,5], then uses float32 arithmetic and truncates to 12 bits.
    configured = B601JointCommandProfile("shoulder_pan", "mit", -150, 150, 10, kp=kp, kd=kd)
    encoded = encode_position(1, -1.2, 0, configured, (8, 14, 9))
    payload = encoded.packet[21:29]
    assert (configured.kp, configured.kd) == (kp, kd)
    assert ((payload[3] & 15) << 8 | payload[4], payload[5] << 4 | payload[6] >> 4) == expected
    assert mit_gain_wire_codes(kp, kd) == expected
    assert int.from_bytes(encoded.packet[13:17], "little") == 1
    assert payload[2] << 4 | payload[3] >> 4 == 2047
    assert (payload[6] & 15) << 8 | payload[7] == 2047


@pytest.mark.parametrize("gain", [None, True, -1, float("nan"), float("inf")])
def test_mit_invalid_gains_are_rejected_before_encoding(gain):
    configured = B601JointCommandProfile("shoulder_pan", "mit", -150, 150, 10, kp=45, kd=gain)
    with pytest.raises(ValueError):
        configured.validate()


@pytest.mark.parametrize("target", [-100, 100])
def test_mit_position_codes_round_inward_at_existing_profile_and_relative_endpoints(target):
    configured = B601JointCommandProfile("shoulder_pan", "mit", -100, 100, 30, kp=45, kd=12)
    encoded = encode_position(1, target, 0, configured, (8, 14, 9))
    assert encoded.clipped_deg == math.copysign(30, target)
    assert -30 <= encoded.wire_position_deg <= 30
    code = int.from_bytes(encoded.packet[21:23], "big")
    assert encoded.wire_position_deg == math.degrees(code * 16 / 65535 - 8)
    assert abs(encoded.wire_position_deg - encoded.clipped_deg) < math.degrees(16 / 65535)


def test_mit_narrow_interval_accepts_its_single_representable_code():
    position = math.degrees(32768 * 16 / 65535 - 8)
    configured = B601JointCommandProfile(
        "shoulder_pan", "mit", position - 1e-10, position + 1e-10, 1, kp=45, kd=12
    )
    encoded = encode_position(1, position, position, configured, (8, 14, 9))
    assert int.from_bytes(encoded.packet[21:23], "big") == 32768
    assert encoded.wire_position_deg == position


def test_mit_interval_without_representable_code_remains_rejected():
    configured = B601JointCommandProfile("shoulder_pan", "mit", -1e-6, 1e-6, 1, kp=45, kd=12)
    with pytest.raises(ValueError, match="quantization"):
        encode_position(1, 0, 0, configured, (8, 14, 9))


def test_mit_native_default_omits_additional_measured_relative_limit():
    configured = B601JointCommandProfile("shoulder_pan", "mit", -100, 100, kp=45, kd=12)
    assert configured.max_relative_deg is None
    encoded = encode_position(1, 80, 0, configured, (8, 14, 9))
    assert encoded.clipped_deg == 80
    assert abs(encoded.wire_position_deg - 80) < math.degrees(16 / 65535)
    endpoint = encode_position(1, 150, 0, configured, (8, 14, 9))
    assert endpoint.clipped_deg == 100
    assert configured.minimum_deg <= endpoint.wire_position_deg <= configured.maximum_deg


@pytest.mark.parametrize("mode", ["pos_vel", "force_pos"])
def test_position_modes_keep_their_existing_explicit_relative_limit(mode):
    configured = replace(
        profile("gripper" if mode == "force_pos" else "shoulder_pan", mode), max_relative_deg=None
    )
    with pytest.raises(ValueError, match="explicit positive max_relative_deg"):
        configured.validate()


@pytest.mark.parametrize("relative", [0, -1, True, float("nan"), float("inf")])
def test_selected_mit_relative_limits_still_require_positive_finite_values(relative):
    configured = B601JointCommandProfile("shoulder_pan", "mit", -100, 100, relative, kp=45, kd=12)
    with pytest.raises(ValueError):
        configured.validate()
