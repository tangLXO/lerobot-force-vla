"""Configuration dimensions and optional limits use synthetic values only."""

from dataclasses import asdict, replace

import draccus
import pytest

from lerobot.force_control.config import (
    LEGACY_TEACHER_FIELDS,
    SafetyConfig,
    TeacherConfig,
    canonicalize_teacher_fields,
    validate_close_sign,
    validate_profile,
)


def test_defaults_never_supply_hardware_gains_or_limits():
    for config in (TeacherConfig(), SafetyConfig()):
        with pytest.raises(ValueError):
            config.validate()
    validate_profile(TeacherConfig.fake(), SafetyConfig.fake(), 1)


@pytest.mark.parametrize("field", ["kp_deg_per_N", "ki_deg_per_N_s", "integral_deadband_N"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "1 N", True, -1])
def test_invalid_gain_or_deadband(field, value):
    with pytest.raises(ValueError):
        TeacherConfig.fake(**{field: value}).validate()


def test_zero_gains_and_zero_deadband_are_explicitly_valid():
    TeacherConfig.fake(kp_deg_per_N=0, ki_deg_per_N_s=0, integral_deadband_N=0).validate()


@pytest.mark.parametrize(
    "updates",
    [
        {"controller_type": "p"},
        {"controller_type": "pi"},
        {"controller_type": "pid"},
        {"force_target_N": 4},
        {"target_min_N": 4},
        {"release_speed_deg_s": 0},
        {"release_travel_deg": None},
        {"max_release_time_s": 0},
        {"profile_id": ""},
    ],
)
def test_invalid_teacher_profile(updates):
    with pytest.raises(ValueError):
        TeacherConfig.fake(**updates).validate()


def test_old_gain_names_are_rejected_instead_of_reinterpreted():
    with pytest.raises(TypeError):
        TeacherConfig(kp=0.1)


@pytest.mark.parametrize(
    "updates",
    [
        {"q_min_deg": 10},
        {"q_max_deg": float("inf")},
        {"fault_action": None},
        {"max_relative_target_deg": 0},
        {"max_command_tracking_error_deg": None},
        {"hard_force_limit_left_N": "5N"},
        {"supported_object_bench": "yes"},
        {"fault_action": "open_automatically"},
        {"soft_q_min_deg": -1},
        {"soft_q_min_deg": -11, "soft_q_max_deg": 2},
        {"soft_q_min_deg": 2, "soft_q_max_deg": 1},
        {"max_total_travel_deg": 0},
        {"max_run_time_s": float("nan")},
        {"stall_timeout_s": None},
    ],
)
def test_invalid_safety_profile(updates):
    with pytest.raises(ValueError):
        SafetyConfig.fake(**updates).validate()


def test_none_disables_only_documented_optional_limits():
    SafetyConfig.fake(
        acceleration_limit_deg_s2=None,
        max_total_travel_deg=None,
        max_run_time_s=None,
        soft_q_min_deg=None,
        soft_q_max_deg=None,
        stall_position_tolerance_deg=None,
        stall_timeout_s=None,
    ).validate()


@pytest.mark.parametrize("sign", [None, 0, 2, -2, True, "1", 1.0])
def test_close_sign_is_never_guessed(sign):
    with pytest.raises(ValueError):
        validate_close_sign(sign)


def test_target_and_release_share_safety_envelope():
    teacher, safety = TeacherConfig.fake(), SafetyConfig.fake()
    for changed in (replace(safety, hard_force_limit_right_N=3), replace(safety, max_open_speed_deg_s=1)):
        with pytest.raises(ValueError):
            validate_profile(teacher, changed, 1)


def test_legacy_teacher_yaml_decodes_nested_and_encodes_only_canonical_names(tmp_path):
    from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig

    path = tmp_path / "legacy.yaml"
    path.write_text(
        "teacher:\n  approach_speed_deg_s: 0.2\n  approach_force_threshold_N: 0.5\n",
        encoding="utf-8",
    )
    original = path.read_bytes()
    config = draccus.parse(ForceTeacherRunConfig, config_path=path, args=[])
    assert config.teacher.gripper_follow_speed_deg_s == 0.2
    assert config.teacher.contact_force_threshold_N == 0.5
    encoded = draccus.encode(config)["teacher"]
    assert not LEGACY_TEACHER_FIELDS.keys() & encoded.keys()
    assert encoded == asdict(config.teacher)
    assert path.read_bytes() == original


@pytest.mark.parametrize("legacy,current", LEGACY_TEACHER_FIELDS.items())
@pytest.mark.parametrize("value", [0.2, None])
def test_legacy_and_canonical_teacher_names_conflict_even_when_values_match(legacy, current, value):
    payload = {legacy: value, current: value}
    with pytest.raises(ValueError, match="Conflicting Teacher fields"):
        canonicalize_teacher_fields(payload)
    with pytest.raises(draccus.utils.DecodingError, match="Conflicting Teacher fields"):
        draccus.decode(TeacherConfig, payload)
    assert payload == {legacy: value, current: value}


def test_canonical_teacher_config_round_trip_and_replace_have_one_source_of_truth():
    original = TeacherConfig.fake(gripper_follow_speed_deg_s=0.2, contact_force_threshold_N=0.5)
    changed = replace(original, gripper_follow_speed_deg_s=0.3)
    assert changed.gripper_follow_speed_deg_s == 0.3
    assert draccus.decode(TeacherConfig, draccus.encode(changed)) == changed
    assert not LEGACY_TEACHER_FIELDS.keys() & asdict(changed).keys()


def test_legacy_teacher_decode_remains_strict_about_unknown_fields():
    with pytest.raises(draccus.utils.DecodingError, match="unknown_gain"):
        draccus.decode(TeacherConfig, {"approach_speed_deg_s": 0.2, "unknown_gain": 1})
