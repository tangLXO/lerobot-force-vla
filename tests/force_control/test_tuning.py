"""Shared configuration migration is data-only and preserves explicit physical values."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot.force_control import tuning
from lerobot.force_control.config import LEGACY_TEACHER_FIELDS


def test_shared_settings_import_without_tk_or_example_modules():
    script = (
        "import sys; from lerobot.force_control import tuning; "
        "assert 'tkinter' not in sys.modules; "
        "assert 'b601_tuner' not in sys.modules; "
        "assert 'b601_supervised_bench' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_v6_settings_load_without_overwriting_file_or_selecting_arm_limits(settings, calibration, tmp_path):
    document = tuning.settings_document(*settings, calibration.path)
    document["schema_version"] = 6
    document.pop("arm_settings")
    path = tmp_path / "legacy-v6.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    arms = []
    values, enabled, _ = tuning.read_settings(path, on_arm_settings=arms.append)
    assert (values, enabled) == settings
    assert arms == [{}] and path.read_bytes() == original


@pytest.mark.parametrize("bad", [{"supported_arm": True}, {"operator": {}, "joints": {}}, "invalid"])
def test_saved_arm_settings_reject_unknown_fields_and_activation_state(settings, calibration, tmp_path, bad):
    document = tuning.settings_document(*settings, calibration.path)
    document["arm_settings"] = bad
    path = tmp_path / "invalid-arm-settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="全臂参数"):
        tuning.read_settings(path)


def test_panel_uses_the_shared_functions_and_schema(panel):
    for name in ("FIELDS", "build_config", "parse_values", "read_settings", "save_settings", "sensor_config"):
        assert getattr(panel, name) is getattr(tuning, name)


@pytest.mark.parametrize("version", [3, 4])
def test_old_motion_field_names_migrate_readonly_and_save_only_canonical_names(
    calibration, settings, tmp_path, version
):
    values, enabled = settings
    values.update(gripper_follow_speed_deg_s="0.17", contact_force_threshold_N="0.43")
    document = tuning.settings_document(values, enabled, calibration.path)
    document["schema_version"] = version
    for old, new in LEGACY_TEACHER_FIELDS.items():
        document["values"][old] = document["values"].pop(new)
    if version == 3:
        document["values"].pop("position_speed_deg_s")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original, mtime = path.read_bytes(), path.stat().st_mtime_ns
    restored, switches, source = tuning.read_settings(path)
    assert restored == values
    assert switches == enabled and source == Path(calibration.path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == (original, mtime)
    saved = tuning.save_settings(tmp_path / "canonical.json", restored, switches, source)
    encoded = json.loads(saved.read_text(encoding="utf-8"))
    assert encoded["schema_version"] == 9
    assert encoded["values"] == values
    assert not LEGACY_TEACHER_FIELDS.keys() & encoded["values"].keys()
    config = tuning.build_config(restored, switches, calibration, tmp_path / "trace")
    assert config.teacher.gripper_follow_speed_deg_s == 0.17
    assert config.teacher.contact_force_threshold_N == 0.43
    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize("old,new", LEGACY_TEACHER_FIELDS.items())
def test_legacy_json_rejects_duplicate_aliases(calibration, settings, tmp_path, old, new):
    document = tuning.settings_document(*settings, calibration.path)
    document["schema_version"] = 4
    document["values"][old] = document["values"][new]
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting Teacher fields"):
        tuning.read_settings(path)


def test_current_settings_require_canonical_names(calibration, settings, tmp_path):
    document = tuning.settings_document(*settings, calibration.path)
    document["values"]["approach_speed_deg_s"] = document["values"].pop("gripper_follow_speed_deg_s")
    path = tmp_path / "wrong-v5.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="参数字段"):
        tuning.read_settings(path)


def test_profile_before_leader_fields_migrates_without_selecting_a_device(calibration, settings, tmp_path):
    values, enabled = settings
    document = tuning.settings_document(values, enabled, calibration.path)
    document["schema_version"] = 5
    for key in tuning.LEADER_FIELDS:
        document["values"].pop(key)
    path = tmp_path / "before-leader.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    restored, switches, calibration_path = tuning.read_settings(path)
    assert restored == values and switches == enabled
    assert restored["leader_port"] == ""
    assert path.read_bytes() == original
    config = tuning.build_config(restored, switches, calibration, tmp_path / "unused")
    assert config.leader_backend is None and config.leader_port is None
    assert calibration_path == Path(calibration.path)


def test_shared_hybrid_configuration_requires_explicit_leader_and_same_gains(calibration, settings, tmp_path):
    values, enabled = settings
    values.update(gripper_follow_speed_deg_s="0.17", contact_force_threshold_N="0.43")
    with pytest.raises(ValueError, match="leader_port"):
        tuning.build_config(values, enabled, calibration, tmp_path / "unused", initial_mode="hybrid")
    values["leader_port"] = "COM_TEST_LEADER"
    hybrid = tuning.build_config(values, enabled, calibration, tmp_path / "hybrid", initial_mode="hybrid")
    standalone = tuning.build_config(values, enabled, calibration, tmp_path / "standalone")
    assert hybrid.teacher == standalone.teacher and hybrid.safety == standalone.safety
    assert hybrid.leader_backend == "rebot_102" and hybrid.leader_port == "COM_TEST_LEADER"
    assert hybrid.runtime.joint_names == ("gripper",)
    assert standalone.leader_backend is None and standalone.leader_port is None
    assert not (tmp_path / "hybrid").exists()


def test_hybrid_panel_explains_mapped_leader_and_committed_contact_reference(panel):
    assert panel.hybrid_description({"phase": "FORCE_TRACK"}) == ""
    text = panel.hybrid_description(
        {
            "hybrid_state": "FORCE_TRACK",
            "arm_intent": {"gripper.pos": -60},
            "contact_reference": {"follower_gripper_deg": -90},
        }
    )
    assert "主爪映射 -60°" in text and "接触从爪参考 -90°" in text


def test_working_command_profile_is_inside_independent_calibration_hard_limits(
    panel, calibration, settings, tmp_path
):
    values, enabled = settings
    values.update(soft_q_min_deg="-30", soft_q_max_deg="30")
    enabled["soft_limits"] = True
    config = tuning.build_config(values, enabled, calibration, tmp_path / "unused")
    assert (config.safety.q_min_deg, config.safety.q_max_deg) == (-40, 40)
    assert (config.safety.soft_q_min_deg, config.safety.soft_q_max_deg) == (-30, 30)
    profile = config.b601.control.profiles[0]
    assert (profile.minimum_deg, profile.maximum_deg) == (-30, 30)
    text = panel.position_range_description(config)
    assert "工作目标范围 [-30, 30]°" in text and "校准硬保护范围 [-40, 40]°" in text
    assert not (tmp_path / "unused").exists()
