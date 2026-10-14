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
    assert encoded["schema_version"] == 5
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


def test_v5_requires_canonical_names(calibration, settings, tmp_path):
    document = tuning.settings_document(*settings, calibration.path)
    document["values"]["approach_speed_deg_s"] = document["values"].pop("gripper_follow_speed_deg_s")
    path = tmp_path / "wrong-v5.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="参数字段"):
        tuning.read_settings(path)
