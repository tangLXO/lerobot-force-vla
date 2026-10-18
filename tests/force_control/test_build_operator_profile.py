"""Normal preparation uses native arm defaults and preserves the current gripper form."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import draccus
import pytest

from lerobot.force_control.tuning import build_config, read_settings, save_settings
from lerobot.robots.rebot_b601_follower import readonly_transport
from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig


@pytest.fixture
def preparation(tmp_path, settings, monkeypatch):
    pytest.importorskip("serial")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    module = importlib.import_module("build_operator_teleoperation_profile")
    values, enabled = settings
    values.update(
        port="FAKE_FOLLOWER",
        leader_port="FAKE_LEADER",
        motor_velocity_code="1000",
        drive_percent="25",
        gripper_follow_speed_deg_s="0.17",
        contact_force_threshold_N="0.43",
        soft_q_min_deg="-40",
        soft_q_max_deg="40",
    )
    calibration = tmp_path / "native.json"
    calibration.write_text(
        json.dumps(
            {
                item.name: {
                    "id": item.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -40,
                    "range_max": 40,
                }
                for item in readonly_transport.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    saved = save_settings(tmp_path / "saved.json", values, enabled, calibration)
    serial = Mock(side_effect=AssertionError("Preparation must not open hardware"))
    monkeypatch.setattr(readonly_transport.serial, "Serial", serial)
    return module, saved, calibration, serial


def test_normal_draft_preserves_gripper_and_disables_additional_arm_limits(preparation):
    module, saved, calibration, serial = preparation
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (saved, calibration)]
    document, sources = module.build_document(saved)
    control = document["b601"]["control"]
    gripper = control["profiles"][-1]
    assert gripper["velocity_wire_code"] == 1000 and gripper["torque_ratio"] == 0.25
    assert gripper["max_relative_deg"] == 0.4
    assert all(
        row["velocity_deg_s"] is None and row["max_relative_deg"] is None for row in control["profiles"][:-1]
    )
    assert [row["mode"] for row in control["profiles"]] == ["mit"] * 6 + ["force_pos"]
    assert [row["kp"] for row in control["profiles"][:-1]] == [45, 45, 45, 8, 9, 8]
    assert [row["kd"] for row in control["profiles"][:-1]] == [12, 12, 12, 1, 1, 1]
    assert document["teacher"]["controller_type"] == "ip"
    operator = document["b601"]["operator_teleoperation"]
    assert operator["reference_envelope_deg"] == 100
    assert all(
        operator[name] is None
        for name in ("alignment_tolerance_deg", "preload_tolerance_deg", "max_command_gap_s")
    )
    assert operator["supported_arm"] is False
    assert document["b601"]["validation"] is None
    assert document["duration_s"] is document["safety"]["max_run_time_s"] is None
    assert document["safety"]["max_total_travel_deg"] is None
    assert document["runtime"]["handover_tolerance_deg"] is None
    assert sources["fault_budget"]["is_session_duration_limit"] is False
    assert sources["body_per_frame_limits"].startswith("disabled")
    assert sources["alignment_preload_and_command_gap"].startswith("disabled")
    assert sources["settings_sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in (saved, calibration)] == before
    serial.assert_not_called()


def test_current_memory_form_preserves_complete_gripper_configuration(preparation, tmp_path):
    module, saved, calibration_file, serial = preparation
    values, enabled, _ = read_settings(saved)
    values.update(
        target_N="2.3",
        kp_deg_per_N="0.06",
        ki_deg_per_N_s="0.12",
        drive_percent="13.57",
        motor_velocity_code="1234",
        max_run_time_s="123",
        max_total_travel_deg="57",
        soft_q_min_deg="-31",
        soft_q_max_deg="29",
        object_id="current_object",
        contact_condition_id="current_contact",
    )
    enabled.update(duration=True, travel=True, soft_limits=True)
    original_values, original_enabled = copy.deepcopy(values), copy.deepcopy(enabled)
    calibration = B601GripperCalibration.load(calibration_file)
    output = tmp_path / "not_created"
    baseline = build_config(values, enabled, calibration, output, initial_mode="hybrid")
    expected = draccus.encode(baseline)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    document, metadata = module.build_operator_document(
        values, enabled, calibration, supported_arm=True, output=output
    )
    for section in ("teacher", "safety", "sensors"):
        assert document[section] == expected[section]
    assert document["b601"]["control"]["profiles"][-1] == expected["b601"]["control"]["profiles"][0]
    assert document["duration_s"] == expected["duration_s"] == 123
    assert document["safety"]["max_total_travel_deg"] == 57
    assert document["runtime"]["object_id"] == "current_object"
    assert document["runtime"]["contact_condition_id"] == "current_contact"
    assert document["trace_root"] == expected["trace_root"]
    assert document["b601"]["operator_teleoperation"]["supported_arm"] is True
    decoded = draccus.decode(ForceTeacherRunConfig, document)
    decoded.validate()
    assert decoded.b601.preflight_report(decoded.safety, decoded.runtime)["configuration_valid"]
    assert metadata["parameters_source"] == "current in-memory tuner values and switches"
    assert "settings_file" not in metadata
    assert values == original_values and enabled == original_enabled
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()} == before
    assert not output.exists()
    serial.assert_not_called()


def test_arm_limits_intersect_official_defaults_with_calibration(preparation):
    module, saved, calibration_file, serial = preparation
    values, enabled, _ = read_settings(saved)
    raw = json.loads(calibration_file.read_text())
    for name, row in raw.items():
        if name != "gripper":
            row.update(range_min=-400, range_max=400)
    raw["shoulder_pan"].update(range_min=-33, range_max=27)
    calibration_file.write_text(json.dumps(raw), encoding="utf-8")
    calibration = B601GripperCalibration.load(calibration_file)
    document, _ = module.build_operator_document(values, enabled, calibration)
    native = RebotB601FollowerConfig(port="FAKE")
    for row in document["b601"]["control"]["profiles"][:-1]:
        bounds = (-33, 27) if row["name"] == "shoulder_pan" else native.joint_limits[row["name"]]
        assert (row["minimum_deg"], row["maximum_deg"]) == bounds
        assert document["leader_joint_ranges"][row["name"]] == list(bounds)
        assert row["max_relative_deg"] is None
    assert document["b601"]["operator_teleoperation"]["supported_arm"] is False
    serial.assert_not_called()


def test_disjoint_arm_calibration_is_rejected_without_device_access(preparation):
    module, saved, calibration_file, serial = preparation
    raw = json.loads(calibration_file.read_text())
    raw["shoulder_lift"].update(range_min=10, range_max=20)
    calibration_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="no intersection: shoulder_lift"):
        module.build_document(saved)
    serial.assert_not_called()


def test_saved_builder_reuses_memory_form_helper_and_preserves_enabled_limits(preparation, monkeypatch):
    module, saved, _, serial = preparation
    values, enabled, calibration_path = read_settings(saved)
    values.update(max_run_time_s="45", max_total_travel_deg="19")
    enabled.update(duration=True, travel=True)
    save_settings(saved, values, enabled, calibration_path)
    delegate = Mock(wraps=module.build_operator_document)
    monkeypatch.setattr(module, "build_operator_document", delegate)
    document, metadata = module.build_document(saved)
    delegate.assert_called_once()
    assert document["duration_s"] == document["safety"]["max_run_time_s"] == 45
    assert document["safety"]["max_total_travel_deg"] == 19
    assert metadata["settings_file"] == str(saved.resolve())
    assert metadata["settings_sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert document["b601"]["operator_teleoperation"]["supported_arm"] is False
    serial.assert_not_called()


def test_draft_files_are_data_only_and_existing_output_is_preserved(preparation, tmp_path):
    module, saved, _, serial = preparation
    destination = tmp_path / "operator.draft.json"
    output, sources = module.write_draft(saved, destination, supported_arm=True)
    assert json.loads(output.read_text())["b601"]["operator_teleoperation"]["supported_arm"] is True
    assert json.loads(sources.read_text())["hardware_accessed"] is False
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_draft(saved, destination)
    assert output.read_bytes() == original
    serial.assert_not_called()


def test_editor_preserves_a_smaller_explicit_reference_envelope(preparation, tmp_path):
    module, saved, _, serial = preparation
    document, _ = module.build_document(saved)
    document["b601"]["operator_teleoperation"]["reference_envelope_deg"] = 50
    source = tmp_path / "smaller_reference.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    editor = importlib.import_module("b601_operator_panel").OperatorProfileEditor()
    editor.load(source)
    assert editor.document()["b601"]["operator_teleoperation"]["reference_envelope_deg"] == 50
    serial.assert_not_called()
