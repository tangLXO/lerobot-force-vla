"""Offline shared fixtures using temporary synthetic calibration files only."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def calibration_file(tmp_path):
    """A file-backed test range, never the user's physical calibration."""
    path = tmp_path / "b601-calibration.json"
    path.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -10, "range_max": 10}}
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("b601_tuner")


@pytest.fixture
def calibration(panel, tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -40, "range_max": 40}}
        )
    )
    return panel.B601GripperCalibration.load(path)


@pytest.fixture
def settings(panel):
    values = panel.default_values()
    values.update(
        kp_deg_per_N="0.04",
        ki_deg_per_N_s="0.08",
        close_sign="-1",
        target_max_N="9.80665",
        max_relative_target_deg="0.4",
        max_command_tracking_error_deg="0.8",
        stall_position_tolerance_deg="0.01",
        stall_timeout_s="0.3",
    )
    enabled = {key: state for key, (_, state) in panel.OPTIONAL_GROUPS.items()}
    return values, enabled
