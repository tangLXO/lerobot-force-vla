"""File validation and pinned revisions without hardware or GUI fixtures."""

import json
import os

import pytest

from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.robots.rebot_b601_follower.control_protocol import position_wire_bounds


@pytest.mark.parametrize(
    "change",
    [
        {"id": 6},
        {"id": "7"},
        {"drive_mode": 1},
        {"drive_mode": False},
        {"homing_offset": 10},
        {"range_min": 10},
        {"range_max": -20},
        {"range_max": float("nan")},
        {"range_min": True},
        {"range_min": None},
    ],
)
def test_invalid_calibration_is_rejected(calibration_file, change):
    data = json.loads(calibration_file.read_text())
    data["gripper"].update(change)
    calibration_file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        B601GripperCalibration.load(calibration_file)


@pytest.mark.parametrize("content", ["{}", "[]", "not json"])
def test_missing_or_malformed_files_are_rejected(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        B601GripperCalibration.load(path)
    path.unlink()
    with pytest.raises(ValueError, match="无法读取"):
        B601GripperCalibration.load(path)


def test_identical_regeneration_invalidates_reference(calibration_file):
    pin = B601GripperCalibration.load(calibration_file)
    os.utime(calibration_file, ns=(calibration_file.stat().st_atime_ns, pin.mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="已变化"):
        pin.require_unchanged()
    assert B601GripperCalibration.load(calibration_file).sha256 == pin.sha256


def test_changed_deleted_and_replaced_files_fail_closed(calibration_file):
    pin = B601GripperCalibration.load(calibration_file)
    calibration_file.write_text(calibration_file.read_text().replace("-10", "-9"))
    with pytest.raises(ValueError):
        pin.require_unchanged()
    calibration_file.unlink()
    with pytest.raises(ValueError):
        pin.require_unchanged()


def test_wire_bounds_round_inside_calibration(calibration_file):
    pin = B601GripperCalibration.load(calibration_file)
    lower, upper = position_wire_bounds(pin.minimum_deg, pin.maximum_deg)
    assert pin.minimum_deg <= lower < upper <= pin.maximum_deg
    pin.check_travel(0, 1)
    with pytest.raises(ValueError):
        pin.check_travel(10, 1)
