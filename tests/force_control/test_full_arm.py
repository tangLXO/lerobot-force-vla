"""Full-arm proposal/receipt integration and configuration-only readiness checks."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from lerobot.force_control.b601_factory import B601HardwareConfig
from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES, FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.robots.rebot_b601_follower import readonly_transport
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlConfig
from lerobot.scripts import lerobot_force_teacher as runner
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace, profile
from tests.force_control.helpers import run_config


@pytest.fixture
def arm_bench():
    class Clock:
        now = 1_000_000_000

        def __call__(self):
            return self.now

    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    teacher = ForceTeacher(TeacherConfig.fake(gripper_follow_speed_deg_s=5, contact_force_threshold_N=0.5), 1)
    runtime = ForceRuntime(
        device,
        sensor,
        teacher,
        SafetySupervisor(SafetyConfig.fake(), 1),
        MemoryTrace(),
        RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES),
        clock=clock,
    )
    runtime.open()
    yield runtime, device, sensor, clock
    runtime.close()


def arm_tick(bench, *, force=0.1, goals=None):
    runtime, _, sensor, clock = bench
    clock.now += 20_000_000
    sensor._publish_sample(
        {"left.normal_force": force, "right.normal_force": force}, clock(), arrival_timestamp_ns=clock()
    )
    if goals is not None:
        runtime.post_arm_intent(
            ArmIntent(
                tuple(goals.items()),
                clock(),
                runtime.epoch,
                clock(),
                200_000_000,
                json.dumps(
                    {
                        "raw_action_deg": goals,
                        "leader_sequence": clock(),
                        "leader_request_started_ns": clock(),
                    }
                ),
            )
        )
    return runtime.tick()


def start_arm(bench, goals):
    runtime = bench[0]
    runtime.post("arm", version=0, epoch=runtime.epoch)
    arm_tick(bench)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": "hybrid", "position_deg": None})
    return arm_tick(bench, goals=goals)


def test_full_arm_continues_following_while_teacher_owns_gripper(arm_bench):
    goals = {f"{name}.pos": 0.8 if name == "gripper" else 0.1 for name in FAKE_B601_JOINT_NAMES}
    row = start_arm(arm_bench, goals)
    assert row["hybrid_state"] == "FOLLOW"
    assert set(row["submitted_action"]) == set(goals)
    goals["shoulder_pan.pos"] = 0.3
    row = arm_tick(arm_bench, force=0.8, goals=goals)
    assert row["hybrid_state"] == "FORCE_TRACK"
    assert row["submitted_action"]["shoulder_pan.pos"] == 0.3
    assert row["arm_filtered_keys"] == ["gripper.pos"]
    assert row["submitted_action"]["gripper.pos"] != goals["gripper.pos"]
    assert row["hardware_ack"] == "unknown"


def test_full_arm_driver_clipping_is_recorded_as_final_action(arm_bench):
    arm_bench[1].arm_limits["shoulder_pan"] = (-0.1, 0.1)
    goals = {f"{name}.pos": 0.8 for name in FAKE_B601_JOINT_NAMES}
    row = start_arm(arm_bench, goals)
    assert row["submitted_action"]["shoulder_pan.pos"] == 0.1
    assert row["arm_intent"]["shoulder_pan.pos"] == 0.8
    result = next(item for item in row["per_joint_results"] if item["name"] == "shoulder_pan")
    assert (result["candidate_deg"], result["clipped_deg"], result["submitted_deg"]) == (0.8, 0.1, 0.1)
    assert "driver_clip:shoulder_pan" in row["clip_flags"]


def test_full_arm_partial_failure_preserves_submissions_and_faults(arm_bench):
    arm_bench[1].send_failure_joint = "elbow_flex"
    goals = {f"{name}.pos": 0.1 for name in FAKE_B601_JOINT_NAMES}
    row = start_arm(arm_bench, goals)
    assert row["phase"] == "FAULT" and row["send_status"] == "unknown"
    assert row["submitted_action"] == {"shoulder_pan.pos": 0.1, "shoulder_lift.pos": 0.1}
    assert [item["status"] for item in row["per_joint_results"]] == [
        "success",
        "success",
        "unknown",
        "not_attempted",
        "not_attempted",
        "not_attempted",
        "not_attempted",
    ]
    assert not row["label_valid"]


def test_hybrid_missing_arm_joint_fails_before_any_submission(arm_bench):
    goals = {f"{name}.pos": 0.1 for name in FAKE_B601_JOINT_NAMES if name != "wrist_yaw"}
    row = start_arm(arm_bench, goals)
    assert row["phase"] == "FAULT" and not row["submitted_action"]
    assert "hybrid_requires_complete_joint_intent" in row["failure_reason"]
    assert not arm_bench[1].submitted_goals


def test_fake_full_arm_runner_forwards_all_leader_joints(tmp_path):
    config = run_config(tmp_path, armed=True)
    config.initial_mode = "hybrid"
    config.leader_backend = "fake"
    config.teacher = replace(config.teacher, gripper_follow_speed_deg_s=5, contact_force_threshold_N=0.5)
    config.runtime = replace(config.runtime, joint_names=FAKE_B601_JOINT_NAMES)
    report = runner.run(config)
    assert report["valid"], report["errors"]
    rows = [json.loads(line) for line in (config.trace_root / "control.jsonl").read_text().splitlines()]
    submitted = [row for row in rows if row.get("send_status") == "submitted"]
    assert submitted
    assert all(
        set(row["submitted_action"]) == {f"{name}.pos" for name in FAKE_B601_JOINT_NAMES} for row in submitted
    )


def arm_profile(tmp_path):
    path = tmp_path / "arm_calibration.json"
    path.write_text(
        json.dumps(
            {
                spec.name: {
                    "id": spec.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -30,
                    "range_max": 30,
                }
                for spec in readonly_transport.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    profiles = tuple(
        profile(name, "force_pos" if name == "gripper" else "pos_vel") for name in FAKE_B601_JOINT_NAMES
    )
    control = B601ControlConfig(
        profiles,
        1.0,
        0.005,
        0.1,
        0.02,
        "disable_selected",
        FAKE_B601_JOINT_NAMES,
        calibration_file=str(path),
    )
    hardware = B601HardwareConfig(
        "FAKE_MUST_NOT_OPEN",
        joint_specs=readonly_transport.DEFAULT_B601_JOINTS,
        control=control,
        startup_mode="verify_pre_enabled",
    )
    return hardware, path


def test_full_arm_file_preflight_never_opens_or_authorizes_hardware(tmp_path, monkeypatch):
    factory = Mock(side_effect=AssertionError("preflight must never open serial"))
    monkeypatch.setattr(readonly_transport.serial, "Serial", factory)
    hardware, path = arm_profile(tmp_path)
    runtime = RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES)
    safety = SafetyConfig.fake(q_min_deg=-20, q_max_deg=20, fault_action="stop", max_relative_target_deg=2)
    report = hardware.preflight_report(safety, runtime)
    assert report["configuration_valid"], report["errors"]
    assert report["hardware_acceptance_required"] and not report["motion_authorized"]
    assert report["acceptance_status"] == "hardware_acceptance_incomplete"
    assert report["calibration"]["path"] == str(path.resolve())
    with pytest.raises(RuntimeError, match="acceptance is incomplete"):
        hardware.make_device().connect()
    factory.assert_not_called()


@pytest.mark.parametrize("mutation", ["missing_joint", "wrong_id", "offset", "profile_bounds", "mit_arm"])
def test_full_arm_preflight_rejects_unsupported_calibration_and_profiles(tmp_path, mutation):
    hardware, path = arm_profile(tmp_path)
    content = json.loads(path.read_text())
    if mutation == "missing_joint":
        del content["wrist_yaw"]
    elif mutation == "wrong_id":
        content["wrist_yaw"]["id"] = 9
    elif mutation == "offset":
        content["wrist_yaw"]["homing_offset"] = 1
    elif mutation == "profile_bounds":
        content["wrist_yaw"]["range_max"] = 5
    else:
        profiles = tuple(
            profile(p.name, "mit") if p.name == "wrist_yaw" else p for p in hardware.control.profiles
        )
        hardware = replace(hardware, control=replace(hardware.control, profiles=profiles))
    path.write_text(json.dumps(content), encoding="utf-8")
    report = hardware.preflight_report(
        SafetyConfig.fake(q_min_deg=-20, q_max_deg=20), RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES)
    )
    assert not report["configuration_valid"] and report["errors"]
    assert not report["motion_authorized"]


def test_full_arm_calibration_change_requires_new_reference(tmp_path):
    hardware, path = arm_profile(tmp_path)
    calibration = hardware.validate_arm_profile()
    calibration.require_unchanged()
    content = json.loads(path.read_text())
    content["elbow_flex"]["range_min"] = -29
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        calibration.require_unchanged()
    assert isinstance(calibration, B601ArmCalibration)


def test_real_full_arm_requires_explicit_leader_mapping_before_transport(tmp_path):
    config = run_config(tmp_path, armed=True)
    config.backend, config.dry_run, config.allow_motion = "b601", False, True
    config.initial_mode, config.leader_backend, config.leader_port = "hybrid", "rebot_102", "FAKE_LEADER"
    config.runtime = replace(config.runtime, joint_names=FAKE_B601_JOINT_NAMES)
    with pytest.raises(ValueError, match="explicit leader joint mapping"):
        config.validate()
