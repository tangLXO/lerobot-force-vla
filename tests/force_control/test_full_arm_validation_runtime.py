"""Bounded validation uses the common owner without changing normal fake control."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES, FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace


@pytest.fixture
def validation_runtime():
    class Clock:
        now = 1_000_000_000

        def __call__(self):
            return self.now

    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    device.full_arm_validation_config = SimpleNamespace(target_force_N=1.0)
    teacher = ForceTeacher(TeacherConfig.fake(gripper_follow_speed_deg_s=1, contact_force_threshold_N=0.5), 1)
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


def tick(bench, *, command=None):
    runtime, _, sensor, clock = bench
    clock.now += 20_000_000
    sensor._publish_sample(
        {"left.normal_force": 0.1, "right.normal_force": 0.1},
        clock(),
        arrival_timestamp_ns=clock(),
    )
    goals = {f"{name}.pos": 0.0 for name in FAKE_B601_JOINT_NAMES}
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
                    "leader_response_completed_ns": clock(),
                    "leader_config_sha256": "f" * 64,
                }
            ),
        )
    )
    if command is not None:
        runtime.post(command[0], version=clock(), epoch=runtime.epoch, value=command[1])
    return runtime.tick()


def test_validation_start_binds_fresh_leader_and_does_not_label_startup(validation_runtime):
    runtime, device, _, _ = validation_runtime
    received = []

    def start(*, guard, leader_sample):
        guard()
        received.append(leader_sample())
        return runtime.latest_joint_snapshot

    device.start_full_arm_validation = start
    tick(validation_runtime, command=("arm", None))
    row = tick(validation_runtime, command=("start", {"mode": "hybrid", "position_deg": None}))
    assert received[0].source_config_sha256 == "f" * 64
    assert received[0].epoch == runtime.epoch
    assert set(dict(received[0].positions_deg)) == set(FAKE_B601_JOINT_NAMES)
    assert row["phase"] == "FOLLOW" and not row["label_valid"]
    assert row["send_status"] == "none" and not device.submitted_goals
    following = tick(validation_runtime)
    assert following["send_status"] == "submitted"
    assert set(following["submitted_action"]) == {f"{name}.pos" for name in FAKE_B601_JOINT_NAMES}


def test_validation_start_refusal_faults_before_normal_commands(validation_runtime):
    runtime, device, _, _ = validation_runtime

    def refuse(**kwargs):
        raise RuntimeError("full_arm_validation_leader_not_aligned:wrist_roll")

    device.start_full_arm_validation = refuse
    tick(validation_runtime, command=("arm", None))
    row = tick(validation_runtime, command=("start", {"mode": "hybrid", "position_deg": None}))
    assert row["phase"] == "FAULT"
    assert "leader_not_aligned" in row["failure_reason"]
    assert not device.submitted_goals


def test_startup_checks_new_leader_cache_against_completion_time(validation_runtime):
    runtime, device, _, clock = validation_runtime

    def start(*, guard, leader_sample):
        leader_sample()
        clock.now += 10_000_000
        goals = {f"{name}.pos": 0.0 for name in FAKE_B601_JOINT_NAMES}
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
                        "leader_response_completed_ns": clock(),
                        "leader_config_sha256": "f" * 64,
                    }
                ),
            )
        )
        guard()
        return runtime.latest_joint_snapshot

    device.start_full_arm_validation = start
    tick(validation_runtime, command=("arm", None))
    row = tick(validation_runtime, command=("start", {"mode": "hybrid", "position_deg": None}))
    assert row["phase"] == "FOLLOW", row["failure_reason"]
    assert row["arm_command_timestamp_ns"] == row["decision_anchor_ns"] == clock()
    assert not row["submitted_action"]


@pytest.mark.parametrize("kind", ["target", "parameters", "direct_mutation"])
def test_validation_force_target_cannot_change_during_owner_session(validation_runtime, kind):
    runtime, device, _, _ = validation_runtime
    original = runtime.controller.config
    if kind == "parameters":
        command = (
            kind,
            {"teacher": replace(original, force_target_N=2), "safety": runtime.safety.config},
        )
    elif kind == "direct_mutation":
        runtime.controller.config = replace(original, force_target_N=2)
        command = None
    else:
        command = (kind, 2)
    row = tick(validation_runtime, command=command)
    assert row["phase"] == "FAULT"
    assert "full_arm_validation_force_target_changed" in row["failure_reason"]
    assert not device.submitted_goals
    if kind != "direct_mutation":
        assert runtime.controller.config.force_target_N == 1


@pytest.mark.parametrize("changed", ["teacher", "safety"])
def test_validation_parameter_bundle_is_frozen_with_same_force_target(validation_runtime, changed):
    runtime, device, _, _ = validation_runtime
    teacher, safety = runtime.controller.config, runtime.safety.config
    if changed == "teacher":
        teacher = replace(teacher, kp_deg_per_N=teacher.kp_deg_per_N + 1)
    else:
        safety = replace(safety, max_close_speed_deg_s=safety.max_close_speed_deg_s / 2)
    row = tick(validation_runtime, command=("parameters", {"teacher": teacher, "safety": safety}))
    assert row["phase"] == "FAULT"
    assert f"full_arm_validation_{changed}_config_changed" in row["failure_reason"]
    assert not device.submitted_goals
