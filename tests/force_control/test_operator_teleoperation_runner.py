"""The normal operator path uses explicit settings without micro-trial prerequisites."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.force_control.b601_factory import B601HardwareConfig
from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES, FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.robots.rebot_b601_follower import readonly_transport
from lerobot.scripts import lerobot_force_teacher as runner
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace
from tests.force_control.operator_teleoperation_helpers import make_operator_files


@pytest.fixture
def normal_config(tmp_path):
    pytest.importorskip("serial")
    files = make_operator_files(tmp_path)
    names = FAKE_B601_JOINT_NAMES
    config = runner.ForceTeacherRunConfig(
        backend="b601",
        dry_run=False,
        allow_motion=True,
        explicit_arm=True,
        initial_mode="hybrid",
        close_sign=1,
        duration_s=None,
        leader_backend="rebot_102",
        leader_port="FAKE_LEADER",
        leader_max_age_s=files.operator.leader_max_age_s,
        leader_read_budget_s=0.05,
        leader_joint_ids=dict(zip(names, range(7), strict=True)),
        leader_joint_directions=dict.fromkeys(names, 1),
        leader_joint_ranges={name: [-150.0, 150.0] for name in names},
        teacher=TeacherConfig.fake(
            force_target_N=2,
            target_max_N=4,
            gripper_follow_speed_deg_s=5,
            contact_force_threshold_N=0.5,
        ),
        safety=SafetyConfig.fake(
            q_min_deg=-200,
            q_max_deg=200,
            max_relative_target_deg=10,
            joint_feedback_max_age_ms=100,
            fault_action="stop",
            max_run_time_s=None,
            max_total_travel_deg=None,
        ),
        runtime=RuntimeConfig(
            joint_names=names,
            feedback_budget_ms=110,
            tick_deadline_ms=200,
            arm_intent_required=True,
        ),
        sensors={
            "gripper_force": X518SensorConfig(
                host="192.0.2.1",
                max_age_ms=100,
                channels={
                    "left.normal_force": X518ChannelConfig(channel=2),
                    "right.normal_force": X518ChannelConfig(channel=1),
                },
            )
        },
        b601=B601HardwareConfig(
            port=files.operator.port,
            joint_specs=readonly_transport.DEFAULT_B601_JOINTS,
            control=files.control,
            startup_mode="prepare_disabled_operator_teleoperation",
            operator_teleoperation=files.operator,
        ),
    )
    selected = replace(
        files.operator,
        leader_config_sha256=runner.leader_configuration_digest(
            runner.leader_configuration_projection(config)
        ),
    )
    config.b601 = replace(config.b601, operator_teleoperation=selected)
    return config


@pytest.mark.parametrize("duration", [None, 120.0])
def test_normal_preflight_accepts_user_settings_without_diagnostic_files(
    normal_config, monkeypatch, duration
):
    normal_config.duration_s = duration
    serial = Mock(side_effect=AssertionError("Software preflight cannot open a serial port"))
    monkeypatch.setattr(readonly_transport.serial, "Serial", serial)
    normal_config.validate()
    report = normal_config.b601.preflight_report(normal_config.safety, normal_config.runtime)
    assert report["configuration_valid"], report["errors"]
    assert report["acceptance_status"] == "operator_teleoperation_config_only"
    assert report["operator_teleoperation"]["micro_history_required"] is False
    assert report["operator_teleoperation"]["fixed_force_target_N"] is None
    assert normal_config.teacher.force_target_N == 2
    assert all(
        profile.mode == "mit" and profile.velocity_deg_s is None
        for profile in normal_config.b601.control.profiles[:-1]
    )
    assert normal_config.b601.validation is None
    serial.assert_not_called()


def test_unknown_body_mit_gain_refuses_preflight_without_opening_hardware(normal_config, monkeypatch):
    serial = Mock(side_effect=AssertionError("Unknown settings cannot open hardware"))
    monkeypatch.setattr(readonly_transport.serial, "Serial", serial)
    profiles = normal_config.b601.control.profiles
    normal_config.b601 = replace(
        normal_config.b601,
        control=replace(normal_config.b601.control, profiles=(replace(profiles[0], kp=None), *profiles[1:])),
    )
    with pytest.raises(ValueError):
        normal_config.validate()
    serial.assert_not_called()


def test_operator_mapping_is_bound_without_a_micro_history(normal_config):
    normal_config.leader_joint_directions = {**normal_config.leader_joint_directions, "wrist_roll": -1}
    with pytest.raises(ValueError, match="leader configuration changed"):
        normal_config.validate()


def test_operator_runtime_uses_normal_start_and_allows_explicit_force_target_change():
    class Clock:
        now = 1_000_000_000

        def __call__(self):
            return self.now

    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor, device = ManualSensor(plant), FakeGripper(plant, clock=clock)
    device.operator_teleoperation_config = SimpleNamespace()
    runtime = ForceRuntime(
        device,
        sensor,
        ForceTeacher(
            TeacherConfig.fake(
                force_target_N=2, target_max_N=4, gripper_follow_speed_deg_s=1, contact_force_threshold_N=0.5
            ),
            1,
        ),
        SafetySupervisor(SafetyConfig.fake(), 1),
        MemoryTrace(),
        RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES),
        clock=clock,
    )
    starts = []

    def start(*, guard, leader_sample):
        guard()
        starts.append(leader_sample())
        return runtime.latest_joint_snapshot

    device.start_operator_teleoperation = start

    def tick(kind=None, value=None):
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
                        "leader_config_sha256": "a" * 64,
                    }
                ),
            )
        )
        if kind:
            runtime.post(kind, version=clock(), epoch=runtime.epoch, value=value)
        return runtime.tick()

    runtime.open()
    try:
        tick("arm")
        row = tick("start", {"mode": "hybrid", "position_deg": None})
        assert row["phase"] == "FOLLOW" and not row["submitted_action"]
        assert len(starts) == 1 and len(starts[0].positions_deg) == 7
        row = tick("target", 3)
        assert row["phase"] != "FAULT", row["failure_reason"]
        assert row["force_target_requested_N"] == 3
        assert runtime.controller.config.force_target_N == 3
        assert len(row["submitted_action"]) == 7
    finally:
        runtime.close()
