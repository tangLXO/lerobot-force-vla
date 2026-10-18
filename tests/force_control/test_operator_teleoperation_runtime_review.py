"""Normal operator integration with synthetic inputs; no physical ports or cameras."""

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
def bench():
    class Clock:
        now = 1_000_000_000

        def __call__(self):
            return self.now

    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    device.operator_teleoperation_config = SimpleNamespace()
    teacher = ForceTeacher(
        TeacherConfig.fake(force_target_N=2, gripper_follow_speed_deg_s=1, contact_force_threshold_N=0.5), 1
    )
    runtime = ForceRuntime(
        device,
        sensor,
        teacher,
        SafetySupervisor(SafetyConfig.fake(max_run_time_s=None, max_total_travel_deg=None), 1),
        MemoryTrace(),
        RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES),
        clock=clock,
    )
    samples = []

    def start(*, guard, leader_sample):
        guard()
        samples.append(leader_sample())
        return runtime.latest_joint_snapshot

    device.start_operator_teleoperation = start
    runtime.open()
    value = SimpleNamespace(
        runtime=runtime, device=device, sensor=sensor, clock=clock, startup_samples=samples
    )
    yield value
    runtime.close()


def tick(bench, *, command=None, force=0.1):
    bench.clock.now += 20_000_000
    now = bench.clock()
    bench.sensor._publish_sample(
        {"left.normal_force": force, "right.normal_force": force}, now, arrival_timestamp_ns=now
    )
    goals = {f"{name}.pos": 0.0 for name in FAKE_B601_JOINT_NAMES}
    bench.runtime.post_arm_intent(
        ArmIntent(
            tuple(goals.items()),
            now,
            bench.runtime.epoch,
            now,
            200_000_000,
            json.dumps(
                {
                    "raw_action_deg": goals,
                    "leader_sequence": now,
                    "leader_request_started_ns": now,
                    "leader_response_completed_ns": now,
                    "leader_config_sha256": "f" * 64,
                }
            ),
        )
    )
    if command is not None:
        bench.runtime.post(command[0], version=now, epoch=bench.runtime.epoch, value=command[1])
    return bench.runtime.tick()


def start(bench):
    tick(bench, command=("arm", None))
    return tick(bench, command=("start", {"mode": "hybrid", "position_deg": None}))


def test_normal_operator_preserves_explicit_user_target_and_live_parameter_updates(bench):
    start(bench)
    row = tick(bench, command=("target", 2.5))
    assert row["phase"] != "FAULT", row["failure_reason"]
    assert bench.runtime.controller.config.force_target_N == 2.5
    teacher = replace(bench.runtime.controller.config, kp_deg_per_N=0.2)
    safety = replace(bench.runtime.safety.config, max_close_speed_deg_s=4.0)
    row = tick(bench, command=("parameters", {"teacher": teacher, "safety": safety}))
    assert row["phase"] != "FAULT", row["failure_reason"]
    assert bench.runtime.controller.config == teacher and bench.runtime.safety.config == safety
    assert bench.runtime.safety.config.max_run_time_s is None
    assert bench.runtime.safety.config.max_total_travel_deg is None


@pytest.mark.parametrize("protection", ["raw_force", "user_stop"])
def test_normal_operator_keeps_raw_force_and_user_stop_priority(bench, protection):
    start(bench)
    submitted = len(bench.device.submitted_goals)
    if protection == "user_stop":
        bench.runtime.request_stop()
    row = tick(bench, force=6 if protection == "raw_force" else 0.1)
    assert row["phase"] == "FAULT"
    expected_reason = "force_hard_limit" if protection == "raw_force" else "stop"
    assert expected_reason in row["failure_reason"]
    assert len(bench.device.submitted_goals) == submitted
