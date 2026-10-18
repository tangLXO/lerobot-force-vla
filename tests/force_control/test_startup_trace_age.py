"""Startup completion may advance the decision anchor without new input reads."""

import json
from types import SimpleNamespace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES, FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.trace import BenchTrace, validate_trace
from tests.force_control.hardware_helpers import Clock, ManualSensor


@pytest.mark.parametrize("full_arm", [False, True])
def test_startup_anchor_recomputes_ages_and_preserves_original_receive_evidence(tmp_path, full_arm):
    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock.perf_counter_ns)
    safety_config = SafetyConfig.fake()
    trace = BenchTrace(
        tmp_path / "trace",
        sensor,
        {
            "backend": "fake",
            "force_max_age_ms": safety_config.force_max_age_ms,
            "joint_feedback_max_age_ms": safety_config.joint_feedback_max_age_ms,
        },
    )
    runtime = ForceRuntime(
        device,
        sensor,
        ForceTeacher(
            TeacherConfig.fake(gripper_follow_speed_deg_s=2, contact_force_threshold_N=0.5), close_sign=1
        ),
        SafetySupervisor(safety_config, close_sign=1),
        trace,
        RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES if full_arm else ("gripper",)),
        clock=clock.perf_counter_ns,
    )
    original = {}

    def delay_startup(snapshot, guard):
        original["snapshot"] = snapshot
        original["sample"] = guard()
        clock.now += 10_000_000
        return snapshot.get("gripper").position_deg

    if full_arm:
        device.operator_teleoperation_config = SimpleNamespace()

        def start_operator(*, guard, leader_sample):
            delay_startup(runtime.latest_joint_snapshot, guard)
            return device.read_joint_snapshot(FAKE_B601_JOINT_NAMES, deadline_ns=clock.now + 10_000_000)

        device.start_operator_teleoperation = start_operator
    else:
        device.start_gripper = delay_startup

    def tick():
        clock.now += 10_000_000
        sample = sensor._publish_sample(
            {"left.normal_force": 0.1, "right.normal_force": 0.1},
            clock.now,
            arrival_timestamp_ns=clock.now,
        )
        if full_arm:
            goals = {f"{name}.pos": 0.0 for name in FAKE_B601_JOINT_NAMES}
            runtime.post_arm_intent(
                ArmIntent(
                    tuple(goals.items()),
                    sample.sequence,
                    runtime.epoch,
                    clock.now,
                    100_000_000,
                    json.dumps(
                        {
                            "raw_action_deg": goals,
                            "leader_sequence": sample.sequence,
                            "leader_request_started_ns": clock.now,
                        }
                    ),
                )
            )
        return runtime.tick()

    trace.start()
    runtime.open()
    try:
        tick()
        runtime.post("arm", version=0, epoch=runtime.epoch)
        tick()
        runtime.post(
            "start",
            version=1,
            epoch=runtime.epoch,
            value={"mode": "hybrid" if full_arm else "force", "position_deg": None},
        )
        row = tick()
        tick()
    finally:
        runtime.close()
        trace.close("COMPLETED", complete=True)

    assert row["event"] == "start" and row["phase"] == ("FOLLOW" if full_arm else "FORCE_TRACK")
    feedback = original["snapshot"].get("gripper")
    sample = original["sample"]
    assert row["decision_anchor_ns"] - feedback.receive_timestamp_ns == 10_000_000
    assert row["feedback_age_ns"] == row["joint_age_ns"] == 10_000_000
    assert row["force_age_ns"] == 10_000_000
    assert row["feedback_received_ns"] == feedback.receive_timestamp_ns
    assert row["feedback_sequence"] == feedback.receive_sequence
    assert row["force_timestamp_ns"] == sample.timestamp_ns
    assert row["force_arrival_timestamp_ns"] == sample.arrival_timestamp_ns
    assert row["force_sequence"] == sample.sequence
    assert row["joint_snapshot"]["completed_ns"] == original["snapshot"].completed_ns
    assert row["joint_snapshot"]["joints"][-1]["age_ns"] == feedback.age_ns == 0
    assert not row["label_valid"] and row["send_status"] == "none"
    report = validate_trace(trace.root)
    assert report["valid"], report["errors"]
