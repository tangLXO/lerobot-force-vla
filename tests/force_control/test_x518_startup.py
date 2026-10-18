"""Exercise the production X518 selection API with a simulated gripper only."""

import json
import threading

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig
from tests.force_control.hardware_helpers import Clock, MemoryTrace


@pytest.mark.parametrize("mode,phase", [("force", "FORCE_TRACK"), ("hybrid", "FOLLOW")])
def test_x518_starts_and_runs_gripper_owner_with_explicit_trace_boundary(monkeypatch, mode, phase):
    clock = Clock()
    # No acquisition worker, socket, serial port, discovery or real connect.
    sensor = X518Sensor(
        X518SensorConfig(
            channels={
                "left.normal_force": X518ChannelConfig(channel=1),
                "right.normal_force": X518ChannelConfig(channel=2),
            }
        )
    )
    monkeypatch.setattr(sensor, "connect", lambda: pytest.fail("X518 hardware connect was called"))
    monkeypatch.setattr(sensor._client, "connect", lambda: pytest.fail("X518 transport connect was called"))
    sensor._lifecycle_active = True
    sensor._thread = threading.Thread(target=sensor._stop_event.wait, daemon=True)
    sensor._thread.start()
    plant = FakePlant(response_time_s=0)
    device = FakeGripper(plant, clock=clock.perf_counter_ns)
    controller = ForceTeacher(
        TeacherConfig.fake(gripper_follow_speed_deg_s=2, contact_force_threshold_N=0.5), close_sign=1
    )
    trace = MemoryTrace()
    runtime = ForceRuntime(
        device,
        sensor,
        controller,
        SafetySupervisor(SafetyConfig.fake(), close_sign=1),
        trace,
        clock=clock.perf_counter_ns,
    )
    startup_samples = []

    def simulated_start(snapshot, guard):
        startup_samples.append(guard())
        return snapshot.get("gripper").position_deg

    device.start_gripper = simulated_start

    def tick():
        clock.now += 20_000_000
        sample = sensor._publish_sample(
            {"left.normal_force": 0.1, "right.normal_force": 0.1},
            clock.now,
            arrival_timestamp_ns=clock.now,
        )
        runtime.post_arm_intent(
            ArmIntent(
                goals=(("gripper.pos", 0.1),),
                version=sample.sequence,
                epoch=runtime.epoch,
                captured_ns=clock.now,
                ttl_ns=100_000_000,
                metadata_json=json.dumps(
                    {
                        "raw_action_deg": {"gripper.pos": 0.1},
                        "leader_sequence": sample.sequence,
                        "leader_request_started_ns": clock.now,
                    }
                ),
            )
        )
        return runtime.tick()

    runtime.open()
    try:
        runtime.post("arm", version=0, epoch=runtime.epoch)
        row = tick()
        assert row["phase"] == "ARMED", row.get("failure_reason")
        runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": mode, "position_deg": None})
        row = tick()
        assert row["phase"] == phase and startup_samples
        assert startup_samples[0].sequence == row["force_sequence"]
        assert runtime._startup_guard() is startup_samples[0]
        row = tick()
        assert row["phase"] == phase and row["send_status"] == "submitted"
        assert row["force_sequence"] > startup_samples[0].sequence
        assert sensor._client._socket is None and sensor._process is None
    finally:
        runtime.close()
        sensor.disconnect()
