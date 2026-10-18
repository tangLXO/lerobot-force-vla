"""End-to-end deterministic serial latency replay; never opens a real device."""

import cProfile
import json
import pstats
import time
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport as readonly
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace
from tests.force_control.operator_teleoperation_helpers import (
    JOINT_NAMES,
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)

PROCESS_TIME_NS = time.process_time_ns


class ProfiledWorkClock:
    """Charge explicit measured work costs without coarse Windows CPU-clock jumps."""

    def __init__(self):
        self.now = 1_000_000_000

    def perf_counter_ns(self):
        return self.now


@contextmanager
def latency_scope(tmp_path, monkeypatch, *, fast_timeout):
    files = make_operator_files(tmp_path)
    files.control = replace(
        files.control,
        startup_budget_s=1.0,
        send_budget_s=0.02,
        fault_budget_s=0.2,
        feedback_max_age_s=0.05,
        watchdog_ticks=2000,
    )
    files.operator = files.configuration = replace(
        files.operator,
        watchdog_ticks=2000,
        alignment_tolerance_deg=None,
        preload_tolerance_deg=None,
        leader_max_age_s=0.15,
        max_command_gap_s=None,
    )
    scope = make_operator_scope(tmp_path, monkeypatch, files=files, connect=False)
    serial, clock = scope.serial, scope.clock
    clock.now = 10_000_000_000
    serial.watchdog_readback = 0
    serial.watchdog_overrides[7] = 2000
    serial.dcb_timeout_changes = []
    serial.fast_timeout_changes = []
    serial.reply_roundtrip_ns = 460_000
    serial.first_reply_chunk = False

    class DelayedSerial(type(serial)):
        def __setattr__(self, name, value):
            if name in ("timeout", "write_timeout") and name in self.__dict__:
                self.dcb_timeout_changes.append((name, value))
                clock.now += 1_800_000
            super().__setattr__(name, value)

    serial.__class__ = DelayedSerial
    original_write, original_read = serial.write, serial.read

    def write(packet):
        result = original_write(packet)  # The shared fake charges 0.1 ms.
        serial.first_reply_chunk = bool(serial.pending)
        refresh()
        return result

    def read(size):
        first = serial.first_reply_chunk
        result = original_read(min(size, 1) if first else size)
        serial.first_reply_chunk = False
        if not first and result:
            # Two reads and one write total 0.46 ms, as in the captured trace.
            clock.now += serial.reply_roundtrip_ns - 300_000
        refresh()
        return result

    monkeypatch.setattr(serial, "write", write)
    monkeypatch.setattr(serial, "read", read)

    def set_timeout(port, name, value):
        if fast_timeout:
            port.fast_timeout_changes.append((name, value))
            object.__setattr__(port, name, value)
            clock.now += 20_000  # Simulated native timeout-only update.
        else:
            setattr(port, name, value)

    monkeypatch.setattr(readonly, "set_serial_timeout", set_timeout)
    monkeypatch.setattr(backend, "set_serial_timeout", set_timeout)
    sensor = ManualSensor(FakePlant(response_time_s=0))
    runtime = ForceRuntime(
        scope.device,
        sensor,
        ForceTeacher(TeacherConfig.fake(gripper_follow_speed_deg_s=1, contact_force_threshold_N=0.5), 1),
        SafetySupervisor(SafetyConfig.fake(max_run_time_s=None, max_total_travel_deg=None), 1),
        MemoryTrace(),
        RuntimeConfig(control_hz=100, feedback_budget_ms=15, tick_deadline_ms=40, joint_names=JOINT_NAMES),
        clock=clock.perf_counter_ns,
    )
    version = 0
    last_force_ns = last_leader_ns = None

    def refresh():
        nonlocal version, last_force_ns, last_leader_ns
        now = clock.now
        if last_force_ns is None or now - last_force_ns >= 2_500_000:
            sensor._publish_sample(
                {"left.normal_force": 0.1, "right.normal_force": 0.1},
                now,
                arrival_timestamp_ns=now,
            )
            last_force_ns = now
        if last_leader_ns is None or now - last_leader_ns >= 20_000_000:
            version += 1
            goals = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
            runtime.post_arm_intent(
                ArmIntent(
                    tuple(goals.items()),
                    version,
                    runtime.epoch,
                    now,
                    150_000_000,
                    json.dumps(
                        {
                            "raw_action_deg": goals,
                            "leader_sequence": version,
                            "leader_request_started_ns": now,
                            "leader_response_completed_ns": now,
                            "leader_config_sha256": files.operator.leader_config_sha256,
                        }
                    ),
                )
            )
            last_leader_ns = now

    def tick(kind=None):
        clock.now += 10_000_000
        refresh()
        if kind is not None:
            runtime.post(
                kind,
                version=clock.now,
                epoch=runtime.epoch,
                value={"mode": "hybrid", "position_deg": None} if kind == "start" else None,
            )
        return runtime.tick()

    try:
        runtime.open()
        yield SimpleNamespace(scope=scope, runtime=runtime, serial=serial, tick=tick)
    finally:
        with suppress(Exception):
            runtime.close()
        close_operator_scope(scope)
        assert not scope.device.has_resources


@pytest.mark.parametrize("fast_timeout", [False, True])
def test_cold_timeouts_connect_start_and_first_active_frame_without_larger_budgets(
    tmp_path, monkeypatch, record_property, fast_timeout
):
    wall_clock = time.perf_counter
    with latency_scope(tmp_path, monkeypatch, fast_timeout=fast_timeout) as rig:
        assert rig.serial.write_timeout == 0.02
        assert rig.serial.timeout == 0.002
        cold = rig.tick()
        if not fast_timeout:
            assert cold["phase"] == "FAULT" and not cold["feedback_valid"]
            assert len(rig.serial.dcb_timeout_changes) >= 5
            assert rig.scope.device.startup_receipt is None
            return
        assert cold["feedback_valid"] and cold["phase"] == "DISABLED"
        assert cold["loop_elapsed_s"] < 0.015
        assert len(cold["joint_snapshot"]["joints"]) == 7
        assert rig.tick("arm")["phase"] == "ARMED"
        started = rig.tick("start")
        assert started["phase"] == "FOLLOW", started["failure_reason"]
        assert rig.scope.device.startup_receipt["complete"]
        received = rig.scope.device._latest_snapshot
        before = len(rig.serial.writes)
        begin = wall_clock()
        active = rig.tick()
        active_cpu_ms = (wall_clock() - begin) * 1000
        assert active["phase"] == "FOLLOW" and active["send_status"] == "submitted"
        packets = rig.serial.writes[before:]
        queries = [packet for packet in packets if int.from_bytes(packet[13:17], "little") == 0x7FF]
        commands = [packet for packet in packets if int.from_bytes(packet[13:17], "little") != 0x7FF]
        assert queries == [] and len(commands) == 7
        assert active["feedback_received_ns"] == received.get("gripper").receive_timestamp_ns
        assert active["feedback_sequence"] == received.get("gripper").receive_sequence
        assert active["joint_snapshot"]["completed_ns"] == received.completed_ns
        assert len(active["per_joint_results"]) == 7
        assert all(item["status"] == "success" for item in active["per_joint_results"])
        assert active["loop_elapsed_s"] < 0.01
        assert rig.serial.dcb_timeout_changes == []
        assert rig.serial.fast_timeout_changes
        assert rig.runtime.config.feedback_budget_ms == 15
        assert rig.runtime.config.tick_deadline_ms == 40
        assert rig.scope.device.config.send_budget_s == 0.02

        profile = cProfile.Profile()
        profiled = profile.runcall(rig.tick)
        assert profiled["phase"] == "FOLLOW" and profiled["send_status"] == "submitted"
        functions = {}
        for (_filename, _line, name), (_primitive, calls, own_s, total_s, _callers) in pstats.Stats(
            profile
        ).stats.items():
            if name in ("require_binding", "wait_guard", "_startup_guard", "send", "tick"):
                functions[name] = {
                    "calls": calls,
                    "own_ms": round(own_s * 1000, 3),
                    "inclusive_ms": round(total_s * 1000, 3),
                }
        warm_sensor = ManualSensor(FakePlant(response_time_s=0))
        for index in range(800):
            stamp = rig.scope.clock.now - (799 - index) * 2_500_000
            warm_sensor._publish_sample(
                {"left.normal_force": 0.1, "right.normal_force": 0.1},
                stamp,
                arrival_timestamp_ns=stamp,
            )
        rig.runtime.sensor = warm_sensor
        begin = wall_clock()
        warm = rig.tick()
        warm_cpu_ms = (wall_clock() - begin) * 1000
        assert warm["phase"] == "FOLLOW" and warm["send_status"] == "submitted"
        assert len(warm_sensor._history.snapshot()) == 800
        report = {
            "synthetic_only": True,
            "hardware_accessed": False,
            "cold_frame_virtual_ms": round(cold["loop_elapsed_s"] * 1000, 3),
            "startup_virtual_ms": round(started["loop_elapsed_s"] * 1000, 3),
            "active_frame_virtual_ms": round(active["loop_elapsed_s"] * 1000, 3),
            "active_frame_cpu_wall_ms": round(active_cpu_ms, 3),
            "warm_800_sample_frame_cpu_wall_ms": round(warm_cpu_ms, 3),
            "profiled_functions": functions,
        }
        record_property("offline_serial_latency", json.dumps(report))
        print("OFFLINE_SERIAL_LATENCY " + json.dumps(report, sort_keys=True))


def test_fast_timeout_path_keeps_a_late_response_invalid(tmp_path, monkeypatch):
    with latency_scope(tmp_path, monkeypatch, fast_timeout=True) as rig:
        rig.serial.reply_roundtrip_ns = 18_000_000
        row = rig.tick()
        assert row["phase"] == "FAULT" and not row["feedback_valid"]
        assert row["joint_snapshot"]["completed_ns"] > row["joint_snapshot"]["deadline_ns"]
        assert row["send_status"] == "none" and not row["label_valid"]
        assert rig.scope.device.startup_receipt is None


def test_profiled_work_1000_frames_with_first_command_spike_preserve_real_receive_evidence(
    tmp_path, monkeypatch, record_property
):
    from tests.force_control import operator_teleoperation_helpers

    monkeypatch.setattr(operator_teleoperation_helpers, "Clock", ProfiledWorkClock)
    with latency_scope(tmp_path, monkeypatch, fast_timeout=True) as rig:
        session = rig.scope.device._full_validation_session
        original_guard, original_binding = session.wait_guard, session.require_binding

        def charged_guard():
            rig.scope.clock.now += 50_000
            return original_guard()

        def charged_binding(*args, **kwargs):
            rig.scope.clock.now += 200_000
            return original_binding(*args, **kwargs)

        monkeypatch.setattr(session, "wait_guard", charged_guard)
        monkeypatch.setattr(session, "require_binding", charged_binding)
        # Emulate a bounded trace consumer rather than retaining 1,000 UI frames.
        rig.runtime.trace.rows = deque(maxlen=8)
        original_post = rig.runtime.post_arm_intent

        def post_with_full_diagnostics(intent):
            metadata = json.loads(intent.metadata_json)
            metadata["leader_joint_diagnostics"] = [
                {
                    "name": name,
                    "sdk_reliable": None,
                    "sdk_angle_deg": 0.0,
                    "status": "packet_validated_no_sdk_angle_filter",
                    "receive_timestamp_ns": intent.captured_ns,
                    "receive_sequence": intent.version,
                    "motor_status": 0,
                }
                for name in JOINT_NAMES
            ]
            return original_post(replace(intent, metadata_json=json.dumps(metadata)))

        monkeypatch.setattr(rig.runtime, "post_arm_intent", post_with_full_diagnostics)
        assert rig.tick()["phase"] == "DISABLED"
        assert rig.tick("arm")["phase"] == "ARMED"
        assert rig.tick("start")["phase"] == "FOLLOW"
        original_write = rig.serial.write
        first_command = True

        def spike_first_command(packet):
            nonlocal first_command
            command = int.from_bytes(packet[13:17], "little") != 0x7FF
            rig.serial.reply_roundtrip_ns = 10_440_700 if command and first_command else 460_000
            if command:
                first_command = False
            return original_write(packet)

        monkeypatch.setattr(rig.serial, "write", spike_first_command)
        maximum_ms = maximum_wall_ms = 0.0
        cpu_started = PROCESS_TIME_NS()
        for frame in range(1000):
            previous = rig.scope.device._latest_snapshot
            before = len(rig.serial.writes)
            wall_started = time.perf_counter()
            row = rig.tick()
            maximum_wall_ms = max(maximum_wall_ms, (time.perf_counter() - wall_started) * 1000)
            maximum_ms = max(maximum_ms, row["loop_elapsed_s"] * 1000)
            assert row["phase"] == "FOLLOW", (
                frame,
                row["failure_reason"],
                row.get("joint_error"),
                row["loop_elapsed_s"],
            )
            assert row["send_status"] == "submitted"
            assert row["feedback_received_ns"] == previous.get("gripper").receive_timestamp_ns
            assert row["feedback_sequence"] == previous.get("gripper").receive_sequence
            assert row["feedback_age_ns"] == row["decision_anchor_ns"] - row["feedback_received_ns"]
            packets = rig.serial.writes[before:]
            assert len(packets) == 7 and all(
                int.from_bytes(packet[13:17], "little") != 0x7FF for packet in packets
            )
        assert rig.runtime.config.feedback_budget_ms == 15
        assert rig.runtime.config.tick_deadline_ms == 40
        assert rig.scope.device.config.send_budget_s == 0.02
        record_property("profiled_work_model_max_frame_ms", maximum_ms)
        record_property("observed_max_wall_ms_without_a_hard_gate", maximum_wall_ms)
        record_property(
            "observed_process_cpu_total_ms_without_a_hard_gate", (PROCESS_TIME_NS() - cpu_started) / 1e6
        )
        print(
            "OFFLINE_1000_FRAME_WORK_MODEL "
            + json.dumps(
                {
                    "frames": 1000,
                    "guard_charge_us": 50,
                    "binding_charge_us": 200,
                    "max_model_frame_ms": maximum_ms,
                    "max_wall_ms_observation_only": maximum_wall_ms,
                    "hardware_accessed": False,
                },
                sort_keys=True,
            )
        )
