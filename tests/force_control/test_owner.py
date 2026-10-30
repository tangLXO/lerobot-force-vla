"""P3 ownership, causal frame binding, leader TTL, partial sends and handover tests."""

import threading
import time
from dataclasses import replace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import (
    FAKE_B601_JOINT_NAMES,
    FakeForceSensor,
    FakeGripper,
    FakePlant,
)
from lerobot.force_control.owner import CachedRobotProxy, OwnerService
from lerobot.force_control.runtime import ArmIntent, ArmMailbox, ForceRuntime, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher


class MemoryTrace:
    run_uid = "owner-test"
    start_sequence = 0

    def __init__(self):
        self.rows = []

    def check_health(self):
        pass

    def submit(self, row):
        self.rows.append(row.copy())
        return True


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now


class ManualSensor(FakeForceSensor):
    @property
    def is_connected(self):
        return True


def make_runtime(sensor, device, trace, *, clock=time.perf_counter_ns, config=None):
    return ForceRuntime(
        device,
        sensor,
        ForceTeacher(TeacherConfig.fake(), close_sign=1),
        SafetySupervisor(SafetyConfig.fake(), close_sign=1),
        trace,
        config or RuntimeConfig(joint_names=FAKE_B601_JOINT_NAMES, arm_intent_required=True),
        clock=clock,
    )


@pytest.fixture
def rig():
    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor, device, trace = ManualSensor(plant), FakeGripper(plant, clock=clock), MemoryTrace()
    runtime = make_runtime(sensor, device, trace, clock=clock)
    runtime.open()
    yield runtime, device, sensor, trace, clock
    runtime.close()


def action(gripper=99.0):
    return {f"{name}.pos": (gripper if name == "gripper" else 0.3) for name in FAKE_B601_JOINT_NAMES}


def put(rig, version, *, values=None, captured_ns=None, epoch=None, ttl_ns=200_000_000):
    runtime, _, _, _, clock = rig
    return runtime.post_arm_intent(
        ArmIntent(
            tuple((values or action()).items()),
            version,
            runtime.epoch if epoch is None else epoch,
            clock() if captured_ns is None else captured_ns,
            ttl_ns,
        )
    )


def tick(rig, *, dt=0.02, force=1.0):
    runtime, _, sensor, _, clock = rig
    clock.now += round(dt * 1e9)
    sensor._publish_sample(
        {"left.normal_force": force, "right.normal_force": force}, clock(), arrival_timestamp_ns=clock()
    )
    return runtime.tick()


def activate(rig):
    runtime = rig[0]
    put(rig, 0)
    runtime.post("arm", version=0, epoch=runtime.epoch)
    tick(rig)
    runtime.post("start", version=1, epoch=runtime.epoch)
    return tick(rig)


def test_intertick_rx_failure_is_published_before_next_control_deadline(rig, monkeypatch):
    runtime, device, _, trace, clock = rig
    activate(rig)
    prior_start = runtime.last_tick_start
    calls = []

    def receive():
        calls.append(threading.get_ident())
        raise TimeoutError("missing pipelined feedback")

    monkeypatch.setattr(device, "service_pending_io", receive, raising=False)
    row = runtime.wait_for_next_tick(clock() + 20_000_000)
    assert row["phase"] == "FAULT" and row["communication_only"]
    assert "missing pipelined feedback" in row["failure_reason"]
    assert runtime.last_tick_start == prior_start
    assert trace.rows[-1] == row
    assert calls == [threading.get_ident()]


def test_direct_runtime_wait_services_rx_without_issuing_commands(rig, monkeypatch):
    runtime, device, _, _, clock = rig
    activate(rig)
    before = list(device.calls)
    serviced = []
    monkeypatch.setattr(device, "service_pending_io", lambda: serviced.append(clock()), raising=False)
    monkeypatch.setattr(device, "command_feedback_pending", True, raising=False)
    monkeypatch.setattr(
        time, "sleep", lambda seconds: setattr(clock, "now", clock.now + round(seconds * 1e9))
    )
    assert runtime.wait_for_next_tick(clock() + 1_000_000) is None
    assert len(serviced) >= 5 and device.calls == before


def test_recording_freeze_waits_for_feedback_and_publishes_a_tail(rig, monkeypatch):
    runtime, device, _, _, _ = rig
    activate(rig)
    owner = OwnerService(runtime)
    subscription = owner.subscribe()
    owner._freeze_requests.append(subscription)
    monkeypatch.setattr(device, "command_feedback_pending", True, raising=False)
    owner._settle_subscriptions()
    assert not subscription.closed and subscription.queue.empty()
    monkeypatch.setattr(device, "command_feedback_pending", False)
    owner._settle_subscriptions()
    tail = subscription.get()
    assert subscription.closed and tail.row["communication_only"]
    assert tail.row["send_status"] == "none" and not tail.row["submitted_action"]


def test_owner_filters_leader_gripper_and_submits_explicit_seven_joints(rig):
    row = activate(rig)
    assert row["arm_intent"]["gripper.pos"] == 99.0
    assert row["submitted_action"]["gripper.pos"] == row["q_sent_deg"] == 0.0
    assert row["arm_filtered_keys"] == ["gripper.pos"]
    assert set(row["submitted_action"]) == set(action())
    assert row["submitted_action"]["wrist_yaw.pos"] == 0.3
    assert len(row["per_joint_results"]) == 7
    assert {operation for operation, _ in rig[1].calls} == {"connect", "read", "send_partial"}


def test_partial_command_never_fills_missing_wrist_yaw(rig):
    runtime, *_ = rig
    values = action()
    del values["wrist_yaw.pos"]
    put(rig, 0, values=values)
    runtime.post("arm", version=0, epoch=0)
    tick(rig)
    runtime.post("start", version=1, epoch=0)
    row = tick(rig)
    assert "wrist_yaw.pos" not in row["submitted_action"]
    assert "wrist_yaw" not in {result["name"] for result in row["per_joint_results"]}


def test_fresh_mailbox_arrival_does_not_extend_old_capture_ttl(rig):
    activate(rig)
    runtime, device, _, _, clock = rig
    old_epoch, old_sent = runtime.epoch, runtime.last_sent
    assert put(rig, 1, captured_ns=clock() - 300_000_000)
    before = len(device.calls)
    row = tick(rig)
    assert "ttl_expired" in row["failure_reason"]
    assert row["submitted_action"] == {} and runtime.last_sent == old_sent
    assert [op for op, _ in device.calls[before:]] == ["read", "emergency_stop"]
    assert not put(rig, 999, epoch=old_epoch)


def test_partial_failure_keeps_success_evidence_without_advancing_gripper_reference(rig):
    activate(rig)
    runtime, device, *_ = rig
    previous = runtime.last_sent
    device.send_failure_joint = "elbow_flex"
    row = tick(rig)
    assert runtime.last_sent == previous
    assert row["send_status"] == "unknown" and not row["label_valid"]
    assert set(row["submitted_action"]) == {"shoulder_pan.pos", "shoulder_lift.pos"}
    assert [item["status"] for item in row["per_joint_results"]] == [
        "success",
        "success",
        "unknown",
        "not_attempted",
        "not_attempted",
        "not_attempted",
        "not_attempted",
    ]
    before = len(device.calls)
    tick(rig)
    assert not any(operation == "send_partial" for operation, _ in device.calls[before:])


def test_frozen_arm_joint_faults_even_with_fresh_gripper(rig):
    activate(rig)
    rig[1].freeze_joints.add("wrist_yaw")
    row = tick(rig)
    assert row["failure_reason"] == "RuntimeError('joint_feedback_not_updated:wrist_yaw')"
    assert row["submitted_action"] == {}


def test_teleop_return_needs_new_epoch_fresh_matching_gripper(rig):
    runtime, _, _, _, clock = rig
    runtime.config = replace(runtime.config, handover_tolerance_deg=0.05)
    assert put(rig, 1, values=action(gripper=8.0))
    old_epoch = runtime.epoch
    runtime.post("teleop", version=0, epoch=old_epoch)
    row = tick(rig)
    assert runtime.mode == "HANDOVER_PENDING" and row["submitted_action"] == {}
    assert not put(rig, 99, values=action(gripper=0), epoch=old_epoch)
    # Even a current-epoch re-enqueue of old capture evidence cannot hand control back.
    assert put(rig, 0, values=action(gripper=0), captured_ns=clock() - 10_000_000)
    row = tick(rig)
    assert runtime.mode == "HANDOVER_PENDING" and row["submitted_action"] == {}
    assert put(rig, 1, values=action(gripper=8.0))
    assert tick(rig)["submitted_action"] == {}
    assert put(rig, 2, values=action(gripper=0.0))
    row = tick(rig)
    assert row["mode"] == "TELEOP" and row["event"] == "teleop_synchronized"
    assert row["submitted_action"]["gripper.pos"] == 0.0
    assert not row["teacher_active"] and not row["label_valid"]
    assert row["arm_filtered_keys"] == []
    assert put(rig, 3, values=action(gripper=0.05))
    moving = tick(rig)
    assert moving["phase"] == "DISABLED" and moving["mode"] == "TELEOP"
    assert moving["delta_raw_deg"] == pytest.approx(0.05)
    assert moving["q_sent_deg"] > 0
    runtime.post("takeover", version=1, epoch=runtime.epoch)
    row = tick(rig)
    assert row["mode"] == "TEACHER" and row["phase"] == "ARMED"
    assert row["submitted_action"] == {} and runtime.arm_mailbox.latest() is None


def test_active_teacher_cannot_return_directly_to_teleop(rig):
    activate(rig)
    runtime = rig[0]
    runtime.config = replace(runtime.config, handover_tolerance_deg=0.05)
    runtime.post("teleop", version=2, epoch=runtime.epoch)
    row = tick(rig)
    assert row["phase"] == "FAULT" and "RELEASE" in row["failure_reason"]
    assert row["submitted_action"] == {}


def test_arm_mailbox_latest_wins_and_epoch_invalidation():
    mailbox = ArmMailbox()
    first = ArmIntent((("gripper.pos", 0.0),), 1, 0, 100, 1000)
    assert mailbox.put(first)
    assert not mailbox.put(first)
    assert mailbox.put(replace(first, version=2))
    assert mailbox.latest().version == 2
    mailbox.invalidate(1)
    assert mailbox.latest() is None and not mailbox.put(replace(first, version=999))


def test_thread_owner_progresses_while_external_reader_blocks_and_frame_binds_after_anchor():
    plant = FakePlant(contact_position_deg=-0.5)
    sensor, device, trace = FakeForceSensor(plant), FakeGripper(plant), MemoryTrace()
    config = RuntimeConfig(
        joint_names=FAKE_B601_JOINT_NAMES,
        arm_intent_required=True,
        command_ttl_ms=1000,
        feedback_budget_ms=30,
        tick_deadline_ms=80,
    )
    runtime = make_runtime(sensor, device, trace, config=config)
    service, sensor_started = OwnerService(runtime), False
    blocked, release = threading.Event(), threading.Event()

    def external_reader():
        blocked.set()
        release.wait(2)

    reader = threading.Thread(target=external_reader)
    try:
        sensor.connect()
        sensor_started = True
        sensor.async_read(1000)
        service.submit_arm_intent(action(), captured_ns=time.perf_counter_ns())
        service.start()
        service.wait_for_command(service.post("arm"))
        service.wait_for_command(service.post("start"))
        proxy = CachedRobotProxy(service)
        before = proxy.get_observation()
        anchor = time.perf_counter_ns()
        version = service.submit_arm_intent(action(), captured_ns=anchor)
        bound = service.await_tick_after(anchor, version)
        assert bound.row["decision_anchor_ns"] >= anchor
        assert set(bound.row["submitted_action"]) == set(before)
        assert proxy.last_joint_snapshot.completed_ns <= anchor
        with pytest.raises(RuntimeError, match="timestamped"):
            proxy.send_action(action())
        reader.start()
        assert blocked.wait(1)
        first_tick = service.snapshot().tick_id
        while service.snapshot().tick_id < first_tick + 3:
            version = service.submit_arm_intent(action(), captured_ns=time.perf_counter_ns())
            service.await_tick_after(time.perf_counter_ns(), version)
        assert not release.is_set()
        assert {thread_id for _, thread_id in device.calls} == {service._thread.ident}
        assert threading.get_ident() not in {thread_id for _, thread_id in device.calls}
        copy_of_row = service.snapshot().row
        copy_of_row["submitted_action"].clear()
        assert service.snapshot().row["submitted_action"]
        previous_epoch = runtime.epoch
        runtime.request_stop()
        with pytest.raises(RuntimeError, match="Owner fault"):
            service.await_tick_after(time.perf_counter_ns(), version)
        reset = service.post("reset")
        reset_row = service.wait_for_command(reset).row
        assert reset_row["phase"] == "DISABLED" and reset_row["submitted_action"] == {}
        with pytest.raises(RuntimeError, match="epoch"):
            service.submit_arm_intent(action(), captured_ns=time.perf_counter_ns(), epoch=previous_epoch)
    finally:
        release.set()
        if reader.ident is not None:
            reader.join(1)
        service.stop()
        if sensor_started:
            sensor.disconnect()
    assert not device.connected and device.calls[-1][0] == "disconnect"
    assert any(operation == "emergency_stop" for operation, _ in device.calls)


def test_hardware_capability_rejection_occurs_on_owner_before_connect():
    plant, trace = FakePlant(), MemoryTrace()
    sensor, device = FakeForceSensor(plant), FakeGripper(plant)

    def reject():
        raise RuntimeError("bounded hardware feedback unavailable")

    device.preflight_control = reject
    service = OwnerService(make_runtime(sensor, device, trace))
    with pytest.raises(RuntimeError, match="bounded hardware feedback unavailable"):
        service.start()
    assert device.calls == [] and not sensor.is_connected


def test_trace_rejection_is_immediately_visible_as_fault_not_bindable_success(rig):
    activate(rig)
    rig[3].submit = lambda row: False
    row = tick(rig)
    assert row["phase"] == "FAULT" and row["safety_override"] and not row["label_valid"]
    assert row["failure_reason"] == "control_trace_incomplete"
    assert rig[0].latest_row["phase"] == "FAULT"


def test_fault_origin_survives_later_masked_ticks_and_is_detached(rig):
    runtime, device, *_ = rig
    service = OwnerService(runtime)
    device.snapshot_error = "TimeoutError: first feedback deadline expired"
    first = tick(rig)
    service._publish(first)
    later = tick(rig)
    service._publish(later)
    assert later["phase"] == "FAULT" and later.get("joint_error") is None
    assert service.snapshot().row.get("joint_error") is None
    origin = service.fault_origin
    assert origin.tick_id == first["tick_id"]
    assert origin.row["joint_error"] == "TimeoutError: first feedback deadline expired"
    origin.row["joint_error"] = "caller mutation"
    origin.row["joint_snapshot"]["joints"][0]["error"] = "caller mutation"
    assert service.fault_origin.row["joint_error"] == "TimeoutError: first feedback deadline expired"
    assert (
        service.fault_origin.row["joint_snapshot"]["joints"][0]["error"]
        == "TimeoutError: first feedback deadline expired"
    )
    with pytest.raises(RuntimeError, match="invalid_joint_feedback") as caught:
        service.wait_for_phase("DISABLED")
    assert caught.value.__notes__ == ["joint_error: TimeoutError: first feedback deadline expired"]


def test_reset_clears_fault_origin_and_does_not_reuse_previous_feedback_error(rig):
    runtime, device, *_ = rig
    service = OwnerService(runtime)
    device.snapshot_error = "TimeoutError: old feedback failure"
    service._publish(tick(rig))
    first_epoch = service.fault_origin.epoch
    device.snapshot_error = None
    assert runtime.post("reset", version=0, epoch=runtime.epoch)
    reset = tick(rig)
    service._publish(reset)
    assert reset["phase"] == "DISABLED" and service.fault_origin is None
    force_failure = tick(rig, force=100)
    service._publish(force_failure)
    assert force_failure["phase"] == "FAULT"
    assert service.fault_origin.epoch != first_epoch
    assert service.fault_origin.row.get("joint_error") is None
    with pytest.raises(RuntimeError, match="Owner fault") as caught:
        service.wait_for_phase("DISABLED")
    assert not getattr(caught.value, "__notes__", [])


def test_stop_timeout_retains_worker_resources_until_io_really_exits():
    plant, trace = FakePlant(), MemoryTrace()
    sensor, device = FakeForceSensor(plant), FakeGripper(plant)
    service = OwnerService(make_runtime(sensor, device, trace))
    entered, release = threading.Event(), threading.Event()
    original = device.read_joint_snapshot

    def blocked(names, *, deadline_ns):
        entered.set()
        release.wait(2)
        return original(names, deadline_ns=deadline_ns)

    try:
        sensor.connect()
        sensor.async_read(1000)
        service.start()
        assert not service._thread.daemon
        device.read_joint_snapshot = blocked
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="ownership remains retained"):
            service.stop(timeout_s=0.01)
        assert service.has_resources and not service.is_running
        assert device.connected and not any(op == "disconnect" for op, _ in device.calls)
    finally:
        release.set()
        service.stop()
        sensor.disconnect()
    assert not service.has_resources and not device.connected
    assert device.calls[-1][0] == "disconnect"


def test_dead_owner_with_failed_transport_close_retains_dependencies():
    plant, trace = FakePlant(), MemoryTrace()
    sensor, device = FakeForceSensor(plant), FakeGripper(plant)
    service = OwnerService(make_runtime(sensor, device, trace))

    def fail_close():
        raise OSError("simulated transport close failure")

    sensor.connect()
    try:
        sensor.async_read(1000)
        service.start()
        device.disconnect = fail_close
        with pytest.raises(OSError, match="simulated transport close failure"):
            service.stop()
        assert not service._thread.is_alive() and service.has_resources
        assert not service.runtime.shutdown_evidence["transport_closed"]
    finally:
        # Only the in-memory fixture is reset here; a real transport cannot
        # transfer its owner after a failed close.
        device.connected = False
        sensor.disconnect()


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_names", ("shoulder_pan",)),
        ("joint_names", ("gripper", "gripper")),
        ("arm_intent_required", 1),
        ("handover_tolerance_deg", 0),
    ],
)
def test_owner_config_rejects_ambiguous_contract(field, value):
    with pytest.raises(ValueError):
        replace(RuntimeConfig(), **{field: value}).validate()


def test_stop_signals_startup_guard_without_waiting_for_owner_tick():
    class Runtime:
        def __init__(self):
            self.device = type("Device", (), {"has_resources": False})()
            self.stop_seen = threading.Event()

        def request_stop(self):
            self.stop_seen.set()

    runtime = Runtime()
    service = OwnerService(runtime)
    entered = threading.Event()

    def startup_guard_wait():
        entered.set()
        assert runtime.stop_seen.wait(1)

    service._thread = threading.Thread(target=startup_guard_wait)
    service._thread.start()
    assert entered.wait(1)
    service.stop(timeout_s=1)
    assert runtime.stop_seen.is_set() and not service.has_resources
