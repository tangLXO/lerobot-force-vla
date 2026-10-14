"""Owner, actual submission, causal selection and termination regressions without hardware."""

import threading
from dataclasses import replace

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FakeGripper, FakePlant
from lerobot.force_control.runtime import ForceRuntime, LatestMailbox, OwnerCommand, RuntimeConfig
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from lerobot.force_control.types import ControlPhase
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds=0.02):
        self.now += round(seconds * 1e9)


@pytest.fixture
def bench():
    clock = Clock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    teacher = ForceTeacher(TeacherConfig.fake(), close_sign=1)
    safety = SafetySupervisor(SafetyConfig.fake(), close_sign=1)
    trace = MemoryTrace()
    runtime = ForceRuntime(device, sensor, teacher, safety, trace, clock=clock)
    runtime.open()
    yield runtime, device, sensor, trace, clock
    if runtime._started:
        runtime.close()


def step(bench, left=1.0, right=1.0, *, dt=0.02):
    runtime, _, sensor, _, clock = bench
    clock.advance(dt)
    sensor._publish_sample(
        {"left.normal_force": left, "right.normal_force": right}, clock(), arrival_timestamp_ns=clock()
    )
    return runtime.tick()


def start(bench):
    runtime = bench[0]
    runtime.post("arm", version=0, epoch=runtime.epoch)
    step(bench)
    assert runtime.controller.phase == ControlPhase.ARMED
    runtime.post("start", version=1, epoch=runtime.epoch)
    row = step(bench)
    assert not row["label_valid"]
    return row


def track(bench):
    start(bench)
    for _ in range(4):
        step(bench)
    assert bench[0].controller.phase == ControlPhase.FORCE_TRACK


def test_owner_approach_switch_position_and_repeat_closure(bench):
    runtime = bench[0]
    runtime.safety.config = replace(runtime.safety.config, acceleration_limit_deg_s2=None)
    start(bench)
    runtime.post(
        "approach",
        version=2,
        epoch=runtime.epoch,
        value={
            "gripper_follow_speed_deg_s": 2,
            "contact_force_threshold_N": 0.5,
        },
    )
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "APPROACH" and not row["label_valid"]
    assert row["delta_final_deg"] == pytest.approx(0.04)
    assert row["contact_force_threshold_N"] == 0.5
    row = step(bench, 0.3, 0.3)
    assert row["phase"] == "APPROACH"
    assert row["delta_p_deg"] == pytest.approx(-0.02)
    assert row["delta_final_deg"] == pytest.approx(0.02)
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "APPROACH"
    assert row["delta_p_deg"] == pytest.approx(0.02)
    assert row["delta_final_deg"] == pytest.approx(0.04)  # Rebound cannot exceed approach speed.
    row = step(bench, 0.4, 0.6)
    assert row["phase"] == "FORCE_TRACK" and not row["label_valid"]
    assert row["transition"] == "APPROACH->FORCE_TRACK"
    assert row["delta_p_deg"] == pytest.approx(-0.04)
    assert row["delta_final_deg"] == pytest.approx(0)
    assert row["hardware_ack"] == "unknown"
    runtime.post("position", version=3, epoch=runtime.epoch, value=-1)
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "POSITION_HOLD" and row["hold_target_deg"] == -1
    assert row["q_sent_deg"] > -1  # Approaches at the configured opening rate.
    assert row["delta_final_deg"] == pytest.approx(-0.1)
    runtime.post(
        "approach",
        version=4,
        epoch=runtime.epoch,
        value={
            "gripper_follow_speed_deg_s": 1,
            "contact_force_threshold_N": 0.4,
        },
    )
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "APPROACH" and row["hold_target_deg"] is None


def test_owner_reads_legacy_approach_request_and_reports_canonical_profile(bench):
    runtime = bench[0]
    start(bench)
    runtime.post(
        "approach",
        version=2,
        epoch=runtime.epoch,
        value={"approach_speed_deg_s": 2, "approach_force_threshold_N": 0.5},
    )
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "APPROACH"
    assert row["gripper_follow_speed_deg_s"] == 2
    assert row["contact_force_threshold_N"] == 0.5
    assert "approach_speed_deg_s" not in row["parameter_update"]["teacher"]


def test_owner_rejects_conflicting_approach_alias_without_changing_profile(bench):
    runtime = bench[0]
    start(bench)
    previous = runtime.controller.config
    runtime.post(
        "approach",
        version=2,
        epoch=runtime.epoch,
        value={
            "approach_speed_deg_s": 2,
            "gripper_follow_speed_deg_s": 2,
            "contact_force_threshold_N": 0.5,
        },
    )
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == "FAULT"
    assert runtime.controller.config == previous


def test_owner_contact_alignment_preserves_real_submitted_reference_and_finishes_after_slew_limit(
    bench, monkeypatch
):
    runtime, device, *_ = bench
    runtime.safety.config = replace(runtime.safety.config, acceleration_limit_deg_s2=None)
    start(bench)
    runtime.controller.update_config(replace(runtime.controller.config, ki_deg_per_N_s=0))
    monkeypatch.setattr(device.plant, "advance", lambda: None)  # Fresh feedback, simulated mechanical lag.
    runtime.post(
        "approach",
        version=2,
        epoch=runtime.epoch,
        value={"gripper_follow_speed_deg_s": 5, "contact_force_threshold_N": 0.5},
    )
    for _ in range(7):
        previous = step(bench, 0.1, 0.1)
    row = step(bench, 0.6, 0.6)
    assert row["phase"] == "FORCE_TRACK"
    assert row["q_prev_sent_deg"] == previous["q_sent_deg"]
    assert row["q_measured_deg"] == 0
    assert row["delta_contact_deg"] == pytest.approx(-0.7)
    assert row["delta_p_deg"] == pytest.approx(-0.05)
    assert row["delta_final_deg"] == pytest.approx(-0.1)
    assert row["contact_alignment_pending"] and not row["label_valid"]
    for _ in range(6):
        previous, row = row, step(bench, 0.6, 0.6)
        assert row["q_prev_sent_deg"] == previous["q_sent_deg"]
        assert row["delta_final_deg"] == pytest.approx(-0.1)
        assert not row["label_valid"]
    assert not row["contact_alignment_pending"]
    assert row["q_sent_deg"] == pytest.approx(row["q_measured_deg"])
    row = step(bench, 0.6, 0.6)
    assert row["delta_contact_deg"] == 0 and row["label_valid"]


@pytest.mark.parametrize(
    "mode,target,phase", [("position", -1, "POSITION_HOLD"), ("approach", None, "APPROACH")]
)
def test_owner_starts_requested_motion_without_intermediate_force_track(bench, mode, target, phase):
    runtime = bench[0]
    runtime.controller.update_config(
        replace(
            runtime.controller.config,
            gripper_follow_speed_deg_s=2,
            contact_force_threshold_N=0.5,
        )
    )
    runtime.post("arm", version=0, epoch=runtime.epoch)
    step(bench)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": mode, "position_deg": target})
    row = step(bench, 0.1, 0.1)
    assert row["phase"] == phase
    assert row["delta_i_deg"] == row["delta_p_deg"] == 0


def test_invalid_start_position_is_rejected_before_hardware_start(bench):
    runtime, device, *_ = bench
    started = []
    device.start_gripper = lambda *args: started.append(True)
    runtime.post("arm", version=0, epoch=runtime.epoch)
    step(bench)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": "position", "position_deg": 11})
    row = step(bench)
    assert row["phase"] == "FAULT" and not started
    assert row["q_sent_deg"] is None


@pytest.mark.parametrize("starting", [False, True])
def test_owner_carries_position_speed_at_start_and_during_active_control(bench, starting):
    runtime = bench[0]
    runtime.safety.config = replace(runtime.safety.config, acceleration_limit_deg_s2=None)
    request = {"position_deg": -1, "position_speed_deg_s": 0.5}
    if starting:
        runtime.post("arm", version=0, epoch=runtime.epoch)
        step(bench)
        runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": "position", **request})
        step(bench)
    else:
        start(bench)
        runtime.post("position", version=2, epoch=runtime.epoch, value=request)
        step(bench)
    row = step(bench, dt=0.037)
    assert row["phase"] == "POSITION_HOLD" and row["hold_target_deg"] == -1
    assert row["hold_speed_deg_s"] == 0.5
    assert row["delta_final_deg"] == pytest.approx(-0.5 * 0.037)
    assert row["delta_i_deg"] == row["delta_p_deg"] == 0
    assert row["hardware_ack"] == "unknown"
    runtime.post("position", version=3, epoch=runtime.epoch, value={**request, "position_speed_deg_s": 1})
    row = step(bench)
    assert row["hold_speed_deg_s"] == 1 and row["delta_final_deg"] == pytest.approx(-0.02)


def test_invalid_position_start_speed_never_enables_device(bench):
    runtime, device, *_ = bench
    started = []
    device.start_gripper = lambda *args: started.append(True)
    runtime.post("arm", version=0, epoch=runtime.epoch)
    step(bench)
    runtime.post(
        "start",
        version=1,
        epoch=runtime.epoch,
        value={"mode": "position", "position_deg": -1, "position_speed_deg_s": -1},
    )
    row = step(bench)
    assert row["phase"] == "FAULT" and not started and row["q_sent_deg"] is None


@pytest.mark.parametrize("speed", [None, 0, -1, float("nan"), float("inf"), True])
def test_invalid_position_speed_bundle_is_rejected_without_motion(bench, speed):
    start(bench)
    runtime = bench[0]
    runtime.post(
        "position",
        version=2,
        epoch=runtime.epoch,
        value={"position_deg": -1, "position_speed_deg_s": speed},
    )
    row = step(bench)
    assert row["phase"] == "FAULT" and row["q_sent_deg"] is None


@pytest.mark.parametrize(
    "kind,value",
    [("position", 11), ("approach", {"gripper_follow_speed_deg_s": 2, "contact_force_threshold_N": 5})],
)
def test_invalid_motion_event_never_submits_new_target(bench, kind, value):
    start(bench)
    runtime = bench[0]
    runtime.post(kind, version=2, epoch=runtime.epoch, value=value)
    row = step(bench)
    assert row["phase"] == "FAULT" and row["q_sent_deg"] is None


def test_approach_checks_raw_overforce_before_mode_switch(bench):
    runtime = bench[0]
    start(bench)
    runtime.post(
        "approach",
        version=2,
        epoch=runtime.epoch,
        value={
            "gripper_follow_speed_deg_s": 2,
            "contact_force_threshold_N": 0.5,
        },
    )
    row = step(bench, 5, 0)
    assert row["phase"] == "FAULT" and row["q_sent_deg"] is None


def test_ip_runtime_records_raw_measurement_increments(bench):
    runtime, _, _, trace, _ = bench
    runtime.controller = ForceTeacher(TeacherConfig.fake(kp_deg_per_N=0.1, ki_deg_per_N_s=0.4), 1)
    track(bench)
    row = step(bench, left=1.2, right=1.2)
    assert row["delta_p_deg"] == pytest.approx(-0.1 * 0.2)
    assert row["delta_i_deg"] == pytest.approx(0.4 * 0.02 * -0.2)
    assert trace.rows[-1]["delta_p_deg"] == row["delta_p_deg"]


@pytest.mark.parametrize(
    "budget_ms,expected_phase", [(5, ControlPhase.FAULT), (15, ControlPhase.FORCE_TRACK)]
)
def test_host_delay_obeys_explicit_query_budget_without_disabling_fault_policy(
    bench, monkeypatch, budget_ms, expected_phase
):
    runtime, device, _, _, clock = bench
    runtime.config = replace(runtime.config, feedback_budget_ms=budget_ms)
    track(bench)
    original = device.read_joint_snapshot
    previous = dict(runtime.last_submitted)

    def delayed(names, *, deadline_ns):
        clock.advance(0.011)
        return original(names, deadline_ns=deadline_ns)

    monkeypatch.setattr(device, "read_joint_snapshot", delayed)
    row = step(bench, left=0.7, right=0.7)
    assert runtime.controller.phase == expected_phase
    if expected_phase == ControlPhase.FAULT:
        assert runtime.last_submitted == previous and not row["label_valid"]
    else:
        assert row["send_status"] == "submitted" and row["label_valid"]
        assert row["joint_snapshot"]["completed_ns"] < row["joint_snapshot"]["deadline_ns"]


def test_explicit_arm_does_not_close_and_first_sample_masked(bench):
    row = start(bench)
    assert len([call for call in bench[1].calls if call[0] == "send_gripper"]) == 1
    assert row["q_sent_deg"] == row["q_measured_deg"]
    assert row["prev_successful_tick_id"] is None


def test_labels_use_final_driver_clip_actual_dt_and_success_reference(bench):
    track(bench)
    runtime, device, *_ = bench
    prev = runtime.last_sent
    device.driver_clip = (prev, prev + 0.01)
    row = step(bench, 0.5, 0.5, dt=0.03)
    assert row["label_valid"]
    assert row["q_sent_deg"] == pytest.approx(prev + 0.01)
    assert row["u_label_deg_s"] == pytest.approx(0.01 / 0.03)
    assert "driver_clip" in row["clip_flags"]
    assert row["hardware_ack"] == "unknown"
    assert row["dt_control_s"] == pytest.approx(0.03)


def test_unknown_send_keeps_last_sent_and_invalidates_epoch(bench):
    track(bench)
    runtime, device, *_ = bench
    before, epoch = runtime.last_sent, runtime.epoch
    device.send_failure = True
    row = step(bench, 0.5, 0.5)
    assert runtime.last_sent == before
    assert runtime.epoch == epoch + 1
    assert row["q_sent_deg"] is None and not row["label_valid"]
    assert row["send_status"] == "unknown"
    assert not runtime.post("start", version=999, epoch=epoch)
    calls = len(device.calls)
    step(bench)
    assert all(call[0] != "send_gripper" for call in device.calls[calls:])


def test_fault_reset_enters_disabled_and_requires_fresh_arm(bench):
    track(bench)
    runtime, device, *_ = bench
    device.send_failure = True
    step(bench)
    device.send_failure = False
    runtime.post("reset", version=0, epoch=runtime.epoch)
    row = step(bench)
    assert runtime.controller.phase == ControlPhase.DISABLED
    assert row["send_status"] == "none"
    assert runtime.last_sent is None


def test_reset_after_operator_pause_is_a_masked_boundary(bench):
    start(bench)
    runtime, *_ = bench
    runtime.request_stop()
    step(bench)
    runtime.post("reset", version=0, epoch=runtime.epoch)
    row = step(bench, dt=0.15)
    assert runtime.controller.phase == ControlPhase.DISABLED
    assert row["event"] == "reset" and row["mode_transition"]
    assert row["q_prev_sent_deg"] is None and not row["label_valid"]
    assert step(bench)["event"] is None


def test_adapter_exception_is_an_unknown_attempt_not_no_command(bench, monkeypatch):
    track(bench)
    runtime, device, *_ = bench
    old_sent = runtime.last_sent

    def ambiguous(*args, **kwargs):
        raise OSError("transport failed after possible submission")

    monkeypatch.setattr(device, "send_gripper_position", ambiguous)
    row = step(bench, 0.5, 0.5)
    assert row["send_status"] == "unknown"
    assert row["send_start_ns"] is not None and row["send_end_ns"] is not None
    assert runtime.last_sent == old_sent and not row["label_valid"]


def test_no_catchup_after_long_pause(bench):
    start(bench)
    runtime, device, *_ = bench
    before = len([x for x in device.calls if x[0] == "send_gripper"])
    row = step(bench, dt=0.8)
    assert row["dt_control_s"] == pytest.approx(0.8)
    assert row["deadline_miss"] and runtime.controller.phase == ControlPhase.FAULT
    assert len([x for x in device.calls if x[0] == "send_gripper"]) == before


def test_frozen_feedback_not_relabelled_new(bench):
    start(bench)
    runtime, device, *_ = bench
    old_position, stamp = device.received_position_deg, device.received_ns
    device.freeze_feedback = True
    device.plant.target_deg = 5
    row = step(bench)
    assert row["q_measured_deg"] == old_position
    assert row["joint_receive_timestamp_ns"] == stamp
    assert runtime.controller.phase == ControlPhase.FAULT
    assert "not_updated" in row["failure_reason"]


def test_late_force_is_not_visible_to_earlier_anchor(bench):
    runtime, _, sensor, _, clock = bench
    clock.advance()
    prior = sensor._publish_sample(
        {"left.normal_force": 1.0, "right.normal_force": 1.0},
        clock() - 1000,
        arrival_timestamp_ns=clock() - 1000,
    )
    sensor._publish_sample(
        {"left.normal_force": 4.0, "right.normal_force": 4.0},
        clock() - 500,
        arrival_timestamp_ns=clock() + 1000,
    )
    row = runtime.tick()
    assert row["force_sequence"] == prior.sequence
    assert row["force_mean_N"] == 1


@pytest.mark.parametrize("fault", ["trace", "overforce", "invalid_feedback", "stop"])
def test_fault_disposition_prevents_same_tick_normal_send(bench, fault):
    track(bench)
    runtime, device, _, trace, _ = bench
    if fault == "trace":
        trace.failed = True
    elif fault == "invalid_feedback":
        device.snapshot_error = "injected_missing"
    elif fault == "stop":
        runtime.request_stop()
    before = len(device.calls)
    row = step(bench, left=6 if fault == "overforce" else 1)
    operations = [x[0] for x in device.calls[before:]]
    assert "send_gripper" not in operations
    assert "emergency_stop" in operations
    assert row["safety_override"] and not row["label_valid"]


def test_competing_thread_io_is_rejected(bench):
    runtime, device, *_ = bench
    failures = []

    def compete():
        for function in (runtime.tick, lambda: device.read_joint_snapshot(("gripper",), deadline_ns=1)):
            try:
                function()
            except RuntimeError as exc:
                failures.append(str(exc))

    thread = threading.Thread(target=compete)
    thread.start()
    thread.join()
    assert len(failures) == 2
    step(bench)
    assert {thread_id for _, thread_id in device.calls} == {threading.get_ident()}


def test_external_camera_or_leader_block_does_not_enter_owner(bench):
    entered, leave = threading.Event(), threading.Event()

    def slow_external_io():
        entered.set()
        leave.wait(2)

    thread = threading.Thread(target=slow_external_io)
    thread.start()
    assert entered.wait(1)
    try:
        for _ in range(8):
            step(bench)
        assert bench[0].tick_id == 8
        assert not leave.is_set()
    finally:
        leave.set()
        thread.join()


def test_mailbox_latest_wins_ttl_and_fault_epoch(bench):
    runtime, *_ = bench
    runtime.mailbox.put(OwnerCommand("arm", 1, 0, 1, 1))
    row = step(bench)
    assert "ttl" in row["failure_reason"]
    assert runtime.epoch == 1
    assert not runtime.mailbox.put(OwnerCommand("start", 99, 0, 100, 100))
    mailbox = LatestMailbox()
    assert mailbox.put(OwnerCommand("target", 1, 0, 100, 100, 1.0))
    assert mailbox.put(OwnerCommand("target", 2, 0, 100, 100, 2.0))
    assert mailbox.take().value == 2.0
    assert mailbox.take() is None


def test_release_mode_cannot_be_replaced_by_later_target():
    mailbox = LatestMailbox()
    mailbox.put(OwnerCommand("release", 1, 0, 100, 100))
    mailbox.put(OwnerCommand("target", 2, 0, 100, 100, 2.0))
    assert mailbox.take().kind == "release"


@pytest.mark.parametrize("check_duration_s", [0.021, 0.040])
def test_expired_decision_at_last_send_check_does_not_send(bench, monkeypatch, check_duration_s):
    track(bench)
    runtime, device, _, _, clock = bench
    runtime.safety.config = replace(runtime.safety.config, decision_max_age_ms=20)
    original = runtime.safety.check_inputs
    count = 0

    def expensive_check(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 2:
            clock.advance(check_duration_s)
        return result

    monkeypatch.setattr(runtime.safety, "check_inputs", expensive_check)
    before = len(device.calls)
    step(bench)
    assert not any(op == "send_gripper" for op, _ in device.calls[before:])


def test_adapter_cannot_extend_requested_deadline(bench, monkeypatch):
    runtime, device, _, _, clock = bench
    original = device.read_joint_snapshot

    def late(names, *, deadline_ns):
        snapshot = original(names, deadline_ns=deadline_ns)
        clock.advance(0.006)
        return replace(snapshot, completed_ns=clock(), deadline_ns=clock() + 100_000_000)

    monkeypatch.setattr(device, "read_joint_snapshot", late)
    row = step(bench)
    assert "joint_feedback_deadline" in row["failure_reason"]


def test_successful_submission_evidence_survives_commit_exception(bench, monkeypatch):
    track(bench)
    runtime, *_ = bench

    def broken(*args, **kwargs):
        raise RuntimeError("controller_commit_failure")

    monkeypatch.setattr(runtime.controller, "commit", broken)
    row = step(bench, 0.5, 0.5)
    assert row["send_status"] == "submitted"
    assert runtime.last_sent == row["q_sent_deg"]
    assert runtime.controller.phase == ControlPhase.FAULT
    assert not row["label_valid"]


def test_hardware_template_and_dryrun_fail_before_opening(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No transport may be opened")

    monkeypatch.setattr(FakeGripper, "connect", forbidden)
    with pytest.raises(ValueError, match="Hardware motion"):
        ForceTeacherRunConfig(backend="b601", dry_run=False, observe_only=True).validate()
    with pytest.raises(ValueError, match="dry_run requires"):
        ForceTeacherRunConfig(backend="b601").validate()


def test_teardown_interrupted_fault_still_closes_transport(bench, monkeypatch):
    track(bench)
    runtime, *_ = bench

    def interrupted(reason):
        raise KeyboardInterrupt("simulated fault interruption")

    monkeypatch.setattr(runtime.device, "emergency_stop", interrupted)
    with pytest.raises(KeyboardInterrupt, match="simulated fault interruption"):
        runtime.close()
    assert runtime.shutdown_evidence["transport_closed"]
    assert not runtime.device.has_resources


def test_units_rejected_before_runtime_connect(bench):
    runtime, *_ = bench
    from lerobot.sensors import SensorFeature

    class WrongUnitSensor(ManualSensor):
        @property
        def features(self):
            return {name: SensorFeature("float32", "kg") for name in super().features}

    with pytest.raises(ValueError, match="in N"):
        ForceRuntime(
            runtime.device,
            WrongUnitSensor(runtime.device.plant),
            runtime.controller,
            runtime.safety,
            runtime.trace,
            RuntimeConfig(),
        )


def test_freeze_small_step_approaches_one_goal_then_resume_has_no_p_kick(bench):
    runtime = bench[0]
    track(bench)
    runtime.post("freeze", version=2, epoch=runtime.epoch)
    frozen = step(bench, 3, 3)
    goal = frozen["hold_target_deg"]
    assert frozen["phase"] == "POSITION_HOLD"
    assert frozen["q_sent_deg"] == goal
    runtime.safety.config = replace(
        runtime.safety.config, max_close_speed_deg_s=0.25, acceleration_limit_deg_s2=None
    )
    runtime.post("step", version=3, epoch=runtime.epoch, value=0.1)
    moved = step(bench, 3, 3)
    assert moved["hold_target_deg"] == pytest.approx(goal + 0.1)
    assert moved["q_sent_deg"] == pytest.approx(goal + 0.005)
    for _ in range(22):
        moved = step(bench, 3, 3)
    assert moved["hold_target_deg"] == pytest.approx(goal + 0.1)
    assert moved["q_sent_deg"] == pytest.approx(goal + 0.1)
    assert moved["delta_i_deg"] == moved["delta_p_deg"] == 0
    runtime.post("resume", version=4, epoch=runtime.epoch)
    resumed = step(bench, 2, 2)
    assert resumed["phase"] == "FORCE_TRACK" and resumed["hold_target_deg"] is None
    assert resumed["delta_p_deg"] == 0
    assert resumed["delta_i_deg"] == pytest.approx(-4 * 0.02)
    assert resumed["prev_successful_tick_id"] is None and not resumed["label_valid"]


def test_manual_step_requires_submitted_and_measured_hold_to_settle(bench):
    runtime = bench[0]
    track(bench)
    runtime.post("freeze", version=2, epoch=runtime.epoch)
    step(bench)
    runtime.last_sent = runtime.controller.hold_target_deg - 0.05
    runtime.post("step", version=3, epoch=runtime.epoch, value=0.1)
    row = step(bench)
    assert row["phase"] == "FAULT"
    assert "manual_step_requires_settled_hold" in row["failure_reason"]
    assert row["send_status"] == "none"


def test_live_parameter_bundle_applies_atomically_with_receipt_and_direct_target(bench):
    runtime = bench[0]
    track(bench)
    teacher = replace(
        runtime.controller.config,
        force_target_N=2,
        kp_deg_per_N=0.2,
        ki_deg_per_N_s=3,
        integral_deadband_N=0.05,
    )
    safety = replace(
        runtime.safety.config, max_close_speed_deg_s=4, max_open_speed_deg_s=3, acceleration_limit_deg_s2=None
    )
    runtime.post("parameters", version=2, epoch=runtime.epoch, value={"teacher": teacher, "safety": safety})
    row = step(bench)
    assert runtime.controller.config == teacher and runtime.safety.config == safety
    assert row["force_target_requested_N"] == row["force_target_effective_N"] == 2
    assert row["delta_i_deg"] == pytest.approx(3 * 0.02)
    assert row["delta_p_deg"] == 0
    assert row["parameter_update"]["teacher"]["kp_deg_per_N"] == 0.2
    assert row["parameter_update"]["safety"]["acceleration_limit_deg_s2"] is None


@pytest.mark.parametrize("invalid", ["threshold", "gain", "missing"])
def test_rejected_parameter_bundle_never_partially_updates(bench, invalid):
    runtime = bench[0]
    track(bench)
    old_teacher, old_safety = runtime.controller.config, runtime.safety.config
    teacher = replace(old_teacher, force_target_N=2)
    safety = old_safety
    if invalid == "threshold":
        safety = replace(safety, hard_force_limit_left_N=6)
    elif invalid == "gain":
        teacher = replace(teacher, ki_deg_per_N_s=-1)
    bundle = {"teacher": teacher, "safety": safety}
    if invalid == "missing":
        bundle.pop("safety")
    runtime.post("parameters", version=2, epoch=runtime.epoch, value=bundle)
    row = step(bench)
    assert row["phase"] == "FAULT" and row["parameter_update"] is None
    assert runtime.controller.config == old_teacher and runtime.safety.config == old_safety
    assert row["send_status"] == "none"
