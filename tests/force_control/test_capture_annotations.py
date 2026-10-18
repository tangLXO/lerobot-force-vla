"""Causal capture labels and complete fake grasp histories, without hardware I/O."""

import json

import pytest

from lerobot.force_control.capture_annotations import annotate_capture
from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.fake_hardware import FakeGripper, FakePlant
from lerobot.force_control.runtime import ArmIntent, ForceRuntime
from lerobot.force_control.safety import SafetySupervisor
from lerobot.force_control.teacher import ForceTeacher
from tests.force_control.hardware_helpers import ManualSensor, MemoryTrace
from tests.force_control.trace_helpers import control_row


def annotated_row(**changes):
    row = control_row(
        1,
        phase_at_tick_start="FORCE_TRACK",
        phase_before="FORCE_TRACK",
        force_valid=True,
        force_max_age_ms=100.0,
        joint_feedback_max_age_ms=100.0,
        contact_force_threshold_N=0.5,
        delta_contact_deg=0.0,
        contact_alignment_pending=False,
    )
    row.update(changes)
    return row


def test_annotation_is_pure_and_keeps_contact_separate_from_control_phase():
    row = annotated_row(phase="FOLLOW", phase_before="FOLLOW", label_valid=False)
    original = row.copy()
    result = annotate_capture(row)
    assert row == original
    assert result["annotation_schema"] == "force_capture_annotations_v1"
    assert result["capture_stage"] == "approach"
    assert result["contact_estimate"] == "contact"
    assert result["contact_estimate_valid"] is True
    assert result["contact_estimate_rule"] == "mean_force_threshold_v1"
    assert result["action_label_valid"] is True
    assert row["label_valid"] is False
    assert row["hardware_ack"] == "unknown"


@pytest.mark.parametrize(
    "changes,stage",
    [
        ({"phase": "DISABLED"}, "idle"),
        ({"phase": "ARMED"}, "idle"),
        ({"phase": "APPROACH", "phase_before": "APPROACH"}, "approach"),
        ({"phase": "FOLLOW", "phase_before": "FOLLOW"}, "approach"),
        ({"phase_before": "APPROACH"}, "contact_transition"),
        ({"phase_before": "FOLLOW"}, "contact_transition"),
        ({"contact_alignment_pending": True}, "contact_transition"),
        ({"delta_contact_deg": -0.1}, "contact_transition"),
        ({"delta_contact_deg": 0.1}, "contact_transition"),
        ({}, "force_control"),
        ({"phase": "FOLLOW_WAIT_CLEAR"}, "release"),
        ({"phase": "RELEASE"}, "release"),
        ({"phase": "POSITION_HOLD"}, "position"),
        ({"phase": "FAULT", "contact_alignment_pending": True}, "fault"),
    ],
)
def test_capture_stages_preserve_control_phase_vocabulary(changes, stage):
    row = annotated_row(**changes)
    result = annotate_capture(row)
    assert result["capture_stage"] == stage
    assert result["phase_after"] == row["phase"]
    assert result["phase_changed"] is (row["phase_at_tick_start"] != row["phase"])


@pytest.mark.parametrize(
    "changes",
    [
        {"force_valid": False},
        {"force_valid": None},
        {"force_sequence": None},
        {"force_left_N": None},
        {"force_right_N": float("nan")},
        {"force_left_N": float("inf")},
        {"force_timestamp_ns": 1_021_000_001},
        {"force_arrival_timestamp_ns": 1_021_000_001},
        {"force_timestamp_ns": 800_000_000},
        {"force_max_age_ms": None},
        {"contact_force_threshold_N": None},
        {"contact_force_threshold_N": float("nan")},
        {"contact_force_threshold_N": 0},
    ],
)
def test_missing_or_noncausal_contact_evidence_is_unknown_not_no_contact(changes):
    result = annotate_capture(annotated_row(**changes))
    assert result["contact_estimate"] == "unknown"
    assert result["contact_estimate_valid"] is False


def test_contact_rule_uses_both_sides_at_the_selected_anchor():
    row = annotated_row(force_left_N=0.2, force_right_N=0.8, force_mean_N=0.5)
    assert annotate_capture(row)["contact_estimate"] == "contact"
    row.update(force_left_N=0.1, force_right_N=0.3, force_mean_N=0.2)
    result = annotate_capture(row)
    assert result["contact_estimate"] == "no_contact"
    assert result["capture_stage"] == "force_control"


@pytest.mark.parametrize(
    "changes",
    [
        {"send_status": "none"},
        {"send_status": "unknown"},
        {"feedback_valid": False},
        {"feedback_received_ns": 800_000_000},
        {"feedback_received_ns": 1_021_000_001},
        {"force_valid": False},
        {"force_timestamp_ns": 800_000_000},
        {"safety_override": True},
        {"deadline_miss": True},
        {"prev_successful_tick_id": None},
        {"prev_successful_tick_id": -1},
        {"failure_reason": "injected_commit_failure"},
        {"u_label_deg_s": 1.0},
        {"u_label_deg_s": None},
        {"q_sent_deg": float("nan")},
        {"q_prev_sent_deg": None},
        {"dt_control_s": 0},
        {"dt_control_s": float("nan")},
    ],
)
def test_action_mask_rejects_incomplete_or_unsafe_evidence(changes):
    assert annotate_capture(annotated_row(**changes))["action_label_valid"] is False


def test_missing_contact_threshold_does_not_discard_valid_action_evidence():
    result = annotate_capture(annotated_row(contact_force_threshold_N=None, label_valid=False))
    assert result["contact_estimate"] == "unknown"
    assert result["action_label_valid"] is True


@pytest.mark.parametrize("field", ["force_timestamp_ns", "feedback_received_ns"])
def test_contact_can_be_fresh_at_decision_while_action_evidence_expires_before_submission(field):
    row = annotated_row(
        label_valid=False,
        send_start_ns=1_023_000_000,
        send_end_ns=1_023_000_100,
    )
    row[field] = row["decision_anchor_ns"] - 99_000_000
    result = annotate_capture(row)
    assert result["contact_estimate"] == "contact"
    assert result["contact_estimate_valid"] is True
    assert result["action_label_valid"] is False


class CaptureClock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now


@pytest.fixture
def capture_bench():
    clock = CaptureClock()
    plant = FakePlant(response_time_s=0)
    sensor = ManualSensor(plant)
    device = FakeGripper(plant, clock=clock)
    controller = ForceTeacher(
        TeacherConfig.fake(
            gripper_follow_speed_deg_s=2,
            contact_force_threshold_N=0.5,
            release_travel_deg=0.12,
            ki_deg_per_N_s=0,
        ),
        close_sign=1,
    )
    runtime = ForceRuntime(
        device,
        sensor,
        controller,
        SafetySupervisor(SafetyConfig.fake(acceleration_limit_deg_s2=None), close_sign=1),
        MemoryTrace(),
        clock=clock,
    )
    runtime.open()
    yield runtime, device, sensor, clock
    runtime.close()


def capture_tick(bench, force=0.1, *, leader=None, publish=True):
    runtime, _, sensor, clock = bench
    clock.now += 20_000_000
    if publish:
        sensor._publish_sample(
            {"left.normal_force": force, "right.normal_force": force},
            clock(),
            arrival_timestamp_ns=clock(),
        )
    if leader is not None:
        metadata = {
            "leader_sequence": clock(),
            "leader_request_started_ns": clock(),
            "raw_action_deg": {"gripper.pos": leader + 10},
        }
        runtime.post_arm_intent(
            ArmIntent(
                (("gripper.pos", leader),),
                clock(),
                runtime.epoch,
                clock(),
                200_000_000,
                json.dumps(metadata),
            )
        )
    return runtime.tick()


def begin_capture(bench, mode):
    runtime = bench[0]
    capture_tick(bench)
    runtime.post("arm", version=0, epoch=runtime.epoch)
    capture_tick(bench)
    runtime.post("start", version=1, epoch=runtime.epoch, value={"mode": mode, "position_deg": None})
    return capture_tick(bench, leader=0.8 if mode == "hybrid" else None)


def test_runtime_retains_precontact_transition_tracking_and_release_without_widening_teacher_mask(
    capture_bench,
):
    runtime = capture_bench[0]
    started = begin_capture(capture_bench, "approach")
    assert started["phase_at_tick_start"] == "ARMED"
    assert started["phase_before"] == "APPROACH"
    assert started["phase_after"] == "APPROACH"
    assert started["phase_changed"] is True
    capture_tick(capture_bench)
    approach = capture_tick(capture_bench)
    assert approach["capture_stage"] == "approach"
    assert approach["contact_estimate"] == "no_contact"
    assert approach["action_label_valid"] and not approach["label_valid"]

    contact = capture_tick(capture_bench, 0.6)
    assert contact["phase_at_tick_start"] == contact["phase_before"] == "APPROACH"
    assert contact["phase_after"] == "FORCE_TRACK"
    assert contact["capture_stage"] == "contact_transition"
    assert contact["contact_estimate"] == "contact"
    assert contact["action_label_valid"] and not contact["label_valid"]
    for _ in range(4):
        tracking = capture_tick(capture_bench, 0.6)
    assert tracking["capture_stage"] == "force_control"
    assert tracking["action_label_valid"] and tracking["label_valid"]

    runtime.post("release", version=2, epoch=runtime.epoch)
    release = capture_tick(capture_bench, 0.6)
    assert release["phase_at_tick_start"] == "FORCE_TRACK"
    assert release["phase_before"] == release["phase_after"] == "RELEASE"
    assert release["capture_stage"] == "release"
    assert not release["label_valid"]
    for _ in range(10):
        if runtime.latest_row["phase"] == "DISABLED":
            break
        capture_tick(capture_bench, 0.0)
    history = runtime.trace.rows
    assert history[-1]["phase_after"] == "DISABLED"
    assert [row["tick_id"] for row in history] == list(range(len(history)))
    assert {row["capture_stage"] for row in history} >= {
        "idle",
        "approach",
        "contact_transition",
        "force_control",
        "release",
    }
    assert any(row["action_label_valid"] for row in history if row["capture_stage"] == "release")
    assert all(row["hardware_ack"] == "unknown" for row in history)


def test_hybrid_release_rearm_and_force_loss_keep_distinct_phase_and_contact_labels(capture_bench):
    runtime = capture_bench[0]
    begin_capture(capture_bench, "hybrid")
    capture_tick(capture_bench, leader=0.8)
    capture_tick(capture_bench, 0.6, leader=0.8)
    for _ in range(3):
        capture_tick(capture_bench, 0.6, leader=0.8)
    lost_contact = capture_tick(capture_bench, 0.1, leader=0.8)
    assert lost_contact["phase_after"] == "FORCE_TRACK"
    assert lost_contact["capture_stage"] == "force_control"
    assert lost_contact["contact_estimate"] == "no_contact"

    released = capture_tick(capture_bench, 0.6, leader=-0.8)
    assert released["phase_after"] == "FOLLOW_WAIT_CLEAR"
    assert released["capture_stage"] == "release"
    assert released["contact_estimate"] == "contact"
    assert released["action_label_valid"] and not released["label_valid"]
    capture_tick(capture_bench, 0.1, leader=-0.9)
    rearmed = capture_tick(capture_bench, 0.1, leader=-0.85)
    assert rearmed["phase_after"] == "FOLLOW"
    assert rearmed["capture_stage"] == "approach"
    assert rearmed["contact_estimate"] == "no_contact"
    assert rearmed["phase_changed"] is True
    assert len(runtime.trace.rows) == runtime.tick_id


def test_runtime_expired_force_is_recorded_as_unknown_with_no_training_action(capture_bench):
    runtime, _, _, clock = capture_bench
    begin_capture(capture_bench, "approach")
    capture_tick(capture_bench)
    clock.now += 200_000_000
    row = capture_tick(capture_bench, publish=False)
    assert row["phase_after"] == "FAULT"
    assert row["capture_stage"] == "fault"
    assert row["contact_estimate"] == "unknown"
    assert not row["contact_estimate_valid"]
    assert not row["action_label_valid"] and not row["label_valid"]
    assert runtime.trace.rows[-1] == row
