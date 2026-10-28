#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Bench evidence tests: no hardware, SDK, external services or GPU."""

import json
import threading

import pytest

from lerobot.force_control.trace import BenchTrace, TraceError, validate_trace
from tests.force_control.trace_helpers import TraceSensor, control_row


def write_run(tmp_path, rows=None, *, complete=True):
    sensor = TraceSensor()
    trace = BenchTrace(
        tmp_path / "run",
        sensor,
        {
            "backend": "fake",
            "force_max_age_ms": 100.0,
            "joint_feedback_max_age_ms": 100.0,
        },
    )
    trace.start()
    rows = [control_row(0), control_row(1)] if rows is None else rows
    for tick, row in enumerate(rows):
        sensor.publish(1_000_000_000 + tick * 20_000_000)
        assert trace.submit(row)
    manifest = trace.close("COMPLETED", complete=complete)
    return trace, sensor, manifest


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "FOLLOW"},
        {"phase": "FOLLOW_WAIT_CLEAR"},
        {"hybrid_transition": "contact"},
        {"delta_contact_deg": -0.1},
    ],
)
def test_follow_and_contact_transition_cannot_claim_force_labels(tmp_path, changes):
    trace, _, _ = write_run(tmp_path, [control_row(0), control_row(1, **changes)])
    report = validate_trace(trace.root)
    assert not report["valid"]
    assert any("must be masked" in error for error in report["errors"])


def test_roundtrip_native_raw_causal_labels_stats_and_read_only(tmp_path):
    trace, sensor, manifest = write_run(tmp_path)
    assert manifest["complete"] is True
    assert sensor.subscriber_count == 0
    assert not sensor._recorder_leases
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in trace.root.iterdir()}
    report = validate_trace(trace.root)
    after = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in trace.root.iterdir()}
    assert before == after
    assert report["valid"], report["errors"]
    stats = report["statistics"]
    assert stats["measured_control_hz"] == pytest.approx(50)
    assert stats["dt_control_s"]["p99"] == pytest.approx(0.02)
    assert stats["force_track_mae_N"] == pytest.approx(0.1)
    assert stats["force_sample_updates"] == stats["feedback_updates"] == 2
    raw = json.loads((trace.root / "raw_force.jsonl").read_text().splitlines()[0])
    assert raw["native_values"] == {"register_left": 42}
    assert raw["native_payload_base64"] == "AP8="
    assert manifest["sensor_features"]["left.normal_force"]["unit"] == "N"
    assert "sidecar_schema_version" not in manifest


def test_cleanup_finish_marks_complete_only_after_recorder_release(tmp_path):
    trace, sensor, sealed = write_run(tmp_path, complete=False)
    assert sealed["complete"] is False and trace.closed and not trace.has_resources
    assert not sensor._recorder_leases
    assert not validate_trace(trace.root)["valid"]
    final = trace.finish_cleanup(complete=True, errors=[])
    assert final["complete"] is True and final["files"] == sealed["files"]
    assert validate_trace(trace.root)["valid"]


def test_cleanup_error_is_recorded_and_cannot_be_erased_on_retry(tmp_path):
    trace, _, sealed = write_run(tmp_path, complete=False)
    first = trace.finish_cleanup(complete=False, errors=["Sensor disconnect failed"])
    second = trace.finish_cleanup(complete=True, errors=[])
    assert not second["complete"] and second["failure_reason"] == first["failure_reason"]
    assert second["metadata"]["cleanup_errors"] == ["Sensor disconnect failed"]
    assert second["files"] == sealed["files"]
    assert not validate_trace(trace.root)["valid"]


def test_cleanup_sealing_failure_keeps_incomplete_file_and_remains_failed_after_retry(tmp_path, monkeypatch):
    trace, _, _ = write_run(tmp_path, complete=False)
    original = trace._write_manifest

    def reject_seal():
        raise OSError("injected cleanup disk failure")

    monkeypatch.setattr(trace, "_write_manifest", reject_seal)
    with pytest.raises(TraceError, match="cleanup disk failure"):
        trace.finish_cleanup(complete=True, errors=[])
    assert not json.loads((trace.root / "manifest.json").read_text())["complete"]
    monkeypatch.setattr(trace, "_write_manifest", original)
    final = trace.finish_cleanup(complete=True, errors=[])
    assert not final["complete"] and "cleanup disk failure" in final["failure_reason"]


def test_submit_detaches_values_and_only_worker_writes(tmp_path):
    threads = []

    class CheckingTrace(BenchTrace):
        def _write_row(self, handle, row):
            threads.append(threading.get_ident())
            super()._write_row(handle, row)

    sensor = TraceSensor()
    trace = CheckingTrace(tmp_path / "run", sensor, {})
    trace.start()
    sensor.publish()
    row = control_row()
    assert trace.submit(row)
    row["q_sent_deg"] = 999.0
    trace.close("COMPLETED", True)
    assert set(threads) == {trace._thread.ident}
    assert trace._thread.ident != threading.get_ident()
    assert validate_trace(trace.root)["valid"]


class BlockingTrace(BenchTrace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def _write_row(self, handle, row):
        self.entered.set()
        assert self.release.wait(5)
        super()._write_row(handle, row)


@pytest.mark.parametrize("queue_kind", ["control", "raw"])
def test_bounded_overflow_latches_incomplete_without_blocking_owner(tmp_path, queue_kind):
    sensor = TraceSensor()
    trace = BlockingTrace(tmp_path / "run", sensor, {}, capacity=1, raw_capacity=1)
    trace.start()
    assert trace.submit(control_row())
    assert trace.entered.wait(2)
    try:
        if queue_kind == "control":
            assert trace.submit(control_row(1))
            assert not trace.submit(control_row(2))
        else:
            sensor.publish()
            sensor.publish(1_020_000_000)
        assert trace.failed
        with pytest.raises(TraceError, match="overflow"):
            trace.check_health()
        assert not trace.submit(control_row(3))
    finally:
        trace.release.set()
        manifest = trace.close("FAULT", True)
    assert manifest["complete"] is False
    assert "overflow" in manifest["failure_reason"]
    assert not validate_trace(trace.root)["valid"]


def test_writer_failure_is_reported_and_never_marked_complete(tmp_path):
    failed = threading.Event()

    class BrokenTrace(BenchTrace):
        def _write_row(self, handle, row):
            failed.set()
            raise OSError("injected disk full")

    sensor = TraceSensor()
    trace = BrokenTrace(tmp_path / "run", sensor, {})
    trace.start()
    trace.submit(control_row())
    assert failed.wait(2)
    trace._thread.join(2)
    with pytest.raises(TraceError, match="disk full"):
        trace.check_health()
    manifest = trace.close("FAULT", True)
    assert not manifest["complete"]
    assert not sensor._recorder_leases


def test_shutdown_timeout_retains_lease_until_worker_stops(tmp_path):
    sensor = TraceSensor()
    trace = BlockingTrace(tmp_path / "run", sensor, {}, join_timeout_s=0.01)
    trace.start()
    trace.submit(control_row())
    assert trace.entered.wait(2)
    try:
        with pytest.raises(TraceError, match="lease retained"):
            trace.close("DISABLED", True)
        assert trace._thread.is_alive()
        assert sensor._recorder_leases
        with pytest.raises(TraceError, match="recorder lease"):
            trace.finish_cleanup(complete=True, errors=[])
        assert not json.loads((trace.root / "manifest.json").read_text())["complete"]
    finally:
        trace.release.set()
        trace._thread.join(2)
        manifest = trace.close("FAULT", False)
    assert not sensor._recorder_leases
    assert not manifest["complete"]


def test_sensor_reset_attempt_invalidates_recording(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {})
    trace.start()
    with pytest.raises(RuntimeError, match="recorder ownership"):
        sensor._reset_framework_state()
    with pytest.raises(TraceError, match="lease failed"):
        trace.check_health()
    sealed = trace.close("FAULT", True)
    assert not sealed["complete"]
    cleanup = trace.finish_cleanup(complete=True, errors=["Sensor cleanup also failed"])
    assert not cleanup["complete"] and cleanup["failure_reason"] == sealed["failure_reason"]
    assert cleanup["metadata"]["cleanup_errors"] == ["Sensor cleanup also failed"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"force_arrival_timestamp_ns": 2_000_000_000}, "force reference mismatch"),
        ({"force_sequence": 55}, "missing raw force reference"),
        ({"force_left_N": 99.0}, "raw semantic value mismatch"),
        ({"u_label_deg_s": 10.0, "dt_send_s": 0.01}, "dt_control_s"),
        ({"epoch": 1}, "predecessor crosses boundary"),
        ({"segment_id": "next"}, "predecessor crosses boundary"),
        ({"q_prev_sent_deg": 0.0}, "last-sent reference mismatch"),
        ({"safety_override": True}, "must be masked"),
        ({"feedback_valid": False}, "must be masked"),
        ({"deadline_miss": True}, "must be masked"),
        ({"send_status": "unknown"}, "unconfirmed submission"),
        ({"hardware_ack": "confirmed"}, "acknowledgement"),
        ({"send_start_ns": 0}, "submission timing"),
        ({"action_override": True}, "must be masked"),
        ({"mode_transition": True}, "must be masked"),
        ({"feedback_received_ns": 2_000_000_000}, "causal receive"),
        ({"feedback_age_ns": 0}, "causal receive"),
        ({"feedback_sequence": None}, "causal receive"),
        ({"phase": "FAULT"}, "must be masked"),
        ({"phase": "APPROACH"}, "must be masked"),
        ({"mode": "TELEOP"}, "must be masked"),
        ({"per_joint_results": [{"name": "gripper", "status": "unknown"}]}, "software receipt"),
        ({"tick_id": 2}, "tick sequence gap"),
        ({"label_valid": 1}, "must be a boolean"),
        ({"close_sign": True}, "calibrated integer"),
        ({"joint_snapshot": {"joints": []}}, "visible joint snapshot"),
    ],
)
def test_validator_rejects_bad_causality_labels_and_masks(tmp_path, changes, expected):
    trace, _, _ = write_run(tmp_path, [control_row(), control_row(1, **changes)])
    report = validate_trace(trace.root)
    assert not report["valid"]
    assert any(expected in error for error in report["errors"]), report


def test_validator_rejects_late_arrival_even_with_earlier_measurement(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {})
    trace.start()
    sensor.publish(1_000_000_000, 1_002_000_000)
    trace.submit(control_row(force_arrival_timestamp_ns=1_002_000_000))
    trace.close("DISABLED", True)
    assert any("noncausal" in error for error in validate_trace(trace.root)["errors"])


def test_first_command_requires_mask_and_duplicate_ticks_are_rejected(tmp_path):
    trace, _, _ = write_run(tmp_path, [control_row(label_valid=True), control_row(0)])
    errors = validate_trace(trace.root)["errors"]
    assert any("must be masked" in error for error in errors)
    assert any("Nonunique" in error for error in errors)


def test_failed_send_does_not_advance_last_successful_reference(tmp_path):
    rows = [
        control_row(),
        control_row(
            1,
            send_status="unknown",
            q_sent_deg=None,
            label_valid=False,
            u_label_deg_s=None,
            phase="FAULT",
            safety_override=True,
        ),
        control_row(
            2,
            epoch=1,
            segment_id="reset",
            phase="DISABLED",
            event="reset",
            send_status="none",
            prev_successful_tick_id=None,
            q_prev_sent_deg=None,
            q_sent_deg=None,
            label_valid=False,
            u_label_deg_s=None,
            teacher_active=False,
        ),
        control_row(3, epoch=1, segment_id="reset", prev_successful_tick_id=None, label_valid=False),
        control_row(4, epoch=1, segment_id="reset"),
    ]
    trace, _, _ = write_run(tmp_path, rows)
    report = validate_trace(trace.root)
    assert report["valid"], report


def test_file_tampering_and_absent_terminal_manifest_are_rejected_read_only(tmp_path):
    trace, _, _ = write_run(tmp_path)
    path = trace.root / "control.jsonl"
    with path.open("a") as handle:
        handle.write("{}\n")
    assert any("integrity" in error for error in validate_trace(trace.root)["errors"])
    (trace.root / "manifest.json").unlink()
    before = set(trace.root.iterdir())
    assert not validate_trace(trace.root)["valid"]
    assert set(trace.root.iterdir()) == before


def test_interrupted_run_is_not_training_complete_and_cannot_overwrite(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {})
    trace.start()
    sensor.publish()
    trace.submit(control_row())
    trace.close("STOPPED", complete=False)
    assert not validate_trace(trace.root)["valid"]
    assert validate_trace(trace.root, require_complete=False)["valid"]
    with pytest.raises(FileExistsError):
        BenchTrace(trace.root, sensor, {}).start()


def test_invalid_raw_attempts_are_preserved_not_filled_with_zero(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {})
    trace.start()
    sensor.publish(is_valid=False)
    trace.submit(
        control_row(
            force_sequence=None,
            force_timestamp_ns=None,
            force_arrival_timestamp_ns=None,
            force_left_N=None,
            force_right_N=None,
            force_mean_N=None,
            feedback_valid=False,
            q_sent_deg=None,
            send_status="none",
            label_valid=False,
            phase="FAULT",
        )
    )
    trace.close("FAULT", True)
    raw = json.loads((trace.root / "raw_force.jsonl").read_text())
    assert raw["values"] == {}
    assert not raw["is_valid"] and raw["error"] == "injected timeout"
    assert validate_trace(trace.root)["valid"]


def test_actual_control_dt_is_verified_independently_of_send_dt(tmp_path):
    rows = [
        control_row(tick_start_ns=1_000_000_000),
        control_row(1, tick_start_ns=1_030_000_000),
    ]
    trace, _, _ = write_run(tmp_path, rows)
    assert any("actual tick-start interval" in error for error in validate_trace(trace.root)["errors"])


def test_reusing_an_old_segment_does_not_restore_a_predecessor(tmp_path):
    rows = [
        control_row(),
        control_row(1, segment_id="segment-1", prev_successful_tick_id=None, label_valid=False),
        control_row(2, prev_successful_tick_id=0, q_prev_sent_deg=0.1, u_label_deg_s=10),
    ]
    trace, _, _ = write_run(tmp_path, rows)
    assert any("predecessor crosses boundary" in error for error in validate_trace(trace.root)["errors"])


def test_band_and_sampled_recovery_need_explicit_thresholds(tmp_path):
    trace, _, _ = write_run(tmp_path)
    stats = validate_trace(trace.root)["statistics"]
    assert stats["band_N"] is None and stats["force_track_in_band_fraction"] is None
    assert stats["segment_recovery"] == []
    stats = validate_trace(trace.root, band_n=0.2, recovery_dwell_s=0.02)["statistics"]
    assert stats["force_track_in_band_fraction"] == 1.0
    assert stats["force_track_max_overshoot_N"] == pytest.approx(0.1)
    assert stats["measured_feedback_update_hz"] == pytest.approx(50)
    assert stats["segment_recovery"][0]["recovery_time_s"] == 0.0
    assert stats["segment_recovery"][0]["dwell_confirmed_ns"] == 1_021_000_000
    stats = validate_trace(trace.root, band_n=0.01, recovery_dwell_s=0.02)["statistics"]
    assert stats["segment_recovery"][0]["recovery_time_s"] is None
    with pytest.raises(ValueError, match="requires"):
        validate_trace(trace.root, recovery_dwell_s=0.02)
    with pytest.raises(ValueError, match="finite"):
        validate_trace(trace.root, band_n=float("nan"))


def test_causal_but_expired_force_cannot_be_a_training_label(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {"force_max_age_ms": 0.1})
    trace.start()
    for tick in (0, 1):
        sensor.publish(1_000_000_000 + tick * 20_000_000)
        trace.submit(control_row(tick))
    trace.close("COMPLETED", True)
    assert any("stale force_timestamp_ns" in error for error in validate_trace(trace.root)["errors"])


@pytest.mark.parametrize("new_epoch", [0, 1])
def test_normal_commands_after_failed_submission_require_explicit_reset(tmp_path, new_epoch):
    rows = [
        control_row(),
        control_row(1, phase="FAULT", send_status="unknown", q_sent_deg=None, label_valid=False),
        control_row(2, epoch=new_epoch, prev_successful_tick_id=None, label_valid=False),
    ]
    trace, _, _ = write_run(tmp_path, rows)
    assert any("requires explicit disabled reset" in error for error in validate_trace(trace.root)["errors"])


def test_terminal_manifest_cannot_claim_disabled_while_last_tick_tracks(tmp_path):
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, {})
    trace.start()
    sensor.publish()
    trace.submit(control_row())
    trace.close("DISABLED", True)
    assert any("Terminal manifest disagrees" in error for error in validate_trace(trace.root)["errors"])


def test_mask_does_not_hide_successful_submission_from_trace(tmp_path):
    row = control_row(
        send_status="none",
        q_sent_deg=None,
        label_valid=False,
        per_joint_results=[{"name": "gripper", "status": "success", "submitted_deg": 0.1}],
    )
    trace, _, _ = write_run(tmp_path, [row])
    assert any("software receipt" in error for error in validate_trace(trace.root)["errors"])


@pytest.mark.parametrize("failed_axis", [0, 1, 5, 6])
def test_partial_arm_stop_preserves_unknown_frame_and_unattempted_gripper(tmp_path, failed_axis):
    from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES

    results = [
        {
            "name": name,
            "status": "success"
            if index < failed_axis
            else "unknown"
            if index == failed_axis
            else "not_attempted",
            "submitted_deg": 0.1 if index < failed_axis else None,
        }
        for index, name in enumerate(JOINT_NAMES)
    ]
    row = control_row(
        phase="FAULT",
        send_status="unknown",
        q_sent_deg=None,
        label_valid=False,
        failure_reason="user_stop",
        per_joint_results=results,
    )
    trace, _, _ = write_run(tmp_path, [row])
    errors = validate_trace(trace.root)["errors"]
    assert not any("software receipt" in error for error in errors)


def test_partial_action_cannot_claim_complete_submission(tmp_path):
    row = control_row(
        per_joint_results=[
            {"name": "shoulder_pan", "status": "unknown", "submitted_deg": None},
            {"name": "gripper", "status": "success", "submitted_deg": 0.1},
        ]
    )
    trace, _, _ = write_run(tmp_path, [row])
    assert any("software receipt" in error for error in validate_trace(trace.root)["errors"])


def test_validator_refuses_labels_with_missing_age_budget_or_wrong_units(tmp_path):
    trace, _, _ = write_run(tmp_path)
    path = trace.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"].pop("force_max_age_ms")
    manifest["sensor_features"]["left.normal_force"]["unit"] = "kg"
    path.write_text(json.dumps(manifest))
    errors = validate_trace(trace.root)["errors"]
    assert any("recorded positive age budget" in error for error in errors)
    assert any("declare N units" in error for error in errors)
