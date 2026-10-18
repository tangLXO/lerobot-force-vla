#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Read-only capture annotation checks with fake evidence and no hardware."""

import copy
import json

import pytest

from lerobot.force_control.capture_annotations import ANNOTATION_SCHEMA, annotate_capture
from lerobot.force_control.trace import (
    CAPTURE_ANNOTATION_FIELDS,
    CAPTURE_INPUT_FIELDS,
    BenchTrace,
    validate_trace,
)
from tests.force_control.trace_helpers import TraceSensor, control_row


def capture_row(tick=0, **changes):
    row = control_row(tick)
    row.update(
        phase_at_tick_start=row["phase"],
        phase_before=row["phase"],
        force_valid=True,
        force_max_age_ms=100.0,
        joint_feedback_max_age_ms=100.0,
        contact_force_threshold_N=0.5,
        failure_reason=None,
    )
    row.update(changes)
    row.update(annotate_capture(row))
    return row


def write_capture(tmp_path, rows, *, schema=ANNOTATION_SCHEMA, complete=True):
    metadata = {"backend": "fake", "force_max_age_ms": 100.0, "joint_feedback_max_age_ms": 100.0}
    if schema is not None:
        metadata["capture_annotation_schema"] = schema
    sensor = TraceSensor()
    trace = BenchTrace(tmp_path / "run", sensor, metadata)
    trace.start()
    for tick, row in enumerate(rows):
        sensor.publish(1_000_000_000 + tick * 20_000_000)
        assert trace.submit(row)
    trace.close("COMPLETED", complete=complete)
    return trace.root


def snapshot(root):
    return {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in root.iterdir()}


def test_full_capture_coverage_retains_masked_rows_and_is_read_only(tmp_path):
    rows = [
        capture_row(0, phase="APPROACH", phase_at_tick_start="ARMED", phase_before="APPROACH"),
        capture_row(1, phase_at_tick_start="APPROACH", phase_before="APPROACH", label_valid=False),
        capture_row(2),
        capture_row(
            3, phase_at_tick_start="FORCE_TRACK", phase_before="RELEASE", phase="RELEASE", label_valid=False
        ),
        capture_row(
            4,
            phase_at_tick_start="RELEASE",
            phase_before="RELEASE",
            phase="DISABLED",
            transition="RELEASE->DISABLED",
            send_status="none",
            q_sent_deg=None,
            send_start_ns=None,
            send_end_ns=None,
            u_label_deg_s=None,
            teacher_active=False,
            label_valid=False,
        ),
    ]
    original = copy.deepcopy(rows)
    root = write_capture(tmp_path, rows)
    before = snapshot(root)
    report = validate_trace(root)
    assert report["valid"], report["errors"]
    assert rows == original and snapshot(root) == before
    assert len((root / "control.jsonl").read_text().splitlines()) == len(rows)
    summary = report["capture_summary"]
    assert summary["available"] and summary["file_complete"]
    assert summary["control_tick_count"] == summary["annotated_tick_count"] == 5
    assert summary["capture_stage_counts"] == {
        "approach": 1,
        "contact_transition": 1,
        "force_control": 1,
        "release": 1,
        "idle": 1,
    }
    assert summary["phase_transition_counts"] == {
        "ARMED->APPROACH": 1,
        "APPROACH->FORCE_TRACK": 1,
        "FORCE_TRACK->RELEASE": 1,
        "RELEASE->DISABLED": 1,
    }
    assert summary["phase_transition_count"] == 4
    assert summary["all_stages_observed"] and summary["normal_release_completed"]
    assert summary["normal_release_completed_count"] == 1
    assert summary["force_label_counts"] == {"valid": 1, "invalid": 4, "unavailable": 0}
    assert summary["action_label_counts"] == {"valid": 3, "invalid": 2, "unavailable": 0}
    assert summary["contact_estimate_counts"] == {"contact": 5}
    assert summary["software_evidence_only"]


@pytest.mark.parametrize("field", sorted(CAPTURE_INPUT_FIELDS | CAPTURE_ANNOTATION_FIELDS))
def test_declared_capture_schema_rejects_each_missing_field(tmp_path, field):
    row = capture_row()
    del row[field]
    root = write_capture(tmp_path, [row])
    report = validate_trace(root)
    assert not report["valid"]
    assert any("missing capture annotation fields" in error and field in error for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("annotation_schema", "force_capture_annotations_v2"),
        ("phase_after", "RELEASE"),
        ("phase_changed", True),
        ("capture_stage", "release"),
        ("contact_estimate", "no_contact"),
        ("contact_estimate_valid", False),
        ("contact_estimate_rule", "guessed_bilateral_contact"),
        ("action_label_valid", True),
        ("phase_changed", 0),
        ("contact_estimate_valid", 1),
        ("action_label_valid", 0),
    ],
)
def test_capture_annotations_are_recomputed_not_trusted(tmp_path, field, bad_value):
    row = capture_row()
    row[field] = bad_value
    root = write_capture(tmp_path, [row])
    report = validate_trace(root)
    assert not report["valid"]
    assert any("capture annotation mismatch" in error and field in error for error in report["errors"])


@pytest.mark.parametrize("schema", [None, "unrecognized"])
def test_annotated_rows_require_matching_manifest_declaration(tmp_path, schema):
    root = write_capture(tmp_path, [capture_row()], schema=schema)
    report = validate_trace(root)
    assert not report["valid"]
    assert any("capture annotation schema in metadata" in error for error in report["errors"])


def test_capture_schema_cannot_mix_legacy_rows(tmp_path):
    root = write_capture(tmp_path, [capture_row(), control_row(1)])
    report = validate_trace(root)
    assert not report["valid"]
    assert any("Tick 1: missing capture annotation fields" in error for error in report["errors"])


def test_legacy_trace_remains_valid_and_does_not_gain_annotations(tmp_path):
    root = write_capture(tmp_path, [control_row(), control_row(1)], schema=None)
    before = snapshot(root)
    report = validate_trace(root)
    assert report["valid"], report["errors"]
    assert snapshot(root) == before
    assert not report["capture_summary"]["available"]
    assert report["capture_summary"]["annotated_tick_count"] == 0
    assert report["capture_summary"]["action_label_counts"]["unavailable"] == 2
    assert all(
        "capture_stage" not in json.loads(line) for line in (root / "control.jsonl").read_text().splitlines()
    )


def test_complete_force_only_file_does_not_imply_full_capture_or_release(tmp_path):
    report = validate_trace(write_capture(tmp_path, [capture_row(), capture_row(1)]))
    assert report["valid"], report["errors"]
    summary = report["capture_summary"]
    assert summary["file_complete"]
    assert summary["observed_stages"] == {
        "approach": False,
        "contact_transition": False,
        "force_control": True,
        "release": False,
    }
    assert not summary["all_stages_observed"] and not summary["normal_release_completed"]


@pytest.mark.parametrize("release_phase", ["FOLLOW_WAIT_CLEAR", "RELEASE"])
def test_disable_after_release_stage_is_not_normal_active_release_completion(tmp_path, release_phase):
    rows = [
        capture_row(0, phase=release_phase, phase_before=release_phase, phase_at_tick_start=release_phase),
        capture_row(
            1,
            phase="DISABLED",
            phase_before="DISABLED",
            phase_at_tick_start=release_phase,
            label_valid=False,
            teacher_active=False,
            send_status="none",
            q_sent_deg=None,
            u_label_deg_s=None,
            send_start_ns=None,
            send_end_ns=None,
        ),
    ]
    report = validate_trace(write_capture(tmp_path, rows))
    assert report["valid"], report["errors"]
    assert report["capture_summary"]["observed_stages"]["release"]
    assert not report["capture_summary"]["normal_release_completed"]


def test_force_valid_flag_cannot_disagree_with_selected_raw_evidence(tmp_path):
    row = capture_row(force_valid=False)
    report = validate_trace(write_capture(tmp_path, [row]))
    assert not report["valid"]
    assert any("force_valid disagrees" in error for error in report["errors"])


def test_unconfigured_contact_threshold_keeps_contact_unknown_without_invalidating_trace(tmp_path):
    root = write_capture(tmp_path, [capture_row(contact_force_threshold_N=None)])
    report = validate_trace(root)
    assert report["valid"], report["errors"]
    assert report["capture_summary"]["contact_estimate_counts"] == {"unknown": 1}


def test_fault_and_unavailable_force_remain_in_complete_evidence(tmp_path):
    row = capture_row(
        phase="FAULT",
        phase_before="FORCE_TRACK",
        force_sequence=None,
        force_valid=False,
        force_timestamp_ns=None,
        force_arrival_timestamp_ns=None,
        force_left_N=None,
        force_right_N=None,
        force_mean_N=None,
        send_status="none",
        q_sent_deg=None,
        send_start_ns=None,
        send_end_ns=None,
        safety_override=True,
        failure_reason="force_unavailable",
    )
    root = write_capture(tmp_path, [row])
    report = validate_trace(root)
    assert report["valid"], report["errors"]
    summary = report["capture_summary"]
    assert summary["capture_stage_counts"] == {"fault": 1}
    assert summary["contact_estimate_counts"] == {"unknown": 1}
    assert summary["action_label_counts"]["invalid"] == 1
    assert summary["force_label_counts"]["invalid"] == 1
    assert report["statistics"]["raw_attempt_count"] == 1
    assert len((root / "control.jsonl").read_text().splitlines()) == 1


def test_partial_annotations_without_schema_cannot_masquerade_as_legacy(tmp_path):
    row = capture_row()
    del row["annotation_schema"]
    report = validate_trace(write_capture(tmp_path, [row], schema=None))
    assert not report["valid"]
    assert any("capture annotation schema in metadata" in error for error in report["errors"])


@pytest.mark.parametrize(
    "field", ["force_max_age_ms", "joint_feedback_max_age_ms", "contact_force_threshold_N"]
)
@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_annotation_configuration_is_rejected(tmp_path, field, value):
    report = validate_trace(write_capture(tmp_path, [capture_row(**{field: value})]))
    assert not report["valid"]
    assert any(field in error for error in report["errors"])
