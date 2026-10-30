import json

import pytest

from examples.force_teacher.compare_command_feedback import summarize


def test_fixed_window_keeps_later_feedback_and_reports_faults(tmp_path):
    rows = []
    for seconds in (0, 20, 40, 60, 80):
        start = 1_000_000_000 + seconds * 1_000_000_000
        rows.append(
            {
                "phase": "FOLLOW",
                "send_status": "submitted",
                "tick_start_ns": start,
                "send_start_ns": start + 1_000_000,
                "send_end_ns": start + 3_000_000,
                "loop_elapsed_s": 0.004,
                "dt_control_s": 0.02,
                "device_io_evidence": {"command_feedback_mode": "pipelined"},
            }
        )
    # Completion may be published outside the selected timing window.
    rows.append(
        {
            "phase": "FAULT",
            "failure_reason": "late fault remains visible",
            "device_io_evidence": {
                "consumed_command_feedback_group": {
                    "send_start_ns": rows[2]["send_start_ns"],
                    "completed_ns": rows[2]["send_start_ns"] + 5_000_000,
                }
            },
        }
    )
    path = tmp_path / "control.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = summarize(path)
    assert report["window_complete"]
    assert report["follow_submissions"] == 3
    assert report["raw_follow_submissions"] == 5
    assert report["raw_follow_span_s"] == 80
    assert report["timing"]["whole_group_feedback_ms"]["mean"] == 5
    assert report["feedback_groups_missing_or_incomplete"] == 2
    assert report["faults"] == ["late fault remains visible"]


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_window_rejected(tmp_path, seconds):
    with pytest.raises(ValueError, match="positive and finite"):
        summarize(tmp_path / "absent", seconds)


def test_empty_run_is_not_a_completed_window(tmp_path):
    path = tmp_path / "control.jsonl"
    path.write_text("", encoding="utf-8")
    report = summarize(path)
    assert not report["window_complete"]
    assert report["follow_submissions"] == 0
