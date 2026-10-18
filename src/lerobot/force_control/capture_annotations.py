"""Versioned, observational labels for complete FORCE_POS capture timelines.

These annotations never select sensors, change a command, or filter recorded rows.
The capture stage describes the completed tick; only decision-time fields may be
used as inputs to the action that was selected during that tick.
"""

from __future__ import annotations

import math
from typing import Any

ANNOTATION_SCHEMA = "force_capture_annotations_v1"
CONTACT_RULE = "mean_force_threshold_v1"


def _finite(value: Any) -> bool:
    """Accept finite scalar numbers, excluding boolean flags and missing values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fresh(timestamp: Any, anchor: Any, max_age_ms: Any) -> bool:
    """Check evidence against its decision anchor, never the serialization time."""
    return (
        type(timestamp) is int
        and timestamp >= 0
        and type(anchor) is int
        and _finite(max_age_ms)
        and max_age_ms > 0
        and 0 <= anchor - timestamp <= max_age_ms * 1e6
    )


def _force_usable(row: dict[str, Any]) -> bool:
    """Use only the single causal, valid force sample already selected by owner."""
    anchor = row.get("decision_anchor_ns")
    return (
        row.get("force_valid") is True
        and type(row.get("force_sequence")) is int
        and row["force_sequence"] >= 0
        and all(_finite(row.get(key)) for key in ("force_left_N", "force_right_N", "force_mean_N"))
        and _fresh(row.get("force_timestamp_ns"), anchor, row.get("force_max_age_ms"))
        and type(row.get("force_arrival_timestamp_ns")) is int
        and 0 <= row["force_arrival_timestamp_ns"] <= anchor
    )


def _capture_stage(row: dict[str, Any]) -> str:
    """Group controller phases without claiming physical contact or stable hold."""
    phase = row.get("phase")
    if phase == "FAULT":
        return "fault"
    if phase == "FORCE_TRACK":
        if (
            row.get("phase_before") in ("APPROACH", "FOLLOW")
            or row.get("delta_contact_deg", 0) != 0
            or row.get("contact_alignment_pending") is True
        ):
            return "contact_transition"
        return "force_control"
    return {
        "DISABLED": "idle",
        "ARMED": "idle",
        "APPROACH": "approach",
        "FOLLOW": "approach",
        "FOLLOW_WAIT_CLEAR": "release",
        "RELEASE": "release",
        "POSITION_HOLD": "position",
    }.get(phase, "unknown")


def _action_label_valid(row: dict[str, Any], force_usable: bool) -> bool:
    """Allow bounded motion imitation across phases, never unknown submissions.

    This is deliberately separate from the existing force-only ``label_valid``.
    An interruption in successful commands masks the first new increment, whose dt
    would otherwise not describe a continuous command trajectory.
    """
    tick = row.get("tick_id")
    previous = row.get("prev_successful_tick_id")
    if not (
        row.get("send_status") == "submitted"
        and row.get("phase") != "FAULT"
        and row.get("feedback_valid") is True
        and force_usable
        and row.get("safety_override") is False
        and row.get("deadline_miss") is False
        and row.get("failure_reason") is None
        and type(tick) is int
        and type(previous) is int
        and 0 <= previous == tick - 1
        and _fresh(
            row.get("feedback_received_ns"),
            row.get("decision_anchor_ns"),
            row.get("joint_feedback_max_age_ms"),
        )
        and all(
            _finite(row.get(key))
            for key in ("q_prev_sent_deg", "q_sent_deg", "q_measured_deg", "u_label_deg_s", "dt_control_s")
        )
        and row["dt_control_s"] > 0
        and type(row.get("close_sign")) is not bool
        and row.get("close_sign") in (-1, 1)
        and type(row.get("send_start_ns")) is int
        and type(row.get("send_end_ns")) is int
        and row["decision_anchor_ns"] <= row["send_start_ns"] <= row["send_end_ns"]
        and _fresh(row.get("force_timestamp_ns"), row["send_start_ns"], row.get("force_max_age_ms"))
        and _fresh(
            row.get("feedback_received_ns"), row["send_start_ns"], row.get("joint_feedback_max_age_ms")
        )
    ):
        return False
    expected = row["close_sign"] * (row["q_sent_deg"] - row["q_prev_sent_deg"]) / row["dt_control_s"]
    return math.isclose(row["u_label_deg_s"], expected, rel_tol=1e-8, abs_tol=1e-8)


def annotate_capture(row: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic annotations without changing the source control row.

    Contact is an estimate using the configured mean-force threshold, not a claim of
    bilateral contact. No new threshold or temporal filter is imposed on control.
    Missing configuration or unusable force produces UNKNOWN rather than no-contact.
    The force-only training mask is kept unchanged, including on transition rows.
    """
    force_usable = _force_usable(row)
    threshold = row.get("contact_force_threshold_N")
    contact_valid = force_usable and _finite(threshold) and threshold > 0
    contact = "unknown"
    if contact_valid:
        contact = "contact" if row["force_mean_N"] >= threshold else "no_contact"
    return {
        "annotation_schema": ANNOTATION_SCHEMA,
        "phase_after": row.get("phase"),
        "phase_changed": row.get("phase_at_tick_start") != row.get("phase"),
        "capture_stage": _capture_stage(row),
        "contact_estimate": contact,
        "contact_estimate_valid": bool(contact_valid),
        "contact_estimate_rule": CONTACT_RULE,
        "action_label_valid": _action_label_valid(row, force_usable),
    }
