"""Read-only comparable communication timing/contact report from real or synthetic logs."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def distribution(values):
    """Nearest-rank P99; missing samples remain missing, never zero-filled."""
    ordered = sorted(values)
    return (
        None
        if not ordered
        else {
            "count": len(ordered),
            "mean": statistics.fmean(ordered),
            "p99": ordered[math.ceil(0.99 * len(ordered)) - 1],
            "max": ordered[-1],
        }
    )


def summarize(path, window_seconds=60.0):
    """Use FOLLOW submissions only for timing, and report contact separately."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    follow = [
        row
        for row in rows
        if row.get("phase") == "FOLLOW"
        and row.get("send_status") == "submitted"
        and not row.get("communication_only")
    ]
    raw_follow = follow
    window_start = follow[0]["tick_start_ns"] if follow else None
    window_end = window_start + round(window_seconds * 1e9) if follow else None
    follow = [row for row in follow if row["tick_start_ns"] < window_end]
    starts = {row["send_start_ns"] for row in follow}
    groups = {}
    for row in rows:
        io = row.get("device_io_evidence") or {}
        for key in ("command_feedback_group", "consumed_command_feedback_group"):
            group = io.get(key)
            if group and group["send_start_ns"] in starts:
                groups[group["send_start_ns"]] = group
    modes = sorted(
        {row.get("device_io_evidence", {}).get("command_feedback_mode", "unrecorded") for row in follow}
    )
    metrics = defaultdict(list)
    for row in follow:
        metrics["send_function_ms"].append((row["send_end_ns"] - row["send_start_ns"]) / 1e6)
        metrics["in_tick_work_ms"].append(row["loop_elapsed_s"] * 1000)
        metrics["control_interval_ms"].append(row["dt_control_s"] * 1000)
        if "intertick_receive_work_ns" in row:
            metrics["intertick_receive_work_ms"].append(row["intertick_receive_work_ns"] / 1e6)
        joints = (row.get("joint_snapshot") or {}).get("joints", [])
        if joints:
            metrics["oldest_joint_feedback_age_ms"].append(
                max(row["decision_anchor_ns"] - j["receive_timestamp_ns"] for j in joints) / 1e6
            )
    for group in groups.values():
        start = group["send_start_ns"]
        end = group.get("completed_ns")
        if end is not None:
            metrics["whole_group_feedback_ms"].append((end - start) / 1e6)
        commands = group.get("commands", {})
        last_write = group.get("last_write_completed_ns")
        if len(commands) == 7 and all(c.get("submission") == "complete" for c in commands.values()):
            last_write = max(c["write_completed_ns"] for c in commands.values())
        if last_write is not None:
            metrics["seven_axis_submit_window_ms"].append((last_write - start) / 1e6)
        metrics["actual_axis_write_interval_us"].extend(
            value / 1000 for value in group.get("actual_write_intervals_ns", [])
        )
    contact = [
        r
        for r in rows
        if r.get("phase") == "FORCE_TRACK"
        and r.get("force_valid")
        and r.get("force_target_effective_N") is not None
    ]
    errors = [r["force_mean_N"] - r["force_target_effective_N"] for r in contact]
    return {
        "source": str(path.resolve()),
        "modes": modes,
        "window_seconds": window_seconds,
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "window_complete": bool(raw_follow and raw_follow[-1]["tick_start_ns"] >= window_end),
        "raw_follow_submissions": len(raw_follow),
        "raw_follow_span_s": (raw_follow[-1]["tick_start_ns"] - raw_follow[0]["tick_start_ns"]) / 1e9
        if len(raw_follow) > 1
        else 0,
        "follow_submissions": len(follow),
        "follow_span_s": (follow[-1]["tick_start_ns"] - follow[0]["tick_start_ns"]) / 1e9
        if len(follow) > 1
        else 0,
        "timing": {name: distribution(values) for name, values in metrics.items()},
        "feedback_groups_missing_or_incomplete": len(starts)
        - sum(g.get("completed_ns") is not None for g in groups.values()),
        "faults": [r.get("failure_reason") for r in rows if r.get("phase") == "FAULT"],
        "contact": {
            "samples": len(contact),
            "absolute_target_error_N": distribution([abs(e) for e in errors]),
            "maximum_overshoot_N": max([0, *errors]),
            "transitions": dict(Counter(r["hybrid_transition"] for r in rows if r.get("hybrid_transition"))),
        },
        "stop_results": [
            r["fault_disposition"] for r in rows if r.get("fault_disposition") and r.get("communication_only")
        ],
    }


def main():
    """Each --run is one segment; do not pool distinct trials or infer hardware qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", type=Path, required=True, help="One control.jsonl per segment"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    args = parser.parse_args()
    reports = [summarize(path, args.window_seconds) for path in args.run]
    counts = Counter(mode for report in reports for mode in report["modes"])
    result = {
        "runs": reports,
        "runs_by_mode": dict(counts),
        "interpretation": "Host submission and RX timings are separate; RX has no command ACK/token. Software/synthetic results do not qualify hardware. Compare matching 60-second FOLLOW segments, recording/save/resume and operator contact evidence separately.",
        "default_mode": "sequential",
        "hardware_qualified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
