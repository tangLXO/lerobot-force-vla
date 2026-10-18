"""Small detached UI views; the full startup evidence remains in trace/result files."""

import copy


def startup_summary(receipt: dict | None) -> dict | None:
    """Copy settings and state without repeatedly copying per-packet startup evidence."""
    if receipt is None:
        return None
    fields = (
        "kind",
        "session_id",
        "complete",
        "error",
        "hardware_ack",
        "production_arm_authorized",
        "watchdog_semantics_verified",
        "enabled_joint_names",
        "target_force_N",
        "command_settings",
        "watchdog_ticks",
        "parameter_write_settle_s",
        "wire_position_deg",
        "arm_mit_gains",
    )
    summary = copy.deepcopy({key: receipt[key] for key in fields if key in receipt})
    for field in ("commands", "mode_writes", "watchdog_writes", "io_stages"):
        summary[f"{field}_count"] = receipt.get(f"{field}_count", len(receipt.get(field) or ()))
    summary["evidence_view"] = "summary; full evidence in bench trace and run result"
    return summary


def control_row_summary(row: dict) -> dict:
    """Leave the canonical trace row unchanged while publishing a bounded startup view."""
    view = dict(row)
    if view.get("startup_evidence") is not None:
        view["startup_evidence"] = startup_summary(view["startup_evidence"])
    evidence = view.get("device_io_evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("events"), list):
        events = evidence["events"]
        view["device_io_evidence"] = {key: value for key, value in evidence.items() if key != "events"}
        view["device_io_evidence"].update(
            events_count=len(events),
            first_sequence=events[0].get("sequence") if events else None,
            last_sequence=events[-1].get("sequence") if events else None,
            evidence_view="summary; full events in canonical control trace",
        )
    return view
