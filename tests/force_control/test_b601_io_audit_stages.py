"""Audit startup/active/cleanup chronology without losing or reordering host evidence."""

import copy

import pytest

from lerobot.force_control.trace import _validate_b601_io
from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable


def stream(*events):
    return {
        "events": list(events),
        "complete": True,
        "correlation": "host_rx_only_no_request_token",
        "hardware_ack": "unknown",
    }


def event(sequence, kind="command_rx"):
    return {
        "sequence": sequence,
        "timestamp_ns": sequence * 100,
        "kind": kind,
        "data_hex": encode_disable(1).hex() if kind == "command_attempt" else "aa",
    }


@pytest.fixture
def audit():
    metadata = {
        "backend": "b601",
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {
                "status": "submitted",
                "hardware_ack": "unknown",
                "hardware_stop_confirmed": False,
            },
            "device_io_evidence": stream(event(6, "post_disable_unexpected_rx")),
        },
    }
    rows = [
        {"device_io_evidence": stream(event(1))},
        {
            "startup_evidence": {"io_stages": [stream(event(2, "command_attempt")), stream(event(3))]},
            "device_io_evidence": stream(event(4), event(5, "full_validation_stop_rx")),
        },
    ]
    return metadata, rows


def test_complete_nested_startup_and_stop_events_are_valid_and_not_mutated(audit):
    original = copy.deepcopy(audit)
    errors = []
    _validate_b601_io(*audit, errors)
    assert errors == [] and audit == original


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "time", "kind", "bytes", "incomplete", "stage_type"]
)
def test_startup_cannot_hide_corrupt_or_missing_evidence(audit, corruption):
    metadata, rows = audit
    stage = rows[1]["startup_evidence"]["io_stages"][0]
    if corruption == "missing":
        stage["events"] = []
    elif corruption == "duplicate":
        stage["events"][0]["sequence"] = 1
    elif corruption == "time":
        stage["events"][0]["timestamp_ns"] = 0
    elif corruption == "kind":
        stage["events"][0]["kind"] = "invented_rx"
    elif corruption == "bytes":
        stage["events"][0]["data_hex"] = "not-hex"
    elif corruption == "incomplete":
        stage["complete"] = False
    else:
        rows[1]["startup_evidence"]["io_stages"] = "invalid"
    errors = []
    _validate_b601_io(metadata, rows, errors)
    assert errors and len(errors) == len(set(errors))


def test_one_gap_does_not_generate_one_error_per_later_event(audit):
    metadata, rows = audit
    rows[1]["startup_evidence"]["io_stages"] = []
    rows[1]["device_io_evidence"] = stream(*(event(i) for i in range(4, 1004)))
    metadata["shutdown_evidence"]["device_io_evidence"] = stream()
    errors = []
    _validate_b601_io(metadata, rows, errors)
    assert errors == ["B601 I/O event sequence/time is not contiguous and monotonic"]
