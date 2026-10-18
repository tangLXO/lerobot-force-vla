"""Closed query-only connection failures permit retry without inventing motor-off evidence."""

import copy
import hashlib
import json
import struct

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import (
    B601JointCommandProfile,
    encode_disable,
    encode_enable,
    encode_mode_query,
    encode_mode_write,
    encode_position,
    encode_timeout_write,
)
from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES
from tests.force_control import test_default_mit_entry

tk_root = test_default_mit_entry.tk_root


@pytest.fixture
def connection_failure():
    """Mirror the recorded 35-query/105-event failure using synthetic packets only."""
    events = []

    def event(kind, data):
        sequence = len(events) + 1
        events.append(
            {"sequence": sequence, "kind": kind, "timestamp_ns": 1000 + sequence, "data_hex": data.hex()}
        )

    for registers in ((21, 22, 23), (10, 9)):
        for motor_id in range(1, 8):
            for register_id in registers:
                query = bytearray(encode_mode_query(motor_id))
                query[24] = register_id
                event("query_attempt", bytes(query))
                value = (
                    struct.pack("<f", 12.5)
                    if register_id in (21, 22, 23)
                    else struct.pack("<I", (4 if motor_id == 7 else 2) if register_id == 10 else 0)
                )
                reply = (
                    b"\xaa\x11\x08"
                    + struct.pack("<I", motor_id + 0x10)
                    + bytes((motor_id, 0, 0x33, register_id))
                    + value
                    + b"\x55"
                )
                event("query_rx", reply[:1])
                event("query_rx", reply[1:])
    return {
        "errors": ["RuntimeError: Full-arm control requires existing mode=1 and TIMEOUT=2000: shoulder_pan"],
        "config": {"runtime": {"joint_names": list(JOINT_NAMES)}, "b601": {"port": "FAKE_FOLLOWER"}},
        "startup_evidence": None,
        "hardware_ack": "unknown",
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {
                "action": "disable_selected",
                "status": "not_attempted",
                "reason": "owner_connection_failed",
                "hardware_ack": "unknown",
                "hardware_stop_confirmed": False,
                "send_start_ns": None,
                "send_end_ns": None,
                "error": "RuntimeError: No active motion session for fault submission",
                "per_joint_results": [],
                "outstanding_query_accounted": True,
            },
            "device_io_evidence": {
                "complete": True,
                "correlation": "host_rx_only_no_request_token",
                "hardware_ack": "unknown",
                "outstanding_query": None,
                "query_write_inflight": None,
                "events": events,
            },
        },
    }


def test_complete_closed_connection_failure_permits_retry_without_claiming_disable(
    panel, connection_failure, tmp_path
):
    original = copy.deepcopy(connection_failure)
    assert len(connection_failure["shutdown_evidence"]["device_io_evidence"]["events"]) == 105
    assert panel.full_arm_result(connection_failure)
    assert panel.query_only_connection_failure(connection_failure)
    assert panel.exit_confirmation(connection_failure)
    report = panel.stop_report(connection_failure, tmp_path / "synthetic-result.json")
    assert "未提交使能或运动命令" in report["summary"]
    assert "串口已关闭" in report["summary"]
    assert "重新只读连接后可重试" in report["summary"]
    assert "已观察全臂七关节失能" not in report["summary"]
    assert "退出失能未确认" not in report["summary"]
    assert connection_failure == original


@pytest.mark.parametrize(
    "path,value",
    [
        (("startup_evidence",), {}),
        (("shutdown_evidence", "transport_closed"), False),
        (("shutdown_evidence", "transport_closed"), 1),
        (("shutdown_evidence", "fault_disposition", "reason"), "runtime_failure"),
        (("shutdown_evidence", "fault_disposition", "status"), "unknown"),
        (("shutdown_evidence", "fault_disposition", "status"), "submitted"),
        (("shutdown_evidence", "fault_disposition", "per_joint_results"), [{"status": "unknown"}]),
        (("shutdown_evidence", "device_io_evidence", "complete"), False),
        (("shutdown_evidence", "device_io_evidence", "complete"), 1),
        (("shutdown_evidence", "device_io_evidence", "events"), []),
        (("shutdown_evidence", "device_io_evidence", "events"), [None]),
        (("shutdown_evidence", "device_io_evidence", "events"), "invalid"),
        (("shutdown_evidence", "device_io_evidence"), None),
        (("shutdown_evidence", "fault_disposition"), []),
        (("shutdown_evidence",), []),
    ],
)
def test_missing_or_conflicting_query_only_evidence_stays_unconfirmed(panel, connection_failure, path, value):
    target = connection_failure
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert panel.query_only_connection_failure(connection_failure) is False


@pytest.mark.parametrize("result", [None, [], "invalid"])
def test_malformed_result_is_not_query_only_evidence(panel, result):
    assert panel.query_only_connection_failure(result) is False


@pytest.mark.parametrize("change", ["drained_prefix", "missing_middle", "duplicate", "missing_sequence"])
def test_incomplete_audit_cannot_be_mislabeled_complete(panel, connection_failure, change):
    events = connection_failure["shutdown_evidence"]["device_io_evidence"]["events"]
    if change == "drained_prefix":
        del events[:3]
    elif change == "missing_middle":
        del events[10]
    elif change == "duplicate":
        events.insert(10, copy.deepcopy(events[9]))
    else:
        del events[0]["sequence"]
    assert panel.query_only_connection_failure(connection_failure) is False
    assert panel.exit_confirmation(connection_failure) is False


@pytest.mark.parametrize(
    "packet",
    [
        encode_enable(1),
        encode_disable(1),
        encode_mode_write(1, "mit"),
        encode_timeout_write(1, 2000),
        encode_position(
            1,
            0,
            0,
            B601JointCommandProfile("shoulder_pan", "mit", -150, 150, kp=45, kd=12),
            (12.5, 10, 28),
        ).packet,
    ],
    ids=["enable", "disable", "mode_write", "timeout_write", "position"],
)
def test_command_disguised_as_query_is_rejected(panel, connection_failure, packet):
    connection_failure["shutdown_evidence"]["device_io_evidence"]["events"][0]["data_hex"] = packet.hex()
    assert panel.query_only_connection_failure(connection_failure) is False
    assert panel.exit_confirmation(connection_failure) is False


@pytest.mark.parametrize("kind", ["command_attempt", "command_rx", "parameter_write_unattributed_rx"])
def test_any_non_query_event_keeps_motion_disposition_unconfirmed(panel, connection_failure, kind):
    connection_failure["shutdown_evidence"]["device_io_evidence"]["events"][0]["kind"] = kind
    assert panel.query_only_connection_failure(connection_failure) is False
    assert panel.exit_confirmation(connection_failure) is False


@pytest.mark.parametrize("data_hex", [None, "zz", "", "00" * 30])
def test_invalid_query_packet_is_rejected(panel, connection_failure, data_hex):
    connection_failure["shutdown_evidence"]["device_io_evidence"]["events"][0]["data_hex"] = data_hex
    assert panel.query_only_connection_failure(connection_failure) is False


def test_rx_without_any_query_attempt_is_not_sufficient(panel, connection_failure):
    events = connection_failure["shutdown_evidence"]["device_io_evidence"]["events"]
    events[:] = [row for row in events if row["kind"] == "query_rx"]
    for sequence, row in enumerate(events, start=1):
        row["sequence"] = sequence
    assert panel.query_only_connection_failure(connection_failure) is False


def write_trace_closure(closure):
    """Write a temporary, hash-bound trace; never modify the user's historical logs."""
    path, rows, result = closure["path"], closure["rows"], closure["result"]
    stream = path.parent / "stream"
    stream.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (stream / "control.jsonl").write_bytes(raw)
    manifest = {
        "complete": False,
        "terminal_state": "FAULT",
        "run_uid": "synthetic-prestart-run",
        "metadata": {"config": copy.deepcopy(result["config"])},
        "files": {
            "control.jsonl": {
                "rows": len(rows),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    }
    (stream / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path.write_text(json.dumps(result), encoding="utf-8")


@pytest.fixture
def prestart_trace(connection_failure, tmp_path):
    """Mirror ac095's DISABLED tick-0 failure and drained shutdown audit."""
    result = connection_failure
    audit = copy.deepcopy(result["shutdown_evidence"]["device_io_evidence"])
    events = audit["events"]

    def event(kind, data):
        sequence = len(events) + 1
        events.append(
            {"sequence": sequence, "kind": kind, "timestamp_ns": 1000 + sequence, "data_hex": data.hex()}
        )

    for motor_id in [*range(1, 8), *range(1, 7)]:
        query = bytearray(encode_mode_query(motor_id))
        query[23:25] = bytes((0xCC, 0))
        event("query_attempt", bytes(query))
        if len(events) != 142:
            reply = (
                b"\xaa\x11\x08"
                + struct.pack("<I", motor_id + 0x10)
                + bytes((motor_id,))
                + b"\0" * 7
                + b"\x55"
            )
            event("query_rx", reply[:1])
            event("query_rx", reply[1:])
    assert len(events) == 142
    result["errors"] = [
        "RuntimeError: Owner fault: invalid_joint_feedback; joint_error: TimeoutError: B601 read-only deadline expired"
    ]
    result["config"]["initial_mode"] = "hybrid"
    result["shutdown_evidence"]["fault_disposition"].update(
        reason="invalid_joint_feedback", outstanding_query_accounted=False
    )
    result["shutdown_evidence"]["device_io_evidence"]["events"] = []
    result["shutdown_evidence"]["device_io_evidence"]["outstanding_query"] = {
        "opcode": 0xCC,
        "packet_hex": events[-1]["data_hex"],
        "submission": "complete",
    }
    closure = {
        "path": tmp_path / "synthetic-session" / "run" / "result.json",
        "result": result,
        "rows": [
            {
                "run_uid": "synthetic-prestart-run",
                "tick_id": 0,
                "phase_before": "DISABLED",
                "phase_after": "FAULT",
                "startup_evidence": None,
                "send_status": "none",
                "send_start_ns": None,
                "send_end_ns": None,
                "q_sent_deg": None,
                "submitted_action": {},
                "device_io_evidence": audit,
            }
        ],
    }
    write_trace_closure(closure)
    return closure


def test_prestart_feedback_failure_requires_complete_readonly_trace_closure(panel, prestart_trace):
    result, path = prestart_trace["result"], prestart_trace["path"]
    before = {
        file: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in path.parent.rglob("*")
        if file.is_file()
    }
    assert panel.query_only_connection_failure(result) is False
    assert panel.exit_confirmation(result) is False
    assert panel.query_only_connection_failure(result, log_path=path)
    assert panel.exit_confirmation(result, log_path=path)
    summary = panel.stop_report(result, path)["summary"]
    assert "未提交使能或运动命令" in summary and "串口已关闭" in summary
    assert "已观察全臂七关节失能" not in summary
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in before} == before


def test_trace_closure_accepts_empty_or_contiguous_shutdown_tail(panel, prestart_trace):
    result = prestart_trace["result"]
    events = prestart_trace["rows"][0]["device_io_evidence"]["events"]
    result["shutdown_evidence"]["device_io_evidence"]["events"] = events[-4:]
    del events[-4:]
    write_trace_closure(prestart_trace)
    assert panel.query_only_connection_failure(result, log_path=prestart_trace["path"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("send_status", "unknown"),
        ("send_status", "success"),
        ("send_start_ns", 1200),
        ("send_end_ns", 1201),
        ("q_sent_deg", 0),
        ("submitted_action", {"gripper.pos": 0}),
        ("startup_evidence", {}),
        ("run_uid", "another-run"),
    ],
)
def test_trace_closure_rejects_any_row_with_command_or_other_run(panel, prestart_trace, field, value):
    prestart_trace["rows"][0][field] = value
    write_trace_closure(prestart_trace)
    assert (
        panel.query_only_connection_failure(prestart_trace["result"], log_path=prestart_trace["path"])
        is False
    )


@pytest.mark.parametrize(
    "change", ["command", "configuration", "missing_prefix", "incomplete", "missing_audit"]
)
def test_bound_trace_still_requires_unbroken_query_only_audit(panel, prestart_trace, change):
    row = prestart_trace["rows"][0]
    audit = row["device_io_evidence"]
    if change == "command":
        audit["events"][-1]["kind"] = "command_attempt"
    elif change == "configuration":
        audit["events"][-1]["data_hex"] = encode_timeout_write(6, 2000).hex()
    elif change == "missing_prefix":
        del audit["events"][0]
    elif change == "incomplete":
        audit["complete"] = False
    else:
        del row["device_io_evidence"]
    write_trace_closure(prestart_trace)
    assert panel.exit_confirmation(prestart_trace["result"], log_path=prestart_trace["path"]) is False


@pytest.mark.parametrize(
    "change", ["hash", "size", "rows", "run_uid", "configuration", "malformed", "missing"]
)
def test_trace_manifest_must_bind_the_exact_file_and_session(panel, prestart_trace, change):
    path = prestart_trace["path"].parent / "stream" / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "hash":
        manifest["files"]["control.jsonl"]["sha256"] = "0" * 64
    elif change in ("size", "rows"):
        key = "size_bytes" if change == "size" else "rows"
        manifest["files"]["control.jsonl"][key] += 1
    elif change == "run_uid":
        manifest["run_uid"] = "another-run"
    elif change == "configuration":
        manifest["metadata"]["config"]["b601"]["port"] = "ANOTHER_PORT"
    elif change == "malformed":
        path.write_text("{broken-json", encoding="utf-8")
    else:
        path.unlink()
    if change not in ("malformed", "missing"):
        path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        panel.query_only_connection_failure(prestart_trace["result"], log_path=prestart_trace["path"])
        is False
    )


def test_trace_closure_does_not_follow_caller_selected_trace_root(panel, prestart_trace, tmp_path):
    prestart_trace["result"]["trace_root"] = str(tmp_path / "unrelated-stream")
    write_trace_closure(prestart_trace)
    assert panel.query_only_connection_failure(prestart_trace["result"], log_path=prestart_trace["path"])
    (prestart_trace["path"].parent / "stream" / "control.jsonl").unlink()
    assert (
        panel.query_only_connection_failure(prestart_trace["result"], log_path=prestart_trace["path"])
        is False
    )


def test_trace_closure_binds_the_provided_result_and_normalizes_queue_tuples(panel, prestart_trace):
    result, path = prestart_trace["result"], prestart_trace["path"]
    result["config"]["runtime"]["joint_names"] = tuple(result["config"]["runtime"]["joint_names"])
    assert panel.query_only_connection_failure(result, log_path=path)
    result["errors"].append("a result not recorded at the supplied path")
    assert panel.query_only_connection_failure(result, log_path=path) is False


def test_latest_history_and_result_poll_use_the_bound_trace_path(
    panel, prestart_trace, calibration, monkeypatch, tk_root
):
    monkeypatch.setattr(panel, "OUTPUTS", prestart_trace["path"].parents[2])

    def forbidden(*args, **kwargs):
        pytest.fail("Recovery classification must not connect devices or launch a process")

    for name in ("ReadonlyB601Transport", "X518Sensor", "ProcessLive", "run_in_process"):
        monkeypatch.setattr(panel, name, forbidden)
    app = panel.Tuner(tk_root, calibration_path=calibration.path)
    try:
        assert app.exit_confirmed
        assert "未提交使能或运动命令" in app.last_stop.get()
        app.exit_confirmed = False
        app.unconfirmed_full_arm = True
        app.messages.put(("result", (prestart_trace["result"], prestart_trace["path"])))
        app.poll()
        assert app.exit_confirmed and not app.unconfirmed_full_arm
        assert not app.running and not app.monitoring and app.next_run is None
    finally:
        for timer in tk_root.tk.call("after", "info"):
            tk_root.after_cancel(timer)
        for widget in tk_root.winfo_children():
            widget.destroy()
