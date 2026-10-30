"""Deterministic seven-axis asynchronous fault tests; no hardware evidence."""

import os
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower import control_backend, readonly_transport
from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable
from tests.force_control.operator_teleoperation_helpers import (
    JOINT_NAMES,
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.fixture
def scope(tmp_path, monkeypatch, request):
    files = make_operator_files(tmp_path)
    parameters = getattr(request, "param", 120)
    mode, gap, age = (
        (*parameters, None)
        if isinstance(parameters, tuple) and len(parameters) == 2
        else parameters
        if isinstance(parameters, tuple)
        else ("pipelined", parameters, None)
    )
    files.control = replace(
        files.control,
        command_feedback_mode=mode,
        command_tx_gap_us=gap,
        feedback_max_age_s=files.control.feedback_max_age_s if age is None else age,
    )
    value = make_operator_scope(tmp_path, monkeypatch, files=files)
    monkeypatch.setattr(
        time, "sleep", lambda seconds: setattr(value.clock, "now", value.clock.now + round(seconds * 1e9))
    )
    value.snapshot = value.device.start_operator_teleoperation(guard=lambda: None, leader_sample=value.leader)
    value.device.take_io_evidence()
    pipeline = value.device._full_validation_session.pipeline
    original_poll = pipeline.poll

    def progressing_poll():
        # Model time spent in real RX/guards even when a fake queue is empty.
        value.clock.now += 10_000
        return original_poll()

    monkeypatch.setattr(pipeline, "poll", progressing_poll)
    yield value
    close_operator_scope(value)


def send(scope):
    return scope.device.send_joint_positions_partial(
        {name: scope.snapshot.get(name).position_deg for name in JOINT_NAMES},
        feedback_snapshot=scope.snapshot,
        **(
            {"deadline_ns": scope.clock.now + 40_000_000}
            if scope.device.config.command_feedback_mode == "cached"
            else {}
        ),
    )


def hold_replies(scope, monkeypatch, *, partial_axis=None, late_axis=None):
    original = scope.serial.write
    replies = []

    def write(packet):
        result = original(packet)
        identifier = int.from_bytes(packet[13:17], "little")
        if identifier != 0x7FF and packet[21:29] not in (b"\xff" * 7 + b"\xfd", b"\xff" * 7 + b"\xfc"):
            replies.append(bytes(scope.serial.pending))
            scope.serial.pending.clear()
            if identifier & 15 == partial_axis:
                return result - 1
            if identifier & 15 == late_axis:
                scope.clock.now += 50_000_000
        return result

    monkeypatch.setattr(scope.serial, "write", write)
    return replies


def consume(scope):
    return scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 15_000_000)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_feedback_reuses_one_missing_axis_until_original_receive_expires(scope, monkeypatch):
    first = consume(scope)
    assert first.is_valid
    replies = hold_replies(scope, monkeypatch)
    scope.snapshot = first
    assert send(scope).all_submitted
    scope.serial.pending.extend(b"".join(replies[:6]))
    scope.device.service_pending_io()
    scope.clock.now += 20_000_000
    second = consume(scope)
    assert second.is_valid
    assert second.get("gripper").receive_sequence == first.get("gripper").receive_sequence
    assert second.get("gripper").receive_timestamp_ns == first.get("gripper").receive_timestamp_ns
    assert second.get("shoulder_pan").receive_sequence > first.get("shoulder_pan").receive_sequence
    scope.clock.now += 51_000_000
    with pytest.raises(RuntimeError, match="expired"):
        consume(scope)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_send_accepts_complete_12ms_host_wait_with_two_period_deadline(scope, monkeypatch):
    scope.snapshot = consume(scope)
    original = scope.serial.write
    count = 0

    def delayed_write(packet):
        nonlocal count
        count += 1
        result = original(packet)
        if count == 5:
            scope.clock.now += 12_100_000
        return result

    monkeypatch.setattr(scope.serial, "write", delayed_write)
    started = scope.clock.now
    receipt = scope.device.send_joint_positions_partial(
        {name: scope.snapshot.get(name).position_deg for name in JOINT_NAMES},
        feedback_snapshot=scope.snapshot,
        deadline_ns=started + 40_000_000,
    )
    assert receipt.all_submitted
    assert receipt.send_end_ns - started > 12_100_000
    assert receipt.send_end_ns <= started + 40_000_000


@pytest.mark.parametrize("scope", [("cached", 1000, 0.05)], indirect=True)
def test_cached_multi_axis_writes_continue_past_soft_period_with_original_hard_deadline(scope, monkeypatch):
    scope.snapshot = consume(scope)
    original = scope.serial.write
    started = scope.clock.now
    writes = []

    def delayed_write(packet):
        result = original(packet)
        axis = int.from_bytes(packet[13:17], "little")
        if axis != 0x7FF and packet[21:29] not in (b"\xff" * 7 + b"\xfd", b"\xff" * 7 + b"\xfc"):
            writes.append((axis, scope.clock.now))
            if axis == 1:
                scope.clock.now += 12_100_000
            elif axis == 3:
                scope.clock.now += 10_000_000
        return result

    monkeypatch.setattr(scope.serial, "write", delayed_write)
    receipt = scope.device.send_joint_positions_partial(
        {name: scope.snapshot.get(name).position_deg for name in JOINT_NAMES},
        feedback_snapshot=scope.snapshot,
        deadline_ns=started + 40_000_000,
    )
    assert receipt.all_submitted and scope.device.failure is None
    assert 20_000_000 < receipt.send_end_ns - started < 40_000_000
    assert len(writes) == 7
    assert writes[-1][1] > started + 20_000_000
    assert scope.device._full_validation_session.pipeline.cycle["deadline_ns"] == started + 40_000_000


@pytest.mark.parametrize("scope", [("cached", 1000, 0.05)], indirect=True)
@pytest.mark.parametrize("reason", ["user_stop", "left_force_hard_limit", "leader_expired"])
def test_cached_guard_failure_between_axes_preserves_prefix_and_clears_cache(scope, reason):
    scope.snapshot = consume(scope)
    failed = False

    def guard():
        if failed:
            raise RuntimeError(reason)

    def on_position(axis):
        nonlocal failed
        if axis == 3:
            failed = True

    scope.device._full_validation_session.guard = guard
    scope.serial.on_position = on_position
    receipt = send(scope)
    assert [item.status for item in receipt.results] == ["success"] * 3 + ["not_attempted"] * 4
    assert reason in scope.device.failure
    assert not scope.device._full_validation_session.pipeline.cached_rows
    assert scope.device.fault_receipt["fresh_disabled_verification_complete"]
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_fragment_crosses_tick_and_late_feedback_never_changes_frozen_snapshot(scope, monkeypatch):
    scope.snapshot = consume(scope)
    replies = hold_replies(scope, monkeypatch)
    assert send(scope).all_submitted
    scope.serial.pending.extend(replies[0][:5])
    scope.device.service_pending_io()
    frozen = consume(scope)
    scope.snapshot = frozen
    assert send(scope).all_submitted  # An incomplete earlier reply must not block TX.
    scope.serial.pending.extend(replies[0][5:] + b"".join(reversed(replies[1:7])))
    scope.device.service_pending_io()
    assert scope.device._latest_snapshot is frozen
    updated = consume(scope)
    assert all(updated.get(n).receive_sequence > frozen.get(n).receive_sequence for n in JOINT_NAMES)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_age_boundary_does_not_refresh_receive_identity(scope, monkeypatch):
    first = consume(scope)
    pipeline = scope.device._full_validation_session.pipeline
    monkeypatch.setattr(pipeline, "poll", lambda: None)
    oldest = min(row.receive_timestamp_ns for row in first.joints)
    scope.clock.now = oldest + 50_000_000
    boundary = consume(scope)
    for row in boundary.joints:
        assert row.receive_timestamp_ns == first.get(row.name).receive_timestamp_ns
        assert row.receive_sequence == first.get(row.name).receive_sequence
    scope.clock.now += 1
    with pytest.raises(RuntimeError, match="expired"):
        consume(scope)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
@pytest.mark.parametrize("damage", ["status", "corrupt", "unknown", "overflow"])
def test_cached_invalid_reply_is_not_hidden_by_normal_cache(scope, monkeypatch, damage):
    scope.snapshot = consume(scope)
    replies = hold_replies(scope, monkeypatch)
    assert send(scope).all_submitted
    payload = bytearray(replies[0])
    if damage == "status":
        payload[7] = 0x81
    elif damage == "corrupt":
        payload[0] = 0
    elif damage == "unknown":
        payload[3] = 0xFF
    else:
        payload *= 100
    scope.serial.pending.extend(payload)
    with pytest.raises(RuntimeError):
        consume(scope)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
@pytest.mark.parametrize("axis, delay_ns", [(3, 41_000_000), (7, 41_000_000), (3, 12_100_000)])
def test_cached_original_deadline_and_frozen_age_bound_every_axis(scope, monkeypatch, axis, delay_ns):
    scope.snapshot = consume(scope)
    if delay_ns < 40_000_000:
        # New incoming replies cannot extend the frozen input's remaining life.
        scope.clock.now = min(j.receive_timestamp_ns for j in scope.snapshot.joints) + 40_000_000
    started = scope.clock.now
    original = scope.serial.write

    def delayed(packet):
        result = original(packet)
        if (
            int.from_bytes(packet[13:17], "little") == (0x307 if axis == 7 else axis)
            and packet[21:29] != b"\xff" * 7 + b"\xfd"
        ):
            scope.clock.now += delay_ns
        return result

    monkeypatch.setattr(scope.serial, "write", delayed)
    receipt = scope.device.send_joint_positions_partial(
        {n: scope.snapshot.get(n).position_deg for n in JOINT_NAMES},
        feedback_snapshot=scope.snapshot,
        deadline_ns=started + 40_000_000,
    )
    assert [r.status for r in receipt.results] == ["success"] * axis + ["not_attempted"] * (7 - axis)
    assert scope.device.failure is not None
    assert receipt.send_end_ns >= started + delay_ns
    assert all(encode_disable(i) in scope.serial.writes for i in range(1, 8))


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_partial_write_never_retries(scope, monkeypatch):
    scope.snapshot = consume(scope)
    hold_replies(scope, monkeypatch, partial_axis=3)
    before = scope.serial.position_counts.copy()
    receipt = send(scope)
    assert [r.status for r in receipt.results] == ["success"] * 2 + ["unknown"] + ["not_attempted"] * 4
    assert scope.serial.position_counts[3] - before[3] == 1
    assert all(scope.serial.position_counts[i] == before[i] for i in range(4, 8))


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_poison_cannot_be_hidden_by_an_empty_receive_queue(scope):
    consume(scope)
    scope.device._rx_poison = "injected retained parsing fault"
    with pytest.raises(RuntimeError, match="retained parsing fault"):
        consume(scope)


@pytest.mark.parametrize("scope", [("cached", 120, 0.05)], indirect=True)
def test_cached_stop_during_final_write_is_checked_before_control_can_commit(scope):
    scope.snapshot = consume(scope)
    stopped = False

    def guard():
        if stopped:
            raise RuntimeError("injected stop during final write")

    def on_position(axis):
        nonlocal stopped
        if axis == 7:
            stopped = True

    scope.device._full_validation_session.guard = guard
    scope.serial.on_position = on_position
    receipt = send(scope)
    assert receipt.all_submitted  # Actual host write facts remain true.
    assert "stop during final write" in scope.device.failure
    assert scope.device.fault_receipt["fresh_disabled_verification_complete"]


def test_cached_disabled_queries_remain_strict_before_explicit_start(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)
    files.control = replace(files.control, command_feedback_mode="cached", feedback_max_age_s=0.05)
    value = make_operator_scope(tmp_path, monkeypatch, files=files)
    try:
        first = value.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=value.clock.now + 50_000_000)
        second = value.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=value.clock.now + 50_000_000)
        assert first.is_valid and second.is_valid
        assert all(second.get(n).receive_sequence > first.get(n).receive_sequence for n in JOINT_NAMES)
        assert not value.device._full_validation_session.pipeline.cached_rows
    finally:
        close_operator_scope(value)


def test_seven_writes_before_any_response_then_out_of_order_fragmented_next_tick(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    writes = len(scope.serial.writes)
    old = scope.device._latest_snapshot
    receipt = send(scope)
    assert receipt.all_submitted and len(scope.serial.writes) - writes == 7
    assert len(replies) == 7 and scope.device._latest_snapshot is old
    pipeline = scope.device._full_validation_session.pipeline
    assert pipeline.pending and len(pipeline.cycle["feedback"]) == 0
    assert min(pipeline.cycle["actual_write_intervals_ns"]) >= 120_000
    assert all(row.hardware_ack == "unknown" for row in receipt.results)
    scope.clock.now += 2_000_000
    scope.serial.fragment_size = 3
    scope.serial.pending.extend(b"".join(reversed(replies)))
    scope.device.service_pending_io()
    assert not pipeline.pending and scope.device._latest_snapshot is old
    scope.clock.now += 5_000_000
    result = consume(scope)
    assert result.is_valid and all(row.age_ns >= 5_000_000 for row in result.joints)
    assert result.completed_ns < scope.clock.now
    assert all(result.get(name).receive_sequence > old.get(name).receive_sequence for name in JOINT_NAMES)
    with pytest.raises(RuntimeError, match="already consumed"):
        consume(scope)
    evidence = scope.device.take_io_evidence()
    assert evidence["consumed_command_feedback_group"]["completed_ns"] <= pipeline.cycle["deadline_ns"]


def test_partial_frame_across_service_calls_and_snapshot_published_atomically(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    old = scope.device._latest_snapshot
    scope.serial.pending.extend(replies[0][:5])
    scope.device.service_pending_io()
    assert len(scope.device._rx_pending) == 5
    assert scope.device._latest_snapshot is old
    scope.serial.pending.extend(replies[0][5:] + b"".join(replies[1:]))
    scope.device.service_pending_io()
    assert consume(scope).is_valid


def test_queued_last_frame_can_complete_with_less_than_one_ms_remaining(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(b"".join(replies[:6]))
    scope.device.service_pending_io()
    pipeline = scope.device._full_validation_session.pipeline
    assert len(pipeline.cycle["feedback"]) == 6 and pipeline.pending
    scope.clock.now = pipeline.cycle["deadline_ns"] - 500_000
    scope.serial.pending.extend(replies[6])
    scope.device.service_pending_io()
    assert not pipeline.pending
    assert pipeline.cycle["completed_ns"] < pipeline.cycle["deadline_ns"]
    assert scope.serial.timeout == 0.002


def test_immediate_read_cannot_accept_bytes_returned_after_deadline(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(b"".join(replies[:6]))
    scope.device.service_pending_io()
    pipeline = scope.device._full_validation_session.pipeline
    scope.clock.now = pipeline.cycle["deadline_ns"] - 500_000
    scope.serial.pending.extend(replies[6])
    original = scope.serial.read

    def late_read(size):
        data = original(size)
        scope.clock.now = pipeline.cycle["deadline_ns"] + 1
        return data

    monkeypatch.setattr(scope.serial, "read", late_read)
    with pytest.raises(TimeoutError, match="after deadline"):
        scope.device.service_pending_io()
    assert pipeline.pending and len(scope.device._rx_pending) == 16


def test_timeout_restore_failure_keeps_read_bytes_and_audit_before_fault(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(replies[0])

    @contextmanager
    def fail_restore(port):
        yield
        raise OSError("injected read-mode restore failure")

    monkeypatch.setattr(readonly_transport, "nonblocking_serial_read", fail_restore)
    with pytest.raises(OSError, match="restore failure"):
        scope.device.service_pending_io()
    assert bytes(scope.device._rx_pending) == replies[0]
    assert any(event["data_hex"] == replies[0].hex() for event in scope.device.take_io_evidence()["events"])


def test_queue_count_is_sampled_once_per_drain_iteration_with_all_live_guards(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(replies[0])
    events = []
    getter = type(scope.serial).in_waiting.fget
    original_read = scope.serial.read

    def queued(port):
        events.append("queue")
        return getter(port)

    def read(size):
        events.append(("read", size))
        return original_read(size)

    scope.device._full_validation_session.guard = lambda: events.append("guard")
    monkeypatch.setattr(type(scope.serial), "in_waiting", property(queued))
    monkeypatch.setattr(scope.serial, "read", read)
    scope.device.service_pending_io()
    assert events.count("queue") == 2  # One frame, then the empty drain iteration.
    assert events.count("guard") == 5  # Service, poll, loop, guarded read, next loop.
    assert events.index(("read", 16)) > events.index("queue")
    assert len(scope.device._full_validation_session.pipeline.cycle["feedback"]) == 1


def test_cached_queued_bytes_never_allow_a_read_after_guard_expires_deadline(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(replies[0])
    pipeline = scope.device._full_validation_session.pipeline
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == 4:
            scope.clock.now = pipeline.cycle["deadline_ns"] + 1

    scope.device._full_validation_session.guard = guard
    with pytest.raises(TimeoutError, match="deadline expired"):
        scope.device.service_pending_io()
    assert not scope.device._rx_pending
    assert bytes(scope.serial.pending) == replies[0]


def test_configured_native_read_avoids_mode_switch_and_reuses_only_known_byte_lower_bound(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(replies[0][:3])
    calls, sizes = 0, []
    original_read = scope.serial.read

    def guard():
        nonlocal calls
        calls += 1
        if calls == 4:
            scope.serial.pending.extend(replies[0][3:] + b"".join(replies[1:]))

    def read(size):
        sizes.append(size)
        return original_read(size)

    def unexpected_mode_switch(_port):
        pytest.fail("Configured native reads must not switch or restore read modes per chunk")

    scope.device._full_validation_session.guard = guard
    monkeypatch.setattr(scope.serial, "read", read)
    monkeypatch.setattr(readonly_transport, "native_first_byte_read_ready", lambda _port: True)
    monkeypatch.setattr(readonly_transport, "nonblocking_serial_read", unexpected_mode_switch)
    scope.device.service_pending_io()
    assert sizes[:2] == [3, 13]
    assert consume(scope).is_valid
    assert scope.serial.timeout == 0.002


def test_first_byte_setup_failure_closes_transport_and_calibration_before_any_command(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)
    files.control = replace(files.control, command_feedback_mode="pipelined")
    value = make_operator_scope(tmp_path, monkeypatch, files=files, connect=False)

    def fail_setup(_port):
        raise OSError("injected native first-byte setup failure")

    monkeypatch.setattr(control_backend, "configure_native_first_byte_read", fail_setup)
    with pytest.raises(OSError, match="native first-byte setup failure"):
        value.device.open()
    assert not value.device.has_resources
    assert not value.device._control_session
    assert not value.serial.writes


@pytest.mark.parametrize("scope", [1000], indirect=True)
def test_one_millisecond_tx_gap_keeps_safety_checks_and_original_deadline(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    checks = []
    scope.device._full_validation_session.guard = lambda: checks.append(scope.clock.now)
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("TX pacing must keep servicing RX"))
    receipt = send(scope)
    assert receipt.all_submitted and len(replies) == 7
    pipeline = scope.device._full_validation_session.pipeline
    assert min(pipeline.cycle["actual_write_intervals_ns"]) >= 1_000_000
    assert len(checks) > 7
    original_deadline = pipeline.cycle["deadline_ns"]
    assert original_deadline - receipt.send_start_ns == round(scope.device.config.send_budget_s * 1e9)
    scope.serial.pending.extend(b"".join(replies))
    scope.device.service_pending_io()
    assert consume(scope).is_valid
    assert pipeline.cycle["deadline_ns"] == original_deadline


def test_pending_rx_service_diagnostics_preserve_empty_polls_and_service_gaps(scope, monkeypatch):
    hold_replies(scope, monkeypatch)
    send(scope)
    pipeline = scope.device._full_validation_session.pipeline
    before = pipeline.evidence()["rx_service"]
    scope.clock.now += 3_000_000
    scope.device.service_pending_io()
    after = pipeline.evidence()["rx_service"]
    assert after["poll_count"] > before["poll_count"]
    assert after["empty_poll_count"] > before["empty_poll_count"]
    assert after["max_service_gap_ns"] >= 3_000_000
    assert after["last_driver_bytes"] == after["last_retained_bytes"] == 0
    scope.clock.now = pipeline.cycle["deadline_ns"] + 1
    with pytest.raises(TimeoutError, match="deadline expired"):
        scope.device.service_pending_io()
    failed = pipeline.evidence()["rx_service"]
    assert failed["last_poll_completed_ns"] > pipeline.cycle["deadline_ns"]
    assert failed["poll_count"] > after["poll_count"]


def test_diagnostic_signature_references_preserve_context_and_detach(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(b"".join(replies))
    scope.device.service_pending_io()
    pipeline = scope.device._full_validation_session.pipeline
    evidence = pipeline.evidence()
    signature = evidence["session_signature"]
    for name in JOINT_NAMES:
        command = evidence["commands"][name]
        context = evidence["feedback"][name]["rx_context"]
        assert command.pop("session_signature_ref") == "group.session_signature"
        assert context.pop("session_signature_ref") == "group.session_signature"
        assert {**command, "session_signature": signature} == pipeline.cycle["commands"][name]
        assert {**context, "session_signature": signature} == pipeline.cycle["feedback"][name]["rx_context"]
    evidence["commands"]["gripper"]["submission"] = "changed by external reader"
    assert pipeline.cycle["commands"]["gripper"]["submission"] == "complete"
    # Even inconsistent diagnostic context is retained, not silently replaced.
    pipeline.cycle["feedback"]["gripper"]["rx_context"]["session_signature"] = ("different",)
    assert pipeline.evidence()["feedback"]["gripper"]["rx_context"]["session_signature"] == ("different",)


def test_no_overlap_or_query_fallback_when_group_missing(scope, monkeypatch):
    hold_replies(scope, monkeypatch)
    send(scope)
    pipeline = scope.device._full_validation_session.pipeline
    deadline = pipeline.cycle["deadline_ns"]
    with pytest.raises(RuntimeError, match="not been consumed"):
        pipeline.begin(scope.clock.now, deadline + 100_000_000)
    scope.clock.now = deadline + 1
    before = len(scope.serial.writes)
    with pytest.raises(TimeoutError, match="deadline expired"):
        consume(scope)
    assert len(scope.serial.writes) == before


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "corrupt", "status", "overflow", "late"])
def test_invalid_rx_fails_closed_and_retains_evidence(scope, monkeypatch, damage):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    payload = bytearray(b"".join(replies))
    if damage == "duplicate":
        payload[16:32] = payload[:16]
    elif damage == "unknown":
        payload[3] = 0xFF
    elif damage == "corrupt":
        payload[0] = 0
    elif damage == "status":
        payload[7] = 0x81
    elif damage == "overflow":
        payload *= 3
    elif damage == "late":
        scope.clock.now = scope.device._full_validation_session.pipeline.cycle["deadline_ns"] + 1
    scope.serial.pending.extend(payload)
    with pytest.raises((RuntimeError, TimeoutError)):
        scope.device.service_pending_io()
    assert scope.device._full_validation_session.pipeline.pending
    scope.device.emergency_stop("injected_receive_fault")
    assert all(encode_disable(i) in scope.serial.writes for i in range(1, 8))
    assert scope.device.fault_receipt["pending_feedback_group_before_disable"] is not None


def test_partial_write_classification_and_all_disable_attempts(scope, monkeypatch):
    hold_replies(scope, monkeypatch, partial_axis=3)
    receipt = send(scope)
    assert [row.status for row in receipt.results] == ["success", "success", "unknown"] + [
        "not_attempted"
    ] * 4
    assert all(encode_disable(i) in scope.serial.writes for i in range(1, 8))
    assert scope.device.fault_receipt["fresh_disabled_verification_complete"]


def test_late_complete_write_preserves_host_submission(scope, monkeypatch):
    hold_replies(scope, monkeypatch, late_axis=3)
    receipt = send(scope)
    assert [row.status for row in receipt.results] == ["success"] * 3 + ["not_attempted"] * 4
    assert scope.device.failure is not None


def test_stop_during_receive_preserves_tail_and_separates_disabled_probes(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(b"".join(replies))
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise RuntimeError("injected emergency stop")

    scope.device._full_validation_session.guard = guard
    with pytest.raises(RuntimeError, match="emergency stop"):
        scope.device.service_pending_io()
    scope.device.emergency_stop("emergency stop")
    receipt = scope.device.fault_receipt
    assert len(receipt["per_joint_results"]) == 7
    assert receipt["fresh_disabled_verification_complete"]
    assert receipt["pending_feedback_group_before_disable"]["completed_ns"] is None
    assert all(entry["historical_only"] for entry in receipt["drained_responses"])
    assert receipt["hardware_ack"] == "unknown"


def test_calibration_content_checked_before_every_pipelined_write(scope, monkeypatch):
    original = scope.serial.write
    pin = scope.device._full_validation_session.calibration_identity
    changed = False

    def write(packet):
        nonlocal changed
        result = original(packet)
        if not changed:
            changed = True
            stat = os.stat(pin.path)
            with open(pin.path, "rb") as handle:
                contents = handle.read()
            with open(pin.path, "wb") as handle:
                handle.write(contents.replace(b"200.0", b"199.0", 1))
            os.utime(pin.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return result

    monkeypatch.setattr(scope.serial, "write", write)
    receipt = send(scope)
    assert receipt.results[0].status == "success"
    assert all(row.status == "not_attempted" for row in receipt.results[1:])
    assert "contents changed" in receipt.results[1].error
    assert scope.device.fault_receipt["fresh_disabled_verification_complete"]


def test_close_with_pending_feedback_archives_tail_and_closes_owner(scope, monkeypatch):
    replies = hold_replies(scope, monkeypatch)
    send(scope)
    scope.serial.pending.extend(b"".join(replies))
    scope.device.disconnect()
    assert not scope.device.has_resources
    assert scope.device.fault_receipt["pending_feedback_group_before_disable"] is not None
    assert scope.device.fault_receipt["fresh_disabled_verification_complete"]
