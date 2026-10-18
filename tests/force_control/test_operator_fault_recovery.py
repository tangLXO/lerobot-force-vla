"""Fault-only RX quarantine followed by independent seven-axis state probes."""

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable
from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES
from tests.force_control.operator_teleoperation_helpers import close_operator_scope, make_operator_scope


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = make_operator_scope(tmp_path, monkeypatch)
    yield value
    close_operator_scope(value)


def start(scope):
    return scope.device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)


@pytest.mark.parametrize("prefix_size", [0, 1, 15])
def test_active_late_query_is_quarantined_then_new_disabled_state_is_verified(
    scope, monkeypatch, prefix_size
):
    start(scope)
    # Consume the post-enable snapshot so this read performs a genuinely new query.
    scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 50_000_000)
    device, serial, clock = scope.device, scope.serial, scope.clock
    original_timeout, original_read = device._bounded_serial_timeout, serial.read
    deadlines, injected = [], []

    def timeout(name, cap_s, deadline_ns):
        if name == "timeout":
            deadlines[:] = [deadline_ns]
            if (
                prefix_size == 0
                and not injected
                and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x15\x00\x00\x00"
            ):
                injected.append(True)
                clock.now = deadline_ns - 500_000
        return original_timeout(name, cap_s, deadline_ns)

    def read(size):
        if prefix_size and not injected and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x15\x00\x00\x00":
            injected.append(True)
            data = original_read(min(size, prefix_size))
            clock.now = deadlines[0] - 500_000
            return data
        return original_read(size)

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(serial, "read", read)
    failed = device.read_joint_snapshot(JOINT_NAMES, deadline_ns=clock.now + 15_000_000)
    assert not failed.is_valid and injected
    before = len(serial.writes)
    receipt = device.emergency_stop("invalid_joint_feedback")
    assert serial.writes[before : before + 7] == [encode_disable(index) for index in range(1, 8)]
    assert receipt["all_joints_disabled_observed"] and receipt["fresh_disabled_verification_complete"]
    assert receipt["transactions_retired_for_cleanup"] and receipt["status"] == "submitted"
    assert receipt["hardware_ack"] == "unknown" and not receipt["hardware_stop_confirmed"]
    assert len(serial.writes[before + 7 :]) == 7  # independent fresh seven-joint probe
    if prefix_size == 0:
        assert not receipt["outstanding_query_accounted"]
        transaction = receipt["unresolved_transactions"][0]
        assert transaction["kind"] == "query" and transaction["evidence"]["name"] == "wrist_yaw"
        assert transaction["outcome"] == "unresolved_retired_after_fault_rx_drain"
    else:
        assert receipt["outstanding_query_accounted"]
    assert any(bytes.fromhex(item["packet_hex"])[7] == 0x15 for item in receipt["drained_responses"])


def test_fresh_enabled_probe_never_becomes_a_successful_disabled_verification(scope, monkeypatch):
    start(scope)
    original = scope.serial.write

    def write(packet):
        result = original(packet)
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[21:24] == bytes((7, 0, 0xCC)):
            scope.serial.pending[-9] = 0x17  # valid ENABLED/id7 in the new probe reply
        return result

    monkeypatch.setattr(scope.serial, "write", write)
    receipt = scope.device.emergency_stop("fresh_probe_enabled")
    assert receipt["transactions_retired_for_cleanup"]
    assert not receipt["all_joints_disabled_observed"]
    assert not receipt["fresh_disabled_verification_complete"]
    assert receipt["status"] == "unknown" and "drive state" in receipt["error"]


@pytest.mark.parametrize("failure", ["no_rx", "partial_write"])
def test_unknown_motion_submission_stays_unknown_after_independent_disabled_verification(
    scope, monkeypatch, failure
):
    snapshot = start(scope)
    original_timeout, original_write = scope.device._bounded_serial_timeout, scope.serial.write
    injected = []

    def timeout(name, cap_s, deadline_ns):
        if (
            name == "timeout"
            and not injected
            and scope.device._full_validation_session.pending_response is not None
        ):
            injected.append(True)
            scope.clock.now = deadline_ns - 500_000
        return original_timeout(name, cap_s, deadline_ns)

    def write(packet):
        result = original_write(packet)
        if (
            not injected
            and int.from_bytes(packet[13:17], "little") == 1
            and packet[21:29] != b"\xff" * 7 + b"\xfd"
        ):
            injected.append(True)
            return result - 1
        return result

    monkeypatch.setattr(
        scope.device, "_bounded_serial_timeout", timeout if failure == "no_rx" else original_timeout
    )
    monkeypatch.setattr(scope.serial, "write", write if failure == "partial_write" else original_write)
    submitted = scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 0.1), feedback_snapshot=snapshot
    )
    assert not submitted.all_submitted and injected
    receipt = scope.device.fault_receipt
    assert receipt["all_joints_disabled_observed"] and receipt["fresh_disabled_verification_complete"]
    assert receipt["status"] == "submitted" and not receipt["outstanding_query_accounted"]
    transaction = receipt["unresolved_transactions"][0]
    assert transaction["kind"] == "command" and transaction["evidence"]["name"] == "shoulder_pan"
    assert transaction["evidence"]["submission"] == ("complete" if failure == "no_rx" else "entered_unknown")
    assert receipt["hardware_ack"] == "unknown" and not receipt["hardware_stop_confirmed"]


def test_a_timeout_in_the_new_disabled_probe_remains_an_unconfirmed_state(scope, monkeypatch):
    start(scope)
    original = scope.device._bounded_serial_timeout
    injected = []

    def timeout(name, cap_s, deadline_ns):
        outstanding = scope.device._outstanding_query
        if (
            name == "timeout"
            and outstanding is not None
            and outstanding["name"] == "wrist_roll"
            and not injected
        ):
            injected.append(True)
            scope.clock.now = deadline_ns - 500_000
        return original(name, cap_s, deadline_ns)

    monkeypatch.setattr(scope.device, "_bounded_serial_timeout", timeout)
    receipt = scope.device.emergency_stop("fresh_probe_timeout")
    assert injected and receipt["transactions_retired_for_cleanup"]
    assert not receipt["all_joints_disabled_observed"] and not receipt["fresh_disabled_verification_complete"]
    assert receipt["status"] == "unknown" and receipt["error"]


@pytest.mark.parametrize("failure", ["framing", "unknown_id", "register", "overflow"])
def test_malformed_history_is_not_silently_discarded_to_obtain_a_false_clean_state(
    scope, monkeypatch, failure
):
    start(scope)
    original = scope.serial.write

    def write(packet):
        result = original(packet)
        if packet == encode_disable(1):
            if failure == "framing":
                scope.serial.pending[15] = 0x54
            elif failure == "unknown_id":
                scope.serial.pending[3] = 0x18
            elif failure == "register":
                scope.serial.pending[7:15] = bytes((1, 0, 0x33, 9, 0, 0, 0, 0))
            else:
                scope.serial.pending[:0] = bytes(scope.serial.pending[:16]) * 17
        return result

    monkeypatch.setattr(scope.serial, "write", write)
    before = len(scope.serial.writes)
    receipt = scope.device.emergency_stop("malformed_fault_history")
    assert scope.serial.writes[before:] == [encode_disable(index) for index in range(1, 8)]
    assert not receipt["all_joints_disabled_observed"]
    assert not receipt["transactions_retired_for_cleanup"]
    assert receipt["status"] == "unknown" and receipt["error"]
