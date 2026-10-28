"""Whole-arm bounded validation lifecycle using synthetic receipts and fake serial only."""

import builtins
import math
import os
import struct
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable, encode_enable
from lerobot.robots.rebot_b601_follower.full_arm_validation import JOINT_NAMES
from tests.force_control.full_arm_backend_helpers import close_full_scope, make_full_scope


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = make_full_scope(tmp_path, monkeypatch)
    yield value
    close_full_scope(value)


def motion_packets(serial):
    return [packet for packet in serial.writes if int.from_bytes(packet[13:17], "little") != 0x7FF]


def start(scope):
    return scope.device.start_full_arm_validation(guard=lambda: None, leader_sample=scope.leader)


def test_connection_remains_query_only_until_explicit_all_joint_start(scope, monkeypatch):
    device, serial = scope.device, scope.serial
    assert not motion_packets(serial)
    enabled_states = []
    original = serial.write

    def write(packet):
        result = original(packet)
        if packet[21:29] == b"\xff" * 7 + b"\xfc":
            enabled_states.append(tuple(index for index, status in serial.statuses.items() if status == 1))
        return result

    monkeypatch.setattr(serial, "write", write)
    final = start(scope)
    assert final.is_valid and len(final.joints) == 7
    assert enabled_states == [tuple(range(1, count + 1)) for count in range(1, 8)]
    packets = motion_packets(serial)
    first_enable = next(index for index, packet in enumerate(packets) if packet == encode_enable(1))
    assert first_enable >= 7
    assert [int.from_bytes(packet[13:17], "little") & 15 for packet in packets[:7]] == list(range(1, 8))
    assert all(packet[21:29] != b"\xff" * 7 + b"\xfc" for packet in packets[:7])
    assert not any(
        packet[23] == 0x55 for packet in serial.writes if int.from_bytes(packet[13:17], "little") == 0x7FF
    )
    receipt = device.startup_receipt
    assert receipt["complete"] and receipt["kind"] == "bounded_full_arm_validation"
    assert not receipt["production_arm_authorized"] and receipt["target_force_N"] == 1
    assert all(len(check["joints"]) == 7 for check in receipt["alignment_checks"])
    assert receipt["io_stages"] and all(stage["complete"] for stage in receipt["io_stages"])
    assert device.config.io_event_capacity == 256  # bounded stage drains preserve the existing capacity


def test_full_frame_receipts_report_actual_clipping_and_wire_positions(scope):
    snapshot = start(scope)
    before = len(motion_packets(scope.serial))
    goals = dict.fromkeys(JOINT_NAMES, 100.0)
    receipt = scope.device.send_joint_positions_partial(goals, feedback_snapshot=snapshot)
    assert receipt.all_submitted and len(receipt.results) == 7
    expected_ids = [0x100 + index for index in range(1, 7)] + [0x307]
    packets = motion_packets(scope.serial)[before:]
    assert [int.from_bytes(packet[13:17], "little") for packet in packets] == expected_ids
    for result, packet in zip(receipt.results, packets, strict=True):
        profile = scope.device._profiles[result.name]
        assert result.candidate_deg == 100
        assert result.clipped_deg == pytest.approx(
            snapshot.get(result.name).position_deg + profile.max_relative_deg
        )
        assert result.submitted_deg == pytest.approx(math.degrees(struct.unpack("<f", packet[21:25])[0]))
        assert result.hardware_ack == "unknown"


def test_normal_stop_attempts_all_seven_before_first_receive_and_closes(scope):
    start(scope)
    serial, device = scope.serial, scope.device
    before = len(serial.calls)
    receipt = device.emergency_stop("normal_validation_stop")
    operations = serial.calls[before:]
    first_read = next(index for index, operation in enumerate(operations) if operation[0] == "read")
    assert sum(operation[0] == "write" for operation in operations[:first_read]) == 7
    assert receipt["status"] == "submitted" and receipt["error"] is None
    assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]
    assert {item["name"] for item in receipt["per_joint_results"]} == set(JOINT_NAMES)
    assert all(status == 0 for status in serial.statuses.values())
    count = sum(packet == encode_disable(index) for packet in serial.writes for index in range(1, 8))
    device.disconnect()
    assert not device.has_resources
    assert sum(packet == encode_disable(index) for packet in serial.writes for index in range(1, 8)) == count


def test_disconnect_of_active_full_validation_performs_same_all_joint_cleanup(scope):
    start(scope)
    scope.device.disconnect()
    assert not scope.device.has_resources and all(status == 0 for status in scope.serial.statuses.values())
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


def test_alignment_refusal_reports_all_joint_angles_and_sends_no_enable(scope):
    scope.positions["wrist_roll"] = 20
    scope.positions["gripper"] = -20
    with pytest.raises(RuntimeError, match="alignment"):
        start(scope)
    assert not any(packet == encode_enable(index) for packet in scope.serial.writes for index in range(1, 8))
    check = scope.device.startup_receipt["alignment_checks"][-1]
    assert len(check["joints"]) == 7
    failed = {row["name"] for row in check["joints"] if not row["aligned"]}
    assert failed == {"wrist_roll", "gripper"}
    assert all(
        {"leader_deg", "follower_deg", "difference_deg", "tolerance_deg"} <= row.keys()
        for row in check["joints"]
    )


def test_session_expiry_prevents_another_position_frame_and_disables_all(scope):
    snapshot = start(scope)
    before = len(motion_packets(scope.serial))
    scope.clock.now += round(31 * 1e9)
    with pytest.raises(TimeoutError, match="duration"):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0.0), feedback_snapshot=snapshot)
    assert motion_packets(scope.serial)[before:] == [encode_disable(index) for index in range(1, 8)]
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize(
    "field,value", [("send_budget_s", 1.0), ("feedback_max_age_s", 1.0), ("fault_budget_s", 1.0)]
)
def test_changed_command_budgets_cannot_expand_the_admitted_session(scope, field, value):
    snapshot = start(scope)
    scope.device.config = replace(scope.device.config, **{field: value})
    before = len(motion_packets(scope.serial))
    with pytest.raises(RuntimeError, match="configuration/budgets"):
        scope.device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0.0), feedback_snapshot=snapshot)
    assert motion_packets(scope.serial)[before:] == [encode_disable(index) for index in range(1, 8)]
    assert scope.device.fault_receipt["all_joints_disabled_observed"]


@pytest.mark.parametrize("failure", ["unknown_id", "malformed", "duplicate", "register"])
def test_full_cleanup_never_ignores_unaccounted_frames(scope, monkeypatch, failure):
    start(scope)
    original = scope.serial.write
    injected = []

    def write(packet):
        result = original(packet)
        if packet == encode_disable(1) and not injected:
            injected.append(True)
            frame = bytes(scope.serial.pending[:16])
            if failure == "unknown_id":
                frame = frame[:3] + struct.pack("<I", 0x18) + bytes((8,)) + frame[8:]
            elif failure == "malformed":
                frame = frame[:-1] + b"\x54"
            elif failure == "register":
                frame = frame[:7] + bytes((1, 0, 0x33, 9)) + struct.pack("<I", 2000) + frame[15:]
            scope.serial.pending[:16] = frame
            if failure == "duplicate":
                scope.serial.pending[:0] = frame
        return result

    monkeypatch.setattr(scope.serial, "write", write)
    before = len(scope.serial.writes)
    receipt = scope.device.emergency_stop("unexpected_full_stop_RX")
    assert receipt["status"] == "unknown" and receipt["error"]
    assert not receipt["all_joints_disabled_observed"]
    assert scope.serial.writes[before:] == [encode_disable(index) for index in range(1, 8)]
    assert all(status == 0 for status in scope.serial.statuses.values())


@pytest.mark.parametrize("prefix_size", [0, 1, 15])
def test_full_cleanup_accounts_proven_query_bytes_and_marks_zero_prefix_ambiguity(
    scope, monkeypatch, prefix_size
):
    start(scope)
    device, serial, clock = scope.device, scope.serial, scope.clock
    original_timeout, original_read = device._bounded_serial_timeout, serial.read
    deadlines, injected = [], []

    def timeout(name, cap_s, deadline_ns):
        if name == "timeout":
            deadlines[:] = [deadline_ns]
            if (
                prefix_size == 0
                and not injected
                and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x14\x00\x00\x00"
            ):
                injected.append(True)
                clock.now = deadline_ns - 500_000
        return original_timeout(name, cap_s, deadline_ns)

    def read(size):
        if prefix_size > 0 and not injected and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x14\x00\x00\x00":
            injected.append(True)
            data = original_read(min(size, prefix_size))
            clock.now = deadlines[0] - 500_000
            return data
        return original_read(size)

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(serial, "read", read)
    snapshot = device.read_joint_snapshot(JOINT_NAMES, deadline_ns=clock.now + 50_000_000)
    assert not snapshot.is_valid and injected
    before = len(serial.writes)
    receipt = device.emergency_stop("known_query_fault")
    assert serial.writes[before : before + 7] == [encode_disable(index) for index in range(1, 8)]
    if prefix_size:
        assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]
        assert receipt["historical_responses"][0]["outstanding_query"]["motor_id"] == 4
        assert receipt["historical_responses"][0]["historical_only"]
    else:
        assert receipt["status"] == "unknown" and "ambiguous" in receipt["error"]
        assert not receipt["all_joints_disabled_observed"]
        assert len(serial.writes[before:]) == 7


@pytest.mark.parametrize("prefix_size", [0, 1])
def test_full_control_failure_preserves_command_response_debt_and_attempts_every_disable(
    scope, monkeypatch, prefix_size
):
    snapshot = start(scope)
    device, serial, clock = scope.device, scope.serial, scope.clock
    original_timeout, original_read = device._bounded_serial_timeout, serial.read
    deadlines, injected = [], []

    def timeout(name, cap_s, deadline_ns):
        if name == "timeout":
            deadlines[:] = [deadline_ns]
            if (
                prefix_size == 0
                and not injected
                and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x11\x00\x00\x00"
            ):
                injected.append(True)
                clock.now = deadline_ns - 500_000
        return original_timeout(name, cap_s, deadline_ns)

    def read(size):
        if prefix_size and not injected and bytes(serial.pending[:7]) == b"\xaa\x11\x08\x11\x00\x00\x00":
            injected.append(True)
            data = original_read(min(size, prefix_size))
            clock.now = deadlines[0] - 500_000
            return data
        return original_read(size)

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(serial, "read", read)
    before = len(motion_packets(serial))
    receipt = device.send_joint_positions_partial(dict.fromkeys(JOINT_NAMES, 0.1), feedback_snapshot=snapshot)
    assert not receipt.all_submitted and injected
    packets = motion_packets(serial)[before:]
    assert packets[-7:] == [encode_disable(index) for index in range(1, 8)]
    fault = device.fault_receipt
    if prefix_size:
        assert fault["all_joints_disabled_observed"] and fault["outstanding_query_accounted"]
        assert fault["historical_responses"][0]["pending_command"]["name"] == "shoulder_pan"
    else:
        assert fault["status"] == "unknown" and "ambiguous" in fault["error"]


def test_full_calibration_hash_still_rejects_same_stat_changed_bytes(scope):
    admission = scope.device._full_validation_session.admission
    pin = next(item for item in admission.file_identities if item.path == admission.calibration.path)
    with open(pin.path, "rb") as handle:
        original = handle.read()
    changed = original.replace(b"30.0", b"29.0", 1)
    assert changed != original and len(changed) == len(original)
    stat = os.stat(pin.path)
    with open(pin.path, "wb") as handle:
        handle.write(changed)
    os.utime(pin.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    pin.require_unchanged()
    with pytest.raises(ValueError, match="contents changed"):
        pin.require_contents_unchanged()


def test_each_full_write_binding_only_reads_calibration_contents_after_admission(scope, monkeypatch):
    reads, opens = [], []
    original = builtins.open
    reader = scope.device._full_validation_session._calibration_reader
    stream = reader._stream

    class ObservedStream:
        closed = False

        def seek(self, offset):
            return stream.seek(offset)

        def read(self, size):
            reads.append(size)
            return stream.read(size)

        def close(self):
            stream.close()

    def opened(file, *args, **kwargs):
        opens.append(os.fspath(file))
        return original(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(reader, "_stream", ObservedStream())
    for _ in range(3):
        scope.device._require_scope()
    assert reads == [reader.identity.size_bytes + 1] * 3
    assert opens == []


def test_disabled_state_proof_retains_position_violation_independently(scope):
    start(scope)
    scope.serial.positions[1] = 100
    receipt = scope.device.emergency_stop("external_mechanical_overtravel")
    assert receipt["all_joints_disabled_observed"] and receipt["status"] == "submitted"
    assert receipt["position_limit_violations"][0]["name"] == "shoulder_pan"
    assert receipt["hardware_ack"] == "unknown"
