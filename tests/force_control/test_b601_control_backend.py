"""Unified transport tests with a fake serial factory; never open a real port."""

import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.control_io import B601BusLease
from lerobot.robots.rebot_b601_follower.control_protocol import (
    encode_disable,
    encode_enable,
    encode_mode_query,
    encode_mode_write,
    encode_position,
)
from tests.force_control.hardware_helpers import (
    Clock,
    ControlSerial,
    bridge_packet,
    commands,
    profile,
    snapshot,
)


@pytest.fixture
def setup_backend(monkeypatch, tmp_path):
    """Patch serial before replacing the offline acceptance gate for byte-level testing."""
    clock = Clock()
    monkeypatch.setattr(backend.time, "perf_counter_ns", clock.perf_counter_ns)
    fake = ControlSerial(clock)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(readonly.serial, "Serial", factory)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -30, "range_max": 30}}
        ),
        encoding="utf-8",
    )
    config = backend.B601ControlConfig(
        (profile(),), 0.5, 0.005, 0.005, 0.02, "disable_gripper", calibration_file=str(calibration)
    )
    device = backend.B601ControlBackend(
        "FAKE_CONTROL", readonly.DEFAULT_B601_JOINTS[-1:], config, lock_root=tmp_path
    )
    yield device, fake, clock, factory
    fake.close_fails = False
    fake.interrupt_motor = None
    if device.has_resources:
        with suppress(RuntimeError):
            device.disconnect()


def connected(setup_backend, monkeypatch):
    """Lift only the test gate after the fixture has replaced the serial factory."""
    device, fake, clock, factory = setup_backend
    monkeypatch.setattr(backend, "require_b601_control_acceptance", lambda: None)
    device.connect()
    device.start_gripper(snapshot(device, clock), lambda: None)
    fake.writes.clear()
    device.take_io_evidence()
    return device, fake, clock, factory


def timed_out_query_prefix(device, fake, clock, monkeypatch, packet):
    """Read only AA before a query expires, leaving its tail queued on the bridge."""
    fake.feedback_transform = lambda _packet, _motor_id: packet
    original_read = fake.read

    def read_prefix(size):
        data = original_read(min(size, 1))
        clock.now += 4_000_000
        return data

    monkeypatch.setattr(fake, "read", read_prefix)
    feedback = snapshot(device, clock)
    monkeypatch.setattr(fake, "read", original_read)
    assert not feedback.is_valid and "TimeoutError" in feedback.error
    assert bytes(device._rx_pending) == packet[:1]
    assert bytes(fake.pending) == packet[1:]
    return feedback


@pytest.mark.parametrize("fragment_size", [999, 1])
def test_fault_reassembles_timed_out_query_prefix_without_reauditing_or_committing(
    setup_backend, monkeypatch, fragment_size
):
    """Replay 8550432c's AA + old tail15 + next AA on one unchanged owner."""
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=1)
    sequences = dict(device._sequences)
    feedback = timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    failure = device.failure
    prefix_events = device.take_io_evidence()["events"]
    prefix = next(event for event in prefix_events if event["kind"] == "query_rx")
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    fake.fragment_size = fragment_size
    started = clock.now
    stop = device.emergency_stop("query_timeout")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    assert stop["hardware_ack"] == "unknown" and not stop["hardware_stop_confirmed"]
    history = item["intermediate_feedback"]
    assert len(history) == 1 and history[0]["packet_hex"] == old.hex()
    assert history[0]["historical_only"]
    context = history[0]["rx_context"]
    assert context["source"] == "query_rx" and context["inherited"]
    assert context["inherited_prefix_hex"] == "aa"
    assert context["deadline_ns"] == feedback.deadline_ns
    assert context["first_receive_timestamp_ns"] == prefix["timestamp_ns"]
    assert context["first_audit_sequence"] == prefix["sequence"]
    assert not item["feedback"]["rx_context"]["inherited"]
    assert item["receive_deadline_ns"] == started + int(device.config.feedback_max_age_s * 1e9)
    assert device._sequences == sequences and device.failure == failure
    tail_events = device.take_io_evidence()["events"]
    rx = [event for event in tail_events if event["kind"] == "command_rx"]
    assert bytes.fromhex("".join(event["data_hex"] for event in rx)) == old[1:] + bridge_packet(7, status=0)
    assert context["last_audit_sequence"] == rx[0 if fragment_size == 999 else 14]["sequence"]
    assert not device._rx_pending
    assert device.emergency_stop("again") == stop
    device.disconnect()
    assert commands(fake) == [encode_disable(7)]


def test_inherited_disabled_prefix_alone_cannot_confirm_current_fault(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=0)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    fake.auto_command_feedback = False
    started = clock.now
    stop = device.emergency_stop("old_disabled_query")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "unknown" and "TimeoutError" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert "feedback" not in item and device._last_command_feedback is None
    assert item["intermediate_feedback"][0]["historical_only"]
    assert item["intermediate_feedback"][0]["status_code"] == 0
    assert item["receive_deadline_ns"] == started + int(device.config.feedback_max_age_s * 1e9)
    assert commands(fake) == [encode_disable(7)]


def test_inherited_tail_after_fault_deadline_remains_unknown_and_uncommitted(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=1)
    sequences = dict(device._sequences)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    fake.late_read = True
    stop = device.emergency_stop("late_old_tail")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "unknown" and "arrived after deadline" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert not item["intermediate_feedback"] and "feedback" not in item
    assert device._sequences == sequences
    assert bytes(device._rx_pending) == old
    assert bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


def test_late_complete_disabled_query_is_history_not_current_fault_confirmation(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=0)
    sequences = dict(device._sequences)
    fake.feedback_transform = lambda _packet, _motor_id: old
    fake.late_read = True
    feedback = snapshot(device, clock)
    assert not feedback.is_valid and bytes(device._rx_pending) == old
    fake.late_read = False
    fake.auto_command_feedback = False
    stop = device.emergency_stop("late_disabled_query")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "unknown" and not stop.get("disabled_feedback_observed", False)
    assert "feedback" not in item
    row = item["intermediate_feedback"][0]
    assert row["status_code"] == 0 and row["historical_only"]
    assert row["rx_context"]["inherited_prefix_hex"] == old.hex()
    assert row["rx_context"]["last_receive_timestamp_ns"] > feedback.deadline_ns
    assert device._sequences == sequences and commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("source", ["command", "prequery", "parameter"])
def test_every_receiver_preserves_partial_frame_for_fault_history(setup_backend, monkeypatch, source):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=0)
    fake.pending.extend(old)
    original_read = fake.read

    def read_prefix(size):
        data = original_read(min(size, 1))
        clock.now += 4_000_000
        return data

    monkeypatch.setattr(fake, "read", read_prefix)
    deadline = clock.now + 5_000_000
    with pytest.raises(TimeoutError):
        if source == "command":
            device._receive_gripper_feedback(deadline, 1, packet_hex=encode_enable(7).hex())
        elif source == "prequery":
            device._prepare_query(deadline)
        else:
            device._capture_parameter_rx(10, deadline)
    monkeypatch.setattr(fake, "read", original_read)
    assert bytes(device._rx_pending) == b"\xaa" and bytes(fake.pending) == old[1:]
    stop = device.emergency_stop("interrupted_receive")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    row = item["intermediate_feedback"][0]
    assert row["historical_only"] and row["status_code"] == 0
    assert (
        row["rx_context"]["source"]
        == {
            "command": "command_feedback",
            "prequery": "pre_query_unattributed_rx",
            "parameter": "parameter_write_unattributed_rx",
        }[source]
    )
    assert not item["feedback"]["rx_context"]["inherited"]
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize(
    "invalid",
    [
        bridge_packet(6, status=1),
        bridge_packet(7, payload=b"\x07\x00\x33\x0a" + struct.pack("<I", 4)),
        bridge_packet(7, payload=b"\x07\x00\x55\x0a" + struct.pack("<I", 4)),
        bridge_packet(7, status=8),
        b"\x00" * 16,
    ],
)
def test_inherited_invalid_frame_never_scans_ahead_to_disabled(setup_backend, monkeypatch, invalid):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    timed_out_query_prefix(device, fake, clock, monkeypatch, invalid)
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    stop = device.emergency_stop("bad_old_frame")
    assert stop["status"] == "unknown" and not stop.get("disabled_feedback_observed", False)
    assert not stop["per_joint_results"][0]["intermediate_feedback"]
    assert bytes(device._rx_pending) == invalid
    assert bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("change", ["mapping", "calibration", "scope"])
def test_inherited_frame_rejects_changed_internal_decoding_context(setup_backend, monkeypatch, change):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    timed_out_query_prefix(device, fake, clock, monkeypatch, bridge_packet(7, status=1))
    if change == "mapping":
        key = ("gripper", 21)
        device._mapping[key] = replace(device._mapping[key], value=9.0)
    elif change == "calibration":
        device.calibration = replace(device.calibration, maximum_deg=29)
    else:
        device.joint_specs = (replace(device.joint_specs[0], model="4340P"),)
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    stop = device.emergency_stop("changed_internal_context")
    if change == "scope":
        assert stop["status"] == "not_attempted" and "scope changed" in stop["error"]
        assert commands(fake) == []
        return
    assert stop["status"] == "unknown" and "decoding context changed" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert commands(fake) == [encode_disable(7)]


def test_inherited_frame_counts_toward_visible_fault_intermediate_limit(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=1)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    device.config = replace(device.config, prequery_byte_limit=16)
    fake.pending.extend(old)
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    stop = device.emergency_stop("inherited_limit")
    assert stop["status"] == "unknown" and "bounded byte limit" in stop["error"]
    rows = stop["per_joint_results"][0]["intermediate_feedback"]
    assert len(rows) == 1 and rows[0]["historical_only"]
    assert not stop.get("disabled_feedback_observed", False)
    assert bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


def test_fault_position_validation_cannot_complete_after_receive_deadline(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    original_check = device._check_gripper_position

    def slow_check(position):
        original_check(position)
        clock.now += 30_000_000

    monkeypatch.setattr(device, "_check_gripper_position", slow_check)
    stop = device.emergency_stop("slow_validation")
    assert stop["status"] == "unknown" and "deadline expired" in stop["error"]
    # Raw status0 was observed in time; its late validation still cannot confirm exit.
    assert stop["disabled_feedback_observed"] and stop["error"] is not None
    assert bytes(device._rx_pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


def test_normal_position_validation_cannot_submit_after_receive_deadline(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    original_check = device._check_gripper_position

    def slow_check(position):
        original_check(position)
        clock.now += 30_000_000

    monkeypatch.setattr(device, "_check_gripper_position", slow_check)
    receipt = device.send_gripper_position(0.05, feedback_snapshot=feedback)
    assert not receipt.all_submitted and "deadline expired" in receipt.get("gripper").error
    assert receipt.get("gripper").submitted_deg is None
    assert bytes(device._rx_pending) == bridge_packet(7, status=1)


def test_historical_position_validation_does_not_consume_after_fault_deadline(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=0)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    fake.feedback_transform = lambda _packet, _motor_id: bridge_packet(7, status=fake.status)
    original_check = device._check_gripper_position

    def slow_check(position):
        original_check(position)
        clock.now += 30_000_000

    monkeypatch.setattr(device, "_check_gripper_position", slow_check)
    stop = device.emergency_stop("slow_history_validation")
    assert stop["status"] == "unknown" and "deadline expired" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert not stop["per_joint_results"][0]["intermediate_feedback"]
    assert bytes(device._rx_pending) == old and bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("kind", ["mode", "register"])
def test_backend_register_consumer_rolls_back_when_final_query_checkpoint_expires(
    setup_backend, monkeypatch, kind
):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    previous_modes = dict(device._modes)
    original_finish = device._finish_rx_frame

    def late_finish():
        original_finish()
        clock.now += 5_000_000

    monkeypatch.setattr(device, "_finish_rx_frame", late_finish)
    with pytest.raises(TimeoutError, match="deadline expired"):
        if kind == "mode":
            device._read_modes(clock.now + 5_000_000)
        else:
            device._read_gripper_register(10, clock.now + 5_000_000)
    assert device._modes == previous_modes
    assert device.mode_evidence == ()
    assert bytes(device._rx_pending) == bridge_packet(
        7, payload=b"\x07\x00\x33\x0a" + struct.pack("<I", fake.mode)
    )
    assert device._rx_context["session_signature"] == device._rx_session_signature()


def test_unresolved_policy_still_refuses_when_acceptance_gate_mocked(setup_backend, monkeypatch):
    device, _, _, factory = setup_backend
    device.config = replace(device.config, fault_action=None)
    monkeypatch.setattr(backend, "require_b601_control_acceptance", lambda: None)
    with pytest.raises(RuntimeError, match="policy is unresolved"):
        device.connect()
    stop = device.emergency_stop("test")
    assert stop["action"] == "unresolved" and stop["status"] == "not_attempted"
    factory.assert_not_called()


def test_low_level_write_refuses_joined_frames_before_io(setup_backend):
    device, fake, clock, factory = setup_backend
    with pytest.raises(ValueError, match="exactly one 30-byte frame"):
        device._write_once(encode_enable(7) + encode_disable(7), clock.now + 5_000_000)
    factory.assert_not_called()
    assert not fake.writes and not device.take_io_evidence()["events"]


def test_write_timeout_is_reused_and_never_exceeds_remaining_budget(setup_backend, monkeypatch):
    device, fake, clock, _ = setup_backend
    device.connect_observe_only()
    assigned = []
    stored = [fake.write_timeout]

    def assign(_serial, value):
        assigned.append(value)
        stored[0] = value

    monkeypatch.setattr(
        ControlSerial, "write_timeout", property(lambda _serial: stored[0], assign), raising=False
    )
    packet = encode_mode_query(7)
    device._write_once(packet, clock.now + 19_950_000)
    device._write_once(packet, clock.now + 19_750_000)
    assert assigned == [0.019]
    device._write_once(packet, clock.now + 4_900_000)
    assert assigned == [0.019, 0.004]
    writes = len(fake.writes)
    with pytest.raises(TimeoutError, match="one millisecond"):
        device._write_once(packet, clock.now + 999_999)
    assert len(fake.writes) == writes


def test_observe_session_cannot_gain_motion_by_mocking_gate(setup_backend, monkeypatch):
    device, fake, clock, _ = setup_backend
    device.connect_observe_only()
    assert device.read_mapping().is_valid
    feedback = snapshot(device, clock)
    monkeypatch.setattr(backend, "require_b601_control_acceptance", lambda: None)
    with pytest.raises(RuntimeError, match="Observe-only"):
        device.send_gripper_position(0.5, feedback_snapshot=feedback)
    assert device.emergency_stop("test")["status"] == "not_attempted"
    device.disconnect()
    assert not commands(fake)


def test_gripper_runtime_contract_uses_one_serial_and_no_implicit_joint(setup_backend, monkeypatch):
    device, fake, clock, factory = connected(setup_backend, monkeypatch)
    assert factory.call_count == 1
    assert not commands(fake)  # connect is read-only, including mode verification
    feedback = snapshot(device, clock)
    receipt = device.send_gripper_position(0.5, feedback_snapshot=feedback)
    assert receipt.all_submitted and len(receipt.results) == 1
    result = receipt.get("gripper")
    packet = commands(fake)[0]
    assert int.from_bytes(packet[13:17], "little") == 0x307
    assert result.submitted_deg == math.degrees(struct.unpack("<f", packet[21:25])[0])
    assert result.hardware_ack == "unknown"
    assert feedback.get("gripper").time_source == "host_serial_read_return_perf_counter_ns"
    assert {item[1] for item in fake.calls} == {device._owner_thread.ident}
    device.disconnect()
    assert commands(fake)[-1] == encode_disable(7)
    assert not device.has_resources


def test_integer_speed_is_reported_and_submitted_without_degree_round_trip(setup_backend, tmp_path):
    original, fake, clock, factory = setup_backend
    configured = replace(profile(), velocity_deg_s=None, velocity_wire_code=53)
    device = backend.B601ControlBackend(
        "FAKE_CONTROL",
        readonly.DEFAULT_B601_JOINTS[-1:],
        replace(original.config, profiles=(configured,)),
        lock_root=tmp_path,
    )
    factory.assert_not_called()
    settings = device.command_settings
    assert settings["requested"]["velocity_wire_code"] == 53
    assert settings["requested"]["motor_velocity_deg_s"] == math.degrees(0.53)
    assert settings["applied"]["velocity_wire_code"] == settings["encoded"]["velocity_wire_code"] == 53
    assert settings["applied"]["motor_velocity_deg_s"] == math.degrees(0.53)
    try:
        device.connect()
        device.start_gripper(snapshot(device, clock), lambda: None)
        assert device.startup_receipt["command_settings"]["encoded"]["velocity_wire_code"] == 53
        receipt = device.send_gripper_position(0.5, feedback_snapshot=snapshot(device, clock))
        assert receipt.all_submitted
        packets = [packet for packet in commands(fake) if int.from_bytes(packet[13:17], "little") == 0x307]
        assert packets and all(struct.unpack("<H", packet[25:27])[0] == 53 for packet in packets)
    finally:
        device.disconnect()
    assert not device.has_resources


def test_fully_open_endpoint_can_be_submitted_repeatedly_and_then_closed(setup_backend, tmp_path):
    """Replay the -270 degree limit with a fake bus and preserve the actual wire receipt."""
    original, fake, clock, _ = setup_backend
    calibration_path = Path(original.config.calibration_file)
    calibration_path.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -270, "range_max": 0}}
        ),
        encoding="utf-8",
    )
    configured = replace(profile(), minimum_deg=-270, maximum_deg=0, max_relative_deg=100)
    fake.mapping = {21: 12.5, 22: 30.0, 23: 10.0}
    position_uint = round((math.radians(-250.84) + 12.5) * 65535 / 25)
    fake.feedback_transform = lambda _packet, motor_id: bridge_packet(
        motor_id, status=fake.status, position_uint=position_uint
    )
    device = backend.B601ControlBackend(
        "FAKE_CONTROL",
        readonly.DEFAULT_B601_JOINTS[-1:],
        replace(original.config, profiles=(configured,)),
        lock_root=tmp_path,
    )
    try:
        device.connect()
        device.start_gripper(snapshot(device, clock), lambda: None)
        for requested in (-269.0331191043383, -270.0, -270.0, -269.0):
            receipt = device.send_gripper_position(requested, feedback_snapshot=snapshot(device, clock))
            assert receipt.all_submitted
            result = receipt.get("gripper")
            assert result.candidate_deg == result.clipped_deg == requested
            assert -270 <= result.submitted_deg <= 0
            packet = commands(fake)[-1]
            assert int.from_bytes(packet[13:17], "little") == 0x307
            assert result.submitted_deg == math.degrees(struct.unpack("<f", packet[21:25])[0])
            assert result.hardware_ack == "unknown"
        assert device.failure is None
    finally:
        device.disconnect()
    assert not device.has_resources


def test_degrees_speed_receipt_preserves_requested_and_quantized_values(setup_backend):
    device, _, _, factory = setup_backend
    settings = device.command_settings
    assert settings["requested"]["velocity_wire_code"] is None
    assert settings["requested"]["motor_velocity_deg_s"] == 10
    assert settings["applied"]["velocity_wire_code"] == settings["encoded"]["velocity_wire_code"] == 17
    assert settings["applied"]["motor_velocity_deg_s"] == math.degrees(0.17)
    factory.assert_not_called()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0"])
def test_invalid_goal_never_writes(setup_backend, monkeypatch, bad):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    with pytest.raises(ValueError):
        device.send_gripper_position(bad, feedback_snapshot=snapshot(device, clock))
    assert not commands(fake)


def test_foreign_and_expired_snapshots_never_write(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    with pytest.raises(ValueError, match="latest valid"):
        device.send_gripper_position(0.5, feedback_snapshot=replace(feedback))
    clock.now += 30_000_000
    with pytest.raises(ValueError, match="Fresh"):
        device.send_gripper_position(0.5, feedback_snapshot=feedback)
    assert not commands(fake)


@pytest.mark.parametrize("kind", ["partial", "late", "exception"])
def test_uncertain_submission_latches_motion_and_retains_receipt(setup_backend, monkeypatch, kind):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    if kind == "partial":
        fake.fail_motor = 7
    elif kind == "late":
        fake.write_delay_ns = 10_000_000
    else:
        monkeypatch.setattr(fake, "write", Mock(side_effect=OSError("transport failure")))
    receipt = device.send_gripper_position(0.5, feedback_snapshot=feedback)
    assert not receipt.all_submitted
    assert receipt.get("gripper").status == "unknown"
    assert receipt.get("gripper").submitted_deg is None
    assert device.last_receipt is receipt
    with pytest.raises(RuntimeError, match="fault is latched"):
        device.send_gripper_position(0.5, feedback_snapshot=feedback)
    stop = device.emergency_stop("failed send")
    assert stop["status"] == "unknown" and not stop["hardware_stop_confirmed"]


def test_interrupt_preserves_unknown_receipt_and_cleanup(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    fake.interrupt_motor = 7
    with pytest.raises(KeyboardInterrupt):
        device.send_gripper_position(0.5, feedback_snapshot=snapshot(device, clock))
    assert device.last_receipt.get("gripper").status == "unknown"
    with pytest.raises(KeyboardInterrupt):
        device.emergency_stop("interrupt")
    assert device._fault_receipt["status"] == "unknown"
    assert not device._fault_receipt["hardware_stop_confirmed"]
    count = len(commands(fake))
    with pytest.raises(RuntimeError, match="unconfirmed"):
        device.disconnect()
    assert len(commands(fake)) == count and not device.has_resources


def test_fault_once_never_claims_ack_or_physical_stop(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    first = device.emergency_stop("overforce")
    assert first["status"] == "submitted" and first["hardware_ack"] == "unknown"
    assert first["hardware_stop_confirmed"] is False
    first["status"] = "changed by caller"
    assert device.emergency_stop("second reason")["status"] == "submitted"
    device.disconnect()
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("fragment_size", [999, 3, 1])
def test_fault_waits_for_disabled_after_valid_enabled_feedback_without_resending(
    setup_backend, monkeypatch, fragment_size
):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    fake.fragment_size = fragment_size
    prior = bridge_packet(7, status=1)
    fake.pending.extend(prior)
    started = clock.now
    stop = device.emergency_stop("pending_query")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    assert stop["hardware_ack"] == "unknown" and not stop["hardware_stop_confirmed"]
    assert item["feedback"]["status_code"] == 0
    assert [row["packet_hex"] for row in item["intermediate_feedback"]] == [prior.hex()]
    assert item["intermediate_feedback_byte_limit"] == device.config.prequery_byte_limit
    assert item["receive_budget_s"] == device.config.feedback_max_age_s
    assert item["receive_deadline_ns"] == started + int(device.config.feedback_max_age_s * 1e9)
    assert stop["send_end_ns"] <= item["receive_deadline_ns"]
    rx = [event for event in device.take_io_evidence()["events"] if event["kind"] == "command_rx"]
    assert bytes.fromhex("".join(event["data_hex"] for event in rx)) == prior + bridge_packet(7, status=0)
    assert device.emergency_stop("second_stop") == stop
    device.disconnect()
    assert commands(fake) == [encode_disable(7)]


def test_fault_enabled_only_remains_unknown_at_fixed_receive_deadline(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    fake.auto_command_feedback = False
    fake.pending.extend(bridge_packet(7, status=1))
    started = clock.now
    stop = device.emergency_stop("missing_disabled")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "unknown" and not stop.get("disabled_feedback_observed", False)
    assert item["feedback"]["status_code"] == 1
    assert len(item["intermediate_feedback"]) == 1
    assert "TimeoutError" in stop["error"]
    assert item["receive_deadline_ns"] == started + int(device.config.feedback_max_age_s * 1e9)
    assert clock.now <= item["receive_deadline_ns"]
    assert commands(fake) == [encode_disable(7)]


def test_fault_send_and_receive_keep_independent_budgets(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    fake.write_delay_ns = 4_000_000
    fake.pending.extend(bridge_packet(7, status=1) * 2)
    started = clock.now
    stop = device.emergency_stop("pending_query_near_send_deadline")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    assert item["receive_deadline_ns"] == (
        started + fake.write_delay_ns + int(device.config.feedback_max_age_s * 1e9)
    )
    assert len(item["intermediate_feedback"]) == 2
    assert item["receive_budget_s"] == device.config.feedback_max_age_s
    assert commands(fake) == [encode_disable(7)]


def test_fault_limits_intermediate_frames_even_if_disabled_follows(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    device.config = replace(device.config, prequery_byte_limit=32)
    fake.pending.extend(bridge_packet(7, status=1) * 3)
    stop = device.emergency_stop("excess_enabled_feedback")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "unknown" and "bounded byte limit" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert len(item["intermediate_feedback"]) == 2
    assert item["feedback"]["status_code"] == 1
    assert bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize(
    ("invalid", "reason"),
    [
        (bridge_packet(6, status=1), "framing/ID"),
        (bridge_packet(7, payload=b"\x07\x00\x33\x0a" + struct.pack("<I", 4)), "not feedback"),
        (bridge_packet(7, payload=b"\x07\x00\x55\x0a" + struct.pack("<I", 4)), "not feedback"),
        (bridge_packet(7, status=8), "Unexpected gripper drive status"),
        (b"\x00" * 16, "framing/ID"),
    ],
)
def test_fault_does_not_skip_invalid_feedback_to_find_disabled(setup_backend, monkeypatch, invalid, reason):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    fake.pending.extend(invalid)
    stop = device.emergency_stop("invalid_preceding_feedback")
    assert stop["status"] == "unknown" and reason in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert not stop["per_joint_results"][0]["intermediate_feedback"]
    assert bytes(fake.pending) == bridge_packet(7, status=0)
    assert commands(fake) == [encode_disable(7)]


def test_fault_does_not_accept_disabled_feedback_after_deadline(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    fake.late_read = True
    stop = device.emergency_stop("late_disabled")
    assert stop["status"] == "unknown" and "TimeoutError" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    assert device._last_command_feedback is None
    assert commands(fake) == [encode_disable(7)]


def test_ordinary_position_does_not_skip_disabled_to_reach_enabled(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    fake.pending.extend(bridge_packet(7, status=0))
    receipt = device.send_gripper_position(0.05, feedback_snapshot=feedback)
    assert not receipt.all_submitted and "Unexpected gripper drive status" in receipt.get("gripper").error
    assert receipt.get("gripper").submitted_deg is None
    assert bytes(fake.pending) == bridge_packet(7, status=1)


def test_fault_read_timeout_reuses_short_value_and_checks_slow_setter(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    assigned = []
    stored = [0.001]

    def assign(_serial, value):
        assigned.append(value)
        stored[0] = value
        clock.now += 10_000_000

    monkeypatch.setattr(ControlSerial, "timeout", property(lambda _serial: stored[0], assign), raising=False)
    fake.pending.extend(bridge_packet(7, status=1))
    stop = device.emergency_stop("short_timeout")
    assert stop["status"] == "submitted" and assigned == []


def test_fault_read_stops_before_io_when_timeout_setter_exhausts_deadline(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    assigned = []
    stored = [1.0]

    def assign(_serial, value):
        assigned.append(value)
        stored[0] = value
        clock.now += 30_000_000

    monkeypatch.setattr(ControlSerial, "timeout", property(lambda _serial: stored[0], assign), raising=False)
    before = sum(name == "read" for name, _ in fake.calls)
    stop = device.emergency_stop("slow_timeout_setter")
    assert stop["status"] == "unknown" and "TimeoutError" in stop["error"]
    assert assigned == [0.002]
    assert sum(name == "read" for name, _ in fake.calls) == before
    assert commands(fake) == [encode_disable(7)]


def test_close_failure_retains_lease_without_replaying_stop(setup_backend, monkeypatch, tmp_path):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    fake.close_fails = True
    with pytest.raises(OSError, match="close failed"):
        device.disconnect()
    assert device.has_resources
    contender = B601BusLease("damiao", "FAKE_CONTROL", lock_root=tmp_path)
    with pytest.raises(RuntimeError, match="already has an owner"):
        contender.acquire()
    assert commands(fake) == [encode_disable(7)]
    fake.close_fails = False
    device.disconnect()
    assert commands(fake) == [encode_disable(7)]
    contender.acquire()
    contender.release()


def test_wrong_owner_cannot_read_send_stop_or_close(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    count = len(fake.calls)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for operation in (
            lambda: snapshot(device, clock),
            lambda: device.send_gripper_position(0.5, feedback_snapshot=feedback),
            lambda: device.emergency_stop("wrong owner"),
            device.disconnect,
            device.connect,
        ):
            with pytest.raises(RuntimeError, match="owner"):
                pool.submit(operation).result()
    assert len(fake.calls) == count


def test_partial_snapshot_only_queries_requested_joint(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    count = len(fake.writes)
    feedback = snapshot(device, clock)
    assert feedback.is_valid and len(fake.writes) == count + 1
    assert fake.writes[-1] == readonly.encode_readonly_query(7)
    assert feedback.get("gripper").receive_timestamp_ns <= feedback.completed_ns <= feedback.deadline_ns
    clock.now += 10_000_000
    newer = snapshot(device, clock)
    assert newer.get("gripper").receive_sequence > feedback.get("gripper").receive_sequence


def test_protocol_golden_packets_and_quantized_submission():
    encoded = encode_position(7, 0, 0, profile(), (8, 14, 9))
    assert encoded.packet.hex() == "55aa1e03010000000a000000000703000000080000000000001100bc0200"
    assert encode_mode_query(7).hex() == "55aa1e03010000000a00000000ff070000000800000700330a0000000000"
    assert encode_disable(7).hex() == "55aa1e03010000000a000000000700000000080000fffffffffffffffd00"
    mit = encode_position(1, 0, 0, profile("shoulder_pan", "mit"), (8, 14, 9))
    assert mit.packet[21:29].hex() == "7fff7ff0410f57ff"
    assert mit.wire_position_deg < 0  # midpoint is quantized, never labelled as exact zero


def test_startup_encoders_match_vendor_single_motor_commands_without_zero_or_store():
    assert encode_enable(7).hex() == "55aa1e03010000000a000000000700000000080000fffffffffffffffc00"
    assert (
        encode_mode_write(7, "force_pos").hex()
        == "55aa1e03010000000a00000000ff070000000800000700550a0400000000"
    )
    for mode, code in (("mit", 1), ("pos_vel", 2), ("force_pos", 4)):
        packet = encode_mode_write(7, mode)
        assert len(packet) == 30
        assert packet[21:25] == bytes((7, 0, 0x55, 10))
        assert struct.unpack("<I", packet[25:29])[0] == code


@pytest.mark.parametrize("mode", ["velocity", "FORCE_POS", 4, True, None])
def test_startup_encoder_refuses_unreviewed_or_ambiguous_mode(mode):
    with pytest.raises(ValueError):
        encode_mode_write(7, mode)


@pytest.mark.parametrize("kwargs", [{"kd": -1}, {"kp": None}, {"kp": True}])
def test_mit_rejects_negative_or_missing_gains(kwargs):
    with pytest.raises(ValueError):
        replace(profile("shoulder_pan", "mit"), **kwargs).validate()


def test_position_endpoint_quantization_stays_inside_configured_limits():
    p = replace(profile(), minimum_deg=0.1, maximum_deg=1)
    encoded = encode_position(7, 0.1, 0.1, p, (8, 14, 9))
    assert encoded.clipped_deg == 0.1
    assert p.minimum_deg <= encoded.wire_position_deg <= p.maximum_deg
    assert encoded.wire_position_deg == math.degrees(struct.unpack("<f", encoded.packet[21:25])[0])


@pytest.mark.parametrize(
    "kwargs", [{"velocity_deg_s": 0.001}, {"torque_ratio": 0.000001}, {"velocity_deg_s": 900}]
)
def test_position_parameters_cannot_silently_saturate_or_round_to_zero(kwargs):
    with pytest.raises(ValueError):
        encode_position(7, 0, 0, replace(profile(), **kwargs), (8, 14, 9))


@pytest.mark.parametrize("traffic", ["ordinary", "fragment", "fault", "wrong_id", "overflow"])
def test_prequery_rx_is_audited_but_never_reused_as_current_feedback(setup_backend, monkeypatch, traffic):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    device.take_io_evidence()
    packet = bridge_packet(7, status=1, position_uint=32000)
    if traffic == "fragment":
        packet = packet[:8]
    elif traffic == "fault":
        packet = bridge_packet(7, status=9)
    elif traffic == "wrong_id":
        packet = bridge_packet(1, status=1)
    elif traffic == "overflow":
        packet *= 17
    fake.pending.extend(packet)
    resets = sum(op == "reset_input_buffer" for op, _ in fake.calls)
    feedback = snapshot(device, clock)
    events = device.take_io_evidence()["events"]
    assert events[0]["kind"] == "pre_query_unattributed_rx"
    assert packet.startswith(bytes.fromhex(events[0]["data_hex"]))
    assert sum(op == "reset_input_buffer" for op, _ in fake.calls) == resets
    if traffic == "ordinary":
        assert feedback.is_valid
        assert abs(feedback.get("gripper").position_deg) < 0.1
        assert any(e["kind"] == "query_attempt" for e in events)
    else:
        assert not feedback.is_valid and device.failure is not None
        assert not any(e["kind"] == "query_attempt" for e in events)


def test_audit_overflow_latches_but_does_not_suppress_explicit_fault_attempt(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    device.take_io_evidence()
    device.config = replace(device.config, io_event_capacity=1)
    assert not snapshot(device, clock).is_valid
    assert device.failure is not None
    stop = device.emergency_stop("audit overflow")
    assert stop["status"] == "submitted" and not stop["hardware_stop_confirmed"]
    assert commands(fake) == [encode_disable(7)]
    evidence = device.take_io_evidence()
    assert not evidence["complete"] and len(evidence["events"]) == 1
    assert not device.take_io_evidence()["complete"]


@pytest.mark.parametrize("failure", [False, True])
def test_selected_fault_disposition_keeps_each_submission_result(
    setup_backend, monkeypatch, tmp_path, failure
):
    _, fake, _, _ = setup_backend
    monkeypatch.setattr(backend, "require_b601_control_acceptance", lambda: None)
    specs = readonly.DEFAULT_B601_JOINTS[:3]
    names = tuple(s.name for s in specs)
    config = backend.B601ControlConfig(
        tuple(profile(name, "pos_vel") for name in names),
        0.1,
        0.005,
        0.005,
        0.02,
        "disable_selected",
        names,
    )
    device = backend.B601ControlBackend("FAKE_CONTROL", specs, config, lock_root=tmp_path)
    fake.mode = 2
    fake.status = 1
    fake.auto_command_feedback = False
    device.connect()
    fake.fail_motor = 2 if failure else None
    stop = device.emergency_stop("simulated fault")
    assert [item["status"] for item in stop["per_joint_results"]] == (
        ["submitted", "unknown", "not_attempted"] if failure else ["submitted"] * 3
    )
    assert commands(fake) == [encode_disable(i) for i in (range(1, 3) if failure else range(1, 4))]
    assert stop["hardware_ack"] == "unknown" and not stop["hardware_stop_confirmed"]
    count = len(fake.writes)
    assert device.emergency_stop("second fault") == stop
    if failure:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            device.disconnect()
    else:
        device.disconnect()
    assert len(fake.writes) == count and not device.has_resources


def test_fault_interrupt_and_close_error_are_both_retained(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    fake.interrupt_motor = 7
    fake.close_fails = True
    with pytest.raises(BaseExceptionGroup) as caught:
        device.disconnect()
    assert isinstance(caught.value.exceptions[0], KeyboardInterrupt)
    assert isinstance(caught.value.exceptions[1], OSError)
    assert device.has_resources and device.fault_receipt["status"] == "unknown"
    fake.interrupt_motor = None
    fake.close_fails = False
    with pytest.raises(RuntimeError, match="unconfirmed"):
        device.disconnect()
    assert not device.has_resources and commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("scope", ["arm", "wrong_id", "wrong_feedback", "mit"])
def test_only_exact_gripper_scope_can_open_without_arm_acceptance(setup_backend, tmp_path, scope):
    original, fake, _, factory = setup_backend
    specs = readonly.DEFAULT_B601_JOINTS[:1] if scope == "arm" else readonly.DEFAULT_B601_JOINTS[-1:]
    if scope == "wrong_id":
        specs = (replace(specs[0], motor_id=6),)
    elif scope == "wrong_feedback":
        specs = (replace(specs[0], feedback_id=0x16),)
    selected = profile(
        specs[0].name, "pos_vel" if scope == "arm" else "mit" if scope == "mit" else "force_pos"
    )
    config = replace(
        original.config,
        profiles=(selected,),
        fault_action="disable_selected" if scope == "arm" else "disable_gripper",
        fault_joint_names=(specs[0].name,) if scope == "arm" else (),
    )
    device = backend.B601ControlBackend("FAKE_CONTROL", specs, config, lock_root=tmp_path)
    for operation in (device.preflight_control, device.open, device.connect):
        with pytest.raises(RuntimeError, match="acceptance is incomplete"):
            operation()
    factory.assert_not_called()
    assert not fake.writes


def test_scope_cannot_change_after_connect(setup_backend):
    device, fake, clock, _ = setup_backend
    device.connect()
    disabled = snapshot(device, clock)
    device.config = replace(device.config, profiles=(replace(profile(), maximum_deg=21),))
    with pytest.raises(ValueError, match="scope changed"):
        device.start_gripper(disabled, lambda: None)
    assert not commands(fake)


def test_disabled_status_evidence_survives_a_cleanup_position_error(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    monkeypatch.setattr(device, "_check_gripper_position", Mock(side_effect=RuntimeError("out of bounds")))
    stop = device.emergency_stop("test")
    assert stop["status"] == "unknown" and stop["disabled_feedback_observed"]
    assert stop["error"] and not stop["hardware_stop_confirmed"]
    assert stop["hardware_ack"] == "unknown" and commands(fake) == [encode_disable(7)]


@pytest.fixture
def working_range_backend(setup_backend):
    """Distinct working and hard bounds with synthetic, adjustable feedback."""
    original, fake, clock, _ = setup_backend
    calibration_path = Path(original.config.calibration_file)
    calibration_path.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -300, "range_max": 0}}
        ),
        encoding="utf-8",
    )
    configured = replace(profile(), minimum_deg=-270, maximum_deg=0)
    device = backend.B601ControlBackend(
        "FAKE_WORKING_RANGE",
        readonly.DEFAULT_B601_JOINTS[-1:],
        replace(original.config, profiles=(configured,)),
        lock_root=calibration_path.parent,
    )
    position = {"degrees": -270.053}

    def feedback(_packet, motor_id):
        code = round((math.radians(position["degrees"]) + 8.0) / 16.0 * 65535)
        return bridge_packet(motor_id, status=fake.status, position_uint=code)

    fake.feedback_transform = feedback
    yield device, fake, clock, position
    if device.has_resources:
        with suppress(RuntimeError):
            device.disconnect()


def test_startup_holds_hard_safe_feedback_outside_working_command_range(working_range_backend):
    device, fake, clock, _ = working_range_backend
    device.connect()
    reference = snapshot(device, clock)
    measured = reference.get("gripper").position_deg
    assert -300 < measured < -270
    wire = device.start_gripper(reference, lambda: None)
    assert abs(wire - measured) < 0.00003  # Only float32 encoding, no endpoint motion.
    assert wire < -270 and device.startup_receipt["complete"]
    startup_positions = [p for p in commands(fake) if int.from_bytes(p[13:17], "little") == 0x307]
    assert len(startup_positions) == 2
    assert all(math.degrees(struct.unpack("<f", p[21:25])[0]) == wire for p in startup_positions)
    assert device._profiles["gripper"].minimum_deg == -270
    fake.writes.clear()
    receipt = device.send_gripper_position(-270, feedback_snapshot=snapshot(device, clock))
    result = receipt.get("gripper")
    assert receipt.all_submitted and result.clipped_deg == -270
    assert -270 <= result.submitted_deg < -269.9999
    assert result.submitted_deg == math.degrees(struct.unpack("<f", commands(fake)[0][21:25])[0])
    assert result.hardware_ack == "unknown"
    assert device._last_command_feedback["position_deg"] == measured
    device.disconnect()
    assert not device.has_resources


def test_startup_still_rejects_feedback_outside_calibrated_hard_limits(working_range_backend):
    device, fake, _, position = working_range_backend
    position["degrees"] = -300.053
    with pytest.raises(ValueError, match="校准限位"):
        device.connect()
    assert not commands(fake)
    assert not device._gripper_start_attempted


def test_hard_overtravel_blocks_motion_but_preserves_fresh_disabled_observation(working_range_backend):
    device, fake, clock, position = working_range_backend
    device.connect()
    device.start_gripper(snapshot(device, clock), lambda: None)
    reference = snapshot(device, clock)
    fake.writes.clear()
    position["degrees"] = -300.053
    motion = device.send_gripper_position(-270, feedback_snapshot=reference)
    assert not motion.all_submitted
    assert motion.get("gripper").submitted_deg is None
    assert "校准限位" in motion.get("gripper").error
    original_failure = device._failure
    stop = device.emergency_stop("position_hard_limit")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["error"] is None
    assert stop["disabled_feedback_observed"]
    assert stop["hardware_ack"] == "unknown" and not stop["hardware_stop_confirmed"]
    assert stop["reason"] == "position_hard_limit" and device._failure == original_failure
    disabled = item["feedback"]
    assert disabled["status_code"] == 0 and disabled["position_deg"] < -300
    assert not disabled["position_within_limits"] and "hard limits" in disabled["position_limit_error"]
    assert disabled["position_limits_deg"]["calibration"] == [-300, 0]
    assert not disabled["rx_context"]["inherited"]
    historical = item["intermediate_feedback"]
    assert len(historical) == 1 and historical[0]["historical_only"]
    assert historical[0]["status_code"] == 1 and not historical[0]["position_within_limits"]
    device.disconnect()
    assert not device.has_resources and commands(fake).count(encode_disable(7)) == 1


@pytest.mark.parametrize("historical_status", [0, 1])
def test_historical_overtravel_does_not_block_current_disabled_feedback(
    setup_backend, monkeypatch, historical_status
):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=historical_status, position_uint=0)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    fake.feedback_transform = lambda _packet, motor_id: bridge_packet(
        motor_id, status=fake.status, position_uint=0
    )
    stop = device.emergency_stop("out_of_bounds_before_disable")
    item = stop["per_joint_results"][0]
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    assert item["intermediate_feedback"][0]["historical_only"]
    assert item["intermediate_feedback"][0]["status_code"] == historical_status
    assert not item["intermediate_feedback"][0]["position_within_limits"]
    assert not item["feedback"]["rx_context"]["inherited"]
    assert not item["feedback"]["position_within_limits"]
    assert item["feedback"]["status_code"] == 0


def test_historical_out_of_bounds_disabled_alone_never_confirms_new_disable(setup_backend, monkeypatch):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    old = bridge_packet(7, status=0, position_uint=0)
    timed_out_query_prefix(device, fake, clock, monkeypatch, old)
    fake.auto_command_feedback = False
    stop = device.emergency_stop("historical_disabled_only")
    assert stop["status"] == "unknown" and "TimeoutError" in stop["error"]
    assert not stop.get("disabled_feedback_observed", False)
    history = stop["per_joint_results"][0]["intermediate_feedback"]
    assert len(history) == 1 and history[0]["historical_only"]
    assert not history[0]["position_within_limits"]
    assert commands(fake) == [encode_disable(7)]


def test_fault_records_out_of_bounds_intermediate_enabled_before_current_disabled(setup_backend, monkeypatch):
    device, fake, _, _ = connected(setup_backend, monkeypatch)
    fake.pending.extend(bridge_packet(7, status=1, position_uint=65535))
    stop = device.emergency_stop("overtravel_with_pending_enabled")
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    item = stop["per_joint_results"][0]
    intermediate = item["intermediate_feedback"]
    assert len(intermediate) == 1 and intermediate[0]["status_code"] == 1
    assert not intermediate[0]["rx_context"]["inherited"]
    assert not intermediate[0]["position_within_limits"]
    assert intermediate[0]["position_deg"] > 30
    assert item["feedback"]["status_code"] == 0 and item["feedback"]["position_within_limits"]
    assert commands(fake) == [encode_disable(7)]


@pytest.mark.parametrize("fault_attempt,expected_status", [(False, 0), (False, 1), (True, 1)])
def test_only_fault_disable_can_record_out_of_bounds_feedback_without_rejecting(
    setup_backend, monkeypatch, fault_attempt, expected_status
):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    fake.pending.extend(bridge_packet(7, status=expected_status, position_uint=0))
    with pytest.raises(ValueError, match="校准限位"):
        device._receive_gripper_feedback(clock.now + 20_000_000, expected_status, fault_attempt=fault_attempt)


def test_changed_calibration_after_connect_blocks_startup(setup_backend):
    device, fake, clock, _ = setup_backend
    device.connect()
    reference = snapshot(device, clock)
    Path(device.config.calibration_file).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        device.start_gripper(reference, lambda: None)
    assert not commands(fake) and not device._gripper_start_attempted


@pytest.mark.parametrize("change", ["replace", "delete"])
def test_changed_calibration_during_tracking_blocks_position_but_allows_disable(
    setup_backend, monkeypatch, change
):
    device, fake, clock, _ = connected(setup_backend, monkeypatch)
    feedback = snapshot(device, clock)
    path = Path(device.config.calibration_file)
    if change == "replace":
        path.write_text("{}", encoding="utf-8")
    else:
        path.unlink()
    receipt = device.send_gripper_position(0.05, feedback_snapshot=feedback)
    assert not receipt.all_submitted
    assert receipt.get("gripper").submitted_deg is None
    assert commands(fake) == []
    stop = device.emergency_stop("calibration_changed")
    assert stop["status"] == "submitted" and stop["disabled_feedback_observed"]
    assert commands(fake) == [encode_disable(7)]
