"""Bounded single-joint lifecycle tests with fake serial only; never physical acceptance."""

import json
import math
import struct
import threading
import uuid
from contextlib import suppress
from dataclasses import replace
from unittest.mock import Mock

import draccus
import pytest

from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.arm_acceptance import (
    SingleJointAcceptanceConfig,
    command_profile_digest,
    execute_single_joint_acceptance,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_protocol import (
    B601JointCommandProfile,
    encode_disable,
    encode_enable,
)
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet


@pytest.fixture
def acceptance(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(backend.time, "perf_counter_ns", clock.perf_counter_ns)

    class AcceptanceSerial(FakeSerial):
        def __init__(self, clock):
            super().__init__(clock)
            self.mapping = {21: 12.5, 22: 30.0, 23: 10.0}
            self.statuses = dict.fromkeys(range(1, 8), 0)
            self.modes = {index: 4 if index == 7 else 2 for index in range(1, 8)}
            self.timeouts = {index: 2000 if index == 7 else 0 for index in range(1, 8)}
            self.positions = dict.fromkeys(range(1, 8), 0.0)
            self.targets = self.positions.copy()
            self.ignore_timeout = False
            self.on_command = None
            self.drop_enable_feedback = False

        def feedback(self, motor_id):
            position = round((math.radians(self.positions[motor_id]) + 12.5) * 65535 / 25)
            return bridge_packet(motor_id, status=self.statuses[motor_id], position_uint=position)

        def write(self, packet):
            self._call("write")
            self.writes.append(packet)
            self.clock.now += 100_000
            can_id, payload = int.from_bytes(packet[13:17], "little"), packet[21:29]
            motor_id = payload[0] if can_id == 0x7FF else can_id & 15
            if can_id == 0x7FF:
                opcode, rid = payload[2:4]
                if opcode == 0x33:
                    value = (
                        self.modes[motor_id]
                        if rid == 10
                        else self.timeouts[motor_id]
                        if rid == 9
                        else self.mapping[rid]
                    )
                    encoded = struct.pack("<I" if rid in (9, 10) else "<f", value)
                    self.pending.extend(bridge_packet(motor_id, payload=payload[:4] + encoded))
                elif opcode == 0xCC:
                    self.pending.extend(self.feedback(motor_id))
                elif opcode == 0x55 and rid == 9:
                    assert self.statuses[motor_id] == 0
                    if not self.ignore_timeout:
                        self.timeouts[motor_id] = struct.unpack("<I", payload[4:])[0]
                else:
                    raise AssertionError("Acceptance emitted an unauthorized register write")
                return len(packet)
            if payload == b"\xff" * 7 + b"\xfc":
                self.statuses[motor_id] = 1
                self.positions[motor_id] = self.targets[motor_id]
            elif payload == b"\xff" * 7 + b"\xfd":
                self.statuses[motor_id] = 0
            else:
                assert can_id == motor_id + 0x100
                target, velocity = struct.unpack("<ff", payload)
                assert 0 < math.degrees(velocity) <= 1.00001
                self.targets[motor_id] = math.degrees(target)
                if self.statuses[motor_id] == 1:
                    self.positions[motor_id] = self.targets[motor_id]
            if self.on_command is not None:
                self.on_command(motor_id, payload)
            if self.drop_enable_feedback and payload == b"\xff" * 7 + b"\xfc":
                return len(packet)
            self.pending.extend(self.feedback(motor_id))
            return len(packet)

    fake = AcceptanceSerial(clock)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(readonly.serial, "Serial", factory)
    path = tmp_path / "all_calibration.json"
    path.write_text(
        json.dumps(
            {
                spec.name: {
                    "id": spec.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -30,
                    "range_max": 30,
                }
                for spec in readonly.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    profile = B601JointCommandProfile("shoulder_pan", "pos_vel", -20, 20, 0.5, velocity_deg_s=1)
    calibration = B601ArmCalibration.load(path, {s.name: s.motor_id for s in readonly.DEFAULT_B601_JOINTS})
    request = SingleJointAcceptanceConfig(
        str(uuid.uuid4()),
        "explicit_supported_microstep_test",
        "shoulder_pan",
        str(path),
        calibration.sha256,
        command_profile_digest(profile),
        True,
        "disable_selected_supported_arm",
        100,
        -0.5,
        1,
        2,
        5,
        2000,
        True,
    )
    config = backend.B601ControlConfig(
        (profile,),
        1.0,
        0.05,
        0.1,
        0.1,
        "disable_selected",
        ("shoulder_pan",),
        calibration_file=str(path),
        watchdog_ticks=2000,
        parameter_write_settle_s=0,
    )
    device = backend.B601ControlBackend(
        "FAKE_ACCEPTANCE", readonly.DEFAULT_B601_JOINTS, config, lock_root=tmp_path, acceptance=request
    )
    yield device, fake, clock, factory, request, path
    fake.on_command = None
    if device.has_resources:
        with suppress(Exception):
            device.disconnect()


def motion_packets(fake):
    return [packet for packet in fake.writes if int.from_bytes(packet[13:17], "little") != 0x7FF]


def test_profile_digest_survives_typed_integer_to_float_configuration_roundtrip():
    profile = B601JointCommandProfile("shoulder_pan", "pos_vel", -20, 20, 0.5, velocity_deg_s=1)
    decoded = draccus.decode(B601JointCommandProfile, json.loads(json.dumps(draccus.encode(profile))))
    assert type(profile.minimum_deg) is int and type(decoded.minimum_deg) is float
    assert type(profile.velocity_deg_s) is int and type(decoded.velocity_deg_s) is float
    assert command_profile_digest(profile) == command_profile_digest(decoded)
    with pytest.raises(ValueError):
        command_profile_digest(replace(profile, velocity_deg_s=True))


def test_single_joint_preparation_step_return_and_all_disabled_receipt(acceptance):
    device, fake, _, _, _, _ = acceptance
    device.connect()
    assert not motion_packets(fake)
    reference = device.start_acceptance(lambda: None)
    assert fake.timeouts[1] == 2000
    assert fake.statuses[1] == 1 and all(fake.statuses[index] == 0 for index in range(2, 8))
    assert not device.startup_receipt["watchdog_semantics_verified"]
    result = device.acceptance_step(reference - 0.1, lambda: None)
    assert result.all_submitted and result.get("shoulder_pan").submitted_deg == pytest.approx(reference - 0.1)
    device.acceptance_step(reference, lambda: None)
    receipt = device.emergency_stop("test_complete")
    assert receipt["all_joints_disabled_observed"] and receipt["hardware_ack"] == "unknown"
    assert all(value == 0 for value in fake.statuses.values())
    assert all((int.from_bytes(packet[13:17], "little") & 15) == 1 for packet in motion_packets(fake))
    assert all(packet[24] == 9 for packet in fake.writes if packet[23] == 0x55)
    device.disconnect()
    assert not device.has_resources


@pytest.mark.parametrize(
    "change",
    [{"supported_arm": False}, {"displacement_deg": 1}, {"velocity_deg_s": 2}, {"max_duration_s": 6}],
)
def test_oversized_or_unsupported_request_never_constructs_transport(acceptance, change):
    device, _, _, factory, request, _ = acceptance
    with pytest.raises(ValueError):
        backend.B601ControlBackend(
            "FAKE_INVALID", readonly.DEFAULT_B601_JOINTS, device.config, acceptance=replace(request, **change)
        )
    factory.assert_not_called()


def test_generic_motion_and_cross_joint_writes_remain_forbidden(acceptance):
    device, fake, _, _, _, _ = acceptance
    device.connect()
    device.start_acceptance(lambda: None)
    before = len(motion_packets(fake))
    with pytest.raises(RuntimeError, match="generic motion"):
        device.send_joint_positions_partial({"shoulder_pan": 0}, feedback_snapshot=device._latest_snapshot)
    with pytest.raises(RuntimeError, match="nonselected"):
        device._write_once(encode_enable(2), device._acceptance_started_ns + 1_000_000_000)
    assert len(motion_packets(fake)) == before


@pytest.mark.parametrize("failure", ["wrong_mode", "timeout_readback", "other_enabled", "stop_guard"])
def test_startup_failure_never_enables_selected_joint(acceptance, failure):
    device, fake, _, _, _, _ = acceptance
    if failure == "wrong_mode":
        fake.modes[1] = 1
        with pytest.raises(RuntimeError, match="mode"):
            device.connect()
    else:
        device.connect()
        if failure == "timeout_readback":
            fake.ignore_timeout = True
        elif failure == "other_enabled":
            fake.statuses[2] = 1
        guard = Mock(side_effect=RuntimeError("operator stop")) if failure == "stop_guard" else lambda: None
        with pytest.raises(RuntimeError):
            device.start_acceptance(guard)
    assert not any(packet == encode_enable(1) for packet in fake.writes)


@pytest.mark.parametrize("failure", ["overtravel", "other_drift", "duration", "bad_target"])
def test_each_step_checks_fresh_local_bounds_and_duration(acceptance, failure):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    if failure == "overtravel":
        fake.positions[1] = reference - 0.7
    elif failure == "other_drift":
        fake.positions[2] = 0.2
    elif failure == "duration":
        clock.now += 5_100_000_000
    before = len(motion_packets(fake))
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        device.acceptance_step(reference - (1 if failure == "bad_target" else 0.1), lambda: None)
    assert motion_packets(fake)[before:] == [encode_disable(1)]


def test_calibration_change_blocks_motion_but_never_blocks_selected_disable(acceptance):
    device, fake, _, _, _, path = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        device.acceptance_step(reference - 0.1, lambda: None)
    receipt = device.emergency_stop("calibration_changed")
    assert receipt["all_joints_disabled_observed"] and fake.statuses[1] == 0


def test_measured_oscillation_cannot_exceed_total_travel_even_for_constant_targets(acceptance):
    device, fake, _, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    for _ in range(2):
        fake.positions[1] = reference - 0.45
        device.acceptance_step(reference, lambda: None)
    fake.positions[1] = reference - 0.45
    with pytest.raises(RuntimeError, match="measured total-travel"):
        device.acceptance_step(reference, lambda: None)
    assert fake.statuses[1] == 0
    assert device.fault_receipt["all_joints_disabled_observed"]


def test_runner_refuses_missing_manual_stop_or_input_guard_before_open(acceptance):
    device, _, _, factory, _, _ = acceptance
    with pytest.raises(ValueError, match="stop channel"):
        execute_single_joint_acceptance(
            device, stop_event=threading.Event(), event_sink=lambda event: None, input_guard=lambda: None
        )
    factory.assert_not_called()


@pytest.mark.parametrize("operator_stop", [False, True])
def test_lost_enable_reply_is_feedback_bounded_and_stop_guard_remains_live(acceptance, operator_stop):
    device, fake, clock, _, _, _ = acceptance
    device.config = replace(device.config, feedback_max_age_s=0.02)
    device.connect()
    fake.drop_enable_feedback = True
    enabled_ns = []
    observed_guard_times = []

    def command(motor_id, payload):
        if payload == b"\xff" * 7 + b"\xfc":
            enabled_ns.append(clock.now)

    def guard():
        if enabled_ns:
            observed_guard_times.append(clock.now)
            if operator_stop:
                raise RuntimeError("operator stop during enable receive")

    fake.on_command = command
    with pytest.raises((RuntimeError, TimeoutError)):
        device.start_acceptance(guard)
    assert enabled_ns and observed_guard_times
    assert observed_guard_times[-1] - enabled_ns[0] <= 20_500_000
    if operator_stop:
        assert observed_guard_times[0] - enabled_ns[0] <= 2_000_000
    packets = motion_packets(fake)
    enable_index = packets.index(encode_enable(1))
    assert packets[enable_index + 1 :] == [encode_disable(1)]
    assert device.fault_receipt["all_joints_disabled_observed"]


def test_runner_observes_out_and_return_then_disables_on_same_owner(acceptance):
    device, fake, _, _, _, _ = acceptance
    events = []
    inputs = Mock()
    result = execute_single_joint_acceptance(
        device,
        stop_event=threading.Event(),
        event_sink=events.append,
        input_guard=inputs,
        manual_stop_established=True,
        control_hz=50,
    )
    assert result["complete"] and result["returned_command"] and result["transport_closed"]
    assert result["cleanup"]["all_joints_disabled_observed"]
    assert not result["watchdog_semantics_verified"] and not result["production_arm_authorized"]
    assert inputs.call_count > 20
    outbound = [event for event in events if event["stage"] == "outbound"]
    returning = [event for event in events if event["stage"] == "return"]
    assert outbound and returning
    positions = [
        next(item["position_deg"] for item in event["observation"]["feedback"] if item["motor_id"] == 1)
        for event in [outbound[-1], returning[-1]]
    ]
    assert positions[0] == pytest.approx(result["reference_deg"] - 0.5, abs=0.05)
    assert positions[1] == pytest.approx(result["reference_deg"], abs=0.05)
    assert all(status == 0 for status in fake.statuses.values())


@pytest.mark.parametrize("stop_source", ["operator", "force_guard"])
def test_runner_stop_and_live_input_failure_always_disable_and_close(acceptance, stop_source):
    device, fake, _, _, _, _ = acceptance
    stop = threading.Event()
    events = []

    def sink(event):
        events.append(event)
        if stop_source == "operator" and event["stage"] == "outbound":
            stop.set()

    def input_guard():
        if stop_source == "force_guard" and fake.statuses[1] == 1:
            raise RuntimeError("raw force exceeded hard limit")

    with pytest.raises(RuntimeError, match="stop|force"):
        execute_single_joint_acceptance(
            device,
            stop_event=stop,
            event_sink=sink,
            input_guard=input_guard,
            manual_stop_established=True,
            control_hz=50,
        )
    assert fake.statuses[1] == 0 and not device.has_resources
    assert events[-1]["stage"] == "result" and not events[-1]["complete"]
    assert any(event["stage"] == "disable" for event in events)


def test_stop_guard_remains_live_during_all_joint_query_receive(acceptance, monkeypatch):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    missing_since = []
    disable_times = []
    original = fake.write

    def write(packet):
        if packet == encode_disable(1):
            disable_times.append(clock.now)
        if (
            int.from_bytes(packet[13:17], "little") == 0x7FF
            and packet[21:25] == bytes((3, 0, 0xCC, 0))
            and fake.statuses[1] == 1
        ):
            fake.writes.append(packet)
            clock.now += 100_000
            missing_since.append(clock.now)
            return len(packet)
        return original(packet)

    def guard():
        if missing_since:
            raise RuntimeError("operator stop during all-joint feedback")

    monkeypatch.setattr(fake, "write", write)
    before = len(motion_packets(fake))
    with pytest.raises(RuntimeError, match="operator stop"):
        device.acceptance_step(reference - 0.1, guard)
    assert missing_since and disable_times[0] - missing_since[0] < 2_000_000
    assert motion_packets(fake)[before:] == [encode_disable(1)]
    assert not device.fault_receipt.get("all_joints_disabled_observed", False)
    assert device.fault_receipt["status"] == "unknown"


@pytest.mark.parametrize("failed_stage", ["connected_disabled", "startup", "outbound", "disable", "result"])
def test_evidence_sink_failure_cannot_skip_disable_or_close(acceptance, failed_stage):
    device, fake, _, _, _, _ = acceptance

    def sink(event):
        if event["stage"] == failed_stage:
            raise OSError(f"injected sink failure at {failed_stage}")

    with pytest.raises((OSError, BaseExceptionGroup), match="injected sink"):
        execute_single_joint_acceptance(
            device,
            stop_event=threading.Event(),
            event_sink=sink,
            input_guard=lambda: None,
            manual_stop_established=True,
            control_hz=50,
        )
    assert all(status == 0 for status in fake.statuses.values())
    assert not device.has_resources
    assert sum(packet == encode_disable(1) for packet in motion_packets(fake)) <= 1
    assert all((int.from_bytes(packet[13:17], "little") & 15) == 1 for packet in motion_packets(fake))


@pytest.mark.parametrize("late_reply_order", ["before_disable", "after_disable"])
@pytest.mark.parametrize(
    "reply_variant", ["exact_recorded", "unknown_id", "malformed", "drive_fault", "duplicate"]
)
def test_zero_prefix_pending_nonselected_query_is_accounted_during_fault_disable(
    acceptance, monkeypatch, late_reply_order, reply_variant
):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    original_timeout, original_write = device._bounded_serial_timeout, fake.write
    pending_reply = []
    injected = []

    def bounded_timeout(name, cap_s, deadline_ns):
        if (
            name == "timeout"
            and not injected
            and len(motion_packets(fake)) >= 4
            and bytes(fake.pending[:7]) == b"\xaa\x11\x08\x14\x00\x00\x00"
        ):
            injected.append(True)
            recorded = bytes.fromhex("aa110814000000047ff87ff803201e55")
            if reply_variant == "unknown_id":
                recorded = recorded[:3] + struct.pack("<I", 0x15) + bytes((5,)) + recorded[8:]
            elif reply_variant == "malformed":
                recorded = recorded[:-1] + b"\x54"
            elif reply_variant == "drive_fault":
                recorded = recorded[:7] + b"\x84" + recorded[8:]
            fake.pending[:16] = recorded
            if reply_variant == "duplicate":
                fake.pending[:0] = recorded
            if late_reply_order == "after_disable":
                count = 32 if reply_variant == "duplicate" else 16
                pending_reply.append(bytes(fake.pending[:count]))
                del fake.pending[:count]
            clock.now = deadline_ns - 500_000
        return original_timeout(name, cap_s, deadline_ns)

    def write(packet):
        result = original_write(packet)
        if packet == encode_disable(1) and pending_reply:
            fake.pending.extend(pending_reply.pop())
        return result

    monkeypatch.setattr(device, "_bounded_serial_timeout", bounded_timeout)
    monkeypatch.setattr(fake, "write", write)
    with pytest.raises((TimeoutError, RuntimeError), match="less than one millisecond"):
        device.acceptance_step(reference - 0.1, lambda: None)
    assert injected and fake.statuses[1] == 0
    receipt = device.fault_receipt
    if reply_variant != "exact_recorded":
        assert receipt["status"] == "unknown" and receipt["error"]
        assert not receipt.get("all_joints_disabled_observed", False)
        disable_index = fake.writes.index(encode_disable(1))
        assert not any(
            int.from_bytes(packet[13:17], "little") == 0x7FF for packet in fake.writes[disable_index + 1 :]
        )
        return
    assert receipt["all_joints_disabled_observed"] and receipt["error"] is None
    historical = receipt["per_joint_results"][0]["intermediate_feedback"]
    assert len(historical) == 1 and historical[0]["historical_only"]
    assert historical[0]["outstanding_query"]["motor_id"] == 4
    assert historical[0]["packet_hex"].startswith("aa11081400000004")
    assert "less than one millisecond" in device.failure


@pytest.mark.parametrize("query_motor_id", [1, 4])
@pytest.mark.parametrize("prefix_size", [1, 15])
def test_captured_query_prefix_keeps_exact_owner_through_selected_disable(
    acceptance, monkeypatch, query_motor_id, prefix_size
):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    original_timeout, original_read = device._bounded_serial_timeout, fake.read
    latest_deadline, injected = [], []

    def timeout(name, cap_s, deadline_ns):
        if name == "timeout":
            latest_deadline[:] = [deadline_ns]
        return original_timeout(name, cap_s, deadline_ns)

    def read(size):
        if (
            not injected
            and len(motion_packets(fake)) >= 4
            and device._outstanding_query is not None
            and bytes(fake.pending[:7]) == b"\xaa\x11\x08" + struct.pack("<I", 0x10 + query_motor_id)
        ):
            injected.append(True)
            data = original_read(min(size, prefix_size))
            clock.now = latest_deadline[0] - 500_000
            return data
        return original_read(size)

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(fake, "read", read)
    with pytest.raises((RuntimeError, TimeoutError), match="less than one millisecond"):
        device.acceptance_step(reference - 0.1, lambda: None)
    receipt = device.fault_receipt
    assert receipt["all_joints_disabled_observed"] and receipt["error"] is None
    historical = receipt["per_joint_results"][0]["intermediate_feedback"]
    assert historical[0]["outstanding_query"]["motor_id"] == query_motor_id
    assert historical[0]["rx_context"]["inherited"] and historical[0]["historical_only"]


@pytest.mark.parametrize("old_reply_order", ["before_disable", "after_disable"])
def test_same_id_zero_prefix_reply_is_explicitly_ambiguous_and_not_confirmation(
    acceptance, monkeypatch, old_reply_order
):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    original, original_write, injected = device._bounded_serial_timeout, fake.write, []
    delayed = []

    def timeout(name, cap_s, deadline_ns):
        if (
            name == "timeout"
            and not injected
            and len(motion_packets(fake)) >= 4
            and device._outstanding_query is not None
            and bytes(fake.pending[:7]) == b"\xaa\x11\x08\x11\x00\x00\x00"
        ):
            injected.append(True)
            if old_reply_order == "after_disable":
                delayed.append(bytes(fake.pending[:16]))
                del fake.pending[:16]
            clock.now = deadline_ns - 500_000
        return original(name, cap_s, deadline_ns)

    def write(packet):
        result = original_write(packet)
        if packet == encode_disable(1) and delayed:
            fake.pending.extend(delayed.pop())
        return result

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(fake, "write", write)
    with pytest.raises((RuntimeError, TimeoutError), match="less than one millisecond"):
        device.acceptance_step(reference - 0.1, lambda: None)
    receipt = device.fault_receipt
    assert receipt["status"] == "unknown" and "ambiguous" in receipt["error"]
    assert not receipt.get("all_joints_disabled_observed", False)
    assert not receipt.get("disabled_feedback_observed", False)
    historical = receipt["per_joint_results"][0]["intermediate_feedback"]
    assert historical[0]["same_id_response_order_ambiguous"] and historical[0]["historical_only"]
    assert device._outstanding_query is not None and device._rx_pending


def test_rx_waits_do_not_read_file_contents_but_actual_writes_revalidate_binding(acceptance, monkeypatch):
    device, fake, _, _, request, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    calls, write_binding_counts = [], []
    original_binding, original_write = type(request).validate_binding, fake.write

    def binding(self, specs, control):
        calls.append(True)
        return original_binding(self, specs, control)

    def write(packet):
        if int.from_bytes(packet[13:17], "little") != 0x7FF or packet[23] == 0x55:
            write_binding_counts.append(len(calls))
        return original_write(packet)

    monkeypatch.setattr(type(request), "validate_binding", binding)
    monkeypatch.setattr(fake, "write", write)
    live_guard = Mock()
    for _ in range(10):
        device._acceptance_wait_guard(live_guard)
    assert not calls and live_guard.call_count == 10
    device.acceptance_step(reference - 0.1, live_guard)
    assert len(write_binding_counts) == 1 and write_binding_counts[0] >= 1
    previous = write_binding_counts[0]
    device.acceptance_step(reference, live_guard)
    assert write_binding_counts[-1] > previous


@pytest.mark.parametrize("submission", ["full_then_deadline", "partial"])
def test_query_debt_requires_complete_submission_even_when_postwrite_deadline_expires(
    acceptance, monkeypatch, submission
):
    device, fake, clock, _, _, _ = acceptance
    device.connect()
    reference = device.start_acceptance(lambda: None)
    original_timeout, original_write = device._bounded_serial_timeout, fake.write
    write_deadlines, injected = [], []

    def timeout(name, cap_s, deadline_ns):
        if name == "write_timeout":
            write_deadlines[:] = [deadline_ns]
        return original_timeout(name, cap_s, deadline_ns)

    def write(packet):
        result = original_write(packet)
        if (
            not injected
            and len(motion_packets(fake)) >= 4
            and int.from_bytes(packet[13:17], "little") == 0x7FF
            and packet[21:25] == bytes((4, 0, 0xCC, 0))
        ):
            injected.append(True)
            if submission == "partial":
                return result - 1
            clock.now = write_deadlines[0] + 100_000
        return result

    monkeypatch.setattr(device, "_bounded_serial_timeout", timeout)
    monkeypatch.setattr(fake, "write", write)
    with pytest.raises((RuntimeError, TimeoutError, OSError)):
        device.acceptance_step(reference - 0.1, lambda: None)
    receipt = device.fault_receipt
    assert injected and fake.statuses[1] == 0
    if submission == "full_then_deadline":
        assert receipt["all_joints_disabled_observed"] and receipt["outstanding_query_accounted"]
        assert receipt["outstanding_query_before_disable"]["motor_id"] == 4
        assert receipt["outstanding_query_before_disable"]["submission"] == "complete"
    else:
        assert receipt["status"] == "unknown" and not receipt["outstanding_query_accounted"]
        assert receipt["outstanding_query_before_disable"] is None
        assert receipt["query_write_inflight_before_disable"] is not None
        assert not receipt.get("all_joints_disabled_observed", False)


def test_late_unrequested_frame_before_fresh_cleanup_query_is_audited_and_fails(acceptance, monkeypatch):
    device, fake, _, _, _, _ = acceptance
    device.connect()
    device.start_acceptance(lambda: None)
    original = device._acceptance_read_all

    def read_all(deadline_ns, *, selected_status, cleanup=False):
        if cleanup:
            fake.pending.extend(bridge_packet(5, status=0))
        return original(deadline_ns, selected_status=selected_status, cleanup=cleanup)

    monkeypatch.setattr(device, "_acceptance_read_all", read_all)
    before = len(fake.writes)
    receipt = device.emergency_stop("late_extra_frame")
    assert receipt["status"] == "unknown" and "Unexpected queued RX" in receipt["error"]
    assert not receipt.get("all_joints_disabled_observed", False)
    assert fake.writes[before:] == [encode_disable(1)]
    evidence = device.take_io_evidence()
    unexpected = [event for event in evidence["events"] if event["kind"] == "post_disable_unexpected_rx"]
    assert unexpected and unexpected[-1]["data_hex"].startswith("aa11081500000005")
    assert device._rx_pending and device._rx_context["source"] == "post_disable_unexpected_rx"
