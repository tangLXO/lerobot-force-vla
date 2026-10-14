"""B601 query-only wire/ownership tests; no actual serial device is opened."""

import math
import struct
import threading
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

from lerobot.robots.rebot_b601_follower import readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.control_io import B601BusLease
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet


@pytest.fixture
def setup_transport(monkeypatch, tmp_path):
    """Construct every serial handle through a fake factory before open."""
    clock = Clock()
    monkeypatch.setattr(readonly.time, "perf_counter_ns", clock.perf_counter_ns)
    fake = FakeSerial(clock)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(readonly.serial, "Serial", factory)
    transport = readonly.ReadonlyB601Transport(
        "FAKE_B601", readonly.DEFAULT_B601_JOINTS[:2], lock_root=tmp_path
    )
    yield transport, fake, clock, factory
    fake.close_fails = False
    if transport.has_resources:
        transport.close()


def initialized(setup_transport):
    transport, fake, clock, factory = setup_transport
    transport.open()
    mapping = transport.read_mapping()
    assert mapping.is_valid
    return transport, fake, clock, factory


def test_wire_queries_match_reviewed_motorbridge_layout():
    """Only 0x7FF with CC or mapping-read 33 can be produced."""
    query = readonly.encode_readonly_query(7)
    assert query.hex() == "55aa1e03010000000a00000000ff070000000800000700cc000000000000"
    register = readonly.encode_readonly_query(1, register_id=21)
    assert register[:21] == query[:21]
    assert register[21:] == bytes((1, 0, 0x33, 21, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("rid", [0, 8, 10, 20, 24, 255, True, 21.0])
def test_query_cannot_encode_unreviewed_registers(rid):
    with pytest.raises(ValueError, match="mapping-register"):
        readonly.encode_readonly_query(1, register_id=rid)


@pytest.mark.parametrize("motor_id", [0, 16, -1, True, 1.0])
def test_query_rejects_unrepresentable_feedback_ids(motor_id):
    with pytest.raises(ValueError, match="motor_id"):
        readonly.encode_readonly_query(motor_id)


def test_open_close_send_nothing_and_single_use(setup_transport):
    transport, fake, _, factory = setup_transport
    transport.open()
    assert not fake.writes
    assert factory.call_args.kwargs["port"] is None
    assert fake.dtr is False and fake.rts is False
    assert fake.port == "FAKE_B601"
    transport.close()
    assert not fake.writes
    assert not transport.mapping_ready and not transport.has_resources
    with pytest.raises(RuntimeError, match="single-use"):
        transport.open()


def test_fragmented_replies_use_device_mapping_and_packet_sequences(setup_transport):
    transport, fake, clock, _ = setup_transport
    fake.fragment_size = 3
    transport.open()
    mapping = transport.read_mapping()
    assert mapping.is_valid and len(mapping.registers) == 6 and transport.mapping_ready
    assert [item.register_id for item in mapping.registers] == [21, 22, 23] * 2
    for sequence in (4, 5):
        began = clock.now
        batch = transport.read_feedback()
        assert batch.is_valid and len(batch.feedback) == 2
        for item in batch.feedback:
            assert item.receive_sequence == sequence
            assert began <= item.receive_timestamp_ns <= batch.completed_ns
            assert item.position_rad == pytest.approx(32768 * 16 / 65535 - 8)
            assert item.position_deg == pytest.approx(math.degrees(item.position_rad))
            assert item.mapping_values == (8.0, 14.0, 9.0)
            assert item.mapping_verified_on_device
            assert item.status_name == "DISABLED" and item.status_code == 0
            assert item.packet_hex == bridge_packet(item.motor_id).hex()
            assert item.payload_hex == bridge_packet(item.motor_id)[7:15].hex()
        assert batch.hardware_measurement_timestamp is None
        assert batch.hardware_acknowledgement == "unavailable"
    with pytest.raises(FrozenInstanceError):
        batch.feedback[0].position_deg = 0
    assert all(packet[21 + 2] in (0x33, 0xCC) for packet in fake.writes)
    assert all(int.from_bytes(packet[13:17], "little") == 0x7FF for packet in fake.writes)


def test_feedback_requires_actual_mapping_and_never_reads_defaults(setup_transport):
    transport, fake, _, _ = setup_transport
    transport.open()
    with pytest.raises(RuntimeError, match="mapping registers"):
        transport.read_feedback()
    assert not fake.writes
    assert transport.read_mapping().is_valid
    with pytest.raises(RuntimeError, match="already been read"):
        transport.read_mapping()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_mapping_cannot_be_substituted_with_model_defaults(setup_transport, value):
    transport, fake, _, _ = setup_transport
    fake.mapping[22] = value
    transport.open()
    mapping = transport.read_mapping()
    assert not mapping.is_valid and "finite and positive" in mapping.error
    assert len(mapping.registers) == 1 and mapping.registers[0].register_id == 21
    assert mapping.receive_chunks and len(mapping.query_packets_hex) == 2
    assert not transport.mapping_ready
    writes = len(fake.writes)
    batch = transport.read_feedback()
    assert not batch.is_valid and "fault_latched" in batch.error and not batch.feedback
    assert len(fake.writes) == writes


@pytest.mark.parametrize("field", ["can_id", "motor_id", "high_byte", "opcode", "rid"])
def test_mapping_reply_needs_exact_ids_opcode_and_register(setup_transport, field):
    transport, fake, _, _ = setup_transport

    def corrupt(packet, motor_id, rid):
        raw = bytearray(packet)
        raw[{"can_id": 3, "motor_id": 7, "high_byte": 8, "opcode": 9, "rid": 10}[field]] ^= 1
        return bytes(raw)

    fake.register_transform = corrupt
    transport.open()
    batch = transport.read_mapping()
    assert not batch.is_valid and not batch.registers
    assert batch.receive_chunks and batch.error


@pytest.mark.parametrize(
    "failure",
    ["can_id", "motor_id", "duplicate", "rtr", "extended", "dlc", "header", "trailer", "register", "extra"],
)
def test_bad_feedback_retains_wire_evidence_and_latches(setup_transport, failure):
    transport, fake, _, _ = initialized(setup_transport)
    assert transport.read_feedback().is_valid

    def corrupt(packet, motor_id):
        if motor_id == 1:
            return packet
        raw = bytearray(packet)
        if failure == "can_id":
            raw[3] = 0x17
        elif failure == "motor_id":
            raw[7] = 1
        elif failure == "duplicate":
            return bridge_packet(1)
        elif failure in ("rtr", "extended", "dlc"):
            raw[2] = {"rtr": 0x88, "extended": 0x48, "dlc": 7}[failure]
        elif failure == "header":
            raw[0] = 0
        elif failure == "trailer":
            raw[-1] = 0
        elif failure == "register":
            raw[7:15] = bytes((2, 0, 0x33, 21)) + struct.pack("<f", 12.5)
        elif failure == "extra":
            return packet + b"X"
        return bytes(raw)

    fake.feedback_transform = corrupt
    failed = transport.read_feedback()
    assert not failed.is_valid and failed.error and failed.receive_chunks
    assert "shoulder_lift" in failed.missing
    writes = len(fake.writes)
    fake.feedback_transform = lambda packet, motor_id: packet
    after = transport.read_feedback()
    assert not after.is_valid and after.feedback == () and after.receive_chunks == ()
    assert not after.query_packets_hex and len(fake.writes) == writes
    assert "fault_latched" in after.error


@pytest.mark.parametrize("response", [b"", b"\xaa\x11\x08"])
def test_missing_or_partial_frame_preserves_received_joint_not_old_cache(setup_transport, response):
    transport, fake, _, _ = initialized(setup_transport)
    first = transport.read_feedback()
    fake.feedback_transform = lambda packet, motor_id: packet if motor_id == 1 else response
    failed = transport.read_feedback(max_read_duration_s=0.002)
    assert not failed.is_valid and "TimeoutError" in failed.error
    assert [item.name for item in failed.feedback] == ["shoulder_pan"]
    assert failed.feedback[0].receive_sequence > first.feedback[0].receive_sequence
    assert failed.missing == ("shoulder_lift",)


@pytest.mark.parametrize("failure", ["partial_write", "late_read", "reset_fails"])
def test_transport_errors_and_late_complete_response_never_validate(setup_transport, failure):
    transport, fake, _, _ = initialized(setup_transport)
    setattr(fake, failure, True)
    result = transport.read_feedback(max_read_duration_s=0.003)
    assert not result.is_valid and result.error and not result.feedback
    if failure == "late_read":
        assert result.receive_chunks  # Raw late packet evidence is retained.
    assert transport.failure is not None


@pytest.mark.parametrize("has_bytes", [False, True])
def test_mapping_timeout_distinguishes_empty_read_from_actual_late_bytes(setup_transport, has_bytes):
    """An empty timed-out read must never be described as an arrived response."""
    transport, fake, _, _ = setup_transport
    fake.late_read = True
    if not has_bytes:
        fake.register_transform = lambda packet, motor_id, rid: b""
    transport.open()
    result = transport.read_mapping(max_read_duration_s=0.003)
    assert not result.is_valid and not result.registers and len(result.query_packets_hex) == 1
    if has_bytes:
        assert "response arrived after deadline" in result.error
        assert len(result.receive_chunks) == 1
        assert result.receive_chunks[0][0] > result.deadline_ns
        assert bytes.fromhex(result.receive_chunks[0][1]) == bridge_packet(
            1, payload=bytes((1, 0, 0x33, 21)) + struct.pack("<f", 8.0)
        )
    else:
        assert "no response before deadline" in result.error
        assert "arrived" not in result.error and result.receive_chunks == ()
    assert transport.failure is not None


def test_pre_request_buffer_is_not_restamped_as_new_feedback(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    fake.pending.extend(bridge_packet(1) + bridge_packet(2))
    fake.feedback_transform = lambda packet, motor_id: b""
    batch = transport.read_feedback(max_read_duration_s=0.002)
    assert not batch.is_valid and not batch.feedback and not batch.receive_chunks


def test_subset_query_and_absolute_deadline_do_not_read_unspecified_joints(setup_transport):
    transport, fake, clock, _ = initialized(setup_transport)
    before = len(fake.writes)
    deadline = clock.now + 3_000_000
    batch = transport.read_feedback(names=("shoulder_lift",), deadline_ns=deadline)
    assert batch.is_valid and batch.deadline_ns == deadline
    assert [item.name for item in batch.feedback] == ["shoulder_lift"]
    assert fake.writes[before:] == [readonly.encode_readonly_query(2)]


@pytest.mark.parametrize("names", [(), ("missing",), ("shoulder_pan", "shoulder_pan")])
def test_invalid_subset_never_touches_serial(setup_transport, names):
    transport, fake, _, _ = initialized(setup_transport)
    before = len(fake.calls)
    with pytest.raises(ValueError, match="Feedback names"):
        transport.read_feedback(names=names)
    assert len(fake.calls) == before


def test_expired_absolute_deadline_does_not_send_query(setup_transport):
    transport, fake, clock, _ = initialized(setup_transport)
    before = len(fake.writes)
    batch = transport.read_feedback(deadline_ns=clock.now)
    assert not batch.is_valid and "deadline expired" in batch.error
    assert len(fake.writes) == before and not batch.query_packets_hex


def test_reordered_joint_replies_cannot_satisfy_a_different_outstanding_query(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    fake.feedback_transform = lambda packet, motor_id: bridge_packet(3 - motor_id)
    batch = transport.read_feedback()
    assert not batch.is_valid and "unexpected feedback ID" in batch.error
    assert not batch.feedback and len(batch.query_packets_hex) == 1


def test_device_fault_remains_visible_and_invalidates_complete_batch(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    fake.feedback_transform = lambda packet, motor_id: bridge_packet(motor_id, status=13)
    batch = transport.read_feedback()
    assert not batch.is_valid and not batch.missing
    assert all(item.status_name == "LOST_COMM" for item in batch.feedback)
    assert transport.failure


def test_single_owner_rejects_read_and_close_from_another_thread(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    before = len(fake.calls)
    errors = []

    def wrong_thread():
        for method in (transport.read_feedback, transport.close):
            try:
                method()
            except RuntimeError as exc:
                errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(1)
    assert len(errors) == 2 and all("single owner" in error for error in errors)
    assert len(fake.calls) == before
    transport.close()
    assert len({ident for _, ident in fake.calls}) == 1


def test_lease_precedes_open_and_close_failure_retains_it(setup_transport, tmp_path):
    transport, fake, _, factory = setup_transport
    competing = B601BusLease("damiao", "FAKE_B601", lock_root=tmp_path)
    transport.open()
    with pytest.raises(RuntimeError, match="already has an owner"):
        competing.acquire()
    fake.close_fails = True
    with pytest.raises(OSError, match="close failed"):
        transport.close()
    assert transport.has_resources
    with pytest.raises(RuntimeError, match="already has an owner"):
        competing.acquire()
    fake.close_fails = False
    transport.close()
    competing.acquire()
    second = readonly.ReadonlyB601Transport("FAKE_B601", lock_root=tmp_path)
    with pytest.raises(RuntimeError, match="already has an owner"):
        second.open()
    assert factory.call_count == 1
    competing.release()


def test_open_failure_closes_before_releasing_lease(setup_transport, tmp_path):
    transport, fake, _, _ = setup_transport
    fake.open_fails = True
    with pytest.raises(OSError, match="open failed"):
        transport.open()
    assert not transport.has_resources and not fake.opened
    competing = B601BusLease("damiao", "FAKE_B601", lock_root=tmp_path)
    competing.acquire()
    competing.release()


def test_open_and_close_failure_preserve_both_errors_and_lease(setup_transport, tmp_path):
    transport, fake, _, _ = setup_transport
    fake.open_fails = fake.close_fails = True
    with pytest.raises(BaseExceptionGroup) as caught:
        transport.open()
    assert [str(exc) for exc in caught.value.exceptions] == ["open failed", "close failed"]
    assert transport.has_resources
    competing = B601BusLease("damiao", "FAKE_B601", lock_root=tmp_path)
    with pytest.raises(RuntimeError, match="already has an owner"):
        competing.acquire()
    fake.close_fails = False
    transport.close()
    competing.acquire()
    competing.release()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_read_budget_performs_no_io(setup_transport, budget):
    transport, fake, _, _ = setup_transport
    transport.open()
    with pytest.raises(ValueError, match="duration"):
        transport.read_mapping(max_read_duration_s=budget)
    assert not fake.writes


@pytest.mark.parametrize("phase", ["mapping", "feedback"])
def test_keyboard_interrupt_latches_before_propagating_and_keeps_attempt(setup_transport, phase):
    transport, fake, _, _ = setup_transport
    transport.open()
    if phase == "feedback":
        assert transport.read_mapping().is_valid
    fake.read = Mock(side_effect=KeyboardInterrupt("cancelled"))
    with pytest.raises(KeyboardInterrupt):
        getattr(transport, f"read_{phase}")()
    assert transport.failure and "KeyboardInterrupt" in transport.failure
    assert transport.last_batch.query_packets_hex and not transport.last_batch.is_valid
    writes = len(fake.writes)
    result = transport.read_feedback()
    assert "fault_latched" in result.error and not result.feedback
    assert len(fake.writes) == writes


def test_feedback_skips_unchanged_serial_configuration(setup_transport, monkeypatch):
    """Repeated normal reads must not issue costly Windows configuration calls."""
    transport, fake, _, _ = initialized(setup_transport)
    assignments = []

    def assign(self, name, value):
        if name in ("timeout", "write_timeout"):
            assignments.append((name, value))
        object.__setattr__(self, name, value)

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    assert transport.read_feedback().is_valid
    assert transport.read_feedback().is_valid
    assert assignments == []


def test_feedback_still_tightens_write_wait_to_remaining_budget(setup_transport, monkeypatch):
    """Skipping redundant setters must not turn a short budget into a long wait."""
    transport, fake, _, _ = initialized(setup_transport)
    assignments = []

    def assign(self, name, value):
        if name in ("timeout", "write_timeout"):
            assignments.append((name, value))
        object.__setattr__(self, name, value)

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    assert transport.read_feedback(max_read_duration_s=0.003).is_valid
    write_limits = [value for name, value in assignments if name == "write_timeout"]
    assert write_limits and all(0 < value <= 0.003 for value in write_limits)
    assert all(value * 1000 == math.floor(value * 1000) for value in write_limits)
    retained_timeout = fake.write_timeout
    assignments.clear()
    assert transport.read_feedback().is_valid
    assert fake.write_timeout == retained_timeout and assignments == []


@pytest.mark.parametrize(
    "setter_cost_ns,reason", [(4_000_000, "deadline expired"), (2_500_000, "one millisecond")]
)
def test_expensive_write_timeout_setter_cannot_send_an_expired_query(
    setup_transport, monkeypatch, setter_cost_ns, reason
):
    transport, fake, clock, _ = initialized(setup_transport)
    before = len(fake.writes)
    assignments = []

    def assign(self, name, value):
        object.__setattr__(self, name, value)
        if name == "write_timeout":
            assignments.append(value)
            clock.now += setter_cost_ns

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    batch = transport.read_feedback(names=("shoulder_pan",), max_read_duration_s=0.003)
    assert not batch.is_valid and reason in batch.error
    assert assignments == [0.003]
    assert not batch.query_packets_hex and not batch.receive_chunks
    assert len(fake.writes) == before


def test_expensive_read_timeout_setter_cannot_start_an_expired_read(setup_transport, monkeypatch):
    transport, fake, clock, _ = initialized(setup_transport)
    fake.write_timeout, fake.timeout = 0.001, 0.01
    fake.calls.clear()
    assignments = []

    def assign(self, name, value):
        object.__setattr__(self, name, value)
        if name == "timeout":
            assignments.append(value)
            clock.now += 3_000_000

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    batch = transport.read_feedback(names=("shoulder_pan",), max_read_duration_s=0.003)
    assert not batch.is_valid and "deadline expired" in batch.error
    assert assignments == [0.002]
    assert len(batch.query_packets_hex) == 1 and not batch.receive_chunks
    assert [operation for operation, _ in fake.calls] == ["reset_input_buffer", "write"]


def test_fragmented_feedback_reuses_short_timeouts_without_reconfiguring(setup_transport, monkeypatch):
    transport, fake, _, _ = initialized(setup_transport)
    fake.timeout = fake.write_timeout = 0.001
    fake.fragment_size = 1
    fake.calls.clear()
    assignments = []

    def assign(self, name, value):
        if name in ("timeout", "write_timeout"):
            assignments.append((name, value))
        object.__setattr__(self, name, value)

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    batch = transport.read_feedback(max_read_duration_s=0.005)
    assert batch.is_valid and len(batch.receive_chunks) == 32
    assert assignments == []
    assert [operation for operation, _ in fake.calls].count("write") == 2


@pytest.mark.parametrize("remaining_ns", [1, 500_000, 999_999])
def test_submillisecond_query_budget_never_sets_timeout_or_sends(setup_transport, monkeypatch, remaining_ns):
    transport, fake, clock, _ = initialized(setup_transport)
    before = len(fake.writes)
    assignments = []

    def assign(self, name, value):
        if name in ("timeout", "write_timeout"):
            assignments.append((name, value))
        object.__setattr__(self, name, value)

    monkeypatch.setattr(FakeSerial, "__setattr__", assign)
    batch = transport.read_feedback(names=("shoulder_pan",), deadline_ns=clock.now + remaining_ns)
    assert not batch.is_valid and "less than one millisecond" in batch.error
    assert not batch.query_packets_hex and not batch.receive_chunks and assignments == []
    assert len(fake.writes) == before


def test_fragmented_read_with_less_than_one_millisecond_keeps_partial_evidence(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    fake.write_timeout = fake.timeout = 0.001
    fake.fragment_size = 1
    batch = transport.read_feedback(names=("shoulder_pan",), max_read_duration_s=0.002)
    assert not batch.is_valid and "less than one millisecond" in batch.error
    assert len(batch.query_packets_hex) == 1 and batch.receive_chunks
    assert not batch.feedback and batch.missing == ("shoulder_pan",)
    assert sum(len(bytes.fromhex(chunk)) for _, chunk in batch.receive_chunks) < 16


def test_timed_out_prefix_remains_on_same_owner_without_fresh_sequence_or_retry(setup_transport, monkeypatch):
    transport, fake, clock, _ = initialized(setup_transport)
    sequences = dict(transport._sequences)
    original_read = fake.read

    def read_prefix(size):
        data = original_read(min(size, 1))
        clock.now += 4_000_000
        return data

    monkeypatch.setattr(fake, "read", read_prefix)
    batch = transport.read_feedback(names=("shoulder_pan",), max_read_duration_s=0.005)
    monkeypatch.setattr(fake, "read", original_read)
    assert not batch.is_valid and not batch.feedback and "TimeoutError" in batch.error
    assert batch.receive_chunks[0][1] == "aa"
    assert bytes(transport._rx_pending) == b"\xaa"
    assert transport._rx_context["source"] == "query_rx"
    assert transport._rx_context["deadline_ns"] == batch.deadline_ns
    assert transport._rx_context["first_receive_timestamp_ns"] == batch.receive_chunks[0][0]
    assert transport._sequences == sequences
    queued = bytes(fake.pending)
    writes = len(fake.writes)
    resets = sum(name == "reset_input_buffer" for name, _ in fake.calls)
    retry = transport.read_feedback(names=("shoulder_pan",))
    assert not retry.is_valid and "fault_latched" in retry.error
    assert not retry.feedback and not retry.receive_chunks and not retry.query_packets_hex
    assert len(fake.writes) == writes and bytes(fake.pending) == queued
    assert sum(name == "reset_input_buffer" for name, _ in fake.calls) == resets
    with pytest.raises(RuntimeError, match="prior RX remains pending"):
        transport._exchange((readonly.encode_readonly_query(1),), clock.now + 5_000_000, [], [], Mock())
    assert len(fake.writes) == writes and transport._sequences == sequences
    transport.close()
    assert not transport._rx_pending and transport._rx_context is None


def test_late_complete_frame_is_retained_without_query_commit(setup_transport):
    transport, fake, _, _ = initialized(setup_transport)
    sequences = dict(transport._sequences)
    fake.late_read = True
    batch = transport.read_feedback(names=("shoulder_pan",), max_read_duration_s=0.003)
    assert not batch.is_valid and not batch.feedback and "arrived after deadline" in batch.error
    assert bytes(transport._rx_pending) == bridge_packet(1)
    assert transport._rx_context["last_receive_timestamp_ns"] > batch.deadline_ns
    assert transport._sequences == sequences


def test_driver_overread_poison_never_truncates_or_scans_for_another_header(setup_transport, monkeypatch):
    transport, fake, _, _ = initialized(setup_transport)
    packet = bridge_packet(1) + b"X"

    def overread(_size):
        fake.pending.clear()
        return packet

    monkeypatch.setattr(fake, "read", overread)
    batch = transport.read_feedback(names=("shoulder_pan",))
    assert not batch.is_valid and "fixed 16-byte frame slot" in batch.error
    assert bytes.fromhex(batch.receive_chunks[0][1]) == packet
    assert transport._rx_poison is not None and not transport._rx_pending
    assert not batch.feedback


def test_expired_feedback_consumption_rolls_back_only_current_joint(setup_transport, monkeypatch):
    transport, fake, clock, _ = initialized(setup_transport)
    sequences = dict(transport._sequences)
    original_finish = transport._finish_rx_frame

    def finish_after_consumption():
        original_finish()
        if fake.writes[-1][21] == 2:
            clock.now += 5_000_000

    monkeypatch.setattr(transport, "_finish_rx_frame", finish_after_consumption)
    batch = transport.read_feedback(max_read_duration_s=0.005)
    assert not batch.is_valid and "deadline expired" in batch.error
    assert [row.name for row in batch.feedback] == ["shoulder_pan"]
    assert batch.missing == ("shoulder_lift",)
    assert transport._sequences == {
        "shoulder_pan": sequences["shoulder_pan"] + 1,
        "shoulder_lift": sequences["shoulder_lift"],
    }
    assert bytes(transport._rx_pending) == bridge_packet(2)
    assert transport._rx_context["packet_hex"] == readonly.encode_readonly_query(2).hex()
    assert transport._rx_context["session_signature"] == transport._rx_session_signature()


def test_expired_mapping_consumption_restores_mapping_sequence_and_frame(setup_transport, monkeypatch):
    transport, fake, clock, _ = setup_transport
    transport.open()
    original_finish = transport._finish_rx_frame

    def finish_after_consumption():
        original_finish()
        if fake.writes[-1][24] == 22:
            clock.now += 5_000_000

    monkeypatch.setattr(transport, "_finish_rx_frame", finish_after_consumption)
    batch = transport.read_mapping(max_read_duration_s=0.005)
    assert not batch.is_valid and "deadline expired" in batch.error
    assert [row.register_id for row in batch.registers] == [21]
    assert set(transport._mapping) == {("shoulder_pan", 21)}
    assert transport._sequences == {"shoulder_pan": 1, "shoulder_lift": 0}
    assert bytes(transport._rx_pending) == bridge_packet(
        1, payload=bytes((1, 0, 0x33, 22)) + struct.pack("<f", 14.0)
    )
    assert transport._rx_context["session_signature"] == transport._rx_session_signature()


def test_query_rollback_failure_preserves_both_errors_and_poisoned_frame(setup_transport, monkeypatch):
    transport, _, clock, _ = initialized(setup_transport)
    original_finish = transport._finish_rx_frame

    def late_finish():
        original_finish()
        clock.now += 5_000_000

    def consume(_raw, _arbitration_id, _received_ns):
        def rollback():
            raise OSError("rollback failed")

        return rollback

    monkeypatch.setattr(transport, "_finish_rx_frame", late_finish)
    with pytest.raises(BaseExceptionGroup) as failure:
        transport._exchange((readonly.encode_readonly_query(1),), clock.now + 5_000_000, [], [], consume)
    assert len(failure.value.exceptions) == 2
    assert isinstance(failure.value.exceptions[0], TimeoutError)
    assert isinstance(failure.value.exceptions[1], OSError)
    assert bytes(transport._rx_pending) == bridge_packet(1)
    assert transport._rx_poison is not None


@pytest.mark.parametrize("kind", ["mapping", "feedback"])
def test_batch_assembly_after_deadline_is_invalid_but_keeps_prior_query_commit(
    setup_transport, monkeypatch, kind
):
    if kind == "mapping":
        transport, _, clock, _ = setup_transport
        transport.open()
    else:
        transport, _, clock, _ = initialized(setup_transport)
    original_exchange = transport._exchange

    def finish_last_query(requests, deadline_ns, queries, chunks, consume):
        original_exchange(requests, deadline_ns, queries, chunks, consume)
        motor_id, _, _, register_id = requests[-1][21:25]
        if motor_id == 2 and (kind == "feedback" or register_id == 23):
            clock.now = deadline_ns + 1

    monkeypatch.setattr(transport, "_exchange", finish_last_query)
    batch = transport.read_mapping() if kind == "mapping" else transport.read_feedback()
    assert not batch.is_valid and "batch completed after deadline" in batch.error
    assert batch.completed_ns > batch.deadline_ns and transport.failure is not None
    assert not transport._rx_pending
    # A valid, completed batch also cannot be restamped into a late valid view.
    assert not replace(batch, error=None).is_valid


def test_feedback_waits_for_each_reply_before_next_query(setup_transport):
    """The bridge must never receive a burst before earlier replies are consumed."""
    transport, fake, _, _ = initialized(setup_transport)
    fake.calls.clear()
    assert transport.read_feedback().is_valid
    assert [name for name, _ in fake.calls] == ["reset_input_buffer", "write", "read"] * 2
    assert all(len(packet) == 30 for packet in fake.writes)
