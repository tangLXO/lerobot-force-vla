"""Strict 102 freshness, read-only setup, and non-blocking worker tests."""

import struct
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.force_control.leader import LeaderSamplerWorker
from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
from lerobot.teleoperators.rebot_102_leader.readonly_transport import (
    Readonly102Transport,
    encode_monitor_query,
)
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import LeaderActionSnapshot

MODULE = "lerobot.teleoperators.rebot_102_leader.rebot_102_leader"


@pytest.fixture
def readonly_leader(tmp_path):
    """Make a fake SDK with explicit freshness, never open real hardware."""
    config = RebotArm102LeaderTeleopConfig(port="FAKE", calibration_dir=tmp_path)
    bus = MagicMock()
    bus.sync_monitor.return_value = {
        motor_id: SimpleNamespace(angle_deg=5.0, reliable=True) for motor_id in config.joint_ids.values()
    }
    with patch(f"{MODULE}.require_package"), patch(f"{MODULE}.FashionStarServo", return_value=bus):
        leader = RebotArm102Leader(config)
        leader.connect_readonly()
        yield leader, bus
        if leader.is_connected:
            leader.disconnect()


def test_readonly_connect_only_opens_and_disconnects(readonly_leader):
    """Opening the audited SDK transport invokes no motion or configuration API."""
    leader, bus = readonly_leader
    assert bus.method_calls == []
    with pytest.raises(RuntimeError, match="read-only"):
        leader.configure()
    with pytest.raises(RuntimeError, match="read-only"):
        leader.calibrate()
    assert bus.method_calls == []
    leader.disconnect()
    assert [call[0] for call in bus.method_calls] == ["close"]


def test_snapshot_native_mapping_and_immutable_provenance(readonly_leader):
    """Degree mapping matches native teleoperation, with honest host times."""
    leader, _ = readonly_leader
    snapshot = leader.read_action_snapshot()
    assert dict(snapshot.raw_action_deg)["shoulder_pan.pos"] == 5.0
    assert dict(snapshot.action_deg)["shoulder_pan.pos"] == -5.0
    assert dict(snapshot.action_deg)["gripper.pos"] == -30.0
    assert snapshot.request_started_ns <= snapshot.response_completed_ns
    assert snapshot.timestamp_source == "host_sync_monitor_return"
    assert snapshot.sequence == 1
    assert leader.read_action_snapshot().sequence == 2
    assert leader._last_raw_positions == {}
    with pytest.raises(FrozenInstanceError):
        snapshot.sequence = 10


@pytest.mark.parametrize(
    "bad", [None, SimpleNamespace(angle_deg=9.0, reliable=False), SimpleNamespace(angle_deg=9.0)]
)
def test_missing_unreliable_or_unsupported_monitor_rejected(readonly_leader, bad):
    """Never accept the SDK's held-last-value path as fresh leader input."""
    leader, bus = readonly_leader
    leader.read_action_snapshot()
    bus.sync_monitor.return_value[0] = bad
    with pytest.raises(RuntimeError, match="response") as exc:
        leader.read_action_snapshot()
    assert len(exc.value.joint_diagnostics) == 7
    assert exc.value.joint_diagnostics[0].sdk_reliable is not True
    assert exc.value.response_completed_ns >= exc.value.request_started_ns
    assert leader._read_sequence == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_monitor_rejected(readonly_leader, bad):
    """Non-finite servo values never become clamped valid-looking targets."""
    leader, bus = readonly_leader
    bus.sync_monitor.return_value[0].angle_deg = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        leader.read_action_snapshot()


def test_missing_joint_and_read_error_do_not_reuse_cached_action(readonly_leader):
    """Both incomplete dictionaries and transport errors fail closed."""
    leader, bus = readonly_leader
    del bus.sync_monitor.return_value[6]
    with pytest.raises(RuntimeError, match="joint IDs"):
        leader.read_action_snapshot()
    bus.sync_monitor.side_effect = OSError("gone")
    leader._last_raw_positions = dict.fromkeys(leader.motor_names, 1.0)
    with pytest.raises(OSError, match="gone"):
        leader.read_action_snapshot()


def test_late_response_is_invalid_not_fresh_at_return(readonly_leader):
    """A returned batch exceeding the configured response budget is discarded."""
    leader, _ = readonly_leader
    with (
        patch(f"{MODULE}.time.perf_counter_ns", side_effect=[100, 200_000_100]),
        pytest.raises(TimeoutError, match="exceeded"),
    ):
        leader.read_action_snapshot(max_read_duration_s=0.15)


class FakeLeader:
    """Controllable fake with call-thread evidence and an interruptible test gate."""

    def __init__(self):
        self.calls = []
        self.fail = False
        self.sequence = 0
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def connect_readonly(self):
        """Record the worker's transport setup context."""
        self.calls.append(("connect", threading.get_ident()))

    def read_action_snapshot(self, *, max_read_duration_s):
        """Optionally delay a read without any actual device access."""
        self.calls.append(("read", threading.get_ident()))
        started = time.perf_counter_ns()
        self.entered.set()
        self.gate.wait(3)
        if self.fail:
            raise OSError("read_failure")
        self.sequence += 1
        return LeaderActionSnapshot(
            raw_action_deg=(("gripper.pos", 1.0),),
            action_deg=(("gripper.pos", -6.0),),
            request_started_ns=started,
            response_completed_ns=time.perf_counter_ns(),
            sequence=self.sequence,
        )

    def disconnect(self):
        """Record teardown on the same reader thread."""
        self.calls.append(("close", threading.get_ident()))


def test_worker_serializes_all_leader_io_and_stale_does_not_refresh():
    """An immutable cache preserves response time and fails on conservative age."""
    source = FakeLeader()
    worker = LeaderSamplerWorker(source, frequency_hz=1)
    worker.start()
    try:
        intent = worker.wait_for_valid()
        assert dict(intent.action_deg) == {"gripper.pos": -6.0}
        stale = worker.latest(intent.request_started_ns + 200_000_000, max_age_s=0.1)
        assert not stale.is_valid and not stale.action_deg and not stale.raw_action_deg
        assert stale.response_completed_ns == intent.response_completed_ns
        assert not worker.latest(intent.response_completed_ns - 1).is_valid
    finally:
        worker.stop()
    assert len({ident for _, ident in source.calls}) == 1
    assert source.calls[0][0] == "connect" and source.calls[-1][0] == "close"
    assert source.calls[0][1] != threading.get_ident()
    assert not worker.latest().is_valid


def test_failure_replaces_previous_action_with_new_invalid_version():
    """A missing leader response cannot silently repeat a previous command."""
    source = FakeLeader()
    worker = LeaderSamplerWorker(source, frequency_hz=100)
    worker.start()
    try:
        previous = worker.wait_for_valid()
        source.fail = True
        deadline = time.monotonic() + 1
        while worker.latest().is_valid and time.monotonic() < deadline:
            time.sleep(0.001)
        failed = worker.latest()
        assert not failed.is_valid
        assert failed.sequence > previous.sequence
        assert failed.action_deg == failed.raw_action_deg == ()
        assert "read_failure" in failed.error
    finally:
        worker.stop()


def test_blocked_leader_read_does_not_block_latest_or_concurrent_close():
    """A native stall cannot hold the mailbox lock or cause racing SDK close."""
    source = FakeLeader()
    source.gate.clear()
    worker = LeaderSamplerWorker(source)
    assert not worker.has_resources
    worker.start()
    assert source.entered.wait(1)
    assert worker.has_resources
    try:
        start = time.monotonic()
        assert not worker.latest().is_valid
        assert time.monotonic() - start < 0.1
        with pytest.raises(TimeoutError, match="did not stop"):
            worker.stop(timeout_s=0.01)
        assert worker.has_resources
        assert all(call != "close" for call, _ in source.calls)
        assert not worker.latest().is_valid
    finally:
        source.gate.set()
        worker.stop()
    assert not worker.has_resources
    assert not worker.latest().is_valid


def test_unknown_action_processor_rejected():
    """Unknown action spaces cannot be mislabeled as physical joint degrees."""
    with pytest.raises(ValueError, match="absolute degree"):
        LeaderSamplerWorker(FakeLeader(), action_processor=lambda x: x)


def test_wait_for_valid_is_bounded_and_worker_is_single_use():
    """Startup failures remain explicit and do not indefinitely block startup."""
    source = FakeLeader()
    source.fail = True
    worker = LeaderSamplerWorker(source)
    worker.start()
    try:
        with pytest.raises(TimeoutError, match="no valid snapshot"):
            worker.wait_for_valid(timeout_s=0.03)
        with pytest.raises(RuntimeError, match="single-use"):
            worker.start()
    finally:
        worker.stop()


def _monitor_packet(motor_id, angle_deg=0.0, size=16):
    params = bytearray(size)
    params[0] = motor_id
    params[9] = 3
    params[10:14] = struct.pack("<i", round(angle_deg * 10))
    prefix = bytes((5, 28, 22, size)) + params
    return prefix + bytes((sum(prefix) & 255,))


class FakeSerial:
    """Serial-shaped fake that releases test response bytes only after a query."""

    def __init__(self, response, fragment_size=999):
        self.response = response
        self.fragment_size = fragment_size
        self.pending = bytearray()
        self.writes = []
        self.timeout = 0.001
        self.opened = False

    def open(self):
        """Record transport open without any device mutation."""
        self.opened = True

    def close(self):
        """Record transport close."""
        self.opened = False

    def reset_input_buffer(self):
        """Discard old pre-request bytes."""
        self.pending.clear()

    def write(self, packet):
        """Capture actual command bytes and expose one test response."""
        self.writes.append(packet)
        self.pending.extend(self.response)
        return len(packet)

    @property
    def in_waiting(self):
        """Return available test bytes."""
        return min(len(self.pending), self.fragment_size)

    def read(self, size):
        """Supply optional fragmented packets or a bounded missing-read timeout."""
        data = bytes(self.pending[:size])
        del self.pending[:size]
        if not data:
            time.sleep(self.timeout)
        return data


def _protocol_transport(fake, ids=(0, 1)):
    transport = Readonly102Transport("FAKE", 1_000_000, ids)
    with patch("lerobot.teleoperators.rebot_102_leader.readonly_transport.serial.Serial", return_value=fake):
        transport.open()
    return transport


def test_code25_query_matches_audited_sdk_wire_layout():
    """Compare to the concrete vendor request, including its additive checksum."""
    assert encode_monitor_query(tuple(range(7))).hex() == "124c190a16010700010203040506b4"
    with pytest.raises(ValueError, match="unique"):
        encode_monitor_query((1, 1))


def test_protocol_fragmentation_signed_angle_and_fresh_near_zero():
    """New valid zero-angle packets remain new; no 30-sample SDK hold is imported."""
    fake = FakeSerial(_monitor_packet(0, -0.2, size=14) + _monitor_packet(1, 0), fragment_size=2)
    transport = _protocol_transport(fake)
    try:
        for sequence in range(1, 36):
            started = time.perf_counter_ns()
            response = transport.read_monitors(max_read_duration_s=0.05)
            assert response[0].angle_deg == -0.2
            assert response[1].angle_deg == 0.0
            assert response[0].receive_sequence == response[1].receive_sequence == sequence
            assert started <= response[0].receive_timestamp_ns <= time.perf_counter_ns()
            assert response[0].motor_status == 3
        assert all(packet[2:5] == bytes((25, 5, 22)) for packet in fake.writes)
    finally:
        transport.close()


@pytest.mark.parametrize("failure", ["checksum", "wrong_id", "duplicate", "wrong_code", "partial", "lost"])
def test_protocol_failures_latch_and_never_reuse_previous_packet(failure):
    """Bad/missing response evidence cannot become a fresh next-cycle action."""
    first = _monitor_packet(0)
    second = _monitor_packet(1)
    damaged = {
        "checksum": first[:-1] + bytes((first[-1] ^ 1,)) + second,
        "wrong_id": first + _monitor_packet(2),
        "duplicate": first + first,
        "wrong_code": first[:2] + bytes((10,)) + first[3:] + second,
        "partial": first + second[:10],
        "lost": first,
    }[failure]
    fake = FakeSerial(first + second)
    transport = _protocol_transport(fake)
    try:
        transport.read_monitors(max_read_duration_s=0.03)
        fake.response = damaged
        with pytest.raises((RuntimeError, TimeoutError)):
            transport.read_monitors(max_read_duration_s=0.003)
        writes = len(fake.writes)
        fake.response = first + second
        with pytest.raises(RuntimeError, match="fault latched"):
            transport.read_monitors(max_read_duration_s=0.03)
        assert len(fake.writes) == writes
    finally:
        transport.close()


def test_protocol_enforces_its_single_serial_owner():
    """A competing execution context cannot perform even a read query."""
    fake = FakeSerial(_monitor_packet(0) + _monitor_packet(1))
    transport = _protocol_transport(fake)
    errors = []

    def wrong_thread():
        try:
            transport.read_monitors(max_read_duration_s=0.01)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(1)
    assert len(errors) == 1 and "owner" in errors[0]
    assert not fake.writes
    transport.close()


@pytest.mark.parametrize("failure", ["late_complete_response", "partial_write"])
def test_protocol_late_response_and_partial_write_cannot_commit_a_batch(failure):
    """Even correct packets cannot validate an overdue or incompletely sent request."""
    fake = FakeSerial(_monitor_packet(0) + _monitor_packet(1))
    transport = _protocol_transport(fake)
    try:
        if failure == "late_complete_response":
            original_read = fake.read

            def read_late(size):
                time.sleep(0.005)
                return original_read(size)

            fake.read = read_late
        else:
            original_write = fake.write
            fake.write = lambda packet: original_write(packet) - 1
        with pytest.raises((OSError, TimeoutError)):
            transport.read_monitors(max_read_duration_s=0.002)
        count = len(fake.writes)
        with pytest.raises(RuntimeError, match="fault latched"):
            transport.read_monitors(max_read_duration_s=0.05)
        assert len(fake.writes) == count
    finally:
        transport.close()


def test_protocol_driver_worker_uses_real_packet_evidence_and_stops(tmp_path):
    """Configured protocol path never calls SDK setup or exposes held data."""
    config = RebotArm102LeaderTeleopConfig(
        port="FAKE", calibration_dir=tmp_path, readonly_transport="protocol_v1"
    )
    fake = FakeSerial(b"".join(_monitor_packet(i, 0.1) for i in range(7)))
    with patch("lerobot.teleoperators.rebot_102_leader.readonly_transport.serial.Serial", return_value=fake):
        worker = LeaderSamplerWorker(RebotArm102Leader(config), max_read_duration_s=0.02)
        worker.start()
        try:
            intent = worker.wait_for_valid()
            assert intent.timestamp_source == "host_validated_monitor_packet_read_return"
            assert all(d.receive_sequence == 1 and d.sdk_reliable is None for d in intent.joint_diagnostics)
            assert dict(intent.action_deg)["gripper.pos"] == pytest.approx(-0.6)
            fake.response = b""
            time.sleep(0.05)
        finally:
            worker.stop()
        assert not worker.latest().is_valid
        assert not fake.opened
