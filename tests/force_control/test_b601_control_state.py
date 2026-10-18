"""Dedicated query-only control-state inspection never enables or reconfigures drives."""

import struct
from unittest.mock import Mock

import pytest

from lerobot.robots.rebot_b601_follower import control_state, readonly_transport as readonly
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet


@pytest.fixture
def setup_state(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(readonly.time, "perf_counter_ns", clock.perf_counter_ns)
    monkeypatch.setattr(control_state.time, "perf_counter_ns", clock.perf_counter_ns)

    class StateSerial(FakeSerial):
        def write(self, packet):
            self._call("write")
            self.writes.append(packet)
            payload = packet[21:29]
            motor_id, zero, opcode, rid = payload[:4]
            assert zero == 0 and opcode == 0x33 and rid in (9, 10)
            assert int.from_bytes(packet[13:17], "little") == 0x7FF
            value = 2 if rid == 10 else 2000
            response = bridge_packet(
                motor_id, payload=bytes((motor_id, 0, 0x33, rid)) + struct.pack("<I", value)
            )
            self.pending.extend(self.register_transform(response, motor_id, rid))
            return len(packet) - int(self.partial_write)

    fake = StateSerial(clock)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(readonly.serial, "Serial", factory)
    transport = readonly.ReadonlyB601Transport(
        "FAKE_STATE_ONLY", readonly.DEFAULT_B601_JOINTS, lock_root=tmp_path
    )
    transport.open()
    yield transport, fake, clock
    transport.close()


def test_all_joint_mode_timeout_queries_are_uint32_and_no_mapping_mutation(setup_state):
    transport, fake, _ = setup_state
    state = control_state.read_control_state(transport)
    assert state.is_valid and len(state.registers) == 14 and len(fake.writes) == 14
    assert [item.value for item in state.registers] == [2, 2000] * 7
    assert all(type(item.value) is int and item.receive_sequence in (1, 2) for item in state.registers)
    assert not transport.mapping_ready and not transport._mapping
    assert not state.watchdog_semantics_verified and state.hardware_ack == "unknown"
    with pytest.raises(ValueError, match="mapping-register"):
        readonly.encode_readonly_query(1, register_id=9)


@pytest.mark.parametrize("failure", ["wrong_id", "wrong_register", "partial_write"])
def test_query_failure_latches_without_any_followup_write(setup_state, failure):
    transport, fake, _ = setup_state
    if failure == "wrong_id":
        fake.register_transform = lambda packet, motor, rid: packet[:3] + struct.pack("<I", 0x19) + packet[7:]
    elif failure == "wrong_register":
        fake.register_transform = lambda packet, motor, rid: packet[:10] + bytes((8,)) + packet[11:]
    else:
        fake.partial_write = True
    state = control_state.read_control_state(transport)
    assert not state.is_valid and state.error and len(fake.writes) == 1
    retry = control_state.read_control_state(transport)
    assert not retry.is_valid and len(fake.writes) == 1


def test_subset_queries_and_invalid_names_never_add_motor_commands(setup_state):
    transport, fake, _ = setup_state
    state = control_state.read_control_state(transport, names=("wrist_roll",))
    assert state.is_valid and len(fake.writes) == 2
    assert all(packet[21] == 6 for packet in fake.writes)
    with pytest.raises(ValueError, match="names"):
        control_state.read_control_state(transport, names=("unknown",))
    assert len(fake.writes) == 2
