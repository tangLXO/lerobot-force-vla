"""Reusable fake serial full-arm lifecycle; never opens actual hardware."""

import math
import struct
from contextlib import suppress
from types import SimpleNamespace

import pytest

from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.full_arm_validation import JOINT_NAMES, ValidationLeaderSample
from tests.force_control.full_arm_validation_helpers import make_full_validation_files
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet


class FullScope(SimpleNamespace):
    def __iter__(self):
        yield self.device
        yield self.serial
        yield self.files
        yield self.leader
        yield self.positions
        yield self.clock


def make_full_scope(tmp_path, monkeypatch):
    pytest.importorskip("serial")
    files = make_full_validation_files(tmp_path)
    config, control = files
    clock = Clock()
    monkeypatch.setattr(backend.time, "perf_counter_ns", clock.perf_counter_ns)

    class Serial(FakeSerial):
        def __init__(self):
            super().__init__(clock)
            self.statuses = dict.fromkeys(range(1, 8), 0)
            self.positions = dict.fromkeys(range(1, 8), 0.0)
            self.targets = self.positions.copy()
            self.on_position = None
            self.failed_disables = set()
            self.block_first_disable = False
            self.disable_write_timeouts = []
            self.position_counts = dict.fromkeys(range(1, 8), 0)
            self.disable_deadlines = []

        def feedback(self, motor_id):
            position = round(
                (math.radians(self.positions[motor_id]) + self.mapping[21]) * 65535 / (2 * self.mapping[21])
            )
            return bridge_packet(motor_id, status=self.statuses[motor_id], position_uint=position)

        def write(self, packet):
            self._call("write")
            self.writes.append(packet)
            clock.now += 100_000
            can_id = int.from_bytes(packet[13:17], "little")
            payload = packet[21:29]
            if can_id == 0x7FF:
                motor_id, _, opcode, rid = payload[:4]
                if opcode == 0xCC:
                    reply = self.feedback(motor_id)
                else:
                    assert opcode == 0x33, "Validation must never configure modes or write registers"
                    value = (
                        (4 if motor_id == 7 else 2) if rid == 10 else 2000 if rid == 9 else self.mapping[rid]
                    )
                    reply = bridge_packet(
                        motor_id, payload=payload[:4] + struct.pack("<I" if rid in (9, 10) else "<f", value)
                    )
                self.pending.extend(reply)
                return len(packet)
            motor_id = can_id & 15
            if payload == b"\xff" * 7 + b"\xfd":
                self.disable_write_timeouts.append(self.write_timeout)
                if motor_id == 1 and self.block_first_disable:
                    clock.now += round(self.write_timeout * 1e9)
                    raise TimeoutError("Injected first disable consumes its whole write timeout")
                if motor_id in self.failed_disables:
                    raise TimeoutError("Injected disable write failure")
                self.statuses[motor_id] = 0
            elif payload == b"\xff" * 7 + b"\xfc":
                self.statuses[motor_id] = 1
                self.positions[motor_id] = self.targets[motor_id]
            else:
                assert can_id == (0x300 if motor_id == 7 else 0x100) + motor_id
                self.targets[motor_id] = math.degrees(struct.unpack("<f", payload[:4])[0])
                self.position_counts[motor_id] += 1
                if self.statuses[motor_id]:
                    self.positions[motor_id] = self.targets[motor_id]
                if self.on_position is not None:
                    self.on_position(motor_id)
            self.pending.extend(self.feedback(motor_id))
            return len(packet)

    serial = Serial()
    monkeypatch.setattr(readonly.serial, "Serial", lambda **_kwargs: serial)
    device = backend.B601ControlBackend(
        "FAKE_VALIDATION",
        readonly.DEFAULT_B601_JOINTS,
        control,
        lock_root=tmp_path / "locks",
        validation=config,
    )
    positions = dict.fromkeys(JOINT_NAMES, 0.0)
    sequence = 0

    def leader():
        nonlocal sequence
        sequence += 1
        return ValidationLeaderSample(
            tuple(positions.items()), clock.now, clock.now, sequence, config.leader_config_sha256
        )

    device.connect()
    return FullScope(
        device=device, serial=serial, files=files, leader=leader, positions=positions, clock=clock
    )


def close_full_scope(scope):
    if scope.device.has_resources:
        with suppress(Exception):
            scope.device.disconnect()
