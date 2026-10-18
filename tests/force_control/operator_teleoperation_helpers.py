"""Independent operator-mode fake serial fixtures; no diagnostic artifacts or hardware."""

import hashlib
import json
import math
import struct
import uuid
from contextlib import suppress
from types import SimpleNamespace

import pytest

from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport as readonly
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile
from lerobot.robots.rebot_b601_follower.operator_teleoperation import (
    OperatorLeaderSample,
    OperatorTeleoperationConfig,
)
from tests.force_control.hardware_helpers import Clock, FakeSerial, bridge_packet

JOINT_NAMES = tuple(spec.name for spec in readonly.DEFAULT_B601_JOINTS)


def make_operator_files(tmp_path):
    """Write only one native calibration, with explicit non-diagnostic test settings."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    calibration_path = tmp_path / "native_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                spec.name: {
                    "id": spec.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -200.0,
                    "range_max": 200.0,
                }
                for spec in readonly.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    calibration = B601ArmCalibration.load(
        calibration_path, {spec.name: spec.motor_id for spec in readonly.DEFAULT_B601_JOINTS}
    )
    profiles = tuple(
        B601JointCommandProfile(name, "mit", -150.0, 150.0, 10.0, kp=kp, kd=kd)
        for name, kp, kd in zip(JOINT_NAMES[:-1], (45, 45, 45, 8, 9, 8), (12, 12, 12, 1, 1, 1), strict=True)
    ) + (
        B601JointCommandProfile(
            "gripper", "force_pos", -150.0, 150.0, 10.0, torque_ratio=0.02, velocity_wire_code=1000
        ),
    )
    control = backend.B601ControlConfig(
        profiles=profiles,
        startup_budget_s=2.0,
        send_budget_s=0.04,
        fault_budget_s=0.5,
        feedback_max_age_s=0.1,
        fault_action="disable_selected",
        fault_joint_names=JOINT_NAMES,
        calibration_file=str(calibration_path),
        watchdog_ticks=9876,
    )
    configuration = OperatorTeleoperationConfig(
        session_id=str(uuid.uuid4()),
        port="FAKE_OPERATOR",
        calibration_file=str(calibration_path),
        calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        leader_config_sha256="a" * 64,
        reference_envelope_deg=100.0,
        alignment_tolerance_deg=12.0,
        preload_tolerance_deg=0.2,
        leader_max_age_s=0.25,
        max_command_gap_s=0.2,
        watchdog_ticks=9876,
        supported_arm=True,
        stop_policy="disable_all_supported_arm",
    )
    return SimpleNamespace(
        operator=configuration,
        configuration=configuration,
        control=control,
        calibration=calibration,
        calibration_path=calibration_path,
    )


def make_operator_scope(tmp_path, monkeypatch, *, files=None, connect=True):
    """Construct a single fake owner with fresh data; no motor or camera is opened."""
    pytest.importorskip("serial")
    files = make_operator_files(tmp_path) if files is None else files
    configuration, control = files.operator, files.control
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
            self.disable_write_timeouts = []
            self.position_counts = dict.fromkeys(range(1, 8), 0)
            self.mode_readback = {index: 4 if index == 7 else 2 for index in range(1, 8)}
            self.ignored_mode_writes = set()
            self.ignored_watchdog_writes = set()
            self.mode_write_echo = False
            self.mode_write_feedback = False
            self.partial_mode_write = False
            self.partial_watchdog_write = False
            self.watchdog_readback = control.watchdog_ticks
            self.watchdog_overrides = {}

        def feedback(self, motor_id):
            position = round(
                (math.radians(self.positions[motor_id]) + self.mapping[21]) * 65535 / (2 * self.mapping[21])
            )
            return bridge_packet(motor_id, status=self.statuses[motor_id], position_uint=position)

        def write(self, packet):
            self._call("write")
            self.writes.append(packet)
            clock.now += 100_000
            identifier = int.from_bytes(packet[13:17], "little")
            payload = packet[21:29]
            if identifier == 0x7FF:
                motor_id, _, opcode, register_id = payload[:4]
                if opcode == 0xCC:
                    reply = self.feedback(motor_id)
                elif opcode == 0x55:
                    assert register_id in (9, 10)
                    assert not any(self.statuses.values()), "Register preparation precedes every enable"
                    value = struct.unpack("<I", payload[4:])[0]
                    if register_id == 10:
                        assert value == (4 if motor_id == 7 else 1)
                        if motor_id not in self.ignored_mode_writes:
                            self.mode_readback[motor_id] = value
                    else:
                        assert value == control.watchdog_ticks
                        if motor_id not in self.ignored_watchdog_writes:
                            self.watchdog_overrides[motor_id] = value
                    if self.mode_write_echo:
                        self.pending.extend(bridge_packet(motor_id, payload=payload))
                    if self.mode_write_feedback:
                        self.pending.extend(self.feedback(motor_id))
                    partial = self.partial_mode_write if register_id == 10 else self.partial_watchdog_write
                    return len(packet) - int(partial)
                else:
                    assert opcode == 0x33
                    value = (
                        self.mode_readback[motor_id]
                        if register_id == 10
                        else self.watchdog_overrides.get(motor_id, self.watchdog_readback)
                        if register_id == 9
                        else self.mapping[register_id]
                    )
                    reply = bridge_packet(
                        motor_id,
                        payload=payload[:4] + struct.pack("<I" if register_id in (9, 10) else "<f", value),
                    )
                self.pending.extend(reply)
                return len(packet)
            motor_id = identifier & 15
            if payload == b"\xff" * 7 + b"\xfd":
                self.disable_write_timeouts.append(self.write_timeout)
                if motor_id in self.failed_disables:
                    raise TimeoutError("Injected disable write failure")
                self.statuses[motor_id] = 0
            elif payload == b"\xff" * 7 + b"\xfc":
                self.statuses[motor_id] = 1
                self.positions[motor_id] = self.targets[motor_id]
            else:
                assert identifier == (0x300 + motor_id if motor_id == 7 else motor_id)
                self.targets[motor_id] = (
                    math.degrees(struct.unpack("<f", payload[:4])[0])
                    if motor_id == 7
                    else math.degrees(
                        int.from_bytes(payload[:2], "big") * 2 * self.mapping[21] / 65535 - self.mapping[21]
                    )
                )
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
        configuration.port,
        readonly.DEFAULT_B601_JOINTS,
        control,
        lock_root=tmp_path / "locks",
        operator_teleoperation=configuration,
    )
    positions = dict.fromkeys(JOINT_NAMES, 0.0)
    sequence = 0

    def leader():
        nonlocal sequence
        sequence += 1
        return OperatorLeaderSample(
            tuple(positions.items()), clock.now, clock.now, sequence, configuration.leader_config_sha256
        )

    if connect:
        device.connect()
    return SimpleNamespace(
        device=device, serial=serial, files=files, leader=leader, positions=positions, clock=clock
    )


def close_operator_scope(scope):
    if scope.device.has_resources:
        with suppress(Exception):
            scope.device.disconnect()
