"""Shared deterministic hardware doubles; no test-module imports or device I/O."""

import struct
import threading

from lerobot.force_control.fake_hardware import FakeForceSensor
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile


class Clock:
    """Deterministic host clock for deadline and packet receive tests."""

    def __init__(self):
        self.now = 1_000_000_000

    def perf_counter_ns(self):
        return self.now


def bridge_packet(motor_id=1, *, payload=None, feedback_id=None, status=0, position_uint=32768):
    """Build an RX fixture independently of the production query encoder."""
    if payload is None:
        payload = bytes((status << 4 | motor_id, position_uint >> 8, position_uint & 255, 128, 0, 0, 26, 27))
    return b"\xaa\x11\x08" + struct.pack("<I", feedback_id or (0x10 + motor_id)) + payload + b"\x55"


class FakeSerial:
    """Release replies only after queries and retain every operation's thread."""

    def __init__(self, clock):
        self.clock = clock
        self.pending = bytearray()
        self.writes = []
        self.calls = []
        self.fragment_size = 999
        self.mapping = {21: 8.0, 22: 14.0, 23: 9.0}
        self.feedback_transform = lambda packet, motor_id: packet
        self.register_transform = lambda packet, motor_id, rid: packet
        self.partial_write = False
        self.late_read = False
        self.close_fails = False
        self.open_fails = False
        self.reset_fails = False
        self.opened = False
        self.timeout = 0.002
        self.write_timeout = 0.02

    def _call(self, name):
        self.calls.append((name, threading.get_ident()))

    def open(self):
        self._call("open")
        self.opened = True
        if self.open_fails:
            raise OSError("open failed")

    def close(self):
        self._call("close")
        if self.close_fails:
            raise OSError("close failed")
        self.opened = False

    def reset_input_buffer(self):
        self._call("reset_input_buffer")
        if self.reset_fails:
            raise OSError("reset failed")
        self.pending.clear()

    def write(self, packet):
        self._call("write")
        self.writes.append(packet)
        self.clock.now += 100_000
        for offset in range(0, len(packet), 30):
            frame = packet[offset : offset + 30]
            assert len(frame) == 30
            payload = frame[21:29]
            motor_id, _, command, rid = payload[:4]
            if command == 0x33:
                reply = bridge_packet(
                    motor_id, payload=bytes((motor_id, 0, 0x33, rid)) + struct.pack("<f", self.mapping[rid])
                )
                reply = self.register_transform(reply, motor_id, rid)
            else:
                assert command == 0xCC
                reply = self.feedback_transform(bridge_packet(motor_id), motor_id)
            self.pending.extend(reply)
        return len(packet) - int(self.partial_write)

    @property
    def in_waiting(self):
        return min(self.fragment_size, len(self.pending))

    def read(self, size):
        self._call("read")
        self.clock.now += 100_000_000 if self.late_read else 100_000
        data = bytes(self.pending[:size])
        del self.pending[:size]
        return data


def profile(name="gripper", mode="force_pos"):
    """Return test-only values, never an example hardware calibration profile."""
    common = {"name": name, "mode": mode, "minimum_deg": -20, "maximum_deg": 20, "max_relative_deg": 2}
    if mode == "mit":
        return B601JointCommandProfile(**common, kp=8, kd=0.3)
    return B601JointCommandProfile(
        **common, velocity_deg_s=10, torque_ratio=0.07 if mode == "force_pos" else None
    )


class ControlSerial(FakeSerial):
    """Accept control packets and register reads on exactly the query serial instance."""

    def __init__(self, clock):
        super().__init__(clock)
        self.mode = 4
        self.status = 0
        self.watchdog = 2000
        self.auto_command_feedback = True
        self.fail_motor = None
        self.write_delay_ns = 0
        self.interrupt_motor = None
        self.feedback_transform = lambda packet, motor_id: bridge_packet(motor_id, status=self.status)

    def write(self, packet):
        arbitration_id = int.from_bytes(packet[13:17], "little")
        payload = packet[21:29]
        if arbitration_id == 0x7FF:
            if payload[2] == 0x55 and payload[3] in (9, 10):
                self._call("write")
                self.writes.append(packet)
                if payload[3] == 10:
                    self.mode = struct.unpack("<I", payload[4:])[0]
                else:
                    self.watchdog = struct.unpack("<I", payload[4:])[0]
                self.clock.now += 100_000
                return len(packet)
            if payload[2] != 0x33 or payload[3] not in (9, 10):
                return super().write(packet)
            self._call("write")
            self.writes.append(packet)
            motor_id = payload[0]
            value = self.mode if payload[3] == 10 else self.watchdog
            self.pending.extend(bridge_packet(motor_id, payload=payload[:4] + struct.pack("<I", value)))
            self.clock.now += 100_000
            return len(packet)
        self._call("write")
        self.writes.append(packet)
        self.clock.now += self.write_delay_ns
        if arbitration_id & 15 == self.interrupt_motor:
            raise KeyboardInterrupt("interrupt after transport entry")
        if arbitration_id & 15 == self.fail_motor:
            return 17
        if payload == b"\xff" * 7 + b"\xfc":
            self.status = 1
        elif payload == b"\xff" * 7 + b"\xfd":
            self.status = 0
        if self.auto_command_feedback:
            self.pending.extend(
                self.feedback_transform(bridge_packet(arbitration_id & 15), arbitration_id & 15)
            )
        return len(packet)


def snapshot(device, clock, names=("gripper",)):
    """Request fresh feedback with an absolute test deadline."""
    return device.read_joint_snapshot(names, deadline_ns=clock.now + 5_000_000)


def commands(fake):
    """Exclude register/feedback reads when checking motion isolation."""
    return [p for p in fake.writes if int.from_bytes(p[13:17], "little") != 0x7FF]


class PreparationSerial(ControlSerial):
    """Model parameter side effects independently of the production packet encoder."""

    def __init__(self, clock):
        super().__init__(clock)
        self.mode = 2
        self.status = 0
        self.watchdog = 0
        self.parameter_echo = False
        self.corrupt_echo = False
        self.partial_parameter = False
        self.late_parameter = False
        self.interrupt_parameter = False
        self.ignore_parameter = False
        self.enable_after_parameter = False
        self.mode_resets_watchdog = False

    def write(self, packet):
        payload = packet[21:29]
        can_id = int.from_bytes(packet[13:17], "little")
        if can_id == 0x7FF and payload[2] == 0x55:
            assert payload[:2] == b"\x07\x00" and payload[3] in (9, 10)
            self._call("write")
            self.writes.append(packet)
            self.clock.now += 20_000_000 if self.late_parameter else 100_000
            if self.interrupt_parameter:
                raise KeyboardInterrupt("parameter interrupted")
            if self.partial_parameter:
                return 17
            value = struct.unpack("<I", payload[4:])[0]
            if not self.ignore_parameter:
                if payload[3] == 9:
                    self.watchdog = value
                else:
                    self.mode = value
                    if self.mode_resets_watchdog:
                        self.watchdog = 0
            if self.enable_after_parameter:
                self.status = 1
            if self.parameter_echo:
                self.pending.extend(bridge_packet(7, payload=bytes((7, 0, 0x33, payload[3])) + payload[4:]))
            if self.corrupt_echo:
                self.pending.extend(b"\xfe\xfd")
            return len(packet)
        if can_id == 0x7FF and payload[2:4] == b"\x33\x09":
            self._call("write")
            self.writes.append(packet)
            self.pending.extend(bridge_packet(7, payload=payload[:4] + struct.pack("<I", self.watchdog)))
            self.clock.now += 100_000
            return len(packet)
        return super().write(packet)


def parameter_writes(fake):
    return [p for p in fake.writes if p[23] == 0x55]


class ManualSensor(FakeForceSensor):
    @property
    def is_connected(self):
        return True

    def disconnect(self):
        pass


class MemoryTrace:
    run_uid = "test-run"
    start_sequence = 0

    def __init__(self):
        self.rows = []
        self.failed = False

    def check_health(self):
        if self.failed:
            raise RuntimeError("injected_trace_failure")

    def submit(self, row):
        self.rows.append(row.copy())
        return not self.failed
