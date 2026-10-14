"""Disabled gripper startup tests; all serial construction uses test doubles."""

import json
import struct
from contextlib import suppress
from dataclasses import replace
from unittest.mock import Mock

import pytest

from lerobot.robots.rebot_b601_follower import control_backend as backend, readonly_transport
from lerobot.robots.rebot_b601_follower.control_protocol import encode_disable
from tests.force_control.hardware_helpers import (
    Clock,
    PreparationSerial,
    commands,
    parameter_writes,
    profile,
    snapshot,
)


@pytest.fixture
def prepared_backend(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(backend.time, "perf_counter_ns", clock.perf_counter_ns)
    fake = PreparationSerial(clock)
    factory = Mock(return_value=fake)
    monkeypatch.setattr(readonly_transport.serial, "Serial", factory)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {"gripper": {"id": 7, "drive_mode": 0, "homing_offset": 0, "range_min": -30, "range_max": 30}}
        ),
        encoding="utf-8",
    )
    config = backend.B601ControlConfig(
        (profile(),),
        0.5,
        0.005,
        0.005,
        0.02,
        "disable_gripper",
        io_event_capacity=1024,
        calibration_file=str(calibration),
    )
    device = backend.B601ControlBackend(
        "FAKE_PREPARATION", readonly_transport.DEFAULT_B601_JOINTS[-1:], config, lock_root=tmp_path
    )
    yield device, fake, clock, factory
    fake.close_fails = False
    fake.interrupt_motor = None
    if device.has_resources:
        with suppress(RuntimeError):
            device.disconnect()


@pytest.mark.parametrize(
    "mode,watchdog,reset_timeout,echo,expected",
    [
        (2, 0, False, False, [10, 9]),
        (4, 2000, False, True, []),
        (1, 2000, True, True, [10, 9]),
        (4, 0, False, True, [9]),
        (3, 2000, True, True, [10, 9]),
    ],
)
def test_startup_prepares_mode_then_timeout_before_enable(
    prepared_backend, mode, watchdog, reset_timeout, echo, expected
):
    device, fake, clock, factory = prepared_backend
    fake.mode, fake.watchdog = mode, watchdog
    fake.mode_resets_watchdog, fake.parameter_echo = reset_timeout, echo
    device.connect()
    assert not parameter_writes(fake) and not commands(fake)
    guards = []
    wire = device.start_gripper(snapshot(device, clock), lambda: guards.append(clock.now))
    assert factory.call_count == 1
    assert [p[24] for p in parameter_writes(fake)] == expected
    assert len(guards) == 5 + len(expected)
    receipt = device.startup_receipt
    assert receipt["complete"] and receipt["hardware_ack"] == "unknown"
    assert {r["register_id"]: r["value"] for r in receipt["final_registers"]} == {10: 4, 9: 2000}
    control = commands(fake)
    assert control[0] == encode_disable(7)
    assert [int.from_bytes(p[13:17], "little") for p in control] == [7, 0x307, 7, 0x307]
    assert wire == receipt["wire_position_deg"]
    assert all(r["hardware_ack"] == "unknown" for r in receipt["commands"])
    device.disconnect()
    assert commands(fake)[-1] == encode_disable(7) and not device.has_resources
    with pytest.raises(RuntimeError, match="fresh disabled"):
        device.start_gripper(device._latest_snapshot, lambda: None)


def test_disabled_velocity_mode_transitions_in_order_and_uses_only_force_pos(prepared_backend):
    from lerobot.motors.damiao.tables import ControlMode

    assert ControlMode.VEL == 3 and ControlMode.TORQUE_POS == 4
    device, fake, clock, _ = prepared_backend
    fake.mode, fake.watchdog = int(ControlMode.VEL), 0
    device.connect()
    device.start_gripper(snapshot(device, clock), lambda: None)
    startup_writes = [
        packet
        for packet in fake.writes
        if packet[23] == 0x55 or int.from_bytes(packet[13:17], "little") != 0x7FF
    ]
    assert [packet[24] for packet in startup_writes[:2]] == [10, 9]
    assert struct.unpack("<I", startup_writes[0][25:29])[0] == 4
    assert struct.unpack("<I", startup_writes[1][25:29])[0] == 2000
    assert startup_writes[2] == encode_disable(7)
    assert [int.from_bytes(packet[13:17], "little") for packet in startup_writes[3:]] == [
        0x307,
        7,
        0x307,
    ]
    assert startup_writes[4][21:29] == b"\xff" * 7 + b"\xfc"
    assert not any(int.from_bytes(packet[13:17], "little") == 0x207 for packet in fake.writes)
    assert device.startup_receipt["original_registers"][0]["value"] == 3
    assert {row["register_id"]: row["value"] for row in device.startup_receipt["final_registers"]} == {
        10: 4,
        9: 2000,
    }


@pytest.mark.parametrize("mode", [0, 5, 255, 2**32 - 1])
def test_unknown_initial_mode_never_writes_startup_or_enables(prepared_backend, mode):
    device, fake, clock, _ = prepared_backend
    fake.mode = mode
    device.connect()
    with pytest.raises(RuntimeError, match=f"Unsupported initial CTRL_MODE={mode}"):
        device.start_gripper(snapshot(device, clock), lambda: None)
    assert not parameter_writes(fake) and not commands(fake)
    assert not device.startup_receipt["complete"]
    assert device.fault_receipt["status"] == "not_attempted"


@pytest.mark.parametrize("guard_call", [1, 2, 3, 4, 5, 6])
def test_disabled_velocity_conversion_checks_guard_before_enabling(prepared_backend, guard_call):
    device, fake, clock, _ = prepared_backend
    fake.mode, fake.watchdog = 3, 0
    device.connect()
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == guard_call:
            raise RuntimeError("user_stop_during_startup")

    with pytest.raises(RuntimeError, match="user_stop_during_startup"):
        device.start_gripper(snapshot(device, clock), guard)
    assert calls == guard_call and not device.startup_receipt["complete"]
    assert not any(packet[21:29] == b"\xff" * 7 + b"\xfc" for packet in fake.writes)
    assert not any(int.from_bytes(packet[13:17], "little") == 0x207 for packet in fake.writes)
    assert sum(packet == encode_disable(7) for packet in commands(fake)) <= 2


def test_watchdog_and_settle_interval_are_explicit_config(prepared_backend):
    device, fake, clock, _ = prepared_backend
    device.config = replace(device.config, watchdog_ticks=7777, parameter_write_settle_s=0)
    device.connect()
    device.start_gripper(snapshot(device, clock), lambda: None)
    assert fake.watchdog == 7777
    assert device.startup_receipt["watchdog_ticks"] == 7777
    assert device.startup_receipt["parameter_write_settle_s"] == 0


@pytest.mark.parametrize("condition", ["enabled", "fault", "mapping", "calibration"])
def test_bad_initial_state_never_configures_or_enables(prepared_backend, condition):
    device, fake, _, _ = prepared_backend
    if condition == "enabled":
        fake.status = 1
    elif condition == "fault":
        fake.status = 9
    elif condition == "mapping":
        fake.mapping[22] = 0.01
    else:
        device.config = replace(device.config, calibration_file=None)
    with pytest.raises((RuntimeError, ValueError)):
        device.connect()
        device.start_gripper(device._latest_snapshot, lambda: None)
    assert not parameter_writes(fake) and not commands(fake)


@pytest.mark.parametrize(
    "failure",
    [
        "partial_parameter",
        "late_parameter",
        "interrupt_parameter",
        "ignore_parameter",
        "enable_after_parameter",
        "corrupt_echo",
    ],
)
def test_ambiguous_preparation_never_enables_or_retries(prepared_backend, failure):
    device, fake, clock, _ = prepared_backend
    setattr(fake, failure, True)
    device.connect()
    with pytest.raises((RuntimeError, OSError, TimeoutError, KeyboardInterrupt)):
        device.start_gripper(snapshot(device, clock), lambda: None)
    assert not device.startup_receipt["complete"]
    assert len(parameter_writes(fake)) == 1
    assert not any(p[21:29] == b"\xff" * 7 + b"\xfc" for p in fake.writes)
    assert commands(fake) == [encode_disable(7)]
    count = len(fake.writes)
    device.emergency_stop("again")
    with suppress(RuntimeError):
        device.disconnect()
    assert len(fake.writes) == count and not device.has_resources


@pytest.mark.parametrize("guard_call", [1, 2, 3, 4, 5, 6, 7])
def test_guard_checks_fresh_force_stop_before_every_write(prepared_backend, guard_call):
    device, fake, clock, _ = prepared_backend
    device.connect()
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        if calls == guard_call:
            raise RuntimeError("fresh force/stop guard")

    with pytest.raises(RuntimeError, match="guard"):
        device.start_gripper(snapshot(device, clock), guard)
    assert calls == guard_call
    assert not device.startup_receipt["complete"]
    if guard_call < 6:
        assert not any(p[21:29] == b"\xff" * 7 + b"\xfc" for p in fake.writes)
    assert sum(p == encode_disable(7) for p in commands(fake)) <= 2  # One startup probe plus one cleanup.


def test_missing_startup_reply_stops_before_enable(prepared_backend):
    device, fake, clock, _ = prepared_backend
    fake.auto_command_feedback = False
    device.connect()
    with pytest.raises(TimeoutError):
        device.start_gripper(snapshot(device, clock), lambda: None)
    assert not any(p[21:29] == b"\xff" * 7 + b"\xfc" for p in fake.writes)
    assert device.fault_receipt["status"] == "unknown"


@pytest.mark.parametrize("value", [0, -1, True, 2**32, 1.5])
def test_watchdog_invalid_values_reject_before_io(prepared_backend, value):
    device, fake, _, factory = prepared_backend
    device.config = replace(device.config, watchdog_ticks=value)
    with pytest.raises(ValueError):
        device.connect()
    factory.assert_not_called()
    assert not fake.writes
