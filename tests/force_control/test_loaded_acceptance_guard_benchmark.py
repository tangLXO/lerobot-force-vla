"""Structural guard assertions with fake force history and disconnected B601 instances."""

import importlib
import json
import os
from pathlib import Path

import pytest

from lerobot.robots.rebot_b601_follower import readonly_transport
from lerobot.robots.rebot_b601_follower.control_protocol import (
    encode_enable,
    encode_position,
    encode_timeout_write,
)
from tests.force_control import test_recording_arm_acceptance_cli as acceptance_cli

cli = acceptance_cli.cli
configuration = acceptance_cli.configuration


@pytest.fixture
def benchmark(cli, monkeypatch):
    monkeypatch.setattr(
        readonly_transport.serial,
        "Serial",
        lambda *args, **kwargs: pytest.fail("Host benchmark opened a serial transport"),
    )
    return importlib.import_module("benchmark_loaded_acceptance_guard")


@pytest.fixture
def loaded(benchmark, configuration, tmp_path):
    with benchmark.loaded_context(configuration[0], tmp_path / "context", warmup_s=0.02) as context:
        yield context


def test_rx_guard_never_reads_file_contents_but_keeps_real_force_checks(loaded, monkeypatch):
    reads = []
    original_open = Path.open

    def deny(*args, **kwargs):
        pytest.fail("RX-only guard attempted to read file contents")

    def open_checked(path, mode="r", *args, **kwargs):
        if "r" in mode or "+" in mode:
            reads.append(str(path))
            deny()
        return original_open(path, mode, *args, **kwargs)

    before = loaded.guard.count
    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_bytes", deny)
        patched.setattr(Path, "read_text", deny)
        patched.setattr(Path, "open", open_checked)
        for _ in range(30):
            loaded.device._acceptance_wait_guard(loaded.guard)
    assert loaded.guard.count == before + 30 and reads == []
    assert not loaded.device.has_resources


def test_emulated_new_step_reads_calibration_three_times_and_old_step_thirty_three(benchmark, loaded):
    result = benchmark.structural_counts(loaded)
    assert result == {
        "rx_only_30_chunks_content_reads": 0,
        "new_step_3_hypothetical_writes_plus_30_chunks_content_reads": 3,
        "old_step_3_hypothetical_writes_plus_30_chunks_content_reads": 33,
        "valid": True,
    }


def test_every_actual_write_entry_validates_content_before_transport_access(loaded, monkeypatch):
    device = loaded.device
    device._control_session = device._motion_session = True
    monkeypatch.setattr(device, "_check_owner", lambda: None)
    reads = []
    original = Path.read_bytes

    def read(path):
        if path == Path(loaded.calibration.path):
            reads.append(path)
        return original(path)

    def before_transport(deadline):
        raise RuntimeError("stopped before transport access")

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(device, "_remaining", before_transport)
    profile = device.config.profiles[0]
    motor_id = device._specs[profile.name].motor_id
    packets = (
        encode_enable(motor_id),
        encode_timeout_write(motor_id, device.config.watchdog_ticks),
        encode_position(motor_id, 0.01, 0.0, profile, (12.5, 30.0, 10.0)).packet,
    )
    for index, packet in enumerate(packets, start=1):
        device._acceptance_allowed_packet = packet
        with pytest.raises(RuntimeError, match="stopped before transport access"):
            device._write_once(packet, 10**30)
        assert len(reads) == index
    assert not device.has_resources


def test_changed_calibration_with_same_mtime_still_blocks_actual_write(loaded, monkeypatch):
    device = loaded.device
    source = Path(loaded.calibration.path)
    before = source.stat()
    data = source.read_text(encoding="utf-8")
    source.write_text(data.replace('"range_min": -30', '"range_min": -31', 1), encoding="utf-8")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    # The receive callback is deliberately metadata-only. The write entry below
    # must independently catch a same-mtime content change before transport access.
    device._acceptance_wait_guard(loaded.guard)
    device._control_session = device._motion_session = True
    monkeypatch.setattr(device, "_check_owner", lambda: None)
    packet = encode_enable(device._specs[device.config.profiles[0].name].motor_id)
    device._acceptance_allowed_packet = packet
    with pytest.raises(ValueError, match="calibration digest changed"):
        device._write_once(packet, 10**30)
    assert not device.has_resources


def test_short_benchmark_reports_only_host_cost_and_no_hardware(benchmark, configuration, tmp_path):
    output = tmp_path / "report"
    result = benchmark.run_benchmark(
        configuration[1], output, warmup_s=0.02, iterations=2, step_repetitions=2
    )
    assert result["complete"] and not result["hardware_acceptance"]
    assert result["serial_opened"] is False and result["motor_commands_sent"] == 0
    assert result["physical_sensor_opened"] is False and result["fake_sensor_requested_hz"] == 400
    assert result["force_max_age_ms"] == 50
    assert result["command_feedback_budget_ms"] == configuration[0].control.feedback_max_age_s * 1000
    assert result["structural"]["valid"] and result["cleanup"]["fake_sensor_closed"]
    assert json.loads((output / "benchmark.json").read_text(encoding="utf-8"))["complete"]
