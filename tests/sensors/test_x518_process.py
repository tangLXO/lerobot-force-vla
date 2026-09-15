# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Real spawn, loopback Modbus, causal handoff, and failure lifecycle checks."""

import json
import multiprocessing
import queue
import threading
import time

import draccus
import pytest

from lerobot.sensors import SensorConfig
from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig
from lerobot.sensors.x518.acquisition import advance_deadline
from lerobot.sensors.x518.diagnostics import AcquisitionDiagnostics, summarize_capture
from lerobot.utils.errors import DeviceNotConnectedError
from tests.sensors.x518_fake_server import ModbusServer


def config(server, **kwargs):
    return X518SensorConfig(
        host="127.0.0.1",
        port=server.port,
        expected_unit=None,
        sample_rate_hz=400,
        expected_sample_rate_hz=400,
        channels={
            "left.normal_force": X518ChannelConfig(channel=1),
            "right.normal_force": X518ChannelConfig(channel=2),
        },
        **kwargs,
    )


def until(predicate, timeout=5):
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() >= deadline:
            raise AssertionError("condition timed out")
        time.sleep(0.005)


def drain(subscription):
    rows = []
    while True:
        try:
            rows.append(subscription.get_nowait())
        except queue.Empty:
            return rows


def test_default_process_config_and_validation():
    payload = {"type": "x518", "channels": {"left.normal_force": {"channel": 1}}}
    decoded = draccus.decode(SensorConfig, payload)
    assert decoded.acquisition_mode == "process"
    assert decoded.acquisition_queue_capacity == 1024
    for kwargs in (
        {"acquisition_mode": "auto"},
        {"acquisition_queue_capacity": 0},
        {"acquisition_queue_capacity": True},
    ):
        with pytest.raises(ValueError):
            X518SensorConfig(channels=decoded.channels, **kwargs)


def test_deadlines_skip_expired_slots_without_catch_up():
    assert advance_deadline(0, 1, 10) == (10, 0)
    assert advance_deadline(0, 10, 10) == (20, 1)
    assert advance_deadline(0, 36, 10) == (40, 3)


@pytest.mark.parametrize("mode", ["process", "thread"])
def test_socket_acquisition_and_complete_subscription_boundaries(mode):
    with ModbusServer() as server:
        sensor = X518Sensor(config(server, acquisition_mode=mode))
        sensor.connect()
        try:
            for _ in range(2):
                lease = sensor.acquire_recorder()
                sub = sensor.subscribe(1000)
                rows = [sub.get(timeout=2) for _ in range(20)]
                sensor.unsubscribe(sub)
                rows.extend(drain(sub))
                assert [r.sequence for r in rows] == list(range(sub.start_sequence, sub.end_sequence))
                assert all(r.arrival_timestamp_ns >= r.timestamp_ns for r in rows)
                assert all(r.hardware_sequence is None for r in rows)
                assert all(
                    r.values["left.normal_force"] == r.native_values["channel_1.register"] for r in rows
                )
                sensor.release_recorder(lease)
            assert sensor.diagnostics["invalid_attempts"] == 0
            assert sensor.diagnostics["transport_gaps"] == 0
        finally:
            sensor.disconnect()
        assert not sensor.has_resources
        sensor.connect()
        sensor.disconnect()
        assert not sensor.has_resources


def test_delayed_parent_publication_cannot_leak_to_earlier_frame():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        try:
            sensor.async_read(2000)
            with sensor._publication_condition:
                before = time.perf_counter_ns()
                count = server.responses
                until(lambda: server.responses > count + 12)
                anchor = time.perf_counter_ns()
            until(lambda: sensor.read_latest().timestamp_ns > anchor)
            samples = sensor._history.snapshot()
            delayed = [s for s in samples if before < s.timestamp_ns <= anchor]
            assert delayed
            assert all(s.arrival_timestamp_ns > anchor for s in delayed)
            selected = sensor.read_latest_before(anchor, max_age_ms=1000)
            assert selected.timestamp_ns <= before
            assert selected.arrival_timestamp_ns <= anchor
        finally:
            sensor.disconnect()


def test_unsubscribe_waits_for_ipc_tail():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        sub = sensor.subscribe(1000)
        try:
            with sensor._publication_condition:
                count = server.responses
                until(lambda: server.responses > count + 15)
                last_acquired = server.responses
            sensor.unsubscribe(sub)
            rows = drain(sub)
            assert rows[-1].native_values["channel_1.register"] >= last_acquired
            assert [s.sequence for s in rows] == list(range(sub.start_sequence, sub.end_sequence))
        finally:
            sensor.unsubscribe(sub)
            sensor.disconnect()


def test_ipc_overflow_is_fatal_even_when_sample_channel_is_full():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server, acquisition_queue_capacity=16))
        sensor.connect()
        lease = sensor.acquire_recorder()
        try:
            with sensor._publication_condition:
                time.sleep(0.2)
            until(lambda: sensor._fault is not None)
            assert "overflow" in sensor._fault
            assert sensor.diagnostics["ipc_overflow_count"] == 1
            assert lease.error is not None
        finally:
            sensor.release_recorder(lease)
            sensor.disconnect()
        assert not sensor.has_resources


def test_required_stream_does_not_reconnect_and_fault_cleanup_releases_handles():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        lease = sensor.acquire_recorder()
        try:
            server.fail.set()
            until(lambda: lease.error is not None)
            assert "needs reconnect" in lease.error
            assert server.connections == 1
            with pytest.raises(RuntimeError):
                sensor.read_latest()
        finally:
            sensor.release_recorder(lease)
            sensor.disconnect()
        assert not sensor.has_resources


def test_process_death_latches_lease_before_next_read():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        lease = sensor.acquire_recorder()
        try:
            sensor._process.terminate()
            sensor._process.join(2)
            with pytest.raises(RuntimeError):
                sensor._check_recorder_fault()
            assert lease.error is not None
            assert sensor.has_resources and not sensor.is_connected
        finally:
            sensor.release_recorder(lease)
            sensor.disconnect()
        assert not sensor.has_resources


def test_startup_error_has_no_implicit_thread_fallback_or_process_leak():
    with ModbusServer() as server:
        server.unit = 99
        sensor = X518Sensor(config(server, connect_retries=0))
        with pytest.raises(RuntimeError, match="unit code"):
            sensor.connect()
        assert not sensor.has_resources
        assert not any(
            child.name.startswith("X518Acquisition") for child in multiprocessing.active_children()
        )


def test_shutdown_interrupts_io_and_wakes_blocked_read():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server, request_timeout_s=2))
        sensor.connect()
        sensor.async_read(2000)
        server.hold.set()
        time.sleep(0.02)
        sensor._last_consumed_sequence = sensor._next_sequence
        errors = []

        def read():
            try:
                sensor.read()
            except Exception as exc:
                errors.append(exc)

        reader = threading.Thread(target=read)
        reader.start()
        sensor.disconnect()
        reader.join(2)
        assert not reader.is_alive()
        assert isinstance(errors[0], DeviceNotConnectedError)
        assert not sensor.has_resources


def test_duplicate_and_missing_transport_sequences_are_rejected():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor._transport_stats["transport_gaps"] = 0
        sensor._transport_stats["ipc_peak_size"] = 0
        components = {"timestamp_ns": 1, "values": {"left.normal_force": 1, "right.normal_force": 2}}
        sensor._receive_message("sample", (0, components, 0, 1))
        for sequence in (0, 2):
            with pytest.raises(RuntimeError, match="sequence mismatch"):
                sensor._receive_message("sample", (sequence, components, 0, 1))
        assert sensor._next_sequence == 1


def test_faulted_barrier_cannot_create_an_unowned_subscription():
    from unittest.mock import Mock

    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        callback = Mock()
        operation = {"callback": callback, "event": threading.Event(), "error": None}
        sensor._barriers["pending"] = operation
        sensor._set_fault("transport failed")
        sensor._receive_message("barrier", "pending")
        sensor._barriers.clear()
        sensor._receive_message("barrier", "pending")
        assert operation["event"].is_set()
        assert operation["error"] is not None
        callback.assert_not_called()


def test_completed_barrier_keeps_result_ownership_on_later_fault():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        operation = {"callback": lambda: "subscription", "event": threading.Event(), "error": None}
        sensor._barriers["complete"] = operation
        sensor._receive_message("barrier", "complete")
        sensor._set_fault("later failure")
        assert operation["result"] == "subscription"
        assert operation["error"] is None
        assert sensor._fault == "later failure"


def test_low_rate_warns_without_fault_and_statistics_remain_bounded(caplog):
    from lerobot.sensors import SensorSample

    stats = AcquisitionDiagnostics()
    for i in range(16000):
        stats.observe(SensorSample(timestamp_ns=i * 5_000_000, sequence=i), 400)
    assert len(stats.intervals) == 4096
    assert "recording retained" in caplog.text
    assert stats.snapshot()["response_rate_hz"] == 200
    warnings = [record for record in caplog.records if "recording retained" in record.message]
    assert len(warnings) == 3  # at 10, 40 and 70 seconds, no faster than every 30 seconds


def test_reconnect_outside_recording_refreshes_settings():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        try:
            sensor.async_read(2000)
            server.unit = 2
            server.fail.set()
            until(lambda: server.connections >= 2)
            server.fail.clear()
            until(lambda: sensor._device_settings.unit == "kg")
            sample = sensor.async_read(2000)
            assert sample.values["left.normal_force"] == pytest.approx(
                sample.native_values["channel_1.register"] * 9.80665
            )
            assert sensor._fault is None
        finally:
            sensor.disconnect()


def test_recorder_start_during_reconnect_cannot_confirm_healthy_boundary():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server, request_timeout_s=0.2, reconnect_delay_s=1))
        sensor.connect()
        try:
            sensor.async_read(2000)
            server.hold_settings.set()
            server.fail.set()
            until(lambda: server.connections >= 2)
            with pytest.raises(RuntimeError, match="reconnect"):
                sensor.acquire_recorder()
            assert not sensor._recorder_leases
            assert sensor._fault is not None
        finally:
            server.hold_settings.clear()
            sensor.disconnect()


def test_spawn_sidecar_commit_two_episodes_and_full_readonly_verification(tmp_path):
    pytest.importorskip("datasets", reason="Sidecar integration requires lerobot[dataset]")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sensor_stream import SensorStreamRecorder
    from lerobot.datasets.sensor_window import SensorStreamReader
    from lerobot.robots.sensorized_robot import SensorizedRobot
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
    from tests.robots.test_sensorized_robot import FakeRobot

    with ModbusServer() as server:
        # This fixture's server shares the parent's GIL with dataset finalization.
        # Its request budget must tolerate lazy imports during the first save.
        sensor = X518Sensor(config(server, max_age_ms=100, request_timeout_s=2))
        robot = SensorizedRobot(FakeRobot(), {"gripper_force": sensor})
        features = robot.route_observation_dataset_features(
            hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
        )
        root = tmp_path / "dataset"
        dataset = LeRobotDataset.create("test/spawn", fps=30, root=root, features=features)
        recorder = SensorStreamRecorder(root, robot.sensors)
        sensor.connect()
        try:
            for episode in range(2):
                recorder.start_episode(episode, dataset=dataset)
                recorder.wait_until_ready()
                for frame in range(4):
                    obs = robot.get_observation()
                    dataset.add_frame({**build_dataset_frame(features, obs, "observation"), "task": "test"})
                    recorder.record_sync(frame, robot.last_capture_metadata)
                    time.sleep(0.01)
                recorder.save_episode(dataset)
                assert not sensor._recorder_leases
                assert sensor.subscriber_count == 0
        finally:
            recorder.close()
            sensor.disconnect()
            dataset.finalize()
        reopened = LeRobotDataset("test/spawn", root=root)
        before = {p.relative_to(root): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}
        reader = SensorStreamReader(root, instance="gripper_force", verify="full")
        for episode in range(2):
            raw = {row["sequence"]: row for row in reader.read_raw(episode_index=episode)}
            sync = reader.read_sync(episode_index=episode)
            for frame, row in enumerate(sync):
                sample = raw[row["sensors"]["gripper_force"]["sequence"]]
                assert sample["arrival_timestamp_ns"] <= row["frame_anchor_ns"]
                assert reopened[episode * 4 + frame]["observation.tactile"].tolist() == pytest.approx(
                    [sample["values"]["left.normal_force"], sample["values"]["right.normal_force"]]
                )
        after = {p.relative_to(root): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}
        assert after == before


def test_capture_report_clips_to_frames_and_serializes_native_json():
    rows = [
        {
            "timestamp_ns": i * 2_500_000,
            "arrival_timestamp_ns": i * 2_500_000 + 100_000,
            "sequence": i,
            "is_valid": True,
        }
        for i in range(20)
    ]
    result = summarize_capture(rows, start_ns=5_000_000, end_ns=25_000_000)
    assert result["sample_count"] == 9
    assert result["response_rate_hz"] == 400
    assert json.loads(json.dumps(result))["passed"] is True
    rows[6]["sequence"] = 0
    assert not summarize_capture(rows)["passed"]


def test_process_failure_prevents_commit_and_releases_recorder(tmp_path):
    pytest.importorskip("pyarrow", reason="Recorder integration requires lerobot[dataset]")
    from lerobot.datasets.sensor_stream import SensorRecorderError, SensorStreamRecorder

    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        recorder = SensorStreamRecorder(tmp_path / "failed", {"gripper_force": sensor})
        try:
            recorder.start_episode(0)
            recorder.wait_until_ready()
            sensor._process.terminate()
            sensor._process.join(2)
            with pytest.raises(SensorRecorderError):
                recorder.prepare_episode()
            assert not sensor._recorder_leases
            assert sensor.subscriber_count == 0
            journals = list((recorder.root / "meta/sensor_transactions").glob("*.json"))
            assert journals
            assert all(json.loads(p.read_text())["state"] != "COMMITTED" for p in journals)
        finally:
            recorder.close()
            sensor.disconnect()
        assert not sensor.has_resources


def test_shutdown_during_reconnect_is_bounded():
    with ModbusServer() as server:
        sensor = X518Sensor(config(server, reconnect_delay_s=2))
        sensor.connect()
        server.fail.set()
        until(lambda: server.connections >= 2)
        started = time.perf_counter()
        sensor.disconnect()
        assert time.perf_counter() - started < 2
        assert not sensor.has_resources


def test_unresponsive_worker_is_terminated_with_latched_fault():
    from unittest.mock import Mock

    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        sensor.connect()
        process = sensor._process
        # Lose the cooperative stop notification; exercise the real terminate path.
        sensor._process_stop = Mock()
        started = time.perf_counter()
        sensor.disconnect()
        assert time.perf_counter() - started < 6
        assert "shutdown timed out" in sensor.diagnostics["fault"]
        assert not sensor.has_resources
        assert process._closed


def test_unconfirmed_shutdown_retains_ownership_and_blocks_connect():
    from unittest.mock import Mock

    from lerobot.utils.errors import DeviceAlreadyConnectedError

    with ModbusServer() as server:
        sensor = X518Sensor(config(server))
        process = Mock(pid=123)
        process.is_alive.return_value = True
        sensor._process = process
        try:
            with pytest.raises(RuntimeError, match="ownership retained"):
                sensor.disconnect()
            assert sensor.has_resources
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            process.close.assert_not_called()
            with pytest.raises(DeviceAlreadyConnectedError):
                sensor.connect()
        finally:
            process.is_alive.return_value = False
            sensor.disconnect()


def test_real_spawn_connect_refusal_rolls_back_ipc_resources():
    import socket

    # Bound but not listening: the address stays reserved and rejects connects.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        server = type("Address", (), {"port": reservation.getsockname()[1]})()
        sensor = X518Sensor(config(server, connect_retries=0, request_timeout_s=0.05))
        with pytest.raises((OSError, TimeoutError)):
            sensor.connect()
        assert not sensor.has_resources


def test_owner_exit_stops_child_even_during_blocked_io(tmp_path):
    import subprocess
    import sys

    psutil = pytest.importorskip("psutil", reason="Process-liveness check requires psutil")

    with ModbusServer() as server:
        owner = tmp_path / "owner.py"
        owner.write_text(
            "import os, sys, time\n"
            "from lerobot.sensors.x518 import X518Sensor, X518SensorConfig, X518ChannelConfig\n"
            "if __name__ == '__main__':\n"
            "    sensor = X518Sensor(X518SensorConfig(host='127.0.0.1', port=int(sys.argv[1]), "
            "expected_unit=None, request_timeout_s=5, channels={'left.normal_force': X518ChannelConfig(channel=1)}))\n"
            "    sensor.connect()\n"
            "    print(sensor._process.pid, flush=True)\n"
            "    sys.stdin.readline()\n"
            "    os._exit(0)\n",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [sys.executable, str(owner), str(server.port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        child_pid = None
        try:
            child_pid = int(process.stdout.readline())
            server.hold.set()
            process.communicate("exit\n", timeout=10)
            until(lambda: not psutil.pid_exists(child_pid), timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if child_pid is not None and psutil.pid_exists(child_pid):
                psutil.Process(child_pid).kill()
