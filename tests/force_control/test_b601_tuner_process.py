"""Windows-compatible spawn with explicit I-P bundles, priority events and stop."""

import importlib
import multiprocessing
import os
import queue
import threading
import time
from pathlib import Path

import pytest

from lerobot.force_control.config import SafetyConfig, TeacherConfig


def fake_control(output, reviewed_config, live, result_queue):
    live.latest = {"pid": os.getpid()}
    while not live.stop.wait(live.poll_s):
        event = live.take()
        if event:
            live.applied = event
        live.latest = {"pid": os.getpid(), "phase": "POSITION_HOLD"}
    result_queue.put((True, {"pid": os.getpid(), "stopped": True}))


def failed_control(output, reviewed_config, live, results):
    results.put((False, "simulated owner failure"))


def silent_control(output, reviewed_config, live, results):
    pass


def preview_control(output, reviewed_config, live, results):
    live.preview = {"front": {"jpeg": b"j" * 200_000, "timestamp_ns": 123}}
    results.put((True, {"stopped": True}))


def observe_stop_transitions(stop, commands, results):
    for _ in range(3):
        commands.get(timeout=10)
        results.put(stop.is_set())


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("b601_tuner_process")


def wait_for(predicate, live, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live.receive()
        if predicate():
            return
        time.sleep(live.poll_s)
    pytest.fail("IPC event not received")


def test_stop_flag_timeout_wakeup_and_reset(bridge):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        assert not live.stop.wait(0)
        thread = threading.Thread(target=lambda: (time.sleep(0.02), live.stop.set()))
        thread.start()
        assert live.stop.wait(2)
        thread.join()
        assert live.stop.is_set() and live.stop.wait(0)
        live.stop.clear()
        assert not live.stop.is_set() and not live.stop.wait(0)
    finally:
        live.close()


def test_stop_flag_transitions_visible_across_spawn(bridge):
    context = multiprocessing.get_context("spawn")
    live = bridge.ProcessLive(poll_s=0.01)
    commands, results = context.Queue(), context.Queue()
    child = context.Process(target=observe_stop_transitions, args=(live.stop, commands, results))
    child.start()
    try:
        for expected in (False, True, False):
            live.stop.set() if expected else live.stop.clear()
            commands.put(True)
            assert results.get(timeout=10) is expected
        child.join(10)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        commands.close()
        results.close()
        live.close()


def test_spawn_isolates_owner_and_carries_new_parameters_and_priority_stop(bridge, tmp_path):
    live = bridge.ProcessLive(poll_s=0.01)
    results, errors = [], []

    def launch():
        try:
            results.append(
                bridge.run_in_process(tmp_path, reviewed_config=None, live=live, worker=fake_control)
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    try:
        wait_for(lambda: live.latest is not None, live)
        assert live.latest["pid"] != os.getpid()
        bundle = {
            "teacher": TeacherConfig.fake(kp_deg_per_N=0.05, ki_deg_per_N_s=0.2),
            "safety": SafetyConfig.fake(),
        }
        live.submit("parameters", bundle)
        wait_for(lambda: live.applied is not None, live)
        assert live.applied == ("parameters", bundle)
        live.submit("resume")
        wait_for(lambda: live.applied == ("resume", None), live)
        live.submit("target", 3.0)
        wait_for(lambda: live.applied == ("target", 3.0), live)
        live.submit("approach", {"gripper_follow_speed_deg_s": 1, "contact_force_threshold_N": 0.5})
        wait_for(lambda: live.applied[0] == "approach", live)
        assert live.applied[1]["contact_force_threshold_N"] == 0.5
        position = {"position_deg": -2, "position_speed_deg_s": 0.7}
        live.submit("position", position)
        wait_for(lambda: live.applied == ("position", position), live)
        live.submit("release_and_finish")
        wait_for(lambda: live.applied == ("release_and_finish", None), live)
        for kind in ("release_and_finish", "resume", "parameters"):
            with pytest.raises(RuntimeError, match="仅停止"):
                live.submit(kind, bundle if kind == "parameters" else None)
    finally:
        live.stop.set()
        thread.join(10)
    assert not errors and not thread.is_alive()
    assert results[0]["stopped"] and live.closed
    live.close()


@pytest.mark.parametrize(
    "worker,match", [(failed_control, "simulated owner failure"), (silent_control, "意外退出")]
)
def test_owner_failure_reaches_parent_without_claiming_disable(bridge, tmp_path, worker, match):
    live = bridge.ProcessLive(poll_s=0.01)
    with pytest.raises(RuntimeError, match=match):
        bridge.run_in_process(tmp_path, reviewed_config=None, live=live, worker=worker)
    assert live.closed


def test_invalid_bundle_never_enters_mailbox_and_mode_has_separate_slot(bridge):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        with pytest.raises(ValueError):
            live.submit(
                "parameters",
                {"teacher": TeacherConfig.fake(ki_deg_per_N_s=float("nan")), "safety": SafetyConfig.fake()},
            )
        bundle = {"teacher": TeacherConfig.fake(), "safety": SafetyConfig.fake()}
        live.submit("parameters", bundle)
        live.submit("resume")
        time.sleep(0.05)
        assert live.take() == ("resume", None)
        assert live.take() == ("parameters", bundle)
        live.stop.set()
        with pytest.raises(RuntimeError, match="停止"):
            live.submit("resume")
    finally:
        live.close()


@pytest.mark.parametrize("kind", ["freeze", "step"])
def test_removed_commands_never_enter_process_mailbox(bridge, kind):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        with pytest.raises(ValueError, match="Unknown control event"):
            live.submit(kind)
        assert live.take() is None
    finally:
        live.close()


@pytest.mark.parametrize("value", [None, True, "3", -1, float("nan"), float("inf")])
def test_invalid_force_target_never_enters_motion_mailbox(bridge, value):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        with pytest.raises(ValueError, match="目标力"):
            live.submit("target", value)
        assert live.take() is None
    finally:
        live.close()


def test_release_finish_uses_existing_limits_and_requires_an_available_event_slot(bridge):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        with pytest.raises(ValueError, match="已配置"):
            live.submit("release_and_finish", {"release_speed_deg_s": 999})
        assert not live.finishing
        live.submit("resume")
        with pytest.raises(queue.Full):
            live.submit("release_and_finish")
        assert not live.finishing
        time.sleep(0.05)
        assert live.take() == ("resume", None)
        live.submit("release_and_finish")
        assert live.finishing
        time.sleep(0.05)
        assert live.take() == ("release_and_finish", None)
        live.stop.set()
        with pytest.raises(RuntimeError, match="停止"):
            live.submit("release_and_finish")
    finally:
        live.close()


def test_control_process_retains_owner_after_unknown_cleanup_without_retry(bridge, monkeypatch, tmp_path):
    from types import SimpleNamespace

    service = SimpleNamespace(has_resources=True)
    monkeypatch.setattr(bridge.common_runner, "_UNSTOPPED_BENCHES", [(service, object(), object())])

    def finalize():
        if service.has_resources:
            return []
        bridge.common_runner._UNSTOPPED_BENCHES.clear()
        return [{"shutdown_evidence": {"transport_closed": True}, "cleanup_errors": []}]

    monkeypatch.setattr(bridge.common_runner, "finalize_retained_benches", finalize)
    calls = []
    monkeypatch.setattr(bridge, "run", lambda *a, **k: calls.append("run") or {"errors": ["shutdown_failed"]})
    monkeypatch.setattr(bridge.time, "sleep", lambda _: setattr(service, "has_resources", False))
    live = SimpleNamespace(latest={"phase": "FAULT", "failure_reason": "shutdown_failed"}, poll_s=0.01)
    results = queue.Queue()
    bridge.control_worker(tmp_path, None, live, results)
    assert calls == ["run"]
    assert live.latest["cleanup_pending"] and "退出未确认" in live.latest["cleanup_error"]
    assert live.latest["failure_reason"] == "shutdown_failed"
    success, result = results.get_nowait()
    assert success is True and result["valid"] is False
    assert result["shutdown_evidence"]["transport_closed"]
    assert result["errors"] == ["shutdown_failed"]


def test_control_worker_reports_individual_failures_from_exception_group(bridge, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(bridge.common_runner, "_UNSTOPPED_BENCHES", [])

    def fail(*args, **kwargs):
        raise ExceptionGroup(
            "configuration and cleanup", [ValueError("bad profile"), OSError("close failed")]
        )

    monkeypatch.setattr(bridge, "run", fail)
    results = queue.Queue()
    bridge.control_worker(tmp_path, None, SimpleNamespace(), results)
    success, result = results.get_nowait()
    assert success is False
    assert result == "ValueError: bad profile; OSError: close failed"


def test_recording_events_have_bounded_independent_slot_and_never_latch_motion_finish(bridge):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        live.submit("record_start", {"config": {"task": "pick"}})
        with pytest.raises(queue.Full):
            live.submit("record_finish", {"task_result": "success"})
        live.submit("resume")
        time.sleep(0.05)
        assert live.take() == ("resume", None)
        assert live.take() == ("record_start", {"config": {"task": "pick"}})
        live.submit("record_finish", {"task_result": "success"})
        time.sleep(0.05)
        assert live.take() == ("record_finish", {"task_result": "success"})
        assert not live.finishing
        live.submit("record_discard", {"rerecord": True})
        live.stop.set()
        with pytest.raises(RuntimeError, match="停止"):
            live.submit("record_outcome", {"task_result": "failure"})
    finally:
        live.close()


@pytest.mark.parametrize(
    "kind,value",
    [
        ("record_start", None),
        ("record_start", {"config": {}, "enable": True}),
        ("record_start", {"config": {"fps": float("nan")}}),
        ("record_finish", {"task_result": "automatic_success"}),
        ("record_outcome", {"task_result": "success", "release": True}),
        ("record_discard", {"rerecord": 1}),
    ],
)
def test_invalid_recording_requests_do_not_enter_ipc(bridge, kind, value):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        with pytest.raises(ValueError):
            live.submit(kind, value)
        assert live.take() is None
    finally:
        live.close()


def test_preview_is_bounded_and_throttled_without_replacing_control_status(bridge, monkeypatch):
    live = bridge.ProcessLive(poll_s=0.01)
    try:
        monkeypatch.setattr(bridge.time, "monotonic", lambda: 10.0)
        first = {"front": {"jpeg": b"jpeg", "timestamp_ns": 1}}
        live.preview = first
        live.preview = {"front": {"jpeg": b"newer", "timestamp_ns": 2}}
        assert live.preview == first
        with pytest.raises(ValueError, match="有界 JPEG"):
            live.preview = {"front": {"jpeg": b"x" * 262_145, "timestamp_ns": 3}}
        with pytest.raises(ValueError, match="四路"):
            live.preview = {str(index): first["front"] for index in range(5)}
        live.latest = {"phase": "FOLLOW"}
        live.recording = {"state": "recording", "episode_count": 0}
        assert live.latest["phase"] == "FOLLOW"
        assert live.recording["state"] == "recording"
    finally:
        live.close()


def test_spawn_exit_drains_preview_larger_than_windows_pipe_buffer(bridge, tmp_path):
    live = bridge.ProcessLive(poll_s=0.01)
    result = bridge.run_in_process(tmp_path, reviewed_config=None, live=live, worker=preview_control)
    assert result == {"stopped": True}
    assert len(live.preview["front"]["jpeg"]) == 200_000
