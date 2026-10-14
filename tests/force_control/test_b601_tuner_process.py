"""Windows-compatible spawn with explicit I-P bundles, priority events and stop."""

import importlib
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
        live.submit("freeze")
        wait_for(lambda: live.applied[0] == "freeze", live)
        live.submit("step", -0.01)
        wait_for(lambda: live.applied == ("step", -0.01), live)
        live.submit("approach", {"gripper_follow_speed_deg_s": 1, "contact_force_threshold_N": 0.5})
        wait_for(lambda: live.applied[0] == "approach", live)
        assert live.applied[1]["contact_force_threshold_N"] == 0.5
        position = {"position_deg": -2, "position_speed_deg_s": 0.7}
        live.submit("position", position)
        wait_for(lambda: live.applied == ("position", position), live)
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
        live.submit("freeze")
        time.sleep(0.05)
        assert live.take() == ("freeze", None)
        assert live.take() == ("parameters", bundle)
        live.stop.set()
        with pytest.raises(RuntimeError, match="停止"):
            live.submit("resume")
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
