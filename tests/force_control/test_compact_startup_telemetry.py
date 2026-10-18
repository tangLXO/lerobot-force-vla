"""Large canonical startup evidence must not enter the repeated owner/UI copy path."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.force_control.fake_hardware import FakePlant
from lerobot.force_control.owner import OwnerService
from lerobot.force_control.telemetry import control_row_summary, startup_summary
from lerobot.force_control.trace import BenchTrace
from tests.force_control.hardware_helpers import ManualSensor


class ExpensiveEvidence:
    def __deepcopy__(self, memo):
        pytest.fail("Full per-packet startup evidence entered a repeated telemetry copy")


def receipt(*, expensive=False):
    return {
        "kind": "operator_teleoperation",
        "complete": True,
        "hardware_ack": "unknown",
        "enabled_joint_names": ["joint", "gripper"],
        "command_settings": {"requested": {"drive_limit_ratio": 1.0}},
        "commands": [ExpensiveEvidence() if expensive else {"packet_hex": "00"}] * 105,
        "io_stages": [{"events": ["original"]}] * 128,
    }


def test_owner_detaches_small_startup_view_without_copying_packet_tree():
    raw = receipt(expensive=True)
    row = {"tick_id": 2, "phase": "FOLLOW", "startup_evidence": raw}
    runtime = SimpleNamespace(latest_joint_snapshot=None, last_submitted={}, epoch=1, mode="HYBRID")
    owner = OwnerService(runtime)
    subscriber = owner.subscribe()
    owner._publish(row)
    detached = owner.snapshot().row["startup_evidence"]
    assert detached["commands_count"] == 105 and detached["io_stages_count"] == 128
    assert "commands" not in detached and "io_stages" not in detached
    assert subscriber.get(timeout=0.001).row["startup_evidence"] == detached
    detached["enabled_joint_names"].append("caller_mutation")
    assert "caller_mutation" not in owner.snapshot().row["startup_evidence"]["enabled_joint_names"]
    assert row["startup_evidence"] is raw and len(raw["commands"]) == 105


def test_bench_trace_keeps_full_startup_evidence_when_owner_view_is_compact(tmp_path):
    raw = receipt()
    row = {"tick_id": 2, "phase": "FOLLOW", "startup_evidence": raw}
    trace = BenchTrace(tmp_path / "trace", ManualSensor(FakePlant()), {})
    trace._accepting = True
    try:
        assert trace.submit(row)
        stored = trace._queue.get_nowait()
        assert len(stored["startup_evidence"]["commands"]) == 105
        assert stored["startup_evidence"]["io_stages"] == raw["io_stages"]
        assert "commands" not in control_row_summary(row)["startup_evidence"]
        raw["commands"].clear()
        assert len(stored["startup_evidence"]["commands"]) == 105
    finally:
        trace._accepting = False


def test_process_telemetry_uses_the_same_detached_small_view(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    module = importlib.import_module("b601_tuner_process")
    live = module.ProcessLive.__new__(module.ProcessLive)
    live._publish = lambda: None
    raw = receipt(expensive=True)
    live.startup = raw
    live.latest = {"startup_evidence": raw, "phase": "FOLLOW"}
    assert live.startup["commands_count"] == 105
    assert "commands" not in live.latest["startup_evidence"]
    live.startup["command_settings"]["requested"]["drive_limit_ratio"] = 0.5
    assert raw["command_settings"]["requested"]["drive_limit_ratio"] == 1.0


def test_startup_summary_is_idempotent_and_keeps_none():
    value = startup_summary(receipt())
    assert startup_summary(value) == value
    assert startup_summary(None) is None


def test_repeated_telemetry_omits_raw_io_but_trace_retains_every_event(tmp_path):
    events = [
        {"sequence": i, "timestamp_ns": i, "kind": "command_rx", "data_hex": "aa"} for i in range(1, 22)
    ]
    row = {
        "tick_id": 3,
        "device_io_evidence": {"events": events, "complete": True, "hardware_ack": "unknown"},
    }
    trace = BenchTrace(tmp_path / "trace", ManualSensor(FakePlant()), {})
    trace._accepting = True
    try:
        assert trace.submit(row)
        canonical = trace._queue.get_nowait()
        view = control_row_summary(row)
        audit = view["device_io_evidence"]
        assert "events" not in audit and audit["events_count"] == 21
        assert (audit["first_sequence"], audit["last_sequence"]) == (1, 21)
        assert control_row_summary(view) == view
        assert canonical["device_io_evidence"]["events"] == events
        assert row["device_io_evidence"]["events"] is events
    finally:
        trace._accepting = False
