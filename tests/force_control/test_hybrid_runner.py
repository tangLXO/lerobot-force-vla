"""Hybrid session lifecycle with synthetic leader inputs and no physical devices."""

import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.force_control.leader import FakeReadonlyLeader, LeaderIntent, LeaderSamplerWorker
from lerobot.scripts import lerobot_force_teacher as runner
from tests.force_control.helpers import run_config


def hybrid_config(tmp_path):
    config = run_config(tmp_path, armed=True)
    config.initial_mode = "hybrid"
    config.leader_backend = "fake"
    config.leader_frequency_hz = 100
    config.leader_max_age_s = 0.3
    config.teacher = replace(config.teacher, gripper_follow_speed_deg_s=5, contact_force_threshold_N=0.1)
    return config


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"leader_backend": None}, "explicit leader_backend"),
        ({"leader_backend": "rebot_102", "leader_port": "FAKE_ONLY"}, "dry_run"),
        ({"leader_port": "FAKE_ONLY"}, "physical leader_port"),
        ({"leader_readonly_transport": "unknown"}, "leader_readonly_transport"),
        ({"leader_baudrate": True}, "leader_baudrate"),
        ({"leader_frequency_hz": 0}, "leader_frequency_hz"),
        ({"leader_read_budget_s": 0.3}, "freshness margin"),
        ({"explicit_arm": False}, "explicit_arm"),
    ],
)
def test_hybrid_rejects_configuration_before_factories(tmp_path, monkeypatch, changes, error):
    config = hybrid_config(tmp_path)
    for name, value in changes.items():
        setattr(config, name, value)
    factory = Mock(side_effect=AssertionError("must not construct a device"))
    monkeypatch.setattr(runner, "_leader_worker", factory)
    monkeypatch.setattr(runner, "FakeGripper", factory)
    with pytest.raises(ValueError, match=error):
        runner.run(config)
    factory.assert_not_called()


def test_nonhybrid_rejects_unused_leader_and_hybrid_rejects_whole_arm(tmp_path):
    config = run_config(tmp_path, armed=True)
    config.leader_backend = "fake"
    with pytest.raises(ValueError, match="initial_mode=hybrid"):
        config.validate()
    config = hybrid_config(tmp_path)
    config.runtime = replace(config.runtime, joint_names=("gripper", "shoulder_pan"))
    with pytest.raises(ValueError, match="ID-7"):
        config.validate()


def test_leader_forwarding_deduplicates_without_refreshing_capture_time(tmp_path):
    config = hybrid_config(tmp_path)
    intent = LeaderIntent(
        sequence=5,
        is_valid=True,
        raw_action_deg=(("gripper.pos", 2.0),),
        action_deg=(("gripper.pos", -12.0),),
        request_started_ns=1_000,
        response_completed_ns=2_000,
        source_sequence=42,
    )
    worker = SimpleNamespace(latest=lambda **kwargs: intent)
    service = SimpleNamespace(runtime=SimpleNamespace(epoch=0), submit_arm_intent=Mock())
    previous = runner._publish_leader(config, worker, service, None)
    assert runner._publish_leader(config, worker, service, previous) == previous
    assert service.submit_arm_intent.call_count == 1
    args = service.submit_arm_intent.call_args
    assert args.args == ({"gripper.pos": -12.0},)
    assert args.kwargs["captured_ns"] == 1_000
    assert args.kwargs["ttl_ms"] == min(config.runtime.command_ttl_ms, 300)
    assert args.kwargs["metadata"]["leader_sequence"] == 42
    assert args.kwargs["metadata"]["raw_action_deg"] == {"gripper.pos": 2.0}
    service.runtime.epoch = 1
    runner._publish_leader(config, worker, service, previous)
    assert service.submit_arm_intent.call_count == 2
    assert service.submit_arm_intent.call_args.kwargs["captured_ns"] == 1_000
    intent = replace(intent, is_valid=False, error="stale_or_future")
    with pytest.raises(RuntimeError, match="stale_or_future"):
        runner._publish_leader(config, worker, service, previous)
    assert service.submit_arm_intent.call_count == 2


def test_leader_startup_failure_never_connects_follower_or_sensor(tmp_path, monkeypatch):
    class FailingLeader(FakeReadonlyLeader):
        def read_action_snapshot(self, **kwargs):
            raise OSError("synthetic leader unavailable")

    source = FailingLeader()
    worker = LeaderSamplerWorker(source, frequency_hz=100)
    monkeypatch.setattr(runner, "_leader_worker", lambda config: worker)
    connect = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(runner.FakeForceSensor, "connect", connect)
    monkeypatch.setattr(runner.OwnerService, "start", connect)
    config = hybrid_config(tmp_path)
    config.startup_timeout_s = 0.03
    with pytest.raises(TimeoutError, match="no valid snapshot"):
        runner.run(config)
    connect.assert_not_called()
    assert not source.connected and not worker.has_resources
    assert not config.trace_root.exists()


def test_fake_hybrid_session_submits_real_leader_metadata_and_closes(tmp_path, monkeypatch):
    source = FakeReadonlyLeader(position_deg=0.2)
    worker = LeaderSamplerWorker(source, frequency_hz=100)
    monkeypatch.setattr(runner, "_leader_worker", lambda config: worker)
    config = hybrid_config(tmp_path)
    report = runner.run(config)
    assert report["valid"], report["errors"]
    assert not source.connected and not worker.has_resources
    rows = [json.loads(line) for line in (config.trace_root / "control.jsonl").read_text().splitlines()]
    following = [row for row in rows if row.get("hybrid_state") == "FOLLOW"]
    assert following
    assert all(row["arm_intent"]["gripper.pos"] == 0.2 for row in following)
    for row in following:
        metadata = row["arm_intent_metadata"]
        assert metadata["leader_sequence"] > 0
        assert metadata["leader_request_started_ns"] == row["arm_command_timestamp_ns"]
        assert metadata["raw_action_deg"]["gripper.pos"] == pytest.approx(-0.2 / 6)
        assert row["hardware_ack"] == "unknown"
    manifest = json.loads((config.trace_root / "manifest.json").read_text())
    assert not manifest["metadata"]["leader_shutdown"]["resources_retained"]


def test_leader_failure_during_session_stops_owner_and_closes(tmp_path, monkeypatch):
    class FailingLater(FakeReadonlyLeader):
        def read_action_snapshot(self, **kwargs):
            if self.sequence >= 8:
                raise OSError("synthetic running leader failure")
            return super().read_action_snapshot(**kwargs)

    source = FailingLater(position_deg=0.2)
    worker = LeaderSamplerWorker(source, frequency_hz=100)
    monkeypatch.setattr(runner, "_leader_worker", lambda config: worker)
    config = hybrid_config(tmp_path)
    with pytest.raises(RuntimeError, match="leader_input_invalid"):
        runner.run(config)
    assert not source.connected and not worker.has_resources
    manifest = json.loads((config.trace_root / "manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["metadata"]["shutdown_evidence"]["transport_closed"]


def test_blocked_leader_shutdown_retains_then_finalizes_without_reconnecting(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_UNSTOPPED_BENCHES", [])
    entered, release = threading.Event(), threading.Event()

    class BlockingLeader(FakeReadonlyLeader):
        def read_action_snapshot(self, **kwargs):
            if self.sequence >= 5:
                entered.set()
                release.wait(3)
            return super().read_action_snapshot(**kwargs)

    source = BlockingLeader()
    worker = LeaderSamplerWorker(source, frequency_hz=100)
    monkeypatch.setattr(runner, "_leader_worker", lambda config: worker)
    config = hybrid_config(tmp_path)
    config.duration_s = 0.1
    config.shutdown_timeout_s = 0.01
    try:
        with pytest.raises(BaseExceptionGroup, match="cleanup evidence"):
            runner.run(config)
        assert entered.is_set() and worker.has_resources
        assert len(runner._UNSTOPPED_BENCHES) == 1
        assert runner._UNSTOPPED_BENCHES[0][3] is worker
        assert runner.finalize_retained_benches() == []
        with pytest.raises(RuntimeError, match="resources remain retained"):
            runner.run(config)
    finally:
        release.set()
        worker.stop(timeout_s=2)
        runner.finalize_retained_benches()
    assert not runner._UNSTOPPED_BENCHES and not source.connected


def test_close_failure_keeps_leader_resources_visible():
    class CloseFailure(FakeReadonlyLeader):
        def disconnect(self):
            raise OSError("synthetic serial close failure")

    source = CloseFailure()
    worker = LeaderSamplerWorker(source)
    worker.start()
    worker.wait_for_valid()
    with pytest.raises(RuntimeError, match="close failure"):
        worker.stop()
    assert worker.has_resources and source.connected


def test_stop_before_hybrid_run_never_starts_leader(tmp_path, monkeypatch):
    source = FakeReadonlyLeader()
    worker = LeaderSamplerWorker(source)
    start = Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(worker, "start", start)
    monkeypatch.setattr(runner, "_leader_worker", lambda config: worker)
    stop = threading.Event()
    stop.set()
    live = SimpleNamespace(stop=stop, take=lambda: None, latest=None)
    with pytest.raises(RuntimeError, match="user_stop_before_connect"):
        runner.run(hybrid_config(tmp_path), live=live)
    start.assert_not_called()
