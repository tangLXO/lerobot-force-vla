"""Runner wiring must pin the actual producer even when every angle is zero."""

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lerobot.force_control.fake_hardware import FAKE_B601_JOINT_NAMES
from lerobot.robots.rebot_b601_follower.full_arm_validation import leader_configuration_digest
from lerobot.scripts import lerobot_force_teacher as runner
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import LeaderActionSnapshot


@pytest.mark.parametrize(
    "mutation", ["runner_before_open", "runner_during_read", "actual_source_during_read"]
)
def test_real_runner_guard_rejects_mapping_changes_that_zero_pose_alignment_cannot_detect(
    monkeypatch, mutation
):
    config = runner.ForceTeacherRunConfig(
        backend="b601",
        dry_run=False,
        initial_mode="hybrid",
        leader_backend="rebot_102",
        leader_port="FAKE_LEADER_NO_HARDWARE",
        leader_readonly_transport="protocol_v1",
        leader_joint_ids=dict(zip(FAKE_B601_JOINT_NAMES, range(7), strict=True)),
        leader_joint_directions=dict.fromkeys(FAKE_B601_JOINT_NAMES, 1),
        leader_joint_ranges={name: [-20, 20] for name in FAKE_B601_JOINT_NAMES},
    )
    pinned_digest = leader_configuration_digest(runner.leader_configuration_projection(config))
    config.b601 = SimpleNamespace(validation=SimpleNamespace(leader_config_sha256=pinned_digest))

    class ZeroLeader:
        instances = []

        def __init__(self, source_config):
            self.config = source_config
            self.connected = False
            self.opens = self.reads = 0
            self.corrupt = False
            type(self).instances.append(self)

        @property
        def has_resources(self):
            return self.connected

        def connect_readonly(self):
            self.opens += 1
            self.connected = True

        def read_action_snapshot(self, *, max_read_duration_s):
            self.reads += 1
            now = time.perf_counter_ns()
            if self.corrupt:
                if mutation == "runner_during_read":
                    config.leader_joint_directions = {**config.leader_joint_directions, "shoulder_pan": -1}
                else:
                    self.config = replace(
                        self.config,
                        joint_directions={**self.config.joint_directions, "shoulder_pan": -1},
                    )
            zeros = tuple((f"{name}.pos", 0.0) for name in FAKE_B601_JOINT_NAMES)
            return LeaderActionSnapshot(zeros, zeros, now, now, self.reads)

        def disconnect(self):
            self.connected = False

    monkeypatch.setattr(runner, "RebotArm102Leader", ZeroLeader)
    worker = runner._leader_worker(config)
    source = ZeroLeader.instances[0]
    try:
        if mutation == "runner_before_open":
            config.leader_joint_directions = {**config.leader_joint_directions, "shoulder_pan": -1}
            worker.start()
            with pytest.raises(RuntimeError, match="configuration_changed"):
                worker.wait_for_valid(timeout_s=1)
            assert source.opens == source.reads == 0
        else:
            worker.start()
            previous = worker.wait_for_valid(timeout_s=1)
            assert previous.is_valid and len(previous.action_deg) == 7
            assert worker.source_config_sha256 == pinned_digest
            source.corrupt = True
            deadline = time.monotonic() + 1
            while worker.latest().is_valid and time.monotonic() < deadline:
                time.sleep(0.002)
            rejected = worker.latest()
            assert not rejected.is_valid and "configuration_changed" in rejected.error
            assert rejected.sequence > previous.sequence
            assert rejected.action_deg == rejected.raw_action_deg == ()
            if mutation == "actual_source_during_read":
                assert (
                    leader_configuration_digest(runner.leader_configuration_projection(config))
                    == pinned_digest
                )
    finally:
        worker.stop(timeout_s=1)
    assert not source.connected and not worker.has_resources
