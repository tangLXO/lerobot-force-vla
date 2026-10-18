"""Query-only alignment lifecycle with synthetic angles and no physical transports."""

import importlib
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.robots.rebot_b601_follower.control_state import B601ControlRegister, B601ControlState
from lerobot.robots.rebot_b601_follower.readonly_transport import (
    DEFAULT_B601_JOINTS,
    B601ReadonlyBatch,
    B601ReceivedFeedback,
    B601ReceivedRegister,
)
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import LeaderActionSnapshot


@pytest.fixture
def alignment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    return importlib.import_module("full_arm_alignment_hardware_smoke_test")


@pytest.fixture
def configuration(alignment, tmp_path):
    calibration = tmp_path / "synthetic_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                joint.name: {
                    "id": joint.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -30,
                    "range_max": 30,
                }
                for joint in DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    config = alignment.AlignmentConfig(
        "FAKE_FOLLOWER",
        921600,
        str(calibration),
        "FAKE_LEADER",
        1_000_000,
        {joint.name: index for index, joint in enumerate(DEFAULT_B601_JOINTS)},
        dict.fromkeys(alignment.NAMES, 1),
        dict.fromkeys(alignment.NAMES, (-20, 20)),
        "protocol_v1",
        5,
        100,
        0.05,
        0.15,
        1.5,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(draccus.encode(config)), encoding="utf-8")
    return config, path


@pytest.fixture
def devices(alignment, monkeypatch):
    calls, owners = [], []
    state = SimpleNamespace(leader_stale=False, enabled_joint=None, close_failure=False)

    def called(label):
        calls.append(label)
        owners.append(threading.get_ident())

    class Follower:
        def __init__(self, *args, **kwargs):
            self.has_resources = False

        def open(self):
            called("follower_open")
            self.has_resources = True

        def read_mapping(self, *, max_read_duration_s):
            called("mapping")
            now = time.perf_counter_ns()
            registers = tuple(
                B601ReceivedRegister(
                    joint.name, joint.motor_id, joint.feedback_id, joint.model, rid, 12.5, now, 1, "", ""
                )
                for joint in DEFAULT_B601_JOINTS
                for rid in (21, 22, 23)
            )
            return B601ReadonlyBatch(
                "mapping", now, now, now + int(max_read_duration_s * 1e9), registers=registers
            )

        def read_feedback(self, *, max_read_duration_s):
            called("follower_snapshot")
            assert calls.index("leader_snapshot") < calls.index("follower_snapshot")
            now = time.perf_counter_ns()
            feedback = tuple(
                B601ReceivedFeedback(
                    joint.name,
                    joint.motor_id,
                    joint.feedback_id,
                    joint.model,
                    1 if state.enabled_joint == joint.name else 0,
                    "DISABLED",
                    0,
                    0,
                    0,
                    0,
                    20,
                    20,
                    now,
                    2,
                    "",
                    "",
                    (12.5, 30, 10),
                )
                for joint in DEFAULT_B601_JOINTS
            )
            return B601ReadonlyBatch(
                "feedback", now, now, now + int(max_read_duration_s * 1e9), feedback=feedback
            )

        def close(self):
            called("follower_close")
            if state.close_failure:
                state.close_failure = False
                raise OSError("close delayed")
            self.has_resources = False

    class Leader:
        def __init__(self, config):
            assert config.readonly_transport == "protocol_v1"
            self.has_resources = False

        def connect_readonly(self):
            called("leader_readonly_open")
            self.has_resources = True

        def read_action_snapshot(self, *, max_read_duration_s):
            called("leader_snapshot")
            now = time.perf_counter_ns()
            values = tuple(
                (f"{name}.pos", 12.0 if index == 0 else -4.0 if index == 5 else 0.0)
                for index, name in enumerate(alignment.NAMES)
            )
            return LeaderActionSnapshot(
                values, values, now - (200_000_000 if state.leader_stale else 1), now, 1
            )

        def disconnect(self):
            called("leader_close")
            self.has_resources = False

    def control(transport, *, max_read_duration_s):
        called("control_state")
        now = time.perf_counter_ns()
        registers = tuple(
            B601ControlRegister(
                joint.name,
                joint.motor_id,
                joint.feedback_id,
                rid,
                (4 if joint.name == "gripper" else 2) if rid == 10 else 2000,
                now,
                1,
                "",
            )
            for joint in DEFAULT_B601_JOINTS
            for rid in (10, 9)
        )
        return B601ControlState(now, now, now + int(max_read_duration_s * 1e9), registers, (), (), (), None)

    monkeypatch.setattr(alignment, "ReadonlyB601Transport", Follower)
    monkeypatch.setattr(alignment, "RebotArm102Leader", Leader)
    monkeypatch.setattr(alignment, "read_control_state", control)
    return state, calls, owners


def test_unaligned_query_reports_every_difference_and_closes_both_ports(
    alignment, configuration, devices, tmp_path
):
    _, calls, owners = devices
    result = alignment.run(configuration[1], tmp_path / "result")
    assert (
        result["complete"]
        and not result["alignment_within_tolerance"]
        and not result["ready_for_bounded_start"]
    )
    assert len(result["joints"]) == 7
    assert result["joints"][0]["difference_deg"] == 12 and result["joints"][5]["difference_deg"] == -4
    assert result["all_seven_disabled"] and result["control_state_expected"]
    assert result["motion_attempted"] is False and result["hardware_ack"] == "unknown"
    assert result["leader_closed"] and result["follower_closed"] and len(set(owners)) == 1
    assert "leader_readonly_open" in calls and "leader_close" in calls and "follower_close" in calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("tolerance_deg", 5.1),
        ("relative_bound_deg", 101),
        ("read_budget_s", 0.051),
        ("leader_max_age_s", 0.151),
        ("leader_readonly_transport", "sdk"),
        ("leader_joint_ids", {"gripper": 1}),
        ("follower_port", "FAKE_LEADER"),
    ],
)
def test_invalid_mapping_or_budget_rejects_before_any_connection(
    alignment, configuration, devices, tmp_path, field, value
):
    config, path = configuration
    _, calls, _ = devices
    path.write_text(json.dumps(draccus.encode(replace(config, **{field: value}))), encoding="utf-8")
    with pytest.raises(ValueError):
        alignment.run(path, tmp_path / "invalid")
    assert calls == []


def test_stale_leader_is_not_restamped_and_both_query_ports_close(
    alignment, configuration, devices, tmp_path
):
    state, _, _ = devices
    state.leader_stale = True
    result = alignment.run(configuration[1], tmp_path / "stale")
    assert not result["complete"] and not result["ready_for_bounded_start"]
    assert any("stale" in error for error in result["errors"])
    assert result["leader_closed"] and result["follower_closed"]


def test_enabled_feedback_is_reported_without_issuing_disable(alignment, configuration, devices, tmp_path):
    state, calls, _ = devices
    state.enabled_joint = "wrist_roll"
    result = alignment.run(configuration[1], tmp_path / "enabled")
    assert result["complete"] and not result["all_seven_disabled"] and not result["ready_for_bounded_start"]
    assert not any("disable" in call for call in calls)


def test_readonly_close_failure_retains_and_retries_only_closing(alignment, configuration, devices, tmp_path):
    state, calls, _ = devices
    state.close_failure = True
    result = alignment.run(configuration[1], tmp_path / "close")
    assert not result["complete"] and result["leader_closed"] and result["follower_closed"]
    assert calls.count("follower_close") == 2 and calls.count("follower_snapshot") == 1


def test_close_budget_persists_retained_state_without_claiming_closed(
    alignment, configuration, devices, monkeypatch, tmp_path
):
    monkeypatch.setattr(alignment.ReadonlyB601Transport, "close", lambda self: None)
    monkeypatch.setattr(alignment, "CLOSE_BUDGET_S", 0.01)
    output = tmp_path / "retained"
    started = time.monotonic()
    result = alignment.run(configuration[1], output)
    assert time.monotonic() - started < 1
    assert not result["complete"] and not result["follower_closed"]
    assert result["resources_retained"] == ["follower"] and result["cleanup_pending"]
    assert json.loads((output / "alignment.json").read_text(encoding="utf-8"))["resources_retained"] == [
        "follower"
    ]
    assert result["motion_authorized"] is False
    alignment._RETAINED_READONLY.clear()
