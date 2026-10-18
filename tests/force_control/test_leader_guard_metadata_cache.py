"""Repeated I/O guards reuse immutable parsing, never freshness or force decisions."""

import json

import pytest

from lerobot.force_control import runtime as runtime_module
from lerobot.force_control.runtime import ArmIntent
from tests.force_control import test_runtime

bench = test_runtime.bench


def publish_intent(runtime, clock, *, version=1):
    now = clock()
    runtime.post_arm_intent(
        ArmIntent(
            goals=(("gripper.pos", 0.0),),
            version=version,
            epoch=runtime.epoch,
            captured_ns=now,
            ttl_ns=100_000_000,
            metadata_json=json.dumps(
                {
                    "raw_action_deg": {"gripper.pos": 0.0},
                    "leader_sequence": version,
                    "leader_request_started_ns": now,
                    "diagnostics": [{"sample": index} for index in range(100)],
                }
            ),
        )
    )


def test_repeated_guards_parse_once_but_check_ttl_and_detach_visible_metadata(bench, monkeypatch):
    runtime, _, _, _, clock = bench
    runtime._hybrid_leader_required = True
    publish_intent(runtime, clock)
    original = json.loads
    calls = []

    def decode(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(runtime_module.json, "loads", decode)
    for _ in range(100):
        runtime._arm_intent(clock(), None)
    assert len(calls) == 1
    row = {}
    intent = runtime._arm_intent(clock(), row)
    row["arm_intent_metadata"]["raw_action_deg"]["gripper.pos"] = 99
    assert runtime._leader_inputs(intent)["leader_raw_gripper_deg"] == 0
    clock.advance(0.101)
    with pytest.raises(RuntimeError, match="arm_intent_ttl_expired"):
        runtime._arm_intent(clock(), None)
    assert len(calls) == 1
    publish_intent(runtime, clock, version=2)
    runtime._arm_intent(clock(), None)
    assert len(calls) == 2


def test_cached_leader_parsing_never_caches_raw_force_safety(bench):
    runtime, _, sensor, _, clock = bench
    runtime._hybrid_leader_required = True
    publish_intent(runtime, clock)
    sensor._publish_sample(
        {"left.normal_force": 0.0, "right.normal_force": 0.0},
        clock(),
        arrival_timestamp_ns=clock(),
    )
    runtime._startup_guard()
    clock.advance(0.001)
    sensor._publish_sample(
        {
            "left.normal_force": runtime.safety.config.hard_force_limit_left_N + 1,
            "right.normal_force": 0.0,
        },
        clock(),
        arrival_timestamp_ns=clock(),
    )
    with pytest.raises(RuntimeError, match="left_force_hard_limit"):
        runtime._startup_guard()


def test_static_leader_checks_run_once_per_immutable_intent_and_results_are_detached(bench, monkeypatch):
    runtime, _, _, _, clock = bench
    runtime._hybrid_leader_required = True
    publish_intent(runtime, clock)
    original = runtime_module.finite_number
    calls = []

    def check(name, value):
        calls.append(name)
        return original(name, value)

    monkeypatch.setattr(runtime_module, "finite_number", check)
    intent = runtime._arm_intent(clock(), None)
    runtime._leader_inputs(intent)["leader_raw_gripper_deg"] = 999
    for _ in range(100):
        runtime._arm_intent(clock(), None)
    assert calls.count("leader_raw_gripper.pos") == 1
    assert runtime._leader_inputs(intent)["leader_raw_gripper_deg"] == 0
    clock.advance(0.101)
    with pytest.raises(RuntimeError, match="arm_intent_ttl_expired"):
        runtime._arm_intent(clock(), None)
    publish_intent(runtime, clock, version=2)
    runtime._arm_intent(clock(), None)
    assert calls.count("leader_raw_gripper.pos") == 2
