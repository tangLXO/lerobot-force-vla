"""Reuse only unconsumed, real seven-joint operator receive evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from lerobot.robots.rebot_b601_follower.full_arm_control_types import JOINT_NAMES
from tests.force_control.operator_teleoperation_helpers import (
    close_operator_scope,
    make_operator_files,
    make_operator_scope,
)


@pytest.fixture
def scope(tmp_path, monkeypatch):
    files = make_operator_files(tmp_path)
    files.operator = files.configuration = replace(files.operator, max_command_gap_s=None)
    value = make_operator_scope(tmp_path, monkeypatch, files=files)
    yield value
    close_operator_scope(value)


def read(scope):
    return scope.device.read_joint_snapshot(JOINT_NAMES, deadline_ns=scope.clock.now + 15_000_000)


def queries(scope):
    return [
        packet
        for packet in scope.serial.writes
        if int.from_bytes(packet[13:17], "little") == 0x7FF and packet[23] == 0xCC
    ]


def start(scope):
    # The runtime consumed an earlier read before dispatching its Start event.
    read(scope)
    return scope.device.start_operator_teleoperation(guard=lambda: None, leader_sample=scope.leader)


def test_first_active_read_reuses_fresh_post_enable_rx_once_without_restamping(scope):
    received = start(scope)
    before = len(queries(scope))
    scope.clock.now += 1_000_000
    snapshot = read(scope)
    assert len(queries(scope)) == before
    assert snapshot.completed_ns == received.completed_ns
    for row in snapshot.joints:
        original = received.get(row.name)
        assert (row.receive_timestamp_ns, row.receive_sequence, row.time_source, row.position_deg) == (
            original.receive_timestamp_ns,
            original.receive_sequence,
            original.time_source,
            original.position_deg,
        )
        assert row.age_ns == scope.clock.now - row.receive_timestamp_ns
    repeated = read(scope)
    assert len(queries(scope)) == before + 7
    assert all(
        repeated.get(name).receive_sequence > snapshot.get(name).receive_sequence for name in JOINT_NAMES
    )


def test_command_replies_supply_the_next_control_frame_without_seven_redundant_queries(scope):
    start(scope)
    previous = read(scope)
    before = len(queries(scope))
    receipt = scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 1.0), feedback_snapshot=previous
    )
    assert receipt.all_submitted
    received = scope.device._latest_snapshot
    snapshot = read(scope)
    assert len(queries(scope)) == before
    assert snapshot.completed_ns == received.completed_ns
    assert all(
        snapshot.get(name).receive_sequence > previous.get(name).receive_sequence for name in JOINT_NAMES
    )
    assert all(
        snapshot.get(name).receive_timestamp_ns == received.get(name).receive_timestamp_ns
        for name in JOINT_NAMES
    )


@pytest.mark.parametrize(
    "corruption",
    ["expired", "future", "missing_sequence", "repeated_sequence", "status", "missing_joint", "invalid"],
)
def test_any_incomplete_or_invalid_cached_joint_requires_a_new_complete_query(scope, corruption):
    start(scope)
    previous = read(scope)
    assert scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 1.0), feedback_snapshot=previous
    ).all_submitted
    session = scope.device._full_validation_session
    received = scope.device._latest_snapshot
    row = received.get("shoulder_pan")
    if corruption == "expired":
        scope.clock.now += 101_000_000
    elif corruption == "future":
        row = replace(row, receive_timestamp_ns=scope.clock.now + 1)
    elif corruption == "missing_sequence":
        row = replace(row, receive_sequence=None)
    elif corruption == "repeated_sequence":
        row = replace(row, receive_sequence=previous.get(row.name).receive_sequence)
    elif corruption == "status":
        session.observed_statuses[row.name] = 0
    if corruption == "missing_joint":
        received = replace(received, joints=received.joints[:-1])
    elif corruption == "invalid":
        received = replace(received, error="invalid cached evidence")
    else:
        received = replace(
            received, joints=tuple(row if item.name == row.name else item for item in received.joints)
        )
    scope.device._latest_snapshot = received
    before = len(queries(scope))
    snapshot = read(scope)
    assert snapshot.is_valid
    assert len(queries(scope)) == before + 7


def test_cached_replies_cannot_bypass_unaccounted_normal_query_ownership(scope):
    start(scope)
    snapshot = read(scope)
    assert scope.device.send_joint_positions_partial(
        dict.fromkeys(JOINT_NAMES, 1.0), feedback_snapshot=snapshot
    ).all_submitted
    scope.device._outstanding_query = {"packet_hex": "unaccounted"}
    before = len(queries(scope))
    result = read(scope)
    assert not result.is_valid and "unaccounted" in result.error
    assert len(queries(scope)) == before
    scope.device._outstanding_query = None  # test-only debt had no host write or pending bytes


def test_a_cached_snapshot_cannot_be_consumed_from_a_different_owner_thread(scope):
    received = start(scope)
    before = len(queries(scope))
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(RuntimeError, match="single owner"):
        executor.submit(read, scope).result()
    snapshot = read(scope)
    assert len(queries(scope)) == before
    assert snapshot.get("gripper").receive_sequence == received.get("gripper").receive_sequence
