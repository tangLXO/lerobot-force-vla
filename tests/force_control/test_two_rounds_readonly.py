"""No hardware: verify two query rounds cannot reuse feedback or restart their budget."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from examples.force_teacher.two_rounds_readonly_hardware_smoke_test import collect_pair
from lerobot.robots.rebot_b601_follower.readonly_transport import (
    B601ReadonlyBatch,
    B601ReceivedFeedback,
)


def fixture_transport(*, failure=None):
    clock = SimpleNamespace(now=1_000_000_000)
    calls = []

    def read_feedback(**kwargs):
        calls.append(kwargs)
        started = clock.now
        clock.now += 4_000_000 if failure != "slow_first" else 21_000_000
        sequence = len(calls)
        feedback = tuple(
            B601ReceivedFeedback(
                f"axis_{index}",
                index,
                index + 16,
                "4310",
                0,
                "DISABLED",
                0,
                0,
                0,
                0,
                20,
                20,
                clock.now,
                sequence,
                "",
                "",
                (12.5, 30, 10),
            )
            for index in range(1, 8)
        )
        batch = B601ReadonlyBatch("feedback", started, clock.now, kwargs["deadline_ns"], feedback=feedback)
        if failure == "missing_first":
            batch = replace(batch, feedback=feedback[:-1], missing=("axis_7",), error="missing axis")
        elif len(calls) == 2 and failure == "reuse":
            batch = replace(batch, feedback=tuple(replace(item, receive_sequence=1) for item in feedback))
        elif len(calls) == 2 and failure == "enabled":
            batch = replace(batch, feedback=tuple(replace(item, status_code=1) for item in feedback))
        elif len(calls) == 2 and failure == "late_second":
            clock.now = kwargs["deadline_ns"] + 1
            batch = replace(batch, completed_ns=clock.now)
        return batch

    transport = SimpleNamespace(read_feedback=read_feedback)

    def sleep(seconds):
        clock.now += round(seconds * 1e9)

    return transport, clock, calls, sleep


def test_two_fresh_rounds_share_original_deadline_and_complete_in_one_period():
    transport, clock, calls, sleep = fixture_transport()
    row = collect_pair(transport, clock=lambda: clock.now, sleep=sleep)
    assert row["error"] is None
    assert len(calls) == 2 and calls[0]["deadline_ns"] == calls[1]["deadline_ns"] == row["deadline_ns"]
    assert row["rounds"][1]["started_ns"] == row["rounds"][0]["completed_ns"]
    assert row["completed_ns"] - row["started_ns"] == 8_000_000


@pytest.mark.parametrize("failure", ["missing_first", "slow_first"])
def test_failed_first_round_prevents_second_query_group(failure):
    transport, clock, calls, sleep = fixture_transport(failure=failure)
    row = collect_pair(transport, clock=lambda: clock.now, sleep=sleep)
    assert row["error"] and len(calls) == 1


@pytest.mark.parametrize("failure", ["reuse", "enabled", "late_second"])
def test_second_round_cannot_hide_reuse_enabled_state_or_expiry(failure):
    transport, clock, calls, sleep = fixture_transport(failure=failure)
    row = collect_pair(transport, clock=lambda: clock.now, sleep=sleep)
    assert row["error"] and len(calls) == 2
