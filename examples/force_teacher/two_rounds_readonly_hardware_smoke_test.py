"""Measure two complete disabled-only feedback rounds under one 20 ms deadline."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from lerobot.robots.rebot_b601_follower.readonly_transport import ReadonlyB601Transport

PERIOD_NS = 20_000_000


class StrictReadonlyTransport(ReadonlyB601Transport):
    """Never discard unexpected bytes to make the next query appear successful."""

    def _prepare_query(self, deadline_ns):
        self._remaining(deadline_ns)
        if self._rx_pending or self._rx_poison or self._serial.in_waiting:
            raise RuntimeError("Unexpected bytes before a new independent feedback query")


def collect_pair(transport, *, clock=time.perf_counter_ns, sleep=time.sleep):
    """Finish each round before another starts; the second cannot reset the deadline."""
    started = clock()
    deadline = started + PERIOD_NS
    row = {"started_ns": started, "deadline_ns": deadline, "rounds": [], "error": None}
    previous_sequences = {}
    try:
        for index in range(2):
            if clock() >= deadline:
                raise TimeoutError("Original 20 ms pair deadline expired before next round")
            batch = transport.read_feedback(max_read_duration_s=0.02, deadline_ns=deadline)
            row["rounds"].append(asdict(batch))
            if not batch.is_valid or len(batch.feedback) != 7:
                raise RuntimeError(f"Incomplete feedback round {index + 1}: {batch.error}")
            if any(item.status_code != 0 for item in batch.feedback):
                raise RuntimeError("This query experiment requires all seven axes disabled")
            if len({item.name for item in batch.feedback}) != 7 or any(
                item.receive_sequence <= previous_sequences.get(item.name, 0)
                or not batch.started_ns <= item.receive_timestamp_ns <= deadline
                for item in batch.feedback
            ):
                raise RuntimeError("Duplicate, reused, or out-of-window feedback")
            previous_sequences = {item.name: item.receive_sequence for item in batch.feedback}
        if clock() > deadline:
            raise TimeoutError("Both feedback rounds completed after their original deadline")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["completed_ns"] = clock()
    return row


def run(port, periods, output):
    """Open only the existing query adapter; retain failures and close on every path."""
    if not 1 <= periods <= 3000:
        raise ValueError("periods must be in 1..3000")
    output.mkdir(parents=True, exist_ok=False)
    transport = StrictReadonlyTransport(port)
    summary = {
        "queries_only": True,
        "period_ns": PERIOD_NS,
        "second_round_offset_ns": "immediately_after_complete_first_round",
        "requested_periods": periods,
        "completed_periods": 0,
        "scheduled_slots_skipped": 0,
        "error": None,
        "hardware_ack": "unknown",
        "interpretation": "Two independent state-query rounds, not repeated actuator commands or ACKs.",
    }
    try:
        transport.open()
        mapping = transport.read_mapping()
        summary["mapping"] = asdict(mapping)
        if not mapping.is_valid:
            raise RuntimeError(f"Mapping preflight failed: {mapping.error}")
        initial = transport.read_feedback(max_read_duration_s=0.015)
        summary["initial_disabled_feedback"] = asdict(initial)
        if not initial.is_valid or any(item.status_code != 0 for item in initial.feedback):
            raise RuntimeError("Fresh disabled preflight required")
        scheduled = time.perf_counter_ns()
        with (output / "pairs.jsonl").open("x", encoding="utf-8") as handle:
            for index in range(periods):
                row = collect_pair(transport)
                row["period_index"] = index
                row["scheduled_ns"] = scheduled
                handle.write(json.dumps(row) + "\n")
                if row["error"]:
                    raise RuntimeError(row["error"])
                summary["completed_periods"] += 1
                scheduled += PERIOD_NS
                now = time.perf_counter_ns()
                if now > scheduled:
                    skipped = (now - scheduled) // PERIOD_NS + 1
                    summary["scheduled_slots_skipped"] += skipped
                    scheduled += skipped * PERIOD_NS
                time.sleep(max(0, (scheduled - time.perf_counter_ns()) / 1e9))
        final = transport.read_feedback(max_read_duration_s=0.015)
        summary["final_disabled_feedback"] = asdict(final)
        if not final.is_valid or any(item.status_code != 0 for item in final.feedback):
            raise RuntimeError("Final disabled preflight incomplete")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if transport.has_resources:
            transport.close()
        summary["transport_closed"] = not transport.has_resources
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}))
    return 1 if summary["error"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--periods", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.port, args.periods, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
