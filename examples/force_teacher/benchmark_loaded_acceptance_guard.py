#!/usr/bin/env python
"""Host-only loaded guard benchmark; no serial open, motor command or physical Sensor."""

import argparse
import hashlib
import json
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import draccus
from single_joint_arm_acceptance import AcceptanceCLIConfig, EvidenceWriter, ForceGuard

from lerobot.force_control.fake_hardware import FakeForceSensor, FakePlant
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlBackend
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS


class FakeEscapeHook:
    """CPU-workload stand-in only; this never establishes a physical manual stop channel."""

    def check(self):
        pass


class ClockedFakeForceSensor(FakeForceSensor):
    """Use the real fake Sensor publication/history with an absolute benchmark clock.

    Windows Event.wait(2.5 ms) can round to about 15.6 ms. Only this benchmark's
    fake acquisition worker changes; physical Sensor and control code stay fixed.
    """

    def connect(self):
        if self.is_connected:
            raise RuntimeError("Fake benchmark sensor already connected")
        self._reset_framework_state()
        self._stop.clear()
        self.skipped_slots = 0

        def acquire():
            interval = round(1e9 / self.sample_rate_hz)
            deadline = time.perf_counter_ns()
            while not self._stop.is_set():
                remaining = deadline - time.perf_counter_ns()
                if remaining > 0:
                    time.sleep(min(0.005, remaining / 1e9))
                    continue
                self.publish_once()
                deadline += interval
                now = time.perf_counter_ns()
                if deadline <= now:
                    skipped = (now - deadline) // interval + 1
                    self.skipped_slots += skipped
                    deadline += skipped * interval

        self._thread = threading.Thread(target=acquire, name="fake-force-benchmark-clock", daemon=True)
        self._thread.start()


@contextmanager
def loaded_context(config, output, *, warmup_s=2.0):
    """Use real Sensor history and asynchronous evidence while leaving transport disconnected."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = config.validate()
    device = B601ControlBackend(
        config.port,
        DEFAULT_B601_JOINTS,
        config.control,
        baudrate=config.baudrate,
        acceptance=config.acceptance,
    )
    sensor = ClockedFakeForceSensor(FakePlant(), sample_rate_hz=400)
    writer = EvidenceWriter(output / "events.jsonl")
    lease = None
    try:
        writer.start()
        sensor.connect()
        lease = sensor.acquire_recorder()

        def sink(event):
            writer.submit({"recorded_ns": time.perf_counter_ns(), **event})

        guard = ForceGuard(
            config,
            sensor,
            lease,
            threading.Event(),
            FakeEscapeHook(),
            sink,
            writer.check_health,
            output / "STOP",
        )
        started = time.perf_counter()
        while time.perf_counter() - started < warmup_s:
            writer.check_health()
            time.sleep(min(0.01, max(0.001, warmup_s - (time.perf_counter() - started))))
        guard.wait_ready(1)
        history = sensor._history.snapshot()
        context = SimpleNamespace(
            device=device,
            sensor=sensor,
            guard=guard,
            writer=writer,
            calibration=calibration,
            initial_history_rows=len(history),
            warmup_s=time.perf_counter() - started,
            history_span_s=(history[-1].timestamp_ns - history[0].timestamp_ns) / 1e9
            if len(history) > 1
            else 0,
        )
        yield context
    finally:
        if lease is not None:
            sensor.release_recorder(lease)
        if sensor.has_resources:
            sensor.disconnect()
        writer.stop.set()
        while writer.has_resources:
            writer.thread.join(timeout=0.1)
        writer.check_health()
        if device.has_resources:
            raise AssertionError("Host-only benchmark unexpectedly acquired a transport")


def emulate_step(context, *, lightweight=True):
    """Three validation-only hypothetical writes and thirty guarded RX chunks; no send method."""
    receive = context.device._acceptance_wait_guard if lightweight else context.device._acceptance_guard
    for write_index in range(3):
        context.device._acceptance_guard(context.guard)
        context.writer.submit(
            {
                "stage": "hypothetical_write",
                "write_index": write_index,
                "transport_opened": False,
                "motor_command_sent": False,
            }
        )
        for _ in range(10):
            receive(context.guard)


def structural_counts(context):
    """Count content reads independently of the timing samples."""
    original = Path.read_bytes
    calibration_path = Path(context.calibration.path)
    count = [0]

    def read(path):
        if path == calibration_path:
            count[0] += 1
        return original(path)

    with patch.object(Path, "read_bytes", read):
        for _ in range(30):
            context.device._acceptance_wait_guard(context.guard)
        rx_reads = count[0]
        count[0] = 0
        emulate_step(context, lightweight=True)
        new_step_reads = count[0]
        count[0] = 0
        emulate_step(context, lightweight=False)
        old_step_reads = count[0]
    return {
        "rx_only_30_chunks_content_reads": rx_reads,
        "new_step_3_hypothetical_writes_plus_30_chunks_content_reads": new_step_reads,
        "old_step_3_hypothetical_writes_plus_30_chunks_content_reads": old_step_reads,
        "valid": rx_reads == 0 and new_step_reads == 3 and old_step_reads == 33,
    }


def _sample(callback):
    wall = time.perf_counter_ns()
    cpu = time.thread_time_ns()
    callback()
    return (time.perf_counter_ns() - wall) / 1e6, (time.thread_time_ns() - cpu) / 1e6


def _summary(samples):
    result = {"samples": len(samples)}
    for column, name in ((0, "wall_ms"), (1, "owner_thread_cpu_ms")):
        values = sorted(row[column] for row in samples)
        result[name] = {
            "median": statistics.median(values),
            "p95": values[round(0.95 * (len(values) - 1))],
            "max": values[-1],
        }
    return result


def run_benchmark(config_file, output, *, warmup_s=2.0, iterations=1000, step_repetitions=20):
    """Report host cost only; the 50 ms hardware limits and all input checks stay unchanged."""
    config_file, output = Path(config_file).resolve(), Path(output).resolve()
    config_bytes = config_file.read_bytes()
    config = draccus.decode(AcceptanceCLIConfig, json.loads(config_bytes))
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "loaded_acceptance_guard_host_benchmark_v1",
        "complete": False,
        "hardware_acceptance": False,
        "serial_opened": False,
        "motor_commands_sent": 0,
        "physical_sensor_opened": False,
        "fake_sensor_requested_hz": 400,
        "fake_acquisition_scheduler": "absolute host deadlines; real FakeForceSensor.publish_once/history/leases; skipped overdue slots",
        "force_max_age_ms": config.force_max_age_ms,
        "command_feedback_budget_ms": config.control.feedback_max_age_s * 1000,
        "step_target_hz": 20,
        "step_slot_ms": 50,
        "timing_scope": "Host function wall latency only, including local validation and real force-history selection; no serial wait or motor response.",
        "cpu_counter_note": "Windows thread CPU samples may be quantized to 15.625 ms; zero medians do not imply zero CPU work.",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "errors": [],
    }
    try:
        with loaded_context(config, output, warmup_s=warmup_s) as context:
            report.update(
                warmup_s=context.warmup_s,
                initial_history_rows=context.initial_history_rows,
                history_span_s=context.history_span_s,
                fake_sensor_history_measured_hz=(context.initial_history_rows - 1) / context.history_span_s
                if context.history_span_s > 0
                else None,
                calibration=asdict(context.calibration),
                structural=structural_counts(context),
            )
            modes = {
                "full_binding_per_rx": lambda: context.device._acceptance_guard(context.guard),
                "lightweight_rx": lambda: context.device._acceptance_wait_guard(context.guard),
            }
            samples = {name: [] for name in modes}
            for _ in range(iterations):
                for name, callback in modes.items():
                    samples[name].append(_sample(callback))
            steps = {"old_full_per_rx": [], "new_lightweight_rx": []}
            for _ in range(step_repetitions):
                for name, lightweight in (("old_full_per_rx", False), ("new_lightweight_rx", True)):
                    steps[name].append(
                        _sample(lambda light=lightweight: emulate_step(context, lightweight=light))
                    )
            report.update(
                callbacks={name: _summary(rows) for name, rows in samples.items()},
                emulated_steps={name: _summary(rows) for name, rows in steps.items()},
                force_guard_callbacks=context.guard.count,
                force_guard_samples_logged=context.guard.logged_count,
                evidence_queue_overflowed=context.writer.overflowed,
                evidence_worker_error=str(context.writer.error) if context.writer.error else None,
                final_history_rows=len(context.sensor._history),
            )
            report["complete"] = report["structural"]["valid"] and not context.writer.overflowed
        report["cleanup"] = {
            "fake_sensor_closed": not context.sensor.has_resources,
            "evidence_closed": not context.writer.has_resources,
        }
    except BaseException as exc:
        report["complete"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--step-repetitions", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.step_repetitions <= 0:
        parser.error("Iteration counts must be positive")
    report = run_benchmark(
        args.config, args.output, iterations=args.iterations, step_repetitions=args.step_repetitions
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
