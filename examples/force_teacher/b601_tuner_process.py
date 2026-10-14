"""Windows spawn bridge: one control process, bounded mailboxes and priority stop."""

import json
import math
import multiprocessing as mp
import queue
import time
from contextlib import suppress
from pathlib import Path

from b601_supervised_bench import exception_messages, run

from lerobot.scripts import lerobot_force_teacher as common_runner


class ProcessLive:
    """Only the child owns runtime state; the parent publishes explicit requests."""

    def __init__(self, *, poll_s=0.05):
        if isinstance(poll_s, bool) or not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("IPC poll interval must be positive")
        context = mp.get_context("spawn")
        self.poll_s = poll_s
        self.stop = context.Event()
        self.parameters = context.Queue(maxsize=1)
        self.events = context.Queue(maxsize=1)
        self.telemetry = context.Queue(maxsize=1)
        self._latest = self._applied = self._startup = None
        self.closed = False

    def submit(self, kind, value=None):
        """Keep one pending bundle and one mode request; never queue motion steps."""
        if self.closed or self.stop.is_set():
            raise RuntimeError("会话已停止")
        if kind not in ("parameters", "freeze", "resume", "step", "approach", "position"):
            raise ValueError("Unknown control event")
        channel = self.parameters if kind == "parameters" else self.events
        if kind == "parameters":
            value["teacher"].validate()
            value["safety"].validate()
            with suppress(queue.Empty):
                channel.get_nowait()
        try:
            channel.put_nowait((kind, value))
        except queue.Full:
            if kind != "parameters":
                raise
            # Spawn's feeder may not yet expose its one item to get_nowait.
            # The visible IPC interval bounds replacement; old gains never queue.
            with suppress(queue.Empty):
                channel.get(timeout=self.poll_s)
            channel.put_nowait((kind, value))

    def take(self):
        """Mode changes take priority over live parameter updates."""
        for channel in (self.events, self.parameters):
            try:
                return channel.get_nowait()
            except queue.Empty:
                pass
        return None

    def _publish(self):
        if not self.closed:
            with suppress(queue.Full):
                self.telemetry.put_nowait((self._latest, self._applied, self._startup))

    @property
    def startup(self):
        return self._startup

    @startup.setter
    def startup(self, value):
        self._startup = value
        self._publish()

    @property
    def latest(self):
        return self._latest

    @latest.setter
    def latest(self, value):
        self._latest = value
        self._publish()

    @property
    def applied(self):
        return self._applied

    @applied.setter
    def applied(self, value):
        self._applied = value
        self._publish()

    def receive(self):
        if not self.closed:
            with suppress(queue.Empty, ValueError, OSError):
                self._latest, self._applied, self._startup = self.telemetry.get_nowait()

    def close(self):
        if self.closed:
            return
        self.receive()
        self.closed = True
        for channel in (self.parameters, self.events, self.telemetry):
            channel.close()
            channel.cancel_join_thread()


def control_worker(output, reviewed_config, live, result_queue):
    """The common runtime owns connect, execution, disposition and close."""
    try:
        result = run(output, execute=True, reviewed_config=reviewed_config, live=live)
        success = True
    except BaseException as exc:
        result, success = "; ".join(exception_messages(exc)), False
    finally:
        # A delayed owner may still close cooperatively. Only then finalize its
        # Sensor/log dependencies; no fault or motor command is retried here.
        while common_runner._UNSTOPPED_BENCHES:
            completed = common_runner.finalize_retained_benches()
            for evidence in completed:
                if isinstance(result, dict):
                    result.update(shutdown_evidence=evidence["shutdown_evidence"], valid=False)
                    result.setdefault("errors", []).extend(evidence["cleanup_errors"])
                    try:
                        (Path(output) / "result.json").write_text(
                            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        result["errors"].append(f"Deferred result write failed: {exc}")
            if not common_runner._UNSTOPPED_BENCHES:
                break
            row = dict(live.latest or {})
            row.update(cleanup_pending=True, cleanup_error="owner / 传输退出未确认；保留控制进程与串口所有权")
            live.latest = row
            time.sleep(live.poll_s)
    result_queue.put((success, result))


def run_in_process(output, *, reviewed_config, live, worker=control_worker):
    """Wait outside Tk and retain a cleaning-up owner; never forcibly kill it."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=worker, args=(output, reviewed_config, live, result_queue), name="b601-force-owner"
    )
    started = False
    try:
        process.start()
        started = True
        while True:
            try:
                success, result = result_queue.get(timeout=live.poll_s)
                break
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError(f"控制进程意外退出（{process.exitcode}）；退出失能未确认") from None
        process.join()
        if not success or process.exitcode != 0:
            raise RuntimeError(f"控制进程异常：{result}")
        return result
    finally:
        if started and process.is_alive():
            live.stop.set()
            process.join()
        live.close()
        result_queue.close()
        result_queue.cancel_join_thread()
