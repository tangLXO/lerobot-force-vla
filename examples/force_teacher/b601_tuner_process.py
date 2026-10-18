"""Windows spawn bridge: one control process, bounded mailboxes and priority stop."""

import json
import math
import multiprocessing as mp
import queue
import time
from contextlib import suppress
from pathlib import Path

from b601_supervised_bench import exception_messages, run

from lerobot.force_control.telemetry import control_row_summary, startup_summary
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
        self.recording_events = context.Queue(maxsize=1)
        self.telemetry = context.Queue(maxsize=1)
        self._latest = self._applied = self._startup = None
        self._recording = self._preview = None
        self._preview_published_at = 0.0
        self.closed = False
        self.finishing = False
        self.ipc_errors = []
        self._receive_failed = False

    def submit(self, kind, value=None):
        """Keep one pending bundle and one mode request; never queue motion steps."""
        if self.closed or self._queue_call("stop_state", self.stop.is_set):
            raise RuntimeError("会话已停止")
        if self.finishing:
            raise RuntimeError("已请求释放并结束采集；仅停止仍可用")
        if kind not in (
            "parameters",
            "target",
            "freeze",
            "resume",
            "step",
            "approach",
            "position",
            "release_and_finish",
            "record_start",
            "record_finish",
            "record_discard",
            "record_outcome",
        ):
            raise ValueError("Unknown control event")
        if kind == "release_and_finish" and value is not None:
            raise ValueError("释放使用本次会话已配置的行程、速度和时间限制")
        if kind == "target" and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("目标力须为非负有限值")
        if kind.startswith("record_"):
            self._validate_record_event(kind, value)
            channel = self.recording_events
        else:
            channel = self.parameters if kind == "parameters" else self.events
        if kind == "parameters":
            value["teacher"].validate()
            value["safety"].validate()
            with suppress(queue.Empty):
                self._queue_call("parameters_replace", channel.get_nowait)
        try:
            self._queue_call(f"submit:{kind}", channel.put_nowait, (kind, value))
        except queue.Full:
            if kind != "parameters":
                raise
            # Spawn's feeder may not yet expose its one item to get_nowait.
            # The visible IPC interval bounds replacement; old gains never queue.
            with suppress(queue.Empty):
                self._queue_call("parameters_replace_wait", channel.get, timeout=self.poll_s)
            self._queue_call("parameters_replace_submit", channel.put_nowait, (kind, value))
        if kind == "release_and_finish":
            self.finishing = True

    @staticmethod
    def _validate_record_event(kind, value):
        """Recording requests contain JSON data and never encode motion commands."""
        if not isinstance(value, dict):
            raise ValueError("录制请求必须为对象")
        if kind == "record_start":
            valid = set(value) == {"config"} and isinstance(value["config"], dict)
        elif kind in ("record_finish", "record_outcome"):
            valid = set(value) == {"task_result"} and value["task_result"] in (
                "success",
                "failure",
                "unknown",
            )
        else:
            valid = set(value) == {"rerecord"} and type(value["rerecord"]) is bool
        if not valid:
            raise ValueError("录制请求字段无效")
        try:
            payload = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (ValueError, TypeError) as exc:
            raise ValueError("录制请求必须为有限 JSON 数据") from exc
        if len(payload) > 131_072:
            raise ValueError("录制配置过大")

    def take(self):
        """Mode changes take priority over live parameter updates."""
        for channel in (self.events, self.recording_events, self.parameters):
            try:
                return self._queue_call("take", channel.get_nowait)
            except queue.Empty:
                pass
        return None

    def _publish(self):
        if not self.closed:
            with suppress(queue.Full):
                self._queue_call(
                    "publish",
                    self.telemetry.put_nowait,
                    (self._latest, self._applied, self._startup, self._recording, self._preview),
                )

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value):
        self._recording = value
        self._publish()

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        """Bound camera snapshots independently of native image acquisition rates."""
        if value is not None:
            frames = value
            if not isinstance(frames, dict) or len(frames) > 4:
                raise ValueError("预览最多支持四路相机")
            for name, frame in frames.items():
                if (
                    not isinstance(name, str)
                    or not 1 <= len(name) <= 128
                    or not isinstance(frame, dict)
                    or set(frame) != {"jpeg", "timestamp_ns"}
                    or not isinstance(frame.get("jpeg"), bytes)
                    or len(frame["jpeg"]) > 262_144
                    or type(frame.get("timestamp_ns")) is not int
                    or frame["timestamp_ns"] < 0
                ):
                    raise ValueError("预览帧必须包含有界 JPEG 和时间戳")
            now = time.monotonic()
            if self._preview is not None and now - self._preview_published_at < 0.2:
                return
            self._preview_published_at = now
        self._preview = value
        self._publish()

    @property
    def startup(self):
        return self._startup

    @startup.setter
    def startup(self, value):
        self._startup = startup_summary(value)
        self._publish()

    @property
    def latest(self):
        return self._latest

    @latest.setter
    def latest(self, value):
        self._latest = None if value is None else control_row_summary(value)
        self._publish()

    @property
    def applied(self):
        return self._applied

    @applied.setter
    def applied(self, value):
        self._applied = value
        self._publish()

    def receive(self):
        """Keep the last snapshot if the pipe closes; request stop without breaking Tk polling."""
        if self.closed or getattr(self, "_receive_failed", False):
            return
        try:
            snapshot = self.telemetry.get_nowait()
            latest, applied, startup, recording, preview = snapshot
        except queue.Empty:
            return
        except (EOFError, OSError, ValueError, TypeError) as exc:
            self._receive_failed = True
            self._ipc_runtime_error("receive", exc)
            return
        self._latest, self._applied, self._startup, self._recording, self._preview = (
            latest,
            applied,
            startup,
            recording,
            preview,
        )

    def _remember_ipc_error(self, operation, error):
        if not hasattr(self, "ipc_errors"):
            self.ipc_errors = []
        self.ipc_errors.append(f"{operation}: {type(error).__name__}: {error}")
        del self.ipc_errors[:-32]

    def _ipc_runtime_error(self, operation, error):
        """Request the existing priority stop and expose a catchable UI/runtime error."""
        self._remember_ipc_error(operation, error)
        try:
            self.stop.set()
        except Exception as stop_error:
            self._remember_ipc_error("stop", stop_error)
        return RuntimeError(f"控制进程通信失败 ({operation})：{type(error).__name__}: {error}")

    def _queue_call(self, operation, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (EOFError, OSError, ValueError) as exc:
            raise self._ipc_runtime_error(operation, exc) from exc

    def close(self):
        """Close every queue once, retaining errors instead of losing the worker's result."""
        if self.closed:
            return
        try:
            self.receive()
        except Exception as exc:
            self._ipc_runtime_error("close_receive", exc)
        finally:
            self.closed = True
            for name in ("parameters", "events", "recording_events", "telemetry"):
                channel = getattr(self, name, None)
                if channel is None:
                    continue
                for operation in ("close", "cancel_join_thread"):
                    try:
                        getattr(channel, operation)()
                    except Exception as exc:
                        self._ipc_runtime_error(f"{name}.{operation}", exc)


def _note_ipc_error(live, operation, error):
    """Retain diagnostics and request stop without interrupting owner cleanup."""
    handler = getattr(live, "_ipc_runtime_error", None)
    if callable(handler):
        return handler(operation, error)
    errors = getattr(live, "ipc_errors", None)
    if not isinstance(errors, list):
        errors = live.ipc_errors = []
    errors.append(f"{operation}: {type(error).__name__}: {error}")
    del errors[:-32]
    stop = getattr(live, "stop", None)
    if stop is not None:
        with suppress(Exception):
            stop.set()
    return RuntimeError(f"控制进程通信失败 ({operation})：{type(error).__name__}: {error}")


def _receive_during_cleanup(live):
    try:
        live.receive()
    except Exception as exc:
        _note_ipc_error(live, "cleanup_receive", exc)


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
            try:
                live.latest = row
            except Exception as exc:
                _note_ipc_error(live, "cleanup_publish", exc)
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
            except (EOFError, OSError, ValueError) as exc:
                raise _note_ipc_error(live, "result_receive", exc) from exc
        # Drain the one bounded snapshot while the child's queue feeder exits.
        # A JPEG can exceed a Windows pipe buffer; join must not await an unread pipe.
        while process.is_alive():
            _receive_during_cleanup(live)
            process.join(timeout=live.poll_s)
        _receive_during_cleanup(live)
        if not success or process.exitcode != 0:
            raise RuntimeError(f"控制进程异常：{result}")
        return result
    finally:
        if started and process.is_alive():
            try:
                live.stop.set()
            except Exception as exc:
                _note_ipc_error(live, "cleanup_stop", exc)
            while process.is_alive():
                _receive_during_cleanup(live)
                process.join(timeout=live.poll_s)
        for operation, function in (
            ("live.close", live.close),
            ("result_queue.close", result_queue.close),
            ("result_queue.cancel_join_thread", result_queue.cancel_join_thread),
        ):
            try:
                function()
            except Exception as exc:
                _note_ipc_error(live, operation, exc)
