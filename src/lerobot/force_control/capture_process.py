"""Isolate capture, encoding and commit work from the motor interpreter.

The producer retains hardware ownership. Only immutable completed ticks,
original Sensor samples and explicit end boundaries cross the bounded channel.
The child reuses DemonstrationController, LeRobotDataset and Sidecar v2.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import queue
import threading
import time
import uuid
from contextlib import suppress
from types import SimpleNamespace
from typing import TYPE_CHECKING

from lerobot.datasets.sensor_raw_input import RawStreamDescriptor, RawStreamInput
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

from .demonstration import CameraCapture, DemonstrationController, TeacherRecordConfig
from .owner import OwnerSubscription

if TYPE_CHECKING or (_av_available and _datasets_available):
    from lerobot.datasets.sensor_stream import SensorStreamRecorder


class _StopFlag:
    """Monotonic byte without a cross-process condition wait on a crashed child."""

    def __init__(self, context):
        self.value = context.RawValue("b", 0)

    def set(self):
        self.value.value = 1

    def is_set(self):
        return bool(self.value.value)


class _CompletedTicks:
    """Recorder endpoint only; no command posting or motor transport exists here."""

    def __init__(self, joint_names, control_hz):
        self.runtime = SimpleNamespace(config=SimpleNamespace(joint_names=joint_names, control_hz=control_hz))
        self.is_running = False
        self.has_resources = False
        self._latest = None
        self._subscriptions = []
        self._lock = threading.Lock()

    def snapshot(self):
        return self._latest

    def subscribe(self, capacity):
        subscription = OwnerSubscription(capacity)
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription):
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            subscription.closed = True

    def accept(self, snapshot):
        self._latest = snapshot
        with self._lock:
            for subscription in self._subscriptions:
                try:
                    subscription.queue.put_nowait(snapshot)
                except queue.Full as exc:
                    subscription.error = "Recorder completed-tick queue overflow"
                    raise RuntimeError(subscription.error) from exc


class _ChildController(DemonstrationController):
    def __init__(self, *args, **kwargs):
        self.start_processed = False
        super().__init__(*args, **kwargs)

    def _start(self, config):
        try:
            super()._start(config)
        finally:
            self.start_processed = True


def _latest_put(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        # Telemetry may be replaced; samples and boundaries never use this path.
        try:
            channel.get_nowait()
        except queue.Empty:
            return
        with suppress(queue.Full):
            channel.put_nowait(value)


def _capture_worker(config, descriptor, names, hz, instance, provenance, incoming, outgoing, cancel, cameras):
    # This channel contains replaceable status/previews, never samples or commit
    # receipts. A full preview pipe must not keep an otherwise closed child alive.
    outgoing.cancel_join_thread()
    controller = None
    source = RawStreamInput(descriptor, 0)
    owner = _CompletedTicks(names, hz)
    token = released = None
    received_controls = 0
    previous_tick = None
    next_status = next_preview = 0.0
    preview = {}
    parent = multiprocessing.parent_process()

    def recorder_factory(root, _sensors):
        return SensorStreamRecorder.from_raw_inputs(root, {instance: source})

    def emit(**extra):
        status = controller.status() if controller else {"state": "error", "active": False}
        _latest_put(outgoing, {"status": status, "preview": preview, "released": released, **extra})

    try:
        provenance = {
            **provenance,
            "capture_process": {
                "writer_pid": os.getpid(),
                "producer_pid": os.getppid(),
                "transport": "bounded_original_samples_and_completed_ticks_v1",
            },
        }
        controller = _ChildController(
            owner,
            descriptor,
            force_instance=instance,
            provenance=provenance,
            camera_factory=cameras,
            recorder_factory=recorder_factory,
        )
        controller._completion_cancel = cancel
        controller.prepare_before_control(config)
        emit(ready=True)
        while True:
            if parent is not None and not parent.is_alive():
                cancel.set()
                raise RuntimeError("Capture producer exited before recorder shutdown")
            try:
                kind, payload = incoming.get(timeout=0.01)
            except queue.Empty:
                kind, payload = None, None
            if cancel.is_set() and kind is None:
                break
            if kind == "close":
                break
            if kind == "owner":
                owner.is_running, owner._latest = payload
            elif kind == "start":
                if token is not None:
                    raise RuntimeError("Capture producer started before prior episode release")
                token, descriptor, first_sequence, value = payload
                source = RawStreamInput(descriptor, first_sequence)
                controller.sensor = descriptor
                controller._recorder.close()
                controller._recorder = recorder_factory(controller._dataset.root, {})
                controller.start_processed = False
                received_controls, previous_tick = 0, None
                controller.submit("record_start", value)
            elif kind == "data":
                message_token, samples, snapshots = payload
                if message_token != token or token is None:
                    raise RuntimeError("Capture data belongs to another episode")
                for sample in samples:
                    source.accept(sample)
                for snapshot in snapshots:
                    if previous_tick is not None and snapshot.tick_id != previous_tick + 1:
                        raise RuntimeError("Capture IPC control tick gap")
                    previous_tick = snapshot.tick_id
                    received_controls += 1
                    owner.accept(snapshot)
            elif kind == "end":
                message_token, count, end_sequence, command, value = payload
                if message_token != token or count != received_controls:
                    raise RuntimeError("Capture IPC control tail incomplete")
                source.seal(end_sequence)
                controller.submit(command, value)
            elif kind == "outcome":
                controller.submit("record_outcome", payload)
            if token is not None and controller.start_processed and controller._uid is None:
                if controller._cleanup_pending:
                    raise RuntimeError("Capture child retained failed episode resources")
                released, token = token, None
            now = time.monotonic()
            if now >= next_status:
                if now >= next_preview:
                    preview = controller.preview()
                    next_preview = now + 0.25
                emit(ready=True)
                next_status = now + 0.05
    except BaseException as exc:
        cancel.set()
        emit(error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if controller is not None:
                controller.close()
        except BaseException as exc:
            emit(error=f"Capture cleanup failed: {type(exc).__name__}: {exc}")
        emit(closed=True)


class ProcessDemonstrationController:
    """Small producer bridge; all large allocations and file work stay in the child."""

    def __init__(
        self, owner, sensor, *, force_instance="gripper_force", provenance=None, camera_factory=CameraCapture
    ):
        """Allocate bounded channels without connecting hardware or starting a child."""
        self.owner, self.sensor = owner, sensor
        self.force_instance, self.provenance = force_instance, provenance or {}
        self._camera_factory = camera_factory
        self._context = multiprocessing.get_context("spawn")
        self._cancel = _StopFlag(self._context)
        self._incoming = self._context.Queue(maxsize=256)
        self._outgoing = self._context.Queue(maxsize=4)
        self._commands = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._status = {"state": "idle", "active": False, "episode_count": 0, "frame_count": 0, "error": None}
        self._preview = {}
        self._process = self._thread = None
        self._lease = self._raw = self._ticks = None
        self._token = None
        self._count = 0
        self._error = None
        self._ended = False
        self._restart = None
        self._closed = False

    @property
    def has_resources(self):
        """Keep ownership until producer and writer exit have both been confirmed."""
        return not self._closed and self._process is not None

    def status(self):
        """Return the latest detached recorder telemetry."""
        with self._lock:
            return copy.deepcopy(self._status)

    def preview(self):
        """Return small child-generated camera previews."""
        with self._lock:
            return copy.deepcopy(self._preview)

    def prepare_before_control(self, config):
        """Spawn and prewarm the recorder before any motor or sampling worker starts."""
        require_package("datasets", extra="dataset")
        require_package("av", extra="dataset")
        if self.owner.is_running or self.owner.has_resources or self._process is not None:
            raise RuntimeError("Capture process preparation requires an unopened motor owner")
        config.validate()
        self._process = self._context.Process(
            target=_capture_worker,
            args=(
                config,
                RawStreamDescriptor.from_sensor(self.sensor),
                tuple(self.owner.runtime.config.joint_names),
                self.owner.runtime.config.control_hz,
                self.force_instance,
                self.provenance,
                self._incoming,
                self._outgoing,
                self._cancel,
                self._camera_factory,
            ),
            name="teacher-capture-writer",
            daemon=False,
        )
        self._process.start()
        self._thread = threading.Thread(target=self._bridge, name="teacher-capture-producer", daemon=False)
        self._thread.start()
        if not self._ready.wait(60):
            raise TimeoutError("Capture process preparation timed out")
        if self._error:
            raise RuntimeError(self._error)

    def submit(self, kind, value=None):
        """Queue capture intent without waiting on hardware or file I/O."""
        if kind not in ("record_start", "record_finish", "record_discard", "record_outcome"):
            raise ValueError("Unsupported recording event")
        if self._stop.is_set() or self._error or not self._ready.is_set():
            raise RuntimeError(self._error or "Capture process is not ready")
        try:
            self._commands.put_nowait((kind, copy.deepcopy(value or {})))
        except queue.Full as exc:
            raise RuntimeError("Recording event queue is full") from exc

    def _send(self, kind, payload):
        try:
            self._incoming.put_nowait((kind, payload))
        except queue.Full as exc:
            raise RuntimeError("Capture IPC overflow; episode cannot be complete") from exc

    def _receive(self):
        while True:
            try:
                event = self._outgoing.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                self._status = event["status"]
                self._preview = event.get("preview", {})
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("released") == self._token and self._token is not None:
                self._release()
                if self._restart is not None:
                    value, self._restart = self._restart, None
                    self._begin(value)
            if event.get("ready"):
                self._ready.set()

    def _begin(self, value):
        if self._token is not None or not self.owner.is_running:
            raise RuntimeError("Recording requires a running owner and no pending episode")
        config = TeacherRecordConfig(**value["config"])
        config.validate()
        self._record_value = copy.deepcopy(value)
        self._lease = self.sensor.acquire_recorder()
        self._raw = self.sensor.subscribe(self.sensor.config.resolve_recorder_queue_capacity())
        self._lease.start_sequence = self._raw.start_sequence
        self._ticks = self.owner.subscribe(config.queue_capacity)
        self._token, self._count, self._ended = str(uuid.uuid4()), 0, False
        self._send("owner", (self.owner.is_running, self.owner.snapshot()))
        self._send(
            "start",
            (self._token, RawStreamDescriptor.from_sensor(self.sensor), self._raw.start_sequence, value),
        )

    @staticmethod
    def _take(subscription, limit):
        result = []
        for _ in range(limit):
            try:
                result.append(subscription.queue.get_nowait())
            except queue.Empty:
                break
        return result

    def _drain(self):
        if self._token is None or self._ended:
            return False
        self.sensor._check_recorder_fault()
        if self._lease.error or self._raw.overflowed:
            raise RuntimeError(self._lease.error or "Capture Raw producer overflow")
        self._ticks.check_health()
        samples, snapshots = self._take(self._raw, 32), self._take(self._ticks, 8)
        if samples or snapshots:
            self._send("data", (self._token, samples, snapshots))
            self._count += len(snapshots)
            return True
        return False

    def _end(self, kind, value):
        if self._token is None or self._ended:
            raise RuntimeError("No episode is recording")
        if kind == "record_discard" and value.get("rerecord"):
            self._restart = copy.deepcopy(self._record_value)
            value = {**value, "rerecord": False}
        self.owner.unsubscribe(self._ticks)
        self.sensor.unsubscribe(self._raw)
        while self._drain():
            pass
        self._send("end", (self._token, self._count, self._raw.end_sequence, kind, value))
        self._ended = True

    def _freeze(self):
        if self._ticks is not None:
            self.owner.unsubscribe(self._ticks)
        if self._raw is not None:
            self.sensor.unsubscribe(self._raw)

    def _release(self):
        self._freeze()
        if self._lease is not None:
            self.sensor.release_recorder(self._lease)
        self._lease = self._raw = self._ticks = self._token = None

    def _bridge(self):
        next_owner = 0.0
        try:
            while not self._stop.is_set():
                self._receive()
                if self._lease is not None:
                    self.sensor._check_recorder_fault()
                    if self._lease.error:
                        raise RuntimeError(self._lease.error)
                if not self._process.is_alive():
                    raise RuntimeError(f"Capture process exited unexpectedly ({self._process.exitcode})")
                try:
                    kind, value = self._commands.get_nowait()
                except queue.Empty:
                    kind, value = None, None
                if kind == "record_start":
                    self._begin(value)
                elif kind in ("record_finish", "record_discard"):
                    self._end(kind, value)
                elif kind == "record_outcome":
                    self._send("outcome", value)
                self._drain()
                if time.monotonic() >= next_owner:
                    self._send("owner", (self.owner.is_running, self.owner.snapshot(allow_empty=True)))
                    next_owner = time.monotonic() + 0.05
                self._stop.wait(0.002)
        except BaseException as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._cancel.set()
            with self._lock:
                self._status.update(state="error", active=False, file_complete=False, error=self._error)
            self._ready.set()
        finally:
            self._freeze()
            if self._process.is_alive():
                try:
                    if self._token is not None and not self._ended:
                        self._end("record_discard", {})
                    self._send("close", None)
                except Exception:
                    # Shared cancellation already withholds completion. Keep ownership
                    # until confirmed process exit; never claim a missing tail arrived.
                    self._cancel.set()

    def close(self, timeout_s=30):
        """Withhold completion, drain inputs and confirm writer exit before releasing leases."""
        if self._closed:
            return
        self._cancel.set()
        self._stop.set()
        deadline = time.monotonic() + timeout_s
        if self._thread is not None:
            self._thread.join(max(0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise TimeoutError("Capture producer shutdown unconfirmed; ownership retained")
        if self._process is not None:
            self._process.join(max(0, deadline - time.monotonic()))
            if self._process.is_alive():
                raise TimeoutError("Capture writer shutdown unconfirmed; ownership retained")
        self._release()
        for channel in (self._incoming, self._outgoing):
            channel.cancel_join_thread()
            channel.close()
        self._closed = True
