# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""X518 sensor with isolated acquisition and parent-owned causal publication."""

import contextlib
import logging
import multiprocessing
import queue
import threading
import time
import uuid
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..sensor import Sensor, SensorFeature, SensorRecorderLease, SensorSample, SensorSubscription
from .acquisition import X518Acquisition
from .configuration_x518 import X518SensorConfig
from .diagnostics import AcquisitionDiagnostics
from .process_transport import acquisition_worker

logger = logging.getLogger(__name__)


class X518Sensor(Sensor):
    """Acquire X518 force in a spawned process by default, exposing the Sensor API."""

    def __init__(self, config: X518SensorConfig):
        """Create disconnected state; do not open sockets or spawn a worker yet."""
        super().__init__(config)
        self._lifecycle_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._sample_ready = self._publication_condition
        self._stop_event = threading.Event()
        self._receiver_stop = threading.Event()
        self._thread = None
        self._process = None
        self._samples = self._commands = self._status = None
        self._process_stop = self._protected = None
        self._lifecycle_active = False
        self._stopping = False
        self._fault = None
        self._device_settings = None
        self._barriers = {}
        self._transport_sequence = 0
        self._transport_stats = {}
        self._diagnostics = AcquisitionDiagnostics()
        self._engine = X518Acquisition(
            config,
            self._stop_event,
            self._publish_acquired,
            self._settings_changed,
            self._notify_reconnect_required,
        )

    @property
    def _client(self):
        """Thread-mode transport seam retained for protocol unit tests."""
        return self._engine.client

    @_client.setter
    def _client(self, client):
        self._engine.client = client

    @property
    def features(self) -> dict[str, SensorFeature]:
        """Describe the configured semantic channels in newtons."""
        return {name: SensorFeature(dtype="float32", unit="N") for name in self.config.channels}

    @property
    def native_features(self) -> dict[str, SensorFeature]:
        """Describe signed device-native register values."""
        return {
            f"channel_{channel}.register": SensorFeature(dtype="int32", unit="device_count")
            for channel in sorted({mapping.channel for mapping in self.config.channels.values()})
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """Return device identity and configuration, without runtime counters."""
        result = super().provenance
        result.update(
            {
                "hardware_model": "X518",
                "device_id": f"{self.config.host}:{self.config.port}/{self.config.unit_id}",
                "channel_mapping": {name: mapping.channel for name, mapping in self.config.channels.items()},
                "timestamp_source": "time.perf_counter_ns_response_received",
                "acquisition_mode": self.config.acquisition_mode,
            }
        )
        if self._device_settings is not None:
            result["device_settings"] = {
                "unit": self._device_settings.unit,
                "decimal": self._device_settings.decimal,
                "sample_rate_hz": self._device_settings.sample_rate_hz,
            }
        return result

    @property
    def has_resources(self) -> bool:
        """Include stopped workers and IPC handles that still require cleanup."""
        return (
            self._lifecycle_active
            or self._thread is not None
            or self._process is not None
            or self._samples is not None
        )

    @property
    def is_connected(self) -> bool:
        """Return true only while acquisition and publication are both healthy."""
        return bool(
            self._lifecycle_active
            and self._fault is None
            and self._thread is not None
            and self._thread.is_alive()
            and (self._process is None or self._process.is_alive())
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return bounded, detached runtime statistics, including the last stopped run."""
        with self._publication_condition:
            result = self._diagnostics.snapshot()
            result.update(self._transport_stats)
            result.update(
                {
                    "acquisition_mode": self.config.acquisition_mode,
                    "device_sample_rate_hz": self._device_settings.sample_rate_hz
                    if self._device_settings
                    else None,
                    "target_sample_rate_hz": self.sample_rate_hz,
                    "ipc_capacity": self.config.acquisition_queue_capacity
                    if self.config.acquisition_mode == "process"
                    else 0,
                    "fault": self._fault,
                    "resources_owned": self.has_resources,
                    "worker_pid": self._process.pid if self._process is not None else None,
                }
            )
            return result

    def _settings_changed(self, settings, rate):
        with self._publication_condition:
            self._device_settings = settings
            self.sample_rate_hz = rate

    def _apply_device_settings(self, settings):
        self._engine.apply_settings(settings)

    def _read_sample_components(self):
        self._engine.settings = self._device_settings
        return self._engine.read_components()

    def _read_values(self):
        values, _native, _payload, timestamp = self._read_sample_components()
        return values, timestamp

    def _publish_acquired(self, components, missed):
        with self._publication_condition:
            sample = self._publish_sample(**components)
            self._transport_stats["missed_deadlines"] = max(
                missed, self._transport_stats.get("missed_deadlines", 0)
            )
            self._diagnostics.observe(sample, self.sample_rate_hz)

    def read_latest_before(self, target_timestamp_ns: int, max_age_ms: float | None = None) -> SensorSample:
        """Select once using parent availability, retaining bounded age diagnostics."""
        sample = super().read_latest_before(target_timestamp_ns, max_age_ms)
        with self._publication_condition:
            self._diagnostics.ages.append((target_timestamp_ns - sample.timestamp_ns) / 1e6)
        return sample

    def _set_fault(self, reason):
        with self._publication_condition:
            if self._fault is None:
                self._fault = str(reason)
            self._latch_recorder_fault(self._fault)
            for operation in self._barriers.values():
                # A completed subscribe must return its handle to the caller,
                # even if a later sample faults before that caller wakes up.
                if operation["event"].is_set():
                    continue
                operation["error"] = RuntimeError(self._fault)
                operation["event"].set()
        if self._process_stop is not None:
            self._process_stop.set()
        self._stop_event.set()

    def _check_recorder_fault(self):
        super()._check_recorder_fault()
        if self._lifecycle_active and not self._stopping:
            if self._process is not None and not self._process.is_alive():
                self._set_fault(f"X518 acquisition process exited (code={self._process.exitcode}).")
            elif self._thread is not None and not self._thread.is_alive():
                self._set_fault("X518 background reader is not running.")
        if self._fault is not None:
            self._latch_recorder_fault(self._fault)
            raise RuntimeError(self._fault)

    def _require_active(self):
        self._check_recorder_fault()
        if not self._lifecycle_active:
            raise DeviceNotConnectedError(f"{type(self).__name__} is not connected.")
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("X518 background reader is not running.")

    def connect(self) -> None:
        """Start acquisition with a bounded handshake and no implicit thread fallback."""
        with self._lifecycle_lock, self._operation_lock:
            if self.has_resources:
                raise DeviceAlreadyConnectedError("X518 still owns acquisition resources; disconnect first.")
            self._reset_framework_state()
            self._fault = None
            self._stopping = False
            self._stop_event.clear()
            self._receiver_stop.clear()
            self._diagnostics = AcquisitionDiagnostics()
            self._transport_sequence = 0
            self._transport_stats = {
                "ipc_peak_size": 0,
                "ipc_overflow_count": 0,
                "transport_gaps": 0,
                "missed_deadlines": 0,
            }
            self._engine.missed_deadlines = 0
            try:
                if self.config.acquisition_mode == "process":
                    self._connect_process()
                else:
                    self._engine.connect_initially()
                    self._lifecycle_active = True
                    self._thread = threading.Thread(target=self._reader_loop, name="X518Reader", daemon=True)
                    self._thread.start()
            except BaseException:
                self._lifecycle_active = False
                self._stop_event.set()
                self._client.close()
                self._shutdown_process()
                self._device_settings = None
                self.sample_rate_hz = self.config.sample_rate_hz
                raise
        logger.info(
            "%s connected using %s acquisition at %g Hz.",
            self,
            self.config.acquisition_mode,
            self.sample_rate_hz,
        )

    def _connect_process(self):
        context = multiprocessing.get_context("spawn")
        self._samples = context.Queue(self.config.acquisition_queue_capacity)
        command_read, self._commands = context.Pipe(duplex=False)
        self._status, status_write = context.Pipe(duplex=False)
        self._process_stop = context.Event()
        self._protected = context.Event()
        self._process = context.Process(
            target=acquisition_worker,
            args=(
                self.config,
                self._samples,
                command_read,
                status_write,
                self._process_stop,
                self._protected,
            ),
            name=f"X518Acquisition-{self.config.host}",
            daemon=True,
        )
        try:
            self._process.start()
            # Only the worker writes this queue. Closing our duplicate writer
            # makes a killed feeder deliver EOF, including a partial message,
            # instead of leaving the receiver stuck inside Queue.get().
            self._samples._writer.close()
        finally:
            command_read.close()
            status_write.close()
        retries = self.config.connect_retries
        budget = (
            self.config.startup_timeout_s
            + (retries + 1) * self.config.request_timeout_s * 3
            + self.config.connect_backoff_s * (2**retries - 1)
        )
        deadline = time.perf_counter() + budget
        while time.perf_counter() < deadline:
            if self._status.poll(0.02):
                kind, payload = self._status.recv()
                if kind == "error":
                    name, message, _overflow = payload
                    error_type = {
                        "ValueError": ValueError,
                        "TimeoutError": TimeoutError,
                        "OSError": OSError,
                        "ConnectionError": ConnectionError,
                    }.get(name, RuntimeError)
                    raise error_type(message)
                if kind == "ready":
                    self._settings_changed(*payload)
                    break
            if not self._process.is_alive():
                raise RuntimeError(
                    f"X518 acquisition process failed during startup (code={self._process.exitcode})."
                )
        else:
            raise TimeoutError("X518 acquisition process startup timed out.")
        self._lifecycle_active = True
        self._thread = threading.Thread(target=self._receive_loop, name="X518Receiver", daemon=True)
        self._thread.start()

    def _reader_loop(self):
        try:
            self._engine.run()
        except BaseException as exc:
            self._set_fault(exc)
        finally:
            with self._publication_condition:
                self._transport_stats["missed_deadlines"] = self._engine.missed_deadlines
                self._publication_condition.notify_all()

    def _receive_message(self, kind, payload):
        with self._publication_condition:
            return self._receive_message_locked(kind, payload)

    def _receive_message_locked(self, kind, payload):
        if kind == "sample":
            sequence, components, missed, peak = payload
            if sequence != self._transport_sequence:
                self._transport_stats["transport_gaps"] += 1
                raise RuntimeError(f"X518 IPC sequence mismatch: {sequence} != {self._transport_sequence}.")
            self._publish_acquired(components, missed)
            self._transport_sequence += 1
            self._transport_stats["ipc_peak_size"] = max(peak, self._transport_stats["ipc_peak_size"])
        elif kind == "settings":
            self._settings_changed(*payload)
        elif kind == "barrier":
            with self._publication_condition:
                operation = self._barriers.get(payload)
                if operation is None:
                    if self._fault is not None:
                        return True
                    raise RuntimeError("Unknown X518 publication barrier.")
                try:
                    if operation["error"] is None and self._fault is None:
                        operation["result"] = operation["callback"]()
                except BaseException as exc:
                    operation["error"] = exc
                finally:
                    operation["event"].set()
        elif kind == "stopped":
            sequence, missed = payload
            if sequence != self._transport_sequence:
                self._transport_stats["transport_gaps"] += 1
                raise RuntimeError("X518 stop boundary is incomplete.")
            self._transport_stats["missed_deadlines"] = missed
            return False
        else:
            raise RuntimeError(f"Unknown X518 transport message: {kind}.")
        return True

    def _receive_loop(self):
        status_open = True
        try:
            while not self._receiver_stop.is_set():
                if status_open:
                    try:
                        kind, payload = self._status.recv() if self._status.poll() else (None, None)
                    except (EOFError, BrokenPipeError):
                        status_open = False
                    else:
                        if kind == "error":
                            _name, message, overflow = payload
                            with self._publication_condition:
                                self._transport_stats["ipc_overflow_count"] = overflow
                            self._set_fault(message)
                        elif kind == "finished":
                            with self._publication_condition:
                                self._transport_stats.update(payload)
                try:
                    message = self._samples.get(timeout=0.01)
                except (EOFError, OSError) as exc:
                    # Fault status is sent before the child closes the sample
                    # channel, but Queue.get() may wake first. Preserve the cause.
                    with contextlib.suppress(EOFError, OSError):
                        if status_open and self._status.poll(0.1):
                            kind, payload = self._status.recv()
                            if kind == "error":
                                with self._publication_condition:
                                    self._transport_stats["ipc_overflow_count"] = payload[2]
                                self._set_fault(payload[1])
                    if self._fault is None:
                        raise RuntimeError("X518 acquisition channel closed before stop boundary.") from exc
                    break
                except queue.Empty:
                    if not self._process.is_alive():
                        if self._fault is None:
                            raise RuntimeError(
                                f"X518 acquisition process exited without a stop boundary (code={self._process.exitcode})."
                            ) from None
                        break
                    continue
                if not self._receive_message(*message):
                    if not self._stopping:
                        raise RuntimeError("X518 acquisition stopped unexpectedly.")
                    break
        except BaseException as exc:
            self._set_fault(exc)
        finally:
            with self._publication_condition:
                self._publication_condition.notify_all()

    def _barrier(self, callback):
        """Execute a framework mutation atomically at the child stream boundary."""
        self._require_active()
        token = uuid.uuid4().hex
        operation = {"callback": callback, "event": threading.Event(), "error": None}
        with self._publication_condition:
            self._barriers[token] = operation
        try:
            self._commands.send(token)
            if not operation["event"].wait(max(2.0, self.config.request_timeout_s * 3)):
                self._set_fault("X518 publication barrier timed out; stream boundary is unknown.")
            if operation["error"] is not None:
                raise operation["error"]
            return operation["result"]
        except (OSError, EOFError) as exc:
            self._set_fault(exc)
            raise
        finally:
            with self._publication_condition:
                self._barriers.pop(token, None)

    def subscribe(self, capacity: int) -> SensorSubscription:
        """Start after all samples preceding the ordered subscription barrier."""
        with self._operation_lock:
            if self._process is None:
                return super().subscribe(capacity)
            return self._barrier(lambda: super(X518Sensor, self).subscribe(capacity))

    def unsubscribe(self, subscription: SensorSubscription) -> None:
        """Publish the entire pre-barrier tail before closing this subscription."""
        with self._operation_lock:
            if subscription.closed:
                return
            if self._process is not None and self.is_connected:
                try:
                    self._barrier(lambda: super(X518Sensor, self).unsubscribe(subscription))
                    return
                except Exception as exc:
                    self._set_fault(exc)
            # Faulted episodes cannot commit, but their existing queues must still drain.
            super().unsubscribe(subscription)

    def acquire_recorder(self) -> SensorRecorderLease:
        """Protect reconnection before confirming ownership at a stream boundary."""
        with self._operation_lock:
            if self._process is not None and self.config.required:
                self._protected.set()
            lease = super().acquire_recorder()
            if self._process is not None and self.config.required:
                try:
                    self._barrier(lambda: None)
                except BaseException:
                    super().release_recorder(lease)
                    raise
            return lease

    def release_recorder(self, lease: SensorRecorderLease) -> None:
        """Keep the no-reconnect latch through Recorder draining and transaction cleanup."""
        with self._operation_lock:
            super().release_recorder(lease)
            if self._protected is not None and not self._recorder_leases and self._fault is None:
                self._protected.clear()

    def _shutdown_process(self):
        process = self._process
        if self._process_stop is not None:
            self._process_stop.set()
        timeout = max(2.0, self.config.request_timeout_s * 3)
        if process is not None and process.pid is not None:
            process.join(timeout)
            if process.is_alive():
                self._set_fault("X518 acquisition shutdown timed out; terminating worker.")
                process.terminate()
                process.join(timeout)
                if process.is_alive():
                    process.kill()
                    process.join(timeout)
                if process.is_alive():
                    raise RuntimeError("X518 process did not stop; resource ownership retained.")
        thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout)
            if thread.is_alive():
                self._set_fault("X518 receiver shutdown timed out; handoff is incomplete.")
                self._receiver_stop.set()
                thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("X518 receiver did not stop; resource ownership retained.")
        if self._status is not None:
            # The ordered stop marker can arrive before the separate final
            # counters. Read them after both worker and receiver have stopped.
            while True:
                try:
                    if not self._status.poll():
                        break
                    kind, payload = self._status.recv()
                except (EOFError, BrokenPipeError):
                    break
                if kind == "finished":
                    with self._publication_condition:
                        self._transport_stats.update(payload)
                elif kind == "error":
                    self._set_fault(payload[1])
        self._thread = None
        if self._samples is not None:
            self._samples.close()
            self._samples.join_thread()
            self._samples._reader.close()
            self._samples._writer.close()
        for connection in (self._commands, self._status):
            if connection is not None:
                connection.close()
        if process is not None:
            process.close()
        self._process = self._samples = self._commands = self._status = None
        self._process_stop = self._protected = None

    def disconnect(self) -> None:
        """Stop acquisition and release resources even after a worker has died."""
        with self._lifecycle_lock, self._operation_lock:
            if not self.has_resources:
                raise DeviceNotConnectedError("X518 is not connected.")
            self._notify_reconnect_required(ConnectionError("X518 disconnected"))
            self._stopping = True
            self._lifecycle_active = False
            self._stop_event.set()
            with self._publication_condition:
                self._publication_condition.notify_all()
            if self._process is not None or self._samples is not None:
                self._shutdown_process()
            else:
                self._client.close()
                thread = self._thread
                if thread is not None and thread.ident is not None:
                    thread.join(max(2.0, self.config.request_timeout_s * 3))
                    if thread.is_alive():
                        self._set_fault("X518 reader shutdown timed out; ownership retained.")
                        raise RuntimeError(self._fault)
                self._client.close()
                self._thread = None
            logger.info("%s disconnected.", self)

    def __del__(self):
        """Release failed-worker handles when explicit cleanup was omitted."""
        with contextlib.suppress(Exception):
            if self.has_resources:
                self.disconnect()

    def __str__(self):
        """Identify this connection in diagnostics."""
        return f"X518Sensor({self.config.host}:{self.config.port})"
