#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Native-rate Raw/Sync sensor sidecar recording."""

from __future__ import annotations

import contextlib
import copy
import ctypes
import json
import logging
import math
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from lerobot.sensors import Sensor, SensorFeature, SensorSubscription
from lerobot.sensors.diagnostics import SensorQueueDiagnostics
from lerobot.sensors.sensor import SensorRecorderLease
from lerobot.utils.errors import DeviceNotConnectedError

from .sensor_spool import SensorParquetSpoolWriter
from .sensor_transaction import (
    SensorTransaction,
    TransactionRecoveryManager,
    parquet_file_record,
    sha256_file,
    validate_episode_uid,
)

SIDECAR_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


class SensorQueueOverflowError(RuntimeError):
    """Raised when a native-rate recorder subscriber loses samples."""


class SensorRecorderError(RuntimeError):
    """Raised when a recorder worker or sidecar writer fails."""


def _feature_schema(features: dict[str, SensorFeature]) -> list[dict[str, Any]]:
    return [
        {"name": name, "dtype": feature.dtype, "unit": feature.unit, "shape": []}
        for name, feature in features.items()
    ]


def build_sensor_stream_manifest(sensors: dict[str, Sensor]) -> dict[str, Any]:
    """Build the stable Dataset-level contract without hardware provenance."""
    streams: dict[str, Any] = {}
    for instance, sensor in sensors.items():
        selected = (
            list(sensor.features) if sensor.config.state_features is None else sensor.config.state_features
        )
        unknown = set(selected) - set(sensor.features)
        if unknown:
            raise ValueError(f"Sensor {instance!r} state_features contains unknown paths: {sorted(unknown)}.")
        streams[instance] = {
            "semantic_features": _feature_schema(sensor.features),
            "native_features": _feature_schema(sensor.native_features),
            "state_features": list(selected),
            "native_values_capable": bool(sensor.native_features),
            "native_payload_capable": True,
        }
    return {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "storage_layout": {
            "type": "per_episode_parquet",
            "version": 1,
            "raw_path_template": "raw/sensors/{instance}/{episode_uid}.parquet",
            "sync_path_template": "raw/sync/{episode_uid}.parquet",
            "episode_metadata_path_template": "meta/sensor_episodes/{episode_uid}.json",
            "transaction_path_template": "meta/sensor_transactions/{episode_uid}.json",
        },
        "streams": streams,
        "clock": {
            "canonical_domain": "host_monotonic_ns",
            "measurement_field": "timestamp_ns",
            "availability_field": "arrival_timestamp_ns",
        },
        "raw_schema": "typed_semantic_and_native_structs_v1",
        "sync_schema": "one_row_per_main_dataset_frame_v1",
        "selection": {
            "current": "latest_before_timestamp_and_arrival_with_static_max_age",
            "window": "causal_previous_hold",
            "default_max_age_ms": "ceil(3000/static_sample_rate_hz)",
        },
    }


def ensure_sensor_stream_manifest(root: Path, sensors: dict[str, Sensor]) -> dict[str, Any]:
    """Create or exact-match the stable manifest for resume safety."""
    root = Path(root)
    path = root / "meta" / "sensor_streams.json"
    expected = build_sensor_stream_manifest(sensors)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "Sensor semantic schema or state feature selection differs from sensor_streams.json."
            )
        return actual
    _atomic_json(path, expected)
    return expected


class SensorDatasetWriterLock:
    """Exclusive Dataset-level lock for a sensor sidecar writer process."""

    def __init__(self, root: Path) -> None:
        """Acquire an exclusive lock file or fail immediately."""
        self.path = Path(root) / ".sensor-writer.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_token = str(uuid.uuid4())
        self._released = False
        self._acquire()

    def _acquire(self) -> None:
        """Acquire the lock, recovering only a verifiably dead writer."""
        payload = {
            "pid": os.getpid(),
            "created_ns": time.time_ns(),
            "owner_token": self._owner_token,
        }
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                try:
                    owner = json.loads(self.path.read_text(encoding="utf-8"))
                    owner_pid = int(owner["pid"])
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as parse_error:
                    raise SensorRecorderError(
                        f"Sensor Dataset has an unreadable writer lock: {self.path}."
                    ) from parse_error
                if _process_is_alive(owner_pid):
                    raise SensorRecorderError(
                        f"Sensor Dataset already has a live writer (pid={owner_pid}): {self.path}."
                    ) from exc
                with contextlib.suppress(FileNotFoundError):
                    self.path.unlink()
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
                handle.flush()
                os.fsync(handle.fileno())
            return
        raise SensorRecorderError(f"Could not acquire Sensor Dataset writer lock: {self.path}.")

    def release(self) -> None:
        """Release this writer's lock idempotently."""
        if not self._released:
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                owner = None
            except (OSError, json.JSONDecodeError):
                owner = {}
            if owner is not None and owner.get("owner_token") == self._owner_token:
                self.path.unlink(missing_ok=True)
            self._released = True

    def __enter__(self):
        """Return the acquired lock."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release the lock on context exit."""
        self.release()


class SensorStreamRecorder:
    """Drain independent subscriber queues into per-episode Raw/Sync sidecars."""

    def __init__(self, root: Path, sensors: dict[str, Sensor]) -> None:
        """Validate the Dataset contract and acquire its single-writer lock."""
        self.root = Path(root)
        self.sensors = sensors
        for instance, sensor in sensors.items():
            if re.fullmatch(r"[a-z][a-z0-9_]*", instance) is None:
                raise ValueError(f"Invalid Sensor instance namespace: {instance!r}.")
            sensor.config.resolve_recorder_queue_capacity()
        self._writer_lock = SensorDatasetWriterLock(self.root)
        try:
            recovered = TransactionRecoveryManager(self.root).recover(writer_lock=self._writer_lock)
            if recovered is not None and recovered.state == "QUARANTINED":
                raise SensorRecorderError("Active transaction was quarantined during recovery.")
            self.manifest = ensure_sensor_stream_manifest(self.root, sensors)
        except Exception:
            self._writer_lock.release()
            raise
        self._active = False
        self._episode_uid: str | None = None
        self._episode_index: int | None = None
        self._episode_start_ns = 0
        self._main_precondition: dict[str, Any] | None = None
        self._subscriptions: dict[str, SensorSubscription] = {}
        self._recorder_leases: dict[str, SensorRecorderLease] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._spools: dict[str, SensorParquetSpoolWriter] = {}
        self._sync_spool: SensorParquetSpoolWriter | None = None
        self._sync_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._sync_diagnostics = SensorQueueDiagnostics("Sensor Sync")
        self._sync_overflow_count = 0
        self._last_diagnostics = None
        self._sync_count = 0
        self._accept_sync = False
        self._rows_lock = threading.RLock()
        self._worker_errors: dict[str, BaseException] = {}
        self._transaction: SensorTransaction | None = None
        self._minimum_timestamp_ns: int | None = None
        self._closed = False
        self._bound_dataset = None

    @property
    def episode_uid(self) -> str | None:
        """Return the active episode UUID, if any."""
        return self._episode_uid

    @property
    def is_active(self) -> bool:
        """Whether native-rate acquisition is currently subscribed."""
        return self._active

    @property
    def has_prepared_episode(self) -> bool:
        """Whether a stopped episode is waiting for main Dataset commit."""
        return self._transaction is not None and self._transaction.state == "PREPARED"

    @property
    def worker_threads(self) -> tuple[threading.Thread, ...]:
        """Expose worker identities for deterministic cleanup tests."""
        return tuple(self._threads.values())

    @property
    def diagnostics(self) -> dict:
        """Constant-space runtime telemetry, retained for the last completed episode."""
        if self._episode_uid is None and self._last_diagnostics is not None:
            return copy.deepcopy(self._last_diagnostics)
        streams = {}
        for name, spool in self._spools.copy().items():
            streams[name] = spool.diagnostics()
            subscription = self._subscriptions.get(name)
            if subscription is not None:
                streams[name].update(
                    subscription.diagnostics.snapshot(
                        subscription.queue, overflow_count=subscription.overflow_count
                    )
                )
        sync = self._sync_spool.diagnostics() if self._sync_spool else {}
        sync.update(
            self._sync_diagnostics.snapshot(self._sync_queue, overflow_count=self._sync_overflow_count)
        )
        return {
            "episode_uid": self._episode_uid,
            "streams": streams,
            "sync": sync,
            "state": self._transaction.state if self._transaction else None,
            "worker_errors": {name: str(error) for name, error in self._worker_errors.copy().items()},
        }

    def start_episode(self, episode_index: int, *, episode_uid: str | None = None, dataset=None) -> str:
        """Subscribe and start one native-rate worker per sensor."""
        if self._closed:
            raise SensorRecorderError("Sensor recorder is closed.")
        if self._active:
            raise SensorRecorderError("A sensor episode is already active.")
        if self._transaction is not None:
            raise SensorRecorderError("A PREPARED sensor episode must be committed or aborted first.")
        recovered = TransactionRecoveryManager(self.root).recover(writer_lock=self._writer_lock)
        if recovered is not None and recovered.state == "QUARANTINED":
            raise SensorRecorderError("Active transaction was quarantined during recovery.")
        resolved_uid = validate_episode_uid(episode_uid or str(uuid.uuid4()))
        if (self.root / "meta/sensor_transactions" / f"{resolved_uid}.json").exists():
            raise SensorRecorderError("Sensor episode UID already has a transaction.")
        self._episode_uid = resolved_uid
        self._episode_index = episode_index
        self._episode_start_ns = time.perf_counter_ns()
        self._stop_event.clear()
        self._spools = {}
        self._sync_spool = None
        self._sync_count = 0
        self._sync_queue = queue.Queue(maxsize=4096)
        self._sync_diagnostics = SensorQueueDiagnostics("Sensor Sync")
        self._sync_overflow_count = 0
        self._accept_sync = True
        self._worker_errors = {}
        self._transaction = None
        self._minimum_timestamp_ns = None
        try:
            for instance, sensor in self.sensors.items():
                self._recorder_leases[instance] = sensor.acquire_recorder()
            self._active = True
            self._transaction = SensorTransaction.begin(self.root, resolved_uid, episode_index)
            self._main_precondition = self._transaction.journal["main_precondition"]
            if dataset is not None:
                self._bind_dataset(dataset)
            staging = self.root / ".sensor-staging" / resolved_uid
            self._sync_spool = SensorParquetSpoolWriter(
                staging / "spool" / "sync",
                _sync_arrow_schema(self.sensors),
                4096,
                0.5,
                episode_uid=resolved_uid,
                instance="__sync__",
            )
            sync_thread = threading.Thread(
                target=self._drain_sync, name="SensorStreamRecorder-Sync", daemon=True
            )
            self._threads["__sync__"] = sync_thread
            sync_thread.start()
            for instance, sensor in self.sensors.items():
                self._spools[instance] = SensorParquetSpoolWriter(
                    staging / "spool" / instance,
                    _raw_arrow_schema(sensor),
                    sensor.config.recorder_flush_rows,
                    sensor.config.recorder_flush_interval_s,
                    episode_uid=resolved_uid,
                    instance=instance,
                )
                subscription = sensor.subscribe(sensor.config.resolve_recorder_queue_capacity())
                subscription.diagnostics.label = f"Sensor Raw {instance}"
                self._recorder_leases[instance].start_sequence = subscription.start_sequence
                self._subscriptions[instance] = subscription
                thread = threading.Thread(
                    target=self._drain,
                    args=(instance, subscription),
                    name=f"SensorStreamRecorder-{instance}",
                    daemon=True,
                )
                self._threads[instance] = thread
                thread.start()
        except Exception as exc:
            self._stop_workers()
            if self._transaction is not None:
                self._quarantine_failed_episode(exc)
            else:
                self._reset_episode()
            raise
        return self._episode_uid

    def _bind_dataset(self, dataset) -> None:
        writer = getattr(dataset, "writer", None)
        if writer is not None:
            writer.set_sensor_transaction(self._transaction)
        else:
            dataset._sensor_transaction = self._transaction
        self._bound_dataset = dataset

    def _drain(self, instance: str, subscription: SensorSubscription) -> None:
        spool = self._spools[instance]
        try:
            while not self._stop_event.is_set() or not subscription.queue.empty():
                try:
                    sample = subscription.get(timeout=0.05)
                except queue.Empty:
                    spool.flush_due()
                    continue
                spool.append(self._raw_row(instance, sample))
        except BaseException as exc:  # worker failures must reach the control loop
            self._worker_errors[instance] = exc
        finally:
            try:
                spool.close()
            except BaseException as exc:
                self._worker_errors[instance] = exc

    def _drain_sync(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._sync_queue.empty():
                try:
                    row = self._sync_queue.get(timeout=0.05)
                except queue.Empty:
                    self._sync_spool.flush_due()
                    continue
                self._sync_spool.append(row)
        except BaseException as exc:
            self._worker_errors["__sync__"] = exc
        finally:
            try:
                self._sync_spool.close()
            except BaseException as exc:
                self._worker_errors["__sync__"] = exc

    def wait_until_ready(self) -> None:
        """Wait for a valid causal sample from this episode on every required stream."""
        pending = {name for name, sensor in self.sensors.items() if sensor.config.required}
        deadlines = {
            name: time.perf_counter() + self.sensors[name].config.startup_timeout_s for name in pending
        }
        while pending:
            self.check_health()
            for name in tuple(pending):
                sensor = self.sensors[name]
                try:
                    sensor.read_latest_before(time.perf_counter_ns())
                except (RuntimeError, DeviceNotConnectedError):
                    if time.perf_counter() >= deadlines[name]:
                        raise SensorRecorderError(
                            f"Required sensor {name!r} startup timeout for this episode."
                        ) from None
                else:
                    pending.remove(name)
            if pending:
                self._stop_event.wait(0.005)

    def _raw_row(self, instance: str, sample) -> dict[str, Any]:
        sensor_config = self.sensors[instance].config
        return {
            "episode_uid": self._episode_uid,
            "episode_time_ns": sample.timestamp_ns - self._episode_start_ns,
            "timestamp_ns": sample.timestamp_ns,
            "arrival_timestamp_ns": sample.arrival_timestamp_ns,
            "sequence": sample.sequence,
            "hardware_timestamp_ns": sample.hardware_timestamp_ns,
            "hardware_sequence": sample.hardware_sequence,
            "is_valid": sample.is_valid,
            "status": sample.status,
            "error": sample.error,
            "values": sample.values,
            "native_values": sample.native_values if sensor_config.record_native_values else None,
            "native_payload": sample.native_payload if sensor_config.record_native_payload else None,
        }

    def record_sync(self, frame_index: int | None, capture_metadata: dict[str, Any]) -> None:
        """Record synchronization metadata only for a frame handed to add_frame."""
        self.check_health()
        if not self._active or not self._accept_sync:
            raise SensorRecorderError("No active sensor episode.")
        expected_frame_index = self._sync_count
        resolved_frame_index = expected_frame_index if frame_index is None else frame_index
        if resolved_frame_index != expected_frame_index:
            raise SensorRecorderError(
                f"Sensor Sync frame_index must be contiguous: expected {expected_frame_index}, "
                f"got {resolved_frame_index}."
            )
        row = {
            "episode_uid": self._episode_uid,
            "frame_index": resolved_frame_index,
            "observation_start_ns": capture_metadata["observation_start_ns"],
            "frame_anchor_ns": capture_metadata["frame_anchor_ns"],
            "observation_complete_ns": capture_metadata["observation_complete_ns"],
            "hardware_observation_timestamps": capture_metadata.get("hardware_observation_timestamps"),
            "sensors": capture_metadata.get("sensors", {}),
        }
        with self._rows_lock:
            if not self._accept_sync:
                raise SensorRecorderError("Sensor Sync is stopping.")
            try:
                self._sync_queue.put_nowait(copy.deepcopy(row))
                self._sync_diagnostics.observe(self._sync_queue)
            except queue.Full as exc:
                self._sync_overflow_count += 1
                self._worker_errors["__sync__"] = exc
                raise SensorQueueOverflowError("Sensor Sync queue overflowed; episode is invalid.") from exc
            self._sync_count += 1

    def check_health(self) -> None:
        """Raise immediately on worker failure or subscriber overflow."""
        for instance, lease in self._recorder_leases.items():
            if lease.error is not None:
                raise SensorRecorderError(f"Sensor {instance!r} episode fault: {lease.error}")
        if self._worker_errors:
            instance, error = next(iter(self._worker_errors.items()))
            raise SensorRecorderError(f"Sensor recorder worker {instance!r} failed: {error}") from error
        overflowed = [name for name, item in self._subscriptions.items() if item.overflowed]
        if overflowed:
            raise SensorQueueOverflowError(
                f"Sensor recorder queue overflowed for streams {overflowed}; episode is invalid."
            )

    def trim_before(self, timestamp_ns: int, *, max_age_ms: float | None = None) -> None:
        """Trim only with a known earliest grid and finite age; unknown windows retain Raw."""
        if not self._active:
            raise SensorRecorderError("Cannot trim a sensor recorder without an active episode.")
        if max_age_ms is None:
            return
        if isinstance(max_age_ms, bool) or not math.isfinite(max_age_ms) or max_age_ms < 0:
            raise ValueError("Safe trim requires finite non-negative max_age_ms.")
        with self._rows_lock:
            if self._sync_count:
                raise SensorRecorderError("Safe trim must precede retained Sync frames.")
            self._minimum_timestamp_ns = int(timestamp_ns) - int(max_age_ms * 1_000_000)

    def prepare_episode(self, *, task_info: Any = None) -> SensorTransaction:
        """Stop workers, write staging artifacts, and persist PREPARED."""
        if not self._active or self._episode_uid is None or self._episode_index is None:
            raise SensorRecorderError("No active sensor episode.")
        self._stop_workers()
        try:
            return self._prepare_closed_episode(task_info=task_info)
        except Exception as exc:
            if self._episode_uid is not None:
                self._quarantine_failed_episode(exc)
            raise

    def _quarantine_failed_episode(self, exc: Exception) -> None:
        self._transaction.quarantine(str(exc))
        self._reset_episode()

    def _prepare_closed_episode(self, *, task_info: Any) -> SensorTransaction:
        staging_root = self.root / ".sensor-staging" / self._episode_uid
        self.check_health()
        self._validate_spools()
        records: list[dict[str, Any]] = []
        for instance in self.sensors:
            staging = staging_root / "raw" / "sensors" / instance / f"{self._episode_uid}.parquet"
            final = self.root / self.manifest["storage_layout"]["raw_path_template"].format(
                instance=instance, episode_uid=self._episode_uid
            )
            staging.parent.mkdir(parents=True, exist_ok=True)
            self._spools[instance].merge(staging, minimum_timestamp_ns=self._minimum_timestamp_ns)
            records.append(parquet_file_record(self.root, staging, final))

        sync_staging = staging_root / "raw" / "sync" / f"{self._episode_uid}.parquet"
        sync_final = self.root / self.manifest["storage_layout"]["sync_path_template"].format(
            episode_uid=self._episode_uid
        )
        sync_staging.parent.mkdir(parents=True, exist_ok=True)
        self._sync_spool.merge(sync_staging)
        records.append(parquet_file_record(self.root, sync_staging, sync_final))

        metadata = self._episode_metadata()
        metadata_staging = staging_root / "meta" / "sensor_episodes" / f"{self._episode_uid}.json"
        metadata_final = self.root / self.manifest["storage_layout"]["episode_metadata_path_template"].format(
            episode_uid=self._episode_uid
        )
        _atomic_json(metadata_staging, metadata)
        records.append(_generic_file_record(self.root, metadata_staging, metadata_final))

        transaction = self._transaction.prepare_sidecars(
            frame_count=self._sync_count,
            files=records,
            task_info=task_info,
            sequence_boundaries={
                name: subscription.start_sequence for name, subscription in self._subscriptions.items()
            },
            required_streams=[name for name, sensor in self.sensors.items() if sensor.config.required],
        )
        self._transaction = transaction
        try:
            self.check_health()
        except Exception as exc:
            if any(lease.error is not None for lease in self._recorder_leases.values()):
                transaction.quarantine(str(exc))
            else:
                transaction.abort(str(exc))
            self._reset_episode()
            raise
        self._active = False
        return transaction

    def save_episode(self, dataset, *, task_info: Any = None, **save_kwargs) -> None:
        """Commit main episode and sidecars through the durable state machine."""
        self.prepare_episode(task_info=task_info)
        self.commit_prepared(dataset, **save_kwargs)

    def commit_prepared(self, dataset, **save_kwargs) -> None:
        """Save the main episode and finish an already PREPARED transaction."""
        transaction = self._transaction
        if transaction is None or transaction.state != "PREPARED":
            raise SensorRecorderError("No PREPARED sensor transaction to commit.")
        try:
            self.check_health()
        except Exception as exc:
            transaction.quarantine(str(exc))
            self._reset_episode()
            raise
        try:
            self._bind_dataset(dataset)
            dataset.save_episode(**save_kwargs)
            seal_episode_artifacts = getattr(dataset, "seal_episode_artifacts", None)
            if seal_episode_artifacts is not None:
                seal_episode_artifacts()
            self.check_health()
            transaction.mark_main_saved()
            transaction.promote_sidecars()
            self.check_health()
            transaction.commit()
            self.check_health()
        except Exception as exc:
            if any(lease.error is not None for lease in self._recorder_leases.values()):
                transaction.quarantine(str(exc))
            else:
                transaction.replay()
                if transaction.state == "COMMITTED":
                    writer = getattr(dataset, "writer", None)
                    if writer is not None:
                        writer.synchronize_sensor_commit(transaction)
                    return
            raise
        finally:
            if transaction.state == "COMMITTED":
                stats = self.diagnostics
                all_streams = [*stats["streams"].values(), stats["sync"]]
                logger.info(
                    "Sensor COMMITTED %s: Raw=%s rows, Sync=%s frames, %.2f MiB written, %s fragments.",
                    transaction.episode_uid,
                    sum(item["rows"] for item in stats["streams"].values()),
                    stats["sync"]["rows"],
                    sum(item["spool_bytes"] + item["final_bytes"] for item in all_streams) / 1024**2,
                    sum(item["fragments"] for item in all_streams),
                )
            self._reset_episode()

    def abort_prepared(self, reason: str) -> None:
        """Abort an already PREPARED transaction and clear episode state."""
        transaction = self._transaction
        if transaction is None or transaction.state != "PREPARED":
            raise SensorRecorderError("No PREPARED sensor transaction to abort.")
        transaction.abort(reason)
        self._reset_episode()

    def abort_episode(self, reason: str) -> None:
        """Persist an auditable ABORTED transaction for a discarded episode."""
        if not self._active:
            return
        self.prepare_episode(task_info={"aborted_reason": reason})
        self.abort_prepared(reason)

    def close(self) -> None:
        """Unsubscribe, join workers, and release the Dataset writer lock."""
        try:
            if self._active:
                try:
                    self.abort_episode("Recorder closed with an active episode.")
                finally:
                    self._reset_episode()
            elif self._transaction is not None:
                try:
                    self.abort_prepared("Recorder closed with a PREPARED episode.")
                finally:
                    self._reset_episode()
        finally:
            if not any(thread.is_alive() for thread in self._threads.values()):
                self._writer_lock.release()
                self._closed = True

    def _stop_workers(self) -> None:
        with self._rows_lock:
            self._accept_sync = False
        for instance, subscription in tuple(self._subscriptions.items()):
            self.sensors[instance].unsubscribe(subscription)
        self._stop_event.set()
        for thread in self._threads.values():
            if thread.ident is not None:
                thread.join(timeout=5.0)
            if thread.is_alive():
                self._worker_errors[thread.name] = RuntimeError("worker did not stop")
        if any(thread.is_alive() for thread in self._threads.values()):
            raise SensorRecorderError("Sensor recorder worker did not stop; ownership retained.")
        for instance, subscription in self._subscriptions.items():
            if not subscription.closed:
                raise SensorRecorderError(f"Sensor {instance!r} subscription was not removed.")
        # Also close spools whose thread failed to start (or was replaced in failure tests).
        for spool in (*self._spools.values(), self._sync_spool):
            if spool is not None:
                try:
                    spool.close()
                except Exception as exc:
                    self._worker_errors[str(spool.root)] = exc

    def _reset_episode(self) -> None:
        if any(thread.is_alive() for thread in self._threads.values()):
            return
        if self._episode_uid is not None:
            self._last_diagnostics = self.diagnostics
        if self._bound_dataset is not None:
            writer = getattr(self._bound_dataset, "writer", None)
            if writer is not None:
                writer.set_sensor_transaction(None)
            else:
                self._bound_dataset._sensor_transaction = None
            self._bound_dataset = None
        self._active = False
        for instance, lease in self._recorder_leases.items():
            self.sensors[instance].release_recorder(lease)
        self._recorder_leases.clear()
        self._episode_uid = None
        self._episode_index = None
        self._subscriptions.clear()
        self._threads.clear()
        self._stop_event.clear()
        self._transaction = None

    def _episode_metadata(self) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        for instance, sensor in self.sensors.items():
            spool = self._spools[instance]
            with contextlib.closing(sqlite3.connect(spool.index_path)) as db:
                count, valid, first, last = db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(is_valid), 0), MIN(arrival_timestamp_ns), "
                    "MAX(arrival_timestamp_ns) FROM samples WHERE (? IS NULL OR timestamp_ns >= ?)",
                    (self._minimum_timestamp_ns, self._minimum_timestamp_ns),
                ).fetchone()
            duration_ns = last - first if count > 1 else 0
            actual_rate = (count - 1) * 1e9 / duration_ns if duration_ns > 0 else None
            streams[instance] = {
                "provenance": sensor.provenance,
                "resolved_max_age_ms": sensor.config.resolve_max_age_ms(
                    state_features_present=bool(
                        sensor.features
                        if sensor.config.state_features is None
                        else sensor.config.state_features
                    )
                ),
                "resolved_recorder_queue_capacity": sensor.config.resolve_recorder_queue_capacity(),
                "sample_count": count,
                "valid_count": valid,
                "invalid_count": count - valid,
                "sequence_gaps": spool.sequence_gaps,
                "queue_overflow_count": self._subscriptions[instance].overflow_count,
                "actual_sample_rate_hz": actual_rate,
            }
        return {
            "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
            "episode_uid": self._episode_uid,
            "episode_index": self._episode_index,
            "frame_count": self._sync_count,
            "streams": streams,
        }

    def _validate_spools(self) -> None:
        """Validate bounded batches and exact required Sync references before PREPARED."""
        with contextlib.ExitStack() as stack:
            indexes = {
                name: stack.enter_context(contextlib.closing(sqlite3.connect(spool.index_path)))
                for name, spool in self._spools.items()
            }
            for name, spool in self._spools.items():
                subscription = self._subscriptions[name]
                previous = subscription.start_sequence - 1
                indexed = stack.enter_context(
                    contextlib.closing(
                        indexes[name].execute(
                            "SELECT sequence, timestamp_ns, arrival_timestamp_ns, is_valid, hardware_sequence, status "
                            "FROM samples ORDER BY sequence"
                        )
                    )
                )
                for batch in stack.enter_context(contextlib.closing(spool.iter_batches())):
                    for row in batch.to_pylist():
                        if row["episode_uid"] != self._episode_uid or row["sequence"] != previous + 1:
                            raise SensorRecorderError(
                                f"Sensor {name!r} UID or sequence completeness mismatch."
                            )
                        previous = row["sequence"]
                        expected = tuple(
                            row[key]
                            for key in (
                                "sequence",
                                "timestamp_ns",
                                "arrival_timestamp_ns",
                                "is_valid",
                                "hardware_sequence",
                                "status",
                            )
                        )
                        if next(indexed, None) != expected:
                            raise SensorRecorderError(f"Sensor {name!r} Raw disk reference index mismatch.")
                if next(indexed, None) is not None:
                    raise SensorRecorderError(f"Sensor {name!r} Raw disk index contains extra rows.")
                if not subscription.queue.empty() or previous + 1 != subscription.end_sequence:
                    raise SensorRecorderError(f"Sensor {name!r} queue did not drain.")
            frame_count = 0
            for batch in stack.enter_context(contextlib.closing(self._sync_spool.iter_batches())):
                for row in batch.to_pylist():
                    if row["episode_uid"] != self._episode_uid or row["frame_index"] != frame_count:
                        raise SensorRecorderError("Sensor Sync UID or contiguous frame index mismatch.")
                    frame_count += 1
                    anchor = row["frame_anchor_ns"]
                    if not row["observation_start_ns"] <= anchor <= row["observation_complete_ns"]:
                        raise SensorRecorderError("Sensor Sync capture timestamps are not ordered.")
                    for name, sensor in self.sensors.items():
                        ref = (row["sensors"] or {}).get(name)
                        if not sensor.config.required:
                            continue
                        if ref is None or ref["sequence"] is None:
                            raise SensorRecorderError(f"Required Sync reference missing for {name!r}.")
                        raw = (
                            indexes[name]
                            .execute(
                                "SELECT timestamp_ns, arrival_timestamp_ns, is_valid, hardware_sequence, status "
                                "FROM samples WHERE sequence=?",
                                (ref["sequence"],),
                            )
                            .fetchone()
                        )
                        if raw is None or tuple(raw[:2]) != (
                            ref["timestamp_ns"],
                            ref["arrival_timestamp_ns"],
                        ):
                            raise SensorRecorderError(
                                f"Required Sync reference does not match episode Raw: {name!r}."
                            )
                        timestamp, arrival, valid, hardware_sequence, status = raw
                        selected = (
                            sensor.features
                            if sensor.config.state_features is None
                            else sensor.config.state_features
                        )
                        max_age = sensor.config.resolve_max_age_ms(state_features_present=bool(selected))
                        if (
                            not valid
                            or ref["hardware_sequence"] != hardware_sequence
                            or ref["status"] != status
                            or max(timestamp, arrival) > anchor
                            or (max_age is not None and anchor - timestamp > int(max_age * 1_000_000))
                            or ref["age_ns"] != anchor - timestamp
                            or (
                                self._minimum_timestamp_ns is not None
                                and timestamp < self._minimum_timestamp_ns
                            )
                        ):
                            raise SensorRecorderError(
                                f"Required Sync reference is invalid, stale, trimmed or noncausal: {name!r}."
                            )
            if frame_count != self._sync_count or not self._sync_queue.empty():
                raise SensorRecorderError("Sensor Sync queue did not drain completely.")


def _arrow_field(name: str, feature: SensorFeature) -> pa.Field:
    return pa.field(name, pa.from_numpy_dtype(np.dtype(feature.dtype)), nullable=True)


def _raw_arrow_schema(sensor: Sensor) -> pa.Schema:
    values = pa.struct([_arrow_field(name, feature) for name, feature in sensor.features.items()])
    native_values = (
        pa.struct([_arrow_field(name, feature) for name, feature in sensor.native_features.items()])
        if sensor.native_features
        else pa.null()
    )
    return pa.schema(
        [
            pa.field("episode_uid", pa.string(), nullable=False),
            pa.field("episode_time_ns", pa.int64(), nullable=False),
            pa.field("timestamp_ns", pa.int64(), nullable=False),
            pa.field("arrival_timestamp_ns", pa.int64(), nullable=False),
            pa.field("sequence", pa.int64(), nullable=False),
            pa.field("hardware_timestamp_ns", pa.int64()),
            pa.field("hardware_sequence", pa.int64()),
            pa.field("is_valid", pa.bool_(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("error", pa.string()),
            pa.field("values", values),
            pa.field("native_values", native_values),
            pa.field("native_payload", pa.binary()),
        ]
    )


def _sync_arrow_schema(sensors: dict[str, Sensor]) -> pa.Schema:
    selection = pa.struct(
        [
            pa.field("timestamp_ns", pa.int64()),
            pa.field("arrival_timestamp_ns", pa.int64()),
            pa.field("sequence", pa.int64()),
            pa.field("hardware_sequence", pa.int64()),
            pa.field("age_ns", pa.int64()),
            pa.field("status", pa.string()),
        ]
    )
    return pa.schema(
        [
            pa.field("episode_uid", pa.string(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("observation_start_ns", pa.int64(), nullable=False),
            pa.field("frame_anchor_ns", pa.int64(), nullable=False),
            pa.field("observation_complete_ns", pa.int64(), nullable=False),
            pa.field("hardware_observation_timestamps", pa.map_(pa.string(), pa.int64())),
            pa.field("sensors", pa.struct([pa.field(instance, selection) for instance in sensors])),
        ]
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _generic_file_record(root: Path, staging: Path, final: Path) -> dict[str, Any]:
    return {
        "kind": "json",
        "staging_path": staging.relative_to(root).as_posix(),
        "final_path": final.relative_to(root).as_posix(),
        "size": staging.stat().st_size,
        "row_count": None,
        "schema": None,
        "sha256": sha256_file(staging),
    }


def _process_is_alive(pid: int) -> bool:
    """Return conservatively whether a lock owner's process still exists."""
    if pid <= 0:
        return False
    if os.name == "nt":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied still proves the process exists
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
