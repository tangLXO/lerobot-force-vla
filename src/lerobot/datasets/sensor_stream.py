#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Native-rate Raw/Sync sensor sidecar recording."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.sensors import Sensor, SensorFeature, SensorSubscription

from .sensor_transaction import (
    SensorTransaction,
    capture_main_dataset_state,
    parquet_file_record,
    replay_sensor_transactions,
    sha256_file,
    validate_episode_uid,
)

SIDECAR_SCHEMA_VERSION = 1


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
        for sensor in sensors.values():
            sensor.config.resolve_recorder_queue_capacity()
        self._writer_lock = SensorDatasetWriterLock(self.root)
        try:
            replay_sensor_transactions(self.root)
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
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._raw_rows: dict[str, list[dict[str, Any]]] = {}
        self._rows_lock = threading.RLock()
        self._sync_rows: list[dict[str, Any]] = []
        self._worker_errors: dict[str, BaseException] = {}
        self._transaction: SensorTransaction | None = None
        self._minimum_timestamp_ns: int | None = None
        self._closed = False

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
        return self._transaction is not None

    @property
    def worker_threads(self) -> tuple[threading.Thread, ...]:
        """Expose worker identities for deterministic cleanup tests."""
        return tuple(self._threads.values())

    def start_episode(self, episode_index: int, *, episode_uid: str | None = None) -> str:
        """Subscribe and start one native-rate worker per sensor."""
        if self._closed:
            raise SensorRecorderError("Sensor recorder is closed.")
        if self._active:
            raise SensorRecorderError("A sensor episode is already active.")
        if self._transaction is not None:
            raise SensorRecorderError("A PREPARED sensor episode must be committed or aborted first.")
        replay_sensor_transactions(self.root)
        resolved_uid = validate_episode_uid(episode_uid or str(uuid.uuid4()))
        self._active = True
        self._episode_uid = resolved_uid
        self._episode_index = episode_index
        self._episode_start_ns = time.perf_counter_ns()
        self._main_precondition = capture_main_dataset_state(self.root)
        self._stop_event.clear()
        self._raw_rows = {instance: [] for instance in self.sensors}
        self._sync_rows = []
        self._worker_errors = {}
        self._transaction = None
        self._minimum_timestamp_ns = None
        try:
            for instance, sensor in self.sensors.items():
                subscription = sensor.subscribe(sensor.config.resolve_recorder_queue_capacity())
                self._subscriptions[instance] = subscription
                thread = threading.Thread(
                    target=self._drain,
                    args=(instance, subscription),
                    name=f"SensorStreamRecorder-{instance}",
                    daemon=True,
                )
                self._threads[instance] = thread
                thread.start()
        except Exception:
            self._stop_workers()
            self._reset_episode()
            raise
        return self._episode_uid

    def _drain(self, instance: str, subscription: SensorSubscription) -> None:
        try:
            while not self._stop_event.is_set() or not subscription.queue.empty():
                try:
                    sample = subscription.get(timeout=0.05)
                except queue.Empty:
                    continue
                with self._rows_lock:
                    if (
                        self._minimum_timestamp_ns is None
                        or sample.timestamp_ns >= self._minimum_timestamp_ns
                    ):
                        self._raw_rows[instance].append(self._raw_row(instance, sample))
        except BaseException as exc:  # worker failures must reach the control loop
            self._worker_errors[instance] = exc

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
        if not self._active:
            raise SensorRecorderError("No active sensor episode.")
        expected_frame_index = len(self._sync_rows)
        resolved_frame_index = expected_frame_index if frame_index is None else frame_index
        if resolved_frame_index != expected_frame_index:
            raise SensorRecorderError(
                f"Sensor Sync frame_index must be contiguous: expected {expected_frame_index}, "
                f"got {resolved_frame_index}."
            )
        self._sync_rows.append(
            {
                "episode_uid": self._episode_uid,
                "frame_index": resolved_frame_index,
                "observation_start_ns": capture_metadata["observation_start_ns"],
                "frame_anchor_ns": capture_metadata["frame_anchor_ns"],
                "observation_complete_ns": capture_metadata["observation_complete_ns"],
                "hardware_observation_timestamps": capture_metadata.get("hardware_observation_timestamps"),
                "sensors": capture_metadata.get("sensors", {}),
            }
        )

    def check_health(self) -> None:
        """Raise immediately on worker failure or subscriber overflow."""
        if self._worker_errors:
            instance, error = next(iter(self._worker_errors.items()))
            raise SensorRecorderError(f"Sensor recorder worker {instance!r} failed: {error}") from error
        overflowed = [name for name, item in self._subscriptions.items() if item.overflowed]
        if overflowed:
            raise SensorQueueOverflowError(
                f"Sensor recorder queue overflowed for streams {overflowed}; episode is invalid."
            )

    def trim_before(self, timestamp_ns: int) -> None:
        """Discard pre-roll raw rows older than a retained Highlight window."""
        if not self._active:
            raise SensorRecorderError("Cannot trim a sensor recorder without an active episode.")
        with self._rows_lock:
            self._minimum_timestamp_ns = timestamp_ns
            for instance, rows in self._raw_rows.items():
                self._raw_rows[instance] = [row for row in rows if row["timestamp_ns"] >= timestamp_ns]

    def prepare_episode(self, *, task_info: Any = None) -> SensorTransaction:
        """Stop workers, write staging artifacts, and persist PREPARED."""
        if not self._active or self._episode_uid is None or self._episode_index is None:
            raise SensorRecorderError("No active sensor episode.")
        self._stop_workers()
        staging_root = self.root / ".sensor-staging" / self._episode_uid
        if staging_root.exists():
            raise SensorRecorderError(f"Sensor staging directory already exists: {staging_root}.")
        records: list[dict[str, Any]] = []
        for instance, sensor in self.sensors.items():
            staging = staging_root / "raw" / "sensors" / instance / f"{self._episode_uid}.parquet"
            final = self.root / self.manifest["storage_layout"]["raw_path_template"].format(
                instance=instance, episode_uid=self._episode_uid
            )
            staging.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(self._raw_rows[instance], schema=_raw_arrow_schema(sensor)),
                staging,
            )
            records.append(parquet_file_record(self.root, staging, final))

        sync_staging = staging_root / "raw" / "sync" / f"{self._episode_uid}.parquet"
        sync_final = self.root / self.manifest["storage_layout"]["sync_path_template"].format(
            episode_uid=self._episode_uid
        )
        sync_staging.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(self._sync_rows, schema=_sync_arrow_schema(self.sensors)), sync_staging
        )
        records.append(parquet_file_record(self.root, sync_staging, sync_final))

        metadata = self._episode_metadata()
        metadata_staging = staging_root / "meta" / "sensor_episodes" / f"{self._episode_uid}.json"
        metadata_final = self.root / self.manifest["storage_layout"]["episode_metadata_path_template"].format(
            episode_uid=self._episode_uid
        )
        _atomic_json(metadata_staging, metadata)
        records.append(_generic_file_record(self.root, metadata_staging, metadata_final))

        dataset_from_index = int((self._main_precondition or {}).get("total_frames") or 0)
        transaction = SensorTransaction.prepare(
            self.root,
            episode_uid=self._episode_uid,
            episode_index=self._episode_index,
            frame_count=len(self._sync_rows),
            dataset_from_index=dataset_from_index,
            files=records,
            task_info=task_info,
            main_precondition=self._main_precondition,
        )
        self._transaction = transaction
        try:
            self.check_health()
        except Exception as exc:
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
            dataset.save_episode(**save_kwargs)
            seal_episode_artifacts = getattr(dataset, "seal_episode_artifacts", None)
            if seal_episode_artifacts is not None:
                seal_episode_artifacts()
            transaction.mark_main_saved()
            transaction.promote_sidecars()
            transaction.commit()
        except Exception:
            transaction.replay()
            raise
        finally:
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
            self._writer_lock.release()
            self._closed = True

    def _stop_workers(self) -> None:
        self._stop_event.set()
        for instance, subscription in tuple(self._subscriptions.items()):
            self.sensors[instance].unsubscribe(subscription)
        for thread in self._threads.values():
            thread.join(timeout=5.0)
            if thread.is_alive():
                self._worker_errors[thread.name] = RuntimeError("worker did not stop")

    def _reset_episode(self) -> None:
        self._active = False
        self._episode_uid = None
        self._episode_index = None
        self._subscriptions.clear()
        self._threads.clear()
        self._stop_event.clear()
        self._transaction = None

    def _episode_metadata(self) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        for instance, sensor in self.sensors.items():
            rows = self._raw_rows[instance]
            sequences = [int(row["sequence"]) for row in rows]
            sequence_gaps = sum(
                max(0, right - left - 1) for left, right in zip(sequences, sequences[1:], strict=False)
            )
            duration_ns = rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"] if len(rows) > 1 else 0
            actual_rate = (len(rows) - 1) * 1e9 / duration_ns if duration_ns > 0 else None
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
                "sample_count": len(rows),
                "valid_count": sum(bool(row["is_valid"]) for row in rows),
                "invalid_count": sum(not bool(row["is_valid"]) for row in rows),
                "sequence_gaps": sequence_gaps,
                "queue_overflow_count": self._subscriptions[instance].overflow_count,
                "actual_sample_rate_hz": actual_rate,
            }
        return {
            "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
            "episode_uid": self._episode_uid,
            "episode_index": self._episode_index,
            "frame_count": len(self._sync_rows),
            "streams": streams,
        }


def _arrow_field(name: str, feature: SensorFeature) -> pa.Field:
    return pa.field(name, pa.from_numpy_dtype(np.dtype(feature.dtype)), nullable=True)


def _raw_arrow_schema(sensor: Sensor) -> pa.Schema:
    values = pa.struct([_arrow_field(name, feature) for name, feature in sensor.features.items()])
    native_values = pa.struct(
        [_arrow_field(name, feature) for name, feature in sensor.native_features.items()]
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
