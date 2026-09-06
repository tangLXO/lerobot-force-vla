#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Directly discoverable transactions with episode-local logical evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .sensor_transaction import (
    SensorTransaction,
    SensorTransactionError,
    TransactionState,
    _atomic_write_json,
    validate_episode_uid,
)


def light_main_precondition(root: Path) -> dict[str, Any]:
    """Read only the small main info document, never historical artifacts."""
    path = root / "meta/info.json"
    content = path.read_bytes() if path.exists() else b""
    info = json.loads(content) if content else {}
    return {
        "info_sha256": hashlib.sha256(content).hexdigest(),
        "total_frames": info.get("total_frames", 0),
        "total_episodes": info.get("total_episodes", 0),
    }


def _file_stamp(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise SensorTransactionError(f"Unsupported logical digest value: {type(value).__name__}.")


def _row_bytes(row) -> bytes:
    return (
        json.dumps(_canonical(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def _episode_rows(path: Path, episode_index: int):
    """Bound decoding and prune only when statistics prove the episode absent."""
    with path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
        schema = parquet.schema_arrow
        if "episode_index" not in schema.names:
            raise SensorTransactionError(f"Missing episode_index in {path}.")
        column = parquet.schema.names.index("episode_index")
        for group in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(group).column(column).statistics
            if stats is not None and stats.has_min_max and not stats.min <= episode_index <= stats.max:
                continue
            for batch in parquet.iter_batches(batch_size=4096, row_groups=[group]):
                for row in batch.to_pylist():
                    if row["episode_index"] == episode_index:
                        yield row


def capture_logical_evidence(transaction: SensorTransaction, *, path_resolver=None) -> dict[str, Any]:
    """Seal precisely one episode's metadata row, data range, schema and video ranges."""
    root, journal = transaction.root, transaction.journal
    resolve_path = path_resolver or transaction._artifact_path
    expected = journal["expected_main"]
    index = expected["episode_index"]
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("total_frames", 0) < expected["dataset_to_index"] or info.get("total_episodes", 0) <= index:
        raise SensorTransactionError("Main episode totals are not sealed.")
    metadata_paths = [
        item["path"] for item in journal["main_artifacts"] if item["role"] == "episode_metadata"
    ]
    if not metadata_paths:
        raise SensorTransactionError("No registered episode metadata artifact.")
    metadata_row = None
    metadata_path = None
    for relative in metadata_paths:
        path = resolve_path(relative)
        with closing(_episode_rows(path, index)) as rows:
            for row in rows:
                if metadata_row is not None:
                    raise SensorTransactionError("Duplicate main episode identity.")
                metadata_row, metadata_path = row, relative
    if metadata_row is None:
        raise SensorTransactionError("Registered metadata has no matching episode row.")
    for key, value in (
        ("length", expected["frame_count"]),
        ("dataset_from_index", expected["dataset_from_index"]),
        ("dataset_to_index", expected["dataset_to_index"]),
    ):
        if metadata_row.get(key) != value:
            raise SensorTransactionError(f"Main episode {key} mismatch.")
    if expected.get("task_info") is not None and metadata_row.get("tasks") != expected["task_info"]:
        raise SensorTransactionError("Main episode task mismatch.")
    data_relative = info["data_path"].format(
        chunk_index=metadata_row["data/chunk_index"], file_index=metadata_row["data/file_index"]
    )
    if not any(
        item["path"] == data_relative and item["role"] == "data" for item in journal["main_artifacts"]
    ):
        raise SensorTransactionError("Main data artifact was not registered before writing.")
    data_path = resolve_path(data_relative)
    digest = hashlib.sha256(b"lerobot-logical-rows-v1\n")
    with data_path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
        schema = parquet.schema_arrow.remove_metadata()
    digest.update(schema.serialize().to_pybytes())
    count = 0
    with closing(_episode_rows(data_path, index)) as rows:
        for row in rows:
            if row.get("index") != expected["dataset_from_index"] + count:
                raise SensorTransactionError("Main episode logical range is not continuous.")
            digest.update(_row_bytes(row))
            count += 1
    if count != expected["frame_count"]:
        raise SensorTransactionError("Main episode logical row count mismatch.")
    videos = []
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        from .video_utils import get_video_duration_in_s

        prefix = f"videos/{key}"
        relative = info["video_path"].format(
            video_key=key,
            chunk_index=metadata_row[f"{prefix}/chunk_index"],
            file_index=metadata_row[f"{prefix}/file_index"],
        )
        start, end = metadata_row[f"{prefix}/from_timestamp"], metadata_row[f"{prefix}/to_timestamp"]
        if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
            raise SensorTransactionError("Invalid main video time range.")
        path = resolve_path(relative)
        if not any(
            item["path"] == relative and item["role"] == "video" for item in journal["main_artifacts"]
        ):
            raise SensorTransactionError("Main video artifact was not registered before writing.")
        if get_video_duration_in_s(path) + 1 / info.get("fps", 30) < end:
            raise SensorTransactionError("Main video is shorter than its episode range.")
        videos.append({"path": relative, "from_timestamp": start, "to_timestamp": end})
    return {
        "version": 1,
        "metadata_path": metadata_path,
        "metadata_row": _canonical(metadata_row),
        "data_path": data_relative,
        "data_schema": str(schema),
        "logical_rows_sha256": digest.hexdigest(),
        "frame_count": count,
        "videos": videos,
    }


def _require_writer(root: Path) -> None:
    try:
        owner = json.loads((root / ".sensor-writer.lock").read_text(encoding="utf-8"))
        if owner["pid"] == os.getpid():
            return
    except (OSError, ValueError, KeyError):
        pass
    raise SensorTransactionError("Transaction recovery/mutation requires the Dataset writer lock.")


class SensorTransactionV2(SensorTransaction):
    """Process-crash recoverable v2 state; v1 journals are never upgraded."""

    @classmethod
    def begin(cls, root: Path, episode_uid: str, episode_index: int):
        _require_writer(root)
        validate_episode_uid(episode_uid)
        precondition = light_main_precondition(root)
        if episode_index != precondition["total_episodes"]:
            raise SensorTransactionError("Recording episode index differs from main precondition.")
        staging = root / ".sensor-staging" / episode_uid
        staging.mkdir(parents=True, exist_ok=False)
        intent = {
            "episode_uid": episode_uid,
            "intent_id": str(uuid.uuid4()),
            "episode_index": episode_index,
            "precondition": precondition,
        }
        _atomic_write_json(staging / "transaction_intent.json", intent)
        transaction = cls(
            root,
            {
                "journal_version": 2,
                "episode_uid": episode_uid,
                "state": "RECORDING",
                "intent_id": intent["intent_id"],
                "main_precondition": precondition,
                "expected_main": {"episode_index": episode_index},
                "main_artifacts": [],
                "files": [],
                "main_evidence": None,
                "episode_start_sequence": {},
                "reason": None,
            },
        )
        transaction._persist()
        _atomic_write_json(
            root / ".sensor-staging/active_transaction.json",
            {"episode_uid": episode_uid, "intent_id": intent["intent_id"]},
        )
        return transaction

    def _persist(self) -> None:
        _require_writer(self.root)
        super()._persist()

    def prepare_sidecars(self, *, frame_count, files, task_info, sequence_boundaries, required_streams):
        if self.state != TransactionState.RECORDING:
            raise SensorTransactionError("Only a RECORDING transaction can be prepared.")
        intent = json.loads(
            (self.root / ".sensor-staging" / self.episode_uid / "transaction_intent.json").read_text()
        )
        if intent["intent_id"] != self.journal["intent_id"] or intent["episode_uid"] != self.episode_uid:
            raise SensorTransactionError("Staging identity mismatch.")
        start = self.journal["main_precondition"]["total_frames"]
        self.journal["expected_main"].update(
            frame_count=frame_count,
            dataset_from_index=start,
            dataset_to_index=start + frame_count,
            task_info=task_info,
        )
        self.journal.update(
            files=files, episode_start_sequence=sequence_boundaries, required_streams=required_streams
        )
        self._verify_staging_files()
        self._advance(TransactionState.PREPARED)
        return self

    def register_artifact(self, path: Path, role: str) -> None:
        """Persist the actual locator and prewrite stamp before its first modification."""
        _require_writer(self.root)
        if self.state not in (TransactionState.RECORDING, TransactionState.PREPARED):
            raise SensorTransactionError("Cannot register an artifact after main sealing.")
        relative = path.resolve().relative_to(self.root.resolve()).as_posix()
        self._artifact_path(relative)
        if not any(
            item["path"] == relative and item["role"] == role for item in self.journal["main_artifacts"]
        ):
            self.journal["main_artifacts"].append(
                {"path": relative, "role": role, "before": _file_stamp(path)}
            )
            self._persist()

    def mark_main_saved(self, main_postcondition=None) -> None:
        if self.state != TransactionState.PREPARED:
            raise SensorTransactionError(f"Cannot mark main saved from {self.state}.")
        self.journal["main_evidence"] = capture_logical_evidence(self)
        self._advance(TransactionState.MAIN_SAVED)

    def _verify_main_postcondition(self) -> None:
        if self.journal.get("main_evidence") != capture_logical_evidence(self):
            raise SensorTransactionError("Main episode logical evidence mismatch.")

    def _verify_committed(self) -> None:
        self._verify_main_postcondition()
        self._verify_final_files()
        self._verify_sidecar_identity()

    def _clear_active_pointer(self) -> None:
        path = self.root / ".sensor-staging/active_transaction.json"
        if path.exists():
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if pointer.get("episode_uid") != self.episode_uid:
                raise SensorTransactionError("Conflicting active transaction pointer.")
            path.unlink()

    def _cleanup_staging(self) -> None:
        _require_writer(self.root)
        super()._cleanup_staging()
        self._clear_active_pointer()

    def abort(self, reason: str) -> None:
        self.journal["reason"] = reason
        self._advance(TransactionState.ABORTED)
        self._quarantine_staging()
        self._clear_active_pointer()

    def quarantine(self, reason: str) -> None:
        self.journal["reason"] = reason
        self._advance(TransactionState.QUARANTINED)
        self._quarantine_staging()
        self._clear_active_pointer()

    def _definitely_unsaved(self) -> bool:
        if light_main_precondition(self.root) != self.journal["main_precondition"]:
            return False
        for item in self.journal["main_artifacts"]:
            if item["role"] == "temporary" or (item["role"] == "info" and item["path"] == "meta/info.json"):
                continue
            # Size/mtime are diagnostic stamps, not proof against partial same-size overwrite.
            if item["before"] is not None or self._artifact_path(item["path"]).exists():
                return False
        return True

    def replay(self) -> TransactionState:
        _require_writer(self.root)
        try:
            staging = self.root / ".sensor-staging" / self.episode_uid
            if staging.exists() and self.state not in (
                TransactionState.ABORTED,
                TransactionState.QUARANTINED,
            ):
                intent = json.loads((staging / "transaction_intent.json").read_text(encoding="utf-8"))
                if (
                    intent.get("intent_id") != self.journal["intent_id"]
                    or intent.get("episode_uid") != self.episode_uid
                ):
                    raise SensorTransactionError("Staging identity differs from journal.")
            if self.state in (TransactionState.RECORDING, TransactionState.PREPARED):
                if self._definitely_unsaved():
                    self.abort("No registered main artifact changed.")
                    return self.state
                if self.state == TransactionState.RECORDING:
                    raise SensorTransactionError("Main artifacts changed before PREPARED.")
                self.mark_main_saved()
            if self.state == TransactionState.MAIN_SAVED:
                self.promote_sidecars()
            if self.state == TransactionState.SIDECAR_PROMOTED:
                self.commit()
            if self.state == TransactionState.COMMITTED:
                self._verify_committed()
                self._cleanup_staging()
            if self.state in (TransactionState.ABORTED, TransactionState.QUARANTINED):
                self._quarantine_staging()
                self._clear_active_pointer()
            return self.state
        except (OSError, ValueError, KeyError, SensorTransactionError) as exc:
            self.quarantine(str(exc))
            return self.state


class TransactionRecoveryManager:
    """Recover one discoverable active transaction under an explicit writer lock."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def recover_legacy(self):
        """Explicit opt-in to v1's historical snapshot scan, without journal upgrades."""
        from .sensor_stream import SensorDatasetWriterLock
        from .sensor_transaction import replay_sensor_transactions

        with SensorDatasetWriterLock(self.root):
            return replay_sensor_transactions(self.root)

    def recover(self, *, writer_lock=None):
        from .sensor_stream import SensorDatasetWriterLock

        if writer_lock is None:
            with SensorDatasetWriterLock(self.root) as lock:
                return self.recover(writer_lock=lock)
        _require_writer(self.root)
        staging = self.root / ".sensor-staging"
        pointer_path = staging / "active_transaction.json"
        pointer = None
        if pointer_path.exists():
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                validate_episode_uid(pointer["episode_uid"])
            except (ValueError, KeyError, TypeError):
                pointer = None
        candidates = []
        if pointer is not None:
            journal = self.root / "meta/sensor_transactions" / f"{pointer['episode_uid']}.json"
            if journal.exists():
                transaction = SensorTransaction.load(journal)
                if transaction.journal.get("intent_id") == pointer.get("intent_id"):
                    candidates.append(transaction)
        # Only uncleaned staging is inspected. Historical COMMITTED journals are never scanned.
        if staging.exists():
            for directory in staging.iterdir():
                if not directory.is_dir():
                    continue
                uid = validate_episode_uid(directory.name)
                if candidates and candidates[0].episode_uid == uid:
                    continue
                intent_path = directory / "transaction_intent.json"
                if not intent_path.exists():
                    if not any(directory.iterdir()):
                        # mkdir completed but intent creation did not start: no subscription or main write was legal.
                        directory.rmdir()
                        continue
                    raise SensorTransactionError("Unidentified staging requires explicit legacy recovery.")
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
                if intent.get("episode_uid") != uid:
                    raise SensorTransactionError("Staging intent UID mismatch.")
                journal = self.root / "meta/sensor_transactions" / f"{uid}.json"
                if journal.exists():
                    candidates.append(SensorTransaction.load(journal))
                else:
                    candidates.append(
                        SensorTransactionV2(
                            self.root,
                            {
                                "journal_version": 2,
                                "episode_uid": uid,
                                "state": "RECORDING",
                                "intent_id": intent["intent_id"],
                                "main_precondition": intent["precondition"],
                                "expected_main": {"episode_index": intent["episode_index"]},
                                "main_artifacts": [],
                                "files": [],
                                "main_evidence": None,
                            },
                        )
                    )
        if (
            len(candidates) > 1
            and candidates[0].state
            in (TransactionState.COMMITTED, TransactionState.ABORTED, TransactionState.QUARANTINED)
            and not (staging / candidates[0].episode_uid).exists()
        ):
            candidates.pop(0)  # A stale terminal pointer cannot supersede an uncleaned active intent.
        if len(candidates) > 1:
            raise SensorTransactionError(
                "Multiple conflicting active transaction candidates; refusing to guess."
            )
        if not candidates:
            if pointer_path.exists():
                pointer_path.unlink()
            return None
        transaction = candidates[0]
        _atomic_write_json(
            pointer_path,
            {"episode_uid": transaction.episode_uid, "intent_id": transaction.journal["intent_id"]},
        )
        transaction.replay()
        return transaction
