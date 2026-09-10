#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Crash-replayable transaction journal for sensor sidecars."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import uuid
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

JOURNAL_VERSION = 1


class SensorTransactionError(RuntimeError):
    """Raised when a sidecar transaction cannot be proven safe."""


class TransactionState(StrEnum):
    """Persistent sidecar transaction states."""

    RECORDING = "RECORDING"
    PREPARED = "PREPARED"
    MAIN_SAVED = "MAIN_SAVED"
    SIDECAR_PROMOTED = "SIDECAR_PROMOTED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    QUARANTINED = "QUARANTINED"


def validate_episode_uid(episode_uid: str) -> str:
    """Return a canonical UUID4 or reject unsafe/non-contract identifiers."""
    try:
        parsed = uuid.UUID(episode_uid)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Sensor episode_uid must be a canonical UUID4, got {episode_uid!r}.") from exc
    if parsed.version != 4 or str(parsed) != episode_uid:
        raise ValueError(f"Sensor episode_uid must be a canonical UUID4, got {episode_uid!r}.")
    return episode_uid


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_file_record(root: Path, staging_path: Path, final_path: Path) -> dict[str, Any]:
    """Describe a closed staging Parquet file for journal verification."""
    metadata = pq.read_metadata(staging_path)
    return {
        "kind": "parquet",
        "staging_path": staging_path.relative_to(root).as_posix(),
        "final_path": final_path.relative_to(root).as_posix(),
        "size": staging_path.stat().st_size,
        "row_count": metadata.num_rows,
        "schema": str(metadata.schema.to_arrow_schema()),
        "sha256": sha256_file(staging_path),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


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


@dataclass
class SensorTransaction:
    """One crash-replayable main-Dataset plus sidecar commit transaction."""

    root: Path
    journal: dict[str, Any]

    @property
    def episode_uid(self) -> str:
        """Return the transaction's stable UUID string."""
        return str(self.journal["episode_uid"])

    @property
    def state(self) -> TransactionState:
        """Return the transaction state recorded in the journal."""
        return TransactionState(self.journal["state"])

    @property
    def journal_path(self) -> Path:
        """Return the persistent journal location."""
        return self.root / "meta" / "sensor_transactions" / f"{self.episode_uid}.json"

    @classmethod
    def load(cls, path: Path) -> SensorTransaction:
        """Load one existing journal without changing it."""
        path = Path(path)
        journal = json.loads(path.read_text(encoding="utf-8"))
        version = journal.get("journal_version")
        if isinstance(version, bool) or not isinstance(version, int) or version != JOURNAL_VERSION:
            raise SensorTransactionError("Unsupported sensor journal version.")
        episode_uid = validate_episode_uid(journal.get("episode_uid"))
        if path.stem != episode_uid:
            raise ValueError(f"Sensor transaction filename and episode_uid differ: {path}.")
        return cls(path.parents[2], journal)

    @classmethod
    def begin(cls, root: Path, episode_uid: str, episode_index: int) -> SensorTransaction:
        """Persist a discoverable RECORDING transaction before subscriptions begin."""
        root = Path(root)
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
                "journal_version": JOURNAL_VERSION,
                "episode_uid": episode_uid,
                "state": TransactionState.RECORDING,
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

    def prepare_sidecars(
        self, *, frame_count, files, task_info, sequence_boundaries, required_streams
    ) -> SensorTransaction:
        """Close and verify staged sidecars before the main Dataset save."""
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
            files=files,
            episode_start_sequence=sequence_boundaries,
            required_streams=required_streams,
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

    def mark_main_saved(self) -> None:
        """Seal the main episode's logical evidence and advance to MAIN_SAVED."""
        if self.state != TransactionState.PREPARED:
            raise SensorTransactionError(f"Cannot mark main saved from {self.state}.")
        self.journal["main_evidence"] = capture_logical_evidence(self)
        self._advance(TransactionState.MAIN_SAVED)

    def promote_sidecars(self) -> None:
        """Idempotently move verified staging files to their manifest paths."""
        if self.state == TransactionState.PREPARED:
            raise SensorTransactionError("Main Dataset must be saved before sidecar promotion.")
        if self.state in {TransactionState.SIDECAR_PROMOTED, TransactionState.COMMITTED}:
            self._verify_final_files()
            return
        if self.state != TransactionState.MAIN_SAVED:
            raise SensorTransactionError(f"Cannot promote sidecars from {self.state}.")
        self._verify_main_postcondition()
        for record in self.journal["files"]:
            staging = self._artifact_path(record["staging_path"])
            final = self._artifact_path(record["final_path"])
            if final.exists():
                self._verify_file(final, record)
                continue
            if not staging.exists():
                raise SensorTransactionError(f"Missing both staging and final sidecar file: {final}.")
            self._verify_file(staging, record)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            self._verify_file(final, record)
        self._advance(TransactionState.SIDECAR_PROMOTED)

    def commit(self) -> None:
        """Verify both halves and atomically advance to COMMITTED."""
        if self.state == TransactionState.COMMITTED:
            self._verify_committed()
            self._cleanup_staging()
            return
        if self.state != TransactionState.SIDECAR_PROMOTED:
            raise SensorTransactionError(f"Cannot commit from {self.state}.")
        self._verify_committed()
        self._advance(TransactionState.COMMITTED)
        self._cleanup_staging()

    def abort(self, reason: str) -> None:
        """Abort a definitely-unsaved main episode and quarantine its staging data."""
        self.journal["reason"] = reason
        self._advance(TransactionState.ABORTED)
        self._quarantine_staging()
        self._clear_active_pointer()

    def quarantine(self, reason: str) -> None:
        """Quarantine an ambiguous/conflicting transaction and refuse repair."""
        self.journal["reason"] = reason
        self._advance(TransactionState.QUARANTINED)
        self._quarantine_staging()
        self._clear_active_pointer()

    def replay(self) -> TransactionState:
        """Deterministically replay this transaction as far as evidence permits."""
        _require_writer(self.root)
        try:
            staging = self.root / ".sensor-staging" / self.episode_uid
            if staging.exists() and self.state not in (
                TransactionState.ABORTED,
                TransactionState.QUARANTINED,
                TransactionState.COMMITTED,
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

    def _advance(self, state: TransactionState) -> None:
        self.journal["state"] = state
        self._persist()

    def _persist(self) -> None:
        _require_writer(self.root)
        _atomic_write_json(self.journal_path, self.journal)

    def _verify_staging_files(self) -> None:
        for record in self.journal["files"]:
            self._verify_file(self._artifact_path(record["staging_path"]), record)

    def _verify_final_files(self) -> None:
        for record in self.journal["files"]:
            self._verify_file(self._artifact_path(record["final_path"]), record)

    def _artifact_path(self, relative_path: str) -> Path:
        """Resolve a journal path while proving that it stays inside the Dataset root."""
        root = self.root.resolve()
        path = (root / relative_path).resolve()
        if path == root or root not in path.parents:
            raise SensorTransactionError(
                f"Transaction artifact path escapes Dataset root: {relative_path!r}."
            )
        return path

    @staticmethod
    def _verify_file(path: Path, record: dict[str, Any]) -> None:
        if not path.exists():
            raise SensorTransactionError(f"Missing transaction artifact {path}.")
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise SensorTransactionError(f"Transaction artifact digest mismatch: {path}.")
        if record.get("kind", "parquet") == "parquet":
            metadata = pq.read_metadata(path)
            if metadata.num_rows != record["row_count"]:
                raise SensorTransactionError(f"Transaction artifact row-count mismatch: {path}.")
            if str(metadata.schema.to_arrow_schema()) != record["schema"]:
                raise SensorTransactionError(f"Transaction artifact schema mismatch: {path}.")

    def _verify_main_postcondition(self) -> None:
        if self.journal.get("main_evidence") != capture_logical_evidence(self):
            raise SensorTransactionError("Main episode logical evidence mismatch.")

    def _verify_committed(self) -> None:
        self._verify_main_postcondition()
        self._verify_final_files()
        self._verify_sidecar_identity()

    def _verify_sidecar_identity(self) -> None:
        expected_main = self.journal["expected_main"]
        for record in self.journal["files"]:
            path = self._artifact_path(record["final_path"])
            if record.get("kind") == "json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("episode_uid") != self.episode_uid:
                    raise SensorTransactionError(f"Episode metadata UID mismatch: {path}.")
                if int(payload.get("episode_index", -1)) != int(expected_main["episode_index"]):
                    raise SensorTransactionError(f"Episode metadata index mismatch: {path}.")
                if int(payload.get("frame_count", -1)) != int(expected_main["frame_count"]):
                    raise SensorTransactionError(f"Episode metadata frame count mismatch: {path}.")
                continue
            schema_names = set(pq.read_schema(path).names)
            if "episode_uid" not in schema_names:
                continue
            columns = ["episode_uid"]
            if "frame_index" in schema_names:
                columns.append("frame_index")
            count = 0
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                    if any(uid != self.episode_uid for uid in batch.column("episode_uid").to_pylist()):
                        raise SensorTransactionError(f"Sidecar episode UID mismatch: {path}.")
                    if "frame_index" in schema_names:
                        indices = batch.column("frame_index").to_pylist()
                        if indices != list(range(count, count + batch.num_rows)):
                            raise SensorTransactionError(f"Sync frame indices mismatch: {path}.")
                    count += batch.num_rows
            if "frame_index" in schema_names and count != int(expected_main["frame_count"]):
                raise SensorTransactionError(f"Sync frame indices mismatch: {path}.")

    def _clear_active_pointer(self) -> None:
        path = self.root / ".sensor-staging/active_transaction.json"
        if path.exists():
            pointer = json.loads(path.read_text(encoding="utf-8"))
            if pointer.get("episode_uid") != self.episode_uid:
                raise SensorTransactionError("Conflicting active transaction pointer.")
            path.unlink()

    def _cleanup_staging(self) -> None:
        _require_writer(self.root)
        staging_root = self.root / ".sensor-staging" / self.episode_uid
        if staging_root.exists():
            shutil.rmtree(staging_root)
        self._clear_active_pointer()

    def _quarantine_staging(self) -> None:
        staging = self.root / ".sensor-staging" / self.episode_uid
        if not staging.exists():
            return
        quarantine = self.root / ".sensor-quarantine" / self.episode_uid
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            raise SensorTransactionError(f"Quarantine destination already exists: {quarantine}.")
        os.replace(staging, quarantine)


class TransactionRecoveryManager:
    """Recover one discoverable active transaction under an explicit writer lock."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def recover(self, *, writer_lock=None):
        """Discover and replay only the active transaction or uncleaned staging."""
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
                    raise SensorTransactionError("Unidentified staging cannot be recovered safely.")
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
                if intent.get("episode_uid") != uid:
                    raise SensorTransactionError("Staging intent UID mismatch.")
                journal = self.root / "meta/sensor_transactions" / f"{uid}.json"
                if journal.exists():
                    candidates.append(SensorTransaction.load(journal))
                else:
                    candidates.append(
                        SensorTransaction(
                            self.root,
                            {
                                "journal_version": JOURNAL_VERSION,
                                "episode_uid": uid,
                                "state": TransactionState.RECORDING,
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
