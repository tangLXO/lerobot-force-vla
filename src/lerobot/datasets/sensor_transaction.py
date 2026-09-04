#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Crash-replayable transaction journal for sensor sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class SensorTransactionError(RuntimeError):
    """Raised when a sidecar transaction cannot be proven safe."""


class TransactionState(StrEnum):
    """Persistent sidecar transaction states."""

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


def capture_main_dataset_state(root: Path) -> dict[str, Any]:
    """Capture exact main metadata plus all committed main artifact digests."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    artifacts: dict[str, dict[str, Any]] = {}
    for top_level in ("data", "videos", "meta"):
        directory = root / top_level
        if not directory.exists():
            continue
        for path in sorted(file for file in directory.rglob("*") if file.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative == "meta/sensor_streams.json" or relative.startswith(
                ("meta/sensor_episodes/", "meta/sensor_transactions/")
            ):
                continue
            artifacts[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "info_sha256": sha256_file(info_path) if info_path.exists() else None,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "artifacts": artifacts,
    }


def verify_expected_main_episode(root: Path, expected: dict[str, Any]) -> bool:
    """Verify episode metadata, contiguous data indices, and the referenced data file."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_index = int(expected["episode_index"])
    frame_count = int(expected["frame_count"])
    data_from = int(expected["dataset_from_index"])
    data_to = int(expected["dataset_to_index"])
    if data_to - data_from != frame_count:
        return False
    if info.get("total_episodes", 0) < episode_index + 1 or info.get("total_frames", 0) < data_to:
        return False

    episode_rows: list[dict[str, Any]] = []
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        return False
    for path in sorted(episodes_dir.rglob("*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    row = next((item for item in episode_rows if int(item["episode_index"]) == episode_index), None)
    if row is None:
        return False
    if int(row["length"]) != frame_count:
        return False
    if int(row["dataset_from_index"]) != data_from or int(row["dataset_to_index"]) != data_to:
        return False
    task_info = expected.get("task_info")
    if task_info is not None and row.get("tasks") != task_info:
        return False

    info_template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    data_path = root / info_template.format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
    )
    if not data_path.exists():
        return False
    data_rows = pq.read_table(data_path, columns=["index", "episode_index"]).to_pylist()
    matching = [item for item in data_rows if int(item["episode_index"]) == episode_index]
    if len(matching) != frame_count or [int(item["index"]) for item in matching] != list(
        range(data_from, data_to)
    ):
        return False

    video_keys = [
        name for name, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
    ]
    video_template = info.get("video_path")
    for video_key in video_keys:
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        from_key = f"videos/{video_key}/from_timestamp"
        to_key = f"videos/{video_key}/to_timestamp"
        if video_template is None or any(key not in row for key in (chunk_key, file_key, from_key, to_key)):
            return False
        video_path = root / video_template.format(
            video_key=video_key,
            chunk_index=int(row[chunk_key]),
            file_index=int(row[file_key]),
        )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            return False
        if float(row[to_key]) <= float(row[from_key]):
            return False
    return True


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


@dataclass
class SensorTransaction:
    """One durable main-Dataset plus sidecar commit transaction."""

    root: Path
    journal: dict[str, Any]

    @property
    def episode_uid(self) -> str:
        """Return the transaction's stable UUID string."""
        return str(self.journal["episode_uid"])

    @property
    def state(self) -> TransactionState:
        """Return the current durable state."""
        return TransactionState(self.journal["state"])

    @property
    def journal_path(self) -> Path:
        """Return the persistent journal location."""
        return self.root / "meta" / "sensor_transactions" / f"{self.episode_uid}.json"

    @classmethod
    def prepare(
        cls,
        root: Path,
        *,
        episode_uid: str,
        episode_index: int,
        frame_count: int,
        dataset_from_index: int,
        files: list[dict[str, Any]],
        task_info: Any = None,
        main_precondition: dict[str, Any] | None = None,
    ) -> SensorTransaction:
        """Persist PREPARED after every staging artifact has been closed and verified."""
        root = Path(root)
        validate_episode_uid(episode_uid)
        precondition = main_precondition or capture_main_dataset_state(root)
        expected = {
            "episode_index": episode_index,
            "frame_count": frame_count,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_from_index + frame_count,
            "task_info": task_info,
        }
        transaction = cls(
            root=root,
            journal={
                "journal_version": 1,
                "episode_uid": episode_uid,
                "state": TransactionState.PREPARED,
                "main_precondition": precondition,
                "expected_main": expected,
                "main_postcondition": None,
                "files": files,
                "reason": None,
            },
        )
        transaction._verify_staging_files()
        transaction._persist()
        return transaction

    @classmethod
    def load(cls, path: Path) -> SensorTransaction:
        """Load one existing journal without changing it."""
        path = Path(path)
        journal = json.loads(path.read_text(encoding="utf-8"))
        episode_uid = validate_episode_uid(journal.get("episode_uid"))
        if path.stem != episode_uid:
            raise ValueError(f"Sensor transaction filename and episode_uid differ: {path}.")
        return cls(path.parents[2], journal)

    def mark_main_saved(self, main_postcondition: dict[str, Any] | None = None) -> None:
        """Verify the main episode postcondition and advance to MAIN_SAVED."""
        if self.state != TransactionState.PREPARED:
            if self.state in {
                TransactionState.MAIN_SAVED,
                TransactionState.SIDECAR_PROMOTED,
                TransactionState.COMMITTED,
            }:
                return
            raise SensorTransactionError(f"Cannot mark main saved from {self.state}.")
        if not verify_expected_main_episode(self.root, self.journal["expected_main"]):
            raise SensorTransactionError("Main Dataset does not satisfy the prepared episode postcondition.")
        self.journal["main_postcondition"] = main_postcondition or capture_main_dataset_state(self.root)
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

    def _cleanup_staging(self) -> None:
        staging_root = self.root / ".sensor-staging" / self.episode_uid
        if staging_root.exists():
            shutil.rmtree(staging_root)

    def abort(self, reason: str) -> None:
        """Abort a definitely-unsaved main episode and quarantine its staging data."""
        self._quarantine_staging()
        self.journal["reason"] = reason
        self._advance(TransactionState.ABORTED)

    def quarantine(self, reason: str) -> None:
        """Quarantine an ambiguous/conflicting transaction and refuse repair."""
        self._quarantine_staging()
        self.journal["reason"] = reason
        self._advance(TransactionState.QUARANTINED)

    def replay(self) -> TransactionState:
        """Deterministically replay this transaction as far as evidence permits."""
        try:
            if self.state == TransactionState.PREPARED:
                current = capture_main_dataset_state(self.root)
                if current == self.journal["main_precondition"]:
                    self.abort("Main Dataset still exactly matches the PREPARED precondition.")
                    return self.state
                if not verify_expected_main_episode(self.root, self.journal["expected_main"]):
                    self.quarantine("Cannot prove whether the prepared main episode was saved completely.")
                    return self.state
                self.journal["main_postcondition"] = current
                self._advance(TransactionState.MAIN_SAVED)
            if self.state == TransactionState.MAIN_SAVED:
                if not verify_expected_main_episode(self.root, self.journal["expected_main"]):
                    self.quarantine("Main episode no longer matches the journal postcondition.")
                    return self.state
                self.promote_sidecars()
            if self.state == TransactionState.SIDECAR_PROMOTED:
                self.commit()
            if self.state == TransactionState.COMMITTED:
                self._verify_committed()
                self._cleanup_staging()
            return self.state
        except (OSError, ValueError, SensorTransactionError) as exc:
            if self.state not in {TransactionState.ABORTED, TransactionState.QUARANTINED}:
                self.quarantine(str(exc))
            raise SensorTransactionError(
                f"Sensor transaction {self.episode_uid} requires quarantine: {exc}"
            ) from exc

    def _advance(self, state: TransactionState) -> None:
        self.journal["state"] = state
        self._persist()

    def _persist(self) -> None:
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

    def _verify_committed(self) -> None:
        if not verify_expected_main_episode(self.root, self.journal["expected_main"]):
            raise SensorTransactionError("Committed transaction main episode is inconsistent.")
        if self.state != TransactionState.COMMITTED:
            self._verify_main_postcondition()
        self._verify_final_files()
        self._verify_sidecar_identity()

    def _verify_main_postcondition(self) -> None:
        expected = self.journal.get("main_postcondition")
        if expected is None or capture_main_dataset_state(self.root) != expected:
            raise SensorTransactionError("Main Dataset no longer matches the saved postcondition.")

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
            table = pq.read_table(path, columns=columns)
            uids = set(table["episode_uid"].to_pylist())
            if uids and uids != {self.episode_uid}:
                raise SensorTransactionError(f"Sidecar episode UID mismatch: {path}.")
            if "frame_index" in schema_names:
                frame_count = int(expected_main["frame_count"])
                frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
                if int(record["row_count"]) != frame_count or frame_indices != list(range(frame_count)):
                    raise SensorTransactionError(f"Sync frame indices mismatch: {path}.")

    def _quarantine_staging(self) -> None:
        staging = self.root / ".sensor-staging" / self.episode_uid
        if not staging.exists():
            return
        quarantine = self.root / ".sensor-quarantine" / self.episode_uid
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            raise SensorTransactionError(f"Quarantine destination already exists: {quarantine}.")
        os.replace(staging, quarantine)


def replay_sensor_transactions(root: Path, *, refuse_quarantined: bool = True) -> list[SensorTransaction]:
    """Replay every journal before reading or starting a new episode."""
    root = Path(root)
    transactions: list[SensorTransaction] = []
    journal_dir = root / "meta" / "sensor_transactions"
    if not journal_dir.exists():
        return transactions
    for path in sorted(journal_dir.glob("*.json")):
        transaction = SensorTransaction.load(path)
        if transaction.state == TransactionState.QUARANTINED and refuse_quarantined:
            raise SensorTransactionError(
                f"Dataset contains quarantined sensor transaction {transaction.episode_uid}: "
                f"{transaction.journal.get('reason')}"
            )
        transaction.replay()
        if transaction.state == TransactionState.QUARANTINED and refuse_quarantined:
            raise SensorTransactionError(
                f"Dataset contains quarantined sensor transaction {transaction.episode_uid}."
            )
        transactions.append(transaction)
    return transactions
