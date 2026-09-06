#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Strictly read-only sensor evidence verification. No recovery or lock mutation."""

import json
import re
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq

from .sensor_stream import SensorRecorderError, _process_is_alive
from .sensor_transaction import SensorTransactionError, sha256_file
from .sensor_transaction_v2 import _episode_rows, capture_logical_evidence


def check_no_live_writer(root: Path) -> None:
    path = root / ".sensor-writer.lock"
    if not path.exists():
        return
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        pid = int(owner["pid"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SensorRecorderError(
            "Sensor Dataset has an unreadable writer lock; explicit recovery is required."
        ) from exc
    if _process_is_alive(pid):
        raise SensorRecorderError(
            f"Sensor Dataset has a live writer (pid={pid}); local reading is unavailable."
        )


def resolve_artifact(root: Path, relative: str) -> Path:
    path = root / relative
    resolved = path.resolve()
    hub_blobs = root.parent.parent / "blobs"
    in_snapshot = root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", root.name)
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path == root
        or (root not in resolved.parents and not (in_snapshot and hub_blobs in resolved.parents))
    ):
        raise SensorTransactionError(f"Sensor artifact escapes Dataset root: {relative!r}.")
    return path


def file_stamp(path: Path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def verify_sidecar(path: Path, record: dict, uid: str, *, full: bool, frame_count: int):
    before = file_stamp(path)
    if before[0] != record["size"]:
        raise SensorTransactionError(f"Sensor artifact size mismatch: {path}.")
    if (full or record.get("kind") == "json") and sha256_file(path) != record["sha256"]:
        raise SensorTransactionError(f"Sensor artifact digest mismatch: {path}.")
    if record.get("kind") != "json":
        with path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
            if (
                parquet.metadata.num_rows != record["row_count"]
                or str(parquet.schema_arrow) != record["schema"]
            ):
                raise SensorTransactionError(f"Sensor artifact schema/row count mismatch: {path}.")
            if full:
                count = 0
                for batch in parquet.iter_batches(batch_size=4096):
                    if any(value != uid for value in batch.column("episode_uid").to_pylist()):
                        raise SensorTransactionError(f"Sensor artifact UID mismatch: {path}.")
                    if "frame_index" in batch.schema.names and batch.column(
                        "frame_index"
                    ).to_pylist() != list(range(count, count + batch.num_rows)):
                        raise SensorTransactionError("Sensor Sync frame index mismatch.")
                    count += batch.num_rows
                if "frame_index" in parquet.schema_arrow.names and count != frame_count:
                    raise SensorTransactionError("Sensor Sync frame count differs from main episode.")
    if before != file_stamp(path):
        raise SensorTransactionError("Sensor artifact changed during verification.")


def _legacy_logical_view(transaction):
    """Discover one v1 logical range via footer pruning, never old snapshot hashes."""
    expected, root = transaction.journal["expected_main"], transaction.root
    index = expected["episode_index"]
    metadata_path, metadata_row = None, None
    for path in (root / "meta/episodes").rglob("*.parquet"):
        with closing(_episode_rows(path, index)) as rows:
            for row in rows:
                if metadata_row is not None:
                    raise SensorTransactionError("Duplicate v1 main episode identity.")
                metadata_path, metadata_row = path, row
    if metadata_row is None:
        raise SensorTransactionError("Missing v1 main episode metadata.")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    artifacts = [
        {"path": metadata_path.relative_to(root).as_posix(), "role": "episode_metadata"},
        {
            "path": info["data_path"].format(
                chunk_index=metadata_row["data/chunk_index"], file_index=metadata_row["data/file_index"]
            ),
            "role": "data",
        },
    ]
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            artifacts.append(
                {
                    "path": info["video_path"].format(
                        video_key=key,
                        chunk_index=metadata_row[f"videos/{key}/chunk_index"],
                        file_index=metadata_row[f"videos/{key}/file_index"],
                    ),
                    "role": "video",
                }
            )
    # A temporary in-memory view is not a journal upgrade or an inferred sequence boundary.
    from .sensor_transaction import SensorTransaction

    return SensorTransaction(root, {**transaction.journal, "main_artifacts": artifacts})


def verify_main_readonly(transaction, *, shared_conflict=False):
    root = transaction.root.resolve()
    check_no_live_writer(root)
    legacy = transaction.journal["journal_version"] == 1
    if legacy and shared_conflict:
        raise SensorTransactionError(
            "Cannot prove a v1 logical range is unchanged in a conflicting shared artifact."
        )
    view = _legacy_logical_view(transaction) if legacy else transaction
    paths = {
        resolve_artifact(root, item["path"])
        for item in view.journal["main_artifacts"]
        if item["role"] in ("data", "episode_metadata", "video", "info", "tasks", "stats")
    }
    paths.add(root / "meta/info.json")
    before = {path: file_stamp(path) for path in paths}
    try:
        evidence = capture_logical_evidence(
            view, path_resolver=lambda relative: resolve_artifact(root, relative)
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SensorTransactionError(
            "Main episode evidence is incomplete or corrupt; explicit recovery is required."
        ) from exc
    if not legacy and evidence != transaction.journal.get("main_evidence"):
        raise SensorTransactionError("Main episode logical evidence mismatch.")
    if before != {path: file_stamp(path) for path in paths}:
        raise SensorTransactionError("Shared main artifact changed during verification.")
    check_no_live_writer(root)
    return evidence


def main_artifact_paths(transaction):
    view = _legacy_logical_view(transaction) if transaction.journal["journal_version"] == 1 else transaction
    paths = {
        resolve_artifact(transaction.root.resolve(), item["path"])
        for item in view.journal["main_artifacts"]
        if item["role"] != "temporary"
    }
    paths.add(transaction.root.resolve() / "meta/info.json")
    return paths
