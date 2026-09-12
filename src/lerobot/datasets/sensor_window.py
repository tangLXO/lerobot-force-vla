#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Read-only committed sidecars and bounded causal range windows."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lerobot.configs.default import SensorWindowConfig

from .sensor_stream import validate_sidecar_schema_version
from .sensor_transaction import (
    SensorTransaction,
    SensorTransactionError,
    TransactionState,
    validate_episode_uid,
)
from .sensor_verification import (
    check_no_live_writer,
    file_stamp,
    main_artifact_paths,
    resolve_artifact,
    verify_main_readonly,
    verify_sidecar,
)
from .sensor_window_cache import SensorRowGroupCache
from .sensor_window_selection import (
    SensorWindow as SensorWindow,
    candidate_row_groups,
    dense_grid,
    empty_window,
    finite_number,
    reduce_candidates,
)


class SensorStreamReader:
    """Verify and read committed episodes without modifying Dataset or lock state."""

    def __init__(
        self,
        root: Path,
        *,
        instance: str | None = None,
        episode_uid: str | None = None,
        episode_index: int | None = None,
        episodes: list[int] | None = None,
        verify: str = "fast",
        cache_mb: float = 64,
    ):
        self.root = Path(root).resolve()
        finite_number("sensor_window_cache_mb", cache_mb, allow_zero=True)
        self._cache_limit_bytes = int(cache_mb * 1024 * 1024)
        self._pid = None
        self._row_group_cache = None
        if verify not in ("fast", "full"):
            raise ValueError("Sensor verify must be 'fast' or 'full'.")
        self.verify = verify
        check_no_live_writer(self.root)
        self.manifest = json.loads((self.root / "meta/sensor_streams.json").read_text(encoding="utf-8"))
        validate_sidecar_schema_version(self.manifest)
        layout = self.manifest.get("storage_layout", {})
        if layout.get("type") != "per_episode_parquet" or layout.get("version") != 1:
            raise ValueError(f"Unsupported sensor storage layout: {layout}.")
        self.instance = instance
        if instance is not None and instance not in self.manifest["streams"]:
            raise ValueError(f"Unknown sensor instance {instance!r}.")
        if episode_index is not None and episodes is not None:
            raise ValueError("Specify episode_index or episodes, not both.")
        selected_indices = (
            set(episodes)
            if episodes is not None
            else ({episode_index} if episode_index is not None else None)
        )
        if episode_uid is not None:
            validate_episode_uid(episode_uid)
        transactions = [
            SensorTransaction.load(path) for path in (self.root / "meta/sensor_transactions").glob("*.json")
        ]
        self._transactions = {}
        self._episode_metadata = {}
        self._index_to_uid = {}
        self._paths = {}
        self._read_stamps = {}
        self._anchor_cache = OrderedDict()
        self._ensure_process()
        pending = [
            tx
            for tx in transactions
            if tx.state not in (TransactionState.COMMITTED, TransactionState.ABORTED)
        ]
        selected = [
            tx
            for tx in transactions
            if (episode_uid is None or tx.episode_uid == episode_uid)
            and (selected_indices is None or tx.journal["expected_main"]["episode_index"] in selected_indices)
        ]
        for tx in selected:
            if tx.state == TransactionState.ABORTED:
                continue
            if tx.state != TransactionState.COMMITTED:
                raise SensorTransactionError(
                    f"Sensor episode {tx.episode_uid} is {tx.state}; explicit recovery is required."
                )
            uid = tx.episode_uid
            index = tx.journal["expected_main"]["episode_index"]
            if index in self._index_to_uid:
                raise SensorTransactionError(f"Duplicate committed sensor episode_index {index}.")
            self._check_shared_conflicts(tx, pending)
            verify_main_readonly(tx)
            paths = self._verify_selected_sidecars(tx)
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if (
                metadata.get("episode_uid") != uid
                or metadata.get("episode_index") != index
                or metadata.get("frame_count") != tx.journal["expected_main"]["frame_count"]
            ):
                raise SensorTransactionError("Sensor episode metadata identity mismatch.")
            self._transactions[uid], self._episode_metadata[uid], self._paths[uid] = tx, metadata, paths
            self._index_to_uid[index] = uid
            checked_paths = (
                main_artifact_paths(tx)
                | set(paths["raw"].values())
                | {paths["sync"], paths["metadata"], tx.journal_path}
            )
            self._read_stamps[uid] = {path: file_stamp(path) for path in checked_paths}
        self._committed_uids = set(self._transactions)
        if selected_indices is not None and selected_indices != set(self._index_to_uid):
            raise SensorTransactionError("Requested sensor episodes are missing or not COMMITTED.")
        self.episode_uid = self._resolve_episode_uid(episode_uid, episode_index, required=False)
        check_no_live_writer(self.root)

    def _check_shared_conflicts(self, transaction, unfinished):
        if not unfinished:
            return
        selected = main_artifact_paths(transaction)
        for other in unfinished:
            if other.episode_uid == transaction.episode_uid:
                continue
            artifacts = other.journal.get("main_artifacts")
            if artifacts is None:
                raise SensorTransactionError(
                    "Unfinished transaction has no artifact locators; explicit recovery is required."
                )
            # Logical rows prove data and episode metadata. Video time ranges,
            # global metadata and diagnostic stamps cannot prove unchanged content.
            if any(
                resolve_artifact(self.root, item["path"]) in selected
                and item["role"] not in ("data", "episode_metadata", "temporary")
                for item in artifacts
            ):
                raise SensorTransactionError(
                    "Cannot prove shared main artifact content is unchanged; explicit recovery is required."
                )

    def _verify_selected_sidecars(self, transaction):
        layout, uid = self.manifest["storage_layout"], transaction.episode_uid
        records = transaction.journal["files"]
        paths = {"raw": {}}
        for name in self.manifest["streams"]:
            record = next(
                (r for r in records if r.get("instance") == name or f"/sensors/{name}/" in r["final_path"]),
                None,
            )
            if record is None:
                raise SensorTransactionError(f"Missing committed Raw evidence for {name!r}.")
            path = resolve_artifact(
                self.root, layout["raw_path_template"].format(instance=name, episode_uid=uid)
            )
            verify_sidecar(
                path,
                record,
                uid,
                full=self.verify == "full",
                frame_count=transaction.journal["expected_main"]["frame_count"],
            )
            paths["raw"][name] = path
        for name, template, predicate in (
            (
                "sync",
                "sync_path_template",
                lambda r: r.get("kind") != "json" and "frame_index:" in r.get("schema", ""),
            ),
            ("metadata", "episode_metadata_path_template", lambda r: r.get("kind") == "json"),
        ):
            record = next((r for r in records if predicate(r)), None)
            if record is None:
                raise SensorTransactionError(f"Missing committed {name} evidence.")
            path = resolve_artifact(self.root, layout[template].format(episode_uid=uid))
            verify_sidecar(
                path,
                record,
                uid,
                full=self.verify == "full",
                frame_count=transaction.journal["expected_main"]["frame_count"],
            )
            paths[name] = path
        return paths

    @property
    def committed_episode_indices(self):
        return tuple(sorted(self._index_to_uid))

    def _selection(self, instance, episode_uid, episode_index):
        check_no_live_writer(self.root)
        name = instance or self.instance
        if name not in self.manifest["streams"]:
            raise ValueError(f"A known sensor instance is required, got {name!r}.")
        uid = self._resolve_episode_uid(
            episode_uid or (self.episode_uid if episode_index is None else None), episode_index
        )
        self._check_episode_state(uid)
        return name, uid

    def _check_episode_state(self, uid):
        self._ensure_process()
        check_no_live_writer(self.root)
        if any(file_stamp(path) != stamp for path, stamp in self._read_stamps[uid].items()):
            raise SensorTransactionError(
                "Dataset changed since Sensor Reader verification; reopen after explicit recovery."
            )

    def _ensure_process(self):
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._anchor_cache = OrderedDict()
            self._row_group_cache = SensorRowGroupCache(self._cache_limit_bytes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _anchor_cache=OrderedDict(), _row_group_cache=None)
        return state

    def _boundary(self, uid, instance):
        transaction = self._transactions[uid]
        try:
            return int(transaction.journal["episode_start_sequence"][instance])
        except (KeyError, TypeError, ValueError) as exc:
            raise SensorTransactionError("Missing episode sequence boundary.") from exc

    def resolve_max_age(self, uid, instance, requested):
        age = (
            self._episode_metadata[uid]["streams"][instance]["resolved_max_age_ms"]
            if requested is None
            else requested
        )
        finite_number("Dense sensor window max_age_ms", age, allow_zero=True)
        return int(age * 1_000_000)

    def _iter_raw_batches(self, instance, uid, lower, upper, *, valid_only, ranges=None):
        check_no_live_writer(self.root)
        path = self._paths[uid]["raw"][instance]
        with path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
            groups = candidate_row_groups(
                parquet, lower, upper, self._boundary(uid, instance), valid_only=valid_only, ranges=ranges
            )
            for group in groups:
                for batch in self._row_group_cache.batches(parquet, uid, instance, group):
                    if any(value != uid for value in batch.column("episode_uid").to_pylist()):
                        raise SensorTransactionError("Raw row belongs to a different episode UID.")
                    yield batch
        self._check_episode_state(uid)

    def read_raw(self, *, instance=None, episode_uid=None, episode_index=None):
        name, uid = self._selection(instance, episode_uid, episode_index)
        return [
            row
            for batch in self._iter_raw_batches(
                name, uid, np.iinfo(np.int64).min, np.iinfo(np.int64).max, valid_only=False
            )
            for row in batch.to_pylist()
        ]

    def read_sync(self, *, episode_uid=None, episode_index=None):
        check_no_live_writer(self.root)
        uid = self._resolve_episode_uid(
            episode_uid or (self.episode_uid if episode_index is None else None), episode_index
        )
        self._check_episode_state(uid)
        with self._paths[uid]["sync"].open("rb") as handle, pq.ParquetFile(handle) as parquet:
            result = parquet.read().to_pylist()
        check_no_live_writer(self.root)
        return result

    def frame_anchor(self, episode_index, frame_index):
        check_no_live_writer(self.root)
        uid = self._resolve_episode_uid(None, episode_index)
        self._check_episode_state(uid)
        if uid not in self._anchor_cache:
            chunks = []
            count = 0
            with self._paths[uid]["sync"].open("rb") as handle, pq.ParquetFile(handle) as parquet:
                for batch in parquet.iter_batches(
                    batch_size=4096, columns=["episode_uid", "frame_index", "frame_anchor_ns"]
                ):
                    if any(value != uid for value in batch.column("episode_uid").to_pylist()) or batch.column(
                        "frame_index"
                    ).to_pylist() != list(range(count, count + batch.num_rows)):
                        raise SensorTransactionError("Sync frame anchor identity mismatch.")
                    chunks.append(batch.column("frame_anchor_ns").to_numpy(zero_copy_only=False))
                    count += batch.num_rows
            self._anchor_cache[uid] = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
            if len(self._anchor_cache) > 4:
                self._anchor_cache.popitem(last=False)
        self._anchor_cache.move_to_end(uid)
        if not 0 <= frame_index < len(self._anchor_cache[uid]):
            raise ValueError(f"No sensor Sync for episode {episode_index}, frame {frame_index}.")
        return int(self._anchor_cache[uid][frame_index])

    def get_window(
        self,
        end_timestamp_ns,
        duration_ms,
        target_hz=None,
        max_age_ms=None,
        *,
        instance=None,
        episode_uid=None,
        episode_index=None,
    ):
        finite_number("duration_ms", duration_ms)
        if max_age_ms is not None:
            finite_number("max_age_ms", max_age_ms, allow_zero=True)
        name, uid = self._selection(instance, episode_uid, episode_index)
        features = [feature["name"] for feature in self.manifest["streams"][name]["semantic_features"]]
        if target_hz is None:
            start = int(end_timestamp_ns) - int(duration_ms * 1_000_000)
            rows = [
                row
                for batch in self._iter_raw_batches(
                    name, uid, start + 1, int(end_timestamp_ns), valid_only=False
                )
                for row in batch.to_pylist()
                if start < row["timestamp_ns"] <= end_timestamp_ns
                and row["arrival_timestamp_ns"] <= end_timestamp_ns
                and row["sequence"] >= self._boundary(uid, name)
            ]
            targets = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
            window = empty_window(targets, len(features))
            for index, row in enumerate(rows):
                semantic = row["values"] or {}
                complete = all(semantic.get(feature) is not None for feature in features)
                if complete:
                    window.values[index] = [semantic[feature] for feature in features]
                window.valid_mask[index] = bool(row["is_valid"] and complete)
                window.source_timestamp_ns[index], window.sequence[index], window.age_ns[index] = (
                    row["timestamp_ns"],
                    row["sequence"],
                    0,
                )
            return window
        return self._dense_windows([end_timestamp_ns], duration_ms, target_hz, max_age_ms, name, uid)[0]

    def _dense_windows(self, anchors, duration_ms, target_hz, max_age_ms, name, uid):
        """Reduce a row-group union against packed grids, preserving request order."""
        self._check_episode_state(uid)
        if not anchors:
            return []
        grids = [dense_grid(int(anchor), duration_ms, target_hz) for anchor in anchors]
        age = self.resolve_max_age(uid, name, max_age_ms)
        features = [feature["name"] for feature in self.manifest["streams"][name]["semantic_features"]]
        window = empty_window(np.concatenate(grids), len(features))
        ranges = []
        for lower, upper in sorted((int(grid[0]) - age, int(grid[-1])) for grid in grids):
            if ranges and lower <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], upper))
            else:
                ranges.append((lower, upper))
        with closing(self._iter_raw_batches(name, uid, 0, 0, valid_only=True, ranges=ranges)) as batches:
            for batch in batches:
                reduce_candidates(window, batch, features, age, self._boundary(uid, name))
        result, start = [], 0
        for grid in grids:
            end = start + len(grid)
            result.append(
                SensorWindow(**{key: value[start:end].copy() for key, value in window.as_dict().items()})
            )
            start = end
        return result

    def _resolve_episode_uid(self, episode_uid, episode_index, *, required=True):
        if episode_uid is not None:
            if episode_uid not in self._committed_uids:
                raise ValueError(f"Sensor episode {episode_uid!r} is not COMMITTED.")
            if episode_index is not None and self._index_to_uid.get(episode_index) != episode_uid:
                raise ValueError("Sensor episode UID and index disagree.")
            return episode_uid
        if episode_index is not None:
            if episode_index not in self._index_to_uid:
                raise ValueError(f"Sensor episode_index {episode_index} is not COMMITTED.")
            return self._index_to_uid[episode_index]
        if required:
            raise ValueError("episode_uid or episode_index is required.")
        return None


class SensorWindowDataset:
    """Map-style adapter adding dense sensor windows to existing frame items."""

    def __init__(
        self,
        dataset,
        sensor_windows: dict[str, SensorWindowConfig],
        *,
        root: Path | None = None,
        cache_mb: float = 64,
    ):
        for instance, config in sensor_windows.items():
            if config.target_hz is None:
                raise ValueError(f"DataLoader sensor window {instance!r} requires a positive target_hz.")
        self.dataset, self.sensor_windows = dataset, sensor_windows
        self.reader = SensorStreamReader(
            root or dataset.root, episodes=getattr(dataset, "episodes", None), cache_mb=cache_mb
        )
        for uid in self.reader._committed_uids:
            for instance, config in sensor_windows.items():
                if instance not in self.reader.manifest["streams"]:
                    raise ValueError(f"Unknown sensor window instance {instance!r}.")
                self.reader.resolve_max_age(uid, instance, config.max_age_ms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        indices = list(indices)
        if not indices:
            return []
        check_no_live_writer(self.reader.root)
        fetch = getattr(self.dataset, "__getitems__", None)
        base = fetch(indices) if fetch is not None else [self.dataset[index] for index in indices]
        if len(base) != len(indices):
            raise ValueError("Base Dataset batch length differs from requested indices.")
        items = [dict(item, sensor_windows={}) for item in base]
        grouped = {}
        for position, item in enumerate(items):
            episode, frame = int(_scalar(item["episode_index"])), int(_scalar(item["frame_index"]))
            anchor = self.reader.frame_anchor(episode, frame)
            grouped.setdefault(episode, []).append((position, anchor))
        for episode, requests in grouped.items():
            uid = self.reader._resolve_episode_uid(None, episode)
            for instance, config in self.sensor_windows.items():
                windows = self.reader._dense_windows(
                    [anchor for _, anchor in requests],
                    config.duration_ms,
                    config.target_hz,
                    config.max_age_ms,
                    instance,
                    uid,
                )
                for (position, _), window in zip(requests, windows, strict=True):
                    items[position]["sensor_windows"][instance] = window.as_dict()
        return items

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)


def _scalar(value):
    return value.item() if hasattr(value, "item") else value
