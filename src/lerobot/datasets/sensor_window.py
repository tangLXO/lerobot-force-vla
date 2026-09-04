#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Committed sidecar reader and causal temporal-window Dataset adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from lerobot.configs.default import SensorWindowConfig
from lerobot.sensors import SensorSample
from lerobot.sensors.synchronization import latest_causal_sample

from .sensor_stream import SensorDatasetWriterLock
from .sensor_transaction import TransactionState, replay_sensor_transactions


@dataclass(frozen=True)
class SensorWindow:
    """Dense values and causal provenance for one stream window."""

    values: np.ndarray
    valid_mask: np.ndarray
    target_timestamp_ns: np.ndarray
    source_timestamp_ns: np.ndarray
    sequence: np.ndarray
    age_ns: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the stable adapter payload keys."""
        return {
            "values": self.values,
            "valid_mask": self.valid_mask,
            "target_timestamp_ns": self.target_timestamp_ns,
            "source_timestamp_ns": self.source_timestamp_ns,
            "sequence": self.sequence,
            "age_ns": self.age_ns,
        }


class SensorStreamReader:
    """Read only COMMITTED episodes through manifest-provided layout templates."""

    def __init__(
        self,
        root: Path,
        *,
        instance: str | None = None,
        episode_uid: str | None = None,
        episode_index: int | None = None,
    ) -> None:
        """Replay journals and optionally bind one stream/episode selection."""
        self.root = Path(root)
        manifest_path = self.root / "meta" / "sensor_streams.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Sensor sidecar manifest is missing: {manifest_path}.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layout = self.manifest.get("storage_layout", {})
        if layout.get("type") != "per_episode_parquet" or layout.get("version") != 1:
            raise ValueError(f"Unsupported sensor storage layout: {layout}.")
        with SensorDatasetWriterLock(self.root):
            transactions = replay_sensor_transactions(self.root)
        self._committed_uids = {
            tx.episode_uid for tx in transactions if tx.state == TransactionState.COMMITTED
        }
        self._episode_metadata: dict[str, dict[str, Any]] = {}
        self._index_to_uid: dict[int, str] = {}
        template = layout["episode_metadata_path_template"]
        for uid in sorted(self._committed_uids):
            path = self.root / template.format(episode_uid=uid)
            if not path.exists():
                raise FileNotFoundError(f"Committed sensor episode metadata is missing: {path}.")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("episode_uid") != uid:
                raise ValueError(f"Sensor episode UID mismatch in {path}.")
            index = int(metadata["episode_index"])
            if index in self._index_to_uid:
                raise ValueError(f"Duplicate committed sensor episode_index {index}.")
            self._episode_metadata[uid] = metadata
            self._index_to_uid[index] = uid
        self.instance = instance
        self.episode_uid = self._resolve_episode_uid(episode_uid, episode_index, required=False)

    @property
    def committed_episode_indices(self) -> tuple[int, ...]:
        """Return indices derived solely from committed episode metadata."""
        return tuple(sorted(self._index_to_uid))

    def read_raw(
        self,
        *,
        instance: str | None = None,
        episode_uid: str | None = None,
        episode_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read one stream's typed native-rate rows via the manifest adapter."""
        selected_instance = instance or self.instance
        if selected_instance is None or selected_instance not in self.manifest["streams"]:
            raise ValueError(f"A known sensor instance is required, got {selected_instance!r}.")
        uid = self._resolve_episode_uid(episode_uid or self.episode_uid, episode_index)
        path = self.root / self.manifest["storage_layout"]["raw_path_template"].format(
            instance=selected_instance, episode_uid=uid
        )
        if not path.exists():
            raise FileNotFoundError(f"Committed raw sensor sidecar is missing: {path}.")
        return pq.read_table(path).to_pylist()

    def read_sync(
        self, *, episode_uid: str | None = None, episode_index: int | None = None
    ) -> list[dict[str, Any]]:
        """Read one episode's frame synchronization rows."""
        uid = self._resolve_episode_uid(episode_uid or self.episode_uid, episode_index)
        path = self.root / self.manifest["storage_layout"]["sync_path_template"].format(episode_uid=uid)
        if not path.exists():
            raise FileNotFoundError(f"Committed sync sidecar is missing: {path}.")
        return pq.read_table(path).to_pylist()

    def get_window(
        self,
        end_timestamp_ns: int,
        duration_ms: float,
        target_hz: float | None = None,
        max_age_ms: float | None = None,
        *,
        instance: str | None = None,
        episode_uid: str | None = None,
        episode_index: int | None = None,
    ) -> SensorWindow:
        """Return raw records or a dense causal previous-hold window."""
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int | float)
            or not math.isfinite(duration_ms)
            or duration_ms <= 0
        ):
            raise ValueError("duration_ms must be positive and finite.")
        if target_hz is not None and (
            isinstance(target_hz, bool)
            or not isinstance(target_hz, int | float)
            or not math.isfinite(target_hz)
            or target_hz <= 0
        ):
            raise ValueError("target_hz must be positive and finite.")
        if max_age_ms is not None and (
            isinstance(max_age_ms, bool)
            or not isinstance(max_age_ms, int | float)
            or not math.isfinite(max_age_ms)
            or max_age_ms < 0
        ):
            raise ValueError("max_age_ms must be finite and non-negative or None.")
        selected_instance = instance or self.instance
        uid = self._resolve_episode_uid(episode_uid or self.episode_uid, episode_index)
        rows = self.read_raw(instance=selected_instance, episode_uid=uid)
        stream_schema = self.manifest["streams"][selected_instance]
        feature_names = [feature["name"] for feature in stream_schema["semantic_features"]]
        start_ns = end_timestamp_ns - int(duration_ms * 1_000_000)
        samples = [_row_to_sample(row) for row in rows]

        if target_hz is None:
            selected_rows = [
                row
                for row in rows
                if start_ns < int(row["timestamp_ns"]) <= end_timestamp_ns
                and int(row["arrival_timestamp_ns"]) <= end_timestamp_ns
            ]
            values = np.zeros((len(selected_rows), len(feature_names)), dtype=np.float32)
            valid = np.zeros(len(selected_rows), dtype=bool)
            targets = np.asarray([int(row["timestamp_ns"]) for row in selected_rows], dtype=np.int64)
            sources = targets.copy()
            sequences = np.asarray([int(row["sequence"]) for row in selected_rows], dtype=np.int64)
            ages = np.zeros(len(selected_rows), dtype=np.int64)
            for index, row in enumerate(selected_rows):
                row_values = row.get("values") or {}
                complete = all(row_values.get(name) is not None for name in feature_names)
                valid[index] = bool(row["is_valid"] and complete)
                if complete:
                    values[index] = [row_values[name] for name in feature_names]
            return SensorWindow(values, valid, targets, sources, sequences, ages)

        length = math.ceil(duration_ms * target_hz / 1000.0)
        step_ns = 1_000_000_000 / target_hz
        targets = np.asarray(
            [round(end_timestamp_ns - (length - 1 - index) * step_ns) for index in range(length)],
            dtype=np.int64,
        )
        values = np.zeros((length, len(feature_names)), dtype=np.float32)
        valid = np.zeros(length, dtype=bool)
        sources = np.full(length, -1, dtype=np.int64)
        sequences = np.full(length, -1, dtype=np.int64)
        ages = np.full(length, -1, dtype=np.int64)
        if max_age_ms is None:
            max_age_ms = self._episode_metadata[uid]["streams"][selected_instance]["resolved_max_age_ms"]
        for index, target in enumerate(targets):
            selected = latest_causal_sample(samples, int(target), max_age_ms)
            if selected is None or any(name not in selected.values for name in feature_names):
                continue
            values[index] = [selected.values[name] for name in feature_names]
            valid[index] = True
            sources[index] = selected.timestamp_ns
            sequences[index] = selected.sequence
            ages[index] = int(target) - selected.timestamp_ns
        return SensorWindow(values, valid, targets, sources, sequences, ages)

    def _resolve_episode_uid(
        self,
        episode_uid: str | None,
        episode_index: int | None,
        *,
        required: bool = True,
    ) -> str | None:
        if episode_uid is not None:
            if episode_uid not in self._committed_uids:
                raise ValueError(f"Sensor episode {episode_uid!r} is not COMMITTED.")
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
        self, dataset, sensor_windows: dict[str, SensorWindowConfig], *, root: Path | None = None
    ) -> None:
        """Validate fixed-length requests and open the committed sidecar reader."""
        for instance, config in sensor_windows.items():
            if config.target_hz is None or config.target_hz <= 0:
                raise ValueError(
                    f"DataLoader sensor window {instance!r} requires a positive target_hz; "
                    "ragged native-rate data is Reader-only."
                )
        self.dataset = dataset
        self.sensor_windows = sensor_windows
        self.reader = SensorStreamReader(root or dataset.root)

    def __len__(self) -> int:
        """Return the base Dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Attach windows ending at the frame's recorded observation anchor."""
        item = self.dataset[index]
        episode_index = int(_scalar(item["episode_index"]))
        frame_index = int(_scalar(item["frame_index"]))
        sync_rows = self.reader.read_sync(episode_index=episode_index)
        sync = next((row for row in sync_rows if int(row["frame_index"]) == frame_index), None)
        if sync is None:
            raise ValueError(f"No sensor sync row for episode {episode_index}, frame {frame_index}.")
        item = dict(item)
        item["sensor_windows"] = {
            instance: self.reader.get_window(
                int(sync["frame_anchor_ns"]),
                config.duration_ms,
                config.target_hz,
                config.max_age_ms,
                instance=instance,
                episode_index=episode_index,
            ).as_dict()
            for instance, config in self.sensor_windows.items()
        }
        return item

    def __getattr__(self, name: str) -> Any:
        """Delegate Dataset metadata and methods."""
        return getattr(self.dataset, name)


def _row_to_sample(row: dict[str, Any]) -> SensorSample:
    return SensorSample(
        timestamp_ns=int(row["timestamp_ns"]),
        arrival_timestamp_ns=int(row["arrival_timestamp_ns"]),
        sequence=int(row["sequence"]),
        hardware_timestamp_ns=row.get("hardware_timestamp_ns"),
        hardware_sequence=row.get("hardware_sequence"),
        values=row.get("values") or {},
        native_values=row.get("native_values"),
        native_payload=row.get("native_payload"),
        is_valid=bool(row["is_valid"]),
        status=row.get("status") or "unknown",
        error=row.get("error"),
    )


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
