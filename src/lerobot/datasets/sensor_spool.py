#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Bounded Parquet fragments and disk indexes for a single recorder worker."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class SensorParquetSpoolWriter:
    """Single-worker spool; memory is bounded by flush_rows, never episode length."""

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        flush_rows: int,
        flush_interval_s: float,
        *,
        episode_uid: str | None = None,
        instance: str | None = None,
    ):
        self.root = root
        self.schema = schema
        self._fragment_schema = schema.with_metadata(
            {
                **(schema.metadata or {}),
                b"sensor_episode_uid": (episode_uid or "").encode(),
                b"sensor_instance": (instance or "").encode(),
            }
        )
        self.flush_rows = flush_rows
        self.flush_interval_s = flush_interval_s
        self.root.mkdir(parents=True, exist_ok=False)
        self.index_path = root / "index.sqlite"
        # Constructed by the controller, then exclusively used by its worker until join.
        self._db = sqlite3.connect(self.index_path, check_same_thread=False)
        self._db.execute("PRAGMA cache_size=-1024")
        self._db.execute("CREATE TABLE fragments (ordinal INTEGER PRIMARY KEY, path TEXT, rows INTEGER)")
        self._db.execute(
            "CREATE TABLE samples (sequence INTEGER PRIMARY KEY, timestamp_ns INTEGER, "
            "arrival_timestamp_ns INTEGER, is_valid INTEGER, hardware_sequence INTEGER, status TEXT)"
        )
        self._db.execute(
            "CREATE TABLE refs (frame_index INTEGER, instance TEXT, sequence INTEGER, "
            "timestamp_ns INTEGER, arrival_timestamp_ns INTEGER, PRIMARY KEY(frame_index, instance))"
        )
        self._db.commit()
        self._rows: list[dict[str, Any]] = []
        self._last_flush = time.perf_counter()
        self._closed = False
        self.row_count = 0
        self.valid_count = 0
        self.fragment_count = 0
        self.written_bytes = 0
        self.peak_buffer_rows = 0
        self.first_arrival_ns: int | None = None
        self.last_arrival_ns: int | None = None
        self.last_sequence: int | None = None
        self.sequence_gaps = 0
        self.row_group_count = 0
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0
        self.flush_count = 0
        self.flush_seconds = 0.0
        self.max_flush_seconds = 0.0
        self.last_lag_ns = 0
        self.max_lag_ns = 0
        self.total_lag_ns = 0
        self.final_rows = 0
        self.final_bytes = 0
        self.final_row_groups = 0

    def append(self, row: dict[str, Any]) -> None:
        """Buffer one row and close a complete fragment at the row bound."""
        if self._closed:
            raise RuntimeError("Sensor spool is closed.")
        available = row.get("arrival_timestamp_ns", row.get("frame_anchor_ns"))
        if available is not None:
            self.last_lag_ns = max(0, time.perf_counter_ns() - available)
            self.max_lag_ns = max(self.max_lag_ns, self.last_lag_ns)
            self.total_lag_ns += self.last_lag_ns
        if "sequence" in row:
            sequence, arrival = row["sequence"], row["arrival_timestamp_ns"]
            if self.last_sequence is not None:
                if sequence <= self.last_sequence:
                    raise ValueError("Sensor spool sequence must strictly increase.")
                self.sequence_gaps += sequence - self.last_sequence - 1
            if self.last_arrival_ns is not None and arrival < self.last_arrival_ns:
                raise ValueError("Sensor spool arrival must not decrease.")
            if self.first_arrival_ns is None:
                self.first_arrival_ns = arrival
            self.last_arrival_ns = arrival
            self.last_sequence = sequence
            self.valid_count += bool(row["is_valid"])
        self._rows.append(row)
        self.row_count += 1
        self.peak_buffer_rows = max(self.peak_buffer_rows, len(self._rows))
        if len(self._rows) >= self.flush_rows:
            self.flush()
        else:
            self.flush_due()

    def flush_due(self) -> None:
        """Close low-rate fragments on time even when no new sample arrives."""
        if self._rows and time.perf_counter() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        """Close Parquet before adding its discoverable disk-index entry."""
        if not self._rows:
            return
        started = time.perf_counter()
        path = self.root / f"fragment-{self.fragment_count:08d}.parquet"
        table = pa.Table.from_pylist(self._rows, schema=self._fragment_schema)
        pq.write_table(table, path, compression="zstd", row_group_size=self.flush_rows)
        with self._db:
            self._db.execute(
                "INSERT INTO fragments VALUES (?, ?, ?)",
                (self.fragment_count, path.name, len(self._rows)),
            )
            if "sequence" in self.schema.names:
                self._db.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        (
                            r["sequence"],
                            r["timestamp_ns"],
                            r["arrival_timestamp_ns"],
                            r["is_valid"],
                            r.get("hardware_sequence"),
                            r.get("status", "ok"),
                        )
                        for r in self._rows
                    ),
                )
            else:
                self._db.executemany(
                    "INSERT INTO refs VALUES (?, ?, ?, ?, ?)",
                    (
                        (
                            row["frame_index"],
                            instance,
                            ref.get("sequence"),
                            ref.get("timestamp_ns"),
                            ref.get("arrival_timestamp_ns"),
                        )
                        for row in self._rows
                        for instance, ref in row["sensors"].items()
                        if ref is not None
                    ),
                )
        # The disk index owns these rows now, even if diagnostic footer reading fails.
        # Publish the completed-fragment count only after all file access has finished.
        self._rows.clear()
        self.written_bytes += path.stat().st_size
        metadata = pq.read_metadata(path)
        self.row_group_count += metadata.num_row_groups
        for group in range(metadata.num_row_groups):
            columns = metadata.row_group(group)
            for column in range(columns.num_columns):
                sizes = columns.column(column)
                self.compressed_bytes += sizes.total_compressed_size
                self.uncompressed_bytes += sizes.total_uncompressed_size
        elapsed = time.perf_counter() - started
        self.flush_count += 1
        self.flush_seconds += elapsed
        self.max_flush_seconds = max(self.max_flush_seconds, elapsed)
        self._last_flush = time.perf_counter()
        self.fragment_count += 1

    def diagnostics(self):
        """Return a runtime snapshot without reading files or retaining per-row telemetry."""
        return {
            "rows": self.row_count,
            "invalid_samples": self.row_count - self.valid_count if "sequence" in self.schema.names else 0,
            "sequence_gaps": self.sequence_gaps,
            "fragments": self.fragment_count,
            "row_groups": self.row_group_count,
            "spool_bytes": self.written_bytes,
            "compressed_bytes": self.compressed_bytes,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compression_ratio": self.uncompressed_bytes / self.compressed_bytes
            if self.compressed_bytes
            else 0,
            "flush_count": self.flush_count,
            "flush_seconds": self.flush_seconds,
            "max_flush_seconds": self.max_flush_seconds,
            "lag_ns": self.last_lag_ns,
            "max_lag_ns": self.max_lag_ns,
            "mean_lag_ns": self.total_lag_ns / self.row_count if self.row_count else 0,
            "buffer_rows": len(self._rows),
            "peak_buffer_rows": self.peak_buffer_rows,
            "final_rows": self.final_rows,
            "final_bytes": self.final_bytes,
            "final_row_groups": self.final_row_groups,
        }

    def close(self) -> None:
        """Flush and release handles, including after a worker exception."""
        if not self._closed:
            try:
                self.flush()
            finally:
                self._db.close()
                self._closed = True

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """Stream complete fragments in publication order with bounded decoding."""
        if not self._closed:
            raise RuntimeError("Join and close the sensor spool before reading it.")
        with ExitStack() as stack:
            db = stack.enter_context(closing(sqlite3.connect(self.index_path)))
            cursor = stack.enter_context(
                closing(db.execute("SELECT ordinal, path, rows FROM fragments ORDER BY ordinal"))
            )
            for expected_ordinal, (ordinal, name, count) in enumerate(cursor):
                if ordinal != expected_ordinal or name != f"fragment-{ordinal:08d}.parquet":
                    raise ValueError("Sensor spool fragment locator mismatch.")
                with (self.root / name).open("rb") as handle, pq.ParquetFile(handle) as parquet:
                    if (
                        not parquet.schema_arrow.equals(self._fragment_schema)
                        or parquet.schema_arrow.metadata != self._fragment_schema.metadata
                    ):
                        raise ValueError("Sensor spool fragment schema or stream identity mismatch.")
                    if parquet.metadata.num_rows != count:
                        raise ValueError("Sensor spool fragment row count mismatch.")
                    for batch in parquet.iter_batches(batch_size=self.flush_rows):
                        yield batch.replace_schema_metadata(self.schema.metadata)

    def merge(self, destination: Path, *, minimum_timestamp_ns: int | None = None) -> int:
        """Finalize without materializing the episode or a list of fragments."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with pq.ParquetWriter(destination, self.schema, compression="zstd") as writer:
            for batch in self.iter_batches():
                if minimum_timestamp_ns is not None:
                    batch = batch.filter(pc.greater_equal(batch.column("timestamp_ns"), minimum_timestamp_ns))
                count += batch.num_rows
                writer.write_batch(batch, row_group_size=self.flush_rows)
        self.final_rows = count
        self.final_bytes = destination.stat().st_size
        self.final_row_groups = pq.read_metadata(destination).num_row_groups
        return count
