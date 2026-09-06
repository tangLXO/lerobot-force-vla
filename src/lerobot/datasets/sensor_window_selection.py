#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Bounded causal window reduction independent of disk layout and caching."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class SensorWindow:
    """Dense values and causal provenance for one stream window."""

    values: np.ndarray
    valid_mask: np.ndarray
    target_timestamp_ns: np.ndarray
    source_timestamp_ns: np.ndarray
    sequence: np.ndarray
    age_ns: np.ndarray

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "values",
                "valid_mask",
                "target_timestamp_ns",
                "source_timestamp_ns",
                "sequence",
                "age_ns",
            )
        }


def finite_number(name, value, *, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}.")


def dense_grid(anchor: int, duration_ms: float, target_hz: float) -> np.ndarray:
    """Round relative rational offsets, preserving even very large integer anchors."""
    finite_number("duration_ms", duration_ms)
    finite_number("target_hz", target_hz)
    rate = Fraction(str(target_hz))
    length = math.ceil(Fraction(str(duration_ms)) * rate / 1000)
    return np.asarray(
        [int(anchor) - round(Fraction(k * 1_000_000_000, 1) / rate) for k in range(length - 1, -1, -1)],
        dtype=np.int64,
    )


def empty_window(targets: np.ndarray, feature_count: int) -> SensorWindow:
    length = len(targets)
    return SensorWindow(
        np.zeros((length, feature_count), dtype=np.float32),
        np.zeros(length, dtype=bool),
        targets,
        np.full(length, -1, dtype=np.int64),
        np.full(length, -1, dtype=np.int64),
        np.full(length, -1, dtype=np.int64),
    )


def reduce_candidates(
    window: SensorWindow,
    batch: pa.RecordBatch,
    feature_names: list[str],
    max_age_ns: int,
    start_sequence: int,
    *,
    candidate_block=4096,
    grid_block=128,
) -> None:
    """Apply every causal predicate before reducing (measurement timestamp, sequence)."""
    for begin in range(0, batch.num_rows, candidate_block):
        chunk = batch.slice(begin, candidate_block)
        timestamp = chunk.column("timestamp_ns").to_numpy(zero_copy_only=False)
        arrival = chunk.column("arrival_timestamp_ns").to_numpy(zero_copy_only=False)
        sequence = chunk.column("sequence").to_numpy(zero_copy_only=False)
        valid = chunk.column("is_valid").to_numpy(zero_copy_only=False).copy()
        semantic = chunk.column("values")
        valid &= semantic.is_valid().to_numpy(zero_copy_only=False)
        values = np.empty((chunk.num_rows, len(feature_names)), dtype=np.float32)
        for column, name in enumerate(feature_names):
            array = semantic.field(name)
            valid &= array.is_valid().to_numpy(zero_copy_only=False)
            values[:, column] = array.to_numpy(zero_copy_only=False)
        order = np.lexsort((sequence, timestamp))
        timestamp, arrival, sequence, valid, values = (
            array[order] for array in (timestamp, arrival, sequence, valid, values)
        )
        available = np.maximum(timestamp, arrival)
        ranks = np.arange(len(timestamp), dtype=np.int32)
        # A candidate chunk can only affect this interval. Packed random windows
        # may be far apart; omit unrelated grids before allocating the mask.
        # Python integers keep the expiration bound safe near int64 limits.
        usable = valid & (sequence >= start_sequence)
        if not usable.any():
            continue
        earliest = int(available[usable].min())
        latest = min(np.iinfo(np.int64).max, int(timestamp[usable].max()) + max_age_ns)
        grid_positions = np.flatnonzero(
            (window.target_timestamp_ns >= earliest) & (window.target_timestamp_ns <= latest)
        )
        for grid_begin in range(0, len(grid_positions), grid_block):
            selected_positions = grid_positions[grid_begin : grid_begin + grid_block]
            targets = window.target_timestamp_ns[selected_positions]
            lower = np.asarray(
                [max(np.iinfo(np.int64).min, int(g) - max_age_ns) for g in targets], dtype=np.int64
            )
            eligible = (
                (available[None, :] <= targets[:, None])
                & (timestamp[None, :] >= lower[:, None])
                & valid[None, :]
                & (sequence[None, :] >= start_sequence)
            )
            winners = np.where(eligible, ranks[None, :], -1).max(axis=1)
            present = winners >= 0
            positions = selected_positions[present]
            winners = winners[present]
            better = (
                ~window.valid_mask[positions]
                | (timestamp[winners] > window.source_timestamp_ns[positions])
                | (
                    (timestamp[winners] == window.source_timestamp_ns[positions])
                    & (sequence[winners] > window.sequence[positions])
                )
            )
            positions, winners = positions[better], winners[better]
            window.values[positions] = values[winners]
            window.valid_mask[positions] = True
            window.source_timestamp_ns[positions] = timestamp[winners]
            window.sequence[positions] = sequence[winners]
            window.age_ns[positions] = window.target_timestamp_ns[positions] - timestamp[winners]


def candidate_row_groups(
    parquet, lower_ns: int, upper_ns: int, start_sequence: int, *, valid_only=True, ranges=None
) -> list[int]:
    """Prune a range union in one footer pass without assuming measurement order."""
    intervals = [(lower_ns, upper_ns)] if ranges is None else ranges
    columns = {
        name: parquet.schema.names.index(name)
        for name in ("timestamp_ns", "arrival_timestamp_ns", "sequence", "is_valid")
    }
    groups = []
    for group in range(parquet.num_row_groups):
        metadata = parquet.metadata.row_group(group)
        stats = {name: metadata.column(index).statistics for name, index in columns.items()}

        def has(name, stats=stats):
            return stats[name] is not None and stats[name].has_min_max

        if has("sequence") and stats["sequence"].max < start_sequence:
            continue
        if valid_only and has("is_valid") and not stats["is_valid"].max:
            continue
        if any(
            (
                not has("timestamp_ns")
                or (stats["timestamp_ns"].max >= lower and stats["timestamp_ns"].min <= upper)
            )
            and (not has("arrival_timestamp_ns") or stats["arrival_timestamp_ns"].min <= upper)
            for lower, upper in intervals
        ):
            groups.append(group)
    return groups
