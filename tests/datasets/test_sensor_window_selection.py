#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sensor_window_oracle import reference_window

from lerobot.datasets.sensor_window_selection import (
    candidate_row_groups,
    dense_grid,
    empty_window,
    reduce_candidates,
)

SCHEMA = pa.schema(
    [
        ("timestamp_ns", pa.int64()),
        ("arrival_timestamp_ns", pa.int64()),
        ("sequence", pa.int64()),
        ("is_valid", pa.bool_()),
        ("values", pa.struct([("left", pa.float32()), ("right", pa.float32())])),
    ]
)


@pytest.mark.parametrize("case", range(1000))
def test_blocked_selection_matches_independent_oracle(case):
    rng = np.random.default_rng(20260905 + case)
    base = [0, 1_000_000_000, 9_000_000_000_000_000_000][case % 3]
    count = int(rng.integers(0, 80))
    sequence = np.cumsum(rng.integers(1, 5, count))
    arrivals = base - 150_000_000 + np.cumsum(rng.integers(0, 5_000_000, count))
    timestamps = base + rng.integers(-100, 100, count) * 1_000_000
    rows = [
        {
            "timestamp_ns": int(timestamps[i]),
            "arrival_timestamp_ns": int(arrivals[i]),
            "sequence": int(sequence[i]),
            "is_valid": bool(rng.integers(0, 2)),
            "values": {"left": float(i), "right": -float(i)},
        }
        for i in range(count)
    ]
    if count and case % 7 == 0:
        rows[-1]["values"] = None
    targets = np.asarray(
        [base + int(value) for value in rng.integers(-90_000_000, 110_000_000, int(rng.integers(0, 150)))],
        dtype=np.int64,
    )
    age = [0, 1, 1_000_000, 5_000_000, 50_000_000, 500_000_000][case % 6]
    boundary = int(rng.integers(0, max(1, count * 3)))
    expected = reference_window(rows, targets, ["left", "right"], age, boundary)
    result = empty_window(targets, 2)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    for batch in table.to_batches(max_chunksize=17):
        reduce_candidates(result, batch, ["left", "right"], age, boundary, candidate_block=7, grid_block=11)
    for field, value in result.as_dict().items():
        np.testing.assert_array_equal(value, expected[field], err_msg=f"case {case}: {field}")


def test_expired_newer_measurement_does_not_hide_still_valid_candidate():
    # The newer measurement arrives much later; prefix-only selection would leak it or
    # retain an expired candidate across grids. The complete predicates are mandatory.
    rows = [
        {
            "timestamp_ns": t,
            "arrival_timestamp_ns": a,
            "sequence": i,
            "is_valid": True,
            "values": {"left": float(i), "right": float(i)},
        }
        for i, (t, a) in enumerate([(10, 10), (20, 100), (15, 101)])
    ]
    targets = np.array([15, 21, 100, 101], dtype=np.int64)
    result = empty_window(targets, 2)
    reduce_candidates(
        result,
        pa.RecordBatch.from_pylist(rows, schema=SCHEMA),
        ["left", "right"],
        10,
        0,
        candidate_block=1,
        grid_block=1,
    )
    np.testing.assert_array_equal(result.sequence, [0, -1, -1, -1])


@pytest.mark.parametrize("rate", [30, 200, 333.3, 29.97])
def test_grid_keeps_large_integer_anchor_exact(rate):
    anchor = 9_000_000_000_000_000_123
    grid = dense_grid(anchor, 500, rate)
    assert int(grid[-1]) == anchor
    assert len(grid) == -(-Fraction("500") * Fraction(str(rate)) // 1000)
    for i, timestamp in enumerate(reversed(grid)):
        assert anchor - int(timestamp) == round(Fraction(i * 1_000_000_000, 1) / Fraction(str(rate)))


def test_candidate_grid_pruning_preserves_shuffled_duplicates_at_int64_limit():
    limit = np.iinfo(np.int64).max
    rows = [
        {
            "timestamp_ns": int(limit) - 5,
            "arrival_timestamp_ns": int(limit) - 3,
            "sequence": 7,
            "is_valid": True,
            "values": {"left": 1.0, "right": 2.0},
        }
    ]
    targets = np.asarray([limit, 0, limit - 4, limit - 3, limit], dtype=np.int64)
    expected = reference_window(rows, targets, ["left", "right"], 10, 7)
    result = empty_window(targets, 2)
    reduce_candidates(result, pa.RecordBatch.from_pylist(rows, schema=SCHEMA), ["left", "right"], 10, 7)
    for field, value in result.as_dict().items():
        np.testing.assert_array_equal(value, expected[field])


@pytest.mark.parametrize("statistics", [False, True])
def test_range_union_visits_each_footer_group_once(tmp_path, statistics):
    rows = [
        {
            "timestamp_ns": timestamp,
            "arrival_timestamp_ns": 100 + index,
            "sequence": index,
            "is_valid": index % 3 != 0,
            "values": {"left": 1.0, "right": 2.0},
        }
        for index, timestamp in enumerate([110, 10, 90, 50, 70, 30, 130, 20])
    ]
    path = tmp_path / "unordered.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA), path, row_group_size=2, write_statistics=statistics
    )
    ranges = [(0, 15), (101, 108), (129, 135), (101, 108)]
    with pq.ParquetFile(path) as parquet:
        expected = sorted({g for lo, hi in ranges for g in candidate_row_groups(parquet, lo, hi, 2)})
        visits = []

        def row_group(index):
            visits.append(index)
            return parquet.metadata.row_group(index)

        counted = SimpleNamespace(
            schema=parquet.schema,
            num_row_groups=parquet.num_row_groups,
            metadata=SimpleNamespace(row_group=row_group),
        )
        assert candidate_row_groups(counted, 0, 0, 2, ranges=ranges) == expected
        assert visits == list(range(parquet.num_row_groups))
        assert candidate_row_groups(parquet, 0, 0, 2, ranges=[]) == []
