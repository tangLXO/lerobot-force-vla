#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Deliberately slow, independent definition of all dense-window fields."""

import numpy as np


def reference_window(rows, targets, names, age_ns, start_sequence):
    output = {
        "values": np.zeros((len(targets), len(names)), dtype=np.float32),
        "valid_mask": np.zeros(len(targets), dtype=bool),
        "target_timestamp_ns": np.asarray(targets, dtype=np.int64),
        "source_timestamp_ns": np.full(len(targets), -1, dtype=np.int64),
        "sequence": np.full(len(targets), -1, dtype=np.int64),
        "age_ns": np.full(len(targets), -1, dtype=np.int64),
    }
    for index, grid in enumerate(targets):
        eligible = []
        for row in rows:
            semantic = row.get("values") or {}
            if (
                max(row["timestamp_ns"], row["arrival_timestamp_ns"]) <= int(grid)
                and row["timestamp_ns"] >= int(grid) - age_ns
                and row["is_valid"]
                and row["sequence"] >= start_sequence
                and all(semantic.get(name) is not None for name in names)
            ):
                eligible.append(row)
        if not eligible:
            continue
        chosen = max(eligible, key=lambda row: (row["timestamp_ns"], row["sequence"]))
        output["values"][index] = [chosen["values"][name] for name in names]
        output["valid_mask"][index] = True
        output["source_timestamp_ns"][index] = chosen["timestamp_ns"]
        output["sequence"][index] = chosen["sequence"]
        output["age_ns"][index] = int(grid) - chosen["timestamp_ns"]
    return output
