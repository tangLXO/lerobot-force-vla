#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Process-local, decoded-byte bounded row-group cache. No persistent Arrow handles."""

from collections import OrderedDict

import pyarrow as pa


class SensorRowGroupCache:
    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.entries = OrderedDict()
        self.decoded_bytes = 0
        self.hits = 0
        self.misses = 0

    def batches(self, parquet, uid, instance, group):
        key = uid, instance, group
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            yield from self.entries[key].to_batches(max_chunksize=4096)
            return
        self.misses += 1
        # Oversized groups remain streamable without accumulating a whole decoded group.
        if not self.limit_bytes or parquet.metadata.row_group(group).total_byte_size > self.limit_bytes:
            yield from parquet.iter_batches(batch_size=4096, row_groups=[group])
            return
        pending, pending_bytes = [], 0
        resident = True
        for batch in parquet.iter_batches(batch_size=4096, row_groups=[group]):
            if not resident:
                yield batch
                continue
            pending.append(batch)
            pending_bytes += batch.get_total_buffer_size()
            if pending_bytes > self.limit_bytes:
                # Dictionary expansion can exceed the footer's uncompressed estimate.
                yield from pending
                pending.clear()
                resident = False
        if not resident:
            return
        table = pa.Table.from_batches(pending)
        size = table.get_total_buffer_size()
        if size <= self.limit_bytes:
            while self.entries and self.decoded_bytes + size > self.limit_bytes:
                _, removed = self.entries.popitem(last=False)
                self.decoded_bytes -= removed.get_total_buffer_size()
            self.entries[key] = table
            self.decoded_bytes += size
        yield from table.to_batches(max_chunksize=4096)
