# Sensor Dataset Sidecar Format

Status: **Sidecar v1 / transaction journal format v1**

This document defines the native-rate Sensor Sidecar stored beside an ordinary LeRobot
Dataset. The main Dataset remains fixed-FPS: selected scalar Sensor features are part of
`observation.state`, while raw-rate acquisition and real capture timing are stored only in the
Sidecar. Sliding windows are reconstructed and are never persisted.

## Layout and identity

Every Sensor episode has a UUID4 `episode_uid`. Its committed metadata maps that UID to the
main Dataset `episode_index`; Readers build their index only from committed transactions and
episode metadata.

```text
dataset_root/
├── meta/
│   ├── sensor_streams.json
│   ├── sensor_episodes/<episode_uid>.json
│   └── sensor_transactions/<episode_uid>.json
├── raw/
│   ├── sensors/<instance>/<episode_uid>.parquet
│   └── sync/<episode_uid>.parquet
├── .sensor-staging/
│   ├── active_transaction.json
│   └── <episode_uid>/transaction_intent.json, spool/, ...
├── .sensor-quarantine/<episode_uid>/...
└── .sensor-writer.lock
```

Physical paths are examples of the version-1 `per_episode_parquet` adapter. Code must resolve
them from the templates in `sensor_streams.json`, not embed them independently.

## Dataset-level manifest

`meta/sensor_streams.json` is the stable resume contract. It contains:

- Sidecar schema and layout versions plus Raw, Sync, metadata, and journal path templates;
- ordered stream instances;
- ordered semantic and native schemas (`name`, `dtype`, `unit`, scalar `shape`);
- ordered relative `state_features` selection;
- native value/payload capabilities;
- the host-monotonic measurement/availability clock fields;
- logical Raw/Sync schemas and causal selection rules.

The manifest never contains a particular device, driver, firmware, channel mapping,
calibration, resolved timing/queue values, observed sampling rate, sequence gaps, or error
statistics. Those are episode facts. Resume requires an exact match of the stable manifest,
including feature and state-selection order.

## Episode metadata

`meta/sensor_episodes/<episode_uid>.json` records `episode_uid`, `episode_index`, main frame
count, and per-stream episode facts:

- driver/hardware/calibration and clock provenance supplied by the backend;
- resolved static maximum age and recorder queue capacity;
- sample, valid, invalid, sequence-gap, and queue-overflow counts;
- observed sample rate.

Different episodes may use different hardware or calibration only when the stable semantic
schema, units/dtypes, namespace, and state selection still match the Dataset manifest.

## Raw Parquet

`raw/sensors/<instance>/<episode_uid>.parquet` contains one row per acquisition attempt drained
from that episode's independent subscriber queue:

| Field | Meaning |
| --- | --- |
| `episode_uid`, `episode_time_ns` | Stable episode identity and measurement time relative to recorder start. |
| `timestamp_ns`, `arrival_timestamp_ns` | Host-monotonic measurement and framework-availability time. |
| `sequence` | Framework sequence; increments for valid and invalid publication attempts. |
| `hardware_timestamp_ns`, `hardware_sequence` | Nullable device-native timing/sequence. |
| `is_valid`, `status`, `error` | Acquisition validity and diagnostic state. |
| `values` | Typed struct of calibrated semantic/SI values. |
| `native_values` | Nullable typed struct of device-native numeric values. |
| `native_payload` | Nullable binary payload; recorded only when explicitly enabled. |

Raw completeness does not depend on online History duration. Subscriber overflow invalidates
the episode; it may not become `COMMITTED`.

## Sync Parquet

`raw/sync/<episode_uid>.parquet` contains exactly one row for each frame actually passed to
`dataset.add_frame()`. It stores:

- UID and episode-local frame index;
- `observation_start_ns`, fixed `frame_anchor_ns`, and `observation_complete_ns`;
- nullable future Robot/Camera `hardware_observation_timestamps`;
- for every stream, the selected sample's measurement/arrival timestamps,
  framework/hardware sequence, age, and status.

Buffered rollout modes must keep frame data and capture metadata together. A discarded buffer
creates neither a main frame nor a Sync row.

## Transaction journal

`meta/sensor_transactions/<episode_uid>.json` is retained for audit and deterministic recovery.
Transaction journal format v1 records a lightweight precondition, expected episode/frame/logical
data range, required streams, subscription start sequences, staging identity, and actual artifact
locators registered before main-file mutation. Immutable Sidecars have file digests. Main evidence
uses the exact episode metadata row, continuous logical data rows, schema and a versioned
logical-row digest; shared videos have paths and time ranges. Appending another episode does not
invalidate older evidence through a whole shared-file SHA mismatch.

The process-crash-recoverable state machine is:

```text
RECORDING -> PREPARED -> MAIN_SAVED -> SIDECAR_PROMOTED -> COMMITTED
definitely unsaved -> ABORTED
partial writes / identity damage / conflicting evidence -> QUARANTINED
```

Each journal transition is written through a temporary file, flushed/fsynced, and atomically
replaced. One live recorder holds the Dataset writer lock. This is
**process-crash recoverable + replayable on-disk state**, without power-loss durability or concurrent snapshot isolation.

- `RECORDING`: acquire the writer lock, create UID staging and a discovery intent, persist the
  journal, atomically replace the fixed active pointer, then subscribe. Intent and pointer only
  locate the transaction; neither is a second authoritative journal.

- `PREPARED`: workers are stopped/unsubscribed and all staging artifacts are closed and
  verified before the main `save_episode()` call.
- `MAIN_SAVED`: the main writer first seals/rotates its buffered Parquet artifacts, then the
  expected episode row, frame count, data range, contiguous indices, referenced data file,
  and any required video files are verified—not inferred from totals alone.
- `SIDECAR_PROMOTED`: every final artifact matches its prepared digest; partial promotion is
  idempotently continued.
- `COMMITTED`: both halves verify. Only this state is visible to Readers; replay also finishes
  an interrupted post-commit staging cleanup.
- `ABORTED`: available evidence proves no main save occurred; staging is quarantined.
- `QUARANTINED`: state is partial, conflicting, corrupt, or cannot be proven safe. No guessed
  repair or commit is allowed.

Recorder startup and each new episode recover the active transaction while holding the writer
lock. Missing/stale/interrupted pointers are rebuilt from uncleaned staging only, without scanning
historical COMMITTED journals. Conflicting candidates are refused. Terminal journals persist before
staging cleanup and pointer removal. A save exception whose replay reaches COMMITTED is a successful
save, with Writer memory synchronized for the next episode.

Reader opening and read entries never create/remove locks, replay, repair or update journals.
Recovery is explicit when needed:

```python
from pathlib import Path
from lerobot.datasets.sensor_transaction import TransactionRecoveryManager

recovery = TransactionRecoveryManager(Path("dataset_root"))
recovery.recover()  # acquires the writer lock; discovers only the active transaction
```

Earlier prototype journals are unsupported and have no migration path. Recovery discovers only
the active pointer and uncleaned staging; it never scans historical committed journals.
Staging and quarantine directories are excluded from Hub upload; committed Sidecars are part of
a normal whole-Dataset upload.

## Causal Reader and window adapter

`SensorStreamReader.get_window(end_timestamp_ns, duration_ms, target_hz=None,
max_age_ms=None)` reads `(end-duration, end]`.

With `target_hz=None`, it returns native-rate rows in the interval; invalid rows remain present
with `valid_mask=false`. With a positive target rate, output length is
`ceil(duration_ms * target_hz / 1000)`, the final grid point equals `end_timestamp_ns`, and each
point uses causal previous-hold only. A candidate must satisfy:

```text
timestamp_ns <= grid_timestamp_ns
arrival_timestamp_ns <= grid_timestamp_ns
is_valid
grid_timestamp_ns - timestamp_ns <= max_age
sequence >= episode_start_sequence[instance]
```

Among eligible candidates, select the maximum `(timestamp_ns, sequence)`. Row-group statistics
conservatively prune candidates without assuming measurement order; missing statistics disable the
corresponding pruning. Candidate and grid arrays are blocked, with expiration checked for every
grid point. Relative rational grid offsets preserve integer nanoseconds and the exact final anchor.

Missing/stale points are zero-filled with `valid_mask=false`; missing provenance uses `-1`.
The returned fields are `values [T,D]`, `valid_mask`, `target_timestamp_ns`,
`source_timestamp_ns`, `sequence`, and `age_ns`.

`SensorWindowDataset` aligns each requested window to the corresponding recorded Sync
`frame_anchor_ns` and adds `item["sensor_windows"][instance]`. DataLoader windows require a
positive target rate and a finite resolved max-age at initialization; explicit window max-age 0
is legal. Ragged native-rate access is Reader-only. Phase one provides no temporal
encoder. Policies that do not explicitly declare window support reject enabled windows, while
the default path remains the standard `observation.state` interface used by ACT, π0, and
SmolVLA.

`SensorStreamReader(..., verify="fast")` checks selected main logical evidence, small JSON digests,
and Sidecar size/footer/schema/row count. `verify="full"` additionally hashes and validates all
Sidecar rows. Both are read-only. Selected unfinished/quarantined episodes are refused. Unselected
failed transactions normally do not block a subset, but touched shared data/episode metadata must
retain matching logical evidence. Shared video/global metadata conflicts without sufficient content
evidence require explicit recovery. Live local Writers are refused on opening and again at read
entries. Changed file stamps require reopening after recovery; this is not snapshot isolation.

Sync anchors use a four-episode LRU. The Raw row-group LRU defaults to 64 MiB per worker
(`DatasetConfig.sensor_window_cache_mb`); zero disables residency and negative/non-finite values
are invalid. Accounting uses decoded Arrow buffers. Oversized groups stream without residency.
Pickle excludes caches/PID state; PID changes rebuild process-local state. `__getitems__()` fetches
base frames in a batch, coalesces episode/stream ranges, reads their row-group union and restores
input order, including duplicates. Single-item reads use the same path. Factory decides train/eval
episodes before constructing the actual wrappers.

## Acquisition bounds and runtime diagnostics

Raw and Sync use independent bounded queues/workers and complete Parquet fragments in staging.
`SensorConfig.recorder_flush_rows=4096` and `recorder_flush_interval_s=0.5` are runtime settings,
not stable manifest fields. Disk indexes track fragments and Sync references. Final merging uses
Arrow batches, so capture buffers do not grow with episode duration. Shutdown refuses new Sync,
unsubscribes, drains/flushes/closes, joins, then checks removal. Join timeout retains ownership,
workers and files and prevents reuse. Required startup includes raw-only streams; optional streams
do not block. Unknown future windows retain Raw; trimming requires explicit boundaries and finite
max-age and filters late arrivals during final merge.

`recorder.diagnostics` exposes Raw and Sync queue utilization/high-water marks, overflow, lag,
rows/invalid attempts/sequence gaps, spool/final bytes, fragments/row groups, compression and flush
timings, plus worker errors. The last snapshot survives cleanup. Above 75% capacity, the first
warning is immediate and repeats at most every 30 seconds per queue; overflow remains fatal.
COMMITTED emits one concise summary. Additional metrics stay in runtime/log data and do not alter
stable manifest or Sidecar schemas.

## Hub subset localization

`LeRobotDataset(..., episodes=[...], revision=..., token=...)` localizes Sensor files before a
Sensor Reader is constructed, including when main data is cached. Metadata download selects
ordinary metadata and small Sensor identity JSONs, without all journals or Sync files. The loader
discovers UIDs, validates selected COMMITTED journals, then downloads exact required main and
Sidecar paths. Unselected Raw, Sync and journals are not downloaded. Subsequent requests use one
pinned commit, the same token and the same local directory or Hub snapshot cache. Incomplete
closures fail explicitly. Complete local datasets need no Hub request. Readers never download on
demand; remote random window access is outside this contract.

## Benchmark

```bash
uv run python benchmarks/sensor_pipeline.py --root outputs/benchmarks/sensor_pipeline
# Short smoke workload; use a new output directory for every run:
uv run python benchmarks/sensor_pipeline.py --root outputs/benchmarks/sensor_smoke --duration-s 1 --read-frames 12 --batch-size 4
```

Defaults are two 1000 Hz streams, 30 logical minutes, 30 Hz main frames and 500 ms / 200 Hz windows.
Synthetic acquisition is accelerated with producer-side throttling. The JSON report contains
capture statistics plus sequential/random/batch throughput, decoded row-group counts, cache hits
and resident bytes. Synthetic-clock lag is not physical-device latency. The 5× speedup is a non-CI
target, not a guarantee or timing assertion; CI checks structural I/O and correctness.

## Phase-one edit boundary

If `meta/sensor_streams.json` exists, mutating Dataset edit operations are refused because phase
one does not rewrite Sidecar UIDs, indices, schemas, or transactions. Read-only Dataset info is
allowed. Selective Hub episode download is supported as above. Sidecar-aware split/delete/merge
editing remains out of scope; read-only train/eval subsets do not rewrite stored episodes.
