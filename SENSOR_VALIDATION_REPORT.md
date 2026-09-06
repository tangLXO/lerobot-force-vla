# Sensor production validation

The six-stage implementation preserves Sensor/SensorizedRobot APIs, existing policy state
inputs and Sidecar v1 final schemas/layout. Transaction v2 provides **process-crash recoverable
+ replayable on-disk state**; it does not provide power-loss durability or concurrent
Reader/Writer isolation.

## Delivered behavior

1. Non-decreasing arrival publication, complete unseen-interval consuming reads, recorder
   sequence ownership and required-stream reconnect fault latching.
2. Bounded independent Raw/Sync queues and disk spools, required raw-only startup barriers,
   causal episode boundaries, exact Sync references, safe trim and retained ownership on join timeout.
3. Fixed active transaction pointer, RECORDING-before-subscribe, registered main artifact
   locators, logical main evidence, complete video/metadata sealing and idempotent recovery.
4. Strictly read-only fast/full verification, shared-artifact conflict refusal, conservative
   row-group pruning and blocked NumPy window selection with an independent oracle.
5. Worker-local decoded-byte LRU, pickle/PID reset, Windows spawn and batch window construction
   preserving duplicate, shuffled and cross-episode input order. Factory chooses splits first.
6. Pinned-revision Hub subset closure, runtime diagnostics, updated Sensor documentation and
   a configurable capture/read benchmark.

## Validation results

Original baseline: **102 passed**. Independent delivery snapshots were reconstructed without
changing the final worktree; each snapshot was tested against the preceding stages.

| Stage | Validation |
| --- | --- |
| 1 | 118 passed; local commit `2b4717fb` |
| 2 | 252 passed in the final independent snapshot; local commit `784db965` |
| 3 | 343 passed initially; two copied v1 expectations were corrected to v2; all 24 spool tests then passed (346 distinct cases in the resulting gate) |
| 4 | 1,385 passed, including earlier stages and 1,000 independent oracle cases |
| 5 | 1,412 passed, including Windows spawn/batch/cache, split-first factory and config tests |
| 6 | 1,479 passed, 5 platform skips; combined final verification completed |

Ruff, format checks and `git diff --check` pass for all **42** changed/new Python files.
Tests check bounded Raw and Sync buffers, structural open/read/hash counts, read-only tree/mtime
invariants, row-group union I/O, and actual Windows multiprocessing. Crash scenarios use abrupt
subprocess exits and three subsequent recovery passes. A later footer-union optimization passed
1,053 targeted tests; the diagnostic-footer error regression also passed.

### Full pytest accounting

The completed full-suite run reports **3,851 passed, 43 failed, 313 skipped, 4 deselected**.
This is not an all-green full suite.

- Six ACT/VQ-BeT failures were caused by an unwritable external Torch cache. All six passed
  after setting `TORCH_HOME` to the workspace cache.
- The remaining 37 failures were reproduced using original source from `74b75a39`:
  32 OpenCV cases decode existing PNG fixtures as four-channel frames; one Dataset test assumes
  POSIX path separators; one Dataset conversion test deletes a video still open on Windows;
  two checkpoint tests require Windows symlink privileges; one image augmentation test requires
  the missing `cl` compiler.
- Four cases were isolated to prevent a blocking queue receive from aborting the full report.
  RL small-parameter flow passed. Large-parameter flow and both async iterator cases timed out
  in both final and original source. They remain unresolved baseline/environment limitations.

Evidence logs are under `.cache/`: `sensor-pr6-full-completable.log` and its JUnit XML,
`sensor-final-policy-cache.log`, `sensor-final-original-head.log`,
`sensor-pr6-head-failures.log`, `sensor-final-rl-large-head.log`,
`sensor-pr6-head-async.log` and `sensor-pr6-head-async-multiple.log`.

### Benchmark

The complete accelerated capture contains **54,000 main frames**, **1,800,000 Raw rows per
stream**, and 54,000 Sync rows. Capture took 193.50 seconds. Both Raw peak buffers were 4,096
rows; the Sync peak buffer was 430 rows. No overflow or sequence gaps occurred.

Read measurements use 1,024 frames: the first 1,024 sequential frames and a fixed-seed random
sample without replacement across the complete capture. Random single-item and batch reads use
the same indices; batch size is 32. This is a sampled reading benchmark on a full-duration
capture, not a completed all-54,000-frame reading run.

| Mode | Seconds | Frames/s | Decoded row groups | Resident decoded bytes |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 29.57 | 34.63 | 18 | 7,870,752 |
| Random single-item | 34.12 | 30.01 | 1,898 | 66,901,392 |
| Random batch | 10.05 | 101.84 | 1,870 | 66,901,392 |

Random and batch reads each produced 204,800 valid window points. Cache residency stayed below
67,108,864 bytes (64 MiB). Batch speedup was **3.39×**; the non-CI 5× target was not reached.
The full-frame reading run was stopped after capture because it was slow; its incomplete
output is not counted as a completed benchmark. JSON evidence:
`.cache/sensor-benchmark-pruned/benchmark_sampled_results.json`.

## Reproduction environment

Use the existing uv environment with `--no-sync`. Set `UV_CACHE_DIR`, `HF_DATASETS_CACHE`,
`HF_LEROBOT_HOME` and `TORCH_HOME` to workspace `.cache` subdirectories. The local full-suite
run also uses `.cache/ffmpeg-bootstrap` on `PYTHONPATH`: it provides workspace-only aliases to
existing PyAV FFmpeg DLLs so TorchCodec can load. No dependency versions were changed.
All 50 tracked Git LFS test artifacts were materialized and verified against their identities.
Hardware tests were not run against real robots or sensors.

## Local delivery

Branch: `codex/sensor-production`. The user requested six ordered local commits; no GitHub
publication occurs. Each intermediate index tree was compared with its tested independent
snapshot before committing. The temporary approval-service HTTP 503 failure has resolved.

| Commit | Scope |
| --- | --- |
| `2b4717fb` | Core causal reads and recorder sequence ownership |
| `784db965` | Bounded Raw/Sync spool and lifecycle, retaining Transaction v1 |
| `23f5b1fb` | Transaction v2, sealing and crash recovery |
| `36500326` | Strict read-only Reader and causal range windows |
| `3e30d6c3` | Worker-local cache, batch windows and factory integration |
| Final delivery commit | Hub subset, diagnostics, documentation and benchmark |

Final combined regression evidence is `.cache/sensor-production-final.log` and its JUnit XML:
**1,479 passed, 5 skipped**. The skips are unavailable Windows SIGHUP/SIGQUIT signals (four)
and a filesystem symlink requiring privileges (one). The full-suite and sampled-benchmark
limitations above remain part of this delivery; they are not reported as passes.
