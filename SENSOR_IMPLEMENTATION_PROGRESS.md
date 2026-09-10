# Sensor production implementation

Scope: six ordered stages from the user attachment, preserving Sidecar v1 final layout,
Sensor/SensorizedRobot APIs, state inputs, and process-crash recovery (no power-loss guarantee).
The initial user edits to AGENTS.md and three untracked Sensor documents are retained.

The commit IDs in this progress record use the final message-normalized history. The
corresponding implementation trees are identical to those used during validation.

## Verification environment

Use the existing uv environment without dependency synchronization. Windows sandbox caches:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
$env:HF_DATASETS_CACHE = Join-Path $PWD '.cache/hf-datasets'
uv run --no-sync pytest tests/sensors tests/datasets/test_sensor_stream.py tests/datasets/test_sensor_transaction.py tests/datasets/test_sensor_window.py tests/robots/test_sensorized_robot.py tests/robots/test_sensor_policy_compatibility.py -q -p no:cacheprovider --basetemp .cache/sensor-stage-N
```

Use a fresh stage directory for each invocation. Initial baseline: **102 passed**.
Default external uv/pytest/HF caches were not writable; workspace caches resolve this.

## Stage 1 — verified

- Arrival timestamps are non-decreasing; measurement timestamps can be out of order.
  Default arrival time is taken under the publication lock. Rejected publication does not consume sequence.
- Both consuming read APIs inspect the entire unseen interval, return its latest valid sequence,
  and advance the cursor through invalid trailing samples.
- Recorder leases survive unsubscribe, draining, PREPARED, and transaction cleanup.
  Any attempted sequence reset latches an episode failure. Required X518 reconnect failures
  stop background recovery and fault the episode; recovery outside recordings is preserved.
- Faulted recordings become QUARANTINED. Join timeout retains worker identities, files,
  writer lock, and leases; the recorder cannot be reused until cleanup finishes.
- Policy capability is a ClassVar defaulting to false; window max_age_ms=0 is legal.
- Acceptance: **118 passed**, Ruff check passed, Ruff format check passed (20 files),
  `git diff --check` passed. No upstream PR has been published.

## Stage 2 — verified

- Raw and Sync use bounded independent queues/workers and complete Parquet fragments.
  Fragment lists, Raw sequence lookup, and Sync references reside in staging SQLite indexes.
  Raw flush defaults are 4096 rows / 0.5 seconds; Sync uses the same fixed limits.
- Required startup waits include raw-only streams, exclude old-episode history, and share
  the recorder readiness implementation across record and rollout. Optional streams do not block.
- Online selections honor subscription sequence boundaries. Shutdown rejects Sync, unsubscribes,
  drains/closes, joins, then verifies removal; timeout retains ownership and file state.
- Unknown future Highlight windows retain Raw. Explicit finite-age trim applies during bounded
  final merge, including late arrivals with old measurements. Required Sync references are
  checked exactly against this episode's Raw before PREPARED, including timestamps, sequence,
  validity, hardware sequence, status, causality, age, and retained range.
- Empty native schemas are stored as nullable null columns. Corrupt fragments close their
  handles even when Arrow construction raises, allowing Windows quarantine moves.
- Acceptance: **249 passed** (139 Sensor baseline/new tests plus 110 rollout/interactive tests).
  Ruff check, format check (17 changed/new Python files), and `git diff --check` passed.
  Bounded buffer tests cover 256 and 8192 Raw/Sync rows; no whole-episode row or fragment list remains.
  The predecessor transaction implementation remains in use; stage 3 introduces RECORDING and the
  active pointer together.

## Stage 3 — verified

- The finalized transaction implementation records RECORDING before subscriptions, with a fixed
  `.sensor-staging/active_transaction.json` and an episode-local discovery intent.
  The existing journal path remains authoritative. Missing/malformed/stale pointers
  are rebuilt from uncleaned staging; conflicting candidates are refused.
- Main writers register actual data, episode metadata, video, info, tasks and stats
  locators before modification. Size/mtime stamps are diagnostics, never sufficient
  evidence that an existing mutable main artifact was untouched.
- PREPARED contains frame count/range, required streams, sequence boundaries and
  staging identity. Main evidence uses versioned logical row digests and schema,
  the exact episode metadata row, and video paths/time ranges. Appending to shared
  physical data/metadata files preserves previously committed logical evidence.
- Main sealing rotates from the current episode's write position without loading
  historical episodes. Sensor recording seals videos per episode even when ordinary
  main recording is configured for batched encoding; the setting is restored afterward.
- Recovery and mutations require the writer lock. Writer startup discovers only the
  active transaction from its pointer or uncleaned staging; historical journals are never scanned.
  Reader conversion remains stage 4.
- A save exception followed by successful COMMITTED recovery returns success and
  synchronizes the real Writer's in-memory state; the next episode can be recorded.
- Staging fragments carry UID/instance metadata removed from final Sidecar v1 files.
  Raw indexes are cross-checked against fragments. Error paths explicitly close Arrow
  handles, SQLite connections and cursors before Windows quarantine moves.
- Acceptance: **345 passed**, including 40 real subprocess exit scenarios and three
  subsequent idempotent recoveries per scenario, the original Sensor regressions,
  real video sealing, 40 Dataset Writer/Metadata tests and rollout/interactive coverage.
  Ruff check, format check (24 changed/new Python files), and `git diff --check` passed.
- Structural I/O tests compare open counts, read bytes and file hash calls with 0/100
  unrelated historical artifact groups; historical journals/data/metadata are not read.

## Stage 4 — verified

- Reader verification and all read entries are strictly read-only: no lock creation,
  replay, repair or journal writes. Live Writers are refused; dead locks are retained.
  Selected unfinished transactions fail. Unrelated unfinished episodes can be excluded;
  shared data/episode-metadata ranges require matching logical evidence. Conflicts involving
  video or global metadata without sufficient content evidence require explicit recovery.
- Fast verification checks Sidecar size/footer and small JSON digests; full additionally
  hashes and checks every Sidecar row identity. Both verify selected main logical evidence.
  Reads use journal logical evidence and recorded sequence boundaries, never historical
  snapshot hashes.
- Raw range reads conservatively prune row groups without assuming measurement order.
  Dense selection uses bounded candidate/grid blocks and the full causal/age predicate.
  Rational relative grids preserve integer anchors; Sync anchors use a four-episode LRU.
  Unsupported layouts and unresolved dense max-age fail at initialization.
- Acceptance: **1381 passed** at `.cache/sensor-pr4-final`, including all previous gates,
  1000 fixed-seed oracle cases (all six fields), 31 Reader read-only/conflict/I/O tests,
  and the crash matrix. Ruff check, format check (30 Python files) and diff check passed.

## Stage 5 — verified

- DatasetConfig.sensor_window_cache_mb defaults to 64; zero disables residency and
  invalid/non-finite values fail validation. The worker-local Raw row-group LRU accounts
  for decoded Arrow buffers, keys by UID/instance/group, and evicts to its byte budget.
  Oversized groups stream without residency, including dictionary-expanded groups whose
  decoded buffers exceed their footer estimate.
- Reader pickle state excludes both caches and PID state; PID changes rebuild them.
  No Arrow file handles persist on the Reader. Wrapper attribute forwarding is safe
  during unpickling before its base Dataset is restored.
- Explicit batch reads call the base Dataset batch API, group frames by episode/stream,
  coalesce overlapping ranges, decode the row-group union once and reduce packed grids.
  Single-item reads use this path. Output order, duplicate independence and all window
  fields are preserved across arbitrary/cross-episode index lists.
- Factory determines train/eval episodes from metadata before constructing actual
  Datasets/wrappers; no full Reader is discarded. Unsupported window storage/streaming
  requests fail before constructing the main Dataset. Cache settings reach both splits.
- Acceptance: **1408 passed** at `.cache/sensor-pr5-final`, including all prior stages,
  18 new cache/batch/factory tests and 9 existing config tests. The real LeRobotDataset
  spawn test used two Windows workers, shuffle and batch size 3, and compared all six
  window fields with single-item reads. Ruff/format (33 Python files) and diff check passed.

## Stage 6 — verified; see final delivery below

- Hub localization narrows metadata discovery, validates selected COMMITTED journals before
  large downloads, and fetches exact main/Raw/Sync/metadata closure even when main data is cached.
  Requests pin a commit and preserve token/cache destination. Subset reads use logical evidence
  without snapshot verification or upgrades. Complete local data needs no network. Read-only
  snapshot blob links are permitted only within the same Hub repository cache.
- Raw/Sync queue watermarks, utilization and rate-limited >75% warnings are runtime-only.
  Diagnostics include lag, invalid attempts, sequence gaps, spool/final bytes, fragments,
  row groups, compression and flush timings, overflow and worker errors. COMMITTED logs
  one concise summary and retains the last diagnostic snapshot after cleanup.
- Three Sensor documents and AGENTS retain existing material with corrected lifecycle,
  process-crash-only active-transaction recovery, read-only Reader and Hub/cache/batch
  guidance. Markdown local links and Python example syntax are tested.
- `benchmarks/sensor_pipeline.py` implements the default 2x1000 Hz / 30-minute / 30 Hz /
  500 ms / 200 Hz workload and sequential/random/batch measurements. A 1-second smoke run
  captured 1000 rows per stream and 30 Sync frames. CI asserts batch row-group reductions,
  cache bounds and output consistency without a wall-clock speedup gate. The full default
  benchmark is running at `.cache/sensor-benchmark-default` (log `.cache/sensor-benchmark-default.log`);
  54000 main frames have been captured and reading measurements are in progress.
- Combined six-stage gate: **1433 passed, 1 skipped** at `.cache/sensor-pr6-final`.
  The skip is a real filesystem symlink test: this Windows host denies symlink creation.
  Subsequent small changes preserve ordinary non-Sensor download behavior and fix an existing
  Windows signal-test collection bug; final all-suite verification is still outstanding.
- Full pytest initially stopped on TorchCodec DLL loading and eager SIGHUP access in a test.
  Existing PyAV FFmpeg 7 DLLs work with TorchCodec via workspace-only aliases and bootstrap:
  set `PYTHONPATH` to `.cache/ffmpeg-bootstrap` in addition to the cache environment above.
  Video cache tests then passed (8). Signal tests now retain POSIX os.kill behavior and use
  CRT raise_signal on Windows; 8 passed, 4 platform skips. No dependency version was changed.
- All 50 tracked LFS test artifacts (72 MB) have been fetched and materialized. Git status/diff
  may require .git LFS temporary-file access after hydration; the verification succeeded with
  sandbox escalation. Artifact contents match tracked LFS identities.
- The unrestricted full suite reached about 81% before pytest-timeout terminated the process
  in `tests/rl/test_actor_learner.py::test_end_to_end_parameters_flow` (blocking queue receive).
  Its log is `.cache/sensor-pr6-full-unlimited.log`; it has no complete final summary.
  Isolated small-parameter case passed; isolated large case also timed out, without test changes.
  Logs: `.cache/sensor-full-rl-small.log` and `.cache/sensor-full-rl-large.log`.
- All other cases are now running with `-k 'not test_end_to_end_parameters_flow'`, no failure
  cutoff, log `.cache/sensor-pr6-full-without-hanging-rl.log`. Keep the two isolated cases
  separate in the final accounting; do not claim an all-green full suite. Known failures so far
  include unchanged OpenCV tests receiving four-channel PNG images. The detailed final failure
  list is still pending. Live exec handles at this checkpoint: full pytest **34032**, default
  benchmark **39038**. Verify these handles before starting replacement runs.
- Ruff/format pass for 41 changed/new Python files; `git diff --check` passes. Final
  requirement-by-requirement audit and PR delivery remain open.

Do not advance a stage until its regression and lint/format/diff gates pass.

## Final audit checkpoint — 2026-09-06 14:10 CST

- Previous handles 34032 and 39038 are now absent, and no Python/uv process remained
  at 14:06. Their logs stop at 98% of pytest and after benchmark capture, respectively;
  neither is a completed validation result. Do not reuse their unfinished output as a pass.
- Candidate reduction now conservatively omits packed grids outside the candidate chunk's
  possible availability/expiration interval. The complete per-candidate predicate remains
  unchanged. Python integer expiration bounds avoid int64 overflow. The 1000-case oracle,
  shuffled duplicate grids at the int64 limit, batch/spawn and benchmark smoke tests pass:
  **1028 passed** (`.cache/sensor-pr6-pruning`), plus Ruff and format checks.
- Benchmark validates arguments before capture and saves each completed measurement mode.
  Fresh default workload: exec **84448**, `.cache/sensor-benchmark-pruned.log`, output root
  `.cache/sensor-benchmark-pruned`. Full pytest: exec **91058**, verbose per-test log
  `.cache/sensor-pr6-full-audit.log`, requested final JUnit `.cache/sensor-pr6-full-audit.xml`.
  Only the two previously isolated RL queue-flow cases are excluded from this run.
- A 128-frame profile on the complete earlier 30-minute capture confirms bounded cache
  residency (66,901,392 / 67,108,864 decoded bytes). Footer scans and row-group processing
  dominate after candidate pruning; this small profile is not the default benchmark result.

Requirement evidence map (test assertions were inspected; final run outcomes remain pending):

| Requirement | Current source and direct checks |
|---|---|
| Arrival ordering, unseen valid reads, recorder sequence ownership | `sensors/sensor.py`, `buffer.py`; `test_core_correctness.py`, X518 reconnect tests |
| Required raw-only startup, optional nonblocking, shared record/rollout boundaries | `sensor_stream.py`, `sensorized_robot.py`, record/rollout adapters; `test_sensor_spool.py` startup and causal-boundary checks |
| Bounded Raw and Sync, disk fragment/reference index, timed flush | `sensor_spool.py`; 256/8192-row tests inspect both buffer peaks, SQLite reference counts and final footer row counts |
| Shutdown ownership, fatal overflow, trim and exact Sync references | Recorder lifecycle and merge; stream/spool tests inject join timeout, overflow, corrupted references and late rows |
| Fixed pointer, RECORDING ordering, recovery state machine | `sensor_transaction.py`; subprocess exit boundaries, discover-before-subscribe, conflicting staging and three-repeat recovery snapshots |
| Main data/video/metadata seal, append-stable logical evidence | Writer/metadata hooks; real writer next-episode recovery, shared append and video sealing tests |
| No unrelated history scan or shared-main whole-file hash | Metered opens/read bytes/hash calls compare zero versus 100 unrelated historical groups; snapshot scanning must not run |
| Fast/full read-only verification and live/shared conflict refusal | `sensor_verification.py`, Reader; full tree/mtime/content comparison, dead locks, new live writer, selected/unselected and shared-artifact corruption tests |
| Range pruning, exact integer grids, full candidate predicate | Selection module; missing footer statistics, unordered measurement and 1000 independent oracle cases compare all six fields |
| Worker byte LRU, no handles in pickle, PID reset | `sensor_window_cache.py`; eviction, oversized and dictionary-expanded groups, pickle/PID tests |
| Explicit batch, order/duplicates/cross-episode, split-first factory | Wrapper/factory; union I/O counts, independent duplicate arrays, actual Windows spawn with two shuffled workers |
| Hub selected closure, pinned revision/token/cache, cached-main path | `sensor_hub.py` and Dataset localization; fake Hub requests inspect exact paths and reject bad selected journals before large transfers |
| Runtime-only metrics, warning/overflow/commit summary | `diagnostics.py`, spool/Recorder; threshold/rate limit, gaps/lag/flush/compression and manifest invariance tests |
| Three Sensor docs and AGENTS | Updated files exist; local Markdown targets and Python fenced examples validated. Three original untracked docs still need Git inclusion |
| Full regression and six reviewable PR deliveries | Final pytest/benchmark results pending; no commits or PRs have been created. Delivery preference requested |

- Delivery preference is now confirmed: **six local commits**, no GitHub publication.
- Two ordinary Dataset failures were reproduced unchanged against the tree now published as
  `36bb61cd`:
  POSIX-only `"data/" in str(Path)` assertion and deleting an open decoded video on Windows.
  Baseline source was exported read-only from Git to `.cache/sensor-baseline-audit/src` and
  imported via PYTHONPATH (module path checked). Logs: `.cache/sensor-pr6-head-failures.log`
  and `.cache/sensor-pr6-audit-first-failures.log`. Do not attribute other failures to baseline
  until their own evidence is collected.
- Four existing Hub unit tests need adjustment to the new intended contract:
  `test_metadata_without_root_uses_hub_cache_snapshot_download` still expects `meta/`;
  its exact pattern assertion must use the newly restricted metadata paths while retaining
  revision/cache/token checks. Three `test_data_download_forwards_token` cases construct
  an incomplete `__new__` stub without `root`; supply a distinct metadata root and retain
  assertions that download propagates the returned snapshot root to Dataset/meta/reader.
  Reproduction log: `.cache/sensor-pr6-audit-hub-existing.log`. Changes pending until the
  running subprocess crash matrix is complete (source remains frozen).
- Current default benchmark capture completed in 193.50 seconds with 54,000 frames and
  1,800,000 Raw rows per stream. Incremental report exists at
  `.cache/sensor-benchmark-pruned/benchmark_results.json`; reading modes remain in progress.

### Active checkpoint — 2026-09-06 14:24 CST

- Local delivery branch created: `codex/sensor-production`; HEAD is still `36bb61cd`.
  No commits yet. Six local commits are explicitly authorized; do not publish GitHub PRs.
- The four existing Hub tests were updated as described above without removing coverage.
  Full `test_lerobot_dataset.py` plus `test_sensor_hub.py`: **50 passed, 1 skipped**
  (`.cache/sensor-pr6-hub-regression-audit.log`). All 42 changed/new Python files pass Ruff;
  the one new mixed-line-ending format issue was corrected and rechecked. Diff check passed.
- `HF_LEROBOT_HOME=.cache/hf-lerobot-audit` is also required for full-suite runs, besides
  UV_CACHE_DIR and HF_DATASETS_CACHE. With this workspace cache, all Dataset factory cases
  plus forward-slash feature validation pass: **13 passed** in 114.46 seconds
  (`.cache/sensor-pr6-audit-factory-cache.log`). This explains the earlier cache-permission failures.
- Full run 91058 reached 98%, then hung in `test_async_iterator_shapes_basic` at RL
  `buffer.py:372` queue.get. A 15-second isolated run reproduces the hang both in current
  source and original HEAD (logs `.cache/sensor-pr6-audit-async-hang.log` and
  `.cache/sensor-pr6-head-async.log`). The dead producer's exception is swallowed by unchanged
  RL code, leaving the consumer waiting. The live full-test processes were identified by exact
  command line and explicitly terminated; benchmark was retained. Stop-Process itself errored,
  taskkill succeeded and exec 91058 then returned exit 1. It has no final summary.
- Fresh full run **46499** excludes the two RL parameter-flow cases and the two async iterator
  cases. Log `.cache/sensor-pr6-full-completable.log`, final JUnit requested at
  `.cache/sensor-pr6-full-completable.xml`. A workspace-only pytest plugin writes each outcome
  and failure immediately to `.cache/sensor-pr6-full-completable.jsonl`; do not add it to Git.
  This run includes the corrected Hub assertions and writable HF cache.
- Second async-iterator case is being tested separately in current source (exec **78701**,
  `.cache/sensor-pr6-audit-async-multiple.log`) and original HEAD (exec **14322**,
  `.cache/sensor-pr6-head-async-multiple.log`), each with 15-second timeout. Close their handles
  and inspect results. Earlier isolated RL small parameter case passed; large timed out.
- Default benchmark **84448** is still live, reading sequential mode; report currently contains
  capture only. Do not restart solely due lack of per-mode output. Its full capture has no
  overflow/gaps, Raw peak buffers 4096 each, Sync peak buffer 430; decoded-cache profile remains
  bounded. Performance profile indicates repeatedly scanning all footer statistics per requested
  range is now a major batch cost; consider one-pass row-group union pruning after the current
  test subprocesses finish. The non-CI 5x speed target remains unverified.

### Delivery audit — 2026-09-06 (supersedes pending checkpoints above)

- Full-suite run completed: **3,851 passed, 43 failed, 313 skipped, 4 deselected**.
  Six Torch-cache failures pass with workspace TORCH_HOME. All remaining 37 failures
  were reproduced against the tree now published as `36bb61cd`. Large RL flow and both
  async iterator hangs were also reproduced against original source; small RL flow passes.
  See `SENSOR_VALIDATION_REPORT.md` for categorized failures and exact evidence logs.
- Final footer union pruning scans each row-group's statistics once per batch range union,
  including absent statistics and unordered measurements. A new failure regression verifies
  that diagnostic footer errors do not reinsert already indexed spool rows during close.
  The completed-fragment count now becomes visible only after footer access finishes.
- Independent delivery snapshots: stage 1 **118 passed**; stage 2 **251 passed** plus
  the added footer-failure regression; stage 3 has **346** distinct passing cases after
  correcting two copied predecessor-journal expectations and rerunning all **24** spool tests;
  stage 4 **1,385 passed**; stage 5 **1,412 passed**. Final stage 6 combined gate is
  running as exec **47926**, `.cache/sensor-production-final.log` and matching JUnit XML.
- The full 30-minute capture is complete: two streams of 1.8 million rows and 54,000
  Sync/main frames, zero overflow/gaps. The old all-frame reading run was explicitly
  stopped; it is not a completed benchmark result. A fixed-seed 1,024-frame reading sample
  on that complete capture measures sequential **34.63 fps**, random **30.01 fps**, batch
  **101.84 fps**: **3.39x**, below the non-CI 5x target. Random/batch valid-point totals match;
  resident decoded cache is **66,901,392 / 67,108,864 bytes**. Full sampled JSON is
  `.cache/sensor-benchmark-pruned/benchmark_sampled_results.json`.
- Stage 1 commit exists: `a7cde76a`. Remaining Git staging attempts were rejected twice by
  automatic approval review because its service returned HTTP 503 (no available model
  channel), not a code-risk finding. The worktree remains intact and index empty.
  Six review patches are exported to `.cache/sensor-review-patches/`; stages 2–6 pass
  `git apply --check` against their preceding isolated snapshots. No GitHub PR was published.
- All 42 final changed/new Python files pass Ruff and format checks; diff whitespace passes.
  `SENSOR_VALIDATION_REPORT.md` is the concise final validation/delivery reference.

### Final local delivery — 2026-09-06 20:05 CST

- Final combined gate completed: **1,479 passed, 5 platform skips** in 195.14 seconds,
  `.cache/sensor-production-final.log` and `.cache/sensor-production-final.xml`.
- The automatic approval service recovered. Local commits now exist for stages 1–5:
  `a7cde76a`, `21c79b7b`, `15bfa5ea`, `bd24c1f5`, `df8af560`. Every staged tree was
  compared with the corresponding independent snapshot before committing; the final worktree
  was preserved. Stage 2 was reverified as **252 passed** and the stage 3 spool suite as
  **24 passed** after the copied journal-version assertions were corrected.
- Stage 6 includes all remaining implementation and the three originally untracked Sensor
  documents, AGENTS changes, this progress log and `SENSOR_VALIDATION_REPORT.md`.
  No GitHub publication is requested. The final commit carries this report; its completion
  audit checks six commits and a clean working tree. No additional product implementation
  is pending.
- The full-suite accounting and 1,024-frame benchmark measurements above are final. The
  benchmark used the complete 30-minute capture; all-frame reading was not completed and
  the 5x non-CI target was not reached. Preserve those limits in the final user-facing report.
