This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot provides PyTorch-based robot policies, datasets, training, evaluation, and hardware control. See `README.md` and `CONTRIBUTING.md` for general project and contribution guidance.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install
git lfs pull                               # When LFS test artifacts are needed
```

## Key Commands

```bash
uv run pytest tests/<affected_module> -q             # Start with relevant tests
uv run pytest tests -svv --maxfail=10                # Full suite when warranted
uv run pre-commit run --all-files                    # Repository-wide lint/format checks
```

Run checks relevant to the change first and complete applicable CI requirements. Broaden to the full suite for cross-module changes or unresolved integration risks; this command list is not a requirement to run every check for every edit. Report targeted tests, full-suite tests, benchmarks, and real-hardware checks separately.

The Makefile E2E workflow requires a compatible Bash/WSL/Linux environment and its dependencies; do not run Bash assignment syntax directly in PowerShell:

```bash
DEVICE=cuda make test-end-to-end
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`sensors/`** — Non-visual physical sensor abstraction. Before any sensor-related design or implementation, read all of [`SENSOR_FRAMEWORK.md`](./SENSOR_FRAMEWORK.md) and follow it as the locked source of truth unless the user explicitly asks to revise it. If it has already been read in this task and is unchanged, reuse that context instead of rereading it.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`. Sensor unit tests must not require real hardware and use `tests/sensors/test_<sensor>.py`; real-hardware checks belong in `examples/<sensor>/<sensor>_hardware_smoke_test.py`, with an optional Windows launcher named `run_<sensor>_hardware_smoke_test.bat` in the same directory. Do not place temporary archives, logs, or captured sensor data under `src/`.
- **`.github/workflows/`** — Source of truth for current CI checks, test environments, and release workflows.
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **Sensor persistence**: follow [`SENSOR_DATASET_FORMAT.md`](./SENSOR_DATASET_FORMAT.md) and
  [`SENSOR_INTEGRATION_GUIDE.md`](./SENSOR_INTEGRATION_GUIDE.md). Preserve Sidecar v1 final schemas/layout;
  transaction journal format v1 guarantees process-crash recovery and replayable on-disk state, not
  power-loss durability.
  Reader paths are strictly read-only. Recovery belongs to a Writer/RecoveryManager holding the writer lock;
  recovery discovers only the active pointer and uncleaned staging, never historical journals. Keep runtime
  diagnostics out of stable manifests. Validate bounded Raw and Sync memory, subprocess recovery, read-only
  tree/mtime invariants, window oracle, Hub subset closure and Windows spawn/batch behavior. Use structural
  I/O assertions in CI; benchmark speedup is not a hard gate.
