"""Saved acquisition semantics, independent of training and window adapters."""

import hashlib
import platform
from pathlib import Path

from .demonstration import JOINT_NAMES


def capture_source() -> dict:
    """Fingerprint the actual local acquisition code, including uncommitted edits."""
    package = Path(__file__).resolve().parents[1]
    files = {}
    for directory in (
        "force_control",
        "robots/rebot_b601_follower",
        "teleoperators/rebot_102_leader",
        "sensors",
        "cameras",
        "datasets",
    ):
        for path in sorted((package / directory).rglob("*.py")):
            files[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("scripts/lerobot_force_teacher.py", "configs/video.py"):
        files[relative] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
    return {
        "base_commit": "d82f5d52a374e9c7ec63855cddf8eb240af4c932",
        "python": platform.python_version(),
        "files_sha256": files,
    }


def capture_contract() -> dict:
    """Return the stable contract required before appending to a capture root."""
    return {
        "schema": "teacher_capture_contract_v1",
        "base_commit": "d82f5d52a374e9c7ec63855cddf8eb240af4c932",
        "episode_schema": "teacher_capture_episode_v1",
        "sidecar_schema_version": 2,
        "storage_layout_version": 1,
        "control_hz": 50,
        "main_fps": 50,
        "raw_force_requested_hz": 400,
        "joint_order": list(JOINT_NAMES),
        "fields": {
            "observation.state": {"shape": [7], "unit": "deg", "meaning": "observed robot joints only"},
            "action": {
                "shape": [7],
                "unit": "deg",
                "meaning": "all seven targets successfully submitted in this control tick; not a hardware ACK",
            },
            "observation.tactile": {
                "shape": [2],
                "unit": "N",
                "order": ["left.normal_force", "right.normal_force"],
                "meaning": "one causal selection, identical to referenced Raw sample",
            },
            "observation.force_target": {
                "shape": [1],
                "unit": "N",
                "meaning": "effective target for mean of left and right forces",
            },
            "task": {"meaning": "operator task description stored by official task_index mapping"},
        },
        "timestamps": {
            "clock": "host monotonic perf_counter_ns, integer nanoseconds; do not compare across process restarts without session provenance",
            "main_timestamp": "nominal frame_index / 50, for Dataset/video playback only",
            "decision_anchor_ns": "actual observation/decision anchor in teacher frames and Sidecar Sync",
            "camera_timestamp_ns": "host VideoCapture read return; not sensor exposure time",
            "camera_arrival_timestamp_ns": "frame available in camera publication cache",
            "force_timestamp_ns": "host reception of full X518 response; not ADC conversion time",
            "force_arrival_timestamp_ns": "parent Sensor framework publication time",
            "send_start_ns/send_end_ns": "host software submission interval",
        },
        "annotations": {
            "teacher_phase": "completed control phase; full transitions retained in control.jsonl",
            "action_valid": "existing action_label_valid mask; false rows are retained",
            "task_result": "human unknown/success/failure, never inferred from file completeness",
        },
        "raw": "all received publications from independent bounded episode subscription, including invalid attempts; sequence gaps or overflow invalidate commit",
        "history": "Raw only; no persistent windows or force duplication in observation.state",
        "configuration": "per-episode config, provenance.run_config and Sensor episode metadata; actual camera bindings and measured rates retained",
        "completion": "verified main/video/Raw/Sync/Teacher files; frequency statistics and human outcome reported separately",
        "durability": "process-crash recovery, no power-loss durability guarantee",
    }
