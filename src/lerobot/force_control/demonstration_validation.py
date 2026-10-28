"""Read-only verification of unified FORCE_POS demonstration evidence.

File integrity, observed control stages and the operator's task result are
reported independently. Verification never recovers a transaction or opens a
Dataset writer, and it never accesses hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

if TYPE_CHECKING or (_datasets_available and _av_available):
    import pyarrow.parquet as pq

    from lerobot.datasets.sensor_transaction import SensorTransaction, _episode_rows
    from lerobot.datasets.sensor_verification import check_no_live_writer, resolve_artifact
    from lerobot.datasets.sensor_window import SensorStreamReader

if TYPE_CHECKING or _av_available:
    import av


def _json_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL row at {path.name}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path.name}:{line_number}")
            yield row


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _vector(value: Any, expected: Any, size: int) -> bool:
    try:
        actual = np.asarray(value, dtype=np.float32)
        reference = np.asarray(expected, dtype=np.float32)
        return (
            actual.shape == reference.shape == (size,)
            and np.isfinite(actual).all()
            and np.isfinite(reference).all()
            and np.array_equal(actual, reference)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _time(value: Any) -> bool:
    return type(value) is int and value >= 0


def _causal(timestamp: Any, arrival: Any, anchor: Any) -> bool:
    return all(_time(value) for value in (timestamp, arrival, anchor)) and max(timestamp, arrival) <= anchor


def _decode_video(path: Path, start: float, end: float, frame_count: int, fps: float) -> None:
    """Decode the selected episode range, including shared video containers."""
    decoded = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video has no video stream: {path.name}")
        stream = container.streams.video[0]
        # Decoding from the beginning also proves earlier packets cannot conceal
        # corruption. A shared file's later episode remains outside this range.
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video frame has no presentation timestamp: {path.name}")
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 0.25 / fps < start:
                continue
            if timestamp >= end - 0.25 / fps:
                break
            frame.to_ndarray(format="rgb24")
            decoded += 1
    if decoded != frame_count:
        raise ValueError(f"Video decoded {decoded} frames for {frame_count} main frames: {path.name}")


def _verify_episode(
    root: Path, manifest_path: Path, *, verify_video: bool, require_complete: bool
) -> dict[str, Any]:
    manifest_stamp = _stamp(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    uid = str(manifest.get("episode_uid"))
    parsed = uuid.UUID(uid)
    if parsed.version != 4 or str(parsed) != uid or manifest_path.stem != uid:
        raise ValueError("Teacher episode UID is not a matching canonical UUID4")
    if manifest.get("schema") not in ("teacher_demo_episode_v1", "teacher_capture_episode_v1"):
        raise ValueError("Unsupported Teacher demonstration schema")
    capture_version = manifest["schema"] == "teacher_capture_episode_v1"
    index, frame_count = manifest["episode_index"], manifest["frame_count"]
    if type(index) is not int or index < 0 or type(frame_count) is not int or frame_count <= 0:
        raise ValueError("Teacher episode index/frame count is invalid")
    if require_complete and manifest.get("complete") is not True:
        raise ValueError("Teacher episode is incomplete")
    if str(manifest.get("status", "")).lower() in {"interrupted", "discarded", "error", "failed"}:
        raise ValueError("Teacher episode has an unsuccessful recording status")
    task_result = manifest.get("task_result", "unknown")
    if task_result not in {"unknown", "success", "failure"}:
        raise ValueError("Teacher task_result is invalid")
    joint_names = manifest["joint_names"]
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 7
        or len(set(joint_names)) != 7
        or any(not isinstance(name, str) or not name for name in joint_names)
    ):
        raise ValueError("Teacher joint_names must contain seven unique names")

    paths, stamps = {}, {manifest_path: manifest_stamp}
    for name in ("control", "frames"):
        record = manifest["files"][name]
        expected = f"teacher/{uid}/{name}.jsonl"
        if record.get("path") != expected:
            raise ValueError(f"Teacher {name} path must be {expected}")
        path = resolve_artifact(root, record["path"])
        paths[name], stamps[path] = path, _stamp(path)
        if _digest(path) != record["sha256"]:
            raise ValueError(f"Teacher {name} digest mismatch")

    frames = list(_json_rows(paths["frames"]))
    if len(frames) != frame_count or manifest["files"]["frames"]["rows"] != frame_count:
        raise ValueError("Teacher frame row count mismatch")
    ticks: dict[int, dict] = {}
    previous_tick, previous_anchor = None, None
    for i, frame in enumerate(frames):
        tick, anchor = frame["control_tick_id"], frame["decision_anchor_ns"]
        if (
            frame.get("frame_index") != i
            or type(tick) is not int
            or tick < 0
            or not _time(anchor)
            or (previous_tick is not None and tick <= previous_tick)
            or (previous_anchor is not None and anchor <= previous_anchor)
        ):
            raise ValueError("Teacher frame index/tick/anchor is not strictly increasing")
        ticks[tick] = frame
        previous_tick, previous_anchor = tick, anchor

    stage_counts: Counter = Counter()
    force_label_count = action_label_count = control_count = 0
    previous_tick = None
    for row in _json_rows(paths["control"]):
        tick = row.get("tick_id")
        if type(tick) is not int or tick < 0 or (previous_tick is not None and tick != previous_tick + 1):
            raise ValueError("Teacher control timeline has missing or unordered ticks")
        previous_tick = tick
        stage_counts[row.get("capture_stage", "unknown")] += 1
        force_label_count += row.get("label_valid") is True
        action_label_count += row.get("action_label_valid") is True
        control_count += 1
        frame = ticks.pop(tick, None)
        if frame is None:
            continue
        if frame["decision_anchor_ns"] != row.get("decision_anchor_ns") or frame.get("epoch") != row.get(
            "epoch"
        ):
            raise ValueError("Teacher frame references a different control anchor/epoch")
        results = row.get("per_joint_results", [])
        by_name = {item["name"]: item for item in results}
        if set(by_name) != set(joint_names) or len(results) != 7 or row.get("send_status") != "submitted":
            raise ValueError("Teacher frame has no complete seven-joint submission receipt")
        if any(by_name[name].get("status") != "success" for name in joint_names):
            raise ValueError("Teacher frame references an unknown joint submission")
        submitted = [by_name[name].get("submitted_deg") for name in joint_names]
        if not _vector(frame["action_deg"], submitted, 7):
            raise ValueError("Teacher action differs from submitted per-joint receipt")
        snapshot = row.get("joint_snapshot", {})
        joints = snapshot.get("joints", [])
        by_name = {item["name"]: item for item in joints}
        if set(by_name) != set(joint_names) or len(joints) != 7:
            raise ValueError("Teacher frame has no complete joint feedback snapshot")
        if any(by_name[name].get("is_valid") is not True for name in joint_names) or not _vector(
            frame["state_deg"], [by_name[name].get("position_deg") for name in joint_names], 7
        ):
            raise ValueError("Teacher state differs from valid observed joint positions")
        for name in (
            "force_sequence",
            "force_timestamp_ns",
            "force_arrival_timestamp_ns",
            "send_start_ns",
            "send_end_ns",
        ):
            if frame.get(name) != row.get(name):
                raise ValueError(f"Teacher frame/control {name} mismatch")
        if not _causal(
            frame["force_timestamp_ns"],
            frame["force_arrival_timestamp_ns"],
            anchor := frame["decision_anchor_ns"],
        ):
            raise ValueError("Teacher frame force is noncausal")
        if not (
            _time(frame["send_start_ns"])
            and _time(frame["send_end_ns"])
            and anchor <= frame["send_start_ns"] <= frame["send_end_ns"]
        ):
            raise ValueError("Teacher frame send timestamps are not ordered")
        if not _vector(frame["tactile_N"], [row.get("force_left_N"), row.get("force_right_N")], 2):
            raise ValueError("Teacher tactile differs from selected control force")
        if capture_version and (
            frame.get("teacher_phase") != row.get("phase")
            or frame.get("action_valid") != (row.get("action_label_valid") is True)
            or not _vector([frame.get("force_target_N")], [row.get("force_target_effective_N")], 1)
        ):
            raise ValueError("Teacher target/phase/action validity differs from completed control tick")
        if any(
            not _causal(item.get("timestamp_ns"), item.get("arrival_timestamp_ns"), anchor)
            for item in frame.get("cameras", {}).values()
        ):
            raise ValueError("Teacher frame camera is noncausal")
    if ticks or control_count != manifest["files"]["control"]["rows"]:
        raise ValueError("Teacher control row count or frame tick reference mismatch")
    summary = manifest.get("capture_summary", {})
    if (
        summary.get("stage_counts") != dict(stage_counts)
        or summary.get("force_label_count") != force_label_count
        or summary.get("action_label_count") != action_label_count
    ):
        raise ValueError("Teacher capture summary differs from retained control rows")

    reader = SensorStreamReader(root, episode_uid=uid, verify="full", cache_mb=0)
    transaction = SensorTransaction.load(root / "meta/sensor_transactions" / f"{uid}.json")
    expected = transaction.journal["expected_main"]
    if expected["episode_index"] != index or expected["frame_count"] != frame_count:
        raise ValueError("Teacher identity differs from committed Sensor transaction")
    sources = reader.manifest.get("frame_view", {}).get("sources", [])
    if (
        len(sources) != 2
        or [source["feature"] for source in sources] != ["left.normal_force", "right.normal_force"]
        or sources[0]["instance"] != sources[1]["instance"]
    ):
        raise ValueError("Teacher demonstration requires the ordered Sidecar v2 two-force profile")
    instance = sources[0]["instance"]
    metadata = json.loads(
        resolve_artifact(
            root, reader.manifest["storage_layout"]["episode_metadata_path_template"].format(episode_uid=uid)
        ).read_text(encoding="utf-8")
    )
    max_age_ms = metadata["streams"][instance]["resolved_max_age_ms"]
    sync_rows = reader.read_sync(episode_uid=uid)
    raw_path = resolve_artifact(
        root,
        reader.manifest["storage_layout"]["raw_path_template"].format(instance=instance, episode_uid=uid),
    )
    required = {frame["force_sequence"] for frame in frames}
    raw_samples = {}
    with raw_path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if row["sequence"] in required:
                    raw_samples[row["sequence"]] = row
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if capture_version:
        from .capture_contract import capture_contract

        if json.loads((root / "meta/teacher_capture.json").read_text(encoding="utf-8")) != capture_contract():
            raise ValueError("Capture field contract is missing or incompatible")
        if info["fps"] != 50 or manifest["config"]["fps"] != 50:
            raise ValueError("Capture main data must declare 50 Hz")
    for key, vector in (("observation.state", joint_names), ("action", joint_names)):
        spec = info["features"][key]
        if spec.get("names") != [f"{name}.pos" for name in vector] or tuple(spec["shape"]) != (7,):
            raise ValueError(f"Main {key} joint order differs from Teacher metadata")
    tactile_spec = info["features"]["observation.tactile"]
    if (
        tactile_spec.get("dtype") != "float32"
        or tuple(tactile_spec.get("shape", ())) != (2,)
        or tactile_spec.get("names") != [source["qualified_name"] for source in sources]
    ):
        raise ValueError("Main tactile schema differs from ordered Sidecar frame sources")
    video_names = {
        key.removeprefix("observation.images.")
        for key, spec in info["features"].items()
        if spec.get("dtype") == "video"
    }
    if not video_names:
        raise ValueError("Teacher demonstration has no camera video features")
    data_path = resolve_artifact(root, transaction.journal["main_evidence"]["data_path"])
    with closing(_episode_rows(data_path, index)) as main_rows:
        for frame, sync in zip(frames, sync_rows, strict=True):
            main = next(main_rows, None)
            if main is None or main.get("frame_index") != frame["frame_index"]:
                raise ValueError("Main frame indices differ from Teacher metadata")
            if capture_version and not _vector(
                [main.get("observation.force_target")], [frame["force_target_N"]], 1
            ):
                raise ValueError("Main effective force target differs from Teacher frame")
            for key, expected_vector, width in (
                ("observation.state", frame["state_deg"], 7),
                ("action", frame["action_deg"], 7),
                ("observation.tactile", frame["tactile_N"], 2),
            ):
                if not _vector(main.get(key), expected_vector, width):
                    raise ValueError(f"Main {key} differs from Teacher frame")
            if set(frame.get("cameras", {})) != video_names:
                raise ValueError("Teacher camera references differ from main video features")
            ref, raw = sync["sensors"][instance], raw_samples.get(frame["force_sequence"])
            if (
                sync["frame_anchor_ns"] != frame["decision_anchor_ns"]
                or ref["sequence"] != frame["force_sequence"]
            ):
                raise ValueError("Teacher frame and Sensor Sync reference differ")
            if (
                raw is None
                or raw["is_valid"] is not True
                or not _vector(
                    frame["tactile_N"], [raw["values"][source["feature"]] for source in sources], 2
                )
            ):
                raise ValueError("Teacher tactile does not match selected Raw force")
            for field, teacher_field in (
                ("timestamp_ns", "force_timestamp_ns"),
                ("arrival_timestamp_ns", "force_arrival_timestamp_ns"),
            ):
                if ref[field] != frame[teacher_field] or raw[field] != frame[teacher_field]:
                    raise ValueError("Teacher force timing differs from Raw/Sync")
            if not _causal(raw["timestamp_ns"], raw["arrival_timestamp_ns"], frame["decision_anchor_ns"]):
                raise ValueError("Raw force selection is noncausal")
            age_ns = frame["decision_anchor_ns"] - raw["timestamp_ns"]
            if ref["age_ns"] != age_ns or (max_age_ms is not None and age_ns > max_age_ms * 1_000_000):
                raise ValueError("Selected Raw force is stale or its Sync age differs")
        if next(main_rows, None) is not None:
            raise ValueError("Main Dataset contains extra episode frames")
    if verify_video:
        fps = float(info["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Main Dataset FPS is invalid")
        for video in transaction.journal["main_evidence"]["videos"]:
            _decode_video(
                resolve_artifact(root, video["path"]),
                video["from_timestamp"],
                video["to_timestamp"],
                frame_count,
                fps,
            )
    if any(_stamp(path) != before for path, before in stamps.items()):
        raise ValueError("Teacher evidence changed during verification")
    # Public Reader entry rechecks all selected immutable and shared-artifact
    # stamps, including any change after its initial verification.
    reader.frame_anchor(index, 0)
    check_no_live_writer(root)
    return {
        "episode_uid": uid,
        "episode_index": index,
        "frame_count": frame_count,
        "valid": True,
        "files_complete": True,
        "task_result": task_result,
        "complete": manifest.get("complete") is True,
        "successful_demo": manifest.get("complete") is True and task_result == "success",
        "capture_summary": summary,
        "video_decoded": verify_video,
        "errors": [],
    }


def validate_demonstration(
    root: str | Path,
    episode_uid: str | None = None,
    *,
    verify_video: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate saved evidence without creating files, locks or caches.

    Writers may check a sealed ``verifying`` candidate with
    ``require_complete=False`` before publishing its complete marker. Ordinary
    callers retain the strict default that refuses incomplete episodes.
    """
    root = Path(root).resolve()
    result: dict[str, Any] = {"valid": False, "errors": [], "episodes": [], "excluded_episodes": []}
    try:
        require_package("datasets", extra="dataset")
        require_package("av", extra="dataset")
        check_no_live_writer(root)
        if episode_uid is not None:
            parsed = uuid.UUID(episode_uid)
            if parsed.version != 4 or str(parsed) != episode_uid:
                raise ValueError("episode_uid must be a canonical UUID4")
            paths = [root / "meta/teacher_episodes" / f"{episode_uid}.json"]
        else:
            paths = sorted((root / "meta/teacher_episodes").glob("*.json"))
        if not paths:
            raise ValueError("No Teacher demonstration episode metadata found")
        for path in paths:
            try:
                if episode_uid is None:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    uid = metadata.get("episode_uid")
                    if (
                        metadata.get("schema") in ("teacher_demo_episode_v1", "teacher_capture_episode_v1")
                        and metadata.get("complete") is False
                        and metadata.get("status") in {"discarded", "interrupted", "error"}
                        and uid == path.stem
                        and str(uuid.UUID(uid)) == uid
                        and uuid.UUID(uid).version == 4
                    ):
                        journal_path = root / "meta/sensor_transactions" / f"{uid}.json"
                        transaction = SensorTransaction.load(journal_path) if journal_path.exists() else None
                        if transaction is None or transaction.state == "ABORTED":
                            result["excluded_episodes"].append(
                                {
                                    "episode_uid": uid,
                                    "status": metadata["status"],
                                    "reason": "unsaved episode retained for audit",
                                }
                            )
                            continue
                episode = _verify_episode(
                    root, path, verify_video=verify_video, require_complete=require_complete
                )
            except Exception as exc:
                error = f"{path.stem}: {type(exc).__name__}: {exc}"
                episode = {
                    "episode_uid": path.stem,
                    "valid": False,
                    "files_complete": False,
                    "successful_demo": False,
                    "errors": [error],
                }
                result["errors"].append(error)
            result["episodes"].append(episode)
        if episode_uid is None:
            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            indices = [episode["episode_index"] for episode in result["episodes"] if episode["valid"]]
            if sorted(indices) != list(range(info["total_episodes"])):
                result["errors"].append(
                    "Main Dataset episode closure lacks exactly one valid complete Teacher attachment per episode"
                )
            if not result["episodes"]:
                result["errors"].append("No complete Teacher demonstrations found")
        check_no_live_writer(root)
        result["valid"] = not result["errors"]
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result
