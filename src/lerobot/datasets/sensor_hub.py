#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Localize only a selected Sensor episode closure before constructing a Reader."""

import json
import re
from pathlib import Path

from huggingface_hub import HfApi

from .sensor_stream import validate_sidecar_schema_version
from .sensor_transaction import SensorTransaction, SensorTransactionError, sha256_file, validate_episode_uid
from .sensor_verification import resolve_artifact

# Main episode metadata supports normal Dataset selection. Sensor identity JSONs are
# discovery records, not an authoritative episode-index-to-UUID index.
METADATA_DOWNLOAD_PATTERNS = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/**/*.parquet",
    "meta/sensor_streams.json",
    "meta/sensor_episodes/*.json",
    "meta/teacher_capture.json",
]


def pin_sensor_revision(root, repo_id, revision, *, token=None):
    """Pin a Hub Sensor download chain; local complete Datasets need no Hub call."""
    root = Path(root)
    if root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", root.name):
        return root.name
    if re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        return revision
    kwargs = {} if token is None else {"token": token}
    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision, **kwargs).sha


def _exact_path(root, relative):
    if not isinstance(relative, str) or any(char in relative for char in "*?[]\\"):
        raise SensorTransactionError("Sensor Hub artifact locator must be an exact POSIX path.")
    path = resolve_artifact(root.resolve(), relative)
    if Path(relative).is_absolute() or path.relative_to(root.resolve()).as_posix() != relative:
        raise SensorTransactionError("Sensor Hub artifact locator must be relative and canonical.")
    return path


def ensure_sensor_subset(
    root, episodes, total_episodes, download, *, refresh=False, main_paths=(), teacher_required=False
):
    """Discover small identities, validate selected journals, then fetch exact closure.

    ``download`` receives exact POSIX relative paths (except identity discovery). It
    must preserve the caller's pinned revision, token and cache destination. This is
    a Dataset download operation; the Sensor Reader itself performs no localization.
    """
    root = Path(root).resolve()
    manifest_path = root / "meta/sensor_streams.json"
    if not manifest_path.exists():
        if teacher_required or (root / "meta/teacher_capture.json").exists():
            raise SensorTransactionError("Teacher Dataset is missing its Sensor manifest.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_sidecar_schema_version(manifest)
    layout = manifest.get("storage_layout", {})
    if layout.get("type") != "per_episode_parquet" or layout.get("version") != 1:
        raise ValueError("Unsupported Sensor Hub storage layout.")
    wanted = set(episodes if episodes is not None else range(total_episodes))
    if not wanted:
        return
    template = layout["episode_metadata_path_template"]
    # Validate the template before issuing a wildcard discovery request.
    probe_uid = "00000000-0000-4000-8000-000000000000"
    _exact_path(root, template.format(episode_uid=probe_uid))
    pattern = template.replace("{episode_uid}", "*")

    def discover():
        found = {}
        for path in root.glob(pattern):
            identity = json.loads(path.read_text(encoding="utf-8"))
            index = identity.get("episode_index")
            if index not in wanted:
                continue
            uid = validate_episode_uid(identity.get("episode_uid"))
            if path != _exact_path(root, template.format(episode_uid=uid)):
                raise SensorTransactionError("Sensor identity file path and UID disagree.")
            if index in found:
                raise SensorTransactionError("Duplicate Sensor episode identity.")
            found[index] = uid
        return found

    identities = discover()
    if refresh or set(identities) != wanted:
        download([pattern])
        identities = discover()
    if set(identities) != wanted:
        raise SensorTransactionError("Selected Sensor episode identities are missing.")
    journals = [f"meta/sensor_transactions/{uid}.json" for uid in identities.values()]
    missing = [name for name in journals if refresh or not (root / name).is_file()]
    if missing:
        download(missing)
    closure = {Path(path).as_posix() for path in main_paths}
    teacher_episodes = {}
    for index, uid in identities.items():
        transaction = SensorTransaction.load(root / f"meta/sensor_transactions/{uid}.json")
        journal = transaction.journal
        if transaction.state != "COMMITTED" or transaction.episode_uid != uid:
            raise SensorTransactionError(
                "Selected Hub Sensor journal is not COMMITTED or has a different UID."
            )
        expected = journal["expected_main"]
        if expected["episode_index"] != index:
            raise SensorTransactionError("Selected Hub Sensor journal episode index mismatch.")
        teacher_episodes[uid] = expected
        required = {template.format(episode_uid=uid), layout["sync_path_template"].format(episode_uid=uid)}
        required.update(
            layout["raw_path_template"].format(instance=name, episode_uid=uid) for name in manifest["streams"]
        )
        recorded = {item["final_path"] for item in journal["files"]}
        if recorded != required or len(recorded) != len(journal["files"]):
            raise SensorTransactionError(
                "Selected Sensor journal does not describe its complete Sidecar closure."
            )
        closure.update(required)
        evidence = journal.get("main_evidence")
        if not evidence or evidence.get("version") != 1:
            raise SensorTransactionError("Selected Hub Sensor journal lacks supported main evidence.")
        closure.update(item["path"] for item in journal["main_artifacts"] if item["role"] != "temporary")
    teacher_digests = {}
    if teacher_required or (root / "meta/teacher_capture.json").exists():
        teacher_digests = _teacher_closure(root, teacher_episodes, download, refresh=refresh)
        closure.update(teacher_digests)
    for relative in closure:
        _exact_path(root, relative)
    missing = sorted(name for name in closure if refresh or not (root / name).is_file())
    if missing:
        download(missing)
    if any(not (root / name).is_file() for name in closure):
        raise SensorTransactionError("Selected Sensor Hub artifact closure is incomplete after download.")
    for relative, expected_digest in teacher_digests.items():
        if sha256_file(root / relative) != expected_digest:
            raise SensorTransactionError(f"Teacher audit attachment digest mismatch: {relative}.")


def _teacher_closure(root, episodes, download, *, refresh):
    """Resolve selected Teacher attachments through committed Sensor identities.

    Fetch small manifests before trusting any audit attachment path. This also
    fills caches made by older loaders without fetching unselected episode logs.
    """
    metadata_paths = ["meta/teacher_capture.json"]
    metadata_paths.extend(f"meta/teacher_episodes/{uid}.json" for uid in episodes)
    missing = [name for name in metadata_paths if refresh or not _exact_path(root, name).is_file()]
    if missing:
        download(missing)
    if any(not (root / name).is_file() for name in metadata_paths):
        raise SensorTransactionError("Teacher audit metadata is incomplete after download.")
    contract = json.loads((root / "meta/teacher_capture.json").read_text(encoding="utf-8"))
    if contract.get("schema") != "teacher_capture_contract_v1":
        raise SensorTransactionError("Unsupported Teacher capture contract.")
    digests = {}
    for uid, expected in episodes.items():
        metadata = json.loads((root / f"meta/teacher_episodes/{uid}.json").read_text(encoding="utf-8"))
        if (
            metadata.get("schema") != contract.get("episode_schema")
            or metadata.get("schema") not in ("teacher_demo_episode_v1", "teacher_capture_episode_v1")
            or metadata.get("episode_uid") != uid
            or type(metadata.get("episode_index")) is not int
            or metadata["episode_index"] != expected["episode_index"]
            or type(metadata.get("frame_count")) is not int
            or metadata["frame_count"] != expected["frame_count"]
            or metadata.get("complete") is not True
            or metadata.get("status") != "complete"
        ):
            raise SensorTransactionError("Teacher audit metadata disagrees with its committed episode.")
        records = metadata.get("files")
        if not isinstance(records, dict) or set(records) != {"control", "frames"}:
            raise SensorTransactionError("Teacher audit metadata lacks its complete attachment closure.")
        for name, record in records.items():
            relative = f"teacher/{uid}/{name}.jsonl"
            if (
                not isinstance(record, dict)
                or record.get("path") != relative
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record.get("rows")) is not int
                or record["rows"] < expected["frame_count"]
                or (name == "frames" and record["rows"] != expected["frame_count"])
            ):
                raise SensorTransactionError("Invalid Teacher audit attachment path, digest or row count.")
            _exact_path(root, relative)
            digests[relative] = record["sha256"]
    return digests
