#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects
from test_sensor_window_batch import recorded_dataset as recorded_dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_hub import METADATA_DOWNLOAD_PATTERNS, ensure_sensor_subset, pin_sensor_revision
from lerobot.datasets.sensor_transaction import SensorTransactionError
from lerobot.datasets.sensor_verification import resolve_artifact
from lerobot.datasets.sensor_window import SensorStreamReader

COMMIT = "a" * 40


def copy_paths(source, destination, names):
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)


def all_paths(root):
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]


def uid_for(root, index):
    return next(
        json.loads(path.read_text())["episode_uid"]
        for path in (root / "meta/sensor_episodes").glob("*.json")
        if json.loads(path.read_text())["episode_index"] == index
    )


@pytest.mark.parametrize("cached_main", [False, True])
@pytest.mark.parametrize("explicit_root", [False, True])
def test_subset_downloads_only_selected_closure_with_same_revision_token_and_destination(
    recorded_dataset, tmp_path, monkeypatch, cached_main, explicit_root
):
    source = recorded_dataset.root
    cache = tmp_path / "hub-cache"
    destination = (
        tmp_path / "local" if explicit_root else cache / "datasets--test--sensor-batch/snapshots" / COMMIT
    )
    uid = uid_for(source, 1)
    calls = []
    if cached_main:
        names = [
            name
            for name in all_paths(source)
            if name.startswith(("data/", "meta/")) and not name.startswith("meta/sensor_transactions/")
        ]
        copy_paths(source, destination, names)

    def snapshot(repo_id, *, revision, allow_patterns, **kwargs):
        assert repo_id == recorded_dataset.repo_id
        assert revision == COMMIT
        assert kwargs["token"] == "test-token"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs.get("local_dir") == (destination if explicit_root else None)
        assert kwargs.get("cache_dir") == (None if explicit_root else cache)
        names = list(
            filter_repo_objects(
                all_paths(source),
                allow_patterns=allow_patterns,
                ignore_patterns=kwargs.get("ignore_patterns"),
            )
        )
        calls.append((allow_patterns, names))
        copy_paths(source, destination, names)
        return str(destination)

    for module in ("dataset_metadata", "lerobot_dataset"):
        monkeypatch.setattr(f"lerobot.datasets.{module}.snapshot_download", snapshot)
        monkeypatch.setattr(f"lerobot.datasets.{module}.HF_LEROBOT_HUB_CACHE", cache)
    monkeypatch.setattr("lerobot.datasets.dataset_metadata.HF_LEROBOT_HOME", tmp_path / "home")
    dataset = LeRobotDataset(
        recorded_dataset.repo_id,
        root=destination if explicit_root else None,
        revision=COMMIT,
        episodes=[1],
        token="test-token",
    )
    reader = SensorStreamReader(dataset.root, episodes=[1], verify="full")
    assert reader.committed_episode_indices == (1,)
    downloaded = [name for _, names in calls for name in names]
    assert {name for name in downloaded if name.startswith("meta/sensor_transactions/")} == {
        f"meta/sensor_transactions/{uid}.json"
    }
    assert {Path(name).stem for name in downloaded if name.startswith(("raw/", "sync/"))} == {uid}
    assert any(name.startswith("raw/") for name in downloaded)
    if cached_main:
        assert not any(name.startswith("data/") for name in downloaded)
    assert all(patterns is not None and patterns != "meta/" for patterns, _ in calls)
    assert not hasattr(dataset, "token")


def test_unselected_quarantine_is_not_downloaded_or_repaired(recorded_dataset, tmp_path):
    source, destination = recorded_dataset.root, tmp_path / "subset"
    uid = uid_for(source, 2)
    path = source / f"meta/sensor_transactions/{uid}.json"
    payload = json.loads(path.read_text())
    payload["state"] = "QUARANTINED"
    path.write_text(json.dumps(payload))
    copy_paths(
        source,
        destination,
        list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS)),
    )
    downloaded = []

    def fetch(names):
        downloaded.extend(names)
        copy_paths(source, destination, names)

    ensure_sensor_subset(destination, [0], 3, fetch)
    assert not (destination / f"meta/sensor_transactions/{uid}.json").exists()
    assert json.loads(path.read_text())["state"] == "QUARANTINED"
    SensorStreamReader(destination, episodes=[0], verify="full")


@pytest.mark.parametrize("damage", ["state", "index", "closure", "escape", "wildcard"])
def test_invalid_selected_journal_rejected_before_any_raw_download(recorded_dataset, tmp_path, damage):
    source, destination = recorded_dataset.root, tmp_path / "subset"
    uid = uid_for(source, 0)
    path = source / f"meta/sensor_transactions/{uid}.json"
    journal = json.loads(path.read_text())
    if damage == "state":
        journal["state"] = "PREPARED"
    elif damage == "index":
        journal["expected_main"]["episode_index"] = 1
    elif damage == "closure":
        journal["files"].pop()
    else:
        journal["main_artifacts"].append(
            {"path": "../escape" if damage == "escape" else "data/*", "role": "data"}
        )
    path.write_text(json.dumps(journal))
    copy_paths(
        source,
        destination,
        list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS)),
    )
    downloaded = []

    def fetch(names):
        downloaded.extend(names)
        copy_paths(source, destination, names)

    with pytest.raises(SensorTransactionError):
        ensure_sensor_subset(destination, [0], 3, fetch)
    assert downloaded == [f"meta/sensor_transactions/{uid}.json"]


def test_complete_local_subset_needs_no_hub_requests(recorded_dataset):
    ensure_sensor_subset(recorded_dataset.root, [2], 3, lambda *_a: pytest.fail("unnecessary Hub download"))


def test_missing_download_is_an_explicit_error(recorded_dataset, tmp_path):
    source, destination = recorded_dataset.root, tmp_path / "subset"
    uid = uid_for(source, 0)
    names = list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS))
    names.append(f"meta/sensor_transactions/{uid}.json")
    copy_paths(source, destination, names)
    with pytest.raises(SensorTransactionError, match="incomplete after download"):
        ensure_sensor_subset(destination, [0], 3, lambda *_a: None)


def test_dataset_rejects_selected_unfinished_journal_before_large_downloads(
    recorded_dataset, tmp_path, monkeypatch
):
    source, destination = recorded_dataset.root, tmp_path / "new-root"
    uid = uid_for(source, 1)
    path = source / f"meta/sensor_transactions/{uid}.json"
    journal = json.loads(path.read_text())
    journal["state"] = "QUARANTINED"
    path.write_text(json.dumps(journal))
    downloaded = []

    def snapshot(_repo_id, *, allow_patterns, **_kwargs):
        names = list(filter_repo_objects(all_paths(source), allow_patterns=allow_patterns))
        downloaded.extend(names)
        copy_paths(source, destination, names)
        return str(destination)

    monkeypatch.setattr("lerobot.datasets.dataset_metadata.snapshot_download", snapshot)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.snapshot_download", snapshot)
    with pytest.raises(SensorTransactionError, match="not COMMITTED"):
        LeRobotDataset(recorded_dataset.repo_id, root=destination, episodes=[1], revision=COMMIT)
    assert not any(name.startswith(("raw/", "sync/", "data/", "videos/")) for name in downloaded)


def test_unsupported_journal_version_is_rejected_before_large_downloads(recorded_dataset, tmp_path):
    source, destination = recorded_dataset.root, tmp_path / "unsupported-journal"
    uid = uid_for(source, 1)
    path = source / f"meta/sensor_transactions/{uid}.json"
    journal = json.loads(path.read_text())
    journal["journal_version"] = 2
    path.write_text(json.dumps(journal))
    copy_paths(
        source,
        destination,
        list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS)),
    )
    downloaded = []

    def fetch(names):
        downloaded.extend(names)
        copy_paths(source, destination, names)

    with pytest.raises(SensorTransactionError, match="Unsupported sensor journal version"):
        ensure_sensor_subset(destination, [1], 3, fetch)
    assert downloaded == [f"meta/sensor_transactions/{uid}.json"]


def test_branch_downloads_pin_before_journals_and_large_artifacts(recorded_dataset, tmp_path, monkeypatch):
    source, destination = recorded_dataset.root, tmp_path / "branch"
    revisions = []

    def snapshot(_repo_id, *, allow_patterns, revision, **kwargs):
        assert kwargs["token"] is False
        revisions.append((revision, allow_patterns))
        names = list(filter_repo_objects(all_paths(source), allow_patterns=allow_patterns))
        copy_paths(source, destination, names)
        return str(destination)

    monkeypatch.setattr("lerobot.datasets.dataset_metadata.snapshot_download", snapshot)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.snapshot_download", snapshot)
    monkeypatch.setattr(
        "lerobot.datasets.sensor_hub.HfApi",
        lambda: SimpleNamespace(repo_info=lambda *_a, **_k: SimpleNamespace(sha=COMMIT)),
    )
    dataset = LeRobotDataset(
        recorded_dataset.repo_id, root=destination, episodes=[1], revision="main", token=False
    )
    assert revisions[0] == ("main", METADATA_DOWNLOAD_PATTERNS)
    assert all(revision == COMMIT for revision, _ in revisions[1:])
    assert dataset.revision == dataset.meta.revision == COMMIT


def test_revision_resolution_preserves_authentication_and_snapshot_identity(tmp_path, monkeypatch):
    calls = []

    class Api:
        def repo_info(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(sha=COMMIT)

    monkeypatch.setattr("lerobot.datasets.sensor_hub.HfApi", Api)
    assert pin_sensor_revision(tmp_path, "test/repo", "main", token=False) == COMMIT
    assert calls == [(("test/repo",), {"repo_type": "dataset", "revision": "main", "token": False})]
    calls.clear()
    assert pin_sensor_revision(tmp_path / "snapshots" / COMMIT, "test/repo", "main") == COMMIT
    assert pin_sensor_revision(tmp_path, "test/repo", COMMIT) == COMMIT
    assert not calls


def test_readonly_snapshot_blob_links_are_supported_without_allowing_other_escapes(
    recorded_dataset, tmp_path
):
    source = recorded_dataset.root
    repo = tmp_path / "datasets--test--sensor-batch"
    snapshot = repo / "snapshots" / COMMIT
    for index, name in enumerate(all_paths(source)):
        blob = repo / "blobs" / str(index)
        blob.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, blob)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(blob)
        except OSError as exc:
            pytest.skip(f"File symlink creation unavailable on this Windows host: {exc}")
    SensorStreamReader(snapshot, episodes=[1], verify="full")
    outside = tmp_path / "outside"
    outside.write_text("not a cache blob")
    (snapshot / "escape").symlink_to(outside)
    with pytest.raises(SensorTransactionError, match="escapes"):
        resolve_artifact(snapshot, "escape")
