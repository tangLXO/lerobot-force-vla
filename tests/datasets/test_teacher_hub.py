#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects
from test_sensor_hub import COMMIT, all_paths, copy_paths, uid_for
from test_sensor_window_batch import recorded_dataset as recorded_dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_hub import METADATA_DOWNLOAD_PATTERNS, ensure_sensor_subset
from lerobot.datasets.sensor_transaction import SensorTransactionError, sha256_file


@pytest.fixture
def teacher_dataset(request):
    dataset = request.getfixturevalue("recorded_dataset")
    root = dataset.root
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "rebot_b601_teacher"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/teacher_capture.json").write_text(
        json.dumps({"schema": "teacher_capture_contract_v1", "episode_schema": "teacher_capture_episode_v1"}),
        encoding="utf-8",
    )
    for index in range(3):
        uid = uid_for(root, index)
        records = {}
        for name in ("control", "frames"):
            path = root / f"teacher/{uid}/{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" * 4, encoding="utf-8")
            records[name] = {
                "path": path.relative_to(root).as_posix(),
                "rows": 4,
                "sha256": sha256_file(path),
            }
        metadata_path = root / f"meta/teacher_episodes/{uid}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": "teacher_capture_episode_v1",
                    "episode_uid": uid,
                    "episode_index": index,
                    "frame_count": 4,
                    "complete": True,
                    "status": "complete",
                    "files": records,
                }
            ),
            encoding="utf-8",
        )
    return dataset


@pytest.mark.parametrize("cached_main", [False, True])
@pytest.mark.parametrize("explicit_root", [False, True])
@pytest.mark.parametrize("episodes", [None, [1]])
def test_normal_loading_fetches_teacher_closure_including_old_caches(
    teacher_dataset, tmp_path, monkeypatch, cached_main, explicit_root, episodes
):
    source = teacher_dataset.root
    cache = tmp_path / "hub-cache"
    destination = (
        tmp_path / "download" if explicit_root else cache / "datasets--test--sensor-batch/snapshots" / COMMIT
    )
    if cached_main:
        # Reproduce the pre-fix loader: main + complete Sensor closure, no Teacher metadata/logs.
        names = [p for p in all_paths(source) if not p.startswith(("teacher/", "meta/teacher"))]
        copy_paths(source, destination, names)
    calls = []

    def snapshot(repo_id, *, revision, allow_patterns, **kwargs):
        assert repo_id == teacher_dataset.repo_id
        assert revision == COMMIT
        assert kwargs["token"] == "test-token"
        assert kwargs.get("local_dir") == (destination if explicit_root else None)
        assert kwargs.get("cache_dir") == (None if explicit_root else cache)
        names = list(filter_repo_objects(all_paths(source), allow_patterns=allow_patterns))
        calls.extend(names)
        copy_paths(source, destination, names)
        return str(destination)

    for module in ("dataset_metadata", "lerobot_dataset"):
        monkeypatch.setattr(f"lerobot.datasets.{module}.snapshot_download", snapshot)
        monkeypatch.setattr(f"lerobot.datasets.{module}.HF_LEROBOT_HUB_CACHE", cache)
    monkeypatch.setattr("lerobot.datasets.dataset_metadata.HF_LEROBOT_HOME", tmp_path / "home")
    dataset = LeRobotDataset(
        teacher_dataset.repo_id,
        root=destination if explicit_root else None,
        revision=COMMIT,
        episodes=episodes,
        token="test-token",
    )
    assert len(dataset) == (12 if episodes is None else 4)
    wanted_uids = {uid_for(source, i) for i in (range(3) if episodes is None else episodes)}
    assert {p.parent.name for p in (destination / "teacher").rglob("*.jsonl")} == wanted_uids
    assert {p.stem for p in (destination / "meta/teacher_episodes").glob("*.json")} == wanted_uids
    assert {p.split("/")[1] for p in calls if p.startswith("teacher/")} == wanted_uids
    assert (destination / "meta/teacher_capture.json").exists()
    for uid in wanted_uids:
        for name in ("control", "frames"):
            assert sha256_file(destination / f"teacher/{uid}/{name}.jsonl") == sha256_file(
                source / f"teacher/{uid}/{name}.jsonl"
            )


def test_complete_teacher_local_load_is_readonly_without_hub(teacher_dataset, monkeypatch):
    root = teacher_dataset.root
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    for module in ("dataset_metadata", "lerobot_dataset"):
        monkeypatch.setattr(
            f"lerobot.datasets.{module}.snapshot_download", lambda *_a, **_k: pytest.fail("Hub I/O")
        )
    monkeypatch.setattr("lerobot.datasets.sensor_hub.HfApi", lambda: pytest.fail("Hub I/O"))
    dataset = LeRobotDataset(teacher_dataset.repo_id, root=root)
    assert len(dataset) == 12
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "damage",
    [
        "uid",
        "index",
        "frames",
        "incomplete",
        "schema",
        "missing_record",
        "escape",
        "wildcard",
        "digest",
        "rows",
    ],
)
def test_teacher_manifest_damage_rejected_before_large_download(teacher_dataset, tmp_path, damage):
    source, destination = teacher_dataset.root, tmp_path / "subset"
    uid = uid_for(source, 1)
    path = source / f"meta/teacher_episodes/{uid}.json"
    manifest = json.loads(path.read_text())
    if damage == "uid":
        manifest["episode_uid"] = uid_for(source, 0)
    elif damage == "index":
        manifest["episode_index"] = 0
    elif damage == "frames":
        manifest["frame_count"] = 3
    elif damage == "incomplete":
        manifest["complete"] = False
    elif damage == "schema":
        manifest["schema"] = "unknown"
    elif damage == "missing_record":
        manifest["files"].pop("frames")
    elif damage in ("escape", "wildcard"):
        manifest["files"]["control"]["path"] = "../outside" if damage == "escape" else "teacher/*"
    elif damage == "digest":
        manifest["files"]["frames"]["sha256"] = "bad"
    else:
        manifest["files"]["frames"]["rows"] = 3
    path.write_text(json.dumps(manifest), encoding="utf-8")
    copy_paths(
        source,
        destination,
        list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS)),
    )
    fetched = []

    def fetch(names):
        fetched.extend(names)
        copy_paths(source, destination, names)

    with pytest.raises(SensorTransactionError, match="Teacher"):
        ensure_sensor_subset(destination, [1], 3, fetch)
    assert not any(p.startswith(("raw/", "data/", "teacher/", "videos/")) for p in fetched)


@pytest.mark.parametrize("missing", ["contract", "metadata", "attachment"])
def test_missing_remote_teacher_artifact_is_explicit_error(teacher_dataset, tmp_path, missing):
    source, destination = teacher_dataset.root, tmp_path / "missing"
    uid = uid_for(source, 1)
    path = {
        "contract": "meta/teacher_capture.json",
        "metadata": f"meta/teacher_episodes/{uid}.json",
        "attachment": f"teacher/{uid}/frames.jsonl",
    }[missing]
    (source / path).unlink()
    copy_paths(
        source,
        destination,
        list(filter_repo_objects(all_paths(source), allow_patterns=METADATA_DOWNLOAD_PATTERNS)),
    )

    def fetch(names):
        copy_paths(source, destination, [name for name in names if (source / name).is_file()])

    with pytest.raises(SensorTransactionError, match="incomplete after download"):
        ensure_sensor_subset(destination, [1], 3, fetch, teacher_required=True)


def test_cached_teacher_digest_damage_is_not_silently_accepted(teacher_dataset):
    uid = uid_for(teacher_dataset.root, 1)
    (teacher_dataset.root / f"teacher/{uid}/control.jsonl").write_text("corrupt\n")
    with pytest.raises(SensorTransactionError, match="digest mismatch"):
        LeRobotDataset(teacher_dataset.repo_id, root=teacher_dataset.root, episodes=[1])


def test_branch_refresh_keeps_teacher_metadata_and_attachments_at_one_commit(
    teacher_dataset, tmp_path, monkeypatch
):
    source, destination = teacher_dataset.root, tmp_path / "refresh"
    copy_paths(source, destination, all_paths(source))
    calls = []

    def snapshot(_repo_id, *, allow_patterns, revision, **kwargs):
        assert kwargs["token"] is False
        calls.append((revision, allow_patterns))
        names = list(filter_repo_objects(all_paths(source), allow_patterns=allow_patterns))
        copy_paths(source, destination, names)
        return str(destination)

    for module in ("dataset_metadata", "lerobot_dataset"):
        monkeypatch.setattr(f"lerobot.datasets.{module}.snapshot_download", snapshot)
    monkeypatch.setattr(
        "lerobot.datasets.sensor_hub.HfApi",
        lambda: SimpleNamespace(repo_info=lambda *_a, **_k: SimpleNamespace(sha=COMMIT)),
    )
    dataset = LeRobotDataset(
        teacher_dataset.repo_id,
        root=destination,
        episodes=[1],
        revision="main",
        token=False,
        force_cache_sync=True,
    )
    assert dataset.revision == COMMIT
    assert calls[0][0] == "main"
    assert all(revision == COMMIT for revision, _ in calls[1:])
    uid = uid_for(source, 1)
    assert any(f"teacher/{uid}/frames.jsonl" in patterns for _, patterns in calls)


def test_teacher_without_sensor_manifest_is_rejected(tmp_path):
    with pytest.raises(SensorTransactionError, match="missing its Sensor manifest"):
        ensure_sensor_subset(tmp_path, [0], 1, lambda *_a: None, teacher_required=True)
