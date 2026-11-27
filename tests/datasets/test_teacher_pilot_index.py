import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config


@pytest.fixture
def pilot(tmp_path):
    parent = tmp_path / "data" / "b601_demos"
    entries = []
    channels = ["sensor.gripper_force.left.normal_force", "sensor.gripper_force.right.normal_force"]
    for number, value in enumerate([2, 90, 4]):
        name = f"run-{number}"
        root = parent / name
        features = {
            "action": {"dtype": "float32", "shape": (1,), "names": ["gripper.pos"]},
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["gripper.pos"]},
            "observation.tactile": {"dtype": "float32", "shape": (2,), "names": channels},
        }
        writer = LeRobotDataset.create(f"test/{name}", root=root, fps=50, features=features, use_videos=False)
        task = f"hold at {value} N"
        for i in range(3):
            writer.add_frame(
                {
                    "action": np.array([value], dtype=np.float32),
                    "observation.state": np.array([i], dtype=np.float32),
                    "observation.tactile": np.array([i, i + 1], dtype=np.float32),
                    "task": task,
                }
            )
        writer.save_episode()
        writer.finalize()
        control = root / "control.jsonl"
        control.write_text(json.dumps({"run_uid": name}) + "\n", encoding="utf-8")
        control_hash = hashlib.sha256(control.read_bytes()).hexdigest()
        frame_hash = hashlib.sha256(b"frames").hexdigest()
        manifest = {
            "complete": True,
            "episode_uid": f"ep-{number}",
            "episode_index": 0,
            "frame_count": 3,
            "config": {"repo_id": f"test/{name}", "task": task},
            "files": {
                "frames": {"sha256": frame_hash},
                "control": {"path": control.name, "sha256": control_hash},
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        entries.append(
            {
                "root": name,
                "manifest": "manifest.json",
                "frames": 3,
                "task": task,
                "run_uid": name,
                "episode_uid": manifest["episode_uid"],
                "episode_index": 0,
                "frames_sha256": frame_hash,
                "control_sha256": control_hash,
                "gripper_stage_ranges": [
                    {"start": 0, "stop": 1, "gripper_source": "teleop"},
                    {"start": 1, "stop": 2, "gripper_source": "transition"},
                    {"start": 2, "stop": 3, "gripper_source": "teacher"},
                ],
            }
        )
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"episodes": entries}), encoding="utf-8")
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            "test/b601",
            root=str(parent),
            teacher_pilot_index=str(index),
            held_out_run_uids=["run-1"],
            video_backend="pyav",
        ),
        policy=PI0Config(device="cpu", chunk_size=3, n_action_steps=3),
    )


def test_indexed_factory_joins_recordings_without_crossing_runs(pilot):
    train, val = make_train_eval_datasets(pilot)
    assert train.run_uids == ["run-0", "run-2"] and val.run_uids == ["run-1"]
    assert list(train.meta.episodes["dataset_from_index"]) == [0, 3]
    assert list(train.meta.episodes["run_uid"]) == train.run_uids
    assert train.num_frames == 6 and val.num_frames == 3
    np.testing.assert_allclose(train.meta.stats["action"]["mean"], [3])
    assert val.meta.stats is train.meta.stats
    assert train[2]["action"].flatten().tolist() == [2, 2, 2]
    assert train[2]["action_is_pad"].tolist() == [False, True, True]
    assert train[2]["gripper_hold"].tolist() == [True, True, True]
    sample = train[3]
    assert (sample["index"].item(), sample["episode_index"].item(), sample["frame_index"].item()) == (3, 1, 0)
    assert sample["task"] == "hold at 4 N"
    assert sample["task_index"].item() == train.meta.tasks.loc[sample["task"], "task_index"]
    assert val.frame_task_indices().tolist() == [val[0]["task_index"].item()] * 3
    assert sample["gripper_hold"].tolist() == [False, True, True]
    torch.testing.assert_close(sample["observation.tactile"], torch.tensor([0.0, 1.0]))


def test_indexed_factory_rejects_wrong_run_identity(pilot):
    path = Path(pilot.dataset.root) / "run-0/control.jsonl"
    path.write_text(json.dumps({"run_uid": "run-1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="run UID"):
        make_train_eval_datasets(pilot)
