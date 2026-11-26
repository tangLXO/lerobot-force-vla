"""Regression tests for normalization when holding out B601 Teacher episodes."""

import numpy as np
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import _use_pi_teacher_training_stats, make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@pytest.fixture
def teacher_root(tmp_path):
    root = tmp_path / "teacher"
    action = {"dtype": "float32", "shape": (1,), "names": ["gripper.pos"]}
    features = {key: action.copy() for key in ("action", "observation.state", "teacher.action_pre_safety")}
    writer = LeRobotDataset.create("test/teacher", root=root, fps=50, features=features, use_videos=False)
    for value, frames in [(2, 2), (5, 3), (90, 4)]:
        for _ in range(frames):
            frame = {key: np.array([value], dtype=np.float32) for key in features}
            writer.add_frame({**frame, "task": "grasp"})
        writer.save_episode()
    writer.finalize()
    return root


@pytest.mark.parametrize("policy_cls", [PI0Config, PI05Config])
def test_pi_teacher_normalization_uses_only_selected_training_episodes(teacher_root, policy_cls):
    root = teacher_root
    stats_path = root / "meta" / "stats.json"
    original_stats = stats_path.read_bytes()
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            "test/teacher", root=root, eval_split=0.25, use_imagenet_stats=False, video_backend="pyav"
        ),
        policy=policy_cls(device="cpu"),
    )
    train, evaluation = make_train_eval_datasets(cfg)
    assert train.episodes == [0, 1]
    assert evaluation.episodes == [2]
    for dataset in (train, evaluation):
        for key in ("action", "observation.state"):
            stats = dataset.meta.stats[key]
            np.testing.assert_array_equal(stats["count"], [5])
            np.testing.assert_allclose(stats["mean"], [3.8])
            np.testing.assert_array_equal(stats["max"], [5])

    cfg.dataset.eval_split = 0
    cfg.dataset.exclude_episodes = [2]
    subset, evaluation = make_train_eval_datasets(cfg)
    assert evaluation is None
    np.testing.assert_allclose(subset.meta.stats["action"]["mean"], [3.8])

    cfg.dataset.exclude_episodes = None
    full, _ = make_train_eval_datasets(cfg)
    np.testing.assert_array_equal(full.meta.stats["action"]["count"], [9])
    np.testing.assert_array_equal(full.meta.stats["action"]["max"], [90])
    assert stats_path.read_bytes() == original_stats


def test_legacy_teacher_metadata_recognized_by_capture_contract(teacher_root):
    cfg = TrainPipelineConfig(dataset=DatasetConfig("test/teacher", root=teacher_root), policy=PI0Config())
    meta = LeRobotDatasetMetadata("test/teacher", root=teacher_root)
    del meta.features["teacher.action_pre_safety"]
    assert not _use_pi_teacher_training_stats(cfg, meta, [0, 1])
    np.testing.assert_array_equal(meta.stats["action"]["count"], [9])

    (teacher_root / "meta" / "teacher_capture.json").write_text("{}", encoding="utf-8")
    assert _use_pi_teacher_training_stats(cfg, meta, [0, 1])
    np.testing.assert_array_equal(meta.stats["action"]["count"], [5])
