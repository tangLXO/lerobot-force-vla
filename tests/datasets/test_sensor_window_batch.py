#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import os
import pickle
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_sensor_reader_readonly import create_episode
from test_sensor_stream import FakeSensor, capture
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig, SensorWindowConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.datasets.sensor_window import SensorStreamReader, SensorWindowDataset
from lerobot.datasets.sensor_window_cache import SensorRowGroupCache
from lerobot.policies.act.configuration_act import ACTConfig


def assert_windows_equal(first, second):
    assert first.keys() == second.keys()
    for stream in first:
        for field in first[stream]:
            np.testing.assert_array_equal(first[stream][field], second[stream][field])


@pytest.fixture
def recorded_dataset(tmp_path):
    dataset = LeRobotDataset.create(
        "test/sensor-batch",
        root=tmp_path / "dataset",
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            "observation.tactile": {
                "dtype": "float32",
                "shape": (2,),
                "names": [
                    "sensor.left.left.normal_force",
                    "sensor.right.right.normal_force",
                ],
            },
        },
    )
    sensors = {name: FakeSensor() for name in ("left", "right")}
    sensors["left"].config.frame_features = ["left.normal_force"]
    sensors["right"].config.frame_features = ["right.normal_force"]
    for sensor in sensors.values():
        sensor.config.recorder_flush_rows = 2
    recorder = SensorStreamRecorder(dataset.root, sensors)
    for episode in range(3):
        recorder.start_episode(episode, dataset=dataset)
        for frame in range(4):
            timestamp = 1_000_000_000 + (episode * 4 + frame) * 30_000_000
            for sensor in sensors.values():
                sensor._publish_sample(
                    {"left.normal_force": float(frame), "right.normal_force": float(episode)},
                    timestamp,
                    arrival_timestamp_ns=timestamp + 1,
                )
            metadata = capture(timestamp + 2, timestamp, episode * 4 + frame)
            selected = metadata["sensors"].pop("gripper_force")
            metadata["sensors"] = {name: dict(selected) for name in sensors}
            dataset.add_frame(
                {
                    "observation.state": np.array([frame], dtype=np.float32),
                    "observation.tactile": np.array([frame, episode], dtype=np.float32),
                    "task": "test",
                }
            )
            recorder.record_sync(frame, metadata)
        recorder.save_episode(dataset, task_info=["test"])
    recorder.close()
    dataset.finalize()
    return LeRobotDataset("test/sensor-batch", root=dataset.root)


def wrap(dataset, *, cache_mb=64, cls=SensorWindowDataset):
    return cls(
        dataset, {name: SensorWindowConfig(90, 100, 40) for name in ("left", "right")}, cache_mb=cache_mb
    )


@pytest.mark.parametrize("cache_mb", [0, 0.002, 64])
def test_batch_matches_single_with_duplicates_out_of_order_and_cross_episode(
    recorded_dataset, cache_mb, monkeypatch
):
    wrapper = wrap(recorded_dataset, cache_mb=cache_mb)
    indices = [9, 1, 9, 7, 0, 4, 11]
    single = [wrapper[i] for i in indices]
    original = recorded_dataset.__getitems__
    calls = []

    def batch_get(indices):
        calls.append(indices)
        return original(indices)

    monkeypatch.setattr(recorded_dataset, "__getitems__", batch_get)
    batch = wrapper.__getitems__(indices)
    assert calls == [indices]
    assert [int(item["index"]) for item in batch] == indices
    for expected, actual in zip(single, batch, strict=True):
        assert_windows_equal(expected["sensor_windows"], actual["sensor_windows"])
        np.testing.assert_array_equal(expected["observation.tactile"], actual["observation.tactile"])
        assert actual["observation.tactile"].shape == (2,)
    batch[0]["sensor_windows"]["left"]["values"][0, 0] = 12345
    assert batch[2]["sensor_windows"]["left"]["values"][0, 0] != 12345
    assert wrapper.__getitems__([]) == []
    assert wrapper.reader._row_group_cache.decoded_bytes <= int(cache_mb * 1024 * 1024)


def test_batch_decodes_row_group_union_once_even_without_cache(tmp_path, monkeypatch):
    create_episode(tmp_path)
    reader = SensorStreamReader(tmp_path, cache_mb=0)
    uid = reader._resolve_episode_uid(None, 0)
    original = pq.ParquetFile
    groups = []

    class Spy(original):
        def iter_batches(self, *args, **kwargs):
            groups.extend(kwargs.get("row_groups", []))
            yield from super().iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    # Nonadjacent ranges must not decode groups lying between them.
    windows = reader._dense_windows(
        [350_000_000, 950_000_000, 350_000_000], 100, 10, 60, "gripper_force", uid
    )
    assert groups == [0, 5]
    assert [window.sequence.tolist() for window in windows] == [[10], [0], [10]]
    assert reader._row_group_cache.decoded_bytes == 0


def test_byte_lru_eviction_oversized_groups_and_instance_keys(tmp_path):
    path = tmp_path / "groups.parquet"
    table = pa.table({"timestamp_ns": np.arange(300), "payload": ["x" * 80 for _ in range(300)]})
    pq.write_table(table, path, row_group_size=100, use_dictionary=False)
    with pq.ParquetFile(path) as parquet:
        size = parquet.read_row_group(0).get_total_buffer_size()
        budget = max(size, parquet.metadata.row_group(0).total_byte_size)
        cache = SensorRowGroupCache(budget)
        for group in (0, 1, 2, 0):
            list(cache.batches(parquet, "uid", "left", group))
            assert cache.decoded_bytes <= budget
            assert len(cache.entries) == 1
        assert cache.misses == 4
        list(cache.batches(parquet, "uid", "left", 0))
        assert cache.hits == 1
        list(cache.batches(parquet, "uid", "right", 0))
        assert cache.misses == 5
        list(cache.batches(parquet, "other-uid", "right", 0))
        assert cache.misses == 6
        small = SensorRowGroupCache(1)
        assert sum(batch.num_rows for batch in small.batches(parquet, "uid", "left", 0)) == 100
        assert not small.entries and small.decoded_bytes == 0


def test_dictionary_expansion_does_not_reside_above_byte_limit(tmp_path):
    path = tmp_path / "dictionary.parquet"
    pq.write_table(pa.table({"payload": ["x" * 500] * 9000}), path, use_dictionary=True)
    cache = SensorRowGroupCache(8192)
    with pq.ParquetFile(path) as parquet:
        assert parquet.metadata.row_group(0).total_byte_size < cache.limit_bytes
        assert sum(batch.num_rows for batch in cache.batches(parquet, "uid", "left", 0)) == 9000
    assert cache.decoded_bytes == 0 and not cache.entries


def test_pickle_and_pid_change_clear_only_process_local_caches(recorded_dataset, monkeypatch):
    wrapper = wrap(recorded_dataset)
    expected = wrapper[0]
    assert wrapper.reader._row_group_cache.entries
    saved = pickle.dumps(wrapper)
    restored = pickle.loads(saved)
    assert restored.reader._pid is None
    assert restored.reader._row_group_cache is None
    assert not restored.reader._anchor_cache
    assert_windows_equal(expected["sensor_windows"], restored[0]["sensor_windows"])
    old = wrapper.reader._row_group_cache
    monkeypatch.setattr("lerobot.datasets.sensor_window.os.getpid", lambda: 987654321)
    wrapper.reader._ensure_process()
    assert wrapper.reader._pid == 987654321
    assert wrapper.reader._row_group_cache is not old
    assert not wrapper.reader._row_group_cache.entries
    assert not wrapper.reader._anchor_cache


class InspectingWindowDataset(SensorWindowDataset):
    def __getitems__(self, indices):
        items = super().__getitems__(indices)
        for item in items:
            item["worker_pid"] = os.getpid()
            item["reader_pid"] = self.reader._pid
            item["cache_bytes"] = self.reader._row_group_cache.decoded_bytes
        return items


def test_real_dataloader_windows_spawn_two_workers_shuffle_and_batch(recorded_dataset):
    wrapper = wrap(recorded_dataset, cache_mb=0.01, cls=InspectingWindowDataset)
    reference = [wrapper[i] for i in range(len(wrapper))]
    loader = DataLoader(
        wrapper,
        num_workers=2,
        multiprocessing_context="spawn",
        batch_size=3,
        shuffle=True,
        generator=torch.Generator().manual_seed(1729),
        timeout=60,
    )
    seen, workers = [], set()
    for batch in loader:
        for position, index in enumerate(batch["index"].tolist()):
            seen.append(index)
            workers.add(int(batch["worker_pid"][position]))
            assert batch["worker_pid"][position] == batch["reader_pid"][position]
            assert batch["cache_bytes"][position] <= int(0.01 * 1024 * 1024)
            actual = {
                stream: {field: values[position].numpy() for field, values in window.items()}
                for stream, window in batch["sensor_windows"].items()
            }
            assert_windows_equal(reference[index]["sensor_windows"], actual)
    assert sorted(seen) == list(range(len(wrapper)))
    assert seen != sorted(seen)
    assert len(workers) == 2 and os.getpid() not in workers


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_cache_configuration_rejected(value):
    with pytest.raises(ValueError, match="sensor_window_cache_mb"):
        DatasetConfig("test/cache", sensor_window_cache_mb=value)


def test_cache_defaults_and_zero_disable():
    assert DatasetConfig("test/cache").sensor_window_cache_mb == 64
    assert DatasetConfig("test/cache", sensor_window_cache_mb=0).sensor_window_cache_mb == 0


def test_split_constructs_only_selected_wrappers(recorded_dataset, monkeypatch):
    import lerobot.datasets.factory as factory

    monkeypatch.setattr(ACTConfig, "supports_sensor_windows", True)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            "test/sensor-batch",
            root=str(recorded_dataset.root),
            eval_split=0.34,
            use_imagenet_stats=False,
            sensor_window_cache_mb=0,
            sensor_windows={"left": SensorWindowConfig(90, 100, 40)},
        ),
        policy=ACTConfig(),
    )
    calls = []
    original = factory.SensorWindowDataset

    def wrapping(dataset, windows, **kwargs):
        calls.append((dataset.episodes, kwargs))
        return original(dataset, windows, **kwargs)

    monkeypatch.setattr(factory, "SensorWindowDataset", wrapping)
    monkeypatch.setattr(factory, "make_dataset", lambda *_a: pytest.fail("discarded full Dataset"))
    train, evaluation = make_train_eval_datasets(cfg)
    assert calls == [([0], {"cache_mb": 0}), ([1, 2], {"cache_mb": 0})]
    assert train.reader.committed_episode_indices == (0,)
    assert evaluation.reader.committed_episode_indices == (1, 2)


def test_empty_unpickling_wrapper_does_not_recurse():
    wrapper = SensorWindowDataset.__new__(SensorWindowDataset)
    with pytest.raises(AttributeError):
        _ = wrapper.dataset
    wrapper.dataset = SimpleNamespace(name="base")
    assert wrapper.name == "base"


def test_streaming_windows_rejected_before_dataset_io(monkeypatch):
    import lerobot.datasets.factory as factory

    monkeypatch.setattr(ACTConfig, "supports_sensor_windows", True)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            "test/no-io", streaming=True, sensor_windows={"left": SensorWindowConfig(90, 100, 40)}
        ),
        policy=ACTConfig(),
    )
    monkeypatch.setattr(factory, "load_dataset_metadata", lambda *_a, **_k: pytest.fail("Dataset I/O"))
    with pytest.raises(ValueError, match="streaming is unsupported"):
        factory.make_dataset(cfg)


@pytest.mark.parametrize("split", [0, 0.5])
def test_nondefault_window_storage_rejected_before_dataset_construction(monkeypatch, split):
    import lerobot.datasets.factory as factory

    monkeypatch.setattr(ACTConfig, "supports_sensor_windows", True)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            "test/no-io", eval_split=split, sensor_windows={"left": SensorWindowConfig(90, 100, 40)}
        ),
        policy=ACTConfig(),
    )
    monkeypatch.setattr(
        factory, "load_dataset_metadata", lambda *_a, **_k: SimpleNamespace(storage_format="lance")
    )
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *_a, **_k: pytest.fail("Dataset construction"))
    with pytest.raises(ValueError, match="local Parquet storage"):
        factory.make_train_eval_datasets(cfg)
