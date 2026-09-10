#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json

import av
import numpy as np
import pytest
from test_sensor_stream import FakeSensor, capture

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.datasets.sensor_transaction import capture_logical_evidence


def encode_video(path, count=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_sensor_writer_seals_video_and_metadata_before_main_saved(tmp_path, monkeypatch):
    video_key = "observation.images.cam"
    dataset = LeRobotDataset.create(
        repo_id="test/sensor-video",
        root=tmp_path / "dataset",
        fps=30,
        batch_encoding_size=10,
        features={
            video_key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        },
    )

    def encode(_key, episode_index):
        path = dataset.root / f"encoded-{episode_index}" / "cam.mp4"
        encode_video(path)
        return path

    monkeypatch.setattr(dataset.writer, "_encode_temporary_episode_video", encode)
    monkeypatch.setattr(
        "lerobot.datasets.dataset_metadata.load_episodes", lambda *_args: pytest.fail("seal scanned history")
    )
    sensor = FakeSensor()
    recorder = SensorStreamRecorder(dataset.root, {"gripper_force": sensor})
    transactions = []
    for episode in range(2):
        uid = recorder.start_episode(episode, dataset=dataset)
        assert dataset.writer._batch_encoding_size == 1
        sensor._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 2.0},
            100 + episode,
            arrival_timestamp_ns=101 + episode,
        )
        for i in range(3):
            dataset.add_frame(
                {
                    video_key: np.zeros((32, 32, 3), dtype=np.uint8),
                    "observation.state": np.array([i], dtype=np.float32),
                    "task": "test",
                }
            )
            recorder.record_sync(i, capture(103, 100 + episode, episode))
        transaction = recorder.prepare_episode(task_info=["test"])
        recorder.commit_prepared(dataset)
        transactions.append(transaction)
        journal = json.loads((dataset.root / "meta/sensor_transactions" / f"{uid}.json").read_text())
        assert journal["state"] == "COMMITTED"
        assert len(journal["main_evidence"]["videos"]) == 1
        assert dataset.writer._batch_encoding_size == 10
        assert dataset.meta.latest_episode is None
    for transaction in transactions:
        assert capture_logical_evidence(transaction) == transaction.journal["main_evidence"]
    recorder.close()
    dataset.finalize()
