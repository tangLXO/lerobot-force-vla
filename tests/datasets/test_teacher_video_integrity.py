"""Teacher's opt-in video gate must run before any main Dataset mutation."""

import queue
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("datasets")

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_dataset(root):
    return LeRobotDataset.create(
        "test/teacher-integrity",
        fps=30,
        root=root,
        streaming_encoding=True,
        strict_video_integrity=True,
        encoder_threads=2,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", g=2),
        features={
            "action": {"dtype": "float32", "shape": (1,)},
            **{
                f"observation.images.{role}": {
                    "dtype": "video",
                    "shape": (16, 16, 3),
                    "names": ["height", "width", "channels"],
                }
                for role in ("front", "wrist")
            },
        },
    )


def frame():
    return {
        "action": np.zeros(1, np.float32),
        "task": "test",
        **{f"observation.images.{role}": np.zeros((16, 16, 3), np.uint8) for role in ("front", "wrist")},
    }


@pytest.mark.parametrize("failure", ["queue_full", "thread_error", "missing_result", "wrong_count"])
def test_strict_video_fault_never_writes_main(tmp_path, monkeypatch, failure):
    dataset = make_dataset(tmp_path / "data")
    try:
        dataset.prepare_episode_recording(collect_garbage=False)
        encoder = dataset.writer._streaming_encoder
        wrist = "observation.images.wrist"
        save_main = Mock(side_effect=AssertionError("main write must not be reached"))
        monkeypatch.setattr(dataset.writer, "_save_episode_data", save_main)
        info_before = (dataset.root / "meta/info.json").read_bytes()
        if failure == "queue_full":
            original = encoder._frame_queues[wrist].put

            def put(value, **kwargs):
                if value is not None:
                    raise queue.Full
                return original(value, **kwargs)

            monkeypatch.setattr(encoder._frame_queues[wrist], "put", put)
            with pytest.raises(RuntimeError, match="lost a frame"):
                dataset.add_frame(frame())
            assert encoder._accepted_frames["observation.images.front"] == 1
        else:
            dataset.add_frame(frame())
            dataset.add_frame(frame())
            result = encoder._result_queues[wrist]
            original_get = result.get

            def get(**kwargs):
                if failure == "missing_result":
                    raise queue.Empty
                if failure == "thread_error":
                    return "error", "injected last-frame failure"
                value = original_get(**kwargs)
                encoder._threads[wrist].encoded_frame_count -= 1
                return value

            monkeypatch.setattr(result, "get", get)
        with pytest.raises((RuntimeError, ValueError)):
            dataset.save_episode()
        assert dataset.num_episodes == 0 and not save_main.called
        assert (dataset.root / "meta/info.json").read_bytes() == info_before
        with pytest.raises((RuntimeError, ValueError)):
            dataset.save_episode()  # failure remains latched
    finally:
        dataset.clear_episode_buffer()
        dataset.finalize()


def test_strict_video_roundtrip_and_skip_only_explicit_gc(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("lerobot.datasets.dataset_writer.gc.collect", lambda: calls.append(True))
    dataset = make_dataset(tmp_path / "data")
    try:
        dataset.prepare_episode_recording()
        for _ in range(2):
            dataset.prepare_episode_recording(collect_garbage=False)
            for _ in range(3):
                dataset.add_frame(frame())
            dataset.save_episode()
        assert dataset.num_episodes == 2 and dataset.num_frames == 6 and calls == [True]
    finally:
        dataset.finalize()
