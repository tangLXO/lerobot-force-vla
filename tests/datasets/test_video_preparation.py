"""Eager codec preparation must preserve real frames and own failed workers."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import av
import numpy as np
import pytest

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import StreamingVideoEncoder, _CameraEncoderThread


def make_encoder():
    return StreamingVideoEncoder(
        fps=25, rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1", preset=12), encoder_threads=2
    )


def test_two_camera_barrier_opens_actual_codecs_before_frames(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    opened = []
    original = _CameraEncoderThread._open_container

    def delayed_open(self, height, width):
        if "slow" in self.video_path.name:
            entered.set()
            assert release.wait(10)
        container, stream = original(self, height, width)
        assert stream.codec_context.is_open
        opened.append(self.video_path.name)
        return container, stream

    monkeypatch.setattr(_CameraEncoderThread, "_open_container", delayed_open)
    encoder = make_encoder()
    try:
        encoder.start_episode(["fast", "slow"], tmp_path, frame_sizes={"fast": (64, 96), "slow": (64, 96)})
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(encoder.wait_until_ready)
            try:
                assert entered.wait(5)
                assert not pending.done()
            finally:
                release.set()
            pending.result(10)
        assert len(opened) == 2
        for i in range(3):
            for key in ("fast", "slow"):
                encoder.feed_frame(key, np.full((64, 96, 3), 40 + i, dtype=np.uint8))
        results = encoder.finish_episode()
        assert len(opened) == 2
        for path, stats in results.values():
            with av.open(path) as video:
                frames = list(video.decode(video=0))
            assert len(frames) == 3
            assert frames[0].pts == 0
            assert [float(frame.time) for frame in frames] == pytest.approx([0, 0.04, 0.08])
            assert np.asarray(stats["min"]).min() == 40
            assert np.asarray(stats["max"]).max() == 42
    finally:
        release.set()
        encoder.close()


def test_initialization_failure_cleans_all_workers(tmp_path, monkeypatch):
    original = _CameraEncoderThread._open_container

    def fail_one(self, height, width):
        if "bad" in self.video_path.name:
            raise ValueError("deliberate codec failure")
        return original(self, height, width)

    monkeypatch.setattr(_CameraEncoderThread, "_open_container", fail_one)
    encoder = make_encoder()
    encoder.start_episode(["ok", "bad"], tmp_path, frame_sizes={"ok": (64, 96), "bad": (64, 96)})
    threads = list(encoder._threads.values())
    with pytest.raises(RuntimeError, match="preparation failed"):
        encoder.wait_until_ready()
    assert not encoder.episode_active
    assert not any(thread.is_alive() for thread in threads)
    assert not list(tmp_path.iterdir())
    encoder.close()


def test_timeout_retains_live_worker_and_refuses_reuse(tmp_path, monkeypatch):
    release = threading.Event()
    original = _CameraEncoderThread._open_container

    def delayed_open(self, height, width):
        assert release.wait(15)
        return original(self, height, width)

    monkeypatch.setattr(_CameraEncoderThread, "_open_container", delayed_open)
    encoder = make_encoder()
    try:
        encoder.start_episode(["cam"], tmp_path, frame_sizes={"cam": (64, 96)})
        thread = encoder._threads["cam"]
        # Deterministically simulate a native codec that cannot be joined yet.
        with patch.object(thread, "join"), pytest.raises(TimeoutError, match="resources retained"):
            encoder.wait_until_ready(timeout_s=0.01)
        assert encoder._threads["cam"] is thread
        assert encoder._video_paths["cam"].parent.exists()
        with pytest.raises(RuntimeError, match="unfinished cancellation"):
            encoder.start_episode(["cam"], tmp_path)
        with pytest.raises(RuntimeError):
            encoder.feed_frame("cam", np.zeros((64, 96, 3), dtype=np.uint8))
    finally:
        release.set()
        encoder.close()
    assert not thread.is_alive()
    assert not list(tmp_path.iterdir())


def make_dataset(root, *, streaming=True, video=True):
    features = {"action": {"dtype": "float32", "shape": (1,), "names": ["joint"]}}
    if video:
        features["observation.images.cam"] = {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        "test/preparation",
        fps=25,
        features=features,
        root=root,
        streaming_encoding=streaming,
        encoder_threads=2,
    )


def add_frames(dataset):
    for _ in range(3):
        dataset.add_frame(
            {
                "action": np.array([1], dtype=np.float32),
                "task": "test",
                "observation.images.cam": np.full((64, 96, 3), 80, dtype=np.uint8),
            }
        )


def test_dataset_preparation_cancel_save_and_resume(tmp_path, monkeypatch):
    root = tmp_path / "data"
    dataset = make_dataset(root)
    encoder = dataset.writer._streaming_encoder
    collected = []

    def collect():
        assert all(worker.ready.is_set() for worker in encoder._threads.values())
        collected.append(True)

    monkeypatch.setattr("lerobot.datasets.dataset_writer.gc.collect", collect)
    try:
        dataset.prepare_episode_recording()
        workers = dict(encoder._threads)
        dataset.prepare_episode_recording()
        assert encoder._threads == workers
        assert collected == [True]
        assert dataset.writer.episode_buffer["size"] == 0
        dataset.clear_episode_buffer()
        assert not encoder.episode_active
        for _ in range(2):
            dataset.prepare_episode_recording()
            workers = dict(encoder._threads)
            add_frames(dataset)
            assert encoder._threads == workers
            with pytest.raises(RuntimeError, match="after frames"):
                dataset.prepare_episode_recording()
            dataset.save_episode()
    finally:
        dataset.finalize()
    dataset = LeRobotDataset.resume("test/preparation", root=root, streaming_encoding=True, encoder_threads=2)
    encoder = dataset.writer._streaming_encoder
    try:
        dataset.prepare_episode_recording()
        add_frames(dataset)
        dataset.save_episode()
        assert dataset.num_episodes == 3
        assert dataset.num_frames == 9
    finally:
        dataset.finalize()
    reader = LeRobotDataset("test/preparation", root=root, video_backend="pyav")
    with pytest.raises(RuntimeError):
        reader.prepare_episode_recording()


@pytest.mark.parametrize("streaming,video", [(False, True), (True, False)])
def test_non_streaming_or_no_video_preparation_is_noop(tmp_path, streaming, video):
    dataset = make_dataset(tmp_path / "data", streaming=streaming, video=video)
    try:
        dataset.prepare_episode_recording()
        assert dataset.writer.episode_buffer["size"] == 0
    finally:
        dataset.finalize()


def test_zero_frame_finalize_removes_prepared_files(tmp_path):
    dataset = make_dataset(tmp_path / "data")
    dataset.prepare_episode_recording()
    encoder = dataset.writer._streaming_encoder
    paths = list(encoder._video_paths.values())
    dataset.finalize()
    assert not encoder.episode_active
    assert all(not path.parent.exists() for path in paths)


def test_prepared_depth_video_preserves_frame_count(tmp_path):
    encoder = make_encoder()
    try:
        encoder.start_episode(
            ["depth"], tmp_path, depth_video_keys=["depth"], frame_sizes={"depth": (64, 96)}
        )
        encoder.wait_until_ready()
        for _ in range(3):
            encoder.feed_frame("depth", np.full((64, 96, 1), 1000, dtype=np.uint16))
        path, stats = encoder.finish_episode()["depth"]
        with av.open(path) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 3
        assert frames[0].pts == 0
        assert np.asarray(stats["min"]).min() == 1000
    finally:
        encoder.close()


def test_preparation_timeout_with_successful_cleanup(tmp_path, monkeypatch):
    original = _CameraEncoderThread._open_container

    def wait_for_cancel(self, height, width):
        assert self.stop_event.wait(5)
        return original(self, height, width)

    monkeypatch.setattr(_CameraEncoderThread, "_open_container", wait_for_cancel)
    encoder = make_encoder()
    encoder.start_episode(["cam"], tmp_path, frame_sizes={"cam": (64, 96)})
    with pytest.raises(TimeoutError, match="preparation timed out"):
        encoder.wait_until_ready(0.01)
    assert not encoder.episode_active
    assert not list(tmp_path.iterdir())
    encoder.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadline_allocates_no_workers(tmp_path, timeout):
    dataset = make_dataset(tmp_path / "data")
    try:
        with pytest.raises(ValueError):
            dataset.prepare_episode_recording(timeout)
        assert not dataset.writer._streaming_encoder.episode_active
    finally:
        dataset.finalize()


@pytest.mark.parametrize("failure_point", ["construct", "start"])
def test_partial_worker_startup_failure_removes_empty_directories(tmp_path, monkeypatch, failure_point):
    encoder = make_encoder()
    workers = []
    method = "__init__" if failure_point == "construct" else "start"
    original = getattr(_CameraEncoderThread, method)
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("partial startup")
        result = original(self, *args, **kwargs)
        workers.append(self)
        return result

    monkeypatch.setattr(_CameraEncoderThread, method, fail_second)
    with pytest.raises(RuntimeError, match="partial startup"):
        encoder.start_episode(["one", "two"], tmp_path, frame_sizes={"one": (64, 96), "two": (64, 96)})
    assert not encoder.episode_active
    assert not any(worker.is_alive() for worker in workers)
    assert not list(tmp_path.iterdir())
    encoder.close()


@pytest.mark.parametrize("named", [False, True])
def test_preparation_supports_channel_first_features(tmp_path, named):
    dataset = LeRobotDataset.create(
        "test/chw",
        fps=25,
        root=tmp_path / "data",
        streaming_encoding=True,
        encoder_threads=2,
        features={
            "observation.images.cam": {
                "dtype": "video",
                "shape": (3, 64, 96),
                "names": ["channels", "height", "width"] if named else None,
            }
        },
    )
    try:
        dataset.prepare_episode_recording()
        for _ in range(3):
            dataset.add_frame(
                {"observation.images.cam": np.ones((3, 64, 96), dtype=np.float32), "task": "test"}
            )
        dataset.save_episode()
        assert dataset.num_frames == 3
    finally:
        dataset.finalize()
