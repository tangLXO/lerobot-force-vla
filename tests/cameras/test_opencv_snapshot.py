"""Atomic capture metadata tests without opening a capture handle."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError


def test_capture_snapshot_pairs_pixels_sequence_and_times_without_consuming_event(monkeypatch):
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=2, height=1))
    camera.stop_event = threading.Event()
    frame = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    def read_frame():
        camera.stop_event.set()
        return frame

    monkeypatch.setattr(camera, "_read_from_hardware", read_frame)
    camera._read_loop()
    camera.thread = SimpleNamespace(is_alive=lambda: True)
    try:
        with patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True):
            first = camera.read_latest_snapshot()
            assert first.sequence == 0
            assert 0 < first.timestamp_ns <= first.arrival_timestamp_ns
            np.testing.assert_array_equal(first.image, frame[..., ::-1])
            assert camera.read_latest_snapshot(after_sequence=first.sequence) is None
            first.image[:] = 0
            np.testing.assert_array_equal(camera.read_latest_snapshot().image, frame[..., ::-1])
            assert camera.new_frame_event.is_set()
    finally:
        camera.thread = None


class QueuedSyntheticCamera(OpenCVCamera):
    """Exercise the actual capture/postprocess/cache loop at a fake device boundary."""

    source_kind = "synthetic"

    def __init__(self):
        super().__init__(OpenCVCameraConfig(index_or_path=0, width=2, height=1))
        self.input = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.start()

    @property
    def is_connected(self):
        return self.thread is not None

    def _read_from_hardware(self):
        value = self.input.get(timeout=2)
        if value is None:
            raise DeviceNotConnectedError("Synthetic device stopped")
        return np.full((1, 2, 3), value, dtype=np.uint8)

    def close(self):
        self.stop_event.set()
        self.input.put(None)
        self._stop_read_thread()


@pytest.fixture
def queued_camera(monkeypatch):
    def forbidden_connect(*args, **kwargs):
        raise AssertionError("Hardware connection is forbidden in snapshot tests")

    monkeypatch.setattr(OpenCVCamera, "connect", forbidden_connect)
    camera = QueuedSyntheticCamera()
    try:
        yield camera
    finally:
        camera.close()


def test_snapshot_early_notification_duplicate_and_legacy_event_are_independent(queued_camera):
    camera = queued_camera
    camera.input.put(17)
    first = camera.wait_for_snapshot(-1, 1000)
    assert first.sequence == 0
    assert camera.async_read(100).mean() == 17  # Clears only the public legacy Event.
    assert not camera.new_frame_event.is_set()
    again = camera.wait_for_snapshot(-1, 0)
    assert again.sequence == first.sequence
    assert again.timestamp_ns == first.timestamp_ns
    assert again.arrival_timestamp_ns == first.arrival_timestamp_ns
    np.testing.assert_array_equal(first.image, again.image)
    assert camera.wait_for_snapshot(first.sequence, 0) is None
    again.image[:] = 0
    assert camera.wait_for_snapshot(-1, 0).image.mean() == 17


def test_snapshot_waiters_share_no_cursor_and_cannot_lose_publication(queued_camera):
    camera = queued_camera
    for value in range(1, 7):
        after = value - 2
        results = []
        threads = [
            threading.Thread(
                target=lambda results=results, after=after: results.append(
                    camera.wait_for_snapshot(after, 1000)
                )
            )
            for _ in range(3)
        ]
        # Publication races entry into wait. The predicate is checked under the
        # same lock as notify, including when capture wins before a waiter starts.
        for thread in threads:
            thread.start()
        camera.input.put(value)
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        assert len(results) == 3
        assert all(item.sequence == value - 1 and item.image.mean() == value for item in results)
        assert len({item.timestamp_ns for item in results}) == 1


def test_snapshot_publication_at_predicate_wait_boundary_is_not_lost(queued_camera, monkeypatch):
    camera = queued_camera
    processed = threading.Event()
    postprocess = camera._postprocess_image
    condition_wait = camera._snapshot_condition.wait

    def process(raw):
        frame = postprocess(raw)
        processed.set()
        return frame

    def boundary_wait(timeout):
        # The predicate has found no capture and still holds the shared lock.
        # Let the actual reader finish postprocessing before releasing that lock
        # in Condition.wait: publication must remain observable on either side.
        camera.input.put(53)
        assert processed.wait(1)
        return condition_wait(timeout)

    monkeypatch.setattr(camera, "_postprocess_image", process)
    monkeypatch.setattr(camera._snapshot_condition, "wait", boundary_wait)
    snapshot = camera.wait_for_snapshot(-1, 1000)
    assert snapshot.sequence == 0 and snapshot.image.mean() == 53


def test_snapshot_spurious_notification_timeout_cancel_and_dead_source(queued_camera):
    camera = queued_camera
    with camera._snapshot_condition:
        camera._snapshot_condition.notify_all()
    assert camera.wait_for_snapshot(-1, 1) is None
    stop = threading.Event()
    stop.set()
    assert camera.wait_for_snapshot(-1, 1000, stop) is None
    started = threading.Event()
    results = []

    def wait():
        started.set()
        results.append(camera.wait_for_snapshot(-1, 10_000, stop))

    stop.clear()
    waiter = threading.Thread(target=wait)
    waiter.start()
    assert started.wait(1)
    stop.set()
    waiter.join(1)
    assert not waiter.is_alive() and results == [None]
    camera.input.put(None)
    camera.thread.join(1)
    with pytest.raises(RuntimeError, match="not running"):
        camera.wait_for_snapshot(-1, 100)


def test_snapshot_background_failure_is_not_idle_timeout(queued_camera):
    camera = queued_camera
    with camera._snapshot_condition:
        camera._snapshot_error = OSError("synthetic capture failed")
        camera._snapshot_condition.notify_all()
    with pytest.raises(RuntimeError, match="capture failed") as caught:
        camera.wait_for_snapshot(-1, 0)
    assert isinstance(caught.value.__cause__, OSError)
