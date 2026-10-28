"""Unconfirmed camera shutdown retains ownership without opening hardware."""

import threading
from unittest.mock import Mock

import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def test_live_reader_retains_capture_and_refuses_reconnect():
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0))
    capture = camera.videocapture = Mock()
    thread = camera.thread = Mock()
    camera.stop_event = threading.Event()
    thread.is_alive.return_value = True
    with pytest.raises(TimeoutError, match="retained"):
        camera.disconnect()
    assert camera.has_resources and camera.thread is thread and camera.videocapture is capture
    capture.release.assert_not_called()
    with pytest.raises(RuntimeError, match="cleanup"):
        camera.connect()
    thread.is_alive.return_value = False
    camera.disconnect()
    assert not camera.has_resources
    capture.release.assert_called_once()
