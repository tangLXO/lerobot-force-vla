"""Optional exposure configuration never accesses a physical camera in tests."""

from unittest.mock import Mock

import cv2
import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def test_unconfigured_exposure_preserves_device_parameters():
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0))
    camera.videocapture = Mock()
    camera._configure_exposure()
    camera.videocapture.set.assert_not_called()
    camera.videocapture.get.assert_not_called()


def test_requested_native_exposure_uses_readback_without_changing_auto_exposure():
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, exposure=-6))
    camera.videocapture = Mock()
    camera.videocapture.set.return_value = True
    camera.videocapture.get.return_value = -6
    camera._configure_exposure()
    camera.videocapture.set.assert_called_once_with(cv2.CAP_PROP_EXPOSURE, -6.0)
    camera.videocapture.get.assert_called_once_with(cv2.CAP_PROP_EXPOSURE)


def test_driver_exposure_rejection_is_reported_without_fabricating_readback(caplog):
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, exposure=-6))
    camera.videocapture = Mock()
    camera.videocapture.set.return_value = False
    camera.videocapture.get.return_value = -5
    camera._configure_exposure()
    assert "returned exposure=-5" in caplog.text
    assert "accepted=False" in caplog.text


@pytest.mark.parametrize("value", [True, "-6", float("nan"), float("inf")])
def test_exposure_rejects_values_that_cannot_be_sent_to_backend(value):
    with pytest.raises(ValueError, match="exposure"):
        OpenCVCameraConfig(index_or_path=0, exposure=value)
