"""Native identity selection is independent of index order and duplicate names."""

import copy

import pytest

from lerobot.cameras.opencv import device_binding
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.force_control.demonstration import TeacherRecordConfig


def test_same_names_bind_two_paths_and_follow_index_changes(monkeypatch):
    raw = {
        role: {"device_path": path, "backend": 700, "fps": 50, "width": 640, "height": 480}
        for role, path in (("front", "usb/front"), ("wrist", "usb/wrist"))
    }
    before = copy.deepcopy(raw)
    for front_index, wrist_index in ((0, 2), (2, 0)):
        monkeypatch.setattr(
            device_binding,
            "enumerate_devices",
            lambda backend, front_index=front_index, wrist_index=wrist_index: [
                {
                    "device_name": "Global Shutter Camera",
                    "device_path": "usb/front",
                    "index": front_index,
                    "backend": backend,
                },
                {
                    "device_name": "Global Shutter Camera",
                    "device_path": "usb/wrist",
                    "index": wrist_index,
                    "backend": backend,
                },
            ],
        )
        configs = TeacherRecordConfig(root="unused", task="test", camera_configs=raw).camera_objects()
        bindings = {
            name: device_binding.resolve_device_path(config.device_path, 700)
            for name, config in configs.items()
        }
        assert configs["front"].device_path == "usb/front"
        assert bindings["front"]["index"] == front_index
        assert bindings["wrist"]["index"] == wrist_index
        assert configs["wrist"].exposure is None
    assert raw == before


def test_missing_path_never_falls_back_to_name_or_index(monkeypatch):
    monkeypatch.setattr(device_binding, "enumerate_devices", lambda _: [])
    with pytest.raises(ValueError, match="exactly one"):
        device_binding.resolve_device_path("disconnected", 700)


def test_selector_validation_and_duplicate_role_rejection(monkeypatch):
    with pytest.raises(ValueError, match="no index_or_path"):
        OpenCVCameraConfig(index_or_path=0, device_path="usb/one", backend=700)
    with pytest.raises(ValueError, match="explicit backend"):
        OpenCVCameraConfig(device_path="usb/one")
    monkeypatch.setattr(
        device_binding,
        "enumerate_devices",
        lambda _: [{"device_name": "same", "device_path": "usb/one", "index": 0, "backend": 700}],
    )
    with pytest.raises(ValueError, match="same device"):
        TeacherRecordConfig(
            root="unused",
            task="test",
            camera_configs={
                name: {"device_path": "usb/one", "backend": 700, "fps": 50, "width": 16, "height": 16}
                for name in ("front", "wrist")
            },
        ).camera_objects()


def test_standard_record_rejects_duplicate_paths_before_constructing_cameras(monkeypatch):
    def unexpected_camera(*args, **kwargs):
        pytest.fail("Duplicate roles must be rejected before any camera is constructed")

    monkeypatch.setattr("lerobot.cameras.opencv.OpenCVCamera", unexpected_camera)
    configs = {
        role: OpenCVCameraConfig(device_path="usb/one", backend=700, fps=50) for role in ("front", "wrist")
    }
    with pytest.raises(ValueError, match="same device path"):
        make_cameras_from_configs(configs)
