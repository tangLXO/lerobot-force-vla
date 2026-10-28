"""Resolve stable device paths without opening a capture or trusting saved indices."""

from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _cv2_enumerate_cameras_available, require_package

if TYPE_CHECKING or _cv2_enumerate_cameras_available:
    from cv2_enumerate_cameras import enumerate_cameras


def enumerate_devices(backend: int) -> list[dict[str, Any]]:
    """List native identities on one explicit backend; never open a camera."""
    if not backend:
        raise ValueError("Device identity selection requires an explicit OpenCV backend")
    require_package("cv2-enumerate-cameras", extra="rebot", import_name="cv2_enumerate_cameras")
    return [
        {"device_name": item.name, "device_path": item.path, "index": item.index, "backend": item.backend}
        for item in enumerate_cameras(backend)
    ]


def resolve_device_path(device_path: str, backend: int) -> dict[str, Any]:
    """Require one exact native identity, even when device names are duplicated."""
    matches = [item for item in enumerate_devices(backend) if item["device_path"] == device_path]
    if len(matches) != 1:
        raise ValueError(f"Camera device_path must match exactly one device; found {len(matches)}")
    return matches[0]
