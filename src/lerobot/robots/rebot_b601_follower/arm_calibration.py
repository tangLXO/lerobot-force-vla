"""Read-only, file-backed native-angle calibration for an explicit B601 arm scope."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class B601ArmCalibration:
    """Pin calibration identity and per-joint bounds without configuring a motor."""

    path: str
    sha256: str
    mtime_ns: int
    # Entries are name, motor ID, minimum degrees, maximum degrees.
    joints: tuple[tuple[str, int, float, float], ...]

    @classmethod
    def load(cls, path: str | Path, joint_ids: dict[str, int]) -> "B601ArmCalibration":
        """Require every selected joint and reject unsupported software offsets."""
        source = Path(path).expanduser().resolve()
        try:
            before = source.stat().st_mtime_ns
            raw = source.read_bytes()
            after = source.stat().st_mtime_ns
            content = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read B601 arm calibration: {source}: {exc}") from exc
        if before != after:
            raise ValueError("Arm calibration changed while loading; reload it")
        if not isinstance(content, dict) or not joint_ids:
            raise ValueError("Arm calibration requires explicitly selected joints")
        joints = []
        for name, motor_id in joint_ids.items():
            item = content.get(name)
            if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] != motor_id:
                raise ValueError(f"Arm calibration has missing/mismatched motor ID for {name}")
            if any(
                type(item.get(key)) is not int or item[key] != 0 for key in ("drive_mode", "homing_offset")
            ):
                raise ValueError(
                    f"Native-angle arm calibration requires zero drive_mode/homing_offset: {name}"
                )
            bounds = [item.get(key) for key in ("range_min", "range_max")]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds):
                raise ValueError(f"Arm calibration requires finite bounds for {name}")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"Arm calibration range must increase for {name}")
            joints.append((name, motor_id, float(bounds[0]), float(bounds[1])))
        return cls(str(source), hashlib.sha256(raw).hexdigest(), after, tuple(joints))

    def require_unchanged(self) -> None:
        """A changed or reloaded calibration always requires a new control session."""
        if self.load(self.path, {name: motor_id for name, motor_id, _, _ in self.joints}) != self:
            raise ValueError("Arm calibration changed; reconnect and acquire fresh references")

    def require_bounds(self, name: str, minimum_deg: float, maximum_deg: float) -> None:
        """Reject explicit profile bounds outside the pinned calibration."""
        _, _, lower, upper = next(item for item in self.joints if item[0] == name)
        if not lower <= minimum_deg < maximum_deg <= upper:
            raise ValueError(f"Command bounds for {name} must fit inside the arm calibration")
