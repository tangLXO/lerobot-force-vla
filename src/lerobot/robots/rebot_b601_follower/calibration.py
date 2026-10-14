"""File-backed limits for the supervised B601 gripper in device-zeroed degrees."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

DEFAULT_GRIPPER_CALIBRATION = HF_LEROBOT_CALIBRATION / "robots/rebot_b601_follower/b601_x518_follower1.json"


@dataclass(frozen=True)
class B601GripperCalibration:
    """Pin the source file, its revision and supported native-angle bounds."""

    path: str
    sha256: str
    mtime_ns: int
    minimum_deg: float
    maximum_deg: float

    @classmethod
    def load(cls, path: str | Path) -> "B601GripperCalibration":
        """Read an existing B601 file without creating files or zeroing motors."""
        source = Path(path).expanduser().resolve()
        try:
            before = source.stat().st_mtime_ns
            raw = source.read_bytes()
            after = source.stat().st_mtime_ns
            content = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取 B601 校准文件：{source}；{exc}") from exc
        if before != after:
            raise ValueError("校准文件读取期间发生变化，请重新加载")
        gripper = content.get("gripper") if isinstance(content, dict) else None
        if not isinstance(gripper, dict) or type(gripper.get("id")) is not int or gripper["id"] != 7:
            raise ValueError("校准文件必须包含 B601 夹爪 gripper，电机 ID=7")
        # The B601 calibrator writes zero into the motor, not a software offset.
        for key in ("drive_mode", "homing_offset"):
            if type(gripper.get(key)) is not int or gripper[key] != 0:
                raise ValueError(f"B601 原生角度控制仅支持校准文件 {key}=0，不会忽略或重复施加偏移")
        bounds = [gripper.get(key) for key in ("range_min", "range_max")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds):
            raise ValueError("校准夹爪限位必须是有限角度数值")
        lower, upper = map(float, bounds)
        if lower >= upper:
            raise ValueError("校准夹爪 range_min 必须小于 range_max")
        return cls(str(source), hashlib.sha256(raw).hexdigest(), after, lower, upper)

    def require_unchanged(self) -> None:
        """Reject replacement or regeneration, even when the JSON bytes are unchanged."""
        if self.load(self.path) != self:
            raise ValueError("校准文件已变化，请重新加载、重新连接并读取新参考后再启动")

    def check_travel(self, reference: float, travel: float) -> None:
        """Reject a requested positive-closing travel instead of silently shortening it."""
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (reference, travel)):
            raise ValueError("校准参考和行程必须是有限角度数值")
        if not self.minimum_deg <= reference <= self.maximum_deg:
            raise ValueError(
                f"夹爪参考 {reference:g}° 超出校准限位 [{self.minimum_deg:g}, {self.maximum_deg:g}]°"
            )
        if travel < 0 or reference + travel > self.maximum_deg:
            raise ValueError(f"闭合行程超出校准限位；当前最多可闭合 {self.maximum_deg - reference:g}°")

    def intersect(self, lower: float, upper: float) -> tuple[float, float]:
        """Keep feedback margins and command bounds inside the file's limits."""
        return max(lower, self.minimum_deg), min(upper, self.maximum_deg)
