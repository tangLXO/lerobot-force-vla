"""Startup-only communication options shared by both desktop entry points."""

import math
import os

DEFAULT_COMMAND_TX_GAP_US = 120.0


def resolve_command_tx_gap(value):
    """Resolve an explicit option before the MotorBridge environment override."""
    source = "命令行"
    if value is None:
        value = os.environ.get("MOTORBRIDGE_TX_GAP_US")
        source = "环境变量 MOTORBRIDGE_TX_GAP_US"
    if value is None:
        return DEFAULT_COMMAND_TX_GAP_US, "官方默认"
    try:
        resolved = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{source}: command TX gap must be finite and nonnegative") from exc
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{source}: command TX gap must be finite and nonnegative")
    return resolved, source


def communication_description(mode, gap, source, *, gripper_only=False):
    """Describe the resolved startup policy without implying a measured interval."""
    if gripper_only:
        mode = "sequential"
    mode = mode or "配置模式"
    if mode == "sequential":
        return "通信模式 sequential；异步发送间隔不适用"
    return f"通信模式 {mode}；最小发送间隔 {gap:g} μs（{source}）"
