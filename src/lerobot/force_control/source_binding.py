"""Reject a shared virtual environment importing a different checkout before I/O."""

import importlib
from pathlib import Path


def require_source_tree(root: Path) -> dict[str, str]:
    """Return verified module paths for parent/Windows-spawn runtime diagnostics."""
    result = {}
    for name in (
        "lerobot.force_control.runtime",
        "lerobot.force_control.capture_process",
        "lerobot.robots.rebot_b601_follower.control_backend",
        "lerobot.robots.rebot_b601_follower.pipelined_control",
    ):
        actual = Path(importlib.import_module(name).__file__).resolve()
        expected = (root / "src" / Path(*name.split("."))).with_suffix(".py").resolve()
        if actual != expected:
            raise RuntimeError(
                f"Source checkout mismatch: {name}: {actual}; expected {expected}. Set PYTHONPATH to this checkout's src."
            )
        result[name] = str(actual)
    return result
