"""Query disabled seven-axis state before cached-control validation; never enable or configure."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lerobot.robots.rebot_b601_follower.control_state import read_control_state
from lerobot.robots.rebot_b601_follower.readonly_transport import ReadonlyB601Transport


def run(port: str, output: Path, samples: int = 10) -> int:
    """Keep complete read-only evidence in a fresh directory, including failures."""
    if not 1 <= samples <= 100:
        raise ValueError("samples must be in 1..100")
    output.mkdir(parents=True, exist_ok=False)
    device = ReadonlyB601Transport(port)
    result = {
        "queries_only": True,
        "port": port,
        "samples": [],
        "error": None,
        "cached_motion_verified": False,
        "loaded_teacher_behavior_verified": False,
    }
    try:
        device.open()
        mapping = device.read_mapping()
        result["mapping"] = asdict(mapping)
        if not mapping.is_valid:
            raise RuntimeError(f"Mapping query failed: {mapping.error}")
        state = read_control_state(device, max_read_duration_s=1.0)
        result["control_state"] = asdict(state)
        if not state.is_valid:
            raise RuntimeError(f"Mode/watchdog query failed: {state.error}")
        for _ in range(samples):
            feedback = device.read_feedback(max_read_duration_s=0.05)
            result["samples"].append(asdict(feedback))
            if not feedback.is_valid or len(feedback.feedback) != 7:
                raise RuntimeError(f"Incomplete disabled query: {feedback.error}")
            if any(row.status_code != 0 for row in feedback.feedback):
                raise RuntimeError("All seven axes must already be disabled")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if device.has_resources:
                device.close()
        except Exception as exc:
            result["close_error"] = f"{type(exc).__name__}: {exc}"
        result["transport_closed"] = not device.has_resources
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("mapping", "control_state", "samples")}))
    return int(result["error"] is not None or not result["transport_closed"] or "close_error" in result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    return run(args.port, args.output, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
