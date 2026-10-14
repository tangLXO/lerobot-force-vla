"""Thin calibrated I-P runner using the common ForceRuntime and OwnerService."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import draccus

from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.robots.rebot_b601_follower.control_protocol import force_pos_wire_codes
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig, run as run_runtime


def load_reviewed(path):
    """Validate a pinned calibration and typed profile without opening devices."""
    draft = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = B601GripperCalibration.load(draft["calibration"]["path"])
    if asdict(calibration) != draft["calibration"]:
        raise ValueError("校准文件已变化；请重新加载并读取新参考")
    config = draccus.decode(ForceTeacherRunConfig, draft["config"])
    config.validate()
    if (config.safety.q_min_deg, config.safety.q_max_deg) != (
        calibration.minimum_deg,
        calibration.maximum_deg,
    ):
        raise ValueError("硬角度范围必须与当前校准文件一致")
    return draft, config


def exception_messages(error):
    """Expose each execution/cleanup failure instead of only the group count."""
    notes = getattr(error, "__notes__", ())
    if isinstance(error, BaseExceptionGroup):
        messages = [message for child in error.exceptions for message in exception_messages(child)]
        return [f"{error.message}: {'; '.join(notes)}", *messages] if notes else messages
    detail = f"{type(error).__name__}: {error}"
    return ["; ".join((detail, *notes))]


def run(output, *, execute=False, reviewed_config=None, live=None):
    """Offline review returns settings; execution has no alternate control loop."""
    if reviewed_config is None:
        raise ValueError("An explicit reviewed configuration is required")
    draft, config = load_reviewed(reviewed_config)
    profile = config.b601.control.profiles[0]
    velocity, drive = force_pos_wire_codes(
        profile.velocity_deg_s, profile.torque_ratio, velocity_wire_code=profile.velocity_wire_code
    )
    if not execute:
        return {
            "transport_opened": False,
            "config": draft["config"],
            "velocity_wire_code": velocity,
            "drive_wire_code": drive,
            "hardware_ack": "unknown",
        }
    if draft.get("executable") is not True or draft.get("consumed") is not False:
        raise ValueError("需要本次只读反馈、人工确认及未使用的配置")
    if live is None:
        raise ValueError("Supervised execution requires a live stop control")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = replace(config, trace_root=output / "stream")
    errors = []
    result = {}
    try:
        result = run_runtime(config, live=live)
    except BaseException as exc:
        errors.extend(exception_messages(exc))
    manifest_path = output / "stream/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    shutdown = manifest.get("metadata", {}).get("shutdown_evidence", {})
    errors = list(result.get("errors") or []) + errors
    result.update(
        errors=errors,
        config=draft["config"],
        calibration=draft["calibration"],
        operator_values=draft.get("operator_values", {}),
        operator_toggles=draft.get("operator_toggles", {}),
        display=draft.get("display", {}),
        reference=draft.get("reference"),
        exit_recovery=draft.get("exit_recovery"),
        startup_evidence=result.get("startup_evidence", getattr(live, "startup", None)),
        shutdown_evidence=result.get("shutdown_evidence", shutdown),
        trace_root=str(config.trace_root.resolve()),
        hardware_ack="unknown",
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, reviewed_config=args.reviewed_config), ensure_ascii=False, indent=2))
