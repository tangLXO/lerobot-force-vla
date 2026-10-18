"""Build normal MIT operator settings from the current gripper form without device access."""

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import draccus

from lerobot.force_control.operator_settings import apply_arm_settings
from lerobot.force_control.tuning import build_config, read_settings
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower.full_arm_control_types import leader_configuration_digest
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from lerobot.scripts.lerobot_force_teacher import leader_configuration_projection
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import RebotArm102LeaderConfig


def build_operator_document(
    values,
    enabled,
    calibration: B601GripperCalibration,
    *,
    supported_arm=False,
    output="outputs/operator_teleoperation",
    arm_settings=None,
):
    """Extend the current form with native arm defaults, preserving every gripper setting.

    The caller supplies the already loaded calibration and its existing support
    confirmation. This function reads calibration files only; it does not save
    settings, create output directories, inspect ports or construct devices.
    """
    if type(supported_arm) is not bool:
        raise ValueError("supported_arm must come from the existing confirmation checkbox")
    calibration.require_unchanged()
    arm_calibration = B601ArmCalibration.load(
        calibration.path, {joint.name: joint.motor_id for joint in DEFAULT_B601_JOINTS}
    )
    config = build_config(values, enabled, calibration, output, initial_mode="hybrid")
    arm_defaults = RebotB601FollowerConfig(port=config.b601.port)
    arm_profiles = []
    for index, (name, _, lower, upper) in enumerate(arm_calibration.joints):
        if name == "gripper":
            continue
        native_lower, native_upper = arm_defaults.joint_limits[name]
        lower, upper = max(lower, native_lower), min(upper, native_upper)
        if lower >= upper:
            raise ValueError(f"Official B601 arm limits and calibration have no intersection: {name}")
        arm_profiles.append(
            {
                "name": name,
                "mode": "mit",
                "minimum_deg": lower,
                "maximum_deg": upper,
                "max_relative_deg": None,
                "velocity_deg_s": None,
                "torque_ratio": None,
                "kp": arm_defaults.mit_kp[index],
                "kd": arm_defaults.mit_kd[index],
                "mit_velocity_deg_s": 0.0,
                "mit_torque_Nm": 0.0,
                "velocity_wire_code": None,
            }
        )
    native = RebotArm102LeaderConfig(port=config.leader_port)
    config.leader_joint_ids = native.joint_ids
    config.leader_joint_directions = native.joint_directions
    config.leader_joint_ranges = {
        profile["name"]: [profile["minimum_deg"], profile["maximum_deg"]] for profile in arm_profiles
    }
    profile = config.b601.control.profiles[0]
    config.leader_joint_ranges["gripper"] = [profile.minimum_deg, profile.maximum_deg]
    document = draccus.encode(config)
    names = tuple(joint.name for joint in DEFAULT_B601_JOINTS)
    document["runtime"].update(
        joint_names=list(names),
        arm_intent_required=True,
        handover_tolerance_deg=None,
    )
    hardware = document["b601"]
    hardware["joint_specs"] = [asdict(joint) for joint in DEFAULT_B601_JOINTS]
    hardware["validation"] = None
    hardware["startup_mode"] = "prepare_disabled_operator_teleoperation"
    hardware["operator_teleoperation"] = {
        "session_id": str(uuid.uuid4()),
        "port": config.b601.port,
        "calibration_file": arm_calibration.path,
        "calibration_sha256": arm_calibration.sha256,
        "leader_config_sha256": leader_configuration_digest(leader_configuration_projection(config)),
        "reference_envelope_deg": 100.0,
        "alignment_tolerance_deg": None,
        "preload_tolerance_deg": None,
        "leader_max_age_s": config.leader_max_age_s,
        "max_command_gap_s": None,
        "watchdog_ticks": config.b601.control.watchdog_ticks,
        "supported_arm": supported_arm,
        "stop_policy": "disable_all_supported_arm",
    }
    control = hardware["control"]
    control["profiles"] = arm_profiles + [asdict(profile)]
    control["fault_action"], control["fault_joint_names"] = "disable_selected", list(names)
    minimum_fault = 7 * control["send_budget_s"] + 0.006 + control["feedback_max_age_s"]
    control["fault_budget_s"] = max(control["fault_budget_s"], math.ceil(minimum_fault * 100) / 100)
    apply_arm_settings(document, arm_settings)
    refresh_operator_mapping(document)
    metadata = {
        "software_only": True,
        "hardware_accessed": False,
        "ready_for_motion": False,
        "parameters_source": "current in-memory tuner values and switches",
        "calibration_file": arm_calibration.path,
        "calibration_sha256": arm_calibration.sha256,
        "reference_envelope_source": (
            f"six arm joints: selected startup reference +/-{hardware['operator_teleoperation']['reference_envelope_deg']:g} "
            "degrees intersected with working and calibration bounds"
        ),
        "gripper_parameters_source": "unchanged current Teacher/Safety/FORCE_POS settings",
        "leader_mapping_source": "existing native project mapping; no new hardware verification claimed",
        "arm_control_source": "native B601 DM MIT: position and explicit saved velocity/torque references; both reference defaults remain zero",
        "arm_gains_source": "explicit saved joint gains or RebotB601FollowerConfig.mit_kp/mit_kd defaults; SDK clamps kp to 500 and kd to 5 during encoding",
        "arm_limits_source": "native B601 DM joint_limits intersected with the loaded calibration",
        "body_per_frame_limits": "disabled as native B601 max_relative_target=None",
        "alignment_preload_and_command_gap": "disabled; no corresponding native B601 parameters",
        "duration_and_cumulative_travel": "preserved from the current gripper values and switches",
        "fault_budget": {
            "source": "all-seven fault cleanup budget derived from current per-write and feedback budgets, rounded up to 0.01 s",
            "minimum_s": minimum_fault,
            "selected_s": control["fault_budget_s"],
            "is_session_duration_limit": False,
        },
        "next_physical_step": "user starts the existing tuner entry; assistant does not access hardware",
        "supported_arm_confirmation": supported_arm,
    }
    calibration.require_unchanged()
    arm_calibration.require_unchanged()
    return document, metadata


def build_document(settings_file, *, supported_arm=False):
    """Use saved settings through the same builder as the current in-memory form."""
    settings_file = Path(settings_file).expanduser().resolve()
    settings_sha = hashlib.sha256(settings_file.read_bytes()).hexdigest()
    arm_settings = []
    values, enabled, calibration_path = read_settings(settings_file, on_arm_settings=arm_settings.append)
    calibration = B601GripperCalibration.load(calibration_path)
    document, metadata = build_operator_document(
        values, enabled, calibration, supported_arm=supported_arm, arm_settings=arm_settings[0]
    )
    if hashlib.sha256(settings_file.read_bytes()).hexdigest() != settings_sha:
        raise ValueError("Saved settings changed while creating the draft")
    metadata.update(settings_file=str(settings_file), settings_sha256=settings_sha)
    return document, metadata


def refresh_operator_mapping(document):
    """Keep edited joint working ranges and the pinned native leader mapping identical."""
    document["leader_joint_ranges"] = {
        profile["name"]: [profile["minimum_deg"], profile["maximum_deg"]]
        for profile in document["b601"]["control"]["profiles"]
    }
    config = SimpleNamespace(**document)
    document["b601"]["operator_teleoperation"]["leader_config_sha256"] = leader_configuration_digest(
        leader_configuration_projection(config)
    )


def write_draft(settings_file, output, *, supported_arm=False):
    output = Path(output).expanduser().resolve()
    sources = output.with_suffix(".sources.json")
    if output.exists() or sources.exists():
        raise FileExistsError("Draft and source report must both use new file names")
    document, metadata = build_document(settings_file, supported_arm=supported_arm)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((output, document), (sources, metadata)):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return output, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--supported-arm-confirmed",
        action="store_true",
        help="Include an explicit operator support confirmation; never enables hardware",
    )
    args = parser.parse_args()
    draft, sources = write_draft(args.settings, args.output, supported_arm=args.supported_arm_confirmed)
    print(json.dumps({"draft": str(draft), "sources": str(sources), "hardware_accessed": False}))


if __name__ == "__main__":
    main()
