"""Create a deliberately incomplete full-arm profile; never connect or enable hardware."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import draccus

from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig


def profile_template():
    """Keep physical gains, bounds, mappings, calibration and budgets unconfigured."""
    document = draccus.encode(ForceTeacherRunConfig())
    document.update(
        backend="b601",
        dry_run=False,
        observe_only=False,
        allow_motion=False,
        explicit_arm=False,
        initial_mode="hybrid",
        close_sign=None,
        leader_backend="rebot_102",
        leader_port=None,
        leader_joint_ids={joint.name: None for joint in DEFAULT_B601_JOINTS},
        leader_joint_directions={joint.name: None for joint in DEFAULT_B601_JOINTS},
        leader_joint_ranges={joint.name: None for joint in DEFAULT_B601_JOINTS},
        recording=None,
        duration_s=None,
    )
    document["runtime"].update(
        joint_names=[joint.name for joint in DEFAULT_B601_JOINTS],
        control_hz=100,
        arm_intent_required=True,
        handover_tolerance_deg=None,
    )
    document["b601"] = {
        "port": None,
        "baudrate": None,
        "joint_specs": [asdict(joint) for joint in DEFAULT_B601_JOINTS],
        "startup_mode": "verify_pre_enabled",
        "control": {
            "profiles": [
                {
                    "name": joint.name,
                    "mode": "force_pos" if joint.name == "gripper" else "pos_vel",
                    "minimum_deg": None,
                    "maximum_deg": None,
                    "max_relative_deg": None,
                    "velocity_deg_s": None,
                    **({"torque_ratio": None, "velocity_wire_code": None} if joint.name == "gripper" else {}),
                }
                for joint in DEFAULT_B601_JOINTS
            ],
            "startup_budget_s": None,
            "send_budget_s": None,
            "fault_budget_s": None,
            "feedback_max_age_s": None,
            "fault_action": "disable_selected",
            "fault_joint_names": [joint.name for joint in DEFAULT_B601_JOINTS],
            "calibration_file": None,
        },
    }
    # The installation's required X518 definition must be copied from reviewed
    # local configuration, rather than inventing ports, channels or calibration.
    document["sensors"] = {}
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New JSON path; existing files are never overwritten")
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(profile_template(), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote non-executable template: {args.output.resolve()}")
    print("Fill null fields and required X518 configuration; motion authorization remains false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
