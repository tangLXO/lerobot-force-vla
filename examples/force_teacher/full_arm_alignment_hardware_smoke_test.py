#!/usr/bin/env python
"""Query seven-joint leader/follower alignment only; never configure or move either arm."""

import argparse
import json
import math
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus

from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_state import read_control_state
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig

NAMES = tuple(joint.name for joint in DEFAULT_B601_JOINTS)
CLOSE_BUDGET_S = 2.0
_RETAINED_READONLY = []


@dataclass(frozen=True)
class AlignmentConfig:
    follower_port: str
    follower_baudrate: int
    calibration_file: str
    leader_port: str
    leader_baudrate: int
    leader_joint_ids: dict[str, int]
    leader_joint_directions: dict[str, int]
    leader_joint_ranges: dict[str, tuple[float, float]]
    leader_readonly_transport: str
    tolerance_deg: float
    relative_bound_deg: float
    read_budget_s: float
    leader_max_age_s: float
    setup_query_budget_s: float

    def validate(self):
        for name in ("follower_port", "leader_port", "calibration_file"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be explicit")
        if self.follower_port.strip().casefold() == self.leader_port.strip().casefold():
            raise ValueError("Leader and follower must use different explicit ports")
        for name in ("follower_baudrate", "leader_baudrate"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, maximum in (
            ("tolerance_deg", 5),
            ("relative_bound_deg", 100),
            ("read_budget_s", 0.05),
            ("leader_max_age_s", 0.15),
            ("setup_query_budget_s", 5),
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be positive and <= {maximum}")
        if self.read_budget_s >= self.leader_max_age_s:
            raise ValueError("Leader age must leave margin for the following follower query")
        if self.leader_readonly_transport != "protocol_v1":
            raise ValueError("Alignment only permits the audited query-only protocol_v1 leader")
        for mapping in (self.leader_joint_ids, self.leader_joint_directions, self.leader_joint_ranges):
            if set(mapping) != set(NAMES):
                raise ValueError("Leader mapping must explicitly cover all seven standard joint names")
        ids = tuple(self.leader_joint_ids.values())
        if len(set(ids)) != 7 or any(type(value) is not int or not 0 <= value <= 253 for value in ids):
            raise ValueError("Leader IDs must be explicit and unique within 0..253")
        if any(type(value) is not int or value == 0 for value in self.leader_joint_directions.values()):
            raise ValueError("Leader directions/scales must be explicit nonzero integers")
        calibration = B601ArmCalibration.load(
            self.calibration_file, {joint.name: joint.motor_id for joint in DEFAULT_B601_JOINTS}
        )
        for name, bounds in self.leader_joint_ranges.items():
            if len(bounds) != 2 or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in bounds
            ):
                raise ValueError(f"Leader {name} requires finite range endpoints")
            calibration.require_bounds(name, *bounds)
        return calibration


def alignment_rows(config, calibration, leader, follower, compared_ns):
    raw, mapped = dict(leader.raw_action_deg), dict(leader.action_deg)
    expected = {f"{name}.pos" for name in NAMES}
    if set(raw) != expected or set(mapped) != expected or len(follower.feedback) != 7:
        raise ValueError("Alignment needs complete seven-joint raw/mapped leader and follower feedback")
    if (
        not 0
        < leader.request_started_ns
        <= leader.response_completed_ns
        <= follower.started_ns
        <= compared_ns
    ):
        raise ValueError(
            "Alignment snapshots are noncausal or out of the required leader-then-follower order"
        )
    if compared_ns - leader.request_started_ns > config.leader_max_age_s * 1e9:
        raise ValueError("Leader snapshot is stale at comparison")
    by_name = {joint.name: joint for joint in follower.feedback}
    if set(by_name) != set(NAMES):
        raise ValueError("Follower snapshot must contain exactly the seven standard joints")
    bounds = {name: (lower, upper) for name, _, lower, upper in calibration.joints}
    result = []
    for name in NAMES:
        joint, key = by_name[name], f"{name}.pos"
        if (
            not 0 < joint.receive_timestamp_ns <= compared_ns
            or compared_ns - joint.receive_timestamp_ns > config.read_budget_s * 1e9
        ):
            raise ValueError(f"Follower feedback is stale/future: {name}")
        values = [float(raw[key]), float(mapped[key]), float(joint.position_deg)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Alignment angle is nonfinite: {name}")
        difference = values[1] - values[2]
        result.append(
            {
                "name": name,
                "motor_id": joint.motor_id,
                "status_code": joint.status_code,
                "leader_raw_deg": values[0],
                "leader_mapped_deg": values[1],
                "follower_deg": values[2],
                "difference_deg": difference,
                "absolute_difference_deg": abs(difference),
                "within_tolerance": abs(difference) <= config.tolerance_deg,
                "within_relative_bound": abs(difference) <= config.relative_bound_deg,
                "follower_within_calibration": bounds[name][0] <= values[2] <= bounds[name][1],
                "follower_received_ns": joint.receive_timestamp_ns,
                "follower_age_ms": (compared_ns - joint.receive_timestamp_ns) / 1e6,
            }
        )
    return result


def run(config_file, output):
    config = draccus.decode(AlignmentConfig, json.loads(Path(config_file).read_text(encoding="utf-8")))
    calibration = config.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "b601_full_arm_alignment_readonly_v1",
        "read_only": True,
        "motion_attempted": False,
        "hardware_ack": "unknown",
        "production_arm_authorized": False,
        "complete": False,
        "alignment_within_tolerance": False,
        "ready_for_bounded_start": False,
        "ready_for_bounded_start_scope": "Alignment/mapped-angle tolerance, mode/watchdog and disabled-state checks only; excludes capability admission, X518 readiness and motion authorization.",
        "motion_authorized": False,
        "errors": [],
        "config": draccus.encode(config),
        "calibration": asdict(calibration),
        "joints": [],
    }
    follower = ReadonlyB601Transport(
        config.follower_port, DEFAULT_B601_JOINTS, baudrate=config.follower_baudrate
    )
    leader = RebotArm102Leader(
        RebotArm102LeaderTeleopConfig(
            port=config.leader_port,
            baudrate=config.leader_baudrate,
            readonly_transport="protocol_v1",
            joint_ids={name: config.leader_joint_ids[name] for name in NAMES},
            joint_directions={name: config.leader_joint_directions[name] for name in NAMES},
            joint_ranges={name: list(config.leader_joint_ranges[name]) for name in NAMES},
            id="alignment_query_only",
            calibration_dir=output / "leader_host_metadata",
        )
    )
    try:
        calibration.require_unchanged()
        follower.open()
        leader.connect_readonly()
        mapping = follower.read_mapping(max_read_duration_s=config.setup_query_budget_s)
        result["mapping"] = asdict(mapping)
        if not mapping.is_valid:
            raise RuntimeError(f"Follower mapping query failed: {mapping.error}")
        state = read_control_state(follower, max_read_duration_s=config.setup_query_budget_s)
        result["control_state"] = asdict(state)
        if not state.is_valid:
            raise RuntimeError(f"Control-state query failed: {state.error}")
        registers = {(item.name, item.register_id): item.value for item in state.registers}
        result["control_state_expected"] = all(
            registers.get((name, 10)) == (4 if name == "gripper" else 2) and registers.get((name, 9)) == 2000
            for name in NAMES
        )
        snapshot = leader.read_action_snapshot(max_read_duration_s=config.read_budget_s)
        feedback = follower.read_feedback(max_read_duration_s=config.read_budget_s)
        result["leader"] = asdict(snapshot)
        result["follower"] = asdict(feedback)
        if not feedback.is_valid:
            raise RuntimeError(f"Follower feedback query failed: {feedback.error}")
        compared = time.perf_counter_ns()
        result["compared_ns"] = compared
        result["leader_age_ms"] = (compared - snapshot.request_started_ns) / 1e6
        result["joints"] = alignment_rows(config, calibration, snapshot, feedback, compared)
        for row in result["joints"]:
            row["control_mode"] = registers.get((row["name"], 10))
            row["watchdog_ticks"] = registers.get((row["name"], 9))
        calibration.require_unchanged()
        result["alignment_within_tolerance"] = all(row["within_tolerance"] for row in result["joints"])
        result["all_seven_disabled"] = all(row["status_code"] == 0 for row in result["joints"])
        result["ready_for_bounded_start"] = bool(
            result["alignment_within_tolerance"]
            and result["all_seven_disabled"]
            and result["control_state_expected"]
            and all(
                row["follower_within_calibration"] and row["within_relative_bound"]
                for row in result["joints"]
            )
        )
        result["complete"] = True
    except BaseException as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_deadline = time.monotonic() + CLOSE_BUDGET_S
        result["resources_retained"] = []

        def persist():
            try:
                (output / "alignment.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
            except OSError as exc:
                message = f"report write: {type(exc).__name__}: {exc}"
                if message not in result["errors"]:
                    result["errors"].append(message)

        for name, resource, close in (
            ("leader", leader, leader.disconnect),
            ("follower", follower, follower.close),
        ):
            while resource.has_resources:
                try:
                    close()
                except BaseException as exc:
                    message = f"{name} close: {type(exc).__name__}: {exc}"
                    if message not in result["errors"]:
                        result["errors"].append(message)
                if resource.has_resources:
                    result["cleanup_pending"] = True
                    result[f"{name}_closed"] = False
                    persist()
                    if time.monotonic() >= cleanup_deadline:
                        result["errors"].append(
                            f"{name} close budget expired; resources retained until process exit"
                        )
                        result["resources_retained"].append(name)
                        _RETAINED_READONLY.append(resource)
                        break
                    with suppress(KeyboardInterrupt):
                        time.sleep(min(0.05, max(0, cleanup_deadline - time.monotonic())))
        result["leader_closed"] = not leader.has_resources
        result["follower_closed"] = not follower.has_resources
        result["cleanup_pending"] = bool(result["resources_retained"])
        if result["errors"]:
            result["complete"] = result["ready_for_bounded_start"] = False
        persist()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args(argv)
    result = run(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
