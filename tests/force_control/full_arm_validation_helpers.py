"""Synthetic admission artifacts for offline tests only; these are never physical evidence."""

import hashlib
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from lerobot.robots.rebot_b601_follower.arm_acceptance import (
    SingleJointAcceptanceConfig,
    command_profile_digest,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlConfig
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile
from lerobot.robots.rebot_b601_follower.full_arm_validation import (
    REQUIRED_SOURCE_PATHS,
    FullArmValidationConfig,
    MicroTrialBinding,
)
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    return _write(Path(path), value)


class ValidationFiles(SimpleNamespace):
    @property
    def capability(self):
        return self.validation

    def __iter__(self):
        yield self.validation
        yield self.control


def make_full_validation_files(tmp_path):
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    names = tuple(joint.name for joint in DEFAULT_B601_JOINTS)
    port = "FAKE_VALIDATION"
    calibration_path = root / "synthetic_calibration.json"
    _write(
        calibration_path,
        {
            joint.name: {
                "id": joint.motor_id,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": -40.0 if joint.name == "gripper" else -30.0,
                "range_max": 40.0 if joint.name == "gripper" else 30.0,
            }
            for joint in DEFAULT_B601_JOINTS
        },
    )
    calibration = B601ArmCalibration.load(
        calibration_path, {joint.name: joint.motor_id for joint in DEFAULT_B601_JOINTS}
    )
    profiles = tuple(
        B601JointCommandProfile(name, "pos_vel", -20.0, 20.0, 0.5, velocity_deg_s=1.0) for name in names[:-1]
    ) + (
        B601JointCommandProfile(
            "gripper", "force_pos", -20.0, 20.0, 0.4, torque_ratio=0.02, velocity_wire_code=1
        ),
    )
    control = B601ControlConfig(
        profiles,
        1.5,
        0.01,
        0.2,
        0.05,
        "disable_selected",
        names,
        calibration_file=str(calibration_path),
        watchdog_ticks=2000,
    )
    bindings = []
    for profile in profiles[:-1]:
        directory = root / profile.name
        directory.mkdir()
        micro_control = replace(control, profiles=(profile,), fault_joint_names=(profile.name,))
        request = SingleJointAcceptanceConfig(
            str(uuid.uuid4()),
            "synthetic_test_only",
            profile.name,
            str(calibration_path),
            calibration.sha256,
            command_profile_digest(profile),
            True,
            "disable_selected_supported_arm",
            100,
            -0.1,
            1.0,
            2.0,
            5.0,
            2000,
            True,
        )
        reviewed = {
            "port": port,
            "baudrate": 921600,
            "control": asdict(micro_control),
            "acceptance": asdict(request),
        }
        reviewed_path = directory / "reviewed_config.json"
        reviewed_sha = _write(reviewed_path, reviewed)
        disabled = [{"name": name, "status_code": 0, "position_deg": 0.0} for name in names]
        cleanup = {
            "status": "submitted",
            "error": None,
            "all_joints_disabled_observed": True,
            "outstanding_query_accounted": True,
            "per_joint_results": [{"name": profile.name, "status": "submitted"}],
            "post_disable_feedback": {"feedback": disabled},
        }
        backend = {
            "schema": "b601_single_joint_acceptance_v1",
            "request": asdict(request),
            "reference_deg": 0.0,
            "returned_command": True,
            "complete": True,
            "cleanup": cleanup,
            "transport_closed": True,
            "error": None,
            "production_arm_authorized": False,
            "endpoint_position_tolerance_deg": 0.05,
        }
        events = [
            {
                "stage": "startup",
                "receipt": {
                    "kind": "bounded_single_joint_acceptance",
                    "complete": True,
                    "request_session_id": request.session_id,
                    "commands": [
                        {"stage": phase, "status": "submitted"}
                        for phase in (
                            "disabled_target_preload",
                            "enable_selected_only",
                            "enabled_reference_target",
                        )
                    ],
                },
            }
        ]
        for phase, endpoint in (("outbound", -0.1), ("return", 0.0)):
            events.append(
                {
                    "stage": phase,
                    "receipt": {
                        "results": [
                            {
                                "name": profile.name,
                                "status": "success",
                                "candidate_deg": endpoint,
                                "clipped_deg": endpoint,
                                "submitted_deg": endpoint,
                            }
                        ]
                    },
                    "observation": {
                        "feedback": [
                            {
                                "name": name,
                                "status_code": 1 if name == profile.name else 0,
                                "position_deg": endpoint if name == profile.name else 0.0,
                            }
                            for name in names
                        ]
                    },
                }
            )
        events.extend(({"stage": "disable", "receipt": cleanup}, {"stage": "result", **backend}))
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
        )
        source_root = directory / "source_snapshot"
        files = []
        for relative in REQUIRED_SOURCE_PATHS:
            source = source_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            data = b"# Synthetic fixture only; not executed hardware source.\n"
            source.write_bytes(data)
            files.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "mtime_ns": source.stat().st_mtime_ns,
                }
            )
        _write(
            source_root / "manifest.json",
            {"schema": "b601_acceptance_source_snapshot_v1", "complete": True, "files": files},
        )
        result = {
            "schema": "b601_single_joint_acceptance_cli_v1",
            "synthetic_fixture": True,
            **dict.fromkeys(
                (
                    "executed",
                    "complete",
                    "transport_closed",
                    "sensor_closed",
                    "sensor_lease_released",
                    "manual_stop_ready",
                    "escape_hook_closed",
                    "evidence_closed",
                ),
                True,
            ),
            "production_arm_authorized": False,
            "cleanup_pending": False,
            "errors": [],
            "source_snapshot_status": "captured",
            "source_snapshot_manifest": "source_snapshot/manifest.json",
            "backend_result": backend,
            "calibration": {"sha256": calibration.sha256},
            "evidence_rows": len(events),
            "config_sha256": reviewed_sha,
        }
        result_path = directory / "result.json"
        result_sha = _write(result_path, result)
        bindings.append(
            MicroTrialBinding(profile.name, str(result_path), result_sha, str(reviewed_path), reviewed_sha)
        )
    validation = FullArmValidationConfig(
        str(uuid.uuid4()),
        "synthetic_test_only",
        port,
        str(calibration_path),
        calibration.sha256,
        tuple((profile.name, command_profile_digest(profile)) for profile in profiles),
        tuple(bindings),
        True,
        "disable_all_supported_arm",
        100,
        5,
        30,
        2000,
        1.0,
        "f" * 64,
    )
    return ValidationFiles(
        validation=validation,
        control=control,
        calibration_path=calibration_path,
        calibration=calibration,
        profiles=profiles,
        bindings=tuple(bindings),
        port=port,
    )
