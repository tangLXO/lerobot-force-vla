#!/usr/bin/env python
"""Record query-only seven-joint feedback, RGB video and X518 for inspection.

This entry point has no enable, disable, position or torque command path. It
requires an explicit --observe-only flag and complete configuration/calibration.
It records observations only, with no action or Teacher-success labels.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pyarrow.parquet as pq

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_stream import SensorStreamRecorder
from lerobot.datasets.sensor_transaction import SensorTransaction, _episode_rows
from lerobot.datasets.sensor_verification import resolve_artifact
from lerobot.datasets.sensor_window import SensorStreamReader
from lerobot.force_control.b601_factory import validate_b601_sensor
from lerobot.force_control.demonstration import JOINT_NAMES, CameraCapture, TeacherRecordConfig, _atomic_json
from lerobot.force_control.demonstration_validation import _decode_video
from lerobot.force_control.runtime import RuntimeConfig
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig
from lerobot.sensors import SensorConfig, make_sensors_from_configs


@dataclass(frozen=True)
class ReadonlyCaptureConfig:
    """Query-only acquisition settings, independent of all motion parameters."""

    port: str
    baudrate: int
    calibration_file: str
    sensors: dict[str, SensorConfig]
    startup_budget_s: float = 1.5
    feedback_budget_ms: float = 50.0
    joint_feedback_max_age_ms: float = 50.0
    force_max_age_ms: float = 50.0
    force_instance: str = "gripper_force"

    def validate(self) -> B601ArmCalibration:
        """Read calibration and validate causal input settings without opening I/O."""
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("Readonly port must be a nonempty string")
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise ValueError("Readonly baudrate must be a positive integer")
        if not isinstance(self.calibration_file, str) or not self.calibration_file.strip():
            raise ValueError("Readonly calibration_file must name an existing native calibration")
        for name in (
            "startup_budget_s",
            "feedback_budget_ms",
            "joint_feedback_max_age_ms",
            "force_max_age_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Readonly {name} must be finite and positive")
        if not isinstance(self.force_instance, str) or not self.force_instance:
            raise ValueError("Readonly force_instance must be a nonempty string")
        runtime = RuntimeConfig(force_instance=self.force_instance, joint_names=JOINT_NAMES)
        validate_b601_sensor(self.sensors, runtime)
        return B601ArmCalibration.load(
            self.calibration_file, {joint.name: joint.motor_id for joint in DEFAULT_B601_JOINTS}
        )

    def runtime_view(self) -> SimpleNamespace:
        """Adapt only query settings to the existing shared capture loop."""
        return SimpleNamespace(
            sensors=self.sensors,
            b601=SimpleNamespace(
                port=self.port,
                baudrate=self.baudrate,
                joint_specs=DEFAULT_B601_JOINTS,
                control=SimpleNamespace(startup_budget_s=self.startup_budget_s),
            ),
            runtime=SimpleNamespace(
                joint_names=JOINT_NAMES,
                force_instance=self.force_instance,
                feedback_budget_ms=self.feedback_budget_ms,
                left_feature="left.normal_force",
                right_feature="right.normal_force",
            ),
            safety=SimpleNamespace(
                force_max_age_ms=self.force_max_age_ms,
                joint_feedback_max_age_ms=self.joint_feedback_max_age_ms,
            ),
        )


def _verify(root: Path, uid: str, frames: list[dict]) -> dict:
    """Read-only verification; the Sensor transaction proves closed main files."""
    reader = SensorStreamReader(root, episode_uid=uid, verify="full", cache_mb=0)
    transaction = SensorTransaction.load(root / "meta/sensor_transactions" / f"{uid}.json")
    evidence = transaction.journal["main_evidence"]
    sync = reader.read_sync(episode_uid=uid)
    sources = reader.manifest["frame_view"]["sources"]
    instance = sources[0]["instance"]
    # This diagnostic is finite (60 seconds by default), and keeps only one
    # referenced native sample per retained observation frame during verification.
    wanted = {frame["force_sequence"] for frame in frames}
    raw = {}
    raw_path = resolve_artifact(
        root,
        reader.manifest["storage_layout"]["raw_path_template"].format(instance=instance, episode_uid=uid),
    )
    with raw_path.open("rb") as handle, pq.ParquetFile(handle) as parquet:
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if row["sequence"] in wanted:
                    raw[row["sequence"]] = row
    with closing(_episode_rows(resolve_artifact(root, evidence["data_path"]), 0)) as rows:
        for frame, reference in zip(frames, sync, strict=True):
            main = next(rows, None)
            if main is None or "action" in main or reference["frame_anchor_ns"] != frame["frame_anchor_ns"]:
                raise ValueError("Readonly frame/Sync identity differs")
            selected = raw[frame["force_sequence"]]
            force = [selected["values"][source["feature"]] for source in sources]
            if not np.array_equal(
                np.asarray(main["observation.state"], dtype=np.float32), frame["state_deg"]
            ):
                raise ValueError("Readonly observed joint state differs")
            if not np.array_equal(
                np.asarray(main["observation.tactile"], dtype=np.float32), np.asarray(force, dtype=np.float32)
            ):
                raise ValueError("Readonly tactile differs from selected Raw sample")
            ref = reference["sensors"][instance]
            if ref["sequence"] != selected["sequence"] or not selected["is_valid"]:
                raise ValueError("Readonly selected force reference differs")
            anchor = frame["frame_anchor_ns"]
            if max(selected["timestamp_ns"], selected["arrival_timestamp_ns"]) > anchor:
                raise ValueError("Readonly force selection is noncausal")
            if any(
                max(item["timestamp_ns"], item["arrival_timestamp_ns"]) > anchor
                for item in frame["cameras"].values()
            ):
                raise ValueError("Readonly camera selection is noncausal")
        if next(rows, None) is not None:
            raise ValueError("Readonly main frame count differs")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    for video in evidence["videos"]:
        _decode_video(
            resolve_artifact(root, video["path"]),
            video["from_timestamp"],
            video["to_timestamp"],
            len(frames),
            info["fps"],
        )
    reader.frame_anchor(0, 0)
    return {"valid": True, "frame_count": len(frames), "video_decoded": True, "raw_sync_verified": True}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Existing full-arm ForceTeacherRunConfig JSON")
    source.add_argument("--readonly-config", type=Path, help="Minimal query-only ReadonlyCaptureConfig JSON")
    parser.add_argument(
        "--recording-config", required=True, type=Path, help="TeacherRecordConfig JSON with named cameras"
    )
    parser.add_argument("--root", required=True, type=Path, help="New local Dataset output directory")
    parser.add_argument("--duration-s", type=float, default=60)
    parser.add_argument("--observe-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    if args.root.exists():
        parser.error("--root must be a new directory")
    recording_document = json.loads(args.recording_config.read_text(encoding="utf-8"))
    recording_document.update(root=str(args.root.resolve()), resume=False)
    recording = TeacherRecordConfig(**recording_document)
    recording.validate()
    if args.readonly_config is not None:
        readonly_config = draccus.decode(
            ReadonlyCaptureConfig, json.loads(args.readonly_config.read_text(encoding="utf-8"))
        )
        calibration = readonly_config.validate()
        config = readonly_config.runtime_view()
        preflight = {
            "configuration_valid": True,
            "read_only": True,
            "motion_authorized": False,
            "hardware_ack": "unknown",
            "joint_names": list(JOINT_NAMES),
            "errors": [],
            "calibration": {
                "path": calibration.path,
                "sha256": calibration.sha256,
                "mtime_ns": calibration.mtime_ns,
            },
        }
    else:
        config = draccus.decode(ForceTeacherRunConfig, json.loads(args.config.read_text(encoding="utf-8")))
        if config.backend != "b601" or config.dry_run or config.b601 is None:
            raise ValueError(
                "Readonly hardware capture requires an explicit backend=b601, dry_run=false config"
            )
        if tuple(config.runtime.joint_names) != JOINT_NAMES:
            raise ValueError("Readonly combined capture requires the full seven-joint mapping")
        # Existing full profiles retain their file/config checks without lifting
        # the physical control gate or calling the motion-session validator.
        preflight = config.b601.preflight_report(config.safety, config.runtime)
        if not preflight["configuration_valid"]:
            raise ValueError("Readonly configuration preflight failed: " + "; ".join(preflight["errors"]))
        calibration = config.b601.validate_arm_profile()
        validate_b601_sensor(config.sensors, config.runtime)
    sensors = make_sensors_from_configs(config.sensors)
    sensor = sensors[config.runtime.force_instance]
    camera_capture = CameraCapture(recording)
    device = ReadonlyB601Transport(config.b601.port, config.b601.joint_specs, baudrate=config.b601.baudrate)
    dataset = recorder = None
    uid, frames, status_counts = str(uuid.uuid4()), [], Counter()
    root, report_path = args.root.resolve(), args.root.resolve() / "readonly_capture_report.json"
    report = {
        "read_only": True,
        "motion_attempted": False,
        "action_labels": False,
        "hardware_ack": "unknown",
        "preflight": preflight,
        "complete": False,
        "episode_uid": uid,
        "errors": [],
    }
    feedback_file = None
    try:
        calibration.require_unchanged()
        sensor.connect()
        camera_capture.start()
        device.open()
        mapping = device.read_mapping(max_read_duration_s=config.b601.control.startup_budget_s)
        report["mapping"] = asdict(mapping)
        if not mapping.is_valid:
            raise RuntimeError(f"Readonly mapping query failed: {mapping.error}")
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"{name}.pos" for name in JOINT_NAMES],
            },
            "observation.tactile": {
                "dtype": "float32",
                "shape": (2,),
                "names": [
                    f"sensor.{config.runtime.force_instance}.{feature}"
                    for feature in (config.runtime.left_feature, config.runtime.right_feature)
                ],
            },
        }
        for name, camera in recording.camera_objects().items():
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channels"],
            }
        dataset = LeRobotDataset.create(
            repo_id=recording.repo_id,
            root=root,
            fps=recording.fps,
            features=features,
            robot_type="b601_readonly_capture",
            streaming_encoding=True,
            encoder_threads=2,
            encoder_queue_maxsize=30,
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23, g=2),
        )
        _atomic_json(report_path, report)
        dataset.prepare_episode_recording()
        recorder = SensorStreamRecorder(root, sensors)
        recorder.start_episode(0, episode_uid=uid, dataset=dataset)
        recorder.wait_until_ready()
        (root / "readonly" / uid).mkdir(parents=True)
        feedback_file = (root / "readonly" / uid / "feedback.jsonl").open("x", encoding="utf-8")
        started, deadline = time.perf_counter(), time.perf_counter()
        while time.perf_counter() - started < args.duration_s:
            calibration.require_unchanged()
            recorder.check_health()
            camera_capture.check_health()
            batch = device.read_feedback(
                max_read_duration_s=config.runtime.feedback_budget_ms / 1000, names=JOINT_NAMES
            )
            feedback_file.write(json.dumps(asdict(batch)) + "\n")
            if not batch.is_valid or {item.name for item in batch.feedback} != set(JOINT_NAMES):
                raise RuntimeError(f"Readonly joint query failed: {batch.error}")
            anchor = time.perf_counter_ns()
            sample = sensor.read_latest_before(anchor, max_age_ms=config.safety.force_max_age_ms)
            by_name = {joint.name: joint for joint in batch.feedback}
            if any(
                not 0 <= anchor - joint.receive_timestamp_ns <= config.safety.joint_feedback_max_age_ms * 1e6
                for joint in batch.feedback
            ):
                raise RuntimeError("Readonly joint feedback is stale")
            state = np.asarray([by_name[name].position_deg for name in JOINT_NAMES], dtype=np.float32)
            frame = {
                "observation.state": state,
                "observation.tactile": np.asarray(
                    [sample.values[config.runtime.left_feature], sample.values[config.runtime.right_feature]],
                    dtype=np.float32,
                ),
                "task": recording.task,
            }
            evidence = {
                "frame_index": len(frames),
                "frame_anchor_ns": anchor,
                "state_deg": state.tolist(),
                "force_sequence": sample.sequence,
                "cameras": {},
            }
            for name, buffer in camera_capture.buffers.items():
                selected = buffer.select(anchor, recording.camera_max_age_ms)
                frame[f"observation.images.{name}"] = selected.image
                evidence["cameras"][name] = {
                    "timestamp_ns": selected.timestamp_ns,
                    "arrival_timestamp_ns": selected.arrival_timestamp_ns,
                }
            capture = {
                "observation_start_ns": batch.started_ns,
                "frame_anchor_ns": anchor,
                "observation_complete_ns": time.perf_counter_ns(),
                "hardware_observation_timestamps": None,
                "sensors": {
                    config.runtime.force_instance: {
                        "timestamp_ns": sample.timestamp_ns,
                        "arrival_timestamp_ns": sample.arrival_timestamp_ns,
                        "sequence": sample.sequence,
                        "hardware_sequence": sample.hardware_sequence,
                        "age_ns": anchor - sample.timestamp_ns,
                        "status": sample.status,
                    }
                },
            }
            dataset.add_frame(frame)
            recorder.record_sync(None, capture)
            frames.append(evidence)
            status_counts.update(joint.status_name for joint in batch.feedback)
            deadline += 1 / recording.fps
            now = time.perf_counter()
            if now >= deadline:
                raise RuntimeError("Readonly capture exceeded its configured frame slot")
            time.sleep(deadline - now)
        recorder.save_episode(dataset, task_info=[recording.task])
        report["capture_duration_s"] = time.perf_counter() - started
    except BaseException as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if feedback_file is not None:
            try:
                feedback_file.close()
            except Exception as exc:
                report["errors"].append(f"Feedback evidence cleanup: {exc}")
        if device.has_resources:
            try:
                device.close()
            except Exception as exc:
                report["errors"].append(f"Readonly transport cleanup: {exc}")
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                report["errors"].append(f"Sensor writer cleanup: {exc}")
                # A live recorder retains the Sensor and Dataset dependencies.
                _atomic_json(report_path, report)
                raise
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception as exc:
                report["errors"].append(f"Dataset cleanup: {exc}")
        try:
            camera_capture.close()
        except Exception as exc:
            report["errors"].append(f"Camera cleanup: {exc}")
        if sensor.has_resources:
            try:
                sensor.disconnect()
            except Exception as exc:
                report["errors"].append(f"Sensor cleanup: {exc}")
    report.update(
        frame_count=len(frames),
        observed_joint_status_counts=dict(status_counts),
        transport_closed=not device.has_resources,
        sensor_closed=not sensor.has_resources,
    )
    if not report["errors"]:
        try:
            report["verification"] = _verify(root, uid, frames)
            report["complete"] = True
        except Exception as exc:
            report["errors"].append(f"Verification: {type(exc).__name__}: {exc}")
    if root.exists():
        _atomic_json(root / "readonly" / uid / "frames.json", frames)
        _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
