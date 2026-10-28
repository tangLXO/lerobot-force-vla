"""Asynchronous Teacher demonstrations using completed owner ticks and Sidecar v2.

This module never connects to a robot or sends a motor command. The owner supplies
an indivisible completed decision, including its selected force and send receipts.
Camera capture, encoding, disk writes and episode finalization run outside it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import queue
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

from .owner import OwnerService, OwnerSnapshot

if TYPE_CHECKING or (_datasets_available and _av_available):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sensor_stream import (
        SensorStreamRecorder,
        build_sensor_stream_manifest,
        preflight_sensor_stream_writer,
    )

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
EPISODE_SCHEMA = "teacher_capture_episode_v1"
RESULTS = ("unknown", "success", "failure")


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported demonstration value: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherRecordConfig:
    """Explicit local recording settings; construction does not open any device."""

    root: str
    task: str
    repo_id: str = "local/force_teacher_demo"
    object_id: str = "unspecified"
    contact_condition_id: str = "unspecified"
    fps: int = 50
    camera_configs: dict[str, dict] = field(default_factory=dict)
    camera_max_age_ms: float = 100.0
    queue_capacity: int = 128
    resume: bool = False
    target_episode_count: int = 10
    h264_encoder: str = "nvenc"

    def video_encoder(self) -> RGBEncoderConfig:
        """Select one explicit encoder; unavailable NVENC never silently falls back."""
        if self.h264_encoder == "nvenc":
            return RGBEncoderConfig(vcodec="h264_nvenc", preset="p4", crf=23, g=2, extra_options={"bf": 0})
        return RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23, g=2)

    def validate(self) -> None:
        """Check recording schema, limits and camera intent without hardware I/O."""
        for name in ("root", "task", "repo_id", "object_id", "contact_condition_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"recording.{name} must be nonempty")
        for name in ("fps", "queue_capacity", "target_episode_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"recording.{name} must be a positive integer")
        if self.fps != 50:
            raise ValueError("Teacher capture requires 50 Hz control and main data")
        if self.h264_encoder not in ("nvenc", "software"):
            raise ValueError("h264_encoder must be nvenc or software")
        if (
            isinstance(self.camera_max_age_ms, bool)
            or not isinstance(self.camera_max_age_ms, (int, float))
            or not math.isfinite(self.camera_max_age_ms)
            or self.camera_max_age_ms <= 0
        ):
            raise ValueError("camera_max_age_ms must be finite and positive")
        if not isinstance(self.resume, bool):
            raise ValueError("recording.resume must be boolean")
        self.camera_objects()

    def camera_objects(self) -> dict[str, OpenCVCameraConfig]:
        """Decode the explicitly configured, timestamp-capable RGB cameras."""
        if not isinstance(self.camera_configs, dict) or not 1 <= len(self.camera_configs) <= 4:
            raise ValueError("Configure one to four named RGB OpenCV cameras")
        configs = {}
        identities = set()
        for name, raw in self.camera_configs.items():
            if not isinstance(name, str) or not name.isidentifier() or not isinstance(raw, dict):
                raise ValueError("Camera entries require bare names and configuration objects")
            values = dict(raw)
            if values.pop("type", "opencv") != "opencv":
                raise ValueError("Teacher recording currently supports timestamped OpenCV RGB cameras")
            config = OpenCVCameraConfig(**values)
            if config.device_path is not None:
                if config.device_path in identities:
                    raise ValueError("Camera roles must not select the same device path")
                identities.add(config.device_path)
            if str(config.color_mode.value) != "rgb":
                raise ValueError("Teacher recording cameras must use RGB color_mode")
            for dimension in (config.width, config.height):
                if type(dimension) is not int or dimension <= 0:
                    raise ValueError("Camera width and height must be explicit positive integers")
            if config.fps is None or config.fps < self.fps:
                raise ValueError("Camera capture fps must be at least the Dataset fps")
            configs[name] = config
        return configs


@dataclass(frozen=True)
class CameraFrame:
    """RGB pixels and their host capture/availability times in the owner clock domain."""

    image: np.ndarray
    timestamp_ns: int
    arrival_timestamp_ns: int


class CausalCameraBuffer:
    """Bounded camera history; late or future frames cannot enter an earlier tick."""

    def __init__(self, capacity: int = 16):
        """Allocate only a bounded number of retained frame references."""
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Camera history capacity must be positive")
        self._frames: deque[CameraFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.count = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None

    def publish(self, image: np.ndarray, timestamp_ns: int, arrival_timestamp_ns: int) -> None:
        """Retain a detached new capture once, preserving availability time."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Camera must publish RGB uint8[height,width,3]")
        if not 0 < timestamp_ns <= arrival_timestamp_ns:
            raise ValueError("Invalid camera capture/arrival time")
        with self._lock:
            if self.last_ns is not None and timestamp_ns <= self.last_ns:
                return
            self._frames.append(CameraFrame(image.copy(), timestamp_ns, arrival_timestamp_ns))
            self.count += 1
            self.first_ns = timestamp_ns if self.first_ns is None else self.first_ns
            self.last_ns = timestamp_ns

    def select(self, anchor_ns: int, max_age_ms: float) -> CameraFrame:
        """Return the most recent available capture at or before this anchor."""
        with self._lock:
            for frame in reversed(self._frames):
                if (
                    frame.timestamp_ns <= anchor_ns
                    and frame.arrival_timestamp_ns <= anchor_ns
                    and anchor_ns - frame.timestamp_ns <= max_age_ms * 1e6
                ):
                    return frame
        raise TimeoutError("No fresh causal camera frame for the completed owner tick")

    def latest(self) -> CameraFrame | None:
        """Return the latest frame for preview, separate from causal selection."""
        with self._lock:
            return self._frames[-1] if self._frames else None


class CameraCapture:
    """Own camera resources off the motor owner; publish bounded causal RGB histories."""

    def __init__(self, config: TeacherRecordConfig):
        """Construct disconnected cameras and bounded histories."""
        self.config = config
        self.cameras = make_cameras_from_configs(config.camera_objects())
        capacity = max(16, math.ceil(config.camera_max_age_ms * config.fps / 1000) + 4)
        self.buffers = {name: CausalCameraBuffer(capacity) for name in self.cameras}
        self.stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.bindings: dict[str, dict] = {}
        self._sequences: dict[str, int] = {}

    def start(self) -> None:
        """Connect cameras outside the owner and await the first captures."""
        try:
            for name, camera in self.cameras.items():
                if camera.config.device_path is None:
                    raise ValueError(
                        "Teacher hardware cameras require a stable device_path, not a saved index"
                    )
                camera.connect()
                self.bindings[name] = {
                    "device_path": camera.config.device_path,
                    "resolved_index": camera.index_or_path,
                    "backend": int(camera.backend),
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                }
            self._thread = threading.Thread(target=self._run, name="teacher-camera-cache", daemon=False)
            self._thread.start()
            deadline = time.monotonic() + 5
            while any(buffer.latest() is None for buffer in self.buffers.values()):
                if self.error is not None:
                    raise RuntimeError(f"Camera capture failed: {self.error}") from self.error
                if time.monotonic() >= deadline:
                    raise TimeoutError("Cameras did not publish valid frames before recording")
                time.sleep(0.005)
        except BaseException:
            self.close()
            raise

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                for name, camera in self.cameras.items():
                    captured = camera.read_latest_snapshot(self._sequences.get(name))
                    if captured is not None:
                        self._sequences[name] = captured.sequence
                        self.buffers[name].publish(
                            captured.image, captured.timestamp_ns, captured.arrival_timestamp_ns
                        )
                self.stop.wait(0.002)
        except BaseException as exc:
            self.error = exc

    def check_health(self) -> None:
        """Surface capture-thread failures to the recording worker."""
        if self.error is not None:
            raise RuntimeError(f"Camera capture failed: {self.error}") from self.error

    def close(self) -> None:
        """Stop cache publication and release camera resources boundedly."""
        self.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise TimeoutError("Camera cache shutdown unconfirmed; resources retained")
        failures = []
        for camera in self.cameras.values():
            if camera.has_resources:
                try:
                    camera.disconnect()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise ExceptionGroup("Camera cleanup failed", failures)


def completed_tick_frame(
    snapshot: OwnerSnapshot,
    cameras: dict[str, CausalCameraBuffer],
    config: TeacherRecordConfig,
    *,
    force_instance: str = "gripper_force",
    frame_index: int = 0,
) -> tuple[dict, dict, dict]:
    """Pack observed state, selected force and actual submissions from one complete tick."""
    row = snapshot.row
    if set(cameras) != set(config.camera_configs):
        raise ValueError("Causal camera buffers must cover exactly the configured cameras")
    if row.get("send_status") != "submitted" or row.get("phase") == "FAULT":
        raise ValueError("A main frame requires a fully submitted non-faulted owner tick")
    if not row.get("feedback_valid") or not row.get("force_valid"):
        raise ValueError("A main frame requires valid observed joints and selected force")
    anchor = row["decision_anchor_ns"]
    if not isinstance(anchor, int) or anchor <= 0:
        raise ValueError("Missing decision anchor")
    force_timestamp = row["force_timestamp_ns"]
    arrival = row["force_arrival_timestamp_ns"]
    if not 0 < force_timestamp <= anchor or not 0 < arrival <= anchor:
        raise ValueError("Selected force is not causal")
    if anchor - force_timestamp > row["force_max_age_ms"] * 1e6:
        raise ValueError("Selected force is stale")
    observed = {item.name: item for item in snapshot.joint_snapshot.joints}
    receipts = {item["name"]: item for item in row.get("per_joint_results", [])}
    if (
        set(observed) != set(JOINT_NAMES)
        or set(receipts) != set(JOINT_NAMES)
        or len(snapshot.joint_snapshot.joints) != 7
        or len(row.get("per_joint_results", [])) != 7
    ):
        raise ValueError("Complete seven-joint observed state and submission receipts are required")
    state, action = [], []
    for name in JOINT_NAMES:
        joint = observed[name]
        receipt = receipts[name]
        if (
            not joint.is_valid
            or not 0 <= anchor - joint.receive_timestamp_ns <= row["joint_feedback_max_age_ms"] * 1e6
            or receipt.get("status") != "success"
            or receipt.get("submitted_deg") is None
        ):
            raise ValueError(f"Incomplete or stale joint evidence: {name}")
        state.append(float(joint.position_deg))
        action.append(float(receipt["submitted_deg"]))
    if not all(
        math.isfinite(value) for value in [*state, *action, row["force_left_N"], row["force_right_N"]]
    ):
        raise ValueError("Nonfinite observation or submitted action")
    tactile = [row["force_left_N"], row["force_right_N"]]
    target = row.get("force_target_effective_N")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
        raise ValueError("A main frame requires the effective force target in N")
    frame = {
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.tactile": np.asarray(tactile, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "observation.force_target": np.asarray([target], dtype=np.float32),
        "task": config.task,
    }
    camera_metadata = {}
    for name, buffer in cameras.items():
        selected = buffer.select(anchor, config.camera_max_age_ms)
        frame[f"observation.images.{name}"] = selected.image
        camera_metadata[name] = {
            "timestamp_ns": selected.timestamp_ns,
            "arrival_timestamp_ns": selected.arrival_timestamp_ns,
        }
    sensor_metadata = {
        "timestamp_ns": force_timestamp,
        "arrival_timestamp_ns": arrival,
        "sequence": row["force_sequence"],
        "hardware_sequence": row.get("force_hardware_sequence"),
        "age_ns": anchor - force_timestamp,
        "status": row.get("force_status", "ok"),
    }
    capture = {
        "observation_start_ns": row["tick_start_ns"],
        "frame_anchor_ns": anchor,
        "observation_complete_ns": row["send_end_ns"],
        "hardware_observation_timestamps": None,
        "sensors": {force_instance: sensor_metadata},
    }
    evidence = {
        "frame_index": frame_index,
        "control_tick_id": row["tick_id"],
        "decision_anchor_ns": anchor,
        "epoch": snapshot.epoch,
        "state_deg": state,
        "action_deg": action,
        "tactile_N": tactile,
        "force_target_N": target,
        "teacher_phase": row["phase"],
        "action_valid": row.get("action_label_valid") is True,
        "force_sequence": row["force_sequence"],
        "force_timestamp_ns": force_timestamp,
        "force_arrival_timestamp_ns": arrival,
        "cameras": camera_metadata,
        "send_start_ns": row["send_start_ns"],
        "send_end_ns": row["send_end_ns"],
    }
    return frame, capture, evidence


class DemonstrationController:
    """Record/save/discard commands never perform motor I/O or wait on the owner thread."""

    def __init__(
        self,
        owner: OwnerService,
        sensor: Any,
        *,
        force_instance: str = "gripper_force",
        provenance: dict | None = None,
        camera_factory=CameraCapture,
        dataset_factory=None,
        recorder_factory=None,
    ):
        """Start an idle worker without touching robot, Sensor or camera transports."""
        self.owner, self.sensor, self.force_instance = owner, sensor, force_instance
        self.provenance = copy.deepcopy(provenance or {})
        from .capture_contract import capture_source

        self.provenance["source"] = capture_source()
        self._camera_factory = camera_factory
        self._dataset_factory = dataset_factory
        self._recorder_factory = recorder_factory
        self._commands: queue.Queue[tuple[str, dict]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status: dict = {
            "state": "idle",
            "active": False,
            "episode_count": 0,
            "successful_episode_count": 0,
            "frame_count": 0,
            "error": None,
        }
        self._config: TeacherRecordConfig | None = None
        self._dataset = None
        self._recorder = None
        self._cameras = None
        self._subscription = None
        self._control_file = self._frames_file = None
        self._uid: str | None = None
        self._episode_index = 0
        self._frame_count = self._control_count = 0
        self._stages: Counter = Counter()
        self._force_labels = self._action_labels = 0
        self._normal_release = False
        self._first_anchor: int | None = None
        self._last_anchor: int | None = None
        self._first_control: int | None = None
        self._last_control: int | None = None
        self._last_tick_id: int | None = None
        self._epoch: int | None = None
        self._task_result = "unknown"
        self._start_sequence = 0
        self._cleanup_pending = False
        self._timing_misses = 0
        self._maximum_interval_ns = 0
        self._started_utc: str | None = None
        self._save_evidence: dict = {}
        self._thread = threading.Thread(target=self._run, name="teacher-demonstrations", daemon=False)
        self._thread.start()

    @property
    def has_resources(self) -> bool:
        """Retain Sensor dependencies after an unconfirmed writer/camera shutdown."""
        return self._thread.is_alive() or self._cleanup_pending

    def status(self) -> dict:
        """Return detached recording state for bounded GUI telemetry."""
        with self._lock:
            return copy.deepcopy(self._status)

    def prepare_before_control(self, config: TeacherRecordConfig) -> None:
        """Open capture resources synchronously before the motor owner starts.

        Only the session startup caller may use this method, before dispatching
        recording events. Device enumeration and first codec/image imports can
        hold the interpreter long enough to starve an already active controller.
        """
        if self.owner.is_running or self.owner.has_resources or self._uid is not None:
            raise RuntimeError("Capture startup preparation requires an unopened motor owner")
        config.validate()
        self._prepare(config)
        self._dataset.prepare_episode_recording()
        self.preview()
        self._update(state="ready", active=False)

    def _update(self, **values) -> None:
        with self._lock:
            self._status.update(values)

    def submit(self, kind: str, value: dict | None = None) -> None:
        """Enqueue recording intent without sending a motor command."""
        if kind not in ("record_start", "record_finish", "record_discard", "record_outcome"):
            raise ValueError("Unsupported recording event")
        if self._stop.is_set():
            raise RuntimeError("Demonstration controller is stopping")
        try:
            self._commands.put_nowait((kind, copy.deepcopy(value or {})))
        except queue.Full as exc:
            raise RuntimeError("Recording event queue is full") from exc

    def close(self, timeout_s: float = 30) -> None:
        """Interrupt an active episode, then await resource cleanup."""
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("Recording shutdown unconfirmed; sensor/files ownership retained")
        if self._cleanup_pending:
            self._cleanup_resources()
        if self._cleanup_pending:
            raise RuntimeError("Recording resources remain retained after cleanup failure")

    def preview(self) -> dict:
        """Encode small latest previews away from the hardware owner thread."""
        if self._cameras is None:
            return {}
        # Pillow is already a base image dependency; JPEG encoding stays off the owner.
        import io

        from PIL import Image

        result = {}
        for name, buffer in self._cameras.buffers.items():
            frame = buffer.latest()
            if frame is not None:
                image = Image.fromarray(frame.image)
                image.thumbnail((320, 240))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=65)
                result[name] = {"jpeg": encoded.getvalue(), "timestamp_ns": frame.timestamp_ns}
        return result

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    kind, value = self._commands.get(timeout=0.002 if self._subscription else 0.05)
                except queue.Empty:
                    kind = None
                    value = {}
                if kind is not None:
                    try:
                        self._validate_request(kind, value)
                    except (ValueError, TypeError, RuntimeError) as exc:
                        self._update(request_error=f"{type(exc).__name__}: {exc}")
                        kind = None
                try:
                    if kind == "record_start":
                        self._start(TeacherRecordConfig(**value["config"]))
                    elif kind == "record_finish":
                        self._finish(value.get("task_result", "unknown"))
                    elif kind == "record_discard":
                        config = self._config
                        self._abort("Discarded by operator", status="discarded")
                        if value.get("rerecord") is True and config is not None:
                            self._start(config)
                    elif kind == "record_outcome":
                        self._set_outcome(value.get("task_result", "unknown"))
                    if self._subscription is not None:
                        self._subscription.check_health()
                        self._cameras.check_health()
                        self._recorder.check_health()
                        # Bound each drain so stop and finish remain responsive.
                        for _ in range(16):
                            try:
                                snapshot = self._subscription.queue.get_nowait()
                            except queue.Empty:
                                break
                            self._consume(snapshot)
                except Exception as exc:
                    if self._uid is not None:
                        try:
                            self._abort(str(exc), status="error")
                        except Exception as cleanup:
                            self._update(cleanup_error=str(cleanup))
                    self._update(state="error", active=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            failures = []
            if self._uid is not None:
                try:
                    self._abort("Session stopped or interrupted", status="interrupted")
                except Exception as exc:
                    failures.append(str(exc))
            self._cleanup_resources()
            if self._cleanup_pending:
                failures.append("Recording cleanup unconfirmed; dependencies retained")
            self._update(
                state="closed" if not failures else "error",
                active=False,
                cleanup_error="; ".join(failures) or None,
            )

    def _validate_request(self, kind: str, value: dict) -> None:
        if not isinstance(value, dict):
            raise ValueError("Recording events require an object value")
        if kind == "record_start":
            if self._uid is not None or self._cleanup_pending:
                raise RuntimeError("An episode or retained cleanup already owns the recorder")
            if not isinstance(value.get("config"), dict):
                raise ValueError("record_start requires an explicit recording configuration")
            TeacherRecordConfig(**value["config"]).validate()
        elif kind in ("record_finish", "record_discard"):
            if self._uid is None or self._subscription is None:
                raise RuntimeError("No demonstration is recording")
            if kind == "record_finish" and value.get("task_result", "unknown") not in RESULTS:
                raise ValueError("Invalid manual task result")
        elif kind == "record_outcome":
            if value.get("task_result", "unknown") not in RESULTS:
                raise ValueError("Invalid manual task result")
            if self._uid is None and not self.status().get("file_complete"):
                raise RuntimeError("No demonstration is available for manual task review")

    def _cleanup_resources(self) -> None:
        failures = []
        # A retained Sensor worker must stop before Dataset buffers are finalized.
        for name in ("_recorder", "_dataset", "_cameras"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    if name == "_dataset":
                        resource.finalize()
                    else:
                        resource.close()
                    setattr(self, name, None)
                except Exception as exc:
                    failures.append(str(exc))
                    if name == "_recorder":
                        break
        self._cleanup_pending = bool(failures)
        if failures:
            self._update(cleanup_error="; ".join(failures))

    def _features(self, config: TeacherRecordConfig) -> dict:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"{n}.pos" for n in JOINT_NAMES],
            },
            "action": {"dtype": "float32", "shape": (7,), "names": [f"{n}.pos" for n in JOINT_NAMES]},
            "observation.force_target": {"dtype": "float32", "shape": (1,), "names": ["mean_normal_force"]},
            "observation.tactile": {
                "dtype": "float32",
                "shape": (2,),
                "names": [
                    f"sensor.{self.force_instance}.left.normal_force",
                    f"sensor.{self.force_instance}.right.normal_force",
                ],
            },
        }
        for name, camera in config.camera_objects().items():
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _prepare(self, config: TeacherRecordConfig) -> None:
        require_package("datasets", extra="dataset")
        require_package("av", extra="dataset")
        root = Path(config.root).resolve()
        if self._dataset is not None:
            previous = self._config
            if root != Path(previous.root).resolve() or (
                config.repo_id,
                config.fps,
                config.camera_configs,
                config.camera_max_age_ms,
                config.h264_encoder,
            ) != (
                previous.repo_id,
                previous.fps,
                previous.camera_configs,
                previous.camera_max_age_ms,
                previous.h264_encoder,
            ):
                raise ValueError("Dataset root, fps and camera schema are fixed for this session")
            return
        self._update(state="preparing", error=None)
        cameras = self._camera_factory(config)
        try:
            cameras.start()
            factory = self._dataset_factory or LeRobotDataset
            features = self._features(config)
            if (root / "meta/info.json").exists():
                if not config.resume:
                    raise ValueError("Dataset already exists; explicitly enable resume")
                existing_manifest = preflight_sensor_stream_writer(root)
                if existing_manifest != build_sensor_stream_manifest({self.force_instance: self.sensor}):
                    raise ValueError("Existing Sidecar v2 schema differs from the configured force stream")
                self._check_resume(root)
                dataset = factory.resume(
                    repo_id=config.repo_id,
                    root=root,
                    streaming_encoding=True,
                    encoder_queue_maxsize=30,
                    video_backend="pyav",
                    rgb_encoder=config.video_encoder(),
                    strict_video_integrity=True,
                )
                if dataset.fps != config.fps or any(
                    any(
                        (
                            tuple(dataset.features.get(k, {}).get(field, ())) != tuple(expected)
                            if field == "shape"
                            else dataset.features.get(k, {}).get(field) != expected
                        )
                        for field, expected in v.items()
                    )
                    for k, v in features.items()
                ):
                    dataset.finalize()
                    raise ValueError("Existing Dataset fps/features differ from requested recording")
            else:
                if root.exists() and any(root.iterdir()):
                    raise ValueError("A new demonstration needs an empty/new root")
                # Dataset.create requires the root itself not to exist.
                if root.exists():
                    raise ValueError("Choose a new Dataset root directory")
                dataset = factory.create(
                    repo_id=config.repo_id,
                    root=root,
                    fps=config.fps,
                    features=features,
                    robot_type="rebot_b601_teacher",
                    use_videos=True,
                    streaming_encoding=True,
                    encoder_queue_maxsize=30,
                    encoder_threads=2,
                    video_backend="pyav",
                    rgb_encoder=config.video_encoder(),
                    strict_video_integrity=True,
                )
                from .capture_contract import capture_contract

                _atomic_json(root / "meta/teacher_capture.json", capture_contract())
            recorder_factory = self._recorder_factory or SensorStreamRecorder
            recorder = recorder_factory(root, {self.force_instance: self.sensor})
            self._dataset = dataset
            self._recorder = recorder
            self._cameras = cameras
            self._config = config
            self._update(episode_count=dataset.meta.total_episodes)
        except BaseException as original:
            # A failed preparation still owns every resource whose shutdown is
            # unconfirmed. Retain those references so close() can retry them.
            failures = []
            recorder_stopped = True
            if "recorder" in locals():
                try:
                    recorder.close()
                    if self._recorder is recorder:
                        self._recorder = None
                except BaseException as cleanup:
                    failures.append(cleanup)
                    self._recorder = recorder
                    recorder_stopped = False
            if "dataset" in locals():
                if recorder_stopped:
                    try:
                        dataset.finalize()
                        if self._dataset is dataset:
                            self._dataset = None
                    except BaseException as cleanup:
                        failures.append(cleanup)
                        self._dataset = dataset
                else:
                    self._dataset = dataset
            try:
                cameras.close()
                if self._cameras is cameras:
                    self._cameras = None
            except BaseException as cleanup:
                failures.append(cleanup)
                self._cameras = cameras
            if failures:
                self._config = config
                self._cleanup_pending = True
                raise BaseExceptionGroup(
                    "Demonstration preparation and cleanup failed", [original, *failures]
                ) from None
            raise

    def _check_resume(self, root: Path) -> None:
        """Refuse incomplete Teacher roots instead of inventing or repairing evidence."""
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        from .capture_contract import capture_contract

        contract_path = root / "meta/teacher_capture.json"
        if (
            not contract_path.is_file()
            or json.loads(contract_path.read_text(encoding="utf-8")) != capture_contract()
        ):
            raise ValueError("Resume requires this capture version; choose a new root for historical data")
        manifests = list((root / "meta/teacher_episodes").glob("*.json"))
        complete_indices = set()
        for path in manifests:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("complete"):
                if metadata["episode_index"] in complete_indices:
                    raise ValueError("Duplicate complete Teacher episode index")
                complete_indices.add(metadata["episode_index"])
            elif metadata.get("status") not in ("discarded", "interrupted", "error"):
                raise ValueError("An unsealed Teacher episode exists; inspect/recover it before resuming")
        if complete_indices != set(range(info["total_episodes"])):
            raise ValueError("Existing main episodes lack complete Teacher attachment evidence")
        # Existing complete metadata is not trusted merely because it says complete.
        from .demonstration_validation import validate_demonstration

        for path in manifests:
            if json.loads(path.read_text(encoding="utf-8")).get("complete"):
                report = validate_demonstration(root, path.stem)
                if not report["valid"]:
                    raise ValueError(
                        "Existing Teacher episode failed read-only verification: "
                        + "; ".join(report["errors"])
                    )
        successes = sum(
            json.loads(path.read_text(encoding="utf-8")).get("task_result") == "success"
            and json.loads(path.read_text(encoding="utf-8")).get("complete") is True
            for path in manifests
        )
        self._update(successful_episode_count=successes)

    def _start(self, config: TeacherRecordConfig) -> None:
        if self._cleanup_pending:
            raise RuntimeError("Earlier recording cleanup is unconfirmed; stop this session before reuse")
        if self._subscription is not None or self._uid is not None:
            raise RuntimeError("A demonstration is already active")
        config.validate()
        if tuple(self.owner.runtime.config.joint_names) != JOINT_NAMES:
            raise ValueError("Full demonstration recording requires the configured seven-joint owner")
        if self.owner.runtime.config.control_hz != config.fps:
            raise ValueError("Teacher capture requires control_hz = Dataset fps = 50")
        if not self.owner.is_running:
            raise RuntimeError("Start the explicitly armed teleoperation session before recording")
        self._prepare(config)
        self._config = config
        self._uid = str(uuid.uuid4())
        self._episode_index = self._dataset.meta.total_episodes
        self._frame_count = self._control_count = 0
        self._stages.clear()
        self._force_labels = self._action_labels = 0
        self._normal_release = False
        self._first_anchor = self._last_anchor = None
        self._first_control = self._last_control = self._last_tick_id = self._epoch = None
        self._task_result = "unknown"
        self._started_utc = datetime.now(UTC).isoformat()
        self._timing_misses = self._maximum_interval_ns = 0
        self._save_evidence = {}
        self._camera_start_counts = {name: buffer.count for name, buffer in self._cameras.buffers.items()}
        directory = Path(config.root).resolve() / "teacher" / self._uid
        directory.mkdir(parents=True, exist_ok=False)
        self._control_file = (directory / "control.jsonl").open("x", encoding="utf-8")
        self._frames_file = (directory / "frames.jsonl").open("x", encoding="utf-8")
        self._write_manifest(complete=False, status="preparing")
        # Warm codecs before subscriptions without forcing GC while Owner runs.
        self._dataset.prepare_episode_recording(collect_garbage=False)
        self._recorder.start_episode(self._episode_index, episode_uid=self._uid, dataset=self._dataset)
        self._recorder.wait_until_ready()
        self._start_sequence = self._recorder._subscriptions[self.force_instance].start_sequence
        self._subscription = self.owner.subscribe(config.queue_capacity)
        self._write_manifest(complete=False, status="recording")
        self._update(
            state="recording",
            active=True,
            episode_uid=self._uid,
            frame_count=0,
            error=None,
            file_complete=False,
            task_result="unknown",
            target_episode_count=config.target_episode_count,
            request_error=None,
        )

    def _consume(self, snapshot: OwnerSnapshot) -> None:
        row = snapshot.row
        if self._last_tick_id is not None and row["tick_id"] != self._last_tick_id + 1:
            raise RuntimeError("Control tick gap in Teacher demonstration")
        if self._epoch is not None and snapshot.epoch != self._epoch:
            raise RuntimeError("Owner epoch changed during demonstration")
        self._epoch = snapshot.epoch
        self._last_tick_id = row["tick_id"]
        self._control_file.write(_json(row) + "\n")
        self._control_count += 1
        self._stages[row.get("capture_stage", "unknown")] += 1
        self._force_labels += row.get("label_valid") is True
        self._action_labels += row.get("action_label_valid") is True
        self._normal_release |= row.get("phase_at_tick_start") == "RELEASE" and row.get("phase") == "DISABLED"
        anchor = row.get("decision_anchor_ns")
        if isinstance(anchor, int):
            self._first_control = anchor if self._first_control is None else self._first_control
            if self._last_control is not None:
                interval = anchor - self._last_control
                self._maximum_interval_ns = max(self._maximum_interval_ns, interval)
                self._timing_misses += interval > 1.5e9 / self._config.fps
            self._last_control = anchor
        if row.get("phase") == "FAULT" or row.get("failure_reason"):
            raise RuntimeError(f"Owner fault during demonstration: {row.get('failure_reason')}")
        if row.get("send_status") != "submitted":
            # Keep masked/disabled rows in full-rate evidence; they are not fabricated actions.
            return
        if row.get("force_sequence") is None or row["force_sequence"] < self._start_sequence:
            return
        frame, capture, evidence = completed_tick_frame(
            snapshot,
            self._cameras.buffers,
            self._config,
            force_instance=self.force_instance,
            frame_index=self._frame_count,
        )
        self._dataset.add_frame(frame)
        self._recorder.record_sync(None, capture)
        self._frames_file.write(_json(evidence) + "\n")
        self._frame_count += 1
        self._first_anchor = anchor if self._first_anchor is None else self._first_anchor
        self._last_anchor = anchor
        # Owner owns the absolute 50 Hz schedule. Record each submitted tick once;
        # real intervals remain evidence, without a second clock that drops early ticks.
        self._update(frame_count=self._frame_count, stage_counts=dict(self._stages))

    def _freeze_and_drain(self) -> None:
        if self._subscription is None:
            raise RuntimeError("No active demonstration")
        subscription, self._subscription = self._subscription, None
        self.owner.unsubscribe(subscription)
        subscription.check_health()
        while True:
            try:
                self._consume(subscription.queue.get_nowait())
            except queue.Empty:
                break
            except Exception:
                # Remaining completed attempts are still evidence after a bad tick.
                while True:
                    try:
                        row = subscription.queue.get_nowait().row
                    except queue.Empty:
                        break
                    self._control_file.write(_json(row) + "\n")
                    self._control_count += 1
                    self._stages[row.get("capture_stage", "unknown")] += 1
                    self._force_labels += row.get("label_valid") is True
                    self._action_labels += row.get("action_label_valid") is True
                raise

    def _finish(self, task_result: str) -> None:
        if task_result not in RESULTS:
            raise ValueError("Invalid manual task result")
        if self._uid is None:
            raise RuntimeError("No demonstration is recording")
        self._update(state="saving")
        self._save_evidence = {"started_ns": time.perf_counter_ns(), "start_owner": self._owner_save_state()}
        self._freeze_and_drain()
        if self._frame_count == 0:
            raise RuntimeError("Cannot save an empty demonstration")
        self._task_result = task_result
        self._close_files()
        self._write_manifest(complete=False, status="saving")
        if self._stop.is_set():
            raise RuntimeError("Session stopped before episode commit")
        self._recorder.prepare_episode(task_info=[self._config.task])
        self._recorder.commit_prepared(self._dataset)
        if self._stop.is_set():
            raise RuntimeError("Session stopped during episode commit; Teacher completion withheld")
        self._write_manifest(complete=False, status="verifying")
        # The independent reader performs full video/Raw/Sync/action checks. Release
        # the live writer lock only while no Sensor episode is active, then reopen it.
        self._recorder.close()
        self._recorder = None
        from .demonstration_validation import validate_demonstration

        # Validate a candidate complete marker, but never expose it until it passes.
        report = validate_demonstration(Path(self._config.root), self._uid, require_complete=False)
        if not report["valid"]:
            self._write_manifest(complete=False, status="error", error="; ".join(report["errors"]))
            raise RuntimeError("Saved demonstration verification failed: " + "; ".join(report["errors"]))
        if self._stop.is_set():
            raise RuntimeError("Session stopped during verification; Teacher completion withheld")
        self._save_evidence.update(completed_ns=time.perf_counter_ns(), end_owner=self._owner_save_state())
        self._write_manifest(complete=True, status="complete")
        self._recorder = (self._recorder_factory or SensorStreamRecorder)(
            self._dataset.root, {self.force_instance: self.sensor}
        )
        status = self.status()
        self._update(
            state="saved",
            active=False,
            file_complete=True,
            episode_count=self._dataset.meta.total_episodes,
            successful_episode_count=status["successful_episode_count"] + (task_result == "success"),
            task_result=task_result,
            normal_release_completed=self._normal_release,
            stage_counts=dict(self._stages),
            error=None,
        )
        self._uid = None

    def _owner_save_state(self) -> dict | None:
        """Retain control progress around asynchronous save without posting commands."""
        try:
            snapshot = getattr(self.owner, "snapshot", lambda: None)()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if snapshot is None:
            return None
        return {
            "tick_id": snapshot.tick_id,
            "phase": snapshot.row.get("phase"),
            "failure_reason": snapshot.row.get("failure_reason"),
        }

    def _close_files(self) -> None:
        errors = []
        for handle in (self._control_file, self._frames_file):
            if handle is not None and not handle.closed:
                try:
                    handle.flush()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        handle.close()
                    except Exception as exc:
                        errors.append(exc)
        if errors:
            raise ExceptionGroup("Teacher file cleanup failed", errors)

    def _write_manifest(self, *, complete: bool, status: str, error: str | None = None) -> None:
        root = Path(self._config.root).resolve()
        files = {}
        for name, count in (("control", self._control_count), ("frames", self._frame_count)):
            path = root / "teacher" / self._uid / f"{name}.jsonl"
            if path.exists():
                files[name] = {
                    "path": path.relative_to(root).as_posix(),
                    "rows": count,
                    "sha256": _digest(path),
                }
        duration = (self._last_anchor - self._first_anchor) / 1e9 if self._frame_count > 1 else None
        control_duration = (
            (self._last_control - self._first_control) / 1e9
            if self._control_count > 1 and self._last_control != self._first_control
            else None
        )
        data = {
            "schema": EPISODE_SCHEMA,
            "episode_uid": self._uid,
            "episode_index": self._episode_index,
            "frame_count": self._frame_count,
            "complete": complete,
            "status": status,
            "task_result": self._task_result,
            "joint_names": list(JOINT_NAMES),
            "force_instance": self.force_instance,
            "files": files,
            "capture_summary": {
                "stage_counts": dict(self._stages),
                "force_label_count": self._force_labels,
                "action_label_count": self._action_labels,
                "normal_release_completed": self._normal_release,
            },
            "rates": {
                "requested_main_hz": self._config.fps,
                "measured_main_hz": (self._frame_count - 1) / duration if duration else None,
                "measured_control_hz": (self._control_count - 1) / control_duration
                if control_duration
                else None,
            },
            "config": asdict(self._config),
            "provenance": self.provenance,
            "error": error,
            "hardware_ack": "unknown",
            "started_utc": self._started_utc,
            "save": self._save_evidence,
            "camera_bindings": getattr(self._cameras, "bindings", {}),
            "timing": {
                "intervals_over_30ms": self._timing_misses,
                "maximum_control_interval_ns": self._maximum_interval_ns,
                "file_completeness_independent_of_rate": True,
            },
            "durability": "process-crash incomplete marker; no power-loss guarantee",
        }
        sensor_metadata_path = root / "meta/sensor_episodes" / f"{self._uid}.json"
        if sensor_metadata_path.exists():
            sensor_metadata = json.loads(sensor_metadata_path.read_text(encoding="utf-8"))
            data["rates"]["measured_force_hz"] = sensor_metadata["streams"][self.force_instance][
                "actual_sample_rate_hz"
            ]
        _atomic_json(root / "meta/teacher_episodes" / f"{self._uid}.json", data)

    def _abort(self, reason: str, *, status: str) -> None:
        if self._uid is None:
            raise RuntimeError("No demonstration is active")
        failures = []
        if self._subscription is not None:
            subscription, self._subscription = self._subscription, None
            self.owner.unsubscribe(subscription)
            # Preserve queued control attempts, including faults, without adding frames.
            while True:
                try:
                    row = subscription.queue.get_nowait().row
                    self._control_file.write(_json(row) + "\n")
                    self._control_count += 1
                    self._stages[row.get("capture_stage", "unknown")] += 1
                    self._force_labels += row.get("label_valid") is True
                    self._action_labels += row.get("action_label_valid") is True
                except queue.Empty:
                    break
        try:
            self._close_files()
        except Exception as exc:
            failures.append(str(exc))
        try:
            self._write_manifest(complete=False, status=status, error=reason)
        except Exception as exc:
            failures.append(str(exc))
        recorder_stopped = True
        if self._recorder is not None:
            try:
                if self._recorder.is_active:
                    self._recorder.abort_episode(reason)
                elif self._recorder._transaction is not None:
                    self._recorder.abort_prepared(reason)
            except Exception as exc:
                failures.append(str(exc))
                recorder_stopped = False
        if self._dataset is not None and recorder_stopped:
            try:
                self._dataset.clear_episode_buffer()
            except Exception as exc:
                failures.append(str(exc))
        self._cleanup_pending = bool(failures)
        if not failures:
            self._uid = None
        self._update(
            state=status,
            active=False,
            file_complete=False,
            error=reason,
            cleanup_error="; ".join(failures) or None,
        )
        if failures:
            raise RuntimeError("Demonstration abort cleanup failed: " + "; ".join(failures))

    def _set_outcome(self, task_result: str) -> None:
        if task_result not in RESULTS:
            raise ValueError("Invalid manual task result")
        if self._uid is not None:
            self._task_result = task_result
            self._update(task_result=task_result)
            return
        status = self.status()
        uid = status.get("episode_uid")
        if not uid or not status.get("file_complete"):
            raise RuntimeError("No complete demonstration is available for manual review")
        path = Path(self._config.root).resolve() / "meta/teacher_episodes" / f"{uid}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        previous = data["task_result"]
        data["task_result"] = task_result
        _atomic_json(path, data)
        self._update(
            task_result=task_result,
            successful_episode_count=(
                status["successful_episode_count"] + (task_result == "success") - (previous == "success")
            ),
        )
