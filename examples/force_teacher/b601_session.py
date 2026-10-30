"""Headless, asynchronous desktop session coordination with explicit resource ownership.

Opening the controller never opens hardware. All long operations run in its
supervisor; stop uses the existing independent Event, outside the request queue.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import draccus
from b601_full_arm_recovery import query_disabled_arm
from b601_session_support import (
    command_feedback_override,
    exit_confirmation,
    exit_port,
    full_arm_result,
    monitor,
    readonly_exit_reference,
    stop_report,
)
from b601_supervised_bench import run
from b601_tuner_process import ProcessLive, run_in_process
from b601_verification import VerificationWorker
from build_operator_teleoperation_profile import build_operator_document

from lerobot.force_control.config import finite_number, positive, validate_motion_request
from lerobot.force_control.demonstration import TeacherRecordConfig
from lerobot.force_control.operator_settings import checked_arm_settings
from lerobot.force_control.tuning import (
    FIELDS,
    LIVE_FIELDS,
    OPTIONAL_GROUPS,
    build_config,
    default_values,
    read_settings,
    save_settings,
    sensor_config,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.calibration import DEFAULT_GRIPPER_CALIBRATION, B601GripperCalibration
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT / "b601_capture_parameters.json"
BINDINGS_FILE = PROJECT / "b601_camera_bindings.json"
OUTPUTS = PROJECT / "outputs/force_teacher_tuning"
_UI_ACTIONS = frozenset(
    (
        "update_form",
        "verify_dataset",
        "cancel_verification",
        "connect",
        "disconnect",
        "recover",
        "start",
        "preflight",
        "apply",
        "target",
        "resume",
        "approach",
        "position",
        "release",
        "record_start",
        "record_finish",
        "record_discard",
        "record_outcome",
        "load_settings",
        "save_settings",
        "load_calibration",
        "enumerate_cameras",
        "preview_cameras",
        "bind_cameras",
        "close_camera_preview",
    )
)


def freeze(value):
    """Detach recursively so a page cannot mutate backend state through a snapshot."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, deque)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class PanelSnapshot:
    revision: int
    session_id: str
    state: str
    message: str
    error: str
    exit_confirmed: bool
    telemetry: Mapping
    recording: Mapping
    values: Mapping
    toggles: Mapping
    recording_values: Mapping
    camera_configs: Mapping
    arm_settings: Mapping
    devices: tuple
    capabilities: Mapping
    settings_path: str
    calibration_path: str
    log_path: str
    operation_results: tuple
    result_errors: tuple = ()


class SessionController:
    """One UI-independent owner of session intent, lifecycle and snapshot publication."""

    def __init__(
        self,
        *,
        settings_path=None,
        calibration_path=None,
        gripper_only=False,
        demo=False,
        output_root=None,
        services=None,
        command_feedback_mode=None,
        command_tx_gap_us=120.0,
        command_tx_gap_source="配置",
    ):
        self.command_feedback_mode = command_feedback_mode
        self.command_tx_gap_us = command_tx_gap_us
        self.command_tx_gap_source = command_tx_gap_source
        self.demo, self.gripper_only = bool(demo), bool(gripper_only)
        self._services = services or {}
        self._lock = threading.RLock()
        self._commands = queue.Queue(maxsize=32)
        self._completions = queue.SimpleQueue()
        self._thread = self._monitor_thread = self._owner_thread = None
        self._stop_monitor = threading.Event()
        self._cancel = threading.Event()
        self._closing = self._closed = False
        self._generation = self._config_revision = self._revision = 0
        self._session_id = uuid.uuid4().hex
        self._state, self._message, self._error = "loading", "正在加载配置…", ""
        self._result_errors = ()
        self._exit_confirmed = True
        self._values = default_values()
        self._toggles = {key: enabled for key, (_, enabled) in OPTIONAL_GROUPS.items()}
        self._recording_values = {
            "enabled": False,
            "root": str(PROJECT / "outputs/teacher_demonstrations"),
            "h264_encoder": "nvenc",
            "repo_id": "local/force_teacher_demo",
            "task": "",
            "object_id": "",
            "contact_condition_id": "",
            "camera_max_age_ms": "100",
            "queue_capacity": "128",
            "target_episode_count": "10",
            "fps": "50",
            "resume": False,
            "verify_video_on_save": False,
        }
        self._cameras = {
            role: {
                "type": "opencv",
                "backend": 700,
                "fps": 50,
                "width": 640,
                "height": 480,
                "fourcc": "MJPG",
                "device_name": "Global Shutter Camera",
            }
            for role in ("front", "wrist")
        }
        self._arm_settings, self._devices = {}, ()
        self._recording, self._telemetry, self._preview = {}, {}, {}
        self._verification = {"state": "idle"}
        self._verification_worker = None
        self._verification_cancel = threading.Event()
        self._verification_result = None
        self._verification_started = None
        self._verification_baseline = None
        self._verification_check_at = 0.0
        self._results = deque(maxlen=64)
        self._pending_record = None
        self._settings_path = Path(settings_path or DEFAULT_SETTINGS)
        self._calibration_path = Path(calibration_path or DEFAULT_GRIPPER_CALIBRATION)
        self._calibration_override = calibration_path
        self._output_root = Path(output_root or OUTPUTS)
        self._calibration = self._config = self._live = None
        self._last_result = self._last_result_path = None
        self._unconfirmed_port = self._unconfirmed_calibration = None
        self._recovery_evidence = None
        self._retained = []
        self._monitor_token = None
        self._monitor_cleanup = None
        self._monitor_values_pin = None
        self._preview_enabled = True
        self._camera_session = None
        self._camera_selection = None
        self._camera_seen = set()
        self._preview_last = self._demo_last_preview = 0.0
        self._demo_tick = 0
        self._control_is_gripper = self.gripper_only
        self._cached_snapshot = None
        self._changed()

    def prefill_recording(self, path):
        """Validate and preload a recording form before starting the supervisor."""
        if self._thread is not None:
            raise RuntimeError("录制预填须在界面会话启动之前完成")
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        config = TeacherRecordConfig(**document)
        config.validate()
        document = asdict(config)
        self._recording_values.update(
            {
                key: str(value)
                for key, value in document.items()
                if key not in ("camera_configs", "resume", "verify_video_on_save")
            }
        )
        self._recording_values.update(
            enabled=True,
            resume=document.get("resume", False),
            verify_video_on_save=document.get("verify_video_on_save", False),
        )
        self._cameras = copy.deepcopy(document["camera_configs"])
        self._changed()

    def start(self):
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="b601-session-supervisor", daemon=False)
            self._thread.start()

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def snapshot(self):
        with self._lock:
            return self._cached_snapshot

    def _changed(self, message=None, *, state=None, error=None):
        with self._lock:
            if message is not None:
                self._message = message
            if state is not None:
                self._state = state
            if error is not None:
                self._error = error
                self._result_errors = ()
            self._revision += 1
            # Publish detached state here; GUI readers never walk mutable worker state.
            self._cached_snapshot = PanelSnapshot(
                self._revision,
                self._session_id,
                self._state,
                self._message,
                self._error,
                self._exit_confirmed,
                freeze(self._telemetry),
                freeze({**self._recording, "full_verification": dict(self._verification)}),
                freeze(self._values),
                freeze(self._toggles),
                freeze(self._recording_values),
                freeze(self._cameras),
                freeze(self._arm_settings),
                freeze(self._devices),
                freeze(self._capabilities()),
                str(self._settings_path),
                str(self._calibration_path),
                str(self._last_result_path or ""),
                freeze(self._results),
                self._result_errors,
            )

    def request(self, action, payload=None):
        if action == "stop":
            self.stop()
            return uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("程序正在退出")
            generation = self._generation
        self._commands.put_nowait((request_id, action, thaw(payload or {}), generation))
        return request_id

    def stop(self):
        """Never wait for the supervisor, decoding, preflight, save, or a result queue."""
        self._cancel.set()
        self._verification_cancel.set()
        self._stop_monitor.set()
        with self._lock:
            self._generation += 1
            live = self._live
        signal_error = None
        if live is not None:
            try:
                live.stop.set()
            except Exception as exc:
                signal_error = f"停止信号通信失败：{type(exc).__name__}: {exc}"
        with self._lock:
            if self.demo:
                self._recording = {**self._recording, **{"active": False, "busy": False}}
            self._changed(
                signal_error or "停止请求已发送，正在等待资源退出确认…",
                state="stopping",
                error=signal_error if signal_error else self._error,
            )

    def close(self):
        self._closing = True
        self.stop()

    def set_preview_enabled(self, enabled):
        self._preview_enabled = bool(enabled)
        live = self._live
        if live is not None and hasattr(live, "set_preview_enabled"):
            live.set_preview_enabled(self._preview_enabled)

    def preview(self):
        with self._lock:
            return dict(self._preview)

    def _capabilities(self):
        idle = self._state in ("disconnected", "recovery_required")
        monitoring = self._state == "monitoring"
        active = (
            self._state == "running"
            and not self._cancel.is_set()
            and self._telemetry.get("phase")
            in ("FORCE_TRACK", "APPROACH", "POSITION_HOLD", "FOLLOW", "FOLLOW_WAIT_CLEAR")
        )
        stable = idle or monitoring or active
        no_preview = self._camera_session is None
        hardware_edit = idle and no_preview and not self._retained
        reason = "请等待当前操作完成"
        capture = bool(self._recording.get("active"))
        record_state = self._recording.get("state", "idle")
        record_ready = active and not self._recording.get("busy") and self._pending_record is None
        blocked_record = record_state in ("preparing", "saving", "verifying", "commit_unknown", "corrupt")
        result = dict.fromkeys(
            (
                "connect",
                "disconnect",
                "start",
                "recover",
                "apply",
                "target",
                "resume",
                "approach",
                "position",
                "release",
                "record_start",
                "record_finish",
                "record_discard",
                "record_outcome",
                "preflight",
                "enumerate_cameras",
                "preview_cameras",
                "bind_cameras",
                "close_camera_preview",
                "load_settings",
                "save_settings",
                "load_calibration",
                "update_form",
                "edit_values",
                "edit_recording",
                "verify_dataset",
                "cancel_verification",
            ),
            reason,
        )
        result.update(
            stop="",
            update_form="" if stable else reason,
            edit_values="" if hardware_edit else "设备固定参数只能在停止并断开后修改",
            edit_live=""
            if monitoring or (active and self._control_is_gripper)
            else "当前模式只能单独应用目标力",
            edit_recording="" if idle or monitoring else "录制准备后请先停止再修改任务信息",
            connect="" if idle and no_preview and not self._retained else "请先停止并关闭相机预览",
            disconnect="" if monitoring else "当前没有只读连接",
            start="" if monitoring and self._exit_confirmed and no_preview else "请先连接并取得新鲜失能反馈",
            recover="" if idle and not self._exit_confirmed and no_preview else "当前无需复核或资源尚未退出",
            preflight="" if (idle or monitoring) and no_preview else "请先停止并关闭相机预览",
            enumerate_cameras="" if hardware_edit else "请先停止并断开设备",
            preview_cameras="" if idle else "请先停止并断开设备",
            bind_cameras="" if self._camera_session is not None else "请先预览两路画面",
            close_camera_preview="" if self._camera_session is not None else "当前没有相机预览",
            load_settings="" if hardware_edit else "请先停止并断开设备",
            save_settings="" if hardware_edit else "请先停止并断开设备",
            load_calibration="" if hardware_edit else "请先停止并断开设备",
        )
        for key in ("resume", "approach", "position", "release"):
            result[key] = "" if active else "请先启动控制会话"
        if active:
            phase = self._telemetry.get("phase")
            result["resume"] = "" if phase == "POSITION_HOLD" else "仅固定位置模式可恢复力控"
            result["apply"] = (
                "" if self._control_is_gripper else "全臂会话请单独应用目标力；其他参数停止后修改"
            )
            result["target"] = "" if not self._control_is_gripper else "仅全臂遥操支持单独应用目标力"
        else:
            result["apply"] = "" if idle or monitoring else reason
        for key in ("record_start", "record_finish", "record_discard", "record_outcome"):
            result[key] = "请先启用录制准备并启动遥操"
        if record_ready and not blocked_record and self._recording_values.get("enabled"):
            if capture:
                result.update(record_finish="", record_discard="", record_outcome="")
            elif record_state == "saved_pending_verification":
                result.update(record_outcome="")
            else:
                result["record_start"] = ""
                if self._recording.get("file_complete") or self.demo:
                    result["record_outcome"] = ""
        verifying = self._verification_worker is not None
        can_verify = (
            self._state == "disconnected"
            and self._exit_confirmed
            and self._resources_stopped()
            and no_preview
            and not self._retained
            and not self._closing
        )
        result["verify_dataset"] = "" if can_verify and not verifying else "请先停止采集并等待资源退出"
        result["cancel_verification"] = "" if verifying else "当前没有完整核验"
        if verifying:
            for key in (
                "connect",
                "start",
                "recover",
                "record_start",
                "update_form",
                "edit_recording",
                "preview_cameras",
                "enumerate_cameras",
            ):
                result[key] = "请等待完整核验结束或取消"
        return result

    def _assert_current(self, generation):
        if generation != self._generation or self._closing or self._cancel.is_set():
            raise RuntimeError("操作已取消；未启动新的控制")

    def _loop(self):
        try:
            try:
                self._initialize()
            except Exception as exc:
                self._changed(f"加载配置失败：{exc}", state="disconnected", error=str(exc))
            while not self._closed:
                try:
                    command = self._commands.get(timeout=0.02)
                except queue.Empty:
                    command = None
                if command is not None:
                    request_id, action, payload, generation = command
                    success = False
                    entered = False
                    try:
                        if generation != self._generation:
                            raise RuntimeError("旧会话请求已取消")
                        reason = self._capabilities().get(action)
                        if reason:
                            raise RuntimeError(reason)
                        entered = True
                        self._dispatch(action, payload, generation)
                        success = True
                    except Exception as exc:
                        self._changed(str(exc) or type(exc).__name__, error=f"{type(exc).__name__}: {exc}")
                        if entered and self._state in (
                            "preflighting",
                            "connecting",
                            "starting",
                            "recovering",
                        ):
                            self._changed(
                                state="monitoring"
                                if self._monitor_thread and self._monitor_thread.is_alive()
                                else "disconnected"
                                if self._exit_confirmed
                                else "recovery_required"
                            )
                    with self._lock:
                        self._results.append(
                            {
                                "request_id": request_id,
                                "action": action,
                                "success": success,
                                "message": self._message,
                                "session_id": self._session_id,
                            }
                        )
                        self._changed()
                try:
                    self._poll()
                except Exception as exc:
                    self._changed(f"会话通信异常：{exc}", error=str(exc))
                    self.stop()
                if (
                    self._closing
                    and self._resources_stopped()
                    and not self._retained
                    and self._verification_worker is None
                ):
                    try:
                        self._close_camera_preview()
                    except Exception as exc:
                        self._changed(f"相机退出未确认：{exc}", state="closing", error=str(exc))
                        time.sleep(0.05)
                        continue
                    self._closed = True
                    self._changed(state="closed")
        finally:
            if not self._resources_stopped():
                self._changed("资源仍在退出，禁止重新打开", state="recovery_required")

    def _initialize(self):
        if self.demo:
            self._values = {
                **self._values,
                **{"target_N": "5", "kp_deg_per_N": "1.7", "ki_deg_per_N_s": "0.1", "close_sign": "1"},
            }
            self._recording_values.update(
                enabled=True, task="离线演示", object_id="合成物体", contact_condition_id="synthetic"
            )
            self._changed("离线演示 · 合成数据 · 操作不会连接设备或保存正式数据", state="disconnected")
            return
        latest = max(
            self._output_root.glob("*/run/result.json"), key=lambda p: p.stat().st_mtime_ns, default=None
        )
        if latest is not None:
            try:
                result = json.loads(latest.read_text(encoding="utf-8"))
                self._accept_result(result, latest)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                self._exit_confirmed = False
                self._last_result_path = latest
                self._changed(error=f"历史退出证据读取失败：{exc}")
        try:
            if self._settings_path.is_file():
                self._load_settings(self._settings_path)
            if BINDINGS_FILE.is_file():
                self._cameras = json.loads(BINDINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._changed(error=f"配置未加载：{exc}")
        if self._calibration_override:
            self._calibration_path = Path(self._calibration_override)
        try:
            self._calibration = B601GripperCalibration.load(self._calibration_path)
            self._populate_arm_settings()
        except (OSError, ValueError) as exc:
            self._changed(error=f"校准未加载：{exc}")
        self._changed(
            "未连接；请检查设备配置后连接只读监测"
            if self._exit_confirmed
            else "上次退出尚未确认；连接时将先进行只读失能复核",
            state="disconnected" if self._exit_confirmed else "recovery_required",
        )

    def _populate_arm_settings(self):
        if self.gripper_only or self._calibration is None or self._arm_settings:
            return
        try:
            document, _ = build_operator_document(self._values, self._toggles, self._calibration)
            operator = document["b601"]["operator_teleoperation"]
            from lerobot.force_control.operator_settings import ARM_OPERATOR_FIELDS, ARM_PROFILE_FIELDS

            self._arm_settings = {
                "operator": {
                    key: "" if operator.get(key) is None else str(operator[key])
                    for key in ARM_OPERATOR_FIELDS
                },
                "joints": {
                    profile["name"]: {
                        key: "" if profile.get(key) is None else str(profile[key])
                        for key in ARM_PROFILE_FIELDS
                    }
                    for profile in document["b601"]["control"]["profiles"]
                    if profile["name"] != "gripper"
                },
            }
        except (ValueError, TypeError, OSError):
            pass  # An incomplete draft stays editable; preflight explains missing fields.

    def _load_settings(self, path):
        arms = []
        values, toggles, calibration = read_settings(path, on_arm_settings=arms.append)
        loaded_calibration = B601GripperCalibration.load(calibration)
        self._values, self._toggles = values, toggles
        self._arm_settings = arms[0] if arms else {}
        self._calibration_path, self._settings_path = calibration, Path(path)
        self._calibration = loaded_calibration
        self._populate_arm_settings()
        self._config_revision += 1

    def _update_form(self, payload):
        values = {**self._values, **payload.get("values", {})}
        toggles = {**self._toggles, **payload.get("toggles", {})}
        if set(values) != set(default_values()) or any(type(value) is not str for value in values.values()):
            raise ValueError("参数字段或类型无效")
        if set(toggles) != set(OPTIONAL_GROUPS) or any(type(value) is not bool for value in toggles.values()):
            raise ValueError("参数开关无效")
        busy = self._state not in ("disconnected", "recovery_required")
        if busy and any(values[key] != self._values[key] for key in values if key not in LIVE_FIELDS):
            raise ValueError("设备固定参数只能在停止并断开后修改")
        if (
            self._state == "running"
            and not self._control_is_gripper
            and any(values[key] != self._values[key] for key in values if key != "target_N")
        ):
            raise ValueError("运行中的全臂会话只能单独修改目标力")
        if busy and toggles != self._toggles:
            raise ValueError("保护开关只能在停止并断开后修改")
        recording = {**self._recording_values, **payload.get("recording_values", {})}
        if type(recording.get("verify_video_on_save")) is not bool:
            raise ValueError("保存时完整核验视频必须为布尔值")
        cameras = copy.deepcopy(payload.get("camera_configs", self._cameras))
        arms = checked_arm_settings(payload.get("arm_settings", self._arm_settings))
        if self._state not in ("disconnected", "recovery_required", "monitoring") and (
            recording != self._recording_values or cameras != self._cameras
        ):
            raise ValueError("会话已准备，录制配置保持锁定")
        if busy and arms != self._arm_settings:
            raise ValueError("全臂固定参数只能在停止并断开后修改")
        if self._camera_session is not None and cameras != self._cameras:
            raise ValueError("请先关闭相机预览再修改配置")
        if (values, toggles, recording, cameras, arms) == (
            self._values,
            self._toggles,
            self._recording_values,
            self._cameras,
            self._arm_settings,
        ):
            return
        if recording.get("root") != self._recording_values.get("root"):
            self._verification = {"state": "idle"}
        self._values, self._toggles, self._recording_values = values, toggles, recording
        self._cameras, self._arm_settings = cameras, arms
        self._config_revision += 1
        self._changed()

    def _dispatch(self, action, payload, generation):
        if not isinstance(action, str) or action not in _UI_ACTIONS:
            raise ValueError(f"未知界面操作：{action}")
        if action == "update_form":
            self._update_form(payload)
            return
        form = payload.pop("form", None)
        if form is not None:
            self._update_form(form)
        self._cancel.clear()
        if action == "verify_dataset":
            self._start_verification()
            return
        if action == "cancel_verification":
            self._verification_cancel.set()
            self._changed("正在取消完整核验…")
            return
        if action == "connect":
            if not self._exit_confirmed:
                self._recover(generation)
                self._assert_current(generation)
            self._connect(generation)
        elif action == "disconnect":
            self.stop()
        elif action == "recover":
            self._recover(generation)
        elif action == "start":
            self._start_control(payload, generation)
        elif action == "preflight":
            self._changed("正在检查参数与录制目录…", state="preflighting", error="")
            if not self.demo:
                if not self.gripper_only and payload.get("confirmed") is not True:
                    raise ValueError("请先确认机械支撑、闭合方向和力通道，再进行全臂预检")
                self._build_config(
                    "hybrid" if not self.gripper_only else "force", confirmed=payload.get("confirmed") is True
                )
                if self._recording_values.get("enabled"):
                    TeacherRecordConfig(**self._recording_config()).validate()
            self._assert_current(generation)
            self._changed(
                "参数与目录检查通过；启动前仍会检查新鲜反馈",
                state="monitoring"
                if self._monitor_thread and self._monitor_thread.is_alive()
                else "disconnected",
            )
        elif action in ("apply", "target", "resume", "approach", "position", "release"):
            self._control_command(action, payload)
        elif action.startswith("record_"):
            self._record_command(action, payload)
        elif action == "load_settings":
            self._load_settings(Path(payload["path"]))
            self._changed("参数已加载；需要重新连接确认", error="")
        elif action == "save_settings":
            save_settings(
                Path(payload["path"]),
                self._values,
                self._toggles,
                self._calibration_path,
                arm_settings=self._arm_settings,
            )
            self._settings_path = Path(payload["path"])
            self._changed("参数草稿已保存")
        elif action == "load_calibration":
            calibration = B601GripperCalibration.load(payload["path"])
            self._calibration, self._calibration_path = calibration, Path(payload["path"])
            self._arm_settings = {}
            self._populate_arm_settings()
            self._config_revision += 1
            self._changed("校准已加载；需要重新连接确认")
        elif action in ("enumerate_cameras", "preview_cameras", "bind_cameras", "close_camera_preview"):
            self._camera_command(action, payload, generation)
        else:
            raise ValueError(f"未知界面操作：{action}")

    def _monitor_values(self):
        needed = {key for _, key, *_ in FIELDS if key.startswith("sensor_")}
        needed |= {
            "port",
            "baudrate",
            "left_channel",
            "right_channel",
            "raw_queue_capacity",
            "acquisition_queue_capacity",
            "force_max_age_ms",
            "startup_timeout_s",
            "startup_budget_s",
            "monitor_interval_ms",
            "monitor_feedback_budget_ms",
        }
        kinds = {key: kind for _, key, _, _, kind, *_ in FIELDS}
        result = {key: kinds[key](self._values[key]) for key in needed}
        for key in (
            "monitor_interval_ms",
            "monitor_feedback_budget_ms",
            "startup_timeout_s",
            "startup_budget_s",
        ):
            positive(key, result[key])
        return result

    def _connect(self, generation):
        self._assert_current(generation)
        self._changed("正在连接只读监测…", state="connecting", error="")
        if self.demo:
            self._changed("离线演示 · 只读监测已连接", state="monitoring")
            return
        if not self._resources_stopped() or self._retained:
            raise RuntimeError("前次资源尚未退出")
        self._calibration = B601GripperCalibration.load(self._calibration_path)
        values = self._monitor_values()
        sensor_config(values)
        token = uuid.uuid4().hex
        self._monitor_token = token
        self._session_id = token
        self._monitor_cleanup = None
        self._monitor_values_pin = (self._values["port"].casefold(), asdict(self._calibration))
        self._stop_monitor.clear()

        def publish(row):
            with self._lock:
                if token == self._monitor_token and not self._stop_monitor.is_set():
                    self._telemetry = dict(row, monitor_session=token)
                    if self._state == "connecting":
                        self._changed("只读监测已连接；确认机械设置后可启动", state="monitoring")
                    else:
                        self._changed()

        def work():
            error = None
            try:
                self._services.get("monitor", monitor)(
                    self._stop_monitor,
                    publish,
                    values,
                    self._calibration,
                    lambda evidence: self._completions.put(("monitor_cleanup", token, evidence)),
                    retained=self._retained,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._completions.put(("monitor_done", token, error))

        self._assert_current(generation)
        self._monitor_thread = threading.Thread(target=work, name="b601-desktop-readonly", daemon=False)
        self._monitor_thread.start()

    def _build_config(self, mode, *, confirmed, payload=None):
        if self._calibration is None:
            self._calibration = B601GripperCalibration.load(self._calibration_path)
        self._calibration.require_unchanged()
        standalone_hybrid = mode == "gripper_hybrid"
        mode = "hybrid" if standalone_hybrid else mode
        if not self.gripper_only and mode == "hybrid" and not standalone_hybrid:
            document, _ = build_operator_document(
                self._values,
                self._toggles,
                self._calibration,
                supported_arm=bool(confirmed),
                arm_settings=self._arm_settings,
            )
            config = draccus.decode(ForceTeacherRunConfig, document)
        else:
            config = build_config(
                self._values,
                self._toggles,
                self._calibration,
                self._output_root / "pending",
                # The initial position is attached below before the final validation.
                initial_mode="force" if mode == "position" else mode,
            )
        if mode == "position":
            payload = payload or {}
            target = payload.get("position_deg", self._values["position_target_deg"])
            speed = payload.get("position_speed_deg_s", self._values["position_speed_deg_s"])
            target = float(target) if isinstance(target, str) else target
            speed = float(speed) if isinstance(speed, str) else speed
            config = replace(
                config,
                initial_mode=mode,
                initial_position_deg=finite_number("指定绝对位置 (°)", target),
                initial_position_speed_deg_s=positive("定位速度 (°/s)", speed),
            )
        override = command_feedback_override(config, self.command_feedback_mode, self.command_tx_gap_us)
        if override is not None:
            config = replace(
                config,
                b601=replace(
                    config.b601,
                    control=replace(config.b601.control, **override),
                ),
            )
        config.validate()
        return config

    def _recording_config(self):
        values = self._recording_values
        config = {}
        for key in ("root", "repo_id", "task", "object_id", "contact_condition_id"):
            value = str(values.get(key, "")).strip()
            if not value:
                raise ValueError(f"示范录制须填写 {key}")
            config[key] = value
        config["root"] = str(Path(config["root"]).expanduser().resolve())
        config.update(
            fps=50,
            h264_encoder=values.get("h264_encoder", "nvenc"),
            verify_video_on_save=values.get("verify_video_on_save", False),
            resume=values.get("resume") is True,
            camera_max_age_ms=positive("相机最大年龄", float(values["camera_max_age_ms"])),
            queue_capacity=int(values["queue_capacity"]),
            target_episode_count=int(values["target_episode_count"]),
            camera_configs=copy.deepcopy(self._cameras),
        )
        TeacherRecordConfig(**config).validate()
        return config

    def _start_control(self, payload, generation):
        if payload.get("confirmed") is not True:
            raise ValueError("请确认机械设置、闭合方向和力通道，手已移开")
        if self.demo:
            self._control_is_gripper = payload.get("mode") in (
                "force",
                "gripper_hybrid",
                "approach",
                "position",
            )
            self._recording = {
                "state": "ready",
                "active": False,
                "busy": False,
                "frame_count": 0,
                "episode_count": 0,
                "completed_episode_count": 0,
                "simulated": True,
                "training_eligible": False,
            }
            self._changed("离线演示 · 遥操作已启动", state="running", error="")
            return
        reference = dict(self._telemetry)
        age = positive("启动反馈最大年龄", float(self._values["start_feedback_max_age_ms"])) * 1e6
        if not reference or not 0 <= time.perf_counter_ns() - reference.get("joint_received_ns", 0) <= age:
            raise ValueError("请等待新鲜只读反馈")
        if reference.get("phase") != "DISABLED" or not self._exit_confirmed:
            raise ValueError("当前反馈尚未确认失能")
        self._changed("正在检查配置和录制目录…", state="preflighting", error="")
        mode = payload.get("mode", "force" if self.gripper_only else "hybrid")
        config = self._build_config(mode, confirmed=True, payload=payload)
        if len(config.runtime.joint_names) != 7 and self._recording_values.get("enabled"):
            raise ValueError("示范录制需要七轴遥操；单独夹爪控制请关闭录制准备")
        recording = self._recording_config() if self._recording_values.get("enabled") else None
        if recording:
            TeacherRecordConfig(**recording).validate()
        if (config.b601.port.casefold(), asdict(self._calibration)) != self._monitor_values_pin:
            raise ValueError("端口或校准已变化，请重新连接")
        self._assert_current(generation)
        reference = dict(self._telemetry)
        if (
            not 0 <= time.perf_counter_ns() - reference.get("joint_received_ns", 0) <= age
            or reference.get("phase") != "DISABLED"
        ):
            raise ValueError("预检后反馈已过期，请重新取得失能反馈")
        folder = self._output_root / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "reviewed.json"
        config = replace(config, trace_root=folder / "run/stream")
        operator_values = dict(self._values)
        if mode == "position":
            operator_values.update(
                position_target_deg=str(config.initial_position_deg),
                position_speed_deg_s=str(config.initial_position_speed_deg_s),
            )
        draft = {
            "config": draccus.encode(config),
            "calibration": asdict(self._calibration),
            "operator_values": operator_values,
            "operator_toggles": dict(self._toggles),
            "reference": reference,
            "exit_recovery": self._recovery_evidence,
            "command_feedback_override": command_feedback_override(
                config, self.command_feedback_mode, self.command_tx_gap_us
            ),
            "executable": True,
            "consumed": False,
            "display": {key: self._values[key] for key in ("display_hz", "plot_seconds", "ipc_poll_ms")},
        }
        if recording:
            draft["recording"] = recording
        draft["default_full_arm_mit"] = len(config.runtime.joint_names) == 7
        path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._services.get("review", run)(folder / "run", reviewed_config=path)
        self._assert_current(generation)
        with self._lock:
            fresh = dict(self._telemetry)
            if (
                not 0 <= time.perf_counter_ns() - fresh.get("joint_received_ns", 0) <= age
                or fresh.get("phase") != "DISABLED"
            ):
                raise ValueError("预检完成后反馈已过期，请重新连接只读反馈")
            draft["reference"] = fresh
        path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._changed("正在交接只读资源…", state="starting")
        self._stop_monitor.set()
        while self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(0.02)
            self._drain_completions()
            if generation != self._generation or self._closing:
                draft.update(executable=False, consumed=True)
                path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError("交接已取消，正在等待只读资源退出")
        self._drain_completions()
        cleanup = self._monitor_cleanup or {}
        if (
            cleanup.get("transport_closed") is not True
            or cleanup.get("sensor_closed") is not True
            or self._retained
        ):
            self._exit_confirmed = False
            raise RuntimeError("只读资源退出未确认，禁止启动")
        try:
            self._assert_current(generation)
        except RuntimeError:
            draft.update(executable=False, consumed=True)
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        live = self._services.get("live_factory", ProcessLive)(
            poll_s=float(self._values["ipc_poll_ms"]) / 1000
        )
        if hasattr(live, "set_preview_enabled"):
            live.set_preview_enabled(self._preview_enabled)
        token = uuid.uuid4().hex

        def owner():
            try:
                result = self._services.get("run_in_process", run_in_process)(
                    folder / "run", reviewed_config=path, live=live
                )
                self._completions.put(("owner_done", token, (result, folder / "run/result.json")))
            except Exception as exc:
                self._completions.put(("owner_error", token, f"{type(exc).__name__}: {exc}"))

        with self._lock:
            try:
                self._assert_current(generation)
            except RuntimeError:
                live.close()
                raise
            self._live, self._config = live, config
            self._control_is_gripper = len(config.runtime.joint_names) == 1
            self._session_id = token
            self._exit_confirmed = False
            self._unconfirmed_port, self._unconfirmed_calibration = (
                config.b601.port,
                asdict(self._calibration),
            )
            self._telemetry, self._recording = {}, {}
            self._owner_thread = threading.Thread(target=owner, name="b601-desktop-control", daemon=False)
            self._owner_thread.start()
            self._changed("正在启动控制会话…", state="starting")

    def _control_command(self, action, payload):
        if "values" in payload or "toggles" in payload:
            self._update_form({key: payload[key] for key in ("values", "toggles") if key in payload})
        if action == "apply" and self._state != "running":
            if not self.demo:
                self._build_config("force", confirmed=False)
            self._changed("参数有效，下次启动时使用")
            return
        kind = "release_and_finish" if action == "release" else action
        value = None
        if action == "target":
            value = finite_number("目标力", float(payload.get("value", self._values["target_N"])))
        elif action == "apply":
            config = self._build_config("force", confirmed=False)
            kind, value = "parameters", {"teacher": config.teacher, "safety": config.safety}
        elif action == "approach":
            value = {
                key: positive(key, float(payload.get(key, self._values[key])))
                for key in ("gripper_follow_speed_deg_s", "contact_force_threshold_N")
            }
        elif action == "position":
            value = {
                "position_deg": float(payload.get("position_deg", self._values["position_target_deg"])),
                "position_speed_deg_s": float(
                    payload.get("position_speed_deg_s", self._values["position_speed_deg_s"])
                ),
            }
            if not self.demo:
                validate_motion_request(self._config.teacher, self._config.safety, "position", **value)
        if self.demo:
            if action == "target":
                self._values = {**self._values, "target_N": str(value)}
            elif action == "resume":
                self._telemetry = {**self._telemetry, "phase": "FORCE_TRACK"}
            elif action == "release":
                self._changed("离线演示 · 主动释放已结束", state="disconnected")
                return
        else:
            self._live.submit(kind, value)
            if action == "release":
                self._changed("正在主动释放并结束会话；停止 / Esc 始终可用", state="releasing")
                return
        self._changed("操作请求已发送，等待实际应用反馈")

    def _record_command(self, action, payload):
        if action not in ("record_start", "record_finish", "record_discard", "record_outcome"):
            raise ValueError("当前采集后端不支持此录制操作")
        if action == "record_start":
            value = {"config": self._recording_config()} if not self.demo else {"config": {}}
            self._verification = {"state": "idle"}
        elif action == "record_discard":
            value = {"rerecord": payload.get("rerecord", True)}
        else:
            value = {"task_result": payload.get("task_result", "unknown")}
            if value["task_result"] not in ("success", "failure", "unknown"):
                raise ValueError("请选择成功、失败或未确认")
        if self.demo:
            if action == "record_start" or action == "record_discard":
                self._recording = {
                    **self._recording,
                    **{
                        "state": "recording",
                        "active": True,
                        "frame_count": 0,
                        "episode_uid": uuid.uuid4().hex,
                    },
                }
            elif action == "record_finish":
                count = self._recording.get("completed_episode_count", 0) + 1
                self._recording = {
                    **self._recording,
                    **{
                        "state": "saved",
                        "active": False,
                        "completed_episode_count": count,
                        "episode_count": count,
                        "task_result": value.get("task_result", "unknown"),
                        "simulated": True,
                        "file_complete": False,
                    },
                }
            else:
                self._recording = {**self._recording, **{"task_result": value["task_result"]}}
        else:
            self._live.submit(action, value)
            self._pending_record = (action, time.monotonic(), copy.deepcopy(self._recording), value)
        self._changed(
            "离线演示 · 录制状态已更新，未保存正式数据" if self.demo else "录制请求已发送，等待保存与核验反馈"
        )

    def _accept_result(self, result, path):
        self._last_result, self._last_result_path = result, Path(path)
        self._exit_confirmed = exit_confirmation(result, log_path=path)
        self._unconfirmed_port = exit_port(result)
        self._unconfirmed_calibration = result.get("calibration")
        if isinstance(result.get("recording"), dict):
            self._recording = copy.deepcopy(result["recording"])
        report = stop_report(result, path)
        self._result_errors = tuple(str(error) for error in (result.get("errors") or ()))
        self._error = report["summary"] if result.get("errors") else ""
        self._changed(
            report["summary"], state="disconnected" if self._exit_confirmed else "recovery_required"
        )

    def _recover(self, generation):
        if self.demo:
            self._exit_confirmed = True
            self._changed("离线演示 · 只读复核通过", state="disconnected")
            return
        if not self._resources_stopped() or self._retained:
            raise RuntimeError("仍有设备资源，不能重复复核")
        calibration = B601GripperCalibration.load(self._calibration_path)
        port = self._values["port"]
        if not self._unconfirmed_port or port.casefold() != self._unconfirmed_port.casefold():
            raise ValueError("恢复必须使用上次会话原串口")
        if asdict(calibration) != self._unconfirmed_calibration:
            raise ValueError("恢复必须使用上次会话的原校准")
        if self._last_result is not None and not full_arm_result(self._last_result):
            self._recover_gripper(calibration, generation)
            return
        # Always obtain the stronger seven-axis proof; a missing calibration is a refusal.
        full = B601ArmCalibration.load(
            calibration.path, {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
        )
        token = uuid.uuid4().hex
        max_age = positive("只读参考最大年龄", float(self._values["start_feedback_max_age_ms"]) / 1000)
        self._stop_monitor.clear()
        self._changed("正在独占原串口，只读复核七关节…", state="recovering")
        receipt = self._services.get("query_disabled_arm", query_disabled_arm)(
            port,
            full,
            baudrate=int(self._values["baudrate"]),
            mapping_budget_s=float(self._values["startup_budget_s"]),
            feedback_budget_s=float(self._values["monitor_feedback_budget_ms"]) / 1000,
            max_age_s=max_age,
            stop=self._stop_monitor,
            session_id=token,
            transport_factory=self._services.get("readonly_factory", ReadonlyB601Transport),
            retain=lambda device: self._retained.append(device),
        )
        self._assert_current(generation)
        if self._retained:
            raise ValueError("只读传输仍持有资源，不能解除恢复门禁")
        full.require_unchanged()
        calibration.require_unchanged()
        if (
            any(
                receipt.get(key) is not expected
                for key, expected in {
                    "complete": True,
                    "read_only": True,
                    "motion_attempted": False,
                    "parameter_writes": False,
                    "all_joints_disabled_observed": True,
                    "transport_closed": True,
                }.items()
            )
            or receipt.get("errors")
            or receipt.get("session_id") != token
            or receipt.get("port", "").casefold() != port.casefold()
            or receipt.get("calibration") != asdict(full)
        ):
            raise ValueError("七轴失能反馈或资源关闭证据不完整")
        feedback = receipt.get("feedback") or {}
        if feedback.get("is_valid") is False or feedback.get("error"):
            raise ValueError("七轴只读反馈无效")
        rows = feedback.get("feedback")
        started, observed, closed = (receipt.get(key) for key in ("started_ns", "observed_ns", "closed_ns"))
        if (
            any(type(value) is not int for value in (started, observed, closed))
            or not started <= observed <= closed
        ):
            raise ValueError("只读复核时间边界不完整")
        now = time.perf_counter_ns()
        if not closed <= now or not 0 <= now - observed <= max_age * 1e9:
            raise ValueError("只读复核收据已过期或时间来自未来")
        expected = {spec.name: spec for spec in DEFAULT_B601_JOINTS}
        if (
            not isinstance(rows, (list, tuple))
            or len(rows) != 7
            or {row.get("name") for row in rows} != set(expected)
        ):
            raise ValueError("缺少七个不同关节的反馈")
        for row in rows:
            spec = expected[row["name"]]
            stamp = row.get("receive_timestamp_ns")
            sequence = row.get("receive_sequence")
            position = finite_number("关节位置", row.get("position_deg"))
            bounds = next((lower, upper) for name, _, lower, upper in full.joints if name == spec.name)
            if (
                row.get("motor_id") != spec.motor_id
                or row.get("feedback_id") != spec.feedback_id
                or row.get("status_code") != 0
                or type(stamp) is not int
                or not started <= stamp <= observed
                or observed - stamp > max_age * 1e9
                or type(sequence) is not int
                or sequence <= 0
                or not bounds[0] <= position <= bounds[1]
            ):
                raise ValueError(f"关节失能反馈不新鲜：{spec.name}")
        folder = self._output_root / "readonly_recovery"
        folder.mkdir(parents=True, exist_ok=True)
        evidence = dict(receipt, previous_exit_log=str(self._last_result_path))
        if self._last_result_path and self._last_result_path.is_file():
            evidence["previous_exit_sha256"] = hashlib.sha256(self._last_result_path.read_bytes()).hexdigest()
        with (folder / f"{token}-full-arm.json").open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        self._recovery_evidence, self._exit_confirmed = evidence, True
        self._telemetry = {}
        self._changed("七关节只读复核通过且串口已关闭；请重新连接后确认启动", state="disconnected", error="")

    def _recover_gripper(self, calibration, generation):
        """Retain standalone fresh force/disabled proof without requiring arm calibration."""
        token = uuid.uuid4().hex
        started = time.perf_counter_ns()
        observed, cleanup = {}, {}
        self._stop_monitor.clear()
        self._changed("正在原串口只读复核夹爪失能及新鲜力反馈…", state="recovering")

        def publish(row):
            reference = dict(row, monitor_session=token)
            if not readonly_exit_reference(
                reference,
                session_token=token,
                started_ns=started,
                now_ns=time.perf_counter_ns(),
                joint_age_ms=float(self._values["start_feedback_max_age_ms"]),
                force_age_ms=float(self._values["force_max_age_ms"]),
            ):
                raise ValueError("夹爪失能或力反馈未通过新鲜只读复核")
            if not calibration.minimum_deg <= reference["q_measured_deg"] <= calibration.maximum_deg:
                raise ValueError("失能位置超出当前校准范围")
            observed.update(reference)
            self._stop_monitor.set()

        self._services.get("monitor", monitor)(
            self._stop_monitor,
            publish,
            self._monitor_values(),
            calibration,
            cleanup.update,
            retained=self._retained,
        )
        self._assert_current(generation)
        calibration.require_unchanged()
        if (
            not observed
            or self._retained
            or cleanup.get("transport_closed") is not True
            or cleanup.get("sensor_closed") is not True
        ):
            raise ValueError("夹爪只读反馈或资源关闭证据不完整")
        if not readonly_exit_reference(
            observed,
            session_token=token,
            started_ns=started,
            now_ns=time.perf_counter_ns(),
            joint_age_ms=float(self._values["start_feedback_max_age_ms"]),
            force_age_ms=float(self._values["force_max_age_ms"]),
        ):
            raise ValueError("夹爪只读复核收据已过期")
        evidence = {
            "source": "fresh_query_only_monitor",
            "port": self._values["port"],
            "motor_id": 7,
            "read_only": True,
            "motion_attempted": False,
            "parameter_writes": False,
            "hardware_ack": "unknown",
            "disabled_feedback_observed": True,
            "calibration": asdict(calibration),
            "reference": observed,
            "cleanup": cleanup,
            "previous_exit_log": str(self._last_result_path),
        }
        folder = self._output_root / "readonly_recovery"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{token}-gripper.json").open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        self._recovery_evidence, self._exit_confirmed = evidence, True
        self._telemetry = {}
        self._changed(
            "夹爪已只读观察到失能且资源已关闭；请重新连接后确认启动", state="disconnected", error=""
        )

    def _camera_command(self, action, payload, generation):
        if action == "enumerate_cameras":
            if self.demo:
                devices = [
                    {"device_name": f"合成相机 {i + 1}", "device_path": f"synthetic/{role}"}
                    for i, role in enumerate(("front", "wrist"))
                ]
            else:
                from lerobot.cameras.opencv.device_binding import enumerate_devices

                devices = enumerate_devices(700)
            self._assert_current(generation)
            self._devices = tuple(devices)
            self._changed("相机列表已刷新")
        elif action == "close_camera_preview":
            self._close_camera_preview()
            self._changed("相机预览已关闭")
        elif action in ("preview_cameras", "bind_cameras"):
            if (
                set(payload) != {"front", "wrist"}
                or not all(payload.values())
                or len(set(payload.values())) != 2
            ):
                raise ValueError("请为前景和腕部选择两台不同相机")
            config = copy.deepcopy(self._cameras)
            for role, path in payload.items():
                for key in ("device_name", "index_or_path"):
                    config[role].pop(key, None)
                config[role].update(device_path=path, backend=700, fps=50)
            if action == "preview_cameras":
                self._close_camera_preview()
                if self.demo:
                    self._camera_session = "synthetic"
                else:
                    from b601_camera_preview import PreviewSession

                    self._camera_session = PreviewSession()
                    self._camera_session.start(config)
                self._camera_selection = config
                self._camera_seen.clear()
                self._assert_current(generation)
                self._changed("请核对前景和腕部画面，确认后保存绑定")
            else:
                if self._camera_selection != config or self._camera_seen != {"front", "wrist"}:
                    raise ValueError("请先预览当前选择的两路画面")
                self._close_camera_preview()
                self._assert_current(generation)
                if not self.demo:
                    temporary = BINDINGS_FILE.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    temporary.replace(BINDINGS_FILE)
                self._cameras = config
                self._config_revision += 1
                self._changed("相机角色绑定已保存" if not self.demo else "离线演示 · 相机角色已确认")

    def _close_camera_preview(self):
        if self._camera_session is not None and self._camera_session != "synthetic":
            self._camera_session.close()
        self._camera_session = self._camera_selection = None
        self._camera_seen.clear()
        self._preview = {}

    def _drain_completions(self):
        while True:
            try:
                kind, token, value = self._completions.get_nowait()
            except queue.Empty:
                return
            if kind.startswith("monitor"):
                if token != self._monitor_token:
                    continue
                if kind == "monitor_cleanup":
                    self._monitor_cleanup = value
                    if value.get("transport_closed") is not True or value.get("sensor_closed") is not True:
                        self._exit_confirmed = False
                        self._unconfirmed_port = self._values["port"]
                        self._unconfirmed_calibration = asdict(self._calibration)
                elif value:
                    self._changed(f"只读监测停止：{value}", error=value)
                if kind == "monitor_done" and (
                    not self._monitor_cleanup
                    or self._monitor_cleanup.get("transport_closed") is not True
                    or self._monitor_cleanup.get("sensor_closed") is not True
                ):
                    self._exit_confirmed = False
                    self._unconfirmed_port = self._values["port"]
                    self._unconfirmed_calibration = asdict(self._calibration) if self._calibration else None
                    self._changed(
                        "只读资源退出缺少关闭证据，请等待清理或只读复核", error=value or "资源关闭未确认"
                    )
            elif token == self._session_id:
                self._pending_record = None
                if kind == "owner_done":
                    self._accept_result(*value)
                else:
                    self._exit_confirmed = False
                    self._changed(f"控制进程退出未确认：{value}", state="recovery_required", error=value)

    def _start_verification(self):
        from lerobot.datasets.sensor_verification import check_no_live_writer

        root = Path(str(self._recording_values.get("root", ""))).expanduser().resolve()
        if not root.name or not (root / "meta/info.json").is_file():
            raise ValueError("请选择已有的 Teacher 数据目录")
        check_no_live_writer(root)
        self._verification_baseline = self._verification_identity(root)
        self._verification_cancel.clear()
        self._verification_result = None
        self._verification_started = time.perf_counter()
        self._verification = {
            "state": "running",
            "root": str(root),
            "started_utc": datetime.now(UTC).isoformat(),
            "index": 0,
            "total": 0,
            "video": "",
        }
        factory = self._services.get("verification_factory", VerificationWorker)
        try:
            self._verification_worker = factory(PROJECT, root)
        except Exception:
            self._verification = {"state": "idle"}
            raise
        self._changed("正在完整核验数据集；可随时取消", error="")

    @staticmethod
    def _verification_identity(root):
        """Detect appended/changed episode metadata without writing the dataset."""
        paths = [root / "meta/info.json", *sorted((root / "meta/teacher_episodes").glob("*.json"))]
        lock = root / ".sensor-writer.lock"
        if lock.exists():
            paths.append(lock)
        return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)

    def _verification_events(self, events):
        for event in events:
            if event.get("event") == "episode":
                self._verification.update(
                    index=event.get("index", 0),
                    total=event.get("total", 0),
                    episode_uid=event.get("episode_uid", ""),
                    video="",
                )
            elif event.get("event") == "video":
                self._verification["video"] = event.get("video", "")
            elif event.get("event") == "result":
                self._verification_result = event.get("report")

    def _poll_verification(self):
        worker = self._verification_worker
        if worker is None:
            if (
                self._verification.get("state") == "passed"
                and time.monotonic() >= self._verification_check_at
            ):
                self._verification_check_at = time.monotonic() + 1
                try:
                    current = self._verification_identity(Path(self._verification["root"]))
                except OSError:
                    current = None
                if current != self._verification_baseline:
                    self._verification = {"state": "idle"}
                    self._changed("数据集内容已变化，需重新完整核验")
            return
        cancelled = self._verification_cancel.is_set()
        if cancelled and not worker.finished:
            worker.cancel()
        events = worker.poll()
        self._verification_events(events)
        if not worker.finished:
            if events:
                current = self._verification
                message = f"完整核验：第 {current['index']} / {current['total']} 条"
                if current.get("video"):
                    message += f" · {current['video']}"
                self._changed(message)
            return
        self._verification_events(worker.poll())
        worker.close()
        report = self._verification_result
        if cancelled:
            state, errors = "cancelled", ["完整核验已取消，尚未完成"]
        elif worker.process.returncode == 0 and isinstance(report, dict) and report.get("valid") is True:
            state, errors = "passed", []
        else:
            state = "failed"
            errors = list(report.get("errors", [])) if isinstance(report, dict) else []
            if not errors:
                errors = ["核验进程未返回有效结果：" + "；".join(worker.output[-3:])]
        if state == "passed":
            try:
                unchanged = (
                    self._verification_identity(Path(self._verification["root"]))
                    == self._verification_baseline
                )
            except OSError:
                unchanged = False
            if not unchanged:
                state, errors = "failed", ["核验期间数据集发生变化，请停止写入后重新核验"]
        self._verification.update(
            state=state,
            errors=errors,
            elapsed_seconds=time.perf_counter() - self._verification_started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        try:
            root = Path(self._verification["root"])
            directory = root.parent / f"{root.name}_verification_reports"
            resolved = directory.resolve()
            if resolved == root or root in resolved.parents:
                raise ValueError("核验报告目录不能位于数据集内")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"verification_{uuid.uuid4().hex}.json"
            temporary = path.with_suffix(".tmp")
            payload = {**self._verification, "scope": "all_saved_episodes", "result": report}
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            self._verification["report_path"] = str(path)
        except (OSError, ValueError, TypeError) as exc:
            self._verification["state"] = "failed"
            self._verification["errors"] = [*errors, f"无法保存核验报告：{exc}"]
        self._verification_worker = None
        labels = {
            "passed": "完整核验通过",
            "failed": "完整核验失败，请查看原因",
            "cancelled": "完整核验未完成",
        }
        self._changed(labels[self._verification["state"]])

    def _resources_stopped(self):
        return not any(
            thread is not None and thread.is_alive() for thread in (self._monitor_thread, self._owner_thread)
        )

    def _poll(self):
        self._drain_completions()
        self._poll_verification()
        if self.demo:
            self._demo_poll()
        live = self._live
        if live is not None and not live.closed:
            live.receive()
            if live.latest is not None:
                self._telemetry = copy.deepcopy(live.latest)
            if live.recording is not None:
                self._recording = copy.deepcopy(live.recording)
                if self._pending_record and self._record_request_observed():
                    self._pending_record = None
            if self._pending_record and time.monotonic() - self._pending_record[1] > 5:
                # Do not blindly replay: the same episode may already be committed.
                self._changed("录制请求仍待反馈，请检查诊断；不会自动重复保存")
            if self._preview_enabled:
                self._preview = dict(live.preview or {})
            if live.startup and live.startup.get("complete") and self._state == "starting":
                self._changed("遥操作正在运行", state="running")
            if live.ipc_errors:
                self.stop()
                self._changed("控制进程通信中断，正在等待退出证据", error=live.ipc_errors[-1])
            if live.applied:
                teacher = live.applied.get("teacher") or {}
                self._telemetry = {**self._telemetry, "applied_target_N": teacher.get("force_target_N")}
            self._changed()
        if (
            self._camera_session is not None
            and self._preview_enabled
            and time.monotonic() - self._preview_last >= 0.2
        ):
            self._preview_last = time.monotonic()
            if self._camera_session == "synthetic":
                self._demo_preview()
            else:
                from PIL import Image

                for role, camera in self._camera_session.cameras.items():
                    frame = camera.read_latest_snapshot()
                    if frame is None:
                        continue
                    picture = Image.fromarray(frame.image)
                    picture.thumbnail((640, 480))
                    stream = io.BytesIO()
                    picture.save(stream, format="JPEG", quality=70)
                    self._preview[role] = {"jpeg": stream.getvalue(), "timestamp_ns": frame.timestamp_ns}
                    self._camera_seen.add(role)
        if (
            self._state in ("stopping", "connecting", "monitoring")
            and self._resources_stopped()
            and (not self.demo or self._state == "stopping")
        ):
            self._changed(
                "设备已断开" if self._exit_confirmed else "退出未确认，请进行只读复核",
                state="disconnected" if self._exit_confirmed else "recovery_required",
            )

    def _demo_poll(self):
        if self._state not in ("monitoring", "running"):
            return
        self._demo_tick += 1
        now = time.perf_counter_ns()
        target = float(self._values.get("target_N") or 5)
        force = target + 0.2 * math.sin(self._demo_tick * 0.05) if self._state == "running" else 0.02
        phase = self._telemetry.get("phase", "FORCE_TRACK") if self._state == "running" else "DISABLED"
        if phase == "DISABLED" and self._state == "running":
            phase = "FORCE_TRACK"
        self._telemetry = {
            **self._telemetry,
            **{
                "decision_anchor_ns": now,
                "tick_id": self._demo_tick,
                "phase": phase,
                "force_left_N": force + 0.05,
                "force_right_N": force - 0.05,
                "force_mean_N": force,
                "force_target_requested_N": target,
                "force_target_effective_N": target,
                "q_sent_deg": -80.0,
                "q_measured_deg": -80.0,
                "delta_i_deg": 0.01,
                "delta_p_deg": -0.005,
                "delta_final_deg": 0.005,
                "joint_received_ns": now,
                "force_timestamp_ns": now,
                "joint_valid": True,
                "force_valid": True,
                "read_only": self._state == "monitoring",
                "simulated": True,
            },
        }
        if self._recording.get("active"):
            self._recording = {**self._recording, "frame_count": self._recording.get("frame_count", 0) + 1}
        if self._preview_enabled and time.monotonic() - self._demo_last_preview >= 0.1:
            self._demo_preview()
        self._changed()

    def _record_request_observed(self):
        action, _, before, value = self._pending_record
        current = self._recording
        state = current.get("state")
        if current.get("busy") or current.get("error") or state in ("corrupt", "commit_unknown"):
            return True
        if action == "record_start":
            return current.get("active") is True or current.get("episode_uid") != before.get("episode_uid")
        if action == "record_finish":
            return not current.get("active", False) or state in (
                "saving",
                "verifying",
                "saved",
                "saved_pending_verification",
            )
        if action == "record_discard":
            return current.get("episode_uid") != before.get("episode_uid") or not current.get("active", False)
        return current.get("task_result") == value.get("task_result")

    def _demo_preview(self):
        from PIL import Image, ImageDraw

        self._demo_last_preview = time.monotonic()
        for role, color in (("front", (222, 235, 244)), ("wrist", (231, 232, 246))):
            picture = Image.new("RGB", (640, 480), color)
            draw = ImageDraw.Draw(picture)
            x = 240 + int(40 * math.sin(self._demo_tick * 0.04))
            draw.rounded_rectangle((x, 160, x + 140, 340), radius=20, fill=(78, 130, 166))
            draw.line((0, 360, 640, 360), fill=(165, 183, 198), width=3)
            draw.text((20, 20), f"SYNTHETIC / {role.upper()} / {self._demo_tick}", fill=(42, 65, 90))
            stream = io.BytesIO()
            picture.save(stream, format="JPEG", quality=75)
            self._preview[role] = {"jpeg": stream.getvalue(), "timestamp_ns": time.perf_counter_ns()}
            if self._camera_session is not None:
                self._camera_seen.add(role)
