"""Recording form and bounded snapshot rendering; no device is opened by this module."""

import io
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import draccus
from b601_profile_review import (
    config_document,
    full_arm_profile_text,
    operator_teleoperation_summary,
    profile_fingerprint,
    require_profile_pin,
    require_validation_evidence_pin,
    validation_evidence_fingerprint,
)
from draccus.utils import DecodingError

from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig

TASK_RESULTS = {"未确认": "unknown", "成功": "success", "失败": "failure"}


def recording_config(values, cameras_text):
    """Parse an explicit form without connecting cameras or authorizing motion."""
    config = {}
    for name in ("root", "repo_id", "task", "object_id", "contact_condition_id"):
        value = str(values.get(name, "")).strip()
        if not value:
            raise ValueError(f"示范录制须填写 {name}")
        config[name] = value
    for name in ("camera_max_age_ms",):
        value = float(values.get(name, ""))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 须为正有限值")
        config[name] = value
    for name in ("fps", "queue_capacity", "target_episode_count"):
        value = int(values.get(name, ""))
        if value <= 0:
            raise ValueError(f"{name} 须为正整数")
        config[name] = value
    config["resume"] = values.get("resume") is True
    config["verify_video_on_save"] = values.get("verify_video_on_save") is True
    config["h264_encoder"] = str(values.get("h264_encoder", "nvenc"))
    cameras = json.loads(cameras_text)
    if not isinstance(cameras, dict) or not cameras or len(cameras) > 4:
        raise ValueError("配置一至四路相机，JSON 根对象为相机名称映射")
    if any(
        not isinstance(name, str) or not name or not isinstance(item, dict) for name, item in cameras.items()
    ):
        raise ValueError("每个相机须有名称及配置对象")
    json.dumps(cameras, allow_nan=False)
    config["camera_configs"] = cameras
    config["root"] = str(Path(config["root"]).expanduser().resolve())
    return config


def load_full_arm_profile(path):
    """Decode a complete profile; hardware acceptance remains the backend's gate."""
    from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS

    path = Path(path).expanduser().resolve()
    document = config_document(json.loads(path.read_text(encoding="utf-8")))
    try:
        config = draccus.decode(ForceTeacherRunConfig, document)
    except DecodingError as exc:
        raise ValueError(f"全臂配置解码失败：{exc}") from exc
    names = tuple(joint.name for joint in DEFAULT_B601_JOINTS)
    if tuple(config.runtime.joint_names) != names or config.initial_mode != "hybrid":
        raise ValueError("全臂配置须为完整七关节标准顺序且 initial_mode=hybrid")
    if config.recording is not None:
        raise ValueError("面板全臂配置的 recording 须为 null；请在示范录制页填写唯一录制配置")
    return config


def recording_description(snapshot):
    if not isinstance(snapshot, dict):
        return "尚未开始示范录制；控制会话和 episode 独立起止"
    completed = snapshot.get("completed_episode_count", snapshot.get("episode_count", 0))
    target = snapshot.get("target_episode_count", 10)
    successful = snapshot.get("successful_episode_count", 0)
    video_check = {"passed": "已通过", "not_run": "待完整核验"}.get(
        snapshot.get("video_verification"), "未记录"
    )
    counts = snapshot.get("stage_counts") or {}
    coverage = " / ".join(
        f"{label} {counts.get(key, 0)}"
        for key, label in (
            ("approach", "接近"),
            ("contact_transition", "接触交接"),
            ("force_control", "力控"),
            ("release", "释放 / 人工张开"),
        )
    )
    return (
        f"录制状态：{snapshot.get('state', snapshot.get('status', 'unknown'))} | 已保存 {completed} 条 | "
        f"人工成功 {successful} / {target}\n"
        f"文件完整：{snapshot.get('file_complete', '待核验')} | 阶段拍数：{coverage}\n"
        f"保存时视频完整核验：{video_check}\n"
        f"人工任务结果：{snapshot.get('task_result', 'unknown')}\n"
        f"采集错误：{snapshot.get('error') or '; '.join(snapshot.get('errors') or []) or '无'}\n"
        f"操作请求错误：{snapshot.get('request_error') or '无'}"
    )


class RecordingPage:
    """Tk form emits requests; all capture and encoding belongs to backend workers."""

    def __init__(self, parent, command, *, profile_parent=None, before_check=None, source_selected=None):
        self.command = command
        self.before_check = before_check or (lambda: None)
        self.source_selected = source_selected or (lambda path: None)
        self.last_episode_uid = None
        self.checked_profile_pin = None
        self.checked_profile_bounded = False
        self.checked_profile_operator = False
        self.checked_validation_evidence_pin = None
        self.configuration_widgets = []
        self.fields = {}
        self.enabled = tk.BooleanVar(value=False)
        enable = ttk.Checkbutton(parent, text="为下次控制会话准备示范录制", variable=self.enabled)
        enable.pack(anchor="w")
        self.configuration_widgets.append(enable)
        ttk.Label(
            parent,
            text="采集版：控制 / 主数据 / 视频统一 50 Hz，默认 NVENC。\n先在全臂会话中遥操，放下并手动张开后结束保存。",
            wraplength=430,
        ).pack(anchor="w", pady=6)
        form = ttk.Frame(parent)
        form.pack(fill="x")
        definitions = (
            ("root", "数据目录", "outputs/teacher_capture_50hz"),
            ("repo_id", "数据集 ID", "local/force_teacher_demo"),
            ("task", "任务描述", ""),
            ("object_id", "物体", ""),
            ("contact_condition_id", "接触条件", ""),
            ("fps", "主数据 / 视频 Hz", "50"),
            ("h264_encoder", "H.264 编码器 nvenc / software", "nvenc"),
            ("camera_max_age_ms", "相机最大年龄 ms", "100"),
            ("queue_capacity", "录制队列容量", "128"),
            ("target_episode_count", "目标 episode 数", "10"),
        )
        for row, (key, label, default) in enumerate(definitions):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            self.fields[key] = tk.StringVar(value=default)
            entry = ttk.Entry(form, textvariable=self.fields[key], width=28)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.configuration_widgets.append(entry)
        form.columnconfigure(1, weight=1)
        self.resume = tk.BooleanVar(value=False)
        resume = ttk.Checkbutton(parent, text="续录已有数据集（须通过已有内容核验）", variable=self.resume)
        resume.pack(anchor="w", pady=6)
        self.configuration_widgets.append(resume)
        self.verify_video_on_save = tk.BooleanVar(value=False)
        verify_video = ttk.Checkbutton(parent, text="保存时完整核验视频", variable=self.verify_video_on_save)
        verify_video.pack(anchor="w")
        self.configuration_widgets.append(verify_video)
        ttk.Label(parent, text="相机配置 JSON（设备仅在后端准备时打开）").pack(anchor="w")
        self.cameras = tk.Text(parent, height=5, width=48, wrap="word")
        bindings = Path(__file__).resolve().parents[2] / "b601_camera_bindings.json"
        self.cameras.insert("1.0", bindings.read_text(encoding="utf-8") if bindings.is_file() else "{}")
        self.cameras.pack(fill="x")
        self.configuration_widgets.append(self.cameras)
        ttk.Label(
            parent,
            text="按 device_path 绑定 front / wrist，相机接入时重新解析设备索引；双摄均为 50 Hz。缺失设备直接报错。",
            wraplength=430,
        ).pack(anchor="w", pady=4)
        profiles = ttk.LabelFrame(profile_parent or parent, text="全臂明确配置与预检", padding=6)
        profiles.pack(fill="x", pady=8)
        self.full_arm_enabled = tk.BooleanVar(value=False)
        full_arm = ttk.Checkbutton(profiles, text="主从启动使用完整全臂配置", variable=self.full_arm_enabled)
        full_arm.pack(anchor="w")
        self.profile_path = tk.StringVar(value="")
        self.profile_path.trace_add("write", lambda *_: self.reset_profile_review())
        profile_entry = ttk.Entry(profiles, textvariable=self.profile_path)
        profile_entry.pack(fill="x", pady=4)
        choose = ttk.Button(profiles, text="选择完整配置 JSON…", command=self.choose_profile)
        choose.pack(side="left")
        check = ttk.Button(profiles, text="离线预检", command=self.check_profile)
        check.pack(side="left", padx=4)
        self.configuration_widgets.extend((full_arm, profile_entry, choose, check))
        self.profile_status = tk.StringVar(value="正常 operator 必须显式选择参数；micro 诊断为可选独立路径")
        ttk.Label(profile_parent or parent, textvariable=self.profile_status, wraplength=430).pack(
            anchor="w", pady=4
        )
        result_row = ttk.Frame(parent)
        result_row.pack(fill="x", pady=4)
        ttk.Label(result_row, text="人工任务结果").pack(side="left")
        self.task_result = tk.StringVar(value="未确认")
        ttk.Combobox(
            result_row, textvariable=self.task_result, values=tuple(TASK_RESULTS), state="readonly", width=10
        ).pack(side="left", padx=6)
        self.buttons = {}
        actions = ttk.Frame(parent)
        actions.pack(fill="x")
        for row, (kind, label) in enumerate(
            (
                ("record_start", "开始录制（不使能）"),
                ("record_finish", "结束并保存（不追加动作）"),
                ("record_discard", "丢弃并重录"),
                ("record_outcome", "确认当前 / 最近 episode 人工结果"),
            )
        ):
            button = ttk.Button(
                actions, text=label, command=lambda event=kind: self.command(event), state="disabled"
            )
            button.grid(row=row // 2, column=row % 2, sticky="ew", padx=2, pady=3)
            self.buttons[kind] = button
        actions.columnconfigure((0, 1), weight=1)
        self.status = tk.StringVar(value=recording_description(None))
        ttk.Label(parent, textvariable=self.status, wraplength=430).pack(anchor="w", pady=8)
        self.preview_frame = ttk.LabelFrame(parent, text="相机预览（最多四路，≤5 Hz）", padding=4)
        self.preview_frame.pack(fill="x")
        self.preview_labels = {}
        self.preview_images = {}
        self.preview_timestamps = {}

    def config(self):
        values = {key: var.get() for key, var in self.fields.items()}
        values["resume"] = self.resume.get()
        values["verify_video_on_save"] = self.verify_video_on_save.get()
        return recording_config(values, self.cameras.get("1.0", "end"))

    def session_options(self, *, include_full_arm_profile=True):
        options = {"recording": self.config()} if self.enabled.get() else {}
        if self.full_arm_enabled.get() and include_full_arm_profile:
            getattr(self, "before_check", lambda: None)()
            require_profile_pin(self.profile_path.get(), self.checked_profile_pin)
            config = load_full_arm_profile(self.profile_path.get())
            full_arm_profile_text(config)
            config.validate()
            require_validation_evidence_pin(config, self.checked_validation_evidence_pin)
            options["full_arm_profile_file"] = str(Path(self.profile_path.get()).resolve())
            options["full_arm_profile_pin"] = self.checked_profile_pin
            options["full_arm_validation_evidence_pin"] = self.checked_validation_evidence_pin
        return options

    def choose_profile(self):
        path = filedialog.askopenfilename(
            title="完整全臂 ForceTeacherRunConfig", filetypes=[("JSON", "*.json")]
        )
        if path:
            self.profile_path.set(path)
            self.check_profile()

    def reset_profile_review(self):
        self.checked_profile_pin = None
        self.checked_profile_bounded = False
        self.checked_profile_operator = False
        self.checked_validation_evidence_pin = None
        if hasattr(self, "profile_status"):
            self.profile_status.set("配置路径已修改；须重新离线预检并取得新鲜只读参考")

    def check_profile(self):
        description = ""
        self.checked_profile_pin = None
        self.checked_profile_bounded = False
        self.checked_profile_operator = False
        self.checked_validation_evidence_pin = None
        try:
            getattr(self, "source_selected", lambda path: None)(self.profile_path.get())
            getattr(self, "before_check", lambda: None)()
            pin = profile_fingerprint(self.profile_path.get())
            config = load_full_arm_profile(self.profile_path.get())
            description = f"配置目标力 {config.teacher.force_target_N:g} N；"
            description += full_arm_profile_text(config)
            report = config.b601.preflight_report(config.safety, config.runtime) if config.b601 else {}
            description += json.dumps(report, ensure_ascii=False) + "；"
            config.validate()
            evidence_pin = validation_evidence_fingerprint(config)
            require_profile_pin(self.profile_path.get(), pin)
            self.checked_profile_pin = pin
            self.checked_profile_operator = operator_teleoperation_summary(config) is not None
            self.checked_profile_bounded = getattr(config.b601, "validation", None) is not None
            self.checked_validation_evidence_pin = evidence_pin
            self.profile_status.set(description)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            self.profile_status.set(f"{description}全臂配置未通过离线预检：{exc}")

    def update(self, snapshot, *, active, preview=None):
        if isinstance(snapshot, dict):
            snapshot = dict(snapshot)
            snapshot.setdefault("target_episode_count", self.fields["target_episode_count"].get())
            uid = snapshot.get("episode_uid")
            if snapshot.get("active") and isinstance(uid, str) and uid != self.last_episode_uid:
                self.last_episode_uid = uid
                self.task_result.set("未确认")
        self.status.set(recording_description(snapshot))
        state = snapshot.get("state", snapshot.get("status")) if isinstance(snapshot, dict) else None
        capturing = snapshot.get("active", False) if isinstance(snapshot, dict) else False
        ready = active and state not in ("preparing", "saving")
        for kind, button in self.buttons.items():
            allowed = ready and (capturing if kind in ("record_finish", "record_discard") else not capturing)
            if kind == "record_start":
                allowed = allowed and self.enabled.get()
            elif kind == "record_outcome":
                allowed = ready and bool(snapshot and (capturing or snapshot.get("file_complete")))
            button.configure(state="normal" if allowed else "disabled")
        if preview:
            self.render_preview(preview)

    def render_preview(self, snapshots):
        from PIL import Image, ImageTk

        if not isinstance(snapshots, dict) or len(snapshots) > 4:
            return
        for name in tuple(self.preview_labels):
            if name not in snapshots:
                self.preview_labels.pop(name).destroy()
                self.preview_images.pop(name, None)
                self.preview_timestamps.pop(name, None)
        for name, frame in snapshots.items():
            if not isinstance(frame, dict) or frame.get("timestamp_ns") == self.preview_timestamps.get(name):
                continue
            jpeg = frame.get("jpeg")
            if not isinstance(jpeg, bytes) or len(jpeg) > 262_144:
                continue
            try:
                with Image.open(io.BytesIO(jpeg)) as source:
                    if source.width * source.height > 16_777_216:
                        continue
                    source.thumbnail((210, 135))
                    thumbnail = ImageTk.PhotoImage(source.copy())
            except (OSError, ValueError):
                continue
            if name not in self.preview_labels:
                label = ttk.Label(self.preview_frame, compound="top")
                label.bind(
                    "<MouseWheel>",
                    lambda event: self.preview_frame.event_generate("<MouseWheel>", delta=event.delta),
                )
                label.grid(
                    row=len(self.preview_labels) // 2, column=len(self.preview_labels) % 2, padx=2, pady=2
                )
                self.preview_labels[name] = label
            self.preview_labels[name].configure(image=thumbnail, text=name)
            self.preview_images[name] = thumbnail
            self.preview_timestamps[name] = frame.get("timestamp_ns")
