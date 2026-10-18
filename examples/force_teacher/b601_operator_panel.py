"""Explicit normal-teleoperation editing; no device, camera or monitoring operation lives here."""

import copy
import json
import math
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, ttk

from b601_profile_review import config_document, profile_fingerprint, require_profile_pin
from build_operator_teleoperation_profile import refresh_operator_mapping

from lerobot.force_control.operator_settings import (
    ARM_OPERATOR_FIELDS,
    ARM_PROFILE_FIELDS,
    checked_arm_settings,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
DEFAULT_DISABLED = "默认关闭"
OPTIONAL_OPERATOR_FIELDS = {
    "alignment_tolerance_deg",
    "preload_tolerance_deg",
    "max_command_gap_s",
    "handover_tolerance_deg",
}
OPERATOR_FIELDS = (
    ("reference_envelope_deg", "六臂距启动位置允许范围 ±(°)"),
    ("alignment_tolerance_deg", "启动主从对齐容差 (°)"),
    ("preload_tolerance_deg", "失能预载位置容差 (°)"),
    ("handover_tolerance_deg", "返回 / 接管模式对齐容差 (°)"),
    ("leader_max_age_s", "主臂反馈最大年龄 (s)"),
    ("max_command_gap_s", "正常命令最大间隔 (s)"),
    ("watchdog_ticks", "硬件 TIMEOUT ticks"),
)


def optional_number(value, label, *, integer=False, zero_allowed=False, signed=False):
    text = str(value).strip()
    if not text or text == DEFAULT_DISABLED:
        return None
    try:
        number = int(text) if integer else float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须明确填写数值") from exc
    if not math.isfinite(number) or (not signed and (number < 0 or (number == 0 and not zero_allowed))):
        raise ValueError(f"{label} 必须为{'非负' if zero_allowed else '正'}有限值")
    return number


class OperatorProfileEditor:
    """Keep unselected numerical choices empty while retaining saved controller/sensor sources."""

    def __init__(self):
        self.source = None
        self.path = None
        self.source_pin = None
        self.operator_values = dict.fromkeys((key for key, _ in OPERATOR_FIELDS), "")
        self.velocities = dict.fromkeys(JOINT_NAMES, "")
        self.kp = dict.fromkeys(JOINT_NAMES[:-1], "")
        self.kd = dict.fromkeys(JOINT_NAMES[:-1], "")
        self.relative_limits = dict.fromkeys(JOINT_NAMES, "")
        self.minimum_deg = dict.fromkeys(JOINT_NAMES[:-1], "")
        self.maximum_deg = dict.fromkeys(JOINT_NAMES[:-1], "")
        self.mit_velocity_deg_s = dict.fromkeys(JOINT_NAMES[:-1], "0")
        self.mit_torque_Nm = dict.fromkeys(JOINT_NAMES[:-1], "0")
        self.target = ""
        self.supported_arm = False
        self.duration_enabled = self.travel_enabled = False
        self.duration = self.travel = ""
        self.gripper_wire_code = None
        self.converted_to_mit = False

    def load(self, path):
        self.path = Path(path).expanduser().resolve()
        pin = profile_fingerprint(self.path)
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        require_profile_pin(self.path, pin)
        self.source = copy.deepcopy(config_document(envelope))
        self.source_pin = pin
        hardware = self.source.get("b601") or {}
        operator = hardware.get("operator_teleoperation")
        normal = isinstance(operator, dict)
        self.target = self.source.get("teacher", {}).get("force_target_N")
        self.target = "" if self.target is None else str(self.target)
        self.supported_arm = operator.get("supported_arm") is True if normal else False
        for key, _ in OPERATOR_FIELDS:
            value = (
                self.source.get("runtime", {}).get(key)
                if normal and key == "handover_tolerance_deg"
                else operator.get(key, 100.0 if key == "reference_envelope_deg" else None)
                if normal
                else None
            )
            self.operator_values[key] = (
                (DEFAULT_DISABLED if normal and key in OPTIONAL_OPERATOR_FIELDS else "")
                if value is None
                else str(value)
            )
        profiles = {row["name"]: row for row in (hardware.get("control") or {}).get("profiles", [])}
        native = RebotB601FollowerConfig(port=hardware.get("port") or "")
        self.converted_to_mit = normal and any(
            profiles.get(name, {}).get("mode") != "mit" for name in JOINT_NAMES[:-1]
        )
        code = profiles.get("gripper", {}).get("velocity_wire_code")
        self.gripper_wire_code = code if type(code) is int and 1 <= code <= 10000 else None
        for name in JOINT_NAMES:
            profile = profiles.get(name, {})
            velocity = profile.get("velocity_deg_s") if normal and name == "gripper" else None
            # A saved FORCE_POS wire code is a visible prior choice, not a new
            # degree/s choice. It remains in the source until the user enters one.
            self.velocities[name] = "" if velocity is None else str(velocity)
            relative = profile.get("max_relative_deg") if normal else None
            self.relative_limits[name] = (
                (DEFAULT_DISABLED if normal and name != "gripper" else "")
                if relative is None
                else str(relative)
            )
        for index, name in enumerate(JOINT_NAMES[:-1]):
            profile = profiles.get(name, {})
            for field in ("minimum_deg", "maximum_deg"):
                value = profile.get(field)
                getattr(self, field)[name] = "" if value is None else str(value)
            for field in ("mit_velocity_deg_s", "mit_torque_Nm"):
                getattr(self, field)[name] = str(profile.get(field, 0.0))
            for field, defaults in (("kp", native.mit_kp), ("kd", native.mit_kd)):
                gain = profile.get(field) if profile.get("mode") == "mit" else defaults[index]
                getattr(self, field)[name] = "" if gain is None else str(gain)
        safety = self.source.get("safety") or {}
        duration = self.source.get("duration_s") if normal else None
        travel = safety.get("max_total_travel_deg") if normal else None
        self.duration_enabled, self.travel_enabled = duration is not None, travel is not None
        self.duration = "" if duration is None else str(duration)
        self.travel = "" if travel is None else str(travel)
        return normal

    def document(self):
        if self.source is None:
            raise ValueError("先选择已保存的 Teacher / Safety / X518 来源配置")
        config = copy.deepcopy(self.source)
        hardware = config.setdefault("b601", {})
        previous = hardware.get("operator_teleoperation") or {}
        old_validation = hardware.get("validation") or {}
        operator = {
            "session_id": previous.get("session_id") or str(uuid.uuid4()),
            "port": hardware.get("port"),
            "calibration_file": (hardware.get("control") or {}).get("calibration_file"),
            "calibration_sha256": previous.get("calibration_sha256")
            or old_validation.get("calibration_sha256"),
            "leader_config_sha256": previous.get("leader_config_sha256")
            or old_validation.get("leader_config_sha256"),
            "supported_arm": self.supported_arm,
            "stop_policy": "disable_all_supported_arm",
        }
        for key, label in OPERATOR_FIELDS:
            value = optional_number(self.operator_values[key], label, integer=key == "watchdog_ticks")
            if key == "handover_tolerance_deg":
                config["runtime"][key] = value
            else:
                operator[key] = value
        hardware["operator_teleoperation"], hardware["validation"] = operator, None
        hardware["startup_mode"] = "prepare_disabled_operator_teleoperation"
        profiles = (hardware.setdefault("control", {})).get("profiles", [])
        if {row.get("name") for row in profiles} != set(JOINT_NAMES):
            raise ValueError("来源配置必须包含七轴 profile；不会从单轴 micro 自动补造")
        for profile in profiles:
            name = profile["name"]
            if name == "gripper":
                profile["velocity_deg_s"] = optional_number(self.velocities[name], f"{name} 速度")
                profile["velocity_wire_code"] = (
                    self.gripper_wire_code if profile["velocity_deg_s"] is None else None
                )
            else:
                profile.update(mode="mit", velocity_deg_s=None, velocity_wire_code=None, torque_ratio=None)
                profile["kp"] = optional_number(self.kp[name], f"{name} MIT Kp", zero_allowed=True)
                profile["kd"] = optional_number(self.kd[name], f"{name} MIT Kd", zero_allowed=True)
                for field in ("minimum_deg", "maximum_deg", "mit_velocity_deg_s", "mit_torque_Nm"):
                    profile[field] = optional_number(
                        getattr(self, field)[name], f"{name} {field}", signed=True
                    )
            profile["max_relative_deg"] = optional_number(
                self.relative_limits[name], f"{name} 本拍目标相对反馈允许偏差"
            )
        config["teacher"]["force_target_N"] = optional_number(self.target, "目标力", zero_allowed=True)
        hardware["control"]["watchdog_ticks"] = operator["watchdog_ticks"]
        config["leader_max_age_s"] = operator["leader_max_age_s"]
        duration = optional_number(self.duration, "可选运行时长") if self.duration_enabled else None
        config["duration_s"] = config["safety"]["max_run_time_s"] = duration
        config["safety"]["max_total_travel_deg"] = (
            optional_number(self.travel, "夹爪累计行程") if self.travel_enabled else None
        )
        config["recording"] = None
        refresh_operator_mapping(config)
        return config

    def missing(self):
        missing = [
            label
            for key, label in OPERATOR_FIELDS
            if key not in OPTIONAL_OPERATOR_FIELDS and not str(self.operator_values[key]).strip()
        ]
        if not str(self.velocities["gripper"]).strip() and self.gripper_wire_code is None:
            missing.append("gripper 速度")
        missing += [
            f"{name} MIT {field.upper()}"
            for name in JOINT_NAMES[:-1]
            for field in ("kp", "kd")
            if not str(getattr(self, field)[name]).strip()
        ]
        missing += [
            "gripper 本拍相对限制" for name in ("gripper",) if not str(self.relative_limits[name]).strip()
        ]
        if not str(self.target).strip():
            missing.append("Teacher 目标力来源")
        if not self.supported_arm:
            missing.append("机械臂支撑确认")
        return missing


class OperatorArmPage:
    """Visible requested limits and separate owner feedback for manual user operation."""

    def __init__(self, parent, *, apply_target, is_busy):
        self.editor = OperatorProfileEditor()
        self.is_busy, self.apply_target = is_busy, apply_target
        self.reset_review = lambda: None
        self.on_saved = lambda path: None
        self.source_changed = lambda: None
        self.dirty = self.loading = False
        self.use_current_form = False
        self.configuration_widgets = []
        self.profile_frame = ttk.LabelFrame(parent, text="来源配置与离线预检", padding=6)
        self.profile_frame.pack(fill="x")
        self.mode = tk.StringVar(value="正常全臂遥操")
        mode = self.mode_widget = ttk.Combobox(
            parent, textvariable=self.mode, values=("正常全臂遥操", "micro 诊断（可选）"), state="readonly"
        )
        mode.pack(fill="x", pady=5)
        mode.bind("<<ComboboxSelected>>", lambda _: self.mark_dirty())
        self.configuration_widgets.append(mode)
        ttk.Label(
            parent,
            text="六臂直接使用官方 MIT 默认值；夹爪保留 FORCE_POS＋I-P。\n官方未设置的额外相对限制/容差显示为“默认关闭”。",
            wraplength=440,
        ).pack(anchor="w")
        self.source_label = tk.StringVar(value="未选择已保存来源")
        ttk.Label(parent, textvariable=self.source_label, wraplength=440).pack(anchor="w", pady=5)
        self.wire_label = tk.StringVar(value="夹爪速度：未选择来源")
        ttk.Label(parent, textvariable=self.wire_label, wraplength=440).pack(anchor="w", pady=3)
        self.operator_values, self.operator_entries = {}, {}
        form = ttk.LabelFrame(parent, text="运行范围与启动保护（停止并断开后修改）", padding=5)
        form.pack(fill="x", pady=5, before=self.profile_frame)
        for row_index, (key, label) in enumerate(OPERATOR_FIELDS):
            ttk.Label(form, text=label).grid(row=row_index, column=0, sticky="w")
            variable = self.operator_values[key] = tk.StringVar(value="")
            entry = ttk.Entry(form, textvariable=variable, width=16)
            self.operator_entries[key] = entry
            entry.grid(row=row_index, column=1, padx=4, pady=2)
            self.configuration_widgets.append(entry)
            variable.trace_add("write", lambda *_: self.mark_dirty())
        self.protections = tk.StringVar(value="校准、raw 超力、反馈新鲜度和 Stop / Esc 均不可禁用。")
        ttk.Label(parent, textvariable=self.protections, wraplength=440).pack(anchor="w", pady=4)
        self.target = tk.StringVar(value="")
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text="请求目标力 (N)").pack(side="left")
        self.target_entry = ttk.Entry(row, textvariable=self.target, width=10)
        self.target_entry.pack(side="left", padx=5)
        self.target_button = ttk.Button(
            row, text="应用目标力（operator）", command=apply_target, state="disabled"
        )
        self.target_button.pack(side="left")
        self.target.trace_add("write", lambda *_: self.mark_dirty(target_only=True))
        self.effective_target = tk.StringVar(value="owner 生效目标：尚无运行回执；正常模式不固定 1 N")
        ttk.Label(parent, textvariable=self.effective_target, wraplength=440).pack(anchor="w", pady=4)
        table = ttk.LabelFrame(parent, text="七轴本次请求值（未选择留空）", padding=5)
        table.pack(fill="x", pady=5)
        self.velocities, self.relative_limits, self.kp, self.kd = {}, {}, {}, {}
        self.relative_entries = {}
        for column, label in enumerate(("关节", "MIT Kp", "MIT Kd", "目标偏差 °")):
            ttk.Label(table, text=label).grid(row=0, column=column, sticky="w")
        for index, name in enumerate(JOINT_NAMES, start=1):
            ttk.Label(table, text=name).grid(row=index, column=0, sticky="w")
            self.velocities[name] = tk.StringVar(value="")
            if name == "gripper":
                ttk.Label(table, text="速度 °/s").grid(row=index, column=1, sticky="w")
                fields_by_column = ((2, self.velocities), (3, self.relative_limits))
            else:
                fields_by_column = ((1, self.kp), (2, self.kd), (3, self.relative_limits))
            for column, fields in fields_by_column:
                variable = fields[name] = tk.StringVar(value="")
                entry = ttk.Entry(table, textvariable=variable, width=8)
                entry.grid(row=index, column=column, padx=3, pady=2)
                if column == 3:
                    self.relative_entries[name] = entry
                elif name == "gripper":
                    self.gripper_velocity_entry = entry
                self.configuration_widgets.append(entry)
                variable.trace_add("write", lambda *_: self.mark_dirty())
        ttk.Label(
            parent,
            text="MIT 初始速度参考和前馈力矩均为 0，可在下表修改；速度参考不是限速。SDK 编码范围 Kp≤500、Kd≤5（官方前三轴请求 Kd=12，编码为5）。请求值与编码值在启动摘要显示。",
            wraplength=440,
        ).pack(anchor="w", pady=4)
        self.mit_velocity_deg_s, self.mit_torque_Nm = {}, {}
        references = ttk.LabelFrame(parent, text="六臂 MIT 参考（默认 0，停止后修改）", padding=5)
        references.pack(fill="x", pady=5)
        for column, label in enumerate(("关节", "速度参考 °/s", "前馈力矩 N·m")):
            ttk.Label(references, text=label).grid(row=0, column=column, sticky="w")
        for index, name in enumerate(JOINT_NAMES[:-1], 1):
            ttk.Label(references, text=name).grid(row=index, column=0, sticky="w")
            for column, fields in ((1, self.mit_velocity_deg_s), (2, self.mit_torque_Nm)):
                variable = fields[name] = tk.StringVar(value="0")
                entry = ttk.Entry(references, textvariable=variable, width=10)
                entry.grid(row=index, column=column, padx=3, pady=2)
                self.configuration_widgets.append(entry)
                variable.trace_add("write", lambda *_: self.mark_dirty())
        self.tracking_status = tk.StringVar(value="尚无七关节运行反馈")
        ttk.Label(parent, textvariable=self.tracking_status, wraplength=440).pack(anchor="w", pady=4)
        self.minimum_deg, self.maximum_deg, self.calibration_bounds = {}, {}, {}
        ranges = ttk.LabelFrame(parent, text="六臂工作目标范围（须在已加载校准内）", padding=5)
        ranges.pack(fill="x", pady=5)
        for column, label in enumerate(("关节 / 校准硬范围", "工作下限 °", "工作上限 °")):
            ttk.Label(ranges, text=label).grid(row=0, column=column, sticky="w")
        for index, name in enumerate(JOINT_NAMES[:-1], 1):
            line = ttk.Frame(ranges)
            line.grid(row=index, column=0, sticky="w")
            ttk.Label(line, text=name).pack(anchor="w")
            hard_range = self.calibration_bounds[name] = tk.StringVar(value="校准未加载")
            ttk.Label(line, textvariable=hard_range).pack(anchor="w")
            for column, fields in ((1, self.minimum_deg), (2, self.maximum_deg)):
                variable = fields[name] = tk.StringVar(value="")
                entry = ttk.Entry(ranges, textvariable=variable, width=10)
                entry.grid(row=index, column=column, padx=3, pady=3)
                self.configuration_widgets.append(entry)
                variable.trace_add("write", lambda *_: self.mark_dirty())
        ttk.Label(
            parent,
            text="六臂参考范围与上述工作范围取交集；夹爪使用“夹爪控制”页的工作范围和校准硬范围。可选对齐、预载、接管容差及命令间隔留空关闭。顶部“保存参数”同时保存全臂修改，下次启动生效。",
            wraplength=440,
        ).pack(anchor="w", pady=5)
        self.supported = tk.BooleanVar(value=False)
        support = self.support_widget = ttk.Checkbutton(
            parent,
            text="已确认本次机械臂支撑与全七轴失能停止策略",
            variable=self.supported,
            command=self.mark_dirty,
        )
        support.pack(anchor="w", pady=4)
        self.configuration_widgets.append(support)
        self.duration_enabled, self.travel_enabled = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.duration, self.travel = tk.StringVar(value=""), tk.StringVar(value="")
        self.limit_entries, self.limit_checks = {}, {}
        for key, title, toggle, value in (
            ("duration", "启用可选总时长 (s)", self.duration_enabled, self.duration),
            ("travel", "启用夹爪累计行程 (°，非全臂计程)", self.travel_enabled, self.travel),
        ):
            line = ttk.Frame(parent)
            line.pack(fill="x", pady=2)
            check = ttk.Checkbutton(line, text=title, variable=toggle, command=self.mark_dirty)
            check.pack(side="left")
            entry = ttk.Entry(line, textvariable=value, width=10)
            entry.pack(side="right")
            self.limit_entries[key], self.limit_checks[key] = entry, check
            self.configuration_widgets.extend((check, entry))
            value.trace_add("write", lambda *_: self.mark_dirty())
        save = ttk.Button(parent, text="保存 operator 草稿…（停止后预检）", command=self.save_draft)
        save.pack(fill="x", pady=6)
        self.configuration_widgets.append(save)
        self.summary = tk.StringVar(
            value="只读：尚未选择必要参数。校准、原始超力、反馈过期和急停均不可禁用。"
        )
        ttk.Label(parent, textvariable=self.summary, wraplength=440).pack(anchor="w", pady=5)

    def mark_dirty(self, target_only=False):
        if self.loading:
            return
        self.dirty = True
        self.reset_review()
        self.summary.set(
            "默认参数已编辑；下次主从启动将直接校验当前表单，无需另存草稿。"
            if self.use_current_form and not self.is_busy()
            else "目标力已编辑，可显式申请实时应用；下次会话使用当前参数。"
            if target_only and self.is_busy()
            else "配置已编辑；停止后保存草稿并重新预检。不会自动发送/使能。"
        )

    def load_source(self, path):
        if self.editor.path == Path(path).resolve():
            current = profile_fingerprint(path)
            if current == self.editor.source_pin:
                return
            if self.dirty:
                raise ValueError("来源文件已变且存在未保存编辑；先保存编辑或重新选择来源再预检")
        self.use_current_form = False
        self.loading = True
        try:
            normal = self.editor.load(path)
            self.target.set(self.editor.target)
            self.supported.set(self.editor.supported_arm)
            for name in JOINT_NAMES:
                self.velocities[name].set(self.editor.velocities[name])
                self.relative_limits[name].set(self.editor.relative_limits[name])
            for name in JOINT_NAMES[:-1]:
                self.kp[name].set(self.editor.kp[name])
                self.kd[name].set(self.editor.kd[name])
                self.minimum_deg[name].set(self.editor.minimum_deg[name])
                self.maximum_deg[name].set(self.editor.maximum_deg[name])
                self.mit_velocity_deg_s[name].set(self.editor.mit_velocity_deg_s[name])
                self.mit_torque_Nm[name].set(self.editor.mit_torque_Nm[name])
            for key, value in self.editor.operator_values.items():
                self.operator_values[key].set(value)
            self.duration_enabled.set(self.editor.duration_enabled)
            self.travel_enabled.set(self.editor.travel_enabled)
            self.duration.set(self.editor.duration)
            self.travel.set(self.editor.travel)
            code = self.editor.gripper_wire_code
            self.wire_label.set(
                f"夹爪可沿用来源速度码 {code}（单位不同，非 °/s）；主动填写 gripper °/s 才切换编码。"
                if code is not None
                else "夹爪无有效来源速度码，须明确选择速度。"
            )
            self.source_label.set(
                f"Teacher/Safety/X518 来源：{self.editor.path}\n目标力来自 teacher.force_target_N={self.editor.target or '未填写'}；"
                + ("已保存 operator 数值" if normal else "非 operator 来源，运动必要值未继承")
            )
            source = self.editor.source
            control = source.get("b601", {}).get("control") or {}
            try:
                calibration = B601ArmCalibration.load(
                    control.get("calibration_file"), dict(zip(JOINT_NAMES, range(1, 8), strict=True))
                )
                for name, _, lower, upper in calibration.joints:
                    if name in self.calibration_bounds:
                        self.calibration_bounds[name].set(f"硬范围 [{lower:g}, {upper:g}]°")
            except (OSError, ValueError, TypeError):
                for variable in self.calibration_bounds.values():
                    variable.set("需加载有效校准")
            safety, runtime = source.get("safety") or {}, source.get("runtime") or {}
            ranges = "; ".join(
                f"{p['name']} [{p.get('minimum_deg')}, {p.get('maximum_deg')}]°"
                for p in control.get("profiles", [])
            )
            self.protections.set(
                f"来源校准：{control.get('calibration_file')}；绝对 profile 范围：{ranges}\n"
                f"来源六臂参考范围：q0 ±{(source.get('b601', {}).get('operator_teleoperation') or {}).get('reference_envelope_deg', 100)}°，可在上方“运行范围与启动保护”修改。\n"
                f"来源启动/写/全七轴清理预算(s)：{control.get('startup_budget_s')}/{control.get('send_budget_s')}/{control.get('fault_budget_s')}；"
                f"runtime 读取预算/节拍截止(ms)：{runtime.get('feedback_budget_ms')}/{runtime.get('tick_deadline_ms')}；"
                f"主臂读取预算(s)：{source.get('leader_read_budget_s')}\n"
                f"原始硬力左/右(N)：{safety.get('hard_force_limit_left_N')}/{safety.get('hard_force_limit_right_N')}；"
                f"force/joint 最大年龄(ms)：{safety.get('force_max_age_ms')}/{safety.get('joint_feedback_max_age_ms')}；"
                f"底层反馈最大年龄(s)：{control.get('feedback_max_age_s')}。上述保护不可禁用。\n"
                f"显式控制权切换容差：{runtime.get('handover_tolerance_deg')}（仅返回/接管模式；非 hybrid 启动容差）。"
            )
            if self.editor.converted_to_mit:
                self.summary.set("六臂已按官方 MIT 参数准备；请保存草稿后重新离线预检。夹爪参数保留。")
            else:
                self.summary.set(
                    "只读缺项：" + "；".join(self.editor.missing())
                    if self.editor.missing()
                    else "已载入显式数值，尚需离线预检和 owner 生效回执。"
                )
            self.dirty = self.editor.converted_to_mit
        finally:
            self.loading = False

    def require_saved(self):
        if self.mode.get() == "正常全臂遥操" and self.dirty:
            raise ValueError("全臂请求值已编辑；请停止后保存 operator 草稿再预检")
        hardware = (self.editor.source or {}).get("b601") or {}
        if self.mode.get() == "正常全臂遥操" and hardware.get("operator_teleoperation") is None:
            raise ValueError("当前是诊断来源；正常模式须保存 operator 草稿，或明确选择 micro 诊断")
        if self.mode.get() != "正常全臂遥操" and hardware.get("operator_teleoperation") is not None:
            raise ValueError("当前是正常 operator 来源；请选择正常全臂遥操")

    def sync_editor(self):
        self.editor.target, self.editor.supported_arm = self.target.get(), self.supported.get()
        self.editor.velocities = {name: value.get() for name, value in self.velocities.items()}
        self.editor.kp = {name: value.get() for name, value in self.kp.items()}
        self.editor.kd = {name: value.get() for name, value in self.kd.items()}
        self.editor.minimum_deg = {name: value.get() for name, value in self.minimum_deg.items()}
        self.editor.maximum_deg = {name: value.get() for name, value in self.maximum_deg.items()}
        self.editor.mit_velocity_deg_s = {
            name: value.get() for name, value in self.mit_velocity_deg_s.items()
        }
        self.editor.mit_torque_Nm = {name: value.get() for name, value in self.mit_torque_Nm.items()}
        self.editor.relative_limits = {name: value.get() for name, value in self.relative_limits.items()}
        self.editor.operator_values = {key: value.get() for key, value in self.operator_values.items()}
        self.editor.duration_enabled, self.editor.travel_enabled = (
            self.duration_enabled.get(),
            self.travel_enabled.get(),
        )
        self.editor.duration, self.editor.travel = self.duration.get(), self.travel.get()

    def settings_values(self):
        """Save all arm edits with the ordinary parameter file, without support confirmation."""
        if self.editor.source is None:
            return {}

        def draft(variable):
            value = variable.get()
            return "" if value.strip() == DEFAULT_DISABLED else value

        return checked_arm_settings(
            {
                "operator": {key: draft(self.operator_values[key]) for key in ARM_OPERATOR_FIELDS},
                "joints": {
                    name: {
                        key: draft(
                            (self.relative_limits if key == "max_relative_deg" else getattr(self, key))[name]
                        )
                        for key in ARM_PROFILE_FIELDS
                    }
                    for name in JOINT_NAMES[:-1]
                },
            }
        )

    def restore_settings_values(self, settings):
        """Restore saved draft text after loading its numeric source, without confirmation."""
        settings = checked_arm_settings(settings)
        if not settings:
            return
        self.loading = True
        try:
            for key, value in settings["operator"].items():
                self.operator_values[key].set(value)
            for name, profile in settings["joints"].items():
                for key, value in profile.items():
                    fields = self.relative_limits if key == "max_relative_deg" else getattr(self, key)
                    fields[name].set(value)
            self.sync_editor()
            self.dirty = False
        finally:
            self.loading = False

    def save_draft(self):
        if self.is_busy():
            self.summary.set("请停止并断开后保存全臂配置")
            return
        try:
            self.sync_editor()
            document = self.editor.document()
            path = filedialog.asksaveasfilename(
                title="保存正常 operator 配置草稿", defaultextension=".json", filetypes=[("JSON", "*.json")]
            )
            if not path:
                return
            Path(path).write_text(
                json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            self.dirty = False
            self.editor.path = None
            self.load_source(path)
            self.on_saved(path)
            self.summary.set("草稿已保存；缺项保持 null，仅只读，需离线预检后由用户本人启动。")
        except (OSError, ValueError, TypeError) as exc:
            self.summary.set(str(exc))
