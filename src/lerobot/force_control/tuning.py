"""Shared B601 I-P settings and disconnected profile construction.

The panel and Teacher entry points use this module without
importing Tk or opening a transport. Saved drafts never authorize motion.
"""

import json
import math
import os
import tempfile
from pathlib import Path

from lerobot.robots.rebot_b601_follower.control_backend import B601ControlConfig
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile, force_pos_wire_codes
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig

from .b601_factory import B601HardwareConfig
from .config import (
    SafetyConfig,
    TeacherConfig,
    canonicalize_teacher_fields,
    finite_number,
    positive,
    validate_profile,
)
from .operator_settings import checked_arm_settings, migrate_arm_settings
from .runtime import DEFAULT_FEEDBACK_BUDGET_MS, DEFAULT_TICK_DEADLINE_MS, RuntimeConfig

GRAVITY_N_PER_KG = 9.80665
# section, key, label with unit, visible default, kind, live, optional group
FIELDS = (
    ("control", "target_N", "目标平均力 (N)", "4.903325", float, True, None),
    ("control", "kp_deg_per_N", "Kp (°/N) · 必填，可为 0", "", float, True, None),
    ("control", "ki_deg_per_N_s", "Ki (°/(N·s)) · 必填，可为 0", "", float, True, None),
    ("control", "integral_deadband_N", "积分误差死区 (N)，0 关闭", "0", float, True, None),
    (
        "control",
        "gripper_follow_speed_deg_s",
        "夹爪跟随 / 接近速度上限 (°/s) · 使用时必填",
        "",
        float,
        True,
        None,
    ),
    (
        "control",
        "contact_force_threshold_N",
        "接触平均力阈值 (N) · 使用时必填",
        "",
        float,
        True,
        None,
    ),
    ("control", "position_target_deg", "指定绝对位置 (°，校准坐标)", "", float, True, None),
    ("control", "position_speed_deg_s", "定位速度 (°/s) · 点击定位时生效", "", float, True, None),
    ("control", "close_sign", "闭合角度方向 (+1 / -1) · 必填", "", int, False, None),
    ("control", "max_close_speed_deg_s", "最大闭合位置变化率 (°/s)", "0.2", float, True, None),
    ("control", "max_open_speed_deg_s", "最大张开位置变化率 (°/s)", "0.2", float, True, None),
    ("control", "acceleration_limit_deg_s2", "位置变化率加速度 (°/s²)", "0.5", float, True, "acceleration"),
    ("control", "manual_step_deg", "手动小步 (°)，正闭合 / 负张开", "", float, True, None),
    ("control", "release_speed_deg_s", "退出放松速度 (°/s)", "0.2", float, False, None),
    ("control", "release_travel_deg", "退出放松行程 (°)", "0.5", float, False, None),
    ("control", "max_release_time_s", "退出放松最长时间 (s)", "3", float, False, None),
    ("limits", "target_min_N", "允许目标最小平均力 (N)", "0", float, False, None),
    ("limits", "target_max_N", "允许目标最大平均力 (N) · 必填", "", float, False, None),
    ("limits", "hard_force_left_kg", "左侧原始力停止阈值 (kg)", "1.5", float, False, None),
    ("limits", "hard_force_right_kg", "右侧原始力停止阈值 (kg)", "1.5", float, False, None),
    ("limits", "max_relative_target_deg", "目标领先反馈上限 (°) · 必填", "", float, False, None),
    ("limits", "max_command_tracking_error_deg", "跟随误差停止阈值 (°) · 必填", "", float, False, None),
    ("limits", "stall_position_tolerance_deg", "停滞 / 小步到位容差 (°) · 必填", "", float, False, None),
    ("limits", "stall_timeout_s", "持续停滞冻结积分时间 (s) · 必填", "", float, False, None),
    ("limits", "soft_q_min_deg", "工作目标下限 (°)，到达后保持", "", float, False, "soft_limits"),
    ("limits", "soft_q_max_deg", "工作目标上限 (°)，到达后保持", "", float, False, "soft_limits"),
    ("limits", "max_total_travel_deg", "累计绝对行程上限 (°)", "", float, False, "travel"),
    ("limits", "max_run_time_s", "运行时长上限 (s)", "", float, False, "duration"),
    ("limits", "force_max_age_ms", "力反馈最大年龄 (ms)", "50", float, False, None),
    ("limits", "joint_feedback_max_age_ms", "位置反馈最大年龄 (ms)", "50", float, False, None),
    ("limits", "decision_max_age_ms", "发送前决策最大年龄 (ms)", "20", float, False, None),
    ("limits", "max_dt_s", "有效控制周期间隔上限 (s)", "0.06", float, False, None),
    ("limits", "start_feedback_max_age_ms", "只读启动参考最大年龄 (ms)", "150", float, False, None),
    ("device", "port", "B601 串口", "COM22", str, False, None),
    ("device", "baudrate", "B601 波特率", "921600", int, False, None),
    ("device", "leader_port", "102 主臂串口 · Teacher 模式必填", "", str, False, None),
    (
        "device",
        "leader_readonly_transport",
        "102 只读传输 (protocol_v1 / sdk)",
        "protocol_v1",
        str,
        False,
        None,
    ),
    ("device", "leader_baudrate", "102 波特率", "1000000", int, False, None),
    ("device", "leader_frequency_hz", "102 采样频率 (Hz)", "50", float, False, None),
    ("device", "leader_max_age_s", "102 指令最大年龄 (s)", "0.15", float, False, None),
    ("device", "leader_read_budget_s", "102 读取预算 (s)", "0.1", float, False, None),
    ("device", "motor_velocity_code", "FORCE_POS 速度码 (整数 1–10000)", "1", int, False, None),
    ("device", "drive_percent", "FORCE_POS 电流 / 力矩比例 (%)", "2", float, False, None),
    ("device", "watchdog_ticks", "看门狗 TIMEOUT ticks (50 μs/tick)", "2000", int, False, None),
    ("device", "parameter_write_settle_s", "参数写入后等待读回 (s)", "0.01", float, False, None),
    ("device", "control_hz", "控制目标频率 (Hz)", "100", float, False, None),
    ("device", "scheduler_sleep_slice_ms", "调度最长分段休眠 (ms)", "5", float, False, None),
    (
        "device",
        "feedback_budget_ms",
        "反馈查询预算，自本周期开始 (ms)",
        f"{DEFAULT_FEEDBACK_BUDGET_MS:g}",
        float,
        False,
        None,
    ),
    (
        "device",
        "tick_deadline_ms",
        "控制周期工作截止预算 (ms)",
        f"{DEFAULT_TICK_DEADLINE_MS:g}",
        float,
        False,
        None,
    ),
    ("device", "command_ttl_ms", "操作请求有效期 (ms)", "200", float, False, None),
    ("device", "startup_budget_s", "电机连接 / 准备预算 (s)", "1", float, False, None),
    ("device", "send_budget_s", "单次命令发送预算 (s)", "0.02", float, False, None),
    ("device", "fault_budget_s", "故障失能发送预算 (s)", "0.02", float, False, None),
    ("device", "startup_timeout_s", "首个反馈等待 / owner 启动 (s)", "2", float, False, None),
    ("device", "shutdown_timeout_s", "owner 退出等待 (s)", "2", float, False, None),
    ("device", "io_event_capacity", "传输 I/O 证据容量 (条)", "4096", int, False, None),
    ("device", "prequery_byte_limit", "查询前 / 故障未失能接收上限 (bytes)", "256", int, False, None),
    ("device", "queue_capacity", "控制日志队列容量 (条)", "2048", int, False, None),
    ("device", "raw_queue_capacity", "原始力日志队列容量 (条)", "4096", int, False, None),
    ("device", "trace_join_timeout_s", "日志退出等待 (s)", "2", float, False, None),
    ("device", "sensor_host", "X518 IP", "192.168.1.100", str, False, None),
    ("device", "sensor_port", "X518 TCP 端口", "502", int, False, None),
    ("device", "sensor_unit_id", "X518 Modbus unit ID", "1", int, False, None),
    ("device", "left_channel", "左力 X518 原生通道 (1 / 2)", "2", int, False, None),
    ("device", "right_channel", "右力 X518 原生通道 (1 / 2)", "1", int, False, None),
    ("device", "sensor_request_timeout_s", "X518 请求超时 (s)", "0.05", float, False, None),
    ("device", "sensor_reconnect_delay_s", "X518 重连等待 (s)", "0.05", float, False, None),
    ("device", "sensor_connect_retries", "X518 连接重试次数", "5", int, False, None),
    ("device", "sensor_connect_backoff_s", "X518 连接重试等待 (s)", "0.2", float, False, None),
    ("device", "sensor_startup_timeout_s", "X518 采集启动等待 (s)", "5", float, False, None),
    ("device", "sensor_history_duration_s", "X518 在线历史缓存 (s)", "2", float, False, None),
    ("device", "acquisition_queue_capacity", "X518 进程采集队列容量 (条)", "1024", int, False, None),
    ("device", "monitor_interval_ms", "只读监测查询间隔 (ms)", "50", float, False, None),
    ("device", "monitor_feedback_budget_ms", "只读监测查询预算 (ms)", "50", float, False, None),
    ("device", "display_hz", "界面刷新频率 (Hz)", "20", float, False, None),
    ("device", "plot_seconds", "波形显示窗口 (s)", "20", float, False, None),
    ("device", "ipc_poll_ms", "进程结果 / 遥测查询间隔 (ms)", "50", float, False, None),
    ("device", "object_id", "物体记录标识", "unspecified", str, False, None),
    ("device", "contact_condition_id", "接触条件记录标识", "unspecified", str, False, None),
)
OPTIONAL_GROUPS = {
    "acceleration": ("启用位置变化率加速度限制", True),
    "soft_limits": ("启用工作目标范围（校准硬范围内）", False),
    "travel": ("启用累计行程限制", False),
    "duration": ("启用运行时长限制", False),
}
LIVE_FIELDS = frozenset(key for _, key, _, _, _, live, _ in FIELDS if live)
GRIPPER_DEVICE_FIELDS = frozenset(
    {
        "motor_velocity_code",
        "drive_percent",
        "watchdog_ticks",
        "control_hz",
        "scheduler_sleep_slice_ms",
        "feedback_budget_ms",
        "tick_deadline_ms",
        "command_ttl_ms",
        "startup_budget_s",
        "send_budget_s",
        "fault_budget_s",
        "startup_timeout_s",
        "shutdown_timeout_s",
        "parameter_write_settle_s",
    }
)
FIELD_PAGES = {
    key: "gripper" if section != "device" or key in GRIPPER_DEVICE_FIELDS else "other"
    for section, key, *_ in FIELDS
}
SETTINGS_FORMAT = "b601_ip_tuner_settings"
SETTINGS_SCHEMA_VERSION = 8
LEADER_FIELDS = frozenset(key for _, key, *_ in FIELDS if key.startswith("leader_"))
MOTION_FIELDS = {
    "gripper_follow_speed_deg_s",
    "contact_force_threshold_N",
    "position_target_deg",
    "position_speed_deg_s",
}
LEGACY_TARGET_FIELDS = {
    "target_kg": "target_N",
    "target_min_kg": "target_min_N",
    "target_max_kg": "target_max_N",
}


def settings_document(values, enabled, calibration_path, *, arm_settings=None):
    """Save editable drafts, never motor references, activation or cleanup state."""
    if (
        not isinstance(values, dict)
        or set(values) != set(default_values())
        or any(type(value) is not str for value in values.values())
    ):
        raise ValueError("参数字段或单位版本不匹配；旧参数不能自动复用")
    if (
        not isinstance(enabled, dict)
        or set(enabled) != set(OPTIONAL_GROUPS)
        or any(type(value) is not bool for value in enabled.values())
    ):
        raise ValueError("参数开关字段或类型不匹配")
    return {
        "format": SETTINGS_FORMAT,
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "controller_type": "ip",
        "calibration_file": str(Path(calibration_path).resolve()),
        "values": dict(values),
        "enabled": dict(enabled),
        "arm_settings": checked_arm_settings(arm_settings),
    }


def save_settings(path, values, enabled, calibration_path, *, arm_settings=None):
    """Replace one requested JSON atomically; preserve the previous file on failure."""
    document = settings_document(values, enabled, calibration_path, arm_settings=arm_settings)
    path = Path(path).resolve()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_settings(path, *, on_unit_conversion=None, on_arm_settings=None):
    """Strictly read a parameter file; calibration and runtime state are not trusted from JSON."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    keys = {"format", "schema_version", "controller_type", "calibration_file", "values", "enabled"}
    if (
        not isinstance(document, dict)
        or set(document) not in (keys, keys | {"arm_settings"})
        or document["format"] != SETTINGS_FORMAT
        or type(document["schema_version"]) is not int
        or document["schema_version"] not in (1, 2, 3, 4, 5, 6, 7, SETTINGS_SCHEMA_VERSION)
        or document["controller_type"] != "ip"
        or not isinstance(document["calibration_file"], str)
        or not document["calibration_file"].strip()
        or not isinstance(document["values"], dict)
        or not isinstance(document["enabled"], dict)
    ):
        raise ValueError("不是当前 I-P 参数文件，不能导入运行日志或旧控制配置")
    values = dict(document["values"])
    if document["schema_version"] < 8:
        values.setdefault("scheduler_sleep_slice_ms", default_values()["scheduler_sleep_slice_ms"])
    if document["schema_version"] < 5:
        values = canonicalize_teacher_fields(values)
    if document["schema_version"] < 6:
        # Old profiles never select a leader port or authorize a running mode.
        defaults = default_values()
        for key in LEADER_FIELDS:
            values.setdefault(key, defaults[key])
    if document["schema_version"] == 1:
        legacy_keys = (set(default_values()) - MOTION_FIELDS - set(LEGACY_TARGET_FIELDS.values())) | set(
            LEGACY_TARGET_FIELDS
        )
        if set(values) != legacy_keys or any(type(value) is not str for value in values.values()):
            raise ValueError("旧 I-P 参数必须使用完整 kg 目标字段，不能混用 N 字段")
        for old_key, new_key in LEGACY_TARGET_FIELDS.items():
            text = values.pop(old_key).strip()
            values[new_key] = (
                format(finite_number(old_key, float(text) * GRAVITY_N_PER_KG), ".15g") if text else ""
            )
    if document["schema_version"] < 3:
        if set(values) != set(default_values()) - MOTION_FIELDS:
            raise ValueError("旧参数字段不完整或包含未知字段")
        values.update(dict.fromkeys(MOTION_FIELDS, ""))
    elif document["schema_version"] == 3:
        if set(values) != set(default_values()) - {"position_speed_deg_s"}:
            raise ValueError("旧参数字段不完整或包含未知字段")
        values["position_speed_deg_s"] = ""
    calibration_path = Path(document["calibration_file"])
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    arm_settings = document.get("arm_settings")
    if document["schema_version"] < 8:
        arm_settings = migrate_arm_settings(arm_settings)
    checked = settings_document(values, document["enabled"], calibration_path, arm_settings=arm_settings)
    if document["schema_version"] == 1 and on_unit_conversion is not None:
        on_unit_conversion("旧文件目标及范围已按 9.80665 N/kg 换算为 N，原文件未改写")
    if on_arm_settings is not None:
        on_arm_settings(checked["arm_settings"])
    return checked["values"], checked["enabled"], Path(checked["calibration_file"])


def default_values():
    """Return editable defaults without supplying missing physical gains or limits."""
    return {key: default for _, key, _, default, *_ in FIELDS}


def parse_values(values, enabled):
    """Disabled optional fields remain explicitly None; never silently clamp."""
    parsed = {}
    for _, key, label, _, kind, _, group in FIELDS:
        if group and not enabled[group]:
            parsed[key] = None
            continue
        if key in ("manual_step_deg", "position_target_deg", "position_speed_deg_s"):
            continue
        if key in MOTION_FIELDS and not values[key].strip():
            parsed[key] = None
            continue
        try:
            parsed[key] = kind(values[key].strip())
            if kind is not str:
                finite_number(label, parsed[key])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"请填写 {label}") from exc
    for key in (
        "start_feedback_max_age_ms",
        "monitor_interval_ms",
        "monitor_feedback_budget_ms",
        "display_hz",
        "plot_seconds",
        "ipc_poll_ms",
    ):
        positive(key, parsed[key])
    if parsed["close_sign"] not in (-1, 1):
        raise ValueError("闭合方向必须明确填写 +1 或 -1")
    return parsed


def sensor_config(v):
    """Keep X518 semantic forces raw in N; no second calibration."""
    return X518SensorConfig(
        host=v["sensor_host"],
        port=v["sensor_port"],
        unit_id=v["sensor_unit_id"],
        request_timeout_s=v["sensor_request_timeout_s"],
        reconnect_delay_s=v["sensor_reconnect_delay_s"],
        connect_retries=v["sensor_connect_retries"],
        connect_backoff_s=v["sensor_connect_backoff_s"],
        startup_timeout_s=v["sensor_startup_timeout_s"],
        history_duration_s=v["sensor_history_duration_s"],
        acquisition_queue_capacity=v["acquisition_queue_capacity"],
        recorder_queue_capacity=v["raw_queue_capacity"],
        acquisition_mode="process",
        sample_rate_hz=None,
        expected_unit="kg",
        max_age_ms=v["force_max_age_ms"],
        required=True,
        frame_features=["left.normal_force", "right.normal_force"],
        channels={
            "left.normal_force": X518ChannelConfig(channel=v["left_channel"]),
            "right.normal_force": X518ChannelConfig(channel=v["right_channel"]),
        },
    )


def build_config(values, enabled, calibration, output, *, initial_mode="force"):
    """Map every visible control into the common typed runtime profile."""
    calibration.require_unchanged()
    v = parse_values(values, enabled)
    teacher = TeacherConfig(
        controller_type="ip",
        force_target_N=v["target_N"],
        target_min_N=v["target_min_N"],
        target_max_N=v["target_max_N"],
        kp_deg_per_N=v["kp_deg_per_N"],
        ki_deg_per_N_s=v["ki_deg_per_N_s"],
        integral_deadband_N=v["integral_deadband_N"],
        gripper_follow_speed_deg_s=v["gripper_follow_speed_deg_s"],
        contact_force_threshold_N=v["contact_force_threshold_N"],
        release_speed_deg_s=v["release_speed_deg_s"],
        release_travel_deg=v["release_travel_deg"],
        max_release_time_s=v["max_release_time_s"],
        profile_id="b601_force_pos_ip",
    )
    safety_fields = (
        "max_relative_target_deg",
        "max_command_tracking_error_deg",
        "max_close_speed_deg_s",
        "max_open_speed_deg_s",
        "acceleration_limit_deg_s2",
        "force_max_age_ms",
        "joint_feedback_max_age_ms",
        "decision_max_age_ms",
        "max_dt_s",
        "stall_position_tolerance_deg",
        "stall_timeout_s",
        "soft_q_min_deg",
        "soft_q_max_deg",
        "max_total_travel_deg",
        "max_run_time_s",
    )
    safety = SafetyConfig(
        q_min_deg=calibration.minimum_deg,
        q_max_deg=calibration.maximum_deg,
        hard_force_limit_left_N=v["hard_force_left_kg"] * GRAVITY_N_PER_KG,
        hard_force_limit_right_N=v["hard_force_right_kg"] * GRAVITY_N_PER_KG,
        fault_action="stop",
        supported_object_bench=True,
        **{key: v[key] for key in safety_fields},
    )
    validate_profile(teacher, safety, v["close_sign"])
    drive = v["drive_percent"] / 100
    force_pos_wire_codes(None, drive, velocity_wire_code=v["motor_velocity_code"])
    if not math.isclose(v["drive_percent"] * 100, round(v["drive_percent"] * 100), abs_tol=1e-9):
        raise ValueError("驱动力比例必须按 0.01% 步进填写")
    profile = B601JointCommandProfile(
        "gripper",
        "force_pos",
        safety.soft_q_min_deg if safety.soft_q_min_deg is not None else calibration.minimum_deg,
        safety.soft_q_max_deg if safety.soft_q_max_deg is not None else calibration.maximum_deg,
        v["max_relative_target_deg"],
        velocity_wire_code=v["motor_velocity_code"],
        torque_ratio=drive,
    )
    control = B601ControlConfig(
        profiles=(profile,),
        startup_budget_s=v["startup_budget_s"],
        send_budget_s=v["send_budget_s"],
        fault_budget_s=v["fault_budget_s"],
        feedback_max_age_s=v["joint_feedback_max_age_ms"] / 1000,
        fault_action="disable_gripper",
        io_event_capacity=v["io_event_capacity"],
        prequery_byte_limit=v["prequery_byte_limit"],
        calibration_file=calibration.path,
        watchdog_ticks=v["watchdog_ticks"],
        parameter_write_settle_s=v["parameter_write_settle_s"],
    )
    config = ForceTeacherRunConfig(
        backend="b601",
        dry_run=False,
        observe_only=False,
        allow_motion=True,
        explicit_arm=True,
        initial_mode=initial_mode,
        leader_backend="rebot_102" if initial_mode == "hybrid" else None,
        leader_port=(v["leader_port"] or None) if initial_mode == "hybrid" else None,
        leader_readonly_transport=v["leader_readonly_transport"],
        leader_baudrate=v["leader_baudrate"],
        leader_frequency_hz=v["leader_frequency_hz"],
        leader_max_age_s=v["leader_max_age_s"],
        leader_read_budget_s=v["leader_read_budget_s"],
        close_sign=v["close_sign"],
        b601=B601HardwareConfig(
            port=v["port"],
            baudrate=v["baudrate"],
            joint_specs=DEFAULT_B601_JOINTS[-1:],
            control=control,
            startup_mode="prepare_disabled_gripper",
        ),
        sensors={"gripper_force": sensor_config(v)},
        teacher=teacher,
        safety=safety,
        runtime=RuntimeConfig(
            control_hz=v["control_hz"],
            scheduler_sleep_slice_ms=v["scheduler_sleep_slice_ms"],
            feedback_budget_ms=v["feedback_budget_ms"],
            tick_deadline_ms=v["tick_deadline_ms"],
            command_ttl_ms=v["command_ttl_ms"],
            object_id=v["object_id"],
            contact_condition_id=v["contact_condition_id"],
        ),
        trace_root=Path(output),
        queue_capacity=v["queue_capacity"],
        raw_queue_capacity=v["raw_queue_capacity"],
        trace_join_timeout_s=v["trace_join_timeout_s"],
        startup_timeout_s=v["startup_timeout_s"],
        shutdown_timeout_s=v["shutdown_timeout_s"],
        duration_s=v["max_run_time_s"],
    )
    config.validate()
    return config
