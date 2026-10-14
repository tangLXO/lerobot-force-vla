"""B601 FORCE_POS + I-P panel; raw force, explicit limits and one hardware owner."""

import argparse
import json
import math
import os
import queue
import tempfile
import threading
import time
import tkinter as tk
import uuid
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, ttk

import draccus
from b601_supervised_bench import run
from b601_tuner_process import ProcessLive, run_in_process

from lerobot.force_control.b601_factory import B601HardwareConfig
from lerobot.force_control.config import (
    SafetyConfig,
    TeacherConfig,
    finite_number,
    positive,
    validate_motion_request,
    validate_profile,
)
from lerobot.force_control.runtime import DEFAULT_FEEDBACK_BUDGET_MS, DEFAULT_TICK_DEADLINE_MS, RuntimeConfig
from lerobot.robots.rebot_b601_follower.calibration import DEFAULT_GRIPPER_CALIBRATION, B601GripperCalibration
from lerobot.robots.rebot_b601_follower.control_backend import B601ControlConfig
from lerobot.robots.rebot_b601_follower.control_protocol import B601JointCommandProfile, force_pos_wire_codes
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig
from lerobot.sensors.x518 import X518ChannelConfig, X518Sensor, X518SensorConfig

GRAVITY_N_PER_KG = 9.80665
OUTPUTS = Path("outputs/force_teacher_tuning")
_UNSTOPPED_PREVIEWS = []
# section, key, label with unit, visible default, kind, live, optional group
FIELDS = (
    ("control", "target_N", "目标平均力 (N)", "4.903325", float, True, None),
    ("control", "kp_deg_per_N", "Kp (°/N) · 必填，可为 0", "", float, True, None),
    ("control", "ki_deg_per_N_s", "Ki (°/(N·s)) · 必填，可为 0", "", float, True, None),
    ("control", "integral_deadband_N", "积分误差死区 (N)，0 关闭", "0", float, True, None),
    ("control", "approach_speed_deg_s", "恒速闭合速度 (°/s) · 使用时必填", "", float, True, None),
    (
        "control",
        "approach_force_threshold_N",
        "切换 I-P 的平均力阈值 (N) · 使用时必填",
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
    ("limits", "soft_q_min_deg", "软件角度下限 (°)", "", float, False, "soft_limits"),
    ("limits", "soft_q_max_deg", "软件角度上限 (°)", "", float, False, "soft_limits"),
    ("limits", "max_total_travel_deg", "累计绝对行程上限 (°)", "", float, False, "travel"),
    ("limits", "max_run_time_s", "运行时长上限 (s)", "", float, False, "duration"),
    ("limits", "force_max_age_ms", "力反馈最大年龄 (ms)", "50", float, False, None),
    ("limits", "joint_feedback_max_age_ms", "位置反馈最大年龄 (ms)", "50", float, False, None),
    ("limits", "decision_max_age_ms", "发送前决策最大年龄 (ms)", "20", float, False, None),
    ("limits", "max_dt_s", "有效控制周期间隔上限 (s)", "0.06", float, False, None),
    ("limits", "start_feedback_max_age_ms", "只读启动参考最大年龄 (ms)", "150", float, False, None),
    ("device", "port", "B601 串口", "COM22", str, False, None),
    ("device", "baudrate", "B601 波特率", "921600", int, False, None),
    ("device", "motor_velocity_code", "FORCE_POS 速度码 (整数 1–10000)", "1", int, False, None),
    ("device", "drive_percent", "FORCE_POS 电流 / 力矩比例 (%)", "2", float, False, None),
    ("device", "watchdog_ticks", "看门狗 TIMEOUT ticks (50 μs/tick)", "2000", int, False, None),
    ("device", "parameter_write_settle_s", "参数写入后等待读回 (s)", "0.01", float, False, None),
    ("device", "control_hz", "控制目标频率 (Hz)", "100", float, False, None),
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
    "soft_limits": ("启用额外软件角度限位", False),
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
SETTINGS_SCHEMA_VERSION = 4
MOTION_FIELDS = {
    "approach_speed_deg_s",
    "approach_force_threshold_N",
    "position_target_deg",
    "position_speed_deg_s",
}
LEGACY_TARGET_FIELDS = {
    "target_kg": "target_N",
    "target_min_kg": "target_min_N",
    "target_max_kg": "target_max_N",
}


def settings_document(values, enabled, calibration_path):
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
    }


def save_settings(path, values, enabled, calibration_path):
    """Replace one requested JSON atomically; preserve the previous file on failure."""
    document = settings_document(values, enabled, calibration_path)
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


def read_settings(path, *, on_unit_conversion=None):
    """Strictly read a parameter file; calibration and runtime state are not trusted from JSON."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    keys = {"format", "schema_version", "controller_type", "calibration_file", "values", "enabled"}
    if (
        not isinstance(document, dict)
        or set(document) != keys
        or document["format"] != SETTINGS_FORMAT
        or type(document["schema_version"]) is not int
        or document["schema_version"] not in (1, 2, 3, SETTINGS_SCHEMA_VERSION)
        or document["controller_type"] != "ip"
        or not isinstance(document["calibration_file"], str)
        or not document["calibration_file"].strip()
        or not isinstance(document["values"], dict)
        or not isinstance(document["enabled"], dict)
    ):
        raise ValueError("不是当前 I-P 参数文件，不能导入运行日志或旧控制配置")
    values = dict(document["values"])
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
    checked = settings_document(values, document["enabled"], calibration_path)
    if document["schema_version"] == 1 and on_unit_conversion is not None:
        on_unit_conversion("旧文件目标及范围已按 9.80665 N/kg 换算为 N，原文件未改写")
    return checked["values"], checked["enabled"], Path(checked["calibration_file"])


def default_values():
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


def build_config(values, enabled, calibration, output):
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
        approach_speed_deg_s=v["approach_speed_deg_s"],
        approach_force_threshold_N=v["approach_force_threshold_N"],
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
        calibration.minimum_deg,
        calibration.maximum_deg,
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


def exit_confirmation(result):
    """GUI restarts cannot erase an unconfirmed motor disposition."""
    if "trial" in result:  # Historical evidence is read without reusing its control configuration.
        disposition = (result.get("trial") or {}).get("disposition") or {}
        return (
            result.get("motor_transport_closed") is True
            and disposition.get("disabled_feedback_observed") is True
            and disposition.get("status") == "submitted"
            and disposition.get("error") is None
        )
    evidence = result.get("shutdown_evidence") or {}
    disposition = evidence.get("fault_disposition") or {}
    return (
        evidence.get("transport_closed") is True
        and disposition.get("disabled_feedback_observed") is True
        and disposition.get("status") == "submitted"
        and disposition.get("error") is None
    )


def exit_port(result):
    """Bind recovery to the failed session's device rather than another selected port."""
    config = result.get("config")
    hardware = config.get("b601") if isinstance(config, dict) else None
    port = hardware.get("port") if isinstance(hardware, dict) else None
    return port if isinstance(port, str) and port.strip() else None


def readonly_exit_reference(row, *, session_token, started_ns, now_ns, joint_age_ms, force_age_ms):
    """Accept only this live query-only connection's fresh disabled device and raw force evidence."""
    if (
        not isinstance(row, dict)
        or not session_token
        or row.get("monitor_session") != session_token
        or row.get("read_only") is not True
        or row.get("joint_valid") is not True
        or row.get("force_valid") is not True
        or row.get("phase") != "DISABLED"
        or type(row.get("joint_status_code")) is not int
        or row.get("joint_status_code") != 0
    ):
        return False
    try:
        for key in ("q_measured_deg", "force_left_N", "force_right_N"):
            finite_number(key, row.get(key))
        for stamp, age_ms in (
            (row.get("joint_received_ns"), positive("位置年龄", joint_age_ms)),
            (row.get("force_timestamp_ns"), positive("力年龄", force_age_ms)),
        ):
            if (
                type(stamp) is not int
                or type(started_ns) is not int
                or type(now_ns) is not int
                or not 0 <= started_ns <= stamp <= now_ns
                or now_ns - stamp > age_ms * 1e6
            ):
                return False
    except (ValueError, TypeError):
        return False
    return True


def stop_report(result, path):
    reason = "; ".join(result.get("errors") or []) or result.get(
        "terminal_phase", result.get("terminal_state", "会话结束")
    )
    config = result.get("config")
    config = config if isinstance(config, dict) else {}
    runtime = config.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    safety = config.get("safety")
    safety = safety if isinstance(safety, dict) else {}
    query_errors = (
        error.partition("joint_error:")[2].strip()
        for error in result.get("errors") or []
        if "invalid_joint_feedback" in error and "joint_error:" in error
    )
    timeout_messages = (
        "B601 read-only deadline expired",
        "B601 write has less than one millisecond remaining",
        "B601 read has less than one millisecond remaining",
        "B601 response arrived after deadline",
        "B601 no response before deadline",
        "B601 response incomplete at deadline",
    )
    query_timeout = next(
        (
            error
            for error in query_errors
            if error.startswith("TimeoutError:") and any(text in error for text in timeout_messages)
        ),
        None,
    )
    if query_timeout:
        diagnosis = (
            "位置反馈查询预算耗尽（发起查询时剩余时间不足 1 ms）"
            if "B601 write has less than one millisecond remaining" in query_timeout
            else "位置反馈查询超时"
        )
        explanation = (
            f"{diagnosis}：本次查询 / 周期工作预算 "
            f"{parameter_number_text(runtime.get('feedback_budget_ms'))} / "
            f"{parameter_number_text(runtime.get('tick_deadline_ms'))} ms。"
            "停止并断开后检查通信耗时，再调整并保存预算；"
            f"B601 默认 {DEFAULT_FEEDBACK_BUDGET_MS:g} / {DEFAULT_TICK_DEADLINE_MS:g} ms 仅供参照，"
            "不会自动替代本次设置。"
        )
        reason = f"{explanation} 原始原因：{reason}"
    else:
        for side, label in (("left", "左"), ("right", "右")):
            if f"{side}_force_hard_limit" in reason:
                threshold = parameter_number_text(safety.get(f"hard_force_limit_{side}_N"))
                reason = f"{label}侧原始力触发过力保护，停止阈值 {threshold} N；原始原因：{reason}"
                break
    if result.get("valid") is False:
        reason = f"证据核验失败：{reason}"
    return {
        "summary": f"{reason}；{'已观察夹爪失能且串口关闭' if exit_confirmation(result) else '退出失能未确认，禁止新会话'}",
        "details": json.dumps(
            {
                "errors": result.get("errors"),
                "shutdown": result.get("shutdown_evidence"),
                "runtime_config": runtime,
                "safety_config": safety,
                "log": str(path),
            },
            ensure_ascii=False,
            indent=2,
        ),
    }


def parameter_number_text(value, *, scale=1.0):
    """Show recorded values, never substitute a default for missing evidence."""
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return "未填写 / 未记录"
    return (
        f"{number:.12g}" if not isinstance(value, bool) and math.isfinite(number) and number > 0 else "无效"
    )


def communication_budget_notice(values):
    """Explain old file values without rewriting them or blocking operation."""
    try:
        query = float(values["feedback_budget_ms"])
        tick = float(values["tick_deadline_ms"])
    except (KeyError, TypeError, ValueError):
        return ""
    if (
        math.isfinite(query)
        and math.isfinite(tick)
        and (query < DEFAULT_FEEDBACK_BUDGET_MS or tick < DEFAULT_TICK_DEADLINE_MS)
    ):
        return (
            f"已保留文件查询 / 周期预算 {query:g} / {tick:g} ms；"
            f"默认 {DEFAULT_FEEDBACK_BUDGET_MS:g} / {DEFAULT_TICK_DEADLINE_MS:g} ms 未替代文件值"
        )
    return ""


def timing_settings_description(values, config=None, *, startup_complete=False, running=False):
    """Keep editable requests distinct from the reviewed owner configuration."""
    requested = (
        f"请求查询 / 周期工作 / 发送预算：{parameter_number_text(values.get('feedback_budget_ms'))} / "
        f"{parameter_number_text(values.get('tick_deadline_ms'))} / "
        f"{parameter_number_text(values.get('send_budget_s'), scale=1000)} ms"
    )
    if config is None or not hasattr(config, "runtime"):
        return f"{requested}\n尚未生效；只读监测不使用控制运行预算"
    if startup_complete:
        state = "本次 owner 运行生效" if running else "上次 owner 运行配置（已结束，当前未生效）"
    else:
        state = "本次 owner 待启动配置（尚未生效）" if running else "上次提交配置（启动未完成，当前未生效）"
    return (
        f"{requested}\n{state}：查询 {config.runtime.feedback_budget_ms:g} ms / "
        f"周期工作 {config.runtime.tick_deadline_ms:g} ms / "
        f"发送 {config.b601.control.send_budget_s * 1000:g} ms"
    )


def timing_description(row):
    """Show measured tick/query time against its recorded absolute deadline."""
    start = row.get("tick_start_ns")
    if type(start) is not int:
        return ""
    snapshot = row.get("joint_snapshot") or {}
    completed, deadline = snapshot.get("completed_ns"), snapshot.get("deadline_ns")
    query = "本周期未查询"
    if type(completed) is int and type(deadline) is int and min(completed, deadline) >= start:
        query = f"查询实际 / 生效预算 {(completed - start) / 1e6:.3f} / {(deadline - start) / 1e6:g} ms"

    def milliseconds(value):
        return f"{value * 1000:.3f}" if type(value) in (int, float) and math.isfinite(value) else "—"

    return (
        f"{query} | 周期间隔 {milliseconds(row.get('dt_control_s'))} ms | "
        f"本周期工作 {milliseconds(row.get('loop_elapsed_s'))} ms"
    )


def tuning_estimate(q0, q1, f0, f1, t0, t90, alpha_i, alpha_p):
    """Operator-fed calculator; never write gains or command motion."""
    for value in (q0, q1, f0, f1, t0, t90, alpha_i, alpha_p):
        finite_number("调参测量值 / 系数", value)
    if q1 == q0:
        raise ValueError("测量角度变化不能为 0")
    kappa, response = (f1 - f0) / (q1 - q0), t90 - t0
    positive("κ (N/°)", kappa)
    positive("t90 - t0 (s)", response)
    positive("积分系数 αI", alpha_i)
    finite_number("比例系数 αP", alpha_p, minimum=0)
    return kappa, response, alpha_i / (kappa * response), alpha_p / kappa


def startup_description(receipt):
    """Show software settings and the fresh readback, without claiming hardware execution."""
    settings = receipt["command_settings"]
    requested, applied, encoded = (settings[key] for key in ("requested", "applied", "encoded"))
    return (
        f"启动{'完成' if receipt.get('complete') else '未完成'} | FORCE_POS=4 | "
        f"看门狗 {receipt['watchdog_ticks'] * 0.05:g} ms\n"
        f"请求底层速度 {requested['motor_velocity_deg_s']:g}°/s → "
        f"码 {encoded['velocity_wire_code']} / 编码有效 {applied['motor_velocity_deg_s']:g}°/s | "
        f"请求驱动 {requested['drive_limit_ratio'] * 100:g}% → "
        f"码 {encoded['drive_wire_code']} / 编码有效 {applied['drive_limit_ratio'] * 100:g}%\n"
        f"本次读回 PMAX / VMAX / TMAX：{settings.get('device_mapping')}；"
        "以上为设置与读回证据，hardware_ack=unknown"
    )


def monitor(stop, publish, v, calibration, cleanup_publish=None):
    """Query-only preview; close both devices before transferring ownership."""
    calibration.require_unchanged()
    if _UNSTOPPED_PREVIEWS:
        raise RuntimeError("前次只读资源退出未确认，禁止重复打开设备")
    device = ReadonlyB601Transport(v["port"], DEFAULT_B601_JOINTS[-1:], baudrate=v["baudrate"])
    sensor = X518Sensor(sensor_config(v))
    lease = None
    try:
        sensor.connect()
        lease = sensor.acquire_recorder()
        device.open()
        if not device.read_mapping(max_read_duration_s=v["startup_budget_s"]).is_valid:
            raise RuntimeError("电机映射读回失败")
        deadline = time.monotonic() + v["startup_timeout_s"]
        seen = False
        while not stop.is_set():
            if lease.error:
                raise RuntimeError(f"X518 采集失败：{lease.error}")
            batch = device.read_feedback(max_read_duration_s=v["monitor_feedback_budget_ms"] / 1000)
            if not batch.is_valid:
                raise RuntimeError(f"B601 反馈无效：{batch.error}")
            now = time.perf_counter_ns()
            try:
                force = sensor.read_latest_before(now, max_age_ms=v["force_max_age_ms"])
            except (RuntimeError, TimeoutError):
                if seen or time.monotonic() >= deadline:
                    raise
                stop.wait(v["monitor_interval_ms"] / 1000)
                continue
            if not force.is_valid:
                raise RuntimeError("X518 原始读数无效")
            seen = True
            joint = batch.feedback[0]
            left, right = float(force.values["left.normal_force"]), float(force.values["right.normal_force"])
            publish(
                {
                    "decision_anchor_ns": now,
                    "force_left_N": left,
                    "force_right_N": right,
                    "force_mean_N": (left + right) / 2,
                    "q_measured_deg": joint.position_deg,
                    "phase": joint.status_name,
                    "joint_received_ns": joint.receive_timestamp_ns,
                    "joint_status_code": joint.status_code,
                    "joint_valid": True,
                    "force_timestamp_ns": force.timestamp_ns,
                    "force_valid": True,
                    "read_only": True,
                }
            )
            stop.wait(v["monitor_interval_ms"] / 1000)
    finally:
        try:
            if device.has_resources:
                device.close()
        finally:
            try:
                if lease is not None:
                    sensor.release_recorder(lease)
            finally:
                try:
                    if sensor.has_resources:
                        sensor.disconnect()
                finally:
                    cleanup = {
                        "read_only": True,
                        "motion_attempted": False,
                        "transport_closed": not device.has_resources,
                        "sensor_closed": not sensor.has_resources,
                    }
                    if device.has_resources or sensor.has_resources:
                        _UNSTOPPED_PREVIEWS.append((device, sensor, lease))
                    if cleanup_publish is not None:
                        cleanup_publish(cleanup)


class Tuner:
    """Tk renders snapshots; a spawned common runtime executes every motion."""

    def __init__(self, root, *, calibration_path=None):
        self.root, self.worker = root, None
        self.running = self.monitoring = self.closing = False
        self.exit_confirmed = True
        self.unconfirmed_port = None
        self.recovery_evidence = None
        self.monitor_session = None
        self.monitor_started_ns = None
        self.monitor_port = None
        self.monitor_stop = threading.Event()
        self.messages = queue.SimpleQueue()
        self.preview_latest = self.live = self.next_run = self.calibration = None
        self.pending_cleanup = None
        self.preview_cleanup = None
        self.config = self.estimate = self.last_anchor = None
        self.last_applied = None
        self.last_startup = None
        self.calibration_path = Path(calibration_path or DEFAULT_GRIPPER_CALIBRATION)
        self.settings_path = None
        self.points, self.fixed_widgets, self.optional_widgets = deque(), [], []
        self.other_widgets = []
        self.fields, self.toggles = {}, {}
        root.title("B601 · FORCE_POS + I-P 力控制")
        root.geometry("1320x920")
        root.minsize(1060, 740)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.stop())
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="夹爪力控制  FORCE_POS + I-P", font=("Microsoft YaHei UI", 18, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            outer, text="Δθ = Ki · dt · (目标 − 原始平均力) − Kp · Δ原始平均力    |    只累计最终成功提交位置"
        ).pack(anchor="w", pady=(4, 10))
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        self.calibration_label = tk.StringVar(value="校准文件未加载")
        ttk.Label(bar, textvariable=self.calibration_label, wraplength=1050).pack(side="left")
        self.calibration_button = ttk.Button(bar, text="选择 / 重载校准", command=self.choose_calibration)
        self.calibration_button.pack(side="right")
        self.fixed_widgets.append(self.calibration_button)
        files = ttk.Frame(outer)
        files.pack(fill="x", pady=(8, 0))
        self.save_button = ttk.Button(files, text="保存参数…", command=self.save_parameters)
        self.save_button.pack(side="left")
        self.load_button = ttk.Button(files, text="加载参数…", command=self.load_parameters)
        self.load_button.pack(side="left", padx=6)
        self.fixed_widgets.extend((self.save_button, self.load_button))
        self.profile_label = tk.StringVar(value="可保存未填完的参数草稿；加载后需重新只读确认，不自动运动")
        ttk.Label(files, textvariable=self.profile_label, wraplength=950).pack(side="left", padx=8)
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True, pady=10)
        left, right = ttk.Frame(body, width=510), ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)
        pages, page_canvases = {}, {}
        for key, title in (("gripper", "夹爪控制"), ("other", "其他设置")):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            canvas = tk.Canvas(frame, width=490, highlightthickness=0)
            scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            inner = ttk.Frame(canvas, padding=10)
            window = canvas.create_window(0, 0, anchor="nw", window=inner)
            inner.bind("<Configure>", lambda _, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda e, c=canvas, w=window: c.itemconfigure(w, width=e.width))
            inner.bind("<MouseWheel>", lambda e, c=canvas: c.yview_scroll(-int(e.delta / 120), "units"))
            pages[key] = inner
            page_canvases[key] = canvas
        self.other_unlocked = tk.BooleanVar(value=False)
        self.other_unlock_check = ttk.Checkbutton(
            pages["other"],
            text="解锁其他设置（仅停止并断开后）",
            variable=self.other_unlocked,
            command=self.update_editability,
        )
        self.other_unlock_check.pack(anchor="w", pady=(0, 8))
        ttk.Label(pages["other"], text="通信、缓存和显示参数默认锁定，避免调参时误改。", wraplength=440).pack(
            anchor="w", pady=(0, 8)
        )
        sections = {}
        for key, title, page in (
            ("control", "力控制与手动操作", "gripper"),
            ("limits", "运动限制与保护", "gripper"),
            ("force_pos", "底层 FORCE_POS 与控制时序", "gripper"),
            ("device", "设备连接、缓存与显示", "other"),
        ):
            sections[key] = ttk.LabelFrame(pages[page], text=title, padding=8)
            sections[key].pack(fill="x", pady=(0, 10))
        rows, seen_groups = dict.fromkeys(sections, 0), set()
        ttk.Label(
            sections["control"],
            text="目标、Kp/Ki、死区、恒速闭合参数和位置变化率可实时应用；指定位置需点击按钮。方向、硬保护及设备参数停止并断开后修改。",
            wraplength=420,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        rows["control"] = 1
        for section, key, label, default, _, live, group in FIELDS:
            if key in GRIPPER_DEVICE_FIELDS:
                section = "force_pos"
            frame = sections[section]
            if group and group not in seen_groups:
                text, state = OPTIONAL_GROUPS[group]
                var = self.toggles[group] = tk.BooleanVar(value=state)
                toggle = ttk.Checkbutton(frame, text=text, variable=var, command=self.update_optional)
                toggle.grid(row=rows[section], column=0, columnspan=2, sticky="w", pady=(10, 4))
                if not live:
                    self.fixed_widgets.append(toggle)
                rows[section] += 1
                seen_groups.add(group)
            var = self.fields[key] = tk.StringVar(value=default)
            ttk.Label(frame, text=label, wraplength=300).grid(row=rows[section], column=0, sticky="w", pady=4)
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.grid(row=rows[section], column=1, sticky="e", padx=(8, 0))
            entry.bind(
                "<MouseWheel>", lambda e, parent=frame: parent.event_generate("<MouseWheel>", delta=e.delta)
            )
            frame.columnconfigure(0, weight=1)
            if not live:
                self.fixed_widgets.append(entry)
            if FIELD_PAGES[key] == "other":
                self.other_widgets.append(entry)
            if group:
                self.optional_widgets.append((entry, group, live))
            rows[section] += 1
        ttk.Label(
            sections["force_pos"],
            text="固定协议：夹爪 ID 7 / feedback 0x17 / FORCE_POS=4；整数速度码 1–10000，每码 0.01 rad/s，直接按所填码发送。驱动比例步进 0.01%。底层速度是上限，外环位置变化率独立受限。串口计时等待按 1 ms 粒度设置，剩余预算不足 1 ms 时停止 I/O。X518 语义单位 N (9.80665 N/kg)。仅支持单夹爪；全臂物理控制门禁仍关闭。",
            wraplength=440,
        ).grid(row=rows["force_pos"], column=0, columnspan=2, sticky="w", pady=12)
        self.speed_label = tk.StringVar()
        ttk.Label(sections["force_pos"], textvariable=self.speed_label, wraplength=440).grid(
            row=rows["force_pos"] + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.fields["motor_velocity_code"].trace_add("write", lambda *_: self.update_speed_label())
        self.update_speed_label()
        self.calculator = {}
        frame = ttk.LabelFrame(sections["control"], text="κ / t90 计算器 · 不自动运动或写参数", padding=8)
        frame.grid(row=rows["control"], column=0, columnspan=2, sticky="ew", pady=14)
        for index, (key, label) in enumerate(
            (
                ("q0", "闭合坐标 θ0 (°)"),
                ("q1", "闭合坐标 θ1 (°)"),
                ("f0", "稳态原始平均力 F0 (N)"),
                ("f1", "稳态原始平均力 F1 (N)"),
                ("t0", "动作起点时间 (s)"),
                ("t90", "首次到达 90% 的时间 (s)"),
                ("alpha_i", "αI：Ki = αI / (κ·t90)"),
                ("alpha_p", "αP：Kp = αP / κ"),
            )
        ):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
            var = self.calculator[key] = tk.StringVar()
            ttk.Entry(frame, textvariable=var, width=13).grid(row=index, column=1, sticky="e")
        ttk.Button(frame, text="计算（仅显示）", command=self.calculate).grid(
            row=8, column=0, sticky="ew", pady=6
        )
        ttk.Button(frame, text="手动填入 Kp / Ki", command=self.fill_estimate).grid(
            row=8, column=1, sticky="ew"
        )
        self.calculation_label = tk.StringVar(value="κ=(F1−F0)/(θ1−θ0)，响应时间=t90−t0")
        ttk.Label(frame, textvariable=self.calculation_label, wraplength=420).grid(
            row=9, column=0, columnspan=2
        )
        for page, inner in pages.items():
            children = [inner]
            while children:
                widget = children.pop()
                widget.bind(
                    "<MouseWheel>",
                    lambda event, c=page_canvases[page]: c.yview_scroll(-int(event.delta / 120), "units"),
                )
                children.extend(widget.winfo_children())
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(8, 0))
        self.read_button = ttk.Button(actions, text="连接 / 只读监测", command=self.read)
        self.read_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="应用实时参数", command=self.apply).grid(row=0, column=1, sticky="ew")
        self.confirmed = tk.BooleanVar()
        confirm = ttk.Checkbutton(
            actions, text="已确认机械设置、闭合方向和力通道，手已移开", variable=self.confirmed
        )
        confirm.grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        self.confirm_check = confirm
        self.start_button = ttk.Button(actions, text="开始 I-P 控制（使能）", command=self.start)
        self.start_button.grid(row=2, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="停止 / 夹爪失能 [Esc]", command=self.stop).grid(
            row=2, column=1, sticky="ew"
        )
        self.freeze_button = ttk.Button(actions, text="冻结固定位置", command=lambda: self.command("freeze"))
        self.freeze_button.grid(row=3, column=0, sticky="ew", pady=6, padx=(0, 4))
        self.resume_button = ttk.Button(actions, text="恢复 I-P", command=lambda: self.command("resume"))
        self.resume_button.grid(row=3, column=1, sticky="ew")
        self.step_button = ttk.Button(actions, text="执行手动小步（仅固定位置到位后）", command=self.step)
        self.step_button.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.approach_button = ttk.Button(actions, text="恒速闭合 → I-P", command=self.approach)
        self.approach_button.grid(row=5, column=0, sticky="ew", pady=6, padx=(0, 4))
        self.position_button = ttk.Button(actions, text="运动到指定位置", command=self.move_to_position)
        self.position_button.grid(row=5, column=1, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        stopped = ttk.LabelFrame(right, text="最近一次停止原因（重连后保留）", padding=8)
        stopped.pack(fill="x")
        self.last_stop, self.last_stop_details = tk.StringVar(value="暂无停止记录"), ""
        ttk.Label(stopped, textvariable=self.last_stop, wraplength=690).pack(fill="x")
        ttk.Button(stopped, text="查看退出原始证据", command=self.show_stop_details).pack(anchor="e")
        self.settings_label = tk.StringVar(value="底层请求值、编码有效值和全部设置在启动前检查并记录")
        ttk.Label(right, textvariable=self.settings_label, wraplength=700).pack(fill="x", pady=8)
        self.timing_settings_label = tk.StringVar()
        ttk.Label(right, textvariable=self.timing_settings_label, wraplength=700).pack(fill="x", pady=(0, 8))
        for key in ("feedback_budget_ms", "tick_deadline_ms", "send_budget_s"):
            self.fields[key].trace_add("write", lambda *_: self.update_timing_settings())
        self.update_timing_settings()
        self.values_label = tk.StringVar(value="未连接；程序打开不连接设备")
        ttk.Label(right, textvariable=self.values_label, wraplength=700).pack(fill="x", pady=8)
        self.canvas = tk.Canvas(right, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Kp、Ki、闭合方向、目标上限、领先 / 跟随 / 停滞限值须由操作者填写。")
        ttk.Label(outer, textvariable=self.status, wraplength=1280).pack(fill="x")
        latest = max(OUTPUTS.glob("*/run/result.json"), key=lambda p: p.stat().st_mtime, default=None)
        if latest is not None:
            try:
                result = json.loads(latest.read_text(encoding="utf-8"))
                self.show_report(stop_report(result, latest))
                self.exit_confirmed = exit_confirmation(result)
                if not self.exit_confirmed:
                    self.unconfirmed_port = exit_port(result)
                    self.status.set("历史退出未确认；可连接只读监测，重新核对同一夹爪当前失能状态")
            except (ValueError, OSError) as exc:
                self.exit_confirmed = False
                self.last_stop.set(f"历史退出日志读取失败：{exc}")
        self.load_calibration()
        self.update_optional()
        self.poll()

    def values(self):
        return {key: var.get() for key, var in self.fields.items()}

    def enabled(self):
        return {key: var.get() for key, var in self.toggles.items()}

    def update_speed_label(self):
        try:
            code = int(self.fields["motor_velocity_code"].get().strip())
            force_pos_wire_codes(None, 1.0, velocity_wire_code=code)
            self.speed_label.set(
                f"速度码 {code} → {code * 0.01:g} rad/s → {math.degrees(code * 0.01):.6g}°/s"
            )
        except ValueError:
            self.speed_label.set("速度码须为 1–10000 的整数，不自动取整或裁剪")

    def update_editability(self):
        busy = self.settings_busy()
        if busy:
            self.other_unlocked.set(False)
        for widget in self.fixed_widgets:
            widget.configure(state="disabled" if busy else "normal")
        for widget in self.other_widgets:
            widget.configure(state="normal" if self.other_unlocked.get() and not busy else "disabled")
        self.other_unlock_check.configure(state="disabled" if busy else "normal")
        self.update_optional()

    def save_parameters_file(self, path):
        if self.settings_busy():
            raise RuntimeError("请停止并断开后保存参数")
        self.settings_path = save_settings(path, self.values(), self.enabled(), self.calibration_path)
        self.profile_label.set(f"已保存参数草稿：{self.settings_path}（执行前仍会完整校验）")

    def update_timing_settings(self):
        self.timing_settings_label.set(
            timing_settings_description(
                self.values(),
                self.config,
                startup_complete=isinstance(self.last_startup, dict)
                and self.last_startup.get("complete") is True,
                running=self.running,
            )
        )

    def load_parameters_file(self, path):
        if self.settings_busy():
            raise RuntimeError("请停止并断开后加载参数")
        unit_notices = []
        values, enabled, calibration_path = read_settings(path, on_unit_conversion=unit_notices.append)
        calibration = B601GripperCalibration.load(calibration_path)
        for key, value in values.items():
            self.fields[key].set(value)
        for key, value in enabled.items():
            self.toggles[key].set(value)
        self.calibration_path, self.calibration = calibration_path, calibration
        self.settings_path = Path(path).resolve()
        self.config = self.estimate = self.preview_latest = self.last_anchor = None
        self.live = self.last_applied = self.last_startup = None
        self.points.clear()
        self.confirmed.set(False)
        self.other_unlocked.set(False)
        self.values_label.set("未连接；参数已加载，需重新取得新鲜只读反馈")
        self.settings_label.set("参数文件已加载，尚未下发；开始前重新验证配置与设备映射")
        self.calculation_label.set("κ=(F1−F0)/(θ1−θ0)，响应时间=t90−t0")
        self.show_calibration()
        self.update_editability()
        self.update_timing_settings()
        unit_description = unit_notices[0] if unit_notices else "目标及范围单位：N"
        budget_notice = communication_budget_notice(values)
        self.profile_label.set(
            f"已加载：{self.settings_path}；{unit_description}；"
            f"{budget_notice + '；' if budget_notice else ''}需重新只读确认，未应用或运动"
        )
        self.status.set("参数已加载；运行前会重新校验所有值和校准，旧参考及确认已清除")

    def save_parameters(self):
        if self.settings_busy():
            self.status.set("请停止并断开后保存参数")
            return
        path = filedialog.asksaveasfilename(
            title="保存 I-P 参数草稿",
            defaultextension=".json",
            filetypes=[("I-P 参数 JSON", "*.json")],
            initialfile=self.settings_path.name if self.settings_path else "b601_ip_parameters.json",
        )
        if path:
            try:
                self.save_parameters_file(path)
            except (ValueError, OSError, RuntimeError) as exc:
                self.status.set(str(exc))

    def load_parameters(self):
        if self.settings_busy():
            self.status.set("请停止并断开后加载参数")
            return
        path = filedialog.askopenfilename(title="加载 I-P 参数", filetypes=[("I-P 参数 JSON", "*.json")])
        if path:
            try:
                self.load_parameters_file(path)
            except (ValueError, OSError, RuntimeError) as exc:
                self.status.set(str(exc))

    def show_calibration(self):
        c = self.calibration
        self.calibration_label.set(
            f"硬角度范围 [{c.minimum_deg:g}, {c.maximum_deg:g}]° | SHA256 {c.sha256[:12]} | {c.path}"
        )

    def update_optional(self):
        for widget, group, live in self.optional_widgets:
            widget.configure(
                state="normal"
                if self.toggles[group].get() and (live or not self.settings_busy())
                else "disabled"
            )

    def load_calibration(self):
        if self.settings_busy():
            self.status.set("请停止并断开后重新加载校准")
            return False
        self.preview_latest = None
        self.confirmed.set(False)
        self.points.clear()
        try:
            self.calibration = B601GripperCalibration.load(self.calibration_path)
            self.show_calibration()
            for key, value in (
                ("soft_q_min_deg", self.calibration.minimum_deg),
                ("soft_q_max_deg", self.calibration.maximum_deg),
            ):
                if not self.fields[key].get().strip():
                    self.fields[key].set(str(value))
            return True
        except ValueError as exc:
            self.calibration = None
            self.calibration_label.set(str(exc))
            return False

    def choose_calibration(self):
        if not self.settings_busy():
            path = filedialog.askopenfilename(filetypes=[("Calibration JSON", "*.json")])
            if path:
                self.calibration_path = Path(path)
                self.load_calibration()

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def settings_busy(self):
        return self.busy() or self.running or self.monitoring or self.next_run is not None

    def launch(self, function):
        def worker():
            try:
                self.messages.put(("done", function()))
            except BaseException as exc:
                self.messages.put(("error", f"{type(exc).__name__}: {exc}"))
                self.messages.put(("done", None))

        self.worker = threading.Thread(target=worker, name="b601-panel-worker", daemon=False)
        self.worker.start()

    def monitor_values(self):
        values = self.values()
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
        result = {key: kinds[key](values[key]) for key in needed}
        for key in (
            "monitor_interval_ms",
            "monitor_feedback_budget_ms",
            "startup_timeout_s",
            "startup_budget_s",
        ):
            positive(key, result[key])
        return result

    def read(self):
        if self.running:
            return
        if self.worker is not None:
            if self.busy() and self.monitoring:
                self.monitor_stop.set()
            else:
                self.status.set("等待当前 worker 清理完成后再连接")
            return
        if self.next_run is not None or self.pending_cleanup is not None or _UNSTOPPED_PREVIEWS:
            self.status.set("现有资源仍待清理，不能重复连接")
            return
        if not self.load_calibration():
            return
        try:
            values = self.monitor_values()
            sensor_config(values)
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc))
            return
        self.preview_latest = None
        self.preview_cleanup = None
        self.monitor_stop.clear()
        token = self.monitor_session = uuid.uuid4().hex
        self.monitor_started_ns = time.perf_counter_ns()
        self.monitor_port = values["port"]
        self.monitoring = True
        self.status.set(
            "正在只读核对当前失能状态…" if not self.exit_confirmed else "正在连接设备进行只读监测…"
        )

        def publish(row):
            if self.monitor_session == token:
                self.preview_latest = dict(row, monitor_session=token)

        def preview():
            try:
                monitor(
                    self.monitor_stop,
                    publish,
                    values,
                    self.calibration,
                    lambda evidence: self.messages.put(("preview_cleanup", evidence)),
                )
            finally:
                self.monitoring = False
            if self.next_run:
                execute, self.next_run = self.next_run, None
                return execute()
            if self.pending_cleanup is not None:
                self.pending_cleanup()
                self.pending_cleanup = None
            return None

        self.launch(preview)

    def recover_readonly_exit(self, row):
        if (
            self.exit_confirmed
            or not self.monitoring
            or self.running
            or self.worker is None
            or not self.worker.is_alive()
            or self.next_run is not None
            or self.pending_cleanup is not None
            or _UNSTOPPED_PREVIEWS
            or self.monitor_stop.is_set()
        ):
            return False
        if (
            self.unconfirmed_port is None
            or not isinstance(self.monitor_port, str)
            or self.monitor_port.casefold() != self.unconfirmed_port.casefold()
        ):
            self.status.set("只读监测可查看；退出未确认会话的串口须一致，当前不能解除门禁")
            return False
        try:
            if self.calibration is None:
                raise ValueError("请加载有效校准后再只读核对")
            if not readonly_exit_reference(
                row,
                session_token=self.monitor_session,
                started_ns=self.monitor_started_ns,
                now_ns=time.perf_counter_ns(),
                joint_age_ms=float(self.fields["start_feedback_max_age_ms"].get()),
                force_age_ms=float(self.fields["force_max_age_ms"].get()),
            ):
                return False
            self.calibration.require_unchanged()
            if not self.calibration.minimum_deg <= row["q_measured_deg"] <= self.calibration.maximum_deg:
                raise ValueError("当前失能位置超出校准硬范围")
            evidence = {
                "source": "fresh_query_only_monitor",
                "port": self.monitor_port,
                "motor_id": 7,
                "monitor_session": self.monitor_session,
                "observed_ns": time.perf_counter_ns(),
                "disabled_feedback_observed": True,
                "hardware_ack": "unknown",
                "hardware_stop_confirmed": False,
                "calibration": asdict(self.calibration),
                "reference": dict(row),
            }
            folder = OUTPUTS / "readonly_recovery"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.monitor_session}.json"
            path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (ValueError, OSError, TypeError) as exc:
            self.status.set(f"只读核对未通过：{exc}")
            return False
        self.recovery_evidence = evidence
        self.exit_confirmed = True
        self.confirmed.set(False)
        self.last_stop.set(f"当前已只读核对失能；历史记录：{self.last_stop.get()}")
        self.last_stop_details = json.dumps(
            {
                "previous_exit_details": self.last_stop_details,
                "read_only_recovery": evidence,
                "recovery_receipt": str(path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        self.status.set("已重新只读观察到同一夹爪失能；需重新勾选机械 / 方向确认后才可开始")
        return True

    def start(self, mode="force", position_deg=None, position_speed_deg_s=None):
        if self.running:
            return
        try:
            if not self.exit_confirmed:
                raise ValueError("上次退出未确认，禁止开始新会话")
            row = self.preview_latest
            age = positive("启动反馈最大年龄", float(self.fields["start_feedback_max_age_ms"].get())) * 1e6
            if (
                not self.monitoring
                or row is None
                or not 0 <= time.perf_counter_ns() - row["joint_received_ns"] <= age
            ):
                raise ValueError("请连接设备，等候新鲜只读反馈")
            if row["phase"] != "DISABLED" or not self.confirmed.get():
                raise ValueError("需要夹爪失能反馈以及机械 / 方向 / 力通道确认")
            folder = OUTPUTS / uuid.uuid4().hex
            config = build_config(self.values(), self.enabled(), self.calibration, folder / "run/stream")
            config = replace(
                config,
                initial_mode=mode,
                initial_position_deg=position_deg,
                initial_position_speed_deg_s=position_speed_deg_s,
            )
            config.validate()
            folder.mkdir(parents=True, exist_ok=False)
            path = folder / "reviewed.json"
            draft = {
                "config": draccus.encode(config),
                "calibration": asdict(self.calibration),
                "operator_values": self.values(),
                "operator_toggles": self.enabled(),
                "display": {
                    key: self.fields[key].get() for key in ("display_hz", "plot_seconds", "ipc_poll_ms")
                },
                "reference": row,
                "exit_recovery": self.recovery_evidence,
                "executable": True,
                "consumed": False,
            }
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            reviewed = run(folder / "run", reviewed_config=path)
            self.settings_label.set(
                f"FORCE_POS=4 | 看门狗 {config.b601.control.watchdog_ticks * 0.05:g} ms | "
                f"请求速度码 {config.b601.control.profiles[0].velocity_wire_code} → "
                f"码 {reviewed['velocity_wire_code']} / 编码有效 {math.degrees(reviewed['velocity_wire_code'] * 0.01):g}°/s | "
                f"请求驱动 {config.b601.control.profiles[0].torque_ratio * 100:g}% → "
                f"码 {reviewed['drive_wire_code']} / 编码有效 {reviewed['drive_wire_code'] / 100:g}%"
                "\n设备映射将在使能前读回；速度字段和驱动比例是命令设置。"
            )
        except (ValueError, TypeError, OSError) as exc:
            self.status.set(str(exc))
            return
        self.config = config
        self.last_startup = None
        self.live = ProcessLive(poll_s=float(self.fields["ipc_poll_ms"].get()) / 1000)
        self.running, self.exit_confirmed = True, False
        self.update_timing_settings()
        self.unconfirmed_port = config.b601.port
        self.points.clear()
        self.status.set("交接只读资源；控制进程将读取新参考后准备并使能夹爪…")

        def execute():
            self.pending_cleanup = None
            try:
                result = run_in_process(folder / "run", reviewed_config=path, live=self.live)
                self.messages.put(("result", (result, folder / "run/result.json")))
                return result
            finally:
                draft.update(executable=False, consumed=True)
                path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        def cancel_pending():
            self.live.close()
            draft.update(executable=False, consumed=True)
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.exit_confirmed = True  # Preview closed; no control process was started.

        self.pending_cleanup = cancel_pending
        self.next_run = execute
        self.monitor_stop.set()

    def apply(self):
        try:
            if self.calibration is None:
                raise ValueError("请先加载有效校准")
            config = build_config(
                self.values(),
                self.enabled(),
                self.calibration,
                self.config.trace_root if self.config else OUTPUTS,
            )
            if self.running:
                self.live.submit("parameters", {"teacher": config.teacher, "safety": config.safety})
                self.status.set("整组实时参数已发送；以下一次实际应用收据为准")
            else:
                self.status.set("配置有效，将在下次开始时使用")
        except (ValueError, TypeError, RuntimeError, queue.Full) as exc:
            self.status.set(str(exc) or "上一操作尚未接收")

    def command(self, kind, value=None):
        if self.running and self.live is not None:
            try:
                self.live.submit(kind, value)
                self.status.set(f"已发送 {kind} 请求，等待 owner 收据")
            except (ValueError, RuntimeError, queue.Full) as exc:
                self.status.set(str(exc) or "上一操作尚未接收")

    def step(self):
        try:
            value = finite_number("手动小步", float(self.fields["manual_step_deg"].get()))
            if value == 0:
                raise ValueError("手动小步不能为 0")
            self.command("step", value)
        except ValueError as exc:
            self.status.set(str(exc))

    def approach(self):
        """Submit one closure request with the speed and threshold currently displayed."""
        if not self.running:
            self.start("approach")
            return
        try:
            self.calibration.require_unchanged()
            value = {
                key: positive(key, float(self.fields[key].get()))
                for key in ("approach_speed_deg_s", "approach_force_threshold_N")
            }
            self.command("approach", value)
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def move_to_position(self):
        """Use an absolute calibrated angle; the owner applies the existing motion limits."""
        try:
            if self.calibration is None:
                raise ValueError("请先加载有效校准")
            self.calibration.require_unchanged()
            target = finite_number("指定绝对位置 (°)", float(self.fields["position_target_deg"].get()))
            speed = positive("定位速度 (°/s)", float(self.fields["position_speed_deg_s"].get()))
            if self.running:
                validate_motion_request(self.config.teacher, self.config.safety, "position", target, speed)
                self.command("position", {"position_deg": target, "position_speed_deg_s": speed})
            else:
                self.start("position", target, speed)
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def calculate(self):
        try:
            self.estimate = tuning_estimate(*(float(var.get()) for var in self.calculator.values()))
            kappa, response, ki, kp = self.estimate
            self.calculation_label.set(
                f"κ={kappa:g} N/°；t90={response:g} s\nKi={ki:g} °/(N·s)；Kp={kp:g} °/N"
            )
        except ValueError as exc:
            self.estimate = None
            self.calculation_label.set(str(exc))

    def fill_estimate(self):
        if self.estimate is not None:
            self.fields["ki_deg_per_N_s"].set(f"{self.estimate[2]:.10g}")
            self.fields["kp_deg_per_N"].set(f"{self.estimate[3]:.10g}")
            self.status.set("已手动填入计算结果；需点击应用或开始才使用")

    def stop(self):
        self.next_run = None
        self.monitor_stop.set()
        if self.live is not None:
            self.live.stop.set()
        self.status.set("停止请求已发送；等待 owner 失能、退出及日志收据")

    def close(self):
        self.closing = True
        self.stop()

    def show_report(self, report):
        self.last_stop.set(report["summary"])
        self.last_stop_details = report["details"]

    def show_stop_details(self):
        window = tk.Toplevel(self.root)
        window.title("夹爪退出证据")
        text = tk.Text(window, wrap="word", width=100, height=28)
        text.pack(fill="both", expand=True)
        text.insert("1.0", self.last_stop_details)
        text.configure(state="disabled")

    def poll(self):
        if self.live is not None:
            self.live.receive()
            if self.live.startup is not None and self.live.startup != self.last_startup:
                self.last_startup = self.live.startup
                self.settings_label.set(startup_description(self.live.startup))
            applied = self.live.applied
            if isinstance(applied, dict) and applied.get("timestamp_ns") != self.last_applied:
                self.last_applied = applied.get("timestamp_ns")
                teacher = applied.get("teacher") or {}
                self.status.set(
                    f"已实际应用 Kp={teacher.get('kp_deg_per_N')}°/N，"
                    f"Ki={teacher.get('ki_deg_per_N_s')}°/(N·s)，目标={teacher.get('force_target_N')} N"
                )
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            if kind == "result":
                result, path = value
                self.exit_confirmed = exit_confirmation(result)
                if not self.exit_confirmed:
                    self.unconfirmed_port = exit_port(result) or self.unconfirmed_port
                self.show_report(stop_report(result, path))
                self.status.set(f"会话结束；日志 {path.resolve()}")
            elif kind == "error":
                self.last_stop.set(value)
                self.last_stop_details = json.dumps(
                    {"error": value, "readonly_cleanup": self.preview_cleanup}, ensure_ascii=False, indent=2
                )
                self.status.set(value)
            elif kind == "preview_cleanup":
                self.preview_cleanup = value
                if not value["transport_closed"] or not value["sensor_closed"]:
                    self.exit_confirmed = False
                    self.unconfirmed_port = self.monitor_port
                    self.status.set("只读资源退出未确认；资源仍保留，禁止新会话")
            elif kind == "done":
                self.running = False
                self.worker = None
        row = self.live.latest if self.running and self.live is not None else self.preview_latest
        if row is not None:
            self.recover_readonly_exit(row)
        if row and row.get("cleanup_pending"):
            error = row.get("cleanup_error") or "退出清理未确认，保留 owner 所有权"
            self.status.set(error)
            self.last_stop.set(error)
            self.last_stop_details = json.dumps(row, ensure_ascii=False, indent=2)
        if row:
            anchor = row.get("decision_anchor_ns")
            if anchor != self.last_anchor:
                self.points.append(row)
                self.last_anchor = anchor
            self.values_label.set(
                f"{row.get('phase')} | 原始左 / 右 / 平均力："
                f"{self.fmt(row.get('force_left_N'))} / {self.fmt(row.get('force_right_N'))} / {self.fmt(row.get('force_mean_N'))} N\n"
                f"目标 {self.fmt(row.get('force_target_requested_N'))} N | 下发 / 实际位置 "
                f"{self.fmt(row.get('q_sent_deg'))} / {self.fmt(row.get('q_measured_deg'))}°\n"
                f"恒速 {self.fmt(row.get('approach_speed_deg_s'))}°/s → "
                f"阈值 {self.fmt(row.get('approach_force_threshold_N'))} N | "
                f"指定 / 保持目标 {self.fmt(row.get('hold_target_deg'))}° | "
                f"定位速度 {self.fmt(row.get('hold_speed_deg_s'))}°/s\n"
                f"ΔI {self.fmt(row.get('delta_i_deg'))}° | ΔP {self.fmt(row.get('delta_p_deg'))}° | "
                f"原始 / 最终增量 {self.fmt(row.get('delta_raw_deg'))} / {self.fmt(row.get('delta_final_deg'))}°\n"
                f"目标领先 {self.fmt(row.get('command_lead_deg'))}° | 限制 {', '.join(row.get('clip_flags') or ()) or '无'}\n"
                f"{timing_description(row)}"
            )
        self.update_editability()
        self.update_timing_settings()
        self.confirm_check.configure(state="disabled" if self.running else "normal")
        self.update_optional()
        self.calibration_button.configure(state="disabled" if self.settings_busy() else "normal")
        self.start_button.configure(state="disabled" if self.running else "normal")
        self.read_button.configure(text="断开只读监测" if self.monitoring else "连接 / 只读监测")
        active = self.running and row and not row.get("cleanup_pending")
        self.freeze_button.configure(
            state="normal" if active and row.get("phase") in ("FORCE_TRACK", "APPROACH") else "disabled"
        )
        can_move = not self.running or (
            active and row.get("phase") in ("FORCE_TRACK", "APPROACH", "POSITION_HOLD")
        )
        self.approach_button.configure(state="normal" if can_move else "disabled")
        self.position_button.configure(state="normal" if can_move else "disabled")
        hold = active and row.get("phase") == "POSITION_HOLD"
        self.resume_button.configure(state="normal" if hold else "disabled")
        settled = (
            hold
            and row.get("hold_target_deg") is not None
            and row.get("q_measured_deg") is not None
            and row.get("q_sent_deg") is not None
            and abs(row["hold_target_deg"] - row["q_measured_deg"])
            <= self.config.safety.stall_position_tolerance_deg
            and abs(row["hold_target_deg"] - row["q_sent_deg"])
            <= self.config.safety.stall_position_tolerance_deg
        )
        self.step_button.configure(state="normal" if settled else "disabled")
        self.draw()
        if self.closing and not self.busy():
            self.root.destroy()
            return
        try:
            hz = positive("界面频率", float(self.fields["display_hz"].get()))
        except ValueError:
            hz = float(default_values()["display_hz"])
        self.root.after(max(1, round(1000 / hz)), self.poll)

    @staticmethod
    def fmt(value):
        return "—" if value is None else f"{value:.5g}"

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        try:
            window = positive("波形窗口", float(self.fields["plot_seconds"].get()))
        except ValueError:
            return
        now = time.perf_counter_ns()
        while self.points and now - self.points[0]["decision_anchor_ns"] > window * 1e9:
            self.points.popleft()
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 80 or height < 120:
            return
        groups = (
            (
                "原始力 / 目标 (N)",
                (
                    ("force_left_N", "左", "#0891b2"),
                    ("force_right_N", "右", "#ea580c"),
                    ("force_mean_N", "均值", "#7c3aed"),
                    ("force_target_requested_N", "目标", "#16a34a"),
                ),
            ),
            (
                "位置 (°)",
                (
                    ("q_sent_deg", "下发", "#16a34a"),
                    ("q_measured_deg", "反馈", "#2563eb"),
                    ("hold_target_deg", "固定目标", "#d97706"),
                ),
            ),
            (
                "控制增量 (°)",
                (
                    ("delta_i_deg", "I", "#ea580c"),
                    ("delta_p_deg", "P", "#2563eb"),
                    ("delta_raw_deg", "原始", "#7c3aed"),
                    ("delta_final_deg", "最终", "#16a34a"),
                ),
            ),
        )
        for index, (title, curves) in enumerate(groups):
            top, bottom = index * height / 3 + 35, (index + 1) * height / 3 - 18
            canvas.create_text(10, top - 20, text=title, anchor="w", fill="#334155")
            for j, (_, label, color) in enumerate(curves):
                canvas.create_text(180 + j * 63, top - 20, text=label, anchor="w", fill=color)
            data = [row[key] for row in self.points for key, _, _ in curves if row.get(key) is not None]
            low, high = (min(data), max(data)) if data else (0, 1)
            if low == high:
                high = low + 1
            canvas.create_text(4, top, text=f"{high:.4g}", anchor="nw", fill="#64748b")
            canvas.create_text(4, bottom, text=f"{low:.4g}", anchor="sw", fill="#64748b")
            canvas.create_line(88, bottom, width - 10, bottom, fill="#e2e8f0")
            for key, _, color in curves:
                segment = []
                for row in self.points:
                    if row.get(key) is None:
                        if len(segment) >= 4:
                            canvas.create_line(*segment, fill=color, width=1.5)
                        segment = []
                        continue
                    x = 88 + ((row["decision_anchor_ns"] - now) / 1e9 + window) / window * (width - 98)
                    y = bottom - (row[key] - low) / (high - low) * (bottom - top)
                    segment.extend((x, y))
                if len(segment) >= 4:
                    canvas.create_line(*segment, fill=color, width=1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Build and close the UI; no device I/O")
    parser.add_argument("--connect", action="store_true", help="Start query-only monitoring; never enable")
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_GRIPPER_CALIBRATION)
    args = parser.parse_args()
    window = tk.Tk()
    app = Tuner(window, calibration_path=args.calibration_file)
    if args.smoke_test:
        window.after(250, app.close)
    elif args.connect:
        window.after(200, app.read)
    window.mainloop()
