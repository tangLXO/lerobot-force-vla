"""Pure file/profile review helpers shared by Tk and its supervised process bridge."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from lerobot.robots.rebot_b601_follower.control_protocol import mit_gain_wire_codes


def config_document(document):
    """Read a direct profile or a preparation envelope with source annotations."""
    return document["config"] if isinstance(document.get("config"), dict) else document


def operator_teleoperation_summary(config):
    hardware = config.b601
    operator = getattr(hardware, "operator_teleoperation", None) if hardware else None
    if operator is None:
        return None
    from lerobot.robots.rebot_b601_follower.operator_teleoperation import OperatorTeleoperationConfig

    if not isinstance(operator, OperatorTeleoperationConfig):
        raise ValueError("正常全臂遥操必须使用 typed OperatorTeleoperationConfig")
    if hardware.startup_mode != "prepare_disabled_operator_teleoperation":
        raise ValueError("正常全臂遥操须显式配置 prepare_disabled_operator_teleoperation")
    return {
        "kind": "operator_teleoperation",
        "session_id": operator.session_id,
        "target_force_N": config.teacher.force_target_N,
        "reference_envelope_deg": operator.reference_envelope_deg,
        "alignment_tolerance_deg": operator.alignment_tolerance_deg,
        "preload_tolerance_deg": operator.preload_tolerance_deg,
        "leader_max_age_s": operator.leader_max_age_s,
        "max_command_gap_s": operator.max_command_gap_s,
        "watchdog_ticks": operator.watchdog_ticks,
        "duration_s": config.duration_s,
        "max_run_time_s": config.safety.max_run_time_s,
        "gripper_max_total_travel_deg": config.safety.max_total_travel_deg,
        "production_arm_authorized": False,
    }


def full_arm_profile_text(config):
    summary = operator_teleoperation_summary(config)
    if summary is None:
        return bounded_validation_text(config)
    control = config.b601.control
    descriptions = []
    for profile in control.profiles:
        if profile.mode == "mit":
            kp_code, kd_code = mit_gain_wire_codes(profile.kp, profile.kd)
            description = (
                f"{profile.name}:MIT Kp={profile.kp:g}, Kd={profile.kd:g}"
                f"（SDK 编码 Kp={kp_code * 500 / 4095:g}, Kd={kd_code * 5 / 4095:g}；"
                f"v参考={profile.mit_velocity_deg_s:g}°/s, τff={profile.mit_torque_Nm:g}N·m）"
            )
        elif profile.velocity_deg_s is not None:
            description = f"{profile.name}:速度 {profile.velocity_deg_s:g}°/s"
        else:
            description = f"{profile.name}:来源速度码 {profile.velocity_wire_code}（非°/s）"
        descriptions.append(description)
    profiles = "; ".join(descriptions)

    def optional_value(value, unit):
        return "默认关闭" if value is None else f"{value:g}{unit}"

    relative = "; ".join(f"{p.name}={optional_value(p.max_relative_deg, '°')}" for p in control.profiles)
    ranges = "; ".join(f"{p.name}=[{p.minimum_deg:g},{p.maximum_deg:g}]°" for p in control.profiles)
    return (
        f"正常全臂遥操；Teacher 来源目标 {summary['target_force_N']:g} N（可显式修改）；"
        f"六臂启动 q0±{summary['reference_envelope_deg']:g}°∩工作范围∩校准；对齐 {optional_value(summary['alignment_tolerance_deg'], '°')}；"
        f"预载 {optional_value(summary['preload_tolerance_deg'], '°')}；{profiles}；本拍目标相对反馈限制：{relative}；"
        f"返回/接管容差 {optional_value(config.runtime.handover_tolerance_deg, '°')}；"
        f"工作目标范围：{ranges}；"
        f"主臂年龄 {summary['leader_max_age_s']:g}s / 命令间隔 {optional_value(summary['max_command_gap_s'], 's')} / "
        f"TIMEOUT {summary['watchdog_ticks']}；来源启动/写预算 {control.startup_budget_s:g}/{control.send_budget_s:g}s；"
        f"全七轴故障清理预算 {control.fault_budget_s:g}s（不是运行时长）；"
        f"控制反馈最大年龄 {control.feedback_max_age_s:g}s；runtime 读取预算 {config.runtime.feedback_budget_ms:g}ms / "
        f"节拍截止 {config.runtime.tick_deadline_ms:g}ms；主臂读取预算 {config.leader_read_budget_s:g}s；"
        f"Safety force/joint 最大年龄 {config.safety.force_max_age_ms}/{config.safety.joint_feedback_max_age_ms}ms；"
        f"左右原始硬力阈值 {config.safety.hard_force_limit_left_N}/{config.safety.hard_force_limit_right_N}N；"
        f"可选时长 {summary['duration_s']} / "
        f"Safety 时长 {summary['max_run_time_s']}；夹爪累计行程 {summary['gripper_max_total_travel_deg']}；"
        "不继承 micro 约束；并非正式全臂验收。"
    )


def profile_fingerprint(path):
    """Pin the exact reviewed bytes and file identity without opening hardware."""
    path = Path(path).expanduser().resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError("全臂配置在读取期间变化；请重新预检并读取参考")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "mtime_ns": before.st_mtime_ns,
        "size_bytes": len(data),
    }


def require_profile_pin(path, expected):
    if not isinstance(expected, dict) or profile_fingerprint(path) != expected:
        raise ValueError("全臂配置与已预检文件不一致；请停止后重新预检并读取新参考")


def bounded_validation_summary(config):
    """Describe a typed short validation without claiming production acceptance."""
    hardware = config.b601
    validation = getattr(hardware, "validation", None) if hardware is not None else None
    if validation is None:
        return None
    from lerobot.robots.rebot_b601_follower.full_arm_validation import FullArmValidationConfig

    if not isinstance(validation, FullArmValidationConfig):
        raise ValueError("受限全臂验证必须使用 typed FullArmValidationConfig")
    if hardware.startup_mode != "prepare_disabled_full_arm_validation":
        raise ValueError("受限验证必须显式配置 prepare_disabled_full_arm_validation")
    if config.teacher.force_target_N != 1.0 or validation.target_force_N != 1.0:
        raise ValueError("本次受限联合验证须显式绑定 Teacher 目标力与 capability 均为 1 N")
    return {
        "kind": "bounded_full_arm_validation",
        "session_id": validation.session_id,
        "target_force_N": validation.target_force_N,
        "maximum_duration_s": validation.max_duration_s,
        "reference_envelope_deg": validation.reference_envelope_deg,
        "production_arm_authorized": False,
    }


def bounded_validation_text(config):
    summary = bounded_validation_summary(config)
    if summary is None:
        return ""
    return (
        f"受限联合验证 / 非正式全臂验收；显式目标 {summary['target_force_N']:g} N；"
        f"最长 {summary['maximum_duration_s']:g} s；参考包络 ±{summary['reference_envelope_deg']:g}°；"
        "普通生产全臂门禁仍关闭。"
    )


def validation_evidence_fingerprint(config):
    """Pin the complete admitted artifact closure, including event/source manifests."""
    if bounded_validation_summary(config) is None:
        return None
    hardware = config.b601
    admission = hardware.validation.validate_binding(
        hardware.joint_specs, hardware.control, port=hardware.port
    )
    identities = sorted(
        (asdict(identity) for identity in admission.file_identities), key=lambda row: row["path"]
    )
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    admission.require_unchanged()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "file_count": len(identities)}


def require_validation_evidence_pin(config, expected):
    current = validation_evidence_fingerprint(config)
    if current is not None and current != expected:
        raise ValueError("受限验证引用证据与预检不一致；请停止后重新预检并读取新参考")
