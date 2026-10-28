"""B601 FORCE_POS + I-P panel; raw force, explicit limits and one hardware owner."""

import argparse
import hashlib
import json
import math
import queue
import threading
import time
import tkinter as tk
import uuid
from collections import Counter, deque
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, ttk

import draccus
from b601_full_arm_recovery import query_disabled_arm
from b601_operator_panel import OperatorArmPage
from b601_profile_review import full_arm_profile_text, operator_teleoperation_summary, require_profile_pin
from b601_supervised_bench import run
from b601_tuner_process import ProcessLive, run_in_process
from b601_tuner_recording import TASK_RESULTS, RecordingPage, load_full_arm_profile
from build_operator_teleoperation_profile import build_operator_document

from lerobot.force_control.config import (
    finite_number,
    positive,
    validate_motion_request,
)
from lerobot.force_control.runtime import DEFAULT_FEEDBACK_BUDGET_MS, DEFAULT_TICK_DEADLINE_MS
from lerobot.force_control.trace import validate_trace
from lerobot.force_control.tuning import (  # noqa: F401 - existing panel helpers remain importable.
    FIELD_PAGES,
    FIELDS,
    GRAVITY_N_PER_KG,
    GRIPPER_DEVICE_FIELDS,
    LEGACY_TARGET_FIELDS,
    LIVE_FIELDS,
    MOTION_FIELDS,
    OPTIONAL_GROUPS,
    SETTINGS_FORMAT,
    SETTINGS_SCHEMA_VERSION,
    build_config,
    default_values,
    parse_values,
    read_settings,
    save_settings,
    sensor_config,
    settings_document,
)
from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.calibration import DEFAULT_GRIPPER_CALIBRATION, B601GripperCalibration
from lerobot.robots.rebot_b601_follower.control_protocol import force_pos_wire_codes
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig
from lerobot.sensors.x518 import X518Sensor

OUTPUTS = Path("outputs/force_teacher_tuning")
DEFAULT_SETTINGS = Path(__file__).resolve().parents[2] / "b601_capture_parameters.json"
_UNSTOPPED_PREVIEWS = []


def full_arm_result(result):
    config = result.get("config")
    runtime = config.get("runtime") if isinstance(config, dict) else None
    names = runtime.get("joint_names") if isinstance(runtime, dict) else None
    return isinstance(names, (list, tuple)) and len(names) > 1


def closed_transport_evidence(result):
    """Only an explicit final transport fact permits a subsequent read-only owner."""
    evidence = result.get("shutdown_evidence") if isinstance(result, dict) else None
    return isinstance(evidence, dict) and evidence.get("transport_closed") is True


def _query_only_audit(audit, next_sequence):
    """Validate one undropped audit segment, allowing an empty shutdown tail."""
    if (
        not isinstance(audit, dict)
        or audit.get("complete") is not True
        or not isinstance(audit.get("events"), list)
    ):
        return None
    queries = 0
    for event in audit["events"]:
        if (
            not isinstance(event, dict)
            or type(event.get("sequence")) is not int
            or event["sequence"] != next_sequence
        ):
            return None
        next_sequence += 1
        if event.get("kind") == "query_rx":
            continue
        if event.get("kind") != "query_attempt":
            return None
        queries += 1
        try:
            packet = bytes.fromhex(event.get("data_hex", ""))
        except (TypeError, ValueError):
            return None
        if (
            len(packet) != 30
            or packet[:4] != b"\x55\xaa\x1e\x03"
            or int.from_bytes(packet[13:17], "little") != 0x7FF
            or packet[23] not in (0x33, 0xCC)
        ):
            return None
    return next_sequence, queries


def _query_only_control_closure(result, log_path, shutdown_audit):
    """Bind drained control audits to this result's fixed sibling trace files, read-only."""
    try:
        result_path = Path(log_path).expanduser().resolve(strict=True)
        if not result_path.is_file():
            return False
        # The queue retains tuples while JSON stores lists. Compare the same
        # serialized representation before trusting a caller-provided path.
        recorded = json.loads(result_path.read_bytes())
        if recorded != json.loads(json.dumps(result, allow_nan=False)):
            return False
        stream = (result_path.parent / "stream").resolve(strict=True)
        if stream.parent != result_path.parent:
            return False
        manifest_path = (stream / "manifest.json").resolve(strict=True)
        control_path = (stream / "control.jsonl").resolve(strict=True)
        if manifest_path.parent != stream or control_path.parent != stream:
            return False
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict):
            return False
        run_uid = manifest.get("run_uid")
        if not isinstance(run_uid, str) or not run_uid:
            return False
        metadata = manifest.get("metadata")
        configuration = recorded.get("config")
        manifest_config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(configuration, dict) or not isinstance(manifest_config, dict):
            return False
        if any(
            manifest_config.get(key) != configuration.get(key) for key in ("b601", "runtime", "initial_mode")
        ):
            return False
        files = manifest.get("files")
        control = files.get("control.jsonl") if isinstance(files, dict) else None
        if (
            not isinstance(control, dict)
            or type(control.get("rows")) is not int
            or control["rows"] < 0
            or type(control.get("size_bytes")) is not int
            or control["size_bytes"] < 0
            or not isinstance(control.get("sha256"), str)
        ):
            return False
        digest = hashlib.sha256()
        rows = size = queries = 0
        next_sequence = 1
        before = control_path.stat()
        with control_path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                size += len(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if (
                    not isinstance(row, dict)
                    or row.get("run_uid") != run_uid
                    or row.get("startup_evidence") is not None
                    or row.get("send_status") != "none"
                    or row.get("send_start_ns") is not None
                    or row.get("send_end_ns") is not None
                    or row.get("q_sent_deg") is not None
                    or row.get("submitted_action") not in (None, {})
                ):
                    return False
                part = _query_only_audit(row.get("device_io_evidence"), next_sequence)
                if part is None:
                    return False
                next_sequence, count = part
                queries += count
                rows += 1
        after = control_path.stat()
        if (
            (before.st_size, before.st_mtime_ns, before.st_ino)
            != (after.st_size, after.st_mtime_ns, after.st_ino)
            or rows != control["rows"]
            or size != control["size_bytes"]
            or digest.hexdigest() != control["sha256"]
        ):
            return False
        tail = _query_only_audit(shutdown_audit, next_sequence)
        return tail is not None and queries + tail[1] > 0
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def query_only_connection_failure(result, log_path=None):
    """Permit retry after a closed, proven pre-start query failure; never claim motor-off."""
    if not isinstance(result, dict):
        return False
    evidence = result.get("shutdown_evidence") or {}
    if not isinstance(evidence, dict):
        return False
    disposition = evidence.get("fault_disposition") or {}
    audit = evidence.get("device_io_evidence") or {}
    if not isinstance(disposition, dict) or not isinstance(audit, dict):
        return False
    reason = disposition.get("reason")
    if (
        result.get("startup_evidence") is not None
        or evidence.get("transport_closed") is not True
        or reason not in ("owner_connection_failed", "invalid_joint_feedback")
        or disposition.get("status") != "not_attempted"
        or disposition.get("per_joint_results") != []
    ):
        return False
    # Older connection errors retained their complete audit in the result alone.
    # Feedback failures can drain it into control.jsonl, so their cause label is
    # never sufficient without a bound, complete trace closure.
    if reason == "owner_connection_failed":
        standalone = _query_only_audit(audit, 1)
        if standalone is not None and standalone[1] > 0:
            return True
    return log_path is not None and _query_only_control_closure(result, log_path, audit)


def exit_confirmation(result, log_path=None):
    """GUI restarts cannot erase an unconfirmed motor disposition."""
    if not isinstance(result, dict):
        return False
    if query_only_connection_failure(result, log_path=log_path):
        return True
    if "trial" in result:  # Historical evidence is read without reusing its control configuration.
        trial = result.get("trial") or {}
        if not isinstance(trial, dict):
            return False
        disposition = trial.get("disposition") or {}
        if not isinstance(disposition, dict):
            return False
        return (
            result.get("motor_transport_closed") is True
            and disposition.get("disabled_feedback_observed") is True
            and disposition.get("status") == "submitted"
            and disposition.get("error") is None
        )
    evidence = result.get("shutdown_evidence") or {}
    if not isinstance(evidence, dict):
        return False
    disposition = evidence.get("fault_disposition") or {}
    if not isinstance(disposition, dict):
        return False
    disabled = (
        disposition.get("all_joints_disabled_observed") is True
        if full_arm_result(result)
        else disposition.get("disabled_feedback_observed") is True
    )
    return (
        evidence.get("transport_closed") is True
        and disabled
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
    historical_errors = None
    legacy_audit_errors = {
        "B601 I/O event sequence/time is not contiguous and monotonic",
        "B601 I/O event has malformed kind or bytes",
    }
    original_errors = result.get("errors") or []
    config = result.get("config") or {}
    candidate = Path(path).parent / "stream"
    if (
        result.get("valid") is False
        and original_errors
        and isinstance(result.get("run_uid"), str)
        and bool(result["run_uid"])
        and isinstance(config, dict)
        and config.get("backend") == "b601"
        and set(original_errors) <= legacy_audit_errors
        and (candidate / "manifest.json").is_file()
    ):
        # Old results remain evidence. Recheck canonical files with today's
        # strict reader rather than showing a known historical verifier bug
        # forever or rewriting the original result/manifest.
        checked = validate_trace(candidate)
        if checked.get("run_uid") == result.get("run_uid"):
            historical_errors = dict(Counter(original_errors))
            result = {**result, "valid": checked["valid"], "errors": checked["errors"]}
    error_counts = Counter(result.get("errors") or [])
    errors = list(error_counts)
    reason = "; ".join(
        error if count == 1 else f"{error}（{count} 次）" for error, count in list(error_counts.items())[:32]
    ) or result.get("terminal_phase", result.get("terminal_state", "会话结束"))
    shutdown = result.get("shutdown_evidence")
    disposition = shutdown.get("fault_disposition") if isinstance(shutdown, dict) else None
    device_reason = disposition.get("reason") if isinstance(disposition, dict) else None
    startup = result.get("startup_evidence") or {}
    capture = result.get("capture_summary") or {}
    phases = capture.get("phase_counts", {}) if isinstance(capture, dict) else {}
    if (
        device_reason == "user_stop_during_startup"
        and isinstance(startup, dict)
        and startup.get("complete") is True
        and any(
            phases.get(phase, 0) > 0
            for phase in ("FOLLOW", "FORCE_TRACK", "FOLLOW_WAIT_CLEAR", "POSITION_HOLD")
        )
    ):
        device_reason = "user_stop（运行期间停止；历史原始标签为 user_stop_during_startup）"
    if isinstance(device_reason, str) and device_reason and device_reason not in reason:
        reason = f"{reason}；设备停止原因：{device_reason}"
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
    elif historical_errors is not None:
        reason = f"历史日志只读重验通过：{reason}"
    scope_label = "全臂七关节" if full_arm_result(result) else "夹爪"
    exit_text = (
        "仅连接检查失败，未提交使能或运动命令且串口已关闭；重新只读连接后可重试"
        if query_only_connection_failure(result, log_path=path)
        else f"已观察{scope_label}失能且串口关闭"
        if exit_confirmation(result, log_path=path)
        else "退出失能未确认，禁止新会话"
    )
    return {
        "summary": f"{reason}；{exit_text}",
        "details": json.dumps(
            {
                "errors": errors[:32],
                "error_counts": dict(list(error_counts.items())[:32]),
                "total_error_count": sum(error_counts.values()),
                "unique_errors_omitted": max(0, len(errors) - 32),
                "historical_error_counts": historical_errors,
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


def hybrid_description(row):
    """Show the committed boundary in the same calibrated coordinates as the follower."""
    state = row.get("hybrid_state")
    if state is None:
        return ""
    reference = row.get("contact_reference") or {}
    leader = (row.get("arm_intent") or {}).get("gripper.pos")
    boundary = reference.get("follower_gripper_deg")
    leader_text = "—" if leader is None else f"{leader:.5g}"
    boundary_text = "—" if boundary is None else f"{boundary:.5g}"
    return f"\nTeacher {state} | 主爪映射 {leader_text}° | 接触从爪参考 {boundary_text}°"


def position_range_description(config):
    """Keep commanded working limits distinct from measured-position hard protection."""
    if config is None:
        return ""
    safety = config.safety
    lower = safety.q_min_deg if safety.soft_q_min_deg is None else safety.soft_q_min_deg
    upper = safety.q_max_deg if safety.soft_q_max_deg is None else safety.soft_q_max_deg
    scope = "夹爪" if len(config.runtime.joint_names) > 1 else ""
    return (
        f"{scope}工作目标范围 [{lower:g}, {upper:g}]° | "
        f"校准硬保护范围 [{safety.q_min_deg:g}, {safety.q_max_deg:g}]°\n"
    )


def observed_control_hz(rows):
    """Use tick-id span, not GUI refresh count, for sparse telemetry views."""
    points = [r for r in rows if type(r.get("tick_id")) is int and type(r.get("tick_start_ns")) is int]
    if len(points) < 2:
        return None
    last = points[-1]
    points = [r for r in points if 0 <= last["tick_start_ns"] - r["tick_start_ns"] <= 2_000_000_000]
    if len(points) < 2:
        return None
    first = points[0]
    elapsed = (last["tick_start_ns"] - first["tick_start_ns"]) / 1e9
    ticks = last["tick_id"] - first["tick_id"]
    return ticks / elapsed if elapsed > 0 and ticks > 0 else None


def arm_tracking_description(row):
    """Show measured-minus-leader differences without claiming post-send arrival."""
    snapshot, intent = row.get("joint_snapshot") or {}, row.get("arm_intent") or {}
    values = []
    for joint in snapshot.get("joints", []):
        name = joint.get("name")
        measured, target = joint.get("position_deg"), intent.get(f"{name}.pos")
        if name != "gripper" and type(measured) in (int, float) and type(target) in (int, float):
            values.append(f"{name}: {measured - target:+.2f}°")
    return "反馈 − 最新主臂目标（含动态滞后）：\n" + "；".join(values) if values else "尚无七关节运行反馈"


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
    if receipt.get("kind") == "operator_teleoperation":
        return (
            f"正常全臂 owner 启动：{receipt.get('complete')} | 已验证使能反馈 "
            f"{len(receipt.get('enabled_joint_names') or [])} / 7；hardware_ack=unknown\n"
        )
    if receipt.get("kind") == "bounded_full_arm_validation":
        enabled = receipt.get("enabled_joint_names") or []
        return (
            f"受限全臂启动{'完成' if receipt.get('complete') else '未完成'}；非正式全臂验收 | "
            f"显式目标 {receipt.get('target_force_N')} N | 已验证使能反馈 {len(enabled)} / 7\n"
            "逐阶段命令与主从对齐差值见启动原始收据；production_arm_authorized=False，hardware_ack=unknown。"
        )
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

    def __init__(self, root, *, calibration_path=None, settings_path=None, default_full_arm_mit=False):
        self.default_full_arm_mit = default_full_arm_mit
        self.default_mit_profile_file = None
        self.root, self.worker = root, None
        self.running = self.monitoring = self.closing = False
        self.finishing = False
        self.exit_confirmed = True
        self.unconfirmed_port = None
        self.unconfirmed_full_arm = False
        self.unconfirmed_transport_closed = False
        self.unconfirmed_result_path = None
        self.unconfirmed_calibration = None
        self.recovery_evidence = None
        self.recovering_full_arm = False
        self.full_arm_recovery_session = None
        self.full_arm_recovery_max_age_s = None
        self.connection_recovery = None
        self.ui_errors = []
        self._last_ipc_errors = ()
        self._poll_after = None
        self.monitor_session = None
        self.monitor_started_ns = None
        self.monitor_port = None
        self.monitor_stop = threading.Event()
        self.messages = queue.SimpleQueue()
        self.preview_latest = self.live = self.next_run = self.calibration = None
        self.pending_cleanup = None
        self.preview_cleanup = None
        self.pending_start_error = None
        self.config = self.estimate = self.last_anchor = None
        self.last_applied = None
        self.last_startup = None
        self.last_recording = None
        self.recording_final = False
        self.calibration_path = Path(calibration_path or DEFAULT_GRIPPER_CALIBRATION)
        self.settings_path = None
        self.points, self.fixed_widgets, self.optional_widgets = deque(), [], []
        self.other_widgets = []
        self.fields, self.toggles = {}, {}
        root.title("B601 · 50 Hz 采集版 · MIT 六臂 + FORCE_POS / I-P 夹爪")
        root.geometry("1320x920")
        root.minsize(1000, 740)
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
        self.calibration_button = ttk.Button(bar, text="选择 / 重载校准", command=self.choose_calibration)
        self.calibration_button.pack(side="right")
        calibration_text = ttk.Label(bar, textvariable=self.calibration_label, wraplength=1050)
        calibration_text.pack(side="left", fill="x", expand=True)
        bar.bind(
            "<Configure>",
            lambda event: calibration_text.configure(
                wraplength=max(260, event.width - self.calibration_button.winfo_reqwidth() - 12)
            ),
        )
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
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        right.bind("<Configure>", lambda event: self.wrap_right_labels(right, event.width))
        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        pages, page_canvases = {}, {}
        for key, title in (
            ("gripper", "夹爪控制"),
            ("arm", "全臂遥操"),
            ("recording", "示范录制"),
            ("other", "其他设置"),
        ):
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
        self.arm_page = OperatorArmPage(
            pages["arm"], apply_target=self.apply_operator_target, is_busy=self.settings_busy
        )
        self.recording_page = RecordingPage(
            pages["recording"],
            self.recording_command,
            profile_parent=self.arm_page.profile_frame,
            before_check=self.arm_page.require_saved,
            source_selected=self.arm_page.load_source,
        )
        self.arm_page.reset_review = self.recording_page.reset_profile_review
        self.arm_page.on_saved = self.recording_page.profile_path.set
        self.fixed_widgets.extend(self.recording_page.configuration_widgets)
        self.fixed_widgets.extend(self.arm_page.configuration_widgets)
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
        for key in ("leader_max_age_s", "watchdog_ticks"):
            self.arm_page.operator_values[key] = self.fields[key]
            self.arm_page.operator_entries[key].configure(textvariable=self.fields[key])
            self.fields[key].trace_add("write", lambda *_: self.arm_page.mark_dirty())
        self.arm_page.relative_limits["gripper"] = self.fields["max_relative_target_deg"]
        self.arm_page.relative_entries["gripper"].configure(
            textvariable=self.fields["max_relative_target_deg"]
        )
        self.arm_page.target = self.fields["target_N"]
        self.arm_page.target_entry.configure(textvariable=self.fields["target_N"])
        self.fields["target_N"].trace_add("write", lambda *_: self.arm_page.mark_dirty(target_only=True))
        for key, field in (("duration", "max_run_time_s"), ("travel", "max_total_travel_deg")):
            setattr(self.arm_page, key, self.fields[field])
            setattr(self.arm_page, f"{key}_enabled", self.toggles[key])
            self.arm_page.limit_entries[key].configure(textvariable=self.fields[field])
            self.arm_page.limit_checks[key].configure(
                variable=self.toggles[key], command=self.update_shared_limits
            )
        ttk.Label(
            sections["force_pos"],
            text="固定协议：夹爪 ID 7 / feedback 0x17 / FORCE_POS=4；整数速度码 1–10000，每码 0.01 rad/s，直接按所填码发送。驱动比例步进 0.01%。底层速度是上限，外环位置变化率独立受限。串口计时等待按 1 ms 粒度设置，剩余预算不足 1 ms 时停止 I/O。X518 语义单位 N (9.80665 N/kg)。默认主从入口为六臂官方 MIT＋当前夹爪 FORCE_POS/I-P，无需选择草稿。",
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
        self.status = tk.StringVar(value="未连接；先取得新鲜只读反馈，再启动遥操。")
        self.action_status = tk.StringVar(value=self.status.get())
        self.status.trace_add("write", lambda *_: self.action_status.set(self.status.get()[:150]))
        actions = ttk.Frame(left)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.read_button = ttk.Button(actions, text="连接 / 只读监测", command=self.read)
        self.read_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="应用实时参数", command=self.apply).grid(row=0, column=1, sticky="ew")
        self.confirmed = tk.BooleanVar()
        if self.default_full_arm_mit:
            self.arm_page.supported = self.confirmed
            self.arm_page.support_widget.configure(variable=self.confirmed)
            self.arm_page.target = self.fields["target_N"]
            self.arm_page.target_entry.configure(textvariable=self.arm_page.target)
            self.arm_page.target.trace_add("write", lambda *_: self.arm_page.mark_dirty(target_only=True))
            for key in ("leader_max_age_s", "watchdog_ticks"):
                self.arm_page.operator_values[key] = self.fields[key]
                self.arm_page.operator_entries[key].configure(textvariable=self.fields[key])
        confirm = ttk.Checkbutton(
            actions, text="已确认机械设置、闭合方向和力通道，手已移开", variable=self.confirmed
        )
        confirm.grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        self.confirm_check = confirm
        self.start_button = ttk.Button(actions, text="开始 I-P 控制（使能）", command=self.start)
        self.start_button.grid(row=4, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(actions, text="停止 / 夹爪失能 [Esc]", command=self.stop)
        self.stop_button.grid(row=4, column=1, sticky="ew")
        self.freeze_button = ttk.Button(actions, text="冻结固定位置", command=lambda: self.command("freeze"))
        self.freeze_button.grid(row=5, column=0, sticky="ew", pady=6, padx=(0, 4))
        self.resume_button = ttk.Button(actions, text="恢复 I-P", command=lambda: self.command("resume"))
        self.resume_button.grid(row=5, column=1, sticky="ew")
        self.step_button = ttk.Button(actions, text="执行手动小步（仅固定位置到位后）", command=self.step)
        self.step_button.grid(row=6, column=0, columnspan=2, sticky="ew")
        self.approach_button = ttk.Button(actions, text="恒速闭合 → I-P", command=self.approach)
        self.approach_button.grid(row=7, column=0, sticky="ew", pady=6, padx=(0, 4))
        self.position_button = ttk.Button(actions, text="运动到指定位置", command=self.move_to_position)
        self.position_button.grid(row=7, column=1, sticky="ew")
        self.hybrid_button = ttk.Button(
            actions, text="启动主从夹爪遥操（使能）", command=lambda: self.start("hybrid")
        )
        self.hybrid_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.action_status_label = ttk.Label(
            actions, textvariable=self.action_status, wraplength=470, foreground="#9a4b00"
        )
        self.action_status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.release_button = ttk.Button(
            actions, text="释放并结束采集", command=self.release_and_finish, state="disabled"
        )
        self.release_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.full_arm_recovery_button = ttk.Button(
            actions,
            text="全臂只读复核（不使能）",
            command=self.recover_full_arm_readonly,
            state="disabled",
        )
        self.full_arm_recovery_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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
        status_text = ttk.Label(outer, textvariable=self.status, wraplength=1280)
        status_text.pack(fill="x")
        outer.bind("<Configure>", lambda event: status_text.configure(wraplength=max(300, event.width - 4)))
        latest = max(OUTPUTS.glob("*/run/result.json"), key=lambda p: p.stat().st_mtime, default=None)
        if latest is not None:
            try:
                result = json.loads(latest.read_text(encoding="utf-8"))
                self.show_report(stop_report(result, latest))
                self.exit_confirmed = exit_confirmation(result, log_path=latest)
                self.unconfirmed_transport_closed = closed_transport_evidence(result)
                self.unconfirmed_result_path = latest
                self.unconfirmed_calibration = result.get("calibration")
                if not self.exit_confirmed:
                    self.unconfirmed_port = exit_port(result)
                    self.unconfirmed_full_arm = full_arm_result(result)
                    self.status.set(
                        "历史全臂退出未确认；点击连接会自动只读复核七关节，不会使能或运动。"
                        if self.unconfirmed_full_arm
                        else "历史退出未确认；可连接只读监测，重新核对同一夹爪当前失能状态"
                    )
            except (ValueError, OSError) as exc:
                self.exit_confirmed = False
                self.last_stop.set(f"历史退出日志读取失败：{exc}")
        self.load_calibration()
        if settings_path is not None:
            try:
                self.load_parameters_file(settings_path)
            except (ValueError, OSError, RuntimeError) as exc:
                self.status.set(f"默认参数加载失败：{exc}")
        if not self.exit_confirmed and self.unconfirmed_full_arm:
            self.status.set(
                "历史全臂退出未确认；点击连接将自动复核七关节，完成后继续只读监测。"
                if self.unconfirmed_transport_closed
                else "历史全臂退出未确认，原串口关闭证据不足；请先等待旧 owner 停止并退出。"
            )
        self.update_optional()
        self.poll()

    def using_default_full_arm_profile(self):
        """Keep the normal entry automatic while allowing an explicit alternate profile."""
        return bool(
            getattr(self, "default_full_arm_mit", False)
            and self.default_mit_profile_file is not None
            and self.recording_page.full_arm_enabled.get()
            and self.recording_page.profile_path.get() == str(self.default_mit_profile_file)
        )

    def prepare_default_full_arm_profile(self):
        """Populate the normal MIT view from current saved gripper values without device I/O."""
        if self.calibration is None:
            raise ValueError("请先加载当前校准文件")
        document, _ = build_operator_document(
            self.values(),
            self.enabled(),
            self.calibration,
            arm_settings=getattr(self, "loaded_arm_settings", None),
        )
        folder = OUTPUTS / "default_mit_profiles"
        folder.mkdir(parents=True, exist_ok=True)
        path = (folder / f"{uuid.uuid4().hex}.json").resolve()
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.default_mit_profile_file = path
        self.recording_page.profile_path.set(str(path))
        self.recording_page.full_arm_enabled.set(True)
        self.arm_page.load_source(path)
        self.arm_page.use_current_form = True
        self.arm_page.restore_settings_values(getattr(self, "loaded_arm_settings", None))
        self.arm_page.source_label.set(
            f"默认六臂：官方 MIT 参数；夹爪来源：{self.settings_path or '当前面板'}。无需手动选择配置。"
        )
        self.arm_page.summary.set(
            "已自动载入官方 MIT 默认值；主从启动直接使用当前夹爪参数，无需选择或保存草稿。"
        )
        self.recording_page.profile_status.set("内置六臂 MIT＋当前夹爪 FORCE_POS/I-P；启动时自动离线校验。")

    def default_full_arm_config(self, output):
        """Build the session from the live form, keeping gripper settings in one source."""
        if self.calibration is None:
            raise ValueError("请先加载当前校准文件")
        document, _ = build_operator_document(
            self.values(), self.enabled(), self.calibration, supported_arm=self.confirmed.get(), output=output
        )
        self.arm_page.sync_editor()
        editor = self.arm_page.editor
        editor.source = document
        editor.target, editor.supported_arm = self.fields["target_N"].get(), self.confirmed.get()
        for key in ("leader_max_age_s", "watchdog_ticks"):
            editor.operator_values[key] = str(document["b601"]["operator_teleoperation"][key])
        # These remain owned by the current gripper form, including optional limits.
        value = editor.document()
        for key in ("teacher", "safety", "sensors", "duration_s"):
            value[key] = document[key]
        value["b601"]["control"]["profiles"][-1] = document["b601"]["control"]["profiles"][-1]
        return draccus.decode(ForceTeacherRunConfig, value)

    def values(self):
        return {key: var.get() for key, var in self.fields.items()}

    @staticmethod
    def wrap_right_labels(container, width):
        """Keep status text readable when the user narrows the control window."""
        children = list(container.winfo_children())
        while children:
            widget = children.pop()
            if isinstance(widget, ttk.Label):
                widget.configure(wraplength=max(240, width - 24))
            children.extend(widget.winfo_children())

    def enabled(self):
        return {key: var.get() for key, var in self.toggles.items()}

    def update_shared_limits(self):
        """Keep optional controls on both tabs in sync before a new session."""
        self.arm_page.mark_dirty()
        self.update_optional()

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
        if hasattr(self, "arm_page"):
            self.arm_page.mode_widget.configure(state="disabled" if busy else "readonly")
            if self.arm_page.use_current_form:
                self.arm_page.gripper_velocity_entry.configure(state="disabled")
        for widget in self.other_widgets:
            widget.configure(state="normal" if self.other_unlocked.get() and not busy else "disabled")
        self.other_unlock_check.configure(state="disabled" if busy else "normal")
        self.update_optional()

    def save_parameters_file(self, path):
        if self.settings_busy():
            raise RuntimeError("请停止并断开后保存参数")
        arm_page = getattr(self, "arm_page", None)
        arm_settings = (
            arm_page.settings_values() if arm_page is not None else getattr(self, "loaded_arm_settings", None)
        )
        self.settings_path = save_settings(
            path, self.values(), self.enabled(), self.calibration_path, arm_settings=arm_settings
        )
        self.loaded_arm_settings = arm_settings
        self.profile_label.set(f"已保存夹爪及全臂参数：{self.settings_path}（下次启动前完整校验）")

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
        arm_settings = []
        values, enabled, calibration_path = read_settings(
            path, on_unit_conversion=unit_notices.append, on_arm_settings=arm_settings.append
        )
        calibration = B601GripperCalibration.load(calibration_path)
        for key, value in values.items():
            self.fields[key].set(value)
        for key, value in enabled.items():
            self.toggles[key].set(value)
        self.calibration_path, self.calibration = calibration_path, calibration
        self.loaded_arm_settings = arm_settings[0]
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
        if getattr(self, "default_full_arm_mit", False):
            self.prepare_default_full_arm_profile()

    def save_parameters(self):
        if self.settings_busy():
            self.status.set("请停止并断开后保存参数")
            return
        path = filedialog.asksaveasfilename(
            title="保存 I-P 参数草稿",
            defaultextension=".json",
            filetypes=[("I-P 参数 JSON", "*.json")],
            initialfile=self.settings_path.name if self.settings_path else "b601_ip_parameters.json",
            initialdir=str(self.settings_path.parent if self.settings_path else Path.cwd()),
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
            f"校准硬保护范围 [{c.minimum_deg:g}, {c.maximum_deg:g}]° | SHA256 {c.sha256[:12]} | {c.path}"
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
                if self.load_calibration() and getattr(self, "default_full_arm_mit", False):
                    try:
                        self.prepare_default_full_arm_profile()
                    except ValueError as exc:
                        self.status.set(str(exc))

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def settings_busy(self):
        return (
            self.busy()
            or self.running
            or self.monitoring
            or getattr(self, "recovering_full_arm", False)
            or self.next_run is not None
        )

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
            self.status.set("控制会话仍在运行；请先停止，等待 owner 退出后再只读连接。")
            return
        if getattr(self, "connection_recovery", None) is not None or getattr(
            self, "recovering_full_arm", False
        ):
            self.stop()
            self.status.set("只读复核已取消；等待当前查询结束并关闭串口。")
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
        if not self.exit_confirmed and getattr(self, "unconfirmed_full_arm", False):
            self.recover_full_arm_readonly(continue_monitor=True)
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
        self.pending_start_error = None
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
                    lambda evidence: self.messages.put(
                        ("preview_cleanup", dict(evidence, monitor_session=token))
                    ),
                )
            finally:
                self.monitoring = False
            return None

        self.launch(preview)

    def dispatch_pending_start(self):
        """Start once after the preview worker exits and its resources are confirmed closed."""
        if self.worker is not None or self.monitoring:
            return
        if self.next_run is None and self.pending_cleanup is None:
            return
        cleanup = self.preview_cleanup or {}
        closed = (
            cleanup.get("monitor_session") == self.monitor_session
            and cleanup.get("transport_closed") is True
            and cleanup.get("sensor_closed") is True
        )
        stopped = self.next_run is None or self.live is None or self.live.stop.is_set()
        error = getattr(self, "pending_start_error", None)
        if closed and not error and not stopped and not _UNSTOPPED_PREVIEWS:
            execute, self.next_run = self.next_run, None
            self.running = True
            self.recording_final = False
            self.status.set("只读资源已关闭，正在启动遥操控制进程…")
            self.launch(execute)
            return
        self.next_run = None
        cancel, self.pending_cleanup = self.pending_cleanup, None
        try:
            if cancel is not None:
                cancel()
        except Exception as exc:
            error = error or f"取消通信清理失败：{type(exc).__name__}: {exc}"
        finally:
            self.running = False
            self.exit_confirmed = closed and not _UNSTOPPED_PREVIEWS
        self.status.set(
            f"遥操未启动：{error}"
            if error
            else "遥操未启动：只读资源退出未确认，请查看退出证据。"
            if not closed or _UNSTOPPED_PREVIEWS
            else "遥操启动已取消；控制进程未运行。"
        )

    def full_arm_recovery_ready(self):
        """Only a finished owner and recorded port closure permit another read-only owner."""
        return bool(
            not self.exit_confirmed
            and self.unconfirmed_full_arm
            and getattr(self, "unconfirmed_transport_closed", False)
            and self.worker is None
            and not self.running
            and not self.monitoring
            and not getattr(self, "recovering_full_arm", False)
            and self.next_run is None
            and self.pending_cleanup is None
            and not _UNSTOPPED_PREVIEWS
        )

    def connection_source(self):
        """Pin the selected form and calibration through the two read-only owners."""
        self.calibration.require_unchanged()
        return {
            "values": self.values(),
            "enabled": self.enabled(),
            "calibration": asdict(self.calibration),
            "settings_path": str(self.settings_path),
            "profile_path": self.recording_page.profile_path.get(),
            "full_arm_enabled": self.recording_page.full_arm_enabled.get(),
        }

    def dispatch_connection_recovery(self):
        """Continue ordinary monitoring only after the probe worker has really finished."""
        request = self.connection_recovery
        if request is None:
            return
        try:
            unchanged = self.connection_source() == request["source"]
        except (ValueError, TypeError, OSError, AttributeError):
            unchanged = False
        if not unchanged:
            self.connection_recovery = None
            self.monitor_stop.set()
            self.status.set("连接来源或校准已变化，自动续接已取消；请按当前来源重新连接。")
            return
        if not request["worker_done"] or self.worker is not None or self.monitoring:
            return
        self.connection_recovery = None
        if self.closing or self.monitor_stop.is_set():
            self.status.set("连接复核后的自动续接已取消；不会启动控制。")
            return
        if (
            not self.exit_confirmed
            or self.unconfirmed_full_arm
            or (self.recovery_evidence or {}).get("session_id") != request["session_id"]
        ):
            return
        self.read()

    def recover_full_arm_readonly(self, *, continue_monitor=False):
        """Run one explicit seven-axis observation, never a disable or reconfiguration."""
        if not self.full_arm_recovery_ready():
            self.status.set("全臂只读复核未开始：须先等待原 owner 退出并确认原串口已关闭。")
            return
        try:
            if self.calibration is None:
                raise ValueError("请加载与失败会话相同的校准文件")
            self.calibration.require_unchanged()
            if asdict(self.calibration) != self.unconfirmed_calibration:
                raise ValueError("只读复核校准与失败会话不一致，请加载原校准文件")
            port = self.fields["port"].get().strip()
            if (
                not isinstance(self.unconfirmed_port, str)
                or port.casefold() != self.unconfirmed_port.casefold()
            ):
                raise ValueError("只读复核串口须与退出未确认的全臂会话一致")
            calibration = B601ArmCalibration.load(
                self.calibration.path, {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
            )
            values = {
                "baudrate": int(self.fields["baudrate"].get()),
                "mapping_budget_s": positive("映射查询预算", float(self.fields["startup_budget_s"].get())),
                "feedback_budget_s": positive(
                    "只读反馈预算", float(self.fields["monitor_feedback_budget_ms"].get()) / 1000
                ),
                "max_age_s": positive(
                    "只读参考最大年龄", float(self.fields["start_feedback_max_age_ms"].get()) / 1000
                ),
            }
        except (ValueError, TypeError, OSError) as exc:
            self.status.set(f"全臂只读复核未开始：{exc}")
            return
        token = self.full_arm_recovery_session = str(uuid.uuid4())
        self.full_arm_recovery_max_age_s = values["max_age_s"]
        if continue_monitor:
            try:
                self.connection_recovery = {
                    "session_id": token,
                    "source": self.connection_source(),
                    "worker_done": False,
                }
            except (ValueError, TypeError, OSError) as exc:
                self.status.set(f"连接复核未开始：{exc}")
                return
        self.recovering_full_arm = True
        self.monitor_stop.clear()
        self.confirmed.set(False)
        self.status.set("正在只读查询七关节；不会发送使能、失能、位置或参数写入。")

        def inspect():
            receipt = query_disabled_arm(
                port,
                calibration,
                **values,
                stop=self.monitor_stop,
                session_id=token,
                transport_factory=ReadonlyB601Transport,
                retain=lambda device: _UNSTOPPED_PREVIEWS.append((device, None, None)),
            )
            self.messages.put(("full_arm_recovery", receipt))
            return None

        try:
            self.launch(inspect)
        except Exception as exc:
            self.connection_recovery = None
            self.recovering_full_arm = False
            if not self.busy():
                self.worker = None
            self.status.set(f"全臂只读复核未开始：{type(exc).__name__}: {exc}")

    def accept_full_arm_recovery(self, receipt):
        """Save new observed facts before unlocking; preserve the original unknown result."""
        if not isinstance(receipt, dict) or receipt.get("session_id") != self.full_arm_recovery_session:
            return
        self.recovering_full_arm = False
        self.unconfirmed_transport_closed = receipt.get("transport_closed") is True
        if (
            receipt.get("complete") is not True
            or receipt.get("all_joints_disabled_observed") is not True
            or receipt.get("transport_closed") is not True
            or receipt.get("errors")
            or receipt.get("read_only") is not True
            or receipt.get("motion_attempted") is not False
            or receipt.get("parameter_writes") is not False
        ):
            self.connection_recovery = None
            message = "; ".join(receipt.get("errors") or []) or "七轴反馈或关闭证据不完整"
            self.status.set(f"全臂只读复核未通过：{message}")
            self.last_stop_details = json.dumps(
                {"previous_exit_details": self.last_stop_details, "failed_read_only_recovery": receipt},
                ensure_ascii=False,
                indent=2,
            )
            return
        try:
            self.calibration.require_unchanged()
            full_calibration = B601ArmCalibration.load(
                self.calibration.path, {spec.name: spec.motor_id for spec in DEFAULT_B601_JOINTS}
            )
            if (
                receipt.get("port", "").casefold() != self.unconfirmed_port.casefold()
                or receipt.get("calibration") != asdict(full_calibration)
                or asdict(self.calibration) != self.unconfirmed_calibration
            ):
                raise ValueError("复核会话的端口或校准绑定已变化")
            rows = (receipt.get("feedback") or {}).get("feedback")
            expected = {spec.name: spec for spec in DEFAULT_B601_JOINTS}
            if (
                not isinstance(rows, (tuple, list))
                or len(rows) != 7
                or {row["name"] for row in rows} != set(expected)
            ):
                raise ValueError("复核收据缺少七个不同关节的反馈")
            started, observed, closed = (
                receipt.get(key) for key in ("started_ns", "observed_ns", "closed_ns")
            )
            if (
                any(type(value) is not int for value in (started, observed, closed))
                or not started <= observed <= closed
            ):
                raise ValueError("复核收据时间边界不完整")
            for row in rows:
                spec = expected[row["name"]]
                stamp = row.get("receive_timestamp_ns")
                if (
                    row.get("motor_id") != spec.motor_id
                    or row.get("feedback_id") != spec.feedback_id
                    or row.get("status_code") != 0
                    or type(stamp) is not int
                    or not started <= stamp <= observed
                    or observed - stamp > self.full_arm_recovery_max_age_s * 1e9
                ):
                    raise ValueError(f"复核收据的关节反馈未确认：{row['name']}")
            folder = OUTPUTS / "readonly_recovery"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self.full_arm_recovery_session}-full-arm.json"
            evidence = dict(receipt, previous_exit_log=str(self.unconfirmed_result_path))
            previous = self.unconfirmed_result_path
            if previous is not None and Path(previous).is_file():
                evidence["previous_exit_sha256"] = hashlib.sha256(Path(previous).read_bytes()).hexdigest()
            with path.open("x", encoding="utf-8") as handle:
                json.dump(evidence, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except (ValueError, TypeError, OSError, AttributeError, KeyError) as exc:
            self.connection_recovery = None
            self.status.set(f"全臂只读复核收据未保存，仍未解锁：{exc}")
            return
        self.recovery_evidence = evidence
        self.exit_confirmed = True
        self.unconfirmed_full_arm = False
        self.preview_latest = None
        self.confirmed.set(False)
        self.last_stop.set(f"新复核已观察七关节 DISABLED 且串口关闭；历史：{self.last_stop.get()}")
        self.last_stop_details = json.dumps(
            {
                "previous_exit_details": self.last_stop_details,
                "read_only_recovery": evidence,
                "recovery_receipt": str(path),
            },
            ensure_ascii=False,
            indent=2,
        )
        self.status.set(
            "七关节只读复核通过且串口已关闭；等待复核 worker 退出后自动继续只读监测。"
            if self.connection_recovery is not None
            else "七关节已只读观察到 DISABLED 且串口已关闭；请重新连接只读反馈并确认后再启动。"
        )

    def recover_readonly_exit(self, row):
        if getattr(self, "unconfirmed_full_arm", False):
            return False
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
                if self.unconfirmed_full_arm:
                    if self.connection_recovery is not None or self.recovering_full_arm:
                        self.status.set(
                            "正在只读复核七关节；尚未启动控制，复核后请确认新鲜反馈并再次点击启动。"
                        )
                    else:
                        self.read()
                    return
                raise ValueError("上次夹爪退出未确认；请先连接只读监测核对失能状态，再确认启动。")
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
            default_mit = self.using_default_full_arm_profile()
            session_options = (
                self.recording_page.session_options(include_full_arm_profile=False)
                if default_mit
                else self.recording_page.session_options()
            )
            if default_mit and mode == "hybrid":
                config = self.default_full_arm_config(folder / "run/stream")
                session_options["default_full_arm_mit"] = True
                if (
                    not isinstance(self.monitor_port, str)
                    or self.monitor_port.casefold() != config.b601.port.casefold()
                ):
                    raise ValueError("默认全臂端口与本次只读参考不一致；请按当前参数重新连接并取得参考")
            elif "full_arm_profile_file" in session_options:
                if mode != "hybrid":
                    raise ValueError("完整全臂配置须通过 Teacher 主从启动")
                require_profile_pin(
                    session_options["full_arm_profile_file"], session_options["full_arm_profile_pin"]
                )
                config = replace(
                    load_full_arm_profile(session_options["full_arm_profile_file"]),
                    trace_root=folder / "run/stream",
                )
                if (
                    not isinstance(self.monitor_port, str)
                    or self.monitor_port.casefold() != config.b601.port.casefold()
                ):
                    raise ValueError("全臂配置端口与本次只读参考不一致；请停止后按该配置重新连接并取得参考")
            else:
                config = build_config(
                    self.values(),
                    self.enabled(),
                    self.calibration,
                    folder / "run/stream",
                    initial_mode="hybrid" if mode == "hybrid" else "force",
                )
                config = replace(
                    config,
                    initial_mode=mode,
                    initial_position_deg=position_deg,
                    initial_position_speed_deg_s=position_speed_deg_s,
                )
            config.validate()
            validation_notice = full_arm_profile_text(config)
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
                **session_options,
            }
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            reviewed = run(folder / "run", reviewed_config=path)
            self.settings_label.set(
                position_range_description(config)
                + f"FORCE_POS=4 | 看门狗 {config.b601.control.watchdog_ticks * 0.05:g} ms | "
                f"请求速度码 {next(p for p in config.b601.control.profiles if p.name == 'gripper').velocity_wire_code} → "
                f"码 {reviewed['velocity_wire_code']} / 编码有效 {math.degrees(reviewed['velocity_wire_code'] * 0.01):g}°/s | "
                f"请求驱动 {next(p for p in config.b601.control.profiles if p.name == 'gripper').torque_ratio * 100:g}% → "
                f"码 {reviewed['drive_wire_code']} / 编码有效 {reviewed['drive_wire_code'] / 100:g}%"
                "\n设备映射将在使能前读回；速度字段和驱动比例是命令设置。"
                + (f"\n{validation_notice}" if validation_notice else "")
            )
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.status.set(str(exc))
            return
        self.config = config
        self.finishing = False
        self.last_startup = None
        self.last_recording = None
        self.recording_final = False
        try:
            self.live = ProcessLive(poll_s=float(self.fields["ipc_poll_ms"].get()) / 1000)
        except (ValueError, OSError, RuntimeError) as exc:
            draft.update(executable=False, consumed=True)
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.status.set(f"遥操未启动：控制进程通信初始化失败：{exc}")
            return
        self.running, self.exit_confirmed = True, False
        self.unconfirmed_transport_closed = False
        self.unconfirmed_calibration = asdict(self.calibration)
        self._last_ipc_errors = ()
        self.unconfirmed_full_arm = tuple(config.runtime.joint_names) != ("gripper",)
        self.update_timing_settings()
        self.unconfirmed_port = config.b601.port
        self.points.clear()
        self.status.set(
            f"受限全臂诊断（非正式全臂验收）；目标 {config.teacher.force_target_N:g} N；正在交接只读资源…"
            if getattr(config.b601, "validation", None) is not None
            else "正常全臂遥操：正在交接只读资源；随后读取七关节参考并准备 MIT 遥操。"
            if validation_notice
            else "交接只读资源；控制进程将读取新参考后准备并使能夹爪…"
        )

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
            cleanup = self.preview_cleanup or {}
            self.exit_confirmed = (
                cleanup.get("monitor_session") == self.monitor_session
                and cleanup.get("transport_closed") is True
                and cleanup.get("sensor_closed") is True
                and not _UNSTOPPED_PREVIEWS
            )
            self.unconfirmed_full_arm = False

        self.pending_cleanup = cancel_pending
        self.next_run = execute
        self.monitor_stop.set()

    def apply(self):
        if self.finishing:
            self.status.set("正在释放并结束采集；参数不再修改，Esc 可立即停止")
            return
        if self.running and tuple(self.config.runtime.joint_names) != ("gripper",):
            self.status.set("全臂会话使用已预检完整配置的 Teacher / Safety 参数；请停止后修改配置并重新预检")
            return
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
        if self.finishing:
            self.status.set("正在释放并结束采集；Esc 可立即停止")
            return
        if self.running and self.live is not None:
            try:
                self.live.submit(kind, value)
                self.status.set(f"已发送 {kind} 请求，等待 owner 收据")
            except (ValueError, RuntimeError, queue.Full) as exc:
                self.status.set(str(exc) or "上一操作尚未接收")

    def apply_operator_target(self):
        """Only the explicit force target may change live; hardware profiles remain pinned."""
        if not self.running or self.live is None or operator_teleoperation_summary(self.config) is None:
            self.status.set("仅运行中的正常 operator 会话可单独应用目标力")
            return
        try:
            target = finite_number("operator 目标力", float(self.arm_page.target.get()))
            if not self.config.teacher.target_min_N <= target <= self.config.teacher.target_max_N:
                raise ValueError("目标力须在本次已配置 Teacher target_min_N / target_max_N 内")
            self.command("target", target)
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc))

    def release_and_finish(self):
        """Request the configured opening trajectory, then save after the owner disables."""
        if self.finishing:
            self.status.set("已请求释放并结束采集；Esc 可立即停止")
            return
        row = self.live.latest if self.running and self.live is not None else None
        if (
            not row
            or row.get("cleanup_pending")
            or row.get("phase")
            not in ("FORCE_TRACK", "POSITION_HOLD", "APPROACH", "FOLLOW", "FOLLOW_WAIT_CLEAR")
        ):
            self.status.set("仅运行中的夹爪可正常释放；停止 / Esc 始终可用")
            return
        try:
            self.live.submit("release_and_finish")
        except (ValueError, RuntimeError, queue.Full) as exc:
            self.status.set(str(exc) or "上一操作尚未接收")
            return
        self.finishing = True
        self.release_button.configure(state="disabled")
        self.status.set(
            f"已请求释放：行程 {self.config.teacher.release_travel_deg:g}°，"
            f"速度 {self.config.teacher.release_speed_deg_s:g}°/s，"
            f"最长 {self.config.teacher.max_release_time_s:g}s；等待 owner 失能并保存，Esc 可立即停止"
        )

    def recording_command(self, kind):
        """Episode buttons only submit capture requests; enabling stays explicit."""
        if not self.running or self.live is None or self.finishing or self.live.stop.is_set():
            self.status.set("请先启动并确认控制会话；开始录制不会使能机械臂")
            return
        try:
            if kind == "record_start":
                value = {"config": self.recording_page.config()}
            elif kind == "record_discard":
                value = {"rerecord": True}
            elif kind in ("record_finish", "record_outcome"):
                value = {"task_result": TASK_RESULTS[self.recording_page.task_result.get()]}
            else:
                raise ValueError("未知录制操作")
            self.live.submit(kind, value)
            self.status.set("录制请求已发送；等待保存与完整性核验收据，控制会话继续")
        except (ValueError, TypeError, OSError, RuntimeError, queue.Full, KeyError) as exc:
            self.status.set(str(exc) or "上一录制操作尚未接收")

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
                for key in ("gripper_follow_speed_deg_s", "contact_force_threshold_N")
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
        self.connection_recovery = None
        self.next_run = None
        self.monitor_stop.set()
        if self.live is not None:
            try:
                self.live.stop.set()
            except Exception as exc:
                self.status.set(f"停止信号通信失败，退出仍未确认：{type(exc).__name__}: {exc}")
                return
        self.status.set(
            "只读停止 / 断开请求已发送；不会发送运动或失能命令。"
            if getattr(self, "recovering_full_arm", False)
            else "停止请求已发送；等待 owner 失能、退出及日志收据"
        )

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

    def _poll_live(self):
        if self.live is not None:
            self.live.receive()
            errors = tuple(getattr(self.live, "ipc_errors", ()))
            if errors and errors != self._last_ipc_errors:
                self._last_ipc_errors = errors
                self.status.set(f"控制通信已中断，已请求停止；等待 owner 退出证据：{errors[-1]}")
            incoming_recording = getattr(self.live, "recording", None)
            if isinstance(incoming_recording, dict) and not self.recording_final:
                self.last_recording = incoming_recording
            if self.live.startup is not None and self.live.startup != self.last_startup:
                self.last_startup = self.live.startup
                self.settings_label.set(
                    position_range_description(self.config)
                    + startup_description(self.live.startup)
                    + full_arm_profile_text(self.config)
                )
            applied = self.live.applied
            if isinstance(applied, dict) and applied.get("timestamp_ns") != self.last_applied:
                self.last_applied = applied.get("timestamp_ns")
                teacher = applied.get("teacher") or {}
                self.status.set(
                    f"已实际应用 Kp={teacher.get('kp_deg_per_N')}°/N，"
                    f"Ki={teacher.get('ki_deg_per_N_s')}°/(N·s)，目标={teacher.get('force_target_N')} N"
                )

    def _drain_messages(self):
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_message(kind, value)
            except Exception as exc:
                self._report_ui_error(f"message:{kind}", exc)

    def _handle_message(self, kind, value):
        if kind == "result":
            result, path = value
            if isinstance(result.get("recording"), dict):
                self.last_recording = result["recording"]
            self.recording_final = True
            self.exit_confirmed = exit_confirmation(result, log_path=path)
            self.unconfirmed_transport_closed = closed_transport_evidence(result)
            self.unconfirmed_result_path = path
            self.unconfirmed_calibration = result.get("calibration")
            self.unconfirmed_full_arm = full_arm_result(result) and not self.exit_confirmed
            if not self.exit_confirmed:
                self.unconfirmed_port = exit_port(result) or self.unconfirmed_port
            self.show_report(stop_report(result, path))
            errors = result.get("errors") or []
            self.status.set(
                f"遥操启动失败：{errors[0]}"
                if errors and not (result.get("startup_evidence") or {}).get("complete")
                else f"会话停止：{errors[0]}"
                if errors
                else f"会话结束；日志 {path.resolve()}"
            )
        elif kind == "error":
            if self.next_run is not None:
                self.pending_start_error = value
            self.recording_final = True
            self.last_stop.set(value)
            self.last_stop_details = json.dumps(
                {"error": value, "readonly_cleanup": self.preview_cleanup}, ensure_ascii=False, indent=2
            )
            self.status.set(value)
        elif kind == "preview_cleanup":
            if value.get("monitor_session") not in (None, self.monitor_session):
                return
            self.preview_cleanup = value
            if not value["transport_closed"] or not value["sensor_closed"]:
                self.exit_confirmed = False
                self.unconfirmed_port = self.monitor_port
                self.status.set("只读资源退出未确认；资源仍保留，禁止新会话")
        elif kind == "done":
            self.running = False
            self.finishing = False
            self.worker = None
            self.recovering_full_arm = False
            self.recording_final = True
            if self.connection_recovery is not None:
                self.connection_recovery["worker_done"] = True
            if self.live is not None and getattr(self.live, "closed", False):
                self.live = None
        elif kind == "full_arm_recovery":
            self.accept_full_arm_recovery(value)

    def _report_ui_error(self, operation, error):
        message = f"界面 {operation} 错误：{type(error).__name__}: {error}"
        if not self.ui_errors or self.ui_errors[-1] != message:
            self.ui_errors.append(message)
            del self.ui_errors[:-32]
            self.status.set(message + "；停止、只读复核和退出仍可操作。")
        if self.running and self.live is not None:
            with suppress(Exception):
                self.live.stop.set()

    def _settle_finished_worker(self):
        if self.worker is not None and not self.busy() and not self.monitoring:
            self.worker = None
            if self.running and self.next_run is None:
                self.running = self.finishing = False
                self.status.set("控制 worker 已退出；运动状态仍按退出收据核对。")
            self.recovering_full_arm = False

    def poll(self):
        """Keep worker completion and essential controls alive after IPC or rendering errors."""
        if self._poll_after is not None:
            with suppress(tk.TclError):
                self.root.after_cancel(self._poll_after)
            self._poll_after = None
        try:
            for operation in (
                self._poll_live,
                self._drain_messages,
                self._settle_finished_worker,
                self.dispatch_connection_recovery,
                self.dispatch_pending_start,
                self._refresh_widgets,
            ):
                try:
                    operation()
                except Exception as exc:
                    self._report_ui_error(operation.__name__, exc)
        finally:
            self._schedule_poll()

    def _schedule_poll(self):
        try:
            self.stop_button.configure(state="normal")
            self.read_button.configure(state="normal")
            self.full_arm_recovery_button.configure(
                state="normal" if self.full_arm_recovery_ready() else "disabled"
            )
            if self.closing and not self.busy() and not self.monitoring:
                self.root.destroy()
                return
            hz = positive("界面频率", float(self.fields["display_hz"].get()))
        except (ValueError, TypeError):
            hz = float(default_values()["display_hz"])
        except tk.TclError:
            return
        with suppress(tk.TclError):
            self._poll_after = self.root.after(max(1, round(1000 / hz)), self.poll)

    def _refresh_widgets(self):
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
                f"跟随 / 接近上限 {self.fmt(row.get('gripper_follow_speed_deg_s'))}°/s → "
                f"阈值 {self.fmt(row.get('contact_force_threshold_N'))} N | "
                f"指定 / 保持目标 {self.fmt(row.get('hold_target_deg'))}° | "
                f"定位速度 {self.fmt(row.get('hold_speed_deg_s'))}°/s\n"
                f"ΔI {self.fmt(row.get('delta_i_deg'))}° | ΔP {self.fmt(row.get('delta_p_deg'))}° | "
                f"接触回退 {self.fmt(row.get('delta_contact_deg'))}° | "
                f"原始 / 最终增量 {self.fmt(row.get('delta_raw_deg'))} / {self.fmt(row.get('delta_final_deg'))}°\n"
                f"目标领先 {self.fmt(row.get('command_lead_deg'))}° | 限制 {', '.join(row.get('clip_flags') or ()) or '无'}\n"
                f"{timing_description(row)}{hybrid_description(row)}\n"
                f"近期实际控制频率 {self.fmt(observed_control_hz(self.points))} Hz（按控制 tick 计数）"
            )
            self.arm_page.tracking_status.set(arm_tracking_description(row))
        self.update_editability()
        self.update_timing_settings()
        self.confirm_check.configure(state="disabled" if self.running else "normal")
        self.update_optional()
        self.calibration_button.configure(state="disabled" if self.settings_busy() else "normal")
        self.start_button.configure(state="disabled" if self.running else "normal")
        self.hybrid_button.configure(
            state="disabled"
            if self.running
            or (
                self.recording_page.full_arm_enabled.get()
                and not self.using_default_full_arm_profile()
                and not (
                    self.recording_page.checked_profile_bounded
                    or self.recording_page.checked_profile_operator
                )
            )
            else "normal",
            text=(
                "启动全臂遥操（六臂 MIT＋夹爪力控）"
                if self.using_default_full_arm_profile()
                or self.recording_page.checked_profile_operator
                or (self.running and getattr(self.config.b601, "operator_teleoperation", None) is not None)
                else "Teacher 受限全臂验证（可选诊断）"
                if self.recording_page.checked_profile_bounded
                else "全臂只读（必要参数待选 / 预检）"
            )
            if self.recording_page.full_arm_enabled.get()
            else "启动主从夹爪遥操（使能）",
        )
        self.stop_button.configure(
            text="停止 / 全臂失能 [Esc]"
            if (
                tuple(self.config.runtime.joint_names) != ("gripper",)
                if self.running and self.config is not None
                else self.recording_page.full_arm_enabled.get()
            )
            else "停止 / 夹爪失能 [Esc]"
        )
        self.read_button.configure(
            text="取消只读复核"
            if self.connection_recovery is not None or self.recovering_full_arm
            else "断开只读监测"
            if self.monitoring
            else "连接 / 只读监测"
        )
        active = self.running and row and not row.get("cleanup_pending") and not self.finishing
        normal_operator = bool(self.running and operator_teleoperation_summary(self.config))
        self.arm_page.target_button.configure(state="normal" if active and normal_operator else "disabled")
        if normal_operator and row:
            self.arm_page.effective_target.set(
                f"owner 请求目标 {self.fmt(row.get('force_target_requested_N'))} N；"
                f"本拍生效目标 {self.fmt(row.get('force_target_effective_N', row.get('force_target_N')))} N"
            )
        self.recording_page.update(
            self.last_recording,
            active=bool(active and not self.live.stop.is_set()),
            preview=getattr(self.live, "preview", None),
        )
        self.release_button.configure(
            state="normal"
            if active
            and row.get("phase")
            in ("FORCE_TRACK", "APPROACH", "POSITION_HOLD", "FOLLOW", "FOLLOW_WAIT_CLEAR")
            else "disabled"
        )
        self.freeze_button.configure(
            state="normal"
            if active and row.get("phase") in ("FORCE_TRACK", "APPROACH", "FOLLOW", "FOLLOW_WAIT_CLEAR")
            else "disabled"
        )
        can_move = not self.running or (
            active
            and row.get("phase")
            in ("FORCE_TRACK", "APPROACH", "POSITION_HOLD", "FOLLOW", "FOLLOW_WAIT_CLEAR")
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

    @staticmethod
    def fmt(value):
        return "—" if value is None else f"{value:.5g}"

    def draw(self):
        canvas = self.canvas
        try:
            window = positive("波形窗口", float(self.fields["plot_seconds"].get()))
        except ValueError:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 80 or height < 120:
            return
        last_anchor = self.points[-1]["decision_anchor_ns"] if self.points else None
        key = (last_anchor, width, height, window)
        if getattr(self, "_last_plot_key", None) == key:
            return
        self._last_plot_key = key
        # A fast UI poll must not rebuild identical curves while waiting for
        # telemetry. Retain the final waveform after stop until new data arrives.
        now = last_anchor if last_anchor is not None else time.perf_counter_ns()
        while self.points and now - self.points[0]["decision_anchor_ns"] > window * 1e9:
            self.points.popleft()
        canvas.delete("all")
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
                    ("delta_contact_deg", "回退", "#db2777"),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Build and close the UI; no device I/O")
    parser.add_argument("--connect", action="store_true", help="Start query-only monitoring; never enable")
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_GRIPPER_CALIBRATION)
    parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--recording-config", type=Path, help="Prefill the recording form only; does not connect or enable"
    )
    parser.add_argument("--gripper-only", action="store_true", help="Use the standalone gripper entry")
    args = parser.parse_args(argv)
    window = tk.Tk()
    app = Tuner(
        window,
        calibration_path=args.calibration_file,
        settings_path=args.settings_file,
        default_full_arm_mit=not args.gripper_only,
    )
    if args.recording_config is not None:
        document = json.loads(args.recording_config.read_text(encoding="utf-8"))
        from lerobot.force_control.demonstration import TeacherRecordConfig

        TeacherRecordConfig(**document).validate()
        for key, variable in app.recording_page.fields.items():
            if key in document:
                variable.set(str(document[key]))
        app.recording_page.cameras.delete("1.0", "end")
        app.recording_page.cameras.insert(
            "1.0", json.dumps(document["camera_configs"], ensure_ascii=False, indent=2)
        )
        app.recording_page.resume.set(document.get("resume", False))
        app.recording_page.enabled.set(True)
    if args.smoke_test:
        window.after(250, app.close)
    elif args.connect:
        window.after(200, app.read)
    window.mainloop()


if __name__ == "__main__":
    main()
