"""Shared headless evidence, query-only monitoring and panel presentation helpers."""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from lerobot.force_control.config import finite_number, positive
from lerobot.force_control.runtime import DEFAULT_FEEDBACK_BUDGET_MS, DEFAULT_TICK_DEADLINE_MS
from lerobot.force_control.trace import validate_trace
from lerobot.force_control.tuning import sensor_config
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS, ReadonlyB601Transport
from lerobot.sensors.x518 import X518Sensor

_UNSTOPPED_PREVIEWS = []


def command_feedback_override(config, requested_mode, command_tx_gap_us):
    """Keep standalone gripper starts on their supported sequential transport."""
    if requested_mode is None:
        return None
    if tuple(config.runtime.joint_names) == ("gripper",) and requested_mode in ("cached", "pipelined"):
        requested_mode = "sequential"
    return {"command_feedback_mode": requested_mode, "command_tx_gap_us": command_tx_gap_us}


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


def timing_settings_description(
    values, config=None, *, startup_complete=False, running=False, command_feedback_mode=None
):
    """Keep editable requests distinct from the reviewed owner configuration."""
    mode = getattr(
        getattr(getattr(config, "b601", None), "control", None),
        "command_feedback_mode",
        command_feedback_mode,
    )
    if mode == "cached":
        hz = config.runtime.control_hz if config is not None else float(values["control_hz"])
        period_ms = round(1e9 / hz) / 1e6
        state = "本次 owner 运行生效" if startup_complete and running else "待启动或已结束，当前未生效"
        return (
            f"缓存模式：正常目标 {period_ms:g} ms / 整拍硬截止 {2 * period_ms:g} ms；"
            f"连续 3 拍超期停止，决策最多存活 {2 * period_ms:g} ms\n"
            f"{state}；反馈、力和主臂沿用各自时效上限"
        )
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


def monitor(stop, publish, v, calibration, cleanup_publish=None, *, retained=None):
    """Query-only preview; close both devices before transferring ownership."""
    calibration.require_unchanged()
    retained = _UNSTOPPED_PREVIEWS if retained is None else retained
    if retained:
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
                        retained.append((device, sensor, lease))
                    if cleanup_publish is not None:
                        cleanup_publish(cleanup)
