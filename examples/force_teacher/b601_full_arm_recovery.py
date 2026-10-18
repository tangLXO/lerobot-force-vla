"""Explicit seven-joint query-only recovery; no control backend or motor writes."""

import math
import time
import uuid
from dataclasses import asdict

from lerobot.robots.rebot_b601_follower.arm_calibration import B601ArmCalibration
from lerobot.robots.rebot_b601_follower.readonly_transport import (
    DEFAULT_B601_JOINTS,
    B601ReadonlyBatch,
    ReadonlyB601Transport,
)


def query_disabled_arm(
    port,
    calibration: B601ArmCalibration,
    *,
    baudrate,
    mapping_budget_s,
    feedback_budget_s,
    max_age_s,
    stop,
    session_id=None,
    transport_factory=None,
    retain=None,
):
    """Observe all seven disabled drives and close the same read-only owner.

    This function never encodes enable, disable, position or parameter writes.
    A failed observation or close remains an incomplete recovery receipt.
    """
    receipt = {
        "source": "explicit_full_arm_query_only_recovery",
        "session_id": session_id or str(uuid.uuid4()),
        "port": port,
        "read_only": True,
        "motion_attempted": False,
        "parameter_writes": False,
        "hardware_ack": "unknown",
        "hardware_stop_confirmed": False,
        "all_joints_disabled_observed": False,
        "transport_closed": False,
        "complete": False,
        "errors": [],
        "started_ns": time.perf_counter_ns(),
        "calibration": asdict(calibration),
    }
    device = None

    def checkpoint():
        if stop.is_set():
            raise RuntimeError("用户已停止全臂只读复核")
        calibration.require_unchanged()

    try:
        for name, value in (
            ("mapping_budget_s", mapping_budget_s),
            ("feedback_budget_s", feedback_budget_s),
            ("max_age_s", max_age_s),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        expected = {spec.name: spec for spec in DEFAULT_B601_JOINTS}
        if {name: motor_id for name, motor_id, _, _ in calibration.joints} != {
            name: spec.motor_id for name, spec in expected.items()
        }:
            raise ValueError("全臂只读复核需要当前七关节校准")
        checkpoint()
        factory = transport_factory or ReadonlyB601Transport
        device = factory(port, DEFAULT_B601_JOINTS, baudrate=baudrate)
        device.open()
        mapping = device.read_mapping(max_read_duration_s=mapping_budget_s)
        receipt["mapping"] = asdict(mapping)
        if not mapping.is_valid:
            raise RuntimeError(f"全臂映射查询失败：{mapping.error}")
        checkpoint()
        batch = device.read_feedback(names=tuple(expected), max_read_duration_s=feedback_budget_s)
        receipt["feedback"] = asdict(batch)
        if not isinstance(batch, B601ReadonlyBatch) or not batch.is_valid:
            raise RuntimeError(f"全臂反馈查询失败：{batch.error}")
        if len(batch.feedback) != 7 or {row.name for row in batch.feedback} != set(expected):
            raise RuntimeError("全臂反馈未完整包含七个不同关节")
        now = time.perf_counter_ns()
        bounds = {name: (lower, upper) for name, _, lower, upper in calibration.joints}
        for row in batch.feedback:
            spec = expected[row.name]
            if row.motor_id != spec.motor_id or row.feedback_id != spec.feedback_id:
                raise RuntimeError(f"全臂反馈 ID 不一致：{row.name}")
            if row.status_code != 0:
                raise RuntimeError(f"关节未观察到 DISABLED：{row.name}={row.status_name}")
            if (
                type(row.receive_timestamp_ns) is not int
                or not receipt["started_ns"] <= row.receive_timestamp_ns <= now
                or now - row.receive_timestamp_ns > max_age_s * 1e9
                or type(row.receive_sequence) is not int
                or row.receive_sequence <= 0
            ):
                raise RuntimeError(f"关节反馈不是本次新鲜数据：{row.name}")
            if (
                type(row.position_deg) not in (int, float)
                or not math.isfinite(row.position_deg)
                or not bounds[row.name][0] <= row.position_deg <= bounds[row.name][1]
            ):
                raise RuntimeError(f"关节位置超出当前校准硬范围：{row.name}")
        checkpoint()
        receipt.update(all_joints_disabled_observed=True, observed_ns=now)
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if device is not None:
            try:
                if device.has_resources:
                    device.close()
            except Exception as exc:
                receipt["errors"].append(f"close: {type(exc).__name__}: {exc}")
            receipt["transport_closed"] = not device.has_resources
            if device.has_resources and retain is not None:
                retain(device)
        else:
            receipt["transport_closed"] = True
        receipt["closed_ns"] = time.perf_counter_ns()
    try:
        checkpoint()
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}: {exc}")
    receipt["complete"] = (
        receipt["all_joints_disabled_observed"] and receipt["transport_closed"] and not receipt["errors"]
    )
    return receipt
