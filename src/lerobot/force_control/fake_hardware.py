#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""CPU-only regression plant; its numerical parameters are not B601 calibration."""

# SI unit N is capitalized in explicit dimensional field names.
# ruff: noqa: N815

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from lerobot.sensors import Sensor, SensorConfig, SensorFeature

from .hardware_types import JointFeedback, JointSendResult, JointSnapshot, SendReceipt

FAKE_B601_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


@dataclass
class FakePlant:
    """Supported contact with optional compliance, relaxation and command lag."""

    close_sign: int = 1
    position_deg: float = 0.0
    contact_position_deg: float = 1.0
    stiffness_N_deg: float = 1.0
    response_time_s: float = 0.03
    relaxation_N_s: float = 0.0
    disturbance_N: float = 0.0

    def __post_init__(self):
        """Initialize state without opening a device."""
        self._lock = threading.RLock()
        self.target_deg = self.position_deg
        self._updated_ns = time.perf_counter_ns()

    def advance(self):
        """Advance only the simulation's physical position using elapsed host time."""
        with self._lock:
            now = time.perf_counter_ns()
            dt = (now - self._updated_ns) / 1e9
            self._updated_ns = now
            alpha = 1.0 if self.response_time_s == 0 else -math.expm1(-dt / self.response_time_s)
            self.position_deg += alpha * (self.target_deg - self.position_deg)
            self.contact_position_deg += self.close_sign * self.relaxation_N_s * dt / self.stiffness_N_deg

    def forces(self):
        """Return semantic N directly; contact is unilateral in this toy model."""
        with self._lock:
            self.advance()
            force = max(0.0, self.close_sign * (self.position_deg - self.contact_position_deg))
            force = max(0.0, force * self.stiffness_N_deg + self.disturbance_N)
            return force, force


class FakeForceSensor(Sensor):
    """Publish simulated force through the real Sensor history/subscription machinery."""

    def __init__(self, plant: FakePlant, sample_rate_hz: float = 200):
        """Initialize simulated acquisition without starting a worker."""
        super().__init__(SensorConfig(sample_rate_hz=sample_rate_hz, max_age_ms=100))
        self.plant = plant
        self._stop = threading.Event()
        self._thread = None
        self.invalid = False
        self.paused = False

    @property
    def features(self):
        """Expose the unchanged two-force SI contract."""
        return {name: SensorFeature("float32", "N") for name in ("left.normal_force", "right.normal_force")}

    @property
    def provenance(self):
        """Identify simulated host-time data without claiming device timestamps."""
        return {"backend": "fake", "calibration_id": "fake_identity_N", "timestamp_source": "host_simulation"}

    @property
    def is_connected(self):
        """Whether the simulated acquisition worker is alive."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def publish_once(self):
        """Publish one sample, also useful for deterministic failure tests."""
        left, right = self.plant.forces()
        return self._publish_sample(
            {"left.normal_force": left, "right.normal_force": right} if not self.invalid else {},
            timestamp_ns=time.perf_counter_ns(),
            is_valid=not self.invalid,
            status="injected_invalid" if self.invalid else "ok",
            error="fake acquisition failure" if self.invalid else None,
        )

    def connect(self):
        """Start fake acquisition; no serial, CAN, motorbridge or GPU access."""
        if self.is_connected:
            raise RuntimeError("Fake sensor already connected")
        self._reset_framework_state()
        self._stop.clear()

        def acquire():
            while not self._stop.is_set():
                if not self.paused:
                    self.publish_once()
                self._stop.wait(1 / self.sample_rate_hz)

        self._thread = threading.Thread(target=acquire, name="fake-force", daemon=True)
        self._thread.start()

    def disconnect(self):
        """Stop and join the fake acquisition worker."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            if self._thread.is_alive():
                raise TimeoutError("Fake acquisition did not stop")
        self._thread = None


class FakeGripper:
    """Single-owner mock I/O with auditable received snapshots and ambiguous sends."""

    def __init__(self, plant: FakePlant, *, clock=time.perf_counter_ns):
        """Initialize injectable mock I/O without claiming an owner."""
        self.plant = plant
        self.clock = clock
        self.owner = None
        self.connected = False
        self.sequence = 0
        self.received_ns = None
        self.received_position_deg = None
        self.snapshot_error = None
        self.freeze_feedback = False
        self.send_failure = False
        self.stop_failure = False
        self.driver_clip = None
        self.calls = []
        self.arm_positions = {name: 0.0 for name in FAKE_B601_JOINT_NAMES if name != "gripper"}
        self.arm_limits = dict.fromkeys(self.arm_positions, (-180.0, 180.0))
        self.freeze_joints = set()
        self._received_joints = {}
        self.send_failure_joint = None
        self.submitted_goals = []

    @property
    def has_resources(self):
        """Mirror a retained transport handle even when its owner thread has exited."""
        return self.connected

    def bind_io_owner(self):
        """Claim every simulated bus operation for exactly one execution context."""
        thread_id = threading.get_ident()
        if self.owner not in (None, thread_id):
            raise RuntimeError("B601 I/O belongs to another owner")
        self.owner = thread_id

    def preflight_control(self):
        """Authorize only this in-memory mock; never inspect or open hardware."""
        if self.plant.close_sign not in (-1, 1) or not math.isfinite(self.plant.position_deg):
            raise ValueError("Invalid fake plant initial state")

    def _check(self, operation):
        if self.owner != threading.get_ident():
            raise RuntimeError("B601 I/O attempted outside its owner")
        self.calls.append((operation, threading.get_ident()))

    def connect(self):
        """Open only the simulated transport, inside its owner."""
        self._check("connect")
        self.connected = True

    def read_joint_snapshot(self, names, *, deadline_ns):
        """Return missing/stale feedback explicitly; never refresh frozen samples."""
        self._check("read")
        self.plant.advance()
        if not names or len(set(names)) != len(names) or set(names) - set(FAKE_B601_JOINT_NAMES):
            raise ValueError("Fake feedback requires unique known bare joint names")
        if not self.freeze_feedback:
            self.sequence += 1
            self.received_ns = self.clock()
            self.received_position_deg = self.plant.position_deg
        now = self.clock()
        valid = self.received_ns is not None and self.snapshot_error is None and now <= deadline_ns
        joints = []
        for name in names:
            if not self.freeze_feedback and name not in self.freeze_joints:
                self._received_joints[name] = (
                    self.received_position_deg if name == "gripper" else self.arm_positions[name],
                    self.received_ns,
                    self.sequence,
                )
            position, received, sequence = self._received_joints.get(name, (None, None, None))
            joints.append(
                JointFeedback(
                    name,
                    position if valid else None,
                    valid and received is not None,
                    received,
                    sequence,
                    None if received is None else now - received,
                    "fake_host_receive",
                    self.snapshot_error,
                )
            )
        return JointSnapshot(tuple(joints), now, deadline_ns, self.snapshot_error)

    def send_gripper_position(self, position_deg, *, feedback_snapshot):
        """Submit just one motor; a raised transport error has unknown execution."""
        self._check("send_gripper")
        start = self.clock()
        if not math.isfinite(position_deg) or not feedback_snapshot.is_valid:
            raise ValueError("Invalid fake command or snapshot")
        clipped = position_deg
        if self.driver_clip is not None:
            clipped = min(self.driver_clip[1], max(self.driver_clip[0], position_deg))
        if not self.send_failure:
            with self.plant._lock:
                self.plant.target_deg = clipped
        end = self.clock()
        result = JointSendResult(
            "gripper",
            position_deg,
            clipped,
            None if self.send_failure else clipped,
            "unknown" if self.send_failure else "success",
            start,
            end,
            "unknown",
            "injected ambiguous send" if self.send_failure else None,
        )
        return SendReceipt((result,), start, end)

    def send_joint_positions_partial(self, goals_deg, *, feedback_snapshot):
        """Submit only explicit fake joints, preserving sequential partial failure evidence."""
        self._check("send_partial")
        if not goals_deg or set(goals_deg) - set(FAKE_B601_JOINT_NAMES):
            raise ValueError("Fake partial goals require known bare joint names")
        if not feedback_snapshot.is_valid:
            raise ValueError("Invalid fake snapshot")
        clipped = {}
        for name, value in goals_deg.items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Invalid fake partial position")
            if not feedback_snapshot.get(name).is_valid:
                raise ValueError("Missing valid fake joint feedback")
            limits = self.driver_clip if name == "gripper" else self.arm_limits[name]
            clipped[name] = value if limits is None else min(limits[1], max(limits[0], value))
        start, failed, results = self.clock(), False, []
        for name, value in goals_deg.items():
            begin = self.clock()
            status = "not_attempted" if failed else "success"
            if not failed and (self.send_failure or self.send_failure_joint == name):
                status, failed = "unknown", True
            if status == "success":
                if name == "gripper":
                    with self.plant._lock:
                        self.plant.target_deg = clipped[name]
                else:
                    self.arm_positions[name] = clipped[name]
                self.submitted_goals.append((name, clipped[name], begin))
            results.append(
                JointSendResult(
                    name,
                    value,
                    clipped[name],
                    clipped[name] if status == "success" else None,
                    status,
                    begin if status != "not_attempted" else None,
                    self.clock() if status != "not_attempted" else None,
                    "unknown",
                    None if status == "success" else "injected sequential submission failure",
                )
            )
        return SendReceipt(tuple(results), start, self.clock())

    def emergency_stop(self, reason):
        """Cancel a simulated pending target; this proves nothing about B601 stopping."""
        self._check("emergency_stop")
        start = self.clock()
        if self.stop_failure:
            return {
                "action": "fake_hold",
                "status": "unknown",
                "hardware_ack": "unknown",
                "reason": reason,
                "q_submitted_deg": None,
                "send_start_ns": start,
                "send_end_ns": self.clock(),
            }
        with self.plant._lock:
            self.plant.advance()
            self.plant.target_deg = self.plant.position_deg
        return {
            "action": "fake_hold",
            "status": "submitted",
            "hardware_ack": "unknown",
            "reason": reason,
            "q_submitted_deg": self.plant.target_deg,
            "send_start_ns": start,
            "send_end_ns": self.clock(),
        }

    def disconnect(self):
        """Close the simulated transport inside its owner."""
        self._check("disconnect")
        self.connected = False
