#!/usr/bin/env python
"""Write real demonstration files using synthetic joints, force and RGB images.

No hardware connection is created. This checks the asynchronous recording,
encoding, Sidecar and read-only verification path only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.force_control.capture_annotations import annotate_capture
from lerobot.force_control.demonstration import JOINT_NAMES, CausalCameraBuffer, DemonstrationController
from lerobot.force_control.demonstration_validation import validate_demonstration
from lerobot.force_control.fake_hardware import FakeForceSensor, FakePlant
from lerobot.force_control.hardware_types import JointFeedback, JointSendResult, JointSnapshot
from lerobot.force_control.owner import OwnerSnapshot, OwnerSubscription


class SyntheticCameras:
    """Publish synthetic RGB frames; never instantiate an OpenCV camera."""

    def __init__(self, config):
        self.config = config
        self.buffers = {name: CausalCameraBuffer(64) for name in config.camera_objects()}
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        def produce():
            index = 0
            try:
                while not self._stop.is_set():
                    for name, spec in self.config.camera_objects().items():
                        image = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
                        image[:, :, index % 3] = index % 255
                        captured = time.perf_counter_ns()
                        self.buffers[name].publish(image, captured, time.perf_counter_ns())
                    index += 1
                    self._stop.wait(1 / 60)
            except BaseException as exc:
                self.error = exc

        self._thread = threading.Thread(target=produce, name="synthetic-camera", daemon=False)
        self._thread.start()
        deadline = time.monotonic() + 2
        while any(buffer.latest() is None for buffer in self.buffers.values()):
            self.check_health()
            if time.monotonic() >= deadline:
                raise TimeoutError("Synthetic cameras did not start")
            time.sleep(0.005)

    def check_health(self):
        if self.error is not None:
            raise RuntimeError("Synthetic camera producer failed") from self.error

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise TimeoutError("Synthetic camera producer did not stop")


class SyntheticOwner:
    """Emit completed synthetic receipts without constructing a hardware adapter."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.runtime = SimpleNamespace(config=SimpleNamespace(joint_names=JOINT_NAMES, control_hz=50))
        self._subscriptions = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.tick_count = 0
        self._latest = None

    @property
    def has_resources(self):
        return self.is_running

    def snapshot(self, *, allow_empty=False):
        return self._latest

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def subscribe(self, capacity):
        subscription = OwnerSubscription(capacity)
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription):
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            subscription.closed = True

    def start(self):
        self._thread = threading.Thread(target=self._run, name="synthetic-seven-joint-owner", daemon=False)
        self._thread.start()

    def _snapshot(self, tick):
        start = time.perf_counter_ns()
        sample = self.sensor.read_latest_before(start, max_age_ms=100, minimum_sequence=0)
        state = [0.1 * math.sin(tick / 30)] * 6 + [1.2]
        action = [value + 0.01 for value in state[:6]] + [1.2]
        joints = tuple(
            JointFeedback(name, value, True, start, tick, 0, "synthetic_host")
            for name, value in zip(JOINT_NAMES, state, strict=True)
        )
        anchor = time.perf_counter_ns()
        snapshot = JointSnapshot(joints, anchor, anchor + 1_000_000)
        sent = time.perf_counter_ns()
        receipt = [
            asdict(JointSendResult(name, value, value, value, "success", sent, sent))
            for name, value in zip(JOINT_NAMES, action, strict=True)
        ]
        phase = ("FOLLOW", "FORCE_TRACK", "FOLLOW_WAIT_CLEAR")[(tick // 10) % 3]
        contact_transition = tick % 30 == 10
        row = {
            "tick_id": tick,
            "tick_start_ns": start,
            "decision_anchor_ns": anchor,
            "epoch": 0,
            "mode": "TEACHER",
            "phase": phase,
            "phase_before": "FOLLOW" if contact_transition else phase,
            "phase_at_tick_start": "FOLLOW" if contact_transition else phase,
            "joint_snapshot": asdict(snapshot),
            "feedback_valid": True,
            "feedback_received_ns": start,
            "feedback_time_source": "synthetic_host",
            "joint_feedback_max_age_ms": 100,
            "force_max_age_ms": 100,
            "force_valid": sample.is_valid,
            "force_sequence": sample.sequence,
            "force_hardware_sequence": sample.hardware_sequence,
            "force_status": sample.status,
            "force_timestamp_ns": sample.timestamp_ns,
            "force_arrival_timestamp_ns": sample.arrival_timestamp_ns,
            "force_left_N": sample.values["left.normal_force"],
            "force_right_N": sample.values["right.normal_force"],
            "force_mean_N": sum(sample.values.values()) / 2,
            "force_target_requested_N": 0.2,
            "force_target_effective_N": 0.2,
            "contact_force_threshold_N": 0.1,
            "send_status": "submitted",
            "send_start_ns": sent,
            "send_end_ns": sent,
            "per_joint_results": receipt,
            "submitted_action": {
                f"{name}.pos": value for name, value in zip(JOINT_NAMES, action, strict=True)
            },
            "q_measured_deg": 1.2,
            "q_prev_sent_deg": 1.2,
            "q_sent_deg": 1.2,
            "q_candidate_deg": 1.2,
            "q_after_safety_deg": 1.2,
            "u_label_deg_s": 0.0,
            "prev_successful_tick_id": tick - 1 if tick else None,
            "dt_control_s": 0.02,
            "close_sign": 1,
            "safety_override": False,
            "deadline_miss": False,
            "failure_reason": None,
            "label_valid": tick > 0 and phase == "FORCE_TRACK" and not contact_transition,
            "hybrid_transition": "contact" if contact_transition else None,
            "hardware_ack": "unknown",
            "synthetic": True,
        }
        row.update(annotate_capture(row))
        return OwnerSnapshot(row, snapshot, tuple(zip(JOINT_NAMES, action, strict=True)), 0, "TEACHER", tick)

    def _run(self):
        tick, deadline = 0, time.perf_counter()
        try:
            while not self._stop.is_set():
                snapshot = self._snapshot(tick)
                self._latest = snapshot
                with self._lock:
                    for subscription in self._subscriptions:
                        try:
                            subscription.queue.put_nowait(copy.deepcopy(snapshot))
                        except queue.Full:
                            subscription.error = "Synthetic owner capture queue overflow"
                tick += 1
                self.tick_count = tick
                deadline += 0.02
                now = time.perf_counter()
                if now > deadline:
                    deadline += (math.floor((now - deadline) / 0.02) + 1) * 0.02
                self._stop.wait(max(0, deadline - time.perf_counter()))
        except BaseException as exc:
            self.error = exc
            with self._lock:
                for subscription in self._subscriptions:
                    subscription.error = f"Synthetic owner failed: {exc}"

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise TimeoutError("Synthetic owner did not stop")


def _wait_state(controller, owner, states, timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while True:
        status = controller.status()
        if status["state"] in states:
            return status
        if status["state"] == "error" or owner.error is not None:
            raise RuntimeError(status.get("error") or str(owner.error))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Recording state timeout: {status}")
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="New local output Dataset directory")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Append to an existing synthetic smoke Dataset")
    parser.add_argument("--encoder", choices=("software", "nvenc"), default="software")
    parser.add_argument("--verify-video-on-save", action="store_true")
    args = parser.parse_args()
    if args.root.exists() and not args.resume:
        parser.error("--root must be a new directory")
    if args.resume and not (args.root / "meta/info.json").is_file():
        parser.error("--resume requires an existing Dataset")
    if args.episodes <= 0 or not math.isfinite(args.duration_s) or args.duration_s < 0.2:
        parser.error("--episodes must be positive and --duration-s must be at least 0.2")
    sensor = FakeForceSensor(FakePlant(position_deg=1.2, contact_position_deg=1.0), sample_rate_hz=400)
    owner, controller = SyntheticOwner(sensor), None
    saves = []
    try:
        sensor.connect()
        # Establish the initial synthetic sample before the owner can select it.
        sensor.publish_once()
        owner.start()
        controller = DemonstrationController(
            owner,
            sensor,
            camera_factory=SyntheticCameras,
            provenance={"backend": "synthetic", "hardware_not_tested": True, "physical_demonstrations": 0},
        )
        config = {
            "root": str(args.root.resolve()),
            "task": "Synthetic software recording smoke test",
            "object_id": "synthetic_object",
            "contact_condition_id": "synthetic_contact",
            "camera_configs": {
                name: {"type": "opencv", "index_or_path": index, "fps": 50, "width": 640, "height": 480}
                for index, name in enumerate(("front", "wrist"))
            },
            "h264_encoder": args.encoder,
            "verify_video_on_save": args.verify_video_on_save,
            "target_episode_count": args.episodes,
            "resume": args.resume,
        }
        for episode in range(args.episodes):
            controller.submit("record_start", {"config": config})
            _wait_state(controller, owner, {"recording"})
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                if owner.error is not None or controller.status()["state"] == "error":
                    raise RuntimeError(controller.status().get("error") or str(owner.error))
                time.sleep(0.01)
            before_save = owner.tick_count
            controller.submit("record_finish", {"task_result": "unknown"})
            saved = _wait_state(controller, owner, {"saved"})
            saves.append(
                {
                    "episode_uid": saved["episode_uid"],
                    "owner_ticks_during_save": owner.tick_count - before_save,
                }
            )
            print(
                json.dumps(
                    {
                        "software_episode": episode + 1,
                        "frames": saved["frame_count"],
                        "files_complete": saved["file_complete"],
                    }
                ),
                flush=True,
            )
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                owner.close()
            finally:
                sensor.disconnect()
    report = validate_demonstration(args.root)
    report.update(
        hardware_not_tested=True,
        physical_demonstrations=0,
        synthetic_episodes=len(report["episodes"]),
        added_episodes=args.episodes,
        saves=saves,
    )
    report_path = args.root / "software_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] and len(report["episodes"]) >= args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
