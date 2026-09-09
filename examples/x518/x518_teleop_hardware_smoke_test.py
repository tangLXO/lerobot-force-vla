"""Bounded real SO101 teleoperation with X518 and two cameras.

Uses existing motor calibration/configuration. Requires aligned arms, clear workspace,
and an operator at the leader. --check-only never enables torque or sends targets.
Actual runs last 15 seconds, cap each target delta at 0.5 units, and refuse targets
over 15 units from the follower start pose (degrees; gripper percentage points).
Esc stops the loop. Output is diagnostics, not a training dataset.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.processor import make_default_processors
from lerobot.robots.sensorized_robot import attach_sensors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import init_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    init_logging()
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration"
    inner = SO101Follower(
        SO101FollowerConfig(
            port="COM13",
            id="my_follower",
            calibration_dir=calibration / "robots/so_follower",
            max_relative_target=0.5,
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=i, width=640, height=480, fps=30, backend=Cv2Backends.DSHOW
                )
                for i, name in enumerate(("wrist", "front"))
            },
        )
    )
    leader = SO101Leader(
        SO101LeaderConfig(
            port="COM24",
            id="my_leader",
            calibration_dir=calibration / "teleoperators/so_leader",
        )
    )
    robot = attach_sensors(
        inner,
        {
            "gripper_force": X518SensorConfig(
                host="192.168.1.100",
                expected_sample_rate_hz=200,
                expected_unit="kg",
                max_age_ms=100,
                channels={
                    "left.normal_force": X518ChannelConfig(channel=1),
                    "right.normal_force": X518ChannelConfig(channel=2),
                },
            )
        },
    )
    listener = None
    rows = []
    result = {"check_only": args.check_only, "motion_commands": 0, "status": "not_started"}
    torque_owned = False
    try:
        leader.bus.connect()
        inner.bus.connect()
        for arm in (leader, inner):
            if not arm.bus.is_calibrated:
                raise RuntimeError("Existing calibration does not match; recalibration required.")
            if any(arm.bus.sync_read("Torque_Enable", normalize=False).values()):
                raise RuntimeError("Expected both arms initially torque-off.")
            if any(arm.bus.sync_read("Operating_Mode", normalize=False).values()):
                raise RuntimeError("Expected existing position mode on all motors.")
        start = inner.bus.sync_read("Present_Position")
        lead = leader.bus.sync_read("Present_Position")
        delta = {k: lead[k] - start[k] for k in start}
        result["initial_delta"] = delta
        result["alignment_passed"] = all(abs(v) <= 10 for v in delta.values())
        print(json.dumps(result, indent=2), flush=True)
        if not result["alignment_passed"]:
            raise RuntimeError("Arms differ by more than 10 degrees/gripper points; align manually first.")
        if args.check_only:
            result["status"] = "preflight_passed"
            return
        for camera in inner.cameras.values():
            camera.connect()
        for sensor in robot.sensors.values():
            sensor.connect()
        robot._wait_until_ready("gripper_force", robot.sensors["gripper_force"])
        listener, events = init_keyboard_listener()
        if listener is None:
            raise RuntimeError("Keyboard stop listener unavailable.")
        # Recheck after camera/sensor startup, immediately before torque enable.
        start = inner.bus.sync_read("Present_Position")
        lead = leader.bus.sync_read("Present_Position")
        if any(abs(lead[k] - start[k]) > 10 for k in start):
            raise RuntimeError("Alignment changed during startup.")
        inner.bus.sync_write("Goal_Position", start)
        torque_owned = True
        inner.bus.enable_torque()
        if not all(inner.bus.sync_read("Torque_Enable", normalize=False).values()):
            raise RuntimeError("Follower torque enable verification failed.")
        original_observe, original_send = robot.get_observation, robot.send_action
        latest = {}

        def observe():
            obs = original_observe()
            latest.clear()
            latest.update({k: float(v) for k, v in obs.items() if not isinstance(v, np.ndarray)})
            if any(abs(obs[f"{k}.pos"] - start[k]) > 17 for k in start):
                raise RuntimeError("Follower exceeded the diagnostic motion envelope.")
            return obs

        def send(action):
            if events["stop_recording"]:
                raise KeyboardInterrupt
            if any(abs(action[f"{k}.pos"] - start[k]) > 15 for k in start):
                raise RuntimeError("Leader target exceeded 15-degree/point diagnostic envelope.")
            sent = original_send(action)
            result["motion_commands"] += 1
            rows.append(
                {
                    "timestamp_ns": time.perf_counter_ns(),
                    "observation": dict(latest),
                    "target": dict(action),
                    "sent": sent,
                    "force_age_ns": robot.last_capture_metadata["sensors"]["gripper_force"]["age_ns"],
                }
            )
            return sent

        robot.get_observation, robot.send_action = observe, send
        print("FOLLOWING NOW: move leader gently; Esc stops; automatic stop after 15 seconds.", flush=True)
        t, r, o = make_default_processors()
        teleop_loop(leader, robot, 30, t, r, o, duration=15)
        result["status"] = "completed"
    except KeyboardInterrupt:
        result["status"] = "operator_stopped"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        cleanup_errors = []
        # Release motor torque first, even if sensor or camera cleanup later fails.
        if inner.bus.is_connected and torque_owned:
            try:
                inner.bus.disable_torque(num_retry=5)
                result["final_torque"] = inner.bus.sync_read("Torque_Enable", normalize=False)
                if any(result["final_torque"].values()):
                    raise RuntimeError("Follower torque is still enabled.")
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        resources = [listener] if listener is not None else []
        for resource in resources:
            try:
                resource.stop()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        for sensor in robot.sensors.values():
            try:
                if sensor.is_connected:
                    sensor.disconnect()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        for camera in inner.cameras.values():
            try:
                if camera.is_connected or camera.thread is not None:
                    camera.disconnect()
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        for arm in (inner, leader):
            try:
                if arm.bus.is_connected:
                    arm.bus.disconnect(disable_torque=False)
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        result["cleanup_errors"] = cleanup_errors
        result["rows"] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
        if cleanup_errors:
            raise RuntimeError(f"Cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
