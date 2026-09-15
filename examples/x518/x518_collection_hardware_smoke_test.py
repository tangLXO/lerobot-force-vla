"""Read-only SO101/X518/two-camera collection check. Actions are NOT executed.

The output is diagnostic data, not teleoperated training demonstrations.
Run with a new --root each time. No Hub upload or motor-register writes occur.
"""

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.default import SensorWindowConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_window import SensorStreamReader, SensorWindowDataset
from lerobot.robots.sensorized_robot import attach_sensors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts import lerobot_record as entry
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig
from lerobot.sensors.x518.diagnostics import summarize_capture
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import OBS_STATE, OBS_TACTILE
from lerobot.utils.import_utils import _av_available, _psutil_available, require_package

if _av_available:
    import av

if _psutil_available:
    import psutil


class ReadOnlyFollower(SO101Follower):
    def connect(self, calibrate=True):
        try:
            self.bus.connect()
            if not self.bus.is_calibrated:
                raise RuntimeError("Follower calibration does not match; no calibration writes allowed.")
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            self.disconnect()
            raise

    def send_action(self, action):
        # Intentional diagnostic seam: record the leader's target without executing it.
        return action

    def disconnect(self):
        try:
            for camera in self.cameras.values():
                if camera.is_connected or camera.thread is not None:
                    camera.disconnect()
        finally:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)


class ReadOnlyLeader(SO101Leader):
    def connect(self, calibrate=True):
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise RuntimeError("Leader calibration does not match; no calibration writes allowed.")

    def disconnect(self):
        if self.bus.is_connected:
            self.bus.disconnect(disable_torque=False)


def verify_videos(data):
    """Decode every frame once, including timestamps in files shared by episodes."""
    require_package("av", extra="av-dep")
    expected = {}
    for episode in range(data.num_episodes):
        metadata = data.meta.episodes[episode]
        for key in data.meta.video_keys:
            path = data.root / data.meta.get_video_file_path(episode, key)
            offset = metadata[f"videos/{key}/from_timestamp"]
            expected.setdefault(path, []).extend(offset + i / data.fps for i in range(metadata["length"]))
    total = 0
    for path, timestamps in expected.items():
        timestamps.sort()
        count = 0
        with av.open(str(path)) as video:
            for frame in video.decode(video=0):
                assert (frame.width, frame.height) == (640, 480)
                assert count < len(timestamps)
                assert abs(float(frame.time) - timestamps[count]) < 0.5 / data.fps
                count += 1
        assert count == len(timestamps)
        total += count
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--duration-s", type=float, default=3)
    parser.add_argument("--acquisition-mode", choices=("process", "thread"), default="process")
    parser.add_argument(
        "--rate-check", action="store_true", help="One measured episode; skip rerecord/resume exercise"
    )
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Read cameras/arms and capture force without Dataset packing",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify existing output without hardware access"
    )
    args = parser.parse_args()
    if not args.verify_only and not args.observe_only:
        require_package("psutil", extra="accelerate-dep")
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s must be positive and finite")
    if args.root.exists() and not args.verify_only:
        raise FileExistsError("Use a new output directory.")
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration"
    cfg = entry.RecordConfig(
        robot=SO101FollowerConfig(
            port="COM13",
            id="my_follower",
            calibration_dir=calibration / "robots/so_follower",
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=index, width=640, height=480, fps=30, backend=Cv2Backends.DSHOW
                )
                for index, name in enumerate(("wrist", "front"))
            },
        ),
        teleop=SO101LeaderConfig(
            port="COM24", id="my_leader", calibration_dir=calibration / "teleoperators/so_leader"
        ),
        sensors={
            "gripper_force": X518SensorConfig(
                host="192.168.1.100",
                expected_unit="kg",
                expected_sample_rate_hz=400,
                sample_rate_hz=400,
                acquisition_mode=args.acquisition_mode,
                max_age_ms=100,
                frame_features=["left.normal_force", "right.normal_force"],
                channels={
                    "left.normal_force": X518ChannelConfig(channel=1),
                    "right.normal_force": X518ChannelConfig(channel=2),
                },
            )
        },
        dataset=DatasetRecordConfig(
            repo_id="local/READ_ONLY_DIAGNOSTIC_NOT_FOR_TRAINING",
            root=args.root,
            single_task="READ ONLY DIAGNOSTIC: leader targets recorded but NOT executed",
            num_episodes=1,
            episode_time_s=args.duration_s,
            reset_time_s=0,
            push_to_hub=False,
            no_stamp=True,
            streaming_encoding=args.streaming,
            encoder_threads=2,
        ),
        play_sounds=False,
    )
    arms = []

    def make_follower(config):
        arm = ReadOnlyFollower(config)
        arms.append(arm)
        return arm

    def make_leader(config):
        arm = ReadOnlyLeader(config)
        arms.append(arm)
        return arm

    def keyboard():
        return None, {"stop_recording": False, "exit_early": False, "rerecord_episode": False}

    original_loop = entry.record_loop
    discarded = False
    runtime = []
    sensors_used = []

    def loop(*a, **kw):
        nonlocal discarded
        robot = kw["robot"]
        sensors_used.extend(robot.sensors.values())
        recorder = kw.get("sensor_recorder")
        monitor_stop = threading.Event()
        memory = []

        def monitor():
            parent = psutil.Process()
            while not monitor_stop.is_set():
                memory.append(
                    {
                        "elapsed_s": time.perf_counter() - started,
                        "parent_rss_bytes": parent.memory_info().rss,
                        "children_rss_bytes": sum(p.memory_info().rss for p in parent.children()),
                    }
                )
                monitor_stop.wait(5)

        started = time.perf_counter()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            original_loop(*a, **kw)
        finally:
            monitor_stop.set()
            watcher.join(2)
            if recorder is not None:
                runtime.append({"recorder": recorder.diagnostics, "memory": memory})
        if kw.get("dataset") is not None and not discarded and not args.rate_check:
            kw["events"]["rerecord_episode"] = True
            discarded = True

    print("READ ONLY: no motion commands, no torque changes, no Hub upload.", flush=True)
    if args.observe_only:
        if args.verify_only:
            parser.error("--observe-only cannot be combined with --verify-only")
        robot = attach_sensors(ReadOnlyFollower(cfg.robot), cfg.sensors)
        leader = ReadOnlyLeader(cfg.teleop)
        sensor = robot.sensors["gripper_force"]
        subscription = None
        rows = []
        try:
            leader.connect()
            robot.connect(calibrate=False)
            subscription = sensor.subscribe(sensor.config.resolve_recorder_queue_capacity())
            start = time.perf_counter_ns()
            deadline = time.perf_counter() + args.duration_s
            while time.perf_counter() < deadline:
                tick = time.perf_counter()
                robot.get_observation()
                leader.get_action()
                while True:
                    try:
                        rows.append(asdict(subscription.get_nowait()))
                    except queue.Empty:
                        break
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - tick)))
            end = time.perf_counter_ns()
            sensor.unsubscribe(subscription)
            while True:
                try:
                    rows.append(asdict(subscription.get_nowait()))
                except queue.Empty:
                    break
            result = summarize_capture(rows, start_ns=start, end_ns=end)
            result["acquisition"] = sensor.diagnostics
            result["queue_overflow_count"] = subscription.overflow_count
            result["passed"] &= subscription.overflow_count == 0 and sensor.diagnostics["fault"] is None
        finally:
            if subscription is not None:
                sensor.unsubscribe(subscription)
            try:
                robot.disconnect()
            finally:
                if leader.bus.is_connected:
                    leader.disconnect()
        result["acquisition"] = sensor.diagnostics
        result["resources_released"] = not sensor.has_resources
        result["recorder_queue"] = subscription.diagnostics.snapshot(
            subscription.queue, overflow_count=subscription.overflow_count
        )
        result["passed"] &= result["resources_released"] and sensor.diagnostics["fault"] is None
        args.root.mkdir(parents=True, exist_ok=False)
        (args.root / "hardware_validation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("X518 rate acceptance failed; diagnostic report retained.")
        return
    try:
        with (
            patch.object(entry, "make_robot_from_config", make_follower),
            patch.object(entry, "make_teleoperator_from_config", make_leader),
            patch.object(entry, "init_keyboard_listener", keyboard),
            patch.object(entry, "record_loop", loop),
        ):
            if not args.verify_only:
                entry.record(cfg)
                if not args.rate_check:
                    cfg.resume = True
                    entry.record(cfg)
    finally:
        for arm in reversed(arms):
            arm.disconnect()
        if not args.verify_only and args.root.exists():
            (args.root / "acquisition_runtime.json").write_text(
                json.dumps(
                    {
                        "runs": runtime,
                        "resources_released": all(not sensor.has_resources for sensor in sensors_used),
                        "final_acquisition": [sensor.diagnostics for sensor in sensors_used],
                    },
                    indent=2,
                )
            )
    data = LeRobotDataset(cfg.dataset.repo_id, root=args.root, video_backend="pyav")
    reader = SensorStreamReader(args.root, instance="gripper_force", verify="full")
    assert data.num_episodes == (1 if args.rate_check else 2)
    assert tuple(data.features[OBS_STATE]["shape"]) == (6,)
    assert tuple(data.features[OBS_TACTILE]["shape"]) == (2,)
    assert data.features[OBS_TACTILE]["dtype"] == "float32"
    assert data.features[OBS_TACTILE]["names"] == [
        "sensor.gripper_force.left.normal_force",
        "sensor.gripper_force.right.normal_force",
    ]
    assert set(data.features[OBS_STATE]["names"]).isdisjoint(data.features[OBS_TACTILE]["names"])
    assert tuple(data.features["action"]["shape"]) == (6,)
    assert reader.manifest["sidecar_schema_version"] == 2
    assert reader.manifest["frame_view"]["dataset_key"] == OBS_TACTILE
    assert reader.manifest["frame_view"]["shape"] == [2]
    episodes = []
    offset = 0
    video_frames = verify_videos(data)
    windows = SensorWindowDataset(
        data.hf_dataset.select_columns(["episode_index", "frame_index", OBS_TACTILE]),
        {"gripper_force": SensorWindowConfig(duration_ms=5, target_hz=400, max_age_ms=100)},
        root=args.root,
    )
    for episode in range(data.num_episodes):
        sync = reader.read_sync(episode_index=episode)
        anchors = np.array([row["frame_anchor_ns"] for row in sync], dtype=np.int64)
        ages = []
        for first in range(0, len(sync), 128):
            items = windows.__getitems__(range(offset + first, offset + min(first + 128, len(sync))))
            for item, row in zip(items, sync[first : first + 128], strict=True):
                window = item["sensor_windows"]["gripper_force"]
                assert bool(window["valid_mask"][-1])
                assert int(window["sequence"][-1]) == row["sensors"]["gripper_force"]["sequence"]
                np.testing.assert_allclose(np.asarray(item[OBS_TACTILE]), window["values"][-1])
                ages.append(float(window["age_ns"][-1]) / 1e6)
        acquisition_report = summarize_capture(
            reader.read_raw(episode_index=episode), start_ns=int(anchors[0]), end_ns=int(anchors[-1])
        )
        frame_hz = float((len(anchors) - 1) * 1e9 / (anchors[-1] - anchors[0]))
        acquisition_report["checks"].update(
            {
                "frame_rate": 29.7 <= frame_hz <= 30.3,
                "selected_age_p99": float(np.percentile(ages, 99)) <= 15,
                "selected_age_max": max(ages) <= 100,
            }
        )
        acquisition_report["passed"] = all(acquisition_report["checks"].values())
        episodes.append(
            {
                "episode": episode,
                "frames": len(sync),
                "effective_anchor_hz": float((len(anchors) - 1) * 1e9 / (anchors[-1] - anchors[0])),
                "sensor_age_ms_p95": float(np.percentile(ages, 95)),
                "sensor_age_ms_max": max(ages),
                "sensor_age_ms_p99": float(np.percentile(ages, 99)),
                "acquisition": acquisition_report,
            }
        )
        offset += len(sync)
    assert offset == len(data)
    journals = [json.loads(p.read_text()) for p in (args.root / "meta/sensor_transactions").glob("*.json")]
    assert sum(j["state"] == "COMMITTED" for j in journals) == data.num_episodes
    assert sum(j["state"] == "ABORTED" for j in journals) == (0 if args.rate_check else 1)
    result = {
        "diagnostic_only": True,
        "motion_executed": False,
        "streaming": args.streaming,
        "rerecord": "not_run" if args.rate_check else "passed",
        "resume": "not_run" if args.rate_check else "passed",
        "full_verification": "passed",
        "all_video_frames_decoded": True,
        "video_frames_decoded": video_frames,
        "state_is_robot_only": True,
        "tactile_matches_sync_reference": True,
        "force_history_is_sidecar_only": True,
        "episodes": episodes,
    }
    runtime_path = args.root / "acquisition_runtime.json"
    if runtime_path.exists():
        result["runtime"] = json.loads(runtime_path.read_text())
        assert result["runtime"]["resources_released"]
        for snapshot in result["runtime"]["final_acquisition"]:
            assert snapshot["ipc_overflow_count"] == snapshot["transport_gaps"] == 0
            assert snapshot["fault"] is None
        for run in result["runtime"]["runs"]:
            assert run["recorder"]["sync"]["overflow_count"] == 0
            assert all(stream["overflow_count"] == 0 for stream in run["recorder"]["streams"].values())
    result["passed"] = all(item["acquisition"]["passed"] for item in episodes)
    (args.root / "hardware_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.rate_check and not all(item["acquisition"]["passed"] for item in episodes):
        raise RuntimeError("X518 rate acceptance failed; Dataset and diagnostic report retained.")


if __name__ == "__main__":
    main()
