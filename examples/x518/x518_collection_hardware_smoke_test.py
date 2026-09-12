"""Read-only SO101/X518/two-camera collection check. Actions are NOT executed.

The output is diagnostic data, not teleoperated training demonstrations.
Run with a new --root each time. No Hub upload or motor-register writes occur.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sensor_window import SensorStreamReader
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts import lerobot_record as entry
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import OBS_STATE, OBS_TACTILE


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify existing output without hardware access"
    )
    args = parser.parse_args()
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
                expected_sample_rate_hz=200,
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
            episode_time_s=3,
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

    def loop(*a, **kw):
        nonlocal discarded
        original_loop(*a, **kw)
        if kw.get("dataset") is not None and not discarded:
            kw["events"]["rerecord_episode"] = True
            discarded = True

    print("READ ONLY: no motion commands, no torque changes, no Hub upload.", flush=True)
    try:
        with (
            patch.object(entry, "make_robot_from_config", make_follower),
            patch.object(entry, "make_teleoperator_from_config", make_leader),
            patch.object(entry, "init_keyboard_listener", keyboard),
            patch.object(entry, "record_loop", loop),
        ):
            if not args.verify_only:
                entry.record(cfg)
                cfg.resume = True
                entry.record(cfg)
    finally:
        for arm in reversed(arms):
            arm.disconnect()
    data = LeRobotDataset(cfg.dataset.repo_id, root=args.root, video_backend="pyav")
    reader = SensorStreamReader(args.root, instance="gripper_force", verify="full")
    assert data.num_episodes == 2
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
    for episode in range(2):
        sync = reader.read_sync(episode_index=episode)
        anchors = np.array([row["frame_anchor_ns"] for row in sync], dtype=np.int64)
        ages = []
        for index, row in enumerate(sync):
            item = data[offset + index]
            for name in ("wrist", "front"):
                assert tuple(item[f"observation.images.{name}"].shape) == (3, 480, 640)
            window = reader.get_window(
                row["frame_anchor_ns"], 5, target_hz=200, max_age_ms=100, episode_index=episode
            )
            assert bool(window.valid_mask[-1])
            assert int(window.sequence[-1]) == row["sensors"]["gripper_force"]["sequence"]
            np.testing.assert_allclose(item[OBS_TACTILE].numpy(), window.values[-1])
            ages.append(float(window.age_ns[-1]) / 1e6)
        episodes.append(
            {
                "episode": episode,
                "frames": len(sync),
                "effective_anchor_hz": float((len(anchors) - 1) * 1e9 / (anchors[-1] - anchors[0])),
                "sensor_age_ms_p95": float(np.percentile(ages, 95)),
                "sensor_age_ms_max": max(ages),
            }
        )
        offset += len(sync)
    assert offset == len(data)
    journals = [json.loads(p.read_text()) for p in (args.root / "meta/sensor_transactions").glob("*.json")]
    assert sum(j["state"] == "COMMITTED" for j in journals) == 2
    assert sum(j["state"] == "ABORTED" for j in journals) == 1
    result = {
        "diagnostic_only": True,
        "motion_executed": False,
        "streaming": args.streaming,
        "rerecord": "passed",
        "resume": "passed",
        "full_verification": "passed",
        "all_video_frames_decoded": True,
        "state_is_robot_only": True,
        "tactile_matches_sync_reference": True,
        "force_history_is_sidecar_only": True,
        "episodes": episodes,
    }
    (args.root / "hardware_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
