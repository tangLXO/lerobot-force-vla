#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.sensorized_robot import SensorizedRobot
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record, record_loop
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.sensors import Sensor, SensorConfig, SensorFeature
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STATE, OBS_TACTILE
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def _ticks(summary: str) -> int:
    """Sample size out of a cadence report — every other number is an average over it."""
    return int(re.search(r"(\d+) ticks", summary).group(1))


def _step_calls(summary: str, step: str) -> int:
    """How many ticks ran *step*, off the loop-body breakdown of a run summary."""
    return int(re.search(rf"\n\s+{step}\s+.*· (\d+) calls", summary).group(1))


class _RecorderFailureSensor(Sensor):
    """Optional raw-only sensor used to exercise recorder abort cleanup."""

    def __init__(self) -> None:
        super().__init__(SensorConfig(expected_sample_rate_hz=100, required=False, frame_features=[]))
        self.connected = False

    @property
    def features(self) -> dict[str, SensorFeature]:
        return {"probe.normal_force": SensorFeature("float32", "N")}

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False


class _FrameForceSensor(Sensor):
    """Deterministic current-force source for record schema tests."""

    def __init__(self) -> None:
        super().__init__(SensorConfig(expected_sample_rate_hz=100, max_age_ms=1000))
        self.connected = False

    @property
    def features(self) -> dict[str, SensorFeature]:
        return {
            "left.normal_force": SensorFeature("float32", "N"),
            "right.normal_force": SensorFeature("float32", "N"),
        }

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True
        now_ns = time.perf_counter_ns()
        self._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 2.0},
            now_ns,
            arrival_timestamp_ns=now_ns,
        )

    def disconnect(self) -> None:
        self.connected = False


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_dataset_compatibility_check_accepts_read_only_metadata() -> None:
    robot = make_robot_from_config(MockRobotConfig(n_motors=1))
    expected_features = {
        OBS_STATE: {"dtype": "float32", "shape": (1,), "names": ["motor_1.pos"]},
        ACTION: {"dtype": "float32", "shape": (1,), "names": ["motor_1.pos"]},
    }
    metadata = type(
        "MetadataView",
        (),
        {
            "robot_type": robot.robot_type,
            "fps": 30,
            "features": {**expected_features, **DEFAULT_FEATURES},
        },
    )()

    sanity_check_dataset_robot_compatibility(metadata, robot, 30, expected_features)


def test_teleoperate(cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)

    # A teleop session has no episodes, so there is one cadence block for the whole run,
    # and the steps it names are the ones the loop wraps.
    (summary,) = cadence_log
    assert summary.startswith("Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick):")
    for step in ("observe", "teleop", "send"):
        assert step in summary, step


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_routes_current_force_to_tactile_without_state_duplication(tmp_path) -> None:
    inner_robot = make_robot_from_config(MockRobotConfig(random_values=False, static_values=[0.1, 0.2, 0.3]))
    sensorized_robot = SensorizedRobot(inner_robot, {"gripper_force": _FrameForceSensor()})
    recorder = MagicMock()
    recorder.commit_prepared.side_effect = lambda dataset: dataset.save_episode()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_tactile",
        num_episodes=1,
        episode_time_s=0.05,
        reset_time_s=0,
        push_to_hub=False,
        video=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset_cfg,
        teleop=MockTeleopConfig(random_values=False, static_values=[0.1, 0.2, 0.3]),
        play_sounds=False,
    )

    with (
        patch("lerobot.scripts.lerobot_record.attach_sensors", return_value=sensorized_robot),
        patch("lerobot.scripts.lerobot_record.SensorStreamRecorder", return_value=recorder),
        patch(
            "lerobot.scripts.lerobot_record.init_keyboard_listener",
            return_value=(
                None,
                {"stop_recording": False, "exit_early": False, "rerecord_episode": False},
            ),
        ),
    ):
        original_send_action = sensorized_robot.send_action

        def mutate_capture_after_observation(action):
            sent = original_send_action(action)
            sensorized_robot.last_capture_metadata["sensors"]["gripper_force"]["sequence"] = 999
            return sent

        sensorized_robot.send_action = mutate_capture_after_observation
        dataset = record(cfg)

    assert dataset.features[OBS_STATE]["shape"] == (3,)
    assert dataset.features[OBS_TACTILE]["shape"] == (2,)
    assert set(dataset.features[OBS_STATE]["names"]).isdisjoint(dataset.features[OBS_TACTILE]["names"])
    item = dataset[0]
    assert item[OBS_STATE].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert item[OBS_TACTILE].tolist() == pytest.approx([1.0, 2.0])
    captures = [call.args[1] for call in recorder.record_sync.call_args_list]
    assert captures
    assert all(capture["sensors"]["gripper_force"]["sequence"] == 0 for capture in captures)


def test_record_rejects_v1_sidecar_before_dataset_resume(tmp_path) -> None:
    root = tmp_path / "v1_resume"
    manifest_path = root / "meta" / "sensor_streams.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"sidecar_schema_version": 1}), encoding="utf-8")
    sensorized_robot = SensorizedRobot(
        make_robot_from_config(MockRobotConfig()), {"gripper_force": _FrameForceSensor()}
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=root,
            num_episodes=1,
            episode_time_s=0.01,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        resume=True,
    )

    with (
        patch("lerobot.scripts.lerobot_record.attach_sensors", return_value=sensorized_robot),
        patch(
            "lerobot.scripts.lerobot_record.LeRobotDatasetMetadata",
            return_value=MagicMock(root=root),
        ),
        patch("lerobot.scripts.lerobot_record.LeRobotDataset.resume") as resume,
        pytest.raises(ValueError, match="Record into a new Dataset root"),
    ):
        record(cfg)

    resume.assert_not_called()


def test_record_rejects_sidecar_resume_without_configured_sensors(tmp_path) -> None:
    root = tmp_path / "v2_resume_without_sensors"
    manifest_path = root / "meta" / "sensor_streams.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"sidecar_schema_version": 2}), encoding="utf-8")
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=root,
            num_episodes=1,
            episode_time_s=0.01,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        resume=True,
    )

    with (
        patch(
            "lerobot.scripts.lerobot_record.LeRobotDatasetMetadata",
            return_value=MagicMock(root=root),
        ),
        patch("lerobot.scripts.lerobot_record.LeRobotDataset.resume") as resume,
        pytest.raises(ValueError, match="without the configured sensors"),
    ):
        record(cfg)

    resume.assert_not_called()


def test_record_rejects_adding_sidecar_during_dataset_resume(tmp_path) -> None:
    root = tmp_path / "resume_without_existing_sidecar"
    sensorized_robot = SensorizedRobot(
        make_robot_from_config(MockRobotConfig()), {"gripper_force": _FrameForceSensor()}
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=root,
            num_episodes=1,
            episode_time_s=0.01,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        resume=True,
    )

    with (
        patch("lerobot.scripts.lerobot_record.attach_sensors", return_value=sensorized_robot),
        patch(
            "lerobot.scripts.lerobot_record.LeRobotDatasetMetadata",
            return_value=MagicMock(root=root),
        ),
        patch("lerobot.scripts.lerobot_record.LeRobotDataset.resume") as resume,
        pytest.raises(ValueError, match="Cannot add a Sensor Sidecar.*new Dataset root"),
    ):
        record(cfg)

    resume.assert_not_called()


def test_record_materializes_hub_metadata_before_sidecar_resume_preflight(tmp_path) -> None:
    root = tmp_path / "hub_v2_resume"
    sensorized_robot = SensorizedRobot(
        make_robot_from_config(MockRobotConfig()), {"gripper_force": _FrameForceSensor()}
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=root,
            num_episodes=1,
            episode_time_s=0.01,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        resume=True,
    )

    def materialize_metadata(*_args, **_kwargs):
        manifest_path = root / "meta" / "sensor_streams.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"sidecar_schema_version": 2}), encoding="utf-8")
        return MagicMock(root=root)

    with (
        patch("lerobot.scripts.lerobot_record.attach_sensors", return_value=sensorized_robot),
        patch(
            "lerobot.scripts.lerobot_record.LeRobotDatasetMetadata",
            side_effect=materialize_metadata,
        ) as metadata_loader,
        patch(
            "lerobot.scripts.lerobot_record.sanity_check_dataset_robot_compatibility",
            side_effect=ValueError("synthetic main schema mismatch"),
        ),
        patch("lerobot.scripts.lerobot_record.LeRobotDataset.resume") as resume,
        pytest.raises(ValueError, match="synthetic main schema mismatch"),
    ):
        record(cfg)

    metadata_loader.assert_called_once()
    resume.assert_not_called()


def test_record_and_replay(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

    # Replay has to hit the dataset's frame rate or the trajectory plays back at the
    # wrong speed, so it reports its cadence like every other loop.  Its block is the
    # last one and names its own steps.
    assert cadence_log[-1].startswith("Cadence summary — whole run · target 30 Hz")
    assert "read_frame" in cadence_log[-1]


def test_record_reports_a_cadence_summary_per_episode_and_for_the_run(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "cadence",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    record(cfg)

    assert len(cadence_log) == 3
    per_episode, run = cadence_log[:2], cadence_log[2]
    assert [m.split(":")[0] for m in per_episode] == ["Cadence (episode 0)", "Cadence (episode 1)"]
    assert run.startswith("Cadence summary — whole run, 2 episodes")
    # Windows partition the session, so the episodes account for every tick of the run...
    assert _ticks(run) == sum(_ticks(m) for m in per_episode)
    # ...and every one of those ticks wrote a frame.  The reset phase paces at the same
    # fps but records nothing, so it runs on its own timer rather than diluting the
    # numbers that answer "did I record at `fps`?".
    assert _step_calls(run, "record") == _step_calls(run, "observe") == _ticks(run)


def test_record_forwards_compressed_images_setting_to_reset_phase(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "compressed_images",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        display_compressed_images=True,
        play_sounds=False,
    )

    with patch("lerobot.scripts.lerobot_record.record_loop", wraps=record_loop) as mock_record_loop:
        record(cfg)

    # Recording episode 0, resetting, then recording episode 1 should all use the same
    # image representation so visualization backends do not receive mixed message types.
    assert [
        call.kwargs.get("display_compressed_images", False) for call in mock_record_loop.call_args_list
    ] == [True, True, True]


def test_record_loop_without_a_teleoperator_paces_and_terminates():
    # Regression: the no-teleop branch used to `continue` past both the pacing sleep and
    # the `timestamp` update, so a reset phase with no teleop device spun as fast as the
    # CPU allowed and never reached `control_time_s` at all.
    robot = make_robot_from_config(MockRobotConfig())
    robot.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    calls = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal calls
        calls += 1
        assert calls <= 20, "loop is spinning: 20 iterations of a 0.1 s phase at 30 Hz"
        return real_get_observation()

    robot.get_observation = counted_get_observation

    try:
        record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=None,
            control_time_s=0.1,
        )
    finally:
        robot.disconnect()

    # 0.1 s at 30 Hz is 3 ticks; the upper bound is what proves the phase was paced.
    assert 1 <= calls <= 6


def test_record_logs_and_aborts_sensor_episode_on_pre_save_failure(tmp_path):
    sensor = _RecorderFailureSensor()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "pre_save_failure",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset_cfg,
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )

    def attach_failure_sensor(robot, _configs):
        return SensorizedRobot(robot, {"failure_probe": sensor})

    with (
        patch("lerobot.scripts.lerobot_record.logging.exception") as log_exception,
        patch("lerobot.scripts.lerobot_record.attach_sensors", side_effect=attach_failure_sensor),
        patch(
            "lerobot.scripts.lerobot_record.init_keyboard_listener",
            return_value=(
                None,
                {"stop_recording": False, "exit_early": False, "rerecord_episode": False},
            ),
        ),
        patch(
            "lerobot.scripts.lerobot_record.record_loop",
            side_effect=RuntimeError("synthetic pre-save failure"),
        ),
        pytest.raises(RuntimeError, match="synthetic pre-save failure"),
    ):
        record(cfg)

    log_exception.assert_called_once_with("Recording failed before main save; aborting the current episode.")
    journal_paths = list((dataset_cfg.root / "meta" / "sensor_transactions").glob("*.json"))
    assert len(journal_paths) == 1
    assert json.loads(journal_paths[0].read_text())["state"] == "ABORTED"
    assert not sensor.is_connected
    assert not (dataset_cfg.root / ".sensor-writer.lock").exists()
