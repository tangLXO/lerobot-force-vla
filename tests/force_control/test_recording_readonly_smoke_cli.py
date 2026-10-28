"""The combined read-only CLI must reject missing intent before opening I/O."""

import ast
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="Readonly recording integration requires lerobot[dataset]")
pytest.importorskip("pyarrow", reason="Readonly recording integration requires lerobot[dataset]")
pytest.importorskip("av", reason="Readonly recording video integration requires lerobot[dataset]")


@pytest.fixture
def smoke_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/force_teacher/recording_readonly_hardware_smoke_test.py"
    )
    spec = importlib.util.spec_from_file_location("recording_readonly_hardware_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extra", [[], ["--observe-only", "--duration-s", "0"]])
def test_cli_rejects_missing_readonly_intent_or_invalid_duration_before_configuration_or_io(
    smoke_module, tmp_path, monkeypatch, extra
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Rejected CLI invocation must not decode configuration or construct transport")

    monkeypatch.setattr(smoke_module.draccus, "decode", forbidden)
    monkeypatch.setattr(smoke_module, "ReadonlyB601Transport", forbidden)
    with pytest.raises(SystemExit) as error:
        smoke_module.main(
            [
                "--config",
                "unused.json",
                "--recording-config",
                "unused.json",
                "--root",
                str(tmp_path / "new"),
                *extra,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


def test_existing_output_is_never_overwritten(smoke_module, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Existing root must be rejected before opening hardware")

    monkeypatch.setattr(smoke_module, "ReadonlyB601Transport", forbidden)
    with pytest.raises(SystemExit) as error:
        smoke_module.main(
            [
                "--config",
                "unused.json",
                "--recording-config",
                "unused.json",
                "--root",
                str(tmp_path),
                "--observe-only",
            ]
        )
    assert error.value.code == 2


def test_readonly_and_motion_profile_sources_are_mutually_exclusive(smoke_module, tmp_path):
    with pytest.raises(SystemExit) as error:
        smoke_module.main(
            [
                "--config",
                "legacy.json",
                "--readonly-config",
                "readonly.json",
                "--recording-config",
                "cameras.json",
                "--root",
                str(tmp_path / "new"),
                "--observe-only",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", ""),
        ("baudrate", True),
        ("calibration_file", ""),
        ("startup_budget_s", 0),
        ("feedback_budget_ms", float("nan")),
        ("joint_feedback_max_age_ms", True),
        ("force_max_age_ms", -1),
        ("force_instance", ""),
    ],
)
def test_minimal_readonly_config_rejects_invalid_settings_without_transport(
    smoke_module, field, value, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Readonly settings validation must not construct hardware")

    monkeypatch.setattr(smoke_module, "ReadonlyB601Transport", forbidden)
    values = {
        "port": "FAKE",
        "baudrate": 921600,
        "calibration_file": "unused.json",
        "sensors": {},
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        smoke_module.ReadonlyCaptureConfig(**values).validate()


def test_minimal_readonly_schema_does_not_accept_motion_authorization(smoke_module):
    with pytest.raises(Exception, match="allow_motion"):
        smoke_module.draccus.decode(
            smoke_module.ReadonlyCaptureConfig,
            {
                "port": "FAKE",
                "baudrate": 921600,
                "calibration_file": "unused.json",
                "sensors": {},
                "allow_motion": True,
            },
        )


def test_transport_calls_are_restricted_to_query_and_cleanup_methods(smoke_module):
    source = Path(smoke_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "device"
    }
    assert calls == {"open", "read_mapping", "read_feedback", "close"}


def test_observation_only_real_artifacts_roundtrip_without_hardware(smoke_module, tmp_path, monkeypatch):
    import av
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sensor_stream import SensorStreamRecorder
    from tests.force_control.test_demonstration_validation import ManualSensor

    root = tmp_path / "readonly"
    names = smoke_module.JOINT_NAMES
    dataset = LeRobotDataset.create(
        repo_id="test/readonly-combined",
        root=root,
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"{name}.pos" for name in names],
            },
            "observation.tactile": {
                "dtype": "float32",
                "shape": (2,),
                "names": [f"sensor.gripper_force.{name}.normal_force" for name in ("left", "right")],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            },
        },
    )

    def encode(_key, _index):
        path = root / "encoded" / "front.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for _ in range(3):
                frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    monkeypatch.setattr(dataset.writer, "_encode_temporary_episode_video", encode)
    sensor = ManualSensor()
    recorder = SensorStreamRecorder(root, {"gripper_force": sensor})
    uid = recorder.start_episode(0, dataset=dataset)
    frames = []
    for index in range(3):
        timestamp = 1_000_000_000 + index * 40_000_000
        anchor = timestamp + 20
        sample = sensor._publish_sample(
            {"left.normal_force": 1.0, "right.normal_force": 2.0},
            timestamp,
            arrival_timestamp_ns=timestamp + 1,
        )
        state = [float(index)] * 7
        dataset.add_frame(
            {
                "observation.state": np.asarray(state, dtype=np.float32),
                "observation.tactile": np.asarray([1.0, 2.0], dtype=np.float32),
                "observation.images.front": np.zeros((16, 16, 3), dtype=np.uint8),
                "task": "readonly",
            }
        )
        recorder.record_sync(
            index,
            {
                "observation_start_ns": anchor - 1,
                "frame_anchor_ns": anchor,
                "observation_complete_ns": anchor + 1,
                "sensors": {
                    "gripper_force": {
                        "timestamp_ns": timestamp,
                        "arrival_timestamp_ns": timestamp + 1,
                        "sequence": sample.sequence,
                        "hardware_sequence": None,
                        "age_ns": 20,
                        "status": sample.status,
                    }
                },
            },
        )
        frames.append(
            {
                "frame_index": index,
                "frame_anchor_ns": anchor,
                "state_deg": state,
                "force_sequence": sample.sequence,
                "cameras": {"front": {"timestamp_ns": timestamp - 1, "arrival_timestamp_ns": timestamp}},
            }
        )
    recorder.save_episode(dataset, task_info=["readonly"])
    recorder.close()
    dataset.finalize()

    def tree():
        return {
            str(path.relative_to(root)): (
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = tree()
    assert smoke_module._verify(root, uid, frames)["valid"]
    assert tree() == before
    frames[1]["cameras"]["front"]["arrival_timestamp_ns"] = frames[1]["frame_anchor_ns"] + 1
    with pytest.raises(ValueError, match="camera selection is noncausal"):
        smoke_module._verify(root, uid, frames)


@pytest.mark.parametrize("minimal", [False, True])
def test_full_readonly_cli_roundtrip_uses_only_query_transport_with_real_artifacts(
    smoke_module, tmp_path, monkeypatch, minimal
):
    import pyarrow.parquet as pq

    from lerobot.force_control.fake_hardware import FakeForceSensor, FakePlant
    from lerobot.robots.rebot_b601_follower.readonly_transport import B601ReadonlyBatch, B601ReceivedFeedback

    camera_path = Path(smoke_module.__file__).with_name("demonstration_smoke_test.py")
    spec = importlib.util.spec_from_file_location("synthetic_demo_for_readonly_test", camera_path)
    synthetic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synthetic)
    sensor = FakeForceSensor(FakePlant(), sample_rate_hz=400)
    calibration = SimpleNamespace(require_unchanged=lambda: None)
    config = SimpleNamespace(
        backend="b601",
        dry_run=False,
        b601=SimpleNamespace(
            port="FAKE_QUERY_ONLY",
            joint_specs=(),
            baudrate=921600,
            control=SimpleNamespace(startup_budget_s=0.1),
            preflight_report=lambda *_: {
                "configuration_valid": True,
                "motion_authorized": False,
                "errors": [],
            },
            validate_arm_profile=lambda: calibration,
        ),
        runtime=SimpleNamespace(
            joint_names=smoke_module.JOINT_NAMES,
            force_instance="gripper_force",
            feedback_budget_ms=15,
            left_feature="left.normal_force",
            right_feature="right.normal_force",
        ),
        safety=SimpleNamespace(force_max_age_ms=100, joint_feedback_max_age_ms=100),
        sensors={"gripper_force": object()},
    )
    calls = []

    class QueryOnlyDevice:
        def __init__(self, *_args, **_kwargs):
            self.has_resources = False
            self.sequence = 0

        def open(self):
            calls.append("open")
            self.has_resources = True

        def read_mapping(self, **_kwargs):
            calls.append("read_mapping")
            return self._batch("mapping")

        def read_feedback(self, **_kwargs):
            calls.append("read_feedback")
            return self._batch("feedback")

        def _batch(self, kind):
            start = time.perf_counter_ns()
            self.sequence += 1
            joints = tuple(
                B601ReceivedFeedback(
                    name,
                    index,
                    0x10 + index,
                    "4310",
                    0,
                    "DISABLED",
                    math.radians(index),
                    float(index),
                    0.0,
                    0.0,
                    20,
                    20,
                    start,
                    self.sequence,
                    "aa",
                    "00",
                    (1.0, 1.0, 1.0),
                )
                for index, name in enumerate(smoke_module.JOINT_NAMES, 1)
            )
            return B601ReadonlyBatch(kind, start, time.perf_counter_ns(), start + 15_000_000, feedback=joints)

        def close(self):
            calls.append("close")
            self.has_resources = False

    if not minimal:
        monkeypatch.setattr(smoke_module.draccus, "decode", lambda *_: config)
        monkeypatch.setattr(smoke_module, "validate_b601_sensor", lambda *_: None)
    monkeypatch.setattr(smoke_module, "make_sensors_from_configs", lambda *_: {"gripper_force": sensor})
    monkeypatch.setattr(smoke_module, "CameraCapture", synthetic.SyntheticCameras)
    monkeypatch.setattr(smoke_module, "ReadonlyB601Transport", QueryOnlyDevice)
    config_file, camera_file = tmp_path / "profile.json", tmp_path / "recording.json"
    if minimal:
        calibration_path = tmp_path / "native-arm.json"
        calibration_path.write_text(
            json.dumps(
                {
                    name: {
                        "id": index,
                        "drive_mode": 0,
                        "homing_offset": 0,
                        "range_min": -150,
                        "range_max": 150,
                    }
                    for index, name in enumerate(smoke_module.JOINT_NAMES, 1)
                }
            ),
            encoding="utf-8",
        )
        config_file.write_text(
            json.dumps(
                {
                    "port": "FAKE_QUERY_ONLY",
                    "baudrate": 921600,
                    "calibration_file": str(calibration_path),
                    "sensors": {
                        "gripper_force": {
                            "type": "x518",
                            "sample_rate_hz": 400,
                            "max_age_ms": 100,
                            "channels": {
                                "left.normal_force": {"channel": 1},
                                "right.normal_force": {"channel": 2},
                            },
                        }
                    },
                    "startup_budget_s": 0.1,
                    "feedback_budget_ms": 15,
                    "joint_feedback_max_age_ms": 100,
                    "force_max_age_ms": 100,
                }
            ),
            encoding="utf-8",
        )
    else:
        config_file.write_text("{}", encoding="utf-8")
    camera_file.write_text(
        json.dumps(
            {
                "task": "Readonly synthetic integration",
                "repo_id": "test/readonly-cli",
                "h264_encoder": "software",
                "camera_configs": {
                    "front": {"type": "opencv", "index_or_path": 0, "fps": 60, "width": 64, "height": 48}
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "capture"
    assert (
        smoke_module.main(
            [
                "--readonly-config" if minimal else "--config",
                str(config_file),
                "--recording-config",
                str(camera_file),
                "--root",
                str(root),
                "--duration-s",
                "0.12",
                "--observe-only",
            ]
        )
        == 0
    )
    report = json.loads((root / "readonly_capture_report.json").read_text(encoding="utf-8"))
    assert report["complete"] and report["verification"]["valid"]
    assert report["motion_attempted"] is False and report["action_labels"] is False
    assert report["transport_closed"] and report["sensor_closed"] and not sensor.has_resources
    assert calls[:2] == ["open", "read_mapping"] and calls[-1] == "close"
    assert set(calls) == {"open", "read_mapping", "read_feedback", "close"}
    for path in (root / "data").rglob("*.parquet"):
        assert "action" not in pq.read_schema(path).names
