"""Offline recording form validation and episode-only UI request semantics."""

import importlib
import json
import uuid
from types import SimpleNamespace

import pytest


@pytest.fixture
def recording(panel):
    return importlib.import_module("b601_tuner_recording")


@pytest.fixture
def recording_values(tmp_path):
    return {
        "root": str(tmp_path / "dataset"),
        "repo_id": "local/force_teacher_demo",
        "task": "pick and place",
        "object_id": "block",
        "contact_condition_id": "rubber",
        "fps": "30",
        "camera_max_age_ms": "100",
        "queue_capacity": "128",
        "target_episode_count": "10",
        "resume": False,
    }


def test_recording_form_builds_explicit_json_config_without_creating_directory(recording, recording_values):
    from lerobot.force_control.demonstration import TeacherRecordConfig

    cameras = {"front": {"type": "opencv", "index_or_path": 0, "fps": 30, "width": 640, "height": 480}}
    result = recording.recording_config(recording_values, json.dumps(cameras))
    assert result["camera_configs"] == cameras
    assert result["fps"] == 30 and result["target_episode_count"] == 10
    assert result["resume"] is False and result["object_id"] == "block"
    TeacherRecordConfig(**result).validate()
    from pathlib import Path

    assert not Path(result["root"]).exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", ""),
        ("object_id", " "),
        ("fps", "nan"),
        ("camera_max_age_ms", "0"),
        ("queue_capacity", "1.5"),
        ("target_episode_count", "-1"),
    ],
)
def test_recording_form_rejects_missing_identity_and_invalid_limits(recording, recording_values, key, value):
    recording_values[key] = value
    with pytest.raises(ValueError):
        recording.recording_config(recording_values, '{"front":{"type":"opencv"}}')


@pytest.mark.parametrize("cameras", ["{}", "[]", '{"front":0}', '{"front":{"fps":NaN}}'])
def test_recording_form_requires_explicit_finite_camera_configs(recording, recording_values, cameras):
    with pytest.raises(ValueError):
        recording.recording_config(recording_values, cameras)


def test_finish_and_discard_submit_episode_requests_without_motion(panel):
    app = panel.Tuner.__new__(panel.Tuner)
    events = []
    app.running, app.finishing = True, False
    app.live = SimpleNamespace(stop=panel.threading.Event(), submit=lambda *event: events.append(event))
    app.status = SimpleNamespace(set=lambda value: None)
    app.recording_page = SimpleNamespace(
        task_result=SimpleNamespace(get=lambda: "成功"), config=lambda: {"task": "pick"}
    )
    app.recording_command("record_start")
    app.recording_command("record_finish")
    app.recording_command("record_discard")
    app.recording_command("record_outcome")
    assert events == [
        ("record_start", {"config": {"task": "pick"}}),
        ("record_finish", {"task_result": "success"}),
        ("record_discard", {"rerecord": True}),
        ("record_outcome", {"task_result": "success"}),
    ]
    assert not app.live.stop.is_set() and not app.finishing
    app.live.stop.set()
    app.recording_command("record_start")
    assert len(events) == 4


def test_record_start_without_active_control_never_enables(panel):
    app = panel.Tuner.__new__(panel.Tuner)
    app.running, app.live, app.finishing = False, None, False
    messages = []
    app.status = SimpleNamespace(set=messages.append)
    app.start = lambda: pytest.fail("record button enabled hardware")
    app.recording_command("record_start")
    assert "不会使能" in messages[-1]


def test_profile_pin_rejects_even_semantically_equal_file_edits(panel, tmp_path):
    review = importlib.import_module("b601_profile_review")
    path = tmp_path / "profile.json"
    path.write_text('{"same":1}', encoding="utf-8")
    pin = review.profile_fingerprint(path)
    review.require_profile_pin(path, pin)
    path.write_text('{"same":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="重新预检"):
        review.require_profile_pin(path, pin)


def test_typed_validation_summary_requires_explicit_one_newton_binding(panel):
    from lerobot.robots.rebot_b601_follower.full_arm_validation import FullArmValidationConfig

    review = importlib.import_module("b601_profile_review")
    capability = FullArmValidationConfig(
        session_id=str(uuid.uuid4()),
        authorization_id="synthetic_only",
        port="FAKE",
        calibration_file="synthetic-calibration.json",
        calibration_sha256="a" * 64,
        profile_sha256=(),
        micro_trials=(),
        supported_arm=True,
        stop_policy="disable_all_supported_arm",
        reference_envelope_deg=100,
        alignment_tolerance_deg=5,
        max_duration_s=30,
        watchdog_ticks=2000,
        target_force_N=1,
        leader_config_sha256="b" * 64,
    )
    config = SimpleNamespace(
        b601=SimpleNamespace(validation=capability, startup_mode="prepare_disabled_full_arm_validation"),
        teacher=SimpleNamespace(force_target_N=1),
    )
    summary = review.bounded_validation_summary(config)
    assert summary["target_force_N"] == 1 and summary["production_arm_authorized"] is False
    assert "非正式全臂验收" in review.bounded_validation_text(config)
    config.teacher.force_target_N = 2
    with pytest.raises(ValueError, match="1 N"):
        review.bounded_validation_summary(config)


def test_recording_description_separates_success_counts_file_completeness_and_stages(recording):
    text = recording.recording_description(
        {
            "state": "saved",
            "episode_count": 5,
            "successful_episode_count": 3,
            "file_complete": True,
            "task_result": "unknown",
            "stage_counts": {"approach": 100, "contact_transition": 4, "force_control": 200, "release": 40},
        }
    )
    assert "已保存 5 条" in text and "人工成功 3 / 10" in text
    assert "文件完整：True" in text and "人工任务结果：unknown" in text
    for label in ("接近 100", "接触交接 4", "力控 200", "释放 / 人工张开 40"):
        assert label in text
    request_error = recording.recording_description(
        {
            "state": "recording",
            "error": None,
            "request_error": "duplicate start",
        }
    )
    assert "采集错误：无" in request_error and "操作请求错误：duplicate start" in request_error


def test_profile_check_preserves_config_preflight_report_when_motion_gate_fails(recording, monkeypatch):
    page = recording.RecordingPage.__new__(recording.RecordingPage)
    messages = []
    page.profile_path = SimpleNamespace(get=lambda: "explicit-profile.json")
    page.profile_status = SimpleNamespace(set=messages.append)

    def gate():
        raise RuntimeError("hardware acceptance is incomplete")

    config = SimpleNamespace(
        teacher=SimpleNamespace(force_target_N=4.5),
        safety=object(),
        runtime=object(),
        validate=gate,
        b601=SimpleNamespace(
            preflight_report=lambda *args: {
                "configuration_valid": True,
                "motion_authorized": False,
            }
        ),
    )
    monkeypatch.setattr(recording, "load_full_arm_profile", lambda path: config)
    monkeypatch.setattr(recording, "profile_fingerprint", lambda path: {"sha256": "a" * 64})
    page.check_profile()
    assert "4.5 N" in messages[-1] and '"configuration_valid": true' in messages[-1]
    assert "hardware acceptance is incomplete" in messages[-1]


def test_full_arm_template_never_invents_physical_gains_or_enables(panel):
    builder = importlib.import_module("build_full_arm_profile_template")
    document = builder.profile_template()
    assert document["allow_motion"] is False and document["explicit_arm"] is False
    assert document["teacher"]["force_target_N"] is None
    assert document["teacher"]["kp_deg_per_N"] is None
    assert document["teacher"]["ki_deg_per_N_s"] is None
    assert document["sensors"] == {} and document["b601"]["control"]["calibration_file"] is None
    profiles = document["b601"]["control"]["profiles"]
    assert len(profiles) == 7 and profiles[-1]["mode"] == "force_pos"
    assert all(profile["mode"] == "pos_vel" for profile in profiles[:-1])
    for profile in profiles:
        assert profile["minimum_deg"] is None and profile["maximum_deg"] is None
        assert profile["max_relative_deg"] is None and profile["velocity_deg_s"] is None


def test_new_episode_requires_fresh_manual_result_confirmation(recording):
    page = recording.RecordingPage.__new__(recording.RecordingPage)
    selected = ["成功"]
    page.last_episode_uid = "previous"
    page.fields = {"target_episode_count": SimpleNamespace(get=lambda: "10")}
    page.task_result = SimpleNamespace(set=lambda value: selected.__setitem__(0, value))
    page.status = SimpleNamespace(set=lambda value: None)
    page.buttons = {}
    page.update({"state": "recording", "active": True, "episode_uid": "next"}, active=True)
    assert selected == ["未确认"]
    selected[0] = "成功"
    page.update({"state": "recording", "active": True, "episode_uid": "next"}, active=True)
    assert selected == ["成功"]


def test_full_arm_profile_rejects_second_recording_configuration(recording, tmp_path):
    from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS

    path = tmp_path / "profile.json"
    document = {
        "runtime": {"joint_names": [joint.name for joint in DEFAULT_B601_JOINTS]},
        "initial_mode": "hybrid",
        "recording": {"root": str(tmp_path / "dataset"), "task": "synthetic"},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="示范录制页"):
        recording.load_full_arm_profile(path)


def test_outcome_confirmation_available_during_capture_and_after_complete_save(recording):
    page = recording.RecordingPage.__new__(recording.RecordingPage)
    page.fields = {"target_episode_count": SimpleNamespace(get=lambda: "10")}
    page.status = SimpleNamespace(set=lambda value: None)
    page.enabled = SimpleNamespace(get=lambda: True)
    states = {}
    page.buttons = {
        kind: SimpleNamespace(configure=lambda key=kind, **kwargs: states.update({key: kwargs["state"]}))
        for kind in ("record_start", "record_finish", "record_discard", "record_outcome")
    }
    page.update({"state": "recording", "active": True}, active=True)
    assert states == {
        "record_start": "disabled",
        "record_finish": "normal",
        "record_discard": "normal",
        "record_outcome": "normal",
    }
    page.update({"state": "saved", "active": False, "file_complete": True}, active=True)
    assert states["record_outcome"] == "normal" and states["record_start"] == "normal"
    page.update({"state": "saving", "active": True}, active=True)
    assert set(states.values()) == {"disabled"}
