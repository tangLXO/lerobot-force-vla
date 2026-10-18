"""Typed bounded validation reaches the original panel review path without physical I/O."""

import importlib
import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.force_control.b601_factory import B601HardwareConfig
from lerobot.force_control.config import SafetyConfig, TeacherConfig
from lerobot.force_control.runtime import RuntimeConfig
from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from lerobot.robots.rebot_b601_follower.full_arm_validation import leader_configuration_digest
from lerobot.robots.rebot_b601_follower.readonly_transport import DEFAULT_B601_JOINTS
from lerobot.scripts.lerobot_force_teacher import ForceTeacherRunConfig, leader_configuration_projection
from lerobot.sensors.x518 import X518ChannelConfig, X518SensorConfig
from tests.force_control.full_arm_validation_helpers import make_full_validation_files


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def bounded_profile(panel, tmp_path):
    files = make_full_validation_files(tmp_path / "synthetic_admission")
    names = tuple(joint.name for joint in DEFAULT_B601_JOINTS)
    config = ForceTeacherRunConfig(
        backend="b601",
        dry_run=False,
        allow_motion=True,
        explicit_arm=True,
        initial_mode="hybrid",
        leader_backend="rebot_102",
        leader_port="FAKE_LEADER",
        leader_readonly_transport="protocol_v1",
        leader_read_budget_s=0.05,
        leader_joint_ids={name: index for index, name in enumerate(names)},
        leader_joint_directions=dict.fromkeys(names, 1),
        leader_joint_ranges={name: [-20.0, 20.0] for name in names},
        close_sign=1,
        b601=B601HardwareConfig(
            port=files.port,
            joint_specs=DEFAULT_B601_JOINTS,
            control=files.control,
            startup_mode="prepare_disabled_full_arm_validation",
            validation=files.validation,
        ),
        sensors={
            "gripper_force": X518SensorConfig(
                host="192.0.2.1",
                channels={
                    "left.normal_force": X518ChannelConfig(channel=2),
                    "right.normal_force": X518ChannelConfig(channel=1),
                },
                sample_rate_hz=400,
                max_age_ms=50,
            )
        },
        teacher=TeacherConfig.fake(gripper_follow_speed_deg_s=1, contact_force_threshold_N=0.5),
        safety=SafetyConfig.fake(
            q_min_deg=-40.0, q_max_deg=40.0, fault_action="stop", joint_feedback_max_age_ms=50
        ),
        runtime=RuntimeConfig(
            control_hz=20,
            feedback_budget_ms=10,
            tick_deadline_ms=50,
            joint_names=names,
            arm_intent_required=True,
            handover_tolerance_deg=5,
        ),
        duration_s=3.0,
        trace_root=tmp_path / "trace",
    )
    capability = replace(
        config.b601.validation,
        leader_config_sha256=leader_configuration_digest(leader_configuration_projection(config)),
    )
    config = replace(config, b601=replace(config.b601, validation=capability))
    config.validate()
    path = tmp_path / "bounded_profile.json"
    path.write_text(json.dumps(draccus.encode(config)), encoding="utf-8")
    return config, path, B601GripperCalibration.load(files.calibration_path)


def test_bounded_duration_keeps_startup_and_shutdown_margin(bounded_profile):
    config = bounded_profile[0]
    config = replace(config, duration_s=config.b601.validation.max_duration_s)
    with pytest.raises(ValueError, match="startup and shutdown margin"):
        config.validate()


def test_typed_capability_can_review_in_original_bench_without_hardware(
    panel, bounded_profile, monkeypatch, tmp_path
):
    bench = importlib.import_module("b601_supervised_bench")
    review = importlib.import_module("b601_profile_review")
    config, source, calibration = bounded_profile
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {
                "config": draccus.encode(config),
                "calibration": asdict(calibration),
                "full_arm_profile_file": str(source),
                "full_arm_profile_pin": review.profile_fingerprint(source),
                "full_arm_validation_evidence_pin": review.validation_evidence_fingerprint(config),
                "executable": True,
                "consumed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench, "run_runtime", lambda *args, **kwargs: pytest.fail("offline review ran motors")
    )
    result = bench.run(tmp_path / "no_runtime", reviewed_config=path)
    assert result["transport_opened"] is False
    assert result["bounded_validation"]["target_force_N"] == 1
    assert result["production_arm_authorized"] is False
    assert not (tmp_path / "no_runtime").exists()
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重新预检"):
        bench.load_reviewed(path)


def test_real_recording_page_session_options_pin_valid_typed_capability(panel, bounded_profile):
    recording = importlib.import_module("b601_tuner_recording")
    review = importlib.import_module("b601_profile_review")
    _, path, _ = bounded_profile
    page = recording.RecordingPage.__new__(recording.RecordingPage)
    page.enabled = Variable(False)
    page.full_arm_enabled = Variable(True)
    page.profile_path = Variable(str(path))
    page.checked_profile_pin = review.profile_fingerprint(path)
    page.checked_validation_evidence_pin = review.validation_evidence_fingerprint(bounded_profile[0])
    options = page.session_options()
    assert options["full_arm_profile_file"] == str(path)
    assert options["full_arm_profile_pin"] == page.checked_profile_pin


@pytest.mark.parametrize("stale", [False, True])
def test_original_panel_start_reviews_typed_capability_and_queues_only_handoff(
    panel, bounded_profile, settings, monkeypatch, tmp_path, stale
):
    config, source, calibration = bounded_profile
    review = importlib.import_module("b601_profile_review")
    app = panel.Tuner.__new__(panel.Tuner)
    values, enabled = settings
    app.fields = {key: Variable(value) for key, value in values.items()}
    app.toggles = {key: Variable(value) for key, value in enabled.items()}
    app.running, app.monitoring, app.exit_confirmed = False, True, True
    app.monitor_port = config.b601.port
    app.confirmed = Variable(True)
    app.preview_latest = {
        "joint_received_ns": time.perf_counter_ns() - (1_000_000_000 if stale else 0),
        "phase": "DISABLED",
    }
    app.calibration, app.recovery_evidence = calibration, None
    app.monitor_stop = threading.Event()
    app.points = []
    app.settings_label, app.status = Variable(), Variable()
    app.recording_page = SimpleNamespace(
        session_options=lambda: {
            "full_arm_profile_file": str(source),
            "full_arm_profile_pin": review.profile_fingerprint(source),
            "full_arm_validation_evidence_pin": review.validation_evidence_fingerprint(config),
        }
    )
    app.update_timing_settings = lambda: None
    constructed = []
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "panel_runs")
    monkeypatch.setattr(
        panel,
        "ProcessLive",
        lambda **kwargs: constructed.append(kwargs) or SimpleNamespace(close=lambda: None),
    )
    panel.Tuner.start(app, "hybrid")
    if stale:
        assert not app.running and constructed == [] and "新鲜只读反馈" in app.status.get()
        return
    assert app.running and app.next_run is not None and len(constructed) == 1
    assert app.monitor_stop.is_set() and app.config.teacher.force_target_N == 1
    assert "非正式全臂验收" in app.status.get() and "1 N" in app.status.get()
    assert app.config.b601.validation == config.b601.validation


def test_ordinary_full_arm_profile_still_rejects_without_capability(bounded_profile):
    config, _, _ = bounded_profile
    ordinary = replace(config, b601=replace(config.b601, validation=None, startup_mode="verify_pre_enabled"))
    with pytest.raises(RuntimeError, match="acceptance"):
        ordinary.validate()


@pytest.mark.parametrize("artifact", ["events", "source_manifest"])
def test_child_review_rejects_logically_valid_evidence_changes_after_ui_preflight(
    panel, bounded_profile, tmp_path, artifact
):
    bench = importlib.import_module("b601_supervised_bench")
    review = importlib.import_module("b601_profile_review")
    config, source, calibration = bounded_profile
    path = tmp_path / "pinned_review.json"
    path.write_text(
        json.dumps(
            {
                "config": draccus.encode(config),
                "calibration": asdict(calibration),
                "full_arm_profile_file": str(source),
                "full_arm_profile_pin": review.profile_fingerprint(source),
                "full_arm_validation_evidence_pin": review.validation_evidence_fingerprint(config),
                "executable": True,
                "consumed": False,
            }
        ),
        encoding="utf-8",
    )
    root = Path(config.b601.validation.micro_trials[0].result_path).parent
    if artifact == "events":
        event_path = root / "events.jsonl"
        rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["additional_note"] = "logically valid, but changed after preflight"
        event_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        manifest = root / "source_snapshot/manifest.json"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["additional_note"] = "logically valid, but changed after preflight"
        manifest.write_text(json.dumps(document), encoding="utf-8")
    # Re-admission itself still passes; the preserved UI closure must reject it.
    config.validate()
    with pytest.raises(ValueError, match="引用证据与预检不一致"):
        bench.load_reviewed(path)
