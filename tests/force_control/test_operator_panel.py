"""Operator form/profile integration uses temporary files only, without physical I/O."""

import importlib
import json
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import draccus
import pytest

from lerobot.robots.rebot_b601_follower.calibration import B601GripperCalibration
from tests.force_control import test_operator_teleoperation_runner

normal_config = test_operator_teleoperation_runner.normal_config


@pytest.fixture
def source(panel, normal_config, tmp_path):
    normal_config.validate()
    path = tmp_path / "operator.json"
    path.write_text(json.dumps(draccus.encode(normal_config)), encoding="utf-8")
    return path


def editor(panel, source):
    module = importlib.import_module("b601_operator_panel")
    value = module.OperatorProfileEditor()
    value.load(source)
    return value


def tk_root(panel):
    try:
        root = panel.tk.Tk()
    except panel.tk.TclError as exc:
        pytest.skip(f"Tk display/runtime unavailable: {exc}")
    root.withdraw()
    return root


def test_operator_blank_choices_do_not_copy_native_or_diagnostic_defaults(panel, source):
    raw = json.loads(source.read_text())
    raw["b601"]["operator_teleoperation"] = None
    raw["b601"]["validation"] = {"max_duration_s": 30, "alignment_tolerance_deg": 5}
    raw["duration_s"] = raw["safety"]["max_run_time_s"] = 20
    raw["safety"]["max_total_travel_deg"] = 2
    source.write_text(json.dumps(raw))
    value = editor(panel, source)
    assert all(v == "" for v in value.velocities.values())
    assert all(v == "" for v in value.relative_limits.values())
    assert value.operator_values["alignment_tolerance_deg"] == ""
    document = value.document()
    assert document["duration_s"] is document["safety"]["max_run_time_s"] is None
    assert document["safety"]["max_total_travel_deg"] is None
    assert document["b601"]["operator_teleoperation"]["alignment_tolerance_deg"] is None
    assert document["b601"]["control"]["profiles"][0]["velocity_deg_s"] is None
    assert document["b601"]["control"]["profiles"][-1]["velocity_wire_code"] == 1000


def test_saved_normal_values_and_gripper_wire_code_are_preserved(panel, source, normal_config):
    value = editor(panel, source)
    assert value.missing() == []
    raw = value.document()
    decoded = draccus.decode(type(normal_config), raw)
    decoded.validate()
    assert decoded.b601.control.profiles[-1].velocity_wire_code == 1000
    assert (
        decoded.b601.control.profiles[-1].torque_ratio == normal_config.b601.control.profiles[-1].torque_ratio
    )
    assert decoded.teacher == normal_config.teacher
    assert decoded.sensors == normal_config.sensors
    assert decoded.duration_s is decoded.safety.max_run_time_s is decoded.safety.max_total_travel_deg is None


@pytest.mark.parametrize("malformed", ["false", 1])
def test_support_confirmation_requires_exact_json_true(panel, source, malformed):
    raw = json.loads(source.read_text())
    raw["b601"]["operator_teleoperation"]["supported_arm"] = malformed
    source.write_text(json.dumps(raw))
    value = editor(panel, source)
    assert value.supported_arm is False
    assert "机械臂支撑确认" in value.missing()
    assert value.document()["b601"]["operator_teleoperation"]["supported_arm"] is False


def test_explicit_zero_force_target_saves_and_passes_normal_preflight(panel, source, normal_config):
    value = editor(panel, source)
    value.target = "0"
    document = value.document()
    assert document["teacher"]["force_target_N"] == 0
    decoded = draccus.decode(type(normal_config), document)
    decoded.validate()
    assert decoded.b601.operator_teleoperation is not None
    assert decoded.teacher.force_target_N == 0


def test_one_visible_choice_updates_both_watchdog_and_leader_age(panel, source):
    value = editor(panel, source)
    value.operator_values.update(watchdog_ticks="1234", leader_max_age_s="0.4")
    value.velocities["gripper"] = "22"
    value.target = "3"
    value.duration_enabled, value.duration = True, "120"
    value.travel_enabled, value.travel = True, "400"
    raw = value.document()
    operator = raw["b601"]["operator_teleoperation"]
    assert operator["watchdog_ticks"] == raw["b601"]["control"]["watchdog_ticks"] == 1234
    assert operator["leader_max_age_s"] == raw["leader_max_age_s"] == 0.4
    gripper = raw["b601"]["control"]["profiles"][-1]
    assert gripper["velocity_deg_s"] == 22 and gripper["velocity_wire_code"] is None
    assert raw["teacher"]["force_target_N"] == 3
    assert raw["duration_s"] == raw["safety"]["max_run_time_s"] == 120
    assert raw["safety"]["max_total_travel_deg"] == 400


def test_normal_bench_review_requires_no_micro_files_and_pins_source(
    panel, source, normal_config, tmp_path, monkeypatch
):
    bench = importlib.import_module("b601_supervised_bench")
    review = importlib.import_module("b601_profile_review")
    calibration = B601GripperCalibration.load(normal_config.b601.control.calibration_file)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(
        json.dumps(
            {
                "config": draccus.encode(normal_config),
                "calibration": asdict(calibration),
                "full_arm_profile_file": str(source),
                "full_arm_profile_pin": review.profile_fingerprint(source),
            }
        )
    )
    monkeypatch.setattr(bench, "run_runtime", lambda *_a, **_k: pytest.fail("offline preflight ran runtime"))
    result = bench.run(tmp_path / "unused", reviewed_config=reviewed)
    assert result["transport_opened"] is False
    assert result["bounded_validation"] is None
    assert result["operator_teleoperation"]["target_force_N"] == 2
    assert "来源速度码 1000" in review.full_arm_profile_text(normal_config)
    assert "全七轴故障清理预算" in review.full_arm_profile_text(normal_config)
    assert "左右原始硬力阈值" in review.full_arm_profile_text(normal_config)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="已预检文件"):
        bench.load_reviewed(reviewed)


def test_original_panel_normal_start_only_queues_handoff(
    panel, source, normal_config, settings, monkeypatch, tmp_path
):
    review = importlib.import_module("b601_profile_review")
    values, toggles = settings

    def variable(value):
        return SimpleNamespace(get=lambda: value, set=lambda value: None)

    app = panel.Tuner.__new__(panel.Tuner)
    app.fields = {key: variable(value) for key, value in values.items()}
    app.toggles = {key: variable(value) for key, value in toggles.items()}
    app.running, app.monitoring, app.exit_confirmed = False, True, True
    app.monitor_port = normal_config.b601.port
    app.confirmed = variable(True)
    app.preview_latest = {"joint_received_ns": time.perf_counter_ns(), "phase": "DISABLED"}
    app.calibration = B601GripperCalibration.load(normal_config.b601.control.calibration_file)
    app.recovery_evidence, app.points = None, []
    app.monitor_stop = threading.Event()
    messages = []
    app.settings_label = app.status = SimpleNamespace(set=messages.append)
    app.recording_page = SimpleNamespace(
        session_options=lambda: {
            "full_arm_profile_file": str(source),
            "full_arm_profile_pin": review.profile_fingerprint(source),
        }
    )
    app.update_timing_settings = lambda: None
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "runs")
    monkeypatch.setattr(panel, "ProcessLive", lambda **_: SimpleNamespace(close=lambda: None))
    panel.Tuner.start(app, "hybrid")
    assert app.running and app.next_run is not None
    assert app.config.b601.operator_teleoperation is not None
    assert app.config.teacher.force_target_N == 2
    assert "正常全臂遥操" in messages[-1]


def test_apply_normal_force_target_is_only_a_request(panel, normal_config):
    app = panel.Tuner.__new__(panel.Tuner)
    app.running, app.finishing = True, False
    app.config = normal_config
    events, status = [], []
    app.live = SimpleNamespace(submit=lambda *args: events.append(args))
    app.status = SimpleNamespace(set=status.append)
    app.arm_page = SimpleNamespace(target=SimpleNamespace(get=lambda: "3"))
    app.apply_operator_target()
    assert events == [("target", 3)]
    assert app.config.teacher.force_target_N == 2
    assert "等待 owner" in status[-1]
    app.config = replace(normal_config, b601=replace(normal_config.b601, operator_teleoperation=None))
    app.apply_operator_target()
    assert len(events) == 1


def test_operator_page_load_review_edit_and_save_without_device_calls(panel, source, monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "panel_output")
    monkeypatch.setattr(panel, "ReadonlyB601Transport", lambda *_a, **_k: pytest.fail("UI opened hardware"))
    monkeypatch.setattr(panel, "X518Sensor", lambda *_a, **_k: pytest.fail("UI opened force sensor"))
    root = tk_root(panel)
    app = panel.Tuner(
        root, calibration_path=json.loads(source.read_text())["b601"]["control"]["calibration_file"]
    )
    try:
        root.geometry("1000x800")
        app.notebook.select(1)
        app.recording_page.profile_path.set(str(source))
        app.recording_page.check_profile()
        root.update_idletasks()
        assert app.recording_page.checked_profile_operator
        assert not app.recording_page.checked_profile_bounded
        assert app.arm_page.velocities["shoulder_pan"].get() == ""
        assert app.arm_page.kp["shoulder_pan"].get() != ""
        assert app.arm_page.kd["shoulder_pan"].get() != ""
        assert app.arm_page.velocities["gripper"].get() == ""
        assert "1000" in app.arm_page.wire_label.get()
        assert "原始硬力左/右" in app.arm_page.protections.get()
        app.arm_page.operator_values["alignment_tolerance_deg"].set("8")
        assert app.recording_page.checked_profile_pin is None
        app.recording_page.check_profile()
        assert "保存" in app.recording_page.profile_status.get()
        saved = tmp_path / "selected_operator.json"
        module = importlib.import_module("b601_operator_panel")
        monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **_k: str(saved))
        app.arm_page.save_draft()
        app.recording_page.check_profile()
        assert app.recording_page.checked_profile_operator
        assert app.recording_page.profile_path.get() == str(saved)
        assert json.loads(saved.read_text())["b601"]["operator_teleoperation"]["alignment_tolerance_deg"] == 8
        assert not app.running and not app.monitoring
    finally:
        root.destroy()


def test_legacy_operator_migrates_arm_mode_without_changing_gripper_or_limits(panel, source):
    raw = json.loads(source.read_text())
    for profile in raw["b601"]["control"]["profiles"][:-1]:
        profile.update(mode="pos_vel", kp=None, kd=None, velocity_deg_s=12.0)
    source.write_text(json.dumps(raw))
    before = source.read_bytes()
    value = editor(panel, source)
    assert value.converted_to_mit
    document = value.document()
    profiles = document["b601"]["control"]["profiles"]
    assert [row["mode"] for row in profiles] == ["mit"] * 6 + ["force_pos"]
    assert [row["kp"] for row in profiles[:-1]] == [45, 45, 45, 8, 9, 8]
    assert [row["kd"] for row in profiles[:-1]] == [12, 12, 12, 1, 1, 1]
    assert all(row["velocity_deg_s"] is None for row in profiles[:-1])
    assert not any("速度" in item for item in value.missing())
    assert profiles[-1] == raw["b601"]["control"]["profiles"][-1]
    assert [row["max_relative_deg"] for row in profiles] == [
        row["max_relative_deg"] for row in raw["b601"]["control"]["profiles"]
    ]
    assert document["teacher"] == raw["teacher"]
    assert document["safety"] == raw["safety"]
    assert document["b601"]["operator_teleoperation"] == raw["b601"]["operator_teleoperation"]
    assert source.read_bytes() == before


def test_mit_gain_edit_and_sdk_effective_values_are_visible(panel, source, normal_config):
    value = editor(panel, source)
    value.kp["shoulder_pan"], value.kd["shoulder_pan"] = "45", "12"
    decoded = draccus.decode(type(normal_config), value.document())
    decoded.validate()
    review = importlib.import_module("b601_profile_review")
    text = review.full_arm_profile_text(decoded)
    assert "shoulder_pan:MIT Kp=45, Kd=12" in text
    assert "SDK 编码 Kp=44.9328, Kd=5" in text
    assert "来源速度码 None" not in text


def test_official_optional_defaults_are_visible_and_pass_preflight(panel, source, monkeypatch, tmp_path):
    raw = json.loads(source.read_text())
    raw["b601"]["operator_teleoperation"]["alignment_tolerance_deg"] = None
    raw["b601"]["control"]["profiles"][0]["velocity_deg_s"] = None
    source.write_text(json.dumps(raw))
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "draft_output")
    root = tk_root(panel)
    app = panel.Tuner(root, calibration_path=raw["b601"]["control"]["calibration_file"])
    try:
        app.recording_page.profile_path.set(str(source))
        app.recording_page.check_profile()
        assert app.arm_page.operator_values["alignment_tolerance_deg"].get() == "默认关闭"
        assert app.arm_page.velocities["shoulder_pan"].get() == ""
        assert "只读缺项" not in app.arm_page.summary.get()
        assert app.recording_page.checked_profile_operator
        assert app.recording_page.checked_profile_pin is not None
        assert not app.running and not app.monitoring
    finally:
        root.destroy()
