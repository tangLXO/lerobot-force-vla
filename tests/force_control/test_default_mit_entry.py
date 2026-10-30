"""Default panel startup uses saved gripper settings without device access."""

import json
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower.full_arm_control_types import leader_configuration_digest
from lerobot.scripts.lerobot_force_teacher import leader_configuration_projection


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def default_tuner(panel, settings, monkeypatch, tmp_path, tk_root):
    calibration_path = tmp_path / "synthetic-seven-joint-calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                joint.name: {
                    "id": joint.motor_id,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": -40 if joint.name == "gripper" else -180,
                    "range_max": 40 if joint.name == "gripper" else 180,
                }
                for joint in panel.DEFAULT_B601_JOINTS
            }
        ),
        encoding="utf-8",
    )
    values, enabled = settings
    values = dict(
        values,
        port="FAKE_FOLLOWER",
        leader_port="FAKE_LEADER",
        gripper_follow_speed_deg_s="0.1",
        contact_force_threshold_N="0.5",
        soft_q_min_deg="-25",
        soft_q_max_deg="30",
        motor_velocity_code="53",
        drive_percent="7",
    )
    enabled = dict(enabled, soft_limits=True)
    settings_path = panel.save_settings(tmp_path / "saved-settings.json", values, enabled, calibration_path)
    original_settings = settings_path.read_bytes()
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "sessions")

    def forbidden(*args, **kwargs):
        pytest.fail("Default panel startup accessed a device or launched a control process")

    for name in ("ReadonlyB601Transport", "X518Sensor", "ProcessLive", "run_in_process"):
        monkeypatch.setattr(panel, name, forbidden)
    app = panel.Tuner(tk_root, settings_path=settings_path, default_full_arm_mit=True)
    monkeypatch.setattr(app, "launch", forbidden)
    for timer in tk_root.tk.call("after", "info"):
        tk_root.after_cancel(timer)
    try:
        yield app, values, enabled, settings_path
        assert settings_path.read_bytes() == original_settings
    finally:
        for widget in tk_root.winfo_children():
            widget.destroy()


def test_main_uses_saved_settings_and_default_mit_without_connecting(panel, monkeypatch):
    opened = []
    window = SimpleNamespace(mainloop=lambda: opened.append("mainloop"))
    calls = []
    monkeypatch.setattr(panel.tk, "Tk", lambda: window)
    monkeypatch.setattr(panel, "Tuner", lambda root, **kwargs: calls.append((root, kwargs)))
    panel.main([])
    assert opened == ["mainloop"] and len(calls) == 1
    assert calls[0][0] is window
    assert calls[0][1]["default_full_arm_mit"] is True
    assert Path(calls[0][1]["settings_path"]) == (
        Path(__file__).resolve().parents[2] / "b601_capture_parameters.json"
    )


def test_explicit_gripper_entry_keeps_software_only_startup(panel, monkeypatch):
    calls = []
    window = SimpleNamespace(mainloop=lambda: None)
    monkeypatch.setattr(panel.tk, "Tk", lambda: window)
    monkeypatch.setattr(panel, "Tuner", lambda root, **kwargs: calls.append(kwargs))
    panel.main(["--gripper-only"])
    assert calls[0]["default_full_arm_mit"] is False


def test_saved_settings_autoload_official_mit_and_preserve_gripper(default_tuner, panel, tmp_path):
    app, values, enabled, settings_path = default_tuner
    assert app.settings_path == settings_path
    assert app.values() == values and app.enabled() == enabled
    assert app.using_default_full_arm_profile()
    assert app.recording_page.full_arm_enabled.get()
    assert app.worker is None and app.next_run is None
    assert not app.running and not app.monitoring and not app.monitor_stop.is_set()

    config = panel.load_full_arm_profile(app.recording_page.profile_path.get())
    native = RebotB601FollowerConfig(port=values["port"])
    profiles = {profile.name: profile for profile in config.b601.control.profiles}
    for index, joint in enumerate(panel.DEFAULT_B601_JOINTS[:-1]):
        assert profiles[joint.name].mode == "mit"
        assert profiles[joint.name].kp == native.mit_kp[index]
        assert profiles[joint.name].kd == native.mit_kd[index]
        assert profiles[joint.name].max_relative_deg is None
        assert profiles[joint.name].velocity_deg_s is None
        assert app.arm_page.kp[joint.name].get() and app.arm_page.kd[joint.name].get()
    operator = config.b601.operator_teleoperation
    assert operator.alignment_tolerance_deg is None
    assert operator.preload_tolerance_deg is None
    assert operator.max_command_gap_s is None
    gripper = panel.build_config(
        values, enabled, app.calibration, tmp_path / "expected", initial_mode="hybrid"
    )
    assert config.teacher == gripper.teacher and config.safety == gripper.safety
    assert asdict(profiles["gripper"]) == asdict(gripper.b601.control.profiles[0])
    assert draccus.encode(config.sensors) == draccus.encode(gripper.sensors)


@pytest.mark.parametrize("mode,joint_count", [("force", 1), ("hybrid", 7)])
def test_start_freezes_current_form_without_executing_readonly_handoff(
    default_tuner, panel, monkeypatch, tmp_path, mode, joint_count
):
    app, _, _, _ = default_tuner
    app.fields["target_N"].set("3.25")
    app.fields["kp_deg_per_N"].set("0.12")
    app.fields["ki_deg_per_N_s"].set("0.23")
    app.fields["motor_velocity_code"].set("71")
    app.fields["drive_percent"].set("9")
    app.arm_page.kp["shoulder_pan"].set("99")
    app.arm_page.kd["shoulder_pan"].set("4")
    expected = panel.build_config(
        app.values(), app.enabled(), app.calibration, tmp_path / "current", initial_mode="hybrid"
    )
    live = SimpleNamespace(stop=panel.threading.Event(), close=lambda: None)
    monkeypatch.setattr(panel, "ProcessLive", lambda **kwargs: live)
    app.monitoring = True
    app.monitor_port = app.fields["port"].get()
    app.preview_latest = {"joint_received_ns": panel.time.perf_counter_ns(), "phase": "DISABLED"}
    app.confirmed.set(True)
    app.start(mode)
    assert app.running, app.status.get()
    assert app.monitor_stop.is_set() and callable(app.next_run)
    assert app.worker is None and app.live is live
    assert not live.stop.is_set()
    assert app.config.teacher == expected.teacher and app.config.safety == expected.safety
    gripper = next(profile for profile in app.config.b601.control.profiles if profile.name == "gripper")
    assert asdict(gripper) == asdict(expected.b601.control.profiles[0])
    assert len(app.config.runtime.joint_names) == joint_count
    if mode == "hybrid":
        shoulder = next(
            profile for profile in app.config.b601.control.profiles if profile.name == "shoulder_pan"
        )
        assert shoulder.kp == 99 and shoulder.kd == 4
    reviewed_path = next(panel.OUTPUTS.glob("*/reviewed.json"))
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    assert reviewed["config"]["teacher"]["force_target_N"] == 3.25
    assert reviewed["operator_values"] == app.values()
    assert reviewed["reference"] == app.preview_latest


def test_all_arm_limits_save_reload_and_build_the_same_session(default_tuner, panel, tmp_path):
    app, _, _, _ = default_tuner
    choices = {
        "reference_envelope_deg": "150",
        "alignment_tolerance_deg": "3",
        "preload_tolerance_deg": "0.1",
        "max_command_gap_s": "0.08",
        "handover_tolerance_deg": "8",
    }
    for key, value in choices.items():
        app.arm_page.operator_values[key].set(value)
    app.arm_page.minimum_deg["shoulder_pan"].set("-33")
    app.arm_page.maximum_deg["shoulder_pan"].set("27")
    app.arm_page.relative_limits["shoulder_pan"].set("5")
    app.arm_page.kp["shoulder_pan"].set("44")
    app.arm_page.kd["shoulder_pan"].set("3")
    app.arm_page.mit_velocity_deg_s["shoulder_pan"].set("12")
    app.arm_page.mit_torque_Nm["shoulder_pan"].set("0.4")
    app.fields["scheduler_sleep_slice_ms"].set("1")
    # Shared controls must edit the saved source used by the next session.
    app.arm_page.operator_values["leader_max_age_s"].set("0.25")
    app.arm_page.operator_values["watchdog_ticks"].set("2500")
    app.arm_page.relative_limits["gripper"].set("1.2")
    app.arm_page.duration.set("180")
    app.arm_page.duration_enabled.set(True)
    app.arm_page.travel.set("400")
    app.arm_page.travel_enabled.set(True)
    expected_arm = app.arm_page.settings_values()
    saved = tmp_path / "all-control-settings.json"
    app.save_parameters_file(saved)
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["schema_version"] == 10 and document["arm_settings"] == expected_arm
    assert not {"supported_arm", "reference_deg", "confirmed", "enabled"} & expected_arm.keys()
    app.arm_page.operator_values["reference_envelope_deg"].set("10")
    app.load_parameters_file(saved)
    assert app.arm_page.settings_values() == expected_arm
    assert not app.confirmed.get() and app.preview_latest is None
    app.confirmed.set(True)
    config = app.default_full_arm_config(tmp_path / "unused")
    config.b601.operator_teleoperation.validate()
    operator = config.b601.operator_teleoperation
    assert (
        operator.reference_envelope_deg,
        operator.alignment_tolerance_deg,
        operator.preload_tolerance_deg,
    ) == (150, 3, 0.1)
    assert operator.max_command_gap_s == 0.08
    assert config.runtime.handover_tolerance_deg == 8
    assert config.leader_max_age_s == operator.leader_max_age_s == 0.25
    assert config.b601.control.watchdog_ticks == operator.watchdog_ticks == 2500
    profile = config.b601.control.profiles[0]
    assert (profile.minimum_deg, profile.maximum_deg, profile.max_relative_deg, profile.kp, profile.kd) == (
        -33,
        27,
        5,
        44,
        3,
    )
    assert config.leader_joint_ranges["shoulder_pan"] == [-33, 27]
    assert (profile.mit_velocity_deg_s, profile.mit_torque_Nm) == (12, 0.4)
    assert config.runtime.scheduler_sleep_slice_ms == 1
    assert operator.leader_config_sha256 == leader_configuration_digest(
        leader_configuration_projection(config)
    )
    assert config.b601.control.profiles[-1].max_relative_deg == 1.2
    assert config.duration_s == config.safety.max_run_time_s == 180
    assert config.safety.max_total_travel_deg == 400
    # The file-based builder must honor the same saved values as the UI.
    from build_operator_teleoperation_profile import build_document

    saved_config, _ = build_document(saved)
    assert saved_config["b601"]["operator_teleoperation"]["reference_envelope_deg"] == 150
    assert saved_config["b601"]["control"]["profiles"][0]["minimum_deg"] == -33
    assert saved_config["runtime"]["handover_tolerance_deg"] == 8
    assert saved_config["b601"]["control"]["profiles"][0]["mit_torque_Nm"] == 0.4
    assert saved_config["runtime"]["scheduler_sleep_slice_ms"] == 1


def test_v7_arm_settings_gain_zero_references_without_rewriting_old_file(default_tuner, panel, tmp_path):
    app, _, _, _ = default_tuner
    document = panel.settings_document(
        app.values(), app.enabled(), app.calibration_path, arm_settings=app.arm_page.settings_values()
    )
    document["schema_version"] = 7
    document["values"]["manual_step_deg"] = ""
    document["values"].pop("scheduler_sleep_slice_ms")
    for profile in document["arm_settings"]["joints"].values():
        profile.pop("mit_velocity_deg_s")
        profile.pop("mit_torque_Nm")
    path = tmp_path / "old-v7.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    app.load_parameters_file(path)
    assert path.read_bytes() == original
    assert app.fields["scheduler_sleep_slice_ms"].get() == "5"
    assert all(value.get() == "0" for value in app.arm_page.mit_velocity_deg_s.values())
    assert all(value.get() == "0" for value in app.arm_page.mit_torque_Nm.values())


def test_edited_arm_working_range_cannot_exceed_loaded_calibration(default_tuner, tmp_path):
    app, _, _, _ = default_tuner
    app.arm_page.minimum_deg["shoulder_pan"].set("-500")
    app.confirmed.set(True)
    config = app.default_full_arm_config(tmp_path / "unused")
    with pytest.raises(ValueError, match="calibration"):
        config.b601.operator_teleoperation.validate_binding(
            config.b601.joint_specs, config.b601.control, port=config.b601.port
        )


def test_fast_ui_poll_does_not_redraw_an_unchanged_waveform(default_tuner, panel, monkeypatch):
    app, _, _, _ = default_tuner
    app.root.geometry("1320x920+10000+10000")
    app.root.deiconify()
    app.root.update()
    app.points.clear()
    now = panel.time.perf_counter_ns()
    app.points.append({"decision_anchor_ns": now, "force_mean_N": 1.0})
    app._last_plot_key = None
    deleted = []
    original = app.canvas.delete

    def delete(*args):
        deleted.append(args)
        return original(*args)

    monkeypatch.setattr(app.canvas, "delete", delete)
    try:
        app.draw()
        app.draw()
        assert len(deleted) == 1
        app.points.append({"decision_anchor_ns": now + 10_000_000, "force_mean_N": 1.1})
        app.draw()
        assert len(deleted) == 2
        app.fields["plot_seconds"].set("10")
        app.draw()
        assert len(deleted) == 3
    finally:
        app.root.withdraw()


@pytest.mark.parametrize("scaling", [1.25, 1.5])
def test_reference_limit_entry_fits_the_visible_arm_page(default_tuner, scaling):
    app, _, _, _ = default_tuner
    root = app.root
    previous = root.tk.call("tk", "scaling")
    try:
        root.tk.call("tk", "scaling", scaling)
        root.geometry("1000x740+10000+10000")
        root.deiconify()
        app.notebook.select(1)
        root.update()
        entry = app.arm_page.operator_entries["reference_envelope_deg"]
        page = app.notebook.nametowidget(app.notebook.select())
        assert entry.winfo_ismapped()
        x = entry.winfo_rootx() - page.winfo_rootx()
        y = entry.winfo_rooty() - page.winfo_rooty()
        assert 0 <= x < x + entry.winfo_width() <= page.winfo_width()
        assert 0 <= y < y + entry.winfo_height() <= page.winfo_height()
    finally:
        root.withdraw()
        root.tk.call("tk", "scaling", previous)


def test_teleoperation_start_stays_visible_above_every_tab_at_supported_window_sizes(default_tuner):
    app, _, _, _ = default_tuner
    root = app.root
    original_scaling = root.tk.call("tk", "scaling")
    button = app.hybrid_button
    left = app.notebook.master

    def rect(widget, parent):
        x = widget.winfo_rootx() - parent.winfo_rootx()
        y = widget.winfo_rooty() - parent.winfo_rooty()
        return x, y, x + widget.winfo_width(), y + widget.winfo_height()

    try:
        for scaling in (1.25, 1.5):
            root.tk.call("tk", "scaling", scaling)
            for width, height in ((1000, 740), (1320, 920), (1920, 1080)):
                # Map outside the desktop solely to resolve window-manager geometry.
                root.geometry(f"{width}x{height}+10000+10000")
                root.deiconify()
                for tab in app.notebook.tabs():
                    app.notebook.select(tab)
                    root.update()
                    evidence = {
                        "scaling": scaling,
                        "size": (width, height),
                        "tab": app.notebook.tab(tab, "text"),
                        "button_in_root": rect(button, root),
                        "button_in_left": rect(button, left),
                        "notebook_in_left": rect(app.notebook, left),
                    }
                    assert button.winfo_ismapped(), evidence
                    for parent in (root, left):
                        x0, y0, x1, y1 = rect(button, parent)
                        assert 0 <= x0 < x1 <= parent.winfo_width(), evidence
                        assert 0 <= y0 < y1 <= parent.winfo_height(), evidence
                    assert button.winfo_height() >= button.winfo_reqheight(), evidence
                    assert rect(button, left)[3] <= rect(app.notebook, left)[1], evidence
        assert "启动全臂遥操" in button.cget("text")
        assert app.worker is None and app.next_run is None
        assert not app.running and not app.monitoring
    finally:
        root.withdraw()
        root.tk.call("tk", "scaling", original_scaling)
