"""Offline UI mapping, calibration, reporting and visible I-P parameter tests."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.force_control import tuning


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Widget:
    def configure(self, **kwargs):
        pass


@pytest.fixture
def stopped_panel(panel, calibration, settings, monkeypatch):
    """Exercise parameter actions without a window, transport or runtime."""
    values, enabled = settings
    app = panel.Tuner.__new__(panel.Tuner)
    app.fields = {key: _Variable(value) for key, value in values.items()}
    app.toggles = {key: _Variable(value) for key, value in enabled.items()}
    app.worker = app.next_run = None
    app.pending_cleanup = None
    app.monitor_stop = panel.threading.Event()
    app.messages = panel.queue.SimpleQueue()
    app.running = app.monitoring = False
    app.exit_confirmed = False
    app.unconfirmed_port = "COM22"
    app.monitor_port = "COM22"
    app.monitor_session = "offline-monitor"
    app.monitor_started_ns = 1_000_000_000
    app.recovery_evidence = None
    app.calibration, app.calibration_path = calibration, Path(calibration.path)
    app.config = object()
    app.estimate = (1, 2, 3, 4)
    app.live = app.last_applied = app.last_startup = object()
    app.preview_latest = {"q_measured_deg": 3}
    app.last_anchor = 123
    app.points = [app.preview_latest]
    app.confirmed = _Variable(True)
    app.other_unlocked = _Variable(True)
    app.other_unlock_check = _Widget()
    app.fixed_widgets, app.other_widgets, app.optional_widgets = [], [], []
    app.settings_path = None
    app.profile_label, app.status, app.calibration_label = (_Variable("") for _ in range(3))
    app.values_label, app.settings_label, app.calculation_label = (_Variable("") for _ in range(3))
    app.timing_settings_label = _Variable("")
    app.last_stop = _Variable("startup mode 3 rejected; exit unconfirmed")
    app.last_stop_details = '{"errors": ["startup_mode_unsupported"], "transport_closed": true}'
    for name in ("ReadonlyB601Transport", "X518Sensor", "ProcessLive"):
        monkeypatch.setattr(panel, name, lambda *a, **kw: pytest.fail("parameter file opened hardware"))
    for name in ("read", "apply", "start"):
        monkeypatch.setattr(app, name, lambda *a, **kw: pytest.fail("parameter file activated control"))
    return app


def _readonly_row():
    return {
        "monitor_session": "offline-monitor",
        "read_only": True,
        "joint_valid": True,
        "force_valid": True,
        "phase": "DISABLED",
        "joint_status_code": 0,
        "q_measured_deg": 0.0,
        "force_left_N": 0.1,
        "force_right_N": 0.2,
        "joint_received_ns": 1_990_000_000,
        "force_timestamp_ns": 1_990_000_000,
    }


@pytest.fixture
def recovery_panel(panel, stopped_panel, monkeypatch, tmp_path):
    stopped_panel.worker = SimpleNamespace(is_alive=lambda: True)
    stopped_panel.monitoring = True
    stopped_panel.live = None
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])
    monkeypatch.setattr(panel.time, "perf_counter_ns", lambda: 2_000_000_000)
    return stopped_panel


def test_readonly_recovery_reference_accepts_only_current_fresh_disabled_data(panel):
    row = _readonly_row()
    arguments = {
        "session_token": "offline-monitor",
        "started_ns": 1_000_000_000,
        "now_ns": 2_000_000_000,
        "joint_age_ms": 150,
        "force_age_ms": 50,
    }
    assert panel.readonly_exit_reference(row, **arguments)
    row.update(joint_received_ns=1_850_000_000, force_timestamp_ns=1_950_000_000)
    assert panel.readonly_exit_reference(row, **arguments)


@pytest.mark.parametrize(
    "key,value",
    [
        ("monitor_session", "previous-monitor"),
        ("read_only", 1),
        ("joint_valid", False),
        ("force_valid", False),
        ("phase", "ENABLED"),
        ("joint_status_code", True),
        ("joint_status_code", 3),
        ("q_measured_deg", float("nan")),
        ("force_left_N", float("inf")),
        ("force_right_N", None),
        ("joint_received_ns", 999_999_999),
        ("force_timestamp_ns", 999_999_999),
        ("joint_received_ns", 1_849_999_999),
        ("force_timestamp_ns", 1_949_999_999),
        ("joint_received_ns", 2_000_000_001),
        ("force_timestamp_ns", 2_000_000_001),
        ("joint_received_ns", 1_990_000_000.0),
        ("force_timestamp_ns", True),
    ],
)
def test_readonly_recovery_rejects_old_invalid_or_unknown_evidence(panel, key, value):
    row = _readonly_row()
    row[key] = value
    assert not panel.readonly_exit_reference(
        row,
        session_token="offline-monitor",
        started_ns=1_000_000_000,
        now_ns=2_000_000_000,
        joint_age_ms=150,
        force_age_ms=50,
    )


def test_recovery_port_is_taken_from_failed_session_without_guessing(panel):
    assert panel.exit_port({"config": {"b601": {"port": "COM22"}}}) == "COM22"
    for result in ({}, {"config": None}, {"config": {"b601": {"port": ""}}}, {"port": "COM22"}):
        assert panel.exit_port(result) is None


def test_exit_unconfirmed_allows_query_only_reconnection_and_rejects_old_session_callbacks(
    panel, stopped_panel, monkeypatch
):
    launched = []
    monkeypatch.setattr(stopped_panel, "launch", launched.append)
    calls = []

    def query_only(stop, publish, values, calibration, cleanup_publish):
        calls.append(values["port"])
        publish({"phase": "DISABLED", "q_measured_deg": 0})
        assert stopped_panel.preview_latest["monitor_session"] == stopped_panel.monitor_session
        stopped_panel.monitor_session = "newer-monitor"
        publish({"phase": "ENABLED", "q_measured_deg": 9})
        assert stopped_panel.preview_latest["phase"] == "DISABLED"

    monkeypatch.setattr(panel, "monitor", query_only)
    panel.Tuner.read(stopped_panel)
    assert len(launched) == 1 and stopped_panel.monitoring
    assert not stopped_panel.exit_confirmed and not stopped_panel.confirmed.get()
    launched[0]()
    assert calls == ["COM22"] and not stopped_panel.monitoring
    assert not stopped_panel.exit_confirmed


@pytest.mark.parametrize(
    "scope", ["dead_worker", "live_worker", "running", "next_run", "cleanup", "retained"]
)
def test_query_only_reconnection_cannot_duplicate_existing_owner(panel, stopped_panel, monkeypatch, scope):
    launched = []
    monkeypatch.setattr(stopped_panel, "launch", launched.append)
    monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [])
    if scope in {"dead_worker", "live_worker"}:
        stopped_panel.worker = SimpleNamespace(is_alive=lambda: scope == "live_worker")
        stopped_panel.monitoring = scope == "live_worker"
    elif scope == "running":
        stopped_panel.running = True
    elif scope == "next_run":
        stopped_panel.next_run = object()
    elif scope == "cleanup":
        stopped_panel.pending_cleanup = object()
    else:
        monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [object()])
    panel.Tuner.read(stopped_panel)
    assert not launched and not stopped_panel.exit_confirmed
    assert stopped_panel.monitor_stop.is_set() == (scope == "live_worker")


def test_fresh_query_only_recovery_writes_separate_receipt_and_preserves_history(
    panel, recovery_panel, tmp_path
):
    history = tmp_path / "outputs/failed-session/run/result.json"
    history.parent.mkdir(parents=True)
    history.write_bytes(b'{"valid": false, "errors": ["startup_mode_unsupported"]}\n')
    previous_bytes, previous_mtime = history.read_bytes(), history.stat().st_mtime_ns
    previous_details = recovery_panel.last_stop_details
    row = _readonly_row()
    recovery_panel.preview_latest = row
    recovery_panel.monitor_port = "com22"
    assert recovery_panel.recover_readonly_exit(row)
    assert recovery_panel.exit_confirmed and not recovery_panel.confirmed.get()
    receipt = tmp_path / "outputs/readonly_recovery/offline-monitor.json"
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    assert evidence == recovery_panel.recovery_evidence
    assert evidence["port"] == "com22" and evidence["motor_id"] == 7
    assert evidence["disabled_feedback_observed"] is True
    assert evidence["hardware_ack"] == "unknown" and evidence["hardware_stop_confirmed"] is False
    assert evidence["reference"]["monitor_session"] == "offline-monitor"
    details = json.loads(recovery_panel.last_stop_details)
    assert details["previous_exit_details"] == previous_details
    assert details["recovery_receipt"] == str(receipt.resolve())
    assert "startup mode 3 rejected" in recovery_panel.last_stop.get()
    assert history.read_bytes() == previous_bytes and history.stat().st_mtime_ns == previous_mtime
    panel.Tuner.start(recovery_panel)
    assert "确认" in recovery_panel.status.get() and not recovery_panel.running
    assert not recovery_panel.recover_readonly_exit(row)


@pytest.mark.parametrize(
    "scope",
    [
        "different_port",
        "unknown_port",
        "running",
        "not_monitoring",
        "dead_worker",
        "pending",
        "cleanup",
        "retained",
        "stopped",
        "stale_force",
        "outside_calibration",
        "changed_calibration",
    ],
)
def test_readonly_recovery_keeps_gate_locked_for_pending_or_unrelated_evidence(
    panel, recovery_panel, monkeypatch, tmp_path, scope
):
    row = _readonly_row()
    previous_details = recovery_panel.last_stop_details
    if scope == "different_port":
        recovery_panel.monitor_port = "COM18"
    elif scope == "unknown_port":
        recovery_panel.unconfirmed_port = None
    elif scope == "running":
        recovery_panel.running = True
    elif scope == "not_monitoring":
        recovery_panel.monitoring = False
    elif scope == "dead_worker":
        recovery_panel.worker = SimpleNamespace(is_alive=lambda: False)
    elif scope == "pending":
        recovery_panel.next_run = object()
    elif scope == "cleanup":
        recovery_panel.pending_cleanup = object()
    elif scope == "retained":
        monkeypatch.setattr(panel, "_UNSTOPPED_PREVIEWS", [object()])
    elif scope == "stopped":
        recovery_panel.monitor_stop.set()
    elif scope == "stale_force":
        row["force_timestamp_ns"] = 1_949_999_999
    elif scope == "outside_calibration":
        row["q_measured_deg"] = 41
    else:
        Path(recovery_panel.calibration.path).write_text('{"gripper": {}}', encoding="utf-8")
    assert not recovery_panel.recover_readonly_exit(row)
    assert not recovery_panel.exit_confirmed and recovery_panel.recovery_evidence is None
    assert recovery_panel.last_stop_details == previous_details
    assert not list(tmp_path.glob("outputs/readonly_recovery/*.json"))


def test_readonly_recovery_receipt_write_failure_keeps_exit_unconfirmed(panel, recovery_panel, monkeypatch):
    previous_details = recovery_panel.last_stop_details

    def reject_write(path, *args, **kwargs):
        raise OSError("recovery receipt denied")

    monkeypatch.setattr(panel.Path, "write_text", reject_write)
    assert not recovery_panel.recover_readonly_exit(_readonly_row())
    assert not recovery_panel.exit_confirmed and recovery_panel.recovery_evidence is None
    assert recovery_panel.last_stop_details == previous_details
    assert "recovery receipt denied" in recovery_panel.status.get()


def test_parameter_draft_round_trip_preserves_unfilled_and_disabled_values(panel, calibration, tmp_path):
    values = panel.default_values()
    values.update(soft_q_min_deg="draft lower bound", manual_step_deg="-0.03")
    enabled = {key: state for key, (_, state) in panel.OPTIONAL_GROUPS.items()}
    saved = panel.save_settings(tmp_path / "parameters.json", values, enabled, calibration.path)
    restored = panel.read_settings(saved)
    assert saved == (tmp_path / "parameters.json").resolve()
    assert restored == (values, enabled, Path(calibration.path).resolve())
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["format"] == "b601_ip_tuner_settings"
    assert document["controller_type"] == "ip" and document["schema_version"] == 5
    assert document["values"]["kp_deg_per_N"] == ""
    assert (
        not {"reference", "executable", "confirmed", "exit_confirmed", "shutdown_evidence"} & document.keys()
    )


def test_schema_v2_loads_without_guessing_motion_parameters(panel, calibration, settings, tmp_path):
    document = panel.settings_document(*settings, calibration.path)
    document["schema_version"] = 2
    for key in panel.MOTION_FIELDS:
        document["values"].pop(key)
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    values, enabled, _ = panel.read_settings(path)
    assert all(values[key] == "" for key in panel.MOTION_FIELDS)
    config = panel.build_config(values, enabled, calibration, tmp_path / "trace")
    assert config.initial_mode == "force"
    assert config.teacher.gripper_follow_speed_deg_s is None
    assert path.read_bytes() == original


def test_motion_parameters_map_to_profile_and_do_not_activate_on_load(panel, calibration, settings, tmp_path):
    values, enabled = settings
    values.update(
        gripper_follow_speed_deg_s="0.1", contact_force_threshold_N="0.5", position_target_deg="-10"
    )
    config = panel.build_config(values, enabled, calibration, tmp_path / "trace")
    assert config.teacher.gripper_follow_speed_deg_s == 0.1
    assert config.teacher.contact_force_threshold_N == 0.5
    assert config.initial_mode == "force" and config.initial_position_deg is None


def test_schema_v3_preserves_settings_without_guessing_position_speed(panel, calibration, settings, tmp_path):
    document = panel.settings_document(*settings, calibration.path)
    document["schema_version"] = 3
    document["values"].pop("position_speed_deg_s")
    document["values"]["position_target_deg"] = "-10"
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    values, enabled, _ = panel.read_settings(path)
    assert values == {**document["values"], "position_speed_deg_s": ""}
    assert path.read_bytes() == original
    values["position_speed_deg_s"] = "0.7"
    saved = panel.save_settings(tmp_path / "v5.json", values, enabled, calibration.path)
    assert panel.read_settings(saved)[0]["position_speed_deg_s"] == "0.7"


def test_buttons_send_atomic_approach_and_absolute_position(
    panel, stopped_panel, calibration, settings, tmp_path
):
    app = stopped_panel
    app.running = True
    app.config = panel.build_config(*settings, calibration, tmp_path / "trace")
    sent = []
    app.live = SimpleNamespace(submit=lambda *args: sent.append(args))
    app.fields["gripper_follow_speed_deg_s"].set("0.1")
    app.fields["contact_force_threshold_N"].set("0.5")
    app.approach()
    assert sent[-1] == ("approach", {"gripper_follow_speed_deg_s": 0.1, "contact_force_threshold_N": 0.5})
    app.fields["position_target_deg"].set("-10")
    app.fields["position_speed_deg_s"].set("0.7")
    app.move_to_position()
    assert sent[-1] == ("position", {"position_deg": -10, "position_speed_deg_s": 0.7})
    app.fields["position_target_deg"].set("-41")
    app.move_to_position()
    assert len(sent) == 2


def test_position_button_can_start_without_starting_ip(panel, stopped_panel, monkeypatch):
    started = []
    monkeypatch.setattr(stopped_panel, "start", lambda *args: started.append(args))
    stopped_panel.fields["position_target_deg"].set("-10")
    stopped_panel.fields["position_speed_deg_s"].set("0.7")
    stopped_panel.move_to_position()
    stopped_panel.approach()
    assert started == [("position", -10, 0.7), ("approach",)]


@pytest.mark.parametrize("speed", ["", "0", "-1", "nan", "inf"])
@pytest.mark.parametrize("running", [False, True])
def test_position_button_rejects_invalid_speed_before_submitting(stopped_panel, speed, running):
    sent = []
    stopped_panel.running = running
    stopped_panel.live = SimpleNamespace(submit=lambda *args: sent.append(args))
    stopped_panel.start = lambda *args: sent.append(args)
    stopped_panel.fields["position_target_deg"].set("-10")
    stopped_panel.fields["position_speed_deg_s"].set(speed)
    stopped_panel.move_to_position()
    assert not sent


@pytest.mark.parametrize(
    "case",
    [
        "reviewed_log",
        "old_format",
        "unknown_top_key",
        "boolean_version",
        "old_velocity_key",
        "missing_value",
        "numeric_value",
        "numeric_toggle",
        "unknown_toggle",
        "nonstring_calibration",
    ],
)
def test_parameter_reader_rejects_legacy_unknown_and_wrong_types(
    panel, calibration, settings, tmp_path, case
):
    values, enabled = settings
    document = panel.settings_document(values, enabled, calibration.path)
    if case == "reviewed_log":
        document = {"config": document, "executable": True, "reference": {"q_measured_deg": 0}}
    elif case == "old_format":
        document["format"] = "b601_pid_tuner_settings"
    elif case == "unknown_top_key":
        document["executable"] = True
    elif case == "boolean_version":
        document["schema_version"] = True
    elif case == "old_velocity_key":
        document["values"]["motor_velocity_deg_s"] = document["values"].pop("motor_velocity_code")
    elif case == "missing_value":
        document["values"].pop("kp_deg_per_N")
    elif case == "numeric_value":
        document["values"]["motor_velocity_code"] = 53
    elif case == "numeric_toggle":
        document["enabled"]["soft_limits"] = 1
    elif case == "unknown_toggle":
        document["enabled"]["filter"] = True
    elif case == "nonstring_calibration":
        document["calibration_file"] = {"range_min": -50, "range_max": 50}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        panel.read_settings(path)


def test_relative_calibration_path_is_resolved_from_parameter_file(panel, calibration, settings, tmp_path):
    document = panel.settings_document(*settings, calibration.path)
    document["calibration_file"] = Path(calibration.path).name
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert panel.read_settings(path)[2] == Path(calibration.path).resolve()


@pytest.fixture
def legacy_settings_file(panel, calibration, settings, tmp_path):
    document = panel.settings_document(*settings, calibration.path)
    document["schema_version"] = 1
    values = document["values"]
    for key in panel.MOTION_FIELDS:
        values.pop(key)
    values.pop("target_N")
    values.pop("target_min_N")
    values.pop("target_max_N")
    values.update(target_kg="0.5", target_min_kg="0.1", target_max_kg="1.2")
    path = tmp_path / "legacy-kg.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_legacy_kg_parameters_preserve_force_and_show_conversion(
    panel, stopped_panel, calibration, legacy_settings_file, tmp_path
):
    path, _ = legacy_settings_file
    original = path.read_bytes()
    stopped_panel.load_parameters_file(path)
    values = stopped_panel.values()
    assert values["target_N"] == "4.903325"
    assert values["target_min_N"] == "0.980665"
    assert values["target_max_N"] == "11.76798"
    assert "9.80665 N/kg" in stopped_panel.profile_label.get()
    assert "原文件未改写" in stopped_panel.profile_label.get()
    assert path.read_bytes() == original
    config = panel.build_config(values, stopped_panel.enabled(), calibration, tmp_path / "trace")
    assert config.teacher.force_target_N == 4.903325
    assert (config.teacher.target_min_N, config.teacher.target_max_N) == (0.980665, 11.76798)
    assert (config.teacher.kp_deg_per_N, config.teacher.ki_deg_per_N_s) == (0.04, 0.08)
    saved = tmp_path / "newtons.json"
    stopped_panel.save_parameters_file(saved)
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["schema_version"] == 5
    assert not {"target_kg", "target_min_kg", "target_max_kg"} & document["values"].keys()
    assert panel.read_settings(saved)[0] == values
    assert path.read_bytes() == original


def test_legacy_target_unit_conversion_preserves_empty_drafts(panel, legacy_settings_file):
    path, document = legacy_settings_file
    document["values"].update(target_kg="", target_min_kg="", target_max_kg="")
    path.write_text(json.dumps(document), encoding="utf-8")
    values, _, _ = panel.read_settings(path)
    assert all(values[key] == "" for key in ("target_N", "target_min_N", "target_max_N"))


@pytest.mark.parametrize("version", [1, 2])
def test_parameter_reader_rejects_mixed_target_units(panel, legacy_settings_file, version):
    path, document = legacy_settings_file
    document["schema_version"] = version
    document["values"]["target_N"] = "5"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        panel.read_settings(path)


@pytest.mark.parametrize("value", [0.5, "nan", "inf", "bad", "1e308"])
def test_legacy_target_unit_conversion_rejects_invalid_values(panel, legacy_settings_file, value):
    path, document = legacy_settings_file
    document["values"]["target_kg"] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        panel.read_settings(path)


def test_failed_atomic_parameter_replace_preserves_previous_file(
    panel, calibration, settings, tmp_path, monkeypatch
):
    path = panel.save_settings(tmp_path / "parameters.json", *settings, calibration.path)
    previous = path.read_bytes()
    files = set(tmp_path.iterdir())
    values, enabled = settings
    values["target_N"] = "7"

    def reject_replace(source, destination):
        assert Path(destination) == path
        raise OSError("replace denied")

    monkeypatch.setattr(tuning.os, "replace", reject_replace)
    with pytest.raises(OSError, match="replace denied"):
        panel.save_settings(path, values, enabled, calibration.path)
    assert path.read_bytes() == previous and set(tmp_path.iterdir()) == files


def test_gui_parameter_load_is_data_only_and_preserves_unconfirmed_exit(
    panel, stopped_panel, calibration, settings, tmp_path
):
    values, enabled = settings
    values.update(target_N="7", soft_q_min_deg="-25", soft_q_max_deg="30", motor_velocity_code="53")
    enabled["soft_limits"] = True
    path = panel.save_settings(tmp_path / "parameters.json", values, enabled, calibration.path)
    stopped_panel.load_parameters_file(path)
    assert stopped_panel.values() == values and stopped_panel.enabled() == enabled
    assert stopped_panel.calibration.sha256 == calibration.sha256
    assert stopped_panel.calibration.minimum_deg == -40
    assert not stopped_panel.exit_confirmed
    assert stopped_panel.config is stopped_panel.estimate is stopped_panel.preview_latest is None
    assert stopped_panel.live is stopped_panel.last_applied is stopped_panel.last_startup is None
    assert stopped_panel.last_anchor is None and stopped_panel.points == []
    assert not stopped_panel.confirmed.get() and not stopped_panel.other_unlocked.get()
    saved = tmp_path / "second.json"
    stopped_panel.save_parameters_file(saved)
    assert panel.read_settings(saved) == (values, enabled, Path(calibration.path).resolve())
    assert not stopped_panel.exit_confirmed


def test_gui_bad_calibration_load_does_not_partially_apply_parameters(
    panel, stopped_panel, settings, tmp_path
):
    previous_values, previous_enabled = stopped_panel.values(), stopped_panel.enabled()
    previous_config, previous_reference = stopped_panel.config, stopped_panel.preview_latest
    bad_calibration = tmp_path / "invalid-calibration.json"
    bad_calibration.write_text('{"gripper": {}}', encoding="utf-8")
    values, enabled = settings
    values["target_N"] = "9"
    path = panel.save_settings(tmp_path / "parameters.json", values, enabled, bad_calibration)
    with pytest.raises(ValueError):
        stopped_panel.load_parameters_file(path)
    assert stopped_panel.values() == previous_values and stopped_panel.enabled() == previous_enabled
    assert stopped_panel.config is previous_config and stopped_panel.preview_latest is previous_reference
    assert stopped_panel.confirmed.get() and stopped_panel.other_unlocked.get()
    assert not stopped_panel.exit_confirmed


@pytest.mark.parametrize("state", ["worker", "running", "monitoring", "next_run"])
def test_gui_save_and_load_are_blocked_until_fully_stopped(
    panel, stopped_panel, calibration, settings, tmp_path, state
):
    path = panel.save_settings(tmp_path / "parameters.json", *settings, calibration.path)
    before = stopped_panel.values()
    value = {"worker": SimpleNamespace(is_alive=lambda: True), "next_run": lambda: None}.get(state, True)
    setattr(stopped_panel, state, value)
    with pytest.raises(RuntimeError, match="停止并断开"):
        stopped_panel.load_parameters_file(path)
    destination = tmp_path / "blocked-save.json"
    with pytest.raises(RuntimeError, match="停止并断开"):
        stopped_panel.save_parameters_file(destination)
    assert not destination.exists() and stopped_panel.values() == before
    assert not stopped_panel.exit_confirmed


def test_required_new_gain_units_and_motion_limits_never_receive_defaults(panel, calibration, tmp_path):
    values = panel.default_values()
    enabled = {key: state for key, (_, state) in panel.OPTIONAL_GROUPS.items()}
    for key in (
        "kp_deg_per_N",
        "ki_deg_per_N_s",
        "close_sign",
        "target_max_N",
        "max_relative_target_deg",
        "max_command_tracking_error_deg",
        "stall_position_tolerance_deg",
        "stall_timeout_s",
    ):
        assert values[key] == ""
    with pytest.raises(ValueError, match="Kp"):
        panel.build_config(values, enabled, calibration, tmp_path)


def test_new_ip_config_reaches_common_runtime_with_raw_force(panel, calibration, settings, tmp_path):
    values, enabled = settings
    values.update(
        target_N="5.25",
        target_min_N="0.25",
        target_max_N="8",
        max_close_speed_deg_s="0.3",
        max_open_speed_deg_s="0.4",
        integral_deadband_N="0.02",
        hard_force_left_kg="1.6",
        hard_force_right_kg="1.7",
        force_max_age_ms="70",
        joint_feedback_max_age_ms="80",
        watchdog_ticks="1800",
        queue_capacity="1024",
        raw_queue_capacity="5120",
        sensor_host="192.0.2.10",
        sensor_request_timeout_s="0.06",
        drive_percent="3.01",
        motor_velocity_code="53",
    )
    cfg = panel.build_config(values, enabled, calibration, tmp_path)
    assert cfg.teacher.controller_type == "ip"
    assert cfg.teacher.kp_deg_per_N == 0.04 and cfg.teacher.ki_deg_per_N_s == 0.08
    assert cfg.teacher.force_target_N == 5.25
    assert cfg.teacher.target_min_N == 0.25 and cfg.teacher.target_max_N == 8
    assert cfg.teacher.integral_deadband_N == 0.02
    assert cfg.close_sign == -1
    assert (cfg.safety.q_min_deg, cfg.safety.q_max_deg) == (-40, 40)
    assert cfg.safety.max_relative_target_deg == cfg.b601.control.profiles[0].max_relative_deg == 0.4
    assert cfg.safety.max_command_tracking_error_deg == 0.8
    assert cfg.safety.max_close_speed_deg_s == 0.3 and cfg.safety.max_open_speed_deg_s == 0.4
    assert cfg.safety.hard_force_limit_left_N == pytest.approx(1.6 * panel.GRAVITY_N_PER_KG)
    assert cfg.safety.hard_force_limit_right_N == pytest.approx(1.7 * panel.GRAVITY_N_PER_KG)
    assert cfg.b601.control.feedback_max_age_s == 0.08
    assert cfg.b601.control.watchdog_ticks == 1800
    assert cfg.b601.control.calibration_file == calibration.path
    assert cfg.b601.startup_mode == "prepare_disabled_gripper"
    profile = cfg.b601.control.profiles[0]
    assert profile.velocity_wire_code == 53 and profile.velocity_deg_s is None
    assert profile.velocity_limit_deg_s == pytest.approx(math.degrees(0.53))
    assert (
        panel.force_pos_wire_codes(
            profile.velocity_deg_s, profile.torque_ratio, velocity_wire_code=profile.velocity_wire_code
        )[0]
        == 53
    )
    sensor = cfg.sensors["gripper_force"]
    assert sensor.host == "192.0.2.10" and sensor.max_age_ms == 70
    assert sensor.request_timeout_s == 0.06 and sensor.recorder_queue_capacity == 5120
    assert cfg.raw_queue_capacity == 5120 and cfg.queue_capacity == 1024
    assert not {"kp", "ki", "kd", "derivative_filter_tau_s", "target_ramp_N_s"} & asdict(cfg.teacher).keys()
    decoded = draccus.decode(type(cfg), draccus.encode(cfg))
    assert decoded == cfg


def test_optional_limits_use_explicit_none_and_calibration_is_never_enlarged(
    panel, calibration, settings, tmp_path
):
    values, enabled = settings
    enabled.update(acceleration=False, soft_limits=False, duration=False, travel=False)
    values.update(
        soft_q_min_deg="bad", soft_q_max_deg="bad", max_run_time_s="bad", max_total_travel_deg="bad"
    )
    cfg = panel.build_config(values, enabled, calibration, tmp_path)
    assert cfg.duration_s is cfg.safety.max_run_time_s is None
    assert cfg.safety.acceleration_limit_deg_s2 is cfg.safety.max_total_travel_deg is None
    enabled.update(acceleration=True, soft_limits=True, duration=True, travel=True)
    values.update(soft_q_min_deg="-30", soft_q_max_deg="35", max_run_time_s="60", max_total_travel_deg="4")
    cfg = panel.build_config(values, enabled, calibration, tmp_path)
    assert cfg.safety.soft_q_min_deg == -30 and cfg.safety.soft_q_max_deg == 35
    assert cfg.duration_s == 60 and cfg.safety.max_total_travel_deg == 4
    values["soft_q_min_deg"] = "-50"
    with pytest.raises(ValueError, match="soft"):
        panel.build_config(values, enabled, calibration, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("kp_deg_per_N", "nan"),
        ("ki_deg_per_N_s", "-1"),
        ("close_sign", "0"),
        ("max_relative_target_deg", "0"),
        ("hard_force_left_kg", "0.5"),
        ("drive_percent", "1.001"),
        ("watchdog_ticks", "0"),
        ("force_max_age_ms", "inf"),
        ("motor_velocity_code", "0"),
        ("motor_velocity_code", "10001"),
        ("motor_velocity_code", "1.5"),
    ],
)
def test_invalid_visible_values_are_rejected_before_transport(
    panel, calibration, settings, tmp_path, key, value
):
    values, enabled = settings
    values[key] = value
    with pytest.raises(ValueError):
        panel.build_config(values, enabled, calibration, tmp_path)


def test_calibration_replacement_requires_reload(panel, calibration, settings, tmp_path):
    Path(calibration.path).write_text('{"gripper": {}}')
    with pytest.raises(ValueError):
        panel.build_config(*settings, calibration, tmp_path)


def test_estimator_is_pure_and_uses_position_output_gain_dimensions(panel):
    result = panel.tuning_estimate(0, 0.2, 1, 3, 10, 10.5, 0.2, 0.1)
    assert result == pytest.approx((10, 0.5, 0.04, 0.01))
    for args in ((0, 0, 1, 3, 0, 1, 0.2, 0.1), (0, 1, 3, 1, 0, 1, 0.2, 0.1), (0, 1, 1, 3, 1, 1, 0.2, 0.1)):
        with pytest.raises(ValueError):
            panel.tuning_estimate(*args)


def test_exit_report_preserves_uncertain_disable_and_raw_errors(panel, tmp_path):
    result = {
        "errors": ["send_failed"],
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {"disabled_feedback_observed": False},
        },
    }
    assert not panel.exit_confirmation(result)
    report = panel.stop_report(result, tmp_path)
    assert "send_failed" in report["summary"] and "未确认" in report["summary"]
    result["shutdown_evidence"]["fault_disposition"]["disabled_feedback_observed"] = True
    assert not panel.exit_confirmation(result)
    result["shutdown_evidence"]["fault_disposition"].update(status="submitted", error=None)
    assert panel.exit_confirmation(result)
    result["shutdown_evidence"]["fault_disposition"].update(status="unknown", error="late feedback")
    assert not panel.exit_confirmation(result)


def test_panel_build_has_two_tabs_and_other_settings_lock_without_io(
    panel, calibration, monkeypatch, tmp_path
):
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path)
    monkeypatch.setattr(panel, "ReadonlyB601Transport", lambda *a, **kw: pytest.fail("UI opened hardware"))
    monkeypatch.setattr(panel, "X518Sensor", lambda *a, **kw: pytest.fail("UI opened sensor"))
    try:
        root = panel.tk.Tk()
    except panel.tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    app = panel.Tuner(root, calibration_path=calibration.path)
    try:
        assert [app.notebook.tab(tab, "text") for tab in app.notebook.tabs()] == ["夹爪控制", "其他设置"]
        assert set(app.fields) == set(panel.default_values())
        assert set(panel.FIELD_PAGES) == set(app.fields)
        assert panel.FIELD_PAGES["motor_velocity_code"] == "gripper"
        assert panel.FIELD_PAGES["port"] == "other"
        assert not app.other_unlocked.get()
        assert all(str(widget.cget("state")) == "disabled" for widget in app.other_widgets)
        app.other_unlocked.set(True)
        app.update_editability()
        assert all(str(widget.cget("state")) == "normal" for widget in app.other_widgets)
        app.confirmed.set(True)
        app.preview_latest = {"q_measured_deg": 0}
        app.fields["soft_q_min_deg"].set("-25")
        app.fields["soft_q_max_deg"].set("30")
        assert app.load_calibration()
        assert app.preview_latest is None and not app.confirmed.get()
        assert app.fields["soft_q_min_deg"].get() == "-25"
        assert app.fields["soft_q_max_deg"].get() == "30"
        app.last_stop.set("persistent stop")
        app.apply()
        assert app.last_stop.get() == "persistent stop"
        app.worker = SimpleNamespace(is_alive=lambda: True)
        app.monitoring = True
        app.poll()
        assert not app.other_unlocked.get()
        assert all(str(widget.cget("state")) == "disabled" for widget in app.fixed_widgets)
        assert all(str(widget.cget("state")) == "disabled" for widget in app.other_widgets)
        assert str(app.other_unlock_check.cget("state")) == "disabled"
        assert str(app.confirm_check.cget("state")) == "normal"
        app.worker = None
        app.monitoring = False
        app.update_editability()
        assert all(str(widget.cget("state")) == "disabled" for widget in app.other_widgets)
    finally:
        root.destroy()


def test_query_only_monitor_closes_resources_and_preserves_raw_values(
    panel, calibration, settings, monkeypatch
):
    values = panel.parse_values(*settings)
    calls = []
    stop = panel.threading.Event()

    class Device:
        has_resources = True

        def __init__(self, *args, **kwargs):
            pass

        def open(self):
            calls.append("open")

        def read_mapping(self, **kwargs):
            return SimpleNamespace(is_valid=True)

        def read_feedback(self, **kwargs):
            return SimpleNamespace(
                is_valid=True,
                feedback=[
                    SimpleNamespace(
                        position_deg=0,
                        status_name="DISABLED",
                        status_code=0,
                        receive_timestamp_ns=time.perf_counter_ns(),
                    )
                ],
            )

        def close(self):
            calls.append("device_close")

    class Sensor:
        has_resources = True

        def __init__(self, cfg):
            pass

        def connect(self):
            calls.append("sensor_connect")

        def acquire_recorder(self):
            return SimpleNamespace(error=None)

        def read_latest_before(self, now, **kwargs):
            return SimpleNamespace(
                is_valid=True,
                timestamp_ns=now,
                values={"left.normal_force": 2.0, "right.normal_force": 4.0},
            )

        def release_recorder(self, lease):
            calls.append("lease_release")

        def disconnect(self):
            calls.append("sensor_close")

    monkeypatch.setattr(panel, "ReadonlyB601Transport", Device)
    monkeypatch.setattr(panel, "X518Sensor", Sensor)
    rows = []

    def publish(row):
        rows.append(row)
        stop.set()

    panel.monitor(stop, publish, values, calibration)
    assert rows[0]["force_mean_N"] == 3
    assert rows[0]["force_left_N"] == 2 and rows[0]["force_right_N"] == 4
    assert rows[0]["read_only"] is rows[0]["joint_valid"] is rows[0]["force_valid"] is True
    assert rows[0]["joint_status_code"] == 0
    assert rows[0]["force_timestamp_ns"] == rows[0]["decision_anchor_ns"]
    assert calls[-3:] == ["device_close", "lease_release", "sensor_close"]


def test_fresh_startup_settings_distinguish_requests_effective_values_and_mapping(panel):
    receipt = {
        "complete": True,
        "watchdog_ticks": 2000,
        "command_settings": {
            "requested": {"motor_velocity_deg_s": 0.6, "drive_limit_ratio": 0.02},
            "applied": {"motor_velocity_deg_s": 0.572957795, "drive_limit_ratio": 0.02},
            "encoded": {"velocity_wire_code": 1, "drive_wire_code": 200},
            "device_mapping": [12.5, 30.0, 10.0],
        },
    }
    text = panel.startup_description(receipt)
    assert "0.6" in text and "0.572958" in text and "30.0" in text
    assert "unknown" in text and "请求" in text and "编码有效" in text


def test_validation_failure_is_visible_even_with_clean_hardware_exit(panel, tmp_path):
    result = {
        "valid": False,
        "errors": ["raw_source_mismatch"],
        "shutdown_evidence": {
            "transport_closed": True,
            "fault_disposition": {"disabled_feedback_observed": True, "status": "submitted", "error": None},
        },
    }
    assert panel.exit_confirmation(result)
    assert "核验失败" in panel.stop_report(result, tmp_path)["summary"]


def test_query_display_reports_recorded_budget_instead_of_ui_default(panel):
    row = {
        "tick_start_ns": 1_000_000_000,
        "joint_snapshot": {"completed_ns": 1_010_925_300, "deadline_ns": 1_005_000_000},
        "dt_control_s": 0.0214087,
        "loop_elapsed_s": 0.0122469,
    }
    description = panel.timing_description(row)
    assert "10.925 / 5 ms" in description
    assert "21.409 ms" in description and "12.247 ms" in description
    assert "本周期未查询" in panel.timing_description({"tick_start_ns": 1_000_000_000})
    assert panel.timing_description({"read_only": True}) == ""


def test_timing_settings_distinguish_pending_and_running_owner_configuration(
    panel, calibration, settings, tmp_path
):
    values, enabled = settings
    values.update(feedback_budget_ms="5", tick_deadline_ms="30")
    config = panel.build_config(values, enabled, calibration, tmp_path / "trace")
    values.update(feedback_budget_ms="15", tick_deadline_ms="40", send_budget_s="0.03")
    idle = panel.timing_settings_description(values)
    assert "15 / 40 / 30 ms" in idle and "尚未生效" in idle
    pending = panel.timing_settings_description(values, config, running=True)
    assert "待启动配置（尚未生效）" in pending
    assert "查询 5 ms / 周期工作 30 ms / 发送 20 ms" in pending
    active = panel.timing_settings_description(values, config, startup_complete=True, running=True)
    assert "本次 owner 运行生效" in active
    assert "查询 5 ms / 周期工作 30 ms / 发送 20 ms" in active
    stopped = panel.timing_settings_description(values, config, startup_complete=True)
    assert "已结束，当前未生效" in stopped
    values["send_budget_s"] = ""
    assert "未填写 / 未记录" in panel.timing_settings_description(values)


def test_startup_receipt_updates_separate_timing_bar_without_replacing_request_values(
    panel, calibration, settings, tmp_path, monkeypatch
):
    monkeypatch.setattr(panel, "OUTPUTS", tmp_path)
    monkeypatch.setattr(panel, "ReadonlyB601Transport", lambda *a, **kw: pytest.fail("UI opened hardware"))
    monkeypatch.setattr(panel, "X518Sensor", lambda *a, **kw: pytest.fail("UI opened sensor"))
    try:
        root = panel.tk.Tk()
    except panel.tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    app = panel.Tuner(root, calibration_path=calibration.path)
    try:
        values, enabled = settings
        values.update(feedback_budget_ms="5", tick_deadline_ms="30")
        app.config = panel.build_config(values, enabled, calibration, tmp_path / "trace")
        for key, value in values.items():
            app.fields[key].set(value)
        receipt = {
            "complete": False,
            "watchdog_ticks": 2000,
            "command_settings": {
                "requested": {"motor_velocity_deg_s": 0.6, "drive_limit_ratio": 0.02},
                "applied": {"motor_velocity_deg_s": 0.572957795, "drive_limit_ratio": 0.02},
                "encoded": {"velocity_wire_code": 1, "drive_wire_code": 200},
                "device_mapping": [12.5, 30.0, 10.0],
            },
        }
        app.running = True
        app.live = SimpleNamespace(receive=lambda: None, startup=receipt, applied=None, latest=None)
        app.poll()
        assert "启动未完成" in app.settings_label.get()
        assert "待启动配置（尚未生效）" in app.timing_settings_label.get()
        assert "5 / 30 / 20 ms" in app.timing_settings_label.get()
        app.live.startup = dict(receipt, complete=True)
        app.poll()
        assert "启动完成" in app.settings_label.get()
        assert "本次 owner 运行生效" in app.timing_settings_label.get()
        assert "查询 5 ms / 周期工作 30 ms / 发送 20 ms" in app.timing_settings_label.get()
        app.running = False
        app.poll()
        assert "已结束，当前未生效" in app.timing_settings_label.get()
    finally:
        root.destroy()


def test_loading_small_legacy_budget_warns_without_changing_file_or_values(
    panel, stopped_panel, calibration, settings, tmp_path
):
    values, enabled = settings
    values.update(feedback_budget_ms="5", tick_deadline_ms="30")
    path = panel.save_settings(tmp_path / "explicit.json", values, enabled, calibration.path)
    original = path.read_bytes()
    stopped_panel.load_parameters_file(path)
    assert stopped_panel.values() == values and path.read_bytes() == original
    assert "5 / 30 ms" in stopped_panel.profile_label.get()
    assert "默认 15 / 40 ms 未替代文件值" in stopped_panel.profile_label.get()
    assert "5 / 30 / 20 ms" in stopped_panel.timing_settings_label.get()
    assert "尚未生效" in stopped_panel.timing_settings_label.get()
    assert not stopped_panel.confirmed.get() and not stopped_panel.exit_confirmed


@pytest.mark.parametrize(
    "error,diagnosis",
    [
        ("B601 read-only deadline expired", "位置反馈查询超时"),
        ("B601 read has less than one millisecond remaining", "位置反馈查询超时"),
        ("B601 response incomplete at deadline (1/16 bytes received)", "位置反馈查询超时"),
        ("B601 write has less than one millisecond remaining", "发起查询时剩余时间不足 1 ms"),
    ],
)
def test_query_timeout_report_uses_recorded_budget_and_preserves_raw_evidence(
    panel, tmp_path, error, diagnosis
):
    raw = f"RuntimeError: invalid_joint_feedback; joint_error: TimeoutError: {error}"
    result = {"errors": [raw], "config": {"runtime": {"feedback_budget_ms": 5, "tick_deadline_ms": 30}}}
    report = panel.stop_report(result, tmp_path)
    assert diagnosis in report["summary"]
    assert "本次查询 / 周期工作预算 5 / 30 ms" in report["summary"]
    assert "默认 15 / 40 ms 仅供参照" in report["summary"]
    assert "不会自动替代" in report["summary"] and raw in report["summary"]
    assert json.loads(report["details"])["errors"] == [raw]
    result["config"]["runtime"].update(feedback_budget_ms=15, tick_deadline_ms=40)
    assert "本次查询 / 周期工作预算 15 / 40 ms" in panel.stop_report(result, tmp_path)["summary"]


def test_stop_report_never_substitutes_defaults_for_malformed_recorded_config(panel, tmp_path):
    result = {
        "errors": [
            "RuntimeError: invalid_joint_feedback; joint_error: TimeoutError: B601 read-only deadline expired"
        ],
        "config": {"runtime": "bad", "safety": []},
    }
    report = panel.stop_report(result, tmp_path)
    assert "本次查询 / 周期工作预算 未填写 / 未记录 / 未填写 / 未记录 ms" in report["summary"]
    assert json.loads(report["details"])["runtime_config"] == {}


def test_protocol_failure_is_not_misreported_as_query_timeout_from_cleanup(panel, tmp_path):
    result = {
        "errors": [
            "RuntimeError: invalid_joint_feedback; joint_error: ValueError: B601 framing mismatch",
            "TimeoutError: B601 no response before deadline",
        ],
        "config": {"runtime": {"feedback_budget_ms": 5, "tick_deadline_ms": 30}},
    }
    report = panel.stop_report(result, tmp_path)
    assert "framing mismatch" in report["summary"]
    assert "位置反馈查询超时" not in report["summary"] and "默认 15 / 40" not in report["summary"]
    assert json.loads(report["details"])["errors"] == result["errors"]


@pytest.mark.parametrize("side,label,threshold", [("left", "左", 14.709975), ("right", "右", 11.76798)])
def test_overforce_report_displays_that_sides_recorded_threshold(panel, tmp_path, side, label, threshold):
    raw = f"RuntimeError: {side}_force_hard_limit"
    result = {"errors": [raw], "config": {"safety": {f"hard_force_limit_{side}_N": threshold}}}
    report = panel.stop_report(result, tmp_path)
    assert f"{label}侧原始力触发过力保护" in report["summary"]
    assert f"停止阈值 {threshold} N" in report["summary"]
    assert "默认 15 / 40" not in report["summary"]
    assert json.loads(report["details"])["errors"] == [raw]
    result.pop("config")
    assert "未填写 / 未记录 N" in panel.stop_report(result, tmp_path)["summary"]


def test_new_budget_default_does_not_override_loaded_explicit_budget(panel, calibration, settings, tmp_path):
    assert panel.default_values()["feedback_budget_ms"] == "15"
    assert panel.default_values()["tick_deadline_ms"] == "40"
    values, enabled = settings
    values["feedback_budget_ms"] = "5"
    values["tick_deadline_ms"] = "30"
    path = panel.save_settings(tmp_path / "explicit.json", values, enabled, calibration.path)
    loaded_values, loaded_enabled, _ = panel.read_settings(path)
    config = panel.build_config(loaded_values, loaded_enabled, calibration, tmp_path / "trace")
    assert (config.runtime.feedback_budget_ms, config.runtime.tick_deadline_ms) == (5, 30)
