"""Offline desktop behavior; tests construct no serial, sensor or camera objects."""

import gc
import importlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from lerobot.force_control.tuning import FIELDS, OPTIONAL_GROUPS


class FakeController:
    def __init__(self):
        self.calls = []
        self.frames = {}
        self.closed = False
        self.preview_enabled = False
        actions = (
            "connect",
            "disconnect",
            "start",
            "recover",
            "apply",
            "target",
            "resume",
            "approach",
            "position",
            "release",
            "record_start",
            "record_finish",
            "record_discard",
            "record_outcome",
            "record_verify",
            "verify_dataset",
            "cancel_verification",
            "preflight",
            "enumerate_cameras",
            "preview_cameras",
            "bind_cameras",
            "close_camera_preview",
            "load_settings",
            "save_settings",
            "load_calibration",
            "edit_values",
            "edit_live",
            "edit_recording",
        )
        self.view = SimpleNamespace(
            revision=1,
            session_id="offline",
            state="disconnected",
            message="设备未连接；点击连接开始只读监测。",
            error=None,
            exit_confirmed=True,
            telemetry={},
            recording={"state": "ready"},
            values={key: default for _, key, _, default, *_ in FIELDS},
            toggles={key: default for key, (_, default) in OPTIONAL_GROUPS.items()},
            recording_values={
                "root": "outputs/offline_demo",
                "task": "offline task",
                "object_id": "cube",
                "contact_condition_id": "dry",
            },
            camera_configs={},
            devices=(),
            capabilities=dict.fromkeys(actions, ""),
            settings_path=None,
            calibration_path=None,
            log_path=None,
            operation_results=(),
            arm_settings={},
        )

    def snapshot(self):
        return self.view

    def request(self, action, payload=None):
        self.calls.append((action, payload))
        return str(len(self.calls))

    def stop(self):
        self.calls.append(("stop", None))

    def close(self):
        self.closed = True

    def set_preview_enabled(self, enabled):
        self.preview_enabled = enabled

    def preview(self):
        return self.frames


@pytest.fixture(scope="module")
def application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if os.name == "nt" and not QtGui.QFontDatabase.families():
        for filename in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
            QtGui.QFontDatabase.addApplicationFont(str(Path(os.environ["WINDIR"]) / "Fonts" / filename))
    yield app
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


@pytest.fixture
def desktop(application, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    module = importlib.import_module("desktop.window")
    controller = FakeController()
    window = module.DesktopWindow(controller, demo=True)
    window.resize(1366, 768)
    window.show()
    application.processEvents()
    try:
        yield window, controller
    finally:
        controller.closed = True
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        application.processEvents()


@pytest.mark.parametrize(
    "args,mode,gap",
    [
        ([], "cached", 120.0),
        (["--command-feedback-mode", "sequential"], "sequential", 120.0),
        (["--command-feedback-mode", "pipelined", "--command-tx-gap-us", "120"], "pipelined", 120.0),
        (["--command-feedback-mode", "cached", "--command-tx-gap-us", "800"], "cached", 800.0),
    ],
)
@pytest.mark.parametrize("environment", [None, "1000"])
def test_desktop_entry_passes_daily_communication_defaults(
    application, monkeypatch, args, mode, gap, environment
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    entry = importlib.import_module("b601_desktop")
    session = importlib.import_module("b601_session")
    monkeypatch.delenv("MOTORBRIDGE_TX_GAP_US", raising=False)
    if environment is not None:
        monkeypatch.setenv("MOTORBRIDGE_TX_GAP_US", environment)
    if "--command-tx-gap-us" not in args and environment is not None:
        gap = float(environment)

    captured = {}

    class StartupCapturedError(Exception):
        pass

    def capture_controller(**kwargs):
        captured.update(kwargs)
        raise StartupCapturedError

    monkeypatch.setattr(session, "SessionController", capture_controller)
    with pytest.raises(StartupCapturedError):
        entry.main(args)
    assert captured["command_feedback_mode"] == mode
    assert captured["command_tx_gap_us"] == gap
    assert captured["command_tx_gap_source"] == (
        "命令行"
        if "--command-tx-gap-us" in args
        else "环境变量 MOTORBRIDGE_TX_GAP_US"
        if environment is not None
        else "官方默认"
    )


def test_home_operations_and_stop_fit_without_scrolling(desktop):
    window, controller = desktop
    assert not controller.calls
    assert controller.preview_enabled
    assert not isinstance(window.pages.widget(0), QtWidgets.QScrollArea)
    for button in (
        window.stop_button,
        window.connect_button,
        window.start_button,
        window.record_button,
        window.save_button,
        window.verify_button,
        window.cancel_verify_button,
    ):
        assert button.isVisible()
        position = button.mapTo(window, QtCore.QPoint())
        assert window.rect().contains(position)
        assert window.rect().contains(position + QtCore.QPoint(button.width() - 1, button.height() - 1))


@pytest.mark.parametrize("field", ["confirmed", "recording_enabled"])
def test_checkbox_has_visible_states_and_is_clickable(desktop, application, field):
    window, controller = desktop
    controller.view.state = "monitoring"
    controller.view.revision += 1
    window.poll()
    checkbox = getattr(window, field)
    checkbox.setChecked(False)
    application.processEvents()
    assert checkbox.isEnabled()
    option = QtWidgets.QStyleOptionButton()
    checkbox.initStyleOption(option)
    indicator = checkbox.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, option, checkbox
    )
    assert indicator.width() >= 16 and indicator.height() >= 16

    def pixels():
        pixmap = checkbox.grab()
        ratio = pixmap.devicePixelRatio()
        image = pixmap.toImage()
        left, top = round(indicator.left() * ratio), round(indicator.top() * ratio)
        width, height = round(indicator.width() * ratio), round(indicator.height() * ratio)
        return [
            image.pixelColor(x, y).name() for y in range(top, top + height) for x in range(left, left + width)
        ]

    unchecked = pixels()
    assert "#647991" in unchecked and "#ffffff" in unchecked
    QtTest.QTest.mouseClick(checkbox, QtCore.Qt.MouseButton.LeftButton, pos=indicator.center())
    application.processEvents()
    assert checkbox.isChecked()
    checked = pixels()
    assert "#087f71" in checked and "#ffffff" in checked
    checkbox.setEnabled(False)
    application.processEvents()
    disabled_checked = pixels()
    assert "#71978f" in disabled_checked and "#ffffff" in disabled_checked
    QtTest.QTest.mouseClick(checkbox, QtCore.Qt.MouseButton.LeftButton, pos=indicator.center())
    assert checkbox.isChecked()
    checkbox.setChecked(False)
    application.processEvents()
    assert "#e6ebf0" in pixels()
    assert not controller.calls


def test_confirmation_is_available_after_readonly_connection_and_resets_after_disconnect(desktop):
    window, controller = desktop
    assert not window.confirmed.isEnabled()
    controller.view.state = "monitoring"
    controller.view.revision += 1
    window.poll()
    assert window.confirmed.isEnabled()
    assert not window.start_button.isEnabled()
    assert "勾选" in window.start_button.toolTip()
    window.confirmed.setChecked(True)
    assert window.start_button.isEnabled()
    controller.view.state = "disconnected"
    controller.view.revision += 1
    window.poll()
    assert not window.confirmed.isEnabled() and not window.confirmed.isChecked()
    assert not window.start_button.isEnabled()
    assert not controller.calls


def test_recording_and_camera_shortcuts_open_the_matching_section(desktop, application):
    window, controller = desktop
    for link, section in (
        (window.recording_config_link, "recording"),
        (window.camera_config_link, "cameras"),
    ):
        window.navigation.setCurrentRow(0)
        link.click()
        application.processEvents()
        assert window.navigation.currentRow() == 2
        target = window.device_sections[section]
        position = target.mapTo(window.device_page.viewport(), QtCore.QPoint())
        assert 0 <= position.y() < window.device_page.viewport().height()
        assert position.y() + target.height() <= window.device_page.viewport().height()
        assert window.device_shortcuts["devices"].isVisible()
        assert window.device_shortcuts["recording"].isVisible()
        assert window.device_shortcuts["cameras"].isVisible()
    window.device_shortcuts["devices"].click()
    application.processEvents()
    position = window.device_sections["devices"].mapTo(window.device_page.viewport(), QtCore.QPoint())
    assert 0 <= position.y() < window.device_page.viewport().height()
    assert not controller.calls


@pytest.mark.parametrize("width,height", [(1050, 700), (1366, 768), (1366, 900)])
def test_tuning_position_is_in_manual_card_and_arm_expansion_reveals_content(
    desktop, application, width, height
):
    window, controller = desktop
    window.resize(width, height)
    window.navigation.setCurrentRow(1)
    application.processEvents()
    page = window.tuning_page
    for widget in (*window._position_command_edits.values(), window.position_button, window.position_hint):
        assert window.manual_operations.isAncestorOf(widget)
        point = widget.mapTo(window, QtCore.QPoint())
        assert widget.isVisible() and window.rect().contains(point)
        assert window.rect().contains(point + QtCore.QPoint(widget.width() - 1, widget.height() - 1))
    page.ensureWidgetVisible(window.arm_editor.toggle, 0, 0)
    application.processEvents()
    QtTest.QTest.mouseClick(window.arm_editor.toggle, QtCore.Qt.MouseButton.LeftButton)
    QtTest.QTest.qWait(100)
    first_field = next(edit for (section, _), edit in window._arm_edits.items() if section == "operator")
    for widget in (window.arm_editor.toggle, first_field):
        point = widget.mapTo(page.viewport(), QtCore.QPoint())
        assert widget.isVisible() and page.viewport().rect().contains(point)
        assert page.viewport().rect().contains(point + QtCore.QPoint(widget.width() - 1, widget.height() - 1))
    scroll_position = page.verticalScrollBar().value()
    controller.view.state = "running"
    for sample in range(5):
        controller.view.telemetry = {
            "phase": "FOLLOW",
            "decision_anchor_ns": time.perf_counter_ns(),
            "q_sent_deg": -20.0 + sample,
            "q_measured_deg": -20.1 + sample,
        }
        controller.view.revision += 1
        window.poll()
        QtTest.QTest.qWait(50)
        assert window.arm_editor.toggle.isChecked()
        assert page.verticalScrollBar().value() == scroll_position
    page.verticalScrollBar().setValue(0)
    application.processEvents()
    position = window.position_button.mapTo(page.viewport(), QtCore.QPoint())
    assert page.viewport().rect().contains(position)
    assert (
        page.viewport()
        .rect()
        .contains(
            position + QtCore.QPoint(window.position_button.width() - 1, window.position_button.height() - 1)
        )
    )
    assert not controller.calls


@pytest.mark.parametrize("width,height", [(1050, 700), (1366, 900)])
def test_tuning_arm_follows_force_section_and_charts_remain_equal(desktop, application, width, height):
    window, controller = desktop
    window.resize(width, height)
    window.navigation.setCurrentRow(1)
    application.processEvents()
    page = window.tuning_page
    force_toggle = next(
        button
        for button in page.findChildren(QtWidgets.QToolButton)
        if button.text() == "FORCE_POS 与控制时序"
    )
    force_box = force_toggle.parentWidget()

    def assert_arm_below_force():
        force = force_box.mapTo(page.widget(), QtCore.QPoint())
        arm = window.arm_editor.mapTo(page.widget(), QtCore.QPoint())
        assert arm.x() == force.x()
        assert arm.y() >= force.y() + force_box.height()

    def assert_equal_charts():
        plots = (window.tuning_force_plot, window.tuning_position_plot, window.tuning_delta_plot)
        assert len({(plot.width(), plot.height()) for plot in plots}) == 1
        assert plots[0].height() == 210

    assert_arm_below_force()
    assert_equal_charts()
    force_toggle.click()
    application.processEvents()
    assert_arm_below_force()
    force_toggle.click()
    window.arm_editor.toggle.click()
    QtTest.QTest.qWait(100)
    assert_arm_below_force()
    assert_equal_charts()

    table = window.arm_editor.findChild(QtWidgets.QTableWidget)
    page.ensureWidgetVisible(table, 0, 0)
    application.processEvents()
    assert table.horizontalScrollBar().maximum() > 0
    for column in range(table.columnCount()):
        title = table.horizontalHeaderItem(column).text()
        assert table.columnWidth(column) >= table.horizontalHeader().fontMetrics().horizontalAdvance(title)
    table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
    application.processEvents()
    last_field = table.cellWidget(0, table.columnCount() - 1)
    point = last_field.mapTo(table.viewport(), QtCore.QPoint())
    assert table.viewport().rect().contains(point)
    assert (
        table.viewport()
        .rect()
        .contains(point + QtCore.QPoint(last_field.width() - 1, last_field.height() - 1))
    )
    assert not controller.calls


@pytest.mark.parametrize("width,height", [(1050, 700), (1366, 900)])
def test_tuning_plot_time_axes_align_and_follow_each_other(desktop, application, width, height):
    window, controller = desktop
    window.resize(width, height)
    window.navigation.setCurrentRow(1)
    for sample in range(40):
        window.charts.append(
            {
                "decision_anchor_ns": 1_000_000_000 + sample * 20_000_000,
                "force_valid": True,
                "force_left_N": 0.2 + sample * 0.01,
                "force_right_N": 0.4 + sample * 0.01,
                "force_mean_N": 0.3 + sample * 0.01,
                "force_target_requested_N": 2.0,
                "q_sent_deg": -170 + sample,
                "q_measured_deg": -171 + sample,
                "hold_target_deg": -120,
                "delta_i_deg": sample * 0.00001,
                "delta_p_deg": sample * -0.00002,
                "delta_contact_deg": 0,
                "delta_raw_deg": sample * 0.00003,
                "delta_final_deg": sample * 0.00001,
            }
        )
    window.charts.render()
    QtTest.QTest.qWait(100)
    plots = (window.tuning_force_plot, window.tuning_position_plot, window.tuning_delta_plot)
    body = window.tuning_page.widget()
    edges = []
    for plot in plots:
        rect = plot.getViewBox().sceneBoundingRect()
        edges.append(
            (
                plot.viewport().mapTo(body, plot.mapFromScene(rect.topLeft())).x(),
                plot.viewport().mapTo(body, plot.mapFromScene(rect.topRight())).x(),
            )
        )
        assert plot.getAxis("left").textWidth < plot.getAxis("left").width()
    assert max(left for left, _ in edges) - min(left for left, _ in edges) <= 1
    assert max(right for _, right in edges) - min(right for _, right in edges) <= 1

    home_range = window.force_plot.getViewBox().viewRange()[0]
    for index, source in enumerate(plots):
        requested = (-18.0 + index * 3, -8.0 + index * 3)
        source.setXRange(*requested, padding=0)
        application.processEvents()
        for plot in plots:
            assert plot.getViewBox().viewRange()[0] == pytest.approx(requested)
    assert window.force_plot.getViewBox().viewRange()[0] == home_range
    assert not controller.calls


def test_tuning_position_uses_per_command_draft_without_changing_locked_parameters(desktop, application):
    window, controller = desktop
    controller.view.state = "monitoring"
    controller.view.capabilities.update(edit_values="已锁定", edit_live="", position="请先启动控制会话")
    controller.view.revision += 1
    window.poll()
    assert all(edit.isEnabled() for edit in window._position_command_edits.values())
    assert not window.position_button.isEnabled()
    controller.view.state = "running"
    controller.view.capabilities.update(edit_values="已锁定", edit_live="全臂参数已锁定", position="")
    controller.view.revision += 1
    window.poll()
    window.navigation.setCurrentRow(1)
    application.processEvents()
    assert not window._field_edits["position_target_deg"][0].isEnabled()
    assert window.position_hint.text() == "主从时退出跟随"
    for key, value in (("position_target_deg", "-10"), ("position_speed_deg_s", "0.7")):
        edit = window._position_command_edits[key]
        assert edit.isEnabled()
        edit.clear()
        QtTest.QTest.keyClicks(edit, value)
    assert not window._form_dirty and not controller.calls
    controller.view.revision += 1
    window.poll()
    window.position_button.click()
    assert controller.calls[-1] == ("position", {"position_deg": -10.0, "position_speed_deg_s": 0.7})
    assert controller.view.values["position_target_deg"] == ""
    controller.view.values.update(position_target_deg="-8", position_speed_deg_s="0.6")
    window._hydrated = False
    window._hydrate(controller.view)
    assert window._position_command_edits["position_target_deg"].text() == "-8"
    assert window._position_command_edits["position_speed_deg_s"].text() == "0.6"
    window._position_command_edits["position_target_deg"].setText("nan")
    count = len(controller.calls)
    window.position_button.click()
    assert len(controller.calls) == count and "有限数值" in window.feedback.text()
    controller.view.capabilities["position"] = "请先启动控制会话"
    controller.view.revision += 1
    window.poll()
    assert not window.position_button.isEnabled()
    assert "请先启动" in window.position_hint.text()
    window.position_button.click()
    assert len(controller.calls) == count


def test_direct_position_start_requires_confirmation_and_recording_disabled(desktop, application):
    window, controller = desktop
    assert not window.position_button.isEnabled() and "连接只读监测" in window.position_hint.text()
    controller.view.state = "monitoring"
    controller.view.capabilities.update(edit_values="已锁定", edit_live="", position="请先启动控制会话")
    controller.view.revision += 1
    window.poll()
    window.navigation.setCurrentRow(1)
    application.processEvents()
    assert window.position_button.text() == "启动并运动到指定位置"
    assert not window.position_button.isEnabled() and "勾选" in window.position_hint.text()
    window.confirmed.setChecked(True)
    window.recording_enabled.setChecked(True)
    assert not window.position_button.isEnabled() and "关闭录制准备" in window.position_hint.text()
    window.recording_enabled.setChecked(False)
    assert window.position_button.isEnabled() and "不录制" in window.position_hint.text()

    class EnabledChanges(QtCore.QObject):
        count = 0

        def eventFilter(self, obj, event):  # noqa: N802 - Qt virtual method.
            if event.type() == QtCore.QEvent.Type.EnabledChange:
                self.count += 1
            return False

    watcher = EnabledChanges(window)
    window.position_button.installEventFilter(watcher)
    for _ in range(3):
        controller.view.revision += 1
        window.poll()
    assert watcher.count == 0
    window._position_command_edits["position_target_deg"].setText("-10")
    window._position_command_edits["position_speed_deg_s"].setText("0.7")
    assert not controller.calls
    window.position_button.click()
    action, payload = controller.calls[-1]
    assert action == "start"
    assert {key: payload[key] for key in ("mode", "confirmed", "position_deg", "position_speed_deg_s")} == {
        "mode": "position",
        "confirmed": True,
        "position_deg": -10.0,
        "position_speed_deg_s": 0.7,
    }
    assert payload["form"]["recording_values"]["enabled"] is False
    assert not window.position_button.isEnabled()
    window.position_button.click()
    assert len(controller.calls) == 1
    controller.view.operation_results = (
        {"request_id": "1", "action": "start", "success": False, "message": "输入无效"},
    )
    controller.view.revision += 1
    window.poll()
    assert window.position_button.isEnabled()


def test_constant_speed_closure_to_force_control_remains_available(desktop, application):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.capabilities["approach"] = ""
    controller.view.revision += 1
    window.poll()
    window.navigation.setCurrentRow(1)
    application.processEvents()
    button = next(button for button, action in window._buttons if action == "approach")
    assert button.text() == "恒速闭合 → 力控" and window.manual_operations.isAncestorOf(button)
    point = button.mapTo(window.tuning_page.viewport(), QtCore.QPoint())
    assert window.tuning_page.viewport().rect().contains(point)
    window._field_edits["gripper_follow_speed_deg_s"][0].setText("0.7")
    window._field_edits["contact_force_threshold_N"][0].setText("0.5")
    button.click()
    assert controller.calls[-1] == (
        "approach",
        {"gripper_follow_speed_deg_s": 0.7, "contact_force_threshold_N": 0.5},
    )


def test_closing_window_during_arm_expansion_does_not_issue_commands(desktop, application):
    window, controller = desktop
    window.navigation.setCurrentRow(1)
    application.processEvents()
    window.arm_editor.toggle.click()
    controller.closed = True
    window.close()
    QtTest.QTest.qWait(50)
    assert not window.isVisible() and not controller.calls


def test_home_data_summary_tracks_unsent_root_and_resume_drafts(desktop):
    window, controller = desktop
    controller.view.recording_values["resume"] = False
    window._recording_edits["root"].setText("C:/data/another_session")
    window.resume_checkbox.setChecked(True)
    assert "数据目录（草稿）" in window.dataset_path.text()
    assert window.dataset_path.toolTip() == "C:/data/another_session"
    assert window.dataset_resume.text() == "续录：已选（草稿）"
    assert "尚未提交" in window.dataset_resume.toolTip()
    controller.view.recording_values.update(root="C:/data/another_session", resume=True)
    controller.view.revision += 1
    window.poll()
    assert window.dataset_path.text() == "数据目录：C:/data/another_session"
    assert window.dataset_resume.text() == "续录：已选"
    assert not controller.calls


@pytest.mark.parametrize("width,height", [(1050, 700), (1050, 740), (1366, 768), (1050, 850), (1366, 900)])
@pytest.mark.parametrize("mode", ["disconnected", "recording", "error"])
def test_primary_controls_remain_visible_at_supported_sizes(desktop, application, width, height, mode):
    window, controller = desktop
    if mode != "disconnected":
        controller.view.state = "running"
        controller.view.recording_values["enabled"] = True
        controller.view.recording = {"state": "recording", "active": True, "episode_uid": "episode"}
        if mode == "error":
            controller.view.recording.update(error="采集出现异常" * 20, cleanup_pending=True)
        controller.view.revision += 1
        window.poll()
    window.resize(width, height)
    application.processEvents()
    assert window.width() == width and window.height() == height
    discard = next(button for button, action in window._buttons if action == "record_discard")
    for widget in (
        window.stop_button,
        window.problem_button,
        window.record_badge,
        window.confirmed,
        window.start_button,
        window.record_button,
        window.save_button,
        window.verify_button,
        window.cancel_verify_button,
        window.verification_result_button,
        window.verification_panel,
        window.record_state,
        window.preview_status["front"],
        window.preview_status["wrist"],
    ):
        pos = widget.mapTo(window, QtCore.QPoint())
        assert widget.isVisible() and window.rect().contains(pos)
        assert window.rect().contains(pos + QtCore.QPoint(widget.width() - 1, widget.height() - 1))
    for upper, lower in (
        (window._recording_edits["task"], window.outcome),
        (window._recording_edits["object_id"], window.outcome),
        (window.outcome, window.record_button),
        (window.record_button, window.save_button),
        (window.save_button, discard),
        (discard, window.record_state),
        (window.record_state, window.record_count),
        (window.preview_labels["front"], window.preview_status["front"]),
        (window.preview_labels["wrist"], window.preview_status["wrist"]),
    ):
        upper_bottom = upper.mapTo(window, QtCore.QPoint(0, upper.height())).y()
        assert upper_bottom <= lower.mapTo(window, QtCore.QPoint()).y()
    record_card = window.record_button.parentWidget()
    for widget in (window._recording_edits["task"], window.outcome, window.record_state, window.record_count):
        pos = widget.mapTo(record_card, QtCore.QPoint())
        assert record_card.rect().contains(pos)
        assert record_card.rect().contains(pos + QtCore.QPoint(widget.width() - 1, widget.height() - 1))
    record_bottom = record_card.mapTo(window, QtCore.QPoint(0, record_card.height())).y()
    panel_top = window.verification_panel.mapTo(window, QtCore.QPoint()).y()
    assert record_bottom <= panel_top
    if window._verification_layout_mode != "footer":
        panel_bottom = window.verification_panel.mapTo(
            window, QtCore.QPoint(0, window.verification_panel.height())
        ).y()
        assert panel_bottom <= window.dataset_path.mapTo(window, QtCore.QPoint()).y()


def test_recording_badge_and_next_step_follow_actual_state(desktop):
    window, controller = desktop
    window._recording_edits["task"].setText("较长的任务描述可完整查看")
    assert "较长的任务描述可完整查看" in window._recording_edits["task"].toolTip()
    controller.view.state = "running"
    controller.view.recording_values["enabled"] = False
    controller.view.revision += 1
    window.poll()
    assert "未准备录制" in window.next_step.text()
    assert window.record_badge.text() == "录制：未开始"
    controller.view.recording_values["enabled"] = True
    controller.view.recording = {
        "state": "recording",
        "active": True,
        "episode_uid": "one",
        "frame_count": 150,
    }
    controller.view.revision += 1
    window.poll()
    assert "录制中" in window.next_step.text()
    assert window.record_badge.text() == "录制：进行中"
    assert "约 3.0 秒（按帧数）" in window.record_count.text()
    window.navigation.setCurrentRow(1)
    controller.view.recording = {"state": "saving", "active": True, "episode_uid": "one"}
    controller.view.revision += 1
    window.poll()
    assert window.record_badge.isVisible() and window.record_badge.text() == "录制：保存中"
    assert "等待" in window.next_step.text()
    assert not controller.calls


def test_recording_errors_are_visible_and_saved_state_requires_completion_evidence(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.recording = {
        "state": "saved",
        "file_complete": False,
        "request_error": "录制请求未接收",
        "cleanup_error": "录制进程尚未退出",
    }
    controller.view.revision += 1
    window.poll()
    assert "完整性待确认" in window.record_state.text()
    assert "录制请求未接收" in window.readiness.text()
    assert "录制进程尚未退出" in window.diagnostic_error.text()
    assert window.record_badge.text() == "录制：异常"
    controller.view.recording = {"state": "saved", "file_complete": False}
    controller.view.revision += 1
    window.poll()
    assert window.record_badge.text() == "录制：待确认"
    assert "数据集条数" in window.record_count.text()


def test_problem_details_preserve_full_errors_and_recent_issue(desktop, application, tmp_path, monkeypatch):
    window, controller = desktop
    original = "ValueError: Dataset already exists; explicitly enable resume " + "x" * 300
    additional = [f"additional error {index}" for index in range(40)]
    controller.view.error = "RuntimeError: " + original
    controller.view.result_errors = (original, *additional)
    controller.view.log_path = str(tmp_path / "run" / "result.json")
    controller.view.state = "recovery_required"
    controller.view.exit_confirmed = False
    controller.view.revision += 1
    window.poll()
    assert len(window.readiness.text()) < len(original)
    assert window.problem_button.isVisible()
    assert window._problem_dialog is None

    window.problem_button.click()
    dialog = window._problem_dialog
    details = window.problem_details_text.toPlainText()
    assert dialog.isVisible() and not dialog.isModal()
    assert "当前上报的问题" in details and "续录" in details
    assert original in details and additional[-1] in details and "41 条" in details
    assert controller.view.log_path in details and window.problem_log_button.isEnabled()
    copied = []
    monkeypatch.setattr(QtGui.QClipboard, "setText", lambda self, value: copied.append(value))
    dialog.findChild(QtWidgets.QPushButton, "copyProblemDetails").click()
    assert copied == [details]

    controller.view.error = None
    controller.view.result_errors = ()
    controller.view.state = "monitoring"
    controller.view.exit_confirmed = True
    controller.view.revision += 1
    window.poll()
    details = window.problem_details_text.toPlainText()
    assert "最近一次问题" in details and original in details and additional[-1] in details
    controller.view.error = "later cleanup failure"
    controller.view.revision += 1
    window.poll()
    details = window.problem_details_text.toPlainText()
    assert "later cleanup failure" in details and original in details
    assert not window.problem_log_button.isEnabled()  # The old result path is not this new issue's log.
    assert not controller.calls
    dialog.close()


@pytest.mark.parametrize("state", ["stopping", "disconnected"])
def test_operator_stop_does_not_create_a_control_fault_problem(desktop, state):
    window, controller = desktop
    controller.view.state = state
    controller.view.telemetry = {"phase": "FAULT", "failure_reason": "user_stop"}
    controller.view.revision += 1
    window.poll()
    assert not window._problem_history
    assert window._current_problem_key is None
    assert not controller.calls


def test_problem_details_collect_independent_sources_without_changing_controls(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.telemetry = {"phase": "FAULT", "failure_reason": "left_force_hard_limit"}
    controller.view.recording = {
        "state": "error",
        "error": "Camera device_path must match exactly one device; found 0",
        "request_error": "unexpected-detail-" + "z" * 120,
        "cleanup_error": "writer did not exit",
    }
    controller.view.revision += 1
    window.poll()
    window.show_problem_details()
    details = window.problem_details_text.toPlainText()
    assert "相机设备路径无法唯一匹配" in details and "左侧原始力触发硬限保护" in details
    assert "Camera device_path must match exactly one device; found 0" in details
    assert "unexpected-detail-" + "z" * 120 in details and "writer did not exit" in details
    assert "left_force_hard_limit" in details
    assert not controller.calls
    window._problem_dialog.close()


def test_problem_details_keep_local_dispatch_error_after_feedback_changes(desktop, monkeypatch):
    window, controller = desktop

    def fail_request(*args, **kwargs):
        raise RuntimeError("local failure " + "y" * 120)

    monkeypatch.setattr(controller, "request", fail_request)
    assert window.dispatch("connect") is None
    controller.view.message = "状态已刷新"
    controller.view.revision += 1
    window.poll()
    window.show_problem_details()
    text = window.problem_details_text.toPlainText()
    assert "最近一次问题" in text and "连接：RuntimeError: local failure " + "y" * 120 in text
    assert "暂未识别" in text
    window._problem_dialog.close()


def test_problem_history_is_bounded_and_preserves_recent_errors(desktop):
    window, controller = desktop
    for index in range(12):
        controller.view.error = f"independent problem {index:02}"
        controller.view.revision += 1
        window.poll()
    window.show_problem_details()
    text = window.problem_details_text.toPlainText()
    assert len(window._problem_history) == 10
    assert "independent problem 00" not in text and "independent problem 01" not in text
    assert "independent problem 02" in text and "independent problem 11" in text
    window._problem_dialog.close()


def test_cleanup_tail_is_explained_as_possible_secondary_error(desktop):
    window, controller = desktop
    controller.view.error = "RuntimeError: Raw producer tail was not sealed"
    controller.view.revision += 1
    window.poll()
    window.show_problem_details()
    text = window.problem_details_text.toPlainText()
    assert "结束边界未确认" in text and "可能出现在清理阶段" in text
    assert "Raw producer tail was not sealed" in text
    window._problem_dialog.close()


def test_new_episode_resets_result_only_once(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.recording = {"state": "recording", "active": True, "episode_uid": "episode-a"}
    controller.view.revision += 1
    window.poll()
    window.outcome.setCurrentIndex(window.outcome.findData("success"))
    controller.view.revision += 1
    window.poll()
    assert window.outcome.currentData() == "success"
    controller.view.recording["episode_uid"] = "episode-b"
    controller.view.revision += 1
    window.poll()
    assert window.outcome.currentData() == "unknown"
    assert not controller.calls


def test_stale_preview_and_disconnected_feedback_are_explicit(desktop):
    window, controller = desktop
    controller.view.state = "running"
    now = time.perf_counter_ns()
    controller.view.telemetry = {
        "phase": "FOLLOW",
        "decision_anchor_ns": now - 3_000_000_000,
        "q_measured_deg": -20,
        "force_left_N": 3.0,
    }
    controller.view.revision += 1
    window.poll()
    image = QtGui.QImage(24, 24, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("#087f71"))
    window._decoded("front", now - 3_000_000_000, image)
    window._refresh_preview_status()
    assert "画面停更" in window.preview_status["front"].text()
    assert "反馈停更" in window.phase_label.text()
    assert window.metrics["force_left_N"].value.text() == "3.000 N"
    controller.view.state = "disconnected"
    controller.view.revision += 1
    window.poll()
    assert "无实时反馈" in window.phase_label.text()
    assert window.metrics["force_left_N"].value.text() == "—"
    assert window.preview_labels["front"]._source is None
    window._decoded("front", time.perf_counter_ns(), image)  # A queued decode arrived after disconnect.
    assert window.preview_labels["front"]._source is None
    assert not controller.calls


def test_new_session_does_not_show_old_camera_frame(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.revision += 1
    window.poll()
    old_timestamp = time.perf_counter_ns()
    image = QtGui.QImage(24, 24, QtGui.QImage.Format.Format_RGB888)
    window._decoded("front", old_timestamp, image)
    controller.frames = {"front": {"jpeg": b"old", "timestamp_ns": old_timestamp}}
    controller.view.session_id = "next-session"
    controller.view.revision += 1
    window.poll()
    assert window.preview_labels["front"]._source is None
    assert "等待画面" in window.preview_status["front"].text()
    assert not window._decode_busy


def test_invalid_force_or_joint_feedback_is_not_shown_as_a_valid_reading(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.telemetry = {
        "phase": "FOLLOW",
        "decision_anchor_ns": time.perf_counter_ns(),
        "feedback_valid": False,
        "force_valid": False,
        "q_measured_deg": -20,
        "force_left_N": 3,
        "force_target_requested_N": 4,
    }
    controller.view.revision += 1
    window.poll()
    assert "关节 / 力反馈无效" in window.phase_label.text()
    assert window.metrics["force_left_N"].value.text() == "—"
    assert window.metrics["force_target_requested_N"].value.text() == "4.000 N"
    controller.view.telemetry.update(feedback_valid=True, force_valid=True, force_left_N=float("nan"))
    controller.view.revision += 1
    window.poll()
    assert window.metrics["force_left_N"].value.text() == "—"


def test_camera_binding_requires_fresh_decoded_frames(desktop):
    window, controller = desktop
    controller.view.capabilities["bind_cameras"] = ""
    window.navigation.setCurrentRow(2)
    window.dispatch("preview_cameras", {"front": "front", "wrist": "wrist"}, update=False)
    image = QtGui.QImage(24, 24, QtGui.QImage.Format.Format_RGB888)
    previous = window._binding_started_ns - 1
    for role in ("front", "wrist"):
        window._decoded(role, previous, image)
    assert not window._binding_seen
    window._binding_started_ns -= 5_000_000_000
    stale = time.perf_counter_ns() - 3_000_000_000
    for role in ("front", "wrist"):
        window._decoded(role, stale, image)
    window.poll()
    assert not window.bind_button.isEnabled()
    assert "停更" in window.bind_button.toolTip()
    now = time.perf_counter_ns()
    for role in ("front", "wrist"):
        window._decoded(role, now, image)
    window.poll()
    assert window.bind_button.isEnabled()
    window._preview_frame_failed("front", now + 1, "解码失败")
    window.poll()
    assert not window.bind_button.isEnabled()
    assert "解码失败" in window.bind_button.toolTip()
    assert controller.calls == [("preview_cameras", {"front": "front", "wrist": "wrist"})]


def test_communication_label_shows_explicit_startup_selection(application, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    module = importlib.import_module("desktop.window")
    controller = FakeController()
    controller.command_feedback_mode = "cached"
    controller.command_tx_gap_us = 1000.0
    window = module.DesktopWindow(controller, demo=True)
    try:
        assert "通信模式 cached" in window.communication_label.text()
        assert window.home_communication_label.text() == window.communication_label.text()
        assert "1000 μs" in window.communication_label.text()
        assert not controller.calls
    finally:
        controller.closed = True
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        application.processEvents()


def test_button_feedback_is_immediate_and_save_never_releases(desktop):
    window, controller = desktop
    started = time.perf_counter()
    window.save_button.click()
    assert time.perf_counter() - started < 0.1
    assert controller.calls == [("record_finish", {"task_result": "unknown"})]
    assert "保存" in window.feedback.text()
    assert not any(action == "release" for action, _ in controller.calls)


def test_esc_uses_direct_stop_with_pending_form_and_decode(desktop, application):
    window, controller = desktop
    window._form_dirty = True
    window._decode_busy = True
    window.activateWindow()
    window.setFocus()
    application.processEvents()
    QtTest.QTest.keyClick(window, QtCore.Qt.Key.Key_Escape)
    application.processEvents()
    assert controller.calls == [("stop", None)]


def test_running_locks_fixed_fields_but_keeps_live_target_and_stop(desktop):
    window, controller = desktop
    controller.view.state = "running"
    controller.view.capabilities.update(
        edit_values="运行中锁定", edit_recording="录制期间锁定", start="已启动"
    )
    controller.view.revision += 1
    window.poll()
    assert not window._field_edits["port"][0].isEnabled()
    assert window._field_edits["target_N"][0].isEnabled()
    assert not window._recording_edits["task"].isEnabled()
    assert window.stop_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.start_button.toolTip() == "已启动"


def test_hidden_page_disables_preview_without_stopping_controller(desktop):
    window, controller = desktop
    window.navigation.setCurrentRow(1)
    assert not controller.preview_enabled
    window.navigation.setCurrentRow(0)
    assert controller.preview_enabled
    assert not controller.calls


def test_interrupted_recording_and_retained_cleanup_have_explicit_final_labels(desktop):
    window, controller = desktop
    controller.view.state = "recovery_required"
    controller.view.recording = {
        "state": "interrupted",
        "active": False,
        "file_complete": False,
        "cleanup_pending": True,
        "cleanup_error": "writer still running",
    }
    window._render_snapshot(controller.view)
    assert "已中断，数据不完整" in window.record_state.text()
    assert "资源仍被占用" in window.record_state.text()
    assert "录制中" not in window.record_state.text()
    controller.view.recording.update(cleanup_pending=False, cleanup_error=None)
    window._render_snapshot(controller.view)
    assert "资源仍被占用" not in window.record_state.text()
    assert not controller.calls


def test_preview_decode_is_worker_only_and_latest_images_survive_bad_frame(desktop, application):
    window, controller = desktop
    controller.view.state = "monitoring"
    controller.view.revision += 1
    window.poll()
    image = QtGui.QImage(24, 24, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor("#087f71"))
    encoded = QtCore.QByteArray()
    buffer = QtCore.QBuffer(encoded)
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(buffer, "JPEG")
    source_timestamp = time.perf_counter_ns()
    controller.frames = {"front": {"jpeg": bytes(encoded), "timestamp_ns": source_timestamp}}
    window.poll()
    assert window._decode_busy
    deadline = time.monotonic() + 2
    while window._decode_busy and time.monotonic() < deadline:
        application.processEvents()
        QtTest.QTest.qWait(5)
    assert not window._decode_busy
    assert window._preview_timestamps["front"] == source_timestamp
    assert window.preview_labels["front"]._source is not None
    original = window.preview_labels["front"]._source.cacheKey()
    controller.frames = {"front": {"jpeg": b"broken", "timestamp_ns": source_timestamp + 1}}
    window.poll()
    deadline = time.monotonic() + 2
    while window._decode_busy and time.monotonic() < deadline:
        application.processEvents()
        QtTest.QTest.qWait(5)
    assert window.preview_labels["front"]._source.cacheKey() == original
    assert "解码失败" in window.preview_status["front"].text()
    window.poll()
    assert not window._decode_busy  # Do not decode the same broken frame repeatedly.
    assert not controller.calls


def test_plot_cache_is_bounded_and_reuses_items(desktop):
    window, _ = desktop
    assert window.force_plot.getPlotItem().legend.columnCount == 2
    assert len(window.force_plot.getPlotItem().legend.items) == 4
    curves = tuple(window.charts._curves[window.force_plot])
    for tick in range(2000):
        window.charts.append({"decision_anchor_ns": tick * 20_000_000, "force_left_N": 1.0})
    assert len(window.charts.points) <= 1001
    window.charts.render()
    window.charts.render()
    assert tuple(window.charts._curves[window.force_plot]) == curves


@pytest.mark.parametrize("width,height", [(1050, 700), (1366, 768)])
def test_tuning_force_plot_shows_live_values_and_marks_invalid_and_stopped_feedback(
    desktop, application, width, height
):
    window, controller = desktop
    window.resize(width, height)
    window.navigation.setCurrentRow(1)
    controller.view.state = "running"
    controller.view.telemetry = {
        "decision_anchor_ns": time.perf_counter_ns(),
        "phase": "FORCE_TRACK",
        "force_valid": True,
        "force_left_N": 1.2,
        "force_right_N": 1.4,
        "force_mean_N": 1.3,
        "force_target_requested_N": 2.0,
    }
    controller.view.revision += 1
    window.poll()
    application.processEvents()
    window._render_charts()
    for widget in (window.tuning_force_plot, window.tuning_force_status):
        point = widget.mapTo(window.tuning_page.viewport(), QtCore.QPoint())
        assert window.tuning_page.viewport().rect().contains(point)
        assert (
            window.tuning_page.viewport()
            .rect()
            .contains(point + QtCore.QPoint(widget.width() - 1, widget.height() - 1))
        )
    curves = dict(window.charts._curves[window.tuning_force_plot])
    for key, expected in (
        ("force_left_N", 1.2),
        ("force_right_N", 1.4),
        ("force_mean_N", 1.3),
        ("force_target_requested_N", 2.0),
    ):
        assert curves[key].getData()[1][-1] == pytest.approx(expected)
    controller.view.telemetry.update(
        decision_anchor_ns=time.perf_counter_ns(), force_valid=False, force_left_N=99, force_right_N=99
    )
    controller.view.revision += 1
    window.poll()
    window._render_charts()
    assert math.isnan(curves["force_left_N"].getData()[1][-1])
    assert "力反馈无效" in window.tuning_force_status.text()
    controller.view.state = "disconnected"
    controller.view.revision += 1
    window.poll()
    assert "无实时反馈" in window.tuning_force_status.text()
    assert not controller.calls


def test_acceleration_toggle_and_inactive_value_are_clear(desktop, application):
    window, controller = desktop
    controller.view.toggles["acceleration"] = False
    window._hydrated = False
    controller.view.revision += 1
    window.poll()
    window.navigation.setCurrentRow(1)
    application.processEvents()
    toggle = window._toggle_checks["acceleration"]
    edit = window._field_edits["acceleration_limit_deg_s2"][0]
    point = toggle.mapTo(window.tuning_page.viewport(), QtCore.QPoint())
    assert toggle.isVisible() and window.tuning_page.viewport().rect().contains(point)
    assert not toggle.isChecked() and toggle.text() == "已关闭"
    assert edit.text() == "0.5" and not edit.isEnabled()
    assert window._form_payload()["toggles"]["acceleration"] is False
    toggle.setChecked(True)
    assert edit.isEnabled() and toggle.text() == "已启用"
    toggle.setChecked(False)
    assert not edit.isEnabled()
    assert not controller.calls


def test_camera_confirmation_needs_new_frames_for_both_roles(desktop):
    window, controller = desktop
    controller.view.capabilities["bind_cameras"] = ""
    window._last_revision = None
    window.poll()
    assert not window.bind_button.isEnabled()
    for role, choice in window.camera_choices.items():
        choice.addItem(role, role)
        choice.setCurrentIndex(choice.count() - 1)
    window.dispatch("preview_cameras", window._camera_selection(), update=False)
    assert controller.calls[-1] == ("preview_cameras", {"front": "front", "wrist": "wrist"})
    assert window._binding_active and not window._binding_seen


def test_manual_step_and_freeze_are_absent_from_the_workbench(desktop):
    window, controller = desktop
    assert "manual_step_deg" not in window._field_edits
    assert "manual_step_deg" not in window._form_payload()["values"]
    actions = {action for _, action in window._buttons}
    assert not {"step", "freeze"} & actions
    assert {"position", "resume", "approach"} <= actions
    assert not controller.calls


def test_form_and_start_are_one_request_and_failed_draft_is_retained(desktop):
    window, controller = desktop
    window._field_changed("target_N", "bad-value")
    window.confirmed.setChecked(True)
    window.start_button.click()
    assert len(controller.calls) == 1
    action, payload = controller.calls[0]
    assert action == "start"
    assert payload["form"]["values"]["target_N"] == "bad-value"
    assert window._form_dirty
    controller.view.operation_results = (
        {"request_id": "1", "action": "start", "success": False, "message": "参数无效"},
    )
    controller.view.revision += 1
    window.poll()
    assert window._form_dirty
    assert window._field_edits["target_N"][0].text() == "bad-value"


def test_connect_uses_visible_port_draft_in_same_request(desktop):
    window, controller = desktop
    window._field_changed("port", "COM99")
    window.connect_button.click()
    assert len(controller.calls) == 1
    action, payload = controller.calls[0]
    assert action == "connect"
    assert payload["form"]["values"]["port"] == "COM99"


def test_tuning_has_no_calculator_and_gains_remain_manually_editable(desktop, application):
    window, controller = desktop
    window.navigation.setCurrentRow(1)
    application.processEvents()
    assert not any(
        "t90" in button.text().lower() for button in window.tuning_page.findChildren(QtWidgets.QToolButton)
    )
    for key, value in (("kp_deg_per_N", "0.5"), ("ki_deg_per_N_s", "0.25")):
        edit = window._field_edits[key][0]
        assert edit.isVisible() and edit.isEnabled()
        edit.clear()
        QtTest.QTest.keyClicks(edit, value)
        assert window._form_payload()["values"][key] == value
    assert window._form_dirty
    assert not controller.calls


def test_late_form_success_does_not_discard_newer_local_edit(desktop):
    window, controller = desktop
    window._field_changed("target_N", "3")
    window.dispatch("preflight")
    window._field_changed("target_N", "4")
    controller.view.operation_results = (
        {"request_id": "1", "action": "preflight", "success": True, "message": "检查通过"},
    )
    controller.view.revision += 1
    window.poll()
    assert window._form_dirty
    assert window._field_edits["target_N"][0].text() == "4"


def test_failed_session_and_verified_episode_are_shown_independently(desktop):
    window, controller = desktop
    controller.view.error = "左侧力超限"
    controller.view.state = "recovery_required"
    controller.view.exit_confirmed = False
    controller.view.recording = {
        "state": "saved",
        "task_result": "unknown",
        "completed_episode_count": 1,
        "file_complete": True,
    }
    controller.view.revision += 1
    window.poll()
    assert "左侧力超限" in window.diagnostic_error.text()
    assert "已保存，视频核验未记录" in window.diagnostic_data.text()
    assert "未确认" in window.diagnostic_data.text()
    assert "复核" in window.diagnostic_exit.text()
    assert window.record_badge.text() == "录制：已保存"


def test_video_verification_controls_and_state_are_distinct(desktop, application):
    window, controller = desktop
    controller.view.recording = {
        "state": "saved",
        "file_complete": True,
        "video_verification": "not_run",
        "episode_uid": "saved-episode",
        "save_timings": {
            "drain_seconds": 0.1,
            "transaction_seconds": 0.2,
            "non_video_seconds": 0.3,
            "video_decode_seconds": 0.0,
            "total_seconds": 0.7,
        },
        "full_verification": {"state": "running", "index": 1, "total": 2, "video": "front.mp4"},
    }
    controller.view.capabilities["verify_dataset"] = "请等待完整核验结束或取消"
    controller.view.capabilities["cancel_verification"] = ""
    controller.view.revision += 1
    window.poll()
    assert "视频待完整核验" in window.record_state.text()
    assert "排空 0.10s" in window.diagnostic_data.text()
    assert "视频解码 0.00s" in window.operation_log.toPlainText()
    assert "1 / 2" in window.verification_status.text() and "front.mp4" in window.verification_status.text()
    assert not window.verify_button.isEnabled() and window.cancel_verify_button.isEnabled()
    window.verify_video_checkbox.setChecked(True)
    application.processEvents()
    assert window._form_payload()["recording_values"]["verify_video_on_save"] is True
    window.cancel_verify_button.click()
    assert controller.calls[-1][0] == "cancel_verification"
    controller.view.recording["full_verification"] = {"state": "passed"}
    controller.view.revision += 1
    window.poll()
    assert "本次全部通过" in window.verification_status.text()
    assert "视频已完整核验" in window.record_state.text()
    window._recording_edits["root"].setText("another dataset")
    assert "待重新核验" in window.verification_status.text()


def test_full_verification_result_appears_reopens_and_clears_after_append(desktop, tmp_path, application):
    window, controller = desktop
    root = tmp_path / "dataset"
    root.mkdir()
    evidence = root / "evidence.txt"
    evidence.write_text("unchanged", encoding="utf-8")
    before = (evidence.read_bytes(), evidence.stat().st_mtime_ns)
    report_dir = tmp_path / "dataset_verification_reports"
    report_dir.mkdir()
    report_path = report_dir / "verification_example.json"
    report = {
        "scope": "all_saved_episodes",
        "root": str(root),
        "state": "passed",
        "elapsed_seconds": 16.5,
        "result": {
            "valid": True,
            "errors": [],
            "excluded_episodes": [],
            "episodes": [
                {"episode_index": 1, "valid": True, "frame_count": 1304, "video_decoded": True},
                {"episode_index": 0, "valid": True, "frame_count": 1469, "video_decoded": True},
            ],
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    controller.view.recording_values["root"] = str(root)
    window._recording_edits["root"].setText(str(root))
    controller.view.recording = {
        "state": "saved",
        "file_complete": True,
        "video_verification": "not_run",
        "full_verification": {"state": "running", "root": str(root), "index": 1, "total": 2},
    }
    controller.view.revision += 1
    window.poll()
    assert not window.verification_result_button.isEnabled()
    assert "1 / 2" in window.verification_inline.text()
    controller.view.recording["full_verification"] = {
        "state": "passed",
        "root": str(root),
        "started_utc": "start",
        "completed_utc": "finish",
        "elapsed_seconds": 16.5,
        "report_path": str(report_path),
        "errors": [],
    }
    controller.view.revision += 1
    window.poll()
    application.processEvents()
    dialog = window._verification_dialog
    details = dialog.findChild(QtWidgets.QPlainTextEdit, "verificationResultDetails").toPlainText()
    assert dialog.isVisible()
    assert "完整核验结果：通过" in details and "总帧数：2,773 帧" in details
    assert "第 1 条：通过 · 1,469 帧 · 视频已完整解码" in details
    assert "第 2 条：通过 · 1,304 帧 · 视频已完整解码" in details
    assert window.verification_result_button.text() == "结果：通过"
    dialog.close()
    controller.view.revision += 1
    window.poll()
    assert window._verification_dialog is dialog and not dialog.isVisible()
    window.verification_result_button.click()
    assert window._verification_dialog.isVisible()
    controller.view.recording["full_verification"] = {"state": "idle"}
    controller.view.revision += 1
    window.poll()
    assert not window._verification_dialog.isVisible()
    assert window.verification_result_button.text() == "查看结果"
    assert (evidence.read_bytes(), evidence.stat().st_mtime_ns) == before


def test_verification_failure_cancel_and_root_draft_are_visible(desktop, tmp_path, application):
    window, controller = desktop
    root = tmp_path / "dataset"
    report_dir = tmp_path / "dataset_verification_reports"
    report_dir.mkdir()
    report_path = report_dir / "verification_failure.json"
    report_path.write_text(
        json.dumps(
            {
                "scope": "all_saved_episodes",
                "root": str(root),
                "result": {
                    "valid": False,
                    "errors": ["bad-episode: corrupt video"],
                    "episodes": [{"episode_uid": "bad-episode", "valid": False, "errors": ["corrupt video"]}],
                    "excluded_episodes": [],
                },
            }
        ),
        encoding="utf-8",
    )
    controller.view.recording_values["root"] = str(root)
    window._recording_edits["root"].setText(str(root))
    controller.view.recording["full_verification"] = {
        "state": "failed",
        "root": str(root),
        "started_utc": "one",
        "report_path": str(report_path),
        "errors": ["bad-episode: corrupt video"],
    }
    controller.view.revision += 1
    window.poll()
    application.processEvents()
    details = window._verification_dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    assert "完整核验结果：失败" in details and "bad-episode" in details and "corrupt video" in details
    assert window.verification_result_button.text() == "结果：失败"
    window._recording_edits["root"].setText(str(tmp_path / "different"))
    assert not window._verification_dialog.isVisible()
    assert window.verification_result_button.text() == "查看结果"
    window._recording_edits["root"].setText(str(root))
    controller.view.recording["full_verification"] = {
        "state": "cancelled",
        "root": str(root),
        "started_utc": "two",
        "errors": ["核验已取消"],
    }
    controller.view.revision += 1
    window.poll()
    details = window._verification_dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    assert "完整核验结果：未完成" in details and "核验已取消" in details


def test_old_report_can_be_viewed_as_history_without_claiming_current_pass(desktop, tmp_path):
    window, controller = desktop
    root = tmp_path / "dataset"
    report_dir = tmp_path / "dataset_verification_reports"
    report_dir.mkdir()
    (report_dir / "verification_old.json").write_text(
        json.dumps(
            {
                "scope": "all_saved_episodes",
                "root": str(root),
                "state": "passed",
                "result": {
                    "valid": True,
                    "episodes": [{"valid": True, "episode_index": 0, "frame_count": 20}],
                },
            }
        ),
        encoding="utf-8",
    )
    controller.view.recording_values["root"] = str(root)
    controller.view.recording["full_verification"] = {"state": "idle"}
    window._recording_edits["root"].setText(str(root))
    controller.view.revision += 1
    window.poll()
    assert window._verification_dialog is None
    assert window.verification_result_button.text() == "查看结果"
    window.verification_result_button.click()
    details = window._verification_dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    assert "历史报告" in details and "不能作为当前数据集全部通过" in details
    assert "第 1 条：通过 · 20 帧" in details
    assert "本次全部通过" not in window.verification_status.text()


def test_foreign_historical_report_is_not_presented_as_a_pass(desktop, tmp_path):
    window, controller = desktop
    root = tmp_path / "dataset"
    report_dir = tmp_path / "dataset_verification_reports"
    report_dir.mkdir()
    (report_dir / "verification_wrong_root.json").write_text(
        json.dumps({"scope": "all_saved_episodes", "root": str(tmp_path / "other"), "state": "passed"}),
        encoding="utf-8",
    )
    controller.view.recording_values["root"] = str(root)
    window._recording_edits["root"].setText(str(root))
    controller.view.revision += 1
    window.poll()
    window.verification_result_button.click()
    dialog = window._verification_dialog
    details = dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText()
    assert "无法读取报告" in details and "不一致" in details
    assert "完整核验结果：通过" not in details
    assert not dialog.findChild(QtWidgets.QPushButton, "openVerificationReport").isEnabled()


def test_verification_start_failure_is_shown_directly(desktop):
    window, controller = desktop
    controller.view.operation_results = (
        {"request_id": "verify-1", "action": "verify_dataset", "success": False, "message": "目录不存在"},
    )
    controller.view.revision += 1
    window.poll()
    dialog = window._verification_start_error_dialog
    assert dialog.isVisible()
    assert "完整核验未启动" in dialog.windowTitle() and "目录不存在" in dialog.text()


def test_badge_color_distinguishes_fault_busy_and_active_states(desktop):
    window, controller = desktop
    for state, expected_background in (
        ("recovery_required", "#fde5e6"),
        ("connecting", "#fff0d3"),
        ("disconnected", "#e8edf3"),
        ("running", "#e1f3ed"),
    ):
        controller.view.state = state
        controller.view.revision += 1
        window.poll()
        background = window.state_badge.palette().color(QtGui.QPalette.ColorRole.Window).name()
        assert background == expected_background


def test_real_demo_controller_hydrates_then_runs_record_save_without_hardware(application, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/force_teacher"))
    backend = importlib.import_module("b601_session")
    ui = importlib.import_module("desktop.window")

    def reject_hardware(*args, **kwargs):
        pytest.fail("offline desktop opened hardware")

    monkeypatch.setattr(backend, "ReadonlyB601Transport", reject_hardware)
    monkeypatch.setattr(backend, "query_disabled_arm", reject_hardware)
    controller = backend.SessionController(demo=True)
    window = ui.DesktopWindow(controller, demo=True)
    controller.start()
    window.show()

    def wait_until(predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            application.processEvents()
            QtTest.QTest.qWait(5)
            window.poll()
            if predicate():
                return
        pytest.fail(f"desktop did not settle: {controller.snapshot().state}")

    try:
        wait_until(lambda: controller.snapshot().state == "disconnected")
        assert window.recording_enabled.isChecked()
        assert window._field_edits["target_N"][0].text() == "5"
        assert window._recording_edits["task"].text() == "离线演示"
        window.connect_button.click()
        wait_until(lambda: controller.snapshot().state == "monitoring")
        window.confirmed.setChecked(True)
        window.start_button.click()
        wait_until(lambda: controller.snapshot().state == "running")
        wait_until(lambda: window.record_button.isEnabled())
        window.record_button.click()
        wait_until(lambda: controller.snapshot().recording.get("active") is True)
        first_uid = controller.snapshot().recording["episode_uid"]
        window.outcome.setCurrentIndex(window.outcome.findData("success"))
        window.save_button.click()
        wait_until(lambda: controller.snapshot().recording.get("state") == "saved")
        assert controller.snapshot().state == "running"
        assert controller.snapshot().recording.get("task_result") == "success"
        assert controller.snapshot().recording.get("training_eligible") is False
        assert "未写正式数据" in window.record_state.text()
        wait_until(lambda: window.record_button.isEnabled())
        window.record_button.click()
        wait_until(
            lambda: (
                controller.snapshot().recording.get("active") is True
                and controller.snapshot().recording.get("episode_uid") != first_uid
            )
        )
        assert window.outcome.currentData() == "unknown"
        window.save_button.click()
        wait_until(lambda: controller.snapshot().recording.get("state") == "saved")
        assert controller.snapshot().recording.get("task_result") == "unknown"
    finally:
        controller.close()
        wait_until(lambda: controller.closed)
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        application.processEvents()
