"""Chinese operation workspace. Widgets render detached controller snapshots only."""

import json
import math
import queue
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

from b601_session import DEFAULT_SETTINGS
from communication_options import communication_description

from lerobot.force_control.operator_settings import ARM_OPERATOR_FIELDS, ARM_PROFILE_FIELDS
from lerobot.force_control.tuning import FIELDS, GRIPPER_DEVICE_FIELDS, LIVE_FIELDS, OPTIONAL_GROUPS
from lerobot.utils.import_utils import _pyside6_available, require_package

if _pyside6_available:
    from PySide6 import QtCore, QtGui, QtWidgets

from .charts import TelemetryCharts
from .problem_details import explain_problems
from .widgets import CollapsibleBox, MetricCard, PreviewDecoder, PreviewLabel, card, scroll_page

require_package("PySide6", "force-teacher-ui")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
JOINT_LABELS = ("肩部旋转", "肩部抬升", "肘部", "腕部俯仰", "腕部偏转", "腕部旋转")
STATE_LABELS = {
    "loading": "正在加载配置",
    "preflighting": "正在预检",
    "disconnected": "未连接",
    "connecting": "连接中",
    "initializing": "正在加载配置",
    "monitoring": "只读监测",
    "preparing": "准备中",
    "starting": "启动中",
    "running": "遥操运行中",
    "stopping": "停止中",
    "recovering": "只读复核中",
    "recovery_required": "等待退出复核",
    "closing": "正在关闭",
    "releasing": "正在释放并结束采集",
    "closed": "已关闭",
    "fault": "会话异常",
    "ready": "已准备",
    "idle": "未录制",
    "recording": "录制中",
    "saving": "保存中",
    "verifying": "核验中",
    "saved": "已保存",
    "saved_pending_verification": "已保存，待核验",
    "commit_unknown": "保存结果待确认",
    "interrupted": "已中断，数据不完整",
    "corrupt": "数据核验失败",
    "error": "发生异常",
}
OUTCOMES = {"unknown": "未确认", "success": "成功", "failure": "失败"}
ACTION_HELP = {
    "connect": "只读连接设备并监测反馈，不使能机械臂。",
    "disconnect": "断开当前只读监测。",
    "start": "需要先连接并完成本次机械、校准与力通道确认。",
    "record_start": "开始一条示范录制；不会启动或使能机械臂。",
    "record_finish": "结束并保存这一条，遥操继续，不追加运动。",
    "record_discard": "丢弃当前录制并开始新的一条。",
    "record_outcome": "更新当前或最近一条已保存记录的人工任务结果。",
    "position": "以指定速度移动夹爪至校准坐标；在主从模式下会退出自动跟随。",
    "position_start": "启动单夹爪控制并移动至校准坐标；本次不录制示范数据。",
    "verify_dataset": "停止采集并确认资源退出后，完整核验当前数据集。",
    "cancel_verification": "取消正在进行的完整核验。",
    "bind_cameras": "核对两路画面后保存角色绑定。",
}
PHASE_LABELS = {
    "DISABLED": "失能",
    "ARMED": "已准备",
    "FOLLOW": "主从跟随",
    "FOLLOW_WAIT_CLEAR": "等待重新接触",
    "FORCE_TRACK": "力控制",
    "POSITION_HOLD": "固定位置",
    "APPROACH": "接近中",
    "RELEASE": "主动释放中",
    "FAULT": "故障停止",
}

STYLE = """
QMainWindow, QWidget { background: #f3f6fa; color: #243247; font-family: 'Microsoft YaHei UI'; }
QLabel { background: transparent; }
QLabel#title { font-size: 20px; font-weight: 700; color: #12263b; }
QLabel#subtitle, QLabel#caption { color: #64748b; }
QLabel#pageTitle { font-size: 22px; font-weight: 700; }
QLabel#stateBadge { background: #e8edf3; color: #526175; border-radius: 12px; padding: 7px 14px; font-weight: 600; }
QLabel#stateBadge[stateTone="active"] { background: #e1f3ed; color: #076952; }
QLabel#stateBadge[stateTone="busy"] { background: #fff0d3; color: #805313; }
QLabel#stateBadge[stateTone="fault"] { background: #fde5e6; color: #a52632; }
QLabel#previewStatus { color: #64748b; font-size: 11px; }
QLabel#previewStatus[statusTone="fault"] { color: #a52632; font-weight: 600; }
QFrame#topBar { background: #ffffff; border-bottom: 1px solid #dce4ed; }
QFrame#metricCard { background: #ffffff; border: 1px solid #e0e7ef; border-radius: 9px; }
QFrame#verificationCard { background: #ffffff; border: 1px solid #dce4ed; border-radius: 9px; }
QFrame#verificationCompact { background: transparent; border: none; }
QGroupBox { background: #ffffff; border: 1px solid #dce4ed; border-radius: 10px; margin-top: 13px; padding: 14px 10px 9px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0px 5px; background: #f3f6fa; }
QPushButton { background: #ffffff; border: 1px solid #ccd7e4; border-radius: 6px; padding: 8px 12px; min-height: 18px; }
QPushButton:hover { background: #edf4fc; border-color: #82a8d1; }
QPushButton:pressed { background: #dbeafe; }
QPushButton:disabled { background: #f1f4f7; color: #96a2b2; border-color: #e2e8f0; }
QPushButton#primary { color: white; background: #087f71; border-color: #087f71; font-weight: 600; }
QPushButton#primary:hover { background: #066b60; }
QPushButton#primary:disabled { color: #95aeb0; background: #e4eeee; border-color: #d4e4e4; }
QPushButton#positionButton { color: #087f71; border-color: #74aa9f; font-weight: 600; }
QPushButton#positionButton:disabled { color: #52677c; background: #e5ebf1; border-color: #becbd8; }
QPushButton#stopButton { color: white; background: #c43d46; border-color: #c43d46; font-size: 14px; font-weight: 700; min-width: 136px; }
QPushButton#stopButton:hover { background: #a62e36; }
QPushButton#problemDetailsButton[problemTone="fault"] { color: #a52632; border-color: #d8868d; background: #fde5e6; }
QListWidget#navigation { background: #eaf0f7; border: 0px; padding: 12px 7px; font-size: 14px; outline: none; }
QListWidget#navigation::item { padding: 13px 9px; margin: 3px 0px; border-radius: 7px; }
QListWidget#navigation::item:selected { background: #ffffff; color: #087f71; font-weight: 700; }
QLineEdit, QComboBox, QPlainTextEdit, QTableWidget { background: #ffffff; border: 1px solid #d2dce8; border-radius: 5px; padding: 5px 7px; selection-background-color: #b9e0da; }
QLineEdit, QComboBox { min-height: 18px; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: #159586; }
QLineEdit:disabled, QComboBox:disabled { background: #edf1f5; color: #768599; }
QCheckBox { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #647991; border-radius: 3px; background: #ffffff; }
QCheckBox::indicator:hover { border-color: #087f71; }
QCheckBox::indicator:checked { border-color: #087f71; background: #087f71; image: url("__CHECKMARK__"); }
QCheckBox::indicator:disabled { border-color: #8a9aaa; background: #e6ebf0; }
QCheckBox::indicator:checked:disabled { border-color: #71978f; background: #71978f; image: url("__CHECKMARK__"); }
QScrollArea { border: none; background: transparent; }
QLabel#cameraPreview { background: #102b38; color: #bed0d9; border-radius: 7px; font-size: 13px; }
QLabel#notice { background: #fff4df; color: #805b1c; border-radius: 6px; padding: 8px 12px; }
QLabel#notice[statusTone="fault"] { background: #fde5e6; color: #a52632; }
QLabel#feedback { padding: 5px 12px; color: #40556e; background: #eaf0f7; }
QToolButton { border: none; color: #3d6688; padding: 7px 0; background: transparent; }
QHeaderView::section { background: #edf3f8; color: #415772; border: 0; padding: 7px; }
""".replace("__CHECKMARK__", (Path(__file__).resolve().parent / "checkmark.svg").as_posix())


def label(text="", *, name=None, wrap=False):
    widget = QtWidgets.QLabel(text)
    widget.setWordWrap(wrap)
    if name:
        widget.setObjectName(name)
    return widget


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


class DesktopWindow(QtWidgets.QMainWindow):
    """UI command dispatcher with bounded rendering and explicit lifecycle feedback."""

    decode_requested = QtCore.Signal(object)

    def __init__(self, controller, *, demo=False, gripper_only=False):
        super().__init__()
        self.controller = controller
        self.demo, self.gripper_only = demo, gripper_only
        self._snapshot = None
        self._last_revision = None
        self._hydrated = False
        self._form_dirty = self._arm_dirty = False
        self._closing = False
        self._decode_busy = False
        self._preview_interest = None
        self._preview_timestamps = {}
        self._home_preview_timestamps = {}
        self._binding_preview_timestamps = {}
        self._preview_errors = {}
        self._preview_session_id = None
        self._preview_epoch_ns = 0
        self._binding_seen = set()
        self._binding_started_ns = 0
        self._binding_active = False
        self._camera_configs = {}
        self._device_signature = None
        self._settings_file = None
        self._last_results = ()
        self._last_episode_identity = None
        self._last_save_timing_identity = None
        self._verification_layout_mode = None
        self._last_verification_popup = None
        self._verification_dialog = None
        self._verification_dialog_token = None
        self._verification_dialog_historical = False
        self._verification_dialog_root = None
        self._verification_start_error_dialog = None
        self._last_problem = None
        self._problem_history = deque(maxlen=10)
        self._current_problem_key = None
        self._problem_dialog = None
        self._last_root_draft = False
        self._pending_forms = {}
        self._buttons = []
        self._field_edits = {}
        self._field_specs = {field[1]: field for field in FIELDS}
        self._position_command_edits = {}
        self._position_command_dirty = set()
        self._position_start_pending = None
        self._toggle_checks = {}
        self._recording_edits = {}
        self._arm_edits = {}
        self.charts = TelemetryCharts(seconds=20)
        self.setWindowTitle("B601 操作工作台 · 50 Hz 采集版" + (" · 离线演示" if demo else ""))
        self.setMinimumSize(1050, 700)
        self.resize(1366, 900)
        self.setStyleSheet(STYLE)
        self._build()
        self._setup_decoder()
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self.poll)
        self.status_timer.start()
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setInterval(50)
        self.plot_timer.timeout.connect(self._render_charts)
        self.plot_timer.start()
        self.stop_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.stop_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.stop_shortcut.activated.connect(self.stop)
        self.preferences = QtCore.QSettings("LeRobotLocal", "B601Desktop")
        geometry = self.preferences.value("window_geometry")
        if geometry is not None and not demo:
            self.restoreGeometry(geometry)
        self.poll()

    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QtWidgets.QFrame()
        top.setObjectName("topBar")
        top_layout = QtWidgets.QHBoxLayout(top)
        top_layout.setContentsMargins(20, 8, 20, 8)
        brand = QtWidgets.QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(label("B601 操作工作台", name="title"))
        brand.addWidget(label("遥操作 · 力控制 · 示范录制", name="subtitle"))
        top_layout.addLayout(brand)
        top_layout.addStretch()
        self.state_badge = label("正在加载配置", name="stateBadge")
        top_layout.addWidget(self.state_badge)
        self.record_badge = label("录制：未开始", name="stateBadge")
        top_layout.addWidget(self.record_badge)
        self.problem_button = QtWidgets.QPushButton("问题详情")
        self.problem_button.setObjectName("problemDetailsButton")
        self.problem_button.setToolTip("查看当前和最近的问题，复制完整原始报错。")
        self.problem_button.clicked.connect(self.show_problem_details)
        top_layout.addWidget(self.problem_button)
        if self.demo:
            demo_badge = label("离线演示 · 合成数据", name="notice")
            top_layout.addWidget(demo_badge)
        self.stop_button = QtWidgets.QPushButton("停止 / 失能  Esc")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop)
        top_layout.addWidget(self.stop_button)
        outer.addWidget(top)
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QtWidgets.QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setFixedWidth(162)
        self.navigation.addItems(("遥操与录制", "调参", "设备与配置", "诊断与记录"))
        body.addWidget(self.navigation)
        self.pages = QtWidgets.QStackedWidget()
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)
        self.feedback = label("正在加载参数；设备连接需要手动操作。", name="feedback", wrap=True)
        self.feedback.setMinimumHeight(31)
        self.feedback.setMaximumHeight(65)
        outer.addWidget(self.feedback)
        self._build_recording()
        self._build_tuning()
        self._build_devices()
        self._build_diagnostics()
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.currentRowChanged.connect(self._page_changed)
        self.navigation.setCurrentRow(0)

    def _button(self, text, action, *, primary=False, payload=None, update=True):
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.setProperty("action", action)
        button.setToolTip(ACTION_HELP.get(action, ""))
        button.clicked.connect(lambda: self.dispatch(action, payload, update=update))
        self._buttons.append((button, action))
        return button

    def _field(self, key, parent_layout, caption=None):
        edit = self._new_field(key)
        parent_layout.addRow(label(caption or self._field_specs[key][2], wrap=True), edit)
        return edit

    def _new_field(self, key):
        spec = self._field_specs[key]
        edit = QtWidgets.QLineEdit(str(spec[3]))
        edit.setObjectName(f"field_{key}_{len(self._field_edits.get(key, []))}")
        edit.setMinimumWidth(95)
        edit.setMaximumWidth(230)
        edit.textEdited.connect(lambda text, k=key: self._field_changed(k, text))
        self._field_edits.setdefault(key, []).append(edit)
        return edit

    def _field_changed(self, key, text):
        self._form_dirty = True
        for edit in self._field_edits[key]:
            if edit.text() != text:
                with QtCore.QSignalBlocker(edit):
                    edit.setText(text)
        if key in self._position_command_edits and key not in self._position_command_dirty:
            self._position_command_edits[key].setText(text)

    def _acceleration_changed(self, *_):
        self._mark_dirty()
        if self._snapshot is not None:
            self._refresh_acceleration_field(self._snapshot)

    def _refresh_acceleration_field(self, snapshot):
        caps = snapshot.capabilities
        editable = not caps.get("edit_values", "等待当前操作完成") or not caps.get(
            "edit_live", "等待当前操作完成"
        )
        check = self._toggle_checks["acceleration"]
        check.setText("已启用" if check.isChecked() else "已关闭")
        enabled = check.isChecked() and editable and not self._closing
        for edit in self._field_edits["acceleration_limit_deg_s2"]:
            edit.setEnabled(enabled)
            edit.setToolTip("仅勾选加速度限制时生效。")

    def _add_acceleration_field(self, form):
        edit = self._new_field("acceleration_limit_deg_s2")
        edit.setMinimumWidth(60)
        check = QtWidgets.QCheckBox()
        check.setAccessibleName("启用位置变化率加速度限制")
        check.setChecked(OPTIONAL_GROUPS["acceleration"][1])
        check.setToolTip("未勾选时，此数值不会用于限制；勾选后须应用参数。")
        check.toggled.connect(self._acceleration_changed)
        self._toggle_checks["acceleration"] = check
        row = QtWidgets.QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(check)
        form.addRow(label("位置变化率加速度 (°/s²)", wrap=True), row)

    def _build_recording(self):
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(18, 8, 18, 8)
        page_layout.setSpacing(6)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label("遥操与录制", name="pageTitle"))
        heading.addStretch()
        self.home_communication_label = label(
            communication_description(
                getattr(self.controller, "command_feedback_mode", None),
                getattr(self.controller, "command_tx_gap_us", 120),
                getattr(self.controller, "command_tx_gap_source", "配置"),
                gripper_only=getattr(self.controller, "gripper_only", False),
            ),
            name="subtitle",
            wrap=True,
        )
        heading.addWidget(self.home_communication_label)
        page_layout.addLayout(heading)
        self.readiness = label("连接设备后检查反馈、校准与录制准备状态。", name="notice", wrap=True)
        page_layout.addWidget(self.readiness)
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(13)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        metrics = QtWidgets.QHBoxLayout()
        self.metrics = {}
        for key, title, color in (
            ("force_left_N", "左侧力", "#0891b2"),
            ("force_right_N", "右侧力", "#e27728"),
            ("force_mean_N", "平均力", "#7951bc"),
            ("force_target_requested_N", "请求目标", "#087f71"),
        ):
            metric = MetricCard(title, color)
            self.metrics[key] = metric
            metrics.addWidget(metric)
        left.addLayout(metrics)
        cameras = QtWidgets.QHBoxLayout()
        cameras.setSpacing(10)
        self.preview_labels = {}
        self.preview_status = {}
        for role, title in (("front", "前景相机"), ("wrist", "腕部相机")):
            camera_card = card(title)
            layout = QtWidgets.QVBoxLayout(camera_card)
            preview = PreviewLabel(title)
            self.preview_labels[role] = preview
            layout.addWidget(preview)
            status = label("等待画面", name="previewStatus")
            self.preview_status[role] = status
            layout.addWidget(status)
            cameras.addWidget(camera_card)
        left.addLayout(cameras, 4)
        self.force_plot = self.charts.create(
            "夹持力与目标 / N",
            (
                ("force_left_N", "左侧", "#0891b2"),
                ("force_right_N", "右侧", "#e27728"),
                ("force_mean_N", "均值", "#7951bc"),
                ("force_target_requested_N", "目标", "#087f71"),
            ),
            minimum_height=140,
            legend_columns=2,
        )
        left.addWidget(self.force_plot, 2)
        self.phase_label = label("反馈状态：等待连接", name="subtitle")
        left.addWidget(self.phase_label)
        content.addLayout(left, 1)
        right_widget = QtWidgets.QWidget()
        right_widget.setMinimumWidth(270)
        right_widget.setMaximumWidth(335)
        right_widget.setStyleSheet("QPushButton { padding: 6px 10px; }")
        right = QtWidgets.QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        operation = card("1  连接与遥操")
        actions = QtWidgets.QVBoxLayout(operation)
        actions.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        self.connect_button = self._button("连接 / 只读监测", "connect")
        self.disconnect_button = self._button("断开", "disconnect", update=False)
        row.addWidget(self.connect_button, 1)
        row.addWidget(self.disconnect_button)
        actions.addLayout(row)
        self.confirmed = QtWidgets.QCheckBox("已确认机械支撑与校准；\n确认力通道，双手已移开")
        self.confirmed.setStyleSheet("font-size: 12px;")
        self.confirmed.setToolTip("连接并取得只读反馈后，为本次启动勾选确认。")
        self.confirmed.toggled.connect(self._confirmation_changed)
        actions.addWidget(self.confirmed)
        self.recording_enabled = QtWidgets.QCheckBox("为本次遥操准备相机与示范录制")
        self.recording_enabled.setToolTip("勾选后，在启动遥操时准备相机和示范录制。")
        self.recording_enabled.toggled.connect(self._mark_dirty)
        actions.addWidget(self.recording_enabled)
        self.start_button = self._button(
            "启动夹爪控制" if self.gripper_only else "启动全臂遥操",
            "start",
            primary=True,
            payload=lambda: {
                "mode": "force" if self.gripper_only else "hybrid",
                "confirmed": self.confirmed.isChecked(),
            },
        )
        actions.addWidget(self.start_button)
        self.next_step = label("先连接并取得新鲜反馈。", name="subtitle", wrap=True)
        actions.addWidget(self.next_step)
        right.addWidget(operation)
        record = card("2  示范录制")
        record_layout = QtWidgets.QVBoxLayout(record)
        record_layout.setSpacing(3)
        form = QtWidgets.QHBoxLayout()
        form.setSpacing(4)
        for key, caption, default in (
            ("task", "任务", ""),
            ("object_id", "物体", ""),
        ):
            edit = QtWidgets.QLineEdit(default)
            edit.setObjectName(f"record_{key}")
            edit.setAccessibleName(caption)
            edit.setToolTip(f"填写{caption}")
            edit.textChanged.connect(
                lambda text, field=edit, title=caption: field.setToolTip(
                    f"{title}：{text}" if text else f"填写{title}"
                )
            )
            edit.setMinimumWidth(0)
            edit.textEdited.connect(self._mark_dirty)
            self._recording_edits[key] = edit
            form.addWidget(label(caption))
            form.addWidget(edit, 1)
        record_layout.addLayout(form)
        outcome_row = QtWidgets.QHBoxLayout()
        outcome_row.addWidget(label("任务结果"))
        self.outcome = QtWidgets.QComboBox()
        self.outcome.setToolTip("每条录制的人工结果单独选择；新录制默认未确认。")
        for value, caption in OUTCOMES.items():
            self.outcome.addItem(caption, value)
        outcome_row.addWidget(self.outcome, 1)
        record_layout.addLayout(outcome_row)
        self.record_button = self._button("开始录制", "record_start", primary=True)
        self.save_button = self._button("结束并保存", "record_finish", payload=self._outcome_payload)
        record_layout.addWidget(self.record_button)
        record_layout.addWidget(self.save_button)
        extras = QtWidgets.QHBoxLayout()
        extras.addWidget(self._button("丢弃重录", "record_discard", payload={"rerecord": True}))
        extras.addWidget(
            self._button("确认结果", "record_outcome", payload=self._outcome_payload, update=False)
        )
        record_layout.addLayout(extras)
        self.record_state = label("尚未录制", wrap=True)
        self.record_count = label("已保存 0 条 · 人工成功 0 条", name="subtitle", wrap=True)
        record_layout.addWidget(self.record_state)
        record_layout.addWidget(self.record_count)
        right.addWidget(record)
        self._record_column = right
        right.addStretch(1)
        content.addWidget(right_widget)
        page_layout.addLayout(content, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.dataset_path = label("数据目录：正在加载", name="subtitle")
        self.dataset_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.dataset_path.setMinimumWidth(0)
        self.dataset_path.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        bottom.addWidget(self.dataset_path, 1)
        self.dataset_resume = label("续录：未选", name="subtitle")
        bottom.addWidget(self.dataset_resume)
        self._dataset_footer = bottom
        self.verification_panel = QtWidgets.QFrame()
        self.verification_panel_layout = QtWidgets.QVBoxLayout(self.verification_panel)
        self.verification_panel_layout.setSpacing(4)
        self.verification_heading = label("3  完整核验")
        self.verification_heading.setStyleSheet("font-weight: 600;")
        self.verification_inline = label("尚未执行", name="subtitle", wrap=True)
        self.verification_inline.setMaximumHeight(42)
        self.verification_inline.setMinimumWidth(0)
        self.verification_panel_layout.addWidget(self.verification_heading)
        self.verification_panel_layout.addWidget(self.verification_inline)
        verification_actions = QtWidgets.QHBoxLayout()
        verification_actions.setSpacing(4)
        self.verify_button = self._button("完整核验", "verify_dataset")
        self.cancel_verify_button = self._button("取消核验", "cancel_verification", update=False)
        self.verification_result_button = QtWidgets.QPushButton("查看结果")
        self.verification_result_button.clicked.connect(lambda: self._show_verification_result())
        verification_actions.addWidget(self.verify_button)
        verification_actions.addWidget(self.cancel_verify_button)
        verification_actions.addWidget(self.verification_result_button)
        self.verification_panel_layout.addLayout(verification_actions)
        bottom.addWidget(self.verification_panel)
        self.recording_config_link = QtWidgets.QPushButton("录制配置…")
        self.recording_config_link.clicked.connect(lambda: self._show_device_section("recording"))
        bottom.addWidget(self.recording_config_link)
        self.camera_config_link = QtWidgets.QPushButton("相机设置…")
        self.camera_config_link.clicked.connect(lambda: self._show_device_section("cameras"))
        bottom.addWidget(self.camera_config_link)
        page_layout.addLayout(bottom)
        self.pages.addWidget(page)
        self._arrange_verification_panel()

    def resizeEvent(self, event):  # noqa: N802 - Qt virtual method.
        super().resizeEvent(event)
        if hasattr(self, "verification_panel"):
            self._arrange_verification_panel()

    def _arrange_verification_panel(self):
        # At shorter supported sizes, keep all recording actions visible.
        mode = "card" if self.height() >= 850 else "column" if self.height() >= 740 else "footer"
        if mode == self._verification_layout_mode:
            return
        self._verification_layout_mode = mode
        self._record_column.removeWidget(self.verification_panel)
        self._dataset_footer.removeWidget(self.verification_panel)
        if mode == "footer":
            self._dataset_footer.insertWidget(2, self.verification_panel)
        else:
            self._record_column.insertWidget(self._record_column.count() - 1, self.verification_panel)
        expanded = mode == "card"
        self.verification_panel.setObjectName("verificationCard" if expanded else "verificationCompact")
        self.verification_panel_layout.setContentsMargins(*(12, 9, 12, 10) if expanded else (0, 0, 0, 0))
        self.verification_heading.setVisible(expanded)
        self.verification_inline.setVisible(expanded)
        self.verification_panel.style().unpolish(self.verification_panel)
        self.verification_panel.style().polish(self.verification_panel)
        self.verification_panel.show()

    def _build_tuning(self):
        page, _, layout = scroll_page()
        self.tuning_page = page
        layout.setContentsMargins(0, 0, 0, 18)
        container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(18, 10, 18, 8)
        outer.setSpacing(8)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label("调参", name="pageTitle"))
        heading.addStretch()
        heading.addWidget(label("修改后显式应用；当前生效目标显示在首页。", name="subtitle"))
        outer.addLayout(heading)
        outer.addWidget(page, 1)
        columns = QtWidgets.QHBoxLayout()
        parameters = QtWidgets.QVBoxLayout()
        controls = card("夹爪力控制")
        control_form = QtWidgets.QFormLayout(controls)
        for section, key, *_ in FIELDS:
            if section == "control" and key != "acceleration_limit_deg_s2":
                caption = {
                    "position_target_deg": "默认定位角度 (°，参数文件)",
                    "position_speed_deg_s": "默认定位速度 (°/s，参数文件)",
                }.get(key)
                self._field(key, control_form, caption=caption)
                if key == "ki_deg_per_N_s":
                    self._add_acceleration_field(control_form)
        parameters.addWidget(controls)
        limits = CollapsibleBox("运动限制与保护参数")
        limits_form = QtWidgets.QFormLayout(limits.content)
        for section, key, *_ in FIELDS:
            if section == "limits":
                caption = "停滞位置容差 (°) · 必填" if key == "stall_position_tolerance_deg" else None
                self._field(key, limits_form, caption=caption)
        for key, (caption, state) in OPTIONAL_GROUPS.items():
            if key == "acceleration":
                continue
            check = QtWidgets.QCheckBox(caption)
            check.setChecked(state)
            check.toggled.connect(self._mark_dirty)
            self._toggle_checks[key] = check
            limits_form.addRow(check)
        parameters.addWidget(limits)
        low_level = CollapsibleBox("FORCE_POS 与控制时序")
        low_form = QtWidgets.QFormLayout(low_level.content)
        for _, key, *_ in FIELDS:
            if key in GRIPPER_DEVICE_FIELDS:
                self._field(key, low_form)
        parameters.addWidget(low_level)
        self._build_arm_editor(parameters)
        parameters.addWidget(self._button("应用实时参数", "apply", primary=True))
        parameters.addWidget(
            self._button("仅应用目标力", "target", payload=lambda: {"value": self._number("target_N")})
        )
        start_row = QtWidgets.QHBoxLayout()
        start_row.addWidget(
            self._button(
                "启动单夹爪 I-P",
                "start",
                payload=lambda: {"mode": "force", "confirmed": self.confirmed.isChecked()},
            )
        )
        start_row.addWidget(
            self._button(
                "启动主从夹爪",
                "start",
                payload=lambda: {"mode": "gripper_hybrid", "confirmed": self.confirmed.isChecked()},
            )
        )
        parameters.addLayout(start_row)
        parameters.addStretch()
        columns.addLayout(parameters, 5)
        right = QtWidgets.QVBoxLayout()
        manual = card("手动操作")
        self.manual_operations = manual
        manual_layout = QtWidgets.QGridLayout(manual)
        for index, (text, action, payload) in enumerate(
            (
                ("恢复 I-P", "resume", None),
                (
                    "恒速闭合 → 力控",
                    "approach",
                    lambda: {
                        key: self._number(key)
                        for key in ("gripper_follow_speed_deg_s", "contact_force_threshold_N")
                    },
                ),
            )
        ):
            manual_layout.addWidget(self._button(text, action, payload=payload), 0, index)
        position_title = label("本次夹爪定位")
        position_title.setStyleSheet("font-weight: 600;")
        manual_layout.addWidget(position_title, 1, 0, 1, 2)
        position_fields = QtWidgets.QHBoxLayout()
        for key, caption, placeholder in (
            ("position_target_deg", "目标角度 / °", "校准坐标"),
            ("position_speed_deg_s", "速度 / °/s", "定位速度"),
        ):
            position_fields.addWidget(label(caption))
            edit = QtWidgets.QLineEdit()
            edit.setObjectName("command_" + key)
            edit.setAccessibleName(caption)
            edit.setPlaceholderText(placeholder)
            edit.setToolTip("用于本次夹爪定位；初值来自参数配置，修改不会自动保存。")
            edit.setMinimumWidth(60)
            edit.textEdited.connect(lambda text, k=key: self._position_command_dirty.add(k))
            self._position_command_edits[key] = edit
            position_fields.addWidget(edit, 1)
        manual_layout.addLayout(position_fields, 2, 0, 1, 2)
        self.position_button = QtWidgets.QPushButton("运动到指定位置")
        self.position_button.setObjectName("positionButton")
        self.position_button.setProperty("action", "position")
        self.position_button.clicked.connect(self._dispatch_position)
        self._buttons.append((self.position_button, "position"))
        self.confirmed.toggled.connect(self._position_prerequisites_changed)
        self.recording_enabled.toggled.connect(self._position_prerequisites_changed)
        manual_layout.addWidget(self.position_button, 3, 0, 1, 2)
        self.position_hint = label("请先启动控制会话", name="subtitle", wrap=True)
        self.position_hint.setMinimumWidth(0)
        manual_layout.addWidget(self.position_hint, 4, 0, 1, 2)
        manual_layout.addWidget(self._button("释放并结束采集", "release"), 5, 0, 1, 2)
        manual_layout.addWidget(
            label("主动释放会运动；普通结束并保存使用首页按钮。", name="subtitle", wrap=True), 6, 0, 1, 2
        )
        right.addWidget(manual)
        self.tuning_force_plot = self.charts.create(
            "夹持力与目标 / N",
            (
                ("force_left_N", "左侧", "#0891b2"),
                ("force_right_N", "右侧", "#e27728"),
                ("force_mean_N", "均值", "#7951bc"),
                ("force_target_requested_N", "目标", "#087f71"),
            ),
            minimum_height=190,
            legend_columns=2,
        )
        right.addWidget(self.tuning_force_plot)
        self.tuning_force_status = label("反馈状态：等待连接", name="subtitle", wrap=True)
        right.addWidget(self.tuning_force_status)
        self.tuning_position_plot = self.charts.create(
            "位置 / °",
            (
                ("q_sent_deg", "下发", "#087f71"),
                ("q_measured_deg", "反馈", "#2563eb"),
                ("hold_target_deg", "固定目标", "#d97706"),
            ),
            minimum_height=190,
            legend_columns=2,
        )
        self.tuning_delta_plot = self.charts.create(
            "控制增量 / °",
            (
                ("delta_i_deg", "I", "#e27728"),
                ("delta_p_deg", "P", "#2563eb"),
                ("delta_contact_deg", "回退", "#bf4d83"),
                ("delta_raw_deg", "原始", "#7951bc"),
                ("delta_final_deg", "最终", "#087f71"),
            ),
            minimum_height=190,
            legend_columns=3,
        )
        self._tuning_plots = (self.tuning_force_plot, self.tuning_position_plot, self.tuning_delta_plot)
        self._tuning_x_syncing = False
        for plot in self._tuning_plots:
            plot.setFixedHeight(210)
            plot.getAxis("left").setWidth(72)
            plot.getViewBox().sigXRangeChanged.connect(self._sync_tuning_x_range)
        right.addWidget(self.tuning_position_plot)
        right.addWidget(self.tuning_delta_plot)
        right.addStretch()
        columns.addLayout(right, 7)
        layout.addLayout(columns)
        self.pages.addWidget(container)

    def _sync_tuning_x_range(self, source, x_range):
        if self._tuning_x_syncing:
            return
        self._tuning_x_syncing = True
        try:
            for plot in self._tuning_plots:
                view = plot.getViewBox()
                if view is not source:
                    view.setXRange(*x_range, padding=0)
        finally:
            self._tuning_x_syncing = False

    def _build_arm_editor(self, layout):
        arm = CollapsibleBox("全臂参数与范围")
        self.arm_editor = arm
        arm.toggle.toggled.connect(self._arm_editor_toggled)
        content = QtWidgets.QVBoxLayout(arm.content)
        content.addWidget(
            label(
                "沿用当前全臂配置。空白的可选范围表示关闭；修改后须停止并重新预检。六轴表格可左右滚动。",
                name="subtitle",
                wrap=True,
            )
        )
        operators = QtWidgets.QFormLayout()
        captions = (
            "距启动位置允许范围 ±°",
            "主从对齐容差 / °",
            "失能预载容差 / °",
            "正常命令最大间隔 / s",
            "接管模式对齐容差 / °",
        )
        for key, caption in zip(ARM_OPERATOR_FIELDS, captions, strict=True):
            edit = QtWidgets.QLineEdit()
            edit.textEdited.connect(self._arm_changed)
            self._arm_edits[("operator", key)] = edit
            operators.addRow(caption, edit)
        content.addLayout(operators)
        table = QtWidgets.QTableWidget(len(JOINTS), len(ARM_PROFILE_FIELDS) + 1)
        table.setHorizontalHeaderLabels(
            ("关节", "Kp", "Kd", "下限 °", "上限 °", "相对限制 °", "速度 °/s", "力矩 N·m")
        )
        table.verticalHeader().hide()
        header = table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((76, 80, 80, 80, 80, 100, 88, 92)):
            header.resizeSection(column, width)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setMinimumHeight(245)
        for row, (joint, caption) in enumerate(zip(JOINTS, JOINT_LABELS, strict=True)):
            item = QtWidgets.QTableWidgetItem(caption)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, item)
            for column, key in enumerate(ARM_PROFILE_FIELDS, 1):
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText("沿用配置")
                edit.textEdited.connect(self._arm_changed)
                table.setCellWidget(row, column, edit)
                self._arm_edits[(joint, key)] = edit
        content.addWidget(table)
        content.addWidget(self._button("离线预检当前参数", "preflight"))
        layout.addWidget(arm)

    def _arm_editor_toggled(self, expanded):
        if expanded:
            QtCore.QTimer.singleShot(0, self._reveal_arm_editor)

    def _reveal_arm_editor(self):
        if (
            self._closing
            or self.controller.closed
            or not self.arm_editor.toggle.isChecked()
            or not self.tuning_page.isVisible()
        ):
            return
        offset = self.arm_editor.mapTo(self.tuning_page.widget(), QtCore.QPoint()).y()
        self.tuning_page.verticalScrollBar().setValue(max(0, offset - 10))

    def _build_devices(self):
        page, _, layout = scroll_page()
        self.device_page = page
        self.device_sections = {}
        self.device_shortcuts = {}
        container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(18, 10, 18, 0)
        outer.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 18)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label("设备与配置", name="pageTitle"))
        heading.addStretch()
        for title, section in (("设备连接", "devices"), ("录制配置", "recording"), ("相机绑定", "cameras")):
            shortcut = QtWidgets.QPushButton(title)
            shortcut.clicked.connect(lambda checked=False, target=section: self._show_device_section(target))
            self.device_shortcuts[section] = shortcut
            heading.addWidget(shortcut)
        outer.addLayout(heading)
        outer.addWidget(page, 1)
        files = card("参数与校准")
        files_layout = QtWidgets.QVBoxLayout(files)
        self.settings_label = label("参数文件：正在加载", wrap=True)
        self.calibration_label = label("校准文件：正在加载", wrap=True)
        description = self.home_communication_label.text()
        self.communication_label = label(description, name="subtitle", wrap=True)
        files_layout.addWidget(self.settings_label)
        files_layout.addWidget(self.calibration_label)
        files_layout.addWidget(self.communication_label)
        buttons = QtWidgets.QHBoxLayout()
        for text, action in (
            ("加载参数…", "load_settings"),
            ("保存参数…", "save_settings"),
            ("选择 / 重载校准…", "load_calibration"),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, a=action: self.choose_file(a))
            self._buttons.append((button, action))
            buttons.addWidget(button)
        buttons.addWidget(self._button("离线预检", "preflight"))
        files_layout.addLayout(buttons)
        layout.addWidget(files)
        common = card("设备连接")
        self.device_sections["devices"] = common
        common_form = QtWidgets.QFormLayout(common)
        common_keys = {
            "port",
            "baudrate",
            "leader_port",
            "leader_baudrate",
            "sensor_host",
            "sensor_port",
            "left_channel",
            "right_channel",
        }
        for _, key, *_ in FIELDS:
            if key in common_keys:
                self._field(key, common_form)
        layout.addWidget(common)
        recording = card("录制配置")
        self.device_sections["recording"] = recording
        rec_form = QtWidgets.QFormLayout(recording)
        for key, caption, value in (
            ("root", "数据目录", "outputs/teacher_demonstrations"),
            ("repo_id", "数据集 ID", "local/force_teacher_demo"),
            ("contact_condition_id", "接触条件", ""),
            ("target_episode_count", "目标条数", "10"),
            ("camera_max_age_ms", "相机最大年龄 / ms", "100"),
            ("queue_capacity", "录制队列容量", "128"),
            ("h264_encoder", "H264 编码器 / nvenc 或 software", "nvenc"),
        ):
            edit = QtWidgets.QLineEdit(value)
            edit.textEdited.connect(self._mark_dirty)
            self._recording_edits[key] = edit
            if key == "root":
                edit.textChanged.connect(self._update_dataset_summary)
                row = QtWidgets.QHBoxLayout()
                row.addWidget(edit, 1)
                browse = QtWidgets.QPushButton("选择…")
                browse.clicked.connect(self.choose_dataset)
                row.addWidget(browse)
                rec_form.addRow(caption, row)
            else:
                rec_form.addRow(caption, edit)
        self.resume_checkbox = QtWidgets.QCheckBox("续录已有数据集")
        self.resume_checkbox.toggled.connect(self._mark_dirty)
        self.resume_checkbox.toggled.connect(self._update_dataset_summary)
        rec_form.addRow(self.resume_checkbox)
        self.verify_video_checkbox = QtWidgets.QCheckBox("保存时完整核验视频")
        self.verify_video_checkbox.setToolTip(
            "关闭时仍检查编码帧数和数据完整性；完整视频核验可在停止后手动执行。"
        )
        self.verify_video_checkbox.toggled.connect(self._mark_dirty)
        rec_form.addRow(self.verify_video_checkbox)
        rec_form.addRow(label("正式采集 50 Hz · 力采集 400 Hz；显示刷新独立。", name="subtitle", wrap=True))
        layout.addWidget(recording)
        self.device_sections["cameras"] = self._build_cameras(layout)
        advanced = CollapsibleBox("高级设备、缓存与显示参数")
        advanced_form = QtWidgets.QFormLayout(advanced.content)
        for section, key, *_ in FIELDS:
            if section == "device" and key not in common_keys and key not in GRIPPER_DEVICE_FIELDS:
                self._field(key, advanced_form)
        layout.addWidget(advanced)
        layout.addStretch()
        self.pages.addWidget(container)

    def _build_cameras(self, layout):
        camera = card("前景 / 腕部相机")
        box = QtWidgets.QVBoxLayout(camera)
        box.addWidget(
            label("选择设备后预览核对，再保存角色绑定。连接按设备路径识别。", name="subtitle", wrap=True)
        )
        selectors = QtWidgets.QFormLayout()
        self.camera_choices = {}
        for role, caption in (("front", "前景"), ("wrist", "腕部")):
            choice = QtWidgets.QComboBox()
            choice.addItem("先刷新设备列表", None)
            choice.setMinimumContentsLength(30)
            choice.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self.camera_choices[role] = choice
            selectors.addRow(caption, choice)
        box.addLayout(selectors)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("刷新设备列表", "enumerate_cameras", update=False))
        row.addWidget(self._button("预览所选设备", "preview_cameras", payload=self._camera_selection))
        self.bind_button = self._button(
            "确认画面并保存绑定", "bind_cameras", payload=self._camera_selection, update=False
        )
        row.addWidget(self.bind_button)
        row.addWidget(self._button("关闭预览", "close_camera_preview", update=False))
        box.addLayout(row)
        previews = QtWidgets.QHBoxLayout()
        self.binding_labels = {
            role: PreviewLabel(caption) for role, caption in (("front", "前景相机"), ("wrist", "腕部相机"))
        }
        self.binding_preview_status = {}
        for role, preview in self.binding_labels.items():
            preview.setMaximumHeight(260)
            column = QtWidgets.QVBoxLayout()
            column.addWidget(preview)
            status = label("等待画面", name="previewStatus")
            self.binding_preview_status[role] = status
            column.addWidget(status)
            previews.addLayout(column)
        box.addLayout(previews)
        self.binding_status = label("绑定预览关闭后，才能准备正式录制。", name="subtitle", wrap=True)
        box.addWidget(self.binding_status)
        advanced = CollapsibleBox("高级相机 JSON")
        editor_layout = QtWidgets.QVBoxLayout(advanced.content)
        self.camera_json = QtWidgets.QPlainTextEdit()
        self.camera_json.setMaximumHeight(180)
        self.camera_json.setPlaceholderText("当前相机配置")
        editor_layout.addWidget(self.camera_json)
        apply_json = QtWidgets.QPushButton("将 JSON 填入当前配置")
        apply_json.clicked.connect(self.apply_camera_json)
        editor_layout.addWidget(apply_json)
        box.addWidget(advanced)
        layout.addWidget(camera)
        return camera

    def _build_diagnostics(self):
        page, _, layout = scroll_page()
        layout.addWidget(label("诊断与记录", name="pageTitle"))
        state = card("当前状态与下一步")
        state_layout = QtWidgets.QVBoxLayout(state)
        self.diagnostic_state = label("等待状态", wrap=True)
        self.diagnostic_error = label("暂无故障", wrap=True)
        self.diagnostic_error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.diagnostic_exit = label("退出状态：尚未启动", wrap=True)
        self.diagnostic_data = label("数据状态：尚未录制", wrap=True)
        self.verification_status = label("完整核验：尚未执行", wrap=True)
        for widget in (
            self.diagnostic_state,
            self.diagnostic_error,
            self.diagnostic_exit,
            self.diagnostic_data,
            self.verification_status,
        ):
            state_layout.addWidget(widget)
        actions = QtWidgets.QHBoxLayout()
        details_button = QtWidgets.QPushButton("查看问题详情")
        details_button.clicked.connect(self.show_problem_details)
        actions.addWidget(details_button)
        actions.addWidget(self._button("全臂只读复核", "recover"))
        actions.addStretch()
        state_layout.addLayout(actions)
        layout.addWidget(state)
        log = card("操作记录")
        log_layout = QtWidgets.QVBoxLayout(log)
        self.operation_log = QtWidgets.QPlainTextEdit()
        self.operation_log.setReadOnly(True)
        self.operation_log.document().setMaximumBlockCount(500)
        self.operation_log.setMinimumHeight(220)
        log_layout.addWidget(self.operation_log)
        self.log_path = label("运行日志：尚未创建", wrap=True)
        self.log_path.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        log_layout.addWidget(self.log_path)
        open_log = QtWidgets.QPushButton("打开日志目录")
        open_log.clicked.connect(self.open_log_directory)
        log_layout.addWidget(open_log)
        layout.addWidget(log)
        details = CollapsibleBox("原始状态与退出证据")
        details_layout = QtWidgets.QVBoxLayout(details.content)
        self.raw_state = QtWidgets.QPlainTextEdit()
        self.raw_state.setReadOnly(True)
        self.raw_state.setMinimumHeight(240)
        details_layout.addWidget(self.raw_state)
        self.raw_state_toggle = details.toggle
        layout.addWidget(details)
        layout.addStretch()
        self.pages.addWidget(page)

    def _setup_decoder(self):
        self.decoder_thread = QtCore.QThread(self)
        self.decoder = PreviewDecoder()
        self.decoder.moveToThread(self.decoder_thread)
        self.decode_requested.connect(self.decoder.decode, QtCore.Qt.ConnectionType.QueuedConnection)
        self.decoder.decoded.connect(self._decoded)
        self.decoder.failed.connect(self._preview_error)
        self.decoder.frame_failed.connect(self._preview_frame_failed)
        self.decoder.finished.connect(self._decode_finished)
        self.decoder_thread.finished.connect(self.decoder.deleteLater)
        self.decoder_thread.start()

    def _mark_dirty(self, *_):
        self._form_dirty = True

    def _show_device_section(self, section):
        self.navigation.setCurrentRow(2)
        self.device_page.ensureWidgetVisible(self.device_sections[section], xmargin=0, ymargin=12)

    def _update_dataset_summary(self, *_):
        root = self._recording_edits.get("root")
        if root is None or not hasattr(self, "resume_checkbox"):
            return
        snapshot_values = self._snapshot.recording_values if self._snapshot is not None else {}
        path = root.text()
        root_changed = "root" in snapshot_values and str(snapshot_values["root"]) != path
        if root_changed != self._last_root_draft:
            self._last_root_draft = root_changed
            self._last_revision = None
        if self._verification_dialog is not None and self._verification_dialog_root != path:
            self._verification_dialog.close()
        path_text = ("数据目录（草稿）：" if root_changed else "数据目录：") + (path or "未设置")
        if self.dataset_path.text() != path_text:
            self.dataset_path.setText(path_text)
        if self.dataset_path.toolTip() != (path or "尚未设置数据目录"):
            self.dataset_path.setToolTip(path or "尚未设置数据目录")
        if root_changed and hasattr(self, "verification_status"):
            self.verification_status.setText("完整核验：目录已更改，待重新核验")
            if (self._snapshot.recording.get("full_verification") or {}).get("state") == "passed":
                self.feedback.setText("数据目录已更改，需重新完整核验。")
                self.record_state.setText("数据：目录已更改，待重新核验")
        if hasattr(self, "verification_result_button"):
            verification = (self._snapshot.recording.get("full_verification") or {}) if self._snapshot else {}
            self._refresh_verification_controls(verification, root_changed)
        resume = self.resume_checkbox.isChecked()
        resume_changed = "resume" in snapshot_values and bool(snapshot_values["resume"]) != resume
        resume_text = f"续录：{'已选' if resume else '未选'}" + ("（草稿）" if resume_changed else "")
        if self.dataset_resume.text() != resume_text:
            self.dataset_resume.setText(resume_text)
        self.dataset_resume.setToolTip(
            "当前显示的是尚未提交的草稿，会随下一次表单请求提交。"
            if root_changed or resume_changed
            else "当前录制配置的续录选项。"
        )

    def _refresh_verification_controls(self, verification, root_draft):
        state = "idle" if root_draft else verification.get("state", "idle")
        if root_draft:
            summary, button = "目录已更改，请重新核验", "查看结果"
        elif state == "running":
            summary = f"正在核验：{verification.get('index', 0)} / {verification.get('total', 0)} 条"
            if verification.get("video"):
                summary += "\n视频：" + Path(str(verification["video"])).name
            button = "查看结果"
        elif state == "passed":
            summary, button = "本次全部通过，可查看逐条结果", "结果：通过"
        elif state == "failed":
            summary, button = "本次核验失败，请查看具体原因", "结果：失败"
        elif state == "cancelled":
            summary, button = "本次核验已取消，未完成", "结果：未完成"
        else:
            summary, button = "尚未执行；也可查看已有的历史报告", "查看结果"
        self.verification_inline.setText(summary)
        self.verification_inline.setToolTip(summary)
        self.verification_result_button.setText(button)
        self.verification_result_button.setEnabled(state != "running" and not self._closing)
        self.verification_result_button.setToolTip(
            "正在核验，完成后自动显示结果。"
            if state == "running"
            else "查看本次结果；没有本次结果时可查看最近的历史报告。"
        )

    @staticmethod
    def _verification_token(verification):
        return tuple(
            verification.get(key) for key in ("state", "root", "started_utc", "completed_utc", "report_path")
        )

    @staticmethod
    def _verification_details(payload, *, historical=False, read_error=""):
        def items(value):
            if isinstance(value, (tuple, list)):
                return value
            return (value,) if isinstance(value, str) else ()

        state = payload.get("state", "idle")
        titles = {"passed": "通过", "failed": "失败", "cancelled": "未完成", "idle": "尚无结果"}
        lines = ["完整核验结果：" + titles.get(state, "未知")]
        if historical:
            lines.append("这是历史报告；数据可能已经变化，不能作为当前数据集全部通过的结论。")
        if payload.get("root"):
            lines.append("数据目录：" + str(payload["root"]))
        if payload.get("completed_utc"):
            lines.append("完成时间（UTC）：" + str(payload["completed_utc"]))
        elapsed = payload.get("elapsed_seconds")
        if type(elapsed) in (int, float) and math.isfinite(elapsed):
            lines.append(f"耗时：{elapsed:.2f} 秒")
        report = payload.get("result")
        if isinstance(report, Mapping):
            episodes = [episode for episode in items(report.get("episodes")) if isinstance(episode, Mapping)]
            excluded = items(report.get("excluded_episodes"))
            frames = [episode.get("frame_count") for episode in episodes]
            lines.append(
                f"已检查：{len(episodes)} 条；通过 {sum(e.get('valid') is True for e in episodes)} 条"
            )
            if frames and all(type(count) is int and count >= 0 for count in frames):
                lines.append(f"总帧数：{sum(frames):,} 帧")
            lines.append(f"视频完整解码：{sum(e.get('video_decoded') is True for e in episodes)} 条")
            if excluded:
                lines.append(f"未保存记录已排除：{len(excluded)} 条")
            if episodes:
                lines.append("\n逐条结果：")
                ordered = sorted(
                    episodes,
                    key=lambda episode: (
                        type(episode.get("episode_index")) is not int,
                        episode.get("episode_index") if type(episode.get("episode_index")) is int else 0,
                    ),
                )
                for episode in ordered:
                    index = episode.get("episode_index")
                    name = (
                        f"第 {index + 1} 条"
                        if type(index) is int and index >= 0
                        else "集 " + str(episode.get("episode_uid", "未知"))
                    )
                    result = "通过" if episode.get("valid") is True else "失败"
                    count = episode.get("frame_count")
                    frames_text = f" · {count:,} 帧" if type(count) is int and count >= 0 else ""
                    video = " · 视频已完整解码" if episode.get("video_decoded") is True else ""
                    lines.append(f"{name}：{result}{frames_text}{video}")
                    lines.extend("  原因：" + str(error) for error in items(episode.get("errors")))
            errors = items(report.get("errors"))
        else:
            lines.append("没有可显示的逐条明细。")
            errors = ()
        errors = list(dict.fromkeys(str(error) for error in (*items(payload.get("errors")), *errors)))
        if errors:
            lines.append("\n原因：")
            lines.extend("• " + error for error in errors)
        if read_error:
            lines.append("\n无法读取报告：" + read_error)
        if payload.get("report_path"):
            lines.append("\n完整报告：" + str(payload["report_path"]))
        if state == "idle":
            lines.append("停止采集并确认资源退出后，点击“完整核验”。")
        return "\n".join(lines)

    def _show_verification_result(self, verification=None):
        if self._closing:
            return
        root_text = self._recording_edits["root"].text().strip()
        snapshot = self._snapshot
        if verification is None:
            verification = (snapshot.recording.get("full_verification") or {}) if snapshot else {}
        root_draft = snapshot is not None and root_text != str(snapshot.recording_values.get("root", ""))
        current = not root_draft and verification.get("state") in {"passed", "failed", "cancelled"}
        payload = plain(verification) if current else {"state": "idle", "root": root_text}
        historical = False
        report_path = payload.get("report_path") if current else None
        read_error = ""
        try:
            if not current and root_text:
                root = Path(root_text).expanduser().resolve()
                directory = root.parent / f"{root.name}_verification_reports"
                candidates = (path for path in directory.glob("verification_*.json") if path.is_file())
                report_path = max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)
                historical = report_path is not None
            if report_path:
                report_path = Path(report_path)
                stored = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(stored, dict) or stored.get("scope") != "all_saved_episodes":
                    raise ValueError("报告格式不正确")
                if (
                    Path(str(stored.get("root", ""))).expanduser().resolve()
                    != Path(root_text).expanduser().resolve()
                ):
                    raise ValueError("报告的数据目录与当前选择不一致")
                if historical:
                    if stored.get("state") == "passed" and (
                        not isinstance(stored.get("result"), dict)
                        or stored["result"].get("valid") is not True
                    ):
                        raise ValueError("报告的通过状态和核验结果不一致")
                    payload = stored
                else:
                    payload["result"] = stored.get("result")
                payload["report_path"] = str(report_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            read_error = str(exc)
            if historical:
                payload = {"state": "idle", "root": root_text, "report_path": str(report_path)}
        if self._verification_dialog is not None:
            self._verification_dialog.close()
            self._verification_dialog.deleteLater()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("完整核验结果" + ("（历史报告）" if historical else ""))
        dialog.resize(720, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        details = QtWidgets.QPlainTextEdit()
        details.setObjectName("verificationResultDetails")
        details.setReadOnly(True)
        details.setPlainText(
            self._verification_details(payload, historical=historical, read_error=read_error)
        )
        layout.addWidget(details)
        actions = QtWidgets.QHBoxLayout()
        open_report = QtWidgets.QPushButton("打开完整报告")
        open_report.setObjectName("openVerificationReport")
        open_report.setEnabled(bool(report_path) and not read_error)
        if report_path:
            open_report.clicked.connect(
                lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(report_path.resolve())))
            )
        actions.addWidget(open_report)
        actions.addStretch()
        close = QtWidgets.QPushButton("关闭")
        close.clicked.connect(dialog.close)
        actions.addWidget(close)
        layout.addLayout(actions)
        self._verification_dialog = dialog
        self._verification_dialog_token = self._verification_token(verification) if current else None
        self._verification_dialog_historical = historical
        self._verification_dialog_root = root_text
        dialog.show()

    def _show_verification_start_error(self, message):
        if self._closing:
            return
        if self._verification_start_error_dialog is not None:
            self._verification_start_error_dialog.close()
            self._verification_start_error_dialog.deleteLater()
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("完整核验未启动")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText("完整核验未启动：\n" + str(message))
        dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        self._verification_start_error_dialog = dialog
        dialog.show()

    def _arm_changed(self, *_):
        self._arm_dirty = self._form_dirty = True

    def _number(self, key):
        value = float(self._field_edits[key][0].text())
        if not math.isfinite(value):
            raise ValueError(f"{self._field_specs[key][2]} 必须为有限数值")
        return value

    def _position_payload(self):
        values = {}
        for key, name in (
            ("position_target_deg", "position_deg"),
            ("position_speed_deg_s", "position_speed_deg_s"),
        ):
            try:
                value = float(self._position_command_edits[key].text())
            except ValueError as exc:
                raise ValueError(f"{self._field_specs[key][2]} 必须为有限数值") from exc
            if not math.isfinite(value):
                raise ValueError(f"{self._field_specs[key][2]} 必须为有限数值")
            values[name] = value
        return values

    def _dispatch_position(self):
        snapshot = self._snapshot
        if snapshot is None or not self.position_button.isEnabled():
            return
        if str(snapshot.state).lower() == "monitoring":
            request_id = self.dispatch(
                "start",
                lambda: {
                    **self._position_payload(),
                    "mode": "position",
                    "confirmed": self.confirmed.isChecked(),
                },
            )
            if request_id is not None:
                self._position_start_pending = request_id
                self._refresh_position_controls(snapshot)
        else:
            self.dispatch("position", self._position_payload)

    def _position_prerequisites_changed(self, *_):
        if self._snapshot is not None and hasattr(self, "position_button"):
            self._refresh_position_controls(self._snapshot)

    def _refresh_position_controls(self, snapshot):
        caps = snapshot.capabilities
        state = str(snapshot.state).lower()
        direct_start = state == "monitoring"
        if direct_start:
            reason = caps.get("start", "等待当前操作完成")
            if not reason:
                missing = []
                if not self.confirmed.isChecked():
                    missing.append("勾选机械与力通道确认")
                if self.recording_enabled.isChecked():
                    missing.append("关闭录制准备")
                if missing:
                    reason = "请先在首页" + "，并".join(missing)
            caption = "启动并运动到指定位置"
            hint = "启动夹爪（不录制）"
            help_text = ACTION_HELP["position_start"]
        else:
            reason = caps.get("position", "等待当前操作完成")
            if state == "disconnected":
                reason = "请先连接只读监测"
            elif state == "recovery_required":
                reason = "请先完成只读复核"
            caption = "运动到指定位置"
            hint = "主从时退出跟随"
            help_text = ACTION_HELP["position"]
        if self._position_start_pending is not None:
            reason = "定位启动请求处理中"
        self.position_button.setText(caption)
        self.position_button.setEnabled(not bool(reason) and not self._closing)
        self.position_button.setToolTip(str(reason) if reason else help_text)
        self.position_hint.setText(str(reason) if reason else hint)
        self.position_hint.setToolTip(str(reason) if reason else help_text)
        position_input_enabled = any(
            not caps.get(action, "等待当前操作完成") for action in ("edit_values", "edit_live", "position")
        )
        for edit in self._position_command_edits.values():
            edit.setEnabled(position_input_enabled and not self._closing)

    def _outcome_payload(self):
        return {"task_result": self.outcome.currentData() or "unknown"}

    def _form_payload(self):
        result = {
            "values": {key: edits[0].text() for key, edits in self._field_edits.items()},
            "toggles": {key: check.isChecked() for key, check in self._toggle_checks.items()},
            "recording_values": {key: edit.text() for key, edit in self._recording_edits.items()},
            "camera_configs": self._camera_configs,
        }
        result["recording_values"]["resume"] = self.resume_checkbox.isChecked()
        result["recording_values"]["verify_video_on_save"] = self.verify_video_checkbox.isChecked()
        result["recording_values"]["enabled"] = self.recording_enabled.isChecked()
        if self._arm_dirty:
            result["arm_settings"] = {
                "operator": {key: self._arm_edits[("operator", key)].text() for key in ARM_OPERATOR_FIELDS},
                "joints": {
                    joint: {key: self._arm_edits[(joint, key)].text() for key in ARM_PROFILE_FIELDS}
                    for joint in JOINTS
                },
            }
        return result

    def dispatch(self, action, payload=None, *, update=True):
        try:
            if action == "verify_dataset" and self._verification_start_error_dialog is not None:
                self._verification_start_error_dialog.close()
            if callable(payload):
                payload = payload()
            if action == "apply":
                payload = {
                    "values": self._form_payload()["values"],
                    "toggles": self._form_payload()["toggles"],
                }
            elif action == "preflight":
                payload = {**(payload or {}), "confirmed": self.confirmed.isChecked()}
            submitted_form = self._form_payload() if update and self._form_dirty else None
            if submitted_form is not None:
                payload = {**(payload or {}), "form": submitted_form}
            preview_started_ns = time.perf_counter_ns() if action == "preview_cameras" else None
            if action in ("load_settings", "load_calibration"):
                self._hydrated = False
                self._form_dirty = self._arm_dirty = False
            request_id = self.controller.request(action, payload)
            if preview_started_ns is not None:
                self._binding_seen.clear()
                self._binding_started_ns = preview_started_ns
                self._binding_active = True
                self._binding_preview_timestamps.clear()
                self._preview_errors.clear()
                for preview in self.binding_labels.values():
                    preview.clear_image()
            if submitted_form is not None:
                self._pending_forms[request_id] = submitted_form
            text = "请求已提交，正在处理…"
            if action == "record_finish":
                text = "已请求结束并保存，正在等待保存与核验结果。"
            elif action == "start":
                text = "已请求启动，正在检查准备条件。"
            self.feedback.setText(text)
            self.operation_log.appendPlainText(
                f"{time.strftime('%H:%M:%S')}  {self._action_title(action)} · 已提交"
            )
            self._page_changed(self.pages.currentIndex())
            return request_id
        except (ValueError, TypeError, RuntimeError, OSError, queue.Full) as exc:
            message = str(exc) or "上一个请求尚未处理，请稍候。"
            self._remember_local_problem(self._action_title(action), f"{type(exc).__name__}: {message}")
            self.feedback.setText(message)
            return None

    def _remember_snapshot_problems(self, snapshot, problems):
        """Retain the most recently reported problem after its live status clears."""
        result_errors = tuple(str(error) for error in (getattr(snapshot, "result_errors", ()) or ()))
        if not problems and not result_errors:
            if self._current_problem_key is not None:
                self._current_problem_key = None
                self._set_tone(self.problem_button, "problemTone", "normal")
                self._update_problem_dialog()
            return
        key = (snapshot.session_id, tuple(problems), result_errors)
        if key != self._current_problem_key:
            self._current_problem_key = key
            self._last_problem = {
                "key": key,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "state": str(snapshot.state),
                "exit_confirmed": snapshot.exit_confirmed,
                "problems": tuple(problems),
                "result_errors": result_errors,
                "log_path": str(snapshot.log_path or "") if result_errors else "",
            }
            self._problem_history.append(self._last_problem)
            self._set_tone(self.problem_button, "problemTone", "fault")
            self._update_problem_dialog()

    def _remember_local_problem(self, source, message):
        """Keep a UI-side failure even when no controller snapshot contains it."""
        snapshot = self._snapshot
        signature = (getattr(snapshot, "session_id", None), source, str(message))
        if self._last_problem is not None and self._last_problem.get("local_signature") == signature:
            return
        self._last_problem = {
            "key": ("local", time.perf_counter_ns()),
            "local_signature": signature,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": str(snapshot.state) if snapshot is not None else "unknown",
            "exit_confirmed": snapshot.exit_confirmed if snapshot is not None else None,
            "problems": (f"{source}：{message}",),
            "result_errors": (),
            "log_path": "",
        }
        self._problem_history.append(self._last_problem)
        self._set_tone(self.problem_button, "problemTone", "fault")
        self._update_problem_dialog()

    def _problem_text(self):
        if not self._problem_history:
            return "当前窗口尚未收到错误。出现问题后，请在这里查看并复制详细信息。"
        return "\n\n────────────\n\n".join(
            self._problem_record_text(issue) for issue in reversed(self._problem_history)
        )

    def _problem_record_text(self, issue):
        current = issue["key"] == self._current_problem_key
        state = STATE_LABELS.get(issue["state"].lower(), issue["state"])
        exit_status = {True: "是", False: "否", None: "未提供"}.get(issue["exit_confirmed"], "未提供")
        raw = (*issue["problems"], *issue["result_errors"])
        title = (
            "当前上报的问题"
            if current
            else "最近一次问题（当前上报可能已改变）"
            if issue is self._last_problem
            else "之前记录的问题"
        )
        lines = [
            title,
            f"界面收到时间：{issue['time']}",
            f"当时界面状态：{state}",
            f"当时退出确认：{exit_status}",
            "",
            "问题说明：",
            *(f"• {explanation}" for explanation in explain_problems(raw)),
            "",
            "完整原始报错：",
            *(f"• {problem}" for problem in issue["problems"]),
        ]
        if issue["result_errors"]:
            lines.extend(("", f"运行结果中的原始错误（{len(issue['result_errors'])} 条）："))
            lines.extend(f"{index}. {error}" for index, error in enumerate(issue["result_errors"], 1))
        if issue["log_path"]:
            lines.extend(("", f"本次运行结果日志：{issue['log_path']}"))
        else:
            lines.extend(("", "本条问题没有关联的运行结果日志；其他日志可在“诊断与记录”查看。"))
        return "\n".join(lines)

    def _update_problem_dialog(self):
        if self._problem_dialog is None:
            return
        text = self._problem_text()
        if self.problem_details_text.toPlainText() != text:
            self.problem_details_text.setPlainText(text)
        self.problem_log_button.setEnabled(bool(self._last_problem and self._last_problem["log_path"]))

    def show_problem_details(self):
        """Open a non-blocking, selectable view without issuing hardware commands."""
        if self._problem_dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("问题详情")
            dialog.resize(760, 520)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(label("当前和最近 10 条问题及完整原始报错；可复制后用于排查。", wrap=True))
            self.problem_details_text = QtWidgets.QPlainTextEdit()
            self.problem_details_text.setObjectName("problemDetailsText")
            self.problem_details_text.setReadOnly(True)
            layout.addWidget(self.problem_details_text)
            actions = QtWidgets.QHBoxLayout()
            self.problem_log_button = QtWidgets.QPushButton("打开本次日志目录")
            self.problem_log_button.clicked.connect(self._open_problem_log)
            actions.addWidget(self.problem_log_button)
            copy_button = QtWidgets.QPushButton("复制全部")
            copy_button.setObjectName("copyProblemDetails")
            copy_button.clicked.connect(
                lambda: QtWidgets.QApplication.clipboard().setText(self.problem_details_text.toPlainText())
            )
            actions.addWidget(copy_button)
            actions.addStretch()
            close = QtWidgets.QPushButton("关闭")
            close.clicked.connect(dialog.close)
            actions.addWidget(close)
            layout.addLayout(actions)
            self._problem_dialog = dialog
        self._update_problem_dialog()
        self._problem_dialog.show()
        self._problem_dialog.raise_()
        if self._current_problem_key is None:
            self._set_tone(self.problem_button, "problemTone", "normal")

    def _open_problem_log(self):
        if self._last_problem and self._last_problem["log_path"]:
            directory = Path(self._last_problem["log_path"]).expanduser().parent
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))

    @staticmethod
    def _action_title(action):
        return {
            "connect": "连接",
            "disconnect": "断开",
            "start": "启动遥操",
            "recover": "只读复核",
            "apply": "应用参数",
            "record_start": "开始录制",
            "record_finish": "结束并保存",
            "record_discard": "丢弃重录",
            "record_outcome": "确认任务结果",
            "verify_dataset": "完整核验",
            "cancel_verification": "取消核验",
            "preview_cameras": "相机预览",
            "bind_cameras": "保存相机绑定",
            "close_camera_preview": "关闭相机预览",
            "load_settings": "加载参数",
            "save_settings": "保存参数",
            "load_calibration": "加载校准",
        }.get(action, action)

    def stop(self):
        try:
            self.controller.stop()
            self.confirmed.setChecked(False)
            self.feedback.setText("停止信号已发送；正在等待退出与资源关闭结果。")
        except (RuntimeError, OSError) as exc:
            self._remember_local_problem("停止", f"{type(exc).__name__}: {exc}")
            self.feedback.setText(f"停止信号发送失败：{exc}；退出仍待确认。")

    def _page_changed(self, index):
        enabled = (index == 0 or (index == 2 and self._binding_active)) and not self._closing
        if enabled != self._preview_interest:
            self.controller.set_preview_enabled(enabled)
            self._preview_interest = enabled

    def poll(self):
        try:
            snapshot = self.controller.snapshot()
            self._snapshot = snapshot
            if self._last_revision != snapshot.revision:
                self._last_revision = snapshot.revision
                self._render_snapshot(snapshot)
            self._poll_preview()
            self._refresh_preview_status()
            self._refresh_binding_button(snapshot)
            self._refresh_feedback_status(snapshot)
            if self._closing and self.controller.closed:
                self.close()
        except Exception as exc:
            self._remember_local_problem("状态显示", f"{type(exc).__name__}: {exc}")
            self.feedback.setText(f"状态显示失败：{exc}；停止仍可操作。")

    def _render_snapshot(self, snapshot):
        state = str(snapshot.state).lower()
        if self._preview_session_id is None:
            self._preview_session_id = snapshot.session_id
        elif self._preview_session_id != snapshot.session_id:
            self._preview_session_id = snapshot.session_id
            self._preview_epoch_ns = time.perf_counter_ns()
            self._preview_timestamps.clear()
            self._home_preview_timestamps.clear()
            self._preview_errors.clear()
            for preview in self.preview_labels.values():
                preview.clear_image()
        self.state_badge.setText(STATE_LABELS.get(state, snapshot.state))
        tone = (
            "fault"
            if state in {"fault", "error", "recovery_required"}
            else "active"
            if state in {"running", "monitoring", "ready"}
            else "busy"
            if state
            in {
                "loading",
                "initializing",
                "connecting",
                "preflighting",
                "preparing",
                "starting",
                "recovering",
                "stopping",
                "closing",
                "releasing",
            }
            else "neutral"
        )
        self._set_tone(self.state_badge, "stateTone", tone)
        message = snapshot.message or "等待下一步操作"
        self.feedback.setText(message)
        recording = snapshot.recording or {}
        record_problems = []
        for prefix, value in (
            ("采集", recording.get("error")),
            ("录制请求", recording.get("request_error")),
            ("资源清理", recording.get("cleanup_error")),
        ):
            if value:
                record_problems.append(f"{prefix}：{value}")
        if recording.get("cleanup_pending") and not recording.get("cleanup_error"):
            record_problems.append("资源清理：尚未完成")
        problems = ([f"会话：{snapshot.error}"] if snapshot.error else []) + record_problems
        row = snapshot.telemetry or {}
        issue_problems = list(problems)
        if row.get("phase") == "FAULT" and row.get("failure_reason") and row["failure_reason"] != "user_stop":
            issue_problems.append(f"控制故障：{row['failure_reason']}")
        if state in {"error", "fault", "recovery_required"} and not snapshot.error:
            issue_problems.append("会话状态异常或需要退出复核；未提供具体会话报错。")
        issue_record_state = str(recording.get("state", recording.get("status", "idle"))).lower()
        if issue_record_state in {"error", "corrupt", "commit_unknown"} and not record_problems:
            issue_problems.append(
                f"采集状态：{STATE_LABELS.get(issue_record_state, issue_record_state)}；未提供具体报错。"
            )
        verification = recording.get("full_verification") or {}
        verification_errors = (
            (verification.get("errors") or ()) if verification.get("state") == "failed" else ()
        )
        if isinstance(verification_errors, str):
            verification_errors = (verification_errors,)
        issue_problems.extend(f"完整核验：{error}" for error in verification_errors)
        self._remember_snapshot_problems(snapshot, issue_problems)
        notice = "；".join(problems) if problems else message
        self.readiness.setText(notice if len(notice) <= 55 else notice[:48] + "…（点“问题详情”）")
        self.readiness.setToolTip(notice)
        self._set_tone(self.readiness, "statusTone", "fault" if problems else "normal")
        caps = snapshot.capabilities
        if self._position_start_pending is not None and any(
            result.get("request_id") == self._position_start_pending
            for result in snapshot.operation_results or ()
        ):
            self._position_start_pending = None
        for button, action in self._buttons:
            if button is self.position_button:
                continue
            reason = caps.get(action, "等待当前操作完成")
            if action == "bind_cameras":
                reason = reason or self._binding_confirmation_reason()
            elif action == "start" and not self.confirmed.isChecked():
                reason = reason or "请先勾选本次机械与力通道确认"
            enabled = not bool(reason) and not self._closing
            if button.isEnabled() != enabled:
                button.setEnabled(enabled)
            button.setToolTip(str(reason) if reason else ACTION_HELP.get(action, ""))
        self.confirmed.setEnabled(state in {"monitoring", "ready"} and not self._closing)
        if state in {"recovering", "recovery_required", "disconnected"}:
            self.confirmed.setChecked(False)
        self._refresh_next_step(snapshot)
        metadata_locked = bool(caps.get("edit_recording", "")) or self._closing
        fields_locked = bool(caps.get("edit_values", "")) or self._closing
        live = state == "running"
        for key, edits in self._field_edits.items():
            enabled = (
                not fields_locked
                or (key in LIVE_FIELDS and not caps.get("edit_live", "当前不能编辑实时参数"))
                or (live and key == "target_N" and not caps.get("target", "当前不能应用目标力"))
            )
            for edit in edits:
                edit.setEnabled(enabled and not self._closing)
        for edit in self._recording_edits.values():
            edit.setEnabled(not metadata_locked)
        self.resume_checkbox.setEnabled(not metadata_locked)
        self.verify_video_checkbox.setEnabled(not metadata_locked)
        self.recording_enabled.setEnabled(not metadata_locked)
        for check in self._toggle_checks.values():
            check.setEnabled(not fields_locked)
        for edit in self._arm_edits.values():
            edit.setEnabled(not fields_locked)
        for choice in self.camera_choices.values():
            choice.setEnabled(not metadata_locked and not self._binding_active)
        self.camera_json.setReadOnly(metadata_locked)
        settings_changed = self._settings_file != snapshot.settings_path
        if (not self._hydrated or settings_changed) and state not in {"loading", "initializing"}:
            self._hydrate(snapshot)
        elif not self._form_dirty and plain(snapshot.camera_configs) != self._camera_configs:
            self._camera_configs = plain(snapshot.camera_configs)
            self.camera_json.setPlainText(json.dumps(self._camera_configs, ensure_ascii=False, indent=2))
        self._refresh_position_controls(snapshot)
        self._refresh_acceleration_field(snapshot)
        self._update_devices(snapshot)
        offline = state in {"disconnected", "recovery_required", "closed"}
        for key, metric in self.metrics.items():
            force_invalid = key != "force_target_requested_N" and row.get("force_valid") is False
            metric.set_value(None if offline or force_invalid else row.get(key))
        if not offline:
            self.charts.append(row)
        elif not self._binding_active:
            self._home_preview_timestamps.clear()
            self._preview_errors.clear()
            for preview in self.preview_labels.values():
                if preview._source is not None:
                    preview.clear_image()
        record_state = str(recording.get("state", recording.get("status", "idle"))).lower()
        episode_uid = recording.get("episode_uid")
        if recording.get("active") and isinstance(episode_uid, str) and episode_uid:
            identity = (snapshot.session_id, episode_uid)
            if identity != self._last_episode_identity:
                self._last_episode_identity = identity
                self.outcome.setCurrentIndex(self.outcome.findData("unknown"))
        root_draft = self._recording_edits["root"].text() != str(snapshot.recording_values.get("root", ""))
        full_check_passed = (
            not root_draft and (recording.get("full_verification") or {}).get("state") == "passed"
        )
        saved_text = (
            "演示流程已结束（未写正式数据）"
            if recording.get("simulated") and record_state == "saved"
            else "已保存，完整性待确认"
            if record_state == "saved" and recording.get("file_complete") is not True
            else "已保存，视频已完整核验"
            if record_state == "saved"
            and (recording.get("video_verification") == "passed" or full_check_passed)
            else "已保存，视频待完整核验"
            if record_state == "saved" and recording.get("video_verification") == "not_run"
            else "已保存，视频核验未记录"
            if record_state == "saved"
            else STATE_LABELS.get(record_state, record_state)
        )
        if recording.get("cleanup_pending"):
            saved_text += "（清理未完成，资源仍被占用）"
        self.record_state.setText("数据：" + saved_text)
        badge_text = (
            "录制：异常"
            if record_problems or record_state in {"error", "corrupt", "interrupted", "commit_unknown"}
            else "录制：待核验"
            if record_state == "saved_pending_verification"
            else "录制：保存中"
            if record_state in {"saving", "verifying"}
            else "录制：进行中"
            if recording.get("active")
            else "录制：演示完成"
            if recording.get("simulated") and record_state == "saved"
            else "录制：已保存"
            if record_state == "saved" and recording.get("file_complete") is True
            else "录制：待确认"
            if record_state == "saved"
            else "录制：未开始"
        )
        badge_tone = (
            "fault"
            if record_problems or record_state in {"error", "corrupt", "interrupted", "commit_unknown"}
            else "busy"
            if record_state in {"saving", "verifying", "saved_pending_verification"}
            else "active"
            if recording.get("active")
            else "neutral"
        )
        self.record_badge.setText(badge_text)
        self._set_tone(self.record_badge, "stateTone", badge_tone)
        frame_count = recording.get("frame_count", 0)
        frame_count = frame_count if type(frame_count) is int and frame_count >= 0 else 0
        self.record_count.setText(
            f"{'演示完成' if recording.get('simulated') else '数据集条数'} {recording.get('completed_episode_count', recording.get('episode_count', 0))} / {snapshot.recording_values.get('target_episode_count', '10')} 条 · 人工成功 {recording.get('successful_episode_count', 0)} 条\n本条 {frame_count} 帧 · 约 {frame_count / 50:.1f} 秒（按帧数）"
        )
        self._update_dataset_summary()
        verification = recording.get("full_verification") or {}
        root_draft = self._recording_edits["root"].text() != str(snapshot.recording_values.get("root", ""))
        verification_state = verification.get("state", "idle")
        token = self._verification_token(verification)
        if self._verification_dialog is not None and (
            verification_state == "running"
            or (not self._verification_dialog_historical and token != self._verification_dialog_token)
        ):
            self._verification_dialog.close()
        self._refresh_verification_controls(verification, root_draft)
        if not root_draft:
            if verification_state == "running":
                text = f"完整核验：{verification.get('index', 0)} / {verification.get('total', 0)}"
                if verification.get("video"):
                    text += " · " + str(verification["video"])
            elif verification_state == "passed":
                text = "完整核验：本次全部通过"
            elif verification_state == "failed":
                text = "完整核验失败：" + "；".join(str(item) for item in verification.get("errors", ()))
            elif verification_state == "cancelled":
                text = "完整核验：已取消，未完成"
            else:
                text = "完整核验：尚未执行"
            if verification.get("report_path"):
                text += "\n报告：" + str(verification["report_path"])
            self.verification_status.setText(text)
            if (
                verification_state in {"passed", "failed", "cancelled"}
                and token != self._last_verification_popup
                and not self._closing
            ):
                self._last_verification_popup = token
                self._show_verification_result(verification)
        self.settings_label.setText("参数文件：" + str(snapshot.settings_path or "未保存的参数草稿"))
        self.calibration_label.setText("校准文件：" + str(snapshot.calibration_path or "未加载"))
        self.diagnostic_state.setText("会话：" + STATE_LABELS.get(state, state) + "\n" + message)
        self.diagnostic_error.setText("发生原因：" + ("\n".join(problems) if problems else "暂无故障"))
        self.diagnostic_exit.setText(
            "设备退出：" + ("已确认" if snapshot.exit_confirmed else "尚未确认，须完成只读复核")
        )
        data_text = (
            "数据：" + saved_text + "\n人工任务结果：" + OUTCOMES.get(recording.get("task_result"), "未确认")
        )
        timings = recording.get("save_timings") or {}
        labels = (
            ("drain_seconds", "排空"),
            ("transaction_seconds", "提交"),
            ("non_video_seconds", "非视频检查"),
            ("video_decode_seconds", "视频解码"),
            ("total_seconds", "总计"),
        )
        if all(type(timings.get(key)) in (int, float) and math.isfinite(timings[key]) for key, _ in labels):
            timing_text = "保存耗时：" + " · ".join(f"{label} {timings[key]:.2f}s" for key, label in labels)
            data_text += "\n" + timing_text
            identity = (snapshot.session_id, episode_uid)
            if episode_uid and identity != self._last_save_timing_identity:
                self.operation_log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {timing_text}")
                self._last_save_timing_identity = identity
        self.diagnostic_data.setText(data_text)
        self.log_path.setText("运行日志：" + str(snapshot.log_path or "尚未创建"))
        if self.raw_state_toggle.isChecked():
            raw = {
                "session_id": snapshot.session_id,
                "state": snapshot.state,
                "error": snapshot.error,
                "exit_confirmed": snapshot.exit_confirmed,
                "telemetry": plain(row),
                "recording": plain(recording),
            }
            self.raw_state.setPlainText(json.dumps(raw, ensure_ascii=False, indent=2, default=str))
        results = tuple(snapshot.operation_results or ())
        if results != self._last_results:
            previous = {result.get("request_id") for result in self._last_results}
            for result in results:
                if result.get("request_id") not in previous:
                    submitted_form = self._pending_forms.pop(result.get("request_id"), None)
                    if (
                        submitted_form is not None
                        and result.get("success")
                        and submitted_form == self._form_payload()
                    ):
                        self._form_dirty = self._arm_dirty = False
                    if result.get("action") in ("bind_cameras", "close_camera_preview") and result.get(
                        "success"
                    ):
                        self._binding_active = False
                        self._binding_preview_timestamps.clear()
                        self._preview_errors.clear()
                        for preview in self.binding_labels.values():
                            preview.clear_image()
                        self._page_changed(self.pages.currentIndex())
                    elif result.get("action") == "preview_cameras" and not result.get("success"):
                        self._binding_active = not bool(caps.get("close_camera_preview"))
                    elif result.get("action") == "verify_dataset" and not result.get("success"):
                        self._show_verification_start_error(result.get("message", "请检查数据目录和会话状态"))
                    icon = "完成" if result.get("success") else "未完成"
                    self.operation_log.appendPlainText(
                        f"{time.strftime('%H:%M:%S')}  {self._action_title(result.get('action'))} · {icon}：{result.get('message', '')}"
                    )
            self._last_results = results

    @staticmethod
    def _set_tone(widget, property_name, tone):
        if widget.property(property_name) != tone:
            widget.setProperty(property_name, tone)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _confirmation_changed(self, *_):
        snapshot = self._snapshot
        if snapshot is None:
            return
        self._refresh_next_step(snapshot)
        reason = snapshot.capabilities.get("start", "等待当前操作完成")
        if not self.confirmed.isChecked():
            reason = reason or "请先勾选本次机械与力通道确认"
        for button, action in self._buttons:
            if action == "start":
                button.setEnabled(not bool(reason) and not self._closing)
                button.setToolTip(str(reason) if reason else ACTION_HELP["start"])

    def _refresh_next_step(self, snapshot):
        if snapshot is None:
            return
        state = str(snapshot.state).lower()
        recording = snapshot.recording or {}
        record_state = str(recording.get("state", "idle")).lower()
        caps = snapshot.capabilities
        if state == "running":
            if (
                record_state in {"error", "corrupt", "interrupted", "commit_unknown"}
                or recording.get("error")
                or recording.get("request_error")
                or recording.get("cleanup_error")
                or recording.get("cleanup_pending")
            ):
                text = "录制有异常，请查看诊断与记录。"
            elif not snapshot.recording_values.get("enabled"):
                text = "未准备录制；停止后可重新设置。"
            elif record_state == "saved_pending_verification":
                text = "已保存，待核验；请查看诊断。"
            elif record_state in {"saving", "verifying"} or recording.get("busy"):
                text = "正在等待录制保存或核验结果。"
            elif recording.get("active"):
                text = "录制中：选结果后点击结束并保存。"
            elif record_state == "saved" and recording.get("simulated"):
                text = "演示完成，可继续下一条或停止。"
            elif record_state == "saved" and recording.get("file_complete") is not True:
                text = "数据完整性待确认，请查看诊断。"
            elif not caps.get("record_start", "等待录制准备"):
                text = "已保存，可开始下一条，或停止。" if record_state == "saved" else "可开始录制。"
            else:
                text = str(caps.get("record_start", "等待录制准备"))
        elif state == "recovery_required":
            text = "退出尚未确认，请先完成只读复核。"
        elif state == "disconnected":
            text = "先连接设备并取得新鲜只读反馈。"
        elif state == "monitoring":
            reason = caps.get("start", "连接设备并检查准备条件")
            text = (
                str(reason)
                if reason
                else (
                    "可启动遥操。" if self.confirmed.isChecked() else "请确认机械与力通道并勾选，再启动遥操。"
                )
            )
        else:
            text = snapshot.message or "等待当前操作完成。"
        self.next_step.setText(text)

    def _refresh_feedback_status(self, snapshot):
        state = str(snapshot.state).lower()
        row = snapshot.telemetry or {}
        if state in {"disconnected", "recovery_required", "closed"}:
            text = "当前无实时反馈；曲线显示最近历史。"
        elif not row:
            text = "反馈状态：等待连接"
        else:
            invalid = []
            if row.get("feedback_valid") is False or row.get("joint_valid") is False:
                invalid.append("关节")
            if row.get("force_valid") is False:
                invalid.append("力")
            if invalid:
                self._set_feedback_text(" / ".join(invalid) + "反馈无效；对应数值不显示。")
                return
            phase = str(row.get("phase") or "")
            text = (
                f"反馈：{PHASE_LABELS.get(phase, phase) or '未知'}"
                f" · 夹爪位置 {self._format(row.get('q_measured_deg'))}°"
            )
            anchor = row.get("decision_anchor_ns")
            if type(anchor) is int and anchor > 0:
                age = (time.perf_counter_ns() - anchor) / 1e9
                text += (
                    " · 反馈时间异常"
                    if age < 0
                    else f" · 反馈停更 {age:.1f} 秒"
                    if age > 2
                    else f" · 更新于 {age:.1f} 秒前"
                )
            else:
                text += " · 反馈时间未知"
        self._set_feedback_text(text)

    def _set_feedback_text(self, text):
        self.phase_label.setText(text)
        self.tuning_force_status.setText(text)

    @staticmethod
    def _format(value):
        try:
            number = float(value) if not isinstance(value, bool) else math.nan
            return f"{number:.3f}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    def _hydrate(self, snapshot):
        if self._form_dirty and self._hydrated:
            return
        self._position_command_dirty.clear()
        for key, value in snapshot.values.items():
            for edit in self._field_edits.get(key, ()):
                with QtCore.QSignalBlocker(edit):
                    edit.setText(str(value))
            if key in self._position_command_edits:
                self._position_command_edits[key].setText(str(value))
        for key, value in snapshot.toggles.items():
            if key in self._toggle_checks:
                with QtCore.QSignalBlocker(self._toggle_checks[key]):
                    self._toggle_checks[key].setChecked(bool(value))
        for key, value in snapshot.recording_values.items():
            if key in self._recording_edits:
                self._recording_edits[key].setText(str(value))
        self.resume_checkbox.setChecked(snapshot.recording_values.get("resume") is True)
        self.verify_video_checkbox.setChecked(snapshot.recording_values.get("verify_video_on_save") is True)
        self.recording_enabled.setChecked(snapshot.recording_values.get("enabled") is True)
        self._camera_configs = plain(snapshot.camera_configs)
        self.camera_json.setPlainText(json.dumps(self._camera_configs, ensure_ascii=False, indent=2))
        arm = getattr(snapshot, "arm_settings", {}) or {}
        for key, value in arm.get("operator", {}).items():
            edit = self._arm_edits.get(("operator", key))
            if edit is not None:
                edit.setText(str(value))
        for joint, values in arm.get("joints", {}).items():
            for key, value in values.items():
                edit = self._arm_edits.get((joint, key))
                if edit is not None:
                    edit.setText(str(value))
        self._settings_file = snapshot.settings_path
        self._hydrated = True
        self._form_dirty = self._arm_dirty = False

    def _update_devices(self, snapshot):
        devices = tuple(snapshot.devices or ())
        signature = tuple((item.get("device_name"), item.get("device_path")) for item in devices)
        if signature == self._device_signature:
            return
        self._device_signature = signature
        for role, choice in self.camera_choices.items():
            current = choice.currentData() or self._camera_configs.get(role, {}).get("device_path")
            with QtCore.QSignalBlocker(choice):
                choice.clear()
                choice.addItem("选择设备…", None)
                for item in devices:
                    path = item.get("device_path")
                    if path:
                        name = item.get("device_name") or "相机"
                        choice.addItem(f"{name} · {path}", path)
                selected = choice.findData(current)
                if selected >= 0:
                    choice.setCurrentIndex(selected)

    def _poll_preview(self):
        if not self._preview_interest or self._decode_busy:
            return
        if (
            self.pages.currentIndex() == 0
            and not self._binding_active
            and self._snapshot is not None
            and self._snapshot.state in {"disconnected", "recovery_required", "closed"}
        ):
            return
        frames = self.controller.preview()
        latest = {
            role: dict(frame)
            for role, frame in frames.items()
            if role in self.preview_labels
            and isinstance(frame, Mapping)
            and type(frame.get("timestamp_ns")) is int
            and frame["timestamp_ns"] >= self._preview_epoch_ns
            and frame["timestamp_ns"] != self._preview_timestamps.get(role)
        }
        if latest:
            self._decode_busy = True
            self.decode_requested.emit(latest)

    @QtCore.Slot(str, object, QtGui.QImage)
    def _decoded(self, role, timestamp, image):
        if not self._accept_preview_frame(timestamp):
            return
        self._preview_timestamps[role] = timestamp
        if self._preview_errors.pop(role, None) and self._last_problem is not None:
            self._last_problem.pop("local_signature", None)
        if self.pages.currentIndex() == 0:
            self.preview_labels[role].set_image(image)
            self._home_preview_timestamps[role] = timestamp
        if self._binding_active and timestamp >= self._binding_started_ns:
            self.binding_labels[role].set_image(image)
            self._binding_preview_timestamps[role] = timestamp
            self._binding_seen.add(role)
            self.binding_status.setText("请确认两幅画面与前景 / 腕部对应，再保存绑定。")
            self._last_revision = None

    @QtCore.Slot()
    def _decode_finished(self):
        self._decode_busy = False

    @QtCore.Slot(str)
    def _preview_error(self, message):
        self._remember_local_problem("相机预览", message)
        self.feedback.setText(message)

    @QtCore.Slot(str, object, str)
    def _preview_frame_failed(self, role, timestamp, message):
        if not self._accept_preview_frame(timestamp):
            return
        self._preview_timestamps[role] = timestamp
        self._preview_errors[role] = message
        self._remember_local_problem(f"{role} 相机预览", message)
        self._refresh_preview_status()

    def _accept_preview_frame(self, timestamp):
        if not self._preview_interest or type(timestamp) is not int or timestamp < self._preview_epoch_ns:
            return False
        page = self.pages.currentIndex()
        if page == 2 and self._binding_active and timestamp < self._binding_started_ns:
            return False
        return not (
            page == 0
            and not self._binding_active
            and self._snapshot is not None
            and self._snapshot.state in {"disconnected", "recovery_required", "closed"}
        )

    def _refresh_preview_status(self):
        now = time.perf_counter_ns()
        for role in self.preview_labels:
            for status, stamps, enabled in (
                (self.preview_status[role], self._home_preview_timestamps, self.pages.currentIndex() == 0),
                (self.binding_preview_status[role], self._binding_preview_timestamps, self._binding_active),
            ):
                stamp = stamps.get(role)
                error = self._preview_errors.get(role) if enabled else None
                if error:
                    text, tone = error, "fault"
                elif not enabled:
                    text, tone = "预览已暂停", "normal"
                elif type(stamp) is not int:
                    text, tone = "等待画面", "normal"
                else:
                    age = (now - stamp) / 1e9
                    if age < 0:
                        text, tone = "画面时间异常", "fault"
                    elif age > 2:
                        text, tone = f"画面停更 {age:.1f} 秒", "fault"
                    else:
                        text, tone = f"更新于 {age:.1f} 秒前", "normal"
                if status.text() != text:
                    status.setText(text)
                self._set_tone(status, "statusTone", tone)

    def _binding_confirmation_reason(self):
        if not self._binding_active or self._binding_seen != {"front", "wrist"}:
            return "先预览并核对当前选择的两路画面"
        if any(self._preview_errors.get(role) for role in ("front", "wrist")):
            return "预览解码失败，请等待两路新画面"
        now = time.perf_counter_ns()
        for role in ("front", "wrist"):
            stamp = self._binding_preview_timestamps.get(role)
            if type(stamp) is not int or not 0 <= now - stamp <= 2_000_000_000:
                return "预览画面已停更，请等待两路新画面"
        return ""

    def _refresh_binding_button(self, snapshot):
        reason = snapshot.capabilities.get("bind_cameras", "等待当前操作完成")
        reason = reason or self._binding_confirmation_reason()
        enabled = not bool(reason) and not self._closing
        if self.bind_button.isEnabled() != enabled:
            self.bind_button.setEnabled(enabled)
        self.bind_button.setToolTip(str(reason) if reason else ACTION_HELP.get("bind_cameras", ""))

    def _render_charts(self):
        try:
            if self.pages.currentIndex() in (0, 1):
                self.charts.append(self.controller.snapshot().telemetry or {})
            self.charts.render()
        except Exception as exc:
            self.feedback.setText(f"曲线显示失败：{exc}；停止仍可操作。")

    def _camera_selection(self):
        selected = {role: choice.currentData() for role, choice in self.camera_choices.items()}
        if not all(selected.values()) or len(set(selected.values())) != 2:
            raise ValueError("请选择两台不同的前景 / 腕部相机，再预览确认。")
        return selected

    def apply_camera_json(self):
        try:
            value = json.loads(self.camera_json.toPlainText())
            if not isinstance(value, dict) or set(value) != {"front", "wrist"}:
                raise ValueError("相机配置必须包含 front 与 wrist")
            self._camera_configs = value
            self._form_dirty = True
            self.feedback.setText("相机配置已填入草稿；下次预检时验证。")
        except (ValueError, TypeError) as exc:
            self.feedback.setText(str(exc))

    def choose_dataset(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if directory:
            self._recording_edits["root"].setText(directory)
            self._mark_dirty()

    def choose_file(self, action):
        if action == "save_settings":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存参数草稿", str(DEFAULT_SETTINGS), "JSON (*.json)"
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择校准" if action == "load_calibration" else "加载参数", "", "JSON (*.json)"
            )
        if path:
            self.dispatch(action, {"path": path}, update=action == "save_settings")

    def open_log_directory(self):
        from pathlib import Path

        path = getattr(self._snapshot, "log_path", None)
        if path:
            local = Path(path)
            directory = local if local.is_dir() else local.parent
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))

    def closeEvent(self, event):  # noqa: N802 - Qt virtual method.
        if not self.controller.closed:
            if not self._closing:
                self._closing = True
                self.controller.set_preview_enabled(False)
                self.controller.close()
                self.feedback.setText("正在停止并关闭设备，等待资源退出结果…")
                self._last_revision = None
            event.ignore()
            return
        self.status_timer.stop()
        self.plot_timer.stop()
        if not self.demo:
            self.preferences.setValue("window_geometry", self.saveGeometry())
        self.decoder_thread.quit()
        if not self.decoder_thread.wait(2000):
            self.feedback.setText("预览解码尚未结束，继续等待退出…")
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
            return
        event.accept()
