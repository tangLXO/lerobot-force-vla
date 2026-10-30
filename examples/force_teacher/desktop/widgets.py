"""Small desktop widgets and a worker which decodes only disposable preview pixels."""

import math

from lerobot.utils.import_utils import _pyside6_available, require_package

if _pyside6_available:
    from PySide6 import QtCore, QtGui, QtWidgets

require_package("PySide6", "force-teacher-ui")


class PreviewLabel(QtWidgets.QLabel):
    """Keep the source pixmap and scale only when a new frame or size arrives."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self._source = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 85)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setObjectName("cameraPreview")
        self.setText(f"{title}\n等待相机画面")

    def set_image(self, image):
        self._source = QtGui.QPixmap.fromImage(image)
        self._scale()

    def clear_image(self):
        self._source = None
        self.clear()
        self.setText(f"{self.title}\n等待相机画面")

    def _scale(self):
        if self._source is not None:
            self.setPixmap(
                self._source.scaled(
                    self.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
            )

    def resizeEvent(self, event):  # noqa: N802 - Qt virtual method.
        super().resizeEvent(event)
        self._scale()


class PreviewDecoder(QtCore.QObject):
    """QImage decoding is worker-safe; QPixmap creation stays in the GUI thread."""

    decoded = QtCore.Signal(str, object, QtGui.QImage)
    failed = QtCore.Signal(str)
    frame_failed = QtCore.Signal(str, object, str)
    finished = QtCore.Signal()

    @QtCore.Slot(object)
    def decode(self, frames):
        try:
            for role, frame in frames.items():
                jpeg = frame.get("jpeg")
                stamp = frame.get("timestamp_ns")
                if (
                    role not in ("front", "wrist")
                    or not isinstance(jpeg, bytes)
                    or len(jpeg) > 262_144
                    or type(stamp) is not int
                ):
                    continue
                image = QtGui.QImage.fromData(jpeg, "JPEG")
                if image.isNull() or image.width() * image.height() > 16_777_216:
                    self.frame_failed.emit(role, stamp, "画面解码失败，显示的可能是上一帧")
                    continue
                image = image.scaled(
                    640,
                    480,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
                self.decoded.emit(role, stamp, image)
        except Exception as exc:
            self.failed.emit(f"相机预览显示失败：{exc}")
        finally:
            self.finished.emit()


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title, color="#0f766e", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(13, 8, 13, 8)
        layout.setSpacing(1)
        caption = QtWidgets.QLabel(title)
        caption.setObjectName("caption")
        self.value = QtWidgets.QLabel("—")
        self.value.setStyleSheet(f"font-size: 23px; font-weight: 600; color: {color};")
        layout.addWidget(caption)
        layout.addWidget(self.value)

    def set_value(self, value, unit="N"):
        try:
            number = float(value) if not isinstance(value, bool) else math.nan
            text = f"{number:.3f} {unit}" if math.isfinite(number) else "—"
        except (ValueError, TypeError):
            text = "—"
        if self.value.text() != text:
            self.value.setText(text)


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QtWidgets.QToolButton(text=title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.content = QtWidgets.QWidget()
        self.content.hide()
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self._toggle)

    def _toggle(self, expanded):
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )


def card(title, parent=None):
    frame = QtWidgets.QGroupBox(title, parent)
    frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
    return frame


def scroll_page():
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QtWidgets.QWidget()
    scroll.setWidget(body)
    layout = QtWidgets.QVBoxLayout(body)
    layout.setContentsMargins(18, 10, 18, 18)
    layout.setSpacing(14)
    return scroll, body, layout
