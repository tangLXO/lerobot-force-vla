"""Bounded display data; these samples never enter recording or control decisions."""

from collections import deque

import numpy as np

from lerobot.utils.import_utils import _pyqtgraph_available, _pyside6_available, require_package

if _pyqtgraph_available:
    import pyqtgraph as pg
if _pyside6_available:
    from PySide6 import QtWidgets

require_package("PySide6", "force-teacher-ui")
require_package("pyqtgraph", "force-teacher-ui")


class TelemetryCharts:
    def __init__(self, seconds=20):
        self.seconds = seconds
        self.points = deque(maxlen=12000)
        self.last_anchor = None
        self.revision = 0
        self._rendered = {}
        self._plots = []
        self._curves = {}

    def create(self, title, fields, *, minimum_height=150, legend_columns=1):
        plot = pg.PlotWidget(background="w")
        plot.setMinimumHeight(minimum_height)
        plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        plot.setTitle(title, color="#334155", size="11pt")
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.addLegend(offset=(12, 4), labelTextColor="#475569", labelTextSize="9pt", colCount=legend_columns)
        plot.setLabel("bottom", "距最新反馈", units="s")
        plot.setMouseEnabled(x=True, y=True)
        plot.setMenuEnabled(False)
        plot.setClipToView(True)
        plot.setDownsampling(auto=True, mode="peak")
        plot.setXRange(-self.seconds, 0, padding=0)
        curves = []
        for key, label, color in fields:
            curve = plot.plot(name=label, pen=pg.mkPen(color, width=1.5), connect="finite")
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            curves.append((key, curve))
        self._plots.append(plot)
        self._curves[plot] = curves
        return plot

    def append(self, row):
        anchor = row.get("decision_anchor_ns")
        if type(anchor) is not int or anchor == self.last_anchor:
            return
        if self.last_anchor is not None and anchor < self.last_anchor:
            self.points.clear()
        self.last_anchor = anchor
        display_row = dict(row)
        if display_row.get("force_valid") is False:
            for key in ("force_left_N", "force_right_N", "force_mean_N"):
                display_row[key] = None
        self.points.append(display_row)
        oldest = anchor - int(self.seconds * 1e9)
        while self.points and self.points[0]["decision_anchor_ns"] < oldest:
            self.points.popleft()
        self.revision += 1

    def render(self):
        if not self.points:
            return
        visible = [p for p in self._plots if p.isVisible() and self._rendered.get(p) != self.revision]
        if not visible:
            return
        rows = tuple(self.points)
        origin = rows[-1]["decision_anchor_ns"]
        x = np.asarray([(row["decision_anchor_ns"] - origin) / 1e9 for row in rows])
        for plot in visible:
            for key, curve in self._curves[plot]:
                y = np.asarray([self._number(row.get(key)) for row in rows])
                curve.setData(x=x, y=y, connect="finite")
            self._rendered[plot] = self.revision

    @staticmethod
    def _number(value):
        if isinstance(value, bool) or value is None:
            return np.nan
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError):
            return np.nan
