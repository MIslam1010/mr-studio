"""Reusable Qt widgets: joint sliders, matrix display, labelled spin boxes, plots."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

pg.setConfigOptions(antialias=True, background="#16191e", foreground=theme.MUTED)


class SliderRow(QtWidgets.QWidget):
    """Label + slider + spin box for one scalar (displayed in `unit`, stored as float)."""
    valueChanged = QtCore.Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float = 0.0, decimals: int = 1,
                 unit: str = "", scale: float = 1.0, parent=None):
        super().__init__(parent)
        self.scale = scale  # display value = internal value * scale
        self._lo, self._hi = lo, hi
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lbl = QtWidgets.QLabel(label)
        self.lbl.setMinimumWidth(34)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 2000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(lo * scale, hi * scale)
        self.spin.setSingleStep((hi - lo) * scale / 200)
        self.spin.setSuffix(unit)
        self.spin.setMinimumWidth(92)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.lbl)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._block = False
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value, emit=False)

    def _from_slider(self, k):
        if self._block:
            return
        v = self._lo + (self._hi - self._lo) * k / 2000.0
        self._block = True
        self.spin.setValue(v * self.scale)
        self._block = False
        self.valueChanged.emit(v)

    def _from_spin(self, d):
        if self._block:
            return
        v = d / self.scale
        self._block = True
        self.slider.setValue(int(round((v - self._lo) / (self._hi - self._lo) * 2000)))
        self._block = False
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self.spin.value() / self.scale

    def set_value(self, v: float, emit: bool = False):
        v = float(np.clip(v, self._lo, self._hi))
        self._block = True
        self.spin.setValue(v * self.scale)
        self.slider.setValue(int(round((v - self._lo) / (self._hi - self._lo) * 2000)))
        self._block = False
        if emit:
            self.valueChanged.emit(v)


class JointSliders(QtWidgets.QWidget):
    """One slider per joint: revolute joints in degrees on screen (radians in code); prismatic
    joints in metres (on screen and in code)."""
    changed = QtCore.Signal(object)  # np.ndarray (rad / m)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(2)
        self.rows: list[SliderRow] = []

    def set_robot(self, robot, theta=None):
        for r in self.rows:
            r.setParent(None)
            r.deleteLater()
        self.rows = []
        theta = robot.home if theta is None else theta
        prism = getattr(robot, "is_prismatic", np.zeros(robot.n, bool))
        for i in range(robot.n):
            lo, hi = robot.joint_limits[i]
            if prism[i]:   # prismatic joint: S = (0, v), displacement in metres (MR Sec. 3.3.2.2)
                lo, hi = max(lo, -2.0), min(hi, 2.0)
                row = SliderRow(f"θ{_sub(i + 1)}", lo, hi, theta[i], decimals=3, unit=" m", scale=1.0)
                row.setToolTip(f"Prismatic joint {i + 1}: displacement along its screw axis, in metres")
            else:
                lo, hi = max(lo, -2 * np.pi), min(hi, 2 * np.pi)
                row = SliderRow(f"θ{_sub(i + 1)}", lo, hi, theta[i], decimals=1, unit="°", scale=180 / np.pi)
            row.valueChanged.connect(lambda _v: self.changed.emit(self.values()))
            self.lay.addWidget(row)
            self.rows.append(row)

    def values(self) -> np.ndarray:
        return np.array([r.value() for r in self.rows])

    def set_values(self, theta, emit: bool = False):
        for r, v in zip(self.rows, theta):
            r.set_value(float(v))
        if emit:
            self.changed.emit(self.values())


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


class MatrixView(QtWidgets.QTableWidget):
    """Compact read-only numeric matrix display."""

    def __init__(self, rows=4, cols=4, decimals=4, parent=None, row_labels=None, col_labels=None):
        super().__init__(rows, cols, parent)
        self.decimals = decimals
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        if row_labels is None:
            self.verticalHeader().setVisible(False)
        else:
            self.setVerticalHeaderLabels(row_labels)
        if col_labels is None:
            self.horizontalHeader().setVisible(False)
        else:
            self.setHorizontalHeaderLabels(col_labels)
        self._fit()

    def _fit(self):
        # rowHeight()/header height can still be 0 before the widget is first shown, so fall back to
        # the default section size; isHidden() (not isVisible()) also works before the parent is shown
        dflt = self.verticalHeader().defaultSectionSize()
        h = sum(self.rowHeight(i) or dflt for i in range(self.rowCount())) + 6
        if not self.horizontalHeader().isHidden():
            h += self.horizontalHeader().height() or self.horizontalHeader().sizeHint().height()
        self.setFixedHeight(min(h, 340))

    def set_matrix(self, A, highlight=None):
        A = np.atleast_2d(np.asarray(A))
        if A.shape != (self.rowCount(), self.columnCount()):
            self.setRowCount(A.shape[0])
            self.setColumnCount(A.shape[1])
            self._fit()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                txt = v if isinstance(v, str) else f"{v:+.{self.decimals}f}"
                it = self.item(i, j)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    it.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.setItem(i, j, it)
                it.setText(txt)
                if highlight is not None and highlight(i, j, v):
                    it.setForeground(QtGui.QColor(theme.WARN))
                else:
                    it.setForeground(QtGui.QColor(theme.TEXT))

    def set_labels(self, rows=None, cols=None):
        if rows is not None:
            self.verticalHeader().setVisible(True)
            self.setVerticalHeaderLabels(rows)
        if cols is not None:
            self.horizontalHeader().setVisible(True)
            self.setHorizontalHeaderLabels(cols)
        self._fit()


def group(title: str, *widgets, layout="v") -> QtWidgets.QGroupBox:
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g) if layout == "v" else QtWidgets.QHBoxLayout(g)
    lay.setSpacing(4)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return g


def hbox(*widgets) -> QtWidgets.QHBoxLayout:
    lay = QtWidgets.QHBoxLayout()
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        elif w is None:
            lay.addStretch(1)
        else:
            lay.addWidget(w)
    return lay


def value_label(text="") -> QtWidgets.QLabel:
    l = QtWidgets.QLabel(text)
    l.setObjectName("value")
    l.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    l.setWordWrap(True)
    return l


def button(text, slot=None, checkable=False, primary=False, tooltip=None) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.setCheckable(checkable)
    if primary:
        b.setObjectName("primary")
    if slot is not None:
        (b.toggled if checkable else b.clicked).connect(slot)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def checkbox(text, slot=None, checked=False) -> QtWidgets.QCheckBox:
    c = QtWidgets.QCheckBox(text)
    c.setChecked(checked)
    if slot is not None:
        c.toggled.connect(slot)
    return c


def combo(items, slot=None) -> QtWidgets.QComboBox:
    c = QtWidgets.QComboBox()
    c.addItems(list(items))
    if slot is not None:
        c.currentTextChanged.connect(slot)
    return c


def spin(lo, hi, value, decimals=3, step=None, suffix="") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setSingleStep(step if step is not None else (hi - lo) / 100)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def plot(title: str = "", ylabel: str = "", xlabel: str = "t [s]", legend=True) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setTitle(title, color=theme.TEXT, size="10pt")
    w.setLabel("left", ylabel)
    w.setLabel("bottom", xlabel)
    w.showGrid(x=True, y=True, alpha=0.15)
    if legend:
        w.addLegend(offset=(6, 6), labelTextSize="8pt")
    w.setMinimumHeight(150)
    return w


def pen(i: int, width=2.0, style=None):
    kw = dict(color=theme.SERIES[i % len(theme.SERIES)], width=width)
    if style == "dash":
        kw["style"] = QtCore.Qt.DashLine
    return pg.mkPen(**kw)
