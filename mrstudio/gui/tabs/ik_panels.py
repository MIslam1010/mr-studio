"""Qt helpers for the inverse-kinematics tab: control-panel / bottom-pane construction, ghost drawing,
table sizing and the convergence plot."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme
from ..widgets import (JointSliders, MatrixView, button, checkbox, combo, group, hbox, pen, plot, spin,
                       value_label)

TOLS = ["1e-2", "1e-3", "1e-4", "1e-6", "1e-8", "1e-10", "1e-12"]


def _rgba(hexcol: str, a: float):
    c = QtGui.QColor(hexcol)
    return (c.redF(), c.greenF(), c.blueF(), a)


def _ghost(view, robot, key, color, width=2.5, geom=None, theta=None):
    """Translucent stick figure (same look as viewer.GhostRobot) that can be re-coloured without recomputing FK."""
    g = geom if geom is not None else robot.geometry(np.asarray(theta, float))
    view.set_polyline(key + ".line", np.vstack(g["polys"]), color, width)
    view.set_points(key + ".pts", g["joint_pos"], color, 7)
    view.set_points(key + ".ee", g["T_ee"][:3, 3][None], color, 11)
    return g


def _fit_table(mv: MatrixView):
    mv.setMinimumHeight(60)
    mv.setMaximumHeight(16777215)
    mv.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    mv.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)


def build_panels(self):
    """Create all widgets of IKTab (stored as attributes on `self`); returns (controls, bottom)."""
    # ---------------------------------------------------------------- target
    self.target_box = QtWidgets.QWidget()
    self.target_lay = QtWidgets.QVBoxLayout(self.target_box)
    self.target_lay.setContentsMargins(0, 0, 0, 0)
    self.target_lay.setSpacing(2)
    self.target_lbl = value_label()
    tbtns = hbox(button("Target = current EE", self.target_from_ee),
                 button("Random reachable", self.random_target,
                        tooltip="T_sd = FK of a random joint vector, so a solution exists"))
    self.cb_live = checkbox("Live IK while dragging the target (warm start)", None, False)

    # ---------------------------------------------------------------- configuration
    self.sliders = JointSliders()
    self.sliders.changed.connect(self._sliders_moved)
    cbtns = hbox(button("Ready pose", lambda: self._set_theta(self.ctx.robot.home),
                        tooltip="a convenient non-singular start pose; the MR home configuration is θ = 0, T = M"),
                 button("Random", lambda: self._set_theta(self.ctx.robot.random_config())),
                 button("Clear ghosts", self.clear_ghosts))

    # ---------------------------------------------------------------- solver
    self.method = combo(["Newton–Raphson · body frame (IKinBody)", "Newton–Raphson · space frame (IKinSpace)",
                         "Damped least squares (DLS)"], self._method_changed)
    self.lam = spin(0.001, 2.0, 0.05, 3, 0.01)
    self.guess = combo(["current θ (sliders)", "ready-pose θ", "random θ"])
    self.eomg = combo(TOLS)
    self.eomg.setCurrentText("1e-6")
    self.ev = combo(TOLS)
    self.ev.setCurrentText("1e-6")
    for cb in (self.method, self.guess, self.eomg, self.ev):
        cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(8)
    self.maxit = QtWidgets.QSpinBox()
    self.maxit.setRange(1, 500)
    self.maxit.setValue(50)
    form = QtWidgets.QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow("Method", self.method)
    form.addRow("DLS λ", self.lam)
    form.addRow("Initial guess θ⁰", self.guess)
    form.addRow("ε_ω  [rad]", self.eomg)
    form.addRow("ε_v  [m]", self.ev)
    form.addRow("Max iterations", self.maxit)
    self.solve_btn = button("▶  Solve IK", self.solve, primary=True)
    self.result_lbl = value_label()
    self.mr_lbl = value_label()

    # ---------------------------------------------------------------- multiple solutions
    self.all_btn = button("Find all solutions", self.find_all,
                          tooltip="Newton–Raphson from random seeds until 60 starts in a row find no new "
                                  "branch (at most 600); distinct wrapped solutions within the limits kept")
    self.all_cancel = button("Cancel", self._cancel_all)
    self.all_cancel.setVisible(False)
    self.sol_list = QtWidgets.QListWidget()
    self.sol_list.setMaximumHeight(170)
    self.sol_list.itemClicked.connect(self._solution_clicked)
    self.all_lbl = value_label()
    self.analytic_lbl = value_label()

    # ---------------------------------------------------------------- redundancy
    self.self_btn = button("▶ Self-motion (null space)", self.toggle_self_motion, checkable=True)
    self.drift_lbl = value_label()

    # ---------------------------------------------------------------- real-time
    self.circle_btn = button("▶ Follow a circle", self.toggle_circle, checkable=True)
    self.radius = spin(0.02, 0.6, 0.12, 2, 0.01, " m")
    self.period = spin(1.0, 30.0, 6.0, 1, 0.5, " s")
    self.cb_trace = checkbox("Trace the end-effector path", self._trace_toggled, True)
    self.rt_lbl = value_label()

    controls = QtWidgets.QWidget()
    cl = QtWidgets.QVBoxLayout(controls)
    cl.addWidget(group("Target pose T_sd  (drawn as {d})", self.target_box, tbtns, self.cb_live,
                       self.target_lbl))
    cl.addWidget(group("Solver  (MR §6.2)", form, self.solve_btn, self.result_lbl, self.mr_lbl))
    cl.addWidget(group("Multiple solutions  (MR §6.1)", hbox(self.all_btn, self.all_cancel), self.all_lbl,
                       self.sol_list,
                       self.analytic_lbl))
    cl.addWidget(group("Redundancy: null-space self-motion  (MR §6.3)", self.self_btn, self.drift_lbl))
    cl.addWidget(group("Real-time IK", hbox(self.circle_btn, QtWidgets.QLabel("r"), self.radius),
                       hbox(QtWidgets.QLabel("period"), self.period, self.cb_trace), self.rt_lbl))
    cl.addWidget(group("Robot configuration θ  (= current guess)", self.sliders, cbtns))
    cl.addStretch(1)

    # ---------------------------------------------------------------- bottom
    self.conv_plot = plot("Newton–Raphson convergence", "error norm (log)", "iteration i", legend=True)
    self.conv_plot.setLogMode(y=True)
    self.order_lbl = QtWidgets.QLabel()
    self.order_lbl.setObjectName("muted")
    cw = QtWidgets.QWidget()
    cwl = QtWidgets.QVBoxLayout(cw)
    cwl.setContentsMargins(0, 0, 0, 0)
    cwl.addWidget(self.conv_plot, 1)
    cwl.addWidget(self.order_lbl)
    self.iter_table = MatrixView(1, 4, 3)
    self.iter_table.setStyleSheet("QTableWidget { font-size: 11px; }")
    tw = QtWidgets.QWidget()
    twl = QtWidgets.QVBoxLayout(tw)
    twl.setContentsMargins(0, 0, 0, 0)
    cap = QtWidgets.QLabel("Iterates θⁱ (cf. MR Example 6.1)")
    cap.setObjectName("muted")
    twl.addWidget(cap)
    twl.addWidget(self.iter_table, 1)
    bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    bottom.addWidget(cw)
    bottom.addWidget(tw)
    bottom.setStretchFactor(0, 1)
    bottom.setStretchFactor(1, 1)
    bottom.setSizes([430, 420])
    lg = self.conv_plot.plotItem.legend
    if lg is not None:
        lg.anchor((1, 0), (1, 0), offset=(-8, 8))
    return controls, bottom


def plot_convergence(pw, order_lbl, res, args):
    """Draw the error curves of one IK run and the observed order of convergence."""
    pw.clear()
    k = np.arange(len(res.err_w))
    kmax = max(int(k[-1]), 1)
    ew = np.maximum(np.asarray(res.err_w), 1e-17)
    ev = np.maximum(np.asarray(res.err_v), 1e-17)
    if not args["position_only"]:
        f = "s" if res.frame == "space" else "b"
        pw.plot(k, ew, pen=pen(0), symbol="o", symbolSize=6, symbolBrush=theme.SERIES[0],
                symbolPen=None, name=f"‖ω_{f}‖ [rad]")
        pw.plot(k, ev, pen=pen(1), symbol="s", symbolSize=6, symbolBrush=theme.SERIES[1],
                symbolPen=None, name=f"‖v_{f}‖ [m]")
        pw.plot([0, kmax], [args["eomg"]] * 2, pen=pen(0, 1, "dash"), name="ε_ω")
        pw.plot([0, kmax], [args["ev"]] * 2, pen=pen(1, 1, "dash"), name="ε_v")
    else:
        pw.plot(k, ev, pen=pen(1), symbol="s", symbolSize=6, symbolBrush=theme.SERIES[1],
                symbolPen=None, name="‖p_d − p‖ [m]")
        pw.plot([0, kmax], [args["ev"]] * 2, pen=pen(1, 1, "dash"), name="ε_v")
    pw.setXRange(0, kmax, padding=0.04)
    # observed order of convergence  q ≈ log(e_i/e_{i-1}) / log(e_{i-1}/e_{i-2})
    e = np.hypot(res.err_w, res.err_v)
    q = None
    for i in range(2, len(e)):
        if min(e[i], e[i - 1], e[i - 2]) > 1e-7 and e[i] < e[i - 1] < e[i - 2] and e[i - 2] < 0.5:
            q = np.log(e[i] / e[i - 1]) / np.log(e[i - 1] / e[i - 2])
    if q is not None:
        kind = "quadratic" if q > 1.6 else ("superlinear" if q > 1.15 else "linear")
        order_lbl.setText(f"Observed order of convergence q = log(eᵢ/eᵢ₋₁)/log(eᵢ₋₁/eᵢ₋₂) ≈ {q:.2f}"
                               f"  →  {kind}")
    else:
        order_lbl.setText("Observed order of convergence: (too few iterations to estimate)")
