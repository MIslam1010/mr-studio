"""Tab 6 -- planar closed chains: four-bar linkage and five-bar parallel robot (MR Ch. 7, Sec. 8.7)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ... import closed_chains as cc
from .. import theme
from ..base import BaseTab
from ..jobs import JobRunner
from ..viewer import SceneView
from ..widgets import (MatrixView, SliderRow, button, checkbox, combo, group, hbox, plot, spin,
                       value_label)
from .closed_anim import ClosedAnimMixin
from .closed_common import CURVE, FOUR_PRESETS, LINK_COLS, _break_jumps, _deg, _wrapped  # noqa: F401
from .closed_draw import ClosedDrawMixin
from .closed_plots import ClosedPlotsMixin
from .closed_theory import THEORY


class ClosedChainTab(ClosedDrawMixin, ClosedPlotsMixin, ClosedAnimMixin, BaseTab):
    title = "⑥ Closed chains"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        self.mech: cc.ClosedChain | None = None
        self.theta = None
        self.sigma = 1
        self.elbows = cc.WORKING_MODES[0][0]
        self.trace: list = []
        self.sing_hist: list = []
        self.anim = None
        self.anim_dir = 1.0
        self.id_res = None
        self.lam_show = None
        self.id_jobs = JobRunner(self)          # inverse dynamics along a trajectory (background)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        app = QtWidgets.QApplication.instance()
        if app is not None and not hasattr(BaseTab, "shutdown"):   # older BaseTab without the quit hook
            app.aboutToQuit.connect(self.shutdown)
        self._last = time.time()

        # ------------------------------------------------------------ mechanism group
        self.mech_box = combo(["Four-bar linkage (1 DOF)", "Five-bar parallel robot (2 DOF)"])
        self.mech_box.currentIndexChanged.connect(self._mech_changed)
        self.preset_box = combo(list(FOUR_PRESETS), self._preset)
        self.s4 = {}
        f4 = QtWidgets.QFormLayout()
        for key, lab, val in (("d", "ground d = |OD|", 0.40), ("L1", "crank L₁ = |OA|", 0.12),
                              ("L2", "coupler L₂ = |AB|", 0.38), ("L3", "rocker L₃ = |DB|", 0.28),
                              ("u", "coupler pt C: u ∥ AB", 0.19), ("w", "coupler pt C: w ⊥ AB", 0.12)):
            lo = -0.4 if key in ("u", "w") else 0.04
            s = spin(lo, 0.8, val, 3, 0.005, " m")
            s.valueChanged.connect(self._params_changed)
            self.s4[key] = s
            f4.addRow(lab, s)
        p4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(p4)
        l4.setContentsMargins(0, 0, 0, 0)
        l4.addLayout(hbox(QtWidgets.QLabel("Preset:"), self.preset_box))
        l4.addLayout(f4)
        self.s5 = {}
        f5 = QtWidgets.QFormLayout()
        for key, lab, val in (("d", "ground d = |OD|", 0.20), ("L1", "left prox. L₁ = |OA|", 0.25),
                              ("L2", "left distal L₂ = |AP|", 0.35), ("L3", "right distal L₃ = |CP|", 0.35),
                              ("L4", "right prox. L₄ = |DC|", 0.25)):
            s = spin(0.02 if key == "d" else 0.05, 0.6, val, 3, 0.005, " m")
            s.valueChanged.connect(self._params_changed)
            self.s5[key] = s
            f5.addRow(lab, s)
        p5 = QtWidgets.QWidget()
        p5.setLayout(f5)
        f5.setContentsMargins(0, 0, 0, 0)
        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(p4)
        self.param_stack.addWidget(p5)
        self.grashof_lbl = QtWidgets.QLabel()
        self.grashof_lbl.setWordWrap(True)

        # ------------------------------------------------------------ kinematics pages
        self.crank = SliderRow("θ₁", -np.pi, np.pi, np.radians(40), decimals=1, unit="°", scale=180 / np.pi)
        self.crank.valueChanged.connect(lambda _v: self._kin_update())
        self.sigma_box = combo(["σ = +1", "σ = −1"])
        self.sigma_box.setToolTip("assembly mode: B to the left (+1) or right (−1) of the line A→D")
        self.sigma_box.currentIndexChanged.connect(self._sigma_changed)
        self.rot_btn = button("▶ Rotate crank", lambda on: self._anim_toggle("crank", on), checkable=True,
                              primary=True)
        self.cb_curve = checkbox("Coupler curves (both assembly modes)", self._redraw, True)
        self.cb_ghost = checkbox("Ghost of the other assembly mode", self._redraw, True)
        self.cb_labels = checkbox("Point labels", self._labels_toggled, True)
        k4 = QtWidgets.QWidget()
        kl4 = QtWidgets.QVBoxLayout(k4)
        kl4.setContentsMargins(0, 0, 0, 0)
        kl4.addWidget(QtWidgets.QLabel("Actuated joint (crank angle θ₁):"))
        kl4.addWidget(self.crank)
        kl4.addLayout(hbox(QtWidgets.QLabel("Assembly mode"), self.sigma_box))
        kl4.addWidget(self.rot_btn)
        kl4.addWidget(self.cb_curve)
        kl4.addStretch(1)

        self.px = SliderRow("Pₓ", -0.5, 0.7, 0.10, decimals=3, unit=" m")
        self.py = SliderRow("Pᵧ", -0.6, 0.8, 0.42, decimals=3, unit=" m")
        self.px.valueChanged.connect(lambda _v: self._p_changed())
        self.py.valueChanged.connect(lambda _v: self._p_changed())
        self.mode_box = combo([lab for _, lab in cc.WORKING_MODES])
        self.mode_box.setToolTip("signs of the passive elbow angles (θ₂, θ₄): the 4 inverse-kinematics solutions")
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.a1 = SliderRow("θ₁", -np.pi, np.pi, 0.0, decimals=1, unit="°", scale=180 / np.pi)
        self.a3 = SliderRow("θ₃", -np.pi, np.pi, 0.0, decimals=1, unit="°", scale=180 / np.pi)
        self.a1.valueChanged.connect(lambda _v: self._act_changed())
        self.a3.valueChanged.connect(lambda _v: self._act_changed())
        self.flip_btn = button("Flip assembly mode σ", self._flip_sigma5)
        self.circle_btn = button("▶ P traces a circle", lambda on: self._anim_toggle("circle", on), checkable=True,
                                 primary=True)
        self.cb_ws = checkbox("Workspace boundary = Type I loci (orange)", self._redraw, True)
        self.cb_t2 = checkbox("Type II loci for this working mode (red)", self._redraw, True)
        k5 = QtWidgets.QWidget()
        kl5 = QtWidgets.QVBoxLayout(k5)
        kl5.setContentsMargins(0, 0, 0, 0)
        kl5.addWidget(QtWidgets.QLabel("End-effector P → inverse kinematics:"))
        kl5.addWidget(self.px)
        kl5.addWidget(self.py)
        kl5.addLayout(hbox(QtWidgets.QLabel("Working mode"), self.mode_box))
        kl5.addWidget(QtWidgets.QLabel("Actuators → forward kinematics:"))
        kl5.addWidget(self.a1)
        kl5.addWidget(self.a3)
        kl5.addLayout(hbox(self.flip_btn, self.circle_btn))
        kl5.addWidget(self.cb_ws)
        kl5.addWidget(self.cb_t2)
        kl5.addStretch(1)
        self.kin_stack = QtWidgets.QStackedWidget()
        self.kin_stack.addWidget(k4)
        self.kin_stack.addWidget(k5)
        kin_page = QtWidgets.QWidget()
        kp = QtWidgets.QVBoxLayout(kin_page)
        kp.addWidget(self.kin_stack)
        kp.addWidget(self.cb_ghost)
        kp.addWidget(self.cb_labels)
        kp.addStretch(1)

        # ------------------------------------------------------------ dynamics page
        self.tau1 = spin(-20, 20, 0.0, 2, 0.1, " N·m")
        self.tau3 = spin(-20, 20, 0.0, 2, 0.1, " N·m")
        self.tau3_lbl = QtWidgets.QLabel("motor torque τ₃")
        self.damp = spin(0, 1, 0.0, 3, 0.005, " N·m·s")
        self.cb_baum = checkbox("Baumgarte stabilisation", None, True)
        self.alpha = spin(0, 200, 10, 1, 1)
        self.beta = spin(0, 200, 10, 1, 1)
        self.dt = spin(0.0005, 0.01, 0.002, 4, 0.0005, " s")
        self.swing_btn = button("▶ Free swing (live RK4)", lambda on: self._anim_toggle("swing", on),
                                checkable=True, primary=True)
        fd = QtWidgets.QFormLayout()
        fd.addRow("motor torque τ₁", self.tau1)
        fd.addRow(self.tau3_lbl, self.tau3)
        fd.addRow("viscous damping b", self.damp)
        fd.addRow(self.cb_baum)
        fd.addRow("α  (2αφ̇)", self.alpha)
        fd.addRow("β  (β²φ)", self.beta)
        fd.addRow("RK4 step dt", self.dt)
        self.omega = spin(0.5, 30, 2 * np.pi, 2, 0.5, " rad/s")
        self.omega_lbl = QtWidgets.QLabel("crank speed ω")
        self.radius = spin(0.01, 0.2, 0.06, 3, 0.005, " m")
        self.radius_lbl = QtWidgets.QLabel("circle radius")
        self.period = spin(0.3, 10, 1.5, 2, 0.1, " s")
        self.period_lbl = QtWidgets.QLabel("circle period")
        fi = QtWidgets.QFormLayout()
        fi.addRow(self.omega_lbl, self.omega)
        fi.addRow(self.radius_lbl, self.radius)
        fi.addRow(self.period_lbl, self.period)
        self.id_btn = button("Compute inverse dynamics", lambda: self._compute_id())
        self.id_play = button("▶ Play", lambda on: self._anim_toggle("id", on), checkable=True)
        self.dyn_lbl = value_label()
        dyn_page = QtWidgets.QWidget()
        dp = QtWidgets.QVBoxLayout(dyn_page)
        dp.addWidget(_wrapped("<b>Forward dynamics</b> — starts at rest from the kinematic pose"))
        dp.addLayout(fd)
        dp.addWidget(self.swing_btn)
        dp.addWidget(_wrapped("<b>Inverse dynamics</b> — four-bar: constant-speed crank; five-bar: P on a circle"))
        dp.addLayout(fi)
        dp.addLayout(hbox(self.id_btn, self.id_play))
        dp.addWidget(self.dyn_lbl)
        dp.addStretch(1)

        self.mode_tabs = QtWidgets.QTabWidget()
        self.mode_tabs.addTab(kin_page, "Kinematics")
        self.mode_tabs.addTab(dyn_page, "Dynamics")
        self.mode_tabs.currentChanged.connect(self._mode_tab_changed)

        # ------------------------------------------------------------ readouts
        self.state_lbl = value_label()
        self.check_lbl = value_label()
        self.sing_lbl = QtWidgets.QLabel()
        self.sing_lbl.setWordWrap(True)
        self.A_view = MatrixView(2, 3, 3, row_labels=["φx", "φy"], col_labels=["θ₁", "θ₂", "θ₃"])

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Mechanism (tree of branches + cut joint)", self.mech_box, self.param_stack,
                           self.grashof_lbl))
        cl.addWidget(self.mode_tabs)
        cl.addWidget(group("State && singularities", self.state_lbl, self.check_lbl, self.sing_lbl))
        cl.addWidget(group("Constraint Jacobian  A(θ) = ∂φ/∂θ", self.A_view))
        cl.addStretch(1)

        # ------------------------------------------------------------ bottom plots
        self.pk1 = plot("", "", "")
        self.pk2 = plot("", "", "")
        self.pE = plot("Mechanical energy", "J", "t [s]")
        self.pphi = plot("Constraint violation ‖φ‖", "m", "t [s]", legend=False)
        self.pphi.setLogMode(y=True)
        self.pphi.getAxis("left").enableAutoSIPrefix(False)
        self.plam = plot("Cut-pin force λ (on branch 1)", "N", "t [s]")
        self.ptau = plot("Actuator torque", "N·m", "")
        self.pidlam = plot("Pin reaction force λ", "N", "")
        self.ppow = plot("Power balance", "W", "")
        self.bottom = QtWidgets.QTabWidget()
        for name, plots in (("Kinematics plots", (self.pk1, self.pk2)), ("Free swing", (self.pE, self.pphi, self.plam)),
                            ("Inverse dynamics", (self.ptau, self.pidlam, self.ppow))):
            w = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            for p in plots:
                hl.addWidget(p)
            self.bottom.addTab(w, name)
        self._init_dyn_plots()

        self.assemble(controls, self.view, self.bottom, THEORY, controls_width=372, bottom_ratio=(5, 3))
        ctx.backend_changed.connect(self._backend_changed)
        self._mech_changed(0)

    # ================================================================ building
    def _make_mech(self):
        be = self.ctx.backend
        if self.mech_box.currentIndex() == 0:
            s = {k: v.value() for k, v in self.s4.items()}
            return cc.FourBar(s["d"], s["L1"], s["L2"], s["L3"], (s["u"], s["w"]), backend=be)
        s = {k: v.value() for k, v in self.s5.items()}
        return cc.FiveBar(s["d"], s["L1"], s["L2"], s["L3"], s["L4"], backend=be)

    @property
    def four(self) -> bool:
        return self.mech is not None and self.mech.kind == "fourbar"

    def _mech_changed(self, idx):
        self._stop_anims()
        for st in (self.param_stack, self.kin_stack):
            st.setCurrentIndex(idx)
            for i in range(st.count()):  # the stack takes the size of the *current* page only
                pol = QtWidgets.QSizePolicy.Preferred if i == idx else QtWidgets.QSizePolicy.Ignored
                st.widget(i).setSizePolicy(pol, pol)
            st.adjustSize()
        for w in (self.tau3, self.tau3_lbl, self.radius, self.radius_lbl, self.period, self.period_lbl):
            w.setVisible(idx == 1)
        for w in (self.omega, self.omega_lbl):
            w.setVisible(idx == 0)
        self.grashof_lbl.setVisible(idx == 0)
        self.theta = None
        self._rebuild()

    def _preset(self, name):
        vals = FOUR_PRESETS[name]
        for k, v in zip(("d", "L1", "L2", "L3"), vals):
            self.s4[k].blockSignals(True)
            self.s4[k].setValue(v)
            self.s4[k].blockSignals(False)
        self.s4["u"].blockSignals(True)
        self.s4["u"].setValue(vals[2] / 2)
        self.s4["u"].blockSignals(False)
        self._params_changed()

    def _params_changed(self, *_):
        self._stop_anims()
        self._rebuild()

    def _backend_changed(self, be):
        if self.mech is not None:
            self.mech.backend = be

    def _rebuild(self):
        """New mechanism instance from the spin boxes: recompute curves, frame the view, redraw."""
        old = self.theta
        self._cancel_id()
        self.mech = self._make_mech()
        m = self.mech
        self.view.remove_prefix("cc.")
        self.trace, self.sing_hist = [], []
        self.id_res = None
        self.lam_show = None
        for p in (self.ptau, self.pidlam, self.ppow):
            p.clear()
        for c in self.E_curves + self.lam_curves + [self.phi_curve]:
            c.setData([], [])
        self.A_view.set_matrix(np.zeros((2, m.n)))
        self.A_view.set_labels(rows=["φx", "φy"], cols=[f"θ{_sub(i + 1)}" for i in range(m.n)])
        if self.four:
            kind, text, (s, l, p, q) = m.grashof()
            ok = kind not in ("triple-rocker",)
            col = theme.GOOD if kind in ("crank-rocker", "double-crank") else (theme.WARN if ok else theme.BAD)
            self.grashof_lbl.setText(f"<span style='color:{col}'><b>{kind}</b></span> — {text}<br>"
                                     f"s + l = {s + l:.3f} m  vs  p + q = {p + q:.3f} m")
            self.curves = {sg: m.coupler_curve(sg) for sg in (1, -1)}
            self.sweeps = {sg: m.sweep(sg) for sg in (1, -1)}
            th1 = self.crank.value()
            if not m.crank_feasible(th1):
                ok_grid = np.where(m.crank_feasible(np.linspace(-np.pi, np.pi, 721)))[0]
                if len(ok_grid):
                    th1 = np.linspace(-np.pi, np.pi, 721)[ok_grid[len(ok_grid) // 2]]
                    self.crank.set_value(th1)
            pts = [np.array([[0, 0, 0], [m.d, 0, 0]])] + [c for v in self.curves.values() for c in v]
            pts.append(np.array([[m.L1, 0, 0], [-m.L1, 0, 0], [0, m.L1, 0], [0, -m.L1, 0]]))
            pts.append(np.array([[m.d + m.L3, 0, 0], [m.d - m.L3, 0, 0], [m.d, m.L3, 0], [m.d, -m.L3, 0]]))
            P = np.vstack(pts)
            self._init_kin_plots_four()
        else:
            self.t2 = {}
            lo, hi = m.bounds()
            P = np.array([[lo[0], lo[1], 0], [hi[0], hi[1], 0], [0, 0, 0], [m.d, 0, 0]])
            self.ws_arcs = m.workspace_arcs()
            self._init_kin_plots_five()
        lo, hi = P[:, :2].min(0), P[:, :2].max(0)
        center = (lo + hi) / 2
        ext = 0.5 * max(hi - lo) + 0.04
        self.view.frame_scene(ext, planar=True, center=(center[0], center[1], 0))
        self._draw_ground()
        if self.four:
            self._kin_update()
        else:
            ok = old is not None and len(old) == 4 and self._set_theta5(old, check=True)
            if not ok:
                self._p_changed()
        self._redraw()

    # ================================================================ kinematics: four-bar
    def _sigma_changed(self, idx):
        self.sigma = 1 if idx == 0 else -1
        self.trace = []
        if self.four:
            self._kin_update()
            self._update_kin_plots_four()

    def _kin_update(self):
        if not self.four:
            return
        m = self.mech
        th1 = self.crank.value()
        th = m.fk_closed_form([th1], self.sigma)
        if th is None:
            self.sing_lbl.setText(f"<span style='color:{theme.BAD}'>⚠ The loop cannot be closed at "
                                  f"θ₁ = {np.degrees(th1):.1f}°: |AD| is outside [|L₂−L₃|, L₂+L₃].</span>")
            return
        prev = self.theta if (self.theta is not None and len(self.theta) == m.n) else th + 0.05
        th_nr, ok, hist = m.fk_newton([th1], prev)
        self.theta = th
        d = float(np.max(np.abs(cc.wrap(th_nr - th))))
        if ok and d < 1e-8:
            self.check_lbl.setText(f"✓ closed form = Newton–Raphson (from previous pose, {len(hist) - 1} its): "
                                   f"max|Δθ| = {d:.1e}")
            self.check_lbl.setStyleSheet(f"color:{theme.GOOD}")
        else:
            self.check_lbl.setText("Newton–Raphson from the previous pose converged to the other assembly mode"
                                   if ok else "Newton–Raphson did not converge from the previous pose")
            self.check_lbl.setStyleSheet(f"color:{theme.WARN}")
        self._redraw()

    # ================================================================ kinematics: five-bar
    def _mode_changed(self, idx):
        self.elbows = cc.WORKING_MODES[idx][0]
        self.trace = []
        if not self.four and self.mech is not None:
            self._p_changed()
            self._update_type2()

    def _set_theta5(self, th, check=False) -> bool:
        m = self.mech
        phi = m.constraint(th)[0]
        if check and np.linalg.norm(phi) > 1e-9:
            return False
        self.theta = np.asarray(th, float)
        self.sigma = m.assembly_mode(th)
        el = m.elbows(th)
        if el != self.elbows:
            self.elbows = el
            idx = [e for e, _ in cc.WORKING_MODES].index(el)
            self.mode_box.blockSignals(True)
            self.mode_box.setCurrentIndex(idx)
            self.mode_box.blockSignals(False)
            self._update_type2()
        P = m.tip(th)
        self.px.set_value(P[0])
        self.py.set_value(P[1])
        self.a1.set_value(th[0])
        self.a3.set_value(th[2])
        return True

    def _p_changed(self):
        if self.four or self.mech is None:
            return
        m = self.mech
        P = np.array([self.px.value(), self.py.value()])
        th = m.ik(P, self.elbows)
        if th is None:
            self.sing_lbl.setText(f"<span style='color:{theme.BAD}'>⚠ P is outside the workspace "
                                  "(not reachable by both 2R branches).</span>")
            return
        self.theta = th
        self.sigma = m.assembly_mode(th)
        self.a1.set_value(th[0])
        self.a3.set_value(th[2])
        cf = m.fk_closed_form(th[m.actuated], self.sigma)
        nr, ok, hist = m.fk_newton(th[m.actuated], th + 0.05)
        e1 = np.linalg.norm(m.tip(cf)[:2] - P) if cf is not None else np.inf
        e2 = float(np.max(np.abs(cc.wrap(nr - th))))
        self.check_lbl.setText(f"✓ FK(IK(P)): |ΔP| = {e1:.1e} m (closed form σ = {self.sigma:+d});  "
                               f"Newton–Raphson ({len(hist) - 1} its) max|Δθ| = {e2:.1e}")
        self.check_lbl.setStyleSheet(f"color:{theme.GOOD if e1 < 1e-9 and e2 < 1e-8 else theme.WARN}")
        self._redraw()

    def _act_changed(self):
        if self.four or self.mech is None:
            return
        m = self.mech
        th_a = np.array([self.a1.value(), self.a3.value()])
        th = m.fk_closed_form(th_a, self.sigma)
        if th is None:
            self.sing_lbl.setText(f"<span style='color:{theme.BAD}'>⚠ The loop cannot be closed: "
                                  "|AC| > L₂ + L₃ or < |L₂ − L₃|.</span>")
            return
        self._set_theta5(th)
        self.check_lbl.setText(f"closed-form FK, assembly mode σ = {self.sigma:+d}")
        self.check_lbl.setStyleSheet(f"color:{theme.GOOD}")
        self._redraw()

    def _flip_sigma5(self):
        if self.four or self.theta is None:
            return
        th = self.mech.fk_closed_form(self.theta[self.mech.actuated], -self.sigma)
        if th is not None:
            self._set_theta5(th)
            self.trace = []
            self._redraw()

    def _update_type2(self):
        m = self.mech
        if self.four or m is None:
            return
        if self.elbows not in self.t2:
            xs, ys, Z = m.type2_field(self.elbows, 200)
            self.t2[self.elbows] = (xs, ys, Z, m.type2_locus(self.elbows, 200))
        xs, ys, Z, lines = self.t2[self.elbows]
        self.view.remove_prefix("cc.t2")
        if self.cb_t2.isChecked():
            for i, L in enumerate(lines):
                self.view.set_polyline(f"cc.t2{i}", L + [0, 0, -0.018], (1.0, 0.3, 0.3, 0.95), 2.5)
        self.img.setImage(np.abs(Z).T, levels=(0, 1), autoLevels=False)
        self.img.setRect(QtCore.QRectF(xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0]))
        for it in self.img_lines:
            self.pk2.removeItem(it)
        self.img_lines = []
        for L in lines:
            it = pg.PlotDataItem(L[:, 0], L[:, 1], pen=pg.mkPen((255, 80, 80), width=2))
            self.pk2.addItem(it)
            self.img_lines.append(it)

    def _mode_tab_changed(self, idx):
        self._stop_anims()
        if idx == 0:
            self.lam_show = None
            self.bottom.setCurrentIndex(0)
            if self.four:
                self._kin_update()
            elif self.theta is not None:
                self._set_theta5(self.theta)
        self._redraw()

    # ================================================================ hooks
    def on_show(self):
        self._redraw()

    def on_hide(self):
        self._stop_anims()
        self._cancel_id()

    def shutdown(self):
        """Application quit: stop the animation, cancel the inverse-dynamics job (brief wait only)."""
        self._stop_anims()
        self.id_jobs.shutdown(500)
        parent = getattr(super(), "shutdown", None)
        if parent is not None:
            parent()


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
