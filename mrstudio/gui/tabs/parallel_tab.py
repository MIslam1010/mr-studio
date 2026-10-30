"""Tab 10 -- parallel robots: the planar 3xRPR (MR 7.1.1) and the Delta robot (MR Ex. 2.7, Fig. 7.1b, Ex. 7.7)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import delta as dl
from ... import parallel_planar as pp
from .. import theme
from ..base import BaseTab
from ..viewer import SceneView
from ..widgets import (MatrixView, SliderRow, button, checkbox, combo, group, hbox, pen, plot, spin,
                       value_label)

THEORY = r"""
<h2>Parallel robots: 3×RPR and Delta</h2>
<span class="ref">Modern Robotics 7.1.1 (3×RPR), 7.2 (differential kinematics), 7.3 (singularities);
Delta robot: Example 2.7 (Fig. 2.8), Fig. 7.1(b), Exercise 7.7</span>
<p>For parallel mechanisms the inverse kinematics is usually easy and the forward kinematics hard —
the opposite of open chains (7.1).</p>
<h3>3×RPR planar mechanism (7.1.1)</h3>
<p>Three legs R–<b>P</b>–R join base points $a_i$ (in {s}) to platform points $b_i$ (in {b}); the
prismatic joints are actuated. With $p$ and $\phi$ the platform position and angle:</p>
$$d_i=p+R(\phi)\,b_i-a_i\quad(7.1)$$
$$s_i^{2}=(p_x+b_{ix}\cos\phi-b_{iy}\sin\phi-a_{ix})^{2}$$
$$\qquad+(p_y+b_{ix}\sin\phi+b_{iy}\cos\phi-a_{iy})^{2}\quad(7.2)$$
<p><b>Inverse kinematics</b>: evaluate (7.2). <b>Forward kinematics</b>: with $c_i=R(\phi)b_i-a_i$, the
differences of the squared equations are linear in $p$:</p>
$$2\,(c_i-c_1)^{T}p=s_i^{2}-s_1^{2}-|c_i|^{2}+|c_1|^{2},\qquad i=2,3$$
<p>so $p=D(\phi)^{-1}r(\phi)$ and equation 1 leaves one scalar equation; multiplied by $\det D^{2}$ it is
pole-free:</p>
$$g(\phi)=|\mathrm{adj}(D)\,r+\det(D)\,c_1|^{2}-s_1^{2}\det(D)^{2}=0$$
<p>Its real roots (bottom plot) are bracketed on a grid, refined (Brent) and polished by Newton–Raphson.
With $t=\tan(\phi/2)$ the system reduces to a sextic in $t$: <b>at most six</b> assembly modes (ghosts).
Preset “6 real FK solutions” shows a case where all six are real.</p>
<p><b>Velocity kinematics and statics (7.2)</b>, planar twist $\mathcal{V}_s=(\omega,v_x,v_y)$, $x_i=p+Rb_i$:</p>
$$\dot{s}_i=(x_i\times\hat{u}_i)_z\,\omega+\hat{u}_i^{T}v\quad\Rightarrow\quad\dot{s}=J\,\mathcal{V}_s,\qquad\mathcal{F}_s=J^{T}f$$
<p><b>Actuator singularity (7.3)</b>: $\det J=0$ ⇔ the three leg lines meet in one point (or are
parallel): the platform can rotate infinitesimally about that point with all legs locked (cf. Fig. 7.3a).
The red locus is $\det J=0$ at the current orientation $\phi$; the blue cloud is the
constant-orientation workspace $s_{min}\leq s_i\leq s_{max}$.</p>
<h3>Delta robot (Clavel)</h3>
<p>Three R–(SS)$^{2}$ legs: an actuated revolute at the base, an upper arm, and a parallelogram forearm.
Grübler (Example 2.7): $N=17$, $J=21$ (9 R, 12 S):</p>
$$dof=6(17-1-21)+9\cdot1+12\cdot3=15$$
<p>of which 12 are forearm torsions; the platform stays parallel to the base — a pure $x$–$y$–$z$
translator. Elbow and platform joint of leg $i$ ($\alpha_i=0,120,240°$):</p>
$$A_i=B_i+L(\cos\theta_i\,\hat{e}_{r,i}-\sin\theta_i\,\hat{z}),\qquad P_i=p+r\,\hat{e}_{r,i},\qquad|P_i-A_i|=l$$
<p><b>IK (closed form)</b>: with $a=p\cdot\hat{e}_{r}+r-R$, $b=p\cdot\hat{e}_{t}$, $c=p_z$:</p>
$$E\cos\theta+F\sin\theta+G=0,\quad E=-2aL,\ F=2cL,\ G=a^{2}+b^{2}+c^{2}+L^{2}-l^{2}$$
$$t=\tan\frac{\theta}{2}:\qquad(G-E)\,t^{2}+2F\,t+(G+E)=0$$
<p>Two roots per leg (elbow out / in): $2^{3}=8$ IK branches; the app uses elbow-out.
<b>FK (closed form)</b>: $p$ is on the three spheres $|p-C_i|=l$, $C_i=A_i-r\hat{e}_{r,i}$ —
trilateration gives two mirror solutions (the upper one is the ghost).</p>
<p><b>Jacobian</b> (Ex. 7.7c) from $\frac{d}{dt}|P_i-A_i|^{2}=0$ with $n_i=P_i-A_i$:</p>
$$J_x\dot{p}=J_{\theta}\dot{\theta},\quad J_x=\left[n_i^{T}\right],\quad J_{\theta}=\mathrm{diag}\left(n_i^{T}\frac{\partial A_i}{\partial\theta_i}\right),\quad\dot{p}=J_x^{-1}J_{\theta}\dot{\theta}$$
<p>Statics (power balance): motor torques holding a payload force $F$ are $\tau=-J^{T}F$ (massless legs).
<b>Singularities (7.3)</b>: $\det J_x=0$ — actuator singularity (Ex. 7.7d: forearms coplanar/parallel, the
platform moves with locked motors); $n_i^{T}\partial A_i/\partial\theta_i=0$ — arm and forearm aligned:
end-effector singularity = workspace boundary. The cloud is coloured by $\kappa(J)$.</p>
<p><b>Pick-and-place</b>: straight segments up–across–down with quintic time scaling (MR 9.2); the motor
angles follow from the closed-form IK and the rates from $\dot{\theta}=J^{-1}\dot{p}$.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 3 §5.2–5.4, pp. 132–143', 'parallel manipulators; planar Stewart platform (Example 3.18)'),
    ('lavalle', '§7.4, pp. 337–346', 'parallel mechanisms in closed-chain planning (p. 338)'),
    ('lavalle', '§4.4, from p. 167', 'loop-closure constraints'),
])


RPR_COLS = [theme.SERIES[0], theme.SERIES[1], theme.SERIES[2]]
GHOST_HUES = 6


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _page(*widgets):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    for x in widgets:
        if isinstance(x, QtWidgets.QLayout):
            lay.addLayout(x)
        else:
            lay.addWidget(x)
    return w


def _fit_stack(stack: QtWidgets.QStackedWidget, idx: int):
    stack.setCurrentIndex(idx)
    for i in range(stack.count()):
        pol = QtWidgets.QSizePolicy.Preferred if i == idx else QtWidgets.QSizePolicy.Ignored
        stack.widget(i).setSizePolicy(pol, pol)
    stack.adjustSize()


class ParallelRobotsTab(BaseTab):
    title = "⑨ Parallel"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.rpr = pp.ThreeRPR()
        self.dlt = dl.DeltaRobot()
        self.view = SceneView()
        self.q = self.rpr.q_home.copy()
        self.s = self.rpr.ik(self.q)
        self.rpr_sols: list = []
        self.p = self.dlt.p_home.copy()
        self.th = self.dlt.ik(self.p)
        self.dl_sols: list = []
        self.anim = None
        self._t0 = time.time()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._ws_cache = None

        self.robot_box = combo(["3×RPR planar parallel robot (MR 7.1.1)", "Delta robot (Clavel; MR Ex. 2.7, 7.7)"])
        self.robot_box.currentIndexChanged.connect(self._robot_changed_idx)

        # ------------------------------------------------------------ 3xRPR controls
        self.r_mode = combo(["Inverse kinematics: pose → leg lengths", "Forward kinematics: leg lengths → pose"])
        self.r_mode.currentIndexChanged.connect(self._r_mode_changed)
        q0 = self.rpr.q_home
        self.r_pose = [SliderRow("pₓ", q0[0] - 0.35, q0[0] + 0.35, q0[0], 3, " m"),
                       SliderRow("pᵧ", q0[1] - 0.25, q0[1] + 0.55, q0[1], 3, " m"),
                       SliderRow("φ", -np.pi, np.pi, q0[2], 1, "°", 180 / np.pi)]
        for r in self.r_pose:
            r.valueChanged.connect(lambda _v: self._r_pose_changed())
        self.r_legs = []
        for i in range(3):
            r = SliderRow(f"s{_sub(i + 1)}", self.rpr.s_min, self.rpr.s_max, self.s[i], 3, " m")
            r.valueChanged.connect(lambda _v: self._r_legs_changed())
            self.r_legs.append(r)
        self.r_stack = QtWidgets.QStackedWidget()
        self.r_stack.addWidget(_page(*self.r_pose))
        self.r_stack.addWidget(_page(*self.r_legs))
        self.r_home = button("Home", self._r_home)
        self.r_six = button("6 real FK solutions", self._r_six,
                            tooltip="leg lengths for which all six roots of the sextic are real")
        self.r_anim = button("▶ Circle", lambda on: self._anim_toggle("rpr", on), checkable=True, primary=True)
        self.r_ghost = checkbox("Ghosts: all other FK solutions", self._redraw, True)
        self.r_ws = checkbox("Workspace at this φ (blue)", self._r_ws_toggled, True)
        self.r_sing = checkbox("Actuator-singularity locus det J = 0 (red)", self._r_ws_toggled, True)
        self.r_fk_lbl = value_label()
        rpr_page = _page(self.r_mode, self.r_stack, hbox(self.r_home, self.r_six, self.r_anim),
                         self.r_ghost, self.r_ws, self.r_sing, self.r_fk_lbl)

        # ------------------------------------------------------------ Delta controls
        self.d_mode = combo(["Inverse kinematics: position → motor angles", "Forward kinematics: motor angles → position"])
        self.d_mode.currentIndexChanged.connect(self._d_mode_changed)
        p0 = self.dlt.p_home
        self.d_pos = [SliderRow("x", -0.25, 0.25, 0.0, 3, " m"), SliderRow("y", -0.25, 0.25, 0.0, 3, " m"),
                      SliderRow("z", -0.60, -0.22, p0[2], 3, " m")]
        for r in self.d_pos:
            r.valueChanged.connect(lambda _v: self._d_pos_changed())
        self.d_ang = []
        for i in range(3):
            r = SliderRow(f"θ{_sub(i + 1)}", self.dlt.th_min, self.dlt.th_max, self.th[i], 1, "°", 180 / np.pi)
            r.valueChanged.connect(lambda _v: self._d_ang_changed())
            self.d_ang.append(r)
        self.d_stack = QtWidgets.QStackedWidget()
        self.d_stack.addWidget(_page(*self.d_pos))
        self.d_stack.addWidget(_page(*self.d_ang))
        self.d_mass = spin(0, 5, 1.0, 2, 0.1, " kg")
        self.d_mass.valueChanged.connect(lambda *_: (self._d_prepare_path(), self._update()))
        self.d_home = button("Home", self._d_home)
        self.d_anim = button("▶ Pick && place", lambda on: self._anim_toggle("delta", on), checkable=True, primary=True)
        self.d_ghost = checkbox("Ghost: second FK solution (mirror)", self._redraw, True)
        self.d_ws = checkbox("Workspace cloud (colour = κ(J))", self._redraw, False)
        self.d_trace = checkbox("Pick-and-place path", self._redraw, True)
        self.d_fk_lbl = value_label()
        fl = QtWidgets.QFormLayout()
        fl.addRow("payload (statics τ = −Jᵀ F)", self.d_mass)
        delta_page = _page(self.d_mode, self.d_stack, fl, hbox(self.d_home, self.d_anim),
                           self.d_ghost, self.d_ws, self.d_trace, self.d_fk_lbl)

        self.page_stack = QtWidgets.QStackedWidget()
        self.page_stack.addWidget(rpr_page)
        self.page_stack.addWidget(delta_page)

        self.state_lbl = value_label()
        self.sing_lbl = QtWidgets.QLabel()
        self.sing_lbl.setWordWrap(True)
        self.J_view = MatrixView(3, 3, 3, row_labels=["1", "2", "3"], col_labels=["1", "2", "3"])
        self.J_title = QtWidgets.QLabel()
        self.J_title.setWordWrap(True)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Mechanism", self.robot_box))
        cl.addWidget(group("Input", self.page_stack))
        cl.addWidget(group("State && singularity", self.state_lbl, self.sing_lbl))
        cl.addWidget(group("Jacobian", self.J_title, self.J_view))
        cl.addStretch(1)

        # ------------------------------------------------------------ bottom plots
        self.pg_ = plot("FK equation g(φ) = 0", "sign(g)·|g/max g|^⅓", "φ [°]")
        self.pS = plot("Leg lengths along the circle", "s [m]", "t [s]")
        self.pS.getAxis("left").enableAutoSIPrefix(False)
        self.pTh = plot("Motor angles along the pick-and-place cycle", "θ [°]", "t [s]")
        self.pTau = plot("Motor torques holding the payload (statics)", "τ [N·m]", "t [s]")
        self.bottom = QtWidgets.QStackedWidget()
        for plots in ((self.pg_, self.pS), (self.pTh, self.pTau)):
            w = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            for p in plots:
                hl.addWidget(p)
            self.bottom.addWidget(w)
        self._g_curve = self.pg_.plot([], [], pen=pen(4, 2), name="g(φ)")
        self.pg_.addLine(y=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self._g_roots = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(theme.WARN), pen=pg.mkPen(None))
        self.pg_.addItem(self._g_roots)
        self._g_cur = self.pg_.addLine(x=0, pen=pg.mkPen(theme.GOOD, width=1.5))

        self.assemble(controls, self.view, self.bottom, THEORY, controls_width=372, bottom_ratio=(5, 2))
        self._r_prepare_path()
        self._d_prepare_path()
        _fit_stack(self.r_stack, 0)
        _fit_stack(self.d_stack, 0)
        self._robot_changed_idx(0)

    # ================================================================ selection
    @property
    def is_rpr(self) -> bool:
        return self.robot_box.currentIndex() == 0

    def _robot_changed_idx(self, idx):
        self._stop_anim()
        _fit_stack(self.page_stack, idx)
        self.bottom.setCurrentIndex(idx)
        self.view.remove_prefix("pr.")
        self.view.remove_prefix("dl.")
        self.view.grid.resetTransform()
        if idx == 0:
            lo, hi = self.rpr.bounds()
            c = (lo + hi) / 2
            self.view.frame_scene(0.62 * float(np.max(hi - lo)), planar=True, center=(c[0], c[1], 0))
            self._ws_cache = None
            self.J_title.setText("ṡ = J 𝒱ₛ,  rows [(xᵢ × ûᵢ)_z, ûᵢᵀ]  (columns ω, vₓ, vᵧ)")
            self.J_view.set_labels([f"leg {i + 1}" for i in range(3)], ["ω", "vₓ", "vᵧ"])
        else:
            self.view.frame_scene(0.55, planar=False, center=(0, 0, -0.30))
            self.view.grid.translate(0, 0, -0.66)
            self.J_title.setText("ṗ = J θ̇,  J = Jₓ⁻¹ J_θ  (columns θ̇₁, θ̇₂, θ̇₃)")
            self.J_view.set_labels(["ẋ", "ẏ", "ż"], ["θ₁", "θ₂", "θ₃"])
        self.J_view.setFixedHeight(3 * 34 + 36)
        self._update()

    # ================================================================ 3xRPR inputs
    def _r_mode_changed(self, idx):
        self._stop_anim()
        _fit_stack(self.r_stack, idx)
        self.r_fk_lbl.setText("")
        self._r_sync()

    def _r_sync(self):
        for r, x in zip(self.r_pose, self.q):
            r.set_value(x)
        for r, x in zip(self.r_legs, self.s):
            r.set_value(x)

    def _r_pose_changed(self):
        q = np.array([r.value() for r in self.r_pose])
        if abs(q[2] - self.q[2]) > 1e-12:
            self._ws_cache = None
        self.q = q
        self.s = self.rpr.ik(q)
        for r, x in zip(self.r_legs, self.s):
            r.set_value(x)
        self.r_fk_lbl.setText("IK (closed form, Eq. 7.2): s_i = |p + R(φ) b_i − a_i|")
        self.r_fk_lbl.setStyleSheet("")
        self._update()

    def _r_legs_changed(self):
        s = np.array([r.value() for r in self.r_legs])
        self._r_set_legs(s)

    def _r_set_legs(self, s):
        t0 = time.perf_counter()
        q, res = self.rpr.fk_nearest(s, self.q)
        ms = (time.perf_counter() - t0) * 1e3
        if q is None:
            self.r_fk_lbl.setText("⚠ No real solution of g(φ) = 0: these leg lengths cannot be assembled — pose kept.")
            self.r_fk_lbl.setStyleSheet(f"color:{theme.BAD}")
            self._update(fk_res=res, valid=False)
            return
        if abs(q[2] - self.q[2]) > 1e-12:
            self._ws_cache = None
        self.q, self.s = q, s
        for r, x in zip(self.r_pose, q):
            r.set_value(x)
        self.r_fk_lbl.setText(f"✓ {len(res.q)} real FK solution(s) (≤ 6) in {ms:.0f} ms; showing the one nearest "
                              "to the previous pose, the others as ghosts.")
        self.r_fk_lbl.setStyleSheet(f"color:{theme.GOOD}")
        self._update(fk_res=res)

    def _r_home(self):
        self._stop_anim()
        self.q = self.rpr.q_home.copy()
        self.s = self.rpr.ik(self.q)
        self._ws_cache = None
        self._r_sync()
        self.r_fk_lbl.setText("")
        self._update()

    def _r_six(self):
        self._stop_anim()
        if self.r_mode.currentIndex() != 1:
            self.r_mode.setCurrentIndex(1)
        s = pp.SIX_SOLUTION_LEGS.copy()
        for r, x in zip(self.r_legs, s):
            r.set_value(x)
        self._r_set_legs(s)

    def _r_ws_toggled(self, *_):
        self._ws_cache = None
        self._redraw()

    # ================================================================ Delta inputs
    def _d_mode_changed(self, idx):
        self._stop_anim()
        _fit_stack(self.d_stack, idx)
        self.d_fk_lbl.setText("")
        self._d_sync()

    def _d_sync(self):
        for r, x in zip(self.d_pos, self.p):
            r.set_value(x)
        for r, x in zip(self.d_ang, self.th):
            r.set_value(x)

    def _d_pos_changed(self):
        p = np.array([r.value() for r in self.d_pos])
        th = self.dlt.ik(p)
        if th is None:
            self.d_fk_lbl.setText("⚠ Out of reach: |P_i − A_i| = l has no solution for some leg — position kept.")
            self.d_fk_lbl.setStyleSheet(f"color:{theme.BAD}")
            return
        self.p, self.th = p, th
        for r, x in zip(self.d_ang, th):
            r.set_value(x)
        lim = np.any(th < self.dlt.th_min) or np.any(th > self.dlt.th_max)
        self.d_fk_lbl.setText("IK (closed form, elbow-out): tan-half-angle quadratic per leg" +
                              ("\n⚠ a motor angle is outside its range" if lim else ""))
        self.d_fk_lbl.setStyleSheet(f"color:{theme.WARN}" if lim else "")
        self._update()

    def _d_ang_changed(self):
        th = np.array([r.value() for r in self.d_ang])
        sols = self.dlt.fk_all(th)
        if not sols:
            self.d_fk_lbl.setText("⚠ The three spheres |p − C_i| = l do not intersect — no assembly; pose kept.")
            self.d_fk_lbl.setStyleSheet(f"color:{theme.BAD}")
            return
        self.th, self.p = th, sols[0]
        for r, x in zip(self.d_pos, self.p):
            r.set_value(x)
        self.d_fk_lbl.setText(f"✓ FK by sphere trilateration: {len(sols)} solutions (lower one used, upper = ghost)")
        self.d_fk_lbl.setStyleSheet(f"color:{theme.GOOD}")
        self._update()

    def _d_home(self):
        self._stop_anim()
        self.p = self.dlt.p_home.copy()
        self.th = self.dlt.ik(self.p)
        self._d_sync()
        self.d_fk_lbl.setText("")
        self._update()

    # ================================================================ paths
    def _r_prepare_path(self):
        t, Q = self.rpr.circle_path()
        self.r_t, self.r_Q = t, Q
        S = np.array([self.rpr.ik(q) for q in Q])
        self.pS.clear()
        for i in range(3):
            self.pS.plot(t, S[:, i], pen=pen(i, 2), name=f"s{_sub(i + 1)}")
        for y in (self.rpr.s_min, self.rpr.s_max):
            self.pS.addLine(y=y, pen=pg.mkPen(theme.BAD, style=QtCore.Qt.DashLine))
        self.r_cursor = self.pS.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1))

    def _d_prepare_path(self):
        d = self.dlt
        t, P, V = d.pick_place_path()
        TH = np.array([d.ik(p) for p in P])
        F = np.array([0.0, 0.0, -self.d_mass.value() * dl.G_ACC])
        TAU = np.array([d.static_torques(th, p, F) for th, p in zip(TH, P)])
        self.d_t, self.d_P, self.d_TH = t, P, TH
        self.pTh.clear()
        self.pTau.clear()
        for i in range(3):
            self.pTh.plot(t, np.degrees(TH[:, i]), pen=pen(i, 2), name=f"θ{_sub(i + 1)}")
            self.pTau.plot(t, TAU[:, i], pen=pen(i, 2), name=f"τ{_sub(i + 1)}")
        self.pTau.addLine(y=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.d_cursors = [p.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1)) for p in (self.pTh, self.pTau)]

    # ================================================================ animation
    def _anim_toggle(self, kind, on):
        btn = self.r_anim if kind == "rpr" else self.d_anim
        btn.setText("■ Stop" if on else ("▶ Circle" if kind == "rpr" else "▶ Pick && place"))
        if on:
            mode = self.r_mode if kind == "rpr" else self.d_mode
            if mode.currentIndex() != 0:
                mode.setCurrentIndex(0)
            self.anim = kind
            self._t0 = time.time()
            self.timer.start(30)
        else:
            self.anim = None
            self.timer.stop()

    def _stop_anim(self):
        for b in (getattr(self, "r_anim", None), getattr(self, "d_anim", None)):
            if b is not None and b.isChecked():
                b.setChecked(False)

    def _tick(self):
        t_now = time.time() - self._t0
        if self.anim == "rpr":
            t = t_now % self.r_t[-1]
            k = min(int(np.searchsorted(self.r_t, t)), len(self.r_t) - 1)
            self.q = self.r_Q[k].copy()
            self.s = self.rpr.ik(self.q)
            self._ws_cache = None
            self._r_sync()
            self.r_cursor.setValue(self.r_t[k])
        elif self.anim == "delta":
            t = t_now % self.d_t[-1]
            k = min(int(np.searchsorted(self.d_t, t)), len(self.d_t) - 1)
            self.p, self.th = self.d_P[k].copy(), self.d_TH[k].copy()
            self._d_sync()
            for c in self.d_cursors:
                c.setValue(self.d_t[k])
        self._update()

    # ================================================================ computation + drawing
    def _update(self, fk_res=None, valid=True):
        if self.is_rpr:
            self._update_rpr(fk_res, valid)
        else:
            self._update_delta()
        self._redraw()

    def _update_rpr(self, fk_res, valid):
        rob = self.rpr
        res = fk_res if fk_res is not None else rob.fk_all(self.s)
        self.rpr_sols = [q for q in res.q if np.abs(np.r_[q[:2] - self.q[:2],
                                                        np.angle(np.exp(1j * (q[2] - self.q[2])))]).max() > 1e-6]
        self._g_curve.setData(np.degrees(res.g_phi), np.cbrt(res.g_val))
        self._g_roots.setData([np.degrees(q[2]) for q in res.q], [0.0] * len(res.q))
        self._g_cur.setValue(np.degrees(self.q[2]))
        self.pg_.setTitle(f"FK equation g(φ) = 0: {len(res.q)} real root(s)", color=theme.TEXT, size="10pt")
        det, kappa = rob.singularity(self.q)
        det0 = rob.singularity(rob.q_home)[0]
        rel = det / det0
        lim = np.any(self.s < rob.s_min - 1e-9) or np.any(self.s > rob.s_max + 1e-9)
        self.state_lbl.setText(f"p = ({self.q[0]:+.4f}, {self.q[1]:+.4f}) m,  φ = {np.degrees(self.q[2]):+.1f}°\n"
                               f"s = ({self.s[0]:.4f}, {self.s[1]:.4f}, {self.s[2]:.4f}) m"
                               + ("  ⚠ stroke limit" if lim else "") +
                               f"\ndet J_n = {det:+.3f} ({rel:+.2f} × home),  κ(J_n) = {kappa:.1f}")
        self._sing_text(abs(rel), kappa, "the three leg lines are (nearly) concurrent")
        self.J_view.set_matrix(rob.jacobian(self.q))

    def _update_delta(self):
        d = self.dlt
        sols = d.fk_all(self.th)
        self.dl_sols = sols[1:] if len(sols) == 2 else []
        dx, t1, kappa = d.singularity(self.th, self.p)
        dx0 = d.singularity(d.ik(d.p_home), d.p_home)[0]
        F = np.array([0.0, 0.0, -self.d_mass.value() * dl.G_ACC])
        tau = d.static_torques(self.th, self.p, F)
        th_d = np.degrees(self.th)
        self.state_lbl.setText(f"p = ({self.p[0]:+.4f}, {self.p[1]:+.4f}, {self.p[2]:+.4f}) m\n"
                               f"θ = ({th_d[0]:+.1f}°, {th_d[1]:+.1f}°, {th_d[2]:+.1f}°)\n"
                               f"τ = ({tau[0]:+.3f}, {tau[1]:+.3f}, {tau[2]:+.3f}) N·m  (payload {self.d_mass.value():.1f} kg)\n"
                               f"det Jₓ (unit nᵢ) = {dx:+.3f} ({dx / dx0:.2f} × home),  κ(J) = {kappa:.2f}\n"
                               f"min |cos∠(nᵢ, ∂Aᵢ/∂θᵢ)| = {t1:.3f}  (0 = arm ∥ forearm)")
        if t1 < 0.08:
            self.sing_lbl.setText(f"<span style='color:{theme.BAD}'>● End-effector singularity: an arm and its "
                                  "forearm are aligned (workspace boundary)</span>")
        else:
            self._sing_text(abs(dx / dx0), kappa, "forearms nearly coplanar / parallel", kappa_bad=8, kappa_warn=4)
        self.J_view.set_matrix(d.jacobian(self.th, self.p))

    def _sing_text(self, rel, kappa, why, kappa_bad=60, kappa_warn=20):
        if rel < 0.1 or kappa > kappa_bad:
            txt, col = f"⚠ Close to an actuator singularity (det → 0): {why}", theme.BAD
        elif rel < 0.35 or kappa > kappa_warn:
            txt, col = "Poorly conditioned configuration (κ large)", theme.WARN
        else:
            txt, col = "Well-conditioned configuration", theme.GOOD
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")

    def _redraw(self, *_):
        if self.is_rpr:
            self._redraw_rpr()
        else:
            self._redraw_delta()

    # ---------------------------------------------------------------- 3xRPR drawing
    def _draw_rpr(self, q, prefix, ghost_col=None):
        rob, v = self.rpr, self.view
        x, s, u = rob.legs(q)
        z = 0.0 if ghost_col is None else -0.01
        X = np.c_[x, np.full(3, z)]
        A = np.c_[rob.a, np.full(3, z)]
        if ghost_col is not None:
            v.set_polyline(f"{prefix}.plat", np.vstack([X, X[:1]]), ghost_col, 2.5)
            legs = np.empty((6, 3))
            legs[0::2], legs[1::2] = A, X
            v.set_polyline(f"{prefix}.legs", legs, (*ghost_col[:3], 0.45), 1.5, mode="lines")
            return
        tri = np.vstack([X, X[:1]])
        v.set_polyline(f"{prefix}.plat", tri, (0.75, 0.9, 1.0, 1.0), 3)
        c = X.mean(0)
        for i in range(3):
            v.set_tube(f"{prefix}.pe{i}", c, X[i], 0.010, (0.30, 0.62, 0.78, 1.0))
        p3 = np.r_[q[:2], 0.0]
        lim_hi = rob.s_max
        for i in range(3):
            ui = np.r_[u[i], 0.0]
            sleeve = min(0.32, 0.55 * lim_hi)
            col = RPR_COLS[i]
            v.set_tube(f"{prefix}.cyl{i}", A[i] + 0.02 * ui, A[i] + sleeve * ui, 0.022, col)
            v.set_tube(f"{prefix}.rod{i}", A[i] + 0.25 * ui, X[i], 0.009, (0.82, 0.84, 0.88, 1.0))
            v.set_disk(f"{prefix}.aj{i}", A[i], [0, 0, 1], 0.026, 0.03, theme.JOINT)
            v.set_disk(f"{prefix}.bj{i}", X[i], [0, 0, 1], 0.018, 0.03, theme.JOINT)
            v.set_label(f"{prefix}.la{i}", A[i] + [0.03, -0.05, 0.03], f"a{_sub(i + 1)}", theme.MUTED)
            v.set_label(f"{prefix}.lb{i}", X[i] + [0.02, 0.025, 0.03], f"b{_sub(i + 1)}", theme.TEXT)
        v.set_triad(f"{prefix}.fb", _se2(q), 0.12, "{b}")

    def _redraw_rpr(self):
        rob, v = self.rpr, self.view
        # base
        for i, a in enumerate(rob.a):
            h, w = 0.05, 0.035
            G = np.r_[a, -0.02]
            v.set_polyline(f"pr.gnd{i}", np.array([G, G + [-w, -h, 0], G + [w, -h, 0], G]), (0.7, 0.74, 0.8, 1.0), 2)
        v.set_triad("pr.fs", np.eye(4), 0.15, "{s}")
        self._draw_rpr(self.q, "pr.m")
        v.remove_prefix("pr.ghost")
        if self.r_ghost.isChecked():
            for k, q in enumerate(self.rpr_sols):
                col = (*pg.intColor(k, hues=GHOST_HUES).getRgbF()[:3], 0.7)
                self._draw_rpr(q, f"pr.ghost{k}", ghost_col=col)
        if self._ws_cache is None:
            X, Y, inside, _ = rob.workspace_grid(self.q[2], n=90)
            ws = np.c_[X[inside], Y[inside], np.full(int(inside.sum()), -0.03)]
            sing = rob.singular_locus(self.q[2], n=140)
            self._ws_cache = (ws, np.c_[sing, np.full(len(sing), -0.025)])
        ws, sing = self._ws_cache
        if self.r_ws.isChecked() and len(ws):
            v.set_points("pr.ws", ws, (0.30, 0.55, 1.0, 0.22), 4)
        else:
            v.remove("pr.ws")
        if self.r_sing.isChecked() and len(sing):
            v.set_points("pr.sing", sing, (1.0, 0.35, 0.35, 1.0), 4)
        else:
            v.remove("pr.sing")

    # ---------------------------------------------------------------- Delta drawing
    def _redraw_delta(self):
        d, v = self.dlt, self.view
        th, p = self.th, self.p
        A = d.elbows(th)
        P = d.platform_points(p)
        # base plate and motors
        v.set_disk("dl.base", [0, 0, 0.03], [0, 0, 1], d.R * 1.45, 0.025, (0.36, 0.41, 0.48, 1.0))
        for i in range(3):
            B, et = d.B[i], d.et[i]
            v.set_tube(f"dl.motor{i}", B - 0.045 * et, B + 0.045 * et, 0.03, (0.25, 0.28, 0.33, 1.0))
            v.set_tube(f"dl.arm{i}", B, A[i], 0.014, RPR_COLS[i])
            v.set_sphere(f"dl.elb{i}", A[i], 0.018, theme.JOINT)
            w = d.w * et
            v.set_tube(f"dl.fa{i}a", A[i] + w, P[i] + w, 0.006, (0.82, 0.84, 0.88, 1.0))
            v.set_tube(f"dl.fa{i}b", A[i] - w, P[i] - w, 0.006, (0.82, 0.84, 0.88, 1.0))
            v.set_tube(f"dl.bar{i}a", A[i] - w, A[i] + w, 0.007, RPR_COLS[i])
            v.set_tube(f"dl.bar{i}b", P[i] - w, P[i] + w, 0.007, (0.30, 0.62, 0.78, 1.0))
            for sgn, nm in ((1, "a"), (-1, "b")):
                v.set_sphere(f"dl.sj{i}{nm}u", A[i] + sgn * w, 0.009, theme.JOINT)
                v.set_sphere(f"dl.sj{i}{nm}l", P[i] + sgn * w, 0.009, theme.JOINT)
            v.set_label(f"dl.lbl{i}", B + 0.07 * d.er[i] + [0, 0, 0.05], f"θ{_sub(i + 1)}", theme.MUTED)
        v.set_disk("dl.plat", p, [0, 0, 1], d.r * 1.35, 0.016, (0.30, 0.62, 0.78, 1.0))
        v.set_tube("dl.tool", p, p - [0, 0, 0.05], 0.008, (0.75, 0.78, 0.82, 1.0))
        v.set_sphere("dl.tcp", p - [0, 0, 0.055], 0.012, (0.95, 0.75, 0.25, 1.0))
        m = self.d_mass.value()
        if m > 0:
            v.set_arrow("dl.g", p - [0, 0, 0.07], np.array([0, 0, -0.04 - 0.012 * m]), (0.95, 0.75, 0.25, 1.0), 0.006)
        else:
            v.remove_prefix("dl.g")
        v.set_triad("dl.fs", np.eye(4), 0.12, "{s}")
        # ghost: second FK solution (mirror through the plane of the C_i)
        v.remove_prefix("dl.ghost")
        if self.d_ghost.isChecked():
            for k, q in enumerate(self.dl_sols):
                Pg = d.platform_points(q)
                col = (1.0, 0.55, 0.85, 0.55)
                v.set_polyline(f"dl.ghost{k}.plat", np.vstack([Pg, Pg[:1]]), col, 3)
                legs = np.empty((6, 3))
                legs[0::2], legs[1::2] = A, Pg
                v.set_polyline(f"dl.ghost{k}.legs", legs, (*col[:3], 0.4), 1.5, mode="lines")
        if self.d_trace.isChecked():
            v.set_polyline("dl.path", self.d_P - [0, 0, 0.055], (1.0, 0.8, 0.3, 0.8), 2)
        else:
            v.remove("dl.path")
        if self.d_ws.isChecked():
            if not hasattr(self, "_d_cloud"):
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    P, K = d.workspace_cloud(n=17)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                c = np.clip((np.log(K) - np.log(1.3)) / (np.log(6.0) - np.log(1.3)), 0, 1)
                cols = np.c_[0.25 + 0.75 * c, 0.85 - 0.5 * c, 1.0 - 0.75 * c, np.full(len(c), 0.35)]
                self._d_cloud = (P, cols)
            P, cols = self._d_cloud
            v.set_points("dl.ws", P, cols, 3)
        else:
            v.remove("dl.ws")

    # ================================================================ hooks
    def on_show(self):
        self._redraw()

    def on_hide(self):
        self._stop_anim()


def _se2(q) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = pp.rot2(q[2])
    T[:2, 3] = q[:2]
    return T
