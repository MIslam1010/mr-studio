"""Tab 6 -- planar closed chains: four-bar linkage and five-bar parallel robot (MR Ch. 7, Sec. 8.7)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...references import further_reading
from ... import closed_chains as cc
from .. import theme
from ..base import BaseTab
from ..viewer import SceneView
from ..widgets import (MatrixView, SliderRow, button, checkbox, combo, group, hbox, pen, plot, spin,
                       value_label)

THEORY = r"""
<h2>Closed chains — cut joint &amp; loop closure</h2>
<span class="ref">Modern Robotics Ch. 7 (7.1 kinematics, 7.2 differential kinematics, 7.3 singularities)
and Sec. 8.7 (constrained dynamics)</span>
<p>The loop is cut open at one pin — the <b>cut joint</b> (yellow ring). What is left is a tree of
serial branches rooted at ground pivots, each described exactly like an open chain: screw axes
$\mathcal{S}_i$, home pose $M$, CoM frames $M_{i-1,i}$, spatial inertias $\mathcal{G}_i$ (gravity $-y$).
Re-joining the two tips gives the loop-closure constraint</p>
$$\phi(\theta)=p_{1}(\theta)-p_{2}(\theta)=0,\qquad A(\theta)=\frac{\partial\phi}{\partial\theta}\in\mathbb{R}^{2\times n}$$
<p>Grübler: $dof=3(N-1-J)+\sum f_i$ = 1 for the four-bar ($N=4,J=4$) and 2 for the five-bar ($N=5,J=5$).</p>
<h3>Forward kinematics: actuated → passive (7.1)</h3>
<p>Closed form: the actuated joints fix the last passive joint of each branch; the cut pin lies on a
circle around each of them → circle–circle intersection with two <b>assembly modes</b> $\sigma=\pm1$.
The app checks it against Newton–Raphson on the same equations (MR 6.2):</p>
$$\theta_p^{k+1}=\theta_p^{k}-A_p^{\dagger}(\theta^{k})\,\phi(\theta^{k})$$
<h3>Differential kinematics (7.2)</h3>
$$A_a\dot{\theta}_a+A_p\dot{\theta}_p=0\quad\Rightarrow\quad\dot{\theta}_p=-A_p^{-1}A_a\dot{\theta}_a\quad(cf.\ 7.15)$$
<p>The columns of $A$ come from each branch's space Jacobian $J_s$ (point velocity
$\dot{p}=v_s+\omega_s\times p$), and $\dot{A}\dot{\theta}$ from $\dot{J}_{si}=[\mathrm{ad}_{\mathcal{V}_{i-1}}]J_{si}$.</p>
<h3>Singularities (7.3)</h3>
<p><b>Actuator singularity</b> (Type II, parallel): $\det A_p=0$ — locked actuators no longer hold the
mechanism. Four-bar: coupler and rocker aligned, transmission angle $\mu=0$ or $180°$ (dead centre
for the crank), since $|\det A_p|=L_2L_3|\sin\mu|$. Five-bar: the two distal links are collinear.<br>
<b>End-effector singularity</b> (Type I, serial): a five-bar branch fully stretched or folded,
$\sin\theta_2=0$ or $\sin\theta_4=0$ — this is the workspace boundary.<br>
<b>Configuration-space singularity</b>: $A$ itself loses rank (change-point four-bar folded flat).</p>
<h3>Grashof condition (four-bar)</h3>
$$s+l\leq p+q$$
<p>$s,l$ = shortest and longest link, $p,q$ the others. If it holds, the shortest link turns fully:
crank-rocker (shortest = crank), double-crank (shortest = ground), double-rocker (shortest = coupler).
Equality: change-point. Otherwise: triple-rocker. Design rule: $40°\leq\mu\leq140°$.</p>
<h3>Constrained dynamics (8.7)</h3>
<p>$M$ is block-diagonal and $h=c+g$ is stacked from the branches (mr.MassMatrix,
mr.VelQuadraticForces, mr.GravityForces). The Lagrange multiplier $\lambda$ is the force of the cut pin
on branch 1 (arrow):</p>
$$M(\theta)\ddot{\theta}+h(\theta,\dot{\theta})=\tau+A^{T}(\theta)\lambda\quad(cf.\ 8.93)$$
<p>Forward dynamics differentiates the constraint twice, with Baumgarte stabilisation
$\ddot{\phi}+2\alpha\dot{\phi}+\beta^{2}\phi=0$, and solves one $(n+2)\times(n+2)$ KKT system per RK4 stage:</p>
$$M\ddot{\theta}-A^{T}\lambda=\tau-h$$
$$A\ddot{\theta}=-\dot{A}\dot{\theta}-2\alpha A\dot{\theta}-\beta^{2}\phi$$
<p>Inverse dynamics: from $\theta_a(t)$ solve $\theta_p,\dot{\theta}_p,\ddot{\theta}_p$, then with the
selection matrix $S$ of the actuated joints:</p>
$$\left[S\ \ A^{T}\right]\left[\tau_a;\ \lambda\right]=M\ddot{\theta}+h$$
<p>Constraint forces do no work ($\lambda^{T}A\dot{\theta}=0$), hence the power balance shown in the
plots:</p>
$$\tau_a^{T}\dot{\theta}_a=\frac{d}{dt}\left(K+P\right)$$
"""
THEORY += further_reading([
    ('lavalle', '§4.4, from p. 167', 'closed kinematic chains: loop-closure constraints as varieties'),
    ('lavalle', '§7.4, pp. 337–346', 'closed chains; active–passive decomposition (§7.4.2, p. 343)'),
    ('lavalle', '§4.1–4.2, pp. 127–154', 'configuration space and its topology'),
    ('mls', 'Ch. 3 §5.2–5.3, pp. 132–138', 'parallel manipulators; the four-bar linkage (p. 135)'),
    ('featherstone', 'Ch. 8, from p. 141', 'closed-loop dynamics: loop constraints (p. 143), loop forces (p. 148)'),
])


FOUR_PRESETS = {
    "Crank-rocker": (0.40, 0.12, 0.38, 0.28),
    "Double-crank (drag link)": (0.12, 0.30, 0.35, 0.28),
    "Double-rocker (Grashof)": (0.40, 0.30, 0.12, 0.35),
    "Change-point (parallelogram)": (0.40, 0.20, 0.40, 0.20),
    "Triple-rocker (non-Grashof)": (0.50, 0.25, 0.30, 0.28),
}
LINK_COLS = [theme.SERIES[0], theme.SERIES[1], theme.SERIES[2], theme.SERIES[4]]
CURVE = (1.0, 0.35, 0.85, 0.95)


def _deg(x):
    return np.degrees(x)


def _break_jumps(y, jump=180.0):
    y = np.array(y, float)
    d = np.abs(np.diff(y))
    y[1:][d > jump] = np.nan
    return y


class ClosedChainTab(BaseTab):
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
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
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
        self.id_btn = button("Compute inverse dynamics", self._compute_id)
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

    # ================================================================ drawing
    def _draw_ground(self):
        m = self.mech
        v = self.view
        for k, b in enumerate(m.branches):
            G = b.pivot
            h, w = 0.045, 0.03
            tri = np.array([G, G + [-w, -h, 0], G + [w, -h, 0], G]) + [0, 0, -0.02]
            v.set_polyline(f"cc.gnd{k}.tri", tri, (0.75, 0.78, 0.82, 1.0), 2.5)
            hatch = []
            for x in np.linspace(-1.4 * w, 1.4 * w, 7):
                hatch += [G + [x, -h, -0.02], G + [x - 0.012, -h - 0.016, -0.02]]
            hatch += [G + [-1.6 * w, -h, -0.02], G + [1.6 * w, -h, -0.02]]
            v.set_polyline(f"cc.gnd{k}.hatch", hatch, (0.6, 0.63, 0.68, 0.9), 1.5, mode="lines")
        O, D = m.branches[0].pivot, m.branches[1].pivot
        v.set_polyline("cc.gndline", [O + [0, 0, -0.02], D + [0, 0, -0.02]], (0.5, 0.55, 0.6, 0.6), 1.5)

    def _draw_mech(self, th, prefix="cc.m"):
        m, v = self.mech, self.view
        pts = m.points(th)
        r = 0.009
        j = 0
        for bi, P in enumerate(pts):
            for i in range(len(P) - 1):
                v.set_tube(f"{prefix}.link{j}", P[i], P[i + 1], r, LINK_COLS[j % len(LINK_COLS)])
                if i < len(P) - 1:
                    v.set_disk(f"{prefix}.j{bi}.{i}", P[i], [0, 0, 1], 0.016, 0.026, theme.JOINT)
                    v.set_disk(f"{prefix}.pin{bi}.{i}", P[i], [0, 0, 1], 0.006, 0.034, "#1b1e23")
                j += 1
        tip = pts[0][-1]
        v.set_disk(f"{prefix}.cut", tip, [0, 0, 1], 0.021, 0.024, theme.WARN)
        v.set_disk(f"{prefix}.cutpin", tip, [0, 0, 1], 0.008, 0.034, "#1b1e23")
        if self.four:
            A, B, C = pts[0][1], pts[0][2], m.coupler_point(th)
            v.set_tube(f"{prefix}.tri1", A, C, 0.005, LINK_COLS[1])
            v.set_tube(f"{prefix}.tri2", B, C, 0.005, LINK_COLS[1])
            v.set_sphere(f"{prefix}.C", C, 0.012, CURVE)
        return pts

    def _labels_toggled(self, on):
        self.view.set_visible_prefix("cc.lbl", on)
        self._redraw()

    def _redraw(self, *_):
        m, th, v = self.mech, self.theta, self.view
        if m is None or th is None:
            return
        pts = self._draw_mech(th)
        # labels
        if self.cb_labels.isChecked():
            names = (["O", "A", "B"], ["D", "B"]) if self.four else (["O", "A", "P"], ["D", "C", "P"])
            off = np.array([0.018, 0.02, 0.03])
            for bi, P in enumerate(pts):
                for i, nm in enumerate(names[bi]):
                    if bi == 1 and i == len(P) - 1:
                        continue
                    v.set_label(f"cc.lbl{bi}.{i}", P[i] + off, nm, theme.TEXT)
            if self.four:
                v.set_label("cc.lblC", m.coupler_point(th) + off, "C", "#ff8fe0")
            v.set_label("cc.lblcut", pts[0][-1] + np.array([0.025, -0.035, 0.03]), "cut joint", theme.WARN)
        # ghost of the other assembly mode
        other = m.fk_closed_form(th[m.actuated], -m.assembly_mode(th)) if self.cb_ghost.isChecked() else None
        if other is not None and np.max(np.abs(cc.wrap(other - th))) > 1e-6:
            for bi, P in enumerate(m.points(other)):
                v.set_polyline(f"cc.ghost{bi}", P + [0, 0, -0.012], (0.8, 0.85, 1.0, 0.35), 3)
        else:
            v.remove_prefix("cc.ghost")
        if self.four:
            self._redraw_four(th)
        else:
            self._redraw_five(th)
        # constraint Jacobian
        _, A, _ = m.constraint(th)
        self.A_view.set_matrix(A)
        # reaction-force arrow (dynamics)
        if self.lam_show is not None:
            lam, scale = self.lam_show
            v.set_arrow("cc.lam", pts[0][-1] + [0, 0, 0.03], np.r_[lam, 0] * scale, theme.ACCENT, 0.006)
        else:
            v.remove_prefix("cc.lam")

    def _redraw_four(self, th):
        m, v = self.mech, self.view
        if self.cb_curve.isChecked():
            for sg in (1, -1):
                col = CURVE if sg == self.sigma else (1.0, 0.35, 0.85, 0.28)
                for i, c in enumerate(self.curves[sg]):
                    v.set_polyline(f"cc.curve{sg}.{i}", c + [0, 0, -0.016], col, 2.2 if sg == self.sigma else 1.5)
        else:
            v.remove_prefix("cc.curve")
        if len(self.trace) > 1:
            v.set_polyline("cc.trace", np.array(self.trace) + [0, 0, -0.01], (1.0, 0.95, 0.4, 0.9), 2.5)
        else:
            v.remove("cc.trace")
        mu = m.transmission_angle(th)
        det = m.actuator_singularity(th)
        tog = abs(np.sin(th[1]))
        names = "θ₁, θ₂, θ₃"
        self.state_lbl.setText(f"{names} = ({_deg(th[0]):+.1f}°, {_deg(th[1]):+.1f}°, {_deg(th[2]):+.1f}°)\n"
                               f"‖φ(θ)‖ = {np.linalg.norm(m.constraint(th)[0]):.1e} m     σ = {self.sigma:+d}\n"
                               f"transmission angle μ = {_deg(mu):.1f}°   det Aₚ/(L₂L₃) = {det:+.3f}")
        if abs(det) < 0.05:
            txt, col = "ACTUATOR SINGULARITY (dead centre for the crank): coupler ∥ rocker, det Aₚ ≈ 0", theme.BAD
        elif _deg(mu) < 40 or _deg(mu) > 140:
            txt, col = f"Poor transmission: μ = {_deg(mu):.0f}° outside 40°–140°", theme.WARN
        else:
            txt, col = "Well-conditioned: μ within 40°–140°", theme.GOOD
        if tog < 0.03:
            txt += "<br>Toggle position: crank ∥ coupler → rocker at an extreme (dead centre if the rocker drove)"
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")
        x = _deg(th[0])
        for c in self.cursors:
            c.setValue(x)
        self.mu_dot.setData([x], [_deg(mu)])

    def _redraw_five(self, th):
        m, v = self.mech, self.view
        v.remove_prefix("cc.ws")
        if self.cb_ws.isChecked():
            for i, a in enumerate(self.ws_arcs):
                v.set_polyline(f"cc.ws{i}", a + [0, 0, -0.018], (1.0, 0.65, 0.2, 0.85), 2.0)
        if self.elbows not in self.t2:
            self._update_type2()
        if not self.cb_t2.isChecked():
            v.remove_prefix("cc.t2")
        elif not any(k.startswith("cc.t2") for k in v.items_by_key):
            self._update_type2()
        if len(self.trace) > 1:
            v.set_polyline("cc.trace", np.array(self.trace) + [0, 0, -0.01], (1.0, 0.95, 0.4, 0.9), 2.5)
        else:
            v.remove("cc.trace")
        s2, s4, d2 = m.singularity_measures(th)
        P = m.tip(th)
        self.state_lbl.setText(
            f"θ = ({_deg(th[0]):+.1f}°, {_deg(th[1]):+.1f}°, {_deg(th[2]):+.1f}°, {_deg(th[3]):+.1f}°)\n"
            f"P = ({P[0]:+.4f}, {P[1]:+.4f}) m    σ = {self.sigma:+d}\n"
            f"Type I:  sin θ₂ = {s2:+.3f},  sin θ₄ = {s4:+.3f}\nType II: det Aₚ/(L₂L₃) = {d2:+.3f}")
        msgs = []
        worst = 1.0
        if min(abs(s2), abs(s4)) < 0.05:
            msgs.append("TYPE I (serial / end-effector) singularity: a branch is stretched or folded")
        if abs(d2) < 0.05:
            msgs.append("TYPE II (parallel / actuator) singularity: distal links collinear, det Aₚ ≈ 0")
        worst = min(abs(s2), abs(s4), abs(d2))
        if msgs:
            txt, col = "<br>".join(msgs), theme.BAD
        elif worst < 0.2:
            txt, col = "Close to a singularity", theme.WARN
        else:
            txt, col = "Well-conditioned configuration", theme.GOOD
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")
        self.p_dot.setData([P[0]], [P[1]])
        self.sing_hist.append((s2, s4, d2))
        self.sing_hist = self.sing_hist[-400:]
        h = np.array(self.sing_hist)
        x = np.arange(len(h))
        for c, y in zip(self.hist_curves, h.T):
            c.setData(x, y)

    # ================================================================ plots
    def _init_kin_plots_four(self):
        for p in (self.pk1, self.pk2):
            p.clear()
            p.setLogMode(False, False)
            p.setAspectLocked(False)
            p.enableAutoRange()
        m = self.mech
        self.pk1.setTitle("Transmission angle μ(θ₁)", color=theme.TEXT, size="10pt")
        self.pk1.setLabel("left", "μ [°]")
        self.pk1.setLabel("bottom", "crank angle θ₁ [°]")
        band = pg.LinearRegionItem((40, 140), orientation="horizontal", movable=False,
                                   brush=pg.mkBrush(87, 214, 141, 28), pen=pg.mkPen(None))
        self.pk1.addItem(band)
        th1 = np.linspace(-np.pi, np.pi, 721)
        AD2 = m.L1 ** 2 + m.d ** 2 - 2 * m.L1 * m.d * np.cos(th1)
        cmu = (m.L2 ** 2 + m.L3 ** 2 - AD2) / (2 * m.L2 * m.L3)
        mu = np.where(np.abs(cmu) <= 1, np.degrees(np.arccos(np.clip(cmu, -1, 1))), np.nan)
        self.pk1.plot(np.degrees(th1), mu, pen=pen(0, 2.5), name="μ (both σ)", connect="finite")
        self.pk1.addLine(y=90, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.mu_dot = self.pk1.plot([], [], pen=None, symbol="o", symbolBrush=theme.WARN, symbolSize=9)
        self.pk1.setYRange(0, 180)
        self.pk2.setTitle("Passive joints from loop closure", color=theme.TEXT, size="10pt")
        self.pk2.setLabel("left", "angle [°]")
        self.pk2.setLabel("bottom", "crank angle θ₁ [°]")
        self.pass_curves = [self.pk2.plot([], [], pen=pen(1, 2.2), name="θ₂ (coupler)", connect="finite"),
                            self.pk2.plot([], [], pen=pen(2, 2.2), name="θ₃ (rocker)", connect="finite"),
                            self.pk2.plot([], [], pen=pen(1, 1.2, "dash"), connect="finite"),
                            self.pk2.plot([], [], pen=pen(2, 1.2, "dash"), connect="finite")]
        self.cursors = [p.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1)) for p in (self.pk1, self.pk2)]
        self._update_kin_plots_four()

    def _update_kin_plots_four(self):
        if not self.four:
            return
        for k, sg in enumerate((self.sigma, -self.sigma)):
            th1, TH = self.sweeps[sg]
            x = np.degrees(th1)
            self.pass_curves[2 * k].setData(x, _break_jumps(np.degrees(TH[:, 1])))
            self.pass_curves[2 * k + 1].setData(x, _break_jumps(np.degrees(TH[:, 2])))

    def _init_kin_plots_five(self):
        for p in (self.pk1, self.pk2):
            p.clear()
            p.setLogMode(False, False)
            p.enableAutoRange()
        self.pk1.setTitle("Singularity measures (history)", color=theme.TEXT, size="10pt")
        self.pk1.setLabel("left", "")
        self.pk1.setLabel("bottom", "update #")
        self.hist_curves = [self.pk1.plot([], [], pen=pen(1, 2), name="sin θ₂ (Type I, left)"),
                            self.pk1.plot([], [], pen=pen(2, 2), name="sin θ₄ (Type I, right)"),
                            self.pk1.plot([], [], pen=pen(3, 2.5), name="det Aₚ/(L₂L₃) (Type II)")]
        self.pk1.addLine(y=0, pen=pg.mkPen(theme.BAD, style=QtCore.Qt.DashLine))
        self.pk1.setYRange(-1.05, 1.05)
        self.pk2.setTitle("|det Aₚ| over the workspace (this working mode; red = Type II)", color=theme.TEXT,
                          size="10pt")
        self.pk2.setLabel("left", "y [m]")
        self.pk2.setLabel("bottom", "x [m]")
        self.img = pg.ImageItem()
        cmap = pg.ColorMap([0.0, 0.12, 0.35, 1.0], [(255, 70, 70), (255, 190, 70), (60, 110, 170), (18, 40, 80)])
        self.img.setLookupTable(cmap.getLookupTable(0, 1, 256))
        self.pk2.addItem(self.img)
        self.img_lines = []
        self.p_dot = self.pk2.plot([], [], pen=None, symbol="+", symbolSize=16, symbolPen=pg.mkPen("w", width=2))
        self.pk2.setAspectLocked(True)
        for ax in ("left", "bottom"):
            self.pk2.getAxis(ax).enableAutoSIPrefix(False)
        self.cursors = []

    def _init_dyn_plots(self):
        self.E_curves = [self.pE.plot([], [], pen=pen(0, 2), name="kinetic K"),
                         self.pE.plot([], [], pen=pen(1, 2), name="potential P"),
                         self.pE.plot([], [], pen=pen(2, 2.5), name="K + P"),
                         self.pE.plot([], [], pen=pen(5, 2, "dash"), name="K + P − W_motor + W_damping")]
        self.phi_curve = self.pphi.plot([], [], pen=pen(3, 2))
        self.lam_curves = [self.plam.plot([], [], pen=pen(0, 1.8), name="λx"),
                           self.plam.plot([], [], pen=pen(1, 1.8), name="λy"),
                           self.plam.plot([], [], pen=pen(5, 2.2), name="|λ|")]

    # ================================================================ animation / simulation
    def _stop_anims(self):
        for b in (self.rot_btn, self.circle_btn, self.swing_btn, self.id_play):
            if b.isChecked():
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
        self.rot_btn.setText("▶ Rotate crank")
        self.circle_btn.setText("▶ P traces a circle")
        self.swing_btn.setText("▶ Free swing (live RK4)")
        self.id_play.setText("▶ Play")
        self.timer.stop()
        self.anim = None

    def _anim_toggle(self, kind, on):
        texts = {"crank": ("▶ Rotate crank", "■ Stop"), "circle": ("▶ P traces a circle", "■ Stop"),
                 "swing": ("▶ Free swing (live RK4)", "■ Stop simulation"), "id": ("▶ Play", "■ Stop")}
        btns = {"crank": self.rot_btn, "circle": self.circle_btn, "swing": self.swing_btn, "id": self.id_play}
        if not on:
            btns[kind].setText(texts[kind][0])
            self.timer.stop()
            self.anim = None
            if kind == "swing" and self.theta is not None:
                self.dyn_lbl.setText(self.dyn_lbl.text() + "\n(stopped — the mechanism stays in the last simulated pose)")
            return
        self._stop_anims()
        btns[kind].blockSignals(True)
        btns[kind].setChecked(True)
        btns[kind].blockSignals(False)
        btns[kind].setText(texts[kind][1])
        if self.theta is None:
            self._stop_anims()
            return
        self.anim = kind
        self.trace = []
        self.anim_t = 0.0
        if kind == "circle":
            self.circ_c = self.mech.tip(self.theta)[:2] - np.array([0.05, 0.0])
        elif kind == "swing":
            if not self._start_swing():
                self._stop_anims()
                return
        elif kind == "id":
            if self.id_res is None:
                self._compute_id()
            if self.id_res is None:
                self._stop_anims()
                return
            self.bottom.setCurrentIndex(2)
            self.id_k = 0
        self._last = time.time()
        self.timer.start(30)

    def _tick(self):
        try:
            now = time.time()
            dt = min(now - self._last, 0.1)
            self._last = now
            if self.anim == "crank":
                self._tick_crank(dt)
            elif self.anim == "circle":
                self._tick_circle(dt)
            elif self.anim == "swing":
                self._tick_swing()
            elif self.anim == "id":
                self._tick_id()
        except Exception as e:  # never let a timer callback crash the app
            self._stop_anims()
            self.dyn_lbl.setText(f"stopped: {type(e).__name__}: {e}")

    def _tick_crank(self, dt):
        m = self.mech
        w = 2 * np.pi * 0.25 * self.anim_dir
        th1 = self.crank.value() + w * dt
        if not m.crank_feasible(th1):
            # input limit (det A_p = 0): reverse the crank and continue on the other assembly mode
            self.anim_dir *= -1
            th1 = self.crank.value()
            self.sigma_box.blockSignals(True)
            self.sigma_box.setCurrentIndex(1 if self.sigma == 1 else 0)
            self.sigma_box.blockSignals(False)
            self.sigma = -self.sigma
            self._update_kin_plots_four()
        self.crank.set_value(float(cc.wrap(th1)))
        self._kin_update()
        self.trace.append(m.coupler_point(self.theta))
        self.trace = self.trace[-600:]

    def _tick_circle(self, dt):
        self.anim_t += dt
        a = 2 * np.pi * self.anim_t / 4.0
        P = self.circ_c + 0.05 * np.array([np.cos(a), np.sin(a)])
        self.px.set_value(P[0])
        self.py.set_value(P[1])
        self._p_changed()
        if self.theta is not None:
            self.trace.append(self.mech.tip(self.theta))
            self.trace = self.trace[-400:]

    # -- free swing -----------------------------------------------------------------------
    def _tau_vec(self):
        tau = np.zeros(self.mech.n)
        tau[self.mech.actuated[0]] = self.tau1.value()
        if self.mech.na > 1:
            tau[self.mech.actuated[1]] = self.tau3.value()
        return tau

    def _start_swing(self) -> bool:
        m = self.mech
        m.backend = self.ctx.backend
        th0 = self.theta.copy()
        self.sim = dict(t=0.0, th=th0, dth=np.zeros(m.n), W=0.0, D=0.0)
        self.buf = {k: [] for k in ("t", "ke", "pe", "E", "Ec", "phi", "lx", "ly")}
        self.lam_max = 1e-9
        self.bottom.setCurrentIndex(1)
        self.mode_tabs.setCurrentIndex(1)
        return True

    def _tick_swing(self):
        m, s = self.mech, self.sim
        dt = self.dt.value()
        steps = max(1, int(round(0.03 / dt)))
        a, b = (self.alpha.value(), self.beta.value()) if self.cb_baum.isChecked() else (0.0, 0.0)
        tau = self._tau_vec()
        t0 = time.time()
        for k in range(steps):
            th, dth, lam, dW, dD = m.rk4_step(s["th"], s["dth"], s["t"], dt, lambda t, q, dq: tau,
                                                a, b, self.damp.value())
            s.update(th=th, dth=dth, t=s["t"] + dt, W=s["W"] + dW, D=s["D"] + dD)
            if time.time() - t0 > 0.06:   # slow (MR) back-end: stay responsive
                break
        ke, pe = m.energy(s["th"], s["dth"])
        phi = np.linalg.norm(m.constraint(s["th"])[0])
        B = self.buf
        for key, val in (("t", s["t"]), ("ke", ke), ("pe", pe), ("E", ke + pe), ("Ec", ke + pe - s["W"] + s["D"]),
                         ("phi", max(phi, 1e-18)), ("lx", lam[0]), ("ly", lam[1])):
            B[key].append(val)
            del B[key][:-1500]
        t = np.array(B["t"])
        for c, key in zip(self.E_curves, ("ke", "pe", "E", "Ec")):
            c.setData(t, np.array(B[key]))
        self.phi_curve.setData(t, np.array(B["phi"]))
        lx, ly = np.array(B["lx"]), np.array(B["ly"])
        for c, y in zip(self.lam_curves, (lx, ly, np.hypot(lx, ly))):
            c.setData(t, y)
        self.lam_max = max(self.lam_max, float(np.hypot(*lam)))
        self.lam_show = (lam, 0.12 / self.lam_max)
        self.theta = s["th"]
        self._redraw()
        drift = B["Ec"][-1] - B["Ec"][0]
        self.dyn_lbl.setText(f"t = {s['t']:.2f} s   ({m.backend} blocks, dt = {dt * 1e3:.1f} ms)\n"
                             f"K + P − W + D drift = {drift:+.2e} J\n‖φ‖ = {phi:.1e} m    |λ| = {np.hypot(*lam):.2f} N")

    # -- inverse dynamics -----------------------------------------------------------------------
    def _compute_id(self):
        m = self.mech
        if m is None or self.theta is None:
            return
        m.backend = self.ctx.backend
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            t0 = time.time()
            if self.four:
                args = (self.omega.value(), self.sigma, float(self.theta[0]))
                t, TH, DTH, DDTH = m.crank_trajectory(*args, N=181)
                traj = lambda tt: m.crank_trajectory(*args, t=tt)[1:]
                full = m.crank_fully_rotates()
                x = np.degrees(np.unwrap(TH[:, 0])) if full else t
                xl = "crank angle θ₁ [°]" if full else "t [s]"
            else:
                c = m.tip(self.theta)[:2] - np.array([self.radius.value(), 0.0])
                args = (c, self.radius.value(), self.period.value(), self.elbows)
                t, TH, DTH, DDTH = m.circle_trajectory(*args, N=181)
                traj = lambda tt: m.circle_trajectory(*args, t=tt)[1:]
                x, xl = t, "t [s]"
            res = m.inverse_dynamics_trajectory(t, TH, DTH, DDTH, x, xl, traj_fn=traj)
        except (ValueError, np.linalg.LinAlgError) as e:
            self.dyn_lbl.setText(f"⚠ {e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.id_res = res
        for p in (self.ptau, self.pidlam, self.ppow):
            p.clear()
            p.setLabel("bottom", xl)
        names = ["τ₁ (crank)"] if self.four else ["τ₁ (left motor)", "τ₃ (right motor)"]
        for i in range(m.na):
            self.ptau.plot(x, res.tau[:, i], pen=pen(i, 2.4), name=names[i])
        lam = res.lam
        self.pidlam.plot(x, lam[:, 0], pen=pen(0, 1.8), name="λx")
        self.pidlam.plot(x, lam[:, 1], pen=pen(1, 1.8), name="λy")
        self.pidlam.plot(x, np.hypot(lam[:, 0], lam[:, 1]), pen=pen(5, 2.4), name="|λ|")
        self.ppow.plot(x, res.power, pen=pen(2, 3), name="τₐᵀθ̇ₐ (motor power)")
        self.ppow.plot(x, res.dEdt, pen=pen(3, 1.6, "dash"), name="dE/dt (central differences)")
        self.id_cursors = [p.addLine(x=x[0], pen=pg.mkPen(theme.WARN, width=1))
                           for p in (self.ptau, self.pidlam, self.ppow)]
        self.lam_scale_id = 0.10 / max(float(np.hypot(lam[:, 0], lam[:, 1]).max()), 1e-9)
        err = np.abs(res.power - res.dEdt).max() / max(np.abs(res.power).max(), 1e-9)
        self.dyn_lbl.setText(f"Inverse dynamics ({len(t)} samples, {m.backend} blocks, {time.time() - t0:.2f} s)\n"
                             f"peak |τ| = {np.abs(res.tau).max():.2f} N·m,  peak |λ| = "
                             f"{np.hypot(lam[:, 0], lam[:, 1]).max():.1f} N\n"
                             f"power balance: max|τθ̇ − dE/dt| / max|τθ̇| = {err:.1e}")
        self.bottom.setCurrentIndex(2)

    def _tick_id(self):
        r = self.id_res
        k = self.id_k % len(r.t)
        self.id_k += 1
        self.theta = r.theta[k].copy()
        self.lam_show = (r.lam[k], self.lam_scale_id)
        for c in self.id_cursors:
            c.setValue(r.x[k])
        if self.four:
            self.trace.append(self.mech.coupler_point(self.theta))
        else:
            self.trace.append(self.mech.tip(self.theta))
        self.trace = self.trace[-len(r.t):]
        self._redraw()

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


def _wrapped(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    return lbl


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
