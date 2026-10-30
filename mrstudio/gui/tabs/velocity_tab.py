"""Tab 2 -- velocity kinematics, statics, singularities and manipulability (MR Ch. 5)."""
from __future__ import annotations

import modern_robotics as mr
import numpy as np
import pyqtgraph as pg
from matplotlib import colormaps
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.optimize import minimize

from ...references import further_reading
from ... import kinematics as kin
from .. import theme
from ..base import BaseTab
from ..jobs import JobRunner
from ..viewer import RobotGraphic, SceneView
from ..widgets import (JointSliders, MatrixView, button, checkbox, combo, group, hbox, plot, spin,
                       value_label)

TWIST_ROWS = ["ωx", "ωy", "ωz", "vx", "vy", "vz"]
MIXED_LABELS = ["ωx", "ωy", "ωz", "vx", "vy", "vz"]   # EE-centred, {s}-aligned axes
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_CMAP = colormaps["plasma"]
_CMAP_RANGE = (0.12, 0.98)          # skip the near-black end (dark background)


def _sub(i: int) -> str:
    return str(i).translate(_SUB)


def _vec(v, d=3) -> str:
    return "(" + ", ".join(f"{x:+.{d}f}" for x in v) + ")"


def _fit_table(mv: MatrixView, max_h: int = 400):
    """Size a MatrixView to show all its rows (no scrollbar); call again once the widget is polished."""
    mv.resizeRowsToContents()
    h = mv.verticalHeader().length() + 2 * mv.frameWidth() + 4
    if mv.horizontalHeader().isVisible():
        h += max(mv.horizontalHeader().height(), mv.horizontalHeader().sizeHint().height())
    mv.setFixedHeight(min(h, max_h))
    mv.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    mv.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)


def _unfix_table(mv: MatrixView):
    """Let a MatrixView take the space its parent layout gives it."""
    mv.setMinimumHeight(60)
    mv.setMaximumHeight(16777215)


def _clear_layout(lay: QtWidgets.QLayout):
    """Remove every item of a layout: widgets are deleted, sub-layouts cleared recursively, spacers dropped."""
    while lay.count():
        item = lay.takeAt(0)
        if item is None:
            break
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


def _cmap(x: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    lo, hi = _CMAP_RANGE
    c = _CMAP(lo + (hi - lo) * np.clip(x, 0, 1))
    c[:, 3] = alpha
    return c


THEORY = r"""
<h2>Velocity kinematics, statics &amp; singularities</h2>
<span class="ref">Modern Robotics, Ch. 5: §5.1 Jacobian, §5.2 statics, §5.3 singularities, §5.4 manipulability</span>
<h3>Space Jacobian (§5.1.1)</h3>
<p>The end-effector twist is linear in the joint rates, $\mathcal{V}_s=J_s(\theta)\,\dot{\theta}$, with
$J_{s1}=\mathcal{S}_1$ and</p>
$$J_{si}(\theta)=\left[\mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}\right]\mathcal{S}_i\quad(5.11)$$
<p>Column $i$ is the screw axis of joint $i$ <i>moved by joints 1…i−1</i>, written in {s}; it does not
depend on $\theta_i,\ldots,\theta_n$.</p>
<h3>Body Jacobian (§5.1.2, §5.1.4)</h3>
<p>$\mathcal{V}_b=J_b(\theta)\,\dot{\theta}$, with $J_{bn}=\mathcal{B}_n$ and</p>
$$J_{bi}(\theta)=\left[\mathrm{Ad}_{e^{-[\mathcal{B}_n]\theta_n}\cdots e^{-[\mathcal{B}_{i+1}]\theta_{i+1}}}\right]\mathcal{B}_i\quad(5.18)$$
$$J_b(\theta)=\left[\mathrm{Ad}_{T_{bs}}\right]J_s(\theta),\qquad J_s(\theta)=\left[\mathrm{Ad}_{T_{sb}}\right]J_b(\theta)\quad(5.22),\ (5.23)$$
<p>The green line under the tables checks this identity live.  Note that $v_s$ is the velocity of the
point of the body at the <i>origin of {s}</i>; the velocity of the end-effector point (green arrow) is</p>
$$\dot{p}=\omega_s\times p+v_s=R_{sb}\,v_b$$
<h3>Statics (§5.2)</h3>
<p>Power is conserved, $\tau^{T}\dot{\theta}=\mathcal{F}_s^{T}\mathcal{V}_s$, so the joint torques that
let the end-effector apply the wrench $\mathcal{F}=(m,f)$ to the environment (no gravity) are</p>
<p>In the panel, $f$ and $m$ are the tip force and moment acting at the end-effector point $p$, with axes
parallel to {s}; the table shows the same wrench as $\mathcal{F}_s$, i.e. moved to the {s} origin:</p>
$$\tau=J_s^{T}(\theta)\,\mathcal{F}_s=J_b^{T}(\theta)\,\mathcal{F}_b\quad(5.26)$$
$$\mathcal{F}_s=(m_s+p\times f,\ f),\qquad \mathcal{F}_b=\left[\mathrm{Ad}_{T_{sb}}\right]^{T}\mathcal{F}_s$$
<h3>Singularities (§5.3)</h3>
<p>A configuration is singular when $J(\theta)$ loses rank.  With the SVD
$J=U\Sigma V^{T}$, the columns of $U$ whose $\sigma_i=0$ are the end-effector twist directions that
<b>no</b> joint velocity can produce (listed in the banner, drawn in red), while those wrench
directions are resisted by the structure with zero torque.</p>
$$\mathrm{rank}\,J(\theta)<\min(m,n)\ \Leftrightarrow\ \sigma_{min}(J)=0$$
<p>Examples: two collinear revolute axes, three coplanar parallel axes (UR5/PUMA elbow
stretched), the wrist centre on joint axis 1 (shoulder), aligned wrist axes (UR5 $\theta_5=0$).</p>
<p><b>Units.</b> The rank of $J$ does not depend on units, but $\sigma_{min}$ of the full $6\times n$ $J$ does:
its rows mix rad/s ($\omega$) and m/s ($v$), so a fixed threshold would change with the length unit (§5.4).
The test therefore uses the dimensionless $\tilde{J}=\mathrm{diag}(I,\,I/L)\,J$ ($L$ = reach), and also reports
$\sigma_{min}(J_\omega)$ and $\sigma_{min}(J_v)$ separately, each in its own unit.
"Find nearest singularity" minimises $\sigma_{min}^{2}(\tilde{J}(\theta))+w\|\theta-\theta_0\|^{2}$ with SciPy
(in a background thread).</p>
<h3>Manipulability ellipsoid (§5.4)</h3>
<p>The unit sphere of joint rates $\|\dot{\theta}\|=1$ maps to the ellipsoid of end-effector velocities $v$</p>
$$v^{T}A^{-1}v=1,\qquad A=JJ^{T}\quad(5.27)$$
<p>with semi-axes $\sqrt{\lambda_i(A)}=\sigma_i(J)$ along the eigenvectors of $A$.  Linear (angular)
ellipsoid: $J$ = rows $v$ ($\omega$) of $J_b$ rotated into {s}.  The <b>force ellipsoid</b>
$f^{T}Af=1$ ($B=A^{-1}$) has the same axes and reciprocal lengths: fast directions are weak.  At a
singularity a semi-axis of the force ellipsoid is <b>infinite</b> (the structure resists any force along it
with zero torque); it is drawn clamped, in red, marked ∞.  The ellipsoid shape is exact; it is drawn at a
fixed scale set at the ready pose.</p>
$$\mu_1(A)=\sqrt{\frac{\lambda_{max}(A)}{\lambda_{min}(A)}}\geq 1$$
$$\mu_2(A)=\frac{\lambda_{max}(A)}{\lambda_{min}(A)}=\mu_1^{2}$$
$$\mu_3(A)=\sqrt{\lambda_1\lambda_2\cdots}=\sqrt{\det A}$$
<p>$\mu_1=1$ is isotropic; at a singularity $\mu_1,\mu_2\to\infty$ and $\mu_3\to 0$.</p>
<h3>Workspace</h3>
<p>Monte-Carlo: end-effector positions of uniformly random joint vectors, coloured by $\mu_3$ of the linear
ellipsoid (dark purple = near-singular, yellow = most dexterous).</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 3 §4, pp. 115–129', 'spatial/body Jacobian, forces (Jᵀ), singularities, manipulability'),
    ('mls', 'Ch. 2 §5, pp. 61–70', 'wrenches and reciprocal screws'),
    ('gotor', '§5.3, pp. 127–148', 'geometric (spatial and tool) Jacobian; adjoint in §5.3.2, p. 128'),
    ('corke', '§8.1–8.3, pp. 171–186', 'Jacobian, manipulability (p. 177), singularity (p. 182), wrenches'),
])



class VelocityTab(BaseTab):
    title = "② Velocity"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        self.graphic = None
        self.theta = np.zeros(0)
        self._ell_scale: dict = {}
        self.ws_jobs = JobRunner(self)           # workspace sampling (one job at a time)
        self.sing_jobs = JobRunner(self)         # nearest-singularity search (L-BFGS-B, off the GUI thread)
        self._ws_P, self._ws_W = [], []

        # ---------------------------------------------------------------- joints
        self.sliders = JointSliders()
        self.sliders.changed.connect(self._sliders_moved)
        self.move_btn = button("▶ Move", self.toggle_move, checkable=True,
                               tooltip="Integrate θ ← θ + θ̇ Δt and watch ṗ stay tangent to the path")
        btns = hbox(button("Ready pose", lambda: self.set_theta(self.ctx.robot.home),
                           tooltip="a convenient non-singular start pose; the MR home configuration is θ = 0, T = M"),
                    button("Random", lambda: self.set_theta(self.ctx.robot.random_config())),
                    self.move_btn)
        self.sing_btn = button("⚠  Find nearest singularity", self.find_singularity, primary=True,
                               tooltip="scipy.optimize.minimize of σ_min²(J̃(θ)) + w‖θ−θ₀‖² (background "
                                       "thread), then animate")
        self.sing_lbl = value_label()

        # ---------------------------------------------------------------- ellipsoid
        self.cb_ell = checkbox("Show ellipsoid at the end-effector", self.update_scene, True)
        self.ell_kind = combo(["Linear  (ṗ, uses J_v)", "Angular  (ω, uses J_ω)"], self.update_scene)
        self.ell_type = combo(["Velocity ellipsoid  A = J Jᵀ", "Force ellipsoid  B = (J Jᵀ)⁻¹"],
                              self.update_scene)
        for cb in (self.ell_kind, self.ell_type):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
        self.mu_lbl = value_label()
        self.ell_cap = QtWidgets.QLabel("shape exact; drawn at a fixed scale relative to the ready pose")
        self.ell_cap.setObjectName("muted")
        self.ell_cap.setWordWrap(True)

        # ---------------------------------------------------------------- twists
        self.qd_box = QtWidgets.QWidget()
        self.qd_grid = QtWidgets.QGridLayout(self.qd_box)
        self.qd_grid.setContentsMargins(0, 0, 0, 0)
        self.qd_grid.setHorizontalSpacing(4)
        self.qd_spins: list[QtWidgets.QDoubleSpinBox] = []
        self.cb_vel = checkbox("Draw ṗ (green) and ω_s (violet) at the end-effector", self.update_scene, True)
        self.twist_tab = MatrixView(2, 6, 2, row_labels=["V_s", "V_b"], col_labels=TWIST_ROWS)
        self.twist_lbl = value_label()
        theme.notifier().projector_changed.connect(self._retone)
        self.twist_chk = value_label()
        self._tone(self.twist_chk, "good")

        # ---------------------------------------------------------------- statics
        self.f_spins = [spin(-500, 500, 0, 1, 5, " N") for _ in range(3)]
        self.m_spins = [spin(-100, 100, 0, 2, 1, " Nm") for _ in range(3)]
        fgrid = QtWidgets.QGridLayout()
        fgrid.setHorizontalSpacing(4)
        for j, ax in enumerate("xyz"):
            fgrid.addWidget(QtWidgets.QLabel(f"f{ax}"), 0, 2 * j)
            fgrid.addWidget(self.f_spins[j], 0, 2 * j + 1)
            fgrid.addWidget(QtWidgets.QLabel(f"m{ax}"), 1, 2 * j)
            fgrid.addWidget(self.m_spins[j], 1, 2 * j + 1)
        for s in self.f_spins + self.m_spins:
            s.valueChanged.connect(self.update_scene)
        self.cb_force = checkbox("Draw the applied force (orange) / moment (pink)", self.update_scene, True)
        self.tau_tab = MatrixView(1, 6, 1, row_labels=["τ"], col_labels=[str(i + 1) for i in range(6)])
        self.tau_lbl = value_label()
        self.f_cap = QtWidgets.QLabel("The table shows the same wrench about the {s} origin: "
                                      "F_s = (m + p×f, f)  (p = EE point).")
        self.f_cap.setObjectName("muted")
        self.f_cap.setWordWrap(True)
        self.tau_chk = value_label()
        self._tone(self.tau_chk, "good")

        # ---------------------------------------------------------------- singularity tolerance
        self.tol_spin = spin(1e-4, 0.2, 0.01, 4, 0.005)
        self.tol_spin.valueChanged.connect(self.update_scene)
        self.tol_spin.setToolTip("threshold on σ_min of the dimensionless J̃ = diag(I, I/L) J, L = reach")
        self.sing_note = QtWidgets.QLabel("σ_min of the raw 6×n J mixes rad/s and m/s (MR §5.4), so the test uses "
                                          "J̃ = diag(I, I/L)·J (L = reach) and also lists σ_min(J_ω), σ_min(J_v).")
        self.sing_note.setObjectName("muted")
        self.sing_note.setWordWrap(True)

        # ---------------------------------------------------------------- workspace
        self.ws_n = QtWidgets.QSpinBox()
        self.ws_n.setRange(500, 40000)
        self.ws_n.setSingleStep(500)
        self.ws_n.setValue(6000)
        self.ws_n.setSuffix(" samples")
        self.ws_btn = button("Compute workspace", self.compute_workspace)
        self.ws_clear_btn = button("Clear", self.clear_workspace)
        self.ws_bar = QtWidgets.QProgressBar()
        self.ws_bar.setMaximumHeight(14)
        self.ws_bar.setTextVisible(False)
        self.ws_bar.setValue(0)
        lo, hi = _CMAP_RANGE
        stops = ", ".join(
            f"stop:{t:.2f} rgb({int(255 * c[0])},{int(255 * c[1])},{int(255 * c[2])})"
            for t, c in ((t, _CMAP(lo + (hi - lo) * t)) for t in np.linspace(0, 1, 6)))
        self.ws_legend = QtWidgets.QFrame()
        self.ws_legend.setFixedHeight(12)
        self.ws_legend.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});"
                                     "border-radius:3px;")
        self.ws_lo = QtWidgets.QLabel("μ₃ = 0  (singular)")
        self.ws_hi = QtWidgets.QLabel("max")
        self.ws_hi.setAlignment(QtCore.Qt.AlignRight)
        for l in (self.ws_lo, self.ws_hi):
            l.setObjectName("muted")
        self.ws_lbl = value_label()

        # ---------------------------------------------------------------- left column
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Joint coordinates θ", self.sliders, btns, self.sing_btn, self.sing_lbl))
        for sp in self.f_spins + self.m_spins:
            sp.setMaximumWidth(80)
        cl.addWidget(group("Manipulability ellipsoid  (MR §5.4)", self.cb_ell,
                           self.ell_kind, self.ell_type, self.mu_lbl, self.ell_cap))
        cl.addWidget(group("Joint rates θ̇ [rad/s] → twists  (MR §5.1)", self.qd_box, self.cb_vel,
                           self.twist_tab, self.twist_lbl, self.twist_chk))
        cl.addWidget(group("Statics  (MR §5.2)",
                           QtWidgets.QLabel("Tip wrench at the EE point, axes ∥ {s}: f [N], m [N·m]"), fgrid,
                           self.f_cap, self.cb_force, self.tau_tab, self.tau_lbl, self.tau_chk))
        cl.addWidget(group("Singularity test  (MR §5.3)",
                           hbox(QtWidgets.QLabel("σ_min(J̃) tolerance"), self.tol_spin, None), self.sing_note))
        cl.addWidget(group("Workspace (Monte-Carlo, coloured by μ₃)",
                           hbox(self.ws_n, self.ws_btn, self.ws_clear_btn), self.ws_bar,
                           self.ws_legend, hbox(self.ws_lo, None, self.ws_hi), self.ws_lbl))
        cl.addStretch(1)

        # ---------------------------------------------------------------- centre: banner + 3-D view
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setMinimumHeight(30)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(3)
        cv.addWidget(self.banner)
        cv.addWidget(self.view, 1)

        # ---------------------------------------------------------------- bottom: Jacobians + bar charts
        self.Js_view = MatrixView(6, 6, 3, row_labels=TWIST_ROWS, col_labels=[str(i + 1) for i in range(6)])
        self.Jb_view = MatrixView(6, 6, 3, row_labels=TWIST_ROWS, col_labels=[str(i + 1) for i in range(6)])
        jt = QtWidgets.QTabWidget()
        jt.addTab(self.Js_view, "Space Jacobian J_s(θ)  (Eq. 5.11)")
        jt.addTab(self.Jb_view, "Body Jacobian J_b(θ)  (Eq. 5.18)")
        self.jac_chk = value_label()
        self._tone(self.jac_chk, "good")
        jw = QtWidgets.QWidget()
        jl = QtWidgets.QVBoxLayout(jw)
        jl.setContentsMargins(0, 0, 0, 0)
        jl.addWidget(jt, 1)
        jl.addWidget(self.jac_chk)

        self.sv_plot = plot("Singular values of J̃ = diag(I, I/L)·J", "σᵢ (dimensionless)", "i", legend=False)
        self.sv_plot.setMouseEnabled(x=False, y=False)
        self.sv_bars = pg.BarGraphItem(x=[1], height=[0], width=0.6)
        self.sv_plot.addItem(self.sv_bars)
        self.tol_line = pg.InfiniteLine(pos=0.01, angle=0, pen=pg.mkPen(theme.BAD, width=1.5,
                                                                         style=QtCore.Qt.DashLine))
        self.sv_plot.addItem(self.tol_line)
        self.tau_plot = plot("τ = J_sᵀ F_s", "τ [N·m]", "joint", legend=False)
        self.tau_plot.setMouseEnabled(x=False, y=False)
        self.tau_bars = pg.BarGraphItem(x=[1], height=[0], width=0.6)
        self.tau_plot.addItem(self.tau_bars)
        self.tau_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.MUTED, width=1)))

        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(jw)
        bottom.addWidget(self.sv_plot)
        bottom.addWidget(self.tau_plot)
        bottom.setStretchFactor(0, 5)
        bottom.setStretchFactor(1, 2)
        bottom.setStretchFactor(2, 2)
        bottom.setSizes([470, 200, 200])

        self.assemble(controls, center, bottom, THEORY, controls_width=370, bottom_ratio=(5, 3))
        for sa in self.findChildren(QtWidgets.QScrollArea):
            if sa.widget() is controls:
                sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.move_timer = QtCore.QTimer(self)
        self.move_timer.timeout.connect(self._move_step)
        self.sing_timer = QtCore.QTimer(self)
        self.sing_timer.timeout.connect(self._sing_step)
        self._sing_anim = None
        self._trace: list = []
        app = QtWidgets.QApplication.instance()
        if app is not None and not hasattr(BaseTab, "shutdown"):   # older BaseTab without the quit hook
            app.aboutToQuit.connect(self.shutdown)
        self.on_robot_changed(self.ctx.robot)

    # ================================================================== robot / state

    def _tone(self, lbl, tone):
        """Colour a label with the current palette's good/warn/bad colour (None = default)."""
        lbl.setProperty("tone", tone)
        lbl.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")

    def _retone(self, *_):
        for lbl in (self.twist_chk, self.tau_chk, self.jac_chk):
            self._tone(lbl, lbl.property("tone"))

    def on_robot_changed(self, robot):
        self._stop_timers()
        self._cancel_workspace()
        self._cancel_singularity()
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("vel.")
        self.graphic = RobotGraphic(self.view, robot, "robot")
        self.sliders.set_robot(robot)
        self.theta = np.array(robot.home, float)
        self.view.frame_scene((1.4 if robot.planar else 1.15) * robot.reach, planar=robot.planar,
                              center=(0, 0, 0.3 * robot.reach * (not robot.planar)))
        self.view.set_triad("vel.s", np.eye(4), 0.22 * robot.reach, "{s}")
        self._ell_scale = {}
        self._trace = []
        # joint-velocity inputs
        _clear_layout(self.qd_grid)
        self.qd_spins = []
        for i in range(robot.n):
            s = spin(-3.0, 3.0, 0.3 if i < 3 else 0.0, 2, 0.05)
            s.setMaximumWidth(70)
            s.valueChanged.connect(self.update_scene)
            lbl = QtWidgets.QLabel(f"θ̇{_sub(i + 1)}")
            self.qd_grid.addWidget(lbl, i // 3, 2 * (i % 3))
            self.qd_grid.addWidget(s, i // 3, 2 * (i % 3) + 1)
            self.qd_spins.append(s)
        # default wrench: push "down" (against gravity) with 20 N
        g = robot.gravity / (np.linalg.norm(robot.gravity) + 1e-12)
        for s, v in zip(self.f_spins, 20.0 * g):
            s.blockSignals(True)
            s.setValue(round(float(v), 1))
            s.blockSignals(False)
        for s in self.m_spins:
            s.blockSignals(True)
            s.setValue(0.0)
            s.blockSignals(False)
        cols = [f"J{_sub(i + 1)}" for i in range(robot.n)]
        for mv in (self.Js_view, self.Jb_view):
            mv.set_matrix(np.zeros((6, robot.n)))
            mv.set_labels(rows=TWIST_ROWS, cols=cols)
        J0, _ = kin.task_jacobian(robot, robot.home)
        self.sv_plot.setXRange(0.4, min(J0.shape) + 0.6)
        self.tau_tab.set_matrix(np.zeros((1, robot.n)))
        self.tau_tab.set_labels(rows=["τ"], cols=[f"τ{_sub(i + 1)}" for i in range(robot.n)])
        self._fit_tables()
        self.tau_plot.setXRange(0.4, robot.n + 0.6)
        self.ws_lbl.setText("")
        self.ws_bar.setValue(0)
        self.sing_lbl.setText("")
        self.update_scene()

    def _fit_tables(self):
        for mv in (self.Js_view, self.Jb_view):
            _unfix_table(mv)
        for mv in (self.twist_tab, self.tau_tab):
            _fit_table(mv)

    def showEvent(self, ev):
        super().showEvent(ev)
        QtCore.QTimer.singleShot(0, self._fit_tables)

    def _sliders_moved(self, th):
        self.theta = np.asarray(th, float)
        self.update_scene()

    def set_theta(self, th):
        self.theta = np.array(th, float)
        self.sliders.set_values(self.theta)
        self.update_scene()

    def qdot(self) -> np.ndarray:
        return np.array([s.value() for s in self.qd_spins])

    # ================================================================== main update
    def update_scene(self, *_):
        robot = self.ctx.robot
        th = self.theta
        if self.graphic is None or len(th) != robot.n or len(self.qd_spins) != robot.n:
            return
        self.graphic.update(th)
        T = kin.fk_space(robot, th)
        R, p = T[:3, :3], T[:3, 3]
        r = robot.reach
        Js = kin.jacobian_space(robot, th)
        Jb = kin.jacobian_body(robot, th)

        # --- Jacobians and the adjoint identity J_b = [Ad_Tbs] J_s ---------------------------
        dJ = np.abs(Jb - mr.Adjoint(mr.TransInv(T)) @ Js).max()
        self.Js_view.set_matrix(Js)
        self.Jb_view.set_matrix(Jb)
        self.jac_chk.setText(f"✓ J_b = [Ad_T_bs] J_s   ‖Δ‖ = {dJ:.1e}   (mr.JacobianSpace vs mr.JacobianBody)")

        # --- singular values, rank, lost directions ---------------------------------------
        self._update_singularity(robot, th, R, p)

        # --- manipulability / force ellipsoid ------------------------------------------------
        self._update_ellipsoid(robot, th)

        # --- twists ------------------------------------------------------------------------
        qd = self.qdot()
        Vs, Vb = Js @ qd, Jb @ qd
        pdot = np.cross(Vs[:3], p) + Vs[3:]
        pdot_b = R @ Vb[3:]
        dV = np.abs(mr.Adjoint(T) @ Vb - Vs).max()
        self.twist_tab.set_matrix(np.vstack([Vs, Vb]))
        self.twist_lbl.setText(f"EE point velocity ṗ = ω_s×p + v_s\n"
                               f"  = {_vec(pdot)} m/s\n  |ṗ| = {np.linalg.norm(pdot):.4f} m/s")
        self.twist_chk.setText(f"✓ V_s = [Ad_T_sb] V_b   ‖Δ‖ = {dV:.1e}\n"
                               f"✓ ṗ = R_sb v_b          ‖Δ‖ = {np.abs(pdot - pdot_b).max():.1e}")
        if self.cb_vel.isChecked():
            vec = pdot * 1.0                                        # metres travelled in 1 s
            n = np.linalg.norm(vec)
            if n > 1.2 * r:
                vec *= 1.2 * r / n
            self.view.set_arrow("vel.pdot", p, vec, theme.GOOD, 0.35 * robot.link_radius)
            self.view.set_label("vel.pdot.lbl", p + vec * 1.05, f"ṗ {np.linalg.norm(pdot):.2f} m/s", theme.GOOD)
            w = Vs[:3] * 0.3 * r
            wn = np.linalg.norm(w)
            if wn > r:
                w *= r / wn
            self.view.set_arrow("vel.omega", p, w, (0.75, 0.55, 1.0, 1.0), 0.3 * robot.link_radius)
        else:
            for k in ("vel.pdot", "vel.omega"):
                self.view.remove_prefix(k)
        if self.move_timer.isActive():
            self._trace.append(p.copy())
            self._trace = self._trace[-600:]
            if len(self._trace) > 1:
                self.view.set_polyline("vel.trace", self._trace, (0.3, 0.8, 1.0, 0.9), 2)

        # --- statics -------------------------------------------------------------------------
        f = np.array([s.value() for s in self.f_spins])
        m = np.array([s.value() for s in self.m_spins])
        tau = kin.joint_torques_from_wrench(robot, th, f, m)
        Fs = np.r_[m + np.cross(p, f), f]
        Fb = mr.Adjoint(T).T @ Fs
        tau_b = Jb.T @ Fb
        util = np.abs(tau) / robot.tau_max
        self.tau_tab.set_matrix(tau[None, :])
        self.tau_lbl.setText(
            f"F_s = (m_s + p×f, f):\n  m = {_vec(Fs[:3], 2)} N·m\n  f = {_vec(Fs[3:], 2)} N\n"
            f"τ = J_sᵀF_s [N·m];  max |τᵢ|/τmax,ᵢ = {100 * util.max():.0f} %")
        self.tau_chk.setText(f"✓ τ = J_bᵀF_b, F_b = [Ad_T_sb]ᵀF_s\n   agrees: ‖Δτ‖ = {np.abs(tau - tau_b).max():.1e}")
        brushes = [pg.mkBrush(theme.BAD if util[i] > 1 else theme.SERIES[i % len(theme.SERIES)])
                   for i in range(robot.n)]
        self.tau_bars.setOpts(x=np.arange(1, robot.n + 1), height=tau, width=0.6, brushes=brushes)
        ylim = max(1.0, 1.15 * np.abs(tau).max())
        self.tau_plot.setYRange(-ylim, ylim)
        if self.cb_force.isChecked() and np.linalg.norm(f) > 1e-9:
            L = 0.3 * r
            self.view.set_arrow("vel.force", p, L * f / np.linalg.norm(f), theme.JOINT, 0.4 * robot.link_radius)
            self.view.set_label("vel.force.lbl", p + 1.08 * L * f / np.linalg.norm(f),
                                f"f {np.linalg.norm(f):.1f} N", theme.JOINT)
        else:
            self.view.remove_prefix("vel.force")
        if self.cb_force.isChecked() and np.linalg.norm(m) > 1e-9:
            self.view.set_arrow("vel.moment", p, 0.25 * r * m / np.linalg.norm(m), (1.0, 0.45, 0.75, 1.0),
                                0.3 * robot.link_radius)
        else:
            self.view.remove_prefix("vel.moment")

    # ------------------------------------------------------------------ singularity analysis
    def _update_singularity(self, robot, th, R, p):
        J, labels = kin.task_jacobian(robot, th)
        # dimensionless J~ = diag(I, I/L) J: sigma_min of the raw J would mix rad/s and m/s (MR Sec. 5.4)
        J = np.array([1.0 / robot.reach if lab.startswith("v") else 1.0 for lab in labels])[:, None] * J
        U, s, _ = np.linalg.svd(J)
        sw, sv = kin.split_singular_values(robot, th)
        k_w, k_v = min(robot.n, len(sw)), min(robot.n, len(sv))
        blocks = ("  ·  σ_min(J_ω) = " + (f"{sw[k_w - 1]:.3g}" if k_w else "—") + " rad/s per rad/s, "
                  f"σ_min(J_v) = {sv[k_v - 1]:.3g} m/s per rad/s")
        m = min(J.shape)
        tol = self.tol_spin.value()
        rank = int(np.sum(s > tol))
        x = np.arange(1, len(s) + 1)
        brushes = [pg.mkBrush(theme.BAD if v < tol else theme.ACCENT) for v in s]
        self.sv_bars.setOpts(x=x, height=s, width=0.6, brushes=brushes)
        self.tol_line.setValue(tol)
        self.sv_plot.setYRange(0, max(0.2, 1.1 * s.max()))
        task = "planar task (" + ", ".join(labels) + ")" if robot.planar else "6-D twist"
        self.view.remove_prefix("vel.lost")
        if s[m - 1] < tol:
            parts = []
            for k, i in enumerate(range(rank, m)):
                u = U[:, i].copy()
                if robot.planar:
                    lab = labels
                    lin = np.zeros(3)
                    ang = np.zeros(3)
                    for j, name in enumerate(labels):
                        if name == "ω_z":
                            ang[2] = u[j]
                        elif name == "v_x":
                            lin[0] = u[j]
                        elif name == "v_y":
                            lin[1] = u[j]
                else:   # body twist -> EE-centred frame with {s}-aligned axes
                    ang, lin = R @ u[:3], R @ u[3:]
                    u = np.r_[ang, lin]
                    lab = MIXED_LABELS
                big = np.argmax(np.abs(u))
                if u[big] < 0:
                    u, ang, lin = -u, -ang, -lin
                terms = [f"{u[j]:+.2f}·{lab[j]}" for j in np.argsort(-np.abs(u)) if abs(u[j]) > 0.15][:4]
                parts.append(" ".join(terms))
                L = 0.35 * robot.reach
                if np.linalg.norm(lin) > 0.25:
                    d = lin / np.linalg.norm(lin)
                    self.view.set_arrow(f"vel.lost{k}.a", p, L * d, theme.BAD, 0.3 * robot.link_radius)
                    self.view.set_arrow(f"vel.lost{k}.b", p, -L * d, theme.BAD, 0.3 * robot.link_radius)
                if np.linalg.norm(ang) > 0.25:
                    d = ang / np.linalg.norm(ang)
                    self.view.set_polyline(f"vel.lost{k}.w", [p - 1.3 * L * d, p + 1.3 * L * d],
                                           (1.0, 0.42, 0.42, 0.9), 3)
            self.banner.setText(f"⚠  SINGULAR CONFIGURATION — rank J = {rank} < {m}, σ_min(J̃) = {s[m - 1]:.2e}"
                                f"{blocks}.  Lost directions (red, v scaled by 1/L): " + "  |  ".join(parts))
            self.banner.setStyleSheet(f"background:#5c1d24; color:#ffffff; font-weight:700; padding:5px 8px;"
                                      f"border:1px solid {theme.BAD}; border-radius:5px;")
        else:
            self.banner.setText(f"Regular configuration — rank J = {m} (full) for the {task},  "
                                f"σ_min(J̃) = {s[m - 1]:.4f} > tol {tol:g}{blocks}")
            self.banner.setStyleSheet(f"background:#183326; color:{theme.GOOD}; font-weight:600; padding:5px 8px;"
                                      "border:1px solid #2e6b4a; border-radius:5px;")

    # ------------------------------------------------------------------ ellipsoid
    def _update_ellipsoid(self, robot, th):
        kind = "linear" if self.ell_kind.currentIndex() == 0 else "angular"
        force = self.ell_type.currentIndex() == 1
        el = kin.manipulability(robot, th, kind, force)
        key = (kind, force)
        if key not in self._ell_scale:
            rv = kin.manipulability(robot, robot.home, kind, False).radii
            nz = rv[rv > 1e-6]
            g = float(np.exp(np.mean(np.log(nz)))) if nz.size else 1.0      # geometric mean of semi-axes
            self._ell_scale[key] = 0.2 * robot.reach * (g if force else 1.0 / g)
        k = self._ell_scale[key]
        inf = el.unbounded                                       # force ellipsoid at a singularity
        clamp = 1.5 * robot.reach
        radii = np.clip(np.where(inf, clamp, el.radii * k), 0.0, clamp)
        sym = "B" if force else "A"
        planar_ang = robot.planar and kind == "angular"
        mu3 = (1.0 / el.mu3 if el.mu3 > 1e-12 else np.inf) if force else el.mu3
        semi = ", ".join("∞" if not np.isfinite(r) else f"{r:+.3f}" for r in np.sort(el.radii)[::-1])
        txt = (f"μ₁({sym}) = {el.mu1:9.3f}  isotropy    (§5.4)\n"
               f"μ₂({sym}) = {el.mu2:9.3f}  condition   (§5.4)\n"
               f"μ₃({sym}) = {mu3:9.4f}  volume      (§5.4)\n"
               f"semi-axes ({semi})")
        if force:
            txt += "\nB = A⁻¹: same axes, lengths 1/σᵢ"
        if inf.any():
            txt += "\n∞: singular direction — any force along it is resisted with zero torque (drawn clamped, red)"
        if planar_ang:
            txt += "\nplanar arm: ω = ω_z only → segment along z"
        self.mu_lbl.setText(txt)
        self.view.remove_prefix("vel.ellinf")
        if self.cb_ell.isChecked():
            col = (1.0, 0.62, 0.25, 0.33) if force else (0.3, 0.79, 0.94, 0.33)
            self.view.set_ellipsoid("vel.ell", el.center, el.axes, radii, col)
            pts, pinf = [], []
            for i in range(3):
                if radii[i] > 1e-4:
                    seg = [el.center - radii[i] * el.axes[:, i], el.center + radii[i] * el.axes[:, i]]
                    (pinf if inf[i] else pts).extend(seg)
            if pts:
                self.view.set_polyline("vel.ellax", pts, (*col[:3], 0.95), 2, mode="lines")
            else:
                self.view.remove("vel.ellax")
            if pinf:
                self.view.set_polyline("vel.ellinf.ax", pinf, (1.0, 0.35, 0.35, 0.95), 3, mode="lines")
                for j, q in enumerate(pinf[1::2]):
                    self.view.set_label(f"vel.ellinf.lbl{j}", q, "∞", theme.BAD)
        else:
            self.view.remove("vel.ell")
            self.view.remove("vel.ellax")

    # ================================================================== nearest singularity
    def find_singularity(self):
        robot = self.ctx.robot
        self._stop_timers()
        th0 = self.theta.copy()
        lo = np.maximum(robot.joint_limits[:, 0], -2 * np.pi)
        hi = np.minimum(robot.joint_limits[:, 1], 2 * np.pi)
        bounds = list(zip(lo, hi))
        th_start = np.clip(th0, lo, hi)

        def job(progress, cancelled):
            def smin2(th):
                if cancelled():
                    raise InterruptedError
                return kin.scaled_singular_values(robot, th)[-1] ** 2

            r1 = minimize(lambda th: smin2(th) + 0.02 * np.sum((th - th0) ** 2), th_start, method="L-BFGS-B",
                          bounds=bounds, options=dict(maxiter=300))
            progress(0.5)
            r2 = minimize(smin2, r1.x, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=300, ftol=1e-18, gtol=1e-12))
            th1 = r2.x
            s1 = kin.scaled_singular_values(robot, th1)[-1]
            snapped = np.where(np.abs(th1 - th0) < 2e-3, th0, th1)     # joints the optimiser did not use
            if kin.scaled_singular_values(robot, snapped)[-1] <= max(10 * s1, 1e-8):
                th1 = snapped
            return th1, r1.nfev + r2.nfev

        self.sing_btn.setEnabled(False)
        self.sing_lbl.setText("searching (L-BFGS-B in a background thread) …")
        self.sing_jobs.run(job, on_done=lambda out, name=robot.name, th0=th0: self._sing_done(name, th0, *out),
                           on_error=self._sing_failed, label="nearest singularity")

    def _sing_failed(self, err):
        self.sing_btn.setEnabled(True)
        self.sing_lbl.setText("search failed:\n" + err.strip().splitlines()[-1])

    def _sing_done(self, name, th0, th1, nfev):
        self.sing_btn.setEnabled(True)
        robot = self.ctx.robot
        if name != robot.name or len(th0) != robot.n:
            return
        s0 = kin.scaled_singular_values(robot, th0)[-1]
        s1 = kin.scaled_singular_values(robot, th1)[-1]
        d = np.degrees(th1 - th0)
        moved = ", ".join(f"θ{_sub(i + 1)} {d[i]:+.1f}°" for i in np.argsort(-np.abs(d)) if abs(d[i]) > 0.5)
        self.sing_lbl.setText(f"σ_min(J̃): {s0:.3e} → {s1:.2e}  ({nfev} evaluations)\n"
                              f"‖Δθ‖ = {np.linalg.norm(d):.1f}°:  {moved or 'already singular'}")
        self._sing_anim = dict(th0=self.theta.copy(), th1=th1, k=0, N=50)
        self.sing_timer.start(30)

    def _cancel_singularity(self):
        if self.sing_jobs.is_busy():
            self.sing_jobs.cancel()
            self.sing_lbl.setText("")
        self.sing_btn.setEnabled(True)

    def _sing_step(self):
        a = self._sing_anim
        if a is None:
            self.sing_timer.stop()
            return
        a["k"] += 1
        s = min(1.0, a["k"] / a["N"])
        s = s * s * (3 - 2 * s)                                      # smoothstep
        self.set_theta(a["th0"] + s * (a["th1"] - a["th0"]))
        if a["k"] >= a["N"]:
            self.sing_timer.stop()
            self._sing_anim = None

    # ================================================================== move with θ̇
    def toggle_move(self, on):
        self.move_btn.setText("■ Stop" if on else "▶ Move")
        if on:
            self.sing_timer.stop()
            self._trace = []
            self.move_timer.start(30)
        else:
            self.move_timer.stop()
            self.view.remove("vel.trace")

    def _move_step(self):
        robot = self.ctx.robot
        qd = self.qdot()
        if len(qd) != len(self.theta):
            return
        th = self.theta + 0.03 * qd
        lo = np.maximum(robot.joint_limits[:, 0], -2 * np.pi)
        hi = np.minimum(robot.joint_limits[:, 1], 2 * np.pi)
        hit = np.flatnonzero((th <= lo) | (th >= hi))
        if hit.size:
            self.move_btn.setChecked(False)
            self.sing_lbl.setText("Move stopped: " + ", ".join(f"θ{_sub(i + 1)}" for i in hit)
                                  + (" reached its limit" if hit.size == 1 else " reached their limits"))
            return
        self.set_theta(th)

    # ================================================================== workspace
    def compute_workspace(self):
        robot = self.ctx.robot
        self._cancel_workspace()
        self.view.remove("vel.ws")
        self._ws_P, self._ws_W = [], []
        n = self.ws_n.value()
        chunk = 500
        nchunks = int(np.ceil(n / chunk))
        buf: list = []          # chunks of *this* job only (appended by the worker, read by the GUI thread)

        def job(progress, cancelled):
            for k in range(nchunks):
                if cancelled():
                    return None
                buf.append(kin.workspace_samples(robot, min(chunk, n - k * chunk), seed=1000 + k))
                progress((k + 1) / nchunks)
            return robot.name

        self.ws_btn.setEnabled(False)
        self.ws_bar.setRange(0, nchunks)
        self.ws_bar.setValue(0)
        self.ws_lbl.setText("sampling in a background thread …")
        # the runner drops progress / results of superseded jobs, and each job has its own buffer,
        # so chunks of a cancelled job can never leak into the next cloud
        self.ws_jobs.run(job, on_done=lambda name, buf=buf: self._ws_done(name, buf),
                         on_error=self._ws_failed,
                         on_progress=lambda f, buf=buf, name=robot.name: self._ws_progress(name, buf),
                         label="workspace")

    def _ws_failed(self, err):
        self.ws_btn.setEnabled(True)
        self.ws_lbl.setText("workspace failed:\n" + err[-300:])

    def _ws_progress(self, name, buf):
        if name != self.ctx.robot.name:
            return
        chunks = list(buf)
        if not chunks:
            return
        self._ws_P = [c[0] for c in chunks]
        self._ws_W = [c[1] for c in chunks]
        P = np.vstack(self._ws_P)
        W = np.concatenate(self._ws_W)
        wmax = max(W.max(), 1e-12)
        self.view.set_points("vel.ws", P, _cmap(W / wmax), 3.5)
        self.ws_bar.setValue(len(chunks))
        self.ws_hi.setText(f"μ₃ = {wmax:.3f}")
        self.ws_lbl.setText(f"{len(P)} samples · reach of cloud {np.linalg.norm(P, axis=1).max():.3f} m")

    def _ws_done(self, name, buf):
        self.ws_btn.setEnabled(True)
        if name is None or name != self.ctx.robot.name:
            return
        self._ws_progress(name, buf)
        if not self._ws_P:
            return
        P = np.vstack(self._ws_P)
        W = np.concatenate(self._ws_W)
        self.ws_lbl.setText(f"{len(P)} samples · max reach {np.linalg.norm(P, axis=1).max():.3f} m · "
                            f"{100 * np.mean(W < 0.05 * W.max()):.1f} % of samples near-singular (μ₃ < 5 % of max)")

    def clear_workspace(self):
        self._cancel_workspace()
        self.view.remove("vel.ws")
        self._ws_P, self._ws_W = [], []
        self.ws_bar.setValue(0)
        self.ws_lbl.setText("")

    def _cancel_workspace(self):
        if self.ws_jobs.is_busy():
            self.ws_jobs.cancel()
        self.ws_btn.setEnabled(True)

    def shutdown(self):
        """Application quit: stop timers and cancel both background jobs (brief waits only)."""
        self._stop_timers()
        self.ws_jobs.shutdown(500)
        self.sing_jobs.shutdown(500)
        parent = getattr(super(), "shutdown", None)
        if parent is not None:
            parent()

    # ================================================================== show / hide
    def _stop_timers(self):
        self.sing_timer.stop()
        self._sing_anim = None
        if self.move_btn.isChecked():
            self.move_btn.setChecked(False)

    def on_hide(self):
        self._stop_timers()
