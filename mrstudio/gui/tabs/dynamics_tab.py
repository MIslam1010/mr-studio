"""Tab 4 -- inverse and forward dynamics of open chains (MR Ch. 8).

Re-exports helpers shared with the control tabs (defined in gui/jobs.py and gui/widgets.py):
  * AsyncMixin -- one background job per tab (JobRunner), progress bar + cancel, stale results dropped
  * Player     -- QTimer playback of a precomputed time series (play/pause, time slider, speed)
"""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import trajectory as tj
from ...dynamics import DynamicsEngine, simulate
from ...robots import get_robot
from .. import theme
from ..base import BaseTab
from ..jobs import AsyncMixin
from ..viewer import GhostRobot, RobotGraphic, SceneView
from ..widgets import (JointSliders, Player, button, checkbox, combo, compact, default_goal, form, group, hbox,
                       limit_lines, no_si, pen, plot, spin, time_cursor, value_label)

THEORY = r"""
<h2>Robot dynamics — Newton–Euler &amp; Lagrange</h2>
<span class="ref">Modern Robotics, Ch. 8 (8.1 Lagrangian, 8.3 Newton–Euler, 8.4 closed form, 8.5 forward dynamics)</span>
<p>Every open chain obeys the same equations of motion (MR Eq. 8.77):</p>
$$\tau=M(\theta)\ddot{\theta}+c(\theta,\dot{\theta})+g(\theta)+J^{T}(\theta)\mathcal{F}_{tip}\quad(8.77)$$
<p>$M$ = mass matrix, $c$ = Coriolis &amp; centripetal torques, $g$ = gravity torques, $\mathcal{F}_{tip}$ = wrench the
end-effector applies to the environment.  The <b>Decomposition</b> plot shows the four terms summing to $\tau$.</p>
<h3>Lagrangian formulation (8.1)</h3>
$$\mathcal{L}=\mathcal{K}-\mathcal{P},\qquad \tau=\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot{\theta}}-\frac{\partial\mathcal{L}}{\partial\theta}\quad(8.15)$$
$$\mathcal{K}=\frac{1}{2}\dot{\theta}^{T}M(\theta)\dot{\theta},\qquad c=\dot{\theta}^{T}\Gamma(\theta)\dot{\theta}\quad(8.14),\ (8.18)$$
$$\Gamma_{ijk}=\frac{1}{2}\left(\frac{\partial m_{ij}}{\partial\theta_k}+\frac{\partial m_{ik}}{\partial\theta_j}-\frac{\partial m_{jk}}{\partial\theta_i}\right)\quad(8.17)$$
<p>Elegant but costly (symbolic derivatives).  The <b>Newton–Euler</b> recursion obtains the same $\tau$ in $O(n)$.</p>
<h3>Recursive Newton–Euler (RNEA, 8.3.2)</h3>
<p>Link frames {i} sit at the centres of mass; $\mathcal{A}_i$ = screw axis of joint $i$ in {i},
$\mathcal{G}_i$ = 6×6 spatial inertia.  Initialise $\mathcal{V}_0=0$, $\dot{\mathcal{V}}_0=(0,\,-g)$,
$\mathcal{F}_{n+1}=\mathcal{F}_{tip}$.</p>
<p><b>Forward iterations</b>  $i=1\ldots n$:</p>
$$T_{i,i-1}=e^{-[\mathcal{A}_i]\theta_i}M_{i,i-1}\quad(8.50)$$
$$\mathcal{V}_i=[\mathrm{Ad}_{T_{i,i-1}}]\mathcal{V}_{i-1}+\mathcal{A}_i\dot{\theta}_i\quad(8.51)$$
$$\dot{\mathcal{V}}_i=[\mathrm{Ad}_{T_{i,i-1}}]\dot{\mathcal{V}}_{i-1}+[\mathrm{ad}_{\mathcal{V}_i}]\mathcal{A}_i\dot{\theta}_i+\mathcal{A}_i\ddot{\theta}_i\quad(8.52)$$
<p><b>Backward iterations</b>  $i=n\ldots 1$:</p>
$$\mathcal{F}_i=[\mathrm{Ad}_{T_{i+1,i}}]^{T}\mathcal{F}_{i+1}+\mathcal{G}_i\dot{\mathcal{V}}_i-[\mathrm{ad}_{\mathcal{V}_i}]^{T}\mathcal{G}_i\mathcal{V}_i\quad(8.53)$$
$$\tau_i=\mathcal{F}_i^{T}\mathcal{A}_i\quad(8.54)$$
<p>The gravity trick: a fictitious upward acceleration $-g$ of the base produces the gravity torques for free.</p>
<h3>Mass matrix from n RNEA calls (8.5)</h3>
$$M(\theta)\,e_j=\mathrm{ID}(\theta,\ \dot{\theta}=0,\ \ddot{\theta}=e_j;\ g=0,\ \mathcal{F}_{tip}=0)$$
<p>That is <b>mr.MassMatrix</b>; Pinocchio uses the Composite-Rigid-Body Algorithm (same matrix, ~1000× faster —
timings in the <b>Mass matrix</b> tab).</p>
<h3>Forward dynamics (8.5)</h3>
$$\ddot{\theta}=M^{-1}(\theta)\left(\tau-c(\theta,\dot{\theta})-g(\theta)-J^{T}(\theta)\mathcal{F}_{tip}\right)\quad(8.81)$$
<p>mr.ForwardDynamics solves this linear system; Pinocchio's Articulated-Body Algorithm is $O(n)$.
The state $(\theta,\dot{\theta})$ is integrated numerically:</p>
<p><b>Explicit Euler</b> (mr.EulerStep):</p>
$$\theta_{k+1}=\theta_k+\Delta t\,\dot{\theta}_k,\qquad \dot{\theta}_{k+1}=\dot{\theta}_k+\Delta t\,\ddot{\theta}_k$$
<p><b>Semi-implicit (symplectic) Euler</b>:</p>
$$\dot{\theta}_{k+1}=\dot{\theta}_k+\Delta t\,\ddot{\theta}_k,\qquad \theta_{k+1}=\theta_k+\Delta t\,\dot{\theta}_{k+1}$$
<p>plus classical RK4 and adaptive RK45 (Dormand–Prince 5(4), SciPy, rtol = 10⁻¹⁰).</p>
<h3>Energy — a built-in correctness test</h3>
$$\mathcal{P}=-\sum_i m_i\,g^{T}p_{c_i}(\theta),\qquad \frac{d}{dt}(\mathcal{K}+\mathcal{P})=\dot{\theta}^{T}(\tau-B\dot{\theta})$$
<p>With $\tau=0$ and no damping the total energy must stay constant: explicit Euler <i>gains</i> energy every step,
the symplectic semi-implicit Euler keeps it bounded, RK4/RK45 conserve it to many digits
(<b>Compare integrators</b>).  With gravity compensation $\tau=g(\theta)$ the robot "floats": $M\ddot{\theta}+c=0$.</p>
<h3>Structural properties (8.1.2)</h3>
$$M(\theta)=M^{T}(\theta)>0,\qquad x^{T}\left(\dot{M}-2C\right)x=0\ \ \forall x$$
<p>Positive definiteness follows from $\mathcal{K}>0$; skew-symmetry of $\dot{M}-2C$ (passivity) is what makes
PD + gravity compensation stable (Ch. 11).  All three are checked live in the <b>Mass matrix</b> tab.</p>
"""
THEORY += further_reading([
    ('featherstone', 'Ch. 2, pp. 7–38', 'spatial vector algebra (spatial inertia §2.13, p. 32)'),
    ('featherstone', '§5.3, pp. 92–97, Table 5.1 p. 96', 'recursive Newton–Euler algorithm'),
    ('featherstone', '§6.2, pp. 104–108, Table 6.2 p. 107', 'composite-rigid-body algorithm (mass matrix)'),
    ('featherstone', '§7.3, pp. 128–131, Table 7.1 p. 132', 'articulated-body algorithm (forward dynamics)'),
    ('mls', 'Ch. 4 §2–3, pp. 156–179', 'Lagrange equations; open chains; Lemma 4.2 (p. 171): M > 0, Ṁ − 2C skew'),
    ('gotor', '§6.2, pp. 152–170; §6.4–6.5, pp. 172–190', 'Lagrange with PoE; recursive Newton–Euler'),
    ('corke', '§9.1, §9.3, pp. 191–204', 'gravity, inertia, Coriolis terms; forward dynamics'),
])



# ============================================================================ shared helpers
# AsyncMixin now lives in gui/jobs.py (built on JobRunner) and Player / form / compact / default_goal / no_si /
# time_cursor / limit_lines in gui/widgets.py; they are re-exported here for older imports.
__all__ = ["AsyncMixin", "Player", "form", "compact", "default_goal", "no_si", "time_cursor", "limit_lines",
           "DynamicsTab"]


FD_LAWS = ["τ = 0  (falls freely — chaotic pendulum)", "τ = g(θ)  (gravity compensation — floats)",
           "τ = A·τmax  (constant)", "τ = A·τmax·sin(2πft + φᵢ)  (sinusoidal)"]
INTEGRATORS = {"Euler (mr.EulerStep)": "euler", "Semi-implicit (symplectic) Euler": "semi-implicit",
               "Runge–Kutta 4": "rk4", "RK45 adaptive (SciPy Dormand–Prince)": "rk45"}

#: robots offered for the integrator comparison: (label, robot name or None = current robot, Δt [s] or None = setting)
CMP_TARGETS = [("current robot, Δt from the settings", None, None),
               ("Planar 2R arm, Δt = 10 ms (drift clearly visible)", "Planar 2R arm", 0.010),
               ("UR5, Δt = 10 ms", "UR5 (6R industrial)", 0.010)]


# ============================================================================ computations (worker threads)
def compute_inverse(robot, backend, th_s, th_g, Tf, scaling, Ftip, progress, cancelled, N=201):
    """tau(t) along a point-to-point joint trajectory + its decomposition M ddq + c + g + J^T F."""
    eng = DynamicsEngine(robot, backend)        # private engine: Pinocchio data are not thread-safe
    tr = tj.make_trajectory(robot, th_s, th_g, Tf, N, "joint", scaling)
    n = robot.n
    tau, Mdd, c, g, JF = (np.zeros((N, n)) for _ in range(5))
    F = None if Ftip is None or not np.any(Ftip) else np.asarray(Ftip, float)
    t0 = time.perf_counter()
    for k in range(N):
        th, dth, ddth = tr.theta[k], tr.dtheta[k], tr.ddtheta[k]
        tau[k] = eng.inverse(th, dth, ddth, F)
        if k % 20 == 0:
            progress(0.5 * k / N)
            if cancelled():
                return None
    t_id = time.perf_counter() - t0
    for k in range(N):
        th, dth, ddth = tr.theta[k], tr.dtheta[k], tr.ddtheta[k]
        Mdd[k] = eng.mass_matrix(th) @ ddth
        g[k] = eng.gravity_torque(th)
        c[k] = coriolis_vector(eng, th, dth, g[k])
        if F is not None:
            JF[k] = eng.tip_torque(th, F)
        if k % 20 == 0:
            progress(0.5 + 0.5 * k / N)
            if cancelled():
                return None
    ee = np.array([robot.fk(th)[:3, 3] for th in tr.theta])
    return dict(tr=tr, tau=tau, Mdd=Mdd, c=c, g=g, JF=JF, ee=ee, t_id=t_id, backend=backend,
                resid=float(np.abs(tau - (Mdd + c + g + JF)).max()))


def _short_name(name: str) -> str:
    """'UR5 (6R industrial)' -> 'UR5' (label for the comparison plot)."""
    return name.split(" (")[0].replace(" arm", "")


def coriolis_vector(eng, th, dth, g=None):
    """c(theta, dtheta).  MR: mr.VelQuadraticForces.  Pinocchio: RNEA(theta, dtheta, 0) - g(theta)
    (avoids DynamicsEngine.coriolis, which leaves the Pinocchio model's gravity set to zero)."""
    if eng.backend == "mr":
        return eng.coriolis(th, dth)
    g = eng.gravity_torque(th) if g is None else g
    return eng.inverse(th, dth, np.zeros_like(th)) - g


def make_tau_fn(eng, robot, law: int, amp: float, freq: float, add_g: bool):
    tmax = robot.tau_max
    phase = np.arange(robot.n) * 0.9

    def tau_fn(t, th, dth):
        if law == 0:
            return np.zeros(robot.n)
        if law == 1:
            return eng.gravity_torque(th)
        tau = amp * tmax if law == 2 else amp * tmax * np.sin(2 * np.pi * freq * t + phase)
        return tau + eng.gravity_torque(th) if add_g else tau
    return tau_fn


def simulate_chunked(eng, th0, dth0, tau_fn, T, dt, method, damping, progress, cancelled, chunks=20, p0=0.0, p1=1.0):
    """dynamics.simulate() run in chunks so that progress can be reported and the run cancelled.

    Every chunk is a whole number of steps (steps_per_chunk = round(T/dt/chunks)), and the time offset advances
    by exactly steps·dt, so the samples sit on the same grid k·dt as one long simulate() call (no drift)."""
    total = max(1, int(round(T / dt)))
    per = max(1, int(round(total / chunks)))
    parts = []
    th, dth = np.array(th0, float), np.array(dth0, float)
    done, c = 0, 0
    n_chunks = -(-total // per)
    while done < total:
        steps = min(per, total - done)
        off = done * dt
        r = simulate(eng, th, dth, lambda t, q, dq, off=off: tau_fn(t + off, q, dq), steps * dt, dt, method,
                     damping=damping)
        r.t = r.t + off
        parts.append(r if c == 0 else _drop_first(r))
        th, dth = r.theta[-1], r.dtheta[-1]
        done += steps
        c += 1
        progress(p0 + (p1 - p0) * c / n_chunks)
        if cancelled():
            return None
    out = parts[0]
    for p in parts[1:]:
        out.t = np.r_[out.t, p.t]
        out.theta, out.dtheta, out.tau = (np.vstack([a, b]) for a, b in
                                          ((out.theta, p.theta), (out.dtheta, p.dtheta), (out.tau, p.tau)))
        out.ke, out.pe = np.r_[out.ke, p.ke], np.r_[out.pe, p.pe]
    return out


def _drop_first(r):
    r.t, r.theta, r.dtheta, r.tau = r.t[1:], r.theta[1:], r.dtheta[1:], r.tau[1:]
    r.ke, r.pe = r.ke[1:], r.pe[1:]
    return r


def diverged_at(res, limit=1e3) -> int:
    """Index of the first sample where the integration blew up (len(t) if it never did)."""
    bad = ~np.all(np.isfinite(res.dtheta), axis=1) | (np.abs(np.nan_to_num(res.dtheta, nan=1e9)).max(axis=1) > limit)
    if res.ke is not None:
        bad |= ~np.isfinite(res.ke)
    return int(np.argmax(bad)) if bad.any() else len(res.t)


def energy_balance(res, damping):
    """E(t) - W(t): total energy minus the work done by actuators and friction (constant if exact)."""
    b = np.broadcast_to(np.asarray(damping, float), (res.theta.shape[1],))
    P = np.sum((res.tau - b * res.dtheta) * res.dtheta, axis=1)
    W = np.r_[0.0, np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(res.t))]
    return res.ke + res.pe - W


# ============================================================================ the tab
class DynamicsTab(AsyncMixin, BaseTab):
    title = "④ Dynamics"
    needs_dynamics = True
    uses_engine = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._init_async()
        self.view = SceneView()
        self.graphic = None
        self.goal_ghost = None
        self.anim = None          # ("id", dict) or ("fd", dict)

        # ---- configuration -------------------------------------------------------------
        self.start_sl = JointSliders()
        self.goal_sl = JointSliders()
        self.start_sl.changed.connect(self._start_changed)
        self.goal_sl.changed.connect(lambda _: self._draw_static())
        self.cfg_tabs = QtWidgets.QTabWidget()
        self.cfg_tabs.addTab(self.start_sl, "Start θ / initial θ₀")
        self.cfg_tabs.addTab(self.goal_sl, "Goal θ (inverse dyn.)")
        cfg_btns = hbox(button("Home", lambda: self._set_start(self.ctx.robot.home)),
                        button("Random", lambda: self._set_start(self.ctx.robot.random_config())),
                        button("Random goal", lambda: self._set_goal(self.ctx.robot.random_config())))

        # ---- inverse dynamics -------------------------------------------------------------
        self.Tf = spin(0.2, 20, 2.0, 2, 0.1, " s")
        self.scaling = combo(["quintic", "cubic", "trapezoidal"])
        self.fx, self.fy, self.fz = (spin(-500, 500, 0.0, 1, 5, "") for _ in range(3))
        for w in (self.fx, self.fy, self.fz):
            w.setMinimumWidth(60)
        self.dec_joint = combo([])
        self.dec_joint.currentIndexChanged.connect(lambda _: self._plot_decomposition())
        self.id_btn = button("Compute τ(t) by RNEA", self.run_inverse, primary=True)
        self.id_lbl = value_label()
        ftip_lbl = QtWidgets.QLabel("F_tip = force applied by the end-effector, in {b}  (fx, fy, fz) [N]")
        ftip_lbl.setObjectName("muted")
        ftip_lbl.setWordWrap(True)
        id_box = group("Inverse dynamics — RNEA (MR 8.3)",
                       form(("Duration T", self.Tf), ("Time scaling", self.scaling),
                            ("Decompose joint", self.dec_joint)),
                       ftip_lbl, hbox(self.fx, self.fy, self.fz),
                       self.id_btn, self.id_lbl)

        # ---- forward dynamics --------------------------------------------------------------
        self.law = combo(FD_LAWS)
        compact(self.law, 22)
        self.amp = spin(-100, 100, 10, 0, 5, " % τmax")
        self.freq = spin(0.05, 5, 0.5, 2, 0.05, " Hz")
        self.add_g = checkbox("+ g(θ)  (add gravity compensation)", None, True)
        self.dth0 = spin(-360, 360, 0.0, 0, 10, " °/s")
        self.damp = spin(0, 50, 0.0, 2, 0.1, " N·m·s/rad")
        self.integ = combo(list(INTEGRATORS))
        compact(self.integ, 18)
        self.integ.setCurrentIndex(2)
        self.dt = spin(0.5, 50, 2.0, 1, 0.5, " ms")
        self.Tsim = spin(0.2, 30, 4.0, 1, 0.5, " s")
        self.law.currentIndexChanged.connect(self._law_changed)
        self.fd_btn = button("Simulate", self.run_forward, primary=True)
        self.cmp_btn = button("Compare integrators", self.compare_integrators,
                              tooltip="τ = 0, no damping: total energy must be conserved")
        self.cmp_target = compact(combo([lab for lab, _, _ in CMP_TARGETS]), 18)
        self.cmp_target.setToolTip("Robot / time step used by “Compare integrators”. Explicit Euler's energy drift "
                                   "is easiest to see on the planar 2R arm with Δt = 10 ms.")
        self.fd_lbl = value_label()
        fd_box = group("Forward dynamics — simulation (MR 8.5)",
                       form(("Torque law", self.law), ("Amplitude A", self.amp), ("Frequency f", self.freq),
                            self.add_g, ("Initial θ̇₀ (all joints)", self.dth0), ("Viscous damping b", self.damp),
                            ("Integrator", self.integ), ("Time step Δt", self.dt), ("Duration", self.Tsim),
                            ("Compare on", self.cmp_target)),
                       hbox(self.fd_btn, self.cmp_btn), self.fd_lbl)

        # ---- display / playback -------------------------------------------------------------
        self.player = Player(self)
        self.player.frame.connect(self.show_frame)
        self.cb_com = checkbox("Link centres of mass (size ∝ mass)", lambda _: self._draw_static(), True)
        self.cb_trace = checkbox("End-effector trace", lambda _: self._redraw_frame(), True)
        self.status_lbl = value_label()

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Configuration", self.cfg_tabs, cfg_btns))
        cl.addWidget(id_box)
        cl.addWidget(fd_box)
        cl.addWidget(group("Playback", self.player.widget, self.busy_row, self.cb_com, self.cb_trace,
                           self.status_lbl))
        cl.addStretch(1)

        # ---- plots -----------------------------------------------------------------------------
        self.p_tau = plot("Joint torques τ(t) from RNEA  (dashed: ±τmax)", "τ [N·m]")
        self.p_dec = plot("τ = M(θ)θ̈ + c(θ,θ̇) + g(θ) + Jᵀ(θ)F_tip", "τ [N·m]")
        self.p_th = plot("Forward dynamics: joint angles θ(t)", "θ [deg]")
        self.p_en = plot("Energy  (𝒦 kinetic, 𝒫 potential)", "E [J]")
        self.p_cmp = plot("Integrator comparison: E(t) − E(0), τ = 0, no damping", "E(t) − E(0) [J]", legend=False)
        self.cursors = [time_cursor(p) for p in (self.p_tau, self.p_dec, self.p_th, self.p_en)]
        for p in (self.p_tau, self.p_dec, self.p_th, self.p_en, self.p_cmp):
            no_si(p)
        self.cmp_lbl = value_label()
        cmp_page = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(cmp_page)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.p_cmp, 1)
        cv.addWidget(self.cmp_lbl)
        self.mass_page = self._build_mass_page()
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(self.p_tau, "τ(t) inverse dynamics")
        self.bottom.addTab(self.p_dec, "Decomposition")
        self.bottom.addTab(self.p_th, "θ(t) forward dynamics")
        self.bottom.addTab(self.p_en, "Energy")
        self.bottom.addTab(cmp_page, "Compare integrators")
        self.bottom.addTab(self.mass_page, "Mass matrix M(θ)")
        self.bottom.currentChanged.connect(lambda _: self._mass_timer.start())

        controls.setMinimumWidth(330)
        self.assemble(controls, self.view, self.bottom, THEORY, controls_width=375, bottom_ratio=(5, 4))
        self._mass_timer = QtCore.QTimer(self)
        self._mass_timer.setSingleShot(True)
        self._mass_timer.setInterval(60)
        self._mass_timer.timeout.connect(self.update_mass_panel)
        self._law_changed()
        self.on_robot_changed(ctx.robot)

    # ------------------------------------------------------------------ mass-matrix page
    def _build_mass_page(self):
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.mplot = self.glw.addPlot(title="M(θ) at the start configuration")
        self.mplot.invertY(True)
        self.mplot.setAspectLocked(True)
        self.mplot.hideButtons()
        self.mplot.setMouseEnabled(False, False)
        self.mimg = pg.ImageItem(axisOrder="row-major")
        self.mplot.addItem(self.mimg)
        try:
            cmap = pg.colormap.get("CET-D1")
        except Exception:
            cmap = pg.colormap.get("viridis")
        self.mbar = pg.ColorBarItem(values=(-1, 1), colorMap=cmap, width=14, interactive=False)
        self.mbar.setImageItem(self.mimg, insert_in=self.mplot)
        self.mtexts: list = []
        self.mass_lbl = value_label()
        self.mass_lbl.setTextFormat(QtCore.Qt.RichText)
        self.mass_lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        sc = QtWidgets.QScrollArea()
        sc.setWidget(self.mass_lbl)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        h.addWidget(self.glw, 3)
        h.addWidget(sc, 2)
        return page

    def update_mass_panel(self):
        robot = self.ctx.robot
        th = self.start_sl.values()
        if len(th) != robot.n or self.bottom.currentWidget() is not self.mass_page:
            return
        ep = self.ctx.engine(robot, "pinocchio")
        em = self.ctx.engine(robot, "mr")
        t0 = time.perf_counter()
        for _ in range(50):
            Mp = ep.mass_matrix(th)
        t_pin = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        Mm = em.mass_matrix(th)
        t_mr = time.perf_counter() - t0
        M = Mp if self.ctx.backend == "pinocchio" else Mm
        n = robot.n
        vmax = float(np.abs(M).max())
        self.mimg.setImage(M, autoLevels=False)
        self.mimg.setRect(QtCore.QRectF(0, 0, n, n))
        self.mbar.setLevels((-vmax, vmax))
        if len(self.mtexts) != n * n:
            for it in self.mtexts:
                self.mplot.removeItem(it)
            self.mtexts = []
            for i in range(n):
                for j in range(n):
                    it = pg.TextItem("", anchor=(0.5, 0.5), color="#101010")
                    it.setPos(j + 0.5, i + 0.5)
                    self.mplot.addItem(it)
                    self.mtexts.append(it)
            ticks = [[(i + 0.5, str(i + 1)) for i in range(n)]]
            self.mplot.getAxis("left").setTicks(ticks)
            self.mplot.getAxis("bottom").setTicks(ticks)
            self.mplot.setRange(xRange=(0, n), yRange=(0, n), padding=0.02)
        for i in range(n):
            for j in range(n):
                v = M[i, j] if abs(M[i, j]) >= 5e-4 else 0.0
                self.mtexts[i * n + j].setText(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
                self.mtexts[i * n + j].setColor("#101010" if abs(v) < 0.6 * vmax else "#f5f5f5")
        lam = np.linalg.eigvalsh(M)
        sym = float(np.abs(M - M.T).max())
        # passivity: (M_dot - 2C) skew-symmetric, M_dot by central differences along a random velocity
        rng = np.random.default_rng(0)
        dq = rng.uniform(-1, 1, n)
        h = 1e-6
        Md = (ep.mass_matrix(th + h * dq) - ep.mass_matrix(th - h * dq)) / (2 * h)
        Nm = Md - 2 * ep.coriolis_matrix(th, dq)
        skew = float(np.abs(Nm + Nm.T).max())
        ok, bad = f"<span style='color:{theme.good_color()}'>✓</span>", f"<span style='color:{theme.bad_color()}'>✗</span>"
        eng = "Pinocchio (CRBA)" if self.ctx.backend == "pinocchio" else "Modern Robotics (n RNEA calls)"
        self.mass_lbl.setText(
            f"<b>Engine shown:</b> {eng}<br>"
            f"<b>Eigenvalues λ(M)</b> [kg·m²]:<br>&nbsp;&nbsp;{', '.join(f'{x:.4g}' for x in lam)}<br>"
            f"symmetric M = Mᵀ: {ok if sym < 1e-9 else bad} ‖M−Mᵀ‖ = {sym:.1e}<br>"
            f"positive-definite: {ok if lam.min() > 0 else bad} λmin = {lam.min():.4g}<br>"
            f"condition number κ = λmax/λmin = {lam.max() / lam.min():.1f}<br>"
            f"Ṁ − 2C skew-symmetric: {ok if skew < 1e-5 else bad} ‖N+Nᵀ‖ = {skew:.1e}<br>"
            f"engines agree: {ok if np.abs(Mp - Mm).max() < 1e-9 else bad} ‖M_MR − M_Pin‖ = {np.abs(Mp - Mm).max():.1e}"
            f"<br><br><b>Computation time of M(θ)</b><br>"
            f"&nbsp;&nbsp;Modern Robotics (n = {n} RNEA calls): {t_mr * 1e3:.2f} ms<br>"
            f"&nbsp;&nbsp;Pinocchio CRBA (C++): {t_pin * 1e6:.1f} µs<br>"
            f"&nbsp;&nbsp;measured speed-up ×{t_mr / max(t_pin, 1e-9):.0f} (typically ~1000×; depends on n "
            f"and the machine)<br><br>"
            f"<span style='color:{theme.muted_color()}'>Diagonal mᵢᵢ = apparent inertia of joint i; off-diagonal terms "
            f"couple the joints (MR 8.1.3). Move the start sliders to watch M(θ) change.</span>")

    # ------------------------------------------------------------------ robot / engine
    def on_robot_changed(self, robot):
        self.cancel_jobs()
        self.player.stop()
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("dyn.")
        self.graphic = RobotGraphic(self.view, robot, "dyn.robot")
        self.goal_ghost = GhostRobot(self.view, robot, "dyn.goal", (0.55, 1.0, 0.6, 0.45))
        self.start_sl.set_robot(robot)
        self.goal_sl.set_robot(robot, default_goal(robot))
        self.dec_joint.blockSignals(True)
        self.dec_joint.clear()
        self.dec_joint.addItems([f"joint {i + 1}" for i in range(robot.n)])
        self.dec_joint.setCurrentIndex(1 if robot.n > 1 else 0)
        self.dec_joint.blockSignals(False)
        # small arms show Euler's drift with the current settings; for big ones default to the planar 2R demo
        self.cmp_target.setCurrentIndex(0 if robot.n <= 3 else 1)
        self.view.frame_scene(robot.reach, planar=robot.planar,
                              center=(0, 0, 0.25 * robot.reach * (not robot.planar)))
        gdir = robot.gravity / np.linalg.norm(robot.gravity)
        base = np.array([-0.85, 0.9, 0.0]) * robot.reach if robot.planar else np.array([0.55, -0.55, 0.75]) * robot.reach
        self.view.set_arrow("dyn.g", base, 0.22 * robot.reach * gdir, theme.WARN, 0.01 * robot.reach, head=0.35)
        self.view.set_label("dyn.glbl", base + 0.04 * robot.reach * np.array([1, 0, 0]) - 0.1 * robot.reach * gdir,
                            "g", theme.WARN)
        self.anim = None
        for p in (self.p_tau, self.p_dec, self.p_th, self.p_en, self.p_cmp):
            self._clear_plot(p)
        self.id_lbl.setText("Plan a start→goal motion and compute the torques it needs.")
        self.fd_lbl.setText("Release the robot from θ₀ and integrate θ̈ = FD(θ, θ̇, τ).")
        self.cmp_lbl.setText("")
        self.player.lbl.setText("no motion computed yet")
        self.status(f"{robot.name}: {robot.n} joints, link masses "
                    f"{', '.join(f'{m:.2f}' for m in robot.masses)} kg")
        self._draw_static()
        self._mass_timer.start()
        QtCore.QTimer.singleShot(50, self.run_inverse)      # show something meaningful right away

    def on_backend_changed(self, backend):
        self._mass_timer.start()
        name = "Pinocchio" if backend == "pinocchio" else "Modern Robotics"
        self.status(f"Dynamics engine: {name}" + ("  (pure Python — long simulations run in the background)"
                                                   if backend == "mr" else ""))

    def status(self, text, color=None):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color:{color}" if color else "")

    def _clear_plot(self, p):
        p.clear()
        if p.plotItem.legend is not None:
            p.plotItem.legend.clear()
        idx ={id(self.p_tau): 0, id(self.p_dec): 1, id(self.p_th): 2, id(self.p_en): 3}.get(id(p))
        if idx is not None:
            p.addItem(self.cursors[idx], ignoreBounds=True)

    # ------------------------------------------------------------------ configuration
    def _set_start(self, th):
        self.player.stop()
        self.anim = None
        self.start_sl.set_values(th)
        self._start_changed()

    def _set_goal(self, th):
        self.goal_sl.set_values(th)
        self.cfg_tabs.setCurrentIndex(1)
        self._draw_static()

    def _start_changed(self, *_):
        if self.player.playing():
            self.player.stop()
        self.anim = None
        self._draw_static()
        self._mass_timer.start()

    def _law_changed(self, *_):
        law = self.law.currentIndex()
        self.amp.setEnabled(law >= 2)
        self.freq.setEnabled(law == 3)
        self.add_g.setEnabled(law >= 2)

    def _draw_static(self):
        """Robot at the start configuration (+ goal ghost) when no animation is shown."""
        robot = self.ctx.robot
        th = self.start_sl.values()
        if len(th) != robot.n:
            return
        if self.anim is None:
            self.graphic.update(th)
            self._draw_com(th)
            self.goal_ghost.update(self.goal_sl.values())
            self.view.remove("dyn.trace")
        else:
            self._redraw_frame()

    def _draw_com(self, th):
        robot = self.ctx.robot
        if not self.cb_com.isChecked():
            self.view.remove_prefix("dyn.com")
            return
        p = robot.com_positions(th)
        m = robot.masses
        r = robot.link_radius * 1.15 * (m / m.max()) ** (1 / 3) + 0.4 * robot.link_radius
        for i in range(robot.n):
            self.view.set_sphere(f"dyn.com{i}", p[i], float(r[i]), (1.0, 0.82, 0.25, 1.0))

    # ------------------------------------------------------------------ inverse dynamics
    def run_inverse(self):
        robot, backend = self.ctx.robot, self.ctx.backend
        th_s, th_g = self.start_sl.values(), self.goal_sl.values()
        F = np.r_[0, 0, 0, self.fx.value(), self.fy.value(), self.fz.value()]
        Tf, sc = self.Tf.value(), self.scaling.currentText()
        self.player.stop()
        self.run_async("RNEA along the trajectory",
                       lambda prog, canc: compute_inverse(robot, backend, th_s, th_g, Tf, sc, F, prog, canc),
                       self._inverse_done)

    def _inverse_done(self, res):
        if res is None:
            return
        robot = self.ctx.robot
        self.id_res = res
        tr = res["tr"]
        p = self.p_tau
        self._clear_plot(p)
        limit_lines(p, robot)
        for i in range(robot.n):
            p.plot(tr.t, res["tau"][:, i], pen=pen(i), name=f"τ{i + 1}")
        ratio = np.abs(res["tau"]) / robot.tau_max
        worst = ratio.max(axis=0)
        msgs = [f"<b>{'Pinocchio RNEA' if res['backend'] == 'pinocchio' else 'mr.InverseDynamics'}</b>: "
                f"{len(tr.t)} calls in {res['t_id'] * 1e3:.1f} ms"]
        msgs.append("peak |τᵢ|/τmax,ᵢ: " + " ".join(
            f"<span style='color:{theme.bad_color() if w > 1 else theme.text_color()}'>{100 * w:.0f}%</span>" for w in worst))
        if worst.max() > 1:
            j = int(np.argmax(worst))
            k = int(np.argmax(ratio[:, j]))
            msgs.append(f"<span style='color:{theme.bad_color()}'>⚠ joint {j + 1} exceeds its limit "
                        f"({worst[j]:.2f}×τmax at t = {tr.t[k]:.2f} s) — increase T or use TOPP (tab ⑤)</span>")
        else:
            msgs.append(f"<span style='color:{theme.good_color()}'>✓ all torques within actuator limits</span>")
        msgs.append(f"<span style='color:{theme.good_color()}'>✓</span> ‖τ − (Mθ̈+c+g+JᵀF)‖ = {res['resid']:.1e}")
        self.id_lbl.setText("<br>".join(msgs))
        self._plot_decomposition()
        self.anim = ("id", res)
        self.bottom.setCurrentWidget(self.p_tau)
        self.player.set_times(tr.t)

    def _plot_decomposition(self):
        res = getattr(self, "id_res", None)
        if res is None or res["tau"].shape[1] != self.ctx.robot.n:
            return
        j = max(self.dec_joint.currentIndex(), 0)
        t = res["tr"].t
        p = self.p_dec
        self._clear_plot(p)
        p.setTitle(f"Joint {j + 1}:  τ = M(θ)θ̈ + c(θ,θ̇) + g(θ) + Jᵀ(θ)F_tip", color=theme.TEXT, size="10pt")
        cols = ["#4cc9f0", "#ff6b9a", "#ffd166", "#b392f0"]
        for key, name, col in (("Mdd", "M(θ)θ̈  inertial", cols[0]), ("c", "c(θ,θ̇)  Coriolis/centripetal", cols[1]),
                               ("g", "g(θ)  gravity", cols[2]), ("JF", "Jᵀ F_tip", cols[3])):
            if key == "JF" and not np.any(res["JF"]):
                continue
            p.plot(t, res[key][:, j], pen=pg.mkPen(col, width=2), name=name)
        p.plot(t, res["tau"][:, j], pen=pg.mkPen("#ffffff", width=3.0, style=QtCore.Qt.DashLine),
               name="τ (RNEA) = sum")

    # ------------------------------------------------------------------ forward dynamics
    def _fd_settings(self):
        return dict(law=self.law.currentIndex(), amp=self.amp.value() / 100.0, freq=self.freq.value(),
                    add_g=self.add_g.isChecked(), dth0=np.radians(self.dth0.value()), b=self.damp.value(),
                    method=INTEGRATORS[self.integ.currentText()], dt=self.dt.value() / 1000.0, T=self.Tsim.value())

    def run_forward(self):
        robot, backend = self.ctx.robot, self.ctx.backend
        th0 = self.start_sl.values()
        s = self._fd_settings()
        self.player.stop()

        def job(prog, canc):
            eng = DynamicsEngine(robot, backend)
            fn = make_tau_fn(eng, robot, s["law"], s["amp"], s["freq"], s["add_g"])
            t0 = time.perf_counter()
            r = simulate_chunked(eng, th0, np.full(robot.n, s["dth0"]), fn, s["T"], s["dt"], s["method"], s["b"],
                                 prog, canc)
            if r is None:
                return None
            ee = np.array([robot.fk(q)[:3, 3] for q in r.theta])
            return dict(sim=r, ee=ee, elapsed=time.perf_counter() - t0, s=s, backend=backend)
        if backend == "mr":
            eng = self.ctx.engine(robot, "mr")
            t0 = time.perf_counter()
            eng.forward(th0, np.zeros(robot.n), np.zeros(robot.n))
            per = time.perf_counter() - t0
            evals = {"euler": 1, "semi-implicit": 1, "rk4": 4, "rk45": 5}[s["method"]] * s["T"] / s["dt"]
            self.status(f"MR engine: ≈ {evals * per * 1.3:.0f} s for this simulation (Pinocchio: < 1 s) — "
                        "running in the background, you can cancel", theme.warn_color())
        self.run_async(f"Simulating ({s['method']})", job, self._forward_done)

    def _forward_done(self, res):
        if res is None:
            return
        robot = self.ctx.robot
        r, s = res["sim"], res["s"]
        k_bad = diverged_at(r)
        p = self.p_th
        self._clear_plot(p)
        for i in range(robot.n):
            p.plot(r.t[:k_bad], robot.to_display(r.theta[:k_bad])[:, i], pen=pen(i), name=f"θ{i + 1}")
        p = self.p_en
        self._clear_plot(p)
        E = r.ke + r.pe
        sl = slice(0, k_bad)
        p.plot(r.t[sl], r.ke[sl], pen=pg.mkPen(theme.SERIES[0], width=2), name="𝒦 kinetic")
        p.plot(r.t[sl], r.pe[sl], pen=pg.mkPen(theme.SERIES[1], width=2), name="𝒫 potential")
        p.plot(r.t[sl], E[sl], pen=pg.mkPen("#ffffff", width=2.5), name="𝒦 + 𝒫")
        bal = energy_balance(r, s["b"])
        conservative = s["law"] == 0 and s["b"] == 0
        if not conservative:
            p.plot(r.t[sl], bal[sl], pen=pg.mkPen(theme.GOOD, width=2, style=QtCore.Qt.DashLine),
                   name="𝒦 + 𝒫 − ∫(τ − bθ̇)ᵀθ̇ dt")
        eng = "Pinocchio ABA" if res["backend"] == "pinocchio" else "mr.ForwardDynamics"
        head = f"<b>{eng}</b> + {s['method']}: {len(r.t)} samples ({s['T']:.1f} s) in {res['elapsed']:.2f} s<br>"
        if k_bad < len(r.t):
            self.fd_lbl.setText(head + f"<span style='color:{theme.bad_color()}'>⚠ the integration diverged at "
                                f"t = {r.t[k_bad]:.2f} s — Δt too large for this stiff system: reduce Δt "
                                "or use RK4/RK45</span>")
        else:
            drift = float(np.abs(bal - bal[0]).max())
            ok = drift < 1e-3 * max(1.0, np.abs(E).max())
            self.fd_lbl.setText(head + f"energy {'balance ' if not conservative else ''}drift max|ΔE| = "
                                f"<span style='color:{theme.good_color() if ok else theme.warn_color()}'>{drift:.2e} J</span>"
                                + ("" if ok else "  (reduce Δt or use RK4)"))
        res["k_bad"] = k_bad
        self.anim = ("fd", res)
        self.bottom.setCurrentWidget(self.p_th if self.bottom.currentWidget() is not self.p_en else self.p_en)
        self.player.set_times(r.t)

    def compare_integrators(self):
        label, name, dt_fixed = CMP_TARGETS[max(self.cmp_target.currentIndex(), 0)]
        current = self.ctx.robot
        robot = current if name is None or name == current.name else get_robot(name)
        own = robot is current
        th0 = self.start_sl.values() if own else default_goal(robot)
        s = self._fd_settings()
        if dt_fixed is not None:
            s["dt"] = dt_fixed
        self.player.stop()

        def job(prog, canc):
            eng = DynamicsEngine(robot, "pinocchio")     # integrators compared on the fast engine
            out = {}
            methods = list(INTEGRATORS.values())
            for m_i, m in enumerate(methods):
                t0 = time.perf_counter()
                r = simulate_chunked(eng, th0, np.full(robot.n, s["dth0"]), lambda t, q, dq: np.zeros(robot.n),
                                     s["T"], s["dt"], m, 0.0, prog, canc, chunks=5,
                                     p0=m_i / len(methods), p1=(m_i + 1) / len(methods))
                if r is None:
                    return None
                out[m] = (r, time.perf_counter() - t0)
            return dict(runs=out, s=s, robot=robot.name, own=own)
        self.run_async("Comparing integrators", job, self._compare_done)

    def _compare_done(self, res):
        if res is None:
            return
        p = self.p_cmp
        p.clear()
        rows, keys = [], []
        names = {v: k for k, v in INTEGRATORS.items()}
        for i, (m, (r, el)) in enumerate(res["runs"].items()):
            k_bad = diverged_at(r)
            E = r.ke + r.pe
            dE = E[:k_bad] - E[0]                       # signed: explicit Euler gains energy, others oscillate ~0
            p.plot(r.t[:k_bad], dE, pen=pen(i, 2.2))
            txt = (f"max|ΔE| = {np.abs(dE).max():.2e} J, final ΔE = {dE[-1]:+.2e} J" if k_bad == len(r.t) else
                   f"<span style='color:{theme.bad_color()}'>diverged at t = {r.t[k_bad]:.2f} s</span>")
            rows.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {names[m]}: {txt}  ({el:.2f} s)")
            keys.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {names[m].split(' (')[0]}")
        # legend above the plot area (in the title row) so it never hides the curves
        p.setTitle("E(t) − E(0), τ = 0, no damping &nbsp;&nbsp; " + " &nbsp; ".join(keys), color=theme.TEXT,
                   size="10pt")
        s = res["s"]
        robot_txt = _short_name(res["robot"])
        self.cmp_lbl.setText(f"{robot_txt}: τ = 0, b = 0, Δt = {s['dt'] * 1e3:.1f} ms, T = {s['T']:.1f} s "
                             "(Pinocchio ABA). Explicit Euler pumps energy in (ΔE grows); symplectic Euler "
                             "oscillates but stays bounded; RK4/RK45 conserve it.<br>" + "<br>".join(rows))
        self.bottom.setCurrentIndex(4)
        if res["own"] and res["robot"] == self.ctx.robot.name:     # animate only the robot shown in the 3-D view
            r = res["runs"]["rk4"][0]
            ee = np.array([self.ctx.robot.fk(q)[:3, 3] for q in r.theta])
            self.anim = ("fd", dict(sim=r, ee=ee))
            self.player.set_times(r.t)

    # ------------------------------------------------------------------ animation
    def show_frame(self, k):
        if self.anim is None:
            return
        kind, res = self.anim
        if kind == "id":
            th = res["tr"].theta[k]
            t = res["tr"].t[k]
        else:
            k = min(k, res.get("k_bad", len(res["sim"].t)) - 1)
            th = res["sim"].theta[k]
            t = res["sim"].t[k]
        if not np.all(np.isfinite(th)):
            return
        self.graphic.update(th)
        self._draw_com(th)
        if kind == "id":
            self.goal_ghost.update(res["tr"].theta[-1])
        else:
            self.goal_ghost.remove()
        if self.cb_trace.isChecked():
            pts = res["ee"] if kind == "id" else res["ee"][:k + 1]
            if len(pts) > 1:
                self.view.set_polyline("dyn.trace", pts, (0.3, 0.85, 1.0, 0.85), 2)
        else:
            self.view.remove("dyn.trace")
        for c in self.cursors:
            c.setPos(t)

    def _redraw_frame(self):
        if self.anim is not None:
            self.show_frame(self.player.k)

    def on_show(self):
        self._mass_timer.start()

    def on_hide(self):
        self.player.stop()

    def shutdown(self):
        self.player.stop()
        self._mass_timer.stop()
        super().shutdown()
