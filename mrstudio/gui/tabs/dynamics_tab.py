"""Tab 4 -- inverse and forward dynamics of open chains (MR Ch. 8).

Also hosts two small helpers shared with the control tab:
  * Worker  -- QThread running a long computation with progress / cancel (keeps the UI responsive)
  * Player  -- QTimer playback of a precomputed time series (play/pause, time slider, speed)
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import trajectory as tj
from ...dynamics import DynamicsEngine, simulate
from .. import theme
from ..base import BaseTab
from ..viewer import GhostRobot, RobotGraphic, SceneView
from ..widgets import JointSliders, button, checkbox, combo, group, hbox, pen, plot, spin, value_label

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
<p>plus classical RK4 and adaptive RK45 (Dormand–Prince 8(5,3), SciPy).</p>
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
class Worker(QtCore.QThread):
    """Run fn(progress, cancelled) in a background thread.  Results come back through queued signals."""
    progressed = QtCore.Signal(float)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(self.progressed.emit, lambda: self._cancel)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.done.emit(res)


class AsyncMixin:
    """One background job at a time, with a progress bar and a cancel button."""

    def _init_async(self):
        self.worker: Worker | None = None
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(16)
        self.cancel_btn = button("Cancel", self._cancel_job)
        self.busy_row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(self.busy_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.progress, 1)
        lay.addWidget(self.cancel_btn)
        self.busy_row.setVisible(False)
        self._job_gen = 0

    def run_async(self, label: str, fn, on_done):
        """fn(progress, cancelled) runs in a QThread; on_done(result) runs in the GUI thread."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self._job_gen += 1
        gen = self._job_gen
        w = Worker(fn, self)
        w.progressed.connect(lambda f: self.progress.setValue(int(1000 * f)))
        w.done.connect(lambda res: self._job_done(gen, res, on_done))
        w.failed.connect(lambda err: self._job_failed(gen, err))
        w.finished.connect(lambda: self._job_finished(gen))
        self.progress.setValue(0)
        self.progress.setFormat(label + "  %p%")
        self.busy_row.setVisible(True)
        self.worker = w
        w.start()

    def wait_job(self, timeout_s: float = 120.0):
        """Block (processing events) until the current job finished -- used by scripted tests."""
        app = QtWidgets.QApplication.instance()
        t0 = time.time()
        while self.worker is not None and (self.worker.isRunning() or self.busy_row.isVisible()):
            app.processEvents()
            time.sleep(0.005)
            if time.time() - t0 > timeout_s:
                raise TimeoutError("background job did not finish")
        app.processEvents()

    def _job_done(self, gen, res, on_done):
        if gen != self._job_gen:
            return
        try:
            on_done(res)
        except Exception:
            self._job_failed(gen, traceback.format_exc())

    def _job_failed(self, gen, err):
        if gen != self._job_gen:
            return
        print(err)
        self.status(f"⚠ computation failed: {err.strip().splitlines()[-1]}", theme.BAD)

    def _job_finished(self, gen):
        if gen == self._job_gen:
            self.busy_row.setVisible(False)

    def _cancel_job(self):
        if self.worker is not None:
            self.worker.cancel()

    def cancel_jobs(self):
        self._job_gen += 1
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        self.busy_row.setVisible(False)

    def status(self, text, color=None):  # overridden
        pass


class Player(QtCore.QObject):
    """Plays back a precomputed time series in real time (QTimer, 30 ms) -- emits frame(index)."""
    frame = QtCore.Signal(int)
    SPEEDS = ["0.1×", "0.25×", "0.5×", "1×", "2×"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t = np.zeros(1)
        self.k = 0
        self._t_sim = 0.0
        self._t_wall = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.btn = button("▶ Play", self._toggled, checkable=True)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._seek)
        self.speed = combo(self.SPEEDS)
        self.speed.setCurrentText("1×")
        self.loop = checkbox("loop", checked=True)
        self.lbl = value_label("no motion computed yet")
        self.widget = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(self.widget)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(hbox(self.btn, QtWidgets.QLabel("speed"), self.speed, self.loop, None))
        v.addWidget(self.slider)
        v.addWidget(self.lbl)

    def set_times(self, t, autoplay: bool = True):
        self.t = np.asarray(t, float)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.t) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._t_sim = 0.0
        self.set_index(0)
        if autoplay:
            if self.btn.isChecked():
                self._t_wall = time.perf_counter()
            else:
                self.btn.setChecked(True)

    def playing(self) -> bool:
        return self.btn.isChecked()

    def stop(self):
        if self.btn.isChecked():
            self.btn.setChecked(False)

    def _toggled(self, on):
        self.btn.setText("❚❚ Pause" if on else "▶ Play")
        if on and len(self.t) > 1:
            if self.k >= len(self.t) - 1:
                self.k = 0
            self._t_sim = self.t[self.k]
            self._t_wall = time.perf_counter()
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        now = time.perf_counter()
        sp = float(self.speed.currentText().rstrip("×"))
        self._t_sim += (now - self._t_wall) * sp
        self._t_wall = now
        T = self.t[-1]
        if self._t_sim > T:
            if self.loop.isChecked():
                if self._t_sim > T + 0.8 * sp:      # short pause at the end, then restart
                    self._t_sim = 0.0
            else:
                self.set_index(len(self.t) - 1)
                self.stop()
                return
        k = int(np.clip(np.searchsorted(self.t, min(self._t_sim, T)), 0, len(self.t) - 1))
        if k != self.k or self._t_sim == 0.0:
            self.set_index(k)

    def set_index(self, k):
        self.k = int(k)
        self.slider.blockSignals(True)
        self.slider.setValue(self.k)
        self.slider.blockSignals(False)
        self.lbl.setText(f"t = {self.t[self.k]:6.3f} s  /  {self.t[-1]:.3f} s")
        self.frame.emit(self.k)

    def _seek(self, k):
        self._t_sim = self.t[int(k)]
        self.set_index(k)


def form(*rows) -> QtWidgets.QFormLayout:
    f = QtWidgets.QFormLayout()
    f.setContentsMargins(0, 0, 0, 0)
    f.setHorizontalSpacing(8)
    f.setVerticalSpacing(3)
    for r in rows:
        if isinstance(r, tuple):
            lab, w = r
            if isinstance(w, QtWidgets.QLayout):
                f.addRow(lab, w)
            else:
                f.addRow(lab, w)
        else:
            f.addRow(r)
    return f


def compact(w, chars: int = 16):
    """Keep long combo boxes from widening the control column."""
    w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    w.setMinimumContentsLength(chars)
    return w


def default_goal(robot) -> np.ndarray:
    d = np.resize([0.9, -0.7, 0.8, 0.6, -0.6, 0.8, 0.5], robot.n)
    if robot.planar:
        d = np.resize([-1.3, -1.1, 0.9], robot.n)
    lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
    return np.clip(robot.home + d, lo + 0.05, hi - 0.05)


def no_si(pw):
    """Plain axis numbers (pyqtgraph's automatic SI prefixes mangle labels such as '1/s')."""
    for ax in ("left", "bottom"):
        pw.getAxis(ax).enableAutoSIPrefix(False)
    return pw


def time_cursor(pw):
    ln = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen((255, 255, 255, 90), width=1))
    pw.addItem(ln, ignoreBounds=True)
    return ln


def limit_lines(pw, robot, scale=1.0):
    for i in range(robot.n):
        for sgn in (1, -1):
            pw.addItem(pg.InfiniteLine(pos=sgn * robot.tau_max[i] * scale, angle=0, pen=pen(i, 1.0, "dash")),
                       ignoreBounds=True)


FD_LAWS = ["τ = 0  (falls freely — chaotic pendulum)", "τ = g(θ)  (gravity compensation — floats)",
           "τ = A·τmax  (constant)", "τ = A·τmax·sin(2πft + φᵢ)  (sinusoidal)"]
INTEGRATORS = {"Euler (mr.EulerStep)": "euler", "Semi-implicit (symplectic) Euler": "semi-implicit",
               "Runge–Kutta 4": "rk4", "RK45 adaptive (SciPy DOP853)": "rk45"}


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
    """dynamics.simulate() run in chunks so that progress can be reported and the run cancelled."""
    Tc = T / chunks
    parts = []
    th, dth, t_off = np.array(th0, float), np.array(dth0, float), 0.0
    for c in range(chunks):
        off = t_off
        r = simulate(eng, th, dth, lambda t, q, dq, off=off: tau_fn(t + off, q, dq), Tc, dt, method,
                     damping=damping)
        r.t = r.t + off
        parts.append(r if c == 0 else _drop_first(r))
        th, dth, t_off = r.theta[-1], r.dtheta[-1], off + Tc
        progress(p0 + (p1 - p0) * (c + 1) / chunks)
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
        self.fd_lbl = value_label()
        fd_box = group("Forward dynamics — simulation (MR 8.5)",
                       form(("Torque law", self.law), ("Amplitude A", self.amp), ("Frequency f", self.freq),
                            self.add_g, ("Initial θ̇₀ (all joints)", self.dth0), ("Viscous damping b", self.damp),
                            ("Integrator", self.integ), ("Time step Δt", self.dt), ("Duration", self.Tsim)),
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
        self.p_cmp = plot("Integrator comparison: |E(t) − E(0)|, τ = 0, no damping", "|ΔE| [J]")
        self.p_cmp.setLogMode(y=True)
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
        ctx.backend_changed.connect(self._backend_changed)
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
        ok, bad = f"<span style='color:{theme.GOOD}'>✓</span>", f"<span style='color:{theme.BAD}'>✗</span>"
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
            f"&nbsp;&nbsp;speed-up ×{t_mr / max(t_pin, 1e-9):.0f}<br><br>"
            f"<span style='color:{theme.MUTED}'>Diagonal mᵢᵢ = apparent inertia of joint i; off-diagonal terms "
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

    def _backend_changed(self, backend):
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
            f"<span style='color:{theme.BAD if w > 1 else theme.TEXT}'>{100 * w:.0f}%</span>" for w in worst))
        if worst.max() > 1:
            j = int(np.argmax(worst))
            k = int(np.argmax(ratio[:, j]))
            msgs.append(f"<span style='color:{theme.BAD}'>⚠ joint {j + 1} exceeds its limit "
                        f"({worst[j]:.2f}×τmax at t = {tr.t[k]:.2f} s) — increase T or use TOPP (tab ⑤)</span>")
        else:
            msgs.append(f"<span style='color:{theme.GOOD}'>✓ all torques within actuator limits</span>")
        msgs.append(f"<span style='color:{theme.GOOD}'>✓</span> ‖τ − (Mθ̈+c+g+JᵀF)‖ = {res['resid']:.1e}")
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
            evals = {"euler": 1, "semi-implicit": 1, "rk4": 4, "rk45": 12}[s["method"]] * s["T"] / s["dt"]
            self.status(f"MR engine: ≈ {evals * per * 1.3:.0f} s for this simulation (Pinocchio: < 1 s) — "
                        "running in the background, you can cancel", theme.WARN)
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
            self.fd_lbl.setText(head + f"<span style='color:{theme.BAD}'>⚠ the integration diverged at "
                                f"t = {r.t[k_bad]:.2f} s — Δt too large for this stiff system: reduce Δt "
                                "or use RK4/RK45</span>")
        else:
            drift = float(np.abs(bal - bal[0]).max())
            ok = drift < 1e-3 * max(1.0, np.abs(E).max())
            self.fd_lbl.setText(head + f"energy {'balance ' if not conservative else ''}drift max|ΔE| = "
                                f"<span style='color:{theme.GOOD if ok else theme.WARN}'>{drift:.2e} J</span>"
                                + ("" if ok else "  (reduce Δt or use RK4)"))
        res["k_bad"] = k_bad
        self.anim = ("fd", res)
        self.bottom.setCurrentWidget(self.p_th if self.bottom.currentWidget() is not self.p_en else self.p_en)
        self.player.set_times(r.t)

    def compare_integrators(self):
        robot = self.ctx.robot
        th0 = self.start_sl.values()
        s = self._fd_settings()
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
            return dict(runs=out, s=s)
        self.run_async("Comparing integrators", job, self._compare_done)

    def _compare_done(self, res):
        if res is None:
            return
        p = self.p_cmp
        p.clear()
        if p.plotItem.legend is not None:
            p.plotItem.legend.clear()
        rows = []
        names = {v: k for k, v in INTEGRATORS.items()}
        for i, (m, (r, el)) in enumerate(res["runs"].items()):
            k_bad = diverged_at(r)
            E = r.ke + r.pe
            dE = np.abs(E[:k_bad] - E[0]) + 1e-16
            p.plot(r.t[:k_bad], dE, pen=pen(i, 2.2), name=names[m])
            txt = (f"max|ΔE| = {dE.max():.2e} J" if k_bad == len(r.t) else
                   f"<span style='color:{theme.BAD}'>diverged at t = {r.t[k_bad]:.2f} s</span>")
            rows.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {names[m]}: {txt}  ({el:.2f} s)")
        s = res["s"]
        self.cmp_lbl.setText(f"τ = 0, b = 0, Δt = {s['dt'] * 1e3:.1f} ms, T = {s['T']:.1f} s (Pinocchio ABA). "
                             "Explicit Euler pumps energy in; symplectic Euler oscillates but stays bounded; "
                             "RK4/RK45 conserve it.<br>" + "<br>".join(rows))
        self.bottom.setCurrentIndex(4)
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
