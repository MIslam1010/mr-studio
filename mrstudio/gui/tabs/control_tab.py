"""Tab 5 -- trajectory generation (MR Ch. 9) and torque-based motion control (MR Ch. 11)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import control as ctl
from ... import trajectory as tj
from ...dynamics import DynamicsEngine
from .. import theme
from ..base import BaseTab
from ..viewer import GhostRobot, RobotGraphic, SceneView
from ..widgets import JointSliders, button, checkbox, combo, group, hbox, pen, plot, spin, value_label
from .dynamics_tab import AsyncMixin, Player, compact, default_goal, form, limit_lines, no_si, time_cursor

THEORY = r"""
<h2>Trajectories and motion control</h2>
<span class="ref">Modern Robotics, Ch. 9 (trajectory generation) and Ch. 11 (robot control, Sec. 11.4)</span>
<h3>Path × time scaling (9.1)</h3>
<p>A trajectory is a geometric path $\theta(s)$, $s\in[0,1]$, composed with a time scaling $s(t)$:</p>
$$\dot{\theta}=\frac{d\theta}{ds}\dot{s},\qquad \ddot{\theta}=\frac{d\theta}{ds}\ddot{s}+\frac{d^{2}\theta}{ds^{2}}\dot{s}^{2}$$
<h3>Three straight-line paths (9.2.1)</h3>
$$\mathrm{joint}:\ \theta(s)=\theta_{start}+s\,(\theta_{end}-\theta_{start})\quad(9.3)$$
$$\mathrm{screw}:\ X(s)=X_{start}\exp\left(\log(X_{start}^{-1}X_{end})\,s\right)\quad(9.6)$$
$$\mathrm{Cartesian}:\ p(s)=p_{start}+s\,(p_{end}-p_{start})\quad(9.7)$$
$$R(s)=R_{start}\exp\left(\log(R_{start}^{T}R_{end})\,s\right)\quad(9.8)$$
<p>Joint-space lines give curved end-effector paths (orange); the screw path follows a constant twist (purple);
only the Cartesian path is a straight line in space (green).  Task-space paths are mapped to joints by
Newton–Raphson IK warm-started at the previous point (MR 6.2).</p>
<h3>Time scalings (9.2.2)</h3>
$$\mathrm{cubic}:\ s=3\left(\frac{t}{T}\right)^{2}-2\left(\frac{t}{T}\right)^{3}$$
$$\mathrm{quintic}:\ s=10\left(\frac{t}{T}\right)^{3}-15\left(\frac{t}{T}\right)^{4}+6\left(\frac{t}{T}\right)^{5}$$
<p>The quintic also has zero acceleration at both ends (no torque jump).  The trapezoidal profile accelerates at
$a$, cruises at $v$ and decelerates, with $T=(a+v^{2})/(va)$ and $v^{2}/a\leq 1$ (here $vT=1.5$).</p>
<h3>Time-optimal time scaling (9.4)</h3>
<p>Along a path the dynamics become scalar in $s$:</p>
$$\tau=m(s)\ddot{s}+c(s)\dot{s}^{2}+g(s)\quad(9.33)$$
$$m(s)=M\theta',\qquad c(s)=M\theta''+\theta'^{T}\Gamma\,\theta'$$
<p>Each actuator limit $\tau_i^{min}\leq\tau_i\leq\tau_i^{max}$ bounds $\ddot{s}$:  $L(s,\dot{s})\leq\ddot{s}\leq U(s,\dot{s})$.
Where $L=U$ lies the <b>maximum-velocity curve</b> (MVC, red).  The minimum time</p>
$$T=\int_{0}^{1}\frac{ds}{\dot{s}(s)}\quad(9.39)$$
<p>is achieved by keeping $\dot{s}$ as high as possible: <b>bang-bang</b> — maximum acceleration $U$, then maximum
deceleration $L$, touching the MVC at switch points; at every instant at least one actuator is saturated.
<b>TOPP-RA</b> (Pham &amp; Pham) solves it with reachability analysis on a grid, $x=\dot{s}^{2}$, $u=\ddot{s}$:</p>
$$x_{i+1}=x_i+2\Delta_i u_i,\qquad x_i\in\mathcal{K}_i$$
<p>The controllable sets $\mathcal{K}_i$ come from a backward pass; a forward pass then greedily maximises $u_i$.  Velocity limits $|\dot{\theta}_i|\leq v_i$ are included.</p>
<h3>Error dynamics (11.2)</h3>
$$\theta_e=\theta_d-\theta,\qquad \ddot{\theta}_e+K_d\dot{\theta}_e+K_p\theta_e=0$$
$$\omega_n=\sqrt{K_p},\qquad \zeta=\frac{K_d}{2\sqrt{K_p}}$$
<h3>Controllers (11.4)</h3>
$$\mathrm{PID}:\ \tau=K_p\theta_e+K_i\int\theta_e(t)dt+K_d\dot{\theta}_e$$
$$\mathrm{PD+gravity}:\ \tau=K_p\theta_e+K_d\dot{\theta}_e+\tilde{g}(\theta)$$
<p><b>Computed torque</b> (feedforward + feedback linearisation, MR Eq. 11.35, mr.ComputedTorque):</p>
$$\tau=\tilde{M}(\theta)\left(\ddot{\theta}_d+K_p\theta_e+K_i\int\theta_e\,dt+K_d\dot{\theta}_e\right)+\tilde{h}(\theta,\dot{\theta})\quad(11.35)$$
<p>If the model is exact ($\tilde{M}=M$, $\tilde{h}=h$) the closed loop becomes linear and decoupled:</p>
$$\ddot{\theta}_e+K_d\dot{\theta}_e+K_p\theta_e+K_i\int\theta_e(t)dt=0\quad(11.33)$$
<p>The decentralised controllers use torque gains $K\,\mathrm{diag}(\tilde{M}(\theta_d(0)))$ so that all four
controllers have the same nominal bandwidth.  PD alone sags under gravity; the integral term removes the sag
slowly; gravity compensation removes it at once; computed torque also cancels inertia and Coriolis
coupling — until the model is wrong (mass scale ≠ 1) or an unknown payload is added.  PD + gravity compensation
is provably stable for set-points thanks to the skew symmetry of $\dot{M}-2C$.</p>
<p>The real robot is simulated with Pinocchio (ABA + RK4); the controller runs at a fixed period with
zero-order hold, exactly as mr.SimulateControl (validated in the Validation tab).</p>
"""
THEORY += further_reading([
    ('lavalle', '§14.6.3, pp. 846–854', 'path-constrained trajectories, (s, ṡ) plane; bang-bang optimum p. 853'),
    ('gotor', 'Ch. 7, pp. 193–212', 'trapezoidal, polynomial and spline trajectories'),
    ('corke', '§3.1, pp. 43–51; §7.4, pp. 152–158', 'quintic and LSPB scalings; joint and Cartesian paths'),
    ('astrom_murray', '§10.1, §10.3–10.4, pp. 293–307', 'PID: P, I, D actions, tuning, integrator windup'),
    ('astrom_murray', '§6.3, pp. 183–184', 'second-order systems: damping ratio, natural frequency'),
    ('astrom_murray', '§4.3–4.4, pp. 102–119; §5.4, pp. 161–162', 'stability, Lyapunov; feedback linearisation'),
    ('mls', 'Ch. 4 §4–5, pp. 179–200', 'Lyapunov; computed torque (p. 190); PD control (p. 193)'),
    ('corke', '§9.4, pp. 204–215', 'independent joint control; feedforward and computed torque'),
])



class ControlTab(AsyncMixin, BaseTab):
    title = "⑤ Control"
    needs_dynamics = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._init_async()
        self.view = SceneView()
        self.graphic = None
        self.ghost = None
        self.paths: dict = {}
        self.anim = None            # dict(t, theta, theta_d (optional), ee (optional))
        self.topp_res = None

        # ---- start / goal -----------------------------------------------------------------------
        self.start_sl, self.goal_sl = JointSliders(), JointSliders()
        self.start_sl.changed.connect(self._config_changed)
        self.goal_sl.changed.connect(self._config_changed)
        self.cfg_tabs = QtWidgets.QTabWidget()
        self.cfg_tabs.addTab(self.start_sl, "Start θ_start")
        self.cfg_tabs.addTab(self.goal_sl, "Goal θ_end")
        cfg_btns = hbox(button("Home start", lambda: self._set(self.start_sl, self.ctx.robot.home)),
                        button("Random goal", lambda: self._set(self.goal_sl, self.ctx.robot.random_config())),
                        button("Swap", self._swap))

        # ---- Ch. 9 trajectory ------------------------------------------------------------------
        self.path_box = compact(combo([tj.PATH_LABELS[k] for k in tj.PATHS]), 18)
        self.scaling = combo(["quintic", "cubic", "trapezoidal"])
        self.Tf = spin(0.2, 20, 2.0, 2, 0.1, " s")
        self.scaling.currentIndexChanged.connect(lambda _: self._plot_scalings())
        self.Tf.valueChanged.connect(lambda _: self._plot_scalings())
        self.cb_paths = checkbox("Show all three end-effector paths", lambda _: self._draw_paths(), True)
        self.gen_btn = button("Generate trajectory", self.generate, primary=True)
        self.traj_lbl = value_label()
        traj_box = group("Trajectory — MR 9.2", form(("Path", self.path_box), ("Time scaling", self.scaling),
                                                     ("Duration T", self.Tf)),
                         self.cb_paths, self.gen_btn, self.traj_lbl)

        # ---- TOPP -------------------------------------------------------------------------------
        self.tau_scale = spin(5, 300, 100, 0, 5, " % τmax")
        self.vel_scale = spin(5, 500, 100, 0, 10, " % vmax")
        self.topp_btn = button("Time-optimal (TOPP-RA)", self.run_topp, primary=True)
        self.topp_lbl = value_label()
        topp_box = group("Time-optimal scaling — MR 9.4",
                         form(("Torque limits", self.tau_scale), ("Speed limits", self.vel_scale)),
                         self.topp_btn, self.topp_lbl)

        # ---- Ch. 11 control ---------------------------------------------------------------------
        self.ctrl_box = combo(list(ctl.CONTROLLERS))
        self.ctrl_box.setCurrentText("Computed torque")
        self.ref_box = combo(["Ch. 9 trajectory", "Time-optimal (TOPP-RA)"])
        self.Kp = spin(0, 2500, 100, 1, 10, "")
        self.Ki = spin(0, 5000, 100, 1, 10, "")
        self.Kd = spin(0, 200, 20, 1, 2, "")
        for w in (self.Kp, self.Ki, self.Kd):
            w.valueChanged.connect(self._gains_changed)
        self.mass_scale = spin(0.2, 3.0, 1.0, 2, 0.05, " ×")
        self.payload = spin(0, 20, 0.0, 2, 0.25, " kg")
        self.init_err = spin(-30, 30, 2.0, 1, 0.5, " °")
        self.ctrl_dt = spin(0.5, 20, 2.0, 1, 0.5, " ms")
        self.cb_sat = checkbox("Actuator saturation |τ| ≤ τmax", None, False)
        self.gain_lbl = value_label()
        self.run_btn = button("Run controller", self.run_control, primary=True)
        self.cmp_btn = button("Compare controllers", self.compare_controllers)
        self.ctrl_lbl = value_label()
        ctrl_box = group("Feedback control — MR 11.4",
                         form(("Controller", self.ctrl_box), ("Reference", self.ref_box),
                              ("Kp  [1/s²]", self.Kp), ("Ki  [1/s³]", self.Ki), ("Kd  [1/s]", self.Kd)),
                         self.gain_lbl,
                         form(("Model mass scale", self.mass_scale), ("Unknown payload", self.payload),
                              ("Initial error θₑ(0)", self.init_err), ("Control period Δt", self.ctrl_dt)),
                         self.cb_sat, hbox(self.run_btn, self.cmp_btn), self.ctrl_lbl)

        self.player = Player(self)
        self.player.frame.connect(self.show_frame)
        self.status_lbl = value_label()
        controls = QtWidgets.QWidget()
        controls.setMinimumWidth(330)
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Start / goal configurations", self.cfg_tabs, cfg_btns))
        cl.addWidget(traj_box)
        cl.addWidget(topp_box)
        cl.addWidget(ctrl_box)
        cl.addWidget(group("Playback", self.player.widget, self.busy_row, self.status_lbl))
        cl.addStretch(1)

        # ---- plots ----------------------------------------------------------------------------------
        self.p_s = plot("s(t)", "s")
        self.p_sd = plot("ṡ(t)", "ṡ [1/s]")
        self.p_sdd = plot("s̈(t)", "s̈ [1/s²]")
        self.p_th = plot("Desired joint angles θ_d(t)", "θ [deg]")
        self.p_dth = plot("Desired joint velocities θ̇_d(t)", "θ̇ [deg/s]")
        self.p_phase = plot("Phase plane (s, ṡ)", "ṡ [1/s]", "s")
        self.p_tn = plot("Normalised torques τᵢ/τmax,ᵢ along the time-optimal trajectory", "τ/τmax")
        self.p_err = plot("Tracking error θₑ = θ_d − θ", "θₑ [deg]")
        self.p_tau = plot("Joint torques (dashed: ±τmax)", "τ [N·m]")
        self.p_bar = plot("RMS tracking error (all joints)", "RMS θₑ [deg]", "", legend=False)
        self.cursors = {id(p): time_cursor(p) for p in (self.p_th, self.p_dth, self.p_tn, self.p_err, self.p_tau)}
        for p in (self.p_s, self.p_sd, self.p_sdd, self.p_th, self.p_dth, self.p_phase, self.p_tn, self.p_err,
                  self.p_tau, self.p_bar):
            no_si(p)
        self.topp_info = value_label()
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Controller", "RMS θₑ [°]", "max |θₑ| [°]", "final |θₑ| [°]",
                                              "max |τ|/τmax"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cmp_info = value_label()

        def page(*ws, info=None, stretch=None):
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            h = QtWidgets.QHBoxLayout()
            for i, x in enumerate(ws):
                h.addWidget(x, (stretch or [1] * len(ws))[i])
            v.addLayout(h, 1)
            if info is not None:
                v.addWidget(info)
            return w
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(page(self.p_s, self.p_sd, self.p_sdd), "Time scaling s(t)")
        self.bottom.addTab(page(self.p_th, self.p_dth), "Joint trajectory")
        self.bottom.addTab(page(self.p_phase, self.p_tn, info=self.topp_info), "TOPP-RA phase plane")
        self.bottom.addTab(page(self.p_err, self.p_tau), "Tracking")
        self.bottom.addTab(page(self.p_bar, self.table, info=self.cmp_info, stretch=[2, 3]), "Compare controllers")

        self.assemble(controls, self.view, self.bottom, THEORY, controls_width=375, bottom_ratio=(5, 4))
        ctx.backend_changed.connect(lambda b: self.status(
            f"Controller model engine: {'Pinocchio' if b == 'pinocchio' else 'Modern Robotics (slower)'}; "
            "the plant is always Pinocchio"))
        self._gains_changed()
        self.on_robot_changed(ctx.robot)

    # ------------------------------------------------------------------ helpers
    def status(self, text, color=None):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color:{color}" if color else "")

    def _path_kind(self) -> str:
        return tj.PATHS[self.path_box.currentIndex()]

    def _clear(self, p):
        p.clear()
        if p.plotItem.legend is not None:
            p.plotItem.legend.clear()
        if id(p) in self.cursors:
            p.addItem(self.cursors[id(p)], ignoreBounds=True)

    def _gains_changed(self, *_):
        kp, kd = self.Kp.value(), self.Kd.value()
        wn = np.sqrt(kp)
        zeta = kd / (2 * wn) if wn > 0 else np.inf
        self.gain_lbl.setText(f"ωn = √Kp = {wn:.2f} rad/s,  ζ = Kd/(2√Kp) = {zeta:.2f}"
                              + ("  (critically damped)" if abs(zeta - 1) < 0.02 else ""))

    def _set(self, sl, th):
        sl.set_values(th)
        self._config_changed()

    def _swap(self):
        a, b = self.start_sl.values(), self.goal_sl.values()
        self.start_sl.set_values(b)
        self.goal_sl.set_values(a)
        self._config_changed()

    def _config_changed(self, *_):
        self.player.stop()
        self.paths = {}
        self.topp_res = None
        self.anim = None
        self.view.remove_prefix("ctl.path")
        self.view.remove("ctl.trace")
        robot = self.ctx.robot
        a = self.start_sl.values()
        if len(a) != robot.n:
            return
        self.graphic.update(a)
        self.ghost.update(self.goal_sl.values())
        self.traj_lbl.setText("Start/goal changed — press Generate.")

    # ------------------------------------------------------------------ robot
    def on_robot_changed(self, robot):
        self.cancel_jobs()
        self.player.stop()
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("ctl.")
        self.graphic = RobotGraphic(self.view, robot, "ctl.robot")
        self.ghost = GhostRobot(self.view, robot, "ctl.des", (0.55, 1.0, 0.6, 0.55))
        self.start_sl.set_robot(robot)
        self.goal_sl.set_robot(robot, default_goal(robot))
        self.view.frame_scene(robot.reach, planar=robot.planar,
                              center=(0, 0, 0.25 * robot.reach * (not robot.planar)))
        for p in (self.p_th, self.p_dth, self.p_phase, self.p_tn, self.p_err, self.p_tau, self.p_bar):
            self._clear(p)
        self.table.setRowCount(0)
        for l in (self.topp_lbl, self.ctrl_lbl, self.topp_info, self.cmp_info):
            l.setText("")
        self._config_changed()
        self._plot_scalings()
        self.status(f"{robot.name}: τmax = {np.array2string(robot.tau_max, precision=0, separator=', ')} N·m")
        QtCore.QTimer.singleShot(50, self.generate)

    # ------------------------------------------------------------------ Ch. 9: paths and time scalings
    def _plot_scalings(self):
        Tf = self.Tf.value()
        t = np.linspace(0, Tf, 300)
        sel = self.scaling.currentText()
        for p in (self.p_s, self.p_sd, self.p_sdd):
            self._clear(p)
        for i, kind in enumerate(["cubic", "quintic", "trapezoidal"]):
            s, sd, sdd = tj.time_scaling(kind, Tf, t)
            w = 3.0 if kind == sel else 1.4
            for p, y in ((self.p_s, s), (self.p_sd, sd), (self.p_sdd, sdd)):
                p.plot(t, y, pen=pen(i, w), name=kind if p is self.p_s else None)

    def _ensure_paths(self, robot, a, b, kinds):
        """Compute (in a worker) the joint paths that are missing."""
        return {k: tj.joint_path(robot, a, b, k, 81) for k in kinds}

    def generate(self):
        robot = self.ctx.robot
        a, b = self.start_sl.values(), self.goal_sl.values()
        missing = [k for k in tj.PATHS if k not in self.paths]
        self.player.stop()

        def job(prog, canc):
            out = {}
            for i, k in enumerate(missing):
                out[k] = tj.joint_path(robot, a, b, k, 81)
                prog((i + 1) / max(len(missing), 1))
                if canc():
                    return None
            return out
        self.run_async("IK along the task-space paths", job, self._generate_done)

    def _generate_done(self, paths):
        if paths is None:
            return
        self.paths.update(paths)
        robot = self.ctx.robot
        kind = self._path_kind()
        jp = self.paths[kind]
        tr = tj.make_trajectory(robot, jp.theta[0], self.goal_sl.values(), self.Tf.value(), 241, kind,
                                self.scaling.currentText(), jpath=jp)
        self.traj = tr
        self._draw_paths()
        for p in (self.p_th, self.p_dth):
            self._clear(p)
        for i in range(robot.n):
            self.p_th.plot(tr.t, robot.to_display(tr.theta)[:, i], pen=pen(i), name=f"θ{i + 1}")
            self.p_dth.plot(tr.t, robot.to_display(tr.dtheta)[:, i], pen=pen(i), name=f"θ̇{i + 1}")
        msgs = []
        for k in tj.PATHS:
            if k in self.paths:
                p = self.paths[k]
                L = np.sum(np.linalg.norm(np.diff(p.ee, axis=0), axis=1))
                warn = (f"<br>&nbsp;&nbsp;<span style='color:{theme.BAD}'>⚠ {p.ik_failures}/{len(p.s)} path points "
                        "unreachable (IK failed — path leaves the workspace)</span>" if p.ik_failures else "")
                msgs.append(f"<span style='color:{tj.PATH_COLORS[k]}'>■</span> {tj.PATH_LABELS[k]}: "
                            f"EE path length {L:.3f} m{warn}")
        straight = self.paths.get("cartesian")
        if straight is not None:
            p0, p1 = straight.ee_target[0], straight.ee_target[-1]
            if np.linalg.norm(p1 - p0) > 1e-9:
                u = (p1 - p0) / np.linalg.norm(p1 - p0)
                dev = np.linalg.norm(straight.ee - p0 - np.outer((straight.ee - p0) @ u, u), axis=1).max()
                msgs.append(f"Cartesian straightness: max deviation {dev:.1e} m")
        self.traj_lbl.setText("<br>".join(msgs))
        self.anim = dict(t=tr.t, theta=tr.theta, label="desired motion")
        self.bottom.setCurrentIndex(0 if self.bottom.currentIndex() not in (0, 1) else self.bottom.currentIndex())
        self.player.set_times(tr.t)

    def _draw_paths(self):
        self.view.remove_prefix("ctl.path")
        if not self.paths:
            return
        sel = self._path_kind()
        for k, p in self.paths.items():
            if k != sel and not self.cb_paths.isChecked():
                continue
            col = pg.mkColor(tj.PATH_COLORS[k])
            rgba = (col.redF(), col.greenF(), col.blueF(), 1.0 if k == sel else 0.75)
            self.view.set_polyline(f"ctl.path.{k}", p.ee, rgba, 4.0 if k == sel else 2.0)
        p = self.paths[sel]
        self.view.set_sphere("ctl.path.start", p.ee[0], 0.7 * self.ctx.robot.link_radius, theme.ACCENT)
        self.view.set_sphere("ctl.path.goal", p.ee_target[-1], 0.7 * self.ctx.robot.link_radius, theme.GOOD)

    # ------------------------------------------------------------------ Ch. 9.4: TOPP-RA
    def _limits(self, robot):
        return robot.tau_max * self.tau_scale.value() / 100.0, \
            tj.default_velocity_limits(robot) * self.vel_scale.value() / 100.0

    def run_topp(self):
        robot, backend = self.ctx.robot, self.ctx.backend
        a, b = self.start_sl.values(), self.goal_sl.values()
        kind = self._path_kind()
        jp = self.paths.get(kind)
        tau_max, vel_max = self._limits(robot)
        self.player.stop()

        def job(prog, canc):
            path = jp if jp is not None else tj.joint_path(robot, a, b, kind, 81)
            prog(0.2)
            eng = DynamicsEngine(robot, backend)
            t0 = time.perf_counter()
            res = tj.topp(eng, path, tau_max, vel_max)
            return dict(res=res, jp=path, kind=kind, elapsed=time.perf_counter() - t0, backend=backend)
        self.run_async("TOPP-RA", job, self._topp_done)

    def _topp_done(self, out):
        robot = self.ctx.robot
        res, jp = out["res"], out["jp"]
        self.paths[out["kind"]] = jp
        self._draw_paths()
        for p in (self.p_phase, self.p_tn):
            self._clear(p)
        if not res.ok:
            self.topp_lbl.setText(f"<span style='color:{theme.BAD}'>{res.message}</span>")
            return
        self.topp_res = (res, jp)
        sd_max = float(np.nanmax(res.sd_grid))
        mvc = np.minimum(res.mvc, 3 * sd_max)
        self.p_phase.plot(res.grid, mvc, pen=pg.mkPen(theme.BAD, width=2.5), name="MVC (max-velocity curve)")
        self.p_phase.plot(res.grid, np.minimum(res.k_upper, 3 * sd_max),
                          pen=pg.mkPen(theme.MUTED, width=1.5, style=QtCore.Qt.DashLine),
                          name="controllable-set boundary")
        if res.quintic_t is not None:
            sq, sdq, _ = tj.time_scaling("quintic", res.quintic_T, res.quintic_t)
            self.p_phase.plot(sq, sdq, pen=pg.mkPen(theme.SERIES[1], width=2, style=QtCore.Qt.DashLine),
                              name=f"fastest feasible quintic (T = {res.quintic_T:.3f} s)")
        self.p_phase.plot(res.grid, res.sd_grid, pen=pg.mkPen(theme.ACCENT, width=3.5),
                          name=f"time-optimal ṡ(s) (T* = {res.duration:.3f} s)")
        self.p_phase.setYRange(0, 1.6 * sd_max)
        self.p_phase.setXRange(0, 1)
        for v in (1.0, -1.0):
            self.p_tn.addItem(pg.InfiniteLine(pos=v, angle=0, pen=pg.mkPen(theme.BAD, width=1.2,
                                                                           style=QtCore.Qt.DashLine)))
        for i in range(robot.n):
            self.p_tn.plot(res.t, res.tau_norm[:, i], pen=pen(i), name=f"τ{i + 1}/τmax")
        self.p_tn.setYRange(-1.15, 1.15)
        sat = np.max(np.maximum(np.abs(res.tau_norm), res.vel_norm), axis=1)
        gain = 100 * (1 - res.duration / res.quintic_T) if np.isfinite(res.quintic_T) else np.nan
        eng = "Pinocchio" if out["backend"] == "pinocchio" else "MR"
        self.topp_lbl.setText(
            f"T* = <b>{res.duration:.3f} s</b> (fastest quintic: {res.quintic_T:.3f} s, "
            f"<span style='color:{theme.GOOD}'>{gain:.0f}% faster</span>)<br>"
            f"max |τ|/τmax = {np.abs(res.tau_norm).max():.3f}, max |θ̇|/vmax = {res.vel_norm.max():.3f}<br>"
            f"solved in {out['elapsed'] * 1e3:.0f} ms (RNEA: {eng}, {len(res.grid)} grid points)")
        self.topp_info.setText(
            f"Bang-bang: at {100 * np.mean(sat > 0.98):.0f}% of the instants some actuator (or speed limit) is "
            "saturated. The optimal ṡ(s) accelerates as hard as possible, touches the MVC and brakes as hard as "
            "possible; the quintic (orange) must stay below the MVC everywhere with its fixed shape, so it is slower.")
        tr = tj.topp_trajectory(res, jp, 0.004)
        self.anim = dict(t=tr.t, theta=tr.theta, label="time-optimal motion")
        self.bottom.setCurrentIndex(2)
        if res.duration < 1.0:
            self.player.speed.setCurrentText("0.25×")
        self.player.set_times(tr.t)

    # ------------------------------------------------------------------ Ch. 11: control
    def _control_settings(self):
        tau_max, vel_max = self._limits(self.ctx.robot)
        return dict(tau_max=tau_max, vel_max=vel_max, Kp=self.Kp.value(), Ki=self.Ki.value(), Kd=self.Kd.value(), mass=self.mass_scale.value(),
                    payload=self.payload.value(), err0=np.radians(self.init_err.value()),
                    dt=self.ctrl_dt.value() / 1000.0, sat=self.cb_sat.isChecked(), Tf=self.Tf.value(),
                    scaling=self.scaling.currentText(), kind=self._path_kind(),
                    topp=self.ref_box.currentIndex() == 1)

    def _reference(self, robot, a, b, s, jp, backend):
        """Desired trajectory sampled at the control period, followed by a 0.5 s hold at the goal."""
        jp = jp if jp is not None else tj.joint_path(robot, a, b, s["kind"], 81)
        if s["topp"]:
            res = tj.topp(DynamicsEngine(robot, backend), jp, s["tau_max"], s["vel_max"], compare_quintic=False)
            if not res.ok:
                raise RuntimeError(res.message)
            tr = tj.topp_trajectory(res, jp, s["dt"])
        else:
            N = int(round(s["Tf"] / s["dt"])) + 1
            tr = tj.make_trajectory(robot, a, b, s["Tf"], N, s["kind"], s["scaling"], jpath=jp)
        return tj.hold(tr, 0.5, s["dt"]), jp

    def run_control(self, kinds=None):
        robot, backend = self.ctx.robot, self.ctx.backend
        a, b = self.start_sl.values(), self.goal_sl.values()
        s = self._control_settings()
        jp = self.paths.get(s["kind"])
        kinds = [self.ctrl_box.currentText()] if not kinds else list(kinds)
        self.player.stop()

        def job(prog, canc):
            ref, path = self._reference(robot, a, b, s, jp, backend)
            plant = ctl.plant_engine(robot, s["payload"])
            model = ctl.model_engine(robot, backend, s["mass"])
            out = {}
            for i, k in enumerate(kinds):
                t0 = time.perf_counter()
                r = ctl.run_controller(robot, ref, k, s["Kp"], s["Ki"], s["Kd"], plant=plant, model=model,
                                       theta0_offset=s["err0"], saturate=s["sat"],
                                       progress=lambda f, i=i: prog((i + f) / len(kinds)), cancel=canc)
                r.elapsed = time.perf_counter() - t0
                out[k] = r
                if r.cancelled:
                    return None
            return dict(results=out, ref=ref, jp=path, s=s, backend=backend)
        label = "Simulating closed loop" if len(kinds) == 1 else "Comparing controllers"
        self.run_async(label, job, self._control_done)

    def compare_controllers(self):
        self.run_control(kinds=ctl.CONTROLLERS)

    def _control_done(self, out):
        if out is None:
            return
        robot = self.ctx.robot
        results, ref, s = out["results"], out["ref"], out["s"]
        self.paths.setdefault(s["kind"], out["jp"])
        self._draw_paths()
        sel = self.ctrl_box.currentText() if self.ctrl_box.currentText() in results else next(iter(results))
        r = results[sel]
        for p in (self.p_err, self.p_tau):
            self._clear(p)
        limit_lines(self.p_tau, robot, 1.0)
        for i in range(robot.n):
            self.p_err.plot(r.t, np.degrees(r.err[:, i]), pen=pen(i), name=f"θₑ{i + 1}")
            self.p_tau.plot(r.t, r.tau[:, i], pen=pen(i), name=f"τ{i + 1}")
        self.p_err.setTitle(f"Tracking error θₑ = θ_d − θ  —  {sel}", color=theme.TEXT, size="10pt")
        model = ("exact model" if abs(s["mass"] - 1) < 1e-9 else f"model masses ×{s['mass']:.2f}") + \
            (f", unknown payload {s['payload']:.2f} kg" if s["payload"] > 0 else "")
        eng = "Pinocchio" if out["backend"] == "pinocchio" else "MR"
        stable = np.isfinite(r.rms_total)
        self.ctrl_lbl.setText(
            f"<b>{sel}</b> ({model}; model engine {eng})<br>"
            + (f"RMS error {np.degrees(r.rms_total):.3f}°, max {np.degrees(r.max_err):.3f}°, "
               f"final {np.degrees(np.abs(r.err[-1]).max()):.3f}°<br>"
               f"max |τ|/τmax = {np.nanmax(np.abs(r.tau) / robot.tau_max):.2f}"
               + (f", saturated {100 * r.saturated:.0f}% of the time" if s["sat"] else "")
               + f"<br>{len(r.t)} control steps simulated in {r.elapsed:.2f} s"
               if stable else f"<span style='color:{theme.BAD}'>unstable — the simulation diverged</span>"))
        if len(results) > 1:
            self._show_comparison(results, robot)
        else:
            self.bottom.setCurrentIndex(3)
        ee = np.array([robot.fk(q)[:3, 3] if np.all(np.isfinite(q)) else [np.nan] * 3 for q in r.theta])
        stride = max(1, int(round(0.004 / s["dt"])))       # animate at ~250 Hz samples
        idx = np.arange(0, len(r.t), stride)
        self.anim = dict(t=r.t[idx], theta=r.theta[idx], theta_d=r.theta_d[idx], ee=ee[idx],
                         label=f"{sel}: solid = actual robot, green ghost = desired")
        self.player.set_times(r.t[idx])

    def _show_comparison(self, results, robot):
        p = self.p_bar
        p.clear()
        names = list(results)
        rms = np.array([np.degrees(results[k].rms_total) for k in names])
        finite = np.where(np.isfinite(rms), rms, 0)
        cols = [pg.mkBrush(theme.SERIES[i]) for i in range(len(names))]
        p.addItem(pg.BarGraphItem(x=np.arange(len(names)), height=finite, width=0.6, brushes=cols))
        short = {"PD + gravity comp.": "PD + g(θ)", "Computed torque": "Comp. torque"}
        p.getAxis("bottom").setTicks([[(i, short.get(n, n)) for i, n in enumerate(names)]])
        for i, v in enumerate(rms):
            txt = pg.TextItem(f"{v:.3f}°" if np.isfinite(v) else "unstable", anchor=(0.5, 1.0), color=theme.TEXT)
            txt.setPos(i, finite[i])
            p.addItem(txt)
        p.setYRange(0, 1.2 * max(finite.max(), 1e-6))
        self.table.setRowCount(len(names))
        for i, k in enumerate(names):
            r = results[k]
            vals = [k, f"{np.degrees(r.rms_total):.4f}", f"{np.degrees(r.max_err):.4f}",
                    f"{np.degrees(np.abs(r.err[-1]).max()):.4f}",
                    f"{np.nanmax(np.abs(r.tau) / robot.tau_max):.2f}"]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                if j == 0:
                    it.setForeground(pg.mkColor(theme.SERIES[i]))
                self.table.setItem(i, j, it)
        best = names[int(np.nanargmin(np.where(np.isfinite(rms), rms, np.inf)))]
        self.cmp_info.setText(f"Same reference, gains and plant for all controllers. Best: <b>{best}</b>. "
                              "Try a wrong model (mass scale 1.3) or a payload: computed torque degrades gracefully, "
                              "PD sags under gravity, the integral term needs time to remove the steady error.")
        self.bottom.setCurrentIndex(4)

    # ------------------------------------------------------------------ animation
    def show_frame(self, k):
        if self.anim is None:
            return
        th = self.anim["theta"][k]
        if not np.all(np.isfinite(th)):
            return
        self.graphic.update(th)
        if "theta_d" in self.anim:
            self.ghost.update(self.anim["theta_d"][k])
        else:
            self.ghost.update(self.goal_sl.values())
        if "ee" in self.anim and k > 0:
            self.view.set_polyline("ctl.trace", self.anim["ee"][:k + 1], (0.3, 0.85, 1.0, 0.95), 2.5)
        else:
            self.view.remove("ctl.trace")
        t = self.anim["t"][k]
        for c in self.cursors.values():
            c.setPos(t)
        self.player.lbl.setText(f"t = {t:6.3f} s / {self.anim['t'][-1]:.3f} s — {self.anim['label']}")

    def on_hide(self):
        self.player.stop()
