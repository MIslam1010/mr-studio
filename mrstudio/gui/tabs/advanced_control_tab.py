"""Tab 9 -- task-space motion control, force control, hybrid motion-force control and impedance control
(MR Ch. 11: Secs. 11.3.3, 11.4.3, 11.5, 11.6, 11.7).  Physics and controllers live in mrstudio.task_control."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import task_control as tc
from ...control import model_engine
from ...dynamics import DynamicsEngine
from .. import theme
from ..base import BaseTab
from ..viewer import RobotGraphic, SceneView
from ..widgets import button, checkbox, combo, group, hbox, pen, plot, spin, value_label
from .dynamics_tab import AsyncMixin, Player, compact, form, no_si, time_cursor

THEORY = r"""
<h2>Task-space, force, hybrid and impedance control</h2>
<span class="ref">Modern Robotics, Ch. 11 (11.3.3, 11.4.3, 11.5, 11.6, 11.7) and Sec. 8.6 (task-space dynamics)</span>
<p>The end-effector configuration is $X=T_{sb}(\theta)\in SE(3)$ with body twist $[\mathcal{V}_b]=X^{-1}\dot{X}$; the desired
motion is $X_d(t)$ with $[\mathcal{V}_d]=X_d^{-1}\dot{X}_d$.  Subtracting elements of SE(3) makes no sense, so the
configuration error is the twist that takes $X$ to $X_d$ in unit time:</p>
$$[X_e]=\log(X^{-1}X_d)$$
<h3>1. Velocity-input task-space control (11.3.3)</h3>
<p>A velocity-controlled (kinematic) robot receives commanded joint rates.  Feedforward plus PI feedback:</p>
$$\mathcal{V}_b=[\mathrm{Ad}_{X^{-1}X_d}]\mathcal{V}_d+K_pX_e+K_i\int_{0}^{t}X_e\,dt\quad(11.16)$$
$$\dot{\theta}=J_b^{\dagger}(\theta)\mathcal{V}_b$$
<p>$[\mathrm{Ad}_{X^{-1}X_d}]$ expresses the feedforward twist in the <i>actual</i> end-effector frame.  With the
alternative representation $X=(R,p)$ and velocity $(\omega_b,\dot{p})$:</p>
$$\left[\omega_b;\,\dot{p}\right]=\left[R^{T}R_d\,\omega_d;\,\dot{p}_d\right]+K_pX_e+K_i\int X_e\,dt\quad(11.18)$$
$$X_e=\left[\omega_e;\,p_d-p\right]$$
<p>with $[\omega_e]=\log(R^{T}R_d)$: the origin of {b} then moves in straight lines.  For a set-point and
$K_i=0$ the error decays exactly as $X_e(t)=X_e(0)e^{-k_pt}$ (Validation tab).</p>
<h3>Task-space dynamics (8.6)</h3>
$$\mathcal{F}=\Lambda(\theta)\dot{\mathcal{V}}+\eta(\theta,\mathcal{V})\quad(8.89)$$
$$\Lambda=J^{-T}MJ^{-1}\quad(8.90),\qquad \eta=J^{-T}h-\Lambda\dot{J}J^{-1}\mathcal{V}\quad(8.91)$$
<p>For the redundant Panda and near singularities the app uses the general form
$\Lambda=(JM^{-1}J^{T}+\mu^{2}I)^{-1}$, $\bar{J}=M^{-1}J^{T}\Lambda$, $\eta=\bar{J}^{T}h-\Lambda\dot{J}\dot{\theta}$
($\mu=0$ unless $\sigma_{min}(J)$ is small), and the self-motion is damped in the null space with
$(I-J^{T}\bar{J}^{T})(-k_{null}\dot{\theta})$.  For a square, regular $J$ this is exactly (8.90)–(8.91).</p>
<h3>2. Task-space motion control with torques (11.4.3)</h3>
$$\tau=J_b^{T}\left(\tilde{\Lambda}\,a+\tilde{\eta}(\theta,\mathcal{V}_b)\right)\quad(11.47)$$
$$a=\frac{d}{dt}([\mathrm{Ad}_{X^{-1}X_d}]\mathcal{V}_d)+K_pX_e+K_i\int X_e\,dt+K_d\mathcal{V}_e$$
$$\mathcal{V}_e=[\mathrm{Ad}_{X^{-1}X_d}]\mathcal{V}_d-\mathcal{V}$$
<p>The feedforward acceleration is computed exactly: with $T=X^{-1}X_d$ and $\mathcal{V}_T=\mathcal{V}_d-[\mathrm{Ad}_{T^{-1}}]\mathcal{V}_b$,</p>
$$\frac{d}{dt}([\mathrm{Ad}_T]\mathcal{V}_d)=[\mathrm{Ad}_T]\left(\dot{\mathcal{V}}_d+[\mathrm{ad}_{\mathcal{V}_T}]\mathcal{V}_d\right)$$
<p>With an exact model and a set-point, $\ddot{X}_e+K_d\dot{X}_e+K_pX_e=0$ along the error screw.  The alternative to
(11.47) is inverse kinematics (11.44)–(11.46) followed by joint-space computed torque (tab ⑤).</p>
<h3>3. Force control (11.5)</h3>
$$M\ddot{\theta}+c(\theta,\dot{\theta})+g(\theta)+b(\dot{\theta})+J^{T}\mathcal{F}_{tip}=\tau\quad(11.48)$$
<p>Slow motion: $g(\theta)+J^{T}\mathcal{F}_{tip}=\tau$ (11.49).  Without a force sensor: $\tau=\tilde{g}(\theta)+J^{T}\mathcal{F}_d$ (11.50).
With a wrist force–torque sensor, PI force control with feedforward and gravity compensation:</p>
$$\tau=\tilde{g}+J^{T}\left(\mathcal{F}_d+K_{fp}\mathcal{F}_e+K_{fi}\int\mathcal{F}_e\,dt\right)\quad(11.51)$$
$$\mathcal{F}_e=\mathcal{F}_d-\mathcal{F}_{tip},\qquad K_{fp}\mathcal{F}_e+K_{fi}\int\mathcal{F}_e(t)dt=0\quad(11.52)$$
<p>Nothing to push against → the robot accelerates; velocity damping limits this:</p>
$$\tau=\tilde{g}+J^{T}\left(\mathcal{F}_d+K_{fp}\mathcal{F}_e+K_{fi}\int\mathcal{F}_e\,dt-K_{damp}\mathcal{V}\right)\quad(11.54)$$
<p>Here the end-effector starts a few mm above a stiff spring–damper surface, approaches at speed
$\approx F_d/K_{damp}$, touches and settles at $F_d$.  The integral runs only in contact (anti-windup) and removes
the error of a wrong gravity model $\tilde{g}$.</p>
<h3>4. Hybrid motion–force control (11.6)</h3>
<p>A rigid surface imposes $k$ natural constraints $A(\theta)\mathcal{V}=0$ (11.55); here $A=[0\ \ n^{T}R]$
(no velocity along the surface normal $n$).  The constrained dynamics split with</p>
$$P=I-A^{T}(A\Lambda^{-1}A^{T})^{-1}A\Lambda^{-1}\quad(11.60)$$
<p>$P$ (rank $n-k$) keeps the wrenches that move the end-effector tangent to the surface; $I-P$ those that push
against it.  The controller sums a motion and a force controller, each in its own subspace:</p>
$$\tau=J_b^{T}\left(P\,\tilde{\Lambda}\,a+(I-P)\,\mathcal{F}_{f}+\tilde{\eta}\right)\quad(11.61)$$
$$\mathcal{F}_{f}=\mathcal{F}_d+K_{fp}\mathcal{F}_e+K_{fi}\int\mathcal{F}_e\,dt$$
<p>with the same motion command $a$ as in (11.47): motion control, force control, Coriolis and gravity.</p>
<p>As the book requires $A\dot{\mathcal{V}}_d=0$, the requested acceleration is projected onto $A\,a=0$ (the
penetration of the compliant surface is not a motion error).  Checks: $P^{2}=P$, $A\Lambda^{-1}P=0$,
$(I-P)A^{T}=A^{T}$.</p>
<h3>5. Impedance control (11.7)</h3>
<p>Render a virtual mass–spring–damper at the end-effector when a user pushes with $f_{ext}$:</p>
$$M\ddot{x}+B\dot{x}+Kx=f_{ext}\quad(11.64)$$
$$\tau=J^{T}\left(\tilde{\Lambda}\ddot{x}+\tilde{\eta}-(M\ddot{x}+B\dot{x}+Kx)\right)\quad(11.65)$$
<p><b>Without F/T sensor</b> the usual simplification $M=0$, $\tilde{\Lambda}\ddot{x}$ dropped:
$\tau=J^{T}(\tilde{\eta}-B\dot{x}-Kx)$ — the user feels the arm's own inertia $\Lambda(\theta)$.
<b>With the wrist sensor</b> the virtual mass is rendered through the admittance law</p>
$$\ddot{x}_d=M^{-1}(f_{ext}-B\dot{x}-Kx)\quad(11.66)$$
<p>and $\tau=J^{T}(\tilde{\Lambda}\ddot{x}_d+\tilde{\eta}-f_{ext})$, which gives exactly (11.64); the plot compares the
robot with the ideal ODE.  Translation and rotation are decoupled as $(\omega_b,\dot{p})$ (11.18); the rotational
spring uses $\log(R^{T}R_0)$.</p>
<p>All plants are Pinocchio (ABA + RK4, controller at 1 kHz with zero-order hold); the environment acts through
$\mathcal{F}_{tip}$ of (11.48).  Singularities: damped least squares
$J^{\dagger}=J^{T}(JJ^{T}+\lambda^{2}I)^{-1}$ with $\lambda$ growing only when $\sigma_{min}<\epsilon$.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 4 §5.4, p. 195; §6, pp. 200–206', 'workspace control; constrained manipulators, hybrid force/position'),
    ('gotor', '§6.3, pp. 170–172', 'computed-torque and task-space control'),
    ('corke', '§8.2–8.3, pp. 180–186', 'resolved-rate motion control; force relationships'),
    ('astrom_murray', '§6.2–6.3, pp. 175–194', 'state feedback and eigenvalue placement'),
    ('astrom_murray', '§12.1–12.2, pp. 347–357', 'model uncertainty and robust stability'),
])


MODE_HELP = {
    "velocity": "Kinematic robot driven by commanded joint velocities θ̇ = J_b†(θ) 𝒱_b.  Toggle the feedforward term "
                "and K_i to see their effect (Compare variants).",
    "motion": "Torque-controlled robot (Pinocchio plant) tracking X_d(t) with the task-space computed-torque law.",
    "force": "The end-effector starts above a stiff surface (spring–damper) and must press on it with F_d.",
    "hybrid": "Wiping: motion along the surface (P), force F_d normal to it (I − P).",
    "impedance": "The arm renders a virtual mass–spring–damper; a user pushes the end-effector with a force pulse.",
}

# per-mode parameters: key -> (label, lo, hi, value, decimals, step, suffix)
PARAMS = {
    "velocity": {"kp": ("K_p", 0, 50, 5.0, 2, 0.5, " 1/s"), "ki": ("K_i", 0, 100, 1.0, 2, 0.5, " 1/s²")},
    "motion": {"kp": ("K_p", 0, 2500, 100.0, 1, 10, " 1/s²"), "ki": ("K_i", 0, 2000, 0.0, 1, 10, " 1/s³"),
               "kd": ("K_d", 0, 200, 20.0, 1, 2, " 1/s"), "mass": ("Model mass scale", 0.2, 3, 1.0, 2, 0.05, " ×")},
    "force": {"fd": ("Desired force F_d", 0, 100, 15.0, 1, 1, " N"), "kfp": ("K_fp", 0, 20, 1.0, 2, 0.1, ""),
              "kfi": ("K_fi", 0, 200, 20.0, 1, 2, " 1/s"), "kdv": ("K_damp (linear)", 0, 500, 40.0, 1, 5, " N·s/m"),
              "kdw": ("K_damp (angular)", 0, 50, 2.0, 2, 0.5, " N·m·s"), "gap": ("Initial gap", 0, 100, 10.0, 1, 1, " mm"),
              "kenv": ("Surface stiffness", 100, 50000, 5000.0, 0, 500, " N/m"),
              "mass": ("Model mass scale", 0.2, 3, 1.0, 2, 0.05, " ×")},
    "hybrid": {"kp": ("K_p", 0, 2500, 100.0, 1, 10, " 1/s²"), "kd": ("K_d", 0, 200, 20.0, 1, 2, " 1/s"),
               "fd": ("Desired force F_d", 0, 100, 15.0, 1, 1, " N"), "kfp": ("K_fp", 0, 20, 1.0, 2, 0.1, ""),
               "kfi": ("K_fi", 0, 200, 20.0, 1, 2, " 1/s"),
               "kenv": ("Surface stiffness", 100, 50000, 5000.0, 0, 500, " N/m"),
               "mass": ("Model mass scale", 0.2, 3, 1.0, 2, 0.05, " ×")},
    "impedance": {"m": ("Virtual mass M", 0.1, 50, 2.0, 2, 0.5, " kg"), "b": ("Virtual damping B", 0, 500, 20.0, 1, 2, " N·s/m"),
                  "k": ("Virtual stiffness K", 0, 5000, 200.0, 0, 20, " N/m"), "f": ("Push force", 0, 100, 10.0, 1, 1, " N"),
                  "tp": ("Push duration", 0.05, 5, 1.0, 2, 0.1, " s")},
}


class AdvancedControlTab(AsyncMixin, BaseTab):
    title = "⑧ Task-space"
    needs_dynamics = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._init_async()
        self.view = SceneView()
        self.graphic = None
        self.anim = None
        self.results: list = []

        # ---- controller choice ----------------------------------------------------------------
        self.mode_box = compact(combo([tc.MODE_LABELS[m] for m in tc.MODES]), 22)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.mode_help = value_label()
        mode_box = group("Controller — MR Ch. 11", self.mode_box, self.mode_help)

        # ---- desired motion ----------------------------------------------------------------------
        self.shape = combo(list(tc.SHAPES))
        self.size = spin(0.01, 0.5, 0.1, 3, 0.01, " m")
        self.Tf = spin(0.5, 20, 4.0, 1, 0.5, " s")
        self.wrist = spin(0, 90, 20, 0, 5, " °")
        self.rep = combo(["Body twist 𝒱_b  (11.16 / 11.47)", "(ω_b, ṗ)  (11.18)"])
        self.err0 = spin(-20, 20, 1.0, 1, 0.5, " °")
        self.motion_form = form(("Path of the origin", self.shape), ("Size", self.size), ("Period T", self.Tf),
                                ("Wrist rotation ±", self.wrist), ("Representation", self.rep),
                                ("Initial joint error", self.err0))
        self.motion_box = group("Desired end-effector motion X_d(t)", self.motion_form)

        # ---- gains per mode --------------------------------------------------------------------
        self.par: dict = {}
        self.stack = QtWidgets.QStackedWidget()
        self.cb_ff = {}
        for mode in tc.MODES:
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            v.setContentsMargins(0, 0, 0, 0)
            rows = []
            self.par[mode] = {}
            for key, (lab, lo, hi, val, dec, step, suf) in PARAMS[mode].items():
                w = spin(lo, hi, val, dec, step, suf)
                self.par[mode][key] = w
                rows.append((lab, w))
            v.addLayout(form(*rows))
            if mode in ("velocity", "motion"):
                self.cb_ff[mode] = checkbox("Feedforward term [Ad_{X⁻¹X_d}]𝒱_d", None, True)
                v.addWidget(self.cb_ff[mode])
            if mode == "impedance":
                self.push_dir = combo(["−z (down)", "+x", "+y", "+z"])
                self.cb_sensor = checkbox("Wrist F/T sensor: render M (11.66)", None, True)
                v.addLayout(form(("Push direction", self.push_dir)))
                v.addWidget(self.cb_sensor)
            self.stack.addWidget(page)
        self.run_btn = button("Run simulation", self.run, primary=True)
        self.cmp_btn = button("Compare variants", self.compare)
        self.res_lbl = value_label()
        gains_box = group("Gains and environment", self.stack, hbox(self.run_btn, self.cmp_btn), self.res_lbl)

        self.cb_trace = checkbox("Trace actual end-effector path", lambda _: self._redraw(), True)
        self.player = Player(self)
        self.player.frame.connect(self.show_frame)
        self.status_lbl = value_label()
        controls = QtWidgets.QWidget()
        controls.setMinimumWidth(340)
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(mode_box)
        cl.addWidget(self.motion_box)
        cl.addWidget(gains_box)
        cl.addWidget(group("Playback", self.player.widget, self.cb_trace, self.busy_row, self.status_lbl))
        cl.addStretch(1)

        # ---- plots -----------------------------------------------------------------------------
        self.p_w = plot("Orientation error ω_e  (X_e angular part)", "ω_e [deg]")
        self.p_v = plot("Position error", "[mm]")
        self.p_f = plot("Contact force", "F [N]")
        self.p_x = plot("Displacement along the surface normal", "[mm]")
        self.p_cmd = plot("Joint commands", "")
        self.p_s = plot("Smallest singular value of J (singularity measure)", "σ_min", legend=False)
        self.p_cmp = plot("Compare variants", "")
        self.cmp_info = value_label()
        self.cursors = {id(p): time_cursor(p) for p in (self.p_w, self.p_v, self.p_f, self.p_x, self.p_cmd, self.p_s)}
        for p in (self.p_w, self.p_v, self.p_f, self.p_x, self.p_cmd, self.p_s, self.p_cmp):
            no_si(p)

        def page(*ws, info=None):
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            h = QtWidgets.QHBoxLayout()
            for x in ws:
                h.addWidget(x, 1)
            v.addLayout(h, 1)
            if info is not None:
                v.addWidget(info)
            return w
        self.bottom = QtWidgets.QTabWidget()
        self.bottom.addTab(page(self.p_w, self.p_v), "Pose error X_e")
        self.bottom.addTab(page(self.p_f, self.p_x), "Force / impedance")
        self.bottom.addTab(page(self.p_cmd, self.p_s), "Joint commands")
        self.bottom.addTab(page(self.p_cmp, info=self.cmp_info), "Compare variants")

        self.assemble(controls, self.view, self.bottom, THEORY, controls_width=375, bottom_ratio=(5, 4))
        self._mode_changed()
        self.on_robot_changed(ctx.robot)

    # ------------------------------------------------------------------ helpers
    def status(self, text, color=None):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color:{color}" if color else "")

    def mode(self) -> str:
        return tc.MODES[self.mode_box.currentIndex()]

    def _clear(self, p):
        p.clear()
        if p.plotItem.legend is not None:
            p.plotItem.legend.clear()
        if id(p) in self.cursors:
            p.addItem(self.cursors[id(p)], ignoreBounds=True)

    def _mode_changed(self, *_):
        m = self.mode()
        self.stack.setCurrentIndex(tc.MODES.index(m))
        self.mode_help.setText(MODE_HELP[m])
        has_motion = m in ("velocity", "motion", "hybrid")
        self.motion_box.setEnabled(has_motion)
        self.err0.setEnabled(m in ("velocity", "motion"))
        self.rep.setEnabled(m in ("velocity", "motion"))
        self.wrist.setEnabled(m in ("velocity", "motion") and not self.ctx.robot.planar)
        self.player.stop()
        self.anim = None
        self.view.remove_prefix("tsc.env")
        self.view.remove_prefix("tsc.path")
        self.view.remove_prefix("tsc.target")
        self.view.remove("tsc.trace")

    # ------------------------------------------------------------------ robot
    def on_robot_changed(self, robot):
        self.cancel_jobs()
        self.player.stop()
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("tsc.")
        self.graphic = RobotGraphic(self.view, robot, "tsc.robot")
        self.th_start = tc.start_config(robot)
        self.graphic.update(self.th_start)
        self.view.frame_scene(robot.reach, planar=robot.planar,
                              center=(0, 0, 0.25 * robot.reach * (not robot.planar)))
        self.size.setValue(tc.default_size(robot))
        self.wrist.setEnabled(not robot.planar and self.mode() in ("velocity", "motion"))
        if robot.planar and robot.n < 3:
            self.rep.setCurrentIndex(1)
        for p in (self.p_w, self.p_v, self.p_f, self.p_x, self.p_cmd, self.p_s, self.p_cmp):
            self._clear(p)
        self.anim = None
        self.res_lbl.setText("")
        self.cmp_info.setText("")
        rows = tc.task_rows(robot)
        names = ["ω_x", "ω_y", "ω_z", "v_x", "v_y", "v_z"]
        self.status(f"{robot.name}: controlled task components ({len(rows)}): "
                    + ", ".join(names[i] for i in rows)
                    + (" — redundant: null-space damping" if robot.n > len(rows) else ""))

    # ------------------------------------------------------------------ settings / scenario
    def _settings(self, mode=None):
        mode = mode or self.mode()
        robot = self.ctx.robot
        s = {k: w.value() for k, w in self.par[mode].items()}
        s.update(mode=mode, shape=self.shape.currentText(), size=self.size.value(), Tf=self.Tf.value(),
                 wrist=0.0 if robot.planar else np.radians(self.wrist.value()),
                 rep="body" if self.rep.currentIndex() == 0 else "hybrid", err0=np.radians(self.err0.value()),
                 ff=self.cb_ff[mode].isChecked() if mode in self.cb_ff else True)
        if mode == "impedance":
            s["dir"] = self.push_dir.currentIndex()
            s["sensor"] = self.cb_sensor.isChecked()
        return s

    @staticmethod
    def _gains(s) -> tc.Gains:
        g = tc.Gains()
        for k in ("kp", "ki", "kd", "fd", "kfp", "kfi"):
            if k in s:
                setattr(g, k, float(s[k]))
        if "kdv" in s:
            g.kdamp_v, g.kdamp_w = s["kdv"], s["kdw"]
        if s["mode"] == "impedance":
            g.imp_m, g.imp_b, g.imp_k, g.f_sensor = s["m"], s["b"], s["k"], s["sensor"]
        g.feedforward = s["ff"]
        return g

    @staticmethod
    def _simulate(robot, s, th_start, label="", progress=None, cancel=None):
        """Run one scenario (in the worker thread).  Returns a TaskResult."""
        mode = s["mode"]
        if mode in ("force", "hybrid"):
            th_start = tc.start_config(robot, contact=True)
        X0 = robot.fk(th_start)
        g = AdvancedControlTab._gains(s)
        offs = s["err0"] * np.resize([1, -1, 1, -1, 1, -1, 1], robot.n)
        if mode == "velocity":
            tr = tc.make_task_trajectory(robot, X0, s["shape"], s["size"], s["Tf"], 0.002, wrist=s["wrist"])
            return tc.velocity_control(robot, tr, th_start + offs, s["kp"], s["ki"], s["ff"], s["rep"], label=label,
                                       cancel=cancel)
        plant = DynamicsEngine(robot, "pinocchio")
        model = model_engine(robot, "pinocchio", s.get("mass", 1.0))
        if mode == "motion":
            tr = tc.make_task_trajectory(robot, X0, s["shape"], s["size"], s["Tf"], 0.001, wrist=s["wrist"])
            return tc.torque_control("motion", robot, th_start + offs, s["Tf"], g, traj=tr, rep=s["rep"], plant=plant,
                                     model=model, label=label, progress=progress, cancel=cancel)
        if mode == "force":
            wall = tc.make_wall(robot, X0, s["gap"] / 1000.0, k=s["kenv"], b=50.0)
            return tc.torque_control("force", robot, th_start, 3.0, g, wall=wall, plant=plant, model=model,
                                     label=label, progress=progress, cancel=cancel)
        if mode == "hybrid":
            nrm = tc.surface_normal(robot)
            shape = s["shape"] if not robot.planar else "line"
            tr = tc.make_task_trajectory(robot, X0, shape, s["size"], s["Tf"], 0.001,
                                         plane=tc.shape_plane(robot, X0, nrm))
            wall = tc.make_wall(robot, X0, 0.0, k=s["kenv"], b=50.0)
            return tc.torque_control("hybrid", robot, th_start, s["Tf"], g, traj=tr, wall=wall, plant=plant,
                                     model=model, label=label, progress=progress, cancel=cancel)
        dirs = [np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
        d = dirs[s["dir"]]
        if robot.planar and abs(d[2]) > 0:
            d = np.array([0, -1.0, 0])                     # planar: "down" is −y
        T = max(3.0, 0.5 + s["tp"] + 2.0)
        return tc.torque_control("impedance", robot, th_start, T, g, push=(s["f"] * d, 0.5, 0.5 + s["tp"]),
                                 plant=plant, model=model, label=label, progress=progress, cancel=cancel)

    def _variants(self, s):
        m = s["mode"]
        V = []
        if m == "velocity":
            V = [("FF + PI", dict(ff=True)), ("FF + P (K_i = 0)", dict(ff=True, ki=0.0)),
                 ("PI, no feedforward", dict(ff=False)), ("P only, no feedforward", dict(ff=False, ki=0.0))]
        elif m == "motion":
            V = [("exact model", dict(mass=1.0)), ("model masses ×1.3", dict(mass=1.3)),
                 ("×1.3 + integral K_i = 200", dict(mass=1.3, ki=200.0)), ("exact, no feedforward", dict(mass=1.0, ff=False))]
        elif m == "force":
            V = [("PI (11.54), exact g̃", dict(mass=1.0)), ("PI, wrong model g̃ ×1.2", dict(mass=1.2)),
                 ("P only (K_fi = 0), g̃ ×1.2", dict(mass=1.2, kfi=0.0)),
                 ("no sensor: τ = g̃ + JᵀF_d (11.50), g̃ ×1.2", dict(mass=1.2, kfp=0.0, kfi=0.0))]
        elif m == "hybrid":
            V = [("as set", {}), ("model ×1.2", dict(mass=1.2)), ("model ×1.2, K_fi = 0", dict(mass=1.2, kfi=0.0))]
        else:
            V = [("F/T sensor: M rendered (11.66)", dict(sensor=True)), ("no sensor, M̃ = 0 (11.65)", dict(sensor=False))]
        out = []
        for lab, ov in V:
            d = dict(s)
            d.update(ov)
            out.append((lab, d))
        return out

    # ------------------------------------------------------------------ run
    def run(self):
        self._launch([("", self._settings())], compare=False)

    def compare(self):
        self._launch(self._variants(self._settings()), compare=True)

    def _launch(self, runs, compare):
        robot, th0 = self.ctx.robot, self.th_start.copy()
        self.player.stop()

        def job(prog, canc):
            out = []
            for i, (lab, s) in enumerate(runs):
                t0 = time.perf_counter()
                r = self._simulate(robot, s, th0, lab, progress=lambda f, i=i: prog((i + f) / len(runs)), cancel=canc)
                r.info["elapsed"] = time.perf_counter() - t0
                r.info["settings"] = s
                if r.cancelled:
                    return None
                out.append(r)
                prog((i + 1) / len(runs))
            return dict(results=out, compare=compare, robot=robot.name)
        self.run_async("Simulating" if not compare else "Comparing variants", job, self._done)

    def _done(self, out):
        if out is None or out["robot"] != self.ctx.robot.name:
            return
        res = out["results"]
        self.results = res
        self._plot(res[0])
        if out["compare"]:
            self._plot_compare(res)
            self.bottom.setCurrentIndex(3)
        else:
            self.bottom.setCurrentIndex(1 if res[0].mode in ("force", "impedance") else 0)
        self._summary(res[0])
        self._start_anim(res[0])

    # ------------------------------------------------------------------ plots
    def _plot(self, r: tc.TaskResult):
        robot = self.ctx.robot
        for p in (self.p_w, self.p_v, self.p_f, self.p_x, self.p_cmd, self.p_s):
            self._clear(p)
        rows = tc.task_rows(robot)
        names = ["x", "y", "z"]
        rep = r.info.get("rep", "body")
        for i in range(3):
            if i in rows:
                self.p_w.plot(r.t, np.degrees(r.Xe[:, i]), pen=pen(i), name=f"ω_e,{names[i]}")
            if i + 3 in rows:
                lab = (f"v_e,{names[i]} (body)" if rep == "body" else f"(p_d − p)_{names[i]}")
                self.p_v.plot(r.t, 1e3 * r.Xe[:, i + 3], pen=pen(i + 3), name=lab)
        self.p_v.plot(r.t, 1e3 * r.pos_err, pen=pg.mkPen(theme.TEXT, width=2, style=QtCore.Qt.DashLine),
                      name="‖p_d − p‖")
        if r.mode in ("force", "impedance"):
            self.p_w.setTitle("Orientation drift from the start ω_e", color=theme.TEXT, size="10pt")
            self.p_v.setTitle("Position offset from the start", color=theme.TEXT, size="10pt")
        else:
            self.p_w.setTitle("Orientation error ω_e  (angular part of X_e)", color=theme.TEXT, size="10pt")
            self.p_v.setTitle("Position error  (linear part of X_e)", color=theme.TEXT, size="10pt")
        # force / impedance page
        if r.force is not None and r.force_d is not None:
            self.p_f.setTitle("Normal contact force: robot on surface", color=theme.TEXT, size="10pt")
            self.p_f.plot(r.t, r.force_d, pen=pg.mkPen(theme.GOOD, width=2, style=QtCore.Qt.DashLine), name="F_d")
            self.p_f.plot(r.t, r.force, pen=pen(0, 2.5), name="F (sensor)")
            n = r.info["normal"]
            wall = r.info["wall"]
            depth = -(r.p - wall.point) @ n
            self.p_x.setTitle("Penetration into the surface (spring–damper)", color=theme.TEXT, size="10pt")
            self.p_x.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.MUTED, width=1)))
            self.p_x.plot(r.t, 1e3 * depth, pen=pen(1, 2.5), name="depth [mm] (> 0: in contact)")
            self.p_x.plot(r.t, np.full(len(r.t), 1e3 * float(r.force_d[0]) / wall.k),
                          pen=pg.mkPen(theme.GOOD, width=1.5, style=QtCore.Qt.DashLine), name="F_d / k")
        elif r.mode == "impedance":
            self.p_f.setTitle("User force f_ext on the end-effector", color=theme.TEXT, size="10pt")
            fu = r.f_user
            for i in range(3):
                if np.abs(fu[:, i]).max() > 1e-9:
                    self.p_f.plot(r.t, fu[:, i], pen=pen(i, 2.5), name=f"f_{names[i]}")
            self.p_x.setTitle("Displacement along the push vs ideal M ẍ + B ẋ + K x = f", color=theme.TEXT,
                              size="10pt")
            if r.x_ref is not None:
                self.p_x.plot(r.t, 1e3 * r.x_ref, pen=pg.mkPen(theme.GOOD, width=2.5, style=QtCore.Qt.DashLine),
                              name="ideal (11.64)")
                self.p_x.plot(r.t, 1e3 * r.x_act, pen=pen(0, 2.5), name="robot")
        else:
            self.p_f.setTitle("Contact force (force / hybrid modes only)", color=theme.TEXT, size="10pt")
            self.p_x.setTitle("", color=theme.TEXT, size="10pt")
        # commands
        if r.mode == "velocity":
            self.p_cmd.setTitle("Commanded joint velocities θ̇ = J†𝒱", color=theme.TEXT, size="10pt")
            self.p_cmd.setLabel("left", "θ̇ [deg/s]")
            for i in range(robot.n):
                self.p_cmd.plot(r.t, np.degrees(r.cmd[:, i]), pen=pen(i), name=f"θ̇{i + 1}")
        else:
            self.p_cmd.setTitle("Joint torques τ", color=theme.TEXT, size="10pt")
            self.p_cmd.setLabel("left", "τ [N·m]")
            for i in range(robot.n):
                self.p_cmd.plot(r.t, r.cmd[:, i], pen=pen(i), name=f"τ{i + 1}")
        if r.smin is not None and np.any(r.smin > 0):
            self.p_s.plot(r.t, r.smin, pen=pen(4, 2.5))
            self.p_s.addItem(pg.InfiniteLine(pos=0.04, angle=0, pen=pg.mkPen(theme.BAD, width=1.2,
                                                                             style=QtCore.Qt.DashLine)))

    def _plot_compare(self, res):
        p = self.p_cmp
        self._clear(p)
        m = res[0].mode
        lines = []
        for i, r in enumerate(res):
            if m in ("velocity", "motion"):
                y, unit = 1e3 * r.pos_err, "‖p_d − p‖ [mm]"
                lines.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {r.label}: max {y.max():.2f} mm, "
                             f"RMS {np.sqrt(np.mean(y ** 2)):.2f} mm, max |ω_e| {np.degrees(r.err_w.max()):.3f}°")
            elif m in ("force", "hybrid"):
                y, unit = r.force, "normal force F [N]"
                late = r.t > r.t[-1] - 0.5
                lines.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {r.label}: final force "
                             f"{y[late].mean():.2f} N (F_d = {r.force_d[0]:.1f} N), max {y.max():.1f} N")
            else:
                y, unit = 1e3 * r.x_act, "displacement [mm]"
                dev = np.abs(r.x_act - r.x_ref).max() / max(np.abs(r.x_ref).max(), 1e-12)
                lines.append(f"<span style='color:{theme.SERIES[i]}'>■</span> {r.label}: peak {y.max():.1f} mm, "
                             f"deviation from ideal MSD {100 * dev:.1f} %")
            p.plot(r.t, y, pen=pen(i, 2.5), name=r.label)
        if m == "impedance" and res[0].x_ref is not None:
            p.plot(res[0].t, 1e3 * res[0].x_ref, pen=pg.mkPen(theme.TEXT, width=1.5, style=QtCore.Qt.DashLine),
                   name="ideal M ẍ + B ẋ + K x = f")
        if m in ("force", "hybrid"):
            p.plot(res[0].t, res[0].force_d, pen=pg.mkPen(theme.TEXT, width=1.5, style=QtCore.Qt.DashLine), name="F_d")
        p.setLabel("left", unit)
        p.setTitle(f"Compare variants — {tc.MODE_LABELS[m]}", color=theme.TEXT, size="10pt")
        self.cmp_info.setText("<br>".join(lines))

    def _summary(self, r: tc.TaskResult):
        el = r.info.get("elapsed", 0.0)
        txt = []
        if r.diverged:
            txt.append(f"<span style='color:{theme.BAD}'>⚠ unstable — the simulation diverged at t = {r.t[-1]:.2f} s</span>")
        if r.mode == "hybrid":
            n = r.info["normal"]
            e = r.pd - r.p
            e = np.linalg.norm(e - np.outer(e @ n, n), axis=1)
            settle = r.t > 0.5
            txt.append(f"tangential tracking error (motion subspace): max {1e3 * e.max():.2f} mm, "
                       f"after 0.5 s {1e3 * e[settle].max():.3f} mm")
        elif r.mode in ("velocity", "motion"):
            settle = r.t > min(1.0, 0.25 * r.t[-1])
            txt.append(f"max ‖p_d − p‖ = {1e3 * r.pos_err.max():.2f} mm "
                       f"(after {r.t[settle][0]:.2f} s: {1e3 * r.pos_err[settle].max():.3f} mm), "
                       f"max |ω_e| = {np.degrees(r.err_w[settle].max()):.3f}°")
        if r.mode in ("force", "hybrid"):
            late = r.t > r.t[-1] - 0.5
            txt.append(f"final force {r.force[late].mean():.3f} N (F_d = {r.force_d[0]:.1f} N), "
                       f"peak {r.force.max():.1f} N")
        if r.mode == "impedance" and r.x_ref is not None:
            dev = np.abs(r.x_act - r.x_ref).max() / max(np.abs(r.x_ref).max(), 1e-12)
            txt.append(f"peak displacement {1e3 * r.x_act.max():.1f} mm (ideal {1e3 * r.x_ref.max():.1f} mm), "
                       f"deviation from 11.64: {100 * dev:.2f} %")
        if r.smin is not None and len(r.smin) and r.smin.min() < 0.04:
            txt.append(f"<span style='color:{theme.WARN}'>near a singularity: σ_min = {r.smin.min():.3f} "
                       "(damped least squares active)</span>")
        plant = "kinematic robot" if r.mode == "velocity" else "Pinocchio plant, 1 kHz control"
        txt.append(f"{len(r.t)} steps simulated in {el:.2f} s ({plant})")
        self.res_lbl.setText("<br>".join(txt))

    # ------------------------------------------------------------------ animation
    def _start_anim(self, r: tc.TaskResult):
        stride = max(1, int(round(0.01 / (r.t[1] - r.t[0]))))
        idx = np.arange(0, len(r.t), stride)
        self.anim = dict(r=r, idx=idx)
        self._draw_static(r)
        self.player.set_times(r.t[idx])

    def _draw_static(self, r: tc.TaskResult):
        robot = self.ctx.robot
        self.view.remove_prefix("tsc.env")
        self.view.remove_prefix("tsc.path")
        self.view.remove_prefix("tsc.target")
        self.view.remove("tsc.trace")
        if r.mode in ("velocity", "motion", "hybrid"):
            self.view.set_polyline("tsc.path", r.pd, (0.55, 1.0, 0.6, 0.85), 2.5)
        wall = r.info.get("wall")
        if wall is not None:
            R = 0.3 * robot.reach
            th = 0.01 * robot.reach
            if robot.planar:            # in the plane the surface is a line (ground hatching underneath)
                n = wall.normal
                a = np.cross(n, [0, 0, 1.0])
                self.view.set_tube("tsc.env.wall", wall.point - R * a - 0.5 * th * n, wall.point + R * a - 0.5 * th * n,
                                   0.5 * th, (0.62, 0.72, 0.9, 1.0))
                segs = []
                for c in np.linspace(-1, 1, 15):
                    q = wall.point + c * R * a - th * n
                    segs += [q, q - 2.5 * th * (n + a)]
                self.view.set_polyline("tsc.env.hatch", np.array(segs), (0.75, 0.85, 1.0, 0.7), 1.2, mode="lines")
            else:
                self.view.set_disk("tsc.env.wall", wall.point - 0.5 * th * wall.normal, wall.normal, R, th,
                                   (0.62, 0.72, 0.9, 1.0))
            # hatching on the surface so it reads well from any camera angle
            n = wall.normal
            a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(n, a)
            segs = []
            for c in (np.linspace(-0.9, 0.9, 9) if not robot.planar else []):
                h = np.sqrt(max(1 - c * c, 0)) * R
                segs += [wall.point + c * R * a - h * b, wall.point + c * R * a + h * b]
                if not robot.planar:
                    segs += [wall.point + c * R * b - h * a, wall.point + c * R * b + h * a]
            if segs:
                self.view.set_polyline("tsc.env.hatch", np.array(segs) + 1e-3 * n, (0.75, 0.85, 1.0, 0.55), 1.2,
                                       mode="lines")
            self.view.set_label("tsc.env.lbl", wall.point + R * 0.7 * np.cross(wall.normal, [0, 0, 1.0] if robot.planar
                                                                              else [1.0, 0, 0]) - 2 * th * wall.normal,
                                f"surface k = {wall.k:.0f} N/m", theme.MUTED)
        if r.mode == "impedance":
            X0 = r.info["X0"]
            self.view.set_triad("tsc.env.rest", X0, 0.12 * robot.reach, "rest", alpha=0.6)

    def _redraw(self):
        if self.anim is not None:
            self.show_frame(self.player.k)

    def show_frame(self, j):
        if self.anim is None:
            return
        r, idx = self.anim["r"], self.anim["idx"]
        k = int(idx[min(j, len(idx) - 1)])
        th = r.theta[k]
        if not np.all(np.isfinite(th)):
            return
        robot = self.ctx.robot
        self.graphic.update(th)
        p = r.p[k]
        sz = 0.12 * robot.reach
        if r.mode in ("velocity", "motion", "hybrid") and r.Xd is not None:
            self.view.set_triad("tsc.target", r.Xd[k], sz, "X_d", alpha=0.75)
            self.view.set_sphere("tsc.target.pt", r.Xd[k][:3, 3], 0.5 * robot.link_radius, theme.GOOD)
        if self.cb_trace.isChecked() and k > 0:
            self.view.set_polyline("tsc.trace", r.p[:k + 1], (0.3, 0.85, 1.0, 0.95), 2.5)
        else:
            self.view.remove("tsc.trace")
        scale = 0.25 * robot.reach / 30.0          # 30 N -> a quarter of the reach
        wall = r.info.get("wall")
        if wall is not None and r.force is not None:
            n = wall.normal
            F = r.force[k]
            if F > 1e-3:
                self.view.set_arrow("tsc.env.F", p, -n * F * scale, theme.WARN, 0.3 * robot.link_radius)
            else:
                self.view.set_arrow("tsc.env.F", p, np.zeros(3), theme.WARN)
            if r.force_d is not None:
                self.view.set_label("tsc.env.Flbl", p - n * (F * scale + 0.03), f"F = {F:.1f} N", theme.WARN)
        if r.mode == "impedance":
            fu = r.f_user[k]
            if np.linalg.norm(fu) > 1e-3:
                self.view.set_arrow("tsc.env.push", p - fu * scale, fu * scale, theme.BAD, 0.25 * robot.link_radius)
                self.view.set_label("tsc.env.plbl", p - fu * scale, f"push {np.linalg.norm(fu):.1f} N", theme.BAD)
            else:
                self.view.set_arrow("tsc.env.push", p, np.zeros(3), theme.BAD)
                self.view.remove("tsc.env.plbl")
        t = r.t[k]
        for c in self.cursors.values():
            c.setPos(t)
        lab = "solid = actual robot, green triad = desired X_d" if r.mode in ("velocity", "motion", "hybrid") \
            else ("yellow arrow = force of the robot on the surface" if r.mode == "force" else "red arrow = user push")
        self.player.lbl.setText(f"t = {t:6.3f} s / {r.t[-1]:.3f} s — {lab}")

    def on_hide(self):
        self.player.stop()
