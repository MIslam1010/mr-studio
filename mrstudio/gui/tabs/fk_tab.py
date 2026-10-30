"""Tab 1 -- robot description (screw axes) and forward kinematics (MR Ch. 3-4)."""
from __future__ import annotations

import time

import modern_robotics as mr
import numpy as np
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import kinematics as kin
from .. import theme
from ..base import BaseTab
from ..viewer import RobotGraphic, SceneView
from ..widgets import (JointSliders, MatrixView, button, checkbox, group, hbox, value_label)

THEORY = r"""
<h2>Forward kinematics — Product of Exponentials</h2>
<span class="ref">Modern Robotics, Ch. 3 (rigid motions, twists, screws) and Ch. 4 (PoE)</span>
<p>Each joint is a <b>screw axis</b> written once, in the fixed frame {s}, at the home configuration
(all $\theta_i=0$): $\mathcal{S}_i=(\omega_i,\,v_i)$ with $v_i=-\omega_i\times q_i$ for a revolute joint
through point $q_i$.  No link frames are needed.</p>
$$T(\theta)=e^{[\mathcal{S}_1]\theta_1}\,e^{[\mathcal{S}_2]\theta_2}\cdots e^{[\mathcal{S}_n]\theta_n}\,M\quad(4.14)$$
<p>Equivalently in the end-effector (body) frame, with $\mathcal{B}_i=[\mathrm{Ad}_{M^{-1}}]\,\mathcal{S}_i$:</p>
$$T(\theta)=M\,e^{[\mathcal{B}_1]\theta_1}\cdots e^{[\mathcal{B}_n]\theta_n}\quad(4.16)$$
<h3>Matrix exponential (Rodrigues)</h3>
$$e^{[\omega]\theta}=I+\sin\theta\,[\omega]+(1-\cos\theta)\,[\omega]^2\quad(3.51)$$
$$e^{[\mathcal{S}]\theta}=\left[\,e^{[\omega]\theta}\ \ \ G(\theta)\,v\,\right]$$
$$G(\theta)=I\theta+(1-\cos\theta)[\omega]+(\theta-\sin\theta)[\omega]^2\quad(3.87)$$
<h3>Chasles–Mozzi (magenta line)</h3>
<p>The displacement of {b} from home to the current pose is itself <i>one</i> screw motion:</p>
$$T\,M^{-1}=e^{[\mathcal{S}]\theta},\qquad \mathrm{pitch}\ \ h=\omega^{T}v$$
<p>The app computes $[\mathcal{S}]\theta=\log(TM^{-1})$ and draws that screw axis.</p>
<h3>Why PoE and not Denavit–Hartenberg?</h3>
<p>PoE needs only {s} and {b}; there are no per-link frame-assignment rules, prismatic and revolute
joints are treated uniformly and the formula has no special cases for parallel axes.  The robots
from Peter Corke's toolbox (PUMA 560, Panda) are stored as DH tables and are converted
automatically to screw axes — the <b>Validation</b> tab proves both descriptions give identical poses.
The drawn links follow the common normals of consecutive screw axes, i.e. the DH skeleton is
<i>recovered</i> from the screw axes.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 2 §3.2–3.3, pp. 39–50', 'twists, exponential coordinates, screws; Chasles (Thm. 2.11, p. 49)'),
    ('mls', 'Ch. 3 §2.2–2.3, pp. 85–95', 'the product-of-exponentials formula'),
    ('gotor', '§2.3, pp. 24–31; §3.3, pp. 36–48', 'screw exponentials; PoE worked for ABB, SCARA, UR16e, KUKA iiwa'),
    ('corke', '§2.1–2.2, pp. 19–39; §7.1–7.2, pp. 137–146', 'rotations, homogeneous transforms; DH forward kinematics'),
    ('lavalle', '§3.2–3.3, pp. 92–111', 'rigid-body transforms and kinematic chains (DH)'),
    ('featherstone', '§2.8, pp. 20–23', 'Plücker coordinate transforms (= the adjoint map)'),
])



class FKTab(BaseTab):
    title = "① FK"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        self.sliders = JointSliders()
        self.sliders.changed.connect(self.update_scene)
        self.anim_btn = button("▶ Animate", self.toggle_anim, checkable=True)
        btns = hbox(button("Ready pose", lambda: self.set_theta(self.ctx.robot.home),
                           tooltip="a convenient non-singular start pose; the MR home configuration is θ = 0, T = M"),
                    button("Zero (MR home, T = M)", lambda: self.set_theta(np.zeros(self.ctx.robot.n)),
                           tooltip="θ = 0: the home configuration of MR Ch. 4, where T(0) = M"),
                    button("Random", lambda: self.set_theta(self.ctx.robot.random_config())),
                    self.anim_btn)
        self.cb_axes = checkbox("Screw axes 𝒮ᵢ (yellow)", self.update_scene, True)
        self.cb_labels = checkbox("Joint numbers", self.update_scene, True)
        self.cb_chasles = checkbox("Equivalent screw of T·M⁻¹ (Chasles)", self.update_scene, False)
        self.cb_com = checkbox("Link centre-of-mass frames", self.update_scene, False)
        self.cb_trace = checkbox("Trace end-effector path", self._trace_toggled, False)

        self.T_view = MatrixView(4, 4, 4)
        self.pose_lbl = value_label()
        self.check_lbl = value_label()
        self._tone(self.check_lbl, "good")
        self.screw_lbl = value_label()
        theme.notifier().projector_changed.connect(self._retone)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Joint coordinates θ", self.sliders, btns))
        cl.addWidget(group("Display", self.cb_axes, self.cb_labels, self.cb_chasles, self.cb_com, self.cb_trace))
        cl.addWidget(group("End-effector pose  T_sb(θ)  (space PoE)", self.T_view, self.pose_lbl, self.check_lbl))
        cl.addWidget(group("Equivalent screw motion from home", self.screw_lbl))
        cl.addStretch(1)

        # bottom: robot description tables
        self.desc = QtWidgets.QLabel()
        self.desc.setWordWrap(True)
        self.desc.setObjectName("muted")
        self.S_table = MatrixView(1, 7, 4, col_labels=["i", "ωx", "ωy", "ωz", "vx", "vy", "vz"])
        self.B_table = MatrixView(1, 7, 4, col_labels=["i", "ωx", "ωy", "ωz", "vx", "vy", "vz"])
        self.M_table = MatrixView(4, 4, 4)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.S_table, "Space screw axes 𝒮ᵢ (in {s})")
        tabs.addTab(self.B_table, "Body screw axes ℬᵢ (in {b})")
        tabs.addTab(self.M_table, "Home configuration M")
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.desc)
        bl.addWidget(tabs)

        self.assemble(controls, self.view, bottom, THEORY, bottom_ratio=(5, 2))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._anim_step)
        self.trace: list = []
        self.graphic = None
        self.on_robot_changed(self.ctx.robot)

    # ------------------------------------------------------------------ robot

    def _tone(self, lbl, tone):
        """Colour a label with the current palette's good/warn/bad colour (None = default)."""
        lbl.setProperty("tone", tone)
        lbl.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")

    def _retone(self, *_):
        for lbl in (self.check_lbl,):
            self._tone(lbl, lbl.property("tone"))

    def on_robot_changed(self, robot):
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("fk.")
        self.graphic = RobotGraphic(self.view, robot, "robot")
        self.sliders.set_robot(robot)
        self.view.frame_scene(robot.reach, planar=robot.planar, center=(0, 0, 0.25 * robot.reach * (not robot.planar)))
        self.view.set_triad("fk.s", np.eye(4), 0.22 * robot.reach, "{s}")
        self.desc.setText(f"<b>{robot.name}</b> — {robot.description}<br><i>Source: {robot.source}</i>")
        self.S_table.set_matrix([r for r in robot.screw_table("space")])
        self.B_table.set_matrix([r for r in robot.screw_table("body")])
        self.M_table.set_matrix(robot.M)
        self.trace = []
        self.update_scene()

    def set_theta(self, th):
        self.sliders.set_values(th)
        self.update_scene()

    # ------------------------------------------------------------------ drawing
    def update_scene(self, *_):
        robot = self.ctx.robot
        th = self.sliders.values()
        if len(th) != robot.n:
            return
        self.graphic.show_axes = self.cb_axes.isChecked()
        self.graphic.show_labels = self.cb_labels.isChecked()
        g = self.graphic.update(th)
        T = kin.fk_space(robot, th)
        Tb = kin.fk_body(robot, th)
        self.T_view.set_matrix(T)
        rpy = np.degrees(kin.rot_to_rpy(T[:3, :3]))
        p = T[:3, 3]
        self.pose_lbl.setText(f"p = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) m\n"
                              f"roll/pitch/yaw = ({rpy[0]:+.1f}°, {rpy[1]:+.1f}°, {rpy[2]:+.1f}°)")
        self.check_lbl.setText(f"✓ body-form PoE  M·Πe^[ℬ]θ  agrees: ‖ΔT‖ = {np.abs(T - Tb).max():.1e}")

        # Chasles: T M^-1 = exp([S] theta)
        V = mr.se3ToVec(mr.MatrixLog6(T @ mr.TransInv(robot.M)))
        w, v = V[:3], V[3:]
        ang = np.linalg.norm(w)
        if ang > 1e-9:
            wh, vh = w / ang, v / ang
            q = np.cross(wh, vh)
            h = float(wh @ vh)
            self.screw_lbl.setText(f"ω̂ = ({wh[0]:+.3f}, {wh[1]:+.3f}, {wh[2]:+.3f}),  θ = {np.degrees(ang):.1f}°\n"
                                   f"point q = ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}) m,  pitch h = {h:+.4f} m/rad")
            if self.cb_chasles.isChecked():
                ext = robot.reach
                self.view.set_polyline("fk.chasles", [q - ext * wh, q + ext * wh], (1, 0.3, 1, 0.9), 3)
                self.view.set_sphere("fk.chasles.q", q, robot.link_radius * 0.8, (1, 0.3, 1, 1))
            else:
                self.view.remove_prefix("fk.chasles")
        else:
            d = np.linalg.norm(v)
            self.screw_lbl.setText(f"pure translation, |d| = {d:.4f} m" if d > 1e-9 else "home configuration (identity)")
            self.view.remove_prefix("fk.chasles")

        if self.cb_com.isChecked() and robot.has_dynamics:
            Ts = robot.link_transforms(th)
            for i in range(robot.n):
                self.view.set_triad(f"fk.com{i}", Ts[i + 1] @ robot.com_home[i], 0.07 * robot.reach, alpha=0.9)
        else:
            self.view.remove_prefix("fk.com")

        if self.cb_trace.isChecked():
            self.trace.append(p.copy())
            self.trace = self.trace[-800:]
            if len(self.trace) > 1:
                self.view.set_polyline("fk.trace", self.trace, (0.3, 0.8, 1.0, 0.9), 2)

    def _trace_toggled(self, on):
        self.trace = []
        if not on:
            self.view.remove("fk.trace")

    # ------------------------------------------------------------------ animation
    def toggle_anim(self, on):
        self.anim_btn.setText("■ Stop" if on else "▶ Animate")
        if on:
            self.t0 = time.time()
            self.th0 = self.sliders.values()
            self.timer.start(30)
        else:
            self.timer.stop()

    def _anim_step(self):
        robot = self.ctx.robot
        t = time.time() - self.t0
        k = np.arange(robot.n)
        th = self.th0 + 0.6 * np.sin(0.7 * t * (1 + 0.23 * k) + k)
        self.sliders.set_values(th)
        self.update_scene()

    def on_hide(self):
        if self.anim_btn.isChecked():
            self.anim_btn.setChecked(False)
