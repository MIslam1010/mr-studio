"""Tab 7 -- Stewart-Gough platform: IK, Newton-Raphson FK, statics, singularities, dynamics (MR 7.1.2, 7.2.1, 7.3, 8.2.2)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import parallel as par
from .. import theme
from ..base import BaseTab
from ..viewer import SceneView
from ..widgets import (MatrixView, SliderRow, button, checkbox, combo, group, hbox, pen, plot, spin,
                       value_label)

THEORY = r"""
<h2>Stewart–Gough platform</h2>
<span class="ref">Modern Robotics 7.1.2 (inverse &amp; forward kinematics), 7.2.1 (differential kinematics
and statics), 7.3 (singularities), 8.2.2–8.2.3 (rigid-body dynamics)</span>
<p>Six actuated prismatic legs join base points $a_i$ (given in {s}) to platform points $b_i$ (given in
the platform frame {b}) through universal and spherical joints (6-UPS). Grübler with $N=14$ links and
$J=18$ joints:</p>
$$dof=6(N-1-J)+\sum f_i=6(14-1-18)+36=6$$
<h3>Inverse kinematics (closed form)</h3>
<p>For the platform pose $T_{sb}=(R,p)$ the legs are</p>
$$l_i=p+Rb_i-a_i,\qquad s_i=|l_i|,\qquad \hat{u}_i=l_i/s_i$$
<h3>Velocity kinematics and statics (7.2.1)</h3>
<p>The platform point $x_i=p+Rb_i$ moves with $\dot{x}_i=v_s+\omega_s\times x_i$ for the spatial twist
$\mathcal{V}_s=(\omega_s,v_s)$, so</p>
$$\dot{s}_i=\hat{u}_i^{T}\dot{x}_i=(x_i\times\hat{u}_i)^{T}\omega_s+\hat{u}_i^{T}v_s\quad\Rightarrow\quad\dot{s}=J\,\mathcal{V}_s\quad(cf.\ 7.5)$$
<p>Row $i$ of $J$ is $\left[(x_i\times\hat{u}_i)^{T}\ \ \hat{u}_i^{T}\right]$ — the inverse of MR's forward
Jacobian $J_s$. Power balance $f^{T}\dot{s}=\mathcal{F}_s^{T}\mathcal{V}_s$ gives the statics:</p>
$$\mathcal{F}_s=J^{T}f=\sum_{i} f_i\left[x_i\times\hat{u}_i;\ \hat{u}_i\right]$$
<p>Leg forces holding the payload wrench $\mathcal{F}_{g}$ (gravity of payload + platform, $f_i>0$ =
compression, blue; $f_i<0$ = tension, red):</p>
$$f=-J^{-T}\mathcal{F}_{g}$$
<h3>Forward kinematics (Newton–Raphson)</h3>
<p>No closed form in general: iterate on the six leg lengths, updating the pose on SE(3) (MR 6.2):</p>
$$\mathcal{V}=J^{-1}(T_k)\left(s_d-s(T_k)\right),\qquad T_{k+1}=e^{[\mathcal{V}]}\,T_k$$
<p>The general 6-6 platform has up to <b>40</b> (complex) forward-kinematics solutions — the
assembly modes (Raghavan 1993, Husty 1996; Dietmaier 1998 found a design with 40 real ones). Newton–Raphson
converges to the one near its initial guess; random restarts reveal others (ghost platforms), e.g. the
mirror image through the base plane.</p>
<h3>Singularities (7.3)</h3>
<p>The IK map is defined everywhere (no serial singularities inside the stroke limits), but where
$\det J=0$ the platform has an <b>actuator (Type II) singularity</b>: it can move with all legs locked
and finite payloads require infinite leg forces. The app shows $\det J_n$ and the condition number
$\kappa(J_n)$, where $J_n$ divides the angular columns by the platform radius to make $J$ dimensionless.</p>
<h3>Dynamics of the platform (8.2.2)</h3>
<p>Platform + payload are one rigid body with spatial inertia $\mathcal{G}_b$ in the platform frame {b}
(not at the CoM, so each part is moved there with Steiner's theorem in adjoint form); the legs are
assumed <b>massless</b>.</p>
$$\mathcal{G}_b=\sum_{k}\left[\mathrm{Ad}_{T_{c_k b}}\right]^{T}\mathcal{G}_{c_k}\left[\mathrm{Ad}_{T_{c_k b}}\right]\quad(8.42)$$
<p>The leg forces act through the wrench $\mathcal{F}_s=J^{T}f$, i.e. $\mathcal{F}_b=[\mathrm{Ad}_{T_{sb}}]^{T}J^{T}f$.
With gravity $\mathcal{F}_{g,b}$ the twist–wrench equation of motion is</p>
$$\mathcal{F}_b=\mathcal{G}_b\dot{\mathcal{V}}_b-[\mathrm{ad}_{\mathcal{V}_b}]^{T}\mathcal{G}_b\mathcal{V}_b\quad(8.40)$$
<p><b>Inverse dynamics</b> along the dance (exact $\mathcal{V}_b,\dot{\mathcal{V}}_b$ of the quintic
Cartesian segments):</p>
$$[\mathrm{Ad}_{T_{sb}}]^{T}J^{T}f=\mathcal{G}_b\dot{\mathcal{V}}_b-[\mathrm{ad}_{\mathcal{V}_b}]^{T}\mathcal{G}_b\mathcal{V}_b-\mathcal{F}_{g,b}$$
<p>At rest it reduces to the statics $f=-J^{-T}\mathcal{F}_g$ (dashed). Power balance (plotted):</p>
$$f^{T}\dot{s}=\frac{d}{dt}\left(\frac{1}{2}\mathcal{V}_b^{T}\mathcal{G}_b\mathcal{V}_b+mgz_c\right)$$
<p><b>Forward dynamics</b>, integrated with RK4 on SE(3) (Munthe-Kaas, $T_{k+1}=T_k\,e^{[\xi]}$, so the pose
never leaves SE(3)):</p>
$$\dot{\mathcal{V}}_b=\mathcal{G}_b^{-1}\left(\mathcal{F}_b+[\mathrm{ad}_{\mathcal{V}_b}]^{T}\mathcal{G}_b\mathcal{V}_b\right)$$
<p>“Simulate push” starts at rest at home with $f=f_{static}+\Delta f$ on one leg for a short pulse. Open
loop (forces only), small errors grow exponentially — a force-driven platform has no restoring stiffness;
a leg-space PD servo $K_p(s_0-s)-K_d\dot{s}$ makes it return.</p>
<h3>Dance</h3>
<p>Waypoints joined by mr.CartesianTrajectory (decoupled rotation and translation with quintic time
scaling, MR 9.2); the legs follow from the closed-form IK.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 3 §5.4, pp. 138–143', 'Stewart platform: FK/IK, singularities; Exercise 22, p. 153'),
    ('lavalle', '§7.4, p. 338', 'Stewart–Gough platform as a closed chain'),
    ('featherstone', '§2.13–2.14, pp. 32–36', 'spatial inertia and the rigid-body equation of motion'),
])


POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]


def _force_color(c: float):
    """c in [-1, 1]: +1 compression -> blue, -1 tension -> red, 0 -> light grey."""
    c = float(np.clip(c, -1, 1))
    grey = np.array([0.82, 0.84, 0.86])
    tgt = np.array([0.25, 0.55, 1.0]) if c >= 0 else np.array([1.0, 0.28, 0.28])
    k = 0.0 if abs(c) < 1e-6 else 0.5 + 0.5 * abs(c)   # the sign always reads clearly
    return (*(grey + k * (tgt - grey)), 1.0)


class StewartTab(BaseTab):
    title = "⑦ Stewart–Gough"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.sp = par.StewartPlatform()
        self.T = self.sp.T_home.copy()
        self.s = self.sp.ik(self.T)
        self.view = SceneView()
        self.sols: list = []
        self._dance_k = 0

        self.mode_box = combo(["Inverse kinematics: pose → leg lengths", "Forward kinematics: leg lengths → pose"])
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        # pose sliders
        z0 = self.sp.z_home
        lim = [(-0.15, 0.15), (-0.15, 0.15), (z0 - 0.17, z0 + 0.2)]
        self.pose_rows = []
        for i, nm in enumerate(POSE_NAMES):
            if i < 3:
                r = SliderRow(nm, *lim[i], value=(z0 if i == 2 else 0.0), decimals=3, unit=" m")
            else:
                r = SliderRow(nm, -np.radians(30), np.radians(30), 0.0, decimals=1, unit="°", scale=180 / np.pi)
            r.valueChanged.connect(lambda _v: self._pose_changed())
            self.pose_rows.append(r)
        self.leg_rows = []
        for i in range(6):
            r = SliderRow(f"s{_sub(i + 1)}", self.sp.s_min, self.sp.s_max, self.s[i], decimals=3, unit=" m")
            r.valueChanged.connect(lambda _v: self._legs_changed())
            self.leg_rows.append(r)
        pose_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pose_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(2)
        for r in self.pose_rows:
            pl.addWidget(r)
        leg_page = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(leg_page)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        for r in self.leg_rows:
            ll.addWidget(r)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(pose_page)
        self.stack.addWidget(leg_page)
        self.home_btn = button("Home", self._home)
        self.dance_btn = button("▶ Dance", self._dance_toggle, checkable=True, primary=True)

        self.mass = spin(0, 300, 25, 1, 1, " kg")
        self.offx = spin(-0.3, 0.3, 0.0, 3, 0.01, " m")
        self.offy = spin(-0.3, 0.3, 0.0, 3, 0.01, " m")
        for w in (self.mass, self.offx, self.offy):
            w.valueChanged.connect(self._load_changed)
        fl = QtWidgets.QFormLayout()
        fl.addRow("payload mass", self.mass)
        fl.addRow("CoM offset x (in {b})", self.offx)
        fl.addRow("CoM offset y (in {b})", self.offy)
        self.cb_all = checkbox("Show all FK solutions (ghost platforms)", self._toggle_all, False)
        self.cb_frames = checkbox("Frames {s} and {b}", self._redraw, True)
        self.all_lbl = value_label()
        self._sol_timer = QtCore.QTimer(self)
        self._sol_timer.setSingleShot(True)
        self._sol_timer.timeout.connect(self._solve_all)

        self.table = MatrixView(6, 3, 3, row_labels=[f"leg {i + 1}" for i in range(6)],
                                col_labels=["length s [m]", "force f [N]", "state"])
        self.table.setFixedHeight(6 * 27 + 32)
        self.pose_lbl = value_label()
        self.fk_lbl = value_label()
        self.sing_lbl = QtWidgets.QLabel()
        self.sing_lbl.setWordWrap(True)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Input", self.mode_box, self.stack, hbox(self.home_btn, self.dance_btn)))
        cl.addWidget(group("Payload (statics  𝓕ₛ = Jᵀ f)", fl))
        cl.addWidget(group("Legs", self.table))
        cl.addWidget(group("Pose, FK && singularity", self.pose_lbl, self.fk_lbl, self.sing_lbl))
        cl.addWidget(group("Assembly modes", self.cb_all, self.all_lbl, self.cb_frames))
        # dynamics
        self.seg_time = spin(0.4, 3.0, 1.2, 2, 0.1, " s")
        self.seg_time.valueChanged.connect(self._load_changed)
        self.push_leg = combo([f"leg {i + 1}" for i in range(6)])
        self.push_df = spin(-500, 500, 150, 0, 10, " N")
        self.push_dur = spin(0.01, 0.5, 0.1, 2, 0.01, " s")
        self.cb_servo = checkbox("Leg PD servo (Kₚ = 3·10⁴ N/m), else open loop", None, True)
        self.push_btn = button("▶ Simulate push", self._push_toggle, checkable=True, primary=True,
                               tooltip="forward dynamics (RK4 on SE(3)) from rest at home")
        fd = QtWidgets.QFormLayout()
        fd.addRow("dance segment time", self.seg_time)
        fd.addRow("pulse on", self.push_leg)
        fd.addRow("pulse Δf", self.push_df)
        fd.addRow("pulse duration", self.push_dur)
        self.dyn_lbl = value_label()
        self.push_lbl = value_label()
        cl.addWidget(group("Dynamics (MR 8.2.2, Eq. 8.40; massless legs)", fd, self.dyn_lbl, self.cb_servo,
                           self.push_btn, self.push_lbl))
        cl.addStretch(1)

        # bottom plots: the dance
        self.pS = plot("Leg lengths during the dance", "s [m]", "t [s]")
        self.pS.getAxis("left").enableAutoSIPrefix(False)
        self.pF = plot("Leg forces (payload statics)", "f [N]", "t [s]")
        self.pD = plot("Singularity measure", "", "t [s]")
        self.pFd = plot("Dynamic vs static (- -) leg forces", "f [N]", "t [s]")
        self.pPow = plot("Power: fᵀṡ vs d(K+P)/dt", "W", "t [s]")
        self.pPush = plot("Push: leg-length change", "Δs [mm]", "t [s]")
        self.pPush.getAxis("bottom").enableAutoSIPrefix(False)
        bottom = QtWidgets.QTabWidget()
        for name, plots in (("Dance kinematics && statics", (self.pS, self.pF, self.pD)),
                            ("Dynamics (MR 8.2.2)", (self.pFd, self.pPow, self.pPush))):
            w = QtWidgets.QWidget()
            bl = QtWidgets.QHBoxLayout(w)
            bl.setContentsMargins(0, 0, 0, 0)
            for p in plots:
                bl.addWidget(p)
            bottom.addTab(w, name)
        self.bottom_tabs = bottom

        self.assemble(controls, self.view, bottom, THEORY, controls_width=372, bottom_ratio=(5, 2))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._dance_step)
        self.push_timer = QtCore.QTimer(self)
        self.push_timer.timeout.connect(self._push_step)
        self.push_sim = None
        self.view.frame_scene(0.72, planar=False, center=(0, 0, 0.3))
        self._mode_changed(0)
        self._prepare_dance()
        self._update()

    # ================================================================ inputs
    def _mode_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        for i in range(self.stack.count()):
            pol = QtWidgets.QSizePolicy.Preferred if i == idx else QtWidgets.QSizePolicy.Ignored
            self.stack.widget(i).setSizePolicy(pol, pol)
        self.stack.adjustSize()
        self._sync_sliders()

    def _sync_sliders(self):
        v = par.pose_vec(self.T)
        for r, x in zip(self.pose_rows, v):
            r.set_value(x)
        for r, x in zip(self.leg_rows, self.s):
            r.set_value(x)

    def _pose_changed(self):
        v = [r.value() for r in self.pose_rows]
        self.T = par.pose(*v)
        self.s = self.sp.ik(self.T)
        for r, x in zip(self.leg_rows, self.s):
            r.set_value(x)
        self.fk_lbl.setText("IK (closed form): s_i = |p + R b_i − a_i|")
        self._update()

    def _legs_changed(self):
        s = np.array([r.value() for r in self.leg_rows])
        t0 = time.perf_counter()
        res = self.sp.fk(s, self.T)
        ms = (time.perf_counter() - t0) * 1e3
        if res.converged and res.T[2, 3] > 0:
            self.T, self.s = res.T, s
            self.fk_lbl.setText(f"✓ Newton–Raphson FK: {res.iterations} iterations, ‖s − s(T)‖ = "
                                f"{res.history[-1]:.1e} m ({ms:.1f} ms), started from the previous pose")
            self.fk_lbl.setStyleSheet(f"color:{theme.GOOD}")
            for r, x in zip(self.pose_rows, par.pose_vec(self.T)):
                r.set_value(x)
        else:
            self.fk_lbl.setText("⚠ Newton–Raphson found no assembly near the previous pose for these leg "
                                "lengths (incompatible lengths or a singularity) — pose kept.")
            self.fk_lbl.setStyleSheet(f"color:{theme.BAD}")
        self._update()

    def _home(self):
        self._stop_dance()
        self._stop_push()
        self.T = self.sp.T_home.copy()
        self.s = self.sp.ik(self.T)
        self._sync_sliders()
        self._update()

    def _load_changed(self, *_):
        self._prepare_dance()
        self._update()

    # ================================================================ computation + drawing
    def _payload(self, T):
        return self.sp.payload_wrench(T, self.mass.value(), (self.offx.value(), self.offy.value(), 0.06))

    def _update(self):
        sp, T = self.sp, self.T
        f = sp.leg_forces(T, self._payload(T))
        det, kappa = sp.singularity(T)
        det0 = sp.singularity(sp.T_home)[0]
        rel = det / det0
        rows = []
        for i in range(6):
            lim = self.s[i] < sp.s_min - 1e-9 or self.s[i] > sp.s_max + 1e-9
            state = "LIMIT" if lim else ("compr." if f[i] >= 0 else "tension")
            rows.append([f"{self.s[i]:.4f}", f"{f[i]:+.1f}", state])
        self.table.set_matrix(np.array(rows, dtype=object),
                              highlight=lambda i, j, v: (j == 2 and v in ("tension", "LIMIT")))
        pv = par.pose_vec(T)
        self.pose_lbl.setText(f"p = ({pv[0]:+.4f}, {pv[1]:+.4f}, {pv[2]:+.4f}) m\n"
                              f"roll/pitch/yaw = ({np.degrees(pv[3]):+.1f}°, {np.degrees(pv[4]):+.1f}°, "
                              f"{np.degrees(pv[5]):+.1f}°)\n"
                              f"det(Jn) = {det:+.3f} ({rel:.2f} × home)\ncond(Jn) = {kappa:.1f}")
        if abs(rel) < 0.1 or kappa > 60:
            txt, col = "⚠ Close to an actuator (Type II) singularity: det J → 0, leg forces blow up", theme.BAD
        elif abs(rel) < 0.35 or kappa > 20:
            txt, col = "Poorly conditioned pose (κ large)", theme.WARN
        else:
            txt, col = "Well-conditioned pose", theme.GOOD
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")
        self.forces = f
        self._redraw()
        if self.cb_all.isChecked():
            self._sol_timer.start(350)

    def _redraw(self, *_):
        sp, T, v = self.sp, self.T, self.view
        x, s, u = sp.legs(T)
        R, p = T[:3, :3], T[:3, 3]
        # base plate + platform plate
        v.set_disk("sg.base", [0, 0, -0.015], [0, 0, 1], sp.Rb * 1.15, 0.022, (0.36, 0.41, 0.48, 1.0))
        v.set_polyline("sg.base.edge", np.vstack([sp.a, sp.a[:1]]) + [0, 0, 0.0], (0.75, 0.8, 0.88, 0.8), 2)
        v.set_disk("sg.plat", p + R @ [0, 0, 0.016], R[:, 2], sp.Rp * 1.13, 0.022, (0.30, 0.62, 0.78, 0.9))
        hexa = np.vstack([x, x[:1]])
        v.set_polyline("sg.plat.edge", hexa + R[:, 2] * 0.03, (0.75, 0.9, 1.0, 0.9), 2)
        fmax = max(float(np.abs(self.forces).max()), 1e-9)
        for i in range(6):
            a = sp.a[i]
            col = _force_color(self.forces[i] / fmax)
            lim = s[i] < sp.s_min - 1e-9 or s[i] > sp.s_max + 1e-9
            v.set_sphere(f"sg.aj{i}", a, 0.022, theme.JOINT)
            v.set_sphere(f"sg.bj{i}", x[i], 0.018, theme.JOINT)
            v.set_tube(f"sg.cyl{i}", a + 0.02 * u[i], a + 0.36 * u[i], 0.026, col)
            v.set_tube(f"sg.rod{i}", a + 0.30 * u[i], x[i], 0.012, theme.BAD if lim else (0.78, 0.8, 0.84, 1.0))
            v.set_label(f"sg.lbl{i}", a + np.array([0, 0, -0.06]) + 0.06 * a / np.linalg.norm(a), f"{i + 1}",
                        theme.MUTED)
        # payload
        m = self.mass.value()
        c = R @ np.array([self.offx.value(), self.offy.value(), 0.06]) + p
        if m > 0:
            v.set_sphere("sg.load", c, 0.03 + 0.012 * m ** (1 / 3), (0.95, 0.75, 0.25, 1.0))
            v.set_arrow("sg.g", c, np.array([0, 0, -0.12 - 0.0015 * m]), (0.95, 0.75, 0.25, 1.0), 0.008)
        else:
            v.remove_prefix("sg.load")
            v.remove_prefix("sg.g")
        if self.cb_frames.isChecked():
            v.set_triad("sg.fs", np.eye(4), 0.18, "{s}")
            v.set_triad("sg.fb", T, 0.15, "{b}")
        else:
            v.remove_prefix("sg.fs")
            v.remove_prefix("sg.fb")
        self._draw_ghosts()

    # ================================================================ multiple FK solutions
    def _toggle_all(self, on):
        if on:
            self._solve_all()
        else:
            self.sols = []
            self.all_lbl.setText("")
            self.view.remove_prefix("sg.ghost")

    def _solve_all(self):
        if not self.cb_all.isChecked():
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            t0 = time.perf_counter()
            sols = self.sp.fk_all(self.s, n_starts=150, seed=0, include=self.T)
            dt = time.perf_counter() - t0
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.sols = [S for S in sols if np.abs(S - self.T).max() > 1e-6]
        n_up = sum(1 for S in sols if S[2, 3] > 0)
        self.all_lbl.setText(f"{len(sols)} distinct real solutions of s(T) = s from 150 random starts "
                             f"({dt:.2f} s): {n_up} above the base, {len(sols) - n_up} mirrored below.\n"
                             "The general platform has up to 40 (complex) solutions.")
        self._draw_ghosts()

    def _draw_ghosts(self):
        self.view.remove_prefix("sg.ghost")
        if not self.cb_all.isChecked():
            return
        for k, S in enumerate(self.sols):
            x = self.sp.platform_points(S)
            col = (*pg.intColor(k, hues=max(len(self.sols), 3)).getRgbF()[:3], 0.55)
            self.view.set_polyline(f"sg.ghost{k}.plat", np.vstack([x, x[:1]]), col, 3)
            legs = np.empty((12, 3))
            legs[0::2], legs[1::2] = self.sp.a, x
            self.view.set_polyline(f"sg.ghost{k}.legs", legs, (*col[:3], 0.35), 1.5, mode="lines")

    # ================================================================ dance
    def _prepare_dance(self):
        t, Ts = self.sp.dance(seg_time=self.seg_time.value())
        self.d_t, self.d_T = t, Ts
        S = np.array([self.sp.ik(T) for T in Ts])
        F = np.array([self.sp.leg_forces(T, self._payload(T)) for T in Ts])
        det0 = self.sp.singularity(self.sp.T_home)[0]
        D = np.array([self.sp.singularity(T)[0] / det0 for T in Ts])
        for p in (self.pS, self.pF, self.pD):
            p.clear()
        for i in range(6):
            self.pS.plot(t, S[:, i], pen=pen(i, 2), name=f"s{_sub(i + 1)}")
            self.pF.plot(t, F[:, i], pen=pen(i, 2), name=f"f{_sub(i + 1)}")
        for y in (self.sp.s_min, self.sp.s_max):
            self.pS.addLine(y=y, pen=pg.mkPen(theme.BAD, style=QtCore.Qt.DashLine))
        self.pF.addLine(y=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.pD.plot(t, D, pen=pen(4, 2.5), name="det(Jn) / det(Jn) at home")
        self.pD.addLine(y=0, pen=pg.mkPen(theme.BAD, style=QtCore.Qt.DashLine))
        self._prepare_dynamics()
        self.cursors = [p.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1))
                        for p in (self.pS, self.pF, self.pD, self.pFd, self.pPow)]

    def _dance_toggle(self, on):
        self.dance_btn.setText("■ Stop" if on else "▶ Dance")
        if on:
            self._stop_push()
            if self.mode_box.currentIndex() != 0:
                self.mode_box.setCurrentIndex(0)
            self.t0 = time.time()
            self.timer.start(30)
        else:
            self.timer.stop()

    def _stop_dance(self):
        if self.dance_btn.isChecked():
            self.dance_btn.setChecked(False)

    def _dance_step(self):
        t = (time.time() - self.t0) % self.d_t[-1]
        k = int(np.searchsorted(self.d_t, t))
        k = min(k, len(self.d_T) - 1)
        self.T = self.d_T[k]
        self.s = self.sp.ik(self.T)
        for r, x in zip(self.pose_rows, par.pose_vec(self.T)):
            r.set_value(x)
        for r, x in zip(self.leg_rows, self.s):
            r.set_value(x)
        for c in self.cursors:
            c.setValue(self.d_t[k])
        self.fk_lbl.setText(f"Dance: t = {t:.2f} s (mr.CartesianTrajectory, quintic time scaling)")
        self._update()

    # ================================================================ hooks
    def on_show(self):
        self._redraw()

    def on_hide(self):
        self._stop_dance()
        self._stop_push()

    # ================================================================ dynamics
    def _body(self):
        return self.sp.rigid_body(self.mass.value(), (self.offx.value(), self.offy.value(), 0.06))

    def _prepare_dynamics(self):
        body = self._body()
        d = self.sp.dance_dynamics(body, seg_time=self.seg_time.value())
        for p in (self.pFd, self.pPow):
            p.clear()
        for i in range(6):
            self.pFd.plot(d["t"], d["f_dyn"][:, i], pen=pen(i, 2), name=f"f{_sub(i + 1)}")
            self.pFd.plot(d["t"], d["f_static"][:, i], pen=pen(i, 1.2, "dash"))
        self.pFd.addLine(y=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.pPow.plot(d["t"], d["power"], pen=pen(0, 3), name="fᵀṡ (legs)")
        self.pPow.plot(d["t"], d["dE"], pen=pg.mkPen(theme.WARN, width=1.5, style=QtCore.Qt.DashLine),
                       name="d(K+P)/dt")
        m, rc, G = body
        dmax = float(np.abs(d["f_dyn"] - d["f_static"]).max())
        self.dyn_lbl.setText(f"rigid body: m = {m:.1f} kg, r_c = ({rc[0]:+.3f}, {rc[1]:+.3f}, {rc[2]:+.3f}) m\n"
                             f"I_b diag = ({G[0, 0]:.3f}, {G[1, 1]:.3f}, {G[2, 2]:.3f}) kg·m²\n"
                             f"max |f_dyn − f_static| = {dmax:.1f} N\nmax |fᵀṡ − dE/dt| = "
                             f"{float(np.abs(d['power'] - d['dE']).max()):.1e} W")

    def _push_toggle(self, on):
        self.push_btn.setText("■ Stop" if on else "▶ Simulate push")
        if not on:
            self.push_timer.stop()
            return
        self._stop_dance()
        if self.mode_box.currentIndex() != 0:
            self.mode_box.setCurrentIndex(0)
        dur = self.push_dur.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            t0 = time.perf_counter()
            sim = self.sp.push_simulation(self._body(), leg=self.push_leg.currentIndex(), df=self.push_df.value(),
                                          t_pulse=(0.1, 0.1 + dur), t_final=2.5, h=0.004,
                                          servo=self.cb_servo.isChecked())
            ms = (time.perf_counter() - t0) * 1e3
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.push_sim = sim
        ds = (sim["s"] - sim["s"][0]) * 1e3
        self.pPush.clear()
        for i in range(6):
            self.pPush.plot(sim["t"], ds[:, i], pen=pen(i, 2), name=f"Δs{_sub(i + 1)}")
        self.pPush.addLine(x=0.1, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.pPush.addLine(x=0.1 + dur, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.push_cursor = self.pPush.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1))
        self.bottom_tabs.setCurrentIndex(1)
        stopped = sim["t"][-1] < 2.5 - 1e-9
        mode = "PD servo" if self.cb_servo.isChecked() else "open loop"
        txt = f"{mode}: {len(sim['t'])} RK4 steps (h = 4 ms) in {ms:.0f} ms; max |Δs| = {np.abs(ds).max():.1f} mm"
        if stopped:
            txt += f"\n⚠ stopped at t = {sim['t'][-1]:.2f} s: a leg left its stroke (unstable without feedback)"
        self.push_lbl.setText(txt)
        self.push_lbl.setStyleSheet(f"color:{theme.WARN}" if stopped else "")
        self._push_t0 = time.time()
        self.push_timer.start(30)

    def _stop_push(self):
        if self.push_btn.isChecked():
            self.push_btn.setChecked(False)

    def _push_step(self):
        sim = self.push_sim
        t = (time.time() - self._push_t0) * 0.5          # half speed
        if t > sim["t"][-1]:
            self._push_t0 = time.time()
            t = 0.0
        k = min(int(np.searchsorted(sim["t"], t)), len(sim["t"]) - 1)
        self.T = sim["T"][k]
        self.s = self.sp.ik(self.T)
        for r, x in zip(self.pose_rows, par.pose_vec(self.T)):
            r.set_value(x)
        for r, x in zip(self.leg_rows, self.s):
            r.set_value(x)
        self.push_cursor.setValue(sim["t"][k])
        self.fk_lbl.setText(f"Push simulation: t = {sim['t'][k]:.2f} s (shown at half speed)")
        self._update()


def _sub(i: int) -> str:
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
