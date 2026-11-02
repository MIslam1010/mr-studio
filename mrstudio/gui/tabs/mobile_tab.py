"""Wheeled mobile robots tab (Modern Robotics Ch. 13): omnidirectional (three omniwheels, four mecanum
wheels) and nonholonomic (unicycle, diff-drive, car) chassis, closed-loop trajectory tracking with the
book's control laws, odometry with an optional wheel-radius error, and a top-down animation.
The maths lives in mrstudio.mobile."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ... import mobile as mb
from ...references import further_reading
from .. import theme
from ..base import BaseTab
from ..widgets import (MatrixView, Player, SliderRow, checkbox, combo, compact, group, no_si, pen, plot,
                       time_cursor, value_label)

THEORY = r"""
<h2>Wheeled mobile robots</h2>
<span class="ref">Modern Robotics, Ch. 13 (pp. 515–553): Sec. 13.2 omnidirectional robots, Sec. 13.3 nonholonomic
robots, Sec. 13.4 odometry.  Equation numbers and printed pages as in the book.</span>
<p>The chassis frame {b} is at $q=(\phi,x,y)$ in {s}; its planar body twist $\mathcal{V}_{b}=(\omega_{bz},v_{bx},v_{by})$
is, by Eq. (13.1), p. 515, $\omega_{bz}=\dot{\phi}$ and</p>
$$v_{bx}=\dot{x}\cos\phi+\dot{y}\sin\phi,\qquad v_{by}=-\dot{x}\sin\phi+\dot{y}\cos\phi$$
<p>and $\dot{q}$ follows back from $\mathcal{V}_{b}$ by the inverse rotation, Eq. (13.2), p. 515.</p>

<h3>Omnidirectional robots (Sec. 13.2.1)</h3>
<p>An omniwheel ($\gamma=0$) or mecanum wheel ($\gamma=\pm45^{\circ}$) is driven only along its heading; the
rollers let it slide freely in the direction $\gamma$.  Driving speed of a wheel of radius $r$ whose centre moves
with $(v_{x},v_{y})$ in the wheel frame, Eq. (13.4), p. 518:</p>
$$u=\frac{1}{r}(v_{x}+v_{y}\tan\gamma)$$
<p>Wheel $i$ at $(x_{i},y_{i})$ with driving direction $\beta_{i}$ in {b}: $u_{i}=h_{i}(\phi)\dot{q}$ with,
by Eq. (13.6), p. 519,</p>
$$h_{i}(\phi)=\frac{1}{r_{i}\cos\gamma_{i}}\left[a_{i},\ \cos(\beta_{i}+\gamma_{i}+\phi),\ \sin(\beta_{i}+\gamma_{i}+\phi)\right]$$
$$a_{i}=x_{i}\sin(\beta_{i}+\gamma_{i})-y_{i}\cos(\beta_{i}+\gamma_{i})$$
<p>Stacking the rows gives $u=H(\phi)\dot{q}$, Eq. (13.7), p. 520, and $u=H(0)\mathcal{V}_{b}$, Eq. (13.8), p. 520.
$H(0)$ must have rank 3; with all driving and sliding directions aligned its rank drops to 2.  The tab shows
$H(0)$ of the selected robot; for three omniwheels (Fig. 13.5) and four mecanum wheels:</p>
$$h_{1}(0)=\frac{1}{r}(-d,1,0),\ h_{2,3}(0)=\frac{1}{r}\left(-d,-\frac{1}{2},\mp\sin\frac{\pi}{3}\right)$$
<p>which is Eq. (13.9), p. 521, and</p>
$$h_{1,4}(0)=\frac{1}{r}(-\ell-w,1,\mp1),\quad h_{2,3}(0)=\frac{1}{r}(\ell+w,1,\pm1)$$
<p>which is Eq. (13.10), p. 521.  Pure sideways motion: mecanum wheels 1, 3 backwards, 2, 4 forwards.  A limit
$|u_{i}|\leq u_{i,max}$ gives two parallel planes in twist space; the feasible twists form a convex polyhedron with
$2m$ faces (Fig. 13.6).  The tab plots its $\omega_{bz}=0$ slice and scales all wheel speeds together when one
saturates (the twist keeps its direction).</p>
<p><b>Feedback.</b>  Feedforward plus PI on the configuration, Eq. (13.11), p. 522, converted to wheel speeds by (13.7):</p>
$$\dot{q}(t)=\dot{q}_{d}(t)+K_{p}(q_{d}(t)-q(t))+K_{i}\int_{0}^{t}(q_{d}(t)-q(t))\,dt$$
<p>With $q_{d}=0$ this is the linear spring $\dot{q}=-Kq$ of Eq. (13.22), p. 530: the omnidirectional robot is linearly
controllable.</p>

<h3>Nonholonomic robots (Sec. 13.3.1)</h3>
<p>Simplified unicycle with driving speed $u_{1}$ and turning speed $u_{2}$, Eq. (13.13), p. 525:</p>
$$\dot{\phi}=u_{2},\quad\dot{x}=ru_{1}\cos\phi,\quad\dot{y}=ru_{1}\sin\phi$$
<p>Differential drive, wheels $2d$ apart, wheel speeds $u_{L},u_{R}$, Eq. (13.15), p. 526:</p>
$$\dot{\phi}=\frac{r}{2d}(u_{R}-u_{L})$$
$$\dot{x}=\frac{r}{2}(u_{L}+u_{R})\cos\phi,\qquad\dot{y}=\frac{r}{2}(u_{L}+u_{R})\sin\phi$$
<p>Car with wheelbase $\ell$: the steering angle follows from the controls $(v,\omega)$ by Eq. (13.17), p. 527,</p>
$$\psi=\tan^{-1}\left(\frac{\ell\omega}{v}\right)$$
<p>and all three reduce to the canonical model $\dot{q}=G(q)u$ with $u=(v,\omega)$, Eq. (13.18), p. 528:</p>
$$\dot{\phi}=\omega,\quad\dot{x}=v\cos\phi,\quad\dot{y}=v\sin\phi$$
<p>through $u_{1}=v/r,\ u_{2}=\omega$ and $u_{L}=(v-\omega d)/r,\ u_{R}=(v+\omega d)/r$ (p. 528).  They differ only in
the control sets (Fig. 13.11): a rectangle, a diamond, and the car's bowtie $|\omega|\leq|v|\tan\psi_{max}/\ell$.
The constraint is $A(q)\dot{q}=\dot{x}\sin\phi-\dot{y}\cos\phi=0$, so $v_{by}=0$ in the twist plot.</p>
<p><b>Reference point.</b>  A point P at $(x_{r},y_{r})$ in {b} can be given any world velocity; inverting
its velocity, Eq. (13.20), p. 529, gives Eq. (13.21), p. 529:</p>
$$v=\frac{(x_{r}\cos\phi-y_{r}\sin\phi)\dot{x}_{P}+(x_{r}\sin\phi+y_{r}\cos\phi)\dot{y}_{P}}{x_{r}}$$
$$\omega=\frac{-\dot{x}_{P}\sin\phi+\dot{y}_{P}\cos\phi}{x_{r}}$$
<p>which is singular for $x_{r}=0$ (the axle midpoint can only move along the heading).</p>

<h3>Controllability (Sec. 13.3.2)</h3>
<p>Theorem 13.1: since rank $G=2<3$, no continuous time-invariant feedback stabilises a configuration of the
nonholonomic robot.  Yet it is not confined: switching between vector fields produces motion along their Lie
bracket, Eq. (13.28), p. 534:</p>
$$[g_{i},g_{j}](q)=\left(\frac{\partial g_{j}}{\partial q}g_{i}-\frac{\partial g_{i}}{\partial q}g_{j}\right)(q)$$
<p>For $g_{1}=(0,\cos\phi,\sin\phi)$, $g_{2}=(1,0,0)$: $[g_{1},g_{2}]=(0,\sin\phi,-\cos\phi)$, a sideways
"parallel parking" motion of order $\epsilon^{2}$ (Example 13.5).  For left-invariant fields the same bracket is the
matrix commutator $[A,B]=AB-BA$ in se(2), Eq. (13.38), p. 560.</p>

<h3>Trajectory tracking (Sec. 13.3.4)</h3>
<p>Proportional control of the reference point, Eq. (13.29), p. 545, then (13.21):</p>
$$\dot{x}_{P}=k_{p}(x_{Pd}-x_{P}),\quad\dot{y}_{P}=k_{p}(y_{Pd}-y_{P})$$
<p>P converges, but nothing fixes the chassis orientation: the robot may drive backwards and make an unintended cusp
(Fig. 13.19; preset "P control makes a cusp").  The tab can add the desired velocity of P as feedforward.  Error coordinates of {b}
in the reference frame {d}, Eq. (13.30), p. 546:</p>
$$\phi_{e}=\phi-\phi_{d},\qquad x_{e}=(x-x_{d})\cos\phi_{d}+(y-y_{d})\sin\phi_{d}$$
$$y_{e}=-(x-x_{d})\sin\phi_{d}+(y-y_{d})\cos\phi_{d}$$
<p>Nonlinear feedforward plus feedback law, Eq. (13.31), p. 546, with $k_{1},k_{2},k_{3}>0$:</p>
$$v=\frac{v_{d}-k_{1}|v_{d}|(x_{e}+y_{e}\tan\phi_{e})}{\cos\phi_{e}}$$
$$\omega=\omega_{d}-(k_{2}v_{d}y_{e}+k_{3}|v_{d}|\tan\phi_{e})\cos^{2}\phi_{e}$$
<p>It needs $v_{d}\neq0$ and assumes $|\phi_{e}|<\pi/2$ (the tab clips $|\phi_{e}|$ at 1.4 rad inside the law).</p>

<h3>Odometry (Sec. 13.4)</h3>
<p>Wheel-angle increments $\Delta\theta$ over one sample give a constant body twist
$\mathcal{V}_{b}=F\Delta\theta$ with $F=H^{\dagger}(0)$: three omniwheels Eq. (13.32), p. 549; four mecanum
wheels Eq. (13.33), p. 549 (both shown under $H(0)$ in the controls).  For the diff-drive and the car's rear
wheels, Eq. (13.34), p. 549:</p>
$$\mathcal{V}_{b}=r\left(\frac{\Delta\theta_{R}-\Delta\theta_{L}}{2d},\ \frac{\Delta\theta_{L}+\Delta\theta_{R}}{2},\ 0\right)$$
<p>Integrating the constant twist exactly ($T_{bb'}=e^{[\mathcal{V}_{b6}]}$) gives the displacement in {b},
Eq. (13.35), p. 550 (for $\omega_{bz}\neq0$; else $\Delta q_{b}=(0,v_{bx},v_{by})$):</p>
$$\Delta x_{b}=\frac{v_{bx}\sin\omega_{bz}+v_{by}(\cos\omega_{bz}-1)}{\omega_{bz}}$$
$$\Delta y_{b}=\frac{v_{by}\sin\omega_{bz}+v_{bx}(1-\cos\omega_{bz})}{\omega_{bz}}$$
<p>with $\Delta\phi_{b}=\omega_{bz}$; Eq. (13.36), p. 550 rotates $\Delta q_{b}$ by $\phi_{k}$ and $q_{k+1}=q_{k}+\Delta q$.</p>
<p><b>In this tab</b> the wheel speeds are held for 10 ms and the true chassis moves with exactly (13.35)–(13.36).
The odometry uses the nominal wheel radius; a radius error makes it drift, and with
"feedback from odometry" the controller then tracks the estimate while the true robot wanders off.  Geometry is
illustrative (mecanum: youBot-like $r$ = 47.5 mm, $\ell$ = 235 mm, $w$ = 150 mm).</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 7 §2.2, pp. 323–325', 'Lie brackets and the motion obtained by switching vector fields'),
    ('lavalle', '§13.1.2, pp. 722–730', 'simple car, differential drive and unicycle models'),
])

DT = 0.01
FEEDBACK = {"true pose (ideal sensing)": "true", "odometry estimate": "odometry"}
KIND_NAMES = {v: k for k, v in mb.KINDS.items()}


def _poly(pts, T) -> np.ndarray:
    """Body-frame 2-D points -> world points under the planar pose T = (phi, x, y)."""
    P = np.asarray(pts, float)
    c, s = math.cos(T[0]), math.sin(T[0])
    return np.column_stack([T[1] + c * P[:, 0] - s * P[:, 1], T[2] + s * P[:, 0] + c * P[:, 1]])


def _rect(cx, cy, ang, L, W) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    pts = []
    for a, b in ((L, W), (-L, W), (-L, -W), (L, -W), (L, W)):
        pts.append((cx + c * a / 2 - s * b / 2, cy + s * a / 2 + c * b / 2))
    return np.array(pts)


def _circle(cx, cy, R, n=40) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n)
    return np.column_stack([cx + R * np.cos(a), cy + R * np.sin(a)])


def _join(parts) -> np.ndarray:
    """Concatenate polylines with NaN breaks (one PlotDataItem, connect='finite')."""
    out = []
    for p in parts:
        out.append(np.asarray(p, float))
        out.append(np.array([[np.nan, np.nan]]))
    return np.vstack(out) if out else np.zeros((0, 2))


def _arrow(a, vec, head=0.3) -> np.ndarray:
    a, vec = np.asarray(a, float), np.asarray(vec, float)
    L = float(np.hypot(*vec))
    if L < 1e-6:
        return np.zeros((0, 2))
    b = a + vec
    u = vec / L
    n = np.array([-u[1], u[0]])
    h = min(0.06, head * L)
    return np.array([a, b, b - h * u + 0.5 * h * n, b, b - h * u - 0.5 * h * n])


class MobileTab(BaseTab):
    title = "⑬ Mobile robots"
    uses_serial_robot = False
    uses_engine = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self._loading = False
        self.sim: mb.MobileSim | None = None
        self.robot = mb.default_robot("diff")

        # ---------------- scenario
        self.preset_box = compact(combo(list(mb.PRESETS), self.load_preset), 22)
        self.preset_lbl = QtWidgets.QLabel()
        self.preset_lbl.setWordWrap(True)
        self.preset_lbl.setObjectName("muted")

        # ---------------- robot
        self.kind_box = combo(list(mb.KINDS.values()), self._kind_changed)
        self.umax = SliderRow("u_max", 2, 60, 15, decimals=1, unit=" rad/s")
        self.umax.setToolTip("wheel driving-speed limit |u_i| ≤ u_max (all wheels scaled together when exceeded)")
        self.rerr = SliderRow("Δr₁", -5, 5, 0, decimals=1, unit=" %")
        self.rerr.setToolTip("true radius of wheel 1 (left wheel) minus nominal: the controller and the odometry "
                             "use the nominal radius")
        self.geom_lbl = QtWidgets.QLabel()
        self.geom_lbl.setWordWrap(True)
        self.geom_lbl.setObjectName("muted")
        self.H_view = MatrixView(3, 3, decimals=2, col_labels=["ω_bz", "v_bx", "v_by"])
        self.H_lbl = QtWidgets.QLabel()
        self.H_lbl.setWordWrap(True)
        self.H_lbl.setObjectName("muted")

        # ---------------- reference + controller
        self.ref_box = compact(combo([], self._changed), 22)
        self.cb_fixed = checkbox("keep φ_d = 0 (omni only)", self._changed)
        self.cb_fixed.setToolTip("omnidirectional robots: follow the path without turning (heading fixed at 0)")
        self.ctrl_box = combo([], self._ctrl_changed)
        self.fb_box = combo(list(FEEDBACK), self._changed)
        self.fb_box.setToolTip("q(t) used by the controller: the true pose, or the wheel-odometry estimate (13.4)")
        self.k1 = SliderRow("k₁", 0, 5, 1.0, decimals=2)
        self.k2 = SliderRow("k₂", 0, 20, 4.0, decimals=2)
        self.k3 = SliderRow("k₃", 0, 10, 2.0, decimals=2)
        self.kp = SliderRow("k_p", 0, 6, 1.5, decimals=2, unit=" 1/s")
        self.ki = SliderRow("k_i", 0, 3, 0.0, decimals=2, unit=" 1/s²")
        self.xr = SliderRow("x_r", 0.05, 0.6, 0.25, decimals=2, unit=" m")
        self.xr.setToolTip("reference point P on the chassis, (x_r, 0) in {b} (13.19)")
        self.cb_ffP = checkbox("feed forward ṗ_d", self._changed)
        self.cb_ffP.setToolTip("add the desired velocity of P to (13.29) before converting with (13.21)")
        self.dphi = SliderRow("Δφ₀", -180, 180, 0, decimals=0, unit="°")
        self.dx = SliderRow("Δx₀", -1, 1, 0, decimals=2, unit=" m")
        self.dy = SliderRow("Δy₀", -1, 1, 0, decimals=2, unit=" m")
        for w in (self.umax, self.rerr, self.k1, self.k2, self.k3, self.kp, self.ki, self.xr,
                  self.dphi, self.dx, self.dy):
            w.valueChanged.connect(self._changed)

        # ---------------- playback + readout
        self.player = Player(self)
        self.player.frame.connect(self._draw)
        self.cb_odo = checkbox("show odometry estimate", self._redraw, True)
        self.cb_ghost = checkbox("show reference frame {d}", self._redraw, True)
        self.read_lbl = value_label()
        self.sum_lbl = QtWidgets.QLabel()
        self.sum_lbl.setWordWrap(True)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Scenario", self.preset_box, self.preset_lbl))
        cl.addWidget(group("Robot", self.kind_box, self.geom_lbl, self.umax, self.rerr))
        cl.addWidget(group("H(0): wheel speeds u = H(0) V_b  (13.8)", self.H_view, self.H_lbl))
        self.gain_rows = dict(track=(self.k1, self.k2, self.k3), point=(self.kp, self.xr, self.cb_ffP),
                              pi=(self.kp, self.ki))
        cl.addWidget(group("Reference && controller", self.ref_box, self.cb_fixed, self.ctrl_box,
                           self.k1, self.k2, self.k3, self.kp, self.ki, self.xr, self.cb_ffP,
                           QtWidgets.QLabel("feedback from:"), self.fb_box))
        cl.addWidget(group("Initial error q(0) − q_d(0)", self.dphi, self.dx, self.dy))
        cl.addWidget(group("Playback", self.player.widget, self.cb_odo, self.cb_ghost))
        cl.addWidget(group("Now", self.read_lbl, self.sum_lbl))
        cl.addStretch(1)

        # ---------------- top-down view
        self.view = pg.PlotWidget()
        no_si(self.view)
        self.view.setAspectLocked(True)
        self.view.showGrid(x=True, y=True, alpha=0.15)
        self.view.setLabel("bottom", "x [m]")
        self.view.setLabel("left", "y [m]")
        self.view.addLegend(offset=(6, 6), labelTextSize="8pt")
        self.c_ref = self.view.plot(pen=pg.mkPen(theme.MUTED, width=1.5, style=QtCore.Qt.DashLine), name="q_d(t)")
        self.c_path = self.view.plot(pen=pen(0, 2), name="true path")
        self.c_odo = self.view.plot(pen=pen(3, 1.5, "dash"), name="odometry estimate")
        self.c_P = self.view.plot(pen=pen(2, 1.2), name="point P")
        self.c_ghost = self.view.plot(pen=pg.mkPen(theme.MUTED, width=1.2, style=QtCore.Qt.DotLine),
                                      connect="finite")
        self.c_odobody = self.view.plot(pen=pen(3, 1.2, "dash"), connect="finite")
        self.c_body = self.view.plot(pen=pg.mkPen(theme.LINK, width=2), connect="finite")
        self.c_wheels = self.view.plot(pen=pg.mkPen(theme.JOINT, width=2), connect="finite")
        self.c_rollers = self.view.plot(pen=pg.mkPen(theme.MUTED, width=1), connect="finite")
        self.c_axes = self.view.plot(pen=pg.mkPen(theme.BAD, width=2), connect="finite")
        self.c_yaxis = self.view.plot(pen=pg.mkPen(theme.GOOD, width=2), connect="finite")
        self.c_speed = self.view.plot(pen=pg.mkPen(theme.WARN, width=2), connect="finite")
        self.c_Pdot = self.view.plot(pen=None, symbol="o", symbolSize=8, symbolBrush=theme.SERIES[2])
        self.c_Pd = self.view.plot(pen=None, symbol="x", symbolSize=10, symbolPen=pg.mkPen(theme.SERIES[2], width=2))
        center = self.view

        # ---------------- plots
        self.p_xy = plot("Position", "[m]")
        self.p_phi = plot("Heading", "φ [deg]")
        self.p_err = plot("Errors", "[m]")
        self.p_u = plot("Wheel speeds", "u [rad/s]")
        self.p_V = plot("Body twist V_b", "")
        self.p_c = plot("Control set", "", "")
        for p in (self.p_xy, self.p_phi, self.p_err, self.p_u, self.p_V, self.p_c):
            no_si(p)
        for p in (self.p_phi, self.p_err, self.p_u, self.p_V):
            p.setXLink(self.p_xy)
        self.c_x = self.p_xy.plot(pen=pen(0), name="x")
        self.c_y = self.p_xy.plot(pen=pen(1), name="y")
        self.c_xd = self.p_xy.plot(pen=pen(0, 1.2, "dash"), name="x_d")
        self.c_yd = self.p_xy.plot(pen=pen(1, 1.2, "dash"), name="y_d")
        self.c_phi = self.p_phi.plot(pen=pen(4), name="φ")
        self.c_phid = self.p_phi.plot(pen=pen(4, 1.2, "dash"), name="φ_d")
        self.c_phie = self.p_phi.plot(pen=pen(5, 1.5), name="φ_e")
        self.c_et = self.p_err.plot(pen=pen(0), name="|(x_e, y_e)|")
        self.c_eP = self.p_err.plot(pen=pen(2, 1.5), name="|P − P_d|")
        self.c_eo = self.p_err.plot(pen=pen(3, 1.5, "dash"), name="|odometry − true|")
        self.c_u: list = []
        self.u_lim = [pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.BAD, width=1, style=QtCore.Qt.DashLine))
                      for _ in range(2)]
        for ln in self.u_lim:
            self.p_u.addItem(ln, ignoreBounds=True)
        self.c_wz = self.p_V.plot(pen=pen(4), name="ω_bz [rad/s]")
        self.c_vx = self.p_V.plot(pen=pen(0), name="v_bx [m/s]")
        self.c_vy = self.p_V.plot(pen=pen(1), name="v_by [m/s]")
        self.c_set = self.p_c.plot(pen=pg.mkPen(theme.MUTED, width=1.5), fillLevel=None)
        self.c_trace = self.p_c.plot(pen=pen(0, 1.5))
        self.c_now = self.p_c.plot(pen=None, symbol="o", symbolSize=9, symbolBrush=theme.WARN)
        self.cursors = [time_cursor(p) for p in (self.p_xy, self.p_phi, self.p_err, self.p_u, self.p_V)]
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for i, p in enumerate((self.p_xy, self.p_phi, self.p_err, self.p_u, self.p_V, self.p_c)):
            grid.addWidget(p, i // 3, i % 3)
        bottom = QtWidgets.QWidget()
        bottom.setLayout(grid)

        self.assemble(controls, center, bottom, THEORY, controls_width=360, bottom_ratio=(1, 1))

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.recompute)
        self.load_preset(self.preset_box.currentText())

    # ------------------------------------------------------------------ presets / parameters
    def apply_preset(self, key) -> bool:
        """Select a preset by its name, its index, or a case-insensitive part of its name (launch URLs)."""
        names = list(mb.PRESETS)
        key = str(key).strip()
        name = None
        if key.isdigit() and int(key) < len(names):
            name = names[int(key)]
        elif key in mb.PRESETS:
            name = key
        else:
            hits = [n for n in names if key.lower() in n.lower()]
            name = hits[0] if hits else None
        if name is None:
            return False
        if self.preset_box.currentText() == name:
            self.load_preset(name)                # same entry: no currentTextChanged signal
        else:
            self.preset_box.setCurrentText(name)  # emits -> load_preset
        return True

    def load_preset(self, name: str):
        if name not in mb.PRESETS:
            return
        p = mb.PRESETS[name]
        robot, ref, ctrl, q0, g, fb = mb.preset(name)
        self._loading = True
        self.preset_lbl.setText(p["text"])
        self.robot = robot
        self.kind_box.setCurrentText(mb.KINDS[robot.kind])
        self._fill_lists()
        self.umax.set_value(robot.u_max)
        self.rerr.set_value(100 * (robot.radius_error[0] if robot.radius_error else 0.0))
        self.ref_box.setCurrentText(p["ref"])
        self.cb_fixed.setChecked(False)
        self.ctrl_box.setCurrentText(p["ctrl"])
        self.fb_box.setCurrentText({v: k for k, v in FEEDBACK.items()}[fb])
        for row, val in ((self.k1, g.k1), (self.k2, g.k2), (self.k3, g.k3), (self.kp, g.kp), (self.ki, g.ki),
                         (self.xr, g.xr)):
            row.set_value(val)
        self.cb_ffP.setChecked(g.feedforward_P)
        qd0 = ref.sample(0.0)[0]
        dphi = math.degrees(float(mb.wrap(q0[0] - qd0[0])))
        self.dphi.set_value(180.0 if dphi < -179.999 else dphi)
        self.dx.set_value(q0[1] - qd0[1])
        self.dy.set_value(q0[2] - qd0[2])
        self._loading = False
        self._update_visibility()
        self.recompute()

    def _fill_lists(self):
        """Reference and controller lists that make sense for the selected robot."""
        hol = self.robot.holonomic
        refs = [n for n in mb.REFERENCES if hol or "(omni)" not in n]
        ctrls = list(mb.CONTROLLERS_OMNI if hol else mb.CONTROLLERS_NH)
        for box, items in ((self.ref_box, refs), (self.ctrl_box, ctrls)):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(items)
            if cur in items:
                box.setCurrentText(cur)
            box.blockSignals(False)
        self.cb_fixed.setEnabled(hol)
        if not hol:
            self.cb_fixed.setChecked(False)

    def _kind_changed(self, text):
        if self._loading:
            return
        kind = KIND_NAMES.get(text, "diff")
        self.robot = mb.default_robot(kind)
        self._loading = True
        self._fill_lists()
        self._loading = False
        self._changed()

    def _ctrl_changed(self, *_):
        self._update_visibility()
        self._changed()

    def _update_visibility(self):
        code = self._ctrl_code()
        shown = set(self.gain_rows.get(code, ()))
        for w in (self.k1, self.k2, self.k3, self.kp, self.ki, self.xr, self.cb_ffP):
            w.setVisible(w in shown)

    def _ctrl_code(self) -> str:
        return {**mb.CONTROLLERS_NH, **mb.CONTROLLERS_OMNI}.get(self.ctrl_box.currentText(), "open")

    def _changed(self, *_):
        if not self._loading:
            self._update_visibility()
            self._debounce.start(80)

    def current(self):
        """(robot, reference, controller code, q0, gains, feedback) from the widgets."""
        rob = self.robot
        rob.u_max = self.umax.value()
        e = self.rerr.value() / 100.0
        rob.radius_error = (e,) if e else ()
        fixed = 0.0 if (self.cb_fixed.isChecked() and rob.holonomic) else None
        name = self.ref_box.currentText() or "Circle (R = 1 m, 0.3 m/s)"
        ref = mb.make_reference(name, fixed)
        qd0 = ref.sample(0.0)[0]
        q0 = qd0 + np.array([math.radians(self.dphi.value()), self.dx.value(), self.dy.value()])
        g = mb.Gains(k1=self.k1.value(), k2=self.k2.value(), k3=self.k3.value(), kp=self.kp.value(),
                     ki=self.ki.value(), xr=self.xr.value(), feedforward_P=self.cb_ffP.isChecked())
        return rob, ref, self._ctrl_code(), q0, g, FEEDBACK.get(self.fb_box.currentText(), "true")

    # ------------------------------------------------------------------ computation
    def recompute(self):
        rob, ref, ctrl, q0, g, fb = self.current()
        self.sim = mb.simulate(rob, ref, ctrl, q0, g, dt=DT, feedback=fb)
        self._update_static()
        self._update_plots()
        self.player.set_times(self.sim.t)
        self._draw(0)

    def _update_static(self):
        rob = self.robot
        H0 = rob.H0()
        self.H_view.set_matrix(H0)
        self.H_view.set_labels(rows=rob.wheel_names(), cols=["ω_bz", "v_bx", "v_by"])
        rank = int(np.linalg.matrix_rank(H0))
        F = rob.F()
        if rob.kind == "unicycle":
            note = "unicycle: u₁ = v_bx / r, u₂ = ω_bz (13.13); v_by = 0 is not available."
        elif rob.holonomic:
            note = (f"rank H(0) = {rank} (needs 3, p. 520).  Odometry F = H†(0) "
                    f"({'13.32' if rob.kind == 'omni3' else '13.33'}): "
                    + "; ".join("[" + ", ".join(f"{v:+.4f}" for v in row) + "]" for row in F))
        else:
            note = (f"Fixed wheels as rows of (13.6) with γ = 0: rank H(0) = {rank} < 3 — no sideways twist.  "
                    f"F (13.34): ω = r(Δθ_R − Δθ_L)/2d, v = r(Δθ_L + Δθ_R)/2.")
        self.H_lbl.setText(note)
        if rob.kind == "omni3":
            g = f"r = {rob.r * 1e3:.0f} mm, d = {rob.d * 1e3:.0f} mm (Fig. 13.5)"
        elif rob.kind == "mecanum":
            g = f"r = {rob.r * 1e3:.1f} mm, ℓ = {rob.l * 1e3:.0f} mm, w = {rob.w * 1e3:.0f} mm (Fig. 13.5)"
        elif rob.kind == "car":
            g = (f"r = {rob.r * 1e3:.0f} mm, rear half-track d = {rob.d * 1e3:.0f} mm, wheelbase ℓ = "
                 f"{rob.ell * 1e3:.0f} mm, |ψ| ≤ {math.degrees(rob.psi_max):.0f}°")
        elif rob.kind == "unicycle":
            g = f"r = {rob.r * 1e3:.0f} mm, |u₂| ≤ {rob.omega_max:.1f} rad/s"
        else:
            g = f"r = {rob.r * 1e3:.0f} mm, wheels 2d = {2 * rob.d * 1e3:.0f} mm apart"
        self.geom_lbl.setText(g + f";  max speed r·u_max = {rob.max_speed():.2f} m/s")

    def _update_plots(self):
        s = self.sim
        t = s.t
        q = s.q.copy()
        qd = s.q_d.copy()
        self.c_ref.setData(qd[:, 1], qd[:, 2])
        self.c_path.setData(q[:, 1], q[:, 2])
        show_odo = self.cb_odo.isChecked() and (s.feedback == "odometry" or bool(s.robot.radius_error))
        self.c_odo.setData(*(s.q_odo[:, 1:].T if show_odo else ([], [])))
        if s.controller == "point":
            self.c_P.setData(s.P[:, 0], s.P[:, 1])
        else:
            self.c_P.setData([], [])
        pts = np.vstack([q[:, 1:], qd[:, 1:]])
        lo, hi = pts.min(0) - 0.45, pts.max(0) + 0.45
        self.view.setRange(xRange=(lo[0], hi[0]), yRange=(lo[1], hi[1]), padding=0.02)

        self.c_x.setData(t, q[:, 1])
        self.c_y.setData(t, q[:, 2])
        self.c_xd.setData(t, qd[:, 1])
        self.c_yd.setData(t, qd[:, 2])
        ph = np.degrees(np.unwrap(q[:, 0]))
        phd = np.degrees(np.unwrap(qd[:, 0]))
        self.c_phi.setData(t, ph)
        self.c_phid.setData(t, phd)
        self.c_phie.setData(t, np.degrees(s.err[:, 0]))
        self.c_et.setData(t, s.track_err)
        if s.controller == "point":
            self.c_eP.setData(t, np.linalg.norm(s.P - s.P_d, axis=1))
        else:
            self.c_eP.setData([], [])
        self.c_eo.setData(t, s.odo_err)

        for c in self.c_u:
            self.p_u.removeItem(c)
        names = s.robot.wheel_names()
        self.c_u = [self.p_u.plot(t, s.u[:, i], pen=pen(i), name=names[i]) for i in range(s.u.shape[1])]
        lim = s.robot.u_max
        for ln, v in zip(self.u_lim, (lim, -lim)):
            ln.setValue(v)
        self.c_wz.setData(t, s.Vb[:, 0])
        self.c_vx.setData(t, s.Vb[:, 1])
        self.c_vy.setData(t, s.Vb[:, 2])

        poly = s.robot.control_set()
        self.c_set.setData(poly[:, 0], poly[:, 1])
        if s.robot.holonomic:
            self.p_c.setTitle("Feasible twists (ω_bz = 0)", color=theme.text_color(), size="10pt")
            self.p_c.setLabel("bottom", "v_bx [m/s]")
            self.p_c.setLabel("left", "v_by [m/s]")
            self.c_trace.setData(s.Vb[:, 1], s.Vb[:, 2])
        else:
            self.p_c.setTitle("Control set (v, ω)", color=theme.text_color(), size="10pt")
            self.p_c.setLabel("bottom", "v [m/s]")
            self.p_c.setLabel("left", "ω [rad/s]")
            self.c_trace.setData(s.Vb[:, 1], s.Vb[:, 0])
        for p in (self.p_xy, self.p_phi, self.p_err, self.p_u, self.p_V, self.p_c):
            p.enableAutoRange()
        # constant signals: keep a readable y range instead of pyqtgraph's 1e-15 zoom
        for p, arrs, span in ((self.p_u, [s.u], 1.0), (self.p_V, [s.Vb], 0.1), (self.p_err, [s.track_err, s.odo_err], 0.05),
                              (self.p_phi, [ph, phd], 10.0)):
            a = np.concatenate([np.ravel(x) for x in arrs])
            if np.ptp(a) < span:
                mid = 0.5 * (a.max() + a.min())
                p.setYRange(mid - span, mid + span, padding=0.05)
        self._summary()
        self.restyle_plots()

    def _summary(self):
        s = self.sim
        good, bad, warn = theme.good_color(), theme.bad_color(), theme.warn_color()
        e_end = float(s.track_err[-1])
        pe_end = math.degrees(abs(float(s.err[-1, 0])))
        ok = e_end < 0.02 and pe_end < 3.0
        nsat = int(np.sum(s.saturated))
        txt = (f"<span style='color:{good if ok else warn}'>{'✓ tracking' if ok else '• not converged'}</span>: "
               f"final |(x_e, y_e)| = {e_end * 100:.1f} cm, |φ_e| = {pe_end:.1f}°")
        v = s.Vb[:, 1]
        flips = int(np.sum(np.diff(np.sign(v[np.abs(v) > 1e-9])) != 0))
        if not s.robot.holonomic and flips:
            txt += (f"<br><span style='color:{warn}'>cusp: v changed sign {flips}× "
                    f"(reversing for {DT * np.sum(v < -1e-9):.1f} s, Fig. 13.19)</span>")
        if s.controller == "point" and pe_end > 90:
            txt += f"<br><span style='color:{warn}'>robot ends up driving backwards (φ_e ≈ 180°)</span>"
        if nsat:
            txt += f"<br><span style='color:{bad}'>wheel limits active in {100 * nsat / len(s.t):.0f} % of samples</span>"
        if s.robot.radius_error or s.feedback == "odometry":
            txt += f"<br>odometry error at the end: {s.odo_err[-1] * 100:.1f} cm"
        if not s.robot.holonomic:
            txt += f"<br>max |v_by| = {np.max(np.abs(s.Vb[:, 2])):.1e} m/s (nonholonomic)"
        self.sum_lbl.setText(txt)
        self.sum_lbl.setTextFormat(QtCore.Qt.RichText)

    # ------------------------------------------------------------------ drawing
    def _shape(self, rob: mb.MobileRobot, Vb):
        """Body outline, wheel rectangles and roller lines in {b}; wheel driving directions + centres."""
        body, wheels, rollers = [], [], []
        if rob.kind == "diff":
            body.append(_circle(0, 0, 1.3 * rob.d))
            body.append(_circle(-0.9 * rob.d, 0, 0.02, 12))
        elif rob.kind == "unicycle":
            body.append(_circle(0, 0, 0.06, 24))
        elif rob.kind == "car":
            body.append(_rect(rob.ell / 2, 0, 0, rob.ell + 0.16, 2 * rob.d + 0.06))
        elif rob.kind == "omni3":
            body.append(_circle(0, 0, rob.d + 0.03, 48))
        else:
            body.append(_rect(0, 0, 0, 2 * rob.l + 0.12, 2 * rob.w - 0.05))
        for wh in rob.wheels():
            Wd = 0.03 if rob.holonomic else 0.025
            wheels.append(_rect(wh.x, wh.y, wh.beta, 2 * wh.r, Wd))
            if rob.holonomic:
                sd = wh.beta + wh.gamma + math.pi / 2           # free-sliding direction (13.3)
                c, s = math.cos(sd), math.sin(sd)
                L = 0.8 * wh.r
                rollers.append(np.array([[wh.x - c * L, wh.y - s * L], [wh.x + c * L, wh.y + s * L]]))
        if rob.kind == "car":                                    # steered front wheels (Ackermann)
            psi = mb.car_steering_angle(float(Vb[1]), float(Vb[0]), rob.ell)
            for side in (+1, -1):
                if abs(psi) < 1e-6:
                    a = 0.0
                else:
                    R = rob.ell / math.tan(psi)
                    a = math.atan(rob.ell / (R - side * rob.d))
                wheels.append(_rect(rob.ell, side * rob.d, a, 2 * rob.r, 0.025))
        return body, wheels, rollers

    def _redraw(self, *_):
        if self.sim is not None:
            self._update_plots()
            self._draw(self.player.k)

    def _draw(self, k: int):
        s = self.sim
        if s is None:
            return
        k = int(np.clip(k, 0, len(s.t) - 1))
        q, qd, rob = s.q[k], s.q_d[k], s.robot
        Vb = s.Vb[k]
        body, wheels, rollers = self._shape(rob, Vb)
        self.c_body.setData(*_join([_poly(b, q) for b in body]).T)
        self.c_wheels.setData(*_join([_poly(w, q) for w in wheels]).T)
        self.c_rollers.setData(*_join([_poly(r, q) for r in rollers]).T)
        L = 0.18
        self.c_axes.setData(*_join([_poly(_arrow((0, 0), (L, 0)), q)]).T)
        self.c_yaxis.setData(*_join([_poly(_arrow((0, 0), (0, 0.6 * L)), q)]).T)
        # wheel rim speeds r u_i along each driving direction (0.5 s of travel)
        arrows = []
        for i, wh in enumerate(rob.wheels()[: s.u.shape[1]]):
            if rob.kind == "unicycle" and i == 1:
                break
            v = wh.r * s.u[k, i] * 0.5
            arrows.append(_arrow((wh.x, wh.y), (v * math.cos(wh.beta), v * math.sin(wh.beta))))
        self.c_speed.setData(*_join([_poly(a, q) for a in arrows if len(a)]).T)
        if self.cb_ghost.isChecked():
            self.c_ghost.setData(*_join([_poly(b, qd) for b in body] + [_poly(_arrow((0, 0), (L, 0)), qd)]).T)
        else:
            self.c_ghost.setData([], [])
        if self.cb_odo.isChecked() and (s.feedback == "odometry" or bool(rob.radius_error)):
            self.c_odobody.setData(*_join([_poly(b, s.q_odo[k]) for b in body]).T)
        else:
            self.c_odobody.setData([], [])
        if s.controller == "point":
            self.c_Pdot.setData([s.P[k, 0]], [s.P[k, 1]])
            self.c_Pd.setData([s.P_d[k, 0]], [s.P_d[k, 1]])
        else:
            self.c_Pdot.setData([], [])
            self.c_Pd.setData([], [])
        for ln in self.cursors:
            ln.setValue(s.t[k])
        if rob.holonomic:
            self.c_now.setData([Vb[1]], [Vb[2]])
        else:
            self.c_now.setData([Vb[1]], [Vb[0]])
        u = ", ".join(f"{v:+.2f}" for v in s.u[k])
        extra = ""
        if rob.kind == "car":
            extra = f"<br>steering ψ = {math.degrees(mb.car_steering_angle(Vb[1], Vb[0], rob.ell)):+.1f}° (13.17)"
        self.read_lbl.setText(
            f"t = {s.t[k]:.2f} s<br>q = ({math.degrees(q[0]):+.1f}°, {q[1]:+.3f}, {q[2]:+.3f})"
            f"<br>q_d = ({math.degrees(mb.wrap(qd[0])):+.1f}°, {qd[1]:+.3f}, {qd[2]:+.3f})"
            f"<br>V_b = ({Vb[0]:+.3f} rad/s, {Vb[1]:+.3f}, {Vb[2]:+.3f} m/s)"
            f"<br>u = ({u}) rad/s{'  <b>saturated</b>' if s.saturated[k] else ''}{extra}")

    # ------------------------------------------------------------------ show / hide
    def on_hide(self):
        self.player.stop()

    def shutdown(self):
        self.player.stop()
        self._debounce.stop()
