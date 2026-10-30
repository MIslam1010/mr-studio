"""Tab -- motion planning for planar 2R / 3R arms (MR Ch. 10): C-space obstacles, grid search (Dijkstra, A*),
sampling planners (PRM, RRT, RRT*), shortcut smoothing and time scaling of the planned path (MR Ch. 9)."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ... import planning as P
from ...references import further_reading
from ...robots import get_robot
from .. import theme
from ..base import BaseTab
from ..jobs import JobRunner
from ..widgets import (JointSliders, button, checkbox, combo, group, hbox, pen, plot, spin, value_label)

THEORY = r"""
<h2>Motion planning in the configuration space</h2>
<span class="ref">Modern Robotics, Ch. 10 (§10.1–10.5, §10.8) and §9.2 (time scaling)</span>
<p>A <b>path planner</b> finds a collision-free path $q(s)$, $s\in[0,1]$, from $q_{start}$ to $q_{goal}$ (§10.1,
p. 355).  It works in the configuration space $\mathcal{C}$, where the robot is a single point and the
workspace obstacles $\mathcal{B}_i$ become <b>C-obstacles</b> (§10.2.1, p. 360):</p>
$$\mathcal{C}_{obs}=\{\,q\in\mathcal{C}\ :\ \mathcal{A}(q)\cap\mathcal{B}\neq\emptyset\,\}$$
$$\mathcal{C}_{free}=\mathcal{C}\setminus\mathcal{C}_{obs}$$
<p>For the 2R arm, $q=(\theta_1,\theta_2)$ and $\mathcal{C}=S^1\times S^1$ is a <b>torus</b>: the right edge of the
C-space picture is glued to the left edge and the top to the bottom (MR Fig. 10.2).  Hover the C-space to see
the arm at that configuration; an obstacle within reach of link 1 blocks a whole band of $\theta_1$.</p>
<h3>Collision detection (§10.2.2, p. 364)</h3>
<p>Each link is a segment of half-thickness $r$; with link end points from the PoE
$e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M$, $q$ is free iff
$d(\mathrm{link}_i(q),\mathcal{B}_j)>r$ for every link and obstacle.  A straight edge in $\mathcal{C}$ (the local
planner, §10.5.1.3, p. 384) is checked at samples with clearance $c_k$; since no arm point moves more than
$\beta(\Delta\theta)=\sum_j|\Delta\theta_j|\rho_j$ ($\rho_j$ = chain length from joint $j$ to the tip), a step is
certified free when $c_k+c_{k+1}>\beta$, otherwise it is bisected — no tunnelling through thin obstacles.</p>
<h3>Graph search: A* and Dijkstra (§10.2.4.1, p. 367; Algorithm 10.1, p. 369)</h3>
<p>Nodes in OPEN are sorted by the estimated total cost</p>
$$f(n)=\mathrm{past\_cost}(n)+h(n),\qquad 0\leq h(n)\leq h^{*}(n)$$
<p>With the straight-line (torus) distance to the goal as $h$ the heuristic is <b>optimistic</b>, so A* returns
a minimum-cost path — the same cost as Dijkstra ($h\equiv 0$), but it expands fewer nodes (the coloured
dots: early = blue, late = yellow).</p>
<h3>Grid methods (§10.4, p. 371)</h3>
<p>$\mathcal{C}$ is divided into $k^n$ cells; a cell is an obstacle if its centre collides, and a move to a
neighbouring free cell is allowed only if the straight segment between the centres is free (no corner cutting).
8-connected (diagonal moves, cost $\sqrt{2}\,\Delta\theta$) or 4-connected.  Resolution matters: a coarse grid can miss narrow
passages; the cost grows as $k^n$ (curse of dimensionality).</p>
<h3>Sampling methods (§10.5, p. 380)</h3>
<p><b>RRT</b> (Algorithm 10.3, p. 381): sample $q_{samp}$ (the goal with probability $p_{goal}$), find the nearest
tree node $q_{nearest}$, move a step $\delta$ towards $q_{samp}$; if the edge is free add $q_{new}$, stop when
the goal can be connected.</p>
$$q_{new}=q_{nearest}+\min\left(1,\frac{\delta}{\|q_{samp}-q_{nearest}\|}\right)(q_{samp}-q_{nearest})$$
<p><b>RRT*</b> (§10.5.1.4, p. 385): $q_{new}$ takes the parent that minimises the cost-to-come among nodes within
radius $\rho$, then <i>rewires</i> neighbours through $q_{new}$ when cheaper — the cost tends to the optimum as
samples are added.</p>
<p><b>PRM</b> (§10.5.2, Algorithm 10.4, p. 387): sample $N$ free configurations, connect each to its $k$ nearest
neighbours by free straight edges (a roadmap, reusable for many queries), then search it with A*.</p>
<h3>Smoothing (§10.8, p. 396)</h3>
<p>Shortcutting: pick two points on the path; if the straight segment between them is free, replace the
stretch in between.  The straight segment is the shortest connection, so the length never increases.</p>
<h3>From path to trajectory (§9.2.2.1, p. 331)</h3>
<p>The geometric path is parameterised by arc length, $\theta(s)$, and timed with a time scaling, e.g.
the quintic that starts and ends at rest:</p>
$$s(t)=10\left(\frac{t}{T}\right)^{3}-15\left(\frac{t}{T}\right)^{4}+6\left(\frac{t}{T}\right)^{5}$$
$$\theta(t)=\theta(s(t)),\qquad\dot{\theta}=\frac{d\theta}{ds}\,\dot{s}$$
<p>On a polyline $d\theta/ds$ jumps at the corners; "rest at via points" times each segment separately
so the velocity is continuous (zero) there.</p>
"""
THEORY += further_reading([
    ('lavalle', '§4.3, pp. 155–167', 'configuration-space obstacles'),
    ('lavalle', '§2.2.2, pp. 35–38', 'Dijkstra (p. 36) and A* (p. 37) forward search'),
    ('lavalle', '§5.5, from p. 228', 'rapidly exploring dense trees (RRT)'),
    ('lavalle', '§5.6, from p. 237', 'roadmap methods for multiple queries (PRM)'),
])

#: colours
C_OBS = (255, 107, 107)          # C-obstacle
C_LIM = (70, 76, 88)             # outside joint limits
C_FREE = (36, 42, 52)
C_START = theme.GOOD
C_GOAL = theme.BAD
C_PATH = theme.WARN
C_RAW = "#c58af9"
ALL_PLANNERS = P.PLANNERS


def _qcol(c, alpha=255):
    q = QtGui.QColor(c) if isinstance(c, str) else QtGui.QColor(*c)
    q.setAlpha(alpha)
    return q


def _wrap_copies(xy: np.ndarray, wrap):
    """Copies of a curve shifted by +-360 deg on the wrapping axes (the view clips them) -> torus display."""
    shifts = [(-360.0, 0.0, 360.0) if wrap[0] else (0.0,), (-360.0, 0.0, 360.0) if wrap[1] else (0.0,)]
    return [xy + np.array([a, b]) for a in shifts[0] for b in shifts[1]]


class PlanningTab(BaseTab):
    title = "⑩ Planning"
    uses_serial_robot = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self.arm: P.PlanarArm | None = None
        self.world: P.World | None = None
        self.rois: list = []
        self.result: P.PlanResult | None = None
        self.traj = None
        self.jobs = JobRunner(self)             # one planning job at a time; stale generations are dropped
        self._grid_cache = None                 # ((world fingerprint, grid_k), CGrid)
        self._job = 0                           # bumped whenever the world / robot changes or a plan starts
        self.history: list = []

        # ---------------------------------------------------------------- views
        self.ws = pg.PlotWidget()
        self.ws.setAspectLocked(True)
        self.ws.showGrid(x=True, y=True, alpha=0.12)
        self.ws.setLabel("bottom", "x [m]")
        self.ws.setLabel("left", "y [m]")
        for ax in ("left", "bottom"):
            self.ws.getAxis(ax).enableAutoSIPrefix(False)     # metres, never "(x0.001)"
        self.ws.setTitle("Workspace — drag obstacles (handles resize / reshape)", color=theme.TEXT, size="10pt")
        self.ws.setMenuEnabled(False)
        self.ws.setMinimumSize(300, 260)
        self.cs = pg.PlotWidget()
        # no aspect lock: the view box is exactly [-180, 180]^2, so it clips the wrapped copies of curves
        self.cs.setMenuEnabled(False)
        self.cs.setMouseEnabled(False, False)
        self.cs.hideButtons()
        self.cs.setLabel("bottom", "θ₁ [deg]")
        self.cs.setLabel("left", "θ₂ [deg]")
        self.cs.setXRange(-180, 180, padding=0)
        self.cs.setYRange(-180, 180, padding=0)
        self.cs.getPlotItem().vb.setLimits(xMin=-180, xMax=180, yMin=-180, yMax=180)
        for a in ("left", "bottom"):
            self.cs.getAxis(a).setTicks([[(v, f"{v}") for v in range(-180, 181, 90)],
                                         [(v, "") for v in range(-180, 181, 30)]])
        self.cs.setMinimumSize(300, 260)
        self._init_items()

        centre = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        centre.addWidget(self.ws)
        centre.addWidget(self.cs)
        centre.setSizes([500, 500])

        # ---------------------------------------------------------------- controls
        self.robot_box = combo([], self._robot_selected)
        self.robot_note = QtWidgets.QLabel()
        self.robot_note.setObjectName("muted")
        self.robot_note.setWordWrap(True)
        self.scene_box = combo(list(P.SCENES), lambda _t: self.load_scene())
        add_c = button("+ Circle", lambda: self._add_obstacle("circle"))
        add_b = button("+ Box", lambda: self._add_obstacle("box"))
        rm = button("Remove last", self._remove_last)
        reset = button("Reset", self.load_scene)
        self.torus_cb = checkbox("θ wraps at ±180° (torus C-space)", lambda _v: self._world_changed(), True)
        self.torus_cb.setToolTip("Joints without limits live on a circle S¹: the 2R C-space is a torus T² (MR Sec. 2.3).")

        self.start_sl = JointSliders()
        self.goal_sl = JointSliders()
        self.start_sl.changed.connect(lambda _v: self._endpoints_changed())
        self.goal_sl.changed.connect(lambda _v: self._endpoints_changed())
        swap = button("Swap", self._swap)
        rnd = button("Random free goal", self._random_goal)
        self.ep_lbl = value_label()
        click_hint = QtWidgets.QLabel("Click the C-space: goal  ·  Shift-click: start")
        click_hint.setObjectName("muted")

        self.planner_box = combo(ALL_PLANNERS, self._planner_changed)
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(8, 160)
        self.k_spin.setValue(64)
        self.connect_box = combo(["8-connected (diagonals)", "4-connected"])
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(20, 3000)
        self.n_spin.setValue(300)
        self.knn_spin = QtWidgets.QSpinBox()
        self.knn_spin.setRange(2, 40)
        self.knn_spin.setValue(10)
        self.iter_spin = QtWidgets.QSpinBox()
        self.iter_spin.setRange(50, 20000)
        self.iter_spin.setSingleStep(250)
        self.iter_spin.setValue(3000)
        self.step_spin = spin(1, 90, 15, 1, 1, "°")
        self.bias_spin = spin(0, 1, 0.10, 2, 0.05)
        self.rad_spin = spin(2, 180, 30, 1, 1, "°")
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.smooth_cb = checkbox("Shortcut smoothing (MR 10.8)", None, True)
        self.smooth_spin = QtWidgets.QSpinBox()
        self.smooth_spin.setRange(1, 5000)
        self.smooth_spin.setValue(200)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._rows = {}

        def row(key, label, w):
            lab = QtWidgets.QLabel(label)
            form.addRow(lab, w)
            self._rows[key] = (lab, w)
        row("k", "grid k per joint", self.k_spin)
        row("conn", "neighbours", self.connect_box)
        row("N", "samples N", self.n_spin)
        row("knn", "k nearest", self.knn_spin)
        row("iter", "max iterations", self.iter_spin)
        row("step", "step δ", self.step_spin)
        row("bias", "goal bias p", self.bias_spin)
        row("rad", "rewire radius ρ", self.rad_spin)
        row("seed", "random seed", self.seed_spin)
        self.plan_btn = button("Plan", self.plan, primary=True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setMaximumHeight(16)
        self.cancel_btn = button("Cancel", self._cancel)
        self.busy = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(self.busy)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.progress, 1)
        bl.addWidget(self.cancel_btn)
        self.busy.setVisible(False)
        self.res_lbl = value_label()

        self.cb_explored = checkbox("Explored nodes / tree / roadmap", self._redraw_result, True)
        self.cb_raw = checkbox("Path before smoothing", self._redraw_result, True)
        self.cb_hover = checkbox("Ghost arm under the C-space cursor", None, True)

        self.scale_box = combo(["quintic", "cubic", "trapezoidal"], lambda _t: self._retime())
        self.rest_cb = checkbox("Rest at via points", lambda _v: self._retime(), False)
        self.Tf_spin = spin(0.5, 30, 4.0, 1, 0.5, " s")
        self.Tf_spin.valueChanged.connect(lambda _v: self._retime())
        self.anim_btn = button("▶ Animate along path", self.toggle_anim, checkable=True)
        self.anim_btn.setEnabled(False)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Robot && scene", self.robot_box, self.robot_note, self.scene_box,
                           hbox(add_c, add_b, rm, reset), self.torus_cb))
        cl.addWidget(group("Start (green) / goal (red)", QtWidgets.QLabel("start θ"), self.start_sl,
                           QtWidgets.QLabel("goal θ"), self.goal_sl, hbox(swap, rnd), click_hint, self.ep_lbl))
        cl.addWidget(group("Planner", self.planner_box, form, hbox(self.smooth_cb, self.smooth_spin),
                           self.plan_btn, self.busy, self.res_lbl))
        cl.addWidget(group("Display", self.cb_explored, self.cb_raw, self.cb_hover))
        cl.addWidget(group("Trajectory (MR 9.2)", hbox(QtWidgets.QLabel("s(t)"), self.scale_box, self.Tf_spin),
                           self.rest_cb, self.anim_btn))
        cl.addStretch(1)

        # ---------------------------------------------------------------- bottom
        self.p_th = plot("Time-scaled path θ(t)", "θ [deg]")
        self.p_dth = plot("Joint velocities θ̇(t)", "θ̇ [deg/s]")
        self.hist = QtWidgets.QTableWidget(0, 6)
        self.hist.setHorizontalHeaderLabels(["planner", "ok", "length [rad]", "nodes", "coll. checks", "time [s]"])
        self.hist.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.hist.horizontalHeader().setStretchLastSection(True)
        self.hist.verticalHeader().setVisible(False)
        self.hist.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.hist.setMinimumWidth(300)
        self.hist.setToolTip("Every planning run (compare A* with Dijkstra, RRT with RRT* …)")
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(self.p_th)
        bottom.addWidget(self.p_dth)
        bottom.addWidget(self.hist)
        bottom.setSizes([360, 360, 330])

        self.assemble(controls, centre, bottom, THEORY, controls_width=350, bottom_ratio=(5, 2))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._anim_step)
        self._cs_timer = QtCore.QTimer(self)
        self._cs_timer.setSingleShot(True)
        self._cs_timer.timeout.connect(lambda: self._update_cspace(fast=True))
        self._planner_changed(self.planner_box.currentText())
        self.jobs.busy_changed.connect(self._busy_changed)
        self.jobs.progressed.connect(lambda f: self.progress.setValue(int(1000 * min(max(f, 0.0), 1.0))))
        app = QtWidgets.QApplication.instance()
        if app is not None and not hasattr(BaseTab, "shutdown"):   # older BaseTab without the quit hook
            app.aboutToQuit.connect(self.shutdown)
        self._populate_robots()
        self.on_robot_changed(self.ctx.robot)

    # ======================================================================== plot items
    def _init_items(self):
        # workspace
        self.reach_curve = self.ws.plot([], [], pen=pg.mkPen(theme.MUTED, width=1, style=QtCore.Qt.DashLine))
        self.ee_trace = self.ws.plot([], [], pen=pg.mkPen(C_PATH, width=1.5))
        self.ghost_s = self.ws.plot([], [], pen=pg.mkPen(_qcol(C_START, 150), width=5), symbol="o", symbolSize=7,
                                    symbolBrush=_qcol(C_START, 150), symbolPen=None)
        self.ghost_g = self.ws.plot([], [], pen=pg.mkPen(_qcol(C_GOAL, 150), width=5), symbol="o", symbolSize=7,
                                    symbolBrush=_qcol(C_GOAL, 150), symbolPen=None)
        self.ghost_h = self.ws.plot([], [], pen=pg.mkPen((255, 255, 255, 120), width=4, style=QtCore.Qt.DashLine))
        self.arm_curve = self.ws.plot([], [], pen=pg.mkPen(theme.LINK, width=9), symbol="o", symbolSize=11,
                                      symbolBrush=theme.JOINT, symbolPen=None)
        self.arm_curve.setZValue(20)
        self.base_mark = self.ws.plot([0], [0], pen=None, symbol="s", symbolSize=14, symbolBrush=theme.BASE)
        # C-space
        self.img = pg.ImageItem()
        self.img.setRect(QtCore.QRectF(-180, -180, 360, 360))
        self.cs.addItem(self.img)
        self.cs_edges = self.cs.plot([], [], pen=pg.mkPen((120, 170, 230, 110), width=1), connect="pairs")
        self.cs_nodes = pg.ScatterPlotItem(size=4, pen=None)
        self.cs.addItem(self.cs_nodes)
        self.cs_raw = []
        self.cs_path = []
        self.cs_start = self.cs.plot([], [], pen=None, symbol="o", symbolSize=13, symbolBrush=C_START,
                                     symbolPen=pg.mkPen("w", width=1.5))
        self.cs_goal = self.cs.plot([], [], pen=None, symbol="star", symbolSize=18, symbolBrush=C_GOAL,
                                    symbolPen=pg.mkPen("w", width=1.5))
        self.cs_cur = self.cs.plot([], [], pen=None, symbol="o", symbolSize=10, symbolBrush="w")
        for it in (self.cs_start, self.cs_goal, self.cs_cur):
            it.setZValue(30)
        self.cs.scene().sigMouseMoved.connect(self._cs_hover)
        self.cs.scene().sigMouseClicked.connect(self._cs_click)

    # ======================================================================== robot / scene
    def _populate_robots(self):
        names = P.planning_robot_names()
        cur = self.robot_box.currentText()
        self.robot_box.blockSignals(True)
        self.robot_box.clear()
        self.robot_box.addItems(names)
        if cur in names:
            self.robot_box.setCurrentText(cur)
        self.robot_box.blockSignals(False)

    def on_robot_changed(self, robot):
        """Global robot box: use it here if it is a planar revolute 2R/3R arm, else keep the planner's robot."""
        names = [self.robot_box.itemText(i) for i in range(self.robot_box.count())]
        if robot.name not in names:
            self._populate_robots()
            names = [self.robot_box.itemText(i) for i in range(self.robot_box.count())]
        if robot.name in names:
            self.robot_box.setCurrentText(robot.name)
            self._robot_selected(robot.name)
        elif self.arm is None:
            self._robot_selected(self.robot_box.currentText() or "Planar 2R arm")

    def _robot_selected(self, name):
        if not name:
            return
        self._stop_anim()
        self._invalidate_jobs()
        robot = get_robot(name)
        self.robot = robot
        self.arm = P.PlanarArm.from_robot(robot)
        glob = self.ctx.robot.name
        note = f"{robot.n}R, links {', '.join(f'{x:.2f}' for x in self.arm.link_lengths)} m, r = {self.arm.radius * 1e3:.0f} mm"
        if glob != name and not P.is_planar_revolute(self.ctx.robot):
            note += f" — the selected robot ({glob}) is not a planar arm, so the planner uses this one."
        if robot.n == 3:
            note += "  C-space picture = slice at the current θ₃ (path drawn projected)."
        self.robot_note.setText(note)
        self.start_sl.set_robot(robot)
        self.goal_sl.set_robot(robot)
        R = self.arm.reach
        a = np.linspace(0, 2 * np.pi, 160)
        self.reach_curve.setData(self.arm.base[0] + R * np.cos(a), self.arm.base[1] + R * np.sin(a))
        self.ws.setXRange(-1.15 * R, 1.15 * R, padding=0)
        self.ws.setYRange(-1.15 * R, 1.15 * R, padding=0)
        # 3R: k^3 cells with 26 certified edges each -- k = 40 already takes several seconds
        self.k_spin.setMaximum(160 if robot.n == 2 else 40)
        self.k_spin.setValue(64 if robot.n == 2 else 24)
        self.k_spin.setToolTip(f"grid points per joint: {'k²' if robot.n == 2 else 'k³'} in all (MR 10.4)")
        self.load_scene()

    def load_scene(self):
        if self.arm is None:
            return
        self._stop_anim()
        obs, s, g = P.scene(self.scene_box.currentText(), self.arm)
        for r in self.rois:
            self.ws.removeItem(r)
        self.rois = []
        for ob in obs:
            self._make_roi(ob)
        self.start_sl.set_values(s)
        self.goal_sl.set_values(g)
        self._world_changed()

    def _make_roi(self, ob):
        p_obs = pg.mkPen(_qcol(C_OBS, 230), width=2)
        hp = pg.mkPen("w", width=1)
        if isinstance(ob, P.Circle):
            roi = pg.CircleROI(tuple(ob.c - ob.r), (2 * ob.r, 2 * ob.r), pen=p_obs, handlePen=hp, movable=True)
        else:
            roi = pg.PolyLineROI([tuple(p) for p in ob.pts], closed=True, pen=p_obs, handlePen=hp, movable=True)
        roi.setZValue(5)
        roi.sigRegionChanged.connect(lambda *_: self._cs_timer.start(40))
        roi.sigRegionChangeFinished.connect(lambda *_: self._world_changed())
        self.ws.addItem(roi)
        self.rois.append(roi)
        # translucent fill
        return roi

    def _obstacles(self) -> list:
        out = []
        for roi in self.rois:
            if isinstance(roi, pg.CircleROI):
                size = roi.size()
                c = roi.pos() + size / 2
                out.append(P.Circle([c.x(), c.y()], abs(size.x()) / 2))
            else:
                pts = [roi.mapToParent(h.pos()) for h in roi.getHandles()]
                if len(pts) >= 3:
                    out.append(P.Polygon([[p.x(), p.y()] for p in pts]))
        return out

    def _add_obstacle(self, kind):
        R = self.arm.reach
        rng = np.random.default_rng(len(self.rois) + 11)
        c = rng.uniform(-0.6, 0.6, 2) * R
        if kind == "circle":
            self._make_roi(P.Circle(c, 0.09 * R))
        else:
            h = 0.08 * R
            self._make_roi(P.box(c[0] - h, c[1] - h, c[0] + h, c[1] + h))
        self._world_changed()

    def _remove_last(self):
        if self.rois:
            self.ws.removeItem(self.rois.pop())
            self._world_changed()

    def _world_changed(self):
        """Obstacles / topology changed: rebuild the world, the C-space picture, drop the old plan."""
        if self.arm is None:
            return
        # a plan for the old scene is stale -- also one whose (queued) result has not been delivered yet
        self._invalidate_jobs()
        self.world = P.World(self.arm, self._obstacles(), torus=self.torus_cb.isChecked())
        self._grid_cache = None
        self._clear_result("Scene changed — press Plan." if self.result is not None else "")
        self._update_cspace()
        self._endpoints_changed()

    # ======================================================================== C-space picture
    def _slice_q(self):
        th = self.start_sl.values()
        if getattr(self, "_anim_q", None) is not None:
            th = self._anim_q
        return th

    def _update_cspace(self, fast: bool = False):
        if self.arm is None:
            return
        w = self.world if not fast else P.World(self.arm, self._obstacles(), torus=self.torus_cb.isChecked())
        k = 90 if fast else 200
        _, obst, outside = P.cspace_image(w, k, self._slice_q())
        rgb = np.empty((k, k, 3), np.ubyte)
        rgb[:] = C_FREE
        rgb[outside] = C_LIM
        rgb[obst] = C_OBS
        rgb[obst & outside] = (130, 70, 76)
        self.img.setImage(rgb, autoLevels=False)
        self.img.setRect(QtCore.QRectF(-180, -180, 360, 360))
        frac = obst[~outside].mean() if (~outside).any() else 0.0
        txt = f"C_obs ≈ {100 * frac:.0f}% of the joint range"
        if self.arm.n == 3:
            txt += f"   (slice θ₃ = {np.degrees(self._slice_q()[2]):.0f}°)"
        self.cs.setTitle("C-space (θ₁, θ₂) — red: C-obstacles" + ("" if not self.world.space.wrap[:2].any()
                         else ", edges glued (torus)") + ("; grey: beyond limits" if outside.any() else "")
                         + f"<br><span style='color:{theme.MUTED}'>{txt}</span>", color=theme.TEXT, size="10pt")

    # ======================================================================== start / goal
    def _endpoints_changed(self):
        if self.world is None:
            return
        s, g = self.start_sl.values(), self.goal_sl.values()
        if len(s) != self.arm.n or len(g) != self.arm.n:
            return
        self._draw_arm(self.ghost_s, s)
        self._draw_arm(self.ghost_g, g)
        if getattr(self, "_anim_q", None) is None:
            self._draw_arm(self.arm_curve, s)
        self.cs_start.setData([np.degrees(s[0])], [np.degrees(s[1])])
        self.cs_goal.setData([np.degrees(g[0])], [np.degrees(g[1])])
        fs, fg = self.world.free(s), self.world.free(g)
        msg = []
        if not fs:
            msg.append("⚠ start is in collision / outside the limits")
        if not fg:
            msg.append("⚠ goal is in collision / outside the limits")
        self.ep_lbl.setText("\n".join(msg) if msg else f"straight-line C-space distance {self.world.space.dist(s, g):.3f} rad"
                            + ("" if self.world.edge_free(s, g) else "  (blocked)"))
        self.ep_lbl.setStyleSheet(f"color:{theme.WARN}" if msg else "")
        if self.arm.n == 3 and getattr(self, "_last_slice", None) != round(float(s[2]), 4):
            self._last_slice = round(float(s[2]), 4)
            self._update_cspace()

    def _swap(self):
        s, g = self.start_sl.values(), self.goal_sl.values()
        self.start_sl.set_values(g)
        self.goal_sl.set_values(s)
        self._endpoints_changed()

    def _random_goal(self):
        rng = np.random.default_rng()
        for _ in range(200):
            q = self.world.space.sample(rng, 64)
            ok = ~self.world.collides(q)
            if ok.any():
                self.goal_sl.set_values(q[np.flatnonzero(ok)[0]])
                break
        self._endpoints_changed()

    def _cs_point(self, pos):
        if not self.cs.sceneBoundingRect().contains(pos):
            return None
        p = self.cs.getPlotItem().vb.mapSceneToView(pos)
        if -180 <= p.x() <= 180 and -180 <= p.y() <= 180:
            return np.radians([p.x(), p.y()])
        return None

    def _cs_hover(self, pos):
        q2 = self._cs_point(pos)
        if q2 is None or not self.cb_hover.isChecked() or self.arm is None:
            self.ghost_h.setData([], [])
            return
        q = np.r_[q2, self._slice_q()[2:]]
        self._draw_arm(self.ghost_h, q)
        free = self.world.free(q)
        self.ghost_h.setPen(pg.mkPen((255, 255, 255, 140) if free else _qcol(C_OBS, 200), width=4,
                                     style=QtCore.Qt.DashLine))

    def _cs_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        q2 = self._cs_point(ev.scenePos())
        if q2 is None:
            return
        target = self.start_sl if ev.modifiers() & QtCore.Qt.ShiftModifier else self.goal_sl
        q = target.values()
        q[:2] = q2
        target.set_values(q)
        self._endpoints_changed()

    def _draw_arm(self, item, q):
        Pts = self.arm.points(q)[0]
        item.setData(Pts[:, 0], Pts[:, 1])

    # ======================================================================== planning
    def _planner_changed(self, name):
        show = {"A* (grid)": {"k", "conn"}, "Dijkstra (grid)": {"k", "conn"}, "PRM": {"N", "knn", "seed"},
                "RRT": {"iter", "step", "bias", "seed"}, "RRT*": {"iter", "step", "bias", "rad", "seed"}}[name]
        for key, (lab, w) in self._rows.items():
            lab.setVisible(key in show)
            w.setVisible(key in show)
        if name == "RRT*" and self.iter_spin.value() > 2500:
            self.iter_spin.setValue(2000)

    def params(self) -> dict:
        return dict(grid_k=self.k_spin.value(), connect="4" if self.connect_box.currentIndex() == 1 else "8",
                    n_samples=self.n_spin.value(), k_nn=self.knn_spin.value(), max_iter=self.iter_spin.value(),
                    step=np.radians(self.step_spin.value()), goal_bias=self.bias_spin.value(),
                    radius=np.radians(self.rad_spin.value()), seed=self.seed_spin.value(),
                    smooth_iters=self.smooth_spin.value() if self.smooth_cb.isChecked() else 0)

    def _invalidate_jobs(self):
        """Robot or scene changed: cancel a running plan and make every pending result stale (non-blocking)."""
        self._job += 1
        if self.jobs.is_busy():
            self.jobs.cancel()

    def _world_now(self) -> P.World:
        """The world as currently shown (obstacle ROIs may have moved since self.world was built)."""
        return P.World(self.arm, self._obstacles(), torus=self.torus_cb.isChecked())

    def plan(self):
        if self.world is None:
            return
        self._stop_anim()
        planner = self.planner_box.currentText()
        world = self._world_now()
        s, g = self.start_sl.values(), self.goal_sl.values()
        kw = self.params()
        key = (world.fingerprint(), kw["grid_k"])          # a cached grid is only valid for the same world
        cache = self._grid_cache
        if cache is not None and cache[0] == key:
            kw["grid"] = cache[1]

        def job(progress, cancelled):
            return P.plan(world, s, g, planner, progress=progress, cancelled=cancelled, **kw)
        self._job += 1
        gen = self._job
        self.progress.setValue(0)
        self.progress.setFormat(f"{planner}  %p%")
        self.res_lbl.setText(f"planning with {planner} …")
        self.res_lbl.setStyleSheet("")
        self.jobs.run(job, on_done=lambda r, gen=gen, key=key: self._plan_done(gen, r, key),
                      on_error=lambda e, gen=gen: self._plan_failed(gen, e), label=planner)

    def _busy_changed(self, on):
        self.busy.setVisible(on)
        self.plan_btn.setEnabled(not on)

    def wait_job(self, timeout_s: float = 60.0):
        """Block (processing events) until planning finished -- used by tests / scripts."""
        self.jobs.wait(timeout_s)

    def _cancel(self):
        if self.jobs.is_busy():
            self.jobs.cancel()
            self.res_lbl.setText("cancelled")
            self.res_lbl.setStyleSheet("")

    def _plan_failed(self, gen, err):
        if gen != self._job:
            return
        self.res_lbl.setText("planner error:\n" + err.strip().splitlines()[-1])
        print(err)

    def _plan_done(self, gen, res: P.PlanResult, key=None):
        if gen != self._job or not self._result_matches_arm(res):
            return
        if "grid" in res.extra and key is not None and res.extra["grid"].fingerprint == key[0]:
            self._grid_cache = (key, res.extra["grid"])
        self.result = res
        self._add_history(res)
        if res.success:
            txt = (f"✓ {res.planner}: path length {res.cost:.3f} rad"
                   + (f" (before smoothing {res.raw_cost:.3f})" if res.raw_cost > res.cost + 1e-9 else "")
                   + f"\n{res.message}\n{res.n_checks:,} configurations collision-checked, {res.seconds:.2f} s")
            self.res_lbl.setStyleSheet(f"color:{theme.GOOD}")
        else:
            txt = f"✗ {res.planner}: {res.message}\n{res.n_checks:,} collision checks, {res.seconds:.2f} s"
            self.res_lbl.setStyleSheet(f"color:{theme.BAD}")
        self.res_lbl.setText(txt)
        self._redraw_result()
        self._retime()

    def _add_history(self, r: P.PlanResult):
        row = self.hist.rowCount()
        self.hist.insertRow(row)
        vals = [r.planner, "✓" if r.success else "✗", f"{r.cost:.3f}" if r.success else "—", f"{r.n_nodes}",
                f"{r.n_checks:,}", f"{r.seconds:.2f}"]
        for j, v in enumerate(vals):
            it = QtWidgets.QTableWidgetItem(v)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            self.hist.setItem(row, j, it)
        self.hist.scrollToBottom()

    def _clear_result(self, msg=""):
        self._stop_anim()
        self.result = None
        self.traj = None
        self.anim_btn.setEnabled(False)
        self.cs_edges.setData([], [])
        self.cs_nodes.setData([], [])
        for it in self.cs_path + self.cs_raw:
            self.cs.removeItem(it)
        self.cs_path, self.cs_raw = [], []
        self.ee_trace.setData([], [])
        for p in (self.p_th, self.p_dth):
            p.clear()
        if msg:
            self.res_lbl.setText(msg)
            self.res_lbl.setStyleSheet("")

    def _result_matches_arm(self, r) -> bool:
        """A plan computed for another robot or another scene (obstacles, topology) must not be drawn: its
        world fingerprint must equal the fingerprint of the world shown now."""
        if r is None or getattr(r, "path", None) is None or self.world is None:
            return r is not None
        if np.asarray(r.path).shape[-1] != self.world.space.n:
            return False
        fp = getattr(r, "fingerprint", None)
        return fp is not None and fp == self._world_now().fingerprint()

    def _redraw_result(self, *_):
        r = self.result
        if r is not None and not self._result_matches_arm(r):
            self.result = r = None
        for it in self.cs_path + self.cs_raw:
            self.cs.removeItem(it)
        self.cs_path, self.cs_raw = [], []
        self.cs_edges.setData([], [])
        self.cs_nodes.setData([], [])
        if r is None:
            return
        wrap = self.world.space.wrap[:2]
        if self.cb_explored.isChecked():
            if r.edges is not None and len(r.edges):
                E = np.degrees(r.edges[:, :, :2])
                segs = [c for c in _wrap_copies(E.reshape(-1, 2), wrap)]
                xy = np.vstack(segs)
                self.cs_edges.setData(xy[:, 0], xy[:, 1])
            if r.expanded is not None and len(r.expanded):
                # order of expansion (grid / PRM search) or of insertion (RRT tree): early = blue, late = yellow
                X = np.degrees(self.world.space.canon(r.expanded)[:, :2])
                cols = pg.colormap.get("viridis").map(np.linspace(0.05, 1, len(X)), mode="byte")
                brushes = [pg.mkBrush(*c[:3], 200) for c in cols]
                self.cs_nodes.setData(X[:, 0], X[:, 1], brush=brushes, size=4 if len(X) > 600 else 6)
        if not r.success:
            return
        raw = r.extra.get("raw_path")
        if self.cb_raw.isChecked() and raw is not None and len(raw) != len(r.path):
            for c in _wrap_copies(np.degrees(raw[:, :2]), wrap):
                it = self.cs.plot(c[:, 0], c[:, 1], pen=pg.mkPen(C_RAW, width=1.5, style=QtCore.Qt.DashLine))
                self.cs_raw.append(it)
        for c in _wrap_copies(np.degrees(r.path[:, :2]), wrap):
            it = self.cs.plot(c[:, 0], c[:, 1], pen=pg.mkPen(C_PATH, width=3))
            it.setZValue(25)
            self.cs_path.append(it)

    # ======================================================================== trajectory
    def _retime(self):
        r = self.result
        if r is None or not r.success or not self._result_matches_arm(r):
            return
        t, Q, dQ, _ = P.time_scale(r.path, self.Tf_spin.value(), self.scale_box.currentText(), 0.02,
                                   self.rest_cb.isChecked())
        self.traj = (t, Q, dQ)
        for p in (self.p_th, self.p_dth):
            p.clear()
        for i in range(Q.shape[1]):
            self.p_th.plot(t, np.degrees(Q[:, i]), pen=pen(i), name=f"θ{'₁₂₃'[i]}")
            self.p_dth.plot(t, np.degrees(dQ[:, i]), pen=pen(i), name=f"θ̇{'₁₂₃'[i]}")
        self.t_line = [p.addLine(x=0, pen=pg.mkPen(theme.MUTED, width=1)) for p in (self.p_th, self.p_dth)]
        E = self.arm.points(Q)[:, -1]
        self.ee_trace.setData(E[:, 0], E[:, 1])
        self.anim_btn.setEnabled(True)
        self._show_q(Q[0])

    def toggle_anim(self, on):
        self.anim_btn.setText("■ Stop" if on else "▶ Animate along path")
        if on and self.traj is not None:
            self.t0 = time.time()
            self.timer.start(30)
        else:
            self.timer.stop()
            self._anim_q = None
            self.cs_cur.setData([], [])

    def _anim_step(self):
        t, Q, _ = self.traj
        T = t[-1]
        tt = (time.time() - self.t0) % (T + 0.8)
        k = min(int(np.searchsorted(t, min(tt, T))), len(t) - 1)
        self._show_q(Q[k], t[k])

    def _show_q(self, q, tk=None):
        self._anim_q = q if tk is not None else None
        self._draw_arm(self.arm_curve, q)
        c = np.degrees(self.world.space.canon(q))
        self.cs_cur.setData([c[0]], [c[1]])
        if tk is not None:
            for ln in getattr(self, "t_line", []):
                ln.setValue(tk)
            if self.arm.n == 3 and int(tk / 0.02) % 4 == 0:
                self._update_cspace(fast=True)

    def _stop_anim(self):
        if hasattr(self, "anim_btn") and self.anim_btn.isChecked():
            self.anim_btn.setChecked(False)
        self._anim_q = None

    # ======================================================================== hooks
    def on_show(self):
        if not getattr(self, "_shown", False):
            self._shown = True
            if self.result is None:
                QtCore.QTimer.singleShot(100, self.plan)   # A* on the default scene: a few ms
        names = P.planning_robot_names()
        if names != [self.robot_box.itemText(i) for i in range(self.robot_box.count())]:
            self._populate_robots()

    def on_hide(self):
        self._stop_anim()

    def shutdown(self):
        """Application quit: cancel the plan, wait at most half a second for the worker."""
        self._stop_anim()
        self._job += 1
        self.jobs.shutdown(500)
        parent = getattr(super(), "shutdown", None)
        if parent is not None:
            parent()
