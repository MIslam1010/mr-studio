"""Tab 3 -- inverse kinematics (MR Ch. 6): Newton-Raphson (IKinBody / IKinSpace), damped least squares,
multiple solutions, analytic planar IK, redundancy (null-space self-motion) and real-time IK."""
from __future__ import annotations

import time
import traceback

import modern_robotics as mr
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...references import further_reading
from ... import kinematics as kin
from ...robots import trans
from .. import theme
from ..base import BaseTab
from ..viewer import RobotGraphic, SceneView
from ..widgets import (JointSliders, MatrixView, SliderRow, button, checkbox, combo, group, hbox, pen, plot,
                       spin, value_label)

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
TOLS = ["1e-2", "1e-3", "1e-4", "1e-6", "1e-8", "1e-10", "1e-12"]


def _sub(i: int) -> str:
    return str(i).translate(_SUB)


def _deg(th, d=1) -> str:
    return "(" + ", ".join(f"{np.degrees(x):+.{d}f}" for x in th) + ")"


def _rgba(hexcol: str, a: float):
    c = QtGui.QColor(hexcol)
    return (c.redF(), c.greenF(), c.blueF(), a)


def _rotz(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _task(robot) -> str:
    """IK task of a robot: 'pose' (6-D), 'planar' (x, y, φ), 'position' (3-D point) or 'point2d' (x, y)."""
    if robot.planar:
        return "planar" if robot.n >= 3 else "point2d"
    return "pose" if robot.n >= 6 else "position"


def _redundancy(robot):
    """Task used for the self-motion demo, or None if the arm is not redundant for any sensible task."""
    if robot.planar:
        return "point2d" if robot.n > 2 else None
    if robot.n > 6:
        return "pose"
    return "position" if 3 < robot.n < 6 else None


def _textbook_planar(robot) -> bool:
    """Planar 2R/3R arm with all joint axes along z through points of the x axis (closed form applies)."""
    S = robot.Slist
    return bool(robot.planar and robot.n in (2, 3) and np.allclose(S[:3], np.array([[0], [0], [1]]))
                and np.allclose(S[3], 0) and np.allclose(S[5], 0) and np.allclose(robot.M[:3, :3], np.eye(3))
                and abs(robot.M[1, 3]) < 1e-12 and abs(robot.M[2, 3]) < 1e-12)


_TASK_DIM = {"pose": 6, "planar": 3, "position": 3, "point2d": 2}


def _task_error(robot, th, T_d, task):
    """Error vector e and Jacobian J of the task (Newton step: Δθ = J⁺ e)."""
    if task in ("pose", "planar"):
        T = kin.fk_body(robot, th)
        Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_d))
        return Vb, kin.jacobian_body(robot, th)
    T = kin.fk_space(robot, th)
    e = T_d[:3, 3] - T[:3, 3]
    J = kin.point_jacobian(robot, th)
    if task == "point2d":
        return e[:2], J[:2]
    return e, J


def _ghost(view, robot, key, color, width=2.5, geom=None, theta=None):
    """Translucent stick figure (same look as viewer.GhostRobot) that can be re-coloured without recomputing FK."""
    g = geom if geom is not None else robot.geometry(np.asarray(theta, float))
    view.set_polyline(key + ".line", np.vstack(g["polys"]), color, width)
    view.set_points(key + ".pts", g["joint_pos"], color, 7)
    view.set_points(key + ".ee", g["T_ee"][:3, 3][None], color, 11)
    return g


def _fit_table(mv: MatrixView):
    mv.setMinimumHeight(60)
    mv.setMaximumHeight(16777215)
    mv.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    mv.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)


THEORY = r"""
<h2>Inverse kinematics</h2>
<span class="ref">Modern Robotics, Ch. 6: §6.1 analytic IK, §6.2 numerical IK, §6.3 inverse velocity kinematics</span>
<p>Given a desired end-effector configuration $T_{sd}$, find $\theta$ with $T(\theta)=T_{sd}$.  Unlike forward
kinematics there may be <b>no</b> solution, a <b>finite number</b> (up to 8 for a 6R arm) or <b>infinitely many</b>
(redundant arms).</p>
<h3>Analytic IK: planar 2R (§6.1)</h3>
<p>Law of cosines for the elbow, then the shoulder:</p>
$$\cos\theta_2=\frac{x^{2}+y^{2}-L_1^{2}-L_2^{2}}{2L_1L_2}$$
$$\theta_1=\mathrm{atan2}(y,x)-\mathrm{atan2}(L_2\sin\theta_2,\ L_1+L_2\cos\theta_2)$$
<p>The two signs of $\theta_2$ are the elbow-down / elbow-up branches.  For the 3R arm with orientation
$\phi$, apply this to the wrist point $(x-L_3\cos\phi,\ y-L_3\sin\phi)$ and set $\theta_3=\phi-\theta_1-\theta_2$.
A PUMA-type 6R arm has up to $2\times2\times2=8$ solutions (shoulder left/right, elbow up/down, wrist flip).</p>
<h3>Newton–Raphson (§6.2.1)</h3>
<p>To solve $x_d-f(\theta)=0$, expand $f$ to first order about the current guess $\theta^{i}$:</p>
$$x_d=f(\theta^{i})+J(\theta^{i})\,\Delta\theta\ \ \Rightarrow\ \ \Delta\theta=J^{\dagger}(\theta^{i})\left(x_d-f(\theta^{i})\right)\quad(6.4),\ (6.6)$$
<p>Near a solution the error is squared at every step: <b>quadratic convergence</b> (the number of correct
digits doubles — the convergence curves bend straight down on the log plot).</p>
<h3>Algorithm IKinBody (§6.2.2)</h3>
<p>For poses the error is the body twist that moves {b} to {d} in unit time:</p>
$$[\mathcal{V}_b]=\log\left(T_{sb}^{-1}(\theta^{i})\,T_{sd}\right),\qquad \theta^{i+1}=\theta^{i}+J_b^{\dagger}(\theta^{i})\,\mathcal{V}_b$$
<p>until $\|\omega_b\|<\epsilon_\omega$ and $\|v_b\|<\epsilon_v$.  <b>IKinSpace</b> uses
$\mathcal{V}_s=\left[\mathrm{Ad}_{T_{sb}}\right]\mathcal{V}_b$ with $J_s$; since
$J_s=\left[\mathrm{Ad}_{T_{sb}}\right]J_b$ it produces the same iterates.  The app re-runs the official
<b>mr.IKinBody / mr.IKinSpace</b> on the same data and prints $\|\Delta\theta\|$.</p>
<h3>Pseudo-inverse (§6.2.1)</h3>
$$J^{\dagger}=J^{T}\left(JJ^{T}\right)^{-1}\ (n>m),\qquad J^{\dagger}=\left(J^{T}J\right)^{-1}J^{T}\ (n<m)$$
<p>For $n>m$ it returns the minimum-norm $\Delta\theta$ among all that achieve the task.</p>
<h3>Damped least squares</h3>
$$\Delta\theta=J^{T}\left(JJ^{T}+\lambda^{2}I\right)^{-1}\mathcal{V}$$
<p>Each singular direction is scaled by $\sigma_i/(\sigma_i^{2}+\lambda^{2})$ instead of $1/\sigma_i$: steps stay
bounded near singularities, at the price of linear convergence (an extension beyond the MR library).</p>
<h3>Redundancy and self-motion (§6.3)</h3>
<p>For $n>m$ the general solution of $\mathcal{V}=J\dot{\theta}$ is</p>
$$\dot{\theta}=J^{\dagger}\mathcal{V}+\left(I-J^{\dagger}J\right)z$$
<p>The second term lies in the null space of $J$: joints move, the end-effector does not.  The 7-DOF Panda has a
1-D null space for the full pose (elbow swing); the planar 3R arm has one for a position-only task.  A finite step
along $(I-J^{\dagger}J)z$ drifts at second order, so each step is followed by a Newton correction — the drift
readout shows the residual.</p>
<h3>Real-time IK</h3>
<p>"Follow a circle" re-solves IK every 30 ms, warm-started from the previous solution: thanks to quadratic
convergence, one or two iterations per frame suffice.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 3 §3.2–3.3, pp. 99–108', 'Paden–Kahan subproblems and closed-form IK'),
    ('gotor', '§4.2–4.4, pp. 53–114', 'numeric vs. geometric IK; Paden–Kahan and Pardos-Gotor subproblems'),
    ('corke', '§7.3, pp. 146–152; §8.4, pp. 187–188', 'closed-form and numerical IK; Jacobian-based iteration'),
])



class _Worker(QtCore.QThread):
    """Runs fn(worker) in a background thread."""
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancelled = False

    def run(self):
        try:
            out = self.fn(self)
            if not self.cancelled:
                self.done.emit(out)
        except Exception:
            self.failed.emit(traceback.format_exc())


class IKTab(BaseTab):
    title = "③ Inverse Kinematics"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        self.graphic = None
        self.theta = np.zeros(0)
        self.T_target = np.eye(4)
        self.t_rows: dict[str, SliderRow] = {}
        self._workers: list[_Worker] = []
        self._all_worker = None
        self._solutions: list[np.ndarray] = []
        self._ghosts: list = []
        self._anim = None
        self._last = None

        # ---------------------------------------------------------------- target
        self.target_box = QtWidgets.QWidget()
        self.target_lay = QtWidgets.QVBoxLayout(self.target_box)
        self.target_lay.setContentsMargins(0, 0, 0, 0)
        self.target_lay.setSpacing(2)
        self.target_lbl = value_label()
        tbtns = hbox(button("Target = current EE", self.target_from_ee),
                     button("Random reachable", self.random_target,
                            tooltip="T_sd = FK of a random joint vector, so a solution exists"))
        self.cb_live = checkbox("Live IK while dragging the target (warm start)", None, False)

        # ---------------------------------------------------------------- configuration
        self.sliders = JointSliders()
        self.sliders.changed.connect(self._sliders_moved)
        cbtns = hbox(button("Home", lambda: self._set_theta(self.ctx.robot.home)),
                     button("Random", lambda: self._set_theta(self.ctx.robot.random_config())),
                     button("Clear ghosts", self.clear_ghosts))

        # ---------------------------------------------------------------- solver
        self.method = combo(["Newton–Raphson · body frame (IKinBody)", "Newton–Raphson · space frame (IKinSpace)",
                             "Damped least squares (DLS)"], self._method_changed)
        self.lam = spin(0.001, 2.0, 0.05, 3, 0.01)
        self.guess = combo(["current θ (sliders)", "home θ", "random θ"])
        self.eomg = combo(TOLS)
        self.eomg.setCurrentText("1e-6")
        self.ev = combo(TOLS)
        self.ev.setCurrentText("1e-6")
        for cb in (self.method, self.guess, self.eomg, self.ev):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
        self.maxit = QtWidgets.QSpinBox()
        self.maxit.setRange(1, 500)
        self.maxit.setValue(50)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Method", self.method)
        form.addRow("DLS λ", self.lam)
        form.addRow("Initial guess θ⁰", self.guess)
        form.addRow("ε_ω  [rad]", self.eomg)
        form.addRow("ε_v  [m]", self.ev)
        form.addRow("Max iterations", self.maxit)
        self.solve_btn = button("▶  Solve IK", self.solve, primary=True)
        self.result_lbl = value_label()
        self.mr_lbl = value_label()

        # ---------------------------------------------------------------- multiple solutions
        self.all_btn = button("Find all solutions", self.find_all,
                              tooltip="Newton–Raphson from 60 random seeds, distinct wrapped solutions kept")
        self.sol_list = QtWidgets.QListWidget()
        self.sol_list.setMaximumHeight(170)
        self.sol_list.itemClicked.connect(self._solution_clicked)
        self.all_lbl = value_label()
        self.analytic_lbl = value_label()

        # ---------------------------------------------------------------- redundancy
        self.self_btn = button("▶ Self-motion (null space)", self.toggle_self_motion, checkable=True)
        self.drift_lbl = value_label()

        # ---------------------------------------------------------------- real-time
        self.circle_btn = button("▶ Follow a circle", self.toggle_circle, checkable=True)
        self.radius = spin(0.02, 0.6, 0.12, 2, 0.01, " m")
        self.period = spin(1.0, 30.0, 6.0, 1, 0.5, " s")
        self.cb_trace = checkbox("Trace the end-effector path", self._trace_toggled, True)
        self.rt_lbl = value_label()

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Target pose T_sd  (drawn as {d})", self.target_box, tbtns, self.cb_live,
                           self.target_lbl))
        cl.addWidget(group("Solver  (MR §6.2)", form, self.solve_btn, self.result_lbl, self.mr_lbl))
        cl.addWidget(group("Multiple solutions  (MR §6.1)", self.all_btn, self.all_lbl, self.sol_list,
                           self.analytic_lbl))
        cl.addWidget(group("Redundancy: null-space self-motion  (MR §6.3)", self.self_btn, self.drift_lbl))
        cl.addWidget(group("Real-time IK", hbox(self.circle_btn, QtWidgets.QLabel("r"), self.radius),
                           hbox(QtWidgets.QLabel("period"), self.period, self.cb_trace), self.rt_lbl))
        cl.addWidget(group("Robot configuration θ  (= current guess)", self.sliders, cbtns))
        cl.addStretch(1)

        # ---------------------------------------------------------------- bottom
        self.conv_plot = plot("Newton–Raphson convergence", "error norm (log)", "iteration i", legend=True)
        self.conv_plot.setLogMode(y=True)
        self.order_lbl = QtWidgets.QLabel()
        self.order_lbl.setObjectName("muted")
        cw = QtWidgets.QWidget()
        cwl = QtWidgets.QVBoxLayout(cw)
        cwl.setContentsMargins(0, 0, 0, 0)
        cwl.addWidget(self.conv_plot, 1)
        cwl.addWidget(self.order_lbl)
        self.iter_table = MatrixView(1, 4, 3)
        self.iter_table.setStyleSheet("QTableWidget { font-size: 11px; }")
        tw = QtWidgets.QWidget()
        twl = QtWidgets.QVBoxLayout(tw)
        twl.setContentsMargins(0, 0, 0, 0)
        cap = QtWidgets.QLabel("Iterates θⁱ (cf. MR Example 6.1)")
        cap.setObjectName("muted")
        twl.addWidget(cap)
        twl.addWidget(self.iter_table, 1)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(cw)
        bottom.addWidget(tw)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        bottom.setSizes([430, 420])
        lg = self.conv_plot.plotItem.legend
        if lg is not None:
            lg.anchor((1, 0), (1, 0), offset=(-8, 8))

        self.assemble(controls, self.view, bottom, THEORY, controls_width=370, bottom_ratio=(5, 3))
        for sa in self.findChildren(QtWidgets.QScrollArea):
            if sa.widget() is controls:
                sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._anim_step)
        self.self_timer = QtCore.QTimer(self)
        self.self_timer.timeout.connect(self._self_step)
        self.circle_timer = QtCore.QTimer(self)
        self.circle_timer.timeout.connect(self._circle_step)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown_workers)
        self._method_changed()
        self.on_robot_changed(self.ctx.robot)

    # ================================================================== robot
    def on_robot_changed(self, robot):
        self._stop_modes()
        self._cancel_all()
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("ik.")
        self._ghosts = []
        self.graphic = RobotGraphic(self.view, robot, "robot")
        self.sliders.set_robot(robot)
        self.theta = np.array(robot.home, float)
        self.view.frame_scene((1.4 if robot.planar else 1.15) * robot.reach, planar=robot.planar,
                              center=(0, 0, 0.3 * robot.reach * (not robot.planar)))
        self.view.set_triad("ik.s", np.eye(4), 0.2 * robot.reach, "{s}")
        self._build_target_rows(robot)
        # a reachable default target, different from home
        lo = np.maximum(robot.joint_limits[:, 0], -np.pi)
        hi = np.minimum(robot.joint_limits[:, 1], np.pi)
        th_t = np.clip(robot.home + 0.35 * np.where(np.arange(robot.n) % 2, -1.0, 1.0), lo, hi)
        self._set_target(kin.fk_space(robot, th_t))
        # tables / labels
        cols = ["i"] + [f"θ{_sub(i + 1)} [{robot.display_unit(i)}]" for i in range(robot.n)] + ["‖ω_b‖", "‖v_b‖"]
        self.iter_table.set_matrix(np.full((1, len(cols)), "", dtype=object))
        self.iter_table.set_labels(cols=cols)
        _fit_table(self.iter_table)
        self.conv_plot.clear()
        self.order_lbl.setText("")
        self.result_lbl.setText("")
        self.mr_lbl.setText("")
        self.sol_list.clear()
        self._solutions = []
        self.all_lbl.setText("")
        self.analytic_lbl.setText("")
        self.rt_lbl.setText("")
        red = _redundancy(robot)
        self.self_btn.setEnabled(red is not None)
        if red is None:
            self.drift_lbl.setText(f"{robot.name}: n = {robot.n} joints = task dimension → not redundant "
                                   "(only isolated solutions).  Select the Franka Panda (7R) or the planar 3R arm.")
        else:
            what = {"pose": "full 6-D pose", "point2d": "position (x, y) only",
                    "position": "position only"}[red]
            dim = {"pose": 6, "point2d": 2, "position": 3}[red]
            self.drift_lbl.setText(f"task = {what}: n − m = {robot.n} − {dim} = {robot.n - dim} → "
                                   f"{robot.n - dim}-D null space of J.  Press ▶ to swing the arm while the "
                                   "end-effector stays put.")
        self._set_theta(self.theta)

    def _build_target_rows(self, robot):
        for r in self.t_rows.values():
            r.setParent(None)
            r.deleteLater()
        self.t_rows = {}
        task = _task(robot)
        R = 1.1 * robot.reach
        spec = [("x", -R, R, 3, " m", 1.0), ("y", -R, R, 3, " m", 1.0)]
        if not robot.planar:
            spec.append(("z", -R, R, 3, " m", 1.0))
        a = (-np.pi, np.pi, 1, "°", 180 / np.pi)
        if task == "pose":
            spec += [("roll", *a), ("pitch", *a), ("yaw", *a)]
        elif task == "planar":
            spec += [("φ", *a)]
        for name, lo, hi, dec, unit, scale in spec:
            row = SliderRow(name, lo, hi, 0.0, dec, unit, scale)
            row.valueChanged.connect(self._target_slider_moved)
            self.target_lay.addWidget(row)
            self.t_rows[name] = row

    # ================================================================== state helpers
    def _set_theta(self, th, sliders=True):
        robot = self.ctx.robot
        self.theta = np.array(th, float)
        if sliders:
            self.sliders.set_values(self.theta)
        self.graphic.update(self.theta)
        self._update_target_label()

    def _sliders_moved(self, th):
        self._set_theta(th, sliders=False)

    def _set_target(self, T, update_rows=True):
        robot = self.ctx.robot
        self.T_target = np.array(T, float)
        if update_rows:
            p = self.T_target[:3, 3]
            R = self.T_target[:3, :3]
            vals = dict(x=p[0], y=p[1], z=p[2])
            rpy = kin.rot_to_rpy(R)
            vals.update(roll=rpy[0], pitch=rpy[1], yaw=rpy[2], **{"φ": np.arctan2(R[1, 0], R[0, 0])})
            for k, row in self.t_rows.items():
                row.set_value(vals[k])
        task = _task(robot)
        self.view.set_sphere("ik.dsph", self.T_target[:3, 3], 0.75 * robot.link_radius, theme.WARN)
        if task in ("pose", "planar"):
            self.view.set_triad("ik.tgt", self.T_target, 0.16 * robot.reach, "{d}")
        else:
            self.view.remove_prefix("ik.tgt")
        self._update_target_label()

    def _target_from_rows(self):
        v = {k: r.value() for k, r in self.t_rows.items()}
        p = [v["x"], v["y"], v.get("z", 0.0)]
        if "roll" in v:
            R = kin.rpy_to_rot([v["roll"], v["pitch"], v["yaw"]])
        elif "φ" in v:
            R = _rotz(v["φ"])
        else:
            R = self.T_target[:3, :3]
        return trans(p, R)

    def _target_slider_moved(self, *_):
        self._set_target(self._target_from_rows(), update_rows=False)
        if self.cb_live.isChecked() and not self.circle_timer.isActive():
            robot = self.ctx.robot
            args = self._solver_args(robot)
            args["max_iter"] = min(args["max_iter"], 30)
            res = self._robust_ik(robot, self.T_target, args)
            self._set_theta(res.theta)
            self.result_lbl.setText(("✓" if res.success else "✗") + f" live IK: {res.iterations} iterations"
                                    + ("" if res.success else " (target unreachable — best effort shown)"))
            self.result_lbl.setStyleSheet(f"color:{theme.GOOD if res.success else theme.BAD}")

    def _update_target_label(self):
        robot = self.ctx.robot
        if len(self.theta) != robot.n:
            return
        task = _task(robot)
        p = self.T_target[:3, 3]
        T = kin.fk_space(robot, self.theta)
        if task in ("pose", "planar"):
            Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ self.T_target))
            rpy = np.degrees(kin.rot_to_rpy(self.T_target[:3, :3]))
            orient = (f"rpy = ({rpy[0]:+.1f}°, {rpy[1]:+.1f}°, {rpy[2]:+.1f}°)" if task == "pose" else
                      f"φ = {np.degrees(np.arctan2(self.T_target[1, 0], self.T_target[0, 0])):+.1f}°")
            self.target_lbl.setText(f"p_d = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m\n{orient}\n"
                                    f"current error: ‖ω_b‖ = {np.linalg.norm(Vb[:3]):.2e}, "
                                    f"‖v_b‖ = {np.linalg.norm(Vb[3:]):.2e}")
        else:
            e = np.linalg.norm(p - T[:3, 3])
            self.target_lbl.setText(f"p_d = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m\n"
                                    f"position-only task ({'x, y' if robot.planar else 'x, y, z'}); "
                                    f"current error ‖p_d − p‖ = {e:.2e} m")

    def target_from_ee(self):
        self._set_target(kin.fk_space(self.ctx.robot, self.theta))

    def random_target(self):
        robot = self.ctx.robot
        self._set_target(kin.fk_space(robot, robot.random_config()))

    def clear_ghosts(self):
        self.anim_timer.stop()
        self._anim = None
        self.view.remove_prefix("ik.it")
        self.view.remove_prefix("ik.sol")
        self._ghosts = []

    # ================================================================== solver
    def _method_changed(self, *_):
        self.lam.setEnabled(self.method.currentIndex() == 2)

    def _solver_args(self, robot):
        idx = self.method.currentIndex()
        return dict(eomg=float(self.eomg.currentText()), ev=float(self.ev.currentText()),
                    max_iter=self.maxit.value(), frame="space" if idx == 1 else "body",
                    damping=self.lam.value() if idx == 2 else 0.0,
                    position_only=_task(robot) in ("position", "point2d"))

    def _robust_ik(self, robot, T, args):
        """Warm-started IK for real-time use: Newton–Raphson; if it fails (target near a singularity or
        outside the workspace) retry with DLS and keep the best iterate of both runs (never worse than θ)."""
        res = kin.ik_newton_raphson(robot, T, self.theta, **args)
        if res.success:
            return res
        res2 = kin.ik_newton_raphson(robot, T, self.theta,
                                     **dict(args, damping=max(args["damping"], 0.05), max_iter=25))
        best = None
        for r in (res, res2):
            e = np.asarray(r.err_w) + np.asarray(r.err_v)
            k = int(np.argmin(e))
            if best is None or e[k] < best[0]:
                best = (e[k], r, k)
        _, r, k = best
        r.theta, r.iterations = r.history[k].copy(), k
        return r

    def solve(self):
        robot = self.ctx.robot
        self._stop_modes()
        g = self.guess.currentIndex()
        th0 = self.theta.copy() if g == 0 else (robot.home.copy() if g == 1 else robot.random_config())
        args = self._solver_args(robot)
        t = time.perf_counter()
        res = kin.ik_newton_raphson(robot, self.T_target, th0, **args)
        dt = time.perf_counter() - t
        self._last = res
        ok = res.success
        self.result_lbl.setStyleSheet(f"color:{theme.GOOD if ok else theme.BAD}")
        last = (f"‖ω_b‖ = {res.err_w[-1]:.1e}, ‖v_b‖ = {res.err_v[-1]:.1e}" if not args["position_only"]
                else f"‖p_d − p‖ = {res.err_v[-1]:.1e} m")
        th_w = kin.wrap_angles(res.theta)                         # same pose, angles in (−180°, 180°]
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        bad = [i + 1 for i in range(robot.n) if th_w[i] < lo[i] - 1e-9 or th_w[i] > hi[i] + 1e-9]
        self._final = th_w
        self.result_lbl.setText(
            (f"✓ converged in {res.iterations} iterations ({1e3 * dt:.1f} ms)\n" if ok else
             f"✗ not converged after {res.iterations} iterations\n  (unreachable target or poor θ⁰ — try DLS "
             f"or another guess)\n") + f"{res.method}: {last}\nθ = {_deg(th_w, 2)}° (wrapped)"
            + (f"\n⚠ outside the joint limits of joint(s) {', '.join(map(str, bad))}: Newton–Raphson ignores "
               "limits — 'Find all solutions' keeps only feasible branches" if bad else ""))
        self._cross_check(robot, res, th0, args)
        self._plot_convergence(res, args)
        self._fill_iter_table(res)
        self._start_anim(res)

    def _cross_check(self, robot, res, th0, args):
        lines = []
        if not args["position_only"]:
            space = args["frame"] == "space"
            name = "mr.IKinSpace" if space else "mr.IKinBody"
            fn, lst = (mr.IKinSpace, robot.Slist) if space else (mr.IKinBody, robot.Blist)
            th_mr, ok_mr = fn(lst, robot.M, self.T_target, th0, args["eomg"], args["ev"])
            th_mr = np.asarray(th_mr, float)
            if args["damping"] > 0:
                d = np.linalg.norm(kin.wrap_angles(th_mr - res.theta))
                if not ok_mr:
                    same = f"{name} did not converge from this θ⁰" + (" — DLS did ✓" if res.success else "")
                elif not res.success:
                    same = "DLS not converged (increase iterations or lower λ)"
                elif d < 1e-4:
                    same = "same solution ✓"
                else:
                    same = "both converged, to different IK branches ✓"
                lines.append(f"DLS vs {name} (pure Newton, same θ⁰):\n  ‖Δθ‖ = {d:.1e} → {same}")
            else:
                dk = [np.linalg.norm(th_mr - h) for h in res.history]
                k = int(np.argmin(dk))
                if dk[k] < 1e-9:
                    lines.append(f"✓ matches {name}: its output = our iterate θ^{k},  ‖Δθ‖ = {dk[k]:.1e}" +
                                 ("" if ok_mr else f"\n   ({name} stops after 20 iterations, not converged)"))
                else:
                    lines.append(f"≈ {name}: ‖Δθ‖ = {np.linalg.norm(th_mr - res.theta):.1e}"
                                 f" ({'converged' if ok_mr else 'not converged in its 20 iterations'})")
        if _textbook_planar(robot):
            T = self.T_target
            phi = np.arctan2(T[1, 0], T[0, 0]) if robot.n == 3 else None
            an = kin.ik_analytic_planar(robot, T[0, 3], T[1, 3], phi)
            if an:
                d = [np.linalg.norm(kin.wrap_angles(a - res.theta)) for a in an]
                k = int(np.argmin(d))
                lines.append(f"{'✓' if d[k] < 1e-5 else '≈'} analytic IK (law of cosines, "
                             f"{'elbow-down' if k == 0 else 'elbow-up'}): ‖Δθ‖ = {d[k]:.1e}")
            else:
                lines.append("analytic IK: target outside the annular workspace (no solution)")
        elif args["position_only"]:
            lines.append("position-only task: mr.IKinBody (full pose) does not apply")
        self.mr_lbl.setText("\n".join(lines))
        good = all("✓" in l or l.startswith("position-only") for l in lines)
        self.mr_lbl.setStyleSheet(f"color:{theme.GOOD if good else theme.WARN}")

    def _plot_convergence(self, res, args):
        pw = self.conv_plot
        pw.clear()
        k = np.arange(len(res.err_w))
        kmax = max(int(k[-1]), 1)
        ew = np.maximum(np.asarray(res.err_w), 1e-17)
        ev = np.maximum(np.asarray(res.err_v), 1e-17)
        if not args["position_only"]:
            pw.plot(k, ew, pen=pen(0), symbol="o", symbolSize=6, symbolBrush=theme.SERIES[0],
                    symbolPen=None, name="‖ω_b‖ [rad]")
            pw.plot(k, ev, pen=pen(1), symbol="s", symbolSize=6, symbolBrush=theme.SERIES[1],
                    symbolPen=None, name="‖v_b‖ [m]")
            pw.plot([0, kmax], [args["eomg"]] * 2, pen=pen(0, 1, "dash"), name="ε_ω")
            pw.plot([0, kmax], [args["ev"]] * 2, pen=pen(1, 1, "dash"), name="ε_v")
        else:
            pw.plot(k, ev, pen=pen(1), symbol="s", symbolSize=6, symbolBrush=theme.SERIES[1],
                    symbolPen=None, name="‖p_d − p‖ [m]")
            pw.plot([0, kmax], [args["ev"]] * 2, pen=pen(1, 1, "dash"), name="ε_v")
        pw.setXRange(0, kmax, padding=0.04)
        # observed order of convergence  q ≈ log(e_i/e_{i-1}) / log(e_{i-1}/e_{i-2})
        e = np.hypot(res.err_w, res.err_v)
        q = None
        for i in range(2, len(e)):
            if min(e[i], e[i - 1], e[i - 2]) > 1e-7 and e[i] < e[i - 1] < e[i - 2] and e[i - 2] < 0.5:
                q = np.log(e[i] / e[i - 1]) / np.log(e[i - 1] / e[i - 2])
        if q is not None:
            kind = "quadratic" if q > 1.6 else ("superlinear" if q > 1.15 else "linear")
            self.order_lbl.setText(f"Observed order of convergence q = log(eᵢ/eᵢ₋₁)/log(eᵢ₋₁/eᵢ₋₂) ≈ {q:.2f}"
                                   f"  →  {kind}")
        else:
            self.order_lbl.setText("Observed order of convergence: (too few iterations to estimate)")

    def _fill_iter_table(self, res):
        rows = []
        for i, th in enumerate(res.history):
            rows.append([str(i)] + [f"{x:+.3f}" if p else f"{np.degrees(x):+.2f}" for x, p in zip(th, self.ctx.robot.is_prismatic)]
                        + [f"{res.err_w[i]:.2e}", f"{res.err_v[i]:.2e}"])
        A = np.array(rows, dtype=object)
        self.iter_table.set_matrix(A)
        _fit_table(self.iter_table)
        self.iter_table.scrollToBottom()

    # ------------------------------------------------------------------ iterate animation
    def _start_anim(self, res):
        self.clear_ghosts()
        self._anim = dict(res=res, k=0)
        n = len(res.history)
        self.anim_timer.start(int(np.clip(2500 / max(n, 1), 90, 380)))

    def _anim_step(self):
        a = self._anim
        robot = self.ctx.robot
        if a is None:
            self.anim_timer.stop()
            return
        H = a["res"].history
        k = a["k"]
        if k >= len(H):
            self.anim_timer.stop()
            self._anim = None
            self._set_theta(getattr(self, "_final", a["res"].theta))
            return
        g = robot.geometry(H[k])
        self._ghosts.append((f"ik.it{k}", g))
        if k == 0:
            self.view.set_label("ik.it.lbl0", g["T_ee"][:3, 3], "  θ⁰", "#ffffff")
        self._ghosts = self._ghosts[:1] + self._ghosts[1:][-30:]
        N = len(self._ghosts)
        for j, (key, gg) in enumerate(self._ghosts):
            if j == 0:
                col = (1.0, 1.0, 1.0, 0.5)
            else:
                col = _rgba(theme.ACCENT, max(0.12, 0.85 * 0.62 ** (N - 1 - j)))
            _ghost(self.view, robot, key, col, 2.5, geom=gg)
        self._set_theta(H[k])
        a["k"] += 1

    # ================================================================== all solutions
    def find_all(self):
        robot = self.ctx.robot
        self._stop_modes()
        self._cancel_all()
        T = self.T_target.copy()
        po = _task(robot) in ("position", "point2d")
        n_starts = 40 if robot.n > 6 else 60

        def job(w):
            sols = kin.ik_all_solutions(robot, T, n_starts=n_starts, position_only=po)
            return robot.name, T, sols

        w = _Worker(job, self)
        w.done.connect(self._all_done)
        w.failed.connect(lambda e: (self.all_lbl.setText("failed:\n" + e[-400:]), self.all_btn.setEnabled(True)))
        w.finished.connect(lambda w=w: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        self._all_worker = w
        self.all_btn.setEnabled(False)
        self.all_lbl.setText(f"Newton–Raphson from {n_starts} random seeds (background thread) …")
        w.start()

    def _all_done(self, out):
        self.all_btn.setEnabled(True)
        self._all_worker = None
        name, T, sols = out
        robot = self.ctx.robot
        if name != robot.name:
            return
        self.clear_ghosts()
        self.sol_list.clear()
        found = len(sols)
        redundant = robot.n > _TASK_DIM[_task(robot)]
        if found > 8:          # redundant arm: keep 8 well-spread samples (farthest-point sampling)
            chosen = [0]
            D = np.array([[np.linalg.norm(kin.wrap_angles(a - b)) for b in sols] for a in sols])
            while len(chosen) < 8:
                chosen.append(int(np.argmax(D[:, chosen].min(axis=1))))
            sols = [sols[i] for i in chosen]
        sols = sorted(sols, key=lambda s: tuple(np.round(s, 3)))
        self._solutions = sols
        for i, s in enumerate(sols):
            col = theme.SERIES[i % len(theme.SERIES)] if i < len(theme.SERIES) else "#ffffff"
            _ghost(self.view, robot, f"ik.sol{i}", _rgba(col, 0.85), 3.0, theta=s)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(col))
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f"#{i + 1}  {_deg(s)}°")
            it.setData(QtCore.Qt.UserRole, i)
            self.sol_list.addItem(it)
        if found == 0:
            self.all_lbl.setText("No solution found — the target is probably outside the workspace.")
        elif redundant:
            self.all_lbl.setText(f"Redundant arm (n = {robot.n} > {_TASK_DIM[_task(robot)]}): a continuum of "
                                 f"solutions (self-motion manifold). {found} distinct samples found"
                                 + (", 8 well-spread ones shown." if found > 8 else "."))
        else:
            self.all_lbl.setText(f"{found} distinct solution(s) within the joint limits "
                                 f"(max. 8 for a 6R arm; 2 for planar 2R/3R). Click one to move the arm.")
        # analytic vs numeric for the textbook planar arms
        if _textbook_planar(robot):
            phi = np.arctan2(T[1, 0], T[0, 0]) if robot.n == 3 else None
            an = kin.ik_analytic_planar(robot, T[0, 3], T[1, 3], phi)
            lines = ["Analytic IK (MR §6.1) vs numeric:"]
            for a, lab in zip(an, ["elbow-down θ₂>0", "elbow-up   θ₂<0"]):
                d = min((np.linalg.norm(kin.wrap_angles(a - s)) for s in sols), default=np.nan)
                lines.append(f"{lab}: {_deg(a)}°\n   ↔ numeric ‖Δθ‖ = {d:.1e} {'✓' if d < 1e-5 else '✗'}")
            if not an:
                lines.append("target unreachable (|c₂| > 1)")
            self.analytic_lbl.setText("\n".join(lines))
            self.analytic_lbl.setStyleSheet(f"color:{theme.GOOD}")
        else:
            self.analytic_lbl.setText("")

    def _solution_clicked(self, item):
        i = item.data(QtCore.Qt.UserRole)
        if i is None or i >= len(self._solutions):
            return
        robot = self.ctx.robot
        for j, s in enumerate(self._solutions):
            col = theme.SERIES[j % len(theme.SERIES)]
            _ghost(self.view, robot, f"ik.sol{j}", _rgba(col, 1.0 if j == i else 0.35), 6.0 if j == i else 2.0,
                   theta=s)
        self._set_theta(self._solutions[i])

    # ================================================================== self-motion
    def toggle_self_motion(self, on):
        self.self_btn.setText("■ Stop self-motion" if on else "▶ Self-motion (null space)")
        robot = self.ctx.robot
        if not on:
            self.self_timer.stop()
            return
        red = _redundancy(robot)
        if red is None:
            self.self_btn.setChecked(False)
            return
        self.anim_timer.stop()
        if self.circle_btn.isChecked():
            self.circle_btn.setChecked(False)
        self.clear_ghosts()
        self._sm = dict(task=red, T0=kin.fk_space(robot, self.theta), th_start=self.theta.copy(), dir=None,
                        trace=[], steps=0, drift1=0.0)
        self._set_target(self._sm["T0"])
        self.self_timer.start(30)

    def _self_step(self):
        robot = self.ctx.robot
        sm = self._sm
        task, T0 = sm["task"], sm["T0"]
        th = self.theta
        _, J = _task_error(robot, th, T0, task)
        if sm["dir"] is None:                                       # null vector of J from the SVD
            sm["dir"] = np.linalg.svd(J)[2][-1]
        if task == "pose":
            dth = kin.null_space_step(robot, th, sm["dir"])        # (I − J_b⁺ J_b) z
        else:
            dth = (np.eye(robot.n) - np.linalg.pinv(J) @ J) @ sm["dir"]
        nrm = np.linalg.norm(dth)
        if nrm < 1e-9:
            sm["dir"] = None
            return
        dth *= np.radians(0.8) / nrm                               # 0.8° of joint motion per frame
        th1 = th + dth
        e1, _ = _task_error(robot, th1, T0, task)
        sm["drift1"] = float(np.linalg.norm(e1))                   # drift of the pure null-space step
        for _ in range(3):                                         # Newton correction back onto T0
            e, Jc = _task_error(robot, th1, T0, task)
            if np.linalg.norm(e) < 1e-14:
                break
            th1 = th1 + np.linalg.pinv(Jc) @ e
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        inside = (th >= lo) & (th <= hi)                           # only guard joints that start feasible
        if np.any(inside & ((th1 < lo + 1e-3) | (th1 > hi - 1e-3))):   # joint limit → reverse
            sm["dir"] = -dth / np.linalg.norm(dth)
            return
        sm["dir"] = dth / np.linalg.norm(dth)
        sm["steps"] += 1
        self._set_theta(th1)
        g = self.graphic.geom
        j = (robot.n - 1) // 2
        sm["trace"].append(g["joint_pos"][j].copy())
        sm["trace"] = sm["trace"][-500:]
        if len(sm["trace"]) > 1:
            self.view.set_polyline("ik.smtrace", sm["trace"], (1.0, 0.65, 0.3, 0.9), 2)
        T = kin.fk_space(robot, th1)
        dp = np.linalg.norm(T[:3, 3] - T0[:3, 3])
        dw = np.linalg.norm(mr.so3ToVec(mr.MatrixLog3(T0[:3, :3].T @ T[:3, :3])))
        moved = np.degrees(np.linalg.norm(th1 - sm["th_start"]))
        s = (f"EE drift (after Newton correction):\n  ‖Δp‖ = {dp:.1e} m" +
             (f",  ‖Δω‖ = {dw:.1e} rad" if task == "pose" else "") +
             f"\nnull-space step alone would drift {sm['drift1']:.1e} per frame\n"
             f"joint {j + 1} (elbow, orange trace) moving · ‖θ − θ_start‖ = {moved:.1f}°\n"
             f"θ = {_deg(th1)}°")
        self.drift_lbl.setText(s)

    # ================================================================== follow a circle
    def toggle_circle(self, on):
        self.circle_btn.setText("■ Stop" if on else "▶ Follow a circle")
        robot = self.ctx.robot
        if not on:
            self.circle_timer.stop()
            return
        self.anim_timer.stop()
        if self.self_btn.isChecked():
            self.self_btn.setChecked(False)
        self.clear_ghosts()
        T0 = kin.fk_space(robot, self.theta)
        r = self.radius.value()
        # circle through the current EE, bulging towards the base (stays inside the workspace):
        # planar arms -> in the x-y plane; spatial arms -> vertical plane through the base axis
        p0 = T0[:3, 3]
        radial = np.array([p0[0], p0[1], 0.0])
        e1 = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > 1e-6 else np.array([1.0, 0, 0])
        e2 = np.cross([0, 0, 1.0], e1) if robot.planar else np.array([0, 0, 1.0])
        c = p0 - r * e1
        ang = np.linspace(0, 2 * np.pi, 120)
        self.view.set_polyline("ik.circ", c + r * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)),
                               (1.0, 0.82, 0.4, 0.55), 1.5)
        self._circ = dict(T0=T0, c=c, e1=e1, e2=e2, t0=time.perf_counter(), trace=[], its=[], ms=[], err=[])
        self.view.remove("ik.trace")
        self.circle_timer.start(30)

    def _circle_step(self):
        robot = self.ctx.robot
        c = self._circ
        r = self.radius.value()
        t = time.perf_counter() - c["t0"]
        a = 2 * np.pi * t / self.period.value()
        p = c["c"] + r * (np.cos(a) * c["e1"] + np.sin(a) * c["e2"])
        T = trans(p, c["T0"][:3, :3])
        args = self._solver_args(robot)
        args["max_iter"] = 15
        t1 = time.perf_counter()
        res = self._robust_ik(robot, T, args)
        ms = 1e3 * (time.perf_counter() - t1)
        self._set_theta(res.theta)
        self._set_target(T)
        pe = kin.fk_space(robot, res.theta)[:3, 3]
        c["its"].append(res.iterations)
        c["ms"].append(ms)
        c["err"].append(np.linalg.norm(pe - p))
        for k in ("its", "ms", "err"):
            c[k] = c[k][-100:]
        if self.cb_trace.isChecked():
            c["trace"].append(pe.copy())
            c["trace"] = c["trace"][-400:]
            if len(c["trace"]) > 1:
                self.view.set_polyline("ik.trace", c["trace"], (0.3, 0.85, 1.0, 0.95), 2.5)
        self.rt_lbl.setText(f"{'✓' if res.success else '✗'} {res.iterations} iteration(s) this frame, "
                            f"mean {np.mean(c['its']):.2f}\n{ms:.2f} ms per IK solve (mean {np.mean(c['ms']):.2f} ms)\n"
                            f"tracking error ‖p − p_d‖ = {c['err'][-1]:.1e} m")
        self.rt_lbl.setStyleSheet(f"color:{theme.GOOD if res.success else theme.BAD}")

    def _trace_toggled(self, on):
        if not on:
            self.view.remove("ik.trace")
            if hasattr(self, "_circ"):
                self._circ["trace"] = []

    # ================================================================== housekeeping
    def _stop_modes(self):
        self.anim_timer.stop()
        if self._anim is not None:
            self._anim = None
        if self.self_btn.isChecked():
            self.self_btn.setChecked(False)
        if self.circle_btn.isChecked():
            self.circle_btn.setChecked(False)

    def _cancel_all(self):
        if self._all_worker is not None:
            self._all_worker.cancelled = True
            self._all_worker = None
        self.all_btn.setEnabled(True)

    def _shutdown_workers(self):
        for w in list(self._workers):
            w.cancelled = True
            w.wait(5000)

    def on_hide(self):
        self._stop_modes()
