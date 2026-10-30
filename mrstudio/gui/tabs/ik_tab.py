"""Tab 3 -- inverse kinematics (MR Ch. 6): Newton-Raphson (IKinBody / IKinSpace), damped least squares,
multiple solutions, analytic planar IK, redundancy (null-space self-motion) and real-time IK."""
from __future__ import annotations

import time

import modern_robotics as mr
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ... import kinematics as kin
from ...robots import trans
from .. import theme
from ..base import BaseTab
from ..jobs import JobRunner
from ..viewer import RobotGraphic, SceneView
from ..widgets import _sub, SliderRow
from .ik_math import (_TASK_DIM, _deg, _fmt_theta, _redundancy, _task, _textbook_planar, analytic_vs_numeric_lines,
                      circle_frame, cross_check_lines, iterate_rows, pose_from_rows, redundancy_text, robust_ik,
                      self_motion_step, spread_samples, target_label_text, target_row_spec, target_row_values)
from .ik_panels import TOLS, _fit_table, _ghost, _rgba, build_panels, plot_convergence
from .ik_theory import THEORY



class IKTab(BaseTab):
    title = "③ IK"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.view = SceneView()
        self.graphic = None
        self.theta = np.zeros(0)
        self.T_target = np.eye(4)
        self.t_rows: dict[str, SliderRow] = {}
        self.jobs = JobRunner(self)              # "Find all solutions": one background search at a time
        self._all_req = None                     # (robot name, T_sd) of the running / last search
        self._solutions: list[np.ndarray] = []
        self._ghosts: list = []
        self._anim = None
        self._last = None

        controls, bottom = build_panels(self)

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
        if app is not None and not hasattr(BaseTab, "shutdown"):   # older BaseTab without the quit hook
            app.aboutToQuit.connect(self.shutdown)
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
        self.drift_lbl.setText(redundancy_text(robot, red))
        self._set_theta(self.theta)

    def _build_target_rows(self, robot):
        for r in self.t_rows.values():
            r.setParent(None)
            r.deleteLater()
        self.t_rows = {}
        for name, lo, hi, dec, unit, scale in target_row_spec(robot):
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
            vals = target_row_values(self.T_target)
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
        return pose_from_rows({k: r.value() for k, r in self.t_rows.items()}, self.T_target)

    def _target_slider_moved(self, *_):
        self._set_target(self._target_from_rows(), update_rows=False)
        if self.cb_live.isChecked() and not self.circle_timer.isActive():
            robot = self.ctx.robot
            args = self._solver_args(robot)
            args["max_iter"] = min(args["max_iter"], 30)
            res = robust_ik(robot, self.T_target, self.theta, args)
            self._set_theta(res.theta)
            self.result_lbl.setText(("✓" if res.success else "✗") + f" live IK: {res.iterations} iterations"
                                    + ("" if res.success else " (target unreachable — best effort shown)"))
            self.result_lbl.setStyleSheet(f"color:{theme.GOOD if res.success else theme.BAD}")

    def _update_target_label(self):
        robot = self.ctx.robot
        if len(self.theta) != robot.n:
            return
        self.target_lbl.setText(target_label_text(robot, self.theta, self.T_target))

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
        f = "s" if res.frame == "space" else "b"
        last = (f"‖ω_{f}‖ = {res.err_w[-1]:.1e}, ‖v_{f}‖ = {res.err_v[-1]:.1e}" if not args["position_only"]
                else f"‖p_d − p‖ = {res.err_v[-1]:.1e} m")
        # same pose: revolute joints wrapped to [−180°, 180°), shifted by ±360° into the limits when possible
        th_w = kin.wrap_joints(robot, res.theta)
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        bad = [i + 1 for i in range(robot.n) if th_w[i] < lo[i] - 1e-9 or th_w[i] > hi[i] + 1e-9]
        self._final = th_w
        self.result_lbl.setText(
            (f"✓ converged in {res.iterations} iterations ({1e3 * dt:.1f} ms)\n" if ok else
             f"✗ not converged after {res.iterations} iterations\n  (unreachable target or poor θ⁰ — try DLS "
             f"or another guess)\n") + f"{res.method}: {last}\nθ = {_fmt_theta(robot, th_w)} (wrapped)"
            + (f"\n⚠ outside the joint limits of joint(s) {', '.join(map(str, bad))}: Newton–Raphson ignores "
               "limits — 'Find all solutions' keeps only feasible branches" if bad else ""))
        self._cross_check(robot, res, th0, args)
        self._plot_convergence(res, args)
        self._fill_iter_table(res)
        self._start_anim(res)

    def _cross_check(self, robot, res, th0, args):
        lines = cross_check_lines(robot, self.T_target, res, th0, args)
        self.mr_lbl.setText("\n".join(lines))
        good = all("✓" in l or l.startswith("position-only") for l in lines)
        self.mr_lbl.setStyleSheet(f"color:{theme.GOOD if good else theme.WARN}")

    def _plot_convergence(self, res, args):
        plot_convergence(self.conv_plot, self.order_lbl, res, args)

    def _fill_iter_table(self, res):
        A = iterate_rows(self.ctx.robot, res)
        robot = self.ctx.robot
        f = "s" if res.frame == "space" else "b"
        err = ["(unused)", "‖p_d − p‖"] if res.method.endswith("(position)") else [f"‖ω_{f}‖", f"‖v_{f}‖"]
        self.iter_table.set_matrix(A)
        self.iter_table.set_labels(cols=["i"] + [f"θ{_sub(i + 1)} [{robot.display_unit(i)}]"
                                                for i in range(robot.n)] + err)
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
        if self.jobs.is_busy():                      # one search at a time (the button is disabled anyway)
            return
        self._stop_modes()
        T = self.T_target.copy()
        po = _task(robot) in ("position", "point2d")
        redundant = robot.n > _TASK_DIM[_task(robot)]
        # redundant arms have a continuum of solutions: a fixed number of samples, no "until nothing new" rule
        kw = dict(n_starts=40, max_starts=40) if redundant else dict(n_starts=60, patience=60, max_starts=600)

        def job(progress, cancelled):
            return kin.ik_all_solutions(robot, T, position_only=po, cancelled=cancelled, progress=progress, **kw)

        self._all_req = (robot.name, T)
        self.all_btn.setEnabled(False)
        self.all_cancel.setVisible(True)
        self.all_lbl.setStyleSheet("")
        self.all_lbl.setText("Newton–Raphson from random seeds (background thread)"
                             + ("" if redundant else ": until 60 starts in a row find no new branch") + " …")
        self.jobs.run(job, on_done=lambda sols, req=self._all_req: self._all_done(req, sols),
                      on_error=self._all_failed,
                      on_progress=lambda f: self.all_lbl.setText(
                          f"Newton–Raphson from random seeds (background thread) … {100 * f:.0f} % of the "
                          f"{kw['max_starts']}-start cap"),
                      label="IK branches")

    def _all_failed(self, err):
        self._all_finished()
        self.all_lbl.setText("failed:\n" + err[-400:])

    def _all_finished(self):
        self.all_btn.setEnabled(True)
        self.all_cancel.setVisible(False)

    def _all_done(self, req, sols):
        self._all_finished()
        robot = self.ctx.robot
        name, T = req
        if name != robot.name or req is not self._all_req:      # robot or request changed meanwhile
            return
        self.clear_ghosts()
        self.sol_list.clear()
        found = len(sols)
        redundant = robot.n > _TASK_DIM[_task(robot)]
        if found > 8:          # redundant arm: keep 8 well-spread samples (farthest-point sampling)
            sols = spread_samples(robot, sols, 8)
        sols = sorted(sols, key=lambda s: tuple(np.round(s, 3)))
        self._solutions = sols
        for i, s in enumerate(sols):
            col = theme.SERIES[i % len(theme.SERIES)] if i < len(theme.SERIES) else "#ffffff"
            _ghost(self.view, robot, f"ik.sol{i}", _rgba(col, 0.85), 3.0, theta=s)
            pm = QtGui.QPixmap(12, 12)
            pm.fill(QtGui.QColor(col))
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f"#{i + 1}  {_fmt_theta(robot, s)}")
            it.setData(QtCore.Qt.UserRole, i)
            self.sol_list.addItem(it)
        if found == 0:
            self.all_lbl.setText("No solution found by the numerical search — the target is probably outside "
                                 "the workspace (or every branch violates the joint limits).")
        elif redundant:
            self.all_lbl.setText(f"Redundant arm (n = {robot.n} > {_TASK_DIM[_task(robot)]}): a continuum of "
                                 f"solutions (self-motion manifold). {found} distinct samples found"
                                 + (", 8 well-spread ones shown." if found > 8 else "."))
        else:
            self.all_lbl.setText(f"{found} distinct solution(s) within the joint limits, found by numerical "
                                 "search (may be incomplete).  Up to 8 for UR/PUMA-type arms (a general 6R arm "
                                 "can have 16); 2 for planar 2R/3R.  Click one to move the arm.")
        # analytic vs numeric for the textbook planar arms
        if _textbook_planar(robot):
            self.analytic_lbl.setText("\n".join(analytic_vs_numeric_lines(robot, T, sols)))
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
        th1 = self_motion_step(robot, th, sm)
        if th1 is None:
            return
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
        c, e1, e2 = circle_frame(robot, T0, r)
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
        res = robust_ik(robot, T, self.theta, args)
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
        """Drop the running branch search (non-blocking: the worker stops at its next start)."""
        if self.jobs.is_busy():
            self.jobs.cancel()
            self.all_lbl.setText("search cancelled")
        self._all_req = None
        self._all_finished()

    def shutdown(self):
        """Application quit: stop timers, cancel the search (waits at most a fraction of a second)."""
        self._stop_modes()
        self.jobs.shutdown(500)
        parent = getattr(super(), "shutdown", None)
        if parent is not None:
            parent()

    def on_hide(self):
        self._stop_modes()
