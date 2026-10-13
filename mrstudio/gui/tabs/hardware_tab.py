"""Hardware bridge -- run the course arm (or its simulation) for the projects' required hardware sessions.

Streams a joint trajectory to the arm at 50 Hz, logs commanded vs measured joints in the CSV formats of
the SBEG 361 (hwcompare.py) and SBEG 461 (hardware.py) starters, enforces the course-arm hardware safety
checklist and computes the hardware acceptance tests HW-2 ... HW-6.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...hardware import COURSE_ARM, HardwareError, SimulatedArm, course_arm_robot
from ...hardware import metrics as hwm
from ...hardware.logger import FORMATS, load_trajectory
from ...hardware.mycobot import MyCobotDriver, list_serial_ports, sdk_available
from ...hardware.safety import SESSION_LABELS, SESSIONS, SafetySupervisor, retime_to_speed
from ...hardware.session import HardwareRun, concat, quintic_move, with_approach
from ...references import further_reading
from .. import theme
from ..base import BaseTab
from ..viewer import GhostRobot, RobotGraphic, SceneView
from ..widgets import JointSliders, button, checkbox, combo, group, hbox, pen, plot, spin, value_label

THEORY = r"""
<h2>Hardware bridge — simulation meets the real arm</h2>
<span class="ref">Modern Robotics §4.1 (PoE), §5.1 (Jacobian), §9.2.2.1 (polynomial time scaling),
§11.2.2.1 (first-order error dynamics); course-arm safety checklist (course_robots.md)</span>
<p>The arm receives <b>joint set-points</b> $\theta_{cmd}(t)$ at 50 Hz and reports its encoder angles
$\theta_{meas}(t)$.  Both are logged and compared, in joint space and at the tool point.</p>
<h3>Trajectories at a safe speed</h3>
<p>A rest-to-rest move uses the quintic time scaling $s(\tau)=10\tau^3-15\tau^4+6\tau^5$, $\tau=t/T$:</p>
$$\theta(t)=\theta_{start}+s(t)\,(\theta_{end}-\theta_{start})$$
$$\dot{\theta}_{max}=\frac{15}{8T}\,|\theta_{end}-\theta_{start}|\quad(\mathrm{Fig.\ 9.4})$$
<p>so the shortest duration that respects the allowed speed $f\,\dot{\theta}_{rated}$
($f=0.25$ in session H1 and on a first run, $0.5$ otherwise) is</p>
$$T\geq\frac{15}{8}\,\max_i\frac{|\Delta\theta_i|}{f\,\dot{\theta}_{rated,i}}$$
<p>(the tab adds 25 % margin).  A loaded CSV is slowed down uniformly if it is too fast.</p>
<h3>Servo model (Simulated back-end)</h3>
<p>Each joint follows its set-point with <b>first-order error dynamics</b>, $\theta_e=\theta_{cmd}-\theta$:</p>
$$\dot{\theta}_e(t)\,+\,\theta_e(t)/t_c=0\quad(11.6)$$
$$\theta_e(t)=e^{-t/t_c}\,\theta_e(0)\quad(11.7)$$
<p>with $t_c=0.04$ s, speed saturation, 12-bit quantisation ($2\pi/4096$ = 0.088°) and noise.
A ramp of speed $v$ is followed with the lag $v\,t_c$ (plus half a sample of zero-order hold).
After <b>Stop</b> the joints coast to rest, $\theta(t)=\theta_0+\dot{\theta}_0\,t_s\,(1-e^{-t/t_s})$.</p>
<h3>Hardware tests (361 hwcompare.py, 461 hardware.py)</h3>
<p><b>HW-2</b> joint error, RMS and max of $|\theta_{cmd}-\theta_{meas}|$ per joint.<br>
<b>HW-3</b> tool-point error through the space-form PoE:</p>
$$p(\theta)=e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}\,M\,p_b\quad(4.14)$$
$$e_{task}(t)=\left|\,p(\theta_{cmd}(t))-p(\theta_{meas}(t))\,\right|\approx|J_v\,\theta_e|$$
<p><b>HW-4</b> smallest margin of $\theta_{meas}$ to the hardware limits ($\geq 10^{\circ}$).<br>
<b>HW-5</b> peak measured speed / rated speed ($\leq f$).<br>
<b>HW-6</b> time to halt after the stop command and joint motion after it.</p>
<h3>Safety checklist (both courses)</h3>
<p>1 software limits $\geq 10^{\circ}$ inside the hardware limits · 2 keep-out radius = reach + 0.30 m ·
3 speed $\leq 25\,\%$ (H1, first run) / $\leq 50\,\%$ of rated · 4 payload $\leq 0.125$ kg ·
5 induction · 6 nobody in the keep-out zone while the arm can move; a hand enters only in
<b>latched stop</b> with torque off · 7 e-stop tested, supervisor present.
The software enforces 1, 3 and 4 and blocks <b>Run</b> until the people at the bench confirm the rest.
<b>Stop</b> is always active and latches the arm until <i>Reset latch</i>.</p>
"""
THEORY += further_reading([
    ('corke', '§3.1.1, pp. 43–45', 'quintic polynomial trajectories (boundary-value system)'),
])

_SOURCES = ["Move to the joint target", "Quintic A → B", "Commanded joints of a CSV log"]
_BACKENDS = ["Simulated (no hardware)", "myCobot 280 (USB serial)"]
_SHORT = {"induction": "Safety induction done by all at the arm (5)",
          "keepout": "Keep-out zone taped, r = reach + 0.30 m (2)",
          "clear": "Nobody inside the keep-out zone (6)",
          "estop": "E-stop tested this session (7)",
          "supervisor": "Supervisor present for the run (7)",
          "payload": "Payload weighed and entered (4)"}
_RATE_MS = 20          # 50 Hz streaming (briefs: >= 20 Hz)
_POLL_MS = 100         # idle read-back


class HardwareTab(BaseTab):
    title = "⑬ Hardware Bridge"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.spec = COURSE_ARM
        self.robot, self.fallback = course_arm_robot()
        self.sup = SafetySupervisor(self.spec)
        self.driver = None
        self.run: HardwareRun | None = None
        self.last_log = None
        self.q_meas = self.spec.q_home.copy()
        self.pose_a = self.spec.q_home.copy()
        self.pose_b = self.spec.q_home + np.radians([35, 20, -25, 15, 25, -30])
        self.csv_traj = None
        self.view = SceneView()

        # ---- connection ---------------------------------------------------------------------
        self.backend_box = combo(_BACKENDS, self._backend_changed)
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setEditable(True)
        self.port_box.addItems(list_serial_ports() or ["/dev/ttyUSB0", "/dev/tty.usbserial-0001", "COM3"])
        self.connect_btn = button("Connect", self._toggle_connect, checkable=True, primary=True)
        self.conn_lbl = value_label("not connected")
        model_txt = (f"Model: <b>{self.robot.name}</b>" +
                     (" — the registered course-arm model is not available yet; using the built-in "
                      "kinematic model (same screw axes)." if self.fallback else ""))
        self.model_lbl = QtWidgets.QLabel(model_txt)
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setObjectName("muted")

        # ---- session --------------------------------------------------------------------------
        self.session_box = combo([SESSION_LABELS[s] for s in SESSIONS], self._safety_changed)
        self.session_box.setCurrentIndex(0)
        self.first_run_cb = checkbox("First run of this motion (≤ 25 % speed)", self._safety_changed, True)
        self.payload_spin = spin(0.0, self.spec.payload_max, 0.0, 3, 0.005, " kg")
        self.payload_spin.valueChanged.connect(self._safety_changed)
        self.speed_lbl = value_label()

        # ---- checklist -----------------------------------------------------------------------
        self.check_boxes: dict[str, QtWidgets.QCheckBox] = {}
        cl_widgets = []
        for key, text in self.sup.checklist_items():
            cb = checkbox(_SHORT.get(key, text), self._update_ready, False)
            cb.setToolTip(text)
            self.check_boxes[key] = cb
            cl_widgets.append(cb)
        self.auto_lbl = value_label()
        self.ready_lbl = value_label()

        # ---- trajectory ----------------------------------------------------------------------
        self.source_box = combo(_SOURCES, self._source_changed)
        self.sliders = JointSliders()
        self.sliders.changed.connect(self._target_changed)
        self.ab_lbl = value_label()
        self.csv_btn = button("Load CSV…", self._load_csv_dialog)
        self.csv_lbl = value_label("no file loaded")
        self.auto_stop = spin(0.0, 60.0, 0.0, 1, 0.5, " s")
        self.auto_stop.setSpecialValueText("off")
        self.auto_stop.setToolTip("Issue a stop command automatically at this time (HW-6 safe-stop test)")
        self.plan_lbl = value_label()

        # ---- run -------------------------------------------------------------------------------
        self.run_btn = button("▶ Run", self.start_run, primary=True)
        self.stop_btn = button("■ STOP", self.stop_now, tooltip="Halt and latch the arm (always active)")
        self.stop_btn.setStyleSheet(f"QPushButton {{ background:#8b1d24; border:1px solid {theme.BAD}; "
                                    "color:white; font-weight:700; padding:6px 14px; }"
                                    "QPushButton:pressed { background:#c0252f; }")
        self.stop_btn.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape))
        self.reset_btn = button("Reset latch", self.reset_latch)
        self.torque_btn = button("Torque off", self._toggle_torque,
                                 tooltip="Disable servo torque (only in latched stop, rule 6)")
        self.fmt_box = combo(list(FORMATS.values()))
        self.save_btn = button("Save log…", self._save_dialog)
        self.status_lbl = value_label("Connect a back-end to start.")

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Arm connection", hbox(QtWidgets.QLabel("Back-end"), self.backend_box),
                           hbox(QtWidgets.QLabel("Port"), self.port_box), self.connect_btn,
                           self.conn_lbl, self.model_lbl))
        cl.addWidget(group("Session", self.session_box, self.first_run_cb,
                           hbox(QtWidgets.QLabel("Payload on the tool"), self.payload_spin), self.speed_lbl))
        cl.addWidget(group("Safety checklist — tick before Run", *cl_widgets, self.auto_lbl, self.ready_lbl))
        cl.addWidget(group("Trajectory", self.source_box, self.sliders,
                           hbox(button("Home", lambda: self._set_target(self.spec.q_home)),
                                button("Set A ← target", self._set_a), button("Set B ← target", self._set_b)),
                           self.ab_lbl, hbox(self.csv_btn, None), self.csv_lbl,
                           hbox(QtWidgets.QLabel("Stop test (HW-6) at t ="), self.auto_stop), self.plan_lbl))
        cl.addWidget(group("Run", hbox(self.run_btn, self.stop_btn), hbox(self.reset_btn, self.torque_btn),
                           hbox(QtWidgets.QLabel("Log format"), self.fmt_box), self.save_btn, self.status_lbl))
        cl.addStretch(1)

        # ---- bottom: live plots + report -------------------------------------------------------
        self.joint_box = combo(["All joints"] + [f"J{i + 1}" for i in range(self.spec.n)], self._redraw_plots)
        self.p_q = plot("Joints: dashed = cmd, solid = meas", "θ [deg]", legend=True)
        self.p_e = plot("Error  θ_meas − θ_cmd", "[deg]", legend=False)
        self.c_cmd, self.c_meas, self.c_err = [], [], []
        for i in range(self.spec.n):
            self.c_cmd.append(self.p_q.plot([], [], pen=pen(i, 1.6, "dash")))
            self.c_meas.append(self.p_q.plot([], [], pen=pen(i, 2.0), name=f"J{i + 1}"))
            self.c_err.append(self.p_e.plot([], [], pen=pen(i, 1.6)))
        self.stop_line_q = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.BAD, width=1.5, style=QtCore.Qt.DashLine))
        self.stop_line_e = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.BAD, width=1.5, style=QtCore.Qt.DashLine))
        for p, ln in ((self.p_q, self.stop_line_q), (self.p_e, self.stop_line_e)):
            p.addItem(ln)
            ln.setVisible(False)
        self.report = QtWidgets.QTableWidget(0, 4)
        self.report.setHorizontalHeaderLabels(["Test", "Value", "Threshold", "Result"])
        self.report.verticalHeader().setVisible(False)
        self.report.setWordWrap(True)                     # full test names ("HW-3 task RMS [mm]")
        self.report.setTextElideMode(QtCore.Qt.ElideNone)
        self.report.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        hh = self.report.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.report_lbl = value_label("Run a trajectory to get the HW-2 … HW-6 report.")
        rep = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rep)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("<b>Hardware test report</b> (thresholds of the briefs)"))
        rl.addWidget(self.report, 1)
        rl.addWidget(self.report_lbl)
        plots = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addLayout(hbox(QtWidgets.QLabel("Show"), self.joint_box, None))
        ph = QtWidgets.QHBoxLayout()
        ph.addWidget(self.p_q, 3)
        ph.addWidget(self.p_e, 2)
        pl.addLayout(ph, 1)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(plots)
        bottom.addWidget(rep)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)

        for cb_ in (self.backend_box, self.port_box, self.session_box, self.source_box, self.fmt_box):
            cb_.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb_.setMinimumContentsLength(10)
            cb_.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.assemble(controls, self.view, bottom, THEORY, controls_width=380, bottom_ratio=(1, 1))

        # ---- scene ------------------------------------------------------------------------------
        self.graphic = RobotGraphic(self.view, self.robot, "hw.robot")
        self.ghost = GhostRobot(self.view, self.robot, "hw.cmd", color=(1.0, 0.82, 0.4, 0.75), width=3.0)
        self.view.frame_scene(self.sup.keepout_radius * 0.8, planar=False, center=(0, 0, 0.1))
        self.view.set_triad("hw.s", np.eye(4), 0.08, "{s}")
        self._draw_keepout()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.poll = QtCore.QTimer(self)
        self.poll.timeout.connect(self._poll)
        app = QtWidgets.QApplication.instance()
        if app is not None:                       # closing MR Studio must never leave the arm moving
            app.aboutToQuit.connect(self.shutdown)

        soft = dataclasses.replace(self.robot, joint_limits=self.sup.soft_limits)
        self.sliders.set_robot(soft, self.pose_b)
        self._backend_changed(self.backend_box.currentText())
        self._source_changed(self.source_box.currentText())
        self._safety_changed()
        self._update_scene(self.q_meas, None)
        self._update_buttons()

    # ================================================================== scene
    def _draw_keepout(self):
        r = self.sup.keepout_radius
        a = np.linspace(0, 2 * np.pi, 121)
        ring = np.c_[r * np.cos(a), r * np.sin(a), np.zeros_like(a)]
        self.view.set_polyline("hw.keepout", ring, (1.0, 0.42, 0.42, 0.85), 2.5)
        self.view.set_label("hw.keepout.lbl", [r * 0.72, -r * 0.72, 0.0], f"keep-out r = {r:.2f} m", theme.BAD)

    def _update_scene(self, q_meas, q_cmd):
        self.graphic.update(q_meas)
        if q_cmd is not None:
            self.ghost.update(q_cmd)
        else:
            self.ghost.update(self._target())

    # ================================================================== connection
    def _backend_changed(self, txt):
        sim = txt == _BACKENDS[0]
        self.port_box.setEnabled(not sim)
        if not sim and not sdk_available():
            self.conn_lbl.setText("pymycobot is not installed — install it with "
                                  "<tt>.venv/bin/python -m pip install pymycobot</tt> (or use the Simulated back-end).")
            self.conn_lbl.setStyleSheet(f"color:{theme.WARN}")
        elif not (self.driver and self.driver.connected):
            self.conn_lbl.setText("not connected")
            self.conn_lbl.setStyleSheet("")

    def _toggle_connect(self, on):
        if on:
            self.connect_backend()
        else:
            self.disconnect_backend()

    def connect_backend(self) -> bool:
        self.disconnect_backend()
        if self.backend_box.currentText() == _BACKENDS[0]:
            drv = SimulatedArm(self.spec, q0=self.q_meas)
        else:
            drv = MyCobotDriver(self.port_box.currentText().strip(), spec=self.spec)
        try:
            drv.connect()
        except HardwareError as e:
            self.conn_lbl.setText(str(e).replace("\n", "<br>"))
            self.conn_lbl.setStyleSheet(f"color:{theme.BAD}")
            self.connect_btn.blockSignals(True)
            self.connect_btn.setChecked(False)
            self.connect_btn.blockSignals(False)
            self._update_buttons()
            return False
        self.driver = drv
        self.q_meas = drv.read_joints()
        self.conn_lbl.setText(f"connected: {drv.name}" + (f" on {drv.port}" if hasattr(drv, "port") else ""))
        self.conn_lbl.setStyleSheet(f"color:{theme.GOOD}")
        if getattr(drv, "offset_calibrated", True) is False:
            self.conn_lbl.setText(self.conn_lbl.text() + "<br>⚠ servo zero offset not calibrated: the PoE zero "
                                  "pose is assumed to be the servo zero. Check the measured pose against the "
                                  "3-D view before the first run.")
            self.conn_lbl.setStyleSheet(f"color:{theme.WARN}")
        self.connect_btn.blockSignals(True)
        self.connect_btn.setChecked(True)
        self.connect_btn.setText("Disconnect")
        self.connect_btn.blockSignals(False)
        self.status_lbl.setText("Connected. Tick the checklist, choose a trajectory, then Run.")
        self.status_lbl.setStyleSheet("")
        if self.isVisible():
            self.poll.start(_POLL_MS)
        self._update_scene(self.q_meas, None)
        self._update_plan()
        self._update_buttons()
        return True

    def disconnect_backend(self):
        if self.run is not None and not self.run.finished:
            self.stop_now()
            self._finish_run()
        self.poll.stop()
        if self.driver is not None:
            self.driver.disconnect()
        self.driver = None
        self.connect_btn.blockSignals(True)
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("Connect")
        self.connect_btn.blockSignals(False)
        self._backend_changed(self.backend_box.currentText())
        self._update_buttons()

    def _poll(self):
        if self.driver is None or (self.run is not None and not self.run.finished):
            return
        try:
            self.q_meas = self.driver.read_joints()
        except HardwareError as e:
            self.status_lbl.setText(f"read failed: {e}")
            return
        self._update_scene(self.q_meas, None)

    # ================================================================== safety
    def _safety_changed(self, *_):
        self.sup.session = SESSIONS[self.session_box.currentIndex()]
        self.first_run_cb.setEnabled(self.sup.session != "H1")
        self.sup.first_run = self.first_run_cb.isChecked()
        self.sup.payload_kg = self.payload_spin.value()
        f = self.sup.speed_fraction
        self.speed_lbl.setText(f"Speed cap: {100 * f:.0f} % of rated = {np.degrees(self.sup.speed_cap[0]):.0f}°/s "
                               f"(rated {self.spec.rated_speed_deg_s[0]:.0f}°/s, rule 3)")
        lo, hi = np.degrees(self.sup.soft_limits).T
        self.auto_lbl.setText(
            f"✓ software limits ±{hi[0]:.0f}° (J6 ±{hi[-1]:.0f}°) = hardware − 10° (rule 1)\n"
            f"✓ speed capped at {100 * f:.0f} % of rated (rule 3)\n"
            + (f"✓ payload {self.sup.payload_kg:.3f} kg ≤ 0.125 kg (rule 4)" if not self.sup.static_problems()
               else "✗ " + "; ".join(self.sup.static_problems())))
        self._update_plan()
        self._update_ready()

    def ticked(self) -> dict:
        return {k: cb.isChecked() for k, cb in self.check_boxes.items()}

    def _update_ready(self, *_):
        n_ok = sum(self.ticked().values())
        ok = n_ok == len(self.check_boxes) and not self.sup.static_problems()
        self.ready_lbl.setText(f"{n_ok} of {len(self.check_boxes)} confirmed — " +
                               ("ready to run" if ok else "Run is blocked until every item is ticked"))
        self.ready_lbl.setStyleSheet(f"color:{theme.GOOD if ok else theme.WARN}")
        self._update_buttons()

    def _update_buttons(self):
        running = self.run is not None and not self.run.finished
        connected = self.driver is not None and self.driver.connected
        latched = connected and self.driver.latched
        ready = all(self.ticked().values()) and not self.sup.static_problems()
        self.run_btn.setEnabled(connected and not running and not latched and ready)
        self.stop_btn.setEnabled(True)                     # always active
        self.reset_btn.setEnabled(latched and not running)
        self.torque_btn.setEnabled(latched and not running)
        self.torque_btn.setText("Torque on" if connected and not self.driver.torque_enabled else "Torque off")
        self.save_btn.setEnabled(self.last_log is not None and len(self.last_log) > 0)
        self.connect_btn.setEnabled(not running)
        self.backend_box.setEnabled(not running and not connected)
        for w in (self.session_box, self.first_run_cb, self.payload_spin, self.source_box, self.sliders):
            w.setEnabled(not running and (w is not self.first_run_cb or self.sup.session != "H1"))

    # ================================================================== trajectory
    def _source_changed(self, *_):
        src = self.source_box.currentIndex()
        self.csv_btn.setEnabled(src == 2)
        self._update_ab()
        self._update_plan()

    def _target(self) -> np.ndarray:
        return self.sliders.values()

    def _set_target(self, q):
        self.sliders.set_values(q)
        self._target_changed()

    def _target_changed(self, *_):
        if self.run is None or self.run.finished:
            self.ghost.update(self._target())
        self._update_plan()

    def _set_a(self):
        self.pose_a = self._target()
        self._update_ab()
        self._update_plan()

    def _set_b(self):
        self.pose_b = self._target()
        self._update_ab()
        self._update_plan()

    def _update_ab(self):
        fmt = lambda q: "(" + ", ".join(f"{x:.0f}" for x in np.degrees(q)) + ")°"
        self.ab_lbl.setText(f"A = {fmt(self.pose_a)}\nB = {fmt(self.pose_b)}")

    def load_csv(self, path: str, fmt: str = "361") -> bool:
        try:
            t, q = load_trajectory(path, fmt)
        except Exception as e:
            self.csv_lbl.setText(f"could not read {os.path.basename(path)}: {e}")
            self.csv_lbl.setStyleSheet(f"color:{theme.BAD}")
            return False
        if q.shape[1] != self.spec.n:
            self.csv_lbl.setText(f"{os.path.basename(path)} has {q.shape[1]} joints; the course arm has 6")
            self.csv_lbl.setStyleSheet(f"color:{theme.BAD}")
            return False
        self.csv_traj = (t, q)
        self.csv_lbl.setText(f"{os.path.basename(path)}: {len(t)} samples, {t[-1]:.2f} s ({FORMATS[fmt]})")
        self.csv_lbl.setStyleSheet("")
        self.source_box.setCurrentIndex(2)
        self._update_plan()
        return True

    def _fmt_key(self) -> str:
        return list(FORMATS)[self.fmt_box.currentIndex()]

    def _load_csv_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load joint log / trajectory", "", "CSV (*.csv)")
        if fn:
            self.load_csv(fn, self._fmt_key())

    def plan(self):
        """(t, q, note) of the trajectory that Run would send, starting from the measured pose."""
        cap = self.sup.speed_cap
        q_now = self.q_meas
        src = self.source_box.currentIndex()
        note = ""
        if src == 0:
            t, q = quintic_move(q_now, self._target(), cap)
            return t, q, "quintic move from the measured pose"
        if src == 1:
            t, q = quintic_move(self.pose_a, self.pose_b, cap)
            t, q = concat((t, q), (t, q[::-1]), pause=0.5)        # A -> B -> A
            note = "quintic A → B → A"
        else:
            if self.csv_traj is None:
                return None
            t, q = self.csv_traj
            t, k = retime_to_speed(t, q, 0.8 * cap)
            note = f"CSV trajectory" + (f", slowed down ×{k:.2f} to respect the speed cap" if k > 1 else "")
        t, q, approached = with_approach(t, q, q_now, cap)
        if approached:
            note += " (+ quintic approach from the measured pose)"
        return t, q, note

    def _update_plan(self, *_):
        if not hasattr(self, "plan_lbl"):
            return
        p = self.plan()
        if p is None:
            self.plan_lbl.setText("Load a CSV file first.")
            return
        t, q, note = p
        from ...hardware.safety import peak_speeds
        pk = np.degrees(peak_speeds(t, q).max())
        probs = self.sup.trajectory_problems(t, q)
        self.plan_lbl.setText(f"{note}: {t[-1]:.2f} s, peak {pk:.1f}°/s"
                              + ("\n✗ " + "\n✗ ".join(probs) if probs else ""))
        self.plan_lbl.setStyleSheet(f"color:{theme.BAD}" if probs else "")

    # ================================================================== run
    def start_run(self) -> bool:
        if self.driver is None or not self.driver.connected:
            self._status("Connect a back-end first.", theme.WARN)
            return False
        if self.driver.latched:
            self._status("The arm is in latched stop — press Reset latch first.", theme.WARN)
            return False
        try:
            self.q_meas = self.driver.read_joints()
        except HardwareError as e:
            self._status(f"read failed: {e}", theme.BAD)
            return False
        p = self.plan()
        if p is None:
            self._status("No trajectory: load a CSV file first.", theme.WARN)
            return False
        t, q, note = p
        ok, probs = self.sup.preflight(self.ticked(), t, q)
        if not ok:
            self._status("Run blocked:\n• " + "\n• ".join(probs), theme.BAD)
            return False
        self.poll.stop()
        self.run = HardwareRun(self.driver, self.sup, t, q, auto_stop_at=self.auto_stop.value() or None)
        self.last_log = self.run.log
        self.report.setRowCount(0)
        self.report_lbl.setText("running…")
        for ln in (self.stop_line_q, self.stop_line_e):
            ln.setVisible(False)
        self._status(f"Running {note} — {t[-1]:.1f} s at ≤ {100 * self.sup.speed_fraction:.0f} % rated speed. "
                     "STOP (or Esc) halts and latches.", theme.ACCENT)
        self.run.start()
        self.timer.start(_RATE_MS)
        self._update_buttons()
        return True

    def _tick(self):
        if self.run is None:
            self.timer.stop()
            return
        try:
            r = self.run.step()
        except HardwareError as e:
            self.run.request_stop(f"communication error: {e}")
            self._status(f"Communication error — stopped and latched: {e}", theme.BAD)
            r = None
        if r is not None:
            _, cmd, meas = r
            self.q_meas = meas
            self._update_scene(meas, cmd)
            if len(self.run.log) % 3 == 0:
                self._redraw_plots()
        if self.run.finished:
            self._finish_run()

    def _finish_run(self):
        self.timer.stop()
        run = self.run
        self._redraw_plots()
        if run.t_stop is not None:
            self._status(f"Stopped ({run.reason}) at t = {run.t_stop:.2f} s — arm LATCHED. "
                         "Reset latch before the next run.", theme.WARN)
        else:
            self._status(f"Run complete: {len(run.log)} samples, {run.log.t[-1]:.2f} s.", theme.GOOD)
        self.fill_report()
        if self.isVisible() and self.driver is not None:
            self.poll.start(_POLL_MS)
        self._update_buttons()

    def stop_now(self):
        """STOP button: halt + latch, whatever the state."""
        if self.run is not None and not self.run.finished:
            self.run.request_stop("STOP pressed")
        elif self.driver is not None:
            self.driver.stop()
        self._status("STOP — arm halted and latched. Reset latch to continue.", theme.BAD)
        self._update_buttons()

    def reset_latch(self):
        if self.driver is None:
            return
        self.driver.reset_latch()
        self._status("Latch reset — torque on. Check the keep-out zone before the next run.", theme.WARN)
        self._update_buttons()

    def _toggle_torque(self):
        if self.driver is None:
            return
        try:
            self.driver.set_torque(not self.driver.torque_enabled)
        except HardwareError as e:
            self._status(str(e), theme.BAD)
        else:
            self._status("Torque OFF — the arm may sag under gravity; a hand may enter with the safety "
                         "officer's approval (rule 6)."
                         if not self.driver.torque_enabled else "Torque on (still latched).", theme.WARN)
        self._update_buttons()

    def _status(self, txt, color=None):
        self.status_lbl.setText(txt)
        self.status_lbl.setStyleSheet(f"color:{color}" if color else "")

    # ================================================================== plots / report
    def _redraw_plots(self, *_):
        lg = self.last_log
        if lg is None or len(lg) == 0:
            for c in self.c_cmd + self.c_meas + self.c_err:
                c.setData([], [])
            return
        t, cmd, meas = lg.t, np.degrees(lg.cmd), np.degrees(lg.meas)
        sel = self.joint_box.currentIndex() - 1
        for i in range(self.spec.n):
            vis = sel < 0 or sel == i
            self.c_cmd[i].setData(t, cmd[:, i]) if vis else self.c_cmd[i].setData([], [])
            self.c_meas[i].setData(t, meas[:, i]) if vis else self.c_meas[i].setData([], [])
            self.c_err[i].setData(t, meas[:, i] - cmd[:, i]) if vis else self.c_err[i].setData([], [])
        ts = lg.stop_time
        for ln in (self.stop_line_q, self.stop_line_e):
            ln.setVisible(ts is not None)
            if ts is not None:
                ln.setValue(ts)

    def fill_report(self):
        lg = self.last_log
        if lg is None or len(lg) < 3:
            return {}
        res = hwm.session_report(lg, self.robot, self.spec, self.sup.session)
        rows = hwm.report_rows(res, self.sup.session)
        self.report.setRowCount(len(rows))
        for r, (name, val, thr, ok) in enumerate(rows):
            for c, txt in enumerate((name, val, thr, "PASS" if ok else "FAIL")):
                it = QtWidgets.QTableWidgetItem(txt)
                if c:
                    it.setTextAlignment(QtCore.Qt.AlignCenter)
                if c == 3:
                    it.setForeground(QtGui.QColor(theme.GOOD if ok else theme.BAD))
                self.report.setItem(r, c, it)
        self.report.resizeRowsToContents()
        n_fail = sum(not ok for *_, ok in rows)
        extra = ""
        if lg.stop_time is not None:
            extra = f" Stop command at t = {lg.stop_time:.2f} s (461 stop_time t_trigger)."
        else:
            extra = " No stop in this run: set 'Stop test at t =' for HW-6."
        # delay of the joint that moves most (a joint at rest has no correlation peak), before any stop
        k0 = int(np.argmax(lg.stop)) if lg.stop.any() else len(lg)
        span = np.ptp(lg.cmd[:k0], axis=0) if k0 > 0 else np.zeros(self.spec.n)
        j = int(np.argmax(span))
        if k0 > 20 and span[j] > np.radians(1.0):
            delay = hwm.estimate_delay(lg.t[:k0], lg.cmd[:k0, j], lg.meas[:k0, j])
            dtxt = f"Delay meas vs cmd (J{j + 1}): {1e3 * delay:.0f} ms."
        else:
            dtxt = "Delay: no joint moved enough to estimate it."
        self.report_lbl.setText(f"{len(rows) - n_fail}/{len(rows)} tests pass (session {self.sup.session}). "
                                f"{dtxt}{extra}")
        self.report_lbl.setStyleSheet(f"color:{theme.GOOD if n_fail == 0 else theme.WARN}")
        return res

    def save_log(self, path: str, fmt: str | None = None):
        if self.last_log is None:
            raise ValueError("nothing to save")
        self.last_log.save(path, fmt or self._fmt_key())
        self._status(f"Log saved: {path}", theme.GOOD)

    def _save_dialog(self):
        fmt = self._fmt_key()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save joint log", f"hw_{self.sup.session}_log.csv",
                                                      "CSV (*.csv)")
        if fn:
            self.save_log(fn, fmt)

    def shutdown(self):
        """Application quit: stop and latch the arm, then release the connection."""
        try:
            self.timer.stop()
            self.poll.stop()
            if self.driver is not None:
                self.driver.stop()
                self.driver.disconnect()
        except Exception:
            pass
        self.driver = None

    # ================================================================== show / hide
    def on_show(self):
        if self.driver is not None and (self.run is None or self.run.finished):
            self.poll.start(_POLL_MS)

    def on_hide(self):
        self.poll.stop()
        if self.run is not None and not self.run.finished:     # never move the arm while its STOP is hidden
            self.run.request_stop("tab left during a run")
