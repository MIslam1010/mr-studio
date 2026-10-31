"""Hardware bridge -- run the course arm (or its simulation) for the projects' required hardware sessions.

Streams a joint trajectory to the arm at 50 Hz, logs commanded vs measured joints in the CSV formats of
the SBEG 361 (hwcompare.py) and SBEG 461 (hardware.py) starters, enforces the course-arm hardware safety
checklist and computes the hardware acceptance tests HW-2 ... HW-6.

During a run the serial I/O happens on a worker thread (``_StreamWorker``) that owns the driver; the GUI
thread only draws the samples it publishes, so STOP is never queued behind a slow USB transfer.  With a
hand-advanced ``ManualClock`` (tests) the same ``HardwareRun.step`` is driven synchronously by ``_tick``.
"""
from __future__ import annotations

import atexit
import dataclasses
import functools
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...hardware import COURSE_ARM, HardwareError, ManualClock, SimulatedArm, course_arm_robot
from ...hardware import metrics as hwm
from ...hardware.logger import FORMATS, load_trajectory
from ...hardware.mycobot import (BAUD_RATES, DEFAULT_STREAM_SPEED_PCT, MyCobotCalibration, MyCobotDriver,
                                 default_calibration_path, list_serial_ports, sdk_available)
from ...hardware.safety import SESSION_LABELS, SESSIONS, SafetySupervisor, retime_to_speed
from ...hardware.session import HardwareRun, LatestValue, concat, quintic_move, with_approach
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
<h3>Run-time guards</h3>
<p>While streaming, each set-point may move at most $\dot{\theta}_{cap}\,\min(\Delta t, 40\ \mathrm{ms})$
from the previous one; a tick gap above 100 ms holds the trajectory time (no jump ahead after a stall);
$|\theta_{cmd}-\theta_{meas}|>10^{\circ}$ on any joint for more than 150 ms, three failed joint reads in a
row or any communication error stop and latch the arm.  On the myCobot every set-point is mapped with
the commissioned sign and offset, $\theta_{servo}=s\,\theta+o$, and clamped 10° inside the servo travel.</p>
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
_FRESH_MAX_S = 0.2     # Run refuses to start on joint readings older than this
_DEFAULT_PORTS = (["/dev/cu.usbserial-0001", "/dev/cu.wchusbserial0001"] if sys.platform == "darwin"
                  else ["COM3"] if sys.platform.startswith("win") else ["/dev/ttyUSB0", "/dev/ttyACM0"])


def _emergency_stop(driver):
    """atexit hook registered on connect: halt, latch and close the port if MR Studio exits uncleanly."""
    try:
        driver.stop()
    except Exception:
        pass
    try:
        driver.disconnect()
    except Exception:
        pass


class _StreamWorker(QtCore.QThread):
    """Streams one ``HardwareRun`` off the GUI thread.

    It owns the driver I/O for the run: every ``safe_step`` (set-point write + joint read) happens here at
    the stream rate, and each sample is published through a latest-value mailbox plus the
    ``sample_ready`` signal (the GUI draws only the newest).  STOP latches the driver from the GUI thread
    (the latch is checked under the driver's I/O lock before every write) and sets ``wake`` so the
    worker sends the halt and logs the deceleration without waiting for the next period."""
    sample_ready = QtCore.Signal()

    def __init__(self, run: HardwareRun, period_s: float, q_now, parent=None):
        super().__init__(parent)
        self.hw_run = run
        self.period = float(period_s)
        self.q_now = q_now
        self.mailbox = LatestValue()
        self.wake = threading.Event()

    def run(self):
        r = self.hw_run
        try:
            if not r.finished:
                r.start(self.q_now)
                self._publish()
            next_t = time.monotonic()
            while not r.finished:
                next_t += self.period
                delay = next_t - time.monotonic()
                if delay < 0:                       # overran: re-phase instead of bursting to catch up
                    next_t, delay = time.monotonic(), 0.0
                if self.wake.wait(delay):
                    self.wake.clear()
                r.safe_step()
                self._publish()
        except Exception as e:                      # safe_step never raises; belt and braces
            r.fail(e)

    def _publish(self):
        lg = self.hw_run.log
        if len(lg):
            t, cmd, meas = lg.snapshot()
            self.mailbox.put((t[-1], cmd[-1], meas[-1]))
            self.sample_ready.emit()


class HardwareTab(BaseTab):
    title = "⑫ Hardware"
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
        self.worker: _StreamWorker | None = None
        self.threaded = True                      # stream on a worker thread (off with a ManualClock)
        self.sdk_factory = None                   # tests inject a fake pymycobot class here
        self.calib_path = str(default_calibration_path())
        self.session_events: list[dict] = []
        self._atexit_cb = None
        self._n_samples = 0
        self.confirm = self._ask                   # (title, text) -> bool; tests replace it
        self.view = SceneView()

        # ---- connection ---------------------------------------------------------------------
        self.backend_box = combo(_BACKENDS, self._backend_changed)
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setEditable(True)
        self.port_box.addItems(list_serial_ports() or _DEFAULT_PORTS)
        self.port_box.setToolTip("USB-serial ports of the arm first (CP210x / CH340); on macOS use /dev/cu.*")
        self.baud_box = combo([str(b) for b in BAUD_RATES])
        self.baud_box.setToolTip("115200 for the M5 / Pi editions; some boards use 1000000 (see the arm's manual)")
        self.connect_btn = button("Connect", self._toggle_connect, checkable=True, primary=True)
        self.conn_lbl = value_label("not connected")
        self.health_lbl = value_label()
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

        # ---- myCobot calibration (commissioning) -----------------------------------------------
        self.sign_boxes, self.offset_spins = [], []
        cal_grid = QtWidgets.QGridLayout()
        cal_grid.addWidget(QtWidgets.QLabel("Joint"), 0, 0)
        cal_grid.addWidget(QtWidgets.QLabel("servo turns the other way"), 0, 1)
        cal_grid.addWidget(QtWidgets.QLabel("servo offset"), 0, 2)
        for j in range(self.spec.n):
            sb = checkbox("invert")
            sb.setToolTip("q_servo = −q_model + offset for this joint")
            sp = spin(-180.0, 180.0, 0.0, 1, 0.5, "°")
            sp.setToolTip("Servo angle [deg] when the model angle is 0 (q_servo = sign·q_model + offset)")
            self.sign_boxes.append(sb)
            self.offset_spins.append(sp)
            cal_grid.addWidget(QtWidgets.QLabel(f"J{j + 1}"), j + 1, 0)
            cal_grid.addWidget(sb, j + 1, 1)
            cal_grid.addWidget(sp, j + 1, 2)
        self.stream_pct = QtWidgets.QSpinBox()
        self.stream_pct.setRange(1, 100)
        self.stream_pct.setValue(DEFAULT_STREAM_SPEED_PCT)
        self.stream_pct.setSuffix(" %")
        self.stream_pct.setToolTip("Firmware speed % sent with every streamed set-point (independent of the "
                                   "session speed cap).  Measure its deg/s at commissioning.")
        self.stream_pct.valueChanged.connect(self._stream_pct_changed)
        self.calib_save_btn = button("Save calibration", lambda: self.save_calibration())
        self.calib_lbl = value_label()
        self.calib_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        theme.notifier().projector_changed.connect(self._retone)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Arm connection", hbox(QtWidgets.QLabel("Back-end"), self.backend_box),
                           hbox(QtWidgets.QLabel("Port"), self.port_box),
                           hbox(QtWidgets.QLabel("Baud"), self.baud_box), self.connect_btn,
                           self.conn_lbl, self.health_lbl, self.model_lbl))
        self.calib_group = group("myCobot calibration (commissioning)", cal_grid,
                                 hbox(QtWidgets.QLabel("Streaming speed (firmware)"), self.stream_pct),
                                 self.calib_save_btn, self.calib_lbl)
        cl.addWidget(self.calib_group)
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
        title = QtWidgets.QLabel("<b>Hardware test report</b> (thresholds of the SBEG 361/461 course briefs)")
        title.setToolTip("An example course setup: the course handout is authoritative for these thresholds.\n"
                         "They are defined in one place, DEFAULTS in mrstudio/hardware/metrics.py.")
        rl.addWidget(title)
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

        for cb_ in (self.backend_box, self.port_box, self.baud_box, self.session_box, self.source_box, self.fmt_box):
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
        self._load_calibration_ui()
        self._backend_changed(self.backend_box.currentText())
        self._source_changed(self.source_box.currentText())
        self._safety_changed()
        self._update_scene(self.q_meas, None)
        self._update_buttons()

    # ================================================================== scene

    def _tone(self, lbl, tone):
        """Colour a label with the current palette's good/warn/bad colour (None = default)."""
        lbl.setProperty("tone", tone)
        lbl.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")

    def _retone(self, *_):
        for lbl in (self.conn_lbl, self.calib_lbl, self.ready_lbl):
            self._tone(lbl, lbl.property("tone"))

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
        connected = self.driver is not None and self.driver.connected
        self.port_box.setEnabled(not sim and not connected)
        self.baud_box.setEnabled(not sim and not connected)
        self.calib_group.setVisible(not sim)
        if not sim and not sdk_available() and self.sdk_factory is None:
            self.conn_lbl.setText("pymycobot is not installed — install it with "
                                  "<tt>.venv/bin/python -m pip install pymycobot</tt> (or use the Simulated back-end).")
            self._tone(self.conn_lbl, "warn")
        elif not connected:
            self.conn_lbl.setText("not connected")
            self._tone(self.conn_lbl, None)

    def _is_mycobot(self) -> bool:
        return isinstance(self.driver, MyCobotDriver)

    def _toggle_connect(self, on):
        if on:
            self.connect_backend()
        else:
            self.disconnect_backend()

    def _event(self, kind: str, msg: str = "") -> None:
        """Session event for the log sidecar (connect, torque, calibration, idle stops ...)."""
        ev = dict(t=None, wall=datetime.now().isoformat(timespec="milliseconds"), kind=kind, msg=msg)
        self.session_events.append(ev)
        if self.last_log is not None:
            self.last_log.events.append(ev)

    def connect_backend(self) -> bool:
        self.disconnect_backend()
        if self.backend_box.currentText() == _BACKENDS[0]:
            drv = SimulatedArm(self.spec, q0=self.q_meas)
        else:
            cal = self._load_calibration()
            drv = MyCobotDriver(self.port_box.currentText().strip(), int(self.baud_box.currentText()), spec=self.spec,
                                sdk_factory=self.sdk_factory, calibration=cal,
                                stream_speed_pct=self.stream_pct.value())
            if cal is not None:
                drv.calibration_path = self.calib_path
        try:
            drv.connect()
            self.q_meas = drv.read_joints()
        except Exception as e:      # HardwareError, or anything a driver bug lets through
            try:
                drv.disconnect()
            except Exception:
                pass
            self.conn_lbl.setText(str(e).replace("\n", "<br>"))
            self._tone(self.conn_lbl, "bad")
            self.connect_btn.blockSignals(True)
            self.connect_btn.setChecked(False)
            self.connect_btn.blockSignals(False)
            self._event("connect-failed", str(e))
            self._update_buttons()
            return False
        self.driver = drv
        # an unclean exit (exception, window killed, Ctrl-C) must still halt the arm and close the port
        self._atexit_cb = functools.partial(_emergency_stop, drv)
        atexit.register(self._atexit_cb)
        where = f" on {drv.port} at {drv.baud} baud" if hasattr(drv, "port") else ""
        self._event("connect", f"{drv.name}{where}")
        self.conn_lbl.setText(f"connected: {drv.name}{where}")
        self._tone(self.conn_lbl, "good")
        if getattr(drv, "offset_calibrated", True) is False:
            self.conn_lbl.setText(self.conn_lbl.text() + f"<br>⚠ no calibration file ({self.calib_path}): the "
                                  "servo signs and zero offsets are unknown, so Run is blocked. Compare the measured "
                                  "pose with the 3-D view, set the signs and offsets below and Save.")
            self._tone(self.conn_lbl, "warn")
        self.connect_btn.blockSignals(True)
        self.connect_btn.setChecked(True)
        self.connect_btn.setText("Disconnect")
        self.connect_btn.blockSignals(False)
        self._backend_changed(self.backend_box.currentText())
        self.status_lbl.setText("Connected. Tick the checklist, choose a trajectory, then Run.")
        self.status_lbl.setStyleSheet("")
        if self.isVisible():
            self.poll.start(_POLL_MS)
        self._update_health()
        self._update_scene(self.q_meas, None)
        self._update_plan()
        self._update_buttons()
        return True

    def disconnect_backend(self):
        if self.run is not None and not self.run.finished:
            self.stop_now()
            self._join_worker()
            self.run.finished = True          # a worker that did not end in time is abandoned here
            self._finish_run()
        self.poll.stop()
        if self.driver is not None:
            try:
                self.driver.disconnect()
            except Exception:
                pass
            self._event("disconnect")
        self._unregister_atexit()
        self.driver = None
        self.connect_btn.blockSignals(True)
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("Connect")
        self.connect_btn.blockSignals(False)
        self._backend_changed(self.backend_box.currentText())
        self._update_health()
        self._update_buttons()

    def _unregister_atexit(self):
        if self._atexit_cb is not None:
            atexit.unregister(self._atexit_cb)
            self._atexit_cb = None

    def _poll(self):
        if self.driver is None or (self.run is not None and not self.run.finished):
            return
        try:
            self.q_meas = self.driver.read_joints()
        except Exception as e:
            self.status_lbl.setText(f"read failed: {e}")
            self._update_health()
            return
        self._update_health()
        self._update_scene(self.q_meas, None)

    def _update_health(self):
        d = self.driver
        if d is None:
            self.health_lbl.setText("")
            return
        age = d.fresh_age()
        age_txt = "never" if not np.isfinite(age) else f"{1e3 * age:.0f} ms ago"
        txt = f"Joint readings: last fresh {age_txt} · {d.read_failures} failed reads"
        if getattr(d, "clamp_count", 0):
            txt += f" · {d.clamp_count} set-points clamped at the servo limits"
        bad = d.consecutive_read_failures > 0 or age > _FRESH_MAX_S
        self.health_lbl.setText(txt)
        self.health_lbl.setStyleSheet(f"color:{theme.WARN}" if bad else "")

    # ------------------------------------------------------------------ calibration
    def calibration_from_ui(self) -> MyCobotCalibration:
        return MyCobotCalibration(sign=[-1.0 if cb.isChecked() else 1.0 for cb in self.sign_boxes],
                                  offset_deg=[sp.value() for sp in self.offset_spins])

    def calibration_exists(self) -> bool:
        return self._load_calibration() is not None

    def _load_calibration(self) -> MyCobotCalibration | None:
        if not Path(self.calib_path).is_file():
            return None
        try:
            return MyCobotCalibration.load(self.calib_path)
        except HardwareError:
            return None

    def _load_calibration_ui(self):
        cal = self._load_calibration()
        if cal is not None and len(cal.sign) == self.spec.n:
            for cb, sg, sp, off in zip(self.sign_boxes, cal.sign, self.offset_spins, cal.offset_deg):
                cb.setChecked(bool(sg < 0))
                sp.setValue(float(off))
            self.calib_lbl.setText(f"Loaded {self.calib_path}")
            self._tone(self.calib_lbl, None)
        elif Path(self.calib_path).exists():
            self.calib_lbl.setText(f"✗ {self.calib_path} is unreadable — fix the values and Save.")
            self._tone(self.calib_lbl, "bad")
        else:
            self.calib_lbl.setText(f"No calibration file yet ({self.calib_path}). Run on the myCobot is blocked "
                                   "until the signs and offsets are commissioned and saved.")
            self._tone(self.calib_lbl, "warn")

    def save_calibration(self, path: str | None = None) -> bool:
        """Write the calibration of the UI to the JSON file and apply it to a connected myCobot."""
        if self.run is not None and not self.run.finished:
            return False
        if path is not None:
            self.calib_path = str(path)
        try:
            cal = self.calibration_from_ui()
            cal.save(self.calib_path)
        except (OSError, ValueError) as e:
            self.calib_lbl.setText(f"✗ could not save: {e}")
            self._tone(self.calib_lbl, "bad")
            return False
        self.calib_lbl.setText(f"Saved {self.calib_path}")
        self._tone(self.calib_lbl, "good")
        self._event("calibration", f"saved {self.calib_path}: sign {cal.sign.astype(int).tolist()}, "
                                   f"offset {cal.offset_deg.tolist()} deg")
        if self._is_mycobot():
            self.driver.set_calibration(cal, self.calib_path)
            self.driver.stream_speed_pct = int(self.stream_pct.value())
            try:
                self.q_meas = self.driver.read_joints()
            except Exception as e:
                self._status(f"read failed: {e}", theme.BAD)
            self._update_scene(self.q_meas, None)
            self._update_plan()
        self._update_buttons()
        return True

    def _stream_pct_changed(self, v):
        if self._is_mycobot():
            self.driver.stream_speed_pct = int(v)

    def _ask(self, title: str, text: str) -> bool:
        box = QtWidgets.QMessageBox.warning(self, title, text,
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                                            QtWidgets.QMessageBox.Cancel)
        return box == QtWidgets.QMessageBox.Yes

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
        self._tone(self.ready_lbl, "good" if ok else "warn")
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
        if connected and self._is_mycobot() and not self.driver.offset_calibrated:
            self.run_btn.setEnabled(False)                 # no commissioning record: never move the real arm
        # file dialogs are modal: never open one while the arm moves (STOP must stay reachable)
        self.save_btn.setEnabled(not running and self.last_log is not None and len(self.last_log) > 0)
        self.csv_btn.setEnabled(not running and self.source_box.currentIndex() == 2)
        self.fmt_box.setEnabled(not running)
        self.calib_group.setEnabled(not running)
        self.connect_btn.setEnabled(not running)
        self.backend_box.setEnabled(not running and not connected)
        for w in (self.session_box, self.first_run_cb, self.payload_spin, self.source_box, self.sliders):
            w.setEnabled(not running and (w is not self.first_run_cb or self.sup.session != "H1"))

    # ================================================================== trajectory
    def _source_changed(self, *_):
        self._update_ab()
        self._update_plan()
        if hasattr(self, "timer"):
            self._update_buttons()

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
        except Exception as e:
            self._status(f"read failed: {e}", theme.BAD)
            self._update_health()
            return False
        self._update_health()
        blocked = []
        age = self.driver.fresh_age()
        if age > _FRESH_MAX_S:
            blocked.append(f"joint readings are stale (last fresh reading "
                           f"{'never' if not np.isfinite(age) else f'{1e3 * age:.0f} ms ago'}, "
                           f"{self.driver.read_failures} failed reads) — check power and cable")
        if self._is_mycobot() and not (self.calibration_exists() and self.driver.offset_calibrated):
            blocked.append(f"no myCobot calibration file ({self.calib_path}): commission the signs and "
                           "offsets in 'myCobot calibration' and Save")
        p = self.plan()
        if p is None:
            self._status("No trajectory: load a CSV file first.", theme.WARN)
            return False
        t, q, note = p
        ok, probs = self.sup.preflight(self.ticked(), t, q)
        probs = blocked + probs
        if probs:
            self._status("Run blocked:\n• " + "\n• ".join(probs), theme.BAD)
            return False
        self.poll.stop()
        self.run = HardwareRun(self.driver, self.sup, t, q, auto_stop_at=self.auto_stop.value() or None)
        self.run.log.events.extend(self.session_events)
        self.run.log.meta = dict(self.driver.describe(), session=self.sup.session,
                                 speed_fraction=self.sup.speed_fraction, rate_hz=1000 / _RATE_MS, trajectory=note)
        self.last_log = self.run.log
        self._n_samples = 0
        self.report.setRowCount(0)
        self.report_lbl.setText("running…")
        for ln in (self.stop_line_q, self.stop_line_e):
            ln.setVisible(False)
        self._status(f"Running {note} — {t[-1]:.1f} s at ≤ {100 * self.sup.speed_fraction:.0f} % rated speed. "
                     "STOP (or Esc) halts and latches.", theme.ACCENT)
        if self.threaded and not isinstance(getattr(self.driver, "clock", None), ManualClock):
            # real time: the worker owns the driver I/O for the whole run
            self.worker = _StreamWorker(self.run, _RATE_MS / 1000.0, self.q_meas, self)
            self.worker.sample_ready.connect(self._on_sample)
            self.worker.finished.connect(self._worker_done)
            self.worker.start(QtCore.QThread.TimeCriticalPriority)
        else:
            # hand-advanced clock (tests): the same step, driven synchronously by _tick
            self.run.start(self.q_meas)
            self.timer.start(_RATE_MS)
        self._update_buttons()
        return True

    def _show_sample(self, cmd, meas):
        self.q_meas = meas
        self._update_scene(meas, cmd)
        self._n_samples += 1
        if self._n_samples % 3 == 0:
            self._redraw_plots()
        if self._n_samples % 10 == 0:
            self._update_health()

    def _on_sample(self):
        """Worker published a sample (queued to the GUI thread): draw the newest one."""
        s = self.worker.mailbox.take() if self.worker is not None else None
        if s is not None:
            try:
                self._show_sample(s[1], s[2])
            except Exception:          # a drawing problem must never interfere with the run
                pass

    def _worker_done(self):
        if self.run is not None and self.worker is not None and self.sender() is self.worker:
            self._finish_run()

    def _join_worker(self, timeout_ms: int = 3000) -> bool:
        w = self.worker
        if w is None:
            return True
        w.wake.set()
        done = w.wait(timeout_ms) if w.isRunning() else True
        if done:
            w.deleteLater()
            self.worker = None
        return done

    def _tick(self):
        """Synchronous stream step (QTimer with a ManualClock, and the tests)."""
        if self.run is None:
            self.timer.stop()
            return
        try:
            r = self.run.step()
            if r is not None:
                self._show_sample(r[1], r[2])
        except Exception as e:      # serial or any other error: latch, end the run, tell the user
            self.run.fail(e)
        if self.run.finished:
            self._finish_run()

    def _finish_run(self):
        self.timer.stop()
        if self.worker is not None and not self.worker.isRunning():
            self.worker.deleteLater()
            self.worker = None
        run = self.run
        self._redraw_plots()
        self._update_health()
        if run.stop_confirmed is False:
            self._stop_not_confirmed(run.error or run.reason)
        elif run.error:
            self._status(f"Communication error — arm stopped and LATCHED: {run.error}", theme.BAD)
        elif run.t_stop is not None:
            self._status(f"Stopped ({run.reason}) at t = {run.t_stop:.2f} s — arm LATCHED. "
                         "Reset latch before the next run.", theme.WARN)
        elif len(run.log) == 0:
            self._status(f"Stopped before the first set-point ({run.reason}) — arm LATCHED.", theme.WARN)
        else:
            self._status(f"Run complete: {len(run.log)} samples, {run.log.t[-1]:.2f} s."
                         + (f" {run.stalls} timing stall(s): trajectory time was held." if run.stalls else ""),
                         theme.GOOD)
        self.fill_report()
        if self.isVisible() and self.driver is not None:
            self.poll.start(_POLL_MS)
        self._update_buttons()

    def _stop_not_confirmed(self, why: str = ""):
        self.status_lbl.setText("<b>STOP NOT CONFIRMED — switch off the arm's power now.</b><br>"
                                "The halt command failed twice" + (f" ({why})" if why else "") +
                                ". The software latch blocks further set-points, but the arm may still be moving.")
        self.status_lbl.setStyleSheet(f"color:{theme.BAD}; font-weight:700")

    def stop_now(self) -> bool:
        """STOP button: halt + latch, whatever the state.  Returns whether the halt was confirmed."""
        if self.run is not None and not self.run.finished:
            ok = self.run.request_stop("STOP pressed")
            if self.worker is not None:
                self.worker.wake.set()
        elif self.driver is not None:
            try:
                ok = bool(self.driver.stop())
            except Exception:
                ok = False
            self._event("stop", "STOP pressed (idle)" + ("" if ok else " — HALT NOT CONFIRMED"))
        else:
            ok = True
        if ok:
            self._status("STOP — arm halted and latched. Reset latch to continue.", theme.BAD)
        else:
            self._stop_not_confirmed()
        self._update_buttons()
        return ok

    def reset_latch(self):
        if self.driver is None:
            return
        try:
            self.driver.reset_latch()
        except Exception as e:
            self._status(f"Reset failed (still latched): {e}", theme.BAD)
        else:
            self._event("reset-latch")
            self._status("Latch reset — torque on. Check the keep-out zone before the next run.", theme.WARN)
        self._update_buttons()

    def _toggle_torque(self):
        if self.driver is None:
            return
        on = not self.driver.torque_enabled
        if not on and not self.confirm(
                "Release servo torque?",
                "Support the arm by hand before you continue.\n\nWith torque off the servos let go: the arm "
                "drops under gravity (J2 and J3 first).  Is someone holding the arm?"):
            self._status("Torque release cancelled — torque still on.", theme.WARN)
            return
        try:
            self.driver.set_torque(on)
        except Exception as e:
            self._status(f"Torque {'on' if on else 'off'} failed: {e}", theme.BAD)
        else:
            self._event("torque", "on" if on else "off")
            self._status("Torque OFF — keep supporting the arm; a hand may enter with the safety "
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
        t, cmd, meas = lg.snapshot()               # the worker may be appending right now
        cmd, meas = np.degrees(cmd), np.degrees(meas)
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
        """The log in the chosen course format plus the MR Studio sidecars (.meta.json, .diag.csv)."""
        if self.last_log is None:
            raise ValueError("nothing to save")
        if self.run is not None and not self.run.finished:
            raise ValueError("a run is in progress")
        fmt = fmt or self._fmt_key()
        self.last_log.save(path, fmt)
        if self.driver is not None:
            self.last_log.meta.update(self.driver.describe())
        self.last_log.meta["format"] = fmt
        meta_p, diag_p = self.last_log.save_sidecars(path)
        self._status(f"Log saved: {path}\n(+ {meta_p.name}, {diag_p.name})", theme.GOOD)

    def _save_dialog(self):
        fmt = self._fmt_key()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save joint log", f"hw_{self.sup.session}_log.csv",
                                                      "CSV (*.csv)")
        if fn:
            self.save_log(fn, fmt)

    def shutdown(self):
        """Window close / application quit: halt and latch the arm, end the streaming worker, close the
        port.  Idempotent and never raises (also wired to ``aboutToQuit``)."""
        drv = self.driver
        try:
            self.timer.stop()
            self.poll.stop()
            if self.run is not None and not self.run.finished:
                self.run.request_stop("MR Studio closing")
            elif drv is not None:
                drv.stop()
        except Exception:
            pass
        try:
            self._join_worker()
        except Exception:
            pass
        try:
            if drv is not None:
                drv.disconnect()
        except Exception:
            pass
        self._unregister_atexit()
        self.driver = None

    # ================================================================== show / hide
    def on_show(self):
        if self.driver is not None and (self.run is None or self.run.finished):
            self.poll.start(_POLL_MS)

    def on_hide(self):
        self.poll.stop()
        if self.run is not None and not self.run.finished:     # never move the arm while its STOP is hidden
            self.run.request_stop("tab left during a run")
            if self.worker is not None:
                self.worker.wake.set()
