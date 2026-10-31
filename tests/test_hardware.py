"""Hardware bridge: SimulatedArm, safety checklist, log formats, HW tests, run loop and the GUI tab.

Uses SimulatedArm only (with a ManualClock, so runs are deterministic and instantaneous).  Where the
course starters are present on this machine, the log formats and the HW tests are compared with the
starters' own functions (hwcompare.py for SBEG 361, sbeg461_project/hardware.py for SBEG 461).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio.hardware import COURSE_ARM, COURSE_ARM_NAME, HardwareError, ManualClock, SimulatedArm, course_arm_robot
from mrstudio.hardware import logger as L
from mrstudio.hardware import metrics as M
from mrstudio.hardware.mycobot import MyCobotDriver, sdk_available
from mrstudio.hardware.safety import (CHECKLIST, SafetySupervisor, allowed_speed_fraction, keepout_radius,
                                      retime_to_speed, soft_limits)
from mrstudio.hardware.session import quintic_duration, quintic_move, run_offline, with_approach

from course_paths import newest_course_file

# Override with MRSTUDIO_COURSE_SRC=<path to _build/src>; by default the newest course-materials copy of each starter
# (tests/course_paths.py). The starter comparisons are skipped when it is absent.
if os.environ.get("MRSTUDIO_COURSE_SRC"):
    _361 = Path(os.environ["MRSTUDIO_COURSE_SRC"]) / "361/project/starter"
    _461 = Path(os.environ["MRSTUDIO_COURSE_SRC"]) / "461/project/starter"
else:
    _361 = newest_course_file("src/361/project/starter/hwcompare.py").parent
    _461 = newest_course_file("src/461/project/starter/sbeg461_project/hardware.py").parent.parent


def _hwcompare():
    if not (_361 / "hwcompare.py").exists():
        pytest.skip("SBEG 361 starter not on this machine")
    sys.path.insert(0, str(_361))
    try:
        spec = importlib.util.spec_from_file_location("hwcompare_starter", _361 / "hwcompare.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(_361))


def _hw461():
    if not (_461 / "sbeg461_project/hardware.py").exists():
        pytest.skip("SBEG 461 starter not on this machine")
    sys.path.insert(0, str(_461))
    try:
        import sbeg461_project.hardware as h
        return h
    except Exception as e:      # the package imports its own dependencies
        pytest.skip(f"cannot import sbeg461_project: {e}")
    finally:
        sys.path.remove(str(_461))


def _arm(**kw):
    clock = ManualClock()
    arm = SimulatedArm(clock=clock, **kw)
    arm.connect()
    return arm, clock


# ------------------------------------------------------------------------------------- SimulatedArm
def test_simulated_arm_reads_quantised_home():
    arm, _ = _arm()
    q = arm.read_joints()
    assert q.shape == (6,)
    np.testing.assert_allclose(q / COURSE_ARM.enc_res, np.round(q / COURSE_ARM.enc_res), atol=1e-9)
    assert np.max(np.abs(q - COURSE_ARM.q_home)) <= COURSE_ARM.enc_res / 2 + 1e-12


def test_simulated_arm_speed_saturation_and_limits():
    arm, clock = _arm(noise_deg=0, quantise=False)
    target = COURSE_ARM.q_home + np.radians(90)
    arm.send_joints(target, 0.25)
    clock.advance(0.1)
    moved = np.abs(arm.true_position() - COURSE_ARM.q_home)
    np.testing.assert_allclose(moved, 0.25 * COURSE_ARM.rated_speed * 0.1, rtol=1e-6)   # saturated
    arm.send_joints(np.full(6, np.radians(300)), 1.0)
    clock.advance(10.0)
    assert np.all(arm.true_position() <= COURSE_ARM.hw_limits[:, 1] + 1e-12)


def test_simulated_arm_first_order_step():
    arm, clock = _arm(noise_deg=0, quantise=False, q0=np.zeros(6))
    arm.send_joints(np.full(6, 0.01), 1.0)          # small step: never saturates
    clock.advance(arm.lag_tau)
    np.testing.assert_allclose(arm.true_position(), 0.01 * (1 - np.exp(-1)), rtol=1e-9)   # MR Eq. 11.7


def test_latch_blocks_motion_and_torque_rules():
    arm, clock = _arm(noise_deg=0, quantise=False)
    with pytest.raises(HardwareError):
        arm.set_torque(False)                       # only in latched stop
    arm.send_joints(COURSE_ARM.q_home + 0.5, 0.5)
    clock.advance(0.2)
    arm.stop()
    assert arm.latched
    clock.advance(2.0)
    q_rest = arm.true_position()
    assert arm.send_joints(COURSE_ARM.q_home, 0.5) is False
    clock.advance(1.0)
    np.testing.assert_allclose(arm.true_position(), q_rest, atol=1e-12)
    arm.set_torque(False)
    assert not arm.torque_enabled
    arm.reset_latch()
    assert not arm.latched and arm.torque_enabled
    assert arm.send_joints(COURSE_ARM.q_home, 0.5) is True


# ------------------------------------------------------------------------------------- safety
def test_checklist_values_match_course_robots():
    assert allowed_speed_fraction("H1", first_run=False) == 0.25
    assert allowed_speed_fraction("H2", first_run=True) == 0.25
    assert allowed_speed_fraction("H2", first_run=False) == 0.5
    assert allowed_speed_fraction("lab") == 0.5
    sl = np.degrees(soft_limits(COURSE_ARM.hw_limits))
    np.testing.assert_allclose(sl[:5], [[-155, 155]] * 5)
    np.testing.assert_allclose(sl[5], [-165, 165])
    with pytest.raises(ValueError):
        soft_limits(COURSE_ARM.hw_limits, 5.0)
    assert keepout_radius(COURSE_ARM.reach) == pytest.approx(0.58)
    sup = SafetySupervisor(payload_kg=0.13)
    assert sup.static_problems()
    sup.payload_kg = 0.125
    assert not sup.static_problems()
    keys = {k for k, _ in CHECKLIST}
    assert {"induction", "keepout", "clear", "estop", "supervisor"} <= keys


def test_preflight_blocks_until_ticked_and_checks_trajectory():
    sup = SafetySupervisor(session="H2", first_run=False)
    t, q = quintic_move(COURSE_ARM.q_home, COURSE_ARM.q_home + 0.4, sup.speed_cap)
    ok, probs = sup.preflight({}, t, q)
    assert not ok and len(probs) == len(CHECKLIST)
    ticked = {k: True for k, _ in CHECKLIST}
    assert sup.preflight(ticked, t, q) == (True, [])
    # too fast for H1
    sup.session = "H1"
    ok, probs = sup.preflight(ticked, t, q)
    assert not ok and any("rule 3" in p for p in probs)
    # outside software limits
    q2 = q.copy()
    q2[:, 0] = np.radians(160)
    ok, probs = sup.preflight(ticked, t, q2)
    assert any("J1 leaves the software limits" in p for p in probs)


def test_quintic_duration_and_retime():
    cap = 0.25 * COURSE_ARM.rated_speed
    dq = np.radians([60, 0, 0, 0, 0, 0])
    T = quintic_duration(np.zeros(6), dq, cap, margin=1.0)
    assert T == pytest.approx(15 / 8 * dq[0] / cap[0])
    t = np.linspace(0, 1, 101)
    q = np.outer(t, np.full(6, 2.0))                 # 2 rad/s > cap
    t2, k = retime_to_speed(t, q, cap)
    assert k == pytest.approx(2.0 / cap[0])
    assert np.abs(np.gradient(q, t2, axis=0)).max() <= cap[0] + 1e-9


def test_with_approach_prepends_move():
    cap = 0.5 * COURSE_ARM.rated_speed
    t, q = quintic_move(COURSE_ARM.q_home + 0.3, COURSE_ARM.q_home + 0.6, cap)
    t2, q2, added = with_approach(t, q, COURSE_ARM.q_home, cap)
    assert added and np.allclose(q2[0], COURSE_ARM.q_home) and np.allclose(q2[-1], q[-1])
    assert np.all(np.diff(t2) >= 0)


# ------------------------------------------------------------------------------------- logs
def _demo_log():
    lg = L.HardwareLog(6)
    t = np.arange(0, 2, 0.02)
    for k, tk in enumerate(t):
        c = COURSE_ARM.q_home + 0.2 * np.sin(tk)
        lg.append(tk, c, c + 0.001 * np.cos(7 * tk), stop=tk >= 1.5)
    return lg


def test_log_formats_round_trip(tmp_path):
    lg = _demo_log()
    lg.save(tmp_path / "a.csv", "361")
    lg.save(tmp_path / "b.csv", "461")
    r1 = L.read_log(tmp_path / "a.csv", "361")
    r2 = L.read_log(tmp_path / "b.csv", "461")
    np.testing.assert_allclose(r1["cmd"], lg.cmd, atol=np.radians(5e-5) + 1e-12)
    np.testing.assert_array_equal(r1["stop"], lg.stop)
    np.testing.assert_array_equal(r2["meas"], lg.meas)
    header = (tmp_path / "a.csv").read_text().splitlines()[0]
    assert header == "t," + ",".join(f"cmd_{i}" for i in range(1, 7)) + "," + \
        ",".join(f"meas_{i}" for i in range(1, 7)) + ",stop"
    t, q = L.load_trajectory(tmp_path / "a.csv", "361")
    assert t[0] == 0 and q.shape == (len(lg), 6)


def test_361_format_identical_to_hwcompare(tmp_path):
    hc = _hwcompare()
    lg = _demo_log()
    d = lg.as_deg()
    hc.write_log(tmp_path / "theirs.csv", d["t"], d["cmd"], d["meas"], lg.stop)
    lg.save(tmp_path / "ours.csv", "361")
    assert (tmp_path / "ours.csv").read_bytes() == (tmp_path / "theirs.csv").read_bytes()


def test_461_format_identical_to_hardware_py(tmp_path):
    h = _hw461()
    lg = _demo_log()
    h.write_log(tmp_path / "theirs.csv", lg.t, lg.cmd, lg.meas)
    lg.save(tmp_path / "ours.csv", "461")
    assert (tmp_path / "ours.csv").read_bytes() == (tmp_path / "theirs.csv").read_bytes()
    back = h.load_log(tmp_path / "ours.csv")
    np.testing.assert_array_equal(back["cmd"], lg.cmd)


# ------------------------------------------------------------------------------------- metrics
def test_hw_report_identical_to_hwcompare():
    hc = _hwcompare()
    robot, _ = course_arm_robot()
    lg = hc.synthetic_log()
    lim = COURSE_ARM.hw_limits_deg
    kw = dict(limits_deg=lim, rated_speed_deg_s=COURSE_ARM.rated_speed_deg_s, Slist=robot.Slist, M=robot.M)
    for session in ("H1", "H2"):
        ours = M.hw_report(lg, session=session, **kw)
        theirs = hc.hw_report(lg, session=session, **kw)
        assert ours.keys() == theirs.keys()
        for k in ours:
            assert ours[k][1] == theirs[k][1]
            assert ours[k][0] == pytest.approx(theirs[k][0], rel=1e-12, abs=1e-12)


def test_461_helpers_identical():
    h = _hw461()
    t = np.arange(0, 3, 0.01)
    a = np.sin(2 * t)
    b = np.sin(2 * (t - 0.05))
    assert M.estimate_delay(t, a, b) == pytest.approx(h.estimate_delay(t, a, b))
    qd = np.exp(-t)[:, None] * np.ones((1, 3))
    assert M.stop_time(t, qd, 0.5, 0.1) == h.stop_time(t, qd, 0.5, 0.1)


def test_validation_checks_pass():
    checks = M.validation_checks()
    assert len(checks) >= 8
    bad = [(c.name, c.error, c.tol, c.detail) for c in checks if not c.passed]
    assert not bad, bad
    assert all(c.category == "Hardware" for c in checks)


# ------------------------------------------------------------------------------------- session
@pytest.mark.parametrize("session", ["H1", "H2", "lab"])
def test_offline_session_passes_hw_tests(session):
    arm, _ = _arm(seed=1)
    sup = SafetySupervisor(session=session, first_run=False)
    t, q = quintic_move(COURSE_ARM.q_home, COURSE_ARM.q_home + np.radians([30, 15, -20, 10, 20, -25]),
                        sup.speed_cap)
    run = run_offline(arm, sup, t, q, auto_stop_at=0.6 * t[-1])
    assert arm.latched and run.t_stop == pytest.approx(0.6 * t[-1], abs=0.021)
    robot, _ = course_arm_robot()
    res = M.session_report(run.log, robot, COURSE_ARM, session)
    assert res["HW-5 speed / rated"][1]
    assert res["HW-4 limit margin [deg]"][1]
    assert res["HW-6 halt time [s]"][1] and res["HW-6 motion after stop [deg]"][1]
    assert res["HW-2 joint max [deg]"][1]
    # sampling rate >= 20 Hz as the briefs require
    assert np.median(np.diff(run.log.t)) <= 0.05


# ------------------------------------------------------------------------------------- myCobot driver
def test_mycobot_missing_sdk_message():
    if sdk_available():
        pytest.skip("pymycobot is installed here")
    with pytest.raises(HardwareError, match="pip install pymycobot"):
        MyCobotDriver("/dev/null").connect()


class _FakeSDK:
    """pymycobot stand-in.  Failure modes (class attributes, set per test with ``_fake(...)``):
    raise_on_send, fail_n_reads (the next n reads return -1), disconnect_after_k (every SDK call after
    the k-th raises OSError), stop_raises, release_raises.  ``follow``: the servos reach each set-point
    at once (so a streamed run tracks perfectly)."""
    raise_on_send = False
    fail_n_reads = 0
    disconnect_after_k = None
    stop_raises = False
    release_raises = False
    follow = False
    instances: list = []

    def __init__(self, port, baud):
        self.port, self.baud = port, baud
        self.angles = [10.0, -20.0, 30.0, 0.0, 5.0, -5.0]
        self.sent = None
        self.sent_log = []
        self.stopped = self.released = self.closed = False
        self.stop_calls = 0
        self.fail = False
        self.calls = 0
        self.fail_n_reads = type(self).fail_n_reads
        _FakeSDK.instances.append(self)

    def _io(self):
        self.calls += 1
        if self.disconnect_after_k is not None and self.calls > self.disconnect_after_k:
            raise OSError("[Errno 6] Device not configured")

    def power_on(self):
        self._io()

    def get_angles(self):
        self._io()
        if self.fail_n_reads > 0:
            self.fail_n_reads -= 1
            return -1
        return -1 if self.fail else list(self.angles)

    def send_angles(self, a, s):
        self._io()
        if self.raise_on_send:
            raise OSError("write timeout")
        self.sent = (a, s)
        self.sent_log.append(list(a))
        if self.follow:
            self.angles = list(a)

    def stop(self):
        self.stop_calls += 1
        if self.stop_raises:
            raise OSError("write timeout")
        self._io()
        self.stopped = True

    def release_all_servos(self):
        self._io()
        if self.release_raises:
            raise OSError("write timeout")
        self.released = True

    def focus_all_servos(self):
        self._io()
        self.released = False

    def close(self):
        self.closed = True


def _fake(**modes):
    return type("FakeSDK", (_FakeSDK,), modes)


def test_mycobot_driver_with_fake_sdk():
    off = np.radians([5, 0, 0, 0, 0, 0])
    d = MyCobotDriver("fake", servo_zero_offset=off, sdk_factory=_FakeSDK)
    d.connect()
    q = d.read_joints()
    np.testing.assert_allclose(q, np.radians([10, -20, 30, 0, 5, -5]) - off)   # q_model = q_servo − offset
    d.send_joints(np.zeros(6), 0.25)
    np.testing.assert_allclose(d.mc.sent[0], [5, 0, 0, 0, 0, 0])
    assert d.mc.sent[1] == 20                                # fixed streaming %, not the speed fraction
    d.mc.fail = True
    np.testing.assert_allclose(d.read_joints(), q)           # one failed read repeats the last sample …
    assert d.read_failures == 1 and not d.last_read_fresh    # … flagged stale
    d.stop()
    assert d.mc.stopped and d.latched and d.send_joints(np.zeros(6), 0.25) is False
    d.set_torque(False)
    assert d.mc.released


# ------------------------------------------------------------------------------------- model
def test_course_arm_model():
    robot, fallback = course_arm_robot()
    assert robot.n == 6
    from mrstudio.hardware.base import _course_arm_fallback
    fb = _course_arm_fallback()
    np.testing.assert_allclose(robot.Slist, fb.Slist, atol=1e-9)
    np.testing.assert_allclose(robot.M, fb.M, atol=1e-9)
    # MR Ex. 4.5 layout: stretched arm reach (L1 + L2, W1 + W2, H1 − H2)
    np.testing.assert_allclose(fb.M[:3, 3], [0.206, 0.109, 0.0561], atol=1e-12)


# ------------------------------------------------------------------------------------- GUI
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    yield app
    # tear down this module's tabs here (halted, worker joined, port closed) instead of leaving them to the
    # session-wide teardown
    from mrstudio.gui.tabs.hardware_tab import HardwareTab
    for w in app.topLevelWidgets():
        if isinstance(w, HardwareTab):
            w.shutdown()
            w.close()
            w.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_hardware_tab_headless(qapp, tmp_path):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import HardwareTab

    tab = HardwareTab(AppContext())
    assert tab.uses_serial_robot is False
    assert tab.stop_btn.isEnabled()                         # always active
    assert not tab.run_btn.isEnabled()
    assert tab.connect_backend()
    clock = ManualClock(0.0)
    tab.driver.clock, tab.driver._t_last = clock, 0.0
    assert not tab.start_run()                              # checklist not ticked
    assert "Run blocked" in tab.status_lbl.text()
    for cb in tab.check_boxes.values():
        cb.setChecked(True)
    assert tab.run_btn.isEnabled()
    tab.source_box.setCurrentIndex(1)                       # quintic A -> B -> A
    tab.auto_stop.setValue(3.0)
    assert tab.start_run()
    tab.timer.stop()
    assert tab.stop_btn.isEnabled() and not tab.run_btn.isEnabled()
    while not tab.run.finished:
        clock.advance(0.02)
        tab._tick()
    assert tab.driver.latched and not tab.run_btn.isEnabled()
    assert tab.report.rowCount() == 8
    tab.save_log(str(tmp_path / "h.csv"), "361")
    back = L.read_log_361(tmp_path / "h.csv")
    assert back["stop"].max() == 1
    tab.reset_latch()
    assert tab.run_btn.isEnabled()
    tab.stop_now()                                          # STOP while idle latches too
    assert tab.driver.latched
    tab.disconnect_backend()
    assert tab.driver is None


# ------------------------------------------------------------------------------------- review fixes
def test_step_in_command_is_caught_by_speed_rule():
    """Linear interpolation commands the *segment* slope; np.gradient would halve a one-sample step."""
    from mrstudio.hardware.safety import peak_speeds
    sup = SafetySupervisor(session="H1")
    t = np.arange(0.0, 1.0, 0.02)
    q = np.tile(COURSE_ARM.q_home, (len(t), 1))
    q[25:, 0] += np.radians(1.5)                       # 75 deg/s over one 20 ms segment
    assert np.degrees(peak_speeds(t, q)[0]) == pytest.approx(75.0)
    assert any("J1 peak speed" in p for p in sup.trajectory_problems(t, q))
    t2, k = retime_to_speed(t, q, sup.speed_cap)
    assert k > 1.8 and not sup.trajectory_problems(t2, q)
    with pytest.raises(ValueError):
        retime_to_speed([0.0, 0.0, 1.0], np.array([np.zeros(6), np.ones(6), np.ones(6)]), sup.speed_cap)


def test_mycobot_offset_flag():
    assert MyCobotDriver("fake", sdk_factory=_FakeSDK).offset_calibrated is False
    assert MyCobotDriver("fake", servo_zero_offset=np.zeros(6), sdk_factory=_FakeSDK).offset_calibrated


def test_hardware_tab_delay_and_shutdown(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import HardwareTab

    tab = HardwareTab(AppContext())
    assert tab.connect_backend()
    clock = ManualClock(0.0)
    tab.driver.clock, tab.driver._t_last = clock, 0.0
    for cb in tab.check_boxes.values():
        cb.setChecked(True)
    tab.pose_a = COURSE_ARM.q_home.copy()
    tab.pose_b = COURSE_ARM.q_home + np.radians([0, 25, 0, 0, 0, 0])   # J1 does not move
    tab.source_box.setCurrentIndex(1)
    assert tab.start_run()
    tab.timer.stop()
    while not tab.run.finished:
        clock.advance(0.02)
        tab._tick()
    txt = tab.report_lbl.text()
    assert "(J2)" in txt
    d_ms = float(txt.split("(J2):")[1].split("ms")[0])
    assert 20 <= d_ms <= 100                           # first-order lag t_c = 40 ms (+ sample)
    drv = tab.driver
    tab.shutdown()
    assert drv.latched and not drv.connected and tab.driver is None


# ------------------------------------------------------------------------------------- safety review (2026-10)
from mrstudio.hardware.mycobot import (MYCOBOT280_LIMITS_DEG, SERVO_LIMIT_MARGIN_DEG, MyCobotCalibration,  # noqa: E402
                                       rank_ports)
from mrstudio.hardware.session import HardwareRun  # noqa: E402

_CAL = MyCobotCalibration(sign=[1, -1, 1, 1, -1, 1], offset_deg=[30, 0, -10, 0, 5, 0])


def _servo_ok(deg):
    lo, hi = (MYCOBOT280_LIMITS_DEG + np.array([SERVO_LIMIT_MARGIN_DEG, -SERVO_LIMIT_MARGIN_DEG])).T
    return np.all((np.asarray(deg) >= lo - 1e-9) & (np.asarray(deg) <= hi + 1e-9))


def test_mycobot_stale_reads_raise_after_three():
    d = MyCobotDriver("fake", sdk_factory=_FakeSDK, calibration=_CAL)
    d.connect()
    assert d.last_read_fresh and d.fresh_age() < 1.0
    d.mc.fail = True
    d.read_joints()
    d.read_joints()
    assert d.consecutive_read_failures == 2 and not d.last_read_fresh
    with pytest.raises(HardwareError, match="3 failed reads"):
        d.read_joints()
    d.mc.fail = False
    d.read_joints()
    assert d.last_read_fresh and d.consecutive_read_failures == 0 and d.read_failures == 3


def test_mycobot_connect_failure_closes_port():
    _FakeSDK.instances.clear()
    d = MyCobotDriver("fake", sdk_factory=_fake(fail_n_reads=99), calibration=_CAL)
    with pytest.raises(HardwareError, match="did not report"):
        d.connect()
    assert not d.connected and d.mc is None and _FakeSDK.instances[-1].closed
    d2 = MyCobotDriver("fake", sdk_factory=_fake(fail_n_reads=2), calibration=_CAL)
    d2.connect()                                         # the first reads after opening may time out
    assert d2.connected


def test_mycobot_sdk_exceptions_become_hardware_error():
    d = MyCobotDriver("fake", sdk_factory=_fake(raise_on_send=True), calibration=_CAL)
    d.connect()
    with pytest.raises(HardwareError) as ei:
        d.send_joints(np.zeros(6), 0.25)
    assert isinstance(ei.value.__cause__, OSError)
    with pytest.raises(HardwareError, match="non-finite"):
        d.send_joints(np.full(6, np.nan), 0.25)


def test_mycobot_calibration_mapping_limits_and_file(tmp_path):
    d = MyCobotDriver("fake", sdk_factory=_FakeSDK, calibration=_CAL)
    d.connect()
    q = d.read_joints()
    np.testing.assert_allclose(_CAL.to_servo_deg(q), d.mc.angles, atol=1e-9)      # inverse mapping
    d.send_joints(np.radians([10, 20, 0, 0, 0, 0]), 0.5)
    np.testing.assert_allclose(d.mc.sent[0], [40, -20, -10, 0, 5, 0])           # s * q + o
    d.send_joints(np.radians([155, -155, 155, 155, 155, 165]), 0.5)             # model soft limits
    assert _servo_ok(d.mc.sent[0]) and d.clamp_count == 1
    assert d.mc.sent[0][0] == pytest.approx(158.0)                              # 155 + 30 -> 168 − 10
    path = _CAL.save(tmp_path / "cal.json")
    back = MyCobotCalibration.load(path)
    np.testing.assert_array_equal(back.sign, _CAL.sign)
    np.testing.assert_array_equal(back.offset_deg, _CAL.offset_deg)
    with pytest.raises(ValueError):
        MyCobotCalibration(sign=[1, 2, 1, 1, 1, 1], offset_deg=np.zeros(6))
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(HardwareError):
        MyCobotCalibration.load(tmp_path / "bad.json")


def test_failed_halt_is_reported():
    d = MyCobotDriver("fake", sdk_factory=_fake(stop_raises=True), calibration=_CAL)
    d.connect()
    assert d.stop() is False
    assert d.latched and d.halt_confirmed is False and d.mc.stop_calls == 2       # one retry
    assert d.send_joints(np.zeros(6), 0.5) is False                              # still latched
    d.mc.stop_raises = False
    assert d.stop() is True and d.halt_confirmed                                 # STOP again retries


def test_torque_state_only_after_success():
    d = MyCobotDriver("fake", sdk_factory=_fake(release_raises=True), calibration=_CAL)
    d.connect()
    d.stop()
    with pytest.raises(HardwareError):
        d.set_torque(False)
    assert d.torque_enabled


def test_comm_error_latches_and_finishes_run():
    clock = ManualClock()
    d = MyCobotDriver("fake", sdk_factory=_fake(follow=True), calibration=_CAL, clock=clock)
    d.connect()
    sup = SafetySupervisor(session="H2", first_run=False)
    q0 = d.read_joints()
    t, q = quintic_move(q0, q0 + np.radians([20, 10, -10, 5, 5, 5]), sup.speed_cap)
    d.mc.disconnect_after_k = d.mc.calls + 40              # the cable comes out mid-run
    run = run_offline(d, sup, t, q)
    assert run.finished and run.error and "Device not configured" in run.error
    assert d.latched and run.t_stop is not None and run.t_stop < t[-1]
    assert any(e["kind"] == "error" for e in run.log.events)


def test_step_clamp_bounds_every_tick():
    arm, clock = _arm(noise_deg=0, quantise=False)
    sup = SafetySupervisor(session="H1")
    q0 = COURSE_ARM.q_home
    t = np.array([0.0, 0.3, 0.31, 2.0])                    # 20° in 10 ms: 2000°/s
    q = np.array([q0, q0, q0 + np.radians(20), q0 + np.radians(20)])
    run = HardwareRun(arm, sup, t, q, clock=clock)
    run.start(arm.read_joints())
    for dt in [0.02] * 20 + [0.08, 0.005, 0.03] * 10:      # irregular ticks (none is a stall)
        clock.advance(dt)
        run.step()
    lg = run.log
    dcmd = np.abs(np.diff(lg.cmd, axis=0))
    lim = sup.speed_cap * np.minimum(np.diff(lg.t), HardwareRun.STEP_DT_MAX)[:, None]
    assert np.all(dcmd <= lim + 1e-12)
    assert dcmd.max() > 0.5 * lim.max()                     # it really was clamping


def test_stall_freezes_trajectory_time():
    arm, clock = _arm(noise_deg=0, quantise=False)
    sup = SafetySupervisor(session="H2", first_run=False)
    t, q = quintic_move(COURSE_ARM.q_home, COURSE_ARM.q_home + np.radians(30), sup.speed_cap)
    run = HardwareRun(arm, sup, t, q, clock=clock)
    run.start(arm.read_joints())
    for _ in range(10):
        clock.advance(0.02)
        run.step()
    tau, cmd = run.tau, run.log.cmd[-1]
    clock.advance(0.6)                                      # GUI / USB stall
    run.step()
    assert run.stalls == 1 and run.tau == pytest.approx(tau)
    np.testing.assert_allclose(run.log.cmd[-1], cmd)        # no jump ahead
    assert run.log.dt[-1] == pytest.approx(0.6)
    clock.advance(0.02)
    run.step()
    assert run.tau == pytest.approx(tau + 0.02)
    assert any(e["kind"] == "stall" for e in run.log.events)


def test_tracking_watchdog_stops_run():
    arm, _ = _arm(noise_deg=0, quantise=False, lag_tau=3.0)  # a servo that cannot keep up
    sup = SafetySupervisor(session="H2", first_run=False)
    t, q = quintic_move(COURSE_ARM.q_home, COURSE_ARM.q_home + np.radians([60, 0, 0, 0, 0, 0]), sup.speed_cap)
    run = run_offline(arm, sup, t, q)
    assert run.reason.startswith("tracking error J1") and arm.latched and run.t_stop < t[-1]


def test_simulated_arm_sags_with_torque_off():
    arm, clock = _arm(noise_deg=0, quantise=False)
    arm.stop()
    q = arm.true_position()
    arm.set_torque(False)
    clock.advance(1.0)
    d = np.degrees(arm.true_position() - q)
    assert d[1] == pytest.approx(-3.0) and d[2] == pytest.approx(-3.0) and d[0] == 0


def test_rank_ports_prefers_usb_serial():
    from types import SimpleNamespace as P
    ports = [P(device="/dev/tty.usbserial-1", vid=0x10C4, pid=0xEA60),
             P(device="/dev/cu.Bluetooth-Incoming-Port", vid=None, pid=None),
             P(device="/dev/cu.debug-console", vid=None, pid=None),
             P(device="/dev/cu.usbmodem1", vid=0x2341, pid=0x0043),
             P(device="/dev/cu.wchusbserial1", vid=0x1A86, pid=0x7523)]
    assert rank_ports(ports, "darwin") == ["/dev/cu.wchusbserial1", "/dev/cu.usbmodem1"]
    lin = [P(device="/dev/ttyS0", vid=None, pid=None), P(device="/dev/ttyUSB0", vid=0x10C4, pid=0xEA60)]
    assert rank_ports(lin, "linux") == ["/dev/ttyUSB0", "/dev/ttyS0"]


# ------------------------------------------------------------------------------------- GUI, myCobot back-end
def _mycobot_tab(monkeypatch, tmp_path, factory=None, calibrated=True):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import _BACKENDS, HardwareTab
    cal_path = tmp_path / "mycobot_calibration.json"
    monkeypatch.setenv("MRSTUDIO_MYCOBOT_CALIBRATION", str(cal_path))
    if calibrated:
        MyCobotCalibration().save(cal_path)
    tab = HardwareTab(AppContext())
    assert tab.calib_path == str(cal_path)
    tab.sdk_factory = factory or _fake(follow=True)
    tab.backend_box.setCurrentText(_BACKENDS[1])
    tab.port_box.setCurrentText("fake")
    assert tab.connect_backend(), tab.conn_lbl.text()
    clock = ManualClock(0.0)
    tab.driver.clock = clock
    tab.driver.read_joints()
    for cb in tab.check_boxes.values():
        cb.setChecked(True)
    tab.source_box.setCurrentIndex(0)
    tab._set_target(tab.q_meas + np.radians([10, 5, -5, 0, 5, 0]))
    return tab, clock


def _drive(tab, clock, dt=0.02, limit=2000):
    for _ in range(limit):
        if tab.run.finished:
            break
        clock.advance(dt)
        tab._tick()


def test_tab_blocks_run_without_calibration(qapp, monkeypatch, tmp_path):
    tab, clock = _mycobot_tab(monkeypatch, tmp_path, calibrated=False)
    assert not tab.run_btn.isEnabled()
    assert not tab.start_run() and "calibration" in tab.status_lbl.text()
    assert tab.save_calibration()
    assert (tmp_path / "mycobot_calibration.json").is_file()
    assert tab.run_btn.isEnabled() and tab.start_run()
    tab.shutdown()


def test_tab_mycobot_run_logs_sidecars(qapp, monkeypatch, tmp_path):
    import json
    tab, clock = _mycobot_tab(monkeypatch, tmp_path)
    assert tab._atexit_cb is not None                       # emergency stop registered on connect
    assert tab.start_run()
    assert tab.worker is None                               # ManualClock: synchronous path
    assert not tab.save_btn.isEnabled() and not tab.csv_btn.isEnabled()   # no modal dialogs during a run
    _drive(tab, clock)
    assert tab.run.finished and tab.run.reason == "trajectory complete"
    sent = np.array(tab.driver.mc.sent_log)
    assert len(sent) > 10 and _servo_ok(sent)
    assert tab.driver.mc.sent[1] == 20                      # fixed firmware streaming %
    assert tab.save_btn.isEnabled()
    tab.save_log(str(tmp_path / "h.csv"), "361")
    meta = json.loads((tmp_path / "h.meta.json").read_text())
    assert meta["meta"]["port"] == "fake" and meta["meta"]["baud"] == 115200
    assert meta["meta"]["sign"] == [1] * 6 and meta["meta"]["offset_deg"] == [0.0] * 6
    assert "sdk_version" in meta["meta"] and meta["meta"]["stream_speed_pct"] == 20
    kinds = [e["kind"] for e in meta["events"]]
    assert "connect" in kinds and "start" in kinds and "complete" in kinds
    diag = (tmp_path / "h.diag.csv").read_text().splitlines()
    assert diag[0].startswith("t,dt,fresh,stop,servo_cmd_1") and len(diag) == len(tab.last_log) + 1
    tab.disconnect_backend()
    assert tab._atexit_cb is None


def test_tab_comm_error_latches_and_finishes(qapp, monkeypatch, tmp_path):
    tab, clock = _mycobot_tab(monkeypatch, tmp_path)
    assert tab.start_run()
    for _ in range(5):
        clock.advance(0.02)
        tab._tick()
    tab.driver.mc.raise_on_send = True
    clock.advance(0.02)
    tab._tick()
    assert tab.run.finished and tab.driver.latched and not tab.timer.isActive()
    assert "Communication error" in tab.status_lbl.text()
    tab.shutdown()


def test_tab_stale_reads_block_run(qapp, monkeypatch, tmp_path):
    tab, clock = _mycobot_tab(monkeypatch, tmp_path)
    tab.driver.mc.fail = True
    clock.advance(0.5)
    assert not tab.start_run()
    assert "stale" in tab.status_lbl.text() and "failed reads" in tab.health_lbl.text()
    tab.driver.mc.fail = False
    assert tab.start_run()
    tab.shutdown()


def test_tab_failed_halt_shows_red_message(qapp, monkeypatch, tmp_path):
    tab, clock = _mycobot_tab(monkeypatch, tmp_path, factory=_fake(follow=True, stop_raises=True))
    assert tab.start_run()
    clock.advance(0.02)
    tab._tick()
    assert tab.stop_now() is False
    assert "STOP NOT CONFIRMED" in tab.status_lbl.text()
    _drive(tab, clock)
    assert "STOP NOT CONFIRMED" in tab.status_lbl.text()   # the end of the run does not hide it
    tab.shutdown()
    assert tab.driver is None


def test_tab_torque_release_needs_confirmation(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import HardwareTab
    tab = HardwareTab(AppContext())
    assert tab.connect_backend()
    tab.stop_now()
    tab.confirm = lambda *a: False
    tab._toggle_torque()
    assert tab.driver.torque_enabled and "cancelled" in tab.status_lbl.text()
    tab.confirm = lambda *a: True
    tab._toggle_torque()
    assert not tab.driver.torque_enabled
    tab._toggle_torque()                                     # back on: no confirmation needed
    assert tab.driver.torque_enabled
    tab.shutdown()


def test_tab_worker_thread_run_and_shutdown(qapp):
    """Real-time run on the streaming worker thread (SimulatedArm, monotonic clock)."""
    import time as _time
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import HardwareTab
    tab = HardwareTab(AppContext())
    assert tab.connect_backend()
    for cb in tab.check_boxes.values():
        cb.setChecked(True)
    tab.source_box.setCurrentIndex(0)
    tab._set_target(tab.q_meas + np.radians([5, 0, 0, 0, 0, 0]))
    tab.auto_stop.setValue(0.5)
    assert tab.start_run() and tab.worker is not None
    assert not tab.save_btn.isEnabled()
    t_end = _time.monotonic() + 10.0
    while not tab.run.finished and _time.monotonic() < t_end:
        qapp.processEvents()
        _time.sleep(0.01)
    for _ in range(50):                                      # let the queued 'finished' arrive
        qapp.processEvents()
        if tab.worker is None:
            break
        _time.sleep(0.01)
    assert tab.run.finished and tab.driver.latched and tab.worker is None
    assert len(tab.last_log) > 20 and tab.last_log.stop.any()
    assert np.median(np.diff(tab.last_log.t)) <= 0.05
    # close while streaming: halt, latch, join the worker, close the port
    tab.reset_latch()
    assert tab.start_run()
    drv = tab.driver
    tab.shutdown()
    assert drv.latched and not drv.connected and tab.worker is None and tab.driver is None
