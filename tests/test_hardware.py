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

_SRC = Path.home() / "Documents/Edu/Lectures/Robotics/Robotics Track Course Materials/_build/src"
_361 = _SRC / "361/project/starter"
_461 = _SRC / "461/project/starter"


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
    def __init__(self, port, baud):
        self.angles = [10.0, -20.0, 30.0, 0.0, 5.0, -5.0]
        self.sent = None
        self.stopped = self.released = False
        self.fail = False

    def power_on(self):
        pass

    def get_angles(self):
        return -1 if self.fail else list(self.angles)

    def send_angles(self, a, s):
        self.sent = (a, s)

    def stop(self):
        self.stopped = True

    def release_all_servos(self):
        self.released = True

    def focus_all_servos(self):
        self.released = False


def test_mycobot_driver_with_fake_sdk():
    off = np.radians([5, 0, 0, 0, 0, 0])
    d = MyCobotDriver("fake", servo_zero_offset=off, sdk_factory=_FakeSDK)
    d.connect()
    q = d.read_joints()
    np.testing.assert_allclose(q, np.radians([10, -20, 30, 0, 5, -5]) - off)   # q_model = q_servo − offset
    d.send_joints(np.zeros(6), 0.25)
    np.testing.assert_allclose(d.mc.sent[0], [5, 0, 0, 0, 0, 0])
    assert d.mc.sent[1] == 25
    d.mc.fail = True
    np.testing.assert_allclose(d.read_joints(), q)           # failed read keeps the last sample
    assert d.read_failures == 1
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
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


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
