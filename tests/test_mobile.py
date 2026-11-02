"""Wheeled mobile robots (MR Ch. 13): kinematic models, odometry, Lie brackets, controllers, presets and the tab."""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio import mobile as mb


def test_validation_checks_pass():
    checks = mb.validation_checks()
    assert len(checks) >= 8
    assert {c.category for c in checks} == {mb.CATEGORY}
    bad = [(c.name, c.error, c.tol, c.detail) for c in checks if not c.passed]
    assert not bad, bad


def test_book_matrices():
    r, d, l, w = 0.05, 0.2, 0.235, 0.15
    np.testing.assert_allclose(mb.H_matrix(mb.three_omni_wheels(r, d)), mb.H0_three_omni_book(r, d), atol=1e-12)
    np.testing.assert_allclose(mb.H_matrix(mb.four_mecanum_wheels(r, l, w)), mb.H0_mecanum_book(r, l, w), atol=1e-12)
    # F H(0) = I_3 for the omnidirectional robots (13.32), (13.33)
    np.testing.assert_allclose(mb.F_three_omni_book(r, d) @ mb.H0_three_omni_book(r, d), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(mb.F_mecanum_book(r, l, w) @ mb.H0_mecanum_book(r, l, w), np.eye(3), atol=1e-12)
    # diff-drive (13.34): H(0) F = I_2
    H = mb.H_matrix(mb.diff_drive_wheels(r, 0.15))
    np.testing.assert_allclose(H @ mb.F_diff_drive_book(r, 0.15), np.eye(2), atol=1e-12)


def test_h_phi_relation():
    """u = H(phi) qdot = H(0) V_b with V_b from (13.1)."""
    rng = np.random.default_rng(0)
    ws = mb.four_mecanum_wheels(0.0475, 0.235, 0.15)
    for _ in range(10):
        phi = rng.uniform(-3, 3)
        qd = rng.uniform(-1, 1, 3)
        np.testing.assert_allclose(mb.H_matrix(ws, phi) @ qd, mb.H_matrix(ws) @ mb.body_twist(phi, qd), atol=1e-12)


def test_mecanum_motions_p521():
    H = mb.H0_mecanum_book(0.05, 0.2, 0.15)
    fwd = H @ [0, 1, 0]
    assert np.all(fwd > 0) and np.ptp(fwd) == pytest.approx(0)          # all forward, same speed
    side = H @ [0, 0, 1]
    assert np.sign(side).tolist() == [-1, 1, -1, 1]                       # 1, 3 backward; 2, 4 forward
    ccw = H @ [1, 0, 0]
    assert np.sign(ccw).tolist() == [-1, 1, 1, -1]                        # 1, 4 backward; 2, 3 forward


def test_odometry_branches():
    np.testing.assert_allclose(mb.delta_qb([0.0, 0.3, -0.1]), [0.0, 0.3, -0.1])
    for Vb in ([0.7, 0.2, 0.1], [-1.3, 0.0, 0.4], [2 * math.pi, 1.0, 0.0]):
        np.testing.assert_allclose(mb.delta_q(Vb, 0.4), mb.delta_q_expm(Vb, 0.4), atol=1e-12)
    # a full turn in place with forward speed returns to the start position (closed circle)
    dq = mb.delta_qb([2 * math.pi, 1.0, 0.0])
    assert abs(dq[1]) < 1e-12 and abs(dq[2]) < 1e-12


def test_lie_bracket_example_13_5():
    g1 = lambda q: np.array([0.0, math.cos(q[0]), math.sin(q[0])])    # noqa: E731
    g2 = lambda q: np.array([1.0, 0.0, 0.0])                          # noqa: E731
    q = np.array([0.7, 0.2, -0.4])
    np.testing.assert_allclose(mb.lie_bracket(g1, g2, q), [0.0, math.sin(0.7), -math.cos(0.7)], atol=1e-8)
    np.testing.assert_allclose(mb.lie_bracket(g2, g1, q), -mb.lie_bracket(g1, g2, q), atol=1e-8)   # skew


def test_control_transformations_p528():
    v, w, r, d = 0.4, 0.8, 0.05, 0.15
    uL, uR = mb.diff_drive_wheel_speeds(v, w, r, d)
    assert (uL, uR) == pytest.approx(((v - w * d) / r, (v + w * d) / r))
    np.testing.assert_allclose(mb.diff_drive_qdot(0.3, uL, uR, r, d), mb.canonical_qdot(0.3, v, w), atol=1e-12)
    assert mb.car_steering_angle(v, w, 0.3) == pytest.approx(math.atan(0.3 * w / v))


def test_saturation_keeps_twist_direction():
    rob = mb.default_robot("omni3")
    rob.u_max = 5.0
    u, Vb, sat = rob.saturate([0.5, 1.0, 0.5])
    assert sat and np.max(np.abs(u)) == pytest.approx(5.0)
    np.testing.assert_allclose(np.cross(Vb, [0.5, 1.0, 0.5]), 0, atol=1e-12)
    car = mb.default_robot("car")
    _, Vb, sat = car.saturate([3.0, 0.3, 0.0])             # far beyond the steering limit
    assert sat and abs(Vb[0]) == pytest.approx(0.3 * math.tan(car.psi_max) / car.ell)


def test_twist_polygon_mecanum_equal_speeds():
    rob = mb.default_robot("mecanum")
    P = rob.control_set()
    assert np.max(P[:, 0]) == pytest.approx(rob.max_speed())
    assert np.max(P[:, 1]) == pytest.approx(rob.max_speed())          # same sideways speed (p. 521)


def test_tracking_converges_and_nonholonomic_constraint():
    rob = mb.default_robot("diff")
    ref = mb.make_reference("Figure-eight")
    sim = mb.simulate(rob, ref, "track", q0=ref.sample(0)[0] + [0.3, -0.2, 0.2], gains=mb.Gains())
    assert sim.track_err[-1] < 5e-3
    assert np.max(np.abs(sim.Vb[:, 2])) == 0.0                        # v_by = 0 for a diff-drive
    resid = [abs(mb.pfaffian_A(q) @ mb.canonical_qdot(q[0], V[1], V[0])) for q, V in zip(sim.q, sim.Vb)]
    assert max(resid) < 1e-12


def test_odometry_drift_preset():
    robot, ref, ctrl, q0, g, fb = mb.preset("Odometry drift: diff-drive, 2 % radius error")
    sim = mb.simulate(robot, ref, ctrl, q0, g, feedback=fb)
    est_err = np.hypot(*(sim.q_odo[-1, 1:] - sim.q_d[-1, 1:]))
    assert est_err < 0.01                       # the estimate tracks the reference ...
    assert sim.odo_err[-1] > 0.1                # ... while the true robot has drifted away
    # no radius error: odometry equals the true pose
    robot.radius_error = ()
    sim = mb.simulate(robot, ref, ctrl, q0, g, feedback=fb)
    assert np.max(sim.odo_err) < 1e-12


def test_cusp_preset_reverses_then_recovers():
    robot, ref, ctrl, q0, g, fb = mb.preset("Diff-drive: P control makes a cusp (Fig. 13.19)")
    sim = mb.simulate(robot, ref, ctrl, q0, g, feedback=fb)
    v = sim.Vb[:, 1]
    assert np.sum(v < 0) * 0.01 > 2.0           # drives backwards for seconds
    assert v[-1] > 0                            # then a cusp: forward again
    assert abs(sim.err[-1, 0]) < math.radians(20)


@pytest.mark.parametrize("name", list(mb.PRESETS))
def test_presets_run(name):
    robot, ref, ctrl, q0, g, fb = mb.preset(name)
    if ref.omni_only:
        assert robot.holonomic
    sim = mb.simulate(robot, ref, ctrl, q0, g, feedback=fb)
    assert np.all(np.isfinite(sim.q)) and len(sim.t) > 100


# ------------------------------------------------------------------ GUI
PDF = (Path.home() / "Documents/Edu/Lectures/Robotics/Textbooks/Modern Robotics/"
       "MODERN ROBOTICS MECHANICS, PLANNING, AND CONTROL-v2, with Hyperlinks.pdf")


def test_theory_renders_and_cites_book_pages():
    from mrstudio.gui.tabs.mobile_tab import THEORY
    from mrstudio.gui.theory import validate_all
    assert validate_all([THEORY]) == []
    cites = re.findall(r"\((13\.\d+)\)[^.]{0,40}?p\.\s*(\d+)", THEORY)
    assert len(cites) >= 25
    # every (13.n) mentioned in the panel carries its page
    assert {e for e, _ in cites} == set(re.findall(r"\((13\.\d+)\)", THEORY))
    import shutil
    import subprocess
    if not PDF.exists() or shutil.which("pdftotext") is None:
        pytest.skip("MR PDF or pdftotext not available")
    bad = []
    for eq, pg in cites:
        txt = subprocess.run(["pdftotext", "-f", str(int(pg) + 20), "-l", str(int(pg) + 20), str(PDF), "-"],
                             capture_output=True, text=True).stdout
        if f"({eq})" not in txt:
            bad.append((eq, pg))
    assert bad == []


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def test_tab_constructs_and_presets(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.mobile_tab import MobileTab
    tab = MobileTab(AppContext())
    assert tab.title == "⑬ Mobile robots" and tab.uses_serial_robot is False
    assert tab.sim is not None
    for i, name in enumerate(mb.PRESETS):
        assert tab.apply_preset(str(i))
        assert tab.preset_box.currentText() == name
        assert tab.sim.robot.kind == mb.PRESETS[name]["kind"]
        tab._draw(len(tab.sim.t) // 2)
    assert tab.apply_preset("mecanum") and tab.robot.kind == "mecanum"
    assert not tab.apply_preset("no such preset")
    # switching the robot type refreshes the controller list and recomputes
    tab.kind_box.setCurrentText(mb.KINDS["car"])
    tab.recompute()
    assert tab.sim.robot.kind == "car"
    assert tab.ctrl_box.findText("Nonlinear tracking (13.31)") >= 0
    assert all("(omni)" not in tab.ref_box.itemText(i) for i in range(tab.ref_box.count()))
    tab.shutdown()
