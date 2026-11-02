"""Guided lessons (mrstudio.lessons) and the Lessons panel of the Practice tab, tested against a fake PracticeHost."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modern_robotics as mr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio import lessons as LS  # noqa: E402

LESSONS = LS.all_lessons()
STEPS = [(L, k) for L in LESSONS for k in range(len(L.steps))]
IDS = [f"{L.id}-{k + 1}" for L, k in STEPS]


# ------------------------------------------------------------------------------------------------ catalogue

def test_catalogue_size_and_courses():
    assert len(LESSONS) >= 8
    assert all(len(L.steps) >= 4 for L in LESSONS)
    assert len({L.id for L in LESSONS}) == len(LESSONS)
    courses = {L.course for L in LESSONS}
    assert courses == {"SBEG 361", "SBEG 461"}
    for L in LESSONS:
        assert 1 <= L.week <= 14 and L.minutes > 0 and L.title and L.section.startswith("MR ")


def test_targets_name_real_tabs_and_robots():
    from mrstudio.gui.app import TABS
    from mrstudio.robots import ROBOTS
    tabs = {c for _, c in TABS}
    for L in LESSONS:
        for s in L.steps:
            assert s.target.tab in tabs, (L.id, s.target.tab)
            assert s.target.robot is None or s.target.robot in ROBOTS, (L.id, s.target.robot)


# ------------------------------------------------------------------------------------------------ checks

@pytest.mark.parametrize("L,k", STEPS, ids=IDS)
def test_check_accepts_reference_and_rejects_10_percent_off(L, k):
    c = L.steps[k].check
    ref = c.reference
    assert math.isfinite(ref) and ref != 0.0
    assert c.accepts(ref)
    assert c.grade(repr(ref)).ok
    assert c.grade(f"{ref:.4g}").ok, "4 significant figures must pass"
    assert not c.accepts(ref * 1.1) and not c.accepts(ref * 0.9)
    assert not c.grade(repr(ref * 1.1)).ok


def test_references_are_deterministic():
    a = [s.check.compute() for L in LESSONS for s in L.steps]
    b = [s.check.compute() for L in (b() for b in LS._BUILDERS) for s in L.steps]
    assert sorted(a) == sorted(b)
    assert LS.seed_for("x", 1) == LS.seed_for("x", 1) != LS.seed_for("x", 2)


def test_all_html_renders():
    from mrstudio.gui.theory import validate_all
    texts = []
    for L in LESSONS:
        texts += [L.step_html(k) for k in range(len(L.steps))] + [s.hint_html for s in L.steps]
    assert validate_all(texts) == []


def test_parse_answer():
    p = LS.parse_answer
    assert p("0.5") == 0.5 and p(" -2.5e-1 ") == -0.25 and p("1,5") == 1.5
    assert p("30°") == 30 and p("−3") == -3 and p("12 N·m") == 12
    assert p("0.6*cos(pi/3)") == pytest.approx(0.3)
    assert p("sqrt(2)/2") == pytest.approx(math.sqrt(2) / 2)
    assert p("2^3") == 8
    for bad in ("", "abc", "__import__('os')", "1/0", "(1).real", "9**9**9"):
        with pytest.raises(ValueError):
            p(bad)


# ------------------------------------------------------------------------------------------------ independent re-checks

def test_poe_tip_matches_closed_form():
    L = LS.get_lesson("361-w06-poe")
    th = L.data["theta"]
    a = np.array([0.5, 0.4, 0.25])           # Planar 3R arm link lengths (robots.planar_3r)
    c = np.cumsum(th)
    refs = [s.check.reference for s in L.steps]
    assert refs[0] == pytest.approx(-0.9)                       # v_y of S_3 = -(L1 + L2)
    assert refs[1] == pytest.approx(float(a @ np.cos(c)), abs=1e-12)
    assert refs[2] == pytest.approx(float(a @ np.sin(c)), abs=1e-12)
    assert refs[3] == pytest.approx(math.degrees(math.atan2(math.sin(c[-1]), math.cos(c[-1]))), abs=1e-9)
    assert refs[4] == pytest.approx(1.15)                       # v_y of B_1 = L1 + L2 + L3
    for k in (1, 2):                                            # the angles are stated in the instruction
        assert all(f"θ{i + 1} = {np.degrees(t):g}°" in L.steps[k].instruction_html for i, t in enumerate(th))


def test_jacobian_lesson_closed_forms():
    L = LS.get_lesson("361-w09-jacobian")
    (t1, t2), (d1, d2) = L.data["theta"], L.data["dtheta"]
    L1, L2 = 0.6, 0.4
    refs = [s.check.reference for s in L.steps]
    assert refs[0] == pytest.approx(L1 * math.sin(t1), abs=1e-12)
    assert refs[1] == pytest.approx(L1 * math.cos(t2) + L2, abs=1e-12)   # tip-frame sideways lever of joint 1
    xdot = -(L1 * math.sin(t1) + L2 * math.sin(t1 + t2)) * d1 - L2 * math.sin(t1 + t2) * d2
    assert refs[2] == pytest.approx(xdot, abs=1e-12)
    assert refs[3] == pytest.approx((L1 * math.cos(t2) + L2) * d1 + L2 * d2, abs=1e-12)


def test_singularity_lesson_closed_forms():
    L = LS.get_lesson("361-w10-singularities")
    (t1, t2), F, L1, L2 = L.data["theta"], L.data["F"], L.data["L1"], L.data["L2"]
    det, sing, s2, cond, tau1 = (s.check.reference for s in L.steps)
    assert det == pytest.approx(L1 * L2 * math.sin(t2), abs=1e-12)
    assert sing == pytest.approx(180.0, abs=1e-6)
    assert cond == pytest.approx(abs(det) / s2**2, rel=1e-9)         # sigma1 sigma2 = |det J|
    J1 = np.array([-(L1 * math.sin(t1) + L2 * math.sin(t1 + t2)), L1 * math.cos(t1) + L2 * math.cos(t1 + t2)])
    assert tau1 == pytest.approx(float(J1 @ F), abs=1e-12)


def test_ik_lesson_converges_to_the_target():
    L = LS.get_lesson("361-w11-newton-raphson")
    d = L.data
    th = LS._nr_2r(LS._robot(LS.R2_NAME), d["x_d"], d["theta0"], 30)
    assert np.allclose(th, d["theta_d"], atol=1e-9)
    e0, e1 = L.steps[0].check.reference, L.steps[3].check.reference
    assert e1 < e0 / 2
    assert 2 <= L.steps[4].check.reference <= 10


def test_mass_matrix_lesson_closed_form():
    L = LS.get_lesson("461-w01-mass-matrix")
    th2 = L.data["theta"][1]
    dth = L.data["dtheta"]
    L1, L2, m1, m2, r = 0.6, 0.4, 2.0, 1.0, 0.025
    I1, I2 = m1 * (3 * r**2 + L1**2) / 12, m2 * (3 * r**2 + L2**2) / 12
    M22 = I2 + m2 * (L2 / 2) ** 2
    M12 = M22 + m2 * L1 * (L2 / 2) * math.cos(th2)
    M11 = I1 + m1 * (L1 / 2) ** 2 + I2 + m2 * (L1**2 + (L2 / 2) ** 2 + 2 * L1 * (L2 / 2) * math.cos(th2))
    refs = [s.check.reference for s in L.steps]
    assert refs[0] == pytest.approx(I1, rel=1e-12)
    assert refs[1:4] == pytest.approx([M22, M12, M11], rel=1e-10)
    K = 0.5 * (M11 * dth[0] ** 2 + 2 * M12 * dth[0] * dth[1] + M22 * dth[1] ** 2)
    assert refs[4] == pytest.approx(K, rel=1e-10)


def test_rnea_lesson_closed_form():
    """Closed-form 2R Lagrangian with rod links (gravity along -y) vs the RNEA references."""
    L = LS.get_lesson("461-w02-rnea")
    (t1, t2), (d1, d2), (a1, a2) = L.data["theta"], L.data["dtheta"], L.data["ddtheta"]
    L1, L2, m1, m2, r, g = 0.6, 0.4, 2.0, 1.0, 0.025, 9.81
    I1, I2 = m1 * (3 * r**2 + L1**2) / 12, m2 * (3 * r**2 + L2**2) / 12
    l2 = L2 / 2
    M11 = I1 + m1 * (L1 / 2) ** 2 + I2 + m2 * (L1**2 + l2**2 + 2 * L1 * l2 * math.cos(t2))
    M12 = I2 + m2 * (l2**2 + L1 * l2 * math.cos(t2))
    M22 = I2 + m2 * l2**2
    h = m2 * L1 * l2 * math.sin(t2)
    c1, c2 = -h * (2 * d1 * d2 + d2**2), h * d1**2
    g1 = (m1 * L1 / 2 + m2 * L1) * g * math.cos(t1) + m2 * l2 * g * math.cos(t1 + t2)
    g2 = m2 * l2 * g * math.cos(t1 + t2)
    tau1 = M11 * a1 + M12 * a2 + c1 + g1
    tau2 = M12 * a1 + M22 * a2 + c2 + g2
    refs = [s.check.reference for s in L.steps]
    assert refs == pytest.approx([g1, g2, c2, tau1, tau2], abs=1e-9)


def test_pd_lesson_closed_form():
    L = LS.get_lesson("461-w05-pd")
    d = L.data
    wn, zeta = math.sqrt(d["Kp"] / d["M"]), (d["b"] + d["Kd"]) / (2 * math.sqrt(d["Kp"] * d["M"]))
    refs = [s.check.reference for s in L.steps]
    # overshoot cross-checked against problems.second_order_response (simulated)
    from mrstudio.problems import second_order_response
    _, _, os_sim, _ = second_order_response(zeta, wn)
    assert refs[0] == pytest.approx(wn) and refs[1] == pytest.approx(zeta)
    assert refs[2] == pytest.approx(100 * os_sim, abs=0.05)
    assert refs[3] == pytest.approx(4 / (zeta * wn))
    assert refs[4] == pytest.approx(2 * math.sqrt(d["Kp"] * d["M"]) - d["b"])


def test_computed_torque_error_dynamics():
    L = LS.get_lesson("461-w06-computed-torque")
    d = L.data
    e, ed = d["theta_d"] - d["theta"], d["dtheta_d"] - d["dtheta"]
    assert L.steps[0].check.reference == pytest.approx(d["kp"]) and L.steps[1].check.reference == pytest.approx(d["kd"])
    assert L.steps[4].check.reference == pytest.approx(-d["kp"] * e[0] - d["kd"] * ed[0], abs=1e-8)
    # torque = M (qdd_d + kp e + kd ed) + c + g with the robot's own model (independent of mr.ComputedTorque)
    from mrstudio.dynamics import DynamicsEngine
    eng = DynamicsEngine(LS._robot(LS.R2_NAME), backend="mr")
    u = d["ddtheta_d"] + d["kp"] * e + d["kd"] * ed
    tau = eng.mass_matrix(d["theta"]) @ u + eng.coriolis(d["theta"], d["dtheta"]) + eng.gravity_torque(d["theta"])
    assert [L.steps[2].check.reference, L.steps[3].check.reference] == pytest.approx(tau, abs=1e-9)
    assert mr is not None


def test_time_scaling_closed_forms():
    L = LS.get_lesson("461-w10-time-scaling")
    d, T, th0, vmax = L.data["delta_deg"], L.data["T"], L.data["theta0_deg"], L.data["vmax_deg"]
    v_c, a_c, v_q, th_q, T_min = (s.check.reference for s in L.steps)
    assert v_c == pytest.approx(1.5 * d / T, rel=1e-6)
    assert a_c == pytest.approx(6 * d / T**2, rel=1e-6)
    assert v_q == pytest.approx(1.875 * d / T, rel=1e-6)
    assert th_q == pytest.approx(th0 + d * (10 / 4**3 - 15 / 4**4 + 6 / 4**5), rel=1e-12)
    assert T_min == pytest.approx(1.875 * d / vmax, rel=1e-6)


# ------------------------------------------------------------------------------------------------ panel

class FakeHost:
    def __init__(self):
        from mrstudio.gui.base import AppContext
        self.ctx = AppContext("Planar 2R arm")
        self.html: list[tuple[str, str | None]] = []
        self.status: list[str] = []
        self.problems = []

    def show_problem(self, p, reveal_allowed: bool = True):
        self.problems.append((p, reveal_allowed))

    def show_html(self, statement_html, solution_html=None):
        self.html.append((statement_html, solution_html))

    def set_status(self, text):
        self.status.append(text)


@pytest.fixture
def panel(qapp, tmp_path):
    from PySide6 import QtCore
    from mrstudio.gui.tabs.lessons_panel import LessonsPanel
    st = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    host = FakeHost()
    p = LessonsPanel(host, settings=st)
    return p, host, st


def test_panel_constructs_and_shows_first_step(panel):
    p, host, _ = panel
    assert p.lesson_box.count() == len(LESSONS)
    assert p.lesson is LESSONS[0] and p.k == 0
    assert host.html and "Step 1 of" in host.html[-1][0]
    assert p.step_list.count() == len(LESSONS[0].steps)
    assert not p.next_btn.isEnabled()


def test_panel_check_flow_and_progress(panel, tmp_path):
    from PySide6 import QtCore
    p, host, st = panel
    L = p.lesson
    s = L.steps[0]
    p.answer.setText(repr(s.check.reference * 1.1))
    assert not p.check().ok
    assert host.html[-1][1].startswith("<h2>Hint")
    p.answer.setText("not a number")
    assert p.check().value is None
    p.answer.setText(f"{s.check.reference:.5g}")
    assert p.check().ok
    assert p.next_btn.isEnabled()
    assert int(st.value(f"lessons/{L.id}/step")) == 1
    p.next()
    assert p.k == 1
    # progress survives a new panel
    from mrstudio.gui.tabs.lessons_panel import LessonsPanel
    st2 = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    p2 = LessonsPanel(FakeHost(), settings=st2)
    assert p2.lesson.id == L.id and p2.k == 1
    p2.restart()
    assert p2.done() == 0 and p2.k == 0


def test_panel_whole_lesson_and_goto(panel):
    p, host, _ = panel
    p.select_lesson("461-w02-rnea")
    L = p.lesson
    assert L.id == "461-w02-rnea"
    got = []
    p.request_tab.connect(lambda tab, rob: got.append((tab, rob)))
    for k, s in enumerate(L.steps):
        assert p.k == k
        p.goto_btn.click()
        p.answer.setText(repr(s.check.reference))
        assert p.check().ok
        p.next()
    assert p.done() == len(L.steps)
    assert got and all(t == ("DynamicsTab", "Planar 2R arm") for t in got)
    assert "finished" in p.progress_lbl.text()
    assert not host.problems, "lessons never call show_problem"


def test_panel_shows_expected_value_after_three_tries(panel):
    p, _, _ = panel
    s = p.lesson.steps[p.k]
    for _ in range(3):
        p.answer.setText(repr(s.check.reference * 2 + 1))
        p.check()
    assert "Expected value" in p.feedback.text()
