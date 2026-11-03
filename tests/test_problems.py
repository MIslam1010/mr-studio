"""Practice-problem generator (mrstudio/problems.py) and the Practice tab."""
from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modern_robotics as mr
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio import problems as P  # noqa: E402

ALL = [(t, d) for t in P.TOPIC_KEYS for d in (1, 2, 3)]


@pytest.mark.parametrize("topic,diff", ALL)
def test_deterministic_by_seed(topic, diff):
    a, b = P.generate(topic, 7, diff), P.generate(topic, 7, diff)
    assert a.statement == b.statement and a.solution == b.solution and a.final_answer == b.final_answer
    c = P.generate(topic, 8, diff)
    assert c.statement != a.statement        # another seed -> another problem


@pytest.mark.parametrize("topic,diff", ALL)
def test_answers_verified_independently(topic, diff):
    for seed in range(6):
        p = P.generate(topic, seed, diff)
        assert P.verify(p) < 1e-6, (topic, diff, seed)
        assert p.statement.strip() and p.steps and p.final_answer.strip()
        assert isinstance(p.data, dict) and p.answer


def test_all_math_renders_in_gui():
    from mrstudio.gui.theory import validate_all
    for t in P.TOPIC_KEYS:
        for d in (1, 2, 3):
            p = P.generate(t, 3, d)
            assert validate_all(["$" + m + "$" for m in p.all_math()]) == [], (t, d)


def test_mixed_topic_and_bad_input():
    assert [P.generate(P.MIXED, s).topic for s in range(len(P.TOPIC_KEYS))] == P.TOPIC_KEYS
    with pytest.raises(ValueError):
        P.generate("nope", 1)
    with pytest.raises(ValueError):
        P.generate("fk", 1, 4)


# --- spot checks against textbook facts, computed here without the generator's code path -------------

def test_newton_raphson_reproduces_mr_example_6_1():
    M = mr.RpToTrans(np.eye(3), [2, 0, 0])
    B = np.array([[0, 0, 1, 0, 2, 0], [0, 0, 1, 0, 1, 0]], float).T
    Tsd = np.array([[-0.5, -0.866, 0, 0.366], [0.866, -0.5, 0, 1.366], [0, 0, 1, 0], [0, 0, 0, 1]])
    rows, th, ok = P.newton_raphson_table(B, M, Tsd, np.radians([0, 30]))
    assert ok and len(rows) == 4                       # MR p. 230: converges after three iterations
    assert np.allclose(np.degrees(rows[1]["theta"]), [34.23, 79.18], atol=0.01)
    assert np.allclose(np.degrees(th), [30, 90], atol=0.05)


def test_rnea_trace_matches_mr_and_lagrangian():
    Mlist, Glist, Slist, g = P.point_mass_2r(0.7, 0.5, 2.0, 1.5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        th, dth, ddth = rng.uniform(-2, 2, (3, 2))
        V, dV, F, A, _ = P.rnea_trace(th, dth, ddth, g, Mlist, Glist, Slist)
        tau = np.array([F[i] @ A[i] for i in range(2)])
        assert np.allclose(tau, mr.InverseDynamics(th, dth, ddth, g, np.zeros(6), Mlist, Glist, Slist), atol=1e-12)
        assert np.allclose(tau, P.lagrange_2r(0.7, 0.5, 2.0, 1.5, th, dth, ddth)[3], atol=1e-12)


def test_trajectory_peaks_and_control_formulas():
    p = P.generate("trajectory", 2, 1)        # cubic: peak speed 3Δ/(2T), peak accel 6Δ/T² (MR p. 332)
    D = abs(p.data["theta_end"] - p.data["theta_start"])
    T = p.answer["T"]
    assert np.allclose(p.answer["vmax"], 1.5 * D / T) and np.allclose(p.answer["amax"], 6 * D / T**2)
    for s in range(6):
        c = P.generate("control", s, 3)
        # MR rule of thumb t_s ~ 4/(zeta wn) is reported, but the gains meet the *exact* 2 % settling time
        assert abs(4 / (c.answer["zeta"] * c.answer["wn"]) - c.answer["ts"]) < 1e-9
        assert abs(c.answer["ts_exact"] - c.data["ts"]) < 1e-9
        assert abs(c.answer["kd"] - 2 * c.answer["zeta"] * np.sqrt(c.answer["kp"])) < 1e-9
        assert c.answer["zeta"] != pytest.approx(1.0)          # the rule is far off for zeta = 1: avoided


def _free_response(A, t_end, n=200001):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda t, x: A @ x, (0, t_end), [1.0, 0.0], rtol=1e-12, atol=1e-14, dense_output=True)
    tt = np.linspace(0, t_end, n)
    return tt, sol.sol(tt)[0]


def test_exact_settling_time_against_integration():
    for zeta, wn in ((0.2, 3.0), (0.5, 1.0), (0.7, 4.0), (1.0, 2.0), (1.5, 5.0)):
        A = np.array([[0, 1], [-wn**2, -2 * zeta * wn]])
        ts = P.exact_settling_time(zeta, wn)
        tt, e = _free_response(A, 3 * ts)
        last = tt[np.nonzero(np.abs(e) > 0.02)[0][-1]]
        assert abs(last - ts) < 2 * (tt[1] - tt[0]), (zeta, wn)
    # critically damped: exact = 5.834/wn, the 4/wn rule is ~31 % optimistic
    assert P.exact_settling_time(1.0, 1.0) == pytest.approx(5.8339, abs=1e-4)


def test_control_regime_label_and_critical_gain():
    for d in (1, 2, 3):
        for s in range(10):
            c = P.generate("control", s, d)
            dd, a = c.data, c.answer
            if "M" in dd:
                zeta = (dd["b"] + dd["Kd"]) / (2 * np.sqrt(dd["Kp"] * dd["M"]))
            else:
                zeta = dd["kd"] / (2 * np.sqrt(dd["kp"]))
            expect = ("critically damped" if abs(zeta - 1) < 1e-9 else "underdamped" if zeta < 1 else "overdamped")
            assert a["kind"] == expect, (d, s, zeta)
            if d == 2:      # (c) k_d for critical damping: the error dynamics get a double root
                A = np.array([[0, 1], [-dd["kp"], -a["kd_critical"]]])
                lam = np.linalg.eigvals(A)
                assert abs(lam[0] - lam[1]) < 1e-6 * abs(lam[0])
                assert abs(a["kd_critical"] - 2 * np.sqrt(dd["kp"])) < 1e-12
                # the settling time stated for (c) is consistent with zeta = 1
                tt, e = _free_response(A, 20 / np.sqrt(dd["kp"]))
                assert np.all(e > -1e-9)                       # no overshoot when critically damped


def test_jacobian_singular_cases_are_singular():
    seen = set()
    for s in range(40):
        for d in (1, 2, 3):
            p = P.generate("jacobian", s, d)
            seen.add(p.answer["singular"])
            if p.answer["singular"]:
                assert abs(p.answer["det"]) < 1e-9
    assert seen == {True, False}


def test_validation_checks_pass():
    checks = P.validation_checks()
    assert len(checks) >= len(P.TOPIC_KEYS) + 2
    assert all(c.passed for c in checks), [(c.name, c.error) for c in checks if not c.passed]


# --- Ch. 8 topics: inertia, forward dynamics, task-space mass matrix, constrained dynamics -----------

CH8 = ["inertia", "fwd_dynamics", "task_space", "constrained"]


def test_ch8_topics_registered():
    for t in CH8:
        assert t in P.TOPIC_KEYS and P.TOPICS[t]["title"].endswith("(Ch. 8)")
    # the original seven topics keep their random streams (a shared seed still gives the same problem)
    assert [P._RNG_ID[t] for t in ("rotation", "fk", "jacobian", "ik", "dynamics", "trajectory", "control")] == \
        list(range(7))
    assert sorted(P._RNG_ID) == sorted(P.TOPIC_KEYS) and len(set(P._RNG_ID.values())) == len(P.TOPIC_KEYS)


@pytest.mark.parametrize("topic", CH8)
@pytest.mark.parametrize("diff", (1, 2, 3))
def test_ch8_verified_rendered_deterministic(topic, diff):
    from mrstudio.gui.theory import validate_all
    for seed in (1, 2, 3):
        p = P.generate(topic, seed, diff)
        assert p.topic == topic and p.difficulty == diff and p.chapter.startswith("Ch. 8")
        assert P.verify(p) < 1e-6, (topic, diff, seed)
        assert validate_all(["$" + m + "$" for m in p.all_math()]) == [], (topic, diff, seed)
        md = p.markdown(1, with_solution=True)
        assert "### Solution" in md and "**Final answer.**" in md and "Step 1." in md
        q = P.generate(topic, seed, diff)
        assert (q.statement, q.solution) == (p.statement, p.solution)


@pytest.mark.parametrize("topic", CH8)
def test_ch8_verifier_catches_a_wrong_answer(topic):
    p = P.generate(topic, 2, 3)
    key = {"inertia": "KE", "fwd_dynamics": "ddtheta", "task_space": "m_u", "constrained": "lam"}[topic]
    p.answer[key] = p.answer[key] + 1e-3
    assert P.verify(p) > 1e-4


def test_inertia_textbook_facts():
    for s in range(12):
        for d in (1, 2, 3):
            p = P.generate("inertia", s, d)
            dd, a = p.data, p.answer
            m, Ib = dd["m"], a["Ib"]
            if dd["shape"] == "box" and len(set(dd["dims"])) == 1:
                assert np.allclose(np.diag(Ib), m * dd["dims"][0] ** 2 / 6)      # cube: m a^2 / 6
            if dd["shape"] == "cylinder":
                assert np.isclose(Ib[2, 2], m * dd["dims"][0] ** 2 / 2)
            # triangle inequality of principal moments, and G_b = diag(I_b, m I)
            ev = np.sort(np.diag(Ib))
            assert ev[0] + ev[1] >= ev[2] - 1e-12
            assert np.allclose(a["Gb"][3:, 3:], m * np.eye(3)) and np.allclose(a["Gb"][:3, 3:], 0)
            if d >= 2:
                pp = dd["p"]
                # upper-left block of G_a is Steiner's I_a; coupling block is m [p]
                assert np.allclose(a["Ga"][:3, :3], a["Ia"], atol=1e-12)
                assert np.allclose(a["Ga"][:3, 3:], m * mr.VecToso3(pp), atol=1e-12)
                assert np.isclose(np.trace(a["Ia"]), np.trace(Ib) + 2 * m * pp @ pp)
            if d == 3:
                assert np.allclose(np.linalg.eigvalsh(a["Ic"]), np.sort(np.diag(Ib)), atol=1e-12)


def test_fwd_dynamics_euler_steps():
    for s in range(8):
        p = P.generate("fwd_dynamics", s, 2)
        d, a = p.data, p.answer
        th_e, dth_e = mr.EulerStep(d["theta"], d["dtheta"], a["ddtheta"], d["dt"])
        assert np.allclose(th_e, a["theta_explicit"]) and np.allclose(dth_e, a["dtheta_explicit"])
        assert np.allclose(a["dtheta_semi"], a["dtheta_explicit"])
        assert np.allclose(a["theta_semi"] - a["theta_explicit"], d["dt"] ** 2 * a["ddtheta"])
        # point masses: the general closed form reduces to MR (8.9)-(8.10)
        Mm, c, g, _ = P.lagrange_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["theta"], d["dtheta"], np.zeros(2))
        assert np.allclose(Mm, a["M"]) and np.allclose(c, a["c"]) and np.allclose(g, a["g"])
    assert P.generate("fwd_dynamics", 1, 3).data["rods"] is True


def test_task_space_felt_mass_and_reflected_inertia():
    for s in range(8):
        p = P.generate("task_space", s, 3)
        d, a = p.data, p.answer
        assert abs(np.sin(d["theta"][1])) > 0.4                         # away from the elbow singularity
        lo, hi = a["Lam_eig"]
        assert lo - 1e-12 <= a["m_u"] <= hi + 1e-12                    # felt mass between eigenvalues of Lambda
        assert np.allclose(np.diag(a["M_rot"] - a["M"]), d["gear"] ** 2 * d["I_rot"])
        assert a["m_u_rot"] > a["m_u"]                                  # rotors only add inertia
        # an impulse along u: the tip velocity component along u is impulse / felt mass
        v = np.linalg.solve(a["Lam"], d["u"])
        assert np.isclose(d["u"] @ v, 1.0 / a["m_u"])


def test_constrained_multiplier_does_no_work():
    seen = set()
    for s in range(10):
        for diff in (1, 2, 3):
            p = P.generate("constrained", s, diff)
            d, a = p.data, p.answer
            seen.add(d["kind"])
            A = a["A"][0]
            assert abs(A @ d["dtheta"]) < 1e-12 and abs(a["power"]) < 1e-12
            assert abs(A @ a["ddtheta"] + a["Adot_dtheta"][0]) < 1e-9
            if diff == 1:
                assert np.allclose(d["dtheta"], 0)
            if d["kind"] == "orientation":
                assert np.allclose(A, [1, 1]) and np.isclose(a["ddtheta"].sum(), 0)
    assert seen == {"orientation", "line"}


def test_mixed_problem_set_covers_every_topic():
    probs = P.problem_set(len(P.TOPIC_KEYS), P.MIXED, 2, seed=3)
    assert [p.topic for p in probs] == P.TOPIC_KEYS
    assert all(P.verify(p) < 1e-6 for p in probs)


# --- Ch. 5 statics, Ch. 11 force control, Ch. 12 grasping, Ch. 13 mobile robots ----------------------

NEW = {"statics": "Ch. 5", "force_control": "Ch. 11", "grasp_closure": "Ch. 12", "mobile_kin": "Ch. 13"}


def test_new_topics_registered_by_chapter():
    for t, ch in NEW.items():
        assert t in P.TOPIC_KEYS and P.TOPICS[t]["title"].endswith(f"({ch})")
    assert [P._RNG_ID[t] for t in NEW] == [11, 12, 13, 14]
    # the topic list is ordered by chapter
    chapters = [int(re.search(r"\(Ch\. (\d+)\)$", P.TOPICS[t]["title"]).group(1)) for t in P.TOPIC_KEYS]
    assert chapters == sorted(chapters)


def test_existing_seeds_unchanged_by_new_topics():
    """New topics get new random-stream ids, so the earlier topics still give the same problem for a seed."""
    p = P.generate("constrained", 5, 3)
    assert P._rng("constrained", 5, 3).integers(1 << 30) == np.random.default_rng([5, 10, 3]).integers(1 << 30)
    assert p.topic == "constrained" and P.verify(p) < 1e-6


@pytest.mark.parametrize("topic", list(NEW))
@pytest.mark.parametrize("diff", (1, 2, 3))
def test_new_topics_verified_rendered_deterministic(topic, diff):
    from mrstudio.gui.theory import validate_all
    for seed in (1, 2, 3):
        p = P.generate(topic, seed, diff)
        assert p.topic == topic and p.difficulty == diff and p.chapter.startswith(NEW[topic] + " ")
        assert P.verify(p) < 1e-6, (topic, diff, seed)
        assert validate_all(["$" + m + "$" for m in p.all_math()]) == [], (topic, diff, seed)
        md = p.markdown(1, with_solution=True)
        assert "### Solution" in md and "**Final answer.**" in md and "Step 1." in md
        assert p.visual.get("kind") in ("robot", "plot", "xy")
        q = P.generate(topic, seed, diff)
        assert (q.statement, q.solution) == (p.statement, p.solution)


@pytest.mark.parametrize("topic,diff,key", [("statics", 1, "tau"), ("statics", 2, "f_star"), ("statics", 3, "mu1"),
                                            ("force_control", 1, "x_ss"), ("force_control", 2, "B_imp"),
                                            ("force_control", 3, "kfi_max"), ("grasp_closure", 2, "mu_min"),
                                            ("grasp_closure", 1, "closed"), ("mobile_kin", 1, "u_LR"),
                                            ("mobile_kin", 2, "u"), ("mobile_kin", 3, "Vb_odo")])
def test_new_verifiers_catch_a_wrong_answer(topic, diff, key):
    p = P.generate(topic, 2, diff)
    v = p.answer[key]
    p.answer[key] = (not v) if isinstance(v, bool) else v + 1e-3 * (1 + np.abs(v))
    assert P.verify(p) > 1e-4


def test_statics_textbook_facts():
    for s in range(10):
        p = P.generate("statics", s, 1)                  # tau_1 is the moment of the tip force about the base
        assert np.isclose(p.answer["tau"][0], p.answer["Fs"][0])
        p = P.generate("statics", s, 2)
        d, a = p.data, p.answer
        assert np.allclose(a["Jv"].T @ d["f"], a["tau"])  # tau = J_v^T f = J_s^T F_s for a pure tip force
        assert np.allclose(a["Jv"].T @ a["f_star"], d["tau_star"])
        p = P.generate("statics", s, 3)
        d, a = p.data, p.answer
        L, th = d["L"], d["theta"]
        assert np.isclose(np.linalg.det(a["Jb_p"]), L[0] * L[1] * np.sin(th[1]))
        assert np.allclose(a["ell_axes"] * a["force_axes"][::-1], 1.0)   # force ellipse = inverse lengths
        assert a["mu1"] >= 1.0


def test_force_control_facts():
    kinds = set()
    for s in range(12):
        a = P.generate("force_control", s, 1).answer
        kinds.add(P._classify(a["zeta"]))
        d = P.generate("force_control", s, 2)
        assert np.isclose(d.answer["K_imp"] * d.data["delta"], d.data["f"])          # spec deflection met
        assert np.isclose(d.answer["B_imp"], 2 * d.data["zeta_req"] * np.sqrt(d.answer["K_imp"] * d.data["M"]))
        p = P.generate("force_control", s, 3)
        dd, a = p.data, p.answer
        assert a["f_tip_PI"] == dd["fd"]
        assert np.isclose(a["fe_P"], -dd["d"] / (1 + dd["kfp"]))
        roots = np.roots(a["char_poly"])
        assert (roots.real.max() < 0) == a["stable"] == (dd["kfi"] < a["kfi_max"])
    assert len(kinds) >= 2
    stab = {P.generate("force_control", s, 3).answer["stable"] for s in range(20)}
    assert stab == {True, False}


def test_grasp_closure_uses_grasping_module_and_varies():
    from mrstudio import grasping as GR
    for diff in (1, 2, 3):
        seen = set()
        for s in range(16):
            p = P.generate("grasp_closure", s, diff)
            d, a = p.data, p.answer
            seen.add(a["closed"])
            cs = [GR.Contact(pt, n, d["mu"]) for pt, n in d["contacts"]]
            assert np.allclose(GR.wrench_matrix(cs, friction=d["mu"] > 0), a["F"])
            if diff >= 2:
                assert a["closed"] == (d["mu"] > a["mu_min"])
                assert np.isfinite(a["mu_min"])
        assert seen == {True, False}, diff


def test_mobile_kin_matches_book_matrices():
    from mrstudio import mobile as MB
    for s in range(8):
        p = P.generate("mobile_kin", s, 1)
        d, a = p.data, p.answer
        assert np.isclose(a["u_LR"].mean() * d["r"], d["v"])                        # v = r (u_L + u_R) / 2
        p = P.generate("mobile_kin", s, 2)
        d, a = p.data, p.answer
        assert np.allclose(a["H0"], MB.H_matrix(MB.three_omni_wheels(d["r"], d["d"])))
        assert np.allclose(a["u_phi"], a["H0"] @ a["Vb_phi"])
        p = P.generate("mobile_kin", s, 3)
        d, a = p.data, p.answer
        assert np.allclose(a["F_odo"] @ a["H0"], np.eye(3))
        res = d["u_meas"] - a["H0"] @ a["Vb_odo"]
        assert np.allclose(a["H0"].T @ res, 0)                                       # least-squares residual


# --- export -----------------------------------------------------------------------------------------

def test_problem_set_versions_share_topics():
    a = P.problem_set(9, P.MIXED, 2, seed=5, version=0)
    b = P.problem_set(9, P.MIXED, 2, seed=5, version=1)
    assert [p.topic for p in a] == [p.topic for p in b] == [P.TOPIC_KEYS[k % len(P.TOPIC_KEYS)]
                                                             for k in range(9)]
    assert all(x.statement != y.statement for x, y in zip(a, b))
    md = P.problem_set_markdown(a, with_solutions=True, version=0)
    assert "version A" in md and "# Solutions" in md and md.count("## Problem") == 18


def test_export_markdown_and_word(tmp_path):
    probs = P.problem_set(7, P.MIXED, 2, seed=1)
    md = P.export_markdown(str(tmp_path / "q.md"), probs)
    txt = Path(md).read_text(encoding="utf-8")
    assert "\\begin{bmatrix}" in txt and "Solutions" not in txt
    if P.find_pandoc() is None:
        pytest.skip("pandoc not installed")
    doc = P.export_docx(str(tmp_path / "q.docx"), probs, with_solutions=True)
    with zipfile.ZipFile(doc) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    assert "<m:oMath" in xml                  # LaTeX became native Word equations
    assert "Solutions" in xml


# --- GUI -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def test_practice_tab_headless(qapp, tmp_path):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.practice_tab import THEORY, PracticeTab
    from mrstudio.gui.theory import validate_all
    assert validate_all([THEORY]) == []
    tab = PracticeTab(AppContext())
    assert tab.uses_serial_robot is False and "Practice" in tab.title
    for i in range(tab.topic.count()):           # every topic draws (robot, frames or plot)
        tab.topic.setCurrentIndex(i)
        assert tab.problem is not None
    tab.topic.setCurrentIndex(0)
    tab.seed.setValue(3)
    assert tab.problem.seed == 3 and tab.problem.topic == P.TOPIC_KEYS[3]
    tab.reveal_btn.setChecked(True)
    assert "Worked solution" in tab.solution.content() and "Step 1" in tab.solution.content()
    assert tab.verify_lbl.text().startswith("✓")
    tab.seed.setValue(4)                          # a new problem hides the solution again
    assert not tab.reveal_btn.isChecked() and "Try the problem first" in tab.solution.content()
    tab.n_spin.setValue(3)
    tab.v_spin.setValue(2)
    files = tab.export_markdown_to(str(tmp_path / "quiz.md"), key=True)
    assert sorted(os.path.basename(f) for f in files) == ["quiz_A.md", "quiz_A_solutions.md", "quiz_B.md",
                                                          "quiz_B_solutions.md"]
    files = tab.export_word_to(str(tmp_path / "quiz.docx"))
    assert len(files) == 2 and all(os.path.exists(f) for f in files)


# --- review additions ---------------------------------------------------------------------------------

def _inline_math_segments(md: str) -> list[str]:
    out = []
    for block in md.split("\n\n"):
        if block.startswith("$$"):
            continue
        out += re.findall(r"\$([^$]+)\$", block)
    return out


def test_markdown_inline_math_is_pandoc_safe():
    """pandoc ignores ``$x = $`` (space before the closing dollar): Word would show raw TeX."""
    for t in P.TOPIC_KEYS:
        for d in (1, 2, 3):
            p = P.generate(t, 2, d)
            md = p.markdown(1, with_solution=True)
            bad = [m for m in _inline_math_segments(md) if m != m.strip()]
            assert not bad, (t, d, bad[:3])


def test_pandoc_parses_every_formula():
    import json
    import subprocess
    exe = P.find_pandoc()
    if exe is None:
        pytest.skip("pandoc not installed")
    probs = [P.generate(t, s, d) for t in P.TOPIC_KEYS for d in (1, 2, 3) for s in (0, 1)]
    md = P.problem_set_markdown(probs, with_solutions=True)
    r = subprocess.run([exe, "-f", "markdown", "-t", "json"], input=md, capture_output=True, text=True, timeout=120)
    leftovers = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("t") == "Str" and "$" in x["c"] or x.get("t") in ("RawInline", "RawBlock"):
                leftovers.append(x["c"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json.loads(r.stdout))
    assert not leftovers, leftovers[:5]


def test_3rs_tip_on_base_axis_is_named():
    hits = 0
    for s in range(300):
        p = P.generate("jacobian", s, 2)
        if p.visual["spec"].family == "3Rs" and p.answer["singular"] and abs(np.sin(p.data["theta"][2])) > 1e-9:
            assert "base (joint-1) axis" in p.steps[-1].doc.markdown()
            hits += 1
    assert hits >= 1


def test_practice_tab_export_runs_in_worker(qapp, tmp_path):
    from PySide6 import QtCore
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.practice_tab import PracticeTab
    tab = PracticeTab(AppContext())
    tab.n_spin.setValue(2)
    tab.v_spin.setValue(2)
    prm = tab.export_params()
    tab._run_export(lambda: tab._write_markdown(str(tmp_path / "w.md"), prm))
    assert tab._worker is not None and not tab.exp_md_btn.isEnabled()
    loop = QtCore.QEventLoop()
    tab._worker.finished.connect(loop.quit)
    QtCore.QTimer.singleShot(30000, loop.quit)
    if tab._worker.isRunning():
        loop.exec()
    for _ in range(5):
        qapp.processEvents()
    assert tab._worker is None and tab.exp_md_btn.isEnabled()
    assert "Wrote 2 file(s)" in tab.exp_lbl.text()
    assert (tmp_path / "w_A.md").exists() and (tmp_path / "w_B.md").exists()
