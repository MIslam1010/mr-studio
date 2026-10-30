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
    assert [P.generate(P.MIXED, s).topic for s in range(7)] == P.TOPIC_KEYS
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


# --- export -----------------------------------------------------------------------------------------

def test_problem_set_versions_share_topics():
    a = P.problem_set(9, P.MIXED, 2, seed=5, version=0)
    b = P.problem_set(9, P.MIXED, 2, seed=5, version=1)
    assert [p.topic for p in a] == [p.topic for p in b] == [P.TOPIC_KEYS[k % 7] for k in range(9)]
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
