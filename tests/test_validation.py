"""The cross-validation suite must pass completely -- and must be able to fail (negative controls)."""
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere: `pytest tests`

import numpy as np
import pytest

from mrstudio import kinematics as kin
from mrstudio import validation as V
from mrstudio.dynamics import DynamicsEngine
from mrstudio.robots import ROBOTS, get_robot


@pytest.fixture(scope="module")
def run():
    t0 = time.perf_counter()
    checks = V.run_all()
    return checks, time.perf_counter() - t0


def test_all_checks_pass(run):
    checks, _ = run
    failed = [c for c in checks if not V.is_skipped(c) and not c.passed]
    msg = "\n\n".join(f"[{c.category}] {c.name}: error {c.error:.3e} > tol {c.tol:.1e}\n{c.detail}" for c in failed)
    assert not failed, msg


def test_suite_is_fast(run):
    _, elapsed = run
    # 60 s on a desktop; shared CI runners (GitHub sets CI=true) are 2-3x slower, especially on Windows
    limit = 300 if os.environ.get("CI") else 60
    assert elapsed < limit, f"validation suite took {elapsed:.1f} s (limit {limit} s)"


def test_builtin_checks_in_planned_order(run):
    checks, _ = run
    planned = [name for _, name, _ in V.planned_checks()]
    assert len(planned) >= 90
    assert [c.name for c in checks[:len(planned)]] == planned


def test_every_robot_and_family_covered(run):
    checks, _ = run
    names = " ".join(c.name for c in checks)
    for robot in ROBOTS:
        assert V.short(robot) in names
    for key in ("Pinocchio", "Corke", "mr.IKinBody", "Euler–Lagrange", "skew-symmetric", "Energy",
                "Power balance", "finite differences", "CRBA", "ABA", "Statics"):
        assert key in names, key
    cats = {c.category for c in checks}
    assert {V.KIN, V.IKC, V.STAT, V.DYN, V.PHYS} <= cats


def test_progress_and_streaming_callbacks():
    """progress goes 0 -> 1 monotonically; on_result streams every check in order."""
    fracs, streamed = [], []
    tasks = V._task_list()[:3]
    orig = V._task_list
    try:
        V._task_list = lambda: tasks
        saved = V.OPTIONAL_MODULES[:]
        V.OPTIONAL_MODULES[:] = []
        out = V.run_all(progress=lambda f, m: fracs.append(f), on_result=streamed.append)
    finally:
        V._task_list = orig
        V.OPTIONAL_MODULES[:] = saved
    assert out == streamed and len(out) == 3
    assert fracs[0] == 0.0 and fracs[-1] == 1.0 and np.all(np.diff(fracs) >= 0)


def test_html_report(run, tmp_path):
    checks, elapsed = run
    html = V.report_html(checks, elapsed)
    assert html.startswith("<!doctype html>") and "</html>" in html
    assert html.count("<tr class=") == len(checks)
    assert "checks passed" in html
    (tmp_path / "report.html").write_text(html, encoding="utf-8")


def test_missing_optional_module_is_skipped_not_failed():
    out = V._optional_checks("mrstudio.this_module_does_not_exist", "Test")
    assert len(out) == 1 and V.is_skipped(out[0])
    s = V.summary(out)
    assert s["skipped"] == 1 and s["total"] == 0 and s["all_passed"]


#: number of checks in a full run when this test was written (504 before the 2026-10 review fixes, + 3 gravity
#: vs hand-CoM checks, + 3 s̈ finite-difference checks). Adding checks is fine; losing some is not.
MIN_CHECKS = 510


def test_no_optional_module_is_skipped(run):
    """A missing optional module turns into a quiet SKIP in the GUI; in the test suite it must be an error."""
    checks, _ = run
    skipped = [c.name for c in checks if V.is_skipped(c)]
    assert not skipped, skipped
    # every optional module contributed checks (modules may file them under their own category name, e.g.
    # trajectory.py -> "Trajectory & control"; _optional_checks tags the last one with "[module: N checks ...]")
    for modname, cat in V.OPTIONAL_MODULES:
        assert any(f"[{modname}: " in (c.detail or "") for c in checks), f"{modname} ({cat}) contributed no checks"
    assert len(checks) >= MIN_CHECKS, f"{len(checks)} checks < {MIN_CHECKS}"


def test_informational_checks_not_counted(run):
    checks, _ = run
    info = [c for c in checks if V.is_informational(c)]
    assert info and all("Euler" in c.name for c in info)
    s = V.summary(checks)
    assert s["informational"] == len(info)
    assert s["total"] == len(checks) - len(info) - s["skipped"]


def test_run_all_can_be_cancelled():
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 2

    out = V.run_all(cancelled=cancelled)
    assert len(out) == 2


def test_gravity_hand_check_has_teeth(monkeypatch):
    """Moving one link's mass in the hand sum (not in the engines) must be detected."""
    robot = get_robot("UR5 (6R industrial)")
    err, tol, _ = V.gravity_vs_hand_com(robot)
    assert err <= tol
    good = type(robot).masses
    monkeypatch.setattr(type(robot), "masses", property(lambda r: good.fget(r) * np.r_[1.0, 1.0, 1.01, 1, 1, 1]))
    err, tol, _ = V.gravity_vs_hand_com(robot)
    assert err > tol


# ---------------------------------------------------------------- negative controls: the checks have teeth
def test_detects_wrong_jacobian(monkeypatch):
    robot = get_robot("UR5 (6R industrial)")
    good = kin.jacobian_space
    monkeypatch.setattr(kin, "jacobian_space", lambda r, th: good(r, th) * (1 + 1e-4))
    err, tol, _ = V.js_vs_finite_difference(robot)
    assert err > tol


def test_detects_wrong_inertia(monkeypatch):
    """A 1 % error in one link inertia used by ONE implementation must be caught."""
    robot = get_robot("Planar 2R arm")
    G = [g.copy() for g in robot.Glist]
    G[1][2, 2] *= 1.01
    orig = V._eng

    def wrong(r, backend, g=None):
        if backend == "mr":
            return DynamicsEngine(r, "mr", g=g, Glist=G)
        return orig(r, backend, g)

    monkeypatch.setattr(V, "_eng", wrong)
    err, tol, _ = V.rnea_mr_vs_pin(robot)
    assert err > tol
    err, tol, _ = V.lagrange_vs_newton_euler(robot)
    assert err > tol


def test_detects_energy_leak():
    """Adding a tiny unmodelled velocity-dependent force breaks energy conservation."""
    robot = get_robot("Planar 2R arm")
    eng = DynamicsEngine(robot, "pinocchio")
    from mrstudio.dynamics import simulate
    th0, dth0 = V._free_motion(robot)
    s = simulate(eng, th0, dth0, lambda t, q, dq: 0.01 * dq, 1.0, dt=0.01, method="rk45")   # "anti-friction"
    E = s.ke + s.pe
    assert np.abs(E - E[0]).max() / s.ke.max() > 1e-6
