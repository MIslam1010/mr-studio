"""Open-chain kinematics (mrstudio/kinematics.py, MR Ch. 4-6): IK round trips, IKinSpace agreement,
multiple IK branches, angle wrapping into joint limits and the force ellipsoid at a singularity."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import modern_robotics as mr
import numpy as np
import pytest

from mrstudio import kinematics as kin
from mrstudio.robots import get_robot

UR5, PUMA, PANDA, P2R = ("UR5 (6R industrial)", "PUMA 560 (from Corke DH)", "Franka Panda (7R redundant)",
                         "Planar 2R arm")


@pytest.mark.parametrize("name", [UR5, PUMA, P2R])
@pytest.mark.parametrize("frame", ["body", "space"])
def test_ik_round_trip(name, frame):
    r = get_robot(name)
    rng = np.random.default_rng(1)
    po = r.planar and r.n < 3
    for _ in range(20):
        th = r.random_config(rng)
        T = kin.fk_space(r, th)
        res = kin.ik_newton_raphson(r, T, th + rng.normal(0, 0.15, r.n), eomg=1e-10, ev=1e-10, max_iter=60,
                                    frame=frame, position_only=po)
        assert res.success
        T2 = kin.fk_space(r, res.theta)
        assert np.allclose(T2[:3, 3], T[:3, 3], atol=1e-8)
        if not po:
            assert np.allclose(T2[:3, :3], T[:3, :3], atol=1e-8)


@pytest.mark.parametrize("name", [UR5, PUMA])
def test_space_frame_matches_mr_ikinspace(name):
    """frame='space' stops on ||w_s||, ||v_s|| exactly like mr.IKinSpace (20 iterations): same final theta."""
    r = get_robot(name)
    rng = np.random.default_rng(2026)
    n_ok = 0
    for _ in range(100):                     # 2 robots x 100 = 200 seeded cases
        th = r.random_config(rng)
        T = kin.fk_space(r, th)
        th0 = th + rng.normal(0, 0.3, r.n)
        eomg, ev = 10.0 ** rng.uniform(-6, -2), 10.0 ** rng.uniform(-6, -2)
        res = kin.ik_newton_raphson(r, T, th0, eomg=eomg, ev=ev, max_iter=20, frame="space")
        th_mr, ok = mr.IKinSpace(r.Slist, r.M, T, th0, eomg, ev)
        assert res.frame == "space"
        assert np.allclose(res.theta, th_mr, atol=1e-9, rtol=0)
        assert res.success == ok
        n_ok += ok
        # the reported errors are the space-frame errors
        Tsb = mr.FKinSpace(r.M, r.Slist, res.theta)
        Vs = mr.Adjoint(Tsb) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tsb) @ T))
        assert res.err_w[-1] == pytest.approx(np.linalg.norm(Vs[:3]), abs=1e-12)
        assert res.err_v[-1] == pytest.approx(np.linalg.norm(Vs[3:]), abs=1e-12)
    assert n_ok > 50


def test_body_frame_matches_mr_ikinbody():
    r = get_robot(UR5)
    rng = np.random.default_rng(5)
    for _ in range(30):
        th = r.random_config(rng)
        T = kin.fk_space(r, th)
        th0 = th + rng.normal(0, 0.3, r.n)
        res = kin.ik_newton_raphson(r, T, th0, eomg=1e-6, ev=1e-6, max_iter=20, frame="body")
        th_mr, ok = mr.IKinBody(r.Blist, r.M, T, th0, 1e-6, 1e-6)
        assert np.allclose(res.theta, th_mr, atol=1e-9) and res.success == ok


def test_ik_all_solutions_finds_8_ur5_branches():
    r = get_robot(UR5)
    th = np.array([0.4, -1.1, 1.3, -0.9, 0.7, 0.3])
    T = kin.fk_space(r, th)
    sols = kin.ik_all_solutions(r, T, n_starts=60)
    assert len(sols) == 8
    for s in sols:
        assert np.allclose(kin.fk_space(r, s), T, atol=1e-5)
    assert min(kin.joint_distance(r, s, th) for s in sols) < 1e-4       # the original branch is among them
    # pairwise distinct
    for i in range(8):
        for j in range(i):
            assert kin.joint_distance(r, sols[i], sols[j]) > 1e-3


def test_ik_all_solutions_cancel_and_cap():
    r = get_robot(UR5)
    T = kin.fk_space(r, np.array([0.4, -1.1, 1.3, -0.9, 0.7, 0.3]))
    assert kin.ik_all_solutions(r, T, cancelled=lambda: True) == []
    calls = []
    kin.ik_all_solutions(r, T, n_starts=5, patience=0, max_starts=5, progress=calls.append)
    assert calls and all(0 <= f <= 1 for f in calls)


def test_planar_2r_all_solutions_and_analytic():
    r = get_robot(P2R)
    th = np.array([0.5, 1.1])
    T = kin.fk_space(r, th)
    an = kin.ik_analytic_planar(r, T[0, 3], T[1, 3])
    assert len(an) == 2 and min(kin.joint_distance(r, a, th) for a in an) < 1e-12
    sols = kin.ik_all_solutions(r, T, position_only=True)
    assert len(sols) == 2
    for a in an:
        assert min(kin.joint_distance(r, a, s) for s in sols) < 1e-5
    # atan2 form is accurate near the stretched elbow (c2 -> 1), where arccos loses half the digits
    th = np.array([0.3, 1e-7])
    T = kin.fk_space(r, th)
    an = kin.ik_analytic_planar(r, T[0, 3], T[1, 3])
    assert np.allclose(kin.fk_space(r, an[0])[:2, 3], T[:2, 3], atol=1e-12)


def test_wrap_keeps_panda_joint6_solution():
    r = get_robot(PANDA)
    lo, hi = r.joint_limits[:, 0], r.joint_limits[:, 1]
    th = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 3.5, 0.5])                   # theta6 = 200.5 deg is feasible
    assert np.all(th >= lo) and np.all(th <= hi)
    naive = kin.wrap_angles(th)
    assert naive[5] < lo[5]                                                   # plain wrapping breaks it
    w = kin.wrap_joints(r, th)
    assert np.allclose(w, th) and kin.within_limits(r, w)
    w2 = kin.wrap_joints(r, th + np.r_[0, 0, 0, 0, 0, -2 * np.pi, 2 * np.pi])
    assert np.allclose(w2, th)
    # the same pose found by IK from another seed is reported inside the limits
    T = kin.fk_space(r, th)
    res = kin.ik_newton_raphson(r, T, th + 0.05, eomg=1e-10, ev=1e-10, max_iter=60)
    assert res.success and kin.within_limits(r, kin.wrap_joints(r, res.theta))


def test_wrap_never_touches_prismatic_joints():
    names = [n for n in ("Stanford arm (RRPRRR)", "SCARA Cobra 600 (RRPR)") if _exists(n)]
    if not names:
        pytest.skip("no robot with prismatic joints registered")
    for n in names:
        r = get_robot(n)
        th = np.where(r.is_prismatic, 4.0, 5.0)        # 4 m on a prismatic joint must stay 4 m
        w = kin.wrap_joints(r, th, into_limits=False)
        assert np.all(w[r.is_prismatic] == 4.0)
        assert np.allclose(w[~r.is_prismatic], 5.0 - 2 * np.pi)


def _exists(name):
    from mrstudio.robots import ROBOTS
    return name in ROBOTS


def test_force_ellipsoid_infinite_at_singularity():
    r = get_robot(P2R)
    el = kin.manipulability(r, np.array([0.4, 0.0]), "linear", force=True)    # stretched elbow: singular
    assert np.isinf(el.radii[:2]).sum() == 1 and el.unbounded.sum() == 1
    assert el.radii[2] == 0.0 and not el.task[2]                                 # padded z axis of the plane
    el = kin.manipulability(r, np.array([0.4, 1.0]), "linear", force=True)
    assert np.all(np.isfinite(el.radii)) and np.all(el.radii[:2] > 0)
    v = kin.manipulability(r, np.array([0.4, 1.0]), "linear", force=False)
    assert np.allclose(np.sort(el.radii[:2]), np.sort(1 / v.radii[:2]))
    # planar 3R fully stretched: the tip cannot move along the arm -> infinite force semi-axis there
    r3 = get_robot("Planar 3R arm")
    el = kin.manipulability(r3, np.array([0.3, 0.0, 0.0]), "linear", force=True)
    k = int(np.flatnonzero(el.unbounded)[0])
    assert el.unbounded.sum() == 1 and abs(abs(el.axes[:2, k] @ [np.cos(0.3), np.sin(0.3)]) - 1) < 1e-6


def test_singularity_test_is_unit_consistent():
    r = get_robot(UR5)
    assert kin.is_singular(r, np.array([0.2, -0.8, 0.0, -0.5, 0.9, 0.0]))       # elbow singularity
    assert kin.is_singular(r, np.array([0.2, -0.8, 1.0, -0.5, 0.0, 0.0]))       # wrist singularity
    assert not kin.is_singular(r, r.home)
    sw, sv = kin.split_singular_values(r, r.home)
    assert len(sw) == 3 and len(sv) == 3
