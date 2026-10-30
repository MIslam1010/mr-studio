"""Trajectories (MR Ch. 9): speed limits, time-scaling derivatives, TOPP-RA limits / endpoints / cancel."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere: `pytest tests`

import numpy as np
import pytest

from mrstudio import trajectory as tj
from mrstudio.dynamics import DynamicsEngine
from mrstudio.robots import get_robot


# ---------------------------------------------------------------- joint speed limits
def test_default_velocity_limits_use_qdot_max():
    for name in ("Course arm (myCobot-280 class 6R)", "EOD arm (6R on UGV)", "Rehab robot 2R (planar)"):
        r = get_robot(name)
        assert r.qdot_max is not None
        np.testing.assert_allclose(tj.default_velocity_limits(r), np.broadcast_to(r.qdot_max, (r.n,)))
    fake = SimpleNamespace(name="custom", n=3, qdot_max=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(tj.default_velocity_limits(fake), [1.0, 2.0, 3.0])
    scalar = SimpleNamespace(name="custom", n=2, qdot_max=4.0)
    np.testing.assert_allclose(tj.default_velocity_limits(scalar), [4.0, 4.0])


def test_default_velocity_limits_fallbacks():
    np.testing.assert_allclose(tj.default_velocity_limits(SimpleNamespace(name="x", n=2, qdot_max=None)),
                               np.full(2, 2.5 * np.pi))
    np.testing.assert_allclose(tj.default_velocity_limits(SimpleNamespace(name="UR5 (6R industrial)", n=6)),
                               np.full(6, np.pi))
    iiwa = get_robot("KUKA iiwa 7 (7R, URDF)")
    if getattr(iiwa, "qdot_max", None) is None:
        np.testing.assert_allclose(tj.default_velocity_limits(iiwa),
                                   np.radians([98, 98, 100, 130, 140, 180, 180]))


# ---------------------------------------------------------------- time scalings
@pytest.mark.parametrize("kind", tj.SCALINGS)
def test_sddot_matches_finite_differences(kind):
    err, _ = tj.sddot_fd_error(kind, 2.0)
    assert err < 1e-5, err
    # and directly, on a coarser grid, away from the trapezoid's acceleration jumps
    Tf, h = 3.0, 1e-5
    t = np.linspace(0.05, Tf - 0.05, 301)
    if kind == "trapezoidal":
        v, a = tj.trapezoid_params(Tf)
        t = t[(np.abs(t - v / a) > 1e-3) & (np.abs(t - (Tf - v / a)) > 1e-3)]
    _, _, sdd = tj.time_scaling(kind, Tf, t)
    fd = (tj.time_scaling(kind, Tf, t + h)[1] - tj.time_scaling(kind, Tf, t - h)[1]) / (2 * h)
    assert np.abs(fd - sdd).max() < 1e-6 * max(1.0, np.abs(sdd).max())


# ---------------------------------------------------------------- TOPP-RA
def _topp_case(name):
    r = get_robot(name)
    eng = DynamicsEngine(r, "pinocchio")
    lo, hi = r.joint_limits[:, 0] + 0.05, r.joint_limits[:, 1] - 0.05
    goal = np.clip(r.home + np.resize([0.8, -0.6, 0.5, 0.4, -0.5, 0.7], r.n), lo, hi)
    jp = tj.joint_path(r, r.home, goal, "joint", 41)
    vmax = tj.default_velocity_limits(r)
    res = tj.topp(eng, jp, r.tau_max, vmax, n_grid=200, samples=400, compare_quintic=False)
    return r, goal, jp, res


TOPP_ROBOTS = ("Planar 2R arm", "UR5 (6R industrial)", "Course arm (myCobot-280 class 6R)")


@pytest.mark.parametrize("name", TOPP_ROBOTS)
def test_topp_respects_limits(name):
    r, _, _, res = _topp_case(name)
    assert res.ok, res.message
    assert np.abs(res.tau_norm).max() <= 1 + 1e-6
    assert res.vel_norm.max() <= 1 + 1e-6


@pytest.mark.parametrize("name", TOPP_ROBOTS)
def test_topp_endpoints_reach_goal(name):
    r, goal, jp, res = _topp_case(name)
    assert res.ok, res.message
    D = res.duration
    for dt in (D / 10.3, D / 10.7, 0.001):        # grids whose arange stops short of / beyond / near the end
        tr = tj.topp_trajectory(res, jp, dt)
        assert tr.t[-1] == pytest.approx(D, abs=1e-12)
        assert np.all(np.diff(tr.t) > 0)
        assert tr.s[-1] == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(tr.theta[-1], goal, atol=1e-8)
        np.testing.assert_allclose(tr.theta[0], r.home, atol=1e-12)
        assert np.abs(tr.dtheta[-1]).max() < 1e-6 * max(1.0, np.abs(tr.dtheta).max()) + 1e-6


def test_topp_cancel_token():
    r = get_robot("Planar 2R arm")
    eng = DynamicsEngine(r, "pinocchio")
    jp = tj.joint_path(r, r.home, r.home + np.array([-1.2, -1.0]), "joint", 41)
    res = tj.topp(eng, jp, r.tau_max, tj.default_velocity_limits(r), cancelled=lambda: True)
    assert not res.ok and res.message == "cancelled"
    calls = []

    def later():                                   # cancelled after the reachability stage
        calls.append(1)
        return len(calls) > 2

    res = tj.topp(eng, jp, r.tau_max, tj.default_velocity_limits(r), cancelled=later)
    assert not res.ok and res.message == "cancelled"
