"""Closed chains (mrstudio/closed_chains.py, MR Ch. 7 and Sec. 8.7): loop closure of the four-bar and the
five-bar over sweeps, and the sign convention of the cut-pin force lambda."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio import closed_chains as cc


@pytest.mark.parametrize("sigma", [1, -1])
def test_fourbar_loop_closure_over_crank_sweep(sigma):
    fb = cc.FourBar()
    worst, n = 0.0, 0
    for th1 in np.linspace(-np.pi, np.pi, 361):
        th = fb.fk_closed_form([th1], sigma)
        if th is None:
            continue
        n += 1
        phi, _, _ = fb.constraint(th)
        worst = max(worst, float(np.linalg.norm(phi)))
        assert fb.assembly_mode(th) == sigma
    assert n > 300 and worst < 1e-10


def test_fourbar_newton_matches_closed_form_and_velocity_closure():
    fb = cc.FourBar()
    rng = np.random.default_rng(0)
    for th1 in np.linspace(-3.0, 3.0, 25):
        th = fb.fk_closed_form([th1], 1)
        if th is None:
            continue
        guess = th + rng.normal(0, 0.05, fb.n)
        thn, ok, hist = fb.fk_newton([th1], guess)
        assert ok and hist[-1] < 1e-10
        assert np.allclose(np.angle(np.exp(1j * (thn - th))), 0, atol=1e-8)
        dth = fb.passive_rates(th, [1.3])                         # A theta_dot = 0 (MR 7.15)
        assert np.linalg.norm(fb.constraint(th)[1] @ dth) < 1e-10


@pytest.mark.parametrize("elbows", [m for m, _ in cc.WORKING_MODES])
def test_fivebar_loop_closure_over_workspace_sweep(elbows):
    fv = cc.FiveBar()
    lo, hi = fv.bounds()
    worst, n = 0.0, 0
    for x in np.linspace(lo[0], hi[0], 25):
        for y in np.linspace(lo[1], hi[1], 25):
            th = fv.ik([x, y], elbows)
            if th is None:
                continue
            n += 1
            worst = max(worst, float(np.linalg.norm(fv.constraint(th)[0])))
            assert np.allclose(fv.tip(th)[:2], [x, y], atol=1e-10)
            # forward kinematics from the actuated joints reproduces a closed loop through the same tip
            th_f = fv.fk_closed_form(th[fv.actuated], fv.assembly_mode(th))
            assert th_f is not None and np.linalg.norm(fv.constraint(th_f)[0]) < 1e-10
            assert np.allclose(fv.tip(th_f)[:2], [x, y], atol=1e-8)
    assert n > 50 and worst < 1e-10


def test_lambda_is_pin_force_on_branch_1():
    """M th_dd + h = S tau + A^T lambda with A = d(p1 - p2)/d theta: lambda is the force on branch 1's tip,
    i.e. lambda_here = -lambda_MR of MR Eq. (8.93) (tau = M th_dd + h + A^T lambda_MR).  Static check:
    the four-bar held at rest under gravity -- the pin force on branch 1 must balance that branch's
    gravity torque about its base: tau_branch1 = -J1^T lambda + g_1."""
    fb = cc.FourBar()
    th = fb.fk_closed_form([0.7], 1)
    z = np.zeros(fb.n)
    tau_a, lam, _ = fb.inverse_dynamics(th, z, z)
    g = fb.bias(th, z)                                         # gravity only (at rest)
    _, A, _ = fb.constraint(th)
    S = fb.S
    assert np.allclose(S @ tau_a + A.T @ lam, g, atol=1e-10)
    # branch 2 (passive rocker) is in equilibrium under gravity and -lambda alone (no motor):
    s2 = fb.sl[1]
    J2 = -A[:, s2]                                             # A = [J1, -J2] in the planar components
    assert np.allclose(g[s2], J2.T @ (-lam), atol=1e-10)
