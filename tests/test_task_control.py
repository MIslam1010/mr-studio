"""Task-space motion, force, hybrid and impedance control (MR Ch. 11) -- mrstudio.task_control."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from mrstudio import task_control as tc
from mrstudio.robots import get_robot


@pytest.fixture(scope="module")
def checks():
    return tc.validation_checks()


def test_all_validation_checks_pass(checks):
    assert len(checks) >= 20
    failed = [(c.name, c.error, c.tol) for c in checks if not c.passed]
    assert not failed, failed
    assert all(c.category == tc.CAT for c in checks)


def test_projection_properties_random():
    rng = np.random.default_rng(5)
    rob = get_robot("Franka Panda (7R redundant)")
    kin = tc.Kin(rob)
    eng = tc.DynamicsEngine(rob, "pinocchio")
    for _ in range(3):
        th, dth = rob.random_config(rng), rng.uniform(-1, 1, rob.n)
        X, Jb, Vb, ab = kin.state(th, dth)
        rep = tc.TaskRep(rob, "body")
        Lam, eta, NT, h, _ = tc.task_dynamics(eng, th, dth, rep.jac(X, Jb), ab)
        A = rep.constraint(X, np.array([0, 0, 1.0]))
        P = tc.projection(A, Lam)
        assert np.allclose(P @ P, P, atol=1e-9)
        assert np.allclose(A @ np.linalg.inv(Lam) @ P, 0, atol=1e-9)
        # dynamically consistent null space: J M^-1 N^T = 0
        J = rep.jac(X, Jb)
        assert np.allclose(J @ np.linalg.inv(eng.mass_matrix(th)) @ NT, 0, atol=1e-8)


def test_square_task_dynamics_matches_book_formula():
    """Lam = J^-T M J^-1 (MR 8.90) and eta = J^-T h - Lam Jdot J^-1 V (8.91) for a non-redundant arm."""
    rob = get_robot("UR5 (6R industrial)")
    kin, eng = tc.Kin(rob), tc.DynamicsEngine(rob, "pinocchio")
    th, dth = rob.home + 0.1, np.full(6, 0.3)
    X, Jb, Vb, ab = kin.state(th, dth)
    Lam, eta, NT, h, _ = tc.task_dynamics(eng, th, dth, Jb, ab)
    Ji = np.linalg.inv(Jb)
    assert np.allclose(Lam, Ji.T @ eng.mass_matrix(th) @ Ji, rtol=1e-8, atol=1e-8)
    assert np.allclose(eta, Ji.T @ h - Lam @ ab, rtol=1e-8, atol=1e-8)
    assert np.abs(NT).max() < 1e-9


@pytest.mark.parametrize("name", ["SCARA Cobra 600 (RRPR)", "Planar 2R arm"])
def test_underactuated_task_rows_and_motion(name):
    try:
        rob = get_robot(name)
    except Exception:
        pytest.skip(f"{name} not available")
    rows = tc.task_rows(rob)
    assert len(rows) <= rob.n
    th = tc.start_config(rob)
    tr = tc.make_task_trajectory(rob, rob.fk(th), "circle", tc.default_size(rob), 3.0, 1e-3)
    r = tc.torque_control("motion", rob, th, 3.0, tc.Gains(), traj=tr)
    assert not r.diverged and r.pos_err.max() < 2e-3


def test_dls_pinv_bounded_at_singularity():
    J = np.array([[1.0, 1.0], [0.0, 1e-9]])
    Jp, smin, lam = tc.dls_pinv(J)
    assert lam > 0 and np.abs(Jp).max() < 1e3
