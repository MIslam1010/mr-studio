"""Regression tests for bugs found during development."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrstudio.dynamics import DynamicsEngine  # noqa: E402
from mrstudio.robots import ROBOTS, get_robot  # noqa: E402


def test_coriolis_does_not_mutate_gravity():
    """DynamicsEngine.coriolis() (Pinocchio) once zeroed model.gravity for every later call."""
    for name in ROBOTS:
        r = get_robot(name)
        rng = np.random.default_rng(3)
        th, dth = r.random_config(rng), rng.normal(size=r.n)
        eng = DynamicsEngine(r, "pinocchio")
        g0 = eng.gravity_torque(th)
        tau0 = eng.inverse(th, dth, np.zeros(r.n))
        eng.coriolis(th, dth)
        assert np.allclose(eng.gravity_torque(th), g0), name
        assert np.allclose(eng.inverse(th, dth, np.zeros(r.n)), tau0), name
        ref = DynamicsEngine(r, "mr").coriolis(th, dth)
        assert np.allclose(eng.coriolis(th, dth), ref, atol=1e-10), name
