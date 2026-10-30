"""Parallel mechanisms: Stewart dynamics, 3xRPR (MR 7.1.1), Delta robot (MR Ex. 2.7 / 7.7) and the new tab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio import delta, parallel, parallel_planar


@pytest.mark.parametrize("mod", [parallel, parallel_planar, delta])
def test_validation_checks_pass(mod):
    checks = mod.validation_checks()
    assert checks
    bad = [f"{c.name}: {c.error:.3e} > {c.tol:.1e}" for c in checks if not c.passed]
    assert not bad, "\n".join(bad)
    assert all(c.category == "Parallel mechanisms" for c in checks)


def test_stewart_dynamic_equals_static_at_rest_and_differs_when_moving():
    sp = parallel.StewartPlatform()
    body = sp.rigid_body(25.0)
    d = sp.dance_dynamics(body, n=121)
    assert np.allclose(d["f_dyn"][0], d["f_static"][0], atol=1e-8)
    assert np.abs(d["f_dyn"] - d["f_static"]).max() > 1.0
    assert np.abs(d["power"] - d["dE"]).max() < 1e-8


def test_stewart_push_servo_is_stable_open_loop_is_not():
    sp = parallel.StewartPlatform()
    body = sp.rigid_body(25.0)
    on = sp.push_simulation(body, t_final=1.5, h=0.004, servo=True)
    assert on["t"][-1] == pytest.approx(1.5)
    assert np.abs(on["s"][-1] - on["s"][0]).max() < 1e-3
    off = sp.push_simulation(body, t_final=1.5, h=0.004, servo=False)
    assert off["t"][-1] < 1.5


def test_rpr_six_solutions_and_negative_control():
    r = parallel_planar.ThreeRPR()
    assert len(r.fk_all(parallel_planar.SIX_SOLUTION_LEGS).q) == 6
    assert r.fk_all([0.2, 0.2, 1.2]).q == []          # incompatible leg lengths -> no assembly


def test_delta_unreachable_and_branches():
    d = delta.DeltaRobot()
    assert d.ik([0.0, 0.0, -2.0]) is None
    p = d.p_home
    for br in [(0, 0, 0), (1, 1, 1), (0, 1, 0)]:
        th = d.ik(p, br)
        assert np.abs(d.closure(th, p)).max() < 1e-12


def test_parallel_tab_builds_and_buttons_work():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.parallel_tab import THEORY, ParallelRobotsTab
    from mrstudio.gui.theory import validate_all
    assert validate_all([THEORY]) == []
    tab = ParallelRobotsTab(AppContext())
    tab._r_six()
    assert len(tab.rpr_sols) == 5
    tab.r_anim.setChecked(True); tab._tick(); tab.r_anim.setChecked(False)
    tab.robot_box.setCurrentIndex(1)
    tab.d_anim.setChecked(True); tab._tick(); tab.d_anim.setChecked(False)
    tab.d_mode.setCurrentIndex(1)
    tab.d_ang[0].set_value(1.0, emit=True)
    assert np.abs(tab.dlt.closure(tab.th, tab.p)).max() < 1e-10
    tab.d_ws.setChecked(True)
    tab.on_hide()
    app.processEvents()


def test_stewart_tab_push_button():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.stewart_tab import StewartTab
    tab = StewartTab(AppContext())
    tab.push_btn.setChecked(True)
    tab.push_jobs.wait(60)                 # the simulation runs in a background thread
    assert tab.push_sim is not None and tab.push_timer.isActive()
    tab._push_step()
    tab.push_btn.setChecked(False)
    tab.on_hide()
    app.processEvents()


def test_stewart_legs_in_compression_holding_platform_at_home():
    """Physical sign: legs below a platform hold it up by pushing, i.e. compression (f > 0 by the
    convention F_s = J^T f).  Checked against an independent force/moment balance."""
    sp = parallel.StewartPlatform()
    T = sp.T_home
    F = sp.payload_wrench(T, 25.0)                     # gravity wrench (m, f) in {s} on platform + payload
    chk = sp.leg_forces_checked(T, F)
    f = chk.value
    assert chk.warning is None and chk.kappa < 1e3
    assert np.all(f > 0)                               # every leg pushes (compression)
    x, _, u = sp.legs(T)                               # u_i: unit vector from base joint to platform joint
    assert np.all(u[:, 2] > 0)                         # legs point up, so pushing = +u_i on the platform
    force = (f[:, None] * u).sum(axis=0)
    moment = (f[:, None] * np.cross(x, u)).sum(axis=0)
    assert np.allclose(force + F[3:], 0, atol=1e-9) and np.allclose(moment + F[:3], 0, atol=1e-9)
    total = (25.0 + sp.platform_mass) * parallel.G_ACC
    assert f @ u[:, 2] == pytest.approx(total)
    # symmetric home pose, centred load: all six legs share it equally
    assert np.allclose(f, f[0])


def test_stewart_statics_flags_type2_singularity():
    sp = parallel.StewartPlatform()
    # yawing the platform by 90 deg about z at home makes the legs coplanar-ish: Type-II singularity nearby
    best = None
    for yaw in np.radians(np.linspace(60, 120, 241)):
        T = parallel.pose(z=sp.z_home, yaw=yaw)
        k = np.linalg.cond(sp.jacobian_normalised(T))
        if best is None or k > best[0]:
            best = (k, T)
    chk = sp.leg_forces_checked(best[1], sp.payload_wrench(best[1], 10.0))
    assert chk.kappa == pytest.approx(best[0], rel=1e-6)
    assert chk.kappa > parallel.KAPPA_WARN and chk.warning is not None      # yaw = 90 deg: singular
    assert parallel.condition_warning(1e7) is not None and parallel.condition_warning(10.0) is None


def test_delta_statics_condition_flag():
    d = delta.DeltaRobot()
    p = d.p_home
    chk = d.static_torques_checked(d.ik(p), p, [0.0, 0.0, -10.0])
    assert chk.warning is None and np.isfinite(chk.kappa)
    assert np.allclose(chk.value, d.static_torques(d.ik(p), p, [0.0, 0.0, -10.0]))
