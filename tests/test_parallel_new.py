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
    assert tab.push_sim is not None
    tab._push_step()
    tab.push_btn.setChecked(False)
    tab.on_hide()
    app.processEvents()
