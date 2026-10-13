"""Haptics & impedance: the 1-DOF paddle model, sampled-data wall, passivity indicator, presets and the tab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio import haptics as hp


@pytest.fixture(scope="module")
def dev():
    return hp.default_device()


def test_validation_checks_pass():
    checks = hp.validation_checks()
    assert len(checks) >= 10
    bad = [(c.name, c.error, c.detail) for c in checks if not c.passed]
    assert not bad, bad


def test_device_numbers_match_course(dev):
    assert dev.m == pytest.approx(0.0570, abs=1e-4)          # SBEG 461 W12 WE7
    assert dev.k_g == pytest.approx(1.2111, abs=1e-3)
    assert dev.ratio == pytest.approx(17.38, abs=0.01)
    assert dev.force_to_current(1.5) == pytest.approx(0.777, abs=1e-3)
    assert dev.tau_max == pytest.approx(0.18)


def test_colgate_condition(dev):
    assert hp.colgate_K_max(0.15, 0.0, 1e-3) == pytest.approx(300.0)
    assert hp.colgate_margin(0.15, 200.0, 0.0, 1e-3) > 0
    assert hp.colgate_margin(0.15, 200.0, 0.1, 1e-3) < 0      # virtual damping does not help passivity
    assert hp.colgate_K_max(0.15, 0.2, 1e-3) == 0.0


def test_exact_limit_falls_with_delay(dev):
    ks = [hp.max_stable_K(dev.m, dev.b, dev.k_g, 1.0, 1e-3, d) for d in (0, 2, 5)]
    assert ks[0] > ks[1] > ks[2]
    for d, k in zip((0, 2, 5), ks):          # first-order estimate within ~5 %
        assert abs(k - hp.first_order_K_limit(dev.b, 1.0, 1e-3, d)) / k < 0.06


def test_release_test_matches_lecture(dev):
    """Week 12 Sec. 6.2 figure: B = 1, 5 ms delay: K = 100 decays, K = 300 grows."""
    for K, grows in ((100.0, False), (300.0, True)):
        ren = hp.Rendering(shape="spring", K=K, B=1.0, delay_steps=5, saturate=False, quantize=False)
        sim = hp.simulate(dev, ren, hp.Hand(kind="none"), 1.5, x0=0.005)
        assert (hp.oscillation_growth(sim) > 1) == grows
        if grows:
            assert np.nanmax(sim.net) > 1e-3       # the sampled spring injects energy


def test_wall_is_unilateral(dev):
    ren = hp.Rendering(shape="wall", K=200.0, B=1.0, x0=0.005)
    assert ren.env_force(0, 0.004, 1.0) == 0.0
    assert ren.env_force(0, 0.006, 0.0) == pytest.approx(-0.2)  # B v = 0
    assert ren.env_force(0, 0.006, -10.0) == 0.0               # never pulls the hand in


@pytest.mark.parametrize("name", list(hp.PRESETS))
def test_presets_run(dev, name):
    ren, hand, t_end, x0 = hp.preset(name)
    sim = hp.simulate(dev, ren, hand, t_end, x0=x0)
    rep = hp.analyse(dev, ren, hand)
    assert np.all(np.isfinite(sim.x))
    assert np.all(np.abs(sim.F_cmd) <= dev.force_max + 1e-12)
    if name.startswith("Release"):
        assert not rep.stable
    else:
        assert rep.stable and hp.oscillation_growth(sim) < 1


def test_casualty_contact_meets_hw6(dev):
    ren, hand, t_end, x0 = hp.preset("Casualty-handling soft contact (S6)")
    sim = hp.simulate(dev, ren, hand, t_end, x0=x0)
    assert np.max(np.abs(sim.f_hand)) <= 1.5 and abs(sim.f_hand[-1]) <= 0.8


def test_admittance_light_virtual_mass_is_harder(dev):
    """MR p. 447: admittance control struggles to render low mass."""
    hand = hp.Hand(kind="hand", m=0.05, b=0.5, k=100.0, x_end=0.01)
    ok = hp.simulate(dev, hp.Rendering(mode="admittance", shape="spring", K=50, B=1, M_v=0.5, Kp=150,
                                       x0=0.0), hand, 1.5)
    assert np.all(np.isfinite(ok.x)) and hp.oscillation_growth(ok) < 1


def test_impedance_magnitude():
    w = np.array([0.0, 10.0])
    assert hp.impedance_magnitude(1.0, 0.0, 100.0, w)[1] == pytest.approx(0.0)
    assert hp.impedance_magnitude(1.0, 2.0, 100.0, w)[0] == pytest.approx(100.0)


# ------------------------------------------------------------------ GUI
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def test_tab_headless(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.haptics_tab import THEORY, HapticsTab
    from mrstudio.gui.theory import validate_all

    assert validate_all([THEORY]) == []
    tab = HapticsTab(AppContext())
    assert tab.uses_serial_robot is False and "Haptics" in tab.title
    for name in hp.PRESETS:
        tab.preset_box.setCurrentText(name)
        assert tab.sim is not None and len(tab.sim.t) > 10
    tab.mode_box.setCurrentIndex(1)             # admittance
    tab.K.set_value(1200, emit=True)
    tab.recompute()
    assert tab.Mv.isVisibleTo(tab)
    assert "stable" in tab.stab_lbl.text()
    tab._draw(len(tab.sim.t) // 2)
    tab.on_show()
    tab._anim_step()
    tab.on_hide()
    import importlib.util
    assert tab.hw_btn.isEnabled() == (importlib.util.find_spec("serial") is not None)


def test_admittance_verdict_matches_simulation(dev):
    """Review fix: the admittance stability verdict is the exact loop of the admittance controller.
    It used to analyse only the inner spring K_p and said 'stable' while the simulation diverged."""
    hand = hp.Hand(kind="hand", m=0.05, b=1.0, k=150.0, x_end=0.01)
    for Kp, Mv in ((100, 0.05), (800, 0.3), (1500, 0.3), (3000, 0.05)):
        ren = hp.Rendering(mode="admittance", shape="spring", K=200, B=1, M_v=Mv, Kp=Kp,
                           quantize=False, saturate=False)
        rep = hp.analyse(dev, ren, hand)
        sim = hp.simulate(dev, ren, hand, 3.0, x0=0.003)
        assert rep.stable == (hp.oscillation_growth(sim) < 1), (Kp, Mv, rep.rho)


def test_admittance_matrix_matches_simulation(dev):
    hand = hp.Hand(kind="hand", m=0.05, b=1.0, k=150.0, x_end=0.0, t_move=0.0)
    ren = hp.Rendering(mode="admittance", shape="spring", K=120, B=0.8, M_v=0.25, Kp=300, Kd=0.3,
                       delay_steps=1, quantize=False, saturate=False)
    sim = hp.simulate(dev, ren, hand, 0.2, x0=0.004)
    A = hp.admittance_loop_matrix(dev, ren, hand)
    z = np.zeros(7)
    z[0] = z[2] = z[3] = 0.004
    xs = []
    for _ in sim.t:
        xs.append(z[0])
        z = A @ z
    assert np.max(np.abs(np.array(xs) - sim.x)) < 1e-9


def test_impedance_loop_matrix_equals_closed_loop_matrix(dev):
    hand = hp.Hand(kind="hand", m=0.05, b=1.0, k=150.0)
    ren = hp.Rendering(shape="spring", K=400, B=0.5, delay_steps=2)
    A1 = hp.loop_matrix(dev, ren, hand)
    A2 = hp.closed_loop_matrix(dev.m + 0.05, dev.b + 1.0, dev.k_g + 150.0, 400, 0.5, 1e-3, 2)
    assert np.allclose(A1, A2)
    assert hp.max_stable_gain(dev, ren, hp.Hand(kind="none")) == pytest.approx(
        hp.max_stable_K(dev.m, dev.b, dev.k_g, 0.5, 1e-3, 2), abs=1.0)


def test_admittance_leak_is_cancelled_damping(dev):
    """With the ideal admittance force (mass reshaping + gravity compensation) the leak no longer
    contains those state-function terms; what is left is the tracking loop cancelling the device
    damping b: with tight tracking leak -> D_b (the physical damping no longer protects passivity)."""
    hand = hp.Hand(kind="hand", m=0.05, b=1.0, k=150.0, x_end=0.01)
    ren = hp.Rendering(mode="admittance", shape="spring", K=50, B=1, M_v=0.3, Kp=1000, Kd=0.3, T=1e-4,
                       quantize=False, saturate=False)
    sim = hp.simulate(dev, ren, hand, 1.5)
    assert abs(sim.net[-1]) < 0.1 * sim.D_b[-1]
