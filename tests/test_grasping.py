"""Grasping (MR Ch. 12): contact labels, twist cones, form / force closure LP, friction, the tab (offscreen)."""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mrstudio import grasping as gr

MR_PDF = Path.home() / ("Documents/Edu/Lectures/Robotics/Textbooks/Modern Robotics/"
                        "MODERN ROBOTICS MECHANICS, PLANNING, AND CONTROL-v2, with Hyperlinks.pdf")


def test_validation_checks_pass():
    checks = gr.validation_checks()
    assert len(checks) >= 8
    assert {c.category for c in checks} == {gr.CATEGORY}
    bad = [(c.name, c.error, c.detail) for c in checks if not c.passed]
    assert not bad, bad


def test_normal_wrench_example_12_2():
    c = gr.Contact((1, 2), (0, 1))
    assert np.allclose(gr.normal_wrench(c), [1, 0, 1])          # (m_z, f_x, f_y), MR p. 470
    assert gr.contact_label(c, (1, 2, -1)) == gr.ROLL            # v_x - 2w = 0, v_y + w = 0
    assert gr.contact_label(c, (1, 0, -1)) in (gr.SLIDE_L, gr.SLIDE_R)
    assert gr.contact_label(c, (1, 0, 0)) == gr.BREAK
    assert gr.contact_label(c, (-1, 0, 0)) == gr.PENETRATE


def test_slide_direction_convention():
    """Sr / Sl: the movable body slips right / left as seen facing along the inward normal."""
    c = gr.Contact((0, 0), (0, 1))            # fixed body below, movable body above; facing +y, right is +x
    assert gr.contact_label(c, (0, 1, 0)) == gr.SLIDE_R
    assert gr.contact_label(c, (0, -1, 0)) == gr.SLIDE_L
    # CoR on the normal above the contact, counter-clockwise: contact point moves +x -> Sr (Fig. 12.11 '+, Sr')
    assert gr.contact_label(c, gr.twist_from_cor((0, 1), 1.0)) == gr.SLIDE_R
    assert gr.contact_label(c, gr.twist_from_cor((0, -1), 1.0)) == gr.SLIDE_L


def test_moving_finger_relative_twist():
    c = gr.Contact((0, 0), (0, 1))
    assert gr.contact_label(c, (0, 0, 1), VB=(0, 0, 1)) == gr.ROLL       # moving together
    assert gr.contact_label(c, (0, 0, 0), VB=(0, 0, 1)) == gr.PENETRATE  # finger pushes into a still body


def test_cor_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        c, w = rng.normal(size=2), rng.uniform(0.2, 2) * rng.choice([-1, 1])
        V = gr.twist_from_cor(c, w)
        assert np.allclose(gr.cor(V), c)
        assert np.allclose(gr.point_velocity(V, c), 0)
    assert gr.cor((0, 1, 0)) is None


def test_form_closure_example_12_7():
    pre = gr.PRESETS["Ex. 12.7 — two fingers in a hole (form closure)"]
    r = gr.form_closure(pre["contacts"])
    assert r.closed and r.rank == 3
    assert np.allclose(r.k, [2, 1, 2, 1], atol=1e-7)
    assert np.allclose(r.F, [[0, 0, -1, 2], [-1, 0, 1, 0], [0, -1, 0, 1]])
    r2 = gr.form_closure(gr.PRESETS["Ex. 12.7 — right finger moved down (slides)"]["contacts"])
    assert r2.rank == 3 and not r2.closed
    V = gr.nonzero_twist(gr.PRESETS["Ex. 12.7 — right finger moved down (slides)"]["contacts"])
    assert V is not None and V[2] < 0 and abs(V[0]) < 1e-9          # it slides downward


def test_example_12_4_cone_and_cor_region():
    cs = gr.PRESETS["Ex. 12.4 — three stationary contacts (twist cone)"]["contacts"]
    assert np.allclose(gr.wrench_matrix(cs, friction=False), [[-2, 1, 1], [0, -1, 1], [1, 0, 0]])
    edges = gr.twist_cone_edges(cs)
    assert len(edges) == 3
    for ref in ((0, 0, 1), (1, 1, 2), (1, -1, 2)):
        r = np.array(ref, float) / np.linalg.norm(ref)
        assert min(np.linalg.norm(e - r) for e in edges) < 1e-9
    # '+' CoRs: x <= -2, -1 <= y <= 1; no '-' CoRs
    xs, ys = np.linspace(-4, 2, 61), np.linspace(-3, 3, 61)
    plus, minus = gr.cor_grid(cs, xs, ys)
    X, Y = np.meshgrid(xs, ys)
    assert np.array_equal(plus, (X <= -2 + 1e-9) & (np.abs(Y) <= 1 + 1e-9))
    assert not minus.any()


def test_square_presets_theorem_12_6():
    assert gr.form_closure(gr.PRESETS["Square — four frictionless fingers (form closure)"]["contacts"]).closed
    three = gr.PRESETS["Square — three frictionless fingers (never closure)"]["contacts"]
    assert not gr.form_closure(three).closed
    V = gr.nonzero_twist(three)
    assert V is not None and gr.feasible(three, V, tol=1e-7)


def test_friction_cone_edges_and_force_closure():
    c = gr.Contact((0, 0), (0, 1), mu=1.0)
    e = gr.cone_edges(c)
    assert len(e) == 2 and all(abs(np.linalg.norm(v) - 1) < 1e-12 for v in e)
    # half-angle alpha = atan(mu) (12.16)
    assert all(abs(math.acos(v @ c.n) - math.atan(1.0)) < 1e-12 for v in e)
    tri = gr.PRESETS["Triangle — two fingers, μ = 1 (force closure)"]["contacts"]
    assert gr.force_closure(tri).closed
    assert not gr.force_closure(gr.with_mu(tri, 0.25)).closed
    assert gr.min_friction(tri, tol=1e-7) == pytest.approx(math.tan(math.radians(30)), abs=1e-6)
    assert gr.min_friction(gr.PRESETS["Square — four frictionless fingers (form closure)"]["contacts"]) == 0.0


def test_dynamic_grasp_example_12_9():
    cs = gr.PRESETS["Ex. 12.9 — block on two fingers (μ = 1 and μ = 0)"]["contacts"]
    F = gr.wrench_matrix(cs)
    s2 = math.sqrt(2)
    assert np.allclose(F, np.column_stack([np.array([-4, -1, 1]) / s2, np.array([-2, 1, 1]) / s2, [1, -1, 0]]))
    g = 9.81
    assert gr.balance(F, [0, 0, g]) is None                    # cannot hold the block at rest (p. 498)
    k = gr.balance(F, [0, -2 * g, g])                          # fingers accelerate left at 2g (p. 498)
    assert k is not None and np.all(k >= -1e-9)
    assert np.allclose(F @ k, [0, -2 * g, g])
    assert gr.polygon_centroid(gr.PRESETS["Ex. 12.9 — block on two fingers (μ = 1 and μ = 0)"]["poly"])[0] == \
        pytest.approx(0.0, abs=1e-12)


def test_grasp_quality_continuous_at_mu_zero():
    cs = gr.PRESETS["Square — four frictionless fingers (form closure)"]["contacts"]
    q0 = gr.grasp_quality(cs, friction=False)
    q_eps = gr.grasp_quality(gr.with_mu(cs, 1e-6))
    assert q0 > 0 and q_eps == pytest.approx(q0, rel=1e-4)
    assert gr.grasp_quality(gr.PRESETS["Square — three frictionless fingers (never closure)"]["contacts"],
                            friction=False) == 0.0


def test_planar_exp_matches_rotation_about_cor():
    V = gr.twist_from_cor((1.0, 2.0), 0.5)
    T = gr.planar_exp(V, 1.3)
    a = 0.5 * 1.3
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([1.0, 2.0])
    p = np.array([0.3, -0.4])
    assert np.allclose(T[:2, :2] @ p + T[:2, 2], c + R @ (p - c))


def test_theory_renders_and_citations_match_pdf():
    from mrstudio.gui.theory import validate_all
    from mrstudio.gui.tabs.grasping_tab import THEORY
    assert validate_all([THEORY]) == []
    pairs = re.findall(r"\((1[23]\.\d+)\)[^.]{0,40}?p\.\s*(\d+)", THEORY)
    assert len(pairs) >= 10
    if not MR_PDF.exists() or subprocess.run(["which", "pdftotext"], capture_output=True).returncode:
        pytest.skip("Modern Robotics PDF or pdftotext not available")
    bad = []
    for eq, pg in pairs:
        txt = subprocess.run(["pdftotext", "-f", str(int(pg) + 20), "-l", str(int(pg) + 20), str(MR_PDF), "-"],
                             capture_output=True, text=True).stdout
        if f"({eq})" not in txt:
            bad.append((eq, pg))
    assert not bad, bad


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def test_grasping_tab_smoke(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.grasping_tab import GraspingTab
    tab = GraspingTab(AppContext())
    assert tab.title == "⑫ Grasping" and tab.uses_serial_robot is False
    for i, name in enumerate(gr.PRESETS):
        assert tab.apply_preset(i)
        assert tab.preset_box.currentText() == name
        assert len(re.findall(r"Sl|Sr|R|B|X", tab.mode)) == len(tab.contacts)
    assert tab.apply_preset("triangle") and "Triangle" in tab.preset_box.currentText()
    assert not tab.apply_preset("no such preset")
    tab.apply_preset("Ex. 12.7 — two fingers in a hole (form closure)")
    assert tab.form.closed and tab.force.closed
    # edit a contact through the table: move contact 3 down to (2, 0) with normal 0 deg -> no form closure
    tab.table.item(2, 1).setText("0")
    tab.recompute()
    assert not tab.form.closed
    # twist sliders -> contact mode
    tab.apply_preset("Ex. 12.7 — right finger moved down (slides)")
    tab.w.set_value(0.0, emit=True)
    tab.vx.set_value(0.0, emit=True)
    tab.vy.set_value(-1.0, emit=True)
    assert tab.mode == "SlBSrB"
    # friction for all contacts
    tab.apply_preset("Triangle — two fingers, μ = 0.25 (no force closure)")
    assert not tab.force.closed
    tab.mu_all.set_value(1.0, emit=True)
    tab.recompute()
    assert tab.force.closed and abs(tab.mu_min - math.tan(math.radians(30))) < 2e-4
    n = len(tab.contacts)
    tab._add_contact()
    assert len(tab.contacts) == n + 1
    tab._del_contact()
    assert len(tab.contacts) == n
