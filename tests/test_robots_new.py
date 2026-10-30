"""Every robot in the library (incl. prismatic and URDF-imported ones) vs Pinocchio / Corke; URDF import."""
import os
import sys

import modern_robotics as mr
import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrstudio import urdf_import as U  # noqa: E402
from mrstudio.dynamics import to_pinocchio  # noqa: E402
from mrstudio.robots import ROBOTS, custom_robot, get_robot, trans  # noqa: E402

NAMES = list(ROBOTS)
DH = {"PUMA 560 (from Corke DH)": "Puma560", "Franka Panda (7R redundant)": "Panda",
      "Stanford arm (RRPRRR)": "Stanford", "SCARA Cobra 600 (RRPR)": "Cobra600"}


def _configs(r, k=6, seed=0):
    rng = np.random.default_rng(seed)
    return [(r.random_config(rng), rng.uniform(-1, 1, r.n), rng.uniform(-1, 1, r.n)) for _ in range(k)]


def test_new_robots_registered():
    for nm in ("KUKA iiwa 7 (7R, URDF)", "Stanford arm (RRPRRR)", "SCARA Cobra 600 (RRPR)"):
        assert nm in ROBOTS
    assert get_robot("Stanford arm (RRPRRR)").joint_types == "RRPRRR"
    assert get_robot("SCARA Cobra 600 (RRPR)").joint_types == "RRPR"
    assert get_robot("KUKA iiwa 7 (7R, URDF)").n == 7


@pytest.mark.parametrize("name", NAMES)
def test_fk_mr_vs_pinocchio(name):
    r = get_robot(name)
    model = to_pinocchio(r)
    data = model.createData()
    fid = model.getFrameId("ee")
    for th, _, _ in _configs(r):
        pin.framesForwardKinematics(model, data, th)
        assert np.abs(r.fk(th) - data.oMf[fid].homogeneous).max() < 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_rnea_mr_vs_pinocchio_and_spd(name):
    r = get_robot(name)
    assert r.has_dynamics
    model = to_pinocchio(r)
    data = model.createData()
    for th, dth, ddth in _configs(r):
        tau_mr = mr.InverseDynamics(th, dth, ddth, r.gravity, np.zeros(6), r.Mlist, r.Glist, r.Slist)
        tau_pin = pin.rnea(model, data, th, dth, ddth)
        assert np.abs(tau_mr - tau_pin).max() < 1e-9 * max(1.0, np.abs(tau_pin).max())
        Mm = np.asarray(mr.MassMatrix(th, r.Mlist, r.Glist, r.Slist))
        assert np.allclose(Mm, Mm.T, atol=1e-12)
        assert np.linalg.eigvalsh(Mm).min() > 0


@pytest.mark.parametrize("name", [n for n in DH if n in ROBOTS])
def test_dh_robots_fk_vs_rtb(name):
    import roboticstoolbox as rtb
    r = get_robot(name)
    ref = getattr(rtb.models.DH, DH[name])()
    for th, _, _ in _configs(r):
        assert np.abs(r.fk(th) - np.asarray(ref.fkine(th).A)).max() < 1e-12


def test_prismatic_geometry_and_limits():
    r = get_robot("Stanford arm (RRPRRR)")
    assert r.is_prismatic.tolist() == [False, False, True, False, False, False]
    assert np.allclose(r.Slist[:3, 2], 0) and np.isclose(np.linalg.norm(r.Slist[3:, 2]), 1)
    rng = np.random.default_rng(1)
    for _ in range(20):
        th = r.random_config(rng)
        assert r.joint_limits[2, 0] <= th[2] <= r.joint_limits[2, 1]
    # the slide is drawn from the sleeve (fixed on link 2) to the sliding link 3: its length = travel
    a, b = r.home.copy(), r.home.copy()
    a[2], b[2] = 0.4, 0.9
    ga, gb = r.geometry(a), r.geometry(b)
    assert np.allclose(ga["joint_pos"][2], gb["joint_pos"][2])
    La = np.linalg.norm(ga["polys"][3][1] - ga["polys"][3][0])
    Lb = np.linalg.norm(gb["polys"][3][1] - gb["polys"][3][0])
    assert np.isclose(Lb - La, 0.5)


def test_custom_robot_prismatic():
    r = custom_robot("t", [[0, 0, 1], [1, 0, 0], [0, 0, 1]], [[0, 0, 0], [0.3, 0, 0], [0.3, 0, 0]],
                     trans([0.6, 0, 0]), joint_types="RPR")
    assert r.joint_types == "RPR" and r.has_dynamics
    assert np.allclose(r.Slist[:, 1], [0, 0, 0, 1, 0, 0])
    model = to_pinocchio(r)
    data = model.createData()
    th = np.array([0.3, 0.2, -0.5])
    pin.framesForwardKinematics(model, data, th)
    assert np.abs(r.fk(th) - data.oMf[model.getFrameId("ee")].homogeneous).max() < 1e-12


# ---------------------------------------------------------------- URDF import
def test_urdf_validation_checks_pass():
    checks = U.validation_checks()
    assert len(checks) >= 7
    for c in checks:
        assert c.passed, (c.name, c.error)


def test_urdf_string_and_kinds():
    r = U.from_urdf(U.TEST_URDF, tip="tool0")
    assert r.urdf.joint_kinds == ["revolute", "continuous", "prismatic", "revolute"]
    assert r.joint_types == "RRPR"
    assert np.allclose(r.tau_max, [80, 40, 300, 10])
    assert np.allclose(r.joint_limits[2], [-0.1, 0.25])
    e = U.compare_with_pinocchio(r, U.TEST_URDF)
    assert max(e.values()) < 1e-12


def test_urdf_branched_rejected():
    branched = U.TEST_URDF.replace("</robot>", """<joint name="s" type="revolute"><parent link="l1"/>
      <child link="sl"/><origin xyz="0 0.1 0"/><axis xyz="1 0 0"/>
      <limit lower="-1" upper="1" effort="1" velocity="1"/></joint><link name="sl"/></robot>""")
    with pytest.raises(ValueError, match="branched"):
        U.from_urdf(branched)
    r = U.from_urdf(branched, tip="tool0")
    assert r.n == 4 and r.urdf.locked == ["s"]


def test_urdf_examples_convert():
    for label, path in U.example_urdfs().items():
        leaves = U.leaf_links(path)   # branched models (e.g. Panda's two fingers) need an explicit tip
        r = U.from_urdf(path, tip=leaves[0] if len(leaves) > 1 else None)
        e = U.compare_with_pinocchio(r, path, n_samples=3)
        assert max(e.values()) < 1e-12, label
