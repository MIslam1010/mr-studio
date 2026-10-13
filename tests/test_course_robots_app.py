"""SBEG 361/461 course robots in MR Studio: registered under the agreed names, numerically identical to the
course materials' shared ``course_robots.py`` (when that file exists), consistent with Pinocchio, and usable
in every serial-robot tab."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere: `pytest tests`

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio import course_models as C  # noqa: E402
from mrstudio import kinematics as kin  # noqa: E402
from mrstudio.dynamics import DynamicsEngine, simulate  # noqa: E402
from mrstudio.robots import ROBOTS, get_robot  # noqa: E402

NAMES = ["Course arm (myCobot-280 class 6R)", "EOD arm (6R on UGV)", "EOD arm (3R planar)",
         "Arm–exoskeleton 2R", "Rehab robot 2R (planar)", "Hapkit (1-DOF haptic paddle)"]
SHARED = (Path.home() / "Documents/Edu/Lectures/Robotics/Robotics Track Course Materials/_build/shared"
          / "course_robots.py")


def _states(r, k=8, seed=0):
    rng = np.random.default_rng(seed)
    return [(r.random_config(rng), rng.uniform(-1, 1, r.n), rng.uniform(-1, 1, r.n), rng.uniform(-1, 1, 6))
            for _ in range(k)]


# ----------------------------------------------------------------------------- registry
def test_names_registered_and_existing_robots_kept():
    for nm in NAMES:
        assert nm in ROBOTS
    for nm in ("Planar 2R arm", "Planar 3R arm", "UR5 (6R industrial)", "PUMA 560 (from Corke DH)",
               "Franka Panda (7R redundant)", "KUKA iiwa 7 (7R, URDF)", "Stanford arm (RRPRRR)",
               "SCARA Cobra 600 (RRPR)"):
        assert nm in ROBOTS
    assert list(C.COURSE_ROBOTS) == NAMES and set(C.SHARED_KEYS) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_model_is_complete(name):
    r = get_robot(name)
    assert r.has_dynamics and len(r.Mlist) == r.n + 1 and len(r.Glist) == r.n
    assert r.joint_limits.shape == (r.n, 2) and np.all(r.joint_limits[:, 0] < r.joint_limits[:, 1])
    assert np.all(r.home >= r.joint_limits[:, 0]) and np.all(r.home <= r.joint_limits[:, 1])
    assert r.tau_max.shape == (r.n,) and np.all(r.tau_max > 0)
    assert r.description and r.source and r.course_key == C.SHARED_KEYS[name]
    for G in r.Glist:                          # physically valid spatial inertias (MR Eq. 8.32)
        assert np.allclose(G, G.T) and np.all(np.linalg.eigvalsh(G) > 0)
    # the textbook M equals the product of the CoM-frame chain (MR Sec. 8.3.2)
    Mprod = np.linalg.multi_dot(r.Mlist) if r.n + 1 > 1 else r.Mlist[0]
    assert np.allclose(Mprod, r.M, atol=1e-12)


def test_expected_structure():
    n = {nm: get_robot(nm).n for nm in NAMES}
    assert list(n.values()) == [6, 6, 3, 2, 2, 1]
    planar = {nm: get_robot(nm).planar for nm in NAMES}
    assert planar == {NAMES[0]: False, NAMES[1]: False, NAMES[2]: True, NAMES[3]: True, NAMES[4]: True,
                      NAMES[5]: True}
    assert np.allclose(get_robot("Rehab robot 2R (planar)").gravity, [0, 0, -9.81])   # horizontal plane
    assert np.allclose(get_robot("EOD arm (3R planar)").gravity, [0, -9.81, 0])       # vertical plane


# ----------------------------------------------------------------------------- shared source of truth
@pytest.fixture(scope="module")
def shared():
    if not SHARED.exists():
        pytest.skip(f"shared course models not found at {SHARED}")
    spec = importlib.util.spec_from_file_location("_shared_course_robots", SHARED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", NAMES)
def test_numbers_equal_shared_file(shared, name):
    r, s = get_robot(name), shared.get(C.SHARED_KEYS[name])
    assert np.array_equal(r.Slist, s["Slist"]) and np.allclose(r.M, s["M"], atol=1e-15)
    assert len(r.Mlist) == len(s["Mlist"]) and len(r.Glist) == len(s["Glist"])
    for a, b in zip(r.Mlist, s["Mlist"]):
        assert np.allclose(a, b, atol=1e-15)
    for a, b in zip(r.Glist, s["Glist"]):
        assert np.allclose(a, b, atol=1e-15)
    assert np.allclose(r.joint_limits, s["limits"]) and np.allclose(r.tau_max, s["tau_max"])
    assert np.allclose(r.home, s["q_home"]) and np.allclose(r.gravity, s["g"])
    if s["qdot_max"] is not None:
        assert np.allclose(r.qdot_max, s["qdot_max"])
    assert r.joint_types == s["joint_types"]


@pytest.mark.parametrize("name", NAMES)
def test_fk_and_dynamics_equal_shared_functions(shared, name):
    r, s = get_robot(name), shared.get(C.SHARED_KEYS[name])
    engines = [DynamicsEngine(r, "mr"), DynamicsEngine(r, "pinocchio")]
    for q, dq, ddq, F in _states(r):
        assert np.allclose(r.fk(q), shared.fk_space(s, q), atol=1e-13)
        assert np.allclose(kin.fk_body(r, q), shared.fk_body(s, q), atol=1e-12)
        tau = shared.inverse_dynamics(s, q, dq, ddq, F)
        Ms, gs = shared.mass_matrix(s, q), shared.gravity_torques(s, q)
        for e in engines:
            assert np.allclose(e.inverse(q, dq, ddq, F), tau, atol=1e-9)
            assert np.allclose(e.mass_matrix(q), Ms, atol=1e-10)
            assert np.allclose(e.gravity_torque(q), gs, atol=1e-9)


def test_course_arm_dh_table_equals_shared(shared):
    r, s = get_robot(NAMES[0]), shared.course_arm()
    for q, *_ in _states(r):
        assert np.allclose(C.dh_fk(r.dh, q), shared.dh_fk(s["dh"], q), atol=1e-13)


def test_hapkit_params_equal_shared(shared):
    r, s = get_robot(NAMES[-1]), shared.hapkit()
    for k in ("rh", "rs", "rp", "ratio", "reflected_rotor_inertia", "force_max", "x_range"):
        assert r.params[k] == pytest.approx(s["params"][k])


# ----------------------------------------------------------------------------- core modules with each robot
def test_validation_checks_pass():
    checks = C.validation_checks()
    assert len(checks) == 5
    bad = [f"{c.name}: {c.error:.3e} > {c.tol:.1e}\n{c.detail}" for c in checks if not c.passed]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", NAMES)
def test_kinematics_core(name):
    r = get_robot(name)
    for q, *_ in _states(r, 4):
        Js, Jb = kin.jacobian_space(r, q), kin.jacobian_body(r, q)
        assert Js.shape == Jb.shape == (6, r.n)
        assert np.all(np.isfinite(kin.singular_values(r, q)))
        kin.manipulability(r, q)
    # IK from a nearby guess reaches the target (position-only for <= 2 joints)
    rng = np.random.default_rng(3)
    q = r.home + 0.1 * (r.joint_limits[:, 1] - r.joint_limits[:, 0]) * 0.1
    T = r.fk(q)
    res = kin.ik_newton_raphson(r, T, q + rng.normal(0, 0.05, r.n), eomg=1e-8, ev=1e-8, max_iter=100,
                                position_only=r.n <= 2)
    assert res.success and np.allclose(r.fk(res.theta)[:3, 3], T[:3, 3], atol=1e-7)


@pytest.mark.parametrize("name", NAMES)
def test_simulation_conserves_energy(name):
    r = get_robot(name)
    eng = DynamicsEngine(r, "pinocchio")
    s = simulate(eng, r.home, np.zeros(r.n), lambda t, q, dq: np.zeros(r.n), 0.3, 1e-3, method="rk45")
    E = s.ke + s.pe
    assert np.abs(E - E[0]).max() <= 1e-6 * max(1.0, np.abs(s.pe).max())


# ----------------------------------------------------------------------------- GUI: every serial tab
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def _serial_tab_classes():
    from mrstudio.gui.app import TABS
    out = []
    for mod, cls in TABS:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                continue                       # tab not written yet (skipped by the app as well)
            raise
        k = getattr(m, cls)
        if getattr(k, "uses_serial_robot", True):
            out.append(k)
    return out


def test_every_serial_tab_accepts_every_course_robot(qapp):
    from mrstudio.gui.base import AppContext
    errors = []
    old_hook = sys.excepthook
    sys.excepthook = lambda *exc: errors.append(exc)        # exceptions raised inside Qt slots / timers
    try:
        for cls in _serial_tab_classes():
            ctx = AppContext()
            tab = cls(ctx)
            tab.resize(1400, 900)
            for nm in NAMES:
                ctx.set_robot(nm)
                assert ctx.robot.name == nm
                tab.on_show()
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 0.15:
                    qapp.processEvents()
                tab.on_hide()
                qapp.processEvents()
            tab.on_hide()
            tab.deleteLater()
            qapp.processEvents()
    finally:
        sys.excepthook = old_hook
    assert not errors, "\n".join(f"{e[0].__name__}: {e[1]}" for e in errors)


# ----------------------------------------------------------------------------- review additions
def test_short_labels_unique():
    """validation.short() maps both "EOD arm (...)" robots to "EOD"; course_models.SHORT must disambiguate."""
    assert set(C.SHORT) == set(NAMES)
    assert len(set(C.SHORT.values())) == len(NAMES)


def test_course_models_importable_first():
    """course_models and robots import each other; either import order must work in a fresh interpreter."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    for stmt in ("import mrstudio.course_models as C; assert len(C.COURSE_ROBOTS) == 6",
                 "import mrstudio.robots as R; assert 'Hapkit (1-DOF haptic paddle)' in R.ROBOTS"):
        subprocess.run([sys.executable, "-c", stmt], cwd=root, check=True)


def test_hapkit_grip_kinematics():
    """{b} on a circle of radius r_h about the pivot: p(θ) = r_h(−sin θ, cos θ, 0) (MR Eq. 4.14 for one joint)."""
    r = get_robot(NAMES[-1])
    rh = C.HAPKIT["rh"]
    for th in np.linspace(*r.joint_limits[0], 7):
        assert np.allclose(r.fk([th])[:3, 3], rh * np.array([-np.sin(th), np.cos(th), 0.0]), atol=1e-15)
    assert r.joint_limits[0, 1] * rh == pytest.approx(C.HAPKIT["x_range"])          # ±40 mm arc
    assert r.tau_max[0] == pytest.approx(C.HAPKIT["force_max"] * rh)                 # 2 N at the grip


def test_eod_3r_gravity_by_hand():
    """g_i(0) = Σ_{k≥i} m_k g (x_ck − x_i) for the stretched vertical-plane arm (statics, MR Sec. 8.1)."""
    p = C.EOD_3R
    xj = np.r_[0.0, np.cumsum(p["L"])][:3]
    xc = xj + np.asarray(p["r"])
    g_hand = [sum(p["m"][k] * C.G0 * (xc[k] - xj[i]) for k in range(i, 3)) for i in range(3)]
    g_app = DynamicsEngine(get_robot(NAMES[2]), "mr").gravity_torque(np.zeros(3))
    assert np.allclose(g_app, g_hand, atol=1e-12)
    assert np.allclose(g_hand, C.EOD_3R_GRAVITY_AT_ZERO, atol=5e-3)
