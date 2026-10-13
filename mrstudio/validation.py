"""Cross-validation suite: proves every number the app shows is right.

Two kinds of evidence are collected (all with a fixed random seed, so every run is reproducible):

1. **Independent implementations must agree** (to machine precision, ~1e-15):
   * Lynch & Park's own ``modern_robotics`` Python code (the textbook's reference software),
   * Pinocchio (LAAS-CNRS, C++, Featherstone's spatial-algebra algorithms RNEA / CRBA / ABA),
   * Peter Corke's Robotics Toolbox (Denavit-Hartenberg models, its own ``rne``),
   * a symbolic Euler-Lagrange derivation done from scratch with SymPy.
   They share no code, use different conventions (PoE vs DH, (w,v) vs (v,w) twist ordering,
   Newton-Euler vs Lagrange), so agreement can only happen if every one of them is right.

2. **Physical / mathematical laws must hold** (they are true for *any* correct model):
   Jacobian = derivative of FK (finite differences), [Ad] change of frame, M(theta) symmetric
   positive definite, Mdot - 2C skew-symmetric (passivity), conservation of energy without
   torques, power balance tau^T dtheta = dE/dt, statics tau = J^T F (principle of virtual work).

``run_all(progress)`` returns a list of :class:`mrstudio.checks.Check`. It also gathers the
``validation_checks()`` of the optional modules (closed chains, parallel mechanisms,
trajectories, control). ``report_html(checks)`` renders a stand-alone HTML report.
"""
from __future__ import annotations

import html
import importlib
import platform
import time
import traceback
import zlib
from datetime import datetime

import modern_robotics as mr
import numpy as np

from . import kinematics as kin
from .checks import Check
from .dynamics import DynamicsEngine, simulate, to_pinocchio
from .robots import ROBOTS, from_rtb_dh, get_robot, screw_axis_line

SEED = 2017  # Modern Robotics publication year -- every run uses the same "random" samples

# categories (also the colour groups of the Validation tab)
KIN = "Kinematics"
IKC = "Inverse kinematics"
STAT = "Statics"
DYN = "Dynamics"
PHYS = "Physical laws"

#: optional modules written alongside this one; each may expose validation_checks() -> list[Check]
OPTIONAL_MODULES = [
    ("mrstudio.closed_chains", "Closed chains"),
    ("mrstudio.parallel", "Parallel mechanisms"),
    ("mrstudio.trajectory", "Trajectories"),
    ("mrstudio.control", "Control"),
    ("mrstudio.urdf_import", "URDF import"),
    ("mrstudio.parallel_planar", "Parallel mechanisms"),
    ("mrstudio.delta", "Parallel mechanisms"),
    ("mrstudio.task_control", "Task-space & force control"),
    ("mrstudio.featherstone", "Dynamics"),
    ("mrstudio.course_models", "Course robots"),
    ("mrstudio.planning", "Motion planning"),
    ("mrstudio.haptics", "Haptics & impedance"),
    ("mrstudio.hardware.metrics", "Hardware bridge"),
    ("mrstudio.problems", "Practice problems"),
]

#: Corke toolbox DH models -> name of the app robot built from them (None: built here on the fly)
DH_MODELS = {"Puma560": "PUMA 560 (from Corke DH)", "Panda": "Franka Panda (7R redundant)", "UR5": None}

#: one paragraph per family, shown in the details pane and the HTML report
FAMILY_TEXT = {
    KIN: "Forward kinematics and Jacobians are computed by three code bases that share nothing: "
         "Lynch & Park's modern_robotics (product of exponentials), Pinocchio (C++, joint "
         "placements + spatial algebra) and Corke's Robotics Toolbox (Denavit-Hartenberg tables). "
         "The space Jacobian is also checked against a numerical derivative of the forward kinematics.",
    IKC: "Inverse kinematics is checked by plugging the answer back into the forward kinematics, "
         "by comparing with mr.IKinBody (the textbook code) and, for the planar arms, with the "
         "closed-form law-of-cosines solution.",
    STAT: "Statics: the joint torques that balance a tip wrench obtained from tau = J^T F must equal "
          "the torques of Newton-Euler inverse dynamics at rest (principle of virtual work).",
    DYN: "Inverse dynamics, mass matrix and forward dynamics are computed by the textbook's Newton-Euler "
         "code, by Pinocchio's RNEA/CRBA/ABA (Featherstone), by Corke's rne on the original DH model, "
         "by a Lagrangian derived symbolically with SymPy and by a compact pure-NumPy transcription of "
         "Featherstone's own Tables 5.1 / 6.2 / 7.1 (mrstudio/featherstone.py) -- all must agree.",
    PHYS: "Physical laws that any correct model must satisfy: M symmetric positive definite, "
          "Mdot - 2C skew-symmetric (passivity), energy conservation without torques and the power "
          "balance tau^T dtheta - b|dtheta|^2 = dE/dt with motors and friction.",
}


# ============================================================================ helpers
def short(name: str) -> str:
    """'UR5 (6R industrial)' -> 'UR5', 'Planar 2R arm' -> 'Planar 2R'.
    Course robots use course_models.SHORT ('EOD arm (6R on UGV)' and 'EOD arm (3R planar)' would collide)."""
    from .course_models import SHORT as _COURSE_SHORT
    return _COURSE_SHORT.get(name) or name.split(" (")[0].replace(" arm", "")


def _rng(tag: str) -> np.random.Generator:
    """Deterministic generator per check (independent of execution order)."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _pose_err(T1, T2) -> float:
    """Size of the twist that moves T1 to T2 (rad / m) -- frame-independent pose error."""
    return float(np.linalg.norm(mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T1) @ T2))))


_pin_cache: dict = {}


def _pin(robot):
    """(model, data, ee frame id) of the Pinocchio model built from the PoE description."""
    key = (robot.name, id(robot))
    if key not in _pin_cache:
        model = to_pinocchio(robot)
        _pin_cache[key] = (model, model.createData(), model.getFrameId("ee"))
    return _pin_cache[key]


_engines: dict = {}


def _eng(robot, backend: str, g=None) -> DynamicsEngine:
    key = (robot.name, id(robot), backend, None if g is None else tuple(g))
    if key not in _engines:
        _engines[key] = DynamicsEngine(robot, backend, g=g)
    return _engines[key]


def _states(robot, rng, k):
    """k random (theta, dtheta, ddtheta) states: joint angles within limits, rates in [-1, 1]."""
    return [(robot.random_config(rng), rng.uniform(-1, 1, robot.n), rng.uniform(-1, 1, robot.n))
            for _ in range(k)]


_dh_cache: dict = {}


def _dh_pair(rtb_name: str):
    """(Corke DH robot, PoE robot converted from it)."""
    if rtb_name not in _dh_cache:
        import roboticstoolbox as rtb
        r = getattr(rtb.models.DH, rtb_name)()
        app_name = DH_MODELS[rtb_name]
        poe = get_robot(app_name) if app_name else from_rtb_dh(r, f"{rtb_name} (Corke DH -> PoE)")
        _dh_cache[rtb_name] = (r, poe)
    return _dh_cache[rtb_name]


def _dh_label(rtb_name: str) -> str:
    return {"Puma560": "PUMA 560", "Panda": "Franka Panda", "UR5": "UR5 (DH model)"}[rtb_name]


def planar_z_axes(robot) -> bool:
    """Planar arm whose joint axes are all parallel to z (+z or -z): the symbolic Lagrangian applies."""
    return bool(robot.planar and robot.has_dynamics
                and all(abs(abs(robot.Slist[2, i]) - 1) < 1e-9 for i in range(robot.n)))


def analytic_ik_applies(robot) -> bool:
    """Assumptions of kinematics.ik_analytic_planar: 2R/3R, +z axes on the x axis, M a pure x translation."""
    if not robot.planar or robot.n not in (2, 3):
        return False
    S, M = robot.Slist, robot.M
    axes_ok = all(np.allclose(S[:3, i], [0, 0, 1]) and abs(S[3, i]) < 1e-9 and abs(S[5, i]) < 1e-9
                  for i in range(robot.n))
    return bool(axes_ok and np.allclose(M[:3, :3], np.eye(3)) and abs(M[1, 3]) < 1e-9 and abs(M[2, 3]) < 1e-9)


# ============================================================================ Ch. 4-5  kinematics
def fk_space_vs_body(robot):
    rng, err, N = _rng("fk-sb" + robot.name), 0.0, 25
    for _ in range(N):
        th = robot.random_config(rng)
        err = max(err, np.abs(kin.fk_space(robot, th) - kin.fk_body(robot, th)).max())
    return err, 1e-10, (
        f"Space form T = e^[S1]θ1···e^[Sn]θn·M (mr.FKinSpace, MR Eq. 4.14) versus body form "
        f"T = M·e^[B1]θ1···e^[Bn]θn (mr.FKinBody, MR Eq. 4.16) with the body screws obtained as "
        f"B_i = [Ad_M⁻¹]·S_i. {N} random configurations; error = largest |ΔT_ij| (m or dimensionless).")


def fk_vs_pinocchio(robot):
    import pinocchio as pin
    model, data, fid = _pin(robot)
    rng, err, N = _rng("fk-pin" + robot.name), 0.0, 25
    for _ in range(N):
        th = robot.random_config(rng)
        pin.framesForwardKinematics(model, data, th)
        err = max(err, np.abs(data.oMf[fid].homogeneous - kin.fk_space(robot, th)).max())
    return err, 1e-10, (
        f"Pinocchio (C++) model generated from the same screw axes: every joint frame is placed on its "
        f"screw axis and the end-effector is an operational frame 'ee'. pin.framesForwardKinematics "
        f"→ data.oMf['ee'] versus mr.FKinSpace at {N} random configurations; error = max |ΔT_ij|. "
        f"Pinocchio composes SE(3) joint placements, MR multiplies matrix exponentials.")


def dh_fk_vs_poe(rtb_name):
    r, poe = _dh_pair(rtb_name)
    rng, err, N = _rng("dh-fk" + rtb_name), 0.0, 25
    lo, hi = np.asarray(r.qlim, float) if r.qlim is not None else (-np.pi * np.ones(r.n), np.pi * np.ones(r.n))
    for _ in range(N):
        q = rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi))
        err = max(err, np.abs(np.asarray(r.fkine(q).A) - kin.fk_space(poe, q)).max())
    conv = "modified" if r.mdh else "standard"
    return err, 1e-10, (
        f"Peter Corke's Robotics Toolbox model rtb.models.DH.{rtb_name} ({conv} DH table, {r.n} joints): "
        f"robot.fkine(q) (product of DH link transforms A_i) versus the PoE model that robots.from_rtb_dh "
        f"generated automatically (screw axes S_i read from the zero-configuration Jacobian, M = fkine(0)). "
        f"{N} random configurations inside the joint limits; error = max |ΔT_ij|. Proves the DH→PoE "
        f"conversion (MR Appendix C) is exact everywhere, not just at the home pose.")


def dh_jac_vs_poe(rtb_name):
    r, poe = _dh_pair(rtb_name)
    rng, err, N = _rng("dh-jac" + rtb_name), 0.0, 20
    for _ in range(N):
        q = rng.uniform(-np.pi, np.pi, r.n)
        Jc = np.asarray(r.jacob0(q))                      # rows (v_ee, w) in the base frame
        T = kin.fk_space(poe, q)
        Jb = kin.jacobian_body(poe, q)
        J_geo = np.vstack([T[:3, :3] @ Jb[3:], T[:3, :3] @ Jb[:3]])
        err = max(err, np.abs(Jc - J_geo).max())
    return err, 1e-10, (
        f"Corke's geometric Jacobian robot.jacob0(q) of rtb.models.DH.{rtb_name} (rows: linear velocity "
        f"of the tool point, then angular velocity, both in the base frame) versus the MR body Jacobian "
        f"rotated into {{s}}: [R_sb·J_bv; R_sb·J_bω] (MR Sec. 5.1.5). {N} random configurations.")


def js_vs_finite_difference(robot):
    rng, err, N, d = _rng("js-fd" + robot.name), 0.0, 6, 1e-6
    for _ in range(N):
        th = robot.random_config(rng)
        Tinv = mr.TransInv(kin.fk_space(robot, th))
        J_fd = np.zeros((6, robot.n))
        for i in range(robot.n):
            e = np.zeros(robot.n)
            e[i] = d
            Vp = mr.se3ToVec(mr.MatrixLog6(kin.fk_space(robot, th + e) @ Tinv))
            Vm = mr.se3ToVec(mr.MatrixLog6(kin.fk_space(robot, th - e) @ Tinv))
            J_fd[:, i] = (Vp - Vm) / (2 * d)
        err = max(err, np.abs(J_fd - kin.jacobian_space(robot, th)).max())
    return err, 1e-7, (
        f"Definition check of the space Jacobian (MR Sec. 5.1.1): column i is the spatial twist produced by "
        f"joint i alone, [V_s] = Ṫ·T⁻¹. Numerically: V_i ≈ [log(T(θ+δe_i)·T(θ)⁻¹) − log(T(θ−δe_i)·T(θ)⁻¹)]/(2δ) "
        f"(central differences of mr.MatrixLog6, δ = {d:g} rad) versus the analytic "
        f"J_si = [Ad_(e^[S1]θ1···e^[S(i-1)]θ(i-1))]·S_i of mr.JacobianSpace. {N} random configurations; "
        f"truncation + round-off error of the finite difference is ~1e-10.")


def jb_vs_adjoint(robot):
    rng, err, N = _rng("jb-ad" + robot.name), 0.0, 25
    for _ in range(N):
        th = robot.random_config(rng)
        T = kin.fk_space(robot, th)
        err = max(err, np.abs(kin.jacobian_body(robot, th)
                              - mr.Adjoint(mr.TransInv(T)) @ kin.jacobian_space(robot, th)).max())
    return err, 1e-10, (
        f"J_b(θ) computed from the body screw axes (mr.JacobianBody) versus the space Jacobian mapped with "
        f"the adjoint of T_bs = T⁻¹: J_b = [Ad_Tbs]·J_s (MR Eq. 5.22). Tests the adjoint map "
        f"(MR Eqs. 3.83–3.84) and both Jacobian recursions. {N} random configurations.")


def jac_vs_pinocchio(robot):
    import pinocchio as pin
    model, data, fid = _pin(robot)
    rng, err, N = _rng("jac-pin" + robot.name), 0.0, 25
    for _ in range(N):
        th = robot.random_config(rng)
        pin.computeJointJacobians(model, data, th)
        pin.updateFramePlacements(model, data)
        J_loc = np.reshape(pin.getFrameJacobian(model, data, fid, pin.LOCAL), (6, -1))   # (v, w) in {b}; 1-DOF -> (6,)
        J_wld = np.reshape(pin.getFrameJacobian(model, data, fid, pin.WORLD), (6, -1))   # (v, w) of the body point at the {s} origin
        e_b = np.abs(np.vstack([J_loc[3:], J_loc[:3]]) - kin.jacobian_body(robot, th)).max()
        e_s = np.abs(np.vstack([J_wld[3:], J_wld[:3]]) - kin.jacobian_space(robot, th)).max()
        err = max(err, e_b, e_s)
    return err, 1e-10, (
        f"Pinocchio frame Jacobians versus MR. Conventions differ and are converted explicitly: Pinocchio "
        f"stores motions as (v, ω), MR twists as (ω, v). LOCAL frame Jacobian (velocity of 'ee' in its own "
        f"frame) = MR body Jacobian J_b with the two 3-row blocks swapped; WORLD Jacobian (spatial velocity "
        f"at the origin of {{s}}) = MR space Jacobian J_s with blocks swapped. {N} configurations, both checked.")


# ============================================================================ Ch. 6  inverse kinematics
def ik_vs_mr(robot):
    rng, N, tol_nr = _rng("ik-mr" + robot.name), 6, 1e-9
    resid = agree = 0.0
    iters = []
    for _ in range(N):
        th_true = robot.random_config(rng)
        T = kin.fk_space(robot, th_true)
        th0 = th_true + rng.normal(0, 0.15, robot.n)
        r = kin.ik_newton_raphson(robot, T, th0, eomg=tol_nr, ev=tol_nr, max_iter=40)
        th_mr, ok = mr.IKinBody(robot.Blist, robot.M, T, th0, tol_nr, tol_nr)
        if not (r.success and ok):
            return np.inf, 1e-7, f"Newton-Raphson did not converge (ours: {r.success}, mr.IKinBody: {ok})."
        iters.append(r.iterations)
        resid = max(resid, np.abs(kin.fk_space(robot, r.theta) - T).max())
        agree = max(agree, np.abs(_wrap(r.theta - np.asarray(th_mr))).max())
    return max(resid, agree), 1e-7, (
        f"{N} reachable targets T = FK(θ*) with initial guesses θ* + N(0, 0.15 rad). The app's "
        f"kinematics.ik_newton_raphson (MR Sec. 6.2.2, body-frame twist error V_b = log(T⁻¹_sb(θ)·T_sd), "
        f"step Δθ = J_b⁺·V_b) versus the textbook's mr.IKinBody, both with ε_ω = ε_v = {tol_nr:g}. "
        f"Error = max(pose residual max|FK(θ) − T|, joint-angle disagreement with mr.IKinBody) = "
        f"{resid:.1e} / {agree:.1e}. Iterations needed: {min(iters)}–{max(iters)} (quadratic convergence). "
        f"Note: mr.MatrixLog3 computes the angle as arccos((tr R − 1)/2), which cannot resolve rotations "
        f"below ≈ 1.5e-8 rad in double precision — that is the attainable orientation floor, hence tol 1e-7.")


def ik_analytic_vs_numeric(robot):
    rng, N = _rng("ik-an" + robot.name), 12
    e_fk = e_nr = e_true = 0.0
    for _ in range(N):
        th = _wrap(robot.random_config(rng))
        T = kin.fk_space(robot, th)
        x, y, phi = T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])
        sols = kin.ik_analytic_planar(robot, x, y, phi if robot.n >= 3 else None)
        if len(sols) != 2:
            return np.inf, 1e-8, f"analytic IK returned {len(sols)} solutions for a reachable target"
        for s in sols:   # both branches must reproduce the target
            Ts = kin.fk_space(robot, s)
            e_fk = max(e_fk, abs(Ts[0, 3] - x), abs(Ts[1, 3] - y),
                       abs(_wrap(np.arctan2(Ts[1, 0], Ts[0, 0]) - phi)) if robot.n >= 3 else 0.0)
        e_true = max(e_true, min(np.abs(_wrap(s - th)).max() for s in sols))
        r = kin.ik_newton_raphson(robot, T, th + rng.normal(0, 0.1, robot.n), eomg=1e-12, ev=1e-12,
                                  position_only=robot.n == 2)
        e_nr = max(e_nr, min(np.abs(_wrap(s - r.theta)).max() for s in sols))
    return max(e_fk, e_nr, e_true), 1e-8, (
        f"Closed-form planar IK (law of cosines, elbow-up and elbow-down, MR Sec. 6.1) versus numerical "
        f"Newton-Raphson for {N} random targets. Three things are checked: both analytic branches satisfy "
        f"FK(θ) = target ({e_fk:.1e}); the configuration that generated the target is one of the two branches "
        f"({e_true:.1e} rad); Newton-Raphson started nearby converges onto that branch ({e_nr:.1e} rad).")


def ik_round_trip(robot):
    rng, N = _rng("ik-rt" + robot.name), 10
    err, it = 0.0, []
    for _ in range(N):
        th = robot.random_config(rng)
        T = kin.fk_space(robot, th)
        for _attempt in range(3):          # multi-start: a perturbed guess can land beyond a singularity
            r = kin.ik_newton_raphson(robot, T, th + rng.normal(0, 0.3, robot.n), eomg=1e-9, ev=1e-9,
                                      max_iter=100, max_step=0.5, position_only=robot.n == 2)
            if r.success:
                break
        if not r.success:
            return np.inf, 1e-7, "Newton-Raphson failed to converge for one target (3 starts)."
        it.append(r.iterations)
        Tr = kin.fk_space(robot, r.theta)
        # a position-only solve (2-joint arms) controls p only: another elbow branch is a valid answer
        err = max(err, np.abs((Tr - T)[:3, 3] if robot.n == 2 else (Tr - T)).max())
    return err, 1e-7, (
        f"Round trip T → IK(T) → FK(IK(T)) for {N} random reachable targets, initial guesses perturbed by "
        f"N(0, 0.3 rad) and steps limited to 0.5 rad. Error = max |ΔT_ij| between the target and the pose "
        f"actually reached. Iterations: {min(it)}–{max(it)}. (For redundant arms IK(T) is not unique; the "
        f"round trip is the meaningful test.) Floor ≈ 1e-8: rotations below ≈ 1.5e-8 rad are invisible to "
        f"mr.MatrixLog3's arccos, so Newton-Raphson cannot see (or correct) them.")


# ============================================================================ Ch. 5.2 statics
def statics_vs_rnea(robot):
    rng, N = _rng("statics" + robot.name), 5
    e_mr = _eng(robot, "mr", g=np.zeros(3))
    e_pin = _eng(robot, "pinocchio", g=np.zeros(3))
    err = 0.0
    for _ in range(N):
        th = robot.random_config(rng)
        Fb = rng.uniform(-10, 10, 6)                          # tip wrench (m, f) in {b}
        T = kin.fk_space(robot, th)
        Fs = mr.Adjoint(mr.TransInv(T)).T @ Fb                # F_s = [Ad_Tbs]^T F_b
        tau_J = kin.jacobian_space(robot, th).T @ Fs          # tau = J_s^T F_s
        tau_eef = mr.EndEffectorForces(th, Fb, robot.Mlist, robot.Glist, robot.Slist)
        tau_mr = e_mr.inverse(th, np.zeros(robot.n), np.zeros(robot.n), Ftip=Fb)
        tau_pin = e_pin.inverse(th, np.zeros(robot.n), np.zeros(robot.n), Ftip=Fb)
        p = T[:3, 3]
        tau_kin = kin.joint_torques_from_wrench(robot, th, Fs[3:], Fs[:3] - np.cross(p, Fs[3:]))
        err = max(err, *(np.abs(tau_J - x).max() for x in (tau_eef, tau_mr, tau_pin, tau_kin)))
    return err, 1e-9, (
        f"Principle of virtual work (MR Sec. 5.2, Eq. 5.26): τ = J_s(θ)ᵀ·F_s with F_s = [Ad_Tbs]ᵀ·F_b. Compared with "
        f"(i) mr.EndEffectorForces, (ii) MR Newton-Euler inverse dynamics at rest (θ̇ = θ̈ = 0, gravity off, "
        f"F_tip = F_b), (iii) Pinocchio RNEA with the tip wrench applied as an external force on the last "
        f"joint, (iv) kinematics.joint_torques_from_wrench (force at the tool point in {{s}}). {N} random "
        f"configurations and wrenches with components up to 10 N / 10 N·m; error in N·m.")


# ============================================================================ Ch. 8  dynamics
def rnea_mr_vs_pin(robot):
    rng, N = _rng("rnea" + robot.name), 6
    e_mr, e_pin = _eng(robot, "mr"), _eng(robot, "pinocchio")
    err, tmax = 0.0, 0.0
    for k, (th, dth, ddth) in enumerate(_states(robot, rng, N)):
        F = rng.uniform(-5, 5, 6) if k % 2 else None
        a, b = e_mr.inverse(th, dth, ddth, F), e_pin.inverse(th, dth, ddth, F)
        err, tmax = max(err, np.abs(a - b).max()), max(tmax, np.abs(a).max())
    return err, 1e-9, (
        f"Inverse dynamics τ = M(θ)θ̈ + c(θ,θ̇) + g(θ) + Jᵀ F_tip: textbook recursive Newton-Euler "
        f"(mr.InverseDynamics, MR Sec. 8.3.2, twists/wrenches with spatial inertias G_i) versus Pinocchio's "
        f"RNEA (C++, Featherstone). {N} random states (θ in limits, θ̇, θ̈ in [-1, 1]), half of them with a "
        f"random tip wrench. Largest torque {tmax:.1f} N·m; error in N·m.")


def rnea_vs_corke(rtb_name):
    r, poe = _dh_pair(rtb_name)
    rc = r.nofriction(coulomb=True, viscous=True)
    for link in rc.links:
        link.Jm = 0.0
    g = np.array([0.0, 0.0, -9.81])
    rc.gravity = g
    e_mr, e_pin = _eng(poe, "mr", g=g), _eng(poe, "pinocchio", g=g)
    rng, N = _rng("corke" + rtb_name), 6
    err, tmax = 0.0, 0.0
    for _ in range(N):
        q, qd, qdd = rng.uniform(-np.pi, np.pi, r.n), rng.uniform(-1, 1, r.n), rng.uniform(-1, 1, r.n)
        tc = np.asarray(rc.rne(q, qd, qdd), float).ravel()
        err = max(err, np.abs(tc - e_mr.inverse(q, qd, qdd)).max(), np.abs(tc - e_pin.inverse(q, qd, qdd)).max())
        tmax = max(tmax, np.abs(tc).max())
    masses = "point masses (no inertia tensors in Corke's UR5)" if rtb_name == "UR5" else "full inertia tensors"
    return err, 1e-9, (
        f"Corke's own recursive Newton-Euler robot.rne(q, q̇, q̈) on the original {'modified' if r.mdh else 'standard'}"
        f"-DH model rtb.models.DH.{rtb_name} ({masses}; friction removed with .nofriction(), motor inertias "
        f"Jm = 0, gravity (0, 0, −9.81)) versus MR Newton-Euler and Pinocchio RNEA on the PoE model produced by "
        f"robots.from_rtb_dh (CoM frames + spatial inertias G_i converted from the DH link frames). "
        f"{N} random states; largest torque {tmax:.1f} N·m. Proves the inertial conversion, not only the geometry.")


def mass_matrix_mr_vs_crba(robot):
    rng, N = _rng("mm" + robot.name), 4
    e_mr, e_pin = _eng(robot, "mr"), _eng(robot, "pinocchio")
    err = 0.0
    for _ in range(N):
        th = robot.random_config(rng)
        err = max(err, np.abs(e_mr.mass_matrix(th) - e_pin.mass_matrix(th)).max())
    return err, 1e-9, (
        f"Mass matrix M(θ): mr.MassMatrix (n calls of inverse dynamics with θ̇ = 0, g = 0, θ̈ = e_i, "
        f"MR Sec. 8.5) versus Pinocchio's Composite Rigid Body Algorithm (pin.crba). {N} random "
        f"configurations; error in kg·m².")


def mass_matrix_vs_link_jacobians(robot):
    import pinocchio  # noqa: F401  (only to fail loudly if missing)
    rng, N = _rng("mm-j" + robot.name), 4
    e_pin = _eng(robot, "pinocchio")
    err = 0.0
    for _ in range(N):
        th = robot.random_config(rng)
        Ts = robot.link_transforms(th)
        Js = kin.jacobian_space(robot, th)
        M = np.zeros((robot.n, robot.n))
        for i in range(robot.n):
            Tc = Ts[i + 1] @ robot.com_home[i]                 # CoM frame of link i in {s}
            Ji = np.zeros((6, robot.n))
            Ji[:, :i + 1] = mr.Adjoint(mr.TransInv(Tc)) @ Js[:, :i + 1]   # body Jacobian of link i
            M += Ji.T @ robot.Glist[i] @ Ji
        err = max(err, np.abs(M - e_pin.mass_matrix(th)).max())
    return err, 1e-9, (
        f"Closed form M(θ) = Σ_i J_iᵀ(θ)·G_i·J_i(θ) (MR Eq. 8.57, kinetic energy ½ Σ V_iᵀ G_i V_i of every link), "
        f"where J_i = [Ad_(T_ci⁻¹)]·[J_s1 … J_si 0 … 0] maps θ̇ to the twist of link i's centre-of-mass frame. "
        f"Built here from the PoE data alone and compared with Pinocchio's CRBA at {N} configurations.")


def forward_mr_vs_aba(robot):
    rng, N = _rng("fd" + robot.name), 4
    e_mr, e_pin = _eng(robot, "mr"), _eng(robot, "pinocchio")
    err, amax = 0.0, 0.0
    for th, dth, _ in _states(robot, rng, N):
        tau = rng.uniform(-0.3, 0.3, robot.n) * robot.tau_max
        a, b = e_mr.forward(th, dth, tau), e_pin.forward(th, dth, tau)
        err, amax = max(err, np.abs(a - b).max() / max(1.0, np.abs(b).max())), max(amax, np.abs(b).max())
    return err, 1e-9, (
        f"Forward dynamics θ̈ = M⁻¹(τ − c − g): mr.ForwardDynamics (builds M with n Newton-Euler passes and "
        f"solves, MR Sec. 8.5) versus Pinocchio's O(n) Articulated-Body Algorithm (pin.aba) — a completely "
        f"different algorithm that never forms M. {N} random states with torques up to 30 % of τ_max. Error "
        f"relative to max(1, |θ̈|) (largest |θ̈| = {amax:.1f} rad/s²; the PUMA wrist makes M ill-conditioned).")


def lagrange_vs_newton_euler(robot):
    """Derive the planar arm's equations of motion symbolically (Euler-Lagrange) from its inertial data."""
    import sympy as sp
    n = robot.n
    th = sp.symbols(f"theta1:{n + 1}")
    dth = sp.symbols(f"dtheta1:{n + 1}")
    ddth = sp.symbols(f"ddtheta1:{n + 1}")
    q = [screw_axis_line(robot.Slist[:, i])[0][:2] for i in range(n)]   # joint axis points at home
    c = [robot.com_home[i][:2, 3] for i in range(n)]                      # CoM positions at home
    m = [float(G[3, 3]) for G in robot.Glist]
    Izz = [float((robot.com_home[i][:3, :3] @ robot.Glist[i][:3, :3] @ robot.com_home[i][:3, :3].T)[2, 2])
           for i in range(n)]
    gvec = robot.gravity[:2]

    def R(a):
        return sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])

    sgn = [1 if robot.Slist[2, i] > 0 else -1 for i in range(n)]   # axis along +z or -z
    phi = [sum(sgn[j] * th[j] for j in range(i + 1)) for i in range(n)]     # absolute link angles
    dphi = [sum(sgn[j] * dth[j] for j in range(i + 1)) for i in range(n)]
    p_joint = sp.Matrix(q[0])
    K = sp.Integer(0)
    P = sp.Integer(0)
    for i in range(n):
        pc = p_joint + R(phi[i]) * sp.Matrix(c[i] - q[i])
        vc = pc.jacobian(th) * sp.Matrix(dth)
        K += sp.Rational(1, 2) * m[i] * (vc.T * vc)[0] + sp.Rational(1, 2) * Izz[i] * dphi[i] ** 2
        P += -m[i] * (gvec[0] * pc[0] + gvec[1] * pc[1])
        if i + 1 < n:
            p_joint = p_joint + R(phi[i]) * sp.Matrix(q[i + 1] - q[i])
    L = K - P
    tau = []
    for i in range(n):            # tau_i = d/dt(dL/d dtheta_i) - dL/dtheta_i   (MR Eq. 8.3)
        dL = sp.diff(L, dth[i])
        ddt = sum(sp.diff(dL, th[j]) * dth[j] + sp.diff(dL, dth[j]) * ddth[j] for j in range(n))
        tau.append(ddt - sp.diff(L, th[i]))
    f = sp.lambdify((th, dth, ddth), tau, "numpy")
    Msym = sp.hessian(K, dth)

    e_mr = _eng(robot, "mr")
    rng, N = _rng("lag" + robot.name), 6
    err = 0.0
    for a, b, cc in _states(robot, rng, N):
        err = max(err, np.abs(np.asarray(f(a, b, cc), float) - e_mr.inverse(a, b, cc)).max())
    if n == 2:
        Ms = [sp.nsimplify(sp.trigsimp(sp.expand(Msym[i, j])), tolerance=1e-10, rational=False)
              for i, j in ((0, 0), (0, 1), (1, 1))]
        Ms = [sp.N(x, 4) for x in Ms]
        extra = (f" SymPy's mass matrix: M11 = {Ms[0]}, M12 = {Ms[1]}, M22 = {Ms[2]} "
                 f"(kg·m²) — compare with the point-mass 2R of MR Eq. 8.9.")
    else:
        extra = ""
    return err, 1e-9, (
        f"Euler-Lagrange equations τ_i = d/dt(∂L/∂θ̇_i) − ∂L/∂θ_i (MR Sec. 8.1) derived SYMBOLICALLY with "
        f"SymPy from scratch: link CoM positions from plane rotations of the home geometry, L = Σ ½m_i|v_ci|² "
        f"+ ½I_zz,i·φ̇_i² − Σ m_i·g·y_ci, using this robot's own masses {m}, CoM positions and I_zz from G_i. "
        f"Compared with recursive Newton-Euler (mr.InverseDynamics, Sec. 8.3) at {N} random states: two "
        f"completely different derivations of the same physics (energy vs forces).{extra}")


def mass_matrix_spd(robot):
    rng = _rng("spd" + robot.name)
    e_mr, e_pin = _eng(robot, "mr"), _eng(robot, "pinocchio")
    asym, lam_min, cond = 0.0, np.inf, 0.0
    for k in range(60):
        th = robot.random_config(rng)
        if k < 4:     # MR builds M column by column -> symmetry is a genuine test
            M = e_mr.mass_matrix(th)
            asym = max(asym, np.abs(M - M.T).max())
        lam = np.linalg.eigvalsh(e_pin.mass_matrix(th))
        lam_min, cond = min(lam_min, lam[0]), max(cond, lam[-1] / lam[0])
    err = asym if lam_min > 0 else np.inf
    return err, 1e-12, (
        f"M(θ) must be symmetric positive definite (kinetic energy ½θ̇ᵀMθ̇ > 0 for θ̇ ≠ 0, MR Sec. 8.1.2). "
        f"Symmetry: MR's M is assembled column by column from n independent Newton-Euler passes, so "
        f"M = Mᵀ is a real test (error = max |M − Mᵀ| over 4 configurations). Positive definiteness: smallest "
        f"eigenvalue over 60 configurations = {lam_min:.3e} kg·m² (> 0 ✓), worst condition number {cond:.1e}.")


def skew_symmetry(robot):
    rng, N, h = _rng("skew" + robot.name), 5, 1e-6
    e_pin = _eng(robot, "pinocchio")
    err, naive, quad = 0.0, 0.0, 0.0
    for th, dth, _ in _states(robot, rng, N):
        Mdot = (e_pin.mass_matrix(th + h * dth) - e_pin.mass_matrix(th - h * dth)) / (2 * h)
        C = e_pin.coriolis_matrix(th, dth)
        Nm = Mdot - 2 * C
        err = max(err, np.abs(Nm + Nm.T).max())
        quad = max(quad, abs(dth @ Nm @ dth))
        c = C @ dth                                    # naive factorisation with the same c = C θ̇
        Cn = np.outer(c, dth) / (dth @ dth)
        Nn = Mdot - 2 * Cn
        naive = max(naive, np.abs(Nn + Nn.T).max())
    return err, 1e-6, (
        f"Passivity property (MR Sec. 8.1.2, Prop. 8.1): with C(θ,θ̇) built from the Christoffel symbols of M "
        f"(Pinocchio computeCoriolisMatrix), Ṁ − 2C is skew-symmetric. Ṁ is obtained independently by central "
        f"differences of M along θ̇: Ṁ ≈ [M(θ+hθ̇) − M(θ−hθ̇)]/(2h), h = {h:g}. Error = max |N + Nᵀ| for "
        f"N = Ṁ − 2C over {N} random states (finite-difference noise ~1e-10). Contrast: the naive factorisation "
        f"C' = c·θ̇ᵀ/‖θ̇‖², which also satisfies C'θ̇ = c, gives {naive:.2f} — skew-symmetry holds only for the "
        f"Christoffel C. The weaker energy identity θ̇ᵀ(Ṁ − 2C)θ̇ = 0 holds: {quad:.1e}.")


def coriolis_factorisation(robot):
    rng, N = _rng("cor" + robot.name), 5
    e_mr, e_pin = _eng(robot, "mr"), _eng(robot, "pinocchio")
    err = 0.0
    for th, dth, _ in _states(robot, rng, N):
        err = max(err, np.abs(e_pin.coriolis_matrix(th, dth) @ dth - e_mr.coriolis(th, dth)).max())
    return err, 1e-9, (
        f"Velocity-product terms: c(θ,θ̇) from the textbook code (mr.VelQuadraticForces = Newton-Euler with "
        f"θ̈ = 0 and g = 0) versus C(θ,θ̇)·θ̇ with Pinocchio's Coriolis matrix built from Christoffel symbols "
        f"(MR Eqs. 8.17, 8.19). {N} random states; error in N·m.")


def _free_motion(robot):
    rng = _rng("energy" + robot.name)
    return robot.random_config(rng), rng.uniform(-0.5, 0.5, robot.n)


def energy_conservation(robot):
    th0, dth0 = _free_motion(robot)
    eng = _eng(robot, "pinocchio")
    s = simulate(eng, th0, dth0, lambda t, q, dq: np.zeros(robot.n), 1.0, dt=0.01, method="rk45")
    E = s.ke + s.pe
    drift = np.abs(E - E[0]).max() / max(s.ke.max(), 1e-12)
    return drift, 1e-6, (
        f"Free motion (τ = 0, no friction, gravity on) from a random state, 1 s, integrated with "
        f"dynamics.simulate(method='rk45') (SciPy DOP853, rtol = atol = 1e-10) using Pinocchio's ABA. "
        f"Total energy E = ½θ̇ᵀM(θ)θ̇ + Σ m_i·g·h_i(θ) — potential energy from the PoE centre-of-mass positions, "
        f"i.e. independent of the dynamics code. Relative drift = max|E(t) − E(0)| / max KE = {drift:.1e} "
        f"(KE up to {s.ke.max():.2f} J). A wrong Coriolis or gravity term would show up immediately as energy "
        f"being created or destroyed.")


def euler_drift(robot):
    th0, dth0 = _free_motion(robot)
    eng = _eng(robot, "pinocchio")
    out = {}
    for m in ("euler", "semi-implicit", "rk4"):
        s = simulate(eng, th0, dth0, lambda t, q, dq: np.zeros(robot.n), 1.0, dt=1e-3, method=m)
        E = s.ke + s.pe
        out[m] = np.abs(E - E[0]).max() / max(s.ke.max(), 1e-12)
    return out["euler"], 0.5, (
        f"INFORMATIONAL (loose tolerance 50 %): same free motion integrated with explicit Euler, dt = 1 ms "
        f"(mr.EulerStep's scheme). Relative energy drift: explicit Euler {out['euler']:.1e}, symplectic "
        f"(semi-implicit) Euler {out['semi-implicit']:.1e}, classic RK4 {out['rk4']:.1e}, adaptive RK45 "
        f"~1e-10 (previous row). Explicit Euler is only first-order accurate and does not preserve energy: "
        f"for an oscillator every step moves along the tangent of the energy level curve, so the state "
        f"spirals outward and energy is gained (error ∝ dt). This is a numerical artefact of the "
        f"integrator, not an error in the dynamics model.")


def power_balance(robot):
    from scipy.integrate import cumulative_simpson
    rng = _rng("power" + robot.name)
    th0, dth0 = robot.random_config(rng), rng.uniform(-0.5, 0.5, robot.n)
    eng = _eng(robot, "pinocchio")
    amp = 4.0 * np.diag(eng.mass_matrix(th0))       # ~4 rad/s^2 of motor-driven acceleration per joint
    w, ph = rng.uniform(2, 6, robot.n), rng.uniform(0, 2 * np.pi, robot.n)
    d = np.diag(eng.mass_matrix(th0))
    b = np.minimum(0.8 * d / d.max(), d / 0.02)    # friction decay time M_ii/b_i >= 20 ms (40 grid steps)
    s = simulate(eng, th0, dth0, lambda t, q, dq: amp * np.sin(w * t + ph), 0.5, dt=5e-4,
                 method="rk45", damping=b)
    P = np.einsum("ij,ij->i", s.tau - b * s.dtheta, s.dtheta)        # motor power - friction losses
    W = cumulative_simpson(P, x=s.t, initial=0.0)
    E = s.ke + s.pe
    dE = E - E[0]
    err = np.abs(dE - W).max() / max(np.abs(dE).max(), 1e-12)
    dEdt = np.gradient(E, s.t, edge_order=2)
    err_rate = np.abs(dEdt - P)[2:-2].max() / max(np.abs(P).max(), 1e-12)
    return err, 1e-6, (
        f"Power balance (work-energy theorem): dE/dt = τᵀθ̇ − b‖θ̇‖² along a 0.5 s trajectory driven by "
        f"sinusoidal motor torques (amplitude 4 rad/s² × M_ii) plus gravity, with viscous joint friction "
        f"b_i = {b.max():.2f}·M_ii/max M_ii N·m·s/rad, integrated "
        f"with RK45. Integral form E(t) − E(0) = ∫₀ᵗ (τ − bθ̇)ᵀθ̇ dt (Simpson rule on a 0.5 ms grid, quadrature error ∝ h⁴); "
        f"error relative to max|ΔE| = {np.abs(dE).max():.2f} J. Pointwise dE/dt vs power (finite "
        f"differences): {err_rate:.1e}. Confirms the dynamics exchange energy exactly as physics demands.")


# ============================================================================ task list
def _task_list():
    """[(label, category, check name, MR section, fn)] in display order."""
    T = []
    names = list(ROBOTS)          # built-in robots + any robot added with the Robot Builder
    dyn = [nm for nm in names if get_robot(nm).has_dynamics]
    for nm in names:
        T.append((KIN, "FK: space-form PoE vs body-form PoE", "MR 4.1.1–4.1.3", nm, fk_space_vs_body))
    for nm in names:
        T.append((KIN, "FK: Modern Robotics vs Pinocchio", "MR 4.1", nm, fk_vs_pinocchio))
    for dh in DH_MODELS:
        T.append((KIN, "FK: Corke DH model vs automatic DH→PoE", "MR 4.1, App. C", ("dh", dh), dh_fk_vs_poe))
    for dh in DH_MODELS:
        T.append((KIN, "Jacobian: Corke DH jacob0 vs MR", "MR 5.1, App. C", ("dh", dh), dh_jac_vs_poe))
    for nm in names:
        T.append((KIN, "J_s vs finite differences of FK", "MR 5.1.1", nm, js_vs_finite_difference))
    for nm in names:
        T.append((KIN, "J_b = [Ad_Tbs]·J_s", "MR 5.1.4", nm, jb_vs_adjoint))
    for nm in names:
        T.append((KIN, "Jacobians: Modern Robotics vs Pinocchio", "MR 5.1", nm, jac_vs_pinocchio))
    for nm in names:
        T.append((IKC, "IK: Newton–Raphson vs mr.IKinBody", "MR 6.2", nm, ik_vs_mr))
    for nm in names:
        if analytic_ik_applies(get_robot(nm)):
            T.append((IKC, "IK: analytic (law of cosines) vs numerical", "MR 6.1", nm, ik_analytic_vs_numeric))
    for nm in names:
        T.append((IKC, "IK: round trip FK(IK(T)) = T", "MR 6.2", nm, ik_round_trip))
    for nm in dyn:
        T.append((STAT, "Statics: τ = J_sᵀF vs RNEA at rest", "MR 5.2, 8.3", nm, statics_vs_rnea))
    for nm in dyn:
        T.append((DYN, "RNEA: Modern Robotics vs Pinocchio", "MR 8.3", nm, rnea_mr_vs_pin))
    for dh in ("Puma560", "Panda", "UR5"):
        T.append((DYN, "RNEA: Corke rne (DH) vs MR & Pinocchio (PoE)", "MR 8.3, App. C", ("dh", dh), rnea_vs_corke))
    for nm in dyn:
        T.append((DYN, "Mass matrix: MR vs Pinocchio CRBA", "MR 8.4–8.5", nm, mass_matrix_mr_vs_crba))
    for nm in dyn:
        T.append((DYN, "Mass matrix: Σ JᵢᵀGᵢJᵢ vs CRBA", "MR 8.4", nm, mass_matrix_vs_link_jacobians))
    for nm in dyn:
        T.append((DYN, "Forward dynamics: MR vs Pinocchio ABA", "MR 8.5", nm, forward_mr_vs_aba))
    for nm in dyn:
        if planar_z_axes(get_robot(nm)):
            T.append((DYN, "Newton–Euler = symbolic Euler–Lagrange", "MR 8.1 vs 8.3", nm, lagrange_vs_newton_euler))
    for nm in dyn:
        T.append((DYN, "C(θ,θ̇)·θ̇ = c(θ,θ̇)", "MR 8.1.2", nm, coriolis_factorisation))
    for nm in dyn:
        T.append((PHYS, "M(θ) symmetric positive definite", "MR 8.1.2", nm, mass_matrix_spd))
    for nm in dyn:
        T.append((PHYS, "Ṁ − 2C skew-symmetric (passivity)", "MR 8.1.2", nm, skew_symmetry))
    for nm in dyn:
        T.append((PHYS, "Energy conserved in free motion (RK45)", "MR 8.1, 8.5", nm, energy_conservation))
    for nm in [x for x in ("Planar 2R arm", "UR5 (6R industrial)") if x in dyn]:
        T.append((PHYS, "ⓘ Explicit-Euler energy drift (for contrast)", "MR 8.5", nm, euler_drift))
    for nm in dyn:
        T.append((PHYS, "Power balance τᵀθ̇ − b‖θ̇‖² = dE/dt", "MR 8.1, 11.4", nm, power_balance))
    return T


def _run_one(cat, title, chapter, target, fn) -> Check:
    if isinstance(target, tuple):              # Corke DH model
        arg, label = target[1], _dh_label(target[1])
    else:
        arg, label = get_robot(target), short(target)
    name = f"{title} — {label}"
    t0 = time.perf_counter()
    try:
        err, tol, detail = fn(arg)
        detail += f"  [{(time.perf_counter() - t0) * 1000:.0f} ms]"
        return Check(name, chapter, float(err), float(tol), detail, cat)
    except Exception:
        return Check(name, chapter, float("inf"), 0.0, "The check raised an exception:\n" + traceback.format_exc(), cat)


def _optional_checks(modname: str, category: str) -> list[Check]:
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            c = Check(f"{modname}: module not available — checks skipped", "—", float("nan"), float("nan"),
                      f"{modname} is not part of this build (ModuleNotFoundError).", category)
            c.skipped = True
            return [c]
        return [Check(f"{modname}: import failed", "—", float("inf"), 0.0, traceback.format_exc(), category)]
    except Exception:
        return [Check(f"{modname}: import failed", "—", float("inf"), 0.0, traceback.format_exc(), category)]
    fn = getattr(mod, "validation_checks", None)
    if fn is None:
        c = Check(f"{modname}: no validation_checks() — skipped", "—", float("nan"), float("nan"),
                  f"{modname} does not expose validation_checks().", category)
        c.skipped = True
        return [c]
    t0 = time.perf_counter()
    try:
        res = list(fn())
    except Exception:
        return [Check(f"{modname}.validation_checks() raised", "—", float("inf"), 0.0, traceback.format_exc(), category)]
    dt = time.perf_counter() - t0
    out = []
    for c in res:
        if not isinstance(c, Check):
            out.append(Check(f"{modname}: invalid check object {c!r:.60}", "—", float("inf"), 0.0, "", category))
            continue
        if not c.category:
            c.category = category
        try:
            c.error, c.tol = float(c.error), float(c.tol)
        except Exception:
            c.error, c.tol = float("inf"), 0.0
        if np.isnan(c.error):
            c.error = float("inf")
        out.append(c)
    if out:
        out[-1].detail = (out[-1].detail or "") + f"  [{modname}: {len(out)} checks in {dt:.2f} s]"
    return out


def _target_label(target) -> str:
    return _dh_label(target[1]) if isinstance(target, tuple) else short(target)


def planned_checks() -> list[tuple[str, str, str]]:
    """(category, name, MR section) of the built-in checks, in run order (cheap; nothing is computed)."""
    return [(cat, f"{title} — {_target_label(target)}", chapter)
            for cat, title, chapter, target, _ in _task_list()]


def run_all(progress=None, on_result=None) -> list[Check]:
    """Run the complete suite.

    ``progress(fraction, message)`` is called before every step; ``on_result(check)`` (optional) is
    called as soon as each check is finished, in the order of the returned list (used to fill the
    Validation tab's table live)."""
    tasks = _task_list()
    total = len(tasks) + len(OPTIONAL_MODULES)
    out: list[Check] = []

    def emit(cs):
        for c in cs:
            out.append(c)
            if on_result is not None:
                on_result(c)

    for k, (cat, title, chapter, target, fn) in enumerate(tasks):
        if progress is not None:
            progress(k / total, f"{title} — {_target_label(target)}")
        emit([_run_one(cat, title, chapter, target, fn)])
    for j, (modname, cat) in enumerate(OPTIONAL_MODULES):
        if progress is not None:
            progress((len(tasks) + j) / total, f"{cat} ({modname}.validation_checks)")
        emit(_optional_checks(modname, cat))
    if progress is not None:
        progress(1.0, "done")
    return out


# ============================================================================ summary + report
def is_skipped(c: Check) -> bool:
    return bool(getattr(c, "skipped", False))


def summary(checks: list[Check]) -> dict:
    run = [c for c in checks if not is_skipped(c)]
    passed = [c for c in run if c.passed]
    # "agreement" checks: two implementations compared at machine precision (tolerance <= 1e-9)
    precise = [c.error for c in passed if c.tol <= 1e-9 and np.isfinite(c.error)]
    return dict(total=len(run), passed=len(passed), failed=len(run) - len(passed),
                skipped=len(checks) - len(run), max_precise_error=max(precise) if precise else 0.0,
                all_passed=len(passed) == len(run))


def library_versions() -> dict:
    out = {"Python": platform.python_version(), "NumPy": np.__version__}
    for mod, label in (("modern_robotics", "modern_robotics"), ("pinocchio", "Pinocchio"),
                       ("roboticstoolbox", "Robotics Toolbox (Corke)"), ("sympy", "SymPy"), ("scipy", "SciPy")):
        try:
            m = importlib.import_module(mod)
            out[label] = getattr(m, "__version__", "installed")
        except Exception:
            out[label] = "not installed"
    return out


def fmt_err(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if np.isinf(x):
        return "∞"
    return "0" if x == 0 else f"{x:.1e}"


def report_html(checks: list[Check], elapsed: float | None = None) -> str:
    """Stand-alone, printable HTML report of a validation run."""
    s = summary(checks)
    esc = html.escape
    ok = s["all_passed"]
    banner = (f"{s['passed']} / {s['total']} checks passed" + (f" · {s['skipped']} skipped" if s["skipped"] else "")
              + f" — max error {fmt_err(s['max_precise_error'])} among machine-precision comparisons (tolerance ≤ 1e-9)")
    rows = []
    for i, c in enumerate(checks):
        if is_skipped(c):
            status, cls = "skipped", "skip"
        else:
            status, cls = ("✓ pass", "pass") if c.passed else ("✗ FAIL", "fail")
        rows.append(f"<tr class='{cls}'><td>{i + 1}</td><td>{esc(c.category)}</td><td><b>{esc(c.name)}</b>"
                    f"<div class='det'>{esc(c.detail)}</div></td><td>{esc(c.chapter)}</td>"
                    f"<td class='num'>{fmt_err(c.error)}</td><td class='num'>{fmt_err(c.tol)}</td>"
                    f"<td class='st'>{status}</td></tr>")
    vers = "".join(f"<li>{esc(k)}: {esc(str(v))}</li>" for k, v in library_versions().items())
    fam = "".join(f"<li><b>{esc(k)}</b> — {esc(v)}</li>" for k, v in FAMILY_TEXT.items())
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    took = f" in {elapsed:.1f} s" if elapsed is not None else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MR Studio validation report</title>
<style>
 body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; margin: 32px auto; max-width: 1200px;
        color: #1d2329; padding: 0 16px; }}
 h1 {{ margin-bottom: 4px; }} .sub {{ color: #65707c; margin-top: 0; }}
 .banner {{ padding: 14px 18px; border-radius: 8px; font-size: 18px; font-weight: 600; margin: 18px 0;
           background: {'#e3f7ea' if ok else '#fde8e8'}; color: {'#11663a' if ok else '#9b1c1c'};
           border: 1px solid {'#9fd8b4' if ok else '#f3b0b0'}; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
 th {{ background: #f1f3f5; text-align: left; padding: 6px 8px; position: sticky; top: 0; }}
 td {{ border-top: 1px solid #e3e6e9; padding: 6px 8px; vertical-align: top; }}
 td.num {{ font-family: Menlo, Consolas, monospace; white-space: nowrap; }}
 td.st {{ white-space: nowrap; font-weight: 600; }}
 tr.pass td.st {{ color: #11883f; }} tr.fail td {{ background: #fff3f3; }} tr.fail td.st {{ color: #c62828; }}
 tr.skip td {{ color: #8a939c; }}
 .det {{ color: #56606a; font-size: 12px; margin-top: 3px; white-space: pre-wrap; }}
 .cols {{ display: flex; gap: 32px; flex-wrap: wrap; }} .cols > div {{ flex: 1; min-width: 300px; }}
 @media print {{ th {{ position: static; }} }}
</style></head><body>
<h1>MR Studio — cross-validation report</h1>
<p class="sub">Kinematics &amp; dynamics of robot chains following <i>Modern Robotics</i> (Lynch &amp; Park, 2017).
Generated {when}{took}; fixed random seed {SEED} (reproducible).</p>
<div class="banner">{'✓' if ok else '✗'} {esc(banner)}</div>
<div class="cols"><div><h3>Why these checks prove correctness</h3><ul>{fam}</ul>
<p>Independent implementations share no code and use different conventions (PoE vs DH, (ω,v) vs (v,ω),
Newton–Euler vs Lagrange). Agreement to ~1e-15 is therefore only possible if each of them is right.
Physical invariants (energy, passivity, power balance) hold for any correct model, whatever the code.</p></div>
<div><h3>Software</h3><ul>{vers}</ul></div></div>
<h3>All checks</h3>
<table><thead><tr><th>#</th><th>Category</th><th>Check (what was compared and how)</th><th>MR section</th>
<th>max error</th><th>tolerance</th><th>result</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>
"""


if __name__ == "__main__":  # pragma: no cover
    t0 = time.perf_counter()
    res = run_all(lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    for c in res:
        st = "SKIP" if is_skipped(c) else ("PASS" if c.passed else "FAIL")
        print(f"{st}  {fmt_err(c.error):>8} / {fmt_err(c.tol):<6} {c.category:<20} {c.name}")
    print(summary(res), f"{time.perf_counter() - t0:.1f} s")
