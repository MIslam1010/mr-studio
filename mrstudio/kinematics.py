"""Kinematics of open chains following Modern Robotics Chapters 4-6.

Thin, well-documented layer on top of the official `modern_robotics` library
(Lynch & Park).  Where the GUI needs more than the library returns (e.g. the
history of Newton-Raphson iterates to animate convergence) the library's own
building blocks are reused so the numbers are identical to mr.IKinBody/IKinSpace.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np

from .robots import SerialRobot


# ============================================================================ Ch. 4  Forward kinematics
def fk_space(robot: SerialRobot, theta) -> np.ndarray:
    """T(theta) = e^[S1]th1 ... e^[Sn]thn M     (MR Eq. 4.14)."""
    return mr.FKinSpace(robot.M, robot.Slist, np.asarray(theta, float))


def fk_body(robot: SerialRobot, theta) -> np.ndarray:
    """T(theta) = M e^[B1]th1 ... e^[Bn]thn     (MR Eq. 4.16)."""
    return mr.FKinBody(robot.M, robot.Blist, np.asarray(theta, float))


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (roll, pitch, yaw) for display."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def rpy_to_rot(rpy) -> np.ndarray:
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def exp_coords(T: np.ndarray):
    """Screw (exponential) coordinates S*theta of a transform (MR Sec. 3.3.3.2)."""
    V = mr.se3ToVec(mr.MatrixLog6(T))
    return V


# ============================================================================ Ch. 5  Velocity kinematics
def jacobian_space(robot: SerialRobot, theta) -> np.ndarray:
    """J_s(theta), columns J_si = Ad_{e^[S1]th1...e^[Si-1]thi-1}(S_i)  (MR Eq. 5.11)."""
    return mr.JacobianSpace(robot.Slist, np.asarray(theta, float))


def jacobian_body(robot: SerialRobot, theta) -> np.ndarray:
    """J_b(theta) = [Ad_{T_bs}] J_s(theta)      (MR Eqs. 5.18, 5.22)."""
    return mr.JacobianBody(robot.Blist, np.asarray(theta, float))


def point_jacobian(robot: SerialRobot, theta) -> np.ndarray:
    """3xn Jacobian of the end-effector *point* velocity, expressed in {s}: p_dot = R_sb J_bv theta_dot."""
    T = fk_space(robot, theta)
    return T[:3, :3] @ jacobian_body(robot, theta)[3:]


def task_jacobian(robot: SerialRobot, theta) -> tuple[np.ndarray, list[str]]:
    """Jacobian restricted to the task space that the robot can actually span.
    Planar robots: (w_z, v_x, v_y) for 3R or (v_x, v_y) for 2R; spatial: the full 6-D body Jacobian."""
    if robot.planar:
        T = fk_space(robot, theta)
        Jb = jacobian_body(robot, theta)
        Jw = (T[:3, :3] @ Jb[:3])[2:3]
        Jv = (T[:3, :3] @ Jb[3:])[:2]
        if robot.n >= 3:
            return np.vstack([Jw, Jv]), ["ω_z", "v_x", "v_y"]
        return Jv, ["v_x", "v_y"]
    return jacobian_body(robot, theta), ["ω_x", "ω_y", "ω_z", "v_x", "v_y", "v_z"]


@dataclass
class Ellipsoid:
    center: np.ndarray
    axes: np.ndarray      # 3x3, columns = principal directions (in {s})
    radii: np.ndarray     # 3 half-lengths
    mu1: float            # isotropy  sqrt(lmax/lmin)  (MR Sec. 5.4)
    mu2: float            # condition number lmax/lmin (MR Sec. 5.4)
    mu3: float            # volume ~ sqrt(l1 l2 ...)   (MR Sec. 5.4)


def manipulability(robot: SerialRobot, theta, kind: str = "linear", force: bool = False) -> Ellipsoid:
    """Manipulability (or force) ellipsoid at the end-effector, MR Sec. 5.4.

    kind='linear'  uses A = J_v J_v^T  (J_v = R_sb J_bv: point velocity in {s} axes)
    kind='angular' uses A = J_w J_w^T  (J_w = R_sb J_bw)
    force=True     returns the force ellipsoid B = A^{-1} (same axes, reciprocal radii).
    """
    T = fk_space(robot, theta)
    Jb = jacobian_body(robot, theta)
    R = T[:3, :3]
    J = R @ (Jb[3:] if kind == "linear" else Jb[:3])
    axes = np.eye(3)
    if robot.planar and kind == "linear":      # ellipse in the x-y plane
        lam, axes[:2, :2] = np.linalg.eigh(J[:2] @ J[:2].T)
        lam3, task = np.r_[lam, 0.0], lam
    elif robot.planar:                          # only w_z exists
        lam3 = np.array([0.0, 0.0, float(J[2] @ J[2])])
        task = lam3[2:]
    else:
        lam3, axes = np.linalg.eigh(J @ J.T)
        task = lam3
    lam3, task = np.clip(lam3, 0.0, None), np.clip(task, 0.0, None)
    lmax, lmin = task.max(), task.min()
    mu1 = np.sqrt(lmax / lmin) if lmin > 1e-14 else np.inf
    mu2 = lmax / lmin if lmin > 1e-14 else np.inf
    mu3 = float(np.sqrt(np.prod(task)))
    radii = np.sqrt(lam3)
    if force:
        radii = np.where(radii > 1e-6, 1.0 / np.maximum(radii, 1e-6), 0.0)
    return Ellipsoid(T[:3, 3].copy(), axes, radii, float(mu1), float(mu2), mu3)


def singular_values(robot: SerialRobot, theta) -> np.ndarray:
    J, _ = task_jacobian(robot, theta)
    return np.linalg.svd(J, compute_uv=False)


def is_singular(robot: SerialRobot, theta, tol: float = 1e-3) -> bool:
    s = singular_values(robot, theta)
    m = min(robot.n, 6 if not robot.planar else (3 if robot.n >= 3 else 2))
    return len(s) < m or s[m - 1] < tol


def joint_torques_from_wrench(robot: SerialRobot, theta, force_s, moment_s=np.zeros(3)) -> np.ndarray:
    """Statics tau = J_s^T F_s  (MR Eq. 5.26).  F is the wrench the end-effector applies to the
    environment, force applied at the end-effector point, expressed in {s}."""
    T = fk_space(robot, theta)
    p = T[:3, 3]
    F_s = np.r_[np.asarray(moment_s) + np.cross(p, force_s), force_s]
    return jacobian_space(robot, theta).T @ F_s


# ============================================================================ Ch. 6  Inverse kinematics
@dataclass
class IKResult:
    theta: np.ndarray
    success: bool
    iterations: int
    history: list = field(default_factory=list)      # theta at every iterate
    err_w: list = field(default_factory=list)        # ||w_b|| history
    err_v: list = field(default_factory=list)        # ||v_b|| history
    method: str = ""


def ik_newton_raphson(robot: SerialRobot, T_target: np.ndarray, theta0, *, eomg=1e-6, ev=1e-6,
                      max_iter=60, frame="body", damping: float = 0.0, position_only=False,
                      max_step: float | None = None) -> IKResult:
    """Newton-Raphson numerical IK, MR Sec. 6.2.2 / Eq. 6.6 (the algorithm of mr.IKinBody/IKinSpace).

    damping > 0 turns the pseudo-inverse step into damped least squares
    dtheta = J^T (J J^T + lambda^2 I)^-1 V (robust near singularities).
    position_only=True solves only for the end-effector point (3-D task, used for 2R arms).
    """
    th = np.array(theta0, float)
    res = IKResult(th.copy(), False, 0, [th.copy()], method=("DLS" if damping > 0 else "Newton-Raphson")
                   + (" (position)" if position_only else f" ({frame} frame)"))

    def error(th):
        T = mr.FKinBody(robot.M, robot.Blist, th)
        if position_only:
            e = T_target[:3, 3] - T[:3, 3]
            return e, np.zeros(3), e, point_jacobian(robot, th)
        Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_target))
        if frame == "space":
            V = mr.Adjoint(T) @ Vb
            J = mr.JacobianSpace(robot.Slist, th)
        else:
            V, J = Vb, mr.JacobianBody(robot.Blist, th)
        if robot.planar:  # keep only the planar task components (w_z, v_x, v_y)
            idx = [2, 3, 4] if robot.n >= 3 else [3, 4]
            Vp, Jp = V[idx], J[idx]
            return Vp, Vb[:3], Vb[3:], Jp
        return V, Vb[:3], Vb[3:], J

    for k in range(max_iter + 1):
        V, ew, evv, J = error(th)
        res.err_w.append(float(np.linalg.norm(ew)))
        res.err_v.append(float(np.linalg.norm(evv)))
        if res.err_w[-1] < eomg and res.err_v[-1] < ev:
            res.success = True
            break
        if k == max_iter:
            break
        if damping > 0:
            step = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), V)
        else:
            step = np.linalg.pinv(J) @ V
        if max_step is not None and np.linalg.norm(step) > max_step:
            step *= max_step / np.linalg.norm(step)
        th = th + step
        res.history.append(th.copy())
    res.theta = th
    res.iterations = len(res.history) - 1
    return res


def wrap_angles(th) -> np.ndarray:
    return (np.asarray(th) + np.pi) % (2 * np.pi) - np.pi


def ik_all_solutions(robot: SerialRobot, T_target, n_starts: int = 60, seed: int = 0,
                     position_only=False, tol=1e-3) -> list[np.ndarray]:
    """Find the distinct IK branches by running Newton-Raphson from many random seeds
    (a 6R arm such as the UR5 has up to 8 branches; MR Sec. 6.1.1)."""
    rng = np.random.default_rng(seed)
    sols: list[np.ndarray] = []
    for _ in range(n_starts):
        r = ik_newton_raphson(robot, T_target, robot.random_config(rng), max_iter=80,
                              position_only=position_only, max_step=1.0)
        if not r.success:
            continue
        th = wrap_angles(r.theta)
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        if np.any(th < lo - 1e-9) or np.any(th > hi + 1e-9):
            continue
        if all(np.linalg.norm(wrap_angles(th - s)) > tol for s in sols):
            sols.append(th)
    return sols


def ik_analytic_planar(robot: SerialRobot, x: float, y: float, phi: float | None = None) -> list[np.ndarray]:
    """Closed-form IK of the planar 2R / 3R arms (MR Sec. 6.1, law of cosines).
    Returns the elbow-down and elbow-up solutions (empty list if unreachable)."""
    L = np.diff(np.r_[0.0, [screw_axis_x(robot, i) for i in range(1, robot.n)], robot.M[0, 3]])
    if robot.n == 3:
        if phi is None:
            raise ValueError("3R analytic IK needs the orientation phi")
        x, y = x - L[2] * np.cos(phi), y - L[2] * np.sin(phi)  # wrist point
    L1, L2 = L[0], L[1]
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1 + 1e-12:
        return []
    c2 = np.clip(c2, -1, 1)
    out = []
    for s in (+1, -1):
        t2 = s * np.arccos(c2)
        t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
        th = [t1, t2] if robot.n == 2 else [t1, t2, phi - t1 - t2]
        out.append(wrap_angles(np.array(th)))
    return out


def screw_axis_x(robot: SerialRobot, i: int) -> float:
    """x-coordinate of the i-th (0-based) revolute axis of a planar arm at home (v = (0,-x,0))."""
    return float(-robot.Slist[4, i])


def null_space_step(robot: SerialRobot, theta, z, gain=1.0) -> np.ndarray:
    """Self-motion of a redundant arm: dtheta = (I - J^+ J) z  (MR Sec. 6.3)."""
    J = jacobian_body(robot, theta)
    N = np.eye(robot.n) - np.linalg.pinv(J) @ J
    return gain * N @ z


def workspace_samples(robot: SerialRobot, n: int = 4000, seed: int = 1):
    """Monte-Carlo workspace: end-effector positions and the manipulability measure mu3 at each sample."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 3))
    W = np.zeros(n)
    for k in range(n):
        th = robot.random_config(rng)
        T = fk_space(robot, th)
        P[k] = T[:3, 3]
        J = T[:3, :3] @ jacobian_body(robot, th)[3:]
        if robot.planar:
            J = J[:2]
        W[k] = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
    return P, W
