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
    radii: np.ndarray     # 3 half-lengths (force ellipsoid: np.inf along singular task directions)
    mu1: float            # isotropy  sqrt(lmax/lmin)  (MR Sec. 5.4)
    mu2: float            # condition number lmax/lmin (MR Sec. 5.4)
    mu3: float            # volume ~ sqrt(l1 l2 ...)   (MR Sec. 5.4)
    #: bool (3,): principal direction belongs to the task space of the arm (False = padding axis of a planar
    #: arm, e.g. z for the planar linear ellipse; such an axis is drawn with zero length)
    task: np.ndarray = field(default_factory=lambda: np.ones(3, bool))

    @property
    def unbounded(self) -> np.ndarray:
        """bool (3,): infinite semi-axes (force ellipsoid at a singularity: J^T has a null direction, so a
        finite joint torque resists an arbitrarily large tip force along it)."""
        return ~np.isfinite(self.radii)


def manipulability(robot: SerialRobot, theta, kind: str = "linear", force: bool = False) -> Ellipsoid:
    """Manipulability (or force) ellipsoid at the end-effector, MR Sec. 5.4.

    kind='linear'  uses A = J_v J_v^T  (J_v = R_sb J_bv: point velocity in {s} axes)
    kind='angular' uses A = J_w J_w^T  (J_w = R_sb J_bw)
    The manipulability ellipsoid is {v : v^T A^{-1} v = 1} (semi-axes sqrt(lambda_i(A)));
    force=True     returns the force ellipsoid {f : f^T A f = 1} = B = A^{-1} (same axes, reciprocal radii).
    At a singularity lambda_i(A) = 0 and the force semi-axis along that task direction is infinite (np.inf),
    not zero; only the padding axes of planar arms (outside the task space) get radius 0.
    """
    T = fk_space(robot, theta)
    Jb = jacobian_body(robot, theta)
    R = T[:3, :3]
    J = R @ (Jb[3:] if kind == "linear" else Jb[:3])
    axes = np.eye(3)
    in_task = np.ones(3, bool)
    if robot.planar and kind == "linear":      # ellipse in the x-y plane
        lam, axes[:2, :2] = np.linalg.eigh(J[:2] @ J[:2].T)
        lam3, task = np.r_[lam, 0.0], lam
        in_task[2] = False
    elif robot.planar:                          # only w_z exists
        lam3 = np.array([0.0, 0.0, float(J[2] @ J[2])])
        task = lam3[2:]
        in_task[:2] = False
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
        safe = np.maximum(radii, 1e-6)
        radii = np.where(~in_task, 0.0, np.where(radii > 1e-6, 1.0 / safe, np.inf))
    return Ellipsoid(T[:3, 3].copy(), axes, radii, float(mu1), float(mu2), mu3, in_task)


def singular_values(robot: SerialRobot, theta) -> np.ndarray:
    """Singular values of the task Jacobian.  For a spatial arm this is the full 6xn J_b, whose rows mix
    rad/s (omega) and m/s (v), so sigma_min depends on the length unit (MR Sec. 5.4) -- prefer
    :func:`split_singular_values` for a unit-consistent test."""
    J, _ = task_jacobian(robot, theta)
    return np.linalg.svd(J, compute_uv=False)


def split_singular_values(robot: SerialRobot, theta) -> tuple[np.ndarray, np.ndarray]:
    """(sigma(J_w), sigma(J_v)): singular values of the angular block J_w = R_sb J_bw [rad/s per rad/s] and of
    the linear block J_v = R_sb J_bv [m/s per rad/s], each in its own unit (MR Sec. 5.4 recommends looking at
    the two blocks separately).  Planar arms: J_w = the omega_z row (3R+), J_v = the (v_x, v_y) rows."""
    T = fk_space(robot, theta)
    Jb = jacobian_body(robot, theta)
    Jw, Jv = T[:3, :3] @ Jb[:3], T[:3, :3] @ Jb[3:]
    if robot.planar:
        Jw, Jv = Jw[2:3], Jv[:2]
        if robot.n < 3:
            Jw = np.zeros((0, robot.n))
    sw = np.linalg.svd(Jw, compute_uv=False) if Jw.size else np.zeros(0)
    sv = np.linalg.svd(Jv, compute_uv=False)
    return sw, sv


def scaled_singular_values(robot: SerialRobot, theta, length: float | None = None) -> np.ndarray:
    """Singular values of the dimensionless task Jacobian: the linear-velocity rows are divided by a
    characteristic length L (default: the reach), so every row is in rad/s per rad/s and sigma_min no longer
    depends on the choice of metres vs millimetres (MR Sec. 5.4)."""
    J, labels = task_jacobian(robot, theta)
    L = float(robot.reach if length is None else length) or 1.0
    scale = np.array([1.0 / L if lab.startswith("v") else 1.0 for lab in labels])
    return np.linalg.svd(scale[:, None] * J, compute_uv=False)


def is_singular(robot: SerialRobot, theta, tol: float = 1e-3) -> bool:
    """Rank of the task Jacobian below min(n, task dimension), tested on the dimensionless Jacobian
    (v rows / reach) -- a threshold on sigma_min of the raw 6xn J would mix rad/s and m/s (MR Sec. 5.4)."""
    s = scaled_singular_values(robot, theta)
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
    err_w: list = field(default_factory=list)        # ||w_b|| (frame="body") or ||w_s|| (frame="space") history
    err_v: list = field(default_factory=list)        # ||v_b|| (frame="body") or ||v_s|| (frame="space") history
    method: str = ""
    frame: str = "body"                              # frame of the twist error used in the stopping test


def ik_newton_raphson(robot: SerialRobot, T_target: np.ndarray, theta0, *, eomg=1e-6, ev=1e-6,
                      max_iter=60, frame="body", damping: float = 0.0, position_only=False,
                      max_step: float | None = None) -> IKResult:
    """Newton-Raphson numerical IK, MR Sec. 6.2.2 / Eq. 6.6 (the algorithm of mr.IKinBody/IKinSpace).

    frame="body":  V_b = log(T_sb^-1 T_sd), step J_b^+ V_b, stop when ||w_b|| <= eomg and ||v_b|| <= ev.
    frame="space": V_s = [Ad_Tsb] V_b, step J_s^+ V_s, stop when ||w_s|| <= eomg and ||v_s|| <= ev -- exactly
    the test of mr.IKinSpace (||v_s|| differs from ||v_b||, so the two frames can stop at different iterates).
    damping > 0 turns the pseudo-inverse step into damped least squares
    dtheta = J^T (J J^T + lambda^2 I)^-1 V (robust near singularities).
    position_only=True solves only for the end-effector point (3-D task, used for 2R arms).
    """
    th = np.array(theta0, float)
    space = frame == "space" and not position_only
    res = IKResult(th.copy(), False, 0, [th.copy()], method=("DLS" if damping > 0 else "Newton-Raphson")
                   + (" (position)" if position_only else f" ({frame} frame)"),
                   frame="space" if space else "body")

    def error(th):
        if position_only:
            T = mr.FKinBody(robot.M, robot.Blist, th)
            e = T_target[:3, 3] - T[:3, 3]
            return e, np.zeros(3), e, point_jacobian(robot, th)
        if space:   # same operations as mr.IKinSpace
            T = mr.FKinSpace(robot.M, robot.Slist, th)
            V = mr.Adjoint(T) @ mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_target))
            J = mr.JacobianSpace(robot.Slist, th)
        else:       # same operations as mr.IKinBody
            T = mr.FKinBody(robot.M, robot.Blist, th)
            V = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_target))
            J = mr.JacobianBody(robot.Blist, th)
        if robot.planar:  # keep only the planar task components (w_z, v_x, v_y)
            idx = [2, 3, 4] if robot.n >= 3 else [3, 4]
            return V[idx], V[:3], V[3:], J[idx]
        return V, V[:3], V[3:], J

    for k in range(max_iter + 1):
        V, ew, evv, J = error(th)
        res.err_w.append(float(np.linalg.norm(ew)))
        res.err_v.append(float(np.linalg.norm(evv)))
        if res.err_w[-1] <= eomg and res.err_v[-1] <= ev:
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
    """Wrap every entry to [-pi, pi) (use :func:`wrap_joints` for robots with prismatic joints)."""
    return (np.asarray(th) + np.pi) % (2 * np.pi) - np.pi


def wrap_joints(robot: SerialRobot, th, into_limits: bool = True) -> np.ndarray:
    """Same configuration with the *revolute* joints wrapped to [-pi, pi) (prismatic joints untouched).

    With ``into_limits`` each wrapped revolute angle that falls outside its joint limits is then shifted by
    +-2 pi when that brings it inside (e.g. Panda joint 6, limits [-1 deg, 215 deg]: 200 deg must not be
    reported as -160 deg)."""
    th = np.array(th, float)
    rev = ~np.asarray(robot.is_prismatic, bool)
    th[rev] = wrap_angles(th[rev])
    if into_limits:
        lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
        for i in np.flatnonzero(rev):
            if lo[i] - 1e-9 <= th[i] <= hi[i] + 1e-9:
                continue
            for k in (1, -1, 2, -2):
                t = th[i] + 2 * np.pi * k
                if lo[i] - 1e-9 <= t <= hi[i] + 1e-9:
                    th[i] = t
                    break
    return th


def joint_distance(robot: SerialRobot, a, b) -> float:
    """||a - b|| with revolute differences wrapped to [-pi, pi) (the same pose up to full turns is 0 apart)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rev = ~np.asarray(robot.is_prismatic, bool)
    d[rev] = wrap_angles(d[rev])
    return float(np.linalg.norm(d))


def within_limits(robot: SerialRobot, th, tol: float = 1e-9) -> bool:
    th = np.asarray(th, float)
    return bool(np.all(th >= robot.joint_limits[:, 0] - tol) and np.all(th <= robot.joint_limits[:, 1] + tol))


def ik_all_solutions(robot: SerialRobot, T_target, n_starts: int = 60, seed: int = 0,
                     position_only=False, tol=1e-3, patience: int | None = None, max_starts: int = 600,
                     cancelled=None, progress=None) -> list[np.ndarray]:
    """IK branches found by Newton-Raphson from many random seeds (MR Sec. 6.1.1).

    A numerical search, so the list may be incomplete: at least ``n_starts`` starts are made, then sampling
    continues until ``patience`` (default ``n_starts``) consecutive starts found no new branch, up to
    ``max_starts``.  UR/PUMA-type arms have up to 8 branches; a general 6R arm can have 16.
    Revolute joints are wrapped (and shifted by 2 pi into the joint limits when possible); solutions outside
    the limits are dropped.  ``cancelled()`` is polled between starts (returns the branches found so far);
    ``progress(fraction)`` reports the fraction of ``max_starts`` used."""
    rng = np.random.default_rng(seed)
    patience = n_starts if patience is None else patience
    sols: list[np.ndarray] = []
    since_new = 0
    for k in range(max(max_starts, n_starts)):
        if cancelled is not None and cancelled():
            break
        if k >= n_starts and since_new >= patience:
            break
        since_new += 1
        r = ik_newton_raphson(robot, T_target, robot.random_config(rng), max_iter=80,
                              position_only=position_only, max_step=1.0)
        if progress is not None and k % 10 == 0:
            progress(k / max(max_starts, n_starts))
        if not r.success:
            continue
        th = wrap_joints(robot, r.theta)
        if not within_limits(robot, th):
            continue
        if all(joint_distance(robot, th, s) > tol for s in sols):
            sols.append(th)
            since_new = 0
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
    s2 = np.sqrt(max(1.0 - c2 * c2, 0.0))
    for s in (+1, -1):
        t2 = np.arctan2(s * s2, c2)        # atan2 instead of arccos: accurate near c2 = +-1 (MR Sec. 6.1)
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
