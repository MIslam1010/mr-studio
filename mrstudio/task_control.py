"""Task-space motion control, force control, hybrid motion-force control and impedance control
following Modern Robotics Chapter 11 (Secs. 11.3.3, 11.4.3, 11.5, 11.6, 11.7).

Conventions (MR): twists V = (w, v), wrenches F = (m, f); X = T_sb in SE(3).  Two task representations:

* ``"body"``   -- body twist V_b, [V_b] = X^-1 Xdot; configuration error [X_e] = log(X^-1 X_d)     (MR Eq. 11.16)
                  wrenches F_b = (m_b, f_b) in the end-effector frame {b}; tau = J_b^T F_b.
* ``"hybrid"`` -- (w_b, pdot): body angular velocity + linear velocity of the origin of {b} in {s};
                  X_e = (log(R^T R_d), p_d - p)                                                  (MR Eq. 11.18)
                  conjugate wrench (m_b, f_s): moment in {b}, force in {s} at the end-effector point.

Planar arms only control the planar task components: (w_z, v_x, v_y) for 3R, (v_x, v_y) for 2R (the 2R arm
always uses the "hybrid" representation, whose position rows are decoupled from the orientation error).

Implemented controllers
  1. velocity_control     theta_dot = J^+ ( [Ad_{X^-1 X_d}] V_d + K_p X_e + K_i int X_e )               (11.16)
  2. task_motion_control  tau = J^T ( Lam~ (d/dt([Ad]V_d) + K_p X_e + K_i int X_e + K_d V_e) + eta~ )      (11.47)
  3. force_control        tau = g~ + J^T ( F_d + K_fp F_e + K_fi int F_e - K_damp V )                      (11.54)
  4. hybrid_control       tau = J^T ( P Lam~ (...) + (I - P)(F_d + K_fp F_e + K_fi int F_e) + eta~ )       (11.61)
                          P = I - A^T (A Lam^-1 A^T)^-1 A Lam^-1                                            (11.60)
  5. impedance_control    M xdd + B xd + K x = f_ext rendered at the end-effector                           (11.64)
                          * no F/T sensor:  tau = J^T ( eta~ - (B xd + K x) )           [11.65 with M~ = 0]
                          * F/T sensor:     xdd_d = M^-1 (f_ext - B xd - K x)  (11.66),
                                            tau = J^T ( Lam~ xdd_d + eta~ - f_ext )

Task-space dynamics (MR Sec. 8.6): Lam = J^-T M J^-1 (8.90), eta = J^-T h - Lam Jdot J^-1 V (8.91).  For redundant
arms (Panda) and near singularities the generalised, damped form Lam = (J M^-1 J^T + mu^2 I)^-1 is used with the
dynamically consistent inverse Jbar = M^-1 J^T Lam, eta = Jbar^T h - Lam Jdot thetadot; the self-motion is damped
in the null space, tau += (I - J^T Jbar^T)(-k_null thetadot).  For a square, non-singular J this is exactly 8.90/8.91.

The plant is Pinocchio (ABA) integrated with RK4 under a zero-order-hold controller; the environment (stiff
spring-damper wall, a user pushing) acts through the tip wrench of MR Eq. 8.77.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np

from .checks import Check
from .dynamics import DynamicsEngine, to_pinocchio
from .robots import SerialRobot

try:
    import pinocchio as pin
except Exception:  # pragma: no cover
    pin = None

CAT = "Task-space & force control"
MODES = ("velocity", "motion", "force", "hybrid", "impedance")
MODE_LABELS = {
    "velocity": "Velocity-input task-space control (11.3.3)",
    "motion": "Task-space computed torque (11.4.3)",
    "force": "Force control against a wall (11.5)",
    "hybrid": "Hybrid motion–force: wiping (11.6)",
    "impedance": "Impedance control (11.7)",
}
SHAPES = ("circle", "figure-eight", "line")


# ============================================================================ kinematics (Pinocchio, MR order)
class Kin:
    """Fast forward kinematics / body Jacobian / Jdot*thetadot of a SerialRobot (Pinocchio built from the PoE data;
    the validation checks compare it with mr.FKinBody / mr.JacobianBody)."""

    def __init__(self, robot: SerialRobot):
        self.robot = robot
        self.model = to_pinocchio(robot)
        self.data = self.model.createData()
        self.ee = self.model.getFrameId("ee")

    def fk(self, q) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, float))
        return self.data.oMf[self.ee].homogeneous.copy()

    def jacobian(self, q) -> np.ndarray:
        J = pin.computeFrameJacobian(self.model, self.data, np.asarray(q, float), self.ee, pin.LOCAL)
        return np.vstack([J[3:], J[:3]])          # Pinocchio (v, w) -> MR (w, v)

    def state(self, q, dq):
        """X, J_b, V_b = J_b dq, Jdot_b dq (the body acceleration at ddq = 0)."""
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        J = self.jacobian(q)
        pin.forwardKinematics(self.model, self.data, q, dq, np.zeros_like(dq))
        pin.updateFramePlacements(self.model, self.data)
        X = self.data.oMf[self.ee].homogeneous.copy()
        a = pin.getFrameAcceleration(self.model, self.data, self.ee, pin.LOCAL)
        return X, J, J @ dq, np.r_[a.angular, a.linear]


# ============================================================================ task representation
def task_rows(robot: SerialRobot) -> list[int]:
    """Task components the arm can actually control.  Planar: (w_z, v_x, v_y) or (v_x, v_y).  Spatial arms with
    fewer than 6 joints (e.g. a SCARA): drop the twist rows that are identically zero (no joint can produce them,
    e.g. w_x, w_y); otherwise all six."""
    if robot.planar:
        return [2, 3, 4] if robot.n >= 3 else [3, 4]
    if robot.n < 6:
        rng = np.random.default_rng(0)
        Js = [mr.JacobianBody(robot.Blist, robot.random_config(rng)) for _ in range(4)]
        rows = [i for i in range(6) if max(np.abs(J[i]).max() for J in Js) > 1e-9]
        if len(rows) <= robot.n:
            return rows
    return list(range(6))


class TaskRep:
    """Body-twist (Eq. 11.16) or (w_b, pdot) (Eq. 11.18) representation, restricted to the controllable rows."""

    def __init__(self, robot: SerialRobot, kind: str = "body"):
        if robot.planar and robot.n < 3:
            kind = "hybrid"              # 2R: orientation is not controllable, keep position decoupled
        self.kind = kind
        self.rows = task_rows(robot)
        self.m = len(self.rows)

    def full_jac(self, X, Jb):
        if self.kind == "body":
            return Jb
        return np.vstack([Jb[:3], X[:3, :3] @ Jb[3:]])

    def jac(self, X, Jb):
        return self.full_jac(X, Jb)[self.rows]

    def twist6(self, X, Vb):
        return Vb if self.kind == "body" else np.r_[Vb[:3], X[:3, :3] @ Vb[3:]]

    def jdot6(self, X, Vb, ab):
        """Jdot thetadot of this representation from the body quantities."""
        if self.kind == "body":
            return ab
        R = X[:3, :3]
        return np.r_[ab[:3], R @ (ab[3:] + np.cross(Vb[:3], Vb[3:]))]

    def error6(self, X, Xd):
        if self.kind == "body":
            return mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
        R, Rd = X[:3, :3], Xd[:3, :3]
        return np.r_[mr.so3ToVec(mr.MatrixLog3(R.T @ Rd)), Xd[:3, 3] - X[:3, 3]]

    def ff6(self, X, Xd, Vd):
        """Feedforward twist in the actual frame: [Ad_{X^-1 X_d}] V_d (body) or (R^T R_d w_d, pdot_d) (hybrid)."""
        if self.kind == "body":
            return mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd
        return np.r_[X[:3, :3].T @ Xd[:3, :3] @ Vd[:3], Vd[3:]]

    def dff6(self, X, Xd, Vd, dVd, Vb):
        """d/dt of the feedforward twist (the feedforward acceleration of Eq. 11.47).
        body:   d/dt(Ad_T V_d) = Ad_T (dV_d + ad_{V_T} V_d),  T = X^-1 X_d,  V_T = V_d - Ad_{T^-1} V_b
        hybrid: d/dt(R^T R_d w_d) = R^T R_d dw_d - w_b x (R^T R_d w_d)."""
        if self.kind == "body":
            T = mr.TransInv(X) @ Xd
            VT = Vd - mr.Adjoint(mr.TransInv(T)) @ Vb
            return mr.Adjoint(T) @ (dVd + mr.ad(VT) @ Vd)
        Rrel = X[:3, :3].T @ Xd[:3, :3]
        w = Rrel @ Vd[:3]
        return np.r_[Rrel @ dVd[:3] - np.cross(Vb[:3], w), dVd[3:]]

    def wrench6(self, X, f_s, m_b=None):
        """Wrench (in this representation) of a force f_s ({s} coordinates) applied at the end-effector point."""
        m = np.zeros(3) if m_b is None else np.asarray(m_b, float)
        if self.kind == "body":
            return np.r_[m, X[:3, :3].T @ f_s]
        return np.r_[m, f_s]

    def constraint(self, X, normal):
        """Pfaffian constraint A V = 0 (Eq. 11.55): no end-effector velocity along the wall normal."""
        n = np.asarray(normal, float)
        A6 = np.r_[np.zeros(3), X[:3, :3].T @ n] if self.kind == "body" else np.r_[np.zeros(3), n]
        return A6[self.rows][None, :]

    def sel(self, v6):
        return np.asarray(v6)[self.rows]


def dls_pinv(J, eps: float = 0.04, lam_max: float = 0.04):
    """Damped pseudo-inverse with variable damping (Nakamura / Maciejewski): lam = 0 when the smallest singular
    value exceeds eps, growing to lam_max at the singularity.  Returns (J^+, sigma_min, lam)."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    smin = float(s[-1]) if len(s) else 0.0
    lam2 = 0.0 if smin >= eps else (1.0 - (smin / eps) ** 2) * lam_max ** 2
    return Vt.T @ np.diag(s / (s ** 2 + lam2)) @ U.T, smin, float(np.sqrt(lam2))


def task_dynamics(model: DynamicsEngine, q, dq, J, Jd_dq, eps: float = 0.04, lam_max: float = 0.04):
    """Lam~, eta~, Jbar, N^T (= I - J^T Jbar^T), h~ of the controller's model (Sec. 8.6, generalised/damped)."""
    M = model.mass_matrix(q)
    h = model.inverse(q, dq, np.zeros_like(q))            # c + g
    Minv = np.linalg.inv(M)
    Li = J @ Minv @ J.T
    s = np.linalg.svd(J, compute_uv=False)
    smin = float(s[-1])
    mu2 = 0.0 if smin >= eps else (1.0 - (smin / eps) ** 2) * lam_max ** 2 * float(np.trace(Li)) / len(Li)
    Lam = np.linalg.inv(Li + mu2 * np.eye(len(Li)))
    Jbar = Minv @ J.T @ Lam
    eta = Jbar.T @ h - Lam @ Jd_dq
    NT = np.eye(len(q)) - J.T @ Jbar.T
    return Lam, eta, NT, h, smin


def projection(A, Lam):
    """P = I - A^T (A Lam^-1 A^T)^-1 A Lam^-1      (MR Eq. 11.60)."""
    Li = np.linalg.inv(Lam)
    return np.eye(len(Lam)) - A.T @ np.linalg.solve(A @ Li @ A.T, A @ Li)


# ============================================================================ desired end-effector motion
@dataclass
class TaskTrajectory:
    t: np.ndarray
    Xd: np.ndarray          # N x 4 x 4
    Vd_b: np.ndarray        # body twist of X_d
    dVd_b: np.ndarray
    Vd_h: np.ndarray        # (w_d, pdot_d)
    dVd_h: np.ndarray

    def V(self, kind):
        return (self.Vd_b, self.dVd_b) if kind == "body" else (self.Vd_h, self.dVd_h)


def quintic(t, T):
    s = np.clip(t / T, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / T * (t <= T), \
        (60 * s - 180 * s**2 + 120 * s**3) / T**2 * (t <= T)


def shape_plane(robot: SerialRobot, X0, surface_normal=None):
    """Two orthonormal in-plane directions (u1, u2) for the desired path."""
    if robot.planar:                               # in the x-y plane, first moving back towards the base
        if surface_normal is not None:             # wiping the floor: only the line along -x is on the surface
            return np.array([-1.0, 0, 0]), np.zeros(3)
        return np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])
    p = X0[:3, 3]
    radial = np.array([p[0], p[1], 0.0])
    radial = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > 1e-6 else np.array([1.0, 0, 0])
    tang = np.cross([0, 0, 1.0], radial)
    if surface_normal is not None:                 # wiping a horizontal table
        return tang, -radial                       # (the circle bulges towards the base)
    return tang, np.array([0, 0, 1.0])             # vertical plane facing the robot


def make_task_trajectory(robot: SerialRobot, X0, shape="circle", size=0.1, T=4.0, dt=0.002, hold=0.6,
                         wrist=0.0, plane=None, cycles: int = 1) -> TaskTrajectory:
    """X_d(t): the origin follows `shape` (starting and ending at X0, quintic phase, zero end velocities) in the
    plane (u1, u2); the orientation optionally rotates about the end-effector z axis by wrist*sin(phi)."""
    u1, u2 = plane if plane is not None else shape_plane(robot, X0)
    t = np.arange(0.0, T + hold + 1e-9, dt)
    s, sd, sdd = quintic(t, T)
    k = 2 * np.pi * cycles
    phi, dphi, ddphi = k * s, k * sd, k * sdd
    if shape == "circle":
        f = np.c_[np.sin(phi), 1 - np.cos(phi)]           # centre at X0 + size*u2
        df = np.c_[np.cos(phi), np.sin(phi)]
        ddf = np.c_[-np.sin(phi), np.cos(phi)]
    elif shape == "figure-eight":
        f = np.c_[np.sin(phi), 0.5 * np.sin(2 * phi)]
        df = np.c_[np.cos(phi), np.cos(2 * phi)]
        ddf = np.c_[-np.sin(phi), -2 * np.sin(2 * phi)]
    else:                                              # straight line out and back
        f = np.c_[(1 - np.cos(phi)) / 2, 0 * phi]
        df = np.c_[np.sin(phi) / 2, 0 * phi]
        ddf = np.c_[np.cos(phi) / 2, 0 * phi]
    U = np.c_[u1, u2]                                   # 3 x 2
    p = X0[:3, 3] + size * f @ U.T
    pd = size * (df * dphi[:, None]) @ U.T
    pdd = size * (ddf * dphi[:, None] ** 2 + df * ddphi[:, None]) @ U.T
    psi, dpsi, ddpsi = wrist * np.sin(phi), wrist * np.cos(phi) * dphi, \
        wrist * (-np.sin(phi) * dphi ** 2 + np.cos(phi) * ddphi)
    N = len(t)
    Xd = np.zeros((N, 4, 4))
    Vb, dVb, Vh, dVh = (np.zeros((N, 6)) for _ in range(4))
    z = np.array([0, 0, 1.0])
    for i in range(N):
        Rz = mr.MatrixExp3(mr.VecToso3(z * psi[i]))
        R = X0[:3, :3] @ Rz
        Xd[i] = mr.RpToTrans(R, p[i])
        w, dw = z * dpsi[i], z * ddpsi[i]
        v = R.T @ pd[i]
        Vb[i] = np.r_[w, v]
        dVb[i] = np.r_[dw, R.T @ pdd[i] - np.cross(w, v)]
        Vh[i] = np.r_[w, pd[i]]
        dVh[i] = np.r_[dw, pdd[i]]
    return TaskTrajectory(t, Xd, Vb, dVb, Vh, dVh)


def hold_trajectory(X0, T, dt) -> TaskTrajectory:
    t = np.arange(0.0, T + 1e-9, dt)
    z = np.zeros((len(t), 6))
    return TaskTrajectory(t, np.repeat(X0[None], len(t), 0), z, z.copy(), z.copy(), z.copy())


# ============================================================================ environment
@dataclass
class Wall:
    """Stiff spring-damper surface through `point` with outward unit normal `normal` (pointing to the robot).
    Unilateral: pushes only, f = max(0, k d - b n.pdot) along n, d = penetration depth."""
    point: np.ndarray
    normal: np.ndarray
    k: float = 5000.0
    b: float = 50.0

    def force_on_robot(self, p, pdot) -> np.ndarray:
        n = self.normal
        d = -float(n @ (p - self.point))
        if d <= 0:
            return np.zeros(3)
        return n * max(0.0, self.k * d - self.b * float(n @ pdot))


def push_profile(t, t_on, t_off, ramp=0.05):
    """Smooth 0 -> 1 -> 0 pulse (user pushing the end-effector)."""
    a = np.clip((t - t_on) / ramp, 0, 1)
    b = np.clip((t_off - t) / ramp, 0, 1)
    a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    return a * b


# ============================================================================ results
@dataclass
class TaskResult:
    mode: str
    label: str
    t: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    cmd: np.ndarray            # commanded joint velocities (velocity mode) or torques
    Xe: np.ndarray             # N x 6 configuration error (rows not controlled are zero)
    p: np.ndarray              # actual end-effector position
    pd: np.ndarray             # desired position (or rest position)
    Xd: np.ndarray | None = None   # desired frames (N x 4 x 4) for the animation
    force: np.ndarray | None = None      # robot's force on the environment along -normal [N]
    force_d: np.ndarray | None = None
    f_user: np.ndarray | None = None     # disturbance force (impedance), {s}
    x_ref: np.ndarray | None = None      # ideal mass-spring-damper response (impedance) along the push
    x_act: np.ndarray | None = None
    smin: np.ndarray | None = None       # smallest singular value of J along the motion
    info: dict = field(default_factory=dict)
    diverged: bool = False
    cancelled: bool = False

    @property
    def err_w(self):
        return np.linalg.norm(self.Xe[:, :3], axis=1)

    @property
    def err_v(self):
        return np.linalg.norm(self.Xe[:, 3:], axis=1)

    @property
    def pos_err(self):
        return np.linalg.norm(self.pd - self.p, axis=1)


def _store_xe(rep: TaskRep, xe6):
    out = np.zeros(6)
    out[rep.rows] = np.asarray(xe6)[rep.rows]
    return out


# ============================================================================ 1. velocity-input control (11.3.3)
def velocity_control(robot: SerialRobot, traj: TaskTrajectory, theta0, kp=5.0, ki=1.0, feedforward=True,
                     rep="body", kin: Kin | None = None, label="", qdot_max=None, cancel=None) -> TaskResult:
    """Kinematic (velocity-controlled) robot: theta_{k+1} = theta_k + dt J^+ V(t_k)   with
    V = [Ad_{X^-1 X_d}] V_d + K_p X_e + K_i int X_e    (MR Eq. 11.16; Eq. 11.18 for rep='hybrid')."""
    kin = kin or Kin(robot)
    tr = TaskRep(robot, rep)
    t = traj.t
    N, n = len(t), robot.n
    Vd_all, _ = traj.V(tr.kind)
    TH, CMD, XE, P = np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, 6)), np.zeros((N, 3))
    SM = np.zeros(N)
    th = np.array(theta0, float)
    eint = np.zeros(6)
    for k in range(N):
        X = kin.fk(th)
        Jb = kin.jacobian(th)
        xe = tr.error6(X, traj.Xd[k])
        V = kp * xe + ki * eint
        if feedforward:
            V = V + tr.ff6(X, traj.Xd[k], Vd_all[k])
        Jp, SM[k], _ = dls_pinv(tr.jac(X, Jb))
        dth = Jp @ tr.sel(V)
        if qdot_max is not None:
            sc = np.max(np.abs(dth) / qdot_max)
            if sc > 1:
                dth = dth / sc
        TH[k], CMD[k], XE[k], P[k] = th, dth, _store_xe(tr, xe), X[:3, 3]
        if k < N - 1:
            h = t[k + 1] - t[k]
            th = th + h * dth
            eint = eint + h * xe
        if cancel is not None and k % 200 == 0 and cancel():
            break
    res = TaskResult("velocity", label, t, TH, CMD, CMD, XE, P, traj.Xd[:, :3, 3], Xd=traj.Xd, smin=SM)
    res.info = dict(rep=tr.kind)
    return res


# ============================================================================ plant integration
def _rk4(plant: DynamicsEngine, th, dth, tau, h, ftip_fn):
    def f(q, dq):
        return plant.forward(q, dq, tau, ftip_fn(q, dq))
    k1v, k1a = dth, f(th, dth)
    k2v = dth + 0.5 * h * k1a
    k2a = f(th + 0.5 * h * k1v, k2v)
    k3v = dth + 0.5 * h * k2a
    k3a = f(th + 0.5 * h * k2v, k3v)
    k4v = dth + h * k3a
    k4a = f(th + h * k3v, k4v)
    return th + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v), dth + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)


class _Env:
    """Tip wrench (MR convention: wrench the end-effector applies to the environment, in {b}) from the wall and the
    user's push; uses its own Pinocchio data so it can be called inside the RK4 stages."""

    def __init__(self, robot, wall: Wall | None = None, f_user_fn=None):
        self.kin = Kin(robot)
        self.wall, self.f_user_fn = wall, f_user_fn
        self.t = 0.0

    def forces(self, q, dq):
        X = self.kin.fk(q)
        if self.wall is None and self.f_user_fn is None:
            return X, np.zeros(3), np.zeros(3)
        Vb = self.kin.jacobian(q) @ dq
        pdot = X[:3, :3] @ Vb[3:]
        fw = self.wall.force_on_robot(X[:3, 3], pdot) if self.wall is not None else np.zeros(3)
        fu = self.f_user_fn(self.t) if self.f_user_fn is not None else np.zeros(3)
        return X, fw, fu

    def ftip(self, q, dq):
        if self.wall is None and self.f_user_fn is None:
            return None
        X, fw, fu = self.forces(q, dq)
        f_on_robot = fw + fu
        return np.r_[np.zeros(3), -X[:3, :3].T @ f_on_robot]


def _null_damping_kin(J, dq, k_null):
    """(I - J^T J^+T)(-k_null dq) for kinematically redundant arms (zero for square J)."""
    if J.shape[0] >= J.shape[1]:
        return np.zeros_like(dq)
    Jp = np.linalg.pinv(J)
    return (np.eye(len(dq)) - J.T @ Jp.T) @ (-k_null * dq)


# ============================================================================ 2-5. torque-controlled modes
@dataclass
class Gains:
    kp: float = 100.0
    ki: float = 0.0
    kd: float = 20.0
    fd: float = 15.0          # desired normal force [N]
    kfp: float = 1.0
    kfi: float = 20.0
    kdamp_v: float = 40.0     # N s/m      (11.54)
    kdamp_w: float = 2.0      # N m s/rad
    k_null: float = 2.0       # N m s/rad  null-space damping (redundant arms)
    imp_m: float = 2.0        # impedance: virtual mass  [kg]
    imp_b: float = 20.0       #            damping       [N s/m]
    imp_k: float = 200.0      #            stiffness     [N/m]
    rot_scale: float = 0.05   # rotational impedance = translational x rot_scale (kg m^2, N m s/rad, N m/rad)
    feedforward: bool = True
    f_sensor: bool = True     # impedance: use the wrist F/T sensor to render the virtual mass
    force_damping: bool = False  # hybrid: add -K_damp V in the force subspace (as in Eq. 11.54)


def torque_control(mode: str, robot: SerialRobot, theta0, T: float, g: Gains, traj: TaskTrajectory | None = None,
                   rep="body", wall: Wall | None = None, push=None, plant: DynamicsEngine | None = None,
                   model: DynamicsEngine | None = None, dt=0.001, substeps=2, label="", progress=None,
                   cancel=None, dtheta0=None) -> TaskResult:
    """Closed-loop simulation of the torque-controlled modes 'motion' (11.47), 'force' (11.54), 'hybrid' (11.61)
    and 'impedance' (11.65/11.66).  push = (f_vec_s, t_on, t_off) for the impedance disturbance."""
    plant = plant or DynamicsEngine(robot, "pinocchio")
    model = model or plant
    kin = Kin(robot)
    tr = TaskRep(robot, "hybrid" if mode == "impedance" else rep)
    m, n = tr.m, robot.n
    f_user_fn = None
    if push is not None:
        fvec, t_on, t_off = push
        fvec = np.asarray(fvec, float)
        f_user_fn = (lambda tt: fvec * push_profile(tt, t_on, t_off))
    env = _Env(robot, wall, f_user_fn)
    ftip_fn = (lambda q, dq: env.ftip(q, dq))
    t = np.arange(0.0, T + 1e-9, dt) if traj is None else traj.t
    N = len(t)
    dt = float(t[1] - t[0])
    th = np.array(theta0, float)
    dth = np.zeros(n) if dtheta0 is None else np.array(dtheta0, float)
    X0 = kin.fk(th)
    if traj is None:
        traj = hold_trajectory(X0, t[-1], dt)
    Vd_all, dVd_all = traj.V(tr.kind)
    TH, DTH, TAU, XE, P = np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, 6)), np.zeros((N, 3))
    FN, FU, SM = np.zeros(N), np.zeros((N, 3)), np.zeros(N)
    eint, fint = np.zeros(6), np.zeros(m)
    gains_w = np.r_[np.full(3, g.kdamp_w), np.full(3, g.kdamp_v)]
    sc = g.rot_scale
    Mv = np.r_[np.full(3, g.imp_m * sc), np.full(3, g.imp_m)]
    Bv = np.r_[np.full(3, g.imp_b * sc), np.full(3, g.imp_b)]
    Kv = np.r_[np.full(3, g.imp_k * sc), np.full(3, g.imp_k)]
    normal = wall.normal if wall is not None else None
    diverged = cancelled = False
    k_end = N
    for k in range(N):
        env.t = t[k]
        X, Jb, Vb, ab = kin.state(th, dth)
        J = tr.jac(X, Jb)
        V = tr.sel(tr.twist6(X, Vb))
        Jd = tr.sel(tr.jdot6(X, Vb, ab))
        _, fw, fu = env.forces(th, dth)
        # "sensor": wrench the robot applies to the wall, in this representation
        F_tip = tr.sel(tr.wrench6(X, -fw))
        F_user = tr.sel(tr.wrench6(X, fu))               # wrench the user applies ON the robot
        TH[k], DTH[k], P[k], FU[k] = th, dth, X[:3, 3], fu
        FN[k] = float(np.linalg.norm(fw))
        if mode in ("motion", "hybrid", "impedance"):
            Lam, eta, NT, h, SM[k] = task_dynamics(model, th, dth, J, Jd)
        if mode in ("motion", "hybrid"):
            Xd = traj.Xd[k]
            xe = tr.error6(X, Xd)
            ff = tr.ff6(X, Xd, Vd_all[k]) if g.feedforward else np.zeros(6)
            dff = tr.dff6(X, Xd, Vd_all[k], dVd_all[k], Vb) if g.feedforward else np.zeros(6)
            Ve = tr.sel(ff) - V
            a = tr.sel(dff) + g.kp * tr.sel(xe) + g.ki * tr.sel(eint) + g.kd * Ve
            XE[k] = _store_xe(tr, xe)
            if mode == "motion":                                            # Eq. 11.47
                tau = J.T @ (Lam @ a + eta) + NT @ (h - g.k_null * dth)
            else:                                                           # Eq. 11.61
                A = tr.constraint(X, normal)
                # the requested acceleration must satisfy A a = 0 (MR 11.6.2, text after Eq. 11.57): with a
                # compliant wall the normal part of X_e (the penetration) would otherwise leak into the motion
                a = a - np.linalg.pinv(A) @ (A @ a)
                Pm = projection(A, Lam)
                Fd = tr.sel(tr.wrench6(X, -g.fd * normal))
                Fe = Fd - F_tip
                Ff = Fd + g.kfp * Fe + g.kfi * fint
                if g.force_damping:
                    Ff = Ff - tr.sel(gains_w) * V
                Im = np.eye(m) - Pm
                tau = J.T @ (Pm @ (Lam @ a) + Im @ Ff + eta) + NT @ (h - g.k_null * dth)
                fint = fint + dt * (Im @ Fe)
            eint = eint + dt * xe
        elif mode == "force":                                               # Eq. 11.54
            Fd = tr.sel(tr.wrench6(X, -g.fd * normal))
            Fe = Fd - F_tip
            gt = model.gravity_torque(th)
            tau = gt + J.T @ (Fd + g.kfp * Fe + g.kfi * fint - tr.sel(gains_w) * V) \
                + _null_damping_kin(J, dth, g.k_null)
            if FN[k] > 0:                         # integrate only in contact (anti-windup during the approach)
                fint = fint + dt * Fe
            SM[k] = np.linalg.svd(J, compute_uv=False)[-1]
            XE[k] = _store_xe(tr, tr.error6(X, X0))
        elif mode == "impedance":                                           # Eqs. 11.64-11.66
            xe = tr.error6(X, X0)                  # -x: restoring direction (rotation: log(R^T R_0))
            spring = tr.sel(Kv) * tr.sel(xe) - tr.sel(Bv) * V
            if g.f_sensor:
                xdd_d = (F_user + spring) / tr.sel(Mv)
                tau = J.T @ (Lam @ xdd_d + eta - F_user) + NT @ (h - g.k_null * dth)
            else:
                tau = J.T @ (eta + spring) + NT @ (h - g.k_null * dth)
            XE[k] = _store_xe(tr, xe)
        else:
            raise ValueError(mode)
        TAU[k] = tau
        if k == N - 1:
            break
        hstep = (t[k + 1] - t[k]) / substeps
        for _ in range(substeps):
            th, dth = _rk4(plant, th, dth, tau, hstep, ftip_fn)
            env.t += hstep
        if not (np.all(np.isfinite(th)) and np.all(np.isfinite(dth))) or np.abs(dth).max() > 200:
            diverged = True
            k_end = k + 1
            break
        if progress is not None and k % 100 == 0:
            progress(k / N)
        if cancel is not None and k % 100 == 0 and cancel():
            cancelled = True
            k_end = k + 1
            break
    sl = slice(0, k_end)
    res = TaskResult(mode, label, t[sl], TH[sl], DTH[sl], TAU[sl], XE[sl], P[sl], traj.Xd[sl, :3, 3],
                     Xd=traj.Xd[sl], force=FN[sl], f_user=FU[sl], smin=SM[sl], diverged=diverged,
                     cancelled=cancelled)
    res.info = dict(rep=tr.kind, X0=X0, normal=normal, wall=wall)
    if mode in ("force", "hybrid"):
        res.force_d = np.full(k_end, g.fd)
    if mode == "impedance" and push is not None:
        res.x_ref, res.x_act = msd_reference(res, push, g)
    return res


def msd_reference(res: TaskResult, push, g: Gains):
    """Ideal response of M xdd + B xd + K x = f(t) (Eq. 11.64) along the push direction (exact discretisation by
    fine RK4) and the robot's actual displacement along it."""
    fvec, t_on, t_off = push
    fmag = float(np.linalg.norm(fvec))
    u = np.asarray(fvec, float) / max(fmag, 1e-12)
    t = res.t
    x, v = 0.0, 0.0
    xs = np.zeros(len(t))
    sub = 10
    for k in range(len(t) - 1):
        xs[k] = x
        hh = (t[k + 1] - t[k]) / sub
        for j in range(sub):
            tt = t[k] + j * hh

            def acc(tq, xq, vq):
                return (fmag * push_profile(tq, t_on, t_off) - g.imp_b * vq - g.imp_k * xq) / g.imp_m
            k1x, k1v = v, acc(tt, x, v)
            k2x, k2v = v + hh / 2 * k1v, acc(tt + hh / 2, x + hh / 2 * k1x, v + hh / 2 * k1v)
            k3x, k3v = v + hh / 2 * k2v, acc(tt + hh / 2, x + hh / 2 * k2x, v + hh / 2 * k2v)
            k4x, k4v = v + hh * k3v, acc(tt + hh, x + hh * k3x, v + hh * k3v)
            x += hh / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v += hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    xs[-1] = x
    return xs, (res.p - res.p[0]) @ u


# ============================================================================ scenario helpers (used by the GUI)
#: well-conditioned start configurations for the planar arms (their home is close to the stretched singularity
#: once the circle / figure-eight is added); other robots start at their home configuration.
_PLANAR_START = {2: [0.3, 1.5], 3: [0.0, 1.5, 0.6]}


#: planar arms reaching down onto the floor (end-effector below every joint) for the contact tasks
_PLANAR_CONTACT_START = {2: [0.3, -1.6], 3: [0.6, -1.3, -0.9]}


def start_config(robot: SerialRobot, contact: bool = False) -> np.ndarray:
    if contact and robot.planar and robot.n in _PLANAR_CONTACT_START:
        return np.array(_PLANAR_CONTACT_START[robot.n], float)
    if robot.planar and robot.n in _PLANAR_START:
        return np.array(_PLANAR_START[robot.n], float)
    return np.array(robot.home, float)


def surface_normal(robot: SerialRobot) -> np.ndarray:
    """Horizontal table / floor: normal opposite to gravity."""
    g = np.asarray(robot.gravity, float)
    return -g / np.linalg.norm(g)


def make_wall(robot: SerialRobot, X0, gap=0.0, k=5000.0, b=50.0) -> Wall:
    n = surface_normal(robot)
    return Wall(X0[:3, 3] - gap * n, n, k, b)


def default_size(robot: SerialRobot) -> float:
    return round((0.1 if robot.planar else 0.12) * robot.reach, 3)


# ============================================================================ validation
def validation_checks() -> list[Check]:
    from .robots import get_robot
    out: list[Check] = []
    rng = np.random.default_rng(11)
    ur5 = get_robot("UR5 (6R industrial)")
    r3 = get_robot("Planar 3R arm")
    panda = get_robot("Franka Panda (7R redundant)")

    # --- kinematics bridge
    kin = Kin(panda)
    th, dth = panda.home + rng.uniform(-.3, .3, 7), rng.uniform(-1, 1, 7)
    X, Jb, Vb, ab = kin.state(th, dth)
    h = 1e-6
    Jdot = (mr.JacobianBody(panda.Blist, th + h * dth) - mr.JacobianBody(panda.Blist, th - h * dth)) / (2 * h)
    err = max(np.abs(X - mr.FKinBody(panda.M, panda.Blist, th)).max(), np.abs(Jb - mr.JacobianBody(panda.Blist, th)).max())
    out.append(Check("Pinocchio frame FK / body Jacobian = mr.FKinBody / mr.JacobianBody (Panda)", "MR 4.1.3 / 5.1.2",
                     float(err), 1e-10, "LOCAL frame Jacobian reordered (v, ω) → (ω, v)", CAT))
    out.append(Check("J̇_b θ̇ (Pinocchio, θ̈ = 0) = central difference of mr.JacobianBody (Panda)", "MR 8.6 / 11.4.3",
                     float(np.abs(ab - Jdot @ dth).max()), 1e-6, "used in η(θ,V) = J̄ᵀh − ΛJ̇θ̇", CAT))

    # --- feedforward derivative d/dt([Ad_{X^-1 X_d}] V_d)
    tr = make_task_trajectory(ur5, ur5.fk(ur5.home), "figure-eight", 0.1, 2.0, 1e-3, wrist=0.5)
    errs = []
    for kind in ("body", "hybrid"):
        rep = TaskRep(ur5, kind)
        Vd, dVd = tr.V(kind)
        th0, dth0 = ur5.home + 0.05, rng.uniform(-.5, .5, 6)
        k = 700
        hh = tr.t[1] - tr.t[0]

        def ff_at(j, s):
            q = th0 + s * dth0
            return rep.ff6(ur5.fk(q), tr.Xd[j], Vd[j])
        num = (ff_at(k + 1, hh) - ff_at(k - 1, -hh)) / (2 * hh)
        q = th0
        Vb_ = mr.JacobianBody(ur5.Blist, q) @ dth0
        ana = rep.dff6(ur5.fk(q), tr.Xd[k], Vd[k], dVd[k], Vb_)
        errs.append(np.abs(num - ana).max() / max(1.0, np.abs(ana).max()))
    out.append(Check("Feedforward acceleration d/dt([Ad_{X⁻¹X_d}]V_d) analytic = finite difference", "MR 11.4.3 Eq. 11.47",
                     float(max(errs)), 1e-4, "UR5, figure-eight with wrist rotation; body twist and (ω_b, ṗ) forms; "
                     "Ad_T(V̇_d + ad_{V_T}V_d), V_T = V_d − Ad_{T⁻¹}V_b", CAT))
    # desired twist consistency: [V_d] = X_d^-1 dX_d/dt
    k = 500
    dX = (tr.Xd[k + 1] - tr.Xd[k - 1]) / (2 * (tr.t[1] - tr.t[0]))
    Vnum = mr.se3ToVec(mr.TransInv(tr.Xd[k]) @ dX)
    out.append(Check("Desired body twist: [V_d] = X_d⁻¹Ẋ_d (figure-eight trajectory)", "MR 3.3.2 / 11.3.3",
                     float(np.abs(Vnum - tr.Vd_b[k]).max()), 1e-4, "central difference of X_d(t)", CAT))

    # --- 1. velocity input: set-point convergence is exactly first order, X_e(t) = X_e(0) e^{-kp t}
    X0 = ur5.fk(ur5.home)
    th0 = ur5.home + np.array([0.05, -0.04, 0.06, 0.05, -0.05, 0.04])
    ht = hold_trajectory(X0, 1.0, 5e-4)
    res = velocity_control(ur5, ht, th0, kp=5.0, ki=0.0)
    e0 = np.linalg.norm(res.Xe[0])
    pred = e0 * np.exp(-5.0 * res.t)
    dev = float(np.abs(np.linalg.norm(res.Xe, axis=1) - pred).max() / e0)
    out.append(Check("Velocity control (11.16), set-point: ‖X_e(t)‖ = ‖X_e(0)‖e^(−k_p t)", "MR 11.3.3 Eq. 11.16",
                     dev, 2e-3, "UR5, k_p = 5/s, Δt = 0.5 ms; relative deviation (Euler integration only)", CAT))
    # trajectory tracking with and without feedforward
    trj = make_task_trajectory(ur5, X0, "circle", 0.1, 3.0, 1e-3, wrist=0.3)
    r_ff = velocity_control(ur5, trj, ur5.home, kp=5.0, ki=0.0, feedforward=True)
    r_no = velocity_control(ur5, trj, ur5.home, kp=5.0, ki=0.0, feedforward=False)
    r_pi = velocity_control(ur5, trj, ur5.home, kp=5.0, ki=5.0, feedforward=False)
    out.append(Check("Velocity control: feedforward + P tracks a circle (max position error)", "MR 11.3.3 Eq. 11.16",
                     float(r_ff.pos_err.max()), 2e-3, f"UR5, r = 0.1 m in 3 s; without feedforward "
                     f"{1e3 * r_no.pos_err.max():.1f} mm, feedback PI only {1e3 * r_pi.pos_err.max():.1f} mm", CAT))
    out.append(Check("Velocity control: feedforward reduces the tracking error ≥ 10×", "MR 11.3.1.3 / 11.3.3",
                     float(r_ff.pos_err.max() / r_no.pos_err.max()), 0.1, "ratio max error (FF+P)/(P only)", CAT))
    # redundant Panda, planar 3R with (w_b, pdot) form
    trp = make_task_trajectory(panda, panda.fk(panda.home), "figure-eight", 0.1, 3.0, 1e-3, wrist=0.3)
    rp = velocity_control(panda, trp, panda.home, kp=5.0, ki=1.0)
    out.append(Check("Velocity control on the redundant Panda (J⁺, 7 joints → 6-D twist)", "MR 11.3.3 / 6.3",
                     float(max(rp.err_v.max(), rp.err_w.max())), 2e-3, "figure-eight, max ‖X_e‖ components", CAT))

    # --- 2. task-space computed torque (11.47)
    plant = DynamicsEngine(ur5, "pinocchio")
    res = torque_control("motion", ur5, th0, 1.0, Gains(kp=100, ki=0, kd=20), traj=hold_trajectory(X0, 1.0, 5e-4),
                         plant=plant, dt=5e-4, substeps=1)
    e0 = res.Xe[0]
    tt = res.t
    pred = (1 + 10 * tt) * np.exp(-10 * tt)
    dev = float(np.abs(res.Xe - pred[:, None] * e0[None]).max() / np.abs(e0).max())
    out.append(Check("Task-space computed torque (11.47), set-point: X_e(t) = X_e(0)(1+ωt)e^(−ωt)", "MR 11.4.3 Eq. 11.47",
                     dev, 1e-2, "UR5, k_p = 100, k_d = 20 (critically damped, ω = 10 rad/s), exact model, "
                     "Δt = 0.5 ms; relative deviation", CAT))
    for rob, shape in ((ur5, "figure-eight"), (panda, "circle"), (r3, "circle")):
        th_s = start_config(rob)
        Xs = rob.fk(th_s)
        trm = make_task_trajectory(rob, Xs, shape, default_size(rob), 3.0, 1e-3, wrist=0.0 if rob.planar else 0.3)
        rr = torque_control("motion", rob, th_s, 3.0, Gains(), traj=trm)
        out.append(Check(f"Task-space computed torque tracks a {shape} ({rob.name.split(' (')[0]})", "MR 11.4.3 Eq. 11.47",
                         float(rr.pos_err.max()), 1e-3, f"max position error, size {default_size(rob):.3f} m in 3 s, "
                         f"max orientation error {np.degrees(rr.err_w.max()):.3f}°", CAT))

    # --- 3. force control (11.54) against a stiff wall
    for rob in (ur5, r3):
        th_s = start_config(rob, contact=True)
        Xs = rob.fk(th_s)
        wall = make_wall(rob, Xs, gap=0.01, k=5000, b=50)
        g = Gains(fd=15.0)
        rf = torque_control("force", rob, th_s, 3.0, g, wall=wall,
                            model=DynamicsEngine(rob, "pinocchio", Glist=[G * 1.2 for G in rob.Glist]))
        ss = np.abs(rf.force[rf.t > 2.5] - 15.0).max() / 15.0
        out.append(Check(f"Force control (11.54): steady-state force error ({rob.name.split(' (')[0]}, wrong g̃ ×1.2)",
                         "MR 11.5 Eqs. 11.51–11.54", float(ss), 1e-2,
                         "F_d = 15 N into a k = 5000 N/m wall, starting 1 cm away; the integral removes the gravity-model "
                         f"error (the end-effector slides {1e3 * np.linalg.norm((rf.p[-1] - Xs[:3, 3])[:2] if rob.planar else (rf.p[-1] - Xs[:3, 3])[:2]):.0f} mm "
                         "along the frictionless surface: free directions are not controlled — hence hybrid control, 11.6)",
                         CAT))

    # --- 4. hybrid motion-force (11.61): projection properties and wiping
    kin = Kin(ur5)
    th = ur5.home + rng.uniform(-.2, .2, 6)
    dth = rng.uniform(-.5, .5, 6)
    X, Jb, Vb, ab = kin.state(th, dth)
    rep = TaskRep(ur5, "body")
    Lam, *_ = task_dynamics(plant, th, dth, rep.jac(X, Jb), ab)
    A = rep.constraint(X, np.array([0, 0, 1.0]))
    Pm = projection(A, Lam)
    err = max(np.abs(Pm @ Pm - Pm).max(), np.abs(A @ np.linalg.inv(Lam) @ Pm).max(),
              np.abs((np.eye(6) - Pm) @ A.T - A.T).max())
    out.append(Check("Hybrid projection P (11.60): P² = P, AΛ⁻¹P = 0, (I−P)Aᵀ = Aᵀ", "MR 11.6.2 Eq. 11.60",
                     float(err), 1e-10, f"UR5 random state, rank P = {np.linalg.matrix_rank(Pm)} (= 6 − k, k = 1)", CAT))
    out.append(Check("Hybrid projection: rank P = n − k", "MR 11.6.2", float(abs(np.linalg.matrix_rank(Pm) - 5)), 0.0,
                     "one constraint (wall normal)", CAT))
    for rob in (ur5, r3, panda):
        th_s = start_config(rob, contact=True)
        Xs = rob.fk(th_s)
        nrm = surface_normal(rob)
        u1, u2 = shape_plane(rob, Xs, nrm)
        shape = "line" if rob.planar else "circle"
        trw = make_task_trajectory(rob, Xs, shape, default_size(rob), 3.0, 1e-3, plane=(u1, u2))
        wall = make_wall(rob, Xs, 0.0, k=5000, b=50)
        rh = torque_control("hybrid", rob, th_s, 3.0, Gains(fd=15.0), traj=trw, wall=wall)
        late = rh.t > 1.0
        tang = rh.pd - rh.p
        tang = np.linalg.norm(tang - np.outer(tang @ nrm, nrm), axis=1)
        fe = np.abs(rh.force[late] - 15.0).max() / 15.0
        out.append(Check(f"Hybrid wiping ({rob.name.split(' (')[0]}): normal force error after 1 s (relative)",
                         "MR 11.6.2 Eq. 11.61", float(fe), 0.03,
                         f"F_d = 15 N on a k = 5000 N/m surface while wiping a {shape} of size "
                         f"{default_size(rob):.3f} m in 3 s", CAT))
        out.append(Check(f"Hybrid wiping ({rob.name.split(' (')[0]}): tangential tracking error after contact", "MR 11.6.2 Eq. 11.61",
                         float(tang[rh.t > 0.5].max()), 5e-4,
                         f"motion subspace (P); peak {1e3 * tang.max():.2f} mm during the first 0.1 s while the contact "
                         "force builds up (a compliant wall couples through Λ⁻¹Aᵀ)", CAT))

    # --- 5. impedance: rendered mass-spring-damper
    for rob in (get_robot("Planar 2R arm"), ur5):
        g = Gains(imp_m=2.0, imp_b=20.0, imp_k=200.0)
        fdir = np.array([1.0, 0, 0]) if rob.planar else np.array([0, 0, -1.0])
        ri = torque_control("impedance", rob, start_config(rob), 2.5, g, push=(10.0 * fdir, 0.3, 1.3))
        dev = np.abs(ri.x_act - ri.x_ref).max() / np.abs(ri.x_ref).max()
        out.append(Check(f"Impedance control with F/T sensor renders M ẍ + B ẋ + K x = f_ext ({rob.name.split(' (')[0]})",
                         "MR 11.7 Eqs. 11.64–11.66", float(dev), 1e-2,
                         "M = 2 kg, B = 20 N·s/m, K = 200 N/m, 10 N push for 1 s; relative deviation from the ODE", CAT))
    rob = ur5
    g = Gains(imp_m=2.0, imp_b=20.0, imp_k=200.0, f_sensor=False)
    ri = torque_control("impedance", rob, start_config(rob), 3.0, g, push=(np.array([0, 0, -10.0]), 0.3, 1.8))
    x_ss = float(ri.x_act[(ri.t > 1.6) & (ri.t < 1.8)].mean())
    out.append(Check("Impedance (11.65, M̃ = 0, no sensor): static stiffness Δx = f/K", "MR 11.7.1 Eq. 11.65",
                     abs(x_ss - 10.0 / 200.0) / 0.05, 2e-2, f"UR5, 10 N push: Δx = {1e3 * x_ss:.2f} mm (expected 50 mm)",
                     CAT))
    # --- power sanity: tau . thetadot = F . V for the motion law (tau = J^T F + null-space part)
    Lam, eta, NT, hh, _ = task_dynamics(plant, th, dth, rep.jac(X, Jb), ab)
    F = Lam @ rng.normal(size=6) + eta
    tau = rep.jac(X, Jb).T @ F
    out.append(Check("Power balance τᵀθ̇ = Fᵀ V for τ = J_bᵀ F_b", "MR 5.2 / 8.6",
                     float(abs(tau @ dth - F @ (rep.jac(X, Jb) @ dth))), 1e-9, "UR5, random state", CAT))
    return out
