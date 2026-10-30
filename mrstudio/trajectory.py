"""Trajectory generation following Modern Robotics Chapter 9.

A trajectory = a geometric *path* theta(s), s in [0, 1]  composed with a *time scaling* s(t)  (MR Sec. 9.1):

    theta(t) = theta(s(t)),   dtheta = theta'(s) s_dot,   ddtheta = theta''(s) s_dot^2 + theta'(s) s_ddot

* Time scalings (MR 9.2.2): cubic / quintic polynomials (``mr.CubicTimeScaling`` / ``mr.QuinticTimeScaling``)
  and the trapezoidal velocity profile (MR 9.2.2.2).
* Paths between a start and a goal configuration (MR 9.2.1):
    - "joint"      straight line in joint space            (``mr.JointTrajectory``)
    - "screw"      constant-screw end-effector motion        (``mr.ScrewTrajectory``)
    - "cartesian"  straight line for p + geodesic for R      (``mr.CartesianTrajectory``)
  Task-space paths are converted to joint space with warm-started Newton-Raphson IK (MR 6.2).
* Time-optimal time scaling along a joint path (MR Sec. 9.4) with **TOPP-RA** (Pham & Pham 2018):
  the phase plane (s, s_dot), the maximum-velocity curve (MVC) and the bang-bang optimal profile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np
from scipy.interpolate import CubicSpline

from . import kinematics as kin
from .checks import Check
from .robots import SerialRobot

logging.getLogger("toppra").setLevel(logging.ERROR)

SCALINGS = ("cubic", "quintic", "trapezoidal")
PATHS = ("joint", "screw", "cartesian")
PATH_LABELS = {
    "joint": "Joint-space straight line",
    "screw": "Screw motion (constant twist)",
    "cartesian": "Cartesian straight line",
}
PATH_COLORS = {"joint": "#f29e4c", "screw": "#b392f0", "cartesian": "#57d68d"}
CAT = "Trajectory & control"


# ============================================================================ time scalings (MR 9.2.2)
def trapezoid_params(Tf: float, v_ratio: float = 1.5) -> tuple[float, float]:
    """Cruise velocity v and acceleration a of a trapezoidal profile lasting Tf (MR 9.2.2.2).

    Requires 1 < v*Tf <= 2 (v_ratio = v*Tf); then a = v^2 / (v*Tf - 1).  v_ratio=1.5 -> 1/3 accel, 1/3 cruise,
    1/3 decel."""
    v = v_ratio / Tf
    a = v * v / (v * Tf - 1.0)
    return v, a


def time_scaling(kind: str, Tf: float, t, v_ratio: float = 1.5):
    """Return (s, s_dot, s_ddot) arrays at the times t (clipped to [0, Tf])."""
    t = np.clip(np.atleast_1d(np.asarray(t, float)), 0.0, Tf)
    tau = t / Tf
    if kind == "cubic":       # s = 3 tau^2 - 2 tau^3          (mr.CubicTimeScaling)
        s = np.array([mr.CubicTimeScaling(Tf, x) for x in t])
        sd = (6 * tau - 6 * tau**2) / Tf
        sdd = (6 - 12 * tau) / Tf**2
    elif kind == "quintic":   # s = 10 tau^3 - 15 tau^4 + 6 tau^5 (mr.QuinticTimeScaling)
        s = np.array([mr.QuinticTimeScaling(Tf, x) for x in t])
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / Tf
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / Tf**2
    elif kind == "trapezoidal":
        v, a = trapezoid_params(Tf, v_ratio)
        ta = v / a
        s, sd, sdd = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        m1, m3 = t <= ta, t > Tf - ta
        m2 = ~(m1 | m3)
        s[m1], sd[m1], sdd[m1] = 0.5 * a * t[m1]**2, a * t[m1], a
        s[m2], sd[m2] = v * t[m2] - v * v / (2 * a), v
        r = Tf - t[m3]
        s[m3], sd[m3], sdd[m3] = 1.0 - 0.5 * a * r**2, a * r, -a
    else:
        raise ValueError(kind)
    return s, sd, sdd


# ============================================================================ geometric paths (MR 9.2.1)
def pose_on_path(Xs, Xe, s: float, kind: str, _cache=None) -> np.ndarray:
    """End-effector pose at path parameter s -- the same formulas as mr.ScrewTrajectory / mr.CartesianTrajectory."""
    if kind == "screw":      # X(s) = X_start exp(log(X_start^-1 X_end) s)
        L = mr.MatrixLog6(mr.TransInv(Xs) @ Xe) if _cache is None else _cache
        return Xs @ mr.MatrixExp6(L * s)
    Rs, ps = mr.TransToRp(Xs)
    Re, pe = mr.TransToRp(Xe)
    L = mr.MatrixLog3(Rs.T @ Re) if _cache is None else _cache
    return mr.RpToTrans(Rs @ mr.MatrixExp3(L * s), s * pe + (1 - s) * ps)


@dataclass
class JointPath:
    """theta(s) sampled on a uniform s grid + a C2 cubic spline through the samples."""
    kind: str
    s: np.ndarray
    theta: np.ndarray            # N x n
    ee: np.ndarray               # N x 3  end-effector positions actually reached (FK of theta)
    ee_target: np.ndarray        # N x 3  positions the task-space path asked for
    ik_failures: int = 0
    max_ik_error: float = 0.0    # worst pose error of the IK solutions (m or rad)
    spline: CubicSpline = field(init=False, repr=False)

    def __post_init__(self):
        self.spline = CubicSpline(self.s, self.theta, axis=0)

    def __call__(self, s, d: int = 0):
        return self.spline(np.clip(s, 0.0, 1.0), d)


def joint_path(robot: SerialRobot, th_start, th_goal, kind: str = "joint", N: int = 101) -> JointPath:
    """Geometric path from th_start to the pose of th_goal.

    kind="joint": theta(s) = th_start + s (th_goal - th_start)  (exact straight line in joint space)
    kind="screw"/"cartesian": the end-effector poses X(s) of mr.ScrewTrajectory / mr.CartesianTrajectory are
    converted to joint space by Newton-Raphson IK warm-started at the previous solution (MR 6.2.2)."""
    th_s, th_g = np.asarray(th_start, float), np.asarray(th_goal, float)
    s = np.linspace(0.0, 1.0, N)
    if kind == "joint":
        theta = th_s + np.outer(s, th_g - th_s)
        ee = np.array([robot.fk(th)[:3, 3] for th in theta])
        return JointPath(kind, s, theta, ee, ee.copy())
    Xs, Xe = robot.fk(th_s), robot.fk(th_g)
    cache = mr.MatrixLog6(mr.TransInv(Xs) @ Xe) if kind == "screw" else mr.MatrixLog3(Xs[:3, :3].T @ Xe[:3, :3])
    pos_only = robot.planar and robot.n < 3     # a planar 2R arm controls only the point (x, y)
    theta = np.zeros((N, robot.n))
    ee, ee_t = np.zeros((N, 3)), np.zeros((N, 3))
    th, fails, worst = th_s.copy(), 0, 0.0
    for k, sk in enumerate(s):
        X = pose_on_path(Xs, Xe, sk, kind, cache)
        r = kin.ik_newton_raphson(robot, X, th, eomg=1e-9, ev=1e-9, max_iter=40, position_only=pos_only,
                                  damping=0.0, max_step=0.5)
        if not r.success:
            fails += 1
        th = r.theta
        theta[k] = th
        T = robot.fk(th)
        ee[k], ee_t[k] = T[:3, 3], X[:3, 3]
        err = np.linalg.norm(T[:3, 3] - X[:3, 3])
        if not pos_only:
            err = max(err, np.linalg.norm(mr.so3ToVec(mr.MatrixLog3(T[:3, :3].T @ X[:3, :3]))))
        worst = max(worst, float(err))
    return JointPath(kind, s, theta, ee, ee_t, fails, worst)


# ============================================================================ trajectories
@dataclass
class Trajectory:
    t: np.ndarray
    s: np.ndarray
    sd: np.ndarray
    sdd: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    ddtheta: np.ndarray
    path: JointPath
    scaling: str
    Tf: float


def make_trajectory(robot: SerialRobot, th_start, th_goal, Tf: float, N: int = 201, path: str = "joint",
                    scaling: str = "quintic", jpath: JointPath | None = None) -> Trajectory:
    """theta_d(t), dtheta_d(t), ddtheta_d(t) for a path kind and a time scaling (MR Sec. 9.2)."""
    jp = jpath if jpath is not None else joint_path(robot, th_start, th_goal, path)
    t = np.linspace(0.0, Tf, N)
    s, sd, sdd = time_scaling(scaling, Tf, t)
    if path == "joint":   # exactly mr.JointTrajectory + analytic derivatives
        d = np.asarray(th_goal, float) - np.asarray(th_start, float)
        th = np.asarray(th_start, float) + np.outer(s, d)
        dth, ddth = np.outer(sd, d), np.outer(sdd, d)
    else:                 # chain rule on the spline path theta(s)
        q1, q2 = jp(s, 1), jp(s, 2)
        th = jp(s)
        dth = q1 * sd[:, None]
        ddth = q2 * sd[:, None]**2 + q1 * sdd[:, None]
    return Trajectory(t, s, sd, sdd, th, dth, ddth, jp, scaling, Tf)


def hold(traj: Trajectory, t_hold: float, dt: float | None = None) -> Trajectory:
    """Append a constant 'hold at the goal' segment (to watch the controller settle)."""
    if t_hold <= 0:
        return traj
    dt = dt or (traj.t[1] - traj.t[0])
    m = int(round(t_hold / dt))
    tt = traj.t[-1] + dt * np.arange(1, m + 1)
    rep = lambda A, last: np.vstack([A, np.repeat(last[None], m, 0)])
    z = np.zeros(traj.theta.shape[1])
    return Trajectory(np.r_[traj.t, tt], np.r_[traj.s, np.ones(m)], np.r_[traj.sd, np.zeros(m)],
                      np.r_[traj.sdd, np.zeros(m)], rep(traj.theta, traj.theta[-1]), rep(traj.dtheta, z),
                      rep(traj.ddtheta, z), traj.path, traj.scaling, traj.Tf)


#: KUKA LBR iiwa 7 R800 joint speeds (deg/s) -- datasheet, verify against the current KUKA spec sheet
IIWA7_QDOT_MAX_DEG_S = (98.0, 98.0, 100.0, 130.0, 140.0, 180.0, 180.0)


def default_velocity_limits(robot: SerialRobot) -> np.ndarray:
    """Joint speed limits [rad/s]: the robot's own ``qdot_max`` if it defines one (course robots do), else
    manufacturer data where known, else 2.5π rad/s."""
    qd = getattr(robot, "qdot_max", None)
    if qd is not None:
        qd = np.broadcast_to(np.asarray(qd, float), (robot.n,)).copy()
        if np.all(np.isfinite(qd)) and np.all(qd > 0):
            return qd
    name = robot.name
    if "iiwa" in name and robot.n == 7:
        return np.radians(IIWA7_QDOT_MAX_DEG_S)      # datasheet, verify
    if "Panda" in name:
        return np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    if "UR5" in name:
        return np.full(6, np.pi)
    if "PUMA" in name:
        return np.array([4.0, 4.0, 4.0, 6.0, 6.0, 6.0])
    return np.full(robot.n, 2.5 * np.pi)


# ============================================================================ time-optimal scaling (MR 9.4)
@dataclass
class ToppResult:
    ok: bool
    message: str
    duration: float = np.nan
    grid: np.ndarray | None = None          # s_i
    sd_grid: np.ndarray | None = None       # optimal s_dot(s_i)
    mvc: np.ndarray | None = None           # maximum-velocity curve  s_dot_max(s_i)
    k_upper: np.ndarray | None = None       # upper boundary of the controllable sets
    t: np.ndarray | None = None
    s: np.ndarray | None = None
    sd: np.ndarray | None = None
    sdd: np.ndarray | None = None
    theta: np.ndarray | None = None
    dtheta: np.ndarray | None = None
    ddtheta: np.ndarray | None = None
    tau: np.ndarray | None = None
    tau_norm: np.ndarray | None = None      # tau / tau_max
    vel_norm: np.ndarray | None = None      # |dtheta| / v_max
    quintic_T: float = np.nan               # shortest quintic time scaling satisfying the same limits
    quintic_t: np.ndarray | None = None
    quintic_tau_norm: np.ndarray | None = None


def _const_accel_eval(grid, sd_grid, t):
    """s(t), s_dot(t), s_ddot(t) for the piecewise-constant s_ddot solution of TOPP-RA."""
    ds = np.diff(grid)
    u = (sd_grid[1:]**2 - sd_grid[:-1]**2) / (2 * ds)
    dt = 2 * ds / np.maximum(sd_grid[1:] + sd_grid[:-1], 1e-12)
    tg = np.r_[0.0, np.cumsum(dt)]
    i = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, len(ds) - 1)
    tau = t - tg[i]
    s = grid[i] + sd_grid[i] * tau + 0.5 * u[i] * tau**2
    sd = sd_grid[i] + u[i] * tau
    return np.clip(s, 0, 1), np.maximum(sd, 0.0), u[i], tg


def limit_ratio(engine, jp: JointPath, s, sd, sdd, tau_max, vel_max):
    """Normalised torques tau/tau_max and speeds |dtheta|/v_max along path(s(t))."""
    q, q1, q2 = jp(s), jp(s, 1), jp(s, 2)
    dth = q1 * sd[:, None]
    ddth = q2 * sd[:, None]**2 + q1 * sdd[:, None]
    tau = np.array([engine.inverse(q[k], dth[k], ddth[k]) for k in range(len(s))])
    return q, dth, ddth, tau, tau / tau_max, np.abs(dth) / vel_max


def min_duration(engine, jp: JointPath, tau_max, vel_max, scaling="quintic", T0=1.0, samples=81) -> float:
    """Shortest duration Tf for which the polynomial time scaling respects |tau| <= tau_max and
    |dtheta| <= v_max (bisection; torques scale like 1/Tf^2 apart from gravity)."""
    def worst(T):
        t = np.linspace(0, T, samples)
        s, sd, sdd = time_scaling(scaling, T, t)
        *_, tn, vn = limit_ratio(engine, jp, s, sd, sdd, tau_max, vel_max)
        return max(np.abs(tn).max(), vn.max())
    hi = max(T0, 1e-2)
    for _ in range(40):
        if worst(hi) <= 1.0:
            break
        hi *= 1.6
    else:
        return np.inf
    lo = hi / 1.6 if hi > T0 else 1e-3
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if worst(mid) <= 1.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4 * hi:
            break
    return hi


def topp(engine, jp: JointPath, tau_max, vel_max, n_grid: int = 250, samples: int = 300,
         compare_quintic: bool = True, cancelled=None) -> ToppResult:
    """Time-optimal time scaling along the joint path jp subject to |tau_i| <= tau_max_i and
    |dtheta_i| <= v_max_i  (MR Sec. 9.4, solved with TOPP-RA's reachability analysis).

    The torque constraint uses tau = a(s) s_ddot + b(s) s_dot^2 + c(s) (MR Eq. 9.33), with a, b, c evaluated
    by toppra through calls to the inverse dynamics engine.inverse (RNEA).

    ``cancelled()`` (optional) is polled between the stages (reachability, feasible sets, resampling, quintic
    comparison); if it returns True the result is ``ToppResult(False, "cancelled")``."""
    def stop() -> bool:
        return cancelled is not None and bool(cancelled())

    if stop():
        return ToppResult(False, "cancelled")
    try:
        import toppra as ta
        import toppra.algorithm as algo
        import toppra.constraint as const
    except Exception as e:  # pragma: no cover
        return ToppResult(False, f"toppra not available: {e}")
    tau_max = np.asarray(tau_max, float)
    vel_max = np.asarray(vel_max, float)
    path = ta.SplineInterpolator(jp.s, jp.theta)
    pc_v = const.JointVelocityConstraint(np.c_[-vel_max, vel_max])
    pc_t = const.JointTorqueConstraint(lambda q, qd, qdd: engine.inverse(q, qd, qdd), np.c_[-tau_max, tau_max],
                                       np.zeros(len(tau_max)),
                                       discretization_scheme=const.DiscretizationType.Interpolation)
    grid = np.linspace(0.0, 1.0, n_grid + 1)
    inst = algo.TOPPRA([pc_v, pc_t], path, gridpoints=grid, solver_wrapper="seidel")
    if stop():
        return ToppResult(False, "cancelled")
    inst.compute_parameterization(0.0, 0.0)
    pd = inst.problem_data
    if pd.sd_vec is None or "Ok" not in str(pd.return_code):
        return ToppResult(False, f"TOPP-RA failed: {pd.return_code} (gravity alone may exceed the torque limits)")
    if stop():
        return ToppResult(False, "cancelled")
    X = inst.compute_feasible_sets()
    if stop():
        return ToppResult(False, "cancelled")
    grid = np.asarray(pd.gridpoints)
    sd_grid = np.asarray(pd.sd_vec)
    mvc = np.sqrt(np.clip(X[:, 1], 0, None))
    mvc = np.where(np.isfinite(mvc), mvc, np.nan)
    ku = np.sqrt(np.clip(np.asarray(pd.K)[:, 1], 0, None))
    _, _, _, tg = _const_accel_eval(grid, sd_grid, np.zeros(1))
    T = float(tg[-1])
    t = np.linspace(0.0, T, samples)
    s, sd, sdd, _ = _const_accel_eval(grid, sd_grid, t)
    q, dq, ddq, tau, tn, vn = limit_ratio(engine, jp, s, sd, sdd, tau_max, vel_max)
    res = ToppResult(True, "ok", T, grid, sd_grid, mvc, ku, t, s, sd, sdd, q, dq, ddq, tau, tn, vn)
    if compare_quintic and not stop():
        Tq = min_duration(engine, jp, tau_max, vel_max, "quintic", T0=T)
        res.quintic_T = Tq
        if np.isfinite(Tq):
            tq = np.linspace(0, Tq, samples)
            s2, sd2, sdd2 = time_scaling("quintic", Tq, tq)
            *_, tn2, _ = limit_ratio(engine, jp, s2, sd2, sdd2, tau_max, vel_max)
            res.quintic_t, res.quintic_tau_norm = tq, tn2
    return res


def topp_trajectory(res: ToppResult, jp: JointPath, dt: float) -> Trajectory:
    """Resample the time-optimal solution at a fixed period dt (e.g. the control period)."""
    D = float(res.duration)
    t = np.arange(0.0, D + 0.5 * dt, dt)
    t[-1] = min(t[-1], D)
    if t[-1] < D - 1e-12:                 # arange stopped up to dt/2 short of the end: the goal must be reached
        if D - t[-1] < 1e-6 * dt:
            t[-1] = D
        else:
            t = np.r_[t, D]
    s, sd, sdd, _ = _const_accel_eval(res.grid, res.sd_grid, t)
    q1, q2 = jp(s, 1), jp(s, 2)
    return Trajectory(t, s, sd, sdd, jp(s), q1 * sd[:, None], q2 * sd[:, None]**2 + q1 * sdd[:, None], jp,
                      "time-optimal", res.duration)


# ============================================================================ validation
def sddot_fd_error(kind: str, Tf: float = 2.0, h: float = 1e-4) -> tuple[float, str]:
    """max |s̈_analytic − (ṡ(t+h) − ṡ(t−h)) / 2h| over t in (0, T), relative to max|s̈|.

    The trapezoid's s̈ jumps at t_a and T − t_a (and the clipped ends): instants within 2h of a jump are skipped."""
    t = np.linspace(2 * h, Tf - 2 * h, 4001)
    _, _, sdd = time_scaling(kind, Tf, t)
    _, sd_p, _ = time_scaling(kind, Tf, t + h)
    _, sd_m, _ = time_scaling(kind, Tf, t - h)
    fd = (sd_p - sd_m) / (2 * h)
    keep = np.ones_like(t, bool)
    note = f"central differences of ṡ, h = {h * 1e3:.1f} ms, {len(t)} instants"
    if kind == "trapezoidal":
        v, a = trapezoid_params(Tf)
        ta = v / a
        for tj in (ta, Tf - ta):
            keep &= np.abs(t - tj) > 2 * h
        note += f"; instants within 2h of the acceleration jumps at {ta:.3f} s and {Tf - ta:.3f} s skipped"
    err = float(np.abs(fd - sdd)[keep].max() / max(np.abs(sdd).max(), 1e-12))
    return err, note + "; error relative to max|s̈|"


def validation_checks() -> list[Check]:
    from .dynamics import DynamicsEngine
    from .robots import get_robot
    out: list[Check] = []
    Tf = 2.0
    t = np.linspace(0, Tf, 2001)
    for kind in SCALINGS:
        s, sd, sdd = time_scaling(kind, Tf, t)
        bc = [s[0], s[-1] - 1, sd[0], sd[-1]] + ([sdd[0], sdd[-1]] if kind == "quintic" else [])
        out.append(Check(f"{kind.capitalize()} time scaling: boundary conditions s(0)=0, s(T)=1, ṡ=0"
                         + (", s̈=0" if kind == "quintic" else ""), "MR 9.2.2", float(np.max(np.abs(bc))), 1e-12,
                         "evaluated at t=0 and t=T", CAT))
        dfd = np.max(np.abs(np.gradient(s, t, edge_order=2) - sd)[5:-5])
        out.append(Check(f"{kind.capitalize()} time scaling: analytic ṡ vs finite differences of s(t)", "MR 9.2.2",
                         float(dfd), 2e-3 if kind == "trapezoidal" else 1e-5, "central differences, h = 1 ms", CAT))
        err, detail = sddot_fd_error(kind, Tf)
        out.append(Check(f"{kind.capitalize()} time scaling: analytic s̈ vs finite differences of ṡ(t)", "MR 9.2.2",
                         err, 1e-5, detail, CAT))
    rob = get_robot("UR5 (6R industrial)")
    ths, thg = rob.home.copy(), rob.home + np.array([0.6, -0.4, 0.5, 0.3, -0.5, 0.8])
    tr = make_trajectory(rob, ths, thg, 2.0, 51, "joint", "quintic")
    ref = np.asarray(mr.JointTrajectory(ths, thg, 2.0, 51, 5))
    out.append(Check("Joint-space trajectory = mr.JointTrajectory (quintic, UR5)", "MR 9.2.1",
                     float(np.abs(tr.theta - ref).max()), 1e-12, "51 samples, method=5", CAT))
    Xs, Xe = rob.fk(ths), rob.fk(thg)
    for kind, fn in (("screw", mr.ScrewTrajectory), ("cartesian", mr.CartesianTrajectory)):
        ref = fn(Xs, Xe, 2.0, 21, 3)
        s, _, _ = time_scaling("cubic", 2.0, np.linspace(0, 2.0, 21))
        mine = [pose_on_path(Xs, Xe, sk, kind) for sk in s]
        err = max(np.abs(a - b).max() for a, b in zip(mine, ref))
        out.append(Check(f"{kind.capitalize()} path ∘ cubic scaling = mr.{fn.__name__}", "MR 9.2.1",
                         float(err), 1e-12, "21 poses, UR5 start/goal", CAT))
    jp = joint_path(rob, ths, thg, "cartesian", 61)
    p0, p1 = jp.ee_target[0], jp.ee_target[-1]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    dev = jp.ee - p0 - np.outer((jp.ee - p0) @ u, u)
    out.append(Check("Cartesian path is a straight line (FK of the IK solutions)", "MR 9.2.1",
                     float(np.linalg.norm(dev, axis=1).max()), 1e-6,
                     f"max distance of the end-effector from the segment; {jp.ik_failures} IK failures", CAT))
    jps = joint_path(rob, ths, thg, "screw", 61)
    out.append(Check("Screw path tracked by warm-started IK (pose error)", "MR 6.2 / 9.2.1",
                     float(jps.max_ik_error), 1e-6, "Newton-Raphson, 61 poses", CAT))
    for name in ("Planar 2R arm", "UR5 (6R industrial)"):
        r = get_robot(name)
        eng = DynamicsEngine(r, "pinocchio")
        a, b = r.home, r.home + (np.array([-1.2, -1.0]) if r.n == 2 else np.array([0.8, 0.5, -0.6, 0.4, 0.5, -0.7]))
        jpj = joint_path(r, a, b, "joint", 41)
        res = topp(eng, jpj, r.tau_max, default_velocity_limits(r), n_grid=200, samples=400)
        worst = max(np.abs(res.tau_norm).max(), res.vel_norm.max()) if res.ok else np.inf
        out.append(Check(f"TOPP-RA respects the limits: max(|τ|/τmax, |θ̇|/vmax) ≤ 1 ({name})", "MR 9.4",
                         float(max(worst - 1.0, 0.0)), 1e-3,
                         f"T* = {res.duration:.3f} s, evaluated at 400 instants between grid points", CAT))
        sat = np.max(np.maximum(np.abs(res.tau_norm), res.vel_norm), axis=1) if res.ok else np.zeros(1)
        out.append(Check(f"TOPP-RA is bang-bang: some limit active at every instant ({name})", "MR 9.4",
                         float(1.0 - np.percentile(sat, 10)) if res.ok else np.inf, 0.05,
                         "10th percentile of max_i(|τ_i|/τmax_i, |θ̇_i|/vmax_i)", CAT))
        out.append(Check(f"TOPP-RA faster than the fastest feasible quintic ({name})", "MR 9.4",
                         float(max(res.duration - res.quintic_T, 0.0)), 1e-9,
                         f"T* = {res.duration:.3f} s vs quintic {res.quintic_T:.3f} s", CAT))
    return out
