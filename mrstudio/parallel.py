"""Stewart-Gough platform (6-SPS / 6-UPS parallel robot) -- Modern Robotics Secs. 7.1.2, 7.2.1, 7.3.

Geometry: base joints a_i (in {s}) on a circle of radius Rb, platform joints b_i (in the platform
frame {b}) on a circle of radius Rp, both grouped in pairs.  With the platform pose T_sb = (R, p):

    l_i = p + R b_i - a_i,   s_i = ||l_i||,   u_i = l_i / s_i              (inverse kinematics, closed form)

Velocity: with the spatial twist V_s = (w, v) of the platform (MR twist order), the platform point
x_i = p + R b_i moves with x_i_dot = v + w x x_i, hence

    s_i_dot = u_i . x_i_dot = (x_i x u_i)^T w + u_i^T v     =>   s_dot = J(T) V_s,
    row i of J = [(x_i x u_i)^T, u_i^T]          (J = J_s^-1 of MR Sec. 7.2.1)

Statics (power balance f^T s_dot = F_s^T V_s):  F_s = J^T f,  f = leg forces (> 0 = compression).
Forward kinematics: Newton-Raphson on the six leg lengths, updating T <- exp([V]) T with
V = J^-1 (s_target - s(T)).  The general platform has up to 40 (complex) FK solutions.

Dynamics (MR 8.2.2, Eq. 8.40 in the platform frame {b}, Eq. 8.42 for the off-CoM frame): platform +
payload form one rigid body with spatial inertia G_b; the legs are assumed massless.  The leg forces
act on the body through the wrench [Ad_Tsb]^T J^T f, so

    [Ad_Tsb]^T J^T f + F_g,b = G_b Vdot_b - [ad_Vb]^T G_b V_b            (inverse dynamics: solve for f)

and forward dynamics integrates Vdot_b with RK4 on SE(3) (T <- T exp([xi])).
"""
from __future__ import annotations

from dataclasses import dataclass

import modern_robotics as mr
import numpy as np

from .checks import Check
from .closed_chains import exp6
from .kinematics import rpy_to_rot, rot_to_rpy

G_ACC = 9.81
CATEGORY = "Parallel mechanisms"


def pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """T_sb from position and ZYX roll/pitch/yaw (radians)."""
    return mr.RpToTrans(rpy_to_rot([roll, pitch, yaw]), np.array([x, y, z], float))


def pose_vec(T) -> np.ndarray:
    """(x, y, z, roll, pitch, yaw) of a transform."""
    return np.r_[T[:3, 3], rot_to_rpy(T[:3, :3])]


@dataclass
class FKResult:
    T: np.ndarray
    converged: bool
    history: list
    iterations: int


class StewartPlatform:
    """6-6 Stewart-Gough platform with paired joints ("semi-regular hexagons")."""

    def __init__(self, Rb=0.50, Rp=0.30, half_gap_b=10.0, half_gap_p=10.0, z_home=0.55,
                 s_min=0.45, s_max=0.85, platform_mass=4.0):
        self.Rb, self.Rp, self.z_home = Rb, Rp, z_home
        self.s_min, self.s_max = s_min, s_max
        self.platform_mass = platform_mass
        gb, gp = np.radians(half_gap_b), np.radians(half_gap_p)
        base_ang = np.array([-gb, gb, 2 * np.pi / 3 - gb, 2 * np.pi / 3 + gb, 4 * np.pi / 3 - gb, 4 * np.pi / 3 + gb])
        c = np.radians([-60, 60, 60, 180, 180, 300])
        plat_ang = c + np.array([gp, -gp, gp, -gp, gp, -gp])
        self.a = np.c_[Rb * np.cos(base_ang), Rb * np.sin(base_ang), np.zeros(6)]
        self.b = np.c_[Rp * np.cos(plat_ang), Rp * np.sin(plat_ang), np.zeros(6)]
        self.T_home = pose(z=z_home)
        self.Lc = Rp  # characteristic length for a dimensionless Jacobian

    # -- inverse kinematics (MR 7.1.2) ------------------------------------------------------
    def platform_points(self, T) -> np.ndarray:
        return (T[:3, :3] @ self.b.T).T + T[:3, 3]

    def legs(self, T):
        """Platform joint positions x_i, leg lengths s_i and unit leg directions u_i."""
        x = self.platform_points(T)
        l = x - self.a
        s = np.linalg.norm(l, axis=1)
        return x, s, l / s[:, None]

    def ik(self, T) -> np.ndarray:
        return self.legs(T)[1]

    # -- velocity kinematics / statics (MR 7.2.1) ------------------------------------------------
    def jacobian(self, T) -> np.ndarray:
        """s_dot = J V_s with row i = [(x_i x u_i)^T, u_i^T]   (twist V_s = (w, v) in {s})."""
        x, _, u = self.legs(T)
        return np.c_[np.cross(x, u), u]

    def jacobian_normalised(self, T) -> np.ndarray:
        """J with the angular columns divided by the characteristic length Rp (dimensionless)."""
        J = self.jacobian(T).copy()
        J[:, :3] /= self.Lc
        return J

    def singularity(self, T):
        """(det J_n, condition number of J_n) -- Type II (actuator / parallel) singularity when det -> 0."""
        Jn = self.jacobian_normalised(T)
        sv = np.linalg.svd(Jn, compute_uv=False)
        return float(np.linalg.det(Jn)), float(sv[0] / max(sv[-1], 1e-300))

    def wrench_from_legs(self, T, f) -> np.ndarray:
        """Spatial wrench F_s = (m, f) exerted on the platform by leg forces f: F_s = J^T f."""
        return self.jacobian(T).T @ np.asarray(f, float)

    def payload_wrench(self, T, mass, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Gravity wrench (in {s}) of a payload of `mass` kg whose CoM is at `offset` in the platform frame,
        plus the platform's own weight at its centre."""
        F = np.zeros(6)
        for m, off in ((mass, np.asarray(offset, float)), (self.platform_mass, np.zeros(3))):
            c = T[:3, :3] @ off + T[:3, 3]
            fg = np.array([0.0, 0.0, -m * G_ACC])
            F += np.r_[np.cross(c, fg), fg]
        return F

    def leg_forces(self, T, F_ext) -> np.ndarray:
        """Leg forces in static equilibrium J^T f + F_ext = 0 (f > 0: leg pushes = compression)."""
        return np.linalg.solve(self.jacobian(T).T, -np.asarray(F_ext, float))

    # -- forward kinematics ---------------------------------------------------------------------
    def fk(self, s_target, T0=None, tol=1e-12, max_iter=60, max_step=0.3) -> FKResult:
        """Newton-Raphson: V = J^-1 (s_target - s(T)),  T <- exp([V]) T   (MR 6.2 applied to 7.1.2)."""
        T = (self.T_home if T0 is None else np.asarray(T0, float)).copy()
        s_target = np.asarray(s_target, float)
        hist = []
        for it in range(max_iter):
            e = s_target - self.ik(T)
            err = float(np.linalg.norm(e))
            hist.append(err)
            if err < tol:
                return FKResult(T, True, hist, it)
            J = self.jacobian(T)
            try:
                V = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                V = np.linalg.lstsq(J, e, rcond=None)[0]
            nv = np.linalg.norm(V)
            if nv > max_step:
                V *= max_step / nv
            T = exp6(V) @ T
        err = float(np.linalg.norm(s_target - self.ik(T)))
        hist.append(err)
        return FKResult(T, err < 1e-9, hist, max_iter)

    def fk_all(self, s_target, n_starts=150, seed=0, include=None) -> list[np.ndarray]:
        """Multiple FK solutions from random initial poses (the general platform has up to 40)."""
        rng = np.random.default_rng(seed)
        sols: list[np.ndarray] = []
        starts = [] if include is None else [np.asarray(include, float)]
        for _ in range(n_starts):
            p = rng.uniform([-0.6, -0.6, -1.0], [0.6, 0.6, 1.0])
            w = rng.standard_normal(3)
            w *= rng.uniform(0, np.pi) / np.linalg.norm(w)
            starts.append(mr.RpToTrans(mr.MatrixExp3(mr.VecToso3(w)), p))
        for T0 in starts:
            r = self.fk(s_target, T0, tol=1e-11, max_iter=80)
            if not r.converged or not np.all(np.isfinite(r.T)):
                continue
            if all(np.abs(r.T - S).max() > 1e-6 for S in sols):
                sols.append(r.T)
        return sols

    # -- trajectories ---------------------------------------------------------------------------------
    def dance_waypoints(self) -> list[np.ndarray]:
        z = self.z_home
        d = np.radians
        return [pose(z=z), pose(z=z + 0.12), pose(0.08, 0, z + 0.05, 0, d(12), 0),
                pose(-0.08, 0, z + 0.05, 0, d(-12), 0), pose(0, 0.08, z, d(-12), 0, d(20)),
                pose(0, -0.08, z, d(12), 0, d(-20)), pose(0, 0, z - 0.06, 0, 0, d(30)),
                pose(0, 0, z - 0.06, 0, 0, d(-30)), pose(0.06, 0.06, z + 0.08, d(8), d(8), 0), pose(z=z)]

    def dance(self, seg_time=1.2, n_per_seg=40):
        """A smooth 6-DOF 'dance': Cartesian (decoupled) quintic segments between waypoints
        (mr.CartesianTrajectory, MR 9.2).  Returns (t, list of T)."""
        way = self.dance_waypoints()
        Ts, ts = [], []
        for k in range(len(way) - 1):
            seg = mr.CartesianTrajectory(way[k], way[k + 1], seg_time, n_per_seg, 5)
            if k:
                seg = seg[1:]
            Ts += [np.asarray(X) for X in seg]
        t = np.linspace(0, seg_time * (len(way) - 1), len(Ts))
        return t, Ts

    def dance_state(self, t, seg_time=1.2):
        """Analytic (T, V_b, Vdot_b) of the dance at time t -- the exact derivatives of
        mr.CartesianTrajectory: R(t) = R0 exp([w] s(t)), p(t) = p0 + s(t)(p1 - p0), quintic s(t).
        Body twist V_b = (w s_dot, R^T p_dot);  Vdot_b = (w s_ddot, R^T p_ddot - [w_b] R^T p_dot)."""
        way = self.dance_waypoints()
        nseg = len(way) - 1
        t = float(np.clip(t, 0.0, nseg * seg_time))
        k = min(int(t // seg_time), nseg - 1)
        tau = (t - k * seg_time) / seg_time
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        sd = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / seg_time
        sdd = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / seg_time ** 2
        R0, p0 = mr.TransToRp(way[k])
        R1, p1 = mr.TransToRp(way[k + 1])
        w = mr.so3ToVec(mr.MatrixLog3(R0.T @ R1))
        R = R0 @ mr.MatrixExp3(mr.VecToso3(w * s))
        p = p0 + s * (p1 - p0)
        pd, pdd = sd * (p1 - p0), sdd * (p1 - p0)
        wb, wbd = w * sd, w * sdd
        vb = R.T @ pd
        Vb = np.r_[wb, vb]
        dVb = np.r_[wbd, R.T @ pdd - np.cross(wb, vb)]
        return mr.RpToTrans(R, p), Vb, dVb

    # -- dynamics of the platform (MR 8.2.2, Eqs. 8.40, 8.42) ---------------------------------
    def rigid_body(self, payload_mass=0.0, offset=(0.0, 0.0, 0.06)):
        """Total mass, CoM r_c (in {b}) and spatial inertia G_b (6x6, about the {b} origin) of
        platform + payload.  Platform: uniform disk of radius 1.13 Rp, thickness 22 mm, CoM at the
        {b} origin; payload: solid sphere at `offset`.  Each CoM-frame inertia G_c = diag(I_c, m I) is
        moved to {b} with Steiner's theorem in adjoint form G_b = [Ad_Tcb]^T G_c [Ad_Tcb] (Eq. 8.42).
        Assumption: the legs are massless (their inertia is lumped into the platform)."""
        bodies = []
        mp, rp, h = self.platform_mass, self.Rp * 1.13, 0.022
        bodies.append((mp, np.zeros(3), np.diag([mp * (3 * rp ** 2 + h ** 2) / 12] * 2 + [mp * rp ** 2 / 2])))
        if payload_mass > 0:
            r = 0.03 + 0.012 * payload_mass ** (1 / 3)
            bodies.append((payload_mass, np.asarray(offset, float), np.eye(3) * 0.4 * payload_mass * r ** 2))
        G = np.zeros((6, 6))
        m_tot, mc = 0.0, np.zeros(3)
        for m, c, Ic in bodies:
            Gc = np.zeros((6, 6))
            Gc[:3, :3], Gc[3:, 3:] = Ic, m * np.eye(3)
            Tcb = mr.RpToTrans(np.eye(3), -c)          # {b} seen from the CoM frame {c}
            Ad = mr.Adjoint(Tcb)
            G += Ad.T @ Gc @ Ad
            m_tot += m
            mc += m * c
        return m_tot, mc / m_tot, G

    def gravity_wrench_b(self, T, m, rc) -> np.ndarray:
        """Gravity wrench on the body expressed in {b}: (r_c x f_g, f_g), f_g = R^T (0, 0, -m g)."""
        fg = T[:3, :3].T @ np.array([0.0, 0.0, -m * G_ACC])
        return np.r_[np.cross(rc, fg), fg]

    def inverse_dynamics(self, T, Vb, dVb, body) -> np.ndarray:
        """Leg forces f realising (V_b, Vdot_b):  [Ad_Tsb]^T J^T f + F_g,b = G_b Vdot_b - [ad_Vb]^T G_b V_b
        (Eq. 8.40, written in the non-CoM frame {b} as allowed by Eq. 8.42)."""
        m, rc, G = body
        Fb = G @ dVb - mr.ad(Vb).T @ G @ Vb - self.gravity_wrench_b(T, m, rc)
        A = mr.Adjoint(T).T @ self.jacobian(T).T          # body wrench produced by unit leg forces
        return np.linalg.solve(A, Fb)

    def forward_dynamics(self, T, Vb, f, body) -> np.ndarray:
        """Vdot_b = G_b^-1 ( [Ad_Tsb]^T J^T f + F_g,b + [ad_Vb]^T G_b V_b )."""
        m, rc, G = body
        F = mr.Adjoint(T).T @ (self.jacobian(T).T @ np.asarray(f, float)) + self.gravity_wrench_b(T, m, rc)
        return np.linalg.solve(G, F + mr.ad(Vb).T @ G @ Vb)

    def energy(self, T, Vb, body):
        """(kinetic, potential) energy: K = 1/2 V_b^T G_b V_b, P = m g z_c."""
        m, rc, G = body
        zc = (T[:3, :3] @ rc + T[:3, 3])[2]
        return 0.5 * float(Vb @ G @ Vb), m * G_ACC * float(zc)

    def leg_rates(self, T, Vb) -> np.ndarray:
        """s_dot = J V_s with V_s = [Ad_Tsb] V_b."""
        return self.jacobian(T) @ (mr.Adjoint(T) @ Vb)

    def rk4_step(self, T, Vb, t, h, f_fn, body):
        """One Runge-Kutta-Munthe-Kaas RK4 step on SE(3) x R^6.  Locally T = T_k exp([xi]) with
        xi_dot = dexp^-1_{-xi}(V_b) ~ V_b + 1/2 [ad_xi] V_b + 1/12 [ad_xi]^2 V_b (4th order), so the pose
        stays exactly on SE(3) (MatrixExp6)."""
        def deriv(xi, V, tt):
            Tx = T @ exp6(xi)
            a = mr.ad(xi)
            dxi = V + 0.5 * a @ V + (a @ (a @ V)) / 12.0
            return dxi, self.forward_dynamics(Tx, V, f_fn(tt, Tx, V), body)
        x0 = np.zeros(6)
        k1x, k1v = deriv(x0, Vb, t)
        k2x, k2v = deriv(0.5 * h * k1x, Vb + 0.5 * h * k1v, t + 0.5 * h)
        k3x, k3v = deriv(0.5 * h * k2x, Vb + 0.5 * h * k2v, t + 0.5 * h)
        k4x, k4v = deriv(h * k3x, Vb + h * k3v, t + h)
        xi = h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        V = Vb + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        return T @ exp6(xi), V

    def simulate(self, T0, V0, f_fn, t_final, h, body, stop_fn=None):
        """Integrate the platform from (T0, V0) under leg forces f_fn(t, T, V_b).
        Returns dict(t, T (list), V (n x 6), f (n x 6), s (n x 6), ke, pe)."""
        T, V, t = np.asarray(T0, float).copy(), np.asarray(V0, float).copy(), 0.0
        out = dict(t=[], T=[], V=[], f=[], s=[], ke=[], pe=[])
        n = int(round(t_final / h))
        for k in range(n + 1):
            ke, pe = self.energy(T, V, body)
            out["t"].append(t); out["T"].append(T); out["V"].append(V); out["f"].append(f_fn(t, T, V))
            out["s"].append(self.ik(T)); out["ke"].append(ke); out["pe"].append(pe)
            if k == n or (stop_fn is not None and stop_fn(T)):
                break
            T, V = self.rk4_step(T, V, t, h, f_fn, body)
            t = (k + 1) * h
        for key in ("t", "V", "f", "s", "ke", "pe"):
            out[key] = np.asarray(out[key])
        return out

    def energy_rate(self, T, Vb, dVb, body) -> float:
        """dE/dt = V_b^T G_b Vdot_b + m g z_c_dot  (the [ad_V]^T term does no work: (ad_V V)^T G V = 0)."""
        m, rc, G = body
        zc_dot = (T[:3, :3] @ (np.cross(Vb[:3], rc) + Vb[3:]))[2]
        return float(Vb @ G @ dVb) + m * G_ACC * float(zc_dot)

    def dance_dynamics(self, body, seg_time=1.2, n=361):
        """Inverse dynamics along the dance.  Returns dict(t, f_dyn, f_static, power = f^T s_dot,
        dE = d/dt (K + P), ke, pe, s)."""
        nseg = len(self.dance_waypoints()) - 1
        t = np.linspace(0, nseg * seg_time, n)
        out = {k: [] for k in ("f_dyn", "f_static", "power", "dE", "ke", "pe", "s")}
        m, rc, _ = body
        for tt in t:
            T, Vb, dVb = self.dance_state(tt, seg_time)
            f = self.inverse_dynamics(T, Vb, dVb, body)
            out["f_dyn"].append(f)
            out["f_static"].append(self.leg_forces(T, mr.Adjoint(mr.TransInv(T)).T @ self.gravity_wrench_b(T, m, rc)))
            out["power"].append(float(f @ self.leg_rates(T, Vb)))
            out["dE"].append(self.energy_rate(T, Vb, dVb, body))
            ke, pe = self.energy(T, Vb, body)
            out["ke"].append(ke)
            out["pe"].append(pe)
            out["s"].append(self.ik(T))
        out = {k: np.asarray(v) for k, v in out.items()}
        out["t"] = t
        return out

    def push_simulation(self, body, leg=0, df=200.0, t_pulse=(0.1, 0.2), t_final=2.0, h=0.002,
                        servo=True, kp=3.0e4, kd=None):
        """Forward dynamics from rest at home under f = f_static(home) + pulse on one leg
        (+ optional leg-space PD servo  K_p (s_home - s) - K_d s_dot  holding the home leg lengths)."""
        m, rc, G = body
        T0 = self.T_home
        f0 = self.leg_forces(T0, mr.Adjoint(mr.TransInv(T0)).T @ self.gravity_wrench_b(T0, m, rc))
        s0 = self.ik(T0)
        if kd is None:
            kd = 2 * 0.25 * np.sqrt(kp * m)      # ~ 25 % damping of the heave mode

        def f_fn(t, T, V):
            f = f0.copy()
            if t_pulse[0] <= t < t_pulse[1]:
                f[leg] += df
            if servo:
                f += kp * (s0 - self.ik(T)) - kd * self.leg_rates(T, V)
            return f

        def stop(T):
            s = self.ik(T)
            return bool(np.any(s < self.s_min - 0.15) or np.any(s > self.s_max + 0.15) or T[2, 3] < 0.1)
        return self.simulate(T0, np.zeros(6), f_fn, t_final, h, body, stop_fn=stop)


# ============================================================================ validation
def validation_checks() -> list[Check]:
    rng = np.random.default_rng(3)
    sp = StewartPlatform()
    out: list[Check] = []

    def rand_pose():
        return pose(*rng.uniform(-0.08, 0.08, 2), sp.z_home + rng.uniform(-0.08, 0.08),
                    *np.radians(rng.uniform(-15, 15, 3)))

    # 1. FK(IK(T)) round trip
    err, its = 0.0, 0
    for _ in range(25):
        T = rand_pose()
        r = sp.fk(sp.ik(T), sp.T_home)
        err = max(err, float(np.abs(r.T - T).max()) if r.converged else 1.0)
        its = max(its, r.iterations)
    out.append(Check("Stewart platform FK(IK(T)) round trip (Newton-Raphson, T <- exp([V])T)", "MR 7.1.2",
                     err, 1e-9, f"25 random poses, NR started at home, <= {its} iterations", CATEGORY))

    # 2. Jacobian vs finite differences, 3. statics power consistency
    eJ = eP = 0.0
    eps = 1e-6
    for _ in range(10):
        T = rand_pose()
        V = rng.standard_normal(6)
        sp_ = sp.ik(exp6(V * eps) @ T)
        sm_ = sp.ik(exp6(-V * eps) @ T)
        sdot_fd = (sp_ - sm_) / (2 * eps)
        eJ = max(eJ, float(np.abs(sp.jacobian(T) @ V - sdot_fd).max()))
        f = rng.standard_normal(6) * 100
        F = sp.wrench_from_legs(T, f)
        eP = max(eP, abs(float(f @ sdot_fd - F @ V)) / (np.abs(f).max() * np.abs(V).max()))
    out.append(Check("Stewart Jacobian rows [(x_i x u_i)^T, u_i^T] vs finite differences of IK", "MR 7.2.1",
                     eJ, 1e-7, "s_dot = J V_s for random spatial twists", CATEGORY))
    out.append(Check("Statics power consistency f^T s_dot = F_s^T V_s with F_s = J^T f", "MR 7.2.1, 3.4",
                     eP, 1e-7, "s_dot by finite differences (independent of J)", CATEGORY))

    # 4. wrench change of frame with the adjoint
    T = rand_pose()
    f = rng.standard_normal(6) * 50
    Fs = sp.wrench_from_legs(T, f)
    R = T[:3, :3]
    _, _, u = sp.legs(T)
    Fb_direct = np.zeros(6)
    for i in range(6):
        fb = R.T @ (f[i] * u[i])
        Fb_direct += np.r_[np.cross(sp.b[i], fb), fb]
    out.append(Check("Platform wrench: F_b = [Ad_Tsb]^T F_s vs direct sum in {b}", "MR 3.4, Eq. 3.98",
                     float(np.abs(mr.Adjoint(T).T @ Fs - Fb_direct).max()), 1e-10,
                     "leg forces summed in the platform frame", CATEGORY))

    # 5. multiple FK solutions all reproduce the leg lengths
    s = sp.ik(sp.T_home)
    sols = sp.fk_all(s, n_starts=60, seed=1)
    res = max(float(np.abs(sp.ik(X) - s).max()) for X in sols)
    out.append(Check(f"Stewart FK: {len(sols)} distinct assembly modes found from 60 random starts", "MR 7.1.2",
                     res if len(sols) >= 2 else 1.0, 1e-9,
                     "all solutions satisfy s(T) = s (mirror images through the base plane included)", CATEGORY))

    # ---------------------------------------------------------------- dynamics (MR 8.2.2)
    body = sp.rigid_body(25.0, (0.05, -0.03, 0.06))
    m, rc, G = body
    # 6. Steiner's theorem: adjoint form (8.42) vs classical parallel-axis formula
    Gc = np.zeros((6, 6))
    for mi, ci in ((sp.platform_mass, np.zeros(3)), (25.0, np.array([0.05, -0.03, 0.06]))):
        r = 0.03 + 0.012 * 25.0 ** (1 / 3)
        Ic = (np.diag([sp.platform_mass * (3 * (sp.Rp * 1.13) ** 2 + 0.022 ** 2) / 12] * 2
                      + [sp.platform_mass * (sp.Rp * 1.13) ** 2 / 2]) if mi == sp.platform_mass
              else np.eye(3) * 0.4 * mi * r ** 2)
        Gc[:3, :3] += Ic + mi * (ci @ ci * np.eye(3) - np.outer(ci, ci))
        Gc[:3, 3:] += mi * mr.VecToso3(ci)
        Gc[3:, :3] += -mi * mr.VecToso3(ci)
        Gc[3:, 3:] += mi * np.eye(3)
    out.append(Check("Platform+payload spatial inertia: [Ad_Tcb]^T G_c [Ad_Tcb] vs parallel-axis formula",
                     "MR 8.2.3, Eq. 8.42", float(np.abs(G - Gc).max()), 1e-12,
                     "I_b = I_c + m(|r|^2 I - r r^T), coupling blocks m[r]", CATEGORY))
    # 7. static limit of the inverse dynamics
    T = rand_pose()
    eS = float(np.abs(sp.inverse_dynamics(T, np.zeros(6), np.zeros(6), body)
                      - sp.leg_forces(T, sp.payload_wrench(T, 25.0, (0.05, -0.03, 0.06)))).max())
    out.append(Check("Stewart inverse dynamics at rest = statics f = -J^-T F_g", "MR 8.2.2, 7.2.1",
                     eS, 1e-9, "V_b = Vdot_b = 0 in Eq. 8.40 (massless legs)", CATEGORY))
    # 8. analytic dance twist vs mr.CartesianTrajectory + finite differences of log
    tt, Ts = sp.dance()
    eT = max(float(np.abs(sp.dance_state(a)[0] - b).max()) for a, b in zip(tt, Ts))
    eV = 0.0
    for t0 in rng.uniform(0.05, tt[-1] - 0.05, 12):
        Tm, Tp = sp.dance_state(t0 - 1e-5)[0], sp.dance_state(t0 + 1e-5)[0]
        _, Vb, dVb = sp.dance_state(t0)
        Vfd = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Tm) @ Tp)) / 2e-5
        dVfd = (sp.dance_state(t0 + 1e-5)[1] - sp.dance_state(t0 - 1e-5)[1]) / 2e-5
        eV = max(eV, float(np.abs(Vfd - Vb).max()), float(np.abs(dVfd - dVb).max()))
    out.append(Check("Dance: analytic T, V_b, Vdot_b vs mr.CartesianTrajectory and finite differences", "MR 9.2, 3.3",
                     max(eT, eV), 1e-6, f"pose vs mr samples {eT:.1e}; [V_b] = T^-1 T_dot by log differences",
                     CATEGORY))
    # 9. power balance f^T s_dot = d/dt (K + P)   (dE/dt by finite differences, independent)
    eP = 0.0
    for t0 in rng.uniform(0.05, tt[-1] - 0.05, 12):
        T, Vb, dVb = sp.dance_state(t0)
        f = sp.inverse_dynamics(T, Vb, dVb, body)
        Em = sum(sp.energy(*sp.dance_state(t0 - 1e-5)[:2], body))
        Ep = sum(sp.energy(*sp.dance_state(t0 + 1e-5)[:2], body))
        eP = max(eP, abs(float(f @ sp.leg_rates(T, Vb)) - (Ep - Em) / 2e-5))
    out.append(Check("Stewart dynamics power balance f^T s_dot = d/dt(K + P) along the dance", "MR 8.2.2, 7.2.1",
                     eP, 1e-5, "inverse-dynamics leg forces (Eq. 8.40); K = 1/2 V_b^T G_b V_b, P = m g z_c [W]",
                     CATEGORY))
    # 10. inverse -> forward round trip
    f_fn = lambda t, T, V: sp.inverse_dynamics(*sp.dance_state(t), body)
    sim = sp.simulate(sp.T_home, np.zeros(6), f_fn, 1.2, 0.005, body)
    eR = max(float(np.abs(X - sp.dance_state(t)[0]).max()) for t, X in zip(sim["t"], sim["T"]))
    out.append(Check("Stewart inverse -> forward dynamics round trip (RK4 on SE(3), first dance segment)",
                     "MR 8.2.2", eR, 1e-8, "leg forces from Eq. 8.40 fed open loop to the forward dynamics, "
                     "h = 5 ms, 1.2 s; max |T_sim - T_d|", CATEGORY))
    # 11. energy conservation of the forward dynamics with no leg forces
    sim = sp.simulate(sp.T_home, rng.standard_normal(6), lambda t, T, V: np.zeros(6), 0.5, 0.005, body)
    E = sim["ke"] + sim["pe"]
    out.append(Check("Platform forward dynamics (f = 0): energy conservation of RK4-Munthe-Kaas on SE(3)",
                     "MR 8.2.2", float(np.abs(E - E[0]).max() / abs(E[0])), 1e-9,
                     "relative drift of K + P over 0.5 s, h = 5 ms, random initial twist", CATEGORY))
    return out
