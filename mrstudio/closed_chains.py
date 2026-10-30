"""Planar closed chains: the four-bar linkage and the five-bar parallel robot.

Modern Robotics (Lynch & Park): Ch. 7 (kinematics of closed chains, Secs. 7.1-7.3) and
Sec. 8.7 (constrained dynamics with Lagrange multipliers).

"Tree + cut joint" formulation
------------------------------
The loop is cut open at one pin joint (the *cut joint*).  What remains is a tree of serial
branches rooted at ground pivots.  Each branch is an ordinary planar open chain described exactly
as in the book -- screw axes ``Slist`` in the branch base frame {0_b} (located at its ground
pivot, aligned with {s}), link CoM frames ``Mlist`` and spatial inertias ``Glist`` (uniform rods),
gravity along -y.  The cut pin becomes the loop-closure constraint

    phi(theta) = p_tip,1(theta) - p_tip,2(theta) = 0  in R^2
    A(theta)   = d phi / d theta                       (2 x n constraint Jacobian)
    A theta_dot = 0,     A theta_ddot + Adot theta_dot = 0.

* Four-bar  : branch 1 = [crank L1, coupler L2] from O, branch 2 = [rocker L3] from D.
              n = 3 joints, 2 constraints -> 1 DOF (actuated: the crank).
* Five-bar  : branch 1 = [L1, L2] from O, branch 2 = [L4, L3] from D.  n = 4 -> 2 DOF
              (actuated: both ground joints); the end-effector is the common tip P.

Dynamics (cf. MR Sec. 8.7, Eqs. 8.93/8.98):  M(theta) theta_ddot + h(theta, theta_dot) = tau + A^T lambda
with block-diagonal M and stacked h of the branches (``mr.MassMatrix``, ``mr.VelQuadraticForces``,
``mr.GravityForces`` through :class:`mrstudio.dynamics.DynamicsEngine`, backend "mr" or
"pinocchio").  lambda is the force the cut pin exerts on the tip of branch 1 (and -lambda on
branch 2).  Sign convention: MR writes tau = M theta_ddot + h + A^T lambda_MR, so
lambda here = -lambda_MR (same physics; our lambda is directly the pin force acting on branch 1).  Forward dynamics solves the KKT system with Baumgarte stabilisation; inverse
dynamics solves [S, A^T][tau_a; lambda] = M theta_ddot + h.
"""
from __future__ import annotations

from dataclasses import dataclass

import modern_robotics as mr
import numpy as np

from .checks import Check
from .dynamics import DynamicsEngine
from .robots import SerialRobot, rod_inertia, screw_revolute, trans

G_ACC = 9.81
CATEGORY = "Closed chains"


# ============================================================================ helpers
def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def circle_intersection(c1, r1, c2, r2, sigma: int = 1):
    """Intersection of the circles |x-c1| = r1 and |x-c2| = r2 (closed-form loop closure).

    sigma = +1 returns the point to the *left* of the directed line c1 -> c2, sigma = -1 the one
    to the right (the two assembly modes).  Returns None when the loop cannot be closed."""
    c1 = np.asarray(c1, float)[:2]
    c2 = np.asarray(c2, float)[:2]
    dv = c2 - c1
    d = float(np.hypot(*dv))
    if d < 1e-12:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h2 = r1 ** 2 - a ** 2
    if h2 < -1e-10 * max(r1 ** 2, 1e-12):
        return None
    h = np.sqrt(max(h2, 0.0))
    u = dv / d
    return c1 + a * u + sigma * h * np.array([-u[1], u[0]])


def exp6(V) -> np.ndarray:
    """e^[V] for a twist V = (w, v) using mr.MatrixExp6.  mr.MatrixExp6 treats ||w|| < 1e-6 as a pure
    translation (its NearZero threshold) and silently drops tiny rotations, which stalls Newton-Raphson
    near convergence; there the exact series I + [V] + [V]^2/2 + [V]^3/6 (error O(||w||^3)) is used."""
    V = np.asarray(V, float)
    se3 = mr.VecTose3(V)
    if np.linalg.norm(V[:3]) >= 1e-5:
        return mr.MatrixExp6(se3)
    se3_2 = se3 @ se3
    return np.eye(4) + se3 + se3_2 / 2 + se3_2 @ se3 / 6


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _e(angle):
    return np.array([np.cos(angle), np.sin(angle)])


# ============================================================================ branch
class Branch:
    """One serial branch of the cut-open mechanism: a planar nR chain rooted at a ground pivot.

    Described the Modern Robotics way (Sec. 4.1 and 8.3.2) in its base frame {0_b} at the pivot:
    joint i is a revolute screw S_i = (z, -z x q_i) through q_i = (x_i, 0, 0) at home, the tip
    frame is M = Trans(sum L, 0, 0), link CoM frames sit at the rod mid-points."""

    def __init__(self, name: str, pivot, lengths, masses, radius: float = 0.012):
        self.name = name
        self.pivot = np.r_[np.asarray(pivot, float).ravel()[:2], 0.0]
        self.L = np.asarray(lengths, float)
        self.masses = np.asarray(masses, float)
        self.radius = radius
        self.k = len(self.L)
        x = np.r_[0.0, np.cumsum(self.L)]
        self.x_joint = x
        Slist = np.column_stack([screw_revolute([0, 0, 1], [x[i], 0, 0]) for i in range(self.k)])
        com = x[:-1] + self.L / 2
        Mlist = ([trans([com[0], 0, 0])] + [trans([com[i] - com[i - 1], 0, 0]) for i in range(1, self.k)]
                 + [trans([x[-1] - com[-1], 0, 0])])
        Glist = [rod_inertia(m, l, radius) for m, l in zip(self.masses, self.L)]
        self.robot = SerialRobot(name, trans([x[-1], 0, 0]), Slist, Mlist, Glist, planar=True,
                                 gravity=np.array([0.0, -G_ACC, 0.0]), link_radius=radius,
                                 description=f"branch of a closed chain, pivot {self.pivot[:2]}")

    # -- PoE kinematics of the tip --------------------------------------------
    def kinematics(self, th, dth=None):
        """Tip position (world), point Jacobian Jp (3 x k) and velocity-product acceleration c.

        Space Jacobian columns J_si = [Ad_{e^[S1]th1...e^[S_i-1]th_i-1}] S_i (MR Eq. 5.11);
        tip velocity p_dot = v_s + w_s x p  =>  Jp = J_v - [p] J_w.
        p_ddot = Jp th_ddot + c  with  c = (Jdot th_dot)_w x p + (Jdot th_dot)_v + w x p_dot  and
        Jdot_i = [ad_{V_{i-1}}] J_si,  V_{i-1} = sum_{j<i} J_sj th_dot_j  (MR Sec. 8.3 / Ex. 5)."""
        th = np.asarray(th, float)
        S = self.robot.Slist
        T = np.eye(4)
        Js = np.zeros((6, self.k))
        for i in range(self.k):
            Js[:, i] = mr.Adjoint(T) @ S[:, i]
            T = T @ exp6(S[:, i] * th[i])
        p = (T @ self.robot.M)[:3, 3]
        Jp = Js[3:] - mr.VecToso3(p) @ Js[:3]
        c = np.zeros(3)
        if dth is not None:
            dth = np.asarray(dth, float)
            V = np.zeros(6)
            Jd = np.zeros(6)
            for i in range(self.k):
                Jd += (mr.ad(V) @ Js[:, i]) * dth[i]
                V += Js[:, i] * dth[i]
            w, v = V[:3], V[3:]
            pdot = v + np.cross(w, p)
            c = np.cross(Jd[:3], p) + Jd[3:] + np.cross(w, pdot)
        return self.pivot + p, Jp, c

    def points(self, th) -> np.ndarray:
        """World positions of the joints and of the tip ((k+1) x 3), from the PoE link transforms."""
        Ts = self.robot.link_transforms(np.asarray(th, float))
        return np.array([self.pivot + Ts[i][:3, :3] @ np.array([self.x_joint[i], 0, 0]) + Ts[i][:3, 3]
                         for i in range(self.k + 1)])

    def com_positions(self, th) -> np.ndarray:
        return self.robot.com_positions(np.asarray(th, float)) + self.pivot


# ============================================================================ generic closed chain
@dataclass
class ClosedSim:
    t: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    lam: np.ndarray       # pin force on branch 1 (N)
    phi: np.ndarray       # ||phi|| constraint violation (m)
    ke: np.ndarray
    pe: np.ndarray
    work: np.ndarray      # work done by the motors
    dissipated: np.ndarray


@dataclass
class IDResult:
    t: np.ndarray
    x: np.ndarray         # abscissa for plots (crank angle in deg, or time)
    xlabel: str
    theta: np.ndarray
    dtheta: np.ndarray
    ddtheta: np.ndarray
    tau: np.ndarray       # actuator torques (N x n_a)
    lam: np.ndarray       # pin reaction force on branch 1 (N x 2)
    ke: np.ndarray
    pe: np.ndarray
    power: np.ndarray     # tau_a . theta_a_dot
    dEdt: np.ndarray      # numerical derivative of the mechanical energy


class ClosedChain:
    """Tree of planar branches + one cut pin joint: tip of branch 0 == tip of branch 1.

    Each branch must consist of actuated joints followed by exactly one passive joint (true for
    the four-bar and the five-bar); this enables the closed-form circle-circle forward kinematics."""
    kind = "generic"

    def __init__(self, name: str, branches: list[Branch], actuated, backend: str = "mr"):
        self.name = name
        self.branches = branches
        self.k = [b.k for b in branches]
        self.n = int(sum(self.k))
        off = np.r_[0, np.cumsum(self.k)].astype(int)
        self.sl = [slice(int(off[i]), int(off[i + 1])) for i in range(len(branches))]
        self.actuated = np.array(actuated, int)
        self.passive = np.array([i for i in range(self.n) if i not in set(self.actuated.tolist())], int)
        self.na = len(self.actuated)
        self.backend = backend
        self._eng: dict = {}
        self.S = np.zeros((self.n, self.na))
        self.S[self.actuated, np.arange(self.na)] = 1.0
        for s in self.sl:  # structure check for the closed-form FK
            last = s.stop - 1
            assert last in self.passive and all(j in self.actuated for j in range(s.start, last))

    # -- bookkeeping --------------------------------------------------------------------------
    def split(self, th):
        th = np.asarray(th, float)
        return [th[s] for s in self.sl]

    @property
    def dof(self) -> int:
        return self.n - 2

    def joint_names(self) -> list[str]:
        return [f"θ{i + 1}" for i in range(self.n)]

    # -- loop-closure constraint (MR 7.1, 7.2) -----------------------------------------------
    def constraint(self, th, dth=None):
        """phi(theta) (2,), A(theta) = d phi/d theta (2 x n) and Adot theta_dot (2,)."""
        th = np.asarray(th, float)
        dtb = self.split(dth) if dth is not None else [None] * len(self.branches)
        p1, J1, c1 = self.branches[0].kinematics(th[self.sl[0]], dtb[0])
        p2, J2, c2 = self.branches[1].kinematics(th[self.sl[1]], dtb[1])
        A = np.zeros((2, self.n))
        A[:, self.sl[0]] = J1[:2]
        A[:, self.sl[1]] = -J2[:2]
        return (p1 - p2)[:2], A, (c1 - c2)[:2]

    def tip(self, th) -> np.ndarray:
        return self.branches[0].kinematics(np.asarray(th, float)[self.sl[0]])[0]

    def points(self, th) -> list[np.ndarray]:
        return [b.points(t) for b, t in zip(self.branches, self.split(th))]

    def assembly_mode(self, th) -> int:
        """sigma = side of the cut pin w.r.t. the line (passive joint of branch 1 -> passive joint of branch 2)."""
        pts = self.points(th)
        c1, c2, P = pts[0][-2], pts[1][-2], pts[0][-1]
        return 1 if _cross2(c2[:2] - c1[:2], P[:2] - c1[:2]) >= 0 else -1

    # -- forward kinematics: actuated -> passive -------------------------------------------
    def fk_closed_form(self, theta_a, sigma: int = 1):
        """Closed-form FK: the actuated joints fix the passive joint of each branch; the tip lies
        on a circle around each of them (radius = last link) -> circle-circle intersection."""
        th = np.zeros(self.n)
        th[self.actuated] = np.asarray(theta_a, float)
        centers, radii, pre = [], [], []
        for b, s in zip(self.branches, self.sl):
            tb = th[s]
            c = b.pivot[:2].copy()
            ang = 0.0
            for i in range(b.k - 1):
                ang += tb[i]
                c = c + b.L[i] * _e(ang)
            centers.append(c)
            radii.append(b.L[-1])
            pre.append(ang)
        P = circle_intersection(centers[0], radii[0], centers[1], radii[1], sigma)
        if P is None:
            return None
        for s, c, a0 in zip(self.sl, centers, pre):
            d = P - c
            th[s.stop - 1] = wrap(np.arctan2(d[1], d[0]) - a0)
        return th

    def fk_newton(self, theta_a, theta_guess, tol: float = 1e-13, max_iter: int = 40):
        """Newton-Raphson on the loop-closure equations for the passive joints (MR 6.2 applied to 7.1):
        theta_p <- theta_p - A_p^+ phi(theta).  Returns (theta, converged, residual history)."""
        th = np.array(theta_guess, float)
        th[self.actuated] = np.asarray(theta_a, float)
        hist = []
        for _ in range(max_iter):
            phi, A, _ = self.constraint(th)
            e = float(np.linalg.norm(phi))
            hist.append(e)
            if e < tol:
                return th, True, hist
            th[self.passive] -= np.linalg.pinv(A[:, self.passive]) @ phi
        phi = self.constraint(th)[0]
        hist.append(float(np.linalg.norm(phi)))
        return th, hist[-1] < 1e-9, hist

    def passive_rates(self, th, dth_a, ddth_a=None):
        """theta_p_dot = -A_p^-1 A_a theta_a_dot   (cf. MR Eq. 7.15), and
        theta_p_ddot = -A_p^-1 (A_a theta_a_ddot + Adot theta_dot)."""
        _, A, _ = self.constraint(th)
        Aa, Ap = A[:, self.actuated], A[:, self.passive]
        dth = np.zeros(self.n)
        dth[self.actuated] = dth_a
        dth[self.passive] = -np.linalg.solve(Ap, Aa @ np.asarray(dth_a, float))
        if ddth_a is None:
            return dth
        _, _, adq = self.constraint(th, dth)
        ddth = np.zeros(self.n)
        ddth[self.actuated] = ddth_a
        ddth[self.passive] = -np.linalg.solve(Ap, Aa @ np.asarray(ddth_a, float) + adq)
        return dth, ddth

    def actuator_singularity(self, th) -> float:
        """det A_p normalised by the product of the passive-link lengths (0 => actuator singularity, MR 7.3)."""
        _, A, _ = self.constraint(th)
        Ap = A[:, self.passive]
        scale = np.prod([b.L[-1] for b in self.branches])
        return float(np.linalg.det(Ap) / scale)

    # -- dynamics (MR 8.7) ----------------------------------------------------------------
    def engine(self, i: int, backend: str | None = None) -> DynamicsEngine:
        be = backend or self.backend
        key = (i, be)
        if key not in self._eng:
            self._eng[key] = DynamicsEngine(self.branches[i].robot, be)
        return self._eng[key]

    def mass_matrix(self, th, backend=None) -> np.ndarray:
        """Block-diagonal M(theta) = diag(M_1, M_2) of the cut-open tree (mr.MassMatrix per branch)."""
        th = np.asarray(th, float)
        M = np.zeros((self.n, self.n))
        for i, s in enumerate(self.sl):
            M[s, s] = self.engine(i, backend).mass_matrix(th[s])
        return M

    def bias(self, th, dth, backend=None) -> np.ndarray:
        """h(theta, theta_dot) = c + g stacked over the branches (mr.VelQuadraticForces + mr.GravityForces)."""
        th, dth = np.asarray(th, float), np.asarray(dth, float)
        h = np.zeros(self.n)
        for i, s in enumerate(self.sl):
            e = self.engine(i, backend)
            if e.backend == "mr":
                h[s] = e.coriolis(th[s], dth[s]) + e.gravity_torque(th[s])
            else:  # Pinocchio: one RNEA call, h = ID(theta, theta_dot, 0) = c + g
                h[s] = e.inverse(th[s], dth[s], np.zeros(len(th[s])))
        return h

    def energy(self, th, dth, M=None) -> tuple[float, float]:
        th, dth = np.asarray(th, float), np.asarray(dth, float)
        M = self.mass_matrix(th) if M is None else M
        ke = 0.5 * float(dth @ M @ dth)
        pe = 0.0
        for b, tb in zip(self.branches, self.split(th)):
            pe += G_ACC * float(np.sum(b.masses * b.com_positions(tb)[:, 1]))
        return ke, pe

    def forward_dynamics(self, th, dth, tau, alpha: float = 0.0, beta: float = 0.0, backend=None):
        """KKT system  [[M, -A^T], [A, 0]] [theta_ddot; lambda] = [tau - h; -Adot th_dot - 2 a A th_dot - b^2 phi].

        lambda = force of the cut pin on branch 1 = -lambda of MR Eqs. (8.93)/(8.98)."""
        th, dth = np.asarray(th, float), np.asarray(dth, float)
        n = self.n
        M = self.mass_matrix(th, backend)
        h = self.bias(th, dth, backend)
        phi, A, adq = self.constraint(th, dth)
        K = np.zeros((n + 2, n + 2))
        K[:n, :n] = M
        K[:n, n:] = -A.T
        K[n:, :n] = A
        rhs = np.r_[np.asarray(tau, float) - h, -adq - 2 * alpha * (A @ dth) - beta ** 2 * phi]
        sol = np.linalg.solve(K, rhs)
        return sol[:n], sol[n:]

    def rk4_step(self, th, dth, t, dt, tau_fn, alpha=10.0, beta=10.0, damping=0.0, backend=None):
        """One classic RK4 step of the constrained dynamics.  Also integrates the motor work and
        the viscous dissipation with the same weights.  Returns (th, dth, lambda(t), dW, dD)."""

        def f(tt, q, dq):
            tau = np.asarray(tau_fn(tt, q, dq), float)
            dd, lam = self.forward_dynamics(q, dq, tau - damping * dq, alpha, beta, backend)
            return dd, lam, float(tau @ dq), float(damping * dq @ dq)

        a1, lam, P1, D1 = f(t, th, dth)
        v2 = dth + dt / 2 * a1
        a2, _, P2, D2 = f(t + dt / 2, th + dt / 2 * dth, v2)
        v3 = dth + dt / 2 * a2
        a3, _, P3, D3 = f(t + dt / 2, th + dt / 2 * v2, v3)
        v4 = dth + dt * a3
        a4, _, P4, D4 = f(t + dt, th + dt * v3, v4)
        th_new = th + dt / 6 * (dth + 2 * v2 + 2 * v3 + v4)
        dth_new = dth + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        return (th_new, dth_new, lam, dt / 6 * (P1 + 2 * P2 + 2 * P3 + P4),
                dt / 6 * (D1 + 2 * D2 + 2 * D3 + D4))

    def simulate(self, th0, dth0, tau_fn, t_final: float, dt: float = 0.002, alpha=10.0, beta=10.0,
                 damping=0.0, backend=None) -> ClosedSim:
        steps = int(round(t_final / dt))
        th, dth = np.array(th0, float), np.array(dth0, float)
        T, TH, DTH, LAM, PHI, KE, PE, W, D = ([] for _ in range(9))
        w = dd = 0.0
        for k in range(steps + 1):
            t = k * dt
            phi = self.constraint(th)[0]
            ke, pe = self.energy(th, dth)
            if k < steps:
                th_n, dth_n, lam, dw, ddis = self.rk4_step(th, dth, t, dt, tau_fn, alpha, beta, damping, backend)
            else:
                lam = self.forward_dynamics(th, dth, np.asarray(tau_fn(t, th, dth), float) - damping * dth,
                                            alpha, beta, backend)[1]
            T.append(t); TH.append(th); DTH.append(dth); LAM.append(lam)
            PHI.append(np.linalg.norm(phi)); KE.append(ke); PE.append(pe); W.append(w); D.append(dd)
            if k < steps:
                th, dth = th_n, dth_n
                w += dw
                dd += ddis
        return ClosedSim(*(np.array(x) for x in (T, TH, DTH, LAM, PHI, KE, PE, W, D)))

    def inverse_dynamics(self, th, dth, ddth, backend=None):
        """M theta_ddot + h = S tau_a + A^T lambda  =>  [S, A^T][tau_a; lambda] = M theta_ddot + h.

        lambda = force of the cut pin on branch 1 = -lambda of MR Eqs. (8.93)/(8.98)."""
        th, dth, ddth = (np.asarray(x, float) for x in (th, dth, ddth))
        M = self.mass_matrix(th, backend)
        h = self.bias(th, dth, backend)
        _, A, _ = self.constraint(th)
        K = np.c_[self.S, A.T]
        sol = np.linalg.lstsq(K, M @ ddth + h, rcond=None)[0] if K.shape[0] != K.shape[1] \
            else np.linalg.solve(K, M @ ddth + h)
        return sol[:self.na], sol[self.na:], M

    def energy_rate(self, traj_fn, t, delta: float = 1e-5, backend=None) -> np.ndarray:
        """dE/dt by central differences of K + P evaluated on the exact trajectory at t +- delta."""
        t = np.asarray(t, float)
        E = []
        for tt in (t + delta, t - delta):
            TH, DTH, _ = traj_fn(tt)
            E.append(np.array([sum(self.energy(a, b, self.mass_matrix(a, backend))) for a, b in zip(TH, DTH)]))
        return (E[0] - E[1]) / (2 * delta)

    def inverse_dynamics_trajectory(self, t, TH, DTH, DDTH, x=None, xlabel="t [s]", backend=None,
                                    traj_fn=None, cancelled=None, progress=None) -> IDResult:
        """Inverse dynamics along a sampled trajectory.  ``cancelled()`` is polled every sample (raises
        InterruptedError); ``progress(fraction)`` reports the fraction of samples done."""
        N = len(t)
        tau = np.zeros((N, self.na))
        lam = np.zeros((N, 2))
        ke, pe = np.zeros(N), np.zeros(N)
        for k in range(N):
            if cancelled is not None and cancelled():
                raise InterruptedError
            if progress is not None and k % 10 == 0:
                progress(k / N)
            tau[k], lam[k], M = self.inverse_dynamics(TH[k], DTH[k], DDTH[k], backend)
            ke[k], pe[k] = self.energy(TH[k], DTH[k], M)
        power = np.sum(tau * DTH[:, self.actuated], axis=1)
        dEdt = self.energy_rate(traj_fn, t, backend=backend) if traj_fn is not None else np.gradient(ke + pe, t)
        return IDResult(np.asarray(t), np.asarray(t) if x is None else np.asarray(x), xlabel, TH, DTH, DDTH,
                        tau, lam, ke, pe, power, dEdt)


# ============================================================================ four-bar
GRASHOF_TEXT = {
    "crank-rocker": "Grashof crank-rocker: the crank (shortest) turns fully, the rocker oscillates",
    "double-crank": "Grashof double-crank (drag link): ground is shortest, crank and rocker both turn fully",
    "double-rocker": "Grashof double-rocker: coupler is shortest, it turns fully relative to the others",
    "rocker-crank": "Grashof rocker-crank: the output (rocker) is shortest and turns fully; the input rocks",
    "change-point": "Change-point (s + l = p + q): the linkage can fold flat -> C-space singularity (bifurcation)",
    "triple-rocker": "Non-Grashof triple-rocker: no link can make a full turn",
}


class FourBar(ClosedChain):
    """Planar four-bar: O=(0,0), D=(d,0); crank OA (L1), coupler AB (L2), rocker DB (L3).
    Joints: theta1 = crank (actuated), theta2 = coupler relative to crank, theta3 = rocker (absolute)."""
    kind = "fourbar"

    def __init__(self, ground=0.40, crank=0.12, coupler=0.38, rocker=0.28, coupler_point=(0.19, 0.12),
                 density=4.0, backend="mr"):
        self.d, self.L1, self.L2, self.L3 = float(ground), float(crank), float(coupler), float(rocker)
        self.cp = np.asarray(coupler_point, float)
        b1 = Branch("crank + coupler", (0, 0), (crank, coupler), (density * crank, density * coupler))
        b2 = Branch("rocker", (ground, 0), (rocker,), (density * rocker,))
        super().__init__("Four-bar linkage", [b1, b2], actuated=[0], backend=backend)

    def named_points(self, th) -> dict:
        (O, A, B), (D, B2) = self.points(th)
        return dict(O=O, A=A, B=B, D=D, B2=B2, C=self.coupler_point(th))

    def coupler_point(self, th) -> np.ndarray:
        """Point fixed on the coupler: home position (L1 + u, w) in the branch frame, moved by T_2(theta)."""
        b = self.branches[0]
        Ts = b.robot.link_transforms(np.asarray(th, float)[:2])
        return b.pivot + (Ts[2] @ np.r_[self.L1 + self.cp[0], self.cp[1], 0.0, 1.0])[:3]

    def transmission_angle(self, th) -> float:
        """Angle mu between coupler and rocker at B (ideal 90 deg; 0 or 180 deg = actuator singularity)."""
        P = self.named_points(th)
        u, v = P["A"] - P["B"], P["D"] - P["B"]
        return float(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15), -1, 1)))

    def grashof(self):
        links = {"ground": self.d, "crank": self.L1, "coupler": self.L2, "rocker": self.L3}
        v = sorted(links.values())
        s, p, q, l = v
        shortest = min(links, key=links.get)
        if abs(s + l - (p + q)) < 1e-9:
            kind = "change-point"
        elif s + l < p + q:
            kind = {"crank": "crank-rocker", "ground": "double-crank", "coupler": "double-rocker",
                    "rocker": "rocker-crank"}[shortest]
        else:
            kind = "triple-rocker"
        return kind, GRASHOF_TEXT[kind], (s, l, p, q)

    def crank_feasible(self, th1) -> np.ndarray:
        th1 = np.asarray(th1, float)
        AD = np.sqrt(self.L1 ** 2 + self.d ** 2 - 2 * self.L1 * self.d * np.cos(th1))
        return (AD >= abs(self.L2 - self.L3) - 1e-12) & (AD <= self.L2 + self.L3 + 1e-12)

    def crank_fully_rotates(self) -> bool:
        return bool(np.all(self.crank_feasible(np.linspace(-np.pi, np.pi, 2001))))

    def sweep(self, sigma: int, N: int = 721):
        """Closed-form FK over a full crank turn: (theta1 grid, N x 3 thetas with NaN where not assemblable)."""
        th1 = np.linspace(-np.pi, np.pi, N)
        out = np.full((N, 3), np.nan)
        for i, a in enumerate(th1):
            th = self.fk_closed_form([a], sigma)
            if th is not None:
                out[i] = th
        return th1, out

    def coupler_curve(self, sigma: int, N: int = 721) -> list[np.ndarray]:
        """Coupler-point curve(s) for one assembly mode, split where the crank cannot reach."""
        th1, TH = self.sweep(sigma, N)
        segs, cur = [], []
        for th in TH:
            if np.isnan(th[0]):
                if len(cur) > 1:
                    segs.append(np.array(cur))
                cur = []
            else:
                cur.append(self.coupler_point(th))
        if len(cur) > 1:
            segs.append(np.array(cur))
        return segs

    def crank_trajectory(self, omega: float, sigma: int, th1_0: float = 0.0, N: int = 361, t=None):
        """Actuated trajectory for inverse dynamics: constant crank speed if the crank turns fully,
        otherwise a sinusoidal swing over its reachable interval with peak speed omega.
        Returns (t, TH, DTH, DDTH) with the passive joints solved from the loop closure."""
        if self.crank_fully_rotates():
            Tp = 2 * np.pi / abs(omega)
            t = np.linspace(0, Tp, N) if t is None else np.asarray(t, float)
            N = len(t)
            a = th1_0 + omega * t
            da = np.full(N, omega)
            dda = np.zeros(N)
        else:
            grid = np.linspace(-np.pi, np.pi, 3601)
            ok = self.crank_feasible(grid)
            if not ok.any():
                raise ValueError("this linkage cannot be assembled")
            i0 = int(np.argmin(np.abs(wrap(grid - th1_0)) + 1e3 * (~ok)))
            lo = hi = i0
            while lo > 0 and ok[lo - 1]:
                lo -= 1
            while hi < len(grid) - 1 and ok[hi + 1]:
                hi += 1
            margin = np.radians(1.0)
            a_lo, a_hi = grid[lo] + margin, grid[hi] - margin
            mid, amp = (a_lo + a_hi) / 2, max((a_hi - a_lo) / 2, 1e-3)
            W = abs(omega) / amp
            t = np.linspace(0, 2 * np.pi / W, N) if t is None else np.asarray(t, float)
            N = len(t)
            a = mid + amp * np.sin(W * t)
            da = amp * W * np.cos(W * t)
            dda = -amp * W ** 2 * np.sin(W * t)
        TH, DTH, DDTH = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
        for k in range(N):
            th = self.fk_closed_form([a[k]], sigma)
            if th is None:
                raise ValueError("crank trajectory leaves the assemblable range")
            TH[k] = th
            DTH[k], DDTH[k] = self.passive_rates(th, [da[k]], [dda[k]])
        return t, TH, DTH, DDTH


# ============================================================================ five-bar
WORKING_MODES = [((-1, 1), "−/+  elbows out"), ((1, -1), "+/−  elbows in"),
                 ((-1, -1), "−/−  both elbows left"), ((1, 1), "+/+  both elbows right")]


class FiveBar(ClosedChain):
    """Planar five-bar parallel robot: O=(0,0), D=(d,0); left arm O-A-P (L1, L2), right arm D-C-P (L4, L3).
    Joints: theta1 (O, actuated), theta2 (A, relative), theta3 (D, actuated), theta4 (C, relative)."""
    kind = "fivebar"

    def __init__(self, ground=0.20, L1=0.25, L2=0.35, L3=0.35, L4=0.25, density=3.0, backend="mr"):
        self.d, self.L1, self.L2, self.L3, self.L4 = (float(x) for x in (ground, L1, L2, L3, L4))
        b1 = Branch("left arm", (0, 0), (L1, L2), (density * L1, density * L2))
        b2 = Branch("right arm", (ground, 0), (L4, L3), (density * L4, density * L3))
        super().__init__("Five-bar parallel robot", [b1, b2], actuated=[0, 2], backend=backend)

    def named_points(self, th) -> dict:
        (O, A, P), (D, C, P2) = self.points(th)
        return dict(O=O, A=A, P=P, D=D, C=C, P2=P2)

    def ik(self, P, elbows=(-1, 1)):
        """Inverse kinematics: each branch is a planar 2R arm reaching P (elbow sign e_b) -> 4 working modes."""
        th = []
        for b, e in zip(self.branches, elbows):
            r = np.asarray(P, float)[:2] - b.pivot[:2]
            La, Lb = b.L
            c2 = (r @ r - La ** 2 - Lb ** 2) / (2 * La * Lb)
            if abs(c2) > 1 + 1e-12:
                return None
            t2 = e * np.arccos(np.clip(c2, -1, 1))
            t1 = np.arctan2(r[1], r[0]) - np.arctan2(Lb * np.sin(t2), La + Lb * np.cos(t2))
            th += [t1, t2]
        return wrap(np.array(th))

    def elbows(self, th) -> tuple[int, int]:
        return (1 if np.sin(th[1]) >= 0 else -1, 1 if np.sin(th[3]) >= 0 else -1)

    def singularity_measures(self, th):
        """(sin theta2, sin theta4, normalised det A_p): Type I (serial / end-effector singularity, a branch
        stretched or folded) when one of the first two vanishes; Type II (parallel / actuator
        singularity, distal links collinear) when the last vanishes."""
        return float(np.sin(th[1])), float(np.sin(th[3])), self.actuator_singularity(th)

    def annuli(self):
        b1, b2 = self.branches
        return [(b1.pivot[:2], abs(b1.L[0] - b1.L[1]), b1.L[0] + b1.L[1]),
                (b2.pivot[:2], abs(b2.L[0] - b2.L[1]), b2.L[0] + b2.L[1])]

    def in_workspace(self, P) -> np.ndarray:
        P = np.asarray(P, float)
        ok = np.ones(P.shape[:-1], bool)
        for c, rin, rout in self.annuli():
            d = np.linalg.norm(P[..., :2] - c, axis=-1)
            ok &= (d >= rin - 1e-12) & (d <= rout + 1e-12)
        return ok

    def workspace_arcs(self, N: int = 721) -> list[np.ndarray]:
        """Workspace boundary = Type I singularity loci: circles where a branch is fully stretched
        (|P - pivot| = La + Lb) or folded (|La - Lb|), clipped to the other branch's annulus."""
        rings = self.annuli()
        arcs = []
        ang = np.linspace(0, 2 * np.pi, N)
        for i, (c, rin, rout) in enumerate(rings):
            oc, orin, orout = rings[1 - i]
            for r in (rout, rin):
                if r < 1e-6:
                    continue
                pts = c + r * np.c_[np.cos(ang), np.sin(ang)]
                d = np.linalg.norm(pts - oc, axis=1)
                mask = (d >= orin - 1e-9) & (d <= orout + 1e-9)
                cur = []
                for p, ok in zip(pts, mask):
                    if ok:
                        cur.append(p)
                    elif cur:
                        if len(cur) > 1:
                            arcs.append(np.c_[np.array(cur), np.zeros(len(cur))])
                        cur = []
                if len(cur) > 1:
                    arcs.append(np.c_[np.array(cur), np.zeros(len(cur))])
        return arcs

    def bounds(self):
        (c1, _, r1), (c2, _, r2) = self.annuli()
        lo = np.maximum(c1 - r1, c2 - r2)
        hi = np.minimum(c1 + r1, c2 + r2)
        return lo, hi

    def type2_field(self, elbows, n: int = 220):
        """Grid (X, Y, Z) of the Type II measure sin(angle between the distal links) for one working
        mode; NaN outside the workspace (vectorised 2R inverse kinematics)."""
        lo, hi = self.bounds()
        xs = np.linspace(lo[0], hi[0], n)
        ys = np.linspace(lo[1], hi[1], n)
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X, Y], -1)
        valid = np.ones(X.shape, bool)
        elb = []
        for b, e in zip(self.branches, elbows):
            r = P - b.pivot[:2]
            La, Lb = b.L
            c2 = (np.sum(r * r, -1) - La ** 2 - Lb ** 2) / (2 * La * Lb)
            valid &= np.abs(c2) <= 1
            t2 = e * np.arccos(np.clip(c2, -1, 1))
            t1 = np.arctan2(r[..., 1], r[..., 0]) - np.arctan2(Lb * np.sin(t2), La + Lb * np.cos(t2))
            elb.append(b.pivot[:2] + La * np.stack([np.cos(t1), np.sin(t1)], -1))
        Z = _cross2(P - elb[0], P - elb[1]) / (self.L2 * self.L3)
        Z[~valid] = np.nan
        return xs, ys, Z

    def type2_locus(self, elbows, n: int = 220) -> list[np.ndarray]:
        """Type II (parallel) singularity curves = zero level set of the field above (contourpy)."""
        import contourpy
        xs, ys, Z = self.type2_field(elbows, n)
        X, Y = np.meshgrid(xs, ys)
        lines = contourpy.contour_generator(X, Y, np.ma.masked_invalid(Z)).lines(0.0)
        return [np.c_[l, np.zeros(len(l))] for l in lines if len(l) > 1]

    def circle_trajectory(self, center, radius: float, period: float, elbows, N: int = 241, t=None):
        """End-effector P on a circle at constant speed; each branch solved as a 2R arm:
        theta_b = IK(P), theta_b_dot = Jp_b^-1 P_dot, theta_b_ddot = Jp_b^-1 (P_ddot - c_b)."""
        w = 2 * np.pi / period
        t = np.linspace(0, period, N) if t is None else np.asarray(t, float)
        N = len(t)
        TH, DTH, DDTH = np.zeros((N, 4)), np.zeros((N, 4)), np.zeros((N, 4))
        c = np.asarray(center, float)[:2]
        for k, tk in enumerate(t):
            P = c + radius * np.array([np.cos(w * tk), np.sin(w * tk)])
            Pd = radius * w * np.array([-np.sin(w * tk), np.cos(w * tk)])
            Pdd = -radius * w ** 2 * np.array([np.cos(w * tk), np.sin(w * tk)])
            th = self.ik(P, elbows)
            if th is None:
                raise ValueError("circle leaves the workspace")
            TH[k] = th
            for b, s in zip(self.branches, self.sl):
                _, Jp, _ = b.kinematics(th[s])
                dth_b = np.linalg.solve(Jp[:2], Pd)
                _, _, cb = b.kinematics(th[s], dth_b)
                DTH[k, s] = dth_b
                DDTH[k, s] = np.linalg.solve(Jp[:2], Pdd - cb[:2])
        return t, TH, DTH, DDTH


# ============================================================================ validation
def _sim_backend():
    from .dynamics import pin
    return "pinocchio" if pin is not None else "mr"


def validation_checks() -> list[Check]:
    rng = np.random.default_rng(7)
    out: list[Check] = []
    fb = FourBar(backend="mr")
    fv = FiveBar(backend="mr")

    # 1. closed-form vs Newton-Raphson forward kinematics ------------------------------------
    err4, its = 0.0, []
    for sigma in (1, -1):
        for a in np.linspace(-np.pi, np.pi, 25):
            th = fb.fk_closed_form([a], sigma)
            guess = th + 0.2 * rng.standard_normal(3)
            th_nr, ok, hist = fb.fk_newton([a], guess)
            err4 = max(err4, float(np.max(np.abs(wrap(th_nr - th)))) if ok else 1.0)
            its.append(len(hist))
    out.append(Check("Four-bar FK: closed form (circle-circle) vs Newton-Raphson", "MR 7.1, 6.2", err4, 1e-9,
                     f"50 crank angles x 2 assembly modes, NR from perturbed guesses ({max(its)} its max)",
                     CATEGORY))
    err5 = 0.0
    for _ in range(40):
        P = np.array([fv.d / 2, 0.0]) + rng.uniform([-0.15, 0.25], [0.15, 0.45])
        el = WORKING_MODES[rng.integers(4)][0]
        th = fv.ik(P, el)
        if th is None:
            continue
        sigma = fv.assembly_mode(th)
        th_cf = fv.fk_closed_form(th[fv.actuated], sigma)
        th_nr, ok, _ = fv.fk_newton(th[fv.actuated], th_cf + 0.15 * rng.standard_normal(4))
        err5 = max(err5, float(np.max(np.abs(wrap(th_nr - th_cf)))) if ok else 1.0)
    out.append(Check("Five-bar FK: closed form vs Newton-Raphson", "MR 7.1, 6.2", err5, 1e-9,
                     "40 random end-effector points, all working modes", CATEGORY))

    # 2. five-bar FK(IK(P)) round trip ----------------------------------------------------------
    err = 0.0
    for _ in range(60):
        P = np.array([fv.d / 2, 0.0]) + rng.uniform([-0.25, 0.1], [0.25, 0.55])
        for el, _ in WORKING_MODES:
            th = fv.ik(P, el)
            if th is None:
                continue
            sols = [fv.fk_closed_form(th[fv.actuated], s) for s in (1, -1)]
            err = max(err, min(np.linalg.norm(fv.tip(s)[:2] - P) for s in sols if s is not None))
    out.append(Check("Five-bar FK(IK(P)) round trip (4 working modes x 2 assembly modes)", "MR 7.1",
                     err, 1e-12, "min over the two assembly modes of |P_FK - P|", CATEGORY))

    # 3. constraint Jacobian and Adot*theta_dot vs finite differences ----------------------------
    errA = errC = 0.0
    eps = 1e-6
    for mech in (fb, fv):
        for _ in range(5):
            th = rng.uniform(-np.pi, np.pi, mech.n)
            dth = rng.standard_normal(mech.n)
            _, A, adq = mech.constraint(th, dth)
            Afd = np.column_stack([(mech.constraint(th + eps * e)[0] - mech.constraint(th - eps * e)[0]) / (2 * eps)
                                   for e in np.eye(mech.n)])
            Ap = mech.constraint(th + eps * dth)[1]
            Am = mech.constraint(th - eps * dth)[1]
            errA = max(errA, float(np.abs(A - Afd).max()))
            errC = max(errC, float(np.abs((Ap - Am) / (2 * eps) @ dth - adq).max()))
    out.append(Check("Constraint Jacobian A = dphi/dtheta (PoE) vs finite differences", "MR 7.2, 5.1",
                     errA, 1e-7, "four-bar and five-bar, random configurations", CATEGORY))
    out.append(Check("Adot*theta_dot (ad-operator Jacobian derivative) vs finite differences", "MR 8.7, 8.2",
                     errC, 1e-6, "dJ_si/dt = [ad_{V_i-1}] J_si", CATEGORY))

    # 4. MR vs Pinocchio on the KKT solution -----------------------------------------------------
    be = _sim_backend()
    if be == "pinocchio":
        e = 0.0
        for mech in (fb, fv):
            for _ in range(4):
                th = rng.uniform(-np.pi, np.pi, mech.n)
                dth = rng.standard_normal(mech.n)
                tau = rng.standard_normal(mech.n)
                a1, l1 = mech.forward_dynamics(th, dth, tau, 5, 5, backend="mr")
                a2, l2 = mech.forward_dynamics(th, dth, tau, 5, 5, backend="pinocchio")
                e = max(e, float(np.abs(a1 - a2).max()), float(np.abs(l1 - l2).max()))
        out.append(Check("Constrained forward dynamics (KKT): Modern Robotics vs Pinocchio blocks", "MR 8.7",
                         e, 1e-9, "mr.MassMatrix/VelQuadraticForces/GravityForces vs CRBA/RNEA", CATEGORY))

    # 5. forward / inverse dynamics consistency ---------------------------------------------------
    e = 0.0
    for mech in (fb, fv):
        for _ in range(4):
            if mech is fb:
                th = fb.fk_closed_form([rng.uniform(-np.pi, np.pi)], 1)
            else:
                th = fv.ik(np.array([0.1, 0.38]) + rng.uniform(-0.08, 0.08, 2), (-1, 1))
            dth, ddth = mech.passive_rates(th, rng.standard_normal(mech.na), rng.standard_normal(mech.na))
            tau_a, lam, _ = mech.inverse_dynamics(th, dth, ddth)
            dd2, lam2 = mech.forward_dynamics(th, dth, mech.S @ tau_a)
            e = max(e, float(np.abs(dd2 - ddth).max()), float(np.abs(lam2 - lam).max()))
    out.append(Check("Inverse dynamics [S, A^T][tau; lambda] -> forward dynamics (KKT) round trip", "MR 8.7",
                     e, 1e-8, "theta_ddot and pin force lambda recovered", CATEGORY))

    # 6. free swing of the unactuated four-bar: energy & constraint drift -------------------------
    th0 = fb.fk_closed_form([np.radians(60)], 1)
    sim = fb.simulate(th0, np.zeros(3), lambda t, q, dq: np.zeros(3), 1.5, dt=0.002, alpha=10, beta=10,
                      backend=be)
    E = sim.ke + sim.pe
    scale = max(float(np.ptp(sim.pe)), 1e-9)
    out.append(Check("Energy conservation: unactuated four-bar swinging under gravity (RK4)", "MR 8.7, 8.1",
                     float(np.max(np.abs(E - E[0])) / scale), 1e-5,
                     f"max |E(t)-E(0)| / PE swing over 1.5 s, dt = 2 ms ({be} blocks)", CATEGORY))
    out.append(Check("Constraint drift ||phi|| during free simulation (Baumgarte a=b=10)", "MR 8.7",
                     float(sim.phi.max()), 5e-8, "max loop-closure error over 1.5 s [m] (without stabilisation: ~2e-7)", CATEGORY))

    # 7. power balance along an inverse-dynamics trajectory ---------------------------------------
    t, TH, DTH, DDTH = fb.crank_trajectory(2 * np.pi, 1, N=13)
    e = 0.0
    pmax = 1e-9
    dt = 1e-5
    for k in range(len(t) - 1):
        tau_a, _, M = fb.inverse_dynamics(TH[k], DTH[k], DDTH[k])
        P = float(tau_a @ DTH[k, fb.actuated])
        Es = []
        for s in (-1, 1):
            a = TH[k, 0] + s * dt * DTH[k, 0]
            th = fb.fk_closed_form([a], 1)
            dth = fb.passive_rates(th, [DTH[k, 0]])
            Es.append(sum(fb.energy(th, dth)))
        e = max(e, abs(P - (Es[1] - Es[0]) / (2 * dt)))
        pmax = max(pmax, abs(P))
    out.append(Check("Power balance tau_a.theta_a_dot = dE/dt (constant-speed crank, inverse dynamics)",
                     "MR 8.7, 8.1.2", e / pmax, 1e-6,
                     "constraint forces do no work: lambda^T A theta_dot = 0 (relative error)", CATEGORY))
    return out
