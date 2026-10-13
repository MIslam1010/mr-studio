"""3xRPR planar parallel mechanism -- Modern Robotics Sec. 7.1.1 (Fig. 7.2, Eqs. 7.1-7.2), 7.2, 7.3.

Three legs, each Revolute (passive, on the base at a_i) - Prismatic (actuated, length s_i) - Revolute
(passive, on the platform at b_i).  Platform pose q = (p_x, p_y, phi), T_sb in SE(2).

    d_i = p + R(phi) b_i - a_i,      s_i = |d_i|                    (7.1), (7.2)  inverse kinematics

Forward kinematics (s -> q).  For a fixed phi, subtracting equation 1 from equations 2 and 3 of (7.2)
gives two equations *linear* in p:

    2 (c_i - c_1)^T p = s_i^2 - s_1^2 - |c_i|^2 + |c_1|^2,      c_i(phi) = R(phi) b_i - a_i,

so p = p(phi) = adj(D) r / det D, and the first equation leaves one scalar equation in phi.  Multiplying
by (det D)^2 removes the poles:  g(phi) = |adj(D) r + det(D) c_1|^2 - s_1^2 det(D)^2 = 0, a trigonometric
polynomial.  Its real roots are bracketed on a fine grid and refined with Brent's method, then
polished by Newton-Raphson on the three equations (7.2).  MR (7.1.1) shows, via t = tan(phi/2), that the
system reduces to a sextic: at most six real assembly modes.

Velocity kinematics (7.2): with the planar spatial twist V_s = (w, v_x, v_y) and x_i = p + R b_i,
    s_i_dot = u_i^T (v + w k x x_i) = (x_i x u_i)_z w + u_i^T v   ->   s_dot = J(q) V_s,  row i = [(x_i x u_i)_z, u_i^T]
(the planar analogue of the Stewart J_s^-1 of 7.2.1).  Statics: F_s = J^T f.
Actuator singularity (7.3): det J = 0 <=> the three leg lines are concurrent (or all parallel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .checks import Check

CATEGORY = "Parallel mechanisms"


def rot2(phi) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class RPRSolutions:
    q: list            # list of (px, py, phi)
    g_phi: np.ndarray  # grid of phi used for bracketing
    g_val: np.ndarray  # normalised g(phi) on the grid (for plotting)


class ThreeRPR:
    """3xRPR planar parallel robot (MR Fig. 7.2) with a non-similar base and platform triangle."""

    def __init__(self, a=None, b=None, s_min=0.20, s_max=1.20):
        self.a = np.array(a if a is not None else [[-0.50, -0.30], [0.55, -0.30], [0.05, 0.55]], float)
        self.b = np.array(b if b is not None else [[0.391, 0.107], [-0.042, -0.343], [-0.350, 0.236]], float)
        self.s_min, self.s_max = s_min, s_max
        self.q_home = np.array([0.15, -0.35, np.radians(30)])   # well-conditioned (|det J_n| ~ 1.5)

    # -- IK (7.1), (7.2) --------------------------------------------------------------------------
    def platform_points(self, q) -> np.ndarray:
        return (rot2(q[2]) @ self.b.T).T + np.asarray(q[:2], float)

    def legs(self, q):
        x = self.platform_points(q)
        d = x - self.a
        s = np.linalg.norm(d, axis=1)
        return x, s, d / s[:, None]

    def ik(self, q) -> np.ndarray:
        return self.legs(q)[1]

    # -- Jacobians (7.2) ------------------------------------------------------------------------------
    def jacobian(self, q) -> np.ndarray:
        """s_dot = J V_s, V_s = (w, v_x, v_y) planar spatial twist; row i = [(x_i x u_i)_z, u_i^T]."""
        x, _, u = self.legs(q)
        return np.c_[_cross2(x, u), u]

    def jacobian_q(self, q) -> np.ndarray:
        """s_dot = J_q q_dot with q_dot = (px_dot, py_dot, phi_dot); row i = [u_i^T, ((R b_i) x u_i)_z]."""
        x, _, u = self.legs(q)
        rb = x - np.asarray(q[:2])
        return np.c_[u, _cross2(rb, u)]

    def singularity(self, q):
        """(det J_n, cond J_n) with the angular column scaled by the platform size (dimensionless)."""
        J = self.jacobian(q).copy()
        L = float(np.linalg.norm(self.b, axis=1).mean())
        J[:, 0] /= L
        sv = np.linalg.svd(J, compute_uv=False)
        return float(np.linalg.det(J)), float(sv[0] / max(sv[-1], 1e-300))

    # -- FK: all real solutions -------------------------------------------------------------------------
    def _lin(self, phi, s):
        """D(phi) p = r(phi): the two equations linear in p.  Returns (D, r, c) for scalar phi."""
        c = (rot2(phi) @ self.b.T).T - self.a
        D = 2 * np.array([c[1] - c[0], c[2] - c[0]])
        r = np.array([s[1] ** 2 - s[0] ** 2 - c[1] @ c[1] + c[0] @ c[0],
                      s[2] ** 2 - s[0] ** 2 - c[2] @ c[2] + c[0] @ c[0]])
        return D, r, c

    def g(self, phi, s) -> float:
        """g(phi) = |adj(D) r + det(D) c_1|^2 - s_1^2 det(D)^2 (pole-free univariate FK equation)."""
        D, r, c = self._lin(phi, s)
        det = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]
        adj = np.array([[D[1, 1], -D[0, 1]], [-D[1, 0], D[0, 0]]])
        w = adj @ r + det * c[0]
        return float(w @ w - s[0] ** 2 * det ** 2)

    def p_of_phi(self, phi, s):
        D, r, _ = self._lin(phi, s)
        return np.linalg.solve(D, r)

    def newton(self, s, q0, tol=1e-13, max_iter=30):
        q = np.asarray(q0, float).copy()
        for _ in range(max_iter):
            e = np.asarray(s, float) - self.ik(q)
            if np.linalg.norm(e) < tol:
                return q, True
            try:
                q = q + np.linalg.solve(self.jacobian_q(q), e)
            except np.linalg.LinAlgError:
                return q, False
        return q, bool(np.linalg.norm(np.asarray(s) - self.ik(q)) < 1e-10)

    def fk_all(self, s, n_grid=1441) -> RPRSolutions:
        s = np.asarray(s, float)
        ph = np.linspace(-np.pi, np.pi, n_grid)
        gv = np.array([self.g(x, s) for x in ph])
        scale = max(float(np.abs(gv).max()), 1e-300)
        roots = []
        for k in range(n_grid - 1):
            g0, g1 = gv[k], gv[k + 1]
            if g0 == 0.0:
                roots.append(ph[k])
            elif g0 * g1 < 0:
                roots.append(brentq(self.g, ph[k], ph[k + 1], args=(s,), xtol=1e-14))
            elif k > 0 and abs(g0) < 1e-6 * scale and abs(g0) <= abs(gv[k - 1]) and abs(g0) <= abs(g1):
                roots.append(ph[k])      # tangent (double) root candidate -> polished below
        sols = []
        for phi in roots:
            try:
                p = self.p_of_phi(phi, s)
            except np.linalg.LinAlgError:
                continue
            q, ok = self.newton(s, np.r_[p, phi])
            q[2] = (q[2] + np.pi) % (2 * np.pi) - np.pi
            if ok and np.abs(self.ik(q) - s).max() < 1e-9 and \
                    all(np.abs(np.r_[q[:2] - o[:2], np.angle(np.exp(1j * (q[2] - o[2])))]).max() > 1e-7
                        for o in sols):
                sols.append(q)
        return RPRSolutions(sols, ph, gv / scale)

    def fk_nearest(self, s, q_prev):
        """FK solution closest to the previous pose (continuity while dragging the leg sliders)."""
        res = self.fk_all(s)
        if not res.q:
            return None, res
        d = [np.linalg.norm(np.r_[q[:2] - q_prev[:2], 0.3 * np.angle(np.exp(1j * (q[2] - q_prev[2])))])
             for q in res.q]
        return res.q[int(np.argmin(d))], res

    # -- workspace / singular locus at constant orientation -------------------------------------------
    def bounds(self):
        lo, hi = self.a.min(0) - 0.35, self.a.max(0) + 0.35
        return lo, hi

    def workspace_grid(self, phi, n=110):
        """Constant-orientation workspace: grid points p with s_min <= s_i <= s_max, plus det J on the grid."""
        lo, hi = self.bounds()
        X, Y = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
        P = np.c_[X.ravel(), Y.ravel()]
        rb = (rot2(phi) @ self.b.T).T
        x = P[:, None, :] + rb[None]                       # (N, 3, 2)
        d = x - self.a[None]
        s = np.linalg.norm(d, axis=2)
        u = d / s[..., None]
        inside = np.all((s >= self.s_min) & (s <= self.s_max), axis=1)
        J = np.concatenate([_cross2(x, u)[..., None], u], axis=2)   # (N, 3, 3)
        det = np.linalg.det(J).reshape(n, n)
        return X, Y, inside.reshape(n, n), det

    def singular_locus(self, phi, n=110):
        """Points where det J changes sign on the grid (actuator singularity curve, 7.3)."""
        X, Y, inside, det = self.workspace_grid(phi, n)
        pts = []
        sg = np.sign(det)
        for a_, b_ in ((sg[:, 1:] != sg[:, :-1], (slice(None), slice(1, None))),
                       (sg[1:, :] != sg[:-1, :], (slice(1, None), slice(None)))):
            m = a_ & inside[b_]
            pts.append(np.c_[X[b_][m], Y[b_][m]])
        return np.vstack(pts) if pts else np.zeros((0, 2))

    # -- a demo trajectory -------------------------------------------------------------------------------
    def circle_path(self, n=181, radius=0.07, period=4.0, swing=np.radians(15)):
        """Platform centre on a circle around home while the platform rocks by +-swing."""
        t = np.linspace(0, period, n)
        w = 2 * np.pi / period
        q0 = self.q_home
        Q = np.c_[q0[0] + radius * np.cos(w * t), q0[1] + radius * np.sin(w * t), q0[2] + swing * np.sin(2 * w * t)]
        return t, Q


# ============================================================================ validation
def validation_checks() -> list[Check]:
    rob = ThreeRPR()
    rng = np.random.default_rng(7)
    out: list[Check] = []

    def rand_q():
        return rob.q_home + np.r_[rng.uniform(-0.12, 0.12, 2), np.radians(rng.uniform(-30, 30))]

    # 1. FK(IK(q)): the true pose is among the FK solutions
    err, nsol = 0.0, []
    for _ in range(12):
        q = rand_q()
        res = rob.fk_all(rob.ik(q))
        nsol.append(len(res.q))
        e = min((np.abs(np.r_[x[:2] - q[:2], np.angle(np.exp(1j * (x[2] - q[2])))]).max() for x in res.q),
                default=1.0)
        err = max(err, e)
    out.append(Check("3xRPR FK(IK(q)) round trip: true pose among the univariate-polynomial roots",
                     "MR 7.1.1, Eq. 7.2", err, 1e-9,
                     f"12 random poses; real assembly modes found: {min(nsol)}-{max(nsol)} (<= 6)", CATEGORY))
    # 2. every FK solution reproduces s, at most 6
    s6 = SIX_SOLUTION_LEGS
    res = rob.fk_all(s6)
    r = max(float(np.abs(rob.ik(q) - s6).max()) for q in res.q)
    out.append(Check(f"3xRPR FK: {len(res.q)} real solutions for the demo leg lengths (MR: at most 6)",
                     "MR 7.1.1", r if 1 <= len(res.q) <= 6 else 1.0, 1e-9,
                     "every assembly mode satisfies (7.2); a sextic in tan(phi/2) bounds the count", CATEGORY))
    # 3. Jacobians vs finite differences
    eJ = 0.0
    for _ in range(10):
        q, dq, h = rand_q(), rng.standard_normal(3), 1e-6
        fd = (rob.ik(q + h * dq) - rob.ik(q - h * dq)) / (2 * h)
        eJ = max(eJ, float(np.abs(rob.jacobian_q(q) @ dq - fd).max()))
        # twist form: V_s = (w, v), v = p_dot - w k x p  (planar: k x p = (-p_y, p_x))
        w = dq[2]
        V = np.r_[w, dq[0] + w * q[1], dq[1] - w * q[0]]
        eJ = max(eJ, float(np.abs(rob.jacobian(q) @ V - fd).max()))
    out.append(Check("3xRPR Jacobian s_dot = J V_s (rows [(x_i x u_i)_z, u_i^T]) vs finite differences",
                     "MR 7.2", eJ, 1e-8, "both the q_dot form and the spatial-twist form", CATEGORY))
    # 4. singularity: symmetric singular pose of MR Fig. 7.3(a) analogue -> concurrent legs
    sym = ThreeRPR(a=[[0.5 * np.cos(t), 0.5 * np.sin(t)] for t in np.radians([90, 210, 330])],
                   b=[[0.2 * np.cos(t), 0.2 * np.sin(t)] for t in np.radians([90, 210, 330])])
    det_sing = abs(sym.singularity(np.array([0.0, 0.0, 0.0]))[0])
    out.append(Check("3xRPR actuator singularity: concurrent leg lines -> det J = 0 (cf. MR Fig. 7.3a)",
                     "MR 7.3", det_sing, 1e-12, "similar triangles, phi = 0: all legs radial through the centre",
                     CATEGORY))
    return out


# leg lengths with 6 real FK solutions for the default geometry (found by a random search; verified in checks)
SIX_SOLUTION_LEGS = np.array([0.786, 1.011, 0.725])
