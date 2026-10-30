"""Delta robot (Clavel) -- Modern Robotics Example 2.7 (Fig. 2.8), Fig. 7.1(b), Exercise 7.7, Sec. 7.3.

Three legs at angles alpha_i = 0, 120, 240 deg about the vertical axis.  Leg i: an actuated revolute
joint at B_i = R e_r,i (axis along e_t,i), an upper arm of length L, and a parallelogram forearm of
length l whose four spherical joints keep the moving platform parallel to the base (Example 2.7: the
platform is a pure x-y-z translator; Gruebler gives 15 dof of which 12 are internal forearm torsions).

    elbow        A_i = B_i + L (cos th_i e_r,i - sin th_i e_z)       (th_i > 0: arm below horizontal)
    platform pt  P_i = p + r e_r,i
    closure      |P_i - A_i| = l                                       i = 1, 2, 3

Inverse kinematics (closed form, per leg): with a = p.e_r + r - R, b = p.e_t, c = p_z the closure reads
E cos th + F sin th + G = 0, E = -2aL, F = 2cL, G = a^2 + b^2 + c^2 + L^2 - l^2, solved by the tangent
half-angle t = tan(th/2):  (G - E) t^2 + 2F t + (G + E) = 0.  The two roots are mirror images of the elbow
about the line B_i -> P_i - r e_r,i; the 'elbow-out' root is used (2^3 = 8 IK branches in total).

Forward kinematics (closed form): p lies on the three spheres |p - C_i| = l, C_i = A_i - r e_r,i
(trilateration) -> two mirror solutions about the plane through C_1, C_2, C_3; the one below is used.

Velocity kinematics: differentiating |P_i - A_i|^2 = l^2 with n_i = P_i - A_i gives
    J_x p_dot = J_th th_dot,   J_x = [n_i^T] (3x3),   J_th = diag(n_i . dA_i/dth_i)
    p_dot = J th_dot,  J = J_x^-1 J_th.      Statics (power balance tau^T th_dot = F^T p_dot): tau = J^T F.
Singularities (7.3): det J_x = 0 -> actuator singularity (platform moves with locked motors);
n_i . dA_i/dth_i = 0 (arm and forearm coplanar-aligned) -> end-effector singularity, the workspace boundary.
"""
from __future__ import annotations

import numpy as np

from .checks import Check

CATEGORY = "Parallel mechanisms"
G_ACC = 9.81


class DeltaRobot:
    def __init__(self, R=0.20, r=0.05, L=0.20, l=0.46, half_width=0.035,
                 th_min=np.radians(-35), th_max=np.radians(95)):
        self.R, self.r, self.L, self.l, self.w = R, r, L, l, half_width
        self.th_min, self.th_max = th_min, th_max
        al = np.radians([0.0, 120.0, 240.0])
        self.alpha = al
        self.er = np.c_[np.cos(al), np.sin(al), np.zeros(3)]
        self.et = np.c_[-np.sin(al), np.cos(al), np.zeros(3)]
        self.ez = np.array([0.0, 0.0, 1.0])
        self.B = R * self.er
        self.p_home = np.array([0.0, 0.0, -0.38])

    # -- geometry ---------------------------------------------------------------------------------------
    def elbows(self, th) -> np.ndarray:
        th = np.asarray(th, float)
        return self.B + self.L * (np.cos(th)[:, None] * self.er - np.sin(th)[:, None] * self.ez)

    def elbow_derivs(self, th) -> np.ndarray:
        th = np.asarray(th, float)
        return self.L * (-np.sin(th)[:, None] * self.er - np.cos(th)[:, None] * self.ez)

    def platform_points(self, p) -> np.ndarray:
        return np.asarray(p, float) + self.r * self.er

    def closure(self, th, p) -> np.ndarray:
        return np.linalg.norm(self.platform_points(p) - self.elbows(th), axis=1) - self.l

    # -- inverse kinematics (closed form) ----------------------------------------------------------------
    def ik_leg(self, p, i):
        """Both roots (elbow-out, elbow-in) of leg i, or None if p is out of reach for this leg."""
        p = np.asarray(p, float)
        a = p @ self.er[i] + self.r - self.R
        b = p @ self.et[i]
        c = p[2]
        L, l = self.L, self.l
        E, F, G = -2 * a * L, 2 * c * L, a * a + b * b + c * c + L * L - l * l
        disc = E * E + F * F - G * G
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        if abs(G - E) < 1e-14:                               # one root at t = inf (th = pi)
            roots = [np.pi, 2 * np.arctan2(-(G + E), 2 * F)]
        else:
            roots = [2 * np.arctan((-F + sq) / (G - E)), 2 * np.arctan((-F - sq) / (G - E))]
        # elbow-out: elbow on the +e_r side of the line from B_i to the shifted platform point (a, c)
        def side(th):
            return a * (-np.sin(th)) - c * np.cos(th)        # 2-D cross product (a, c) x (cos, -sin)
        roots.sort(key=lambda th: -side(th))
        return roots[0], roots[1]

    def ik(self, p, branch=(0, 0, 0)):
        """Actuator angles th (rad) for platform position p; None if unreachable."""
        th = []
        for i in range(3):
            rt = self.ik_leg(p, i)
            if rt is None:
                return None
            th.append(rt[branch[i]])
        return np.array(th)

    # -- forward kinematics (closed form trilateration) ---------------------------------------------------
    def fk_all(self, th):
        """Both intersection points of the spheres |p - C_i| = l (lower first); [] if none."""
        C = self.elbows(th) - self.r * self.er
        l = self.l
        ex = C[1] - C[0]
        d = np.linalg.norm(ex)
        if d < 1e-12:
            return []
        ex /= d
        i = ex @ (C[2] - C[0])
        ey = C[2] - C[0] - i * ex
        jn = np.linalg.norm(ey)
        if jn < 1e-12:
            return []
        ey /= jn
        ez = np.cross(ex, ey)
        x = d / 2.0                                          # equal radii: (l^2 - l^2 + d^2) / 2d
        y = (i * i + jn * jn - 2 * i * x) / (2 * jn)
        z2 = l * l - x * x - y * y
        if z2 < -1e-12:
            return []
        z = np.sqrt(max(z2, 0.0))
        base = C[0] + x * ex + y * ey
        sols = [base + z * ez, base - z * ez]
        sols.sort(key=lambda q: q[2])
        return sols

    def fk(self, th):
        s = self.fk_all(th)
        return s[0] if s else None

    def fk_newton(self, th, p0, tol=1e-13, max_iter=30):
        """Newton-Raphson on the closure equations (independent cross-check of the trilateration)."""
        p = np.asarray(p0, float).copy()
        A = self.elbows(th)
        for _ in range(max_iter):
            n = self.platform_points(p) - A
            e = 0.5 * (np.sum(n * n, axis=1) - self.l ** 2)
            if np.abs(e).max() < tol:
                break
            p = p - np.linalg.solve(n, e)
        return p

    # -- Jacobian, statics, singularities -------------------------------------------------------------------
    def jacobian_parts(self, th, p):
        n = self.platform_points(p) - self.elbows(th)
        Jth = np.diag(np.sum(n * self.elbow_derivs(th), axis=1))
        return n, Jth

    def jacobian(self, th, p=None) -> np.ndarray:
        """p_dot = J th_dot with J = J_x^-1 J_th."""
        p = self.fk(th) if p is None else p
        Jx, Jth = self.jacobian_parts(th, p)
        return np.linalg.solve(Jx, Jth)

    def singularity(self, th, p):
        """(normalised det J_x, min_i |cos| between n_i and dA_i, cond J)."""
        n, Jth = self.jacobian_parts(th, p)
        nn = n / np.linalg.norm(n, axis=1)[:, None]
        dx = float(np.linalg.det(nn))
        da = self.elbow_derivs(th) / self.L
        t1 = float(np.min(np.abs(np.sum(nn * da, axis=1))))
        try:
            sv = np.linalg.svd(np.linalg.solve(n, Jth), compute_uv=False)
            cond = float(sv[0] / max(sv[-1], 1e-300))
        except np.linalg.LinAlgError:
            cond = np.inf
        return dx, t1, cond

    def static_torques(self, th, p, F) -> np.ndarray:
        """Motor torques balancing an external force F on the platform: tau = -J^T F (massless legs)."""
        return -self.jacobian(th, p).T @ np.asarray(F, float)

    def static_torques_checked(self, th, p, F):
        """:meth:`static_torques` plus the condition number of the (row-normalised) J_x that is inverted in
        J = J_x^-1 J_th, and a warning when kappa > 1e6 (Type-II singularity: torques unbounded).
        Returns a :class:`mrstudio.parallel.Checked`."""
        from .parallel import Checked, _cond, condition_warning
        n, _ = self.jacobian_parts(th, p)
        kappa = _cond(n / np.linalg.norm(n, axis=1)[:, None])
        try:
            tau = self.static_torques(th, p, F)
        except np.linalg.LinAlgError:
            tau, kappa = np.full(3, np.nan), np.inf
        return Checked(tau, kappa, condition_warning(kappa, "motor torques"))

    # -- workspace ------------------------------------------------------------------------------------------
    def workspace_cloud(self, n=17, cancelled=None, progress=None):
        """FK of a grid of actuator angles within the limits; returns (points, cond J).
        ``cancelled()`` is polled per grid slab (raises InterruptedError); ``progress(fraction)``."""
        g = np.linspace(self.th_min, self.th_max, n)
        P, K = [], []
        for i1, t1 in enumerate(g):
            if cancelled is not None and cancelled():
                raise InterruptedError
            if progress is not None:
                progress(i1 / n)
            for t2 in g:
                for t3 in g:
                    th = np.array([t1, t2, t3])
                    p = self.fk(th)
                    if p is None:
                        continue
                    # keep only elbow-out assemblies (the IK branch actually used)
                    th2 = self.ik(p)
                    if th2 is None or np.abs(th2 - th).max() > 1e-6:
                        continue
                    P.append(p)
                    K.append(self.singularity(th, p)[2])
        return np.array(P), np.array(K)

    # -- pick-and-place cycle ---------------------------------------------------------------------------------
    def pick_place_path(self, span=0.24, lift=0.07, seg=0.32, n_seg=24, z=None):
        """'Adept-cycle'-like path: up, across, down, and back; quintic time scaling on each straight segment
        (MR 9.2).  Returns (t, P, P_dot)."""
        z = self.p_home[2] - 0.03 if z is None else z
        u = np.array([np.cos(np.radians(25)), np.sin(np.radians(25)), 0.0])   # oblique: all 3 legs differ
        A, B = np.r_[0.0, 0.0, z] - span / 2 * u, np.r_[0.0, 0.0, z] + span / 2 * u
        way = [A, A + [0, 0, lift], B + [0, 0, lift], B, B + [0, 0, lift], A + [0, 0, lift], A]
        durs = [seg, 1.6 * seg, seg, seg, 1.6 * seg, seg]
        ts, Ps, Vs = [], [], []
        t0 = 0.0
        for k in range(len(way) - 1):
            Tf = durs[k]
            tau = np.linspace(0, 1, n_seg + 1)[(1 if k else 0):]
            s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            sd = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / Tf
            d = way[k + 1] - way[k]
            ts.append(t0 + tau * Tf)
            Ps.append(way[k] + s[:, None] * d)
            Vs.append(sd[:, None] * d)
            t0 += Tf
        return np.concatenate(ts), np.vstack(Ps), np.vstack(Vs)


# ============================================================================ validation
def validation_checks() -> list[Check]:
    d = DeltaRobot()
    rng = np.random.default_rng(11)
    out: list[Check] = []

    def rand_p():
        while True:
            p = d.p_home + rng.uniform([-0.12, -0.12, -0.08], [0.12, 0.12, 0.08])
            th = d.ik(p)
            if th is not None and np.all(th > d.th_min) and np.all(th < d.th_max):
                return p, th

    # 1. FK(IK(p))
    e1 = e2 = e3 = 0.0
    for _ in range(30):
        p, th = rand_p()
        e1 = max(e1, float(np.abs(d.fk(th) - p).max()))
        e2 = max(e2, float(np.abs(d.closure(th, p)).max()))
        e3 = max(e3, float(np.abs(d.fk_newton(th, d.p_home) - p).max()))
    out.append(Check("Delta FK(IK(p)) round trip: closed-form IK (tan half-angle) + sphere trilateration",
                     "MR Ex. 7.7(a,b)", e1, 1e-10, "30 random positions; lower assembly mode", CATEGORY))
    out.append(Check("Delta loop closure |P_i - A_i| = l at the IK solution", "MR 7.1, Ex. 2.7",
                     e2, 1e-12, "all three parallelogram forearms", CATEGORY))
    out.append(Check("Delta FK: trilateration vs Newton-Raphson on the closure equations", "MR 6.2, Ex. 7.7(a)",
                     e3, 1e-10, "Newton started at the home position", CATEGORY))
    # 2. Jacobian vs finite differences of FK
    eJ = eP = 0.0
    for _ in range(10):
        p, th = rand_p()
        dth, h = rng.standard_normal(3), 1e-6
        fd = (d.fk(th + h * dth) - d.fk(th - h * dth)) / (2 * h)
        J = d.jacobian(th, p)
        eJ = max(eJ, float(np.abs(J @ dth - fd).max()))
        F = rng.standard_normal(3) * 20
        tau = J.T @ F
        eP = max(eP, abs(float(tau @ dth - F @ fd)))
    out.append(Check("Delta Jacobian p_dot = J_x^-1 J_th th_dot vs finite differences of FK",
                     "MR Ex. 7.7(c), 7.2", eJ, 1e-8, "", CATEGORY))
    out.append(Check("Delta statics power balance tau^T th_dot = F^T p_dot with tau = J^T F",
                     "MR 7.2.1, 3.4", eP, 1e-7, "p_dot by finite differences (independent of J)", CATEGORY))
    # 3. both FK roots satisfy the closure; mirror symmetry
    p, th = rand_p()
    sols = d.fk_all(th)
    eM = max(float(np.abs(d.closure(th, q)).max()) for q in sols) if len(sols) == 2 else 1.0
    out.append(Check("Delta FK: both sphere-intersection assembly modes satisfy the loop closure",
                     "MR Ex. 7.7(a)", eM, 1e-12, "upper solution = mirror image through the plane of C_1, C_2, C_3",
                     CATEGORY))
    # 4. actuator singularity: forearms of the three legs all horizontal-parallel to the base -> det J_x = 0
    #    (place p on the plane of the elbow centres C_i: the two FK roots merge, z = 0 in trilateration)
    th = np.radians([30.0, 30.0, 30.0])
    C = d.elbows(th) - d.r * d.er
    ctr = C.mean(0)
    rad = np.linalg.norm(C[0] - ctr)
    l_sing = rad                                    # if l equals the circumradius the spheres touch in the plane
    ds = DeltaRobot(l=l_sing)
    ps = ds.fk_all(th)
    det_s = abs(ds.singularity(th, ps[0])[0]) if ps else 1.0
    out.append(Check("Delta actuator singularity: forearms coplanar (FK roots merge) -> det J_x = 0",
                     "MR 7.3, Ex. 7.7(d)", det_s, 1e-6, "l = circumradius of C_1 C_2 C_3: all n_i in one plane",
                     CATEGORY))
    return out
