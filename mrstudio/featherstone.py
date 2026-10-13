"""Independent rigid-body dynamics in Featherstone's spatial-vector form (pure NumPy).

R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer 2008:
  * Ch. 5, Sec. 5.3, Table 5.1 (p. 96)   -- recursive Newton-Euler algorithm (RNEA), inverse dynamics
  * Ch. 6, Sec. 6.2, Table 6.2 (p. 107)  -- composite-rigid-body algorithm (CRBA), joint-space inertia H
  * Ch. 7, Sec. 7.3, Table 7.1 (p. 132)  -- articulated-body algorithm (ABA), forward dynamics
All three are the *body-coordinates* versions of the tables.  Nothing here calls ``modern_robotics``
or Pinocchio: the exponential, the Plücker transforms and the cross operators are written out below,
so agreement with ``mrstudio.dynamics`` (MR's code and Pinocchio) is genuine cross-validation.

Translating the app's Modern-Robotics data (Slist, Mlist, Glist) into Featherstone's model
----------------------------------------------------------------------------------------
* Ordering.  Featherstone's spatial motion vectors are (ω, v_O) and force vectors (n_O, f) in
  Plücker coordinates -- the same ordering as MR's twists V = (ω, v) and wrenches F = (m, f), so no
  re-ordering is needed (Pinocchio, in contrast, stores (v, ω)).
* Body frames.  Body i's coordinate frame is MR's CoM frame {i}; its spatial inertia I_i is MR's
  G_i = diag(I_b, m 1) (Featherstone Eq. 2.63 with c = 0).
* Plücker transforms.  The motion transform from frame A to frame B is ^B X_A = [Ad_{T_BA}]
  (Featherstone Sec. 2.8); the force transform is ^B X_A^* = ^B X_A^{-T}, hence
  ^{λ(i)} X_i^* = (^i X_{λ(i)})^T.
* Tree + joint transform.  ^i X_{λ(i)} = X_J(q_i) X_T(i) (Table 5.1), with the fixed tree transform
  X_T(i) = [Ad_{M_{i-1,i}^{-1}}] and the joint transform X_J = [Ad_{exp(-[A_i] q_i)}], where
  A_i = [Ad_{M_{0,i}^{-1}}] S_i is joint i's screw axis in {i} -- the motion subspace S_i of Table 5.1
  (constant in body coordinates, so c_J = 0).
* Cross products.  v× = [ad_v] (Eq. 2.31) and v×* = -[ad_v]^T (Eq. 2.32).
* Gravity.  a_0 = -a_g with a_g = (0, g) (the fictitious base acceleration of Table 5.1).
* Tip wrench.  MR's F_tip (the wrench the end-effector applies to the environment, in {n+1}) is the
  external force f^x_n = -[Ad_{T_{n+1,n}}]^T F_tip acting on body n.
"""
from __future__ import annotations

import time

import numpy as np

from .checks import Check

CAT = "Dynamics"


# ============================================================================ spatial algebra
def _skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _exp6(S, q) -> np.ndarray:
    """Homogeneous transform exp([S] q) for a unit screw S = (ω, v) (Rodrigues / MR Prop. 3.25)."""
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    T = np.eye(4)
    if np.linalg.norm(w) < 1e-12:          # prismatic
        T[:3, 3] = v * q
        return T
    W = _skew(w)
    T[:3, :3] = np.eye(3) + np.sin(q) * W + (1 - np.cos(q)) * W @ W
    T[:3, 3] = (np.eye(3) * q + (1 - np.cos(q)) * W + (q - np.sin(q)) * W @ W) @ v
    return T


def _inv(T) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ p
    return Ti


def plucker(T) -> np.ndarray:
    """Motion transform ^B X_A for T = T_BA, (ω, v) ordering (Featherstone Eq. 2.24 / Table 2.2)."""
    R, p = T[:3, :3], T[:3, 3]
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = _skew(p) @ R
    return X


def crm(v) -> np.ndarray:
    """Spatial cross product for motion vectors, v× (Featherstone Eq. 2.31)."""
    X = np.zeros((6, 6))
    X[:3, :3] = _skew(v[:3])
    X[3:, 3:] = _skew(v[:3])
    X[3:, :3] = _skew(v[3:])
    return X


def crf(v) -> np.ndarray:
    """Spatial cross product for force vectors, v×* = -(v×)^T (Featherstone Eq. 2.32)."""
    return -crm(v).T


# ============================================================================ model
class FeatherstoneModel:
    """Kinematic chain (λ(i) = i-1) built from a SerialRobot's PoE data, in Featherstone's notation."""

    def __init__(self, robot, g=None):
        if not robot.has_dynamics:
            raise ValueError(f"{robot.name} has no inertial parameters")
        n = robot.n
        self.n = n
        self.I = [np.asarray(G, float) for G in robot.Glist]                       # spatial inertias
        self.XT = [plucker(_inv(np.asarray(robot.Mlist[i], float))) for i in range(n)]  # tree transforms
        Mi = np.eye(4)
        self.S = []                                                                 # motion subspaces
        for i in range(n):
            Mi = Mi @ np.asarray(robot.Mlist[i], float)                            # M_{0,i}
            self.S.append(plucker(_inv(Mi)) @ robot.Slist[:, i])
        self.X_tip = plucker(_inv(np.asarray(robot.Mlist[n], float)))              # ^{n+1}X_n
        g = robot.gravity if g is None else g
        self.a_grav = np.r_[0.0, 0.0, 0.0, np.asarray(g, float)]                   # a_g

    def jcalc(self, i, qi):
        """Joint transform X_J and motion subspace S_i (Featherstone Sec. 4.4, Table 4.1)."""
        return plucker(_exp6(self.S[i], -qi)), self.S[i]

    def _Xup(self, q):
        return [self.jcalc(i, q[i])[0] @ self.XT[i] for i in range(self.n)]       # ^iX_{λ(i)} = X_J X_T

    def _fext(self, Ftip):
        fx = [np.zeros(6) for _ in range(self.n)]
        if Ftip is not None:
            fx[-1] = -self.X_tip.T @ np.asarray(Ftip, float)
        return fx

    # -------------------------------------------------------------- Table 5.1 (RNEA)
    def rnea(self, q, qd, qdd, Ftip=None) -> np.ndarray:
        n, Xup, fx = self.n, self._Xup(q), self._fext(Ftip)
        v, a, f = [None] * n, [None] * n, [None] * n
        v_prev, a_prev = np.zeros(6), -self.a_grav                                # v_0 = 0, a_0 = -a_g
        for i in range(n):
            vJ = self.S[i] * qd[i]
            v[i] = Xup[i] @ v_prev + vJ
            a[i] = Xup[i] @ a_prev + self.S[i] * qdd[i] + crm(v[i]) @ vJ           # c_J = 0
            f[i] = self.I[i] @ a[i] + crf(v[i]) @ self.I[i] @ v[i] - fx[i]
            v_prev, a_prev = v[i], a[i]
        tau = np.zeros(n)
        for i in range(n - 1, -1, -1):
            tau[i] = self.S[i] @ f[i]
            if i > 0:
                f[i - 1] = f[i - 1] + Xup[i].T @ f[i]                              # ^{λ}X_i^* = (^iX_λ)^T
        return tau

    # -------------------------------------------------------------- Table 6.2 (CRBA)
    def crba(self, q) -> np.ndarray:
        n, Xup = self.n, self._Xup(q)
        Ic = [I.copy() for I in self.I]
        for i in range(n - 1, 0, -1):
            Ic[i - 1] = Ic[i - 1] + Xup[i].T @ Ic[i] @ Xup[i]
        H = np.zeros((n, n))
        for i in range(n):
            F = Ic[i] @ self.S[i]
            H[i, i] = self.S[i] @ F
            j = i
            while j > 0:
                F = Xup[j].T @ F
                j -= 1
                H[i, j] = H[j, i] = self.S[j] @ F
        return H

    # -------------------------------------------------------------- Table 7.1 (ABA)
    def aba(self, q, qd, tau, Ftip=None) -> np.ndarray:
        n, Xup, fx = self.n, self._Xup(q), self._fext(Ftip)
        v, c, IA, pA = [None] * n, [None] * n, [None] * n, [None] * n
        v_prev = np.zeros(6)
        for i in range(n):                                                         # pass 1
            vJ = self.S[i] * qd[i]
            v[i] = Xup[i] @ v_prev + vJ
            c[i] = crm(v[i]) @ vJ
            IA[i] = self.I[i].copy()
            pA[i] = crf(v[i]) @ self.I[i] @ v[i] - fx[i]
            v_prev = v[i]
        U, D, u = [None] * n, np.zeros(n), np.zeros(n)
        for i in range(n - 1, -1, -1):                                             # pass 2
            U[i] = IA[i] @ self.S[i]
            D[i] = self.S[i] @ U[i]
            u[i] = tau[i] - self.S[i] @ pA[i]
            if i > 0:
                Ia = IA[i] - np.outer(U[i], U[i]) / D[i]
                pa = pA[i] + Ia @ c[i] + U[i] * u[i] / D[i]
                IA[i - 1] = IA[i - 1] + Xup[i].T @ Ia @ Xup[i]
                pA[i - 1] = pA[i - 1] + Xup[i].T @ pa
        qdd = np.zeros(n)
        a_prev = -self.a_grav                                                      # pass 3
        for i in range(n):
            a1 = Xup[i] @ a_prev + c[i]
            qdd[i] = (u[i] - U[i] @ a1) / D[i]
            a_prev = a1 + self.S[i] * qdd[i]
        return qdd


# ============================================================================ validation
def validation_checks() -> list[Check]:
    """Featherstone RNEA / CRBA / ABA (this module) vs. Modern Robotics' code and Pinocchio."""
    from .dynamics import DynamicsEngine, pin
    from .robots import ROBOTS, get_robot

    out: list[Check] = []
    rng = np.random.default_rng(8)
    backends = ["mr"] + (["pinocchio"] if pin is not None else [])
    for name in ROBOTS:
        r = get_robot(name)
        if not r.has_dynamics:
            continue
        fm = FeatherstoneModel(r)
        engines = {b: DynamicsEngine(r, b) for b in backends}
        err = {(k, b): 0.0 for k in ("rnea", "crba", "aba") for b in backends}
        scale = {k: 1.0 for k in ("rnea", "crba", "aba")}
        t0 = time.perf_counter()
        for _ in range(5):
            q = r.random_config(rng)
            qd, qdd = rng.uniform(-1, 1, r.n), rng.uniform(-1, 1, r.n)
            Ftip = rng.uniform(-2, 2, 6)
            tau_f = fm.rnea(q, qd, qdd, Ftip)
            H_f = fm.crba(q)
            tau_in = rng.uniform(-5, 5, r.n)
            qdd_f = fm.aba(q, qd, tau_in, Ftip)
            scale["rnea"] = max(scale["rnea"], np.abs(tau_f).max())
            scale["crba"] = max(scale["crba"], np.abs(H_f).max())
            scale["aba"] = max(scale["aba"], np.abs(qdd_f).max())
            for b, eng in engines.items():
                err["rnea", b] = max(err["rnea", b], np.abs(tau_f - eng.inverse(q, qd, qdd, Ftip)).max())
                err["crba", b] = max(err["crba", b], np.abs(H_f - eng.mass_matrix(q)).max())
                err["aba", b] = max(err["aba", b], np.abs(qdd_f - eng.forward(q, qd, tau_in, Ftip)).max())
        ms = (time.perf_counter() - t0) * 1e3
        label = {"mr": "Modern Robotics code", "pinocchio": "Pinocchio"}
        what = {"rnea": ("RNEA τ = ID(θ, θ̇, θ̈, F_tip)", "Featherstone Table 5.1 / MR 8.3"),
                "crba": ("CRBA mass matrix H(θ)", "Featherstone Table 6.2 / MR 8.4"),
                "aba": ("ABA θ̈ = FD(θ, θ̇, τ, F_tip)", "Featherstone Table 7.1 / MR 8.5")}
        for k in ("rnea", "crba", "aba"):
            for b in backends:
                e = err[k, b] / max(1.0, scale[k])
                out.append(Check(f"Featherstone {what[k][0]} vs {label[b]} ({name})", what[k][1], float(e), 1e-10,
                                 f"pure-NumPy spatial-vector {k.upper()} (mrstudio/featherstone.py) vs "
                                 f"{label[b]}; 5 random states with tip wrench, relative max error "
                                 f"(÷ max(1, max|value|)); {ms:.0f} ms for all three algorithms", CAT))
    return out
