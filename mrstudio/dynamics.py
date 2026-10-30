"""Dynamics of open chains following Modern Robotics Chapter 8.

    tau = M(theta) theta_ddot + c(theta, theta_dot) + g(theta) + J^T(theta) F_tip      (MR Eq. 8.77)

Two interchangeable back-ends with an identical API:

* ``"mr"``        -- the official Modern Robotics Python code (recursive Newton-Euler,
                     Sec. 8.3; mass matrix by n RNEA calls, Sec. 8.5).  Reference / didactic.
* ``"pinocchio"`` -- Pinocchio (LAAS-CNRS, C++): RNEA, CRBA and Featherstone's
                     Articulated-Body Algorithm.  ~1000× faster, used for long simulations.

The Pinocchio model is *built from the same PoE data* (Slist, Mlist, Glist); the
validation suite shows both engines agree to machine precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import modern_robotics as mr
import numpy as np

from .robots import SerialRobot, screw_axis_line

try:
    import pinocchio as pin
except Exception:  # pragma: no cover
    pin = None


# ============================================================================ Pinocchio bridge
def to_pinocchio(robot: SerialRobot):
    """Build a pinocchio.Model from the Modern Robotics description.

    Joint frame J_i is placed on screw axis i, aligned with {s} at the home configuration,
    so that e^[S_i]th = J_i Rot(w_i, th) J_i^-1 and the PoE formula is reproduced exactly.

    Only pure revolute (pitch 0) and prismatic joints have a Pinocchio counterpart here: a screw axis with
    pitch h = w.v / |w|^2 != 0 raises ValueError instead of being silently turned into a revolute joint."""
    if pin is None:
        raise RuntimeError("pinocchio is not installed")
    for i in range(robot.n):
        w, v = robot.Slist[:3, i], robot.Slist[3:, i]
        nw = float(np.linalg.norm(w))
        if nw > 1e-9:
            pitch = float(w @ v) / nw**2
            if abs(pitch) > 1e-9 * max(1.0, float(np.linalg.norm(v)) / nw):
                raise ValueError(f"{robot.name}: joint {i + 1} is a screw joint with pitch {pitch:.3g} m/rad "
                                 "(w·v ≠ 0); only revolute (pitch 0) and prismatic joints are supported")
    model = pin.Model()
    model.name = robot.name
    parent, J_prev = 0, np.eye(4)
    Mc = robot.com_home if robot.has_dynamics else None
    for i in range(robot.n):
        q, w, rev = screw_axis_line(robot.Slist[:, i])
        J_i = np.eye(4)
        J_i[:3, 3] = q
        P = np.linalg.inv(J_prev) @ J_i
        jm = pin.JointModelRevoluteUnaligned(w) if rev else pin.JointModelPrismaticUnaligned(w)
        jid = model.addJoint(parent, jm, pin.SE3(P[:3, :3], P[:3, 3]), f"joint{i + 1}")
        if Mc is not None:
            G = robot.Glist[i]
            R, pc = Mc[i][:3, :3], Mc[i][:3, 3]
            inertia = pin.Inertia(float(G[3, 3]), pc - q, R @ G[:3, :3] @ R.T)
        else:
            inertia = pin.Inertia.Zero()
        model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())
        parent, J_prev = jid, J_i
    E = np.linalg.inv(J_prev) @ robot.M
    model.addFrame(pin.Frame("ee", parent, pin.SE3(E[:3, :3], E[:3, 3]), pin.FrameType.OP_FRAME))
    model.gravity = pin.Motion(np.r_[robot.gravity, 0.0, 0.0, 0.0])
    model.lowerPositionLimit = robot.joint_limits[:, 0].copy()
    model.upperPositionLimit = robot.joint_limits[:, 1].copy()
    return model


# ============================================================================ Engine
class DynamicsEngine:
    """Rigid-body dynamics of a SerialRobot with a selectable back-end ("mr" or "pinocchio")."""

    def __init__(self, robot: SerialRobot, backend: str = "pinocchio", g=None, Mlist=None, Glist=None):
        if not robot.has_dynamics:
            raise ValueError(f"{robot.name} has no inertial parameters")
        self.robot = robot
        self.backend = backend if (backend == "mr" or pin is not None) else "mr"
        self.g = np.asarray(robot.gravity if g is None else g, float)
        # Mlist/Glist may be overridden to build a deliberately *wrong* model (control robustness)
        self.Mlist = robot.Mlist if Mlist is None else Mlist
        self.Glist = robot.Glist if Glist is None else Glist
        self.S = robot.Slist
        if self.backend == "pinocchio":
            if Mlist is None and Glist is None:
                src = robot
            else:
                src = SerialRobot(robot.name, robot.M, robot.Slist, self.Mlist, self.Glist,
                                  joint_limits=robot.joint_limits, gravity=self.g)
            self.model = to_pinocchio(src)
            self.model.gravity = pin.Motion(np.r_[self.g, 0.0, 0.0, 0.0])
            self.data = self.model.createData()
            self.ee_id = self.model.getFrameId("ee")

    # -- helpers ---------------------------------------------------------------
    def _f_ext(self, theta, Ftip):
        """Pinocchio external forces equivalent to the end-effector applying wrench Ftip
        (MR convention: Ftip = (m, f) expressed in the end-effector frame {n+1})."""
        if Ftip is None or not np.any(Ftip):
            return None
        pin.forwardKinematics(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        fr = self.model.frames[self.ee_id]
        jid = fr.parentJoint
        # wrench exerted ON the robot by the environment = -Ftip, expressed in frame ee -> joint frame
        f_ee = pin.Force(-np.asarray(Ftip[3:], float), -np.asarray(Ftip[:3], float))
        fext = [pin.Force.Zero() for _ in range(self.model.njoints)]
        fext[jid] = fr.placement.act(f_ee)
        return fext

    # -- Ch. 8 quantities ------------------------------------------------------
    def inverse(self, theta, dtheta, ddtheta, Ftip=None) -> np.ndarray:
        """tau = ID(theta, dtheta, ddtheta, F_tip)   (RNEA, MR Eqs. 8.50-8.54)."""
        th, dth, ddth = (np.asarray(x, float) for x in (theta, dtheta, ddtheta))
        if self.backend == "mr":
            F = np.zeros(6) if Ftip is None else np.asarray(Ftip, float)
            return np.asarray(mr.InverseDynamics(th, dth, ddth, self.g, F, self.Mlist, self.Glist, self.S))
        fext = self._f_ext(th, Ftip)
        if fext is None:
            return pin.rnea(self.model, self.data, th, dth, ddth).copy()
        return pin.rnea(self.model, self.data, th, dth, ddth, fext).copy()

    def mass_matrix(self, theta) -> np.ndarray:
        th = np.asarray(theta, float)
        if self.backend == "mr":
            return np.asarray(mr.MassMatrix(th, self.Mlist, self.Glist, self.S))
        Mq = pin.crba(self.model, self.data, th)
        return np.triu(Mq) + np.triu(Mq, 1).T

    def coriolis(self, theta, dtheta) -> np.ndarray:
        """c(theta, dtheta) = C(theta, dtheta) dtheta  (velocity-product terms)."""
        th, dth = np.asarray(theta, float), np.asarray(dtheta, float)
        if self.backend == "mr":
            return np.asarray(mr.VelQuadraticForces(th, dth, self.Mlist, self.Glist, self.S))
        # c = h - g; never mutates model.gravity (thread-safe, no hidden state)
        h = pin.nonLinearEffects(self.model, self.data, th, dth).copy()
        return h - pin.computeGeneralizedGravity(self.model, self.data, th)

    def coriolis_matrix(self, theta, dtheta) -> np.ndarray:
        """C(theta, dtheta) built from Christoffel symbols (Pinocchio) -- used for the passivity check."""
        if self.backend == "mr" or pin is None:
            raise NotImplementedError
        return pin.computeCoriolisMatrix(self.model, self.data, np.asarray(theta, float),
                                         np.asarray(dtheta, float)).copy()

    def gravity_torque(self, theta) -> np.ndarray:
        th = np.asarray(theta, float)
        if self.backend == "mr":
            return np.asarray(mr.GravityForces(th, self.g, self.Mlist, self.Glist, self.S))
        return pin.computeGeneralizedGravity(self.model, self.data, th).copy()

    def tip_torque(self, theta, Ftip) -> np.ndarray:
        """J^T(theta) F_tip  (torques needed to make the end-effector apply Ftip)."""
        th = np.asarray(theta, float)
        return np.asarray(mr.EndEffectorForces(th, np.asarray(Ftip, float), self.Mlist, self.Glist, self.S))

    def forward(self, theta, dtheta, tau, Ftip=None) -> np.ndarray:
        """theta_ddot = M^-1 (tau - c - g - J^T F_tip)   (MR Eq. 8.81; Pinocchio: ABA)."""
        th, dth, t = (np.asarray(x, float) for x in (theta, dtheta, tau))
        if self.backend == "mr":
            F = np.zeros(6) if Ftip is None else np.asarray(Ftip, float)
            return np.asarray(mr.ForwardDynamics(th, dth, t, self.g, F, self.Mlist, self.Glist, self.S))
        fext = self._f_ext(th, Ftip)
        if fext is None:
            return pin.aba(self.model, self.data, th, dth, t).copy()
        return pin.aba(self.model, self.data, th, dth, t, fext).copy()

    # -- energy ------------------------------------------------------------------
    def kinetic_energy(self, theta, dtheta) -> float:
        dth = np.asarray(dtheta, float)
        return float(0.5 * dth @ self.mass_matrix(theta) @ dth)

    def potential_energy(self, theta) -> float:
        """P = -sum_i m_i g . p_ci(theta)   with CoM positions from the PoE formula."""
        rob = SerialRobot(self.robot.name, self.robot.M, self.S, self.Mlist, self.Glist) \
            if self.Mlist is not self.robot.Mlist else self.robot
        p = rob.com_positions(theta)
        m = np.array([G[3, 3] for G in self.Glist])
        return float(-np.sum(m * (p @ self.g)))

    def energy(self, theta, dtheta) -> tuple[float, float]:
        return self.kinetic_energy(theta, dtheta), self.potential_energy(theta)


# ============================================================================ simulation
@dataclass
class SimResult:
    t: np.ndarray
    theta: np.ndarray       # N x n
    dtheta: np.ndarray
    tau: np.ndarray
    ke: np.ndarray | None = None
    pe: np.ndarray | None = None
    method: str = ""


def simulate(engine: DynamicsEngine, theta0, dtheta0, tau_fn, t_final: float, dt: float = 0.005,
             method: str = "rk4", Ftip=None, with_energy: bool = True, damping=None) -> SimResult:
    """Integrate the forward dynamics (MR Sec. 8.5).

    tau_fn(t, theta, dtheta) -> tau.   method in {"euler" (MR's EulerStep), "semi-implicit", "rk4", "rk45"}
    ("rk45" = SciPy's adaptive Dormand-Prince RK45, rtol = atol = 1e-10, sampled every dt).
    damping: optional viscous joint friction coefficients b (tau_f = -b dtheta).
    """
    n = engine.robot.n
    b = np.zeros(n) if damping is None else np.broadcast_to(np.asarray(damping, float), (n,))

    def acc(t, th, dth):
        tau = np.asarray(tau_fn(t, th, dth), float)
        return engine.forward(th, dth, tau - b * dth, Ftip), tau

    if method == "rk45":
        from scipy.integrate import solve_ivp

        def f(t, x):
            a, _ = acc(t, x[:n], x[n:])
            return np.r_[x[n:], a]
        t_eval = np.arange(0.0, t_final + 1e-12, dt)
        sol = solve_ivp(f, (0.0, t_final), np.r_[theta0, dtheta0], t_eval=t_eval, rtol=1e-10, atol=1e-10,
                        method="RK45")
        TH, DTH = sol.y[:n].T, sol.y[n:].T
        TAU = np.array([tau_fn(t, TH[k], DTH[k]) for k, t in enumerate(sol.t)])
        ts = sol.t
    else:
        steps = int(round(t_final / dt))
        ts = np.arange(steps + 1) * dt
        TH = np.zeros((steps + 1, n))
        DTH = np.zeros((steps + 1, n))
        TAU = np.zeros((steps + 1, n))
        th, dth = np.array(theta0, float), np.array(dtheta0, float)
        for k in range(steps + 1):
            TH[k], DTH[k] = th, dth
            if k == steps:
                TAU[k] = tau_fn(ts[k], th, dth)
                break
            t = ts[k]
            if method == "euler":            # mr.EulerStep: explicit Euler
                a, TAU[k] = acc(t, th, dth)
                th, dth = th + dt * dth, dth + dt * a
            elif method == "semi-implicit":  # symplectic Euler
                a, TAU[k] = acc(t, th, dth)
                dth = dth + dt * a
                th = th + dt * dth
            else:                            # classic RK4
                k1a, TAU[k] = acc(t, th, dth)
                k1v = dth
                k2a, _ = acc(t + dt / 2, th + dt / 2 * k1v, dth + dt / 2 * k1a)
                k2v = dth + dt / 2 * k1a
                k3a, _ = acc(t + dt / 2, th + dt / 2 * k2v, dth + dt / 2 * k2a)
                k3v = dth + dt / 2 * k2a
                k4a, _ = acc(t + dt, th + dt * k3v, dth + dt * k3a)
                k4v = dth + dt * k3a
                th = th + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                dth = dth + dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
    res = SimResult(ts, TH, DTH, TAU, method=method)
    if with_energy:
        e = np.array([engine.energy(TH[k], DTH[k]) for k in range(len(ts))])
        res.ke, res.pe = e[:, 0], e[:, 1]
    return res
