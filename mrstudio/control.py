"""Motion control of torque-controlled robots following Modern Robotics Chapter 11 (Sec. 11.4).

Controllers (theta_e = theta_d - theta, all gains in *acceleration* units so they can be compared fairly):

* "PD"                  tau = K'_p theta_e + K'_d dtheta_e                       (MR 11.4.1, decentralised)
* "PID"                 tau = K'_p theta_e + K'_i int(theta_e) + K'_d dtheta_e    (MR 11.4.1 / 11.4.2)
* "PD + gravity comp."  tau = K'_p theta_e + K'_d dtheta_e + g~(theta)            (MR 11.4.2)
* "Computed torque"     tau = M~(theta)(ddtheta_d + K_p theta_e + K_i int theta_e + K_d dtheta_e) + h~(theta, dtheta)
                        (MR Eq. 11.35; identical to mr.ComputedTorque)

For the decentralised (independent-joint) controllers the torque gains are K' = K diag(M~(theta_d(0))), i.e. every
joint gets the same closed-loop bandwidth as the computed-torque controller.

The "real robot" (plant) is simulated with Pinocchio's ABA (engine.forward) and may carry an unknown payload; the
controller may use a deliberately wrong model (scaled inertias) built with DynamicsEngine(robot, backend, Glist=...).
The controller runs at a fixed period dt (zero-order hold); the plant is integrated with RK4 (or explicit Euler,
which reproduces mr.SimulateControl exactly).
"""
from __future__ import annotations

from dataclasses import dataclass

import modern_robotics as mr
import numpy as np

from .checks import Check
from .dynamics import DynamicsEngine
from .robots import SerialRobot, trans

CONTROLLERS = ("PD", "PID", "PD + gravity comp.", "Computed torque")
CAT = "Trajectory & control"


# ============================================================================ models
def payload_model(robot: SerialRobot, m_payload: float, radius: float = 0.03, Mlist=None, Glist=None):
    """(Mlist, Glist) of the robot carrying a solid-sphere payload of mass m_payload centred at the end-effector
    frame origin.  The payload is merged into the last link: new centre of mass and the parallel-axis theorem
    (MR Eqs. 8.27 and 8.42), so the last link frame stays at its (new) centre of mass with a block-diagonal G."""
    Mlist = [np.array(M, float) for M in (robot.Mlist if Mlist is None else Mlist)]
    Glist = [np.array(G, float) for G in (robot.Glist if Glist is None else Glist)]
    if m_payload <= 0:
        return Mlist, Glist
    n = robot.n
    G = Glist[n - 1]
    m, I = G[3, 3], G[:3, :3]
    r_p = Mlist[n][:3, 3]                          # payload position in the last CoM frame {n}
    m_t = m + m_payload
    c = m_payload * r_p / m_t                      # new CoM in {n} coordinates

    def steiner(mass, d):
        return mass * (d @ d * np.eye(3) - np.outer(d, d))
    I_p = 0.4 * m_payload * radius**2 * np.eye(3)
    I_new = I + steiner(m, -c) + I_p + steiner(m_payload, r_p - c)
    G_new = np.zeros((6, 6))
    G_new[:3, :3], G_new[3:, 3:] = I_new, m_t * np.eye(3)
    Glist[n - 1] = G_new
    Mlist[n - 1] = Mlist[n - 1] @ trans(c)
    Mlist[n] = trans(-c) @ Mlist[n]
    return Mlist, Glist


def plant_engine(robot: SerialRobot, m_payload: float = 0.0) -> DynamicsEngine:
    """The 'real robot': Pinocchio model, optionally with a payload the controller does not know about."""
    if m_payload <= 0:
        return DynamicsEngine(robot, "pinocchio")
    Ml, Gl = payload_model(robot, m_payload)
    return DynamicsEngine(robot, "pinocchio", Mlist=Ml, Glist=Gl)


def model_engine(robot: SerialRobot, backend: str = "pinocchio", mass_scale: float = 1.0) -> DynamicsEngine:
    """The controller's (possibly wrong) model M~, h~: all link inertias multiplied by mass_scale."""
    if abs(mass_scale - 1.0) < 1e-12:
        return DynamicsEngine(robot, backend)
    return DynamicsEngine(robot, backend, Glist=[np.asarray(G, float) * mass_scale for G in robot.Glist])


# ============================================================================ controllers
class Controller:
    """tau = controller(theta, dtheta, eint, theta_d, dtheta_d, ddtheta_d).  eint follows mr.ComputedTorque:
    the integral term is Ki (eint + theta_e)."""

    def __init__(self, kind: str, model: DynamicsEngine, Kp: float, Ki: float, Kd: float, theta_ref=None):
        if kind not in CONTROLLERS:
            raise ValueError(kind)
        self.kind, self.model = kind, model
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        th_ref = model.robot.home if theta_ref is None else np.asarray(theta_ref, float)
        self.Jdiag = np.diag(model.mass_matrix(th_ref)).copy()   # apparent joint inertias of the model

    def __call__(self, th, dth, eint, thd, dthd, ddthd):
        e, ed = thd - th, dthd - dth
        if self.kind == "Computed torque":                       # MR Eq. 11.35 == mr.ComputedTorque
            v = self.Kp * e + self.Ki * (eint + e) + self.Kd * ed
            return self.model.mass_matrix(th) @ v + self.model.inverse(th, dth, ddthd)
        J = self.Jdiag
        tau = J * (self.Kp * e + self.Kd * ed)
        if self.kind == "PID":
            tau = tau + J * self.Ki * (eint + e)
        elif self.kind == "PD + gravity comp.":
            tau = tau + self.model.gravity_torque(th)
        return tau


# ============================================================================ closed-loop simulation
@dataclass
class ControlResult:
    kind: str
    t: np.ndarray
    theta: np.ndarray
    dtheta: np.ndarray
    theta_d: np.ndarray
    tau: np.ndarray
    err: np.ndarray            # theta_d - theta
    rms: np.ndarray            # per joint [rad]
    rms_total: float
    max_err: float
    saturated: float = 0.0     # fraction of samples with at least one saturated actuator
    cancelled: bool = False


def simulate_control(plant: DynamicsEngine, controller, t, theta_d, dtheta_d, ddtheta_d, theta0, dtheta0,
                     substeps: int = 2, integrator: str = "rk4", tau_limit=None, Ftip=None,
                     progress=None, cancel=None) -> ControlResult:
    """Digital control loop: at t_k the controller computes tau_k, which is held constant (ZOH) while the plant
    dynamics theta_ddot = FD(theta, dtheta, tau_k) are integrated over [t_k, t_k+1] with `substeps` steps of
    RK4 (or explicit Euler = mr.EulerStep, as in mr.SimulateControl).  Optional actuator saturation |tau|<=tau_limit."""
    t = np.asarray(t, float)
    N, n = len(t), plant.robot.n
    TH, DTH, TAU = np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, n))
    th, dth = np.array(theta0, float), np.array(dtheta0, float)
    eint = np.zeros(n)
    sat = 0
    lim = None if tau_limit is None else np.asarray(tau_limit, float)
    f = (lambda q, dq, tq: plant.forward(q, dq, tq, Ftip))
    cancelled = False
    for k in range(N):
        TH[k], DTH[k] = th, dth
        tau = np.asarray(controller(th, dth, eint, theta_d[k], dtheta_d[k], ddtheta_d[k]), float)
        if lim is not None:
            if np.any(np.abs(tau) > lim):
                sat += 1
            tau = np.clip(tau, -lim, lim)
        TAU[k] = tau
        if k == N - 1:
            break
        h = (t[k + 1] - t[k]) / substeps
        for _ in range(substeps):
            if integrator == "euler":
                a = f(th, dth, tau)
                th, dth = mr.EulerStep(th, dth, a, h)
            else:
                k1v, k1a = dth, f(th, dth, tau)
                k2v = dth + 0.5 * h * k1a
                k2a = f(th + 0.5 * h * k1v, k2v, tau)
                k3v = dth + 0.5 * h * k2a
                k3a = f(th + 0.5 * h * k2v, k3v, tau)
                k4v = dth + h * k3a
                k4a = f(th + h * k3v, k4v, tau)
                th = th + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                dth = dth + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        if not (np.all(np.isfinite(th)) and np.all(np.isfinite(dth))) or np.abs(dth).max() > 1e4:
            TH[k + 1:], DTH[k + 1:], TAU[k + 1:] = np.nan, np.nan, np.nan
            break
        eint = eint + (t[k + 1] - t[k]) * (theta_d[k] - th)     # as in mr.SimulateControl
        if progress is not None and k % 50 == 0:
            progress((k + 1) / N)
        if cancel is not None and k % 50 == 0 and cancel():
            cancelled = True
            TH, DTH, TAU, t = TH[:k + 2], DTH[:k + 2], TAU[:k + 2], t[:k + 2]
            theta_d = theta_d[:k + 2]
            break
    err = np.asarray(theta_d)[:len(t)] - TH
    ok = np.all(np.isfinite(err), axis=1)
    rms = np.sqrt(np.nanmean(err[ok]**2, axis=0)) if ok.any() else np.full(n, np.inf)
    if not ok.all():
        rms = np.full(n, np.inf)
    return ControlResult(getattr(controller, "kind", "custom"), t, TH, DTH, np.asarray(theta_d)[:len(t)], TAU, err,
                         rms, float(np.sqrt(np.mean(rms**2))), float(np.nanmax(np.abs(err))) if ok.any() else np.inf,
                         sat / max(len(t), 1), cancelled)


def run_controller(robot: SerialRobot, traj, kind: str, Kp: float, Ki: float, Kd: float, backend: str = "pinocchio",
                   mass_scale: float = 1.0, payload: float = 0.0, theta0_offset=0.0, saturate: bool = False,
                   substeps: int = 2, progress=None, cancel=None, plant=None, model=None) -> ControlResult:
    """Convenience wrapper: build plant + (wrong) model + controller and simulate tracking of `traj`
    (a trajectory.Trajectory sampled at the control period)."""
    plant = plant if plant is not None else plant_engine(robot, payload)
    model = model if model is not None else model_engine(robot, backend, mass_scale)
    ctrl = Controller(kind, model, Kp, Ki, Kd, theta_ref=traj.theta[0])
    th0 = traj.theta[0] + np.broadcast_to(np.asarray(theta0_offset, float), (robot.n,))
    return simulate_control(plant, ctrl, traj.t, traj.theta, traj.dtheta, traj.ddtheta, th0, traj.dtheta[0],
                            substeps=substeps, tau_limit=robot.tau_max if saturate else None,
                            progress=progress, cancel=cancel)


# ============================================================================ validation
def _mr_simulate_control(thetalist, dthetalist, g, Ftipmat, Mlist, Glist, Slist, thetamatd, dthetamatd,
                         ddthetamatd, gtilde, Mtildelist, Gtildelist, Kp, Ki, Kd, dt, intRes):
    """mr.SimulateControl's computation, line for line, minus its final matplotlib plot.

    The textbook function ends with plt.plot/plt.show (and np.random colours); calling it from the validation
    worker thread would need a process-wide matplotlib backend switch, so the loop is transcribed here using the
    same mr.ComputedTorque / mr.ForwardDynamics / mr.EulerStep calls.  Returns (taumat, thetamat), N x n."""
    Ftipmat = np.array(Ftipmat).T
    thetamatd, dthetamatd, ddthetamatd = (np.array(x).T for x in (thetamatd, dthetamatd, ddthetamatd))
    m, n = thetamatd.shape
    th, dth = np.array(thetalist, float).copy(), np.array(dthetalist, float).copy()
    eint = np.zeros(m)
    taumat, thetamat = np.zeros((m, n)), np.zeros((m, n))
    for i in range(n):
        tau = mr.ComputedTorque(th, dth, eint, gtilde, Mtildelist, Gtildelist, Slist, thetamatd[:, i],
                                dthetamatd[:, i], ddthetamatd[:, i], Kp, Ki, Kd)
        for _ in range(intRes):
            ddth = mr.ForwardDynamics(th, dth, tau, g, Ftipmat[:, i], Mlist, Glist, Slist)
            th, dth = mr.EulerStep(th, dth, ddth, 1.0 * dt / intRes)
        taumat[:, i] = tau
        thetamat[:, i] = th
        eint = eint + dt * (thetamatd[:, i] - th)
    return taumat.T, thetamat.T


def validation_checks() -> list[Check]:
    from . import trajectory as tj
    from .robots import get_robot
    out: list[Check] = []
    rng = np.random.default_rng(3)

    # 1. our computed-torque law == mr.ComputedTorque (MR Eq. 11.35)
    r = get_robot("UR5 (6R industrial)")
    for backend in ("mr", "pinocchio"):
        eng = DynamicsEngine(r, backend)
        ctrl = Controller("Computed torque", eng, 20.0, 5.0, 9.0)
        th, dth, eint = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), rng.uniform(-.1, .1, 6)
        thd, dthd, ddthd = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
        ref = mr.ComputedTorque(th, dth, eint, r.gravity, r.Mlist, r.Glist, r.Slist, thd, dthd, ddthd, 20.0, 5.0, 9.0)
        out.append(Check(f"Computed-torque law = mr.ComputedTorque (UR5, {backend} model)", "MR 11.4.2 Eq. 11.35",
                         float(np.abs(ctrl(th, dth, eint, thd, dthd, ddthd) - ref).max()), 1e-9,
                         "random state, reference and integral error", CAT))

    # 2. closed loop with explicit Euler reproduces mr.SimulateControl
    r2 = get_robot("Planar 2R arm")
    tr = tj.make_trajectory(r2, r2.home, r2.home + np.array([0.8, -0.6]), 0.6, 61, "joint", "cubic")
    dt = tr.t[1] - tr.t[0]
    Gt = [G * 1.2 for G in r2.Glist]
    th0 = tr.theta[0] + 0.05
    taumat, thetamat = _mr_simulate_control(th0, np.zeros(2), r2.gravity, np.zeros((61, 6)), r2.Mlist, r2.Glist,
                                            r2.Slist, tr.theta, tr.dtheta, tr.ddtheta, r2.gravity, r2.Mlist, Gt,
                                            25.0, 4.0, 10.0, dt, 4)
    ctrl = Controller("Computed torque", DynamicsEngine(r2, "mr", Glist=Gt), 25.0, 4.0, 10.0)
    res = simulate_control(DynamicsEngine(r2, "pinocchio"), ctrl, tr.t, tr.theta, tr.dtheta, tr.ddtheta, th0,
                           np.zeros(2), substeps=4, integrator="euler")
    err = max(np.abs(res.theta[1:] - thetamat[:-1]).max(), np.abs(res.tau[:-1] - taumat[:-1]).max() / 10)
    out.append(Check("Closed loop (Pinocchio plant, Euler) reproduces mr.SimulateControl (2R, wrong model ×1.2)",
                     "MR 11.4.2 / 8.5", float(err), 1e-8,
                     "61 control steps, intRes = 4, Ki ≠ 0; reference = mr.SimulateControl's own loop "
                     "(mr.ComputedTorque, mr.ForwardDynamics, mr.EulerStep) without its matplotlib plot", CAT))

    # 3. computed torque with a perfect model tracks exactly; decentralised PD does not
    tr = tj.make_trajectory(r, r.home, r.home + np.array([0.6, -0.5, 0.6, 0.4, -0.5, 0.7]), 1.5, 751, "joint",
                            "quintic")
    plant = DynamicsEngine(r, "pinocchio")
    res_ct = run_controller(r, tr, "Computed torque", 100.0, 0.0, 20.0, plant=plant, model=plant)
    res_pd = run_controller(r, tr, "PD", 100.0, 0.0, 20.0, plant=plant, model=plant)
    out.append(Check("Computed torque with exact model: max tracking error (UR5)", "MR 11.4.2",
                     res_ct.max_err, 5e-4, f"only the O(Δt) zero-order-hold lag remains (Δt = 2 ms); "
                     f"decentralised PD: {res_pd.max_err:.3f} rad", CAT))
    # set-point: the closed loop must follow the linear error dynamics theta_e'' + Kd theta_e' + Kp theta_e = 0
    tt = np.arange(0, 1.0 + 1e-9, 5e-4)
    z = np.zeros((len(tt), 6))
    hold_th = np.repeat(r.home[None], len(tt), 0)
    ctrl = Controller("Computed torque", plant, 100.0, 0.0, 20.0)
    res = simulate_control(plant, ctrl, tt, hold_th, z, z, r.home - 0.05, np.zeros(6), substeps=1)
    pred = 0.05 * (1 + 10 * tt) * np.exp(-10 * tt)          # critically damped: (1 + w t) e^{-w t}, w = 10
    dev = float(np.abs(res.err - pred[:, None]).max())
    out.append(Check("Computed torque: θₑ(t) obeys θ̈ₑ + Kd θ̇ₑ + Kp θₑ = 0 (critically damped)", "MR 11.2 / 11.4.2",
                     dev, 2.5e-4, "initial error 0.05 rad on all joints, ωn = 10 rad/s, ζ = 1, Δt = 0.5 ms: "
                     "max |θₑ − 0.05(1+ωt)e^(−ωt)|", CAT))

    # 4. payload model: parallel-axis theorem reproduces the static load J_b^T F of a point mass
    m_p = 2.0
    Ml, Gl = payload_model(r, m_p, radius=0.0)
    e_pl = DynamicsEngine(r, "mr", Mlist=Ml, Glist=Gl)
    e0 = DynamicsEngine(r, "mr")
    th = rng.uniform(-1, 1, 6)
    T = r.fk(th)
    Fb = np.r_[0, 0, 0, T[:3, :3].T @ (-m_p * r.gravity)]
    err = np.abs(e_pl.gravity_torque(th) - (e0.gravity_torque(th) + e0.tip_torque(th, Fb))).max()
    out.append(Check("Payload model (CoM shift + parallel-axis theorem) = static load J_bᵀF (UR5, 2 kg)", "MR 8.2 / 5.2",
                     float(err), 1e-9, "gravity torques of the merged last link vs added tip wrench", CAT))
    pe = DynamicsEngine(r, "pinocchio", Mlist=Ml, Glist=Gl)
    err = np.abs(pe.mass_matrix(th) - e_pl.mass_matrix(th)).max()
    out.append(Check("Payload model: Pinocchio CRBA = mr.MassMatrix", "MR 8.4", float(err), 1e-9, "UR5 + 2 kg", CAT))

    # 5. passivity property used in the stability proofs of PD + gravity compensation
    th, dth = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
    h = 1e-6
    Md = (plant.mass_matrix(th + h * dth) - plant.mass_matrix(th - h * dth)) / (2 * h)
    Nm = Md - 2 * plant.coriolis_matrix(th, dth)
    out.append(Check("Ṁ − 2C is skew-symmetric (UR5, Pinocchio Coriolis matrix)", "MR 8.1.2 / 11.4.2",
                     float(np.abs(Nm + Nm.T).max()), 1e-6, "Ṁ by central differences along θ̇", CAT))
    return out
