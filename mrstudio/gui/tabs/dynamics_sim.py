"""Pure maths for the Dynamics tab: RNEA trajectory evaluation, torque laws, chunked forward simulation
(cancellable, with progress), divergence detection and the energy balance.  No Qt; safe for worker threads."""
from __future__ import annotations

import time

import numpy as np

from ... import trajectory as tj
from ...dynamics import DynamicsEngine, simulate


FD_LAWS = ["τ = 0  (falls freely — chaotic pendulum)", "τ = g(θ)  (gravity compensation — floats)",
           "τ = A·τmax  (constant)", "τ = A·τmax·sin(2πft + φᵢ)  (sinusoidal)"]
INTEGRATORS = {"Euler (mr.EulerStep)": "euler", "Semi-implicit (symplectic) Euler": "semi-implicit",
               "Runge–Kutta 4": "rk4", "RK45 adaptive (SciPy Dormand–Prince)": "rk45"}

#: robots offered for the integrator comparison: (label, robot name or None = current robot, Δt [s] or None = setting)
CMP_TARGETS = [("current robot, Δt from the settings", None, None),
               ("Planar 2R arm, Δt = 10 ms (drift clearly visible)", "Planar 2R arm", 0.010),
               ("UR5, Δt = 10 ms", "UR5 (6R industrial)", 0.010)]


# ============================================================================ computations (worker threads)
def compute_inverse(robot, backend, th_s, th_g, Tf, scaling, Ftip, progress, cancelled, N=201):
    """tau(t) along a point-to-point joint trajectory + its decomposition M ddq + c + g + J^T F."""
    eng = DynamicsEngine(robot, backend)        # private engine: Pinocchio data are not thread-safe
    tr = tj.make_trajectory(robot, th_s, th_g, Tf, N, "joint", scaling)
    n = robot.n
    tau, Mdd, c, g, JF = (np.zeros((N, n)) for _ in range(5))
    F = None if Ftip is None or not np.any(Ftip) else np.asarray(Ftip, float)
    t0 = time.perf_counter()
    for k in range(N):
        th, dth, ddth = tr.theta[k], tr.dtheta[k], tr.ddtheta[k]
        tau[k] = eng.inverse(th, dth, ddth, F)
        if k % 20 == 0:
            progress(0.5 * k / N)
            if cancelled():
                return None
    t_id = time.perf_counter() - t0
    for k in range(N):
        th, dth, ddth = tr.theta[k], tr.dtheta[k], tr.ddtheta[k]
        Mdd[k] = eng.mass_matrix(th) @ ddth
        g[k] = eng.gravity_torque(th)
        c[k] = coriolis_vector(eng, th, dth, g[k])
        if F is not None:
            JF[k] = eng.tip_torque(th, F)
        if k % 20 == 0:
            progress(0.5 + 0.5 * k / N)
            if cancelled():
                return None
    ee = np.array([robot.fk(th)[:3, 3] for th in tr.theta])
    return dict(tr=tr, tau=tau, Mdd=Mdd, c=c, g=g, JF=JF, ee=ee, t_id=t_id, backend=backend,
                resid=float(np.abs(tau - (Mdd + c + g + JF)).max()))


def _short_name(name: str) -> str:
    """'UR5 (6R industrial)' -> 'UR5' (label for the comparison plot)."""
    return name.split(" (")[0].replace(" arm", "")


def coriolis_vector(eng, th, dth, g=None):
    """c(theta, dtheta).  MR: mr.VelQuadraticForces.  Pinocchio: RNEA(theta, dtheta, 0) - g(theta)
    (avoids DynamicsEngine.coriolis, which leaves the Pinocchio model's gravity set to zero)."""
    if eng.backend == "mr":
        return eng.coriolis(th, dth)
    g = eng.gravity_torque(th) if g is None else g
    return eng.inverse(th, dth, np.zeros_like(th)) - g


def make_tau_fn(eng, robot, law: int, amp: float, freq: float, add_g: bool):
    tmax = robot.tau_max
    phase = np.arange(robot.n) * 0.9

    def tau_fn(t, th, dth):
        if law == 0:
            return np.zeros(robot.n)
        if law == 1:
            return eng.gravity_torque(th)
        tau = amp * tmax if law == 2 else amp * tmax * np.sin(2 * np.pi * freq * t + phase)
        return tau + eng.gravity_torque(th) if add_g else tau
    return tau_fn


def simulate_chunked(eng, th0, dth0, tau_fn, T, dt, method, damping, progress, cancelled, chunks=20, p0=0.0, p1=1.0):
    """dynamics.simulate() run in chunks so that progress can be reported and the run cancelled.

    Every chunk is a whole number of steps (steps_per_chunk = round(T/dt/chunks)), and the time offset advances
    by exactly steps·dt, so the samples sit on the same grid k·dt as one long simulate() call (no drift)."""
    total = max(1, int(round(T / dt)))
    per = max(1, int(round(total / chunks)))
    parts = []
    th, dth = np.array(th0, float), np.array(dth0, float)
    done, c = 0, 0
    n_chunks = -(-total // per)
    while done < total:
        steps = min(per, total - done)
        off = done * dt
        r = simulate(eng, th, dth, lambda t, q, dq, off=off: tau_fn(t + off, q, dq), steps * dt, dt, method,
                     damping=damping)
        r.t = r.t + off
        parts.append(r if c == 0 else _drop_first(r))
        th, dth = r.theta[-1], r.dtheta[-1]
        done += steps
        c += 1
        progress(p0 + (p1 - p0) * c / n_chunks)
        if cancelled():
            return None
    out = parts[0]
    for p in parts[1:]:
        out.t = np.r_[out.t, p.t]
        out.theta, out.dtheta, out.tau = (np.vstack([a, b]) for a, b in
                                          ((out.theta, p.theta), (out.dtheta, p.dtheta), (out.tau, p.tau)))
        out.ke, out.pe = np.r_[out.ke, p.ke], np.r_[out.pe, p.pe]
    return out


def _drop_first(r):
    r.t, r.theta, r.dtheta, r.tau = r.t[1:], r.theta[1:], r.dtheta[1:], r.tau[1:]
    r.ke, r.pe = r.ke[1:], r.pe[1:]
    return r


def diverged_at(res, limit=1e3) -> int:
    """Index of the first sample where the integration blew up (len(t) if it never did)."""
    bad = ~np.all(np.isfinite(res.dtheta), axis=1) | (np.abs(np.nan_to_num(res.dtheta, nan=1e9)).max(axis=1) > limit)
    if res.ke is not None:
        bad |= ~np.isfinite(res.ke)
    return int(np.argmax(bad)) if bad.any() else len(res.t)


def energy_balance(res, damping):
    """E(t) - W(t): total energy minus the work done by actuators and friction (constant if exact)."""
    b = np.broadcast_to(np.asarray(damping, float), (res.theta.shape[1],))
    P = np.sum((res.tau - b * res.dtheta) * res.dtheta, axis=1)
    W = np.r_[0.0, np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(res.t))]
    return res.ke + res.pe - W
