"""Hardware acceptance tests HW-2 ... HW-6, identical to the SBEG 361 starter ``hwcompare.py``
(same functions, thresholds and result keys), plus the SBEG 461 ``hardware.py`` stop-time and delay
helpers.  Angles in **degrees** here, as in the logs the starters read.

  HW-2  joint tracking   RMS / max |cmd - meas| per joint
  HW-3  task tracking    tool-point error |p(cmd) - p(meas)| via space PoE FK (MR Eq. 4.14, mr.FKinSpace)
  HW-4  joint-limit margin of the MEASURED joints (to the hardware limits)
  HW-5  peak MEASURED joint speed / rated  <= 25 % (session H1) or 50 % (otherwise)
  HW-6  safe stop: time to halt and joint motion after the stop command

``validation_checks()`` verifies the pipeline quantitatively (Validation tab, category "Hardware").

This is an example course setup: the course handout is authoritative for the thresholds and rules below, which
mirror it in one place (``DEFAULTS``, ``SESSION_SPEED_FRACTION``) so another course or term can change them here.
"""
from __future__ import annotations

import os
import tempfile

import modern_robotics as mr
import numpy as np

from ..checks import Check

CATEGORY = "Hardware"

DEFAULTS = dict(joint_rms_deg=2.0, joint_max_deg=5.0, task_rms_mm=5.0, task_max_mm=10.0,
                limit_margin_deg=10.0, speed_fraction=0.5, stop_time_s=0.5,
                stop_motion_deg=5.0, still_deg_s=1.0)

SESSION_SPEED_FRACTION = {"H1": 0.25, "H2": 0.5, "LAB": 0.5}
DEFAULT_SESSION = "H2"

#: threshold shown next to each result key in the report table
THRESHOLD_TEXT = {
    "HW-2 joint RMS [deg]": ("≤", "joint_rms_deg"),
    "HW-2 joint max [deg]": ("≤", "joint_max_deg"),
    "HW-3 task RMS [mm]": ("≤", "task_rms_mm"),
    "HW-3 task max [mm]": ("≤", "task_max_mm"),
    "HW-4 limit margin [deg]": ("≥", "limit_margin_deg"),
    "HW-5 speed / rated": ("≤", "speed_fraction"),
    "HW-6 halt time [s]": ("≤", "stop_time_s"),
    "HW-6 motion after stop [deg]": ("≤", "stop_motion_deg"),
}


def speed_fraction(session=DEFAULT_SESSION) -> float:
    try:
        return SESSION_SPEED_FRACTION[str(session).upper()]
    except KeyError:
        raise ValueError(f"unknown session {session!r}; use 'H1', 'H2' or 'lab'") from None


def joint_errors(cmd, meas):
    """Per-joint RMS and maximum absolute error (same units as the inputs)."""
    e = np.asarray(meas, float) - np.asarray(cmd, float)
    return np.sqrt(np.mean(e ** 2, axis=0)), np.max(np.abs(e), axis=0)


def task_errors(Slist, M, cmd_deg, meas_deg, p_b=(0, 0, 0)):
    """Tool-point position error |p(cmd) - p(meas)| [m] with space-form PoE (MR Eq. 4.14).
    Returns (rms, max, errors)."""
    pb = np.r_[np.asarray(p_b, float), 1.0]
    err = []
    for qc, qm in zip(np.deg2rad(cmd_deg), np.deg2rad(meas_deg)):
        pc = (mr.FKinSpace(M, Slist, qc) @ pb)[:3]
        pm = (mr.FKinSpace(M, Slist, qm) @ pb)[:3]
        err.append(np.linalg.norm(pc - pm))
    err = np.array(err)
    return float(np.sqrt(np.mean(err ** 2))), float(err.max()), err


def limit_margins(meas_deg, limits_deg):
    """Smallest distance [deg] of each measured joint from its lower/upper limit."""
    lim = np.asarray(limits_deg, float)
    m = np.asarray(meas_deg, float)
    return np.minimum(m.min(axis=0) - lim[:, 0], lim[:, 1] - m.max(axis=0))


def joint_speeds(t, q):
    """Finite-difference joint speeds (units of q per second)."""
    return np.gradient(np.asarray(q, float), np.asarray(t, float), axis=0)


def stop_check(t, meas_deg, stop, still_deg_s=DEFAULTS["still_deg_s"]):
    """Safe-stop test: (time from the stop command until every joint speed stays below
    ``still_deg_s``, maximum joint motion after the command [deg])."""
    t = np.asarray(t, float)
    stop = np.asarray(stop) > 0.5
    if not stop.any():
        raise ValueError("no stop command in the log")
    k0 = int(np.argmax(stop))
    qd = np.abs(joint_speeds(t, meas_deg)).max(axis=1)
    moving = np.nonzero(qd[k0:] > still_deg_s)[0]
    t_halt = 0.0 if moving.size == 0 else t[k0 + moving[-1]] - t[k0]
    if moving.size and k0 + moving[-1] == len(t) - 1:
        t_halt = np.inf
    drift = np.max(np.abs(np.asarray(meas_deg)[k0:] - np.asarray(meas_deg)[k0]))
    return float(t_halt), float(drift)


def hw_report(log, limits_deg=None, rated_speed_deg_s=None, Slist=None, M=None,
              p_b=(0, 0, 0), thresholds=None, session=DEFAULT_SESSION) -> dict:
    """{test_id: (value, passed)} exactly as hwcompare.hw_report (tests without inputs are left out)."""
    th = dict(DEFAULTS, speed_fraction=speed_fraction(session), **(thresholds or {}))
    t, cmd, meas, stop = log["t"], log["cmd"], log["meas"], log.get("stop")
    if stop is not None and np.any(stop > 0.5):
        k0 = int(np.argmax(stop > 0.5))
    else:
        k0 = len(t)
    rms, mx = joint_errors(cmd[:k0], meas[:k0])
    res = {"HW-2 joint RMS [deg]": (float(rms.max()), bool(rms.max() <= th["joint_rms_deg"])),
           "HW-2 joint max [deg]": (float(mx.max()), bool(mx.max() <= th["joint_max_deg"]))}
    if Slist is not None and M is not None:
        trms, tmax, _ = task_errors(Slist, M, cmd[:k0], meas[:k0], p_b)
        res["HW-3 task RMS [mm]"] = (1e3 * trms, bool(1e3 * trms <= th["task_rms_mm"]))
        res["HW-3 task max [mm]"] = (1e3 * tmax, bool(1e3 * tmax <= th["task_max_mm"]))
    if limits_deg is not None:
        marg = float(limit_margins(meas, limits_deg).min())
        res["HW-4 limit margin [deg]"] = (marg, bool(marg >= th["limit_margin_deg"]))
    if rated_speed_deg_s is not None:
        ratio = np.abs(joint_speeds(t, meas)).max(axis=0) / np.asarray(rated_speed_deg_s, float)
        res["HW-5 speed / rated"] = (float(ratio.max()), bool(ratio.max() <= th["speed_fraction"]))
    if stop is not None and np.any(stop > 0.5):
        t_halt, drift = stop_check(t, meas, stop, th["still_deg_s"])
        res["HW-6 halt time [s]"] = (t_halt, bool(t_halt <= th["stop_time_s"]))
        res["HW-6 motion after stop [deg]"] = (drift, bool(drift <= th["stop_motion_deg"]))
    return res


def report_rows(res: dict, session=DEFAULT_SESSION, thresholds=None) -> list[tuple[str, str, str, bool]]:
    """(test, value, threshold, passed) rows for the report table."""
    th = dict(DEFAULTS, speed_fraction=speed_fraction(session), **(thresholds or {}))
    rows = []
    for k, (v, ok) in res.items():
        op, key = THRESHOLD_TEXT.get(k, ("", None))
        rows.append((k, "∞" if np.isinf(v) else f"{v:.3f}", f"{op} {th[key]:g}" if key else "", ok))
    return rows


# --------------------------------------------------------------------------- SBEG 461 helpers
def estimate_delay(t, a, b, max_delay: float = 0.2) -> float:
    """Lag d (s) of b behind a (b(t) ~ a(t - d)) from the normalised cross-correlation (461 hardware.py)."""
    t = np.asarray(t, float)
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    dt = float(np.median(np.diff(t)))
    kmax = int(round(max_delay / dt))
    lags = np.arange(-kmax, kmax + 1)
    n = len(a)
    score = []
    for k in lags:
        x, y = a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)]
        score.append(np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else -np.inf)
    return float(lags[int(np.argmax(score))] * dt)


def stop_time(t, qdot, t_trigger: float, threshold: float) -> float:
    """Time (s) from the trigger until |qdot| stays below ``threshold`` (461 hardware.py, HWk-4)."""
    t = np.asarray(t, float)
    v = np.max(np.abs(np.asarray(qdot, float).reshape(len(t), -1)), axis=1)
    above = np.nonzero((t >= t_trigger) & (v >= threshold))[0]
    if len(above) == 0:
        return 0.0
    k = above[-1] + 1
    return float(t[k] - t_trigger) if k < len(t) else float("inf")


def session_report(log, robot, spec, session=DEFAULT_SESSION, p_b=(0, 0, 0)) -> dict:
    """HW-2 ... HW-6 for a ``HardwareLog`` of the course arm (hardware limits, rated speed, PoE model)."""
    return hw_report(log.as_deg(), limits_deg=spec.hw_limits_deg, rated_speed_deg_s=spec.rated_speed_deg_s,
                     Slist=robot.Slist, M=robot.M, p_b=p_b, session=session)


# --------------------------------------------------------------------------- validation
def validation_checks() -> list[Check]:
    from .base import COURSE_ARM, course_arm_robot
    from .logger import HardwareLog, read_log_361, read_log_461
    from .safety import SafetySupervisor
    from .session import quintic_move, run_offline
    from .simulated import ManualClock, SimulatedArm

    out: list[Check] = []
    robot, _ = course_arm_robot()
    spec = COURSE_ARM
    rng = np.random.default_rng(3)

    # 1-2. log formats round-trip (361: %.4f degrees; 461: full precision SI)
    t = np.arange(0, 1, 0.02)
    cmd = spec.q_home + 0.3 * np.sin(t[:, None] * (1 + np.arange(6)))
    meas = cmd + rng.normal(0, 1e-3, cmd.shape)
    lg = HardwareLog(6)
    for k in range(len(t)):
        lg.append(t[k], cmd[k], meas[k], stop=k >= 40)
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
        lg.save(p1, "361")
        lg.save(p2, "461")
        r1, r2 = read_log_361(p1), read_log_461(p2)
    e1 = max(np.abs(r1["cmd"] - np.degrees(cmd)).max(), np.abs(r1["meas"] - np.degrees(meas)).max(),
             np.abs(r1["stop"] - lg.stop).max())
    out.append(Check("Log, SBEG 361 format: write → read (deg, %.4f, stop column)", "Course HW log",
                     float(e1), 5.1e-5, "hwcompare.write_log/read_log format; rounding of %.4f is ≤ 5e-5°",
                     CATEGORY))
    e2 = max(np.abs(r2["cmd"] - cmd).max(), np.abs(r2["meas"] - meas).max(), np.abs(r2["t"] - t).max())
    out.append(Check("Log, SBEG 461 format: write → read (SI, full precision)", "Course HW log", float(e2), 1e-15,
                     "hardware.write_log/load_log format (rad)", CATEGORY))

    # 3. HW-3 task error vs first-order prediction |J_v δθ| (MR 5.1: p-dot = J_v θ-dot)
    from .. import kinematics as kin
    worst = 0.0
    for _ in range(5):
        th = robot.random_config(rng) * 0.8
        dth = rng.normal(0, 1e-5, 6)
        _, _, err = task_errors(robot.Slist, robot.M, np.degrees(th)[None], np.degrees(th + dth)[None])
        Jv = kin.point_jacobian(robot, th)
        pred = np.linalg.norm(Jv @ dth)
        worst = max(worst, abs(err[0] - pred) / pred)
    out.append(Check("HW-3 tool-point error (PoE FK) = |J_v·δθ| to first order (course arm)", "MR 4.1 / 5.1",
                     worst, 1e-3, "|p(θ+δθ)−p(θ)| from mr.FKinSpace vs the point Jacobian, δθ ~ 1e-5 rad",
                     CATEGORY))

    # 4. quintic peak speed = 15/(8T)|Δθ| (MR Fig. 9.4) via the same finite differences as HW-5
    q0, q1 = spec.q_home, spec.q_home + np.radians([30, -20, 25, 10, -15, 40])
    tq, qq = quintic_move(q0, q1, 0.25 * spec.rated_speed, dt=0.001)
    T = tq[-1]
    peak = np.abs(joint_speeds(tq, qq)).max(axis=0)
    rel = np.max(np.abs(peak - 15 / (8 * T) * np.abs(q1 - q0)) / (15 / (8 * T) * np.abs(q1 - q0)))
    out.append(Check("Quintic time scaling: peak joint speed = 15|Δθ|/(8T)", "MR 9.2.2.1 (Fig. 9.4)",
                     float(rel), 1e-5, f"mr.JointTrajectory(method=5), T = {T:.2f} s, finite differences",
                     CATEGORY))

    # 5. simulated servo = first-order error dynamics: ramp lag = v·t_c (+ half a sample of ZOH)
    clock = ManualClock()
    arm = SimulatedArm(noise_deg=0.0, quantise=False, clock=clock, q0=np.zeros(6))
    arm.connect()
    v, rate = np.radians(10.0), 200.0
    for k in range(1, 400):
        clock.advance(1 / rate)
        arm.send_joints(np.full(6, v * clock.t), 0.5)
    lag = v * clock.t - arm.true_position()[0]
    pred = v * (arm.lag_tau + 0.5 / rate)
    out.append(Check("Simulated servo: ramp tracking lag = v(t_c + Δt/2)", "MR 11.2.2.1 Eqs. 11.6–11.7",
                     float(abs(lag - pred) / pred), 0.02,
                     f"first-order lag t_c = {arm.lag_tau} s, v = 10°/s, zero-order hold at {rate:.0f} Hz",
                     CATEGORY))

    # 6-8. a full simulated H1 session (quintic at the 25 % speed cap, automatic stop at mid-motion)
    clock = ManualClock()
    arm = SimulatedArm(clock=clock)
    arm.connect()
    sup = SafetySupervisor(session="H1", first_run=True)
    dq = np.radians([40, 20, -30, 20, 30, -40])
    tt, qt = quintic_move(spec.q_home, spec.q_home + dq, sup.speed_cap)
    T = tt[-1]
    run = run_offline(arm, sup, tt, qt, auto_stop_at=0.5 * T)
    res = session_report(run.log, robot, spec, "H1")
    out.append(Check("Simulated H1 session: HW-5 peak measured speed ≤ 25 % of rated", "Course HW tests (rule 3)",
                     res["HW-5 speed / rated"][0], 0.25,
                     f"quintic T = {T:.2f} s planned at 15|Δθ|/(8T) ≤ 0.25·160°/s / 1.25; measured incl. noise",
                     CATEGORY))
    # tracking error predicted by first-order error dynamics: e ≈ θ̇_cmd·(t_c + Δt/2)  (MR 11.2.2.1)
    lg = run.log
    k0 = int(np.argmax(lg.stop)) if lg.stop.any() else len(lg)
    tl, cl, ml = lg.t[:k0], np.degrees(lg.cmd[:k0]), np.degrees(lg.meas[:k0])
    e_meas = np.sqrt(np.mean((ml - cl) ** 2, axis=0))
    e_pred = np.sqrt(np.mean((joint_speeds(tl, cl) * (arm.lag_tau + 0.5 * 0.02)) ** 2, axis=0))
    rel = float(np.max(np.abs(e_meas - e_pred) / e_pred))
    out.append(Check("Simulated H1 session: HW-2 joint RMS error = RMS(θ̇_cmd·(t_c+Δt/2))", "MR 11.2.2.1 Eq. 11.7",
                     rel, 0.15, f"measured {np.array2string(e_meas, precision=2)}° vs first-order prediction "
                     f"{np.array2string(e_pred, precision=2)}° (noise 0.05°, 12-bit quantisation)", CATEGORY))
    # stop at mid-motion (peak speed, zero acceleration): drift = |θ̇(T/2)| τ_s (1 − e^{−1/τ_s})
    # with a zero-order hold of Δt the servo speed just before the next set-point is
    # θ̇·Δt·a/((1−a)t_c), a = e^(−Δt/t_c)  (exact solution of Eq. 11.6 over one sample)
    a = np.exp(-0.02 / arm.lag_tau)
    v_mid = 15 / (8 * T) * np.degrees(np.abs(dq)).max() * 0.02 * a / ((1 - a) * arm.lag_tau)
    drift_pred = v_mid * arm.stop_tau * (1 - np.exp(-run.LOG_AFTER_STOP / arm.stop_tau))
    drift = res["HW-6 motion after stop [deg]"][0]
    out.append(Check("Simulated stop: HW-6 motion after stop = θ̇·τ_s (exponential deceleration)",
                     "Course HW tests (rule 6)", float(abs(drift - drift_pred) / drift_pred), 0.15,
                     f"{drift:.3f}° measured vs {drift_pred:.3f}° predicted; halt time "
                     f"{res['HW-6 halt time [s]'][0]:.2f} s (τ_s = {arm.stop_tau} s)", CATEGORY))
    return out
