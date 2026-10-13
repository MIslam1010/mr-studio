"""Trajectory sources and the (GUI-independent) run loop of a hardware session.

* ``quintic_move`` -- straight-line joint path with the quintic time scaling (MR Sec. 9.2.2.1,
  ``mr.JointTrajectory(..., method=5)``).  Its peak speed is 15/(8T)|Δθ_i| (MR Fig. 9.4, p. 333), so the
  shortest duration that respects the allowed joint speed f·θ̇_rated is
  T = 15 max_i |Δθ_i| / (8 f θ̇_rated,i)   (then rounded up with a safety factor).
* ``HardwareRun`` -- streams a sampled trajectory to an ``ArmDriver`` at a fixed rate (>= 20 Hz as
  the briefs require), reads the joints back and logs (t, cmd, meas, stop).  It does not own a timer:
  the GUI calls ``step()`` from a QTimer, tests call it in a loop with a ``ManualClock``.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

from .base import ArmDriver
from .logger import HardwareLog
from .safety import SafetySupervisor, retime_to_speed


def quintic_duration(q0, q1, vmax, margin: float = 1.25, T_min: float = 0.5) -> float:
    """Shortest quintic duration with peak speed 15/(8T)|Δθ| <= vmax/margin (MR Fig. 9.4)."""
    dq = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    T = 15.0 / 8.0 * float(np.max(dq / np.asarray(vmax, float))) * margin
    return max(T, T_min)


def quintic_move(q0, q1, vmax, dt: float = 0.02, T: float | None = None):
    """(t, q) samples of the rest-to-rest quintic joint motion q0 -> q1 (MR Sec. 9.2.2.1)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if T is None:
        T = quintic_duration(q0, q1, vmax)
    N = max(2, int(np.ceil(T / dt)) + 1)
    q = np.array(mr.JointTrajectory(q0, q1, T, N, 5))
    return np.linspace(0.0, T, N), q


def concat(*segments, pause: float = 0.0):
    """Join (t, q) segments end to start, with ``pause`` seconds of hold between them."""
    ts, qs, t_off = [], [], 0.0
    for i, (t, q) in enumerate(segments):
        t = np.asarray(t, float) - t[0]
        if i:
            t_off += pause
            t = t[1:] if pause == 0 else t
            q = q[1:] if pause == 0 else q
        ts.append(t + t_off)
        qs.append(np.asarray(q, float))
        t_off = ts[-1][-1]
    return np.concatenate(ts), np.vstack(qs)


def with_approach(t, q, q_now, vmax, tol: float = np.radians(1.0), pause: float = 0.3):
    """Prepend a quintic approach from the measured pose when the trajectory starts elsewhere."""
    if np.max(np.abs(np.asarray(q[0]) - np.asarray(q_now))) <= tol:
        return np.asarray(t, float) - t[0], np.asarray(q, float), False
    ta, qa = quintic_move(q_now, q[0], vmax)
    tt, qq = concat((ta, qa), (t, q), pause=pause)
    return tt, qq, True


def sample(t, q, tq: float) -> np.ndarray:
    """Linear interpolation of the sampled trajectory at time tq (held at the ends)."""
    return np.array([np.interp(tq, t, q[:, j]) for j in range(q.shape[1])])


class HardwareRun:
    """One run: trajectory -> driver, with logging and the stop logic (HW-6)."""
    HOLD_AFTER = 0.5          # s of logging after the trajectory ends
    LOG_AFTER_STOP = 1.0      # s of logging after a stop (deceleration for HW-6)

    def __init__(self, driver: ArmDriver, supervisor: SafetySupervisor, t, q, *,
                 auto_stop_at: float | None = None, clock=None):
        self.driver, self.sup = driver, supervisor
        self.t_traj, self.q_traj = np.asarray(t, float), np.asarray(q, float)
        self.auto_stop_at = auto_stop_at if auto_stop_at and auto_stop_at > 0 else None
        self.clock = clock if clock is not None else getattr(driver, "clock", None)
        if self.clock is None:
            import time
            self.clock = time.monotonic
        self.log = HardwareLog(driver.spec.n)
        self.t0 = None
        self.t_stop = None
        self.finished = False
        self.reason = ""

    @property
    def duration(self) -> float:
        return float(self.t_traj[-1])

    def start(self) -> None:
        self.t0 = self.clock()
        self.step()

    def request_stop(self, reason: str = "stop pressed") -> None:
        """Stop button / e-stop: latch the driver now; keep logging the deceleration."""
        self.driver.stop()
        if self.t_stop is None and self.t0 is not None:
            self.t_stop = self.clock() - self.t0
            self.reason = reason

    def step(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        if self.finished or self.t0 is None:
            return None
        t = self.clock() - self.t0
        if self.auto_stop_at is not None and self.t_stop is None and t >= self.auto_stop_at:
            self.request_stop(f"automatic stop test at t = {self.auto_stop_at:.2f} s")
        stopped = self.t_stop is not None
        if stopped:
            cmd = self.log.cmd[-1] if len(self.log) else sample(self.t_traj, self.q_traj, t)
        else:
            cmd = self.sup.clamp(sample(self.t_traj, self.q_traj, t))
            self.driver.send_joints(cmd, self.sup.speed_fraction)
        meas = self.driver.read_joints()
        self.log.append(t, cmd, meas, stop=stopped)
        if stopped and t - self.t_stop >= self.LOG_AFTER_STOP:
            self.finished = True
        elif not stopped and t >= self.duration + self.HOLD_AFTER:
            self.finished = True
            self.reason = "trajectory complete"
        return t, cmd, meas


def run_offline(driver, supervisor, t, q, *, rate: float = 50.0, auto_stop_at=None) -> HardwareRun:
    """Run a whole session against a driver whose clock is a ``ManualClock`` (tests, Validation)."""
    clock = driver.clock
    run = HardwareRun(driver, supervisor, t, q, auto_stop_at=auto_stop_at, clock=clock)
    run.start()
    while not run.finished:
        clock.advance(1.0 / rate)
        run.step()
    return run


__all__ = ["quintic_duration", "quintic_move", "concat", "with_approach", "sample", "HardwareRun",
           "run_offline", "retime_to_speed"]
