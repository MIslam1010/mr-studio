"""Trajectory sources and the (GUI-independent) run loop of a hardware session.

* ``quintic_move`` -- straight-line joint path with the quintic time scaling (MR Sec. 9.2.2.1,
  ``mr.JointTrajectory(..., method=5)``).  Its peak speed is 15/(8T)|Δθ_i| (MR Fig. 9.4, p. 333), so the
  shortest duration that respects the allowed joint speed f·θ̇_rated is
  T = 15 max_i |Δθ_i| / (8 f θ̇_rated,i)   (then rounded up with a safety factor).
* ``HardwareRun`` -- streams a sampled trajectory to an ``ArmDriver`` at a fixed rate (>= 20 Hz as
  the briefs require), reads the joints back and logs (t, cmd, meas, stop) plus diagnostics (fresh flag,
  tick dt, raw servo degrees, events).  It does not own a timer: the GUI's streaming worker thread calls
  ``safe_step()`` at the stream rate; tests (and the GUI with a ``ManualClock``) call it in a loop.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

import modern_robotics as mr
import numpy as np

from .base import ArmDriver, HardwareError
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


class LatestValue:
    """Thread-safe latest-value mailbox: ``put`` overwrites, ``take`` returns the newest value (or None)
    and empties the box.  The streaming worker publishes samples through it, so a slow GUI only ever
    draws the newest sample and never queues up stale ones."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def put(self, value) -> None:
        with self._lock:
            self._value = value

    def take(self):
        with self._lock:
            v, self._value = self._value, None
            return v


class HardwareRun:
    """One run: trajectory -> driver, with logging, the stop logic (HW-6) and the run-time guards.

    Guards (on top of the trajectory checks of ``SafetySupervisor.preflight``):

    * **step clamp** -- |cmd_k − cmd_k−1| <= speed_cap · min(dt, 40 ms) per joint, whatever the trajectory
      or the timing does (the first reference is the measured pose at start);
    * **stall freeze** -- a tick gap > 100 ms (GUI or USB stall) does not advance the trajectory time, so
      the set-point never jumps ahead to catch up with the wall clock;
    * **tracking watchdog** -- |cmd − meas| > 10° on any joint for > 150 ms stops and latches the arm;
    * **communication errors** -- ``safe_step`` turns any exception into a latched stop and ends the run.

    ``step()`` is called by one thread at a time (the streaming worker, or the GUI timer / a test loop);
    ``request_stop()`` may come from any thread.
    """
    HOLD_AFTER = 0.5          # s of logging after the trajectory ends
    LOG_AFTER_STOP = 1.0      # s of logging after a stop (deceleration for HW-6)
    STEP_DT_MAX = 0.04        # s: the per-tick step clamp allows at most speed_cap * 40 ms
    STALL_GAP = 0.10          # s: a longer tick gap freezes the trajectory time
    TRACK_ERR = np.radians(10.0)
    TRACK_TIME = 0.15         # s the tracking error may exceed TRACK_ERR

    def __init__(self, driver: ArmDriver, supervisor: SafetySupervisor, t, q, *,
                 auto_stop_at: float | None = None, clock=None):
        self.driver, self.sup = driver, supervisor
        self.t_traj, self.q_traj = np.asarray(t, float), np.asarray(q, float)
        self.auto_stop_at = auto_stop_at if auto_stop_at and auto_stop_at > 0 else None
        self.clock = clock if clock is not None else getattr(driver, "clock", None)
        if self.clock is None:
            self.clock = time.monotonic
        self.log = HardwareLog(driver.spec.n)
        self.t0 = None
        self.t_stop = None
        self.finished = False
        self.reason = ""
        self.error: str | None = None             # set when the run ended on an exception
        self.stop_confirmed: bool | None = None   # outcome of the last halt request
        self.tau = 0.0                            # trajectory time (frozen during stalls)
        self.stalls = 0
        self._t_prev = 0.0
        self._cmd_prev = None
        self._err_since = np.full(driver.spec.n, np.nan)
        self._stop_lock = threading.Lock()

    @property
    def duration(self) -> float:
        return float(self.t_traj[-1])

    def event(self, kind: str, msg: str = "") -> None:
        t = None if self.t0 is None else round(self.clock() - self.t0, 4)
        self.log.events.append(dict(t=t, wall=datetime.now().isoformat(timespec="milliseconds"),
                                    kind=kind, msg=msg))

    def start(self, q_now=None) -> None:
        """Start the clock and send the first set-point.  ``q_now`` (measured pose) seeds the step clamp."""
        self.t0 = self.clock()
        self._t_prev = 0.0
        self._cmd_prev = None if q_now is None else np.asarray(q_now, float).copy()
        self.event("start", f"duration {self.duration:.2f} s, speed cap {100 * self.sup.speed_fraction:.0f} %")
        self.safe_step()

    def request_stop(self, reason: str = "stop pressed") -> bool:
        """Stop button / e-stop / guard: latch the driver now; keep logging the deceleration.
        Returns whether the halt reached the arm (never raises)."""
        with self._stop_lock:
            first = self.t_stop is None and self.t0 is not None
            if first:
                self.t_stop = self.clock() - self.t0
                self.reason = reason
            elif self.t0 is None and not self.finished:     # stopped before the worker started it
                self.finished = True
                self.reason = reason
        try:
            ok = bool(self.driver.stop())
        except Exception:
            ok = False
        self.stop_confirmed = ok
        if first or not ok:
            self.event("stop", reason + ("" if ok else " — HALT NOT CONFIRMED"))
        return ok

    def fail(self, exc: BaseException) -> None:
        """End the run on an exception: latch (best effort) and stop logging."""
        msg = str(exc) if isinstance(exc, HardwareError) else f"{type(exc).__name__}: {exc}"
        self.error = msg
        self.event("error", msg)
        self.request_stop(f"communication error: {msg}")
        self.finished = True

    def safe_step(self):
        """``step()`` that never raises: any exception latches the arm and finishes the run."""
        try:
            return self.step()
        except Exception as e:
            self.fail(e)
            return None

    def step(self) -> tuple[float, np.ndarray, np.ndarray] | None:
        if self.finished or self.t0 is None:
            return None
        t = self.clock() - self.t0
        gap = t - self._t_prev
        self._t_prev = t
        if gap > self.STALL_GAP:
            self.stalls += 1
            self.event("stall", f"tick gap {1e3 * gap:.0f} ms: trajectory time held at {self.tau:.3f} s")
        elif gap > 0:
            self.tau += gap
        if self.auto_stop_at is not None and self.t_stop is None and self.tau >= self.auto_stop_at:
            self.request_stop(f"automatic stop test at t = {self.auto_stop_at:.2f} s")
        stopped = self.t_stop is not None
        if stopped:
            if self.driver.ensure_halted() and self.stop_confirmed is False:   # halt before any further read
                self.stop_confirmed = True                                        # (a failed one is retried)
                self.event("stop", "halt confirmed on retry")
            if self._cmd_prev is not None:
                cmd = self._cmd_prev
            else:
                cmd = self.log.cmd[-1] if len(self.log) else sample(self.t_traj, self.q_traj, self.tau)
        else:
            cmd = self.sup.clamp(sample(self.t_traj, self.q_traj, self.tau))
            if self._cmd_prev is not None:
                lim = self.sup.speed_cap * min(max(gap, 0.0), self.STEP_DT_MAX)
                cmd = self._cmd_prev + np.clip(cmd - self._cmd_prev, -lim, lim)
            self.driver.send_joints(cmd, self.sup.speed_fraction)
            self._cmd_prev = cmd
        meas = self.driver.read_joints()
        fresh = bool(getattr(self.driver, "last_read_fresh", True))
        self.log.append(t, cmd, meas, stop=stopped, fresh=fresh, dt=gap,
                        servo_cmd=getattr(self.driver, "last_cmd_servo_deg", None),
                        servo_meas=getattr(self.driver, "last_meas_servo_deg", None) if fresh else None)
        if not stopped:
            self._watch_tracking(t, cmd, meas)
            stopped = self.t_stop is not None
        if stopped and t - self.t_stop >= self.LOG_AFTER_STOP:
            self.finished = True
        elif not stopped and self.tau >= self.duration + self.HOLD_AFTER:
            self.finished = True
            self.reason = "trajectory complete"
            self.event("complete")
        return t, cmd, meas

    def _watch_tracking(self, t, cmd, meas) -> None:
        over = np.abs(np.asarray(cmd) - np.asarray(meas)) > self.TRACK_ERR
        self._err_since = np.where(over, np.where(np.isnan(self._err_since), t, self._err_since), np.nan)
        late = over & (t - self._err_since > self.TRACK_TIME)
        if late.any():
            j = int(np.argmax(late))
            self.request_stop(f"tracking error J{j + 1} = {np.degrees(abs(cmd[j] - meas[j])):.1f}° > "
                              f"{np.degrees(self.TRACK_ERR):.0f}° for > {1e3 * self.TRACK_TIME:.0f} ms")


def run_offline(driver, supervisor, t, q, *, rate: float = 50.0, auto_stop_at=None) -> HardwareRun:
    """Run a whole session against a driver whose clock is a ``ManualClock`` (tests, Validation)."""
    clock = driver.clock
    run = HardwareRun(driver, supervisor, t, q, auto_stop_at=auto_stop_at, clock=clock)
    run.start(driver.read_joints())
    while not run.finished:
        clock.advance(1.0 / rate)
        run.safe_step()
    return run


__all__ = ["quintic_duration", "quintic_move", "concat", "with_approach", "sample", "HardwareRun",
           "LatestValue", "run_offline", "retime_to_speed"]
