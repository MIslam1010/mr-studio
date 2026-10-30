"""SimulatedArm: a stand-in for the course arm so the whole hardware workflow runs without hardware.

Servo model (per joint, position-controlled smart servo):

* **first-order lag** towards the set-point, MR Sec. 11.2.2.1 (Eqs. 11.6-11.7):
  theta_e' + theta_e / t_c = 0  ->  theta_e(t) = exp(-t / t_c) theta_e(0),
  integrated *exactly* between samples (no Euler error), so a ramp of speed v is tracked with the
  steady-state lag v * t_c;
* the step per sub-interval is **saturated** at ``speed * rated_speed`` (the speed passed to
  ``send_joints``);
* **hardware limits** clamp the true position;
* the reading is **quantised** to the encoder resolution (12 bit = 0.088 deg) after adding Gaussian
  **noise** while the joint moves (servo vibration; a joint at rest reads a steady count);
* ``stop()`` latches: the joints coast to rest with the deceleration time constant ``stop_tau``
  (q(t) = q0 + v0 * stop_tau * (1 - exp(-t / stop_tau)), the model used by hwcompare.synthetic_log);
* **torque off** (latched, rule 6): J2 and J3 sag slowly (``sag_deg_s``) as a real arm does under
  gravity, so practice with the simulation teaches to support the arm before releasing torque.  The
  direction is illustrative (which way the tool falls depends on the pose); there is no gravity model.

Time comes from ``clock()`` (default ``time.monotonic``); tests pass a manual clock so runs are
deterministic and instantaneous.
"""
from __future__ import annotations

import time

import numpy as np

from .base import COURSE_ARM, ArmDriver, ArmSpec


class ManualClock:
    """A clock you advance by hand (tests and fast offline sessions)."""

    def __init__(self, t0: float = 0.0):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


class SimulatedArm(ArmDriver):
    name = "Simulated"

    def __init__(self, spec: ArmSpec = COURSE_ARM, *, lag_tau: float = 0.04, stop_tau: float = 0.06,
                 noise_deg: float = 0.05, quantise: bool = True, seed: int | None = 0,
                 q0=None, clock=None, substep: float = 1e-3, sag_deg_s: float = 3.0):
        super().__init__(spec)
        self.lag_tau = float(lag_tau)
        self.stop_tau = float(stop_tau)
        self.noise = np.radians(noise_deg)
        self.quantise = quantise
        self.rng = np.random.default_rng(seed)
        self.clock = clock if clock is not None else time.monotonic
        self.substep = float(substep)
        self.q = np.array(spec.q_home if q0 is None else q0, float).copy()
        self.v = np.zeros(spec.n)
        self.target = self.q.copy()
        self.vmax = 0.25 * spec.rated_speed
        self.sag = np.zeros(spec.n)
        if spec.n >= 3:
            self.sag[1:3] = -np.radians(sag_deg_s)     # J2, J3 fall while torque is off
        self._t_last = None

    # -- ArmDriver ---------------------------------------------------------------------------
    def connect(self) -> None:
        self.connected = True
        self._t_last = self.clock()

    def read_joints(self) -> np.ndarray:
        self._advance()
        q = self.q.copy()
        moving = np.abs(self.v) > np.radians(0.5)
        if self.noise > 0 and moving.any():
            q[moving] += self.rng.normal(0.0, self.noise, int(moving.sum()))
        if self.quantise:
            q = np.round(q / self.spec.enc_res) * self.spec.enc_res
        self.last_meas_servo_deg = np.degrees(q)
        self._mark_read(True)
        return q

    def _send(self, q, speed) -> None:
        self._advance()
        lo, hi = self.spec.hw_limits.T
        self.target = np.clip(q, lo, hi)
        self.vmax = speed * self.spec.rated_speed
        self.last_cmd_servo_deg = np.degrees(self.target)

    def stop(self) -> bool:
        if self.connected:
            self._advance()    # integrate up to the stop instant with the servo law, then coast
        return super().stop()

    def _halt(self) -> bool:
        return True

    def _set_torque(self, on: bool) -> None:
        self._advance()        # integrate the coast (or the sag) up to the switching instant

    def reset_latch(self) -> None:
        self._advance()
        super().reset_latch()
        self.target = self.q.copy()
        self.v[:] = 0.0

    # -- physics -------------------------------------------------------------------------------
    def true_position(self) -> np.ndarray:
        self._advance()
        return self.q.copy()

    def _advance(self) -> None:
        now = self.clock()
        if self._t_last is None:
            self._t_last = now
            return
        T = now - self._t_last
        self._t_last = now
        if T <= 0:
            return
        k = max(1, int(np.ceil(T / self.substep)))
        h = T / k
        lo, hi = self.spec.hw_limits.T
        if self._latched or not self._torque:
            if not self._torque:          # torque off: J2/J3 sag at a constant slow rate
                self.q = np.clip(self.q + self.sag * T, lo, hi)
                self.v = self.sag.copy()
                return
            decay = np.exp(-T / self.stop_tau)
            self.q = np.clip(self.q + self.v * self.stop_tau * (1 - decay), lo, hi)
            self.v *= decay
            return
        a = np.exp(-h / self.lag_tau)
        for _ in range(k):
            dq = (self.target - self.q) * (1 - a)                       # exact first-order step
            dq = np.clip(dq, -self.vmax * h, self.vmax * h)             # speed saturation
            self.q = np.clip(self.q + dq, lo, hi)
            self.v = dq / h
