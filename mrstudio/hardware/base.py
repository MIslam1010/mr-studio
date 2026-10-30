"""Hardware-bridge foundations: the course-arm specification and the ``ArmDriver`` interface.

Every driver speaks the same language as the rest of MR Studio: joint angles in **radians**, in the
PoE convention of the model (MR Sec. 4.1: theta = 0 is the home configuration M).  A driver for a
commercial arm converts to and from its servo convention internally (degrees, servo zero offset).

The numbers of ``COURSE_ARM`` are those of the shared course-robot library
(``_build/shared/course_robots.py``: ``COURSE_ARM_GEOMETRY``, ``COURSE_ARM_LIMITS_DEG``,
``COURSE_ARM_RATED_SPEED_DEG_S``, ``COURSE_ARM_PAYLOAD_MAX``) -- illustrative values until the arms
are commissioned.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

DEG = np.pi / 180.0

#: Name under which the course arm is registered in ``mrstudio.robots.ROBOTS`` (by the robots builder).
COURSE_ARM_NAME = "Course arm (myCobot-280 class 6R)"

#: MR Example 4.5 (UR-type) layout scaled to the myCobot-280 class (course_robots.py COURSE_ARM_GEOMETRY).
COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)


class HardwareError(RuntimeError):
    """A driver could not do what was asked (missing SDK, no device, communication failure)."""


@dataclass
class ArmSpec:
    """What the safety layer and the metrics need to know about an arm."""
    name: str
    hw_limits: np.ndarray                 # n x 2, rad (the servo / mechanical limits)
    rated_speed: np.ndarray               # n, rad/s
    payload_max: float                    # kg (rated)
    reach: float                          # m
    enc_res: float                        # rad per encoder count
    q_home: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        self.hw_limits = np.asarray(self.hw_limits, float).reshape(-1, 2)
        self.rated_speed = np.broadcast_to(np.asarray(self.rated_speed, float), (self.n,)).copy()
        self.q_home = np.asarray(self.q_home, float)

    @property
    def n(self) -> int:
        return self.hw_limits.shape[0]

    @property
    def hw_limits_deg(self) -> np.ndarray:
        return np.degrees(self.hw_limits)

    @property
    def rated_speed_deg_s(self) -> np.ndarray:
        return np.degrees(self.rated_speed)


COURSE_ARM = ArmSpec(
    name=COURSE_ARM_NAME,
    hw_limits=np.array([(-165, 165)] * 5 + [(-175, 175)], float) * DEG,
    rated_speed=np.full(6, 160.0 * DEG),           # 160 deg/s, illustrative
    payload_max=0.25,                              # kg, illustrative
    reach=0.28,
    enc_res=2 * np.pi / 4096,                      # 12-bit encoders (0.088 deg)
    q_home=np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0]),
)


def _course_arm_fallback():
    """Kinematic-only course arm (MR Example 4.5 screw table, course geometry) used when the robots
    builder has not registered ``COURSE_ARM_NAME`` yet."""
    from ..robots import SerialRobot
    g = COURSE_ARM_GEOMETRY
    W1, W2, L1, L2, H1, H2 = (g[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    return SerialRobot(name=COURSE_ARM_NAME + " — kinematic fallback", M=M, Slist=S,
                       joint_limits=COURSE_ARM.hw_limits.copy(), home=COURSE_ARM.q_home.copy(),
                       description="Desktop UR-type 6R (myCobot-280 class): MR Example 4.5 layout with "
                                   "W1=63.4, W2=45.6, L1=110, L2=96, H1=131.2, H2=75.1 mm.",
                       source="SBEG 361/461 course_robots.py course_arm() (kinematics only)",
                       link_radius=0.012)


def course_arm_robot():
    """The course-arm SerialRobot: the registered model if present, else the kinematic fallback.
    Returns (robot, is_fallback)."""
    from ..robots import ROBOTS, get_robot
    if COURSE_ARM_NAME in ROBOTS:
        try:
            return get_robot(COURSE_ARM_NAME), False
        except Exception:   # a broken registration must not take the hardware tab down
            pass
    return _course_arm_fallback(), True


class ArmDriver(ABC):
    """Common interface of every arm back-end (simulated or real).

    * ``connect()`` / ``disconnect()``
    * ``read_joints() -> q``          measured joint angles [rad], model (PoE) convention.  ``last_read_fresh``
                                      tells whether that sample is new; ``fresh_age()`` is the time since the
                                      last fresh one (a stale stream must not start or keep a run going).
    * ``send_joints(q, speed)``       position set-point q [rad]; ``speed`` = fraction (0, 1] of the rated
                                      joint speed the servos may use.  Returns False (and does nothing)
                                      while the driver is latched; raises ``HardwareError`` on a non-finite q.
    * ``stop() -> bool``              halt now and enter the **latched stop** (checklist rules 6–7): every
                                      later ``send_joints`` is ignored until ``reset_latch()``.  True only when
                                      the halt command went out without error (one retry on failure).
    * ``latch()``                     alias of ``stop()``;  ``latched`` tells the state.
    * ``set_torque(on)``              disable / enable servo torque (only allowed while latched).

    Threads: a run streams from a worker thread while STOP comes from the GUI thread.  The latch is a
    ``threading.Event`` that ``stop()`` sets *before* it takes the I/O lock, and ``send_joints`` checks it
    under that lock just before every write, so no set-point can follow a STOP.  The halt itself is sent by
    whichever thread takes the lock first (``ensure_halted``), ahead of any further read.
    """
    name = "driver"

    def __init__(self, spec: ArmSpec = COURSE_ARM):
        self.spec = spec
        self._stop_evt = threading.Event()
        self._halt_ok: bool | None = None         # outcome of the halt of the current latch (None = not sent)
        self._torque = True
        self.connected = False
        self.io_lock = threading.RLock()          # guards the SDK / serial object
        self.clock = time.monotonic
        self.read_failures = 0                    # failed reads since connect (shown in the tab)
        self.consecutive_read_failures = 0
        self.last_read_fresh = False
        self.last_fresh_t: float | None = None
        self.last_cmd_servo_deg: np.ndarray | None = None    # raw servo set-point of the last write [deg]
        self.last_meas_servo_deg: np.ndarray | None = None   # raw servo reading of the last read [deg]

    @property
    def _latched(self) -> bool:
        return self._stop_evt.is_set()

    @_latched.setter
    def _latched(self, on: bool) -> None:
        if on:
            self._stop_evt.set()
        else:
            self._stop_evt.clear()

    # -- connection ---------------------------------------------------------------------------
    @abstractmethod
    def connect(self) -> None: ...

    def disconnect(self) -> None:
        self.connected = False

    def describe(self) -> dict:
        """Connection and calibration facts for the log sidecar."""
        return dict(driver=self.name)

    # -- reads -----------------------------------------------------------------------------------
    @abstractmethod
    def read_joints(self) -> np.ndarray: ...

    def _mark_read(self, fresh: bool) -> None:
        self.last_read_fresh = bool(fresh)
        if fresh:
            self.consecutive_read_failures = 0
            self.last_fresh_t = self.clock()
        else:
            self.read_failures += 1
            self.consecutive_read_failures += 1

    def fresh_age(self) -> float:
        """Seconds since the last fresh joint reading (inf before the first one)."""
        if self.last_fresh_t is None:
            return float("inf")
        return max(0.0, self.clock() - self.last_fresh_t)

    # -- motion -------------------------------------------------------------------------------
    @abstractmethod
    def _send(self, q: np.ndarray, speed: float) -> None: ...

    @abstractmethod
    def _halt(self) -> bool:
        """Send the halt command; True when it went out without error."""

    def send_joints(self, q, speed: float) -> bool:
        q = np.asarray(q, float).reshape(self.spec.n)
        if not np.all(np.isfinite(q)):
            raise HardwareError(f"non-finite joint command rejected: {q}")
        speed = float(np.clip(speed, 0.0, 1.0))
        if speed <= 0:
            return False
        with self.io_lock:
            if self._latched or not self.connected:      # checked under the lock, before every write
                return False
            self._send(q, speed)
        return True

    def stop(self) -> bool:
        """Halt and latch (safe even when not connected).  True when the halt is confirmed."""
        self._latched = True
        return self.ensure_halted()

    def ensure_halted(self) -> bool:
        """Send the halt of the current latch unless it already went out (one retry on failure)."""
        if not self._latched or not self.connected:
            return True                 # nothing commanded through this driver is still running
        with self.io_lock:
            if self._halt_ok:
                return True
            ok = False
            for _ in range(2):
                try:
                    ok = self._halt() is not False
                except Exception:
                    ok = False
                if ok:
                    break
            self._halt_ok = ok
            return ok

    @property
    def halt_confirmed(self) -> bool | None:
        """True / False = outcome of the halt of the current latch, None = not latched or not sent yet."""
        return self._halt_ok

    def latch(self) -> bool:
        return self.stop()

    @property
    def latched(self) -> bool:
        return self._latched

    def reset_latch(self) -> None:
        if not self._torque:
            self.set_torque(True)
        self._halt_ok = None
        self._latched = False

    # -- torque ------------------------------------------------------------------------------
    @property
    def torque_enabled(self) -> bool:
        return self._torque

    def set_torque(self, on: bool) -> None:
        if not on and not self._latched:
            raise HardwareError("torque may only be disabled in latched stop (checklist rule 6)")
        with self.io_lock:
            self._set_torque(bool(on))
        self._torque = bool(on)          # only once the servos accepted the command

    def _set_torque(self, on: bool) -> None:
        pass
