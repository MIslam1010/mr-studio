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
    * ``read_joints() -> q``          measured joint angles [rad], model (PoE) convention
    * ``send_joints(q, speed)``       position set-point q [rad]; ``speed`` = fraction (0, 1] of the rated
                                      joint speed the servos may use.  Returns False (and does nothing)
                                      while the driver is latched.
    * ``stop()``                      halt now and enter the **latched stop** (checklist rules 6–7): every
                                      later ``send_joints`` is ignored until ``reset_latch()``.
    * ``latch()``                     alias of ``stop()``;  ``latched`` tells the state.
    * ``set_torque(on)``              disable / enable servo torque (only allowed while latched).
    """
    name = "driver"

    def __init__(self, spec: ArmSpec = COURSE_ARM):
        self.spec = spec
        self._latched = False
        self._torque = True
        self.connected = False

    # -- connection ---------------------------------------------------------------------------
    @abstractmethod
    def connect(self) -> None: ...

    def disconnect(self) -> None:
        self.connected = False

    # -- motion -------------------------------------------------------------------------------
    @abstractmethod
    def read_joints(self) -> np.ndarray: ...

    @abstractmethod
    def _send(self, q: np.ndarray, speed: float) -> None: ...

    @abstractmethod
    def _halt(self) -> None: ...

    def send_joints(self, q, speed: float) -> bool:
        if self._latched or not self.connected:
            return False
        q = np.asarray(q, float).reshape(self.spec.n)
        speed = float(np.clip(speed, 0.0, 1.0))
        if speed <= 0:
            return False
        self._send(q, speed)
        return True

    def stop(self) -> None:
        """Halt and latch (safe even when not connected)."""
        self._latched = True
        if self.connected:
            self._halt()

    def latch(self) -> None:
        self.stop()

    @property
    def latched(self) -> bool:
        return self._latched

    def reset_latch(self) -> None:
        if not self._torque:
            self.set_torque(True)
        self._latched = False

    # -- torque ------------------------------------------------------------------------------
    @property
    def torque_enabled(self) -> bool:
        return self._torque

    def set_torque(self, on: bool) -> None:
        if not on and not self._latched:
            raise HardwareError("torque may only be disabled in latched stop (checklist rule 6)")
        self._torque = bool(on)
        self._set_torque(bool(on))

    def _set_torque(self, on: bool) -> None:
        pass
