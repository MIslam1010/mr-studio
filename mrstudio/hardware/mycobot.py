"""MyCobotDriver: the course arm over USB-serial with Elephant Robotics' ``pymycobot`` SDK.

``pymycobot`` is imported lazily in ``connect()`` -- MR Studio does not depend on it.  If it is
missing, ``connect()`` raises ``HardwareError`` with the install command (``pip install pymycobot``,
which also pulls in ``pyserial``).

Conventions (course_robots.md, ``course_arm()`` docstring): the PoE zero pose (arm stretched along
+x_s) is NOT the servo zero of a commercial arm.  ``servo_zero_offset`` (rad, calibrated at
commissioning) maps   q_model = q_servo - offset   and   q_servo = q_model + offset.
The SDK talks in degrees and in a speed percentage 1-100; MR Studio's ``speed`` fraction of the rated
speed is sent as that percentage (re-check at commissioning that 100 % of the SDK equals the rated
speed of ``COURSE_ARM``).
"""
from __future__ import annotations

import numpy as np

from .base import COURSE_ARM, ArmDriver, ArmSpec, HardwareError

INSTALL_HINT = ("The myCobot back-end needs Elephant Robotics' SDK (pymycobot), which is not available in this "
                "copy of MR Studio.  If you run MR Studio from source, install it into its venv with\n"
                "    .venv/bin/python -m pip install pymycobot\n"
                "If you use the installed app, ask for a build that includes pymycobot "
                "(packaging/README.md).  Until then, use the Simulated back-end.")


def _sdk_class():
    """The arm class of the installed pymycobot (MyCobot280 in recent versions, MyCobot before)."""
    try:
        import pymycobot  # noqa: F401
    except ImportError as e:
        raise HardwareError(INSTALL_HINT) from e
    try:
        from pymycobot import MyCobot280 as cls
    except ImportError:
        try:
            from pymycobot.mycobot import MyCobot as cls
        except ImportError as e:
            raise HardwareError("pymycobot is installed but provides neither MyCobot280 nor MyCobot") from e
    return cls


def sdk_available() -> bool:
    try:
        _sdk_class()
        return True
    except HardwareError:
        return False


def list_serial_ports() -> list[str]:
    """Serial ports that may carry the arm (needs pyserial; empty list without it)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p.device for p in list_ports.comports()]


class MyCobotDriver(ArmDriver):
    name = "myCobot (serial)"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, spec: ArmSpec = COURSE_ARM,
                 servo_zero_offset=None, sdk_factory=None):
        super().__init__(spec)
        self.port, self.baud = port, int(baud)
        self.offset_calibrated = servo_zero_offset is not None
        self.offset = np.zeros(spec.n) if servo_zero_offset is None else np.asarray(servo_zero_offset, float)
        self._factory = sdk_factory           # tests may inject a fake SDK class
        self.mc = None
        self._last = None
        self.read_failures = 0

    def connect(self) -> None:
        cls = self._factory or _sdk_class()
        try:
            self.mc = cls(self.port, self.baud)
            if hasattr(self.mc, "power_on"):
                self.mc.power_on()
        except Exception as e:  # serial errors come in many flavours
            self.mc = None
            raise HardwareError(f"could not open {self.port} at {self.baud} baud: {e}") from e
        self.connected = True
        self._last = None
        self._last = self.read_joints()

    def disconnect(self) -> None:
        self.connected = False
        mc, self.mc = self.mc, None
        try:
            if mc is not None and hasattr(mc, "close"):
                mc.close()
        except Exception:
            pass

    def read_joints(self) -> np.ndarray:
        if self.mc is None:
            raise HardwareError("not connected")
        try:
            ang = self.mc.get_angles()
        except Exception:
            ang = None
        # pymycobot returns -1, None or [] when a read times out: keep the previous sample
        if not isinstance(ang, (list, tuple)) or len(ang) != self.spec.n:
            self.read_failures += 1
            if self._last is None:
                raise HardwareError("the arm did not report its joint angles (check power and cable)")
            return self._last.copy()
        q = np.radians(np.asarray(ang, float)) - self.offset
        self._last = q
        return q.copy()

    def _send(self, q, speed) -> None:
        deg = np.degrees(np.asarray(q, float) + self.offset)
        pct = int(np.clip(round(100 * speed), 1, 100))
        self.mc.send_angles([float(x) for x in deg], pct)

    def _halt(self) -> None:
        try:
            self.mc.stop()
        except Exception:
            pass

    def _set_torque(self, on: bool) -> None:
        if self.mc is None:
            return
        if on:
            (getattr(self.mc, "focus_all_servos", None) or self.mc.power_on)()
        else:
            self.mc.release_all_servos()
