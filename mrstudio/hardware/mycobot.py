"""MyCobotDriver: the course arm over USB-serial with Elephant Robotics' ``pymycobot`` SDK.

``pymycobot`` is imported lazily in ``connect()`` -- MR Studio does not depend on it.  If it is
missing, ``connect()`` raises ``HardwareError`` with the install command (``pip install pymycobot``,
which also pulls in ``pyserial``).

Conventions (course_robots.md, ``course_arm()`` docstring): the PoE zero pose (arm stretched along
+x_s) is NOT the servo zero of a commercial arm, and a servo may turn the other way round than the
model's screw axis.  The commissioning calibration (``MyCobotCalibration``, a small JSON file) holds a
sign s_i = ±1 and an offset o_i [deg] per joint:

    q_servo = s * q_model + o        q_model = s * (q_servo - o)        (degrees, element-wise)

Every SDK call goes through ``_call`` (under the driver's I/O lock), so any serial exception becomes a
``HardwareError`` that the run loop turns into a latched stop.  Set-points are clamped in servo-degree
space to ``MYCOBOT280_LIMITS_DEG`` minus 10° *after* the calibration mapping, i.e. against the real
servo travel whatever the model limits say.

Speed: MR Studio limits the joint speed itself (trajectory time scaling + a per-tick step clamp in
``session.HardwareRun``) and streams set-points at 50 Hz.  The firmware speed percentage of
``send_angles`` only sets how fast a servo chases each set-point, so a fixed low value is sent
(``stream_speed_pct``, default 20 %), independent of the session's speed cap.  The mapping
% -> deg/s is firmware-dependent and not documented reliably: measure it at commissioning (time a 30°
step at the chosen %) and check that it is above the session's speed cap, or the arm lags the
set-points and the tracking watchdog stops the run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import COURSE_ARM, ArmDriver, ArmSpec, HardwareError

INSTALL_HINT = ("The myCobot back-end needs Elephant Robotics' SDK (pymycobot), which is not available in this "
                "copy of MR Studio.  If you run MR Studio from source, install it into its venv with\n"
                "    .venv/bin/python -m pip install pymycobot\n"
                "If you use the installed app, ask for a build that includes pymycobot "
                "(packaging/README.md).  Until then, use the Simulated back-end.")

#: Servo travel of the myCobot 280 [deg] (Elephant Robotics spec sheet, M5/Pi editions).
#: VERIFY ON YOUR UNIT at commissioning -- firmware versions and editions differ.
MYCOBOT280_LIMITS_DEG = np.array([(-168, 168), (-135, 135), (-150, 150),
                                  (-145, 145), (-165, 165), (-175, 175)], float)
SERVO_LIMIT_MARGIN_DEG = 10.0          # set-points stay this far inside the servo travel
BAUD_RATES = (115200, 1000000)         # 115200: M5 / Pi editions; 1000000: some boards (check the manual)
DEFAULT_STREAM_SPEED_PCT = 20          # firmware speed % while streaming (see the module docstring)
MAX_READ_FAILURES = 3                  # consecutive failed reads -> HardwareError (stops the run)
#: USB-serial bridges used on myCobot boards: CP210x (M5 Basic), CH340/CH9102 (M5 Atom / Pi adapters)
USB_SERIAL_IDS = {(0x10C4, 0xEA60): "CP210x", (0x1A86, 0x7523): "CH340", (0x1A86, 0x55D4): "CH9102"}
CALIBRATION_ENV = "MRSTUDIO_MYCOBOT_CALIBRATION"


def default_calibration_path() -> Path:
    """``$MRSTUDIO_MYCOBOT_CALIBRATION`` or ``~/.mrstudio/mycobot_calibration.json``."""
    env = os.environ.get(CALIBRATION_ENV)
    return Path(env).expanduser() if env else Path.home() / ".mrstudio" / "mycobot_calibration.json"


@dataclass
class MyCobotCalibration:
    """Per-joint sign and servo offset of one arm (the commissioning record)."""
    sign: np.ndarray = field(default_factory=lambda: np.ones(6))
    offset_deg: np.ndarray = field(default_factory=lambda: np.zeros(6))
    note: str = ""

    def __post_init__(self):
        self.sign = np.asarray(self.sign, float).reshape(-1)
        self.offset_deg = np.asarray(self.offset_deg, float).reshape(-1)
        if self.sign.shape != self.offset_deg.shape:
            raise ValueError("sign and offset_deg need one entry per joint")
        if not np.all(np.abs(self.sign) == 1.0):
            raise ValueError("every sign must be +1 or -1")
        if not np.all(np.isfinite(self.offset_deg)) or np.any(np.abs(self.offset_deg) > 180.0):
            raise ValueError("offsets must be finite and within ±180°")

    def to_servo_deg(self, q_model) -> np.ndarray:
        return self.sign * np.degrees(np.asarray(q_model, float)) + self.offset_deg

    def to_model(self, q_servo_deg) -> np.ndarray:
        return np.radians(self.sign * (np.asarray(q_servo_deg, float) - self.offset_deg))

    def to_dict(self) -> dict:
        return dict(sign=[int(s) for s in self.sign], offset_deg=[float(o) for o in self.offset_deg],
                    note=self.note)

    def save(self, path=None) -> Path:
        path = Path(path) if path is not None else default_calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(self.to_dict(), format="mrstudio-mycobot-calibration/1"), indent=2))
        os.replace(tmp, path)                    # never leave a half-written file behind
        return path

    @classmethod
    def load(cls, path=None) -> "MyCobotCalibration":
        path = Path(path) if path is not None else default_calibration_path()
        try:
            d = json.loads(Path(path).read_text())
            return cls(sign=d["sign"], offset_deg=d["offset_deg"], note=str(d.get("note", "")))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise HardwareError(f"calibration file {path} is unreadable: {e}") from e


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


def sdk_version() -> str | None:
    """Installed pymycobot version (None without it) -- recorded in the log sidecar."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("pymycobot")
    except (ImportError, PackageNotFoundError):
        return None
    except Exception:
        return None


def rank_ports(ports, platform: str | None = None) -> list[str]:
    """Device names of likely arm ports, best first.

    ``ports`` are pyserial ``ListPortInfo``-like objects (``device``, ``vid``, ``pid``).  Known USB-serial
    bridges of the myCobot boards come first, then other USB ports, then the rest.  On macOS the
    call-out ``/dev/cu.*`` nodes are used (``/dev/tty.*`` blocks on open until carrier detect), and the
    Bluetooth / debug-console ports are dropped."""
    platform = platform or sys.platform
    out = []
    for p in ports:
        dev = str(getattr(p, "device", p))
        if platform == "darwin":
            if dev.startswith("/dev/tty."):
                continue
            if "bluetooth" in dev.lower() or "debug-console" in dev.lower():
                continue
        vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
        rank = 0 if (vid, pid) in USB_SERIAL_IDS else (1 if vid is not None else 2)
        out.append((rank, dev))
    return [d for _, d in sorted(out)]


def list_serial_ports() -> list[str]:
    """Serial ports that may carry the arm, likely ones first (needs pyserial; empty list without it)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    try:
        return rank_ports(list_ports.comports())
    except Exception:
        return []


class MyCobotDriver(ArmDriver):
    name = "myCobot (serial)"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, spec: ArmSpec = COURSE_ARM,
                 servo_zero_offset=None, sdk_factory=None, *, calibration: MyCobotCalibration | None = None,
                 stream_speed_pct: int = DEFAULT_STREAM_SPEED_PCT, clock=None,
                 servo_limits_deg=MYCOBOT280_LIMITS_DEG):
        super().__init__(spec)
        self.port, self.baud = port, int(baud)
        if calibration is None:
            # legacy argument: servo_zero_offset [rad] with sign +1 (q_servo = q_model + offset)
            off = np.zeros(spec.n) if servo_zero_offset is None else np.degrees(np.asarray(servo_zero_offset, float))
            calibration = MyCobotCalibration(sign=np.ones(spec.n), offset_deg=off)
            self.offset_calibrated = servo_zero_offset is not None
        else:
            self.offset_calibrated = True
        self.calibration = calibration
        self.calibration_path: str | None = None          # set by whoever loaded it (for the sidecar)
        self.stream_speed_pct = int(np.clip(stream_speed_pct, 1, 100))
        lim = np.asarray(servo_limits_deg, float).reshape(spec.n, 2)
        self.servo_limits_deg = lim + np.array([SERVO_LIMIT_MARGIN_DEG, -SERVO_LIMIT_MARGIN_DEG])
        self.clamp_count = 0                               # writes that hit the servo-limit clamp
        if clock is not None:
            self.clock = clock
        self._factory = sdk_factory           # tests may inject a fake SDK class
        self.mc = None
        self._last = None

    # -- calibration -------------------------------------------------------------------------
    @property
    def offset(self) -> np.ndarray:
        """Servo zero offset [rad] (q_servo = sign * q_model + offset)."""
        return np.radians(self.calibration.offset_deg)

    @property
    def sign(self) -> np.ndarray:
        return self.calibration.sign.copy()

    def set_calibration(self, cal: MyCobotCalibration, path=None) -> None:
        with self.io_lock:
            self.calibration = cal
            self.offset_calibrated = True
            self.calibration_path = str(path) if path is not None else self.calibration_path
            self._last = None

    def describe(self) -> dict:
        return dict(driver=self.name, port=self.port, baud=self.baud, sdk="pymycobot", sdk_version=sdk_version(),
                    sign=[int(s) for s in self.calibration.sign],
                    offset_deg=[float(o) for o in self.calibration.offset_deg],
                    calibrated=bool(self.offset_calibrated), calibration_file=self.calibration_path,
                    stream_speed_pct=self.stream_speed_pct,
                    servo_limits_deg=self.servo_limits_deg.tolist(),
                    read_failures=self.read_failures, clamp_count=self.clamp_count)

    # -- SDK access ----------------------------------------------------------------------------
    def _call(self, what: str, *args):
        """One SDK call under the I/O lock; any exception becomes a HardwareError."""
        with self.io_lock:
            mc = self.mc
            if mc is None:
                raise HardwareError("not connected")
            try:
                return getattr(mc, what)(*args)
            except Exception as e:
                raise HardwareError(f"{what} failed on {self.port}: {e}") from e

    def _close_port(self) -> None:
        mc, self.mc = self.mc, None
        if mc is None:
            return
        try:
            if hasattr(mc, "close"):
                mc.close()
            elif getattr(mc, "_serial_port", None) is not None:
                mc._serial_port.close()
        except Exception:
            pass

    # -- connection ----------------------------------------------------------------------------
    def connect(self) -> None:
        cls = self._factory or _sdk_class()
        with self.io_lock:
            try:
                self.mc = cls(self.port, self.baud)
                if hasattr(self.mc, "power_on"):
                    self.mc.power_on()
            except Exception as e:  # serial errors come in many flavours
                self._close_port()
                raise HardwareError(f"could not open {self.port} at {self.baud} baud: {e}") from e
            self.connected = True
            self._last = None
            self.consecutive_read_failures = 0
            err = None
            for k in range(MAX_READ_FAILURES):      # the first reads after opening may time out
                try:
                    self._last = self.read_joints()
                    return
                except HardwareError as e:
                    err = e
                    if k + 1 < MAX_READ_FAILURES:
                        time.sleep(0.05)
            # no valid reading: do not leave the port open behind a failed connect.  The servos are left
            # powered (holding their pose) -- powering them off would drop the arm.
            self.disconnect()
            raise HardwareError(f"the arm on {self.port} did not report its joint angles "
                                f"(check power, cable and baud rate): {err}") from err

    def disconnect(self) -> None:
        with self.io_lock:
            self.connected = False
            self._close_port()

    # -- motion --------------------------------------------------------------------------------
    def read_joints(self) -> np.ndarray:
        err = None
        with self.io_lock:
            if self.mc is None:
                raise HardwareError("not connected")
            try:
                ang = self.mc.get_angles()
            except Exception as e:
                ang, err = None, e
        # pymycobot returns -1, None or [] when a read times out
        ok = isinstance(ang, (list, tuple)) and len(ang) == self.spec.n
        if ok:
            try:
                deg = np.asarray(ang, float)
                ok = bool(np.all(np.isfinite(deg)))
            except (TypeError, ValueError):
                ok = False
        if not ok:
            self._mark_read(False)
            if self._last is None or self.consecutive_read_failures >= MAX_READ_FAILURES:
                raise HardwareError(f"the arm did not report its joint angles ({self.consecutive_read_failures} "
                                    "failed reads in a row; check power and cable)") from err
            return self._last.copy()          # flagged stale: last_read_fresh is False
        self.last_meas_servo_deg = deg
        q = self.calibration.to_model(deg)
        self._last = q
        self._mark_read(True)
        return q.copy()

    def _send(self, q, speed) -> None:
        deg = self.calibration.to_servo_deg(q)
        lo, hi = self.servo_limits_deg.T
        sent = np.clip(deg, lo, hi)
        if np.any(sent != deg):
            self.clamp_count += 1
        self.last_cmd_servo_deg = sent
        # ``speed`` (fraction of rated) is enforced by the run loop; the firmware gets a fixed low %
        self._call("send_angles", [float(x) for x in sent], self.stream_speed_pct)

    def _halt(self) -> bool:
        self._call("stop")
        return True

    def _set_torque(self, on: bool) -> None:
        if on:
            self._call("focus_all_servos" if hasattr(self.mc, "focus_all_servos") else "power_on")
        else:
            self._call("release_all_servos")
