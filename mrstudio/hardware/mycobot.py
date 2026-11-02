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

Liveness (pymycobot 4.x, checked against the installed 4.0.7 source): ``MyCobot280._res`` writes a command and
waits for a reply after EVERY command (has_reply or not), up to 3 attempts of 0.5 s each on macOS/Linux
(0.15 s on Windows), and then *returns* -1 instead of raising.  The driver therefore

* treats a -1 / None return of any SDK call as a failure (``send_angles`` -> ``HardwareError`` -> latched stop;
  ``stop`` -> halt NOT confirmed; ``get_angles`` -> failed read; ``power_on`` / ``set_fresh_mode`` -> connect
  fails).  An SDK that does not acknowledge commands (pymycobot < 4) is not supported;
* runs every SDK call on one I/O thread per connection (in order) and waits at most a **deadline** for it
  (``CALL_DEADLINE_S`` for set-points and reads, ``STOP_DEADLINE_S`` for the halt), so a hung controller
  (port open, no replies) latches the run within about ``CALL_DEADLINE_S`` + one tick, whatever the SDK's own
  retry timing.  A call that missed its deadline before it started is dropped (never sent late), except
  ``stop``, which still goes out when the port frees up;
* raises from ``read_joints`` as soon as a read fails and the last good one is older than
  ``STALE_READ_LIMIT_S`` (wall-clock), besides the ``MAX_READ_FAILURES`` count;
* sets the firmware's **fresh mode** (``set_fresh_mode(1)``: execute the latest command, no queue) on connect
  and verifies it with ``get_fresh_mode``, so the arm never works through a backlog of set-points after a
  stall or a cable pull.  ``connect()`` fails if it cannot be set.

``send_angles`` stays synchronous (not ``_async=True``): the 4.x firmware acknowledges every command, and an
unread acknowledgement would be taken by ``read()`` as the reply to the next ``get_angles`` (it returns the
first complete frame), so every read would need a retry.  The acknowledgement is also the per-tick proof
that the set-point arrived.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from concurrent.futures import wait as _wait_futures
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
STALE_READ_LIMIT_S = 0.3               # a failed read with no good one for this long -> HardwareError
CALL_DEADLINE_S = 0.4                  # get_angles / send_angles: no answer within this -> failure
STOP_DEADLINE_S = 0.5                  # stop: not acknowledged within this -> halt not confirmed
SLOW_CALL_DEADLINE_S = 3.0             # connect and torque commands (fits pymycobot's 3 x 0.5 s retries)
FRESH_MODE = 1                         # firmware "fresh" mode: always execute the latest command first
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


def _not_acknowledged(r) -> bool:
    """pymycobot 4.x returns -1 when a command got no (valid) reply after its retries; None means nothing was
    read at all (an ``_async`` call, or an SDK that does not wait for replies)."""
    if r is None:
        return True
    if isinstance(r, (bool, np.bool_)):
        return False
    return isinstance(r, (int, float, np.integer, np.floating)) and r == -1


class _IOWorker:
    """Runs the SDK calls of one connection in order on a daemon thread, so a caller can stop waiting after a
    deadline while the SDK is still inside its own read retries.  Calls queued after ``close()``, and calls
    cancelled by their caller before they started, are never executed."""

    def __init__(self, name: str):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self.closed = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        if self.closed:
            fut.set_exception(HardwareError("not connected"))
        else:
            self._q.put((fut, fn, args))
        return fut

    def close(self) -> None:
        self.closed = True
        self._q.put(None)

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():
                continue                                  # missed its deadline before it started: never sent
            if self.closed:
                fut.set_exception(HardwareError("not connected"))
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - handed to the waiting caller
                fut.set_exception(e)


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
        self._io: _IOWorker | None = None     # runs the SDK calls of the current connection
        self._last = None
        self.call_deadline_s = CALL_DEADLINE_S
        self.stop_deadline_s = STOP_DEADLINE_S
        self.slow_call_deadline_s = SLOW_CALL_DEADLINE_S
        self.stale_read_limit_s = STALE_READ_LIMIT_S
        self.deadline_misses = 0              # SDK calls that did not return within their deadline
        self.io_round_trip_s = 0.0            # duration of the last answered SDK call [s] (seeds HardwareRun)
        self.fresh_mode: int | None = None    # firmware fresh mode read back on connect (None = not verified)

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
                    read_failures=self.read_failures, clamp_count=self.clamp_count,
                    fresh_mode=self.fresh_mode, call_deadline_s=self.call_deadline_s,
                    stop_deadline_s=self.stop_deadline_s, stale_read_limit_s=self.stale_read_limit_s,
                    deadline_misses=self.deadline_misses, io_round_trip_s=round(self.io_round_trip_s, 4))

    # -- SDK access ----------------------------------------------------------------------------
    def _call(self, what: str, *args, deadline: float | None = None, ack: bool = True):
        """One SDK call under the I/O lock, run on the connection's I/O thread and waited for at most
        ``deadline`` seconds.  Any exception, a missed deadline and (with ``ack``) a -1 / None return become a
        ``HardwareError``."""
        deadline = self.call_deadline_s if deadline is None else deadline
        with self.io_lock:
            mc, io = self.mc, self._io
            if mc is None or io is None:
                raise HardwareError("not connected")
            t_call = time.monotonic()
            fut = io.submit(getattr(mc, what), *args)
            done, _ = _wait_futures([fut], timeout=deadline)
            if not done:
                self.deadline_misses += 1
                if what != "stop":         # a late halt is still wanted; a late set-point or read is not
                    fut.cancel()
                raise HardwareError(f"{what}: no answer from the arm on {self.port} within "
                                    f"{1e3 * deadline:.0f} ms (controller hung or cable out?)")
            try:
                r = fut.result()
            except Exception as e:
                raise HardwareError(f"{what} failed on {self.port}: {e}") from e
            self.io_round_trip_s = time.monotonic() - t_call
            if ack and _not_acknowledged(r):
                raise HardwareError(f"{what}: the arm on {self.port} did not acknowledge the command "
                                    f"(pymycobot returned {r!r})")
            return r

    def _close_port(self) -> None:
        io, self._io = self._io, None
        if io is not None:
            io.close()                     # queued calls are dropped, never sent on a closing port
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
            self.fresh_mode = None
            try:
                self.mc = cls(self.port, self.baud)
            except Exception as e:  # serial errors come in many flavours
                self._close_port()
                raise HardwareError(f"could not open {self.port} at {self.baud} baud: {e}") from e
            self._io = _IOWorker(f"mycobot-io {self.port}")
            try:
                if hasattr(self.mc, "power_on"):
                    self._call("power_on", deadline=self.slow_call_deadline_s)
                self._set_fresh_mode()
            except HardwareError as e:
                self._close_port()
                raise HardwareError(f"the arm on {self.port} at {self.baud} baud did not answer as expected "
                                    f"(check power, cable, baud rate and firmware): {e}") from e
            self.connected = True
            self._last = None
            self.consecutive_read_failures = 0
            err = None
            for k in range(MAX_READ_FAILURES):      # the first reads after opening may time out
                try:
                    self._last = self._read_joints(self.slow_call_deadline_s)
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

    def _set_fresh_mode(self) -> None:
        """Fresh mode on (the firmware executes the newest command, no queue); verified when the SDK can read
        it back.  Raises HardwareError when it cannot be set or does not read back as set."""
        if getattr(self.mc, "set_fresh_mode", None) is None:      # required (not a hasattr guard: see dry run)
            raise HardwareError("this pymycobot has no set_fresh_mode(); install the tested version "
                                "(pip install -r requirements-hardware.txt)")
        self._call("set_fresh_mode", FRESH_MODE, deadline=self.slow_call_deadline_s)
        if hasattr(self.mc, "get_fresh_mode"):
            mode = self._call("get_fresh_mode", deadline=self.slow_call_deadline_s)
            try:
                mode = int(mode)
            except (TypeError, ValueError):
                mode = None
            if mode != FRESH_MODE:
                raise HardwareError(f"fresh mode did not stick (get_fresh_mode returned {mode!r}, "
                                    f"expected {FRESH_MODE}); the firmware may queue set-points")
            self.fresh_mode = mode

    def disconnect(self) -> None:
        with self.io_lock:
            self.connected = False
            self._close_port()

    # -- motion --------------------------------------------------------------------------------
    def read_joints(self) -> np.ndarray:
        return self._read_joints(self.call_deadline_s)

    def _read_joints(self, deadline: float) -> np.ndarray:
        err = None
        with self.io_lock:
            if self.mc is None:
                raise HardwareError("not connected")
            try:
                ang = self._call("get_angles", deadline=deadline, ack=False)
            except HardwareError as e:
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
            age = self.fresh_age()
            if (self._last is None or self.consecutive_read_failures >= MAX_READ_FAILURES
                    or age > self.stale_read_limit_s):
                since = f"none for {1e3 * age:.0f} ms" if np.isfinite(age) else "none yet"
                raise HardwareError(f"the arm did not report its joint angles ({self.consecutive_read_failures} "
                                    f"failed reads in a row, {since}; check power and cable)") from err
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
        # synchronous: -1 (no acknowledgement) or a missed deadline raises HardwareError -> latched stop
        self._call("send_angles", [float(x) for x in sent], self.stream_speed_pct, deadline=self.call_deadline_s)

    def _halt(self) -> bool:
        # raises (-> halt NOT confirmed) when the firmware does not acknowledge STOP within the deadline
        self._call("stop", deadline=self.stop_deadline_s)
        return True

    def _set_torque(self, on: bool) -> None:
        if on:
            self._call("focus_all_servos" if hasattr(self.mc, "focus_all_servos") else "power_on",
                       deadline=self.slow_call_deadline_s)
        else:
            self._call("release_all_servos", deadline=self.slow_call_deadline_s)
