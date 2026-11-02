#!/usr/bin/env python3
"""Dry run of the myCobot back-end: everything that can be checked WITHOUT moving the arm.

    .venv/bin/python tools/mycobot_dryrun.py                    # no serial port is opened
    .venv/bin/python tools/mycobot_dryrun.py --calibration FILE  # check another calibration file
    .venv/bin/python tools/mycobot_dryrun.py --probe PORT        # also time read-only calls on PORT

Checks (exit code 1 if any FAIL):

1. pymycobot is importable; its version against the pin in ``requirements-hardware.txt``.
2. Every SDK method that ``mrstudio/hardware/mycobot.py`` calls (found by parsing the driver source, not a
   hand-written list) exists on the class the driver will use (``MyCobot280``, else the legacy ``MyCobot``),
   and the constructor / ``send_angles`` accept the driver's arguments.
3. The driver's servo clamps (servo travel minus 10°) lie inside the angle range the SDK accepts, so a clamped
   set-point is never rejected by the SDK.
4. Serial ports, ranked like the Hardware tab ranks them, with a VID:PID match against ``USB_SERIAL_IDS``
   (listing only: no port is opened).
5. The calibration JSON (``$MRSTUDIO_MYCOBOT_CALIBRATION`` or ``~/.mrstudio/mycobot_calibration.json``):
   a missing file is a WARN (Run stays blocked until commissioning); an unreadable one is a FAIL.
6. Prints the safety checklist, the speed caps of each session type and the run-time guards.

``--probe PORT`` opens PORT with the SDK class and sends READ commands only (``get_angles``,
``is_power_on``) to measure the round trip.  It never sends power, torque or motion commands.  Use it on the
day, before the Hardware tab connects (the tab's Connect powers the servos on).
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mrstudio.hardware import mycobot as drv_mod  # noqa: E402
from mrstudio.hardware.base import COURSE_ARM, HardwareError  # noqa: E402
from mrstudio.hardware.safety import COURSE_ARM_SAFETY, SESSION_LABELS, SafetySupervisor  # noqa: E402
from mrstudio.hardware.session import HardwareRun  # noqa: E402

DRIVER_SRC = Path(drv_mod.__file__)
REQ_HW = ROOT / "requirements-hardware.txt"


class Report:
    def __init__(self):
        self.fails, self.warns = [], []

    def ok(self, msg):
        print(f"  OK    {msg}")

    def warn(self, msg):
        self.warns.append(msg)
        print(f"  WARN  {msg}")

    def fail(self, msg):
        self.fails.append(msg)
        print(f"  FAIL  {msg}")

    @staticmethod
    def head(title):
        print(f"\n== {title}")


# --------------------------------------------------------------------------------------------- driver source
def sdk_methods_called(src: str | None = None) -> dict[str, bool]:
    """SDK methods the driver calls -> True if the call is guarded by ``hasattr`` (optional).

    Found in the driver source: ``self._call("name", ...)`` (also both arms of a conditional expression),
    ``self.mc.name(...)`` / ``mc.name(...)``, and the names tested with ``hasattr(<sdk object>, "name")``."""
    tree = ast.parse(src if src is not None else DRIVER_SRC.read_text(encoding="utf-8"))
    called, guarded = set(), set()

    def consts(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.IfExp):
            return consts(node.body) + consts(node.orelse)
        return []

    def is_sdk(node):          # self.mc or a local alias ``mc``
        return ((isinstance(node, ast.Attribute) and node.attr == "mc") or
                (isinstance(node, ast.Name) and node.id == "mc"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "_call" and node.args:
            called.update(consts(node.args[0]))
        elif isinstance(f, ast.Attribute) and is_sdk(f.value):
            called.add(f.attr)
        elif isinstance(f, ast.Name) and f.id == "hasattr" and len(node.args) == 2 and is_sdk(node.args[0]):
            guarded.update(consts(node.args[1]))
    return {m: (m in guarded) for m in sorted(called | guarded)}


# --------------------------------------------------------------------------------------------- checks
def check_sdk(rep: Report):
    rep.head("1. pymycobot")
    try:
        cls = drv_mod._sdk_class()
    except HardwareError as e:
        rep.fail(f"pymycobot is not usable: {e}")
        return None
    ver = drv_mod.sdk_version()
    rep.ok(f"pymycobot {ver} installed; the driver will use {cls.__module__}.{cls.__name__}")
    pin = None
    if REQ_HW.is_file():
        m = re.search(r"^pymycobot==(\S+)", REQ_HW.read_text(encoding="utf-8"), re.M)
        pin = m.group(1) if m else None
    if pin is None:
        rep.warn("requirements-hardware.txt has no pymycobot pin")
    elif pin != ver:
        rep.warn(f"installed pymycobot {ver} differs from the tested pin {pin} (requirements-hardware.txt)")
    else:
        rep.ok(f"matches the pin in requirements-hardware.txt ({pin})")
    return cls


def check_methods(rep: Report, cls) -> int:
    rep.head("2. SDK methods the driver calls")
    methods = sdk_methods_called()
    classes = [cls]
    try:
        import pymycobot
        legacy = getattr(pymycobot, "MyCobot", None)
        if legacy is not None and legacy is not cls:
            classes.append(legacy)
    except ImportError:
        pass
    missing = 0
    for m, optional in methods.items():
        have = [c.__name__ for c in classes if callable(getattr(c, m, None))]
        tag = " (optional: guarded by hasattr)" if optional else ""
        if cls.__name__ in have:
            rep.ok(f"{m}{tag}: present on {', '.join(have)}")
        elif optional:
            rep.warn(f"{m}{tag}: missing on {cls.__name__} (the driver falls back)")
        else:
            missing += 1
            rep.fail(f"{m}: missing on {cls.__name__}")
    # the driver's call signatures
    for what, fn, args in (("constructor", cls.__init__, (None, "/dev/null", 115200)),
                           ("send_angles", getattr(cls, "send_angles", None), (None, [0.0] * 6, 20))):
        try:
            inspect.signature(fn).bind(*args)
            rep.ok(f"{what} accepts the driver's arguments {inspect.signature(fn)}")
        except (TypeError, ValueError) as e:
            missing += 1
            rep.fail(f"{what} does not accept the driver's arguments: {e}")
    if missing == 0:
        print(f"  all {len(methods)} SDK methods present ({', '.join(methods)})")
    return missing


def check_limits(rep: Report, cls):
    rep.head("3. Servo clamps vs the SDK's accepted angle range")
    clamp = drv_mod.MYCOBOT280_LIMITS_DEG + np.array([drv_mod.SERVO_LIMIT_MARGIN_DEG, -drv_mod.SERVO_LIMIT_MARGIN_DEG])
    try:
        from pymycobot.robot_info import RobotLimit
        lim = RobotLimit.robot_limit[cls.__name__]
        lo, hi = np.asarray(lim["angles_min"], float), np.asarray(lim["angles_max"], float)
    except Exception as e:     # older SDKs keep the table elsewhere
        rep.warn(f"SDK angle table not found ({e}); compare the clamps with the arm's manual by hand")
        return
    for j in range(6):
        txt = (f"J{j + 1}: driver clamp [{clamp[j, 0]:.0f}°, {clamp[j, 1]:.0f}°], "
               f"SDK accepts [{lo[j]:.0f}°, {hi[j]:.0f}°], driver travel table ±{drv_mod.MYCOBOT280_LIMITS_DEG[j, 1]:.0f}°")
        if clamp[j, 0] >= lo[j] and clamp[j, 1] <= hi[j]:
            if (abs(drv_mod.MYCOBOT280_LIMITS_DEG[j, 0] - lo[j]) > 0.5 or
                    abs(drv_mod.MYCOBOT280_LIMITS_DEG[j, 1] - hi[j]) > 0.5):
                rep.ok(txt + "  (tables differ: verify the travel on this arm)")
            else:
                rep.ok(txt)
        else:
            rep.fail(txt + "  -> a clamped set-point would be rejected by the SDK")


def check_ports(rep: Report):
    rep.head("4. Serial ports (listed only, none opened)")
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except Exception as e:
        rep.warn(f"cannot list serial ports: {e}")
        return
    ranked = drv_mod.rank_ports(ports)
    info = {str(p.device): p for p in ports}
    if not ranked:
        rep.warn("no candidate serial port (fine for a dry run; on the day the arm's USB bridge must show here)")
    for dev in ranked:
        p = info.get(dev)
        vid, pid = getattr(p, "vid", None), getattr(p, "pid", None)
        ident = f"{vid:04X}:{pid:04X}" if vid is not None and pid is not None else "no USB id"
        bridge = drv_mod.USB_SERIAL_IDS.get((vid, pid))
        note = f"myCobot USB bridge ({bridge})" if bridge else "not a known myCobot bridge"
        print(f"        {dev:<34} {ident:<10} {note}")
    hits = [d for d in ranked if (getattr(info.get(d), "vid", None), getattr(info.get(d), "pid", None))
            in drv_mod.USB_SERIAL_IDS]
    if hits:
        rep.ok(f"likely arm port: {hits[0]}")
    else:
        rep.warn("no CP210x / CH340 / CH9102 bridge connected (expected when the arm is not plugged in)")


def check_calibration(rep: Report, path=None):
    rep.head("5. Calibration file")
    path = Path(path).expanduser() if path else drv_mod.default_calibration_path()
    if not path.exists():
        rep.warn(f"no calibration file at {path}: Run on the myCobot stays blocked until the signs and offsets "
                 "are commissioned and saved (protocol step 6)")
        return
    try:
        cal = drv_mod.MyCobotCalibration.load(path)
    except HardwareError as e:
        rep.fail(str(e))
        return
    if cal.sign.shape != (COURSE_ARM.n,):
        rep.fail(f"{path}: {cal.sign.size} joints, the arm has {COURSE_ARM.n}")
        return
    rep.ok(f"{path} is valid")
    for j in range(COURSE_ARM.n):
        print(f"        J{j + 1}: sign {int(cal.sign[j]):+d}, offset {cal.offset_deg[j]:+7.2f}°")
    if cal.note:
        print(f"        note: {cal.note}")


def print_safety():
    Report.head("6. Safety checklist and speed caps (enforced by the Hardware tab)")
    sup = SafetySupervisor(session="H1")
    for k, txt in sup.checklist_items():
        print(f"  [ ] {txt}")
    print(f"  keep-out radius {sup.keepout_radius:.2f} m; payload <= {COURSE_ARM_SAFETY['payload_max_kg']:.3f} kg")
    rated = COURSE_ARM.rated_speed_deg_s
    print(f"  rated joint speed {rated[0]:.0f}°/s (illustrative until commissioned)")
    for s in ("H1", "H2", "lab"):
        for first in (True, False):
            sv = SafetySupervisor(session=s, first_run=first)
            print(f"  {SESSION_LABELS[s]:<45} first run={'yes' if first else 'no ':<3}  "
                  f"cap {100 * sv.speed_fraction:.0f} % = {np.degrees(sv.speed_cap[0]):.0f}°/s per joint")
    sl = np.degrees(SafetySupervisor().soft_limits)
    print("  software limits (model, rule 1): " + ", ".join(f"J{j + 1} ±{sl[j, 1]:.0f}°" for j in range(len(sl))))
    print(f"  run-time guards: step clamp cap·{1e3 * HardwareRun.STEP_DT_MAX:.0f} ms per tick, stall freeze after "
          f"{1e3 * HardwareRun.STALL_GAP:.0f} ms gap (stop after {HardwareRun.STALL_STOP_S:.0f} s of them), "
          f"write/read on alternate ticks when both take > {1e3 * HardwareRun.SPLIT_IO_S:.0f} ms, tracking watchdog {np.degrees(HardwareRun.TRACK_ERR):.0f}° for "
          f"{1e3 * HardwareRun.TRACK_TIME:.0f} ms, {drv_mod.MAX_READ_FAILURES} failed reads in a row -> latched stop")
    print(f"  firmware streaming speed {drv_mod.DEFAULT_STREAM_SPEED_PCT} % (measure its deg/s at commissioning)")


SLOW_LINK_MAX_MS = 80.0     # probe: median round trip above this is too slow to stream (stall gap 100 ms)


def probe(rep: Report, cls, port: str, baud: int, n: int = 20) -> None:
    """Read-only round-trip timing on a port (get_angles, is_power_on).  Never sends motion or power commands."""
    rep.head(f"7. Read-only probe of {port} at {baud} baud")
    t0 = time.monotonic()
    try:
        mc = cls(port, baud)
    except Exception as e:
        rep.fail(f"could not open {port}: {e}")
        return
    print(f"        port opened in {time.monotonic() - t0:.2f} s (the SDK waits 1.5 s after opening)")
    try:
        t = time.monotonic()
        pw = mc.is_power_on()
        print(f"        is_power_on -> {pw} ({1e3 * (time.monotonic() - t):.1f} ms)")
        dts, bad, last = [], 0, None
        for _ in range(n):
            t = time.monotonic()
            a = mc.get_angles()
            dts.append(time.monotonic() - t)
            if isinstance(a, (list, tuple)) and len(a) == 6:
                last = a
            else:
                bad += 1
        dts = 1e3 * np.array(dts)
        print(f"        get_angles x{n}: median {np.median(dts):.1f} ms, max {dts.max():.1f} ms, {bad} failed")
        if last is not None:
            print("        servo angles [deg]: " + ", ".join(f"{x:.1f}" for x in last))
        if bad == n:
            rep.fail("no joint reading: check power, Transponder mode, cable and baud rate")
        elif bad:
            rep.warn("failed reads: do not stream until every read succeeds")
        elif np.median(dts) <= 20.0:
            rep.ok("reads are fast enough for the 50 Hz stream")
        elif np.median(dts) <= SLOW_LINK_MAX_MS and dts.max() < 1e3 * HardwareRun.STALL_GAP:
            rep.warn(f"slow reads (median {np.median(dts):.0f} ms): the stream will alternate set-point and read "
                     f"ticks, about {1e3 / (2 * np.median(dts)):.0f} set-points/s instead of 50 (tested with the "
                     f"emulator up to 60 ms round trips); fine for slow H1 moves")
        else:
            rep.warn(f"reads too slow to stream (median {np.median(dts):.0f} ms, max {dts.max():.0f} ms; a tick "
                     f"needs one round trip below {1e3 * HardwareRun.STALL_GAP:.0f} ms): do not stream")
    except Exception as e:
        rep.fail(f"probe failed: {e}")
    finally:
        try:
            mc.close()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="myCobot back-end dry run (no motion).")
    ap.add_argument("--calibration", help="calibration JSON to check (default: the Hardware tab's file)")
    ap.add_argument("--probe", metavar="PORT", help="also open PORT and time read-only calls (no motion)")
    ap.add_argument("--baud", type=int, default=drv_mod.BAUD_RATES[0], help="baud rate for --probe")
    a = ap.parse_args(argv)
    rep = Report()
    print("MR Studio myCobot dry run")
    cls = check_sdk(rep)
    if cls is not None:
        check_methods(rep, cls)
        check_limits(rep, cls)
    check_ports(rep)
    check_calibration(rep, a.calibration)
    print_safety()
    if a.probe and cls is not None:
        probe(rep, cls, a.probe, a.baud)
    print(f"\nResult: {'FAIL' if rep.fails else 'OK'} — {len(rep.fails)} failed, {len(rep.warns)} warnings")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
