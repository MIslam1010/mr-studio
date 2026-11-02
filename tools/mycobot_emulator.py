#!/usr/bin/env python3
"""A software stand-in for the myCobot 280 firmware on a pseudo-terminal (pty).

The emulator opens a pty pair and answers on the master side the serial frames that Elephant Robotics'
``pymycobot`` writes to the slave side.  The *real* ``pymycobot.MyCobot280`` class can therefore talk to the
*real* ``MyCobotDriver`` without an arm: the tests (``tests/test_mycobot_emulator.py``) and a dry run of the
Hardware tab both use it.  It never touches a real serial device.

Frame format and command codes are taken from the installed SDK at run time
(``pymycobot.common.ProtocolCode`` and ``DataProcessor._mesg`` / ``read``), not hard-coded::

    0xFE 0xFE  LEN  CMD  DATA...  0xFA        LEN = len(DATA) + 2

Replies use the same framing.  ``pymycobot.common.read`` accepts a reply when its CMD byte equals the command it
sent; angles are big-endian int16 of degrees x 100 (``DataProcessor._angle2int`` / ``_int2angle``).

Handled commands: POWER_ON, POWER_OFF, IS_POWER_ON, RELEASE_ALL_SERVOS, FOCUS_SERVO (0 = all),
GET_ANGLES, SEND_ANGLE, SEND_ANGLES, STOP, GET_SPEED, SET_SPEED, SET_FRESH_MODE, GET_FRESH_MODE (``fresh_mode``
starts at 0 = queue; ``fresh_mode_supported = False`` makes both unknown, like an old firmware).  Every handled set-type command is
acknowledged with a one-byte status 1 (pymycobot 4.x reads a reply after every command, see
``MyCobot280._res``).  Unknown commands get no reply (the SDK then times out and returns -1) and are counted
in ``unknown``.

Servo model (illustrative, NOT the real firmware): with power on and torque enabled each joint moves to its
set-point at ``deg_s_at_100 * speed% / 100`` deg/s (rate limited, no overshoot); STOP holds the current angle;
with torque released nothing moves (the real arm would sag or fall).  Angles are reported with the firmware's
0.01° resolution.

Serial latency: ``latency_ms`` (``--latency MS``) delays every reply by that much, plus a uniform random
``jitter_ms`` (``--jitter MS``: the delay is drawn from [latency - jitter, latency + jitter], never below 0) -- a
stand-in for a slow USB-serial bridge and firmware loop (a real myCobot 280 answers in tens of ms).  The command
takes effect when it arrives; only its reply is late.  Replies go out in order, one at a time (pymycobot waits
for each before it sends the next command).  ``reply_delays`` records the delays applied.

Fault injection (for the cable-pull and firmware-hang tests):

* ``hangup()`` closes the master side: the slave port then fails on the next read, like a USB cable pulled
  out (macOS: "Device not configured").
* ``silent = True`` keeps the port open but stops answering (a hung controller).
* ``errors = True`` (or a set of command names, e.g. ``{"STOP"}``) answers those commands with a corrupted frame
  (wrong command byte) and does not execute them: pymycobot retries 3 times and returns -1.  Counted in
  ``errored``.

Command line (prints the pty path, runs until Ctrl-C)::

    .venv/bin/python tools/mycobot_emulator.py [--angles 0 0 0 0 0 0] [--deg-s-at-100 500] [--latency 50 --jitter 10]

then choose "myCobot 280 (USB serial)" in the Hardware tab and type the printed path into *Port*.
"""
from __future__ import annotations

import argparse
import os
import random
import select
import struct
import sys
import threading
import time
import tty
from collections import Counter

try:
    from pymycobot.common import ProtocolCode as PC
except ImportError as e:  # pragma: no cover - the CLI and the tests check this first
    raise SystemExit("tools/mycobot_emulator.py needs pymycobot (pip install -r requirements-hardware.txt)") from e

HEADER, FOOTER = PC.HEADER, PC.FOOTER
#: The commands the emulator answers (names of ``pymycobot.common.ProtocolCode`` attributes).
HANDLED = ("POWER_ON", "POWER_OFF", "IS_POWER_ON", "RELEASE_ALL_SERVOS", "FOCUS_SERVO", "GET_ANGLES",
           "SEND_ANGLE", "SEND_ANGLES", "STOP", "GET_SPEED", "SET_SPEED", "SET_FRESH_MODE", "GET_FRESH_MODE")
CODE = {name: getattr(PC, name) for name in HANDLED}
NAME = {v: k for k, v in CODE.items()}


def frame(cmd: int, data: bytes = b"") -> bytes:
    """One frame in pymycobot's serial format (no CRC: the 280 class uses the 0xFA footer)."""
    return bytes([HEADER, HEADER, len(data) + 2, cmd]) + bytes(data) + bytes([FOOTER])


def split_frames(buf: bytearray) -> list[tuple[int, bytes]]:
    """Remove every complete frame from ``buf`` (in place); returns [(cmd, data)].  Garbage before a header
    and frames without the footer are dropped byte by byte (resynchronisation)."""
    out = []
    while True:
        i = buf.find(bytes([HEADER, HEADER]))
        if i < 0:
            del buf[:max(0, len(buf) - 1)]          # keep a trailing 0xFE: it may start the next header
            return out
        del buf[:i]
        if len(buf) < 4:
            return out
        n = buf[2]
        total = n + 3
        if n < 2 or n > 64:
            del buf[:1]
            continue
        if len(buf) < total:
            return out
        if buf[total - 1] != FOOTER:
            del buf[:1]
            continue
        out.append((buf[3], bytes(buf[4:total - 1])))
        del buf[:total]


class MyCobotEmulator:
    """Firmware emulator on a pty.  ``start()`` launches the I/O thread; ``port`` is the slave path."""

    def __init__(self, angles=(0.0,) * 6, deg_s_at_100: float = 500.0, powered: bool = False, verbose: bool = False,
                 latency_ms: float = 0.0, jitter_ms: float = 0.0, seed: int | None = None):
        self.angles = [float(a) for a in angles]          # servo degrees, what GET_ANGLES reports
        self.target = list(self.angles)
        self.speed_pct = 0
        self.deg_s_at_100 = float(deg_s_at_100)
        self.powered = bool(powered)
        self.torque = True                                 # False after RELEASE_ALL_SERVOS
        self.silent = False
        self.errors: bool | set = False                    # True / set of command names: corrupted replies
        self.fresh_mode = 0                                # firmware command mode: 0 queue, 1 fresh (latest first)
        self.fresh_mode_supported = True
        self.errored: Counter = Counter()                  # commands answered with a corrupted frame
        self.verbose = verbose
        if latency_ms < 0 or jitter_ms < 0:
            raise ValueError("latency_ms and jitter_ms must be >= 0")
        self.latency_ms = float(latency_ms)                # reply delay per command [ms] (see the module docstring)
        self.jitter_ms = float(jitter_ms)                  # +- uniform jitter on that delay [ms]
        self._rng = random.Random(seed)
        self.reply_delays: list[float] = []                # delays applied [s], capped like ``log``
        self.counts: Counter = Counter()                   # handled frames per command name
        self.unknown: Counter = Counter()                  # unhandled command codes
        self.log: list[tuple[float, str, tuple]] = []      # (t, command, decoded args), capped
        self.lock = threading.Lock()
        self._master, self._slave = os.openpty()
        tty.setraw(self._slave)                            # no echo / line editing before pyserial opens it
        self.port = os.ttyname(self._slave)
        self._stop_r, self._stop_w = os.pipe()
        self._thread: threading.Thread | None = None
        self._t_last = time.monotonic()
        self._closed = False

    # -- life cycle ------------------------------------------------------------------------------
    def start(self) -> "MyCobotEmulator":
        self._thread = threading.Thread(target=self._loop, name="mycobot-emulator", daemon=True)
        self._thread.start()
        return self

    def hangup(self) -> None:
        """USB cable pulled: close the master side; the slave fails on its next read."""
        with self.lock:
            if self._master is not None:
                os.write(self._stop_w, b"x")
        if self._thread is not None:
            self._thread.join(2.0)
        with self.lock:
            if self._master is not None:
                os.close(self._master)
                self._master = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.hangup()
        for fd in (self._slave, self._stop_r, self._stop_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # -- servo model -----------------------------------------------------------------------------
    def _advance(self) -> None:
        now = time.monotonic()
        dt, self._t_last = now - self._t_last, now
        if not (self.powered and self.torque) or self.speed_pct <= 0:
            return
        step = self.deg_s_at_100 * self.speed_pct / 100.0 * dt
        for j, (a, b) in enumerate(zip(self.angles, self.target)):
            d = b - a
            self.angles[j] = b if abs(d) <= step else a + (step if d > 0 else -step)

    def reported_angles(self) -> list[float]:
        with self.lock:
            self._advance()
            return [round(a, 2) for a in self.angles]

    def reply_delay(self) -> float:
        """Delay [s] before the next reply: latency +- uniform jitter, never negative."""
        ms = self.latency_ms
        if self.jitter_ms > 0:
            ms += self._rng.uniform(-self.jitter_ms, self.jitter_ms)
        return max(0.0, ms) / 1000.0

    # -- protocol --------------------------------------------------------------------------------
    def handle(self, cmd: int, data: bytes) -> bytes | None:
        """Apply one command; returns the reply frame (None = no reply)."""
        name = NAME.get(cmd)
        if name in ("SET_FRESH_MODE", "GET_FRESH_MODE") and not self.fresh_mode_supported:
            name = None
        if name is None:
            self.unknown[cmd] += 1
            return None
        if self.errors is True or (self.errors and name in self.errors):
            with self.lock:
                self.errored[name] += 1
            return frame(0x00, b"\x00")                    # wrong command byte: the SDK rejects the reply
        ack = frame(cmd, b"\x01")
        with self.lock:
            self._advance()
            self.counts[name] += 1
            args: tuple = ()
            if name == "POWER_ON":
                self.powered, self.torque = True, True
                reply = ack
            elif name == "POWER_OFF":
                self.powered = False
                reply = ack
            elif name == "IS_POWER_ON":
                reply = frame(cmd, bytes([1 if self.powered else 0]))
            elif name == "RELEASE_ALL_SERVOS":
                self.torque = False
                self.target = list(self.angles)
                reply = ack
            elif name == "FOCUS_SERVO":
                if not data or data[0] in (0, *range(1, 7)):
                    self.torque = True
                    self.target = list(self.angles)
                reply = ack
            elif name == "GET_ANGLES":
                ints = [int(round(a * 100)) for a in self.angles]
                reply = frame(cmd, struct.pack(">6h", *ints))
            elif name == "SEND_ANGLES":
                if len(data) < 13:
                    return None
                deg = [v / 100.0 for v in struct.unpack(">6h", data[:12])]
                args = (deg, data[12])
                if self.powered and self.torque:
                    self.target, self.speed_pct = deg, data[12]
                reply = ack
            elif name == "SEND_ANGLE":
                if len(data) < 4:
                    return None
                j = data[0]
                deg = struct.unpack(">h", data[1:3])[0] / 100.0
                args = (j, deg, data[3])
                if 1 <= j <= 6 and self.powered and self.torque:
                    self.target[j - 1], self.speed_pct = deg, data[3]
                reply = ack
            elif name == "STOP":
                self.target = list(self.angles)
                reply = ack
            elif name == "GET_SPEED":
                reply = frame(cmd, bytes([min(255, self.speed_pct)]))
            elif name == "SET_SPEED":
                if data:
                    self.speed_pct = data[0]
                reply = ack
            elif name == "SET_FRESH_MODE":
                if data and data[0] in (0, 1):
                    self.fresh_mode = data[0]
                args = (data[0] if data else None,)
                reply = ack
            elif name == "GET_FRESH_MODE":
                reply = frame(cmd, bytes([self.fresh_mode]))
            else:  # pragma: no cover - HANDLED and the branches above agree
                return None
            if len(self.log) < 100000:
                self.log.append((time.monotonic(), name, args))
        if self.verbose:
            print(f"{name:<18} {args if args else ''}", flush=True)
        return reply

    def _loop(self) -> None:
        buf = bytearray()
        while True:
            master = self._master
            if master is None:
                return
            try:
                r, _, _ = select.select([master, self._stop_r], [], [], 0.05)
            except (OSError, ValueError):
                return
            if self._stop_r in r:
                return
            if master not in r:
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:            # slave closed by the client: wait for the next open
                time.sleep(0.01)
                continue
            if not chunk:
                time.sleep(0.01)
                continue
            buf.extend(chunk)
            for cmd, data in split_frames(buf):
                reply = self.handle(cmd, data)
                if reply is None:
                    continue
                delay = self.reply_delay()
                if delay > 0:
                    if len(self.reply_delays) < 100000:
                        self.reply_delays.append(delay)
                    try:                                   # wait, but wake at once for hangup() / close()
                        r, _, _ = select.select([self._stop_r], [], [], delay)
                    except (OSError, ValueError):
                        return
                    if r:
                        return
                    master = self._master
                    if master is None:
                        return
                if not self.silent:
                    try:
                        os.write(master, reply)
                    except OSError:
                        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="myCobot 280 firmware emulator on a pseudo-terminal (no hardware).")
    ap.add_argument("--angles", type=float, nargs=6, default=[0.0] * 6, metavar="DEG",
                    help="initial servo angles [deg] (default: all 0)")
    ap.add_argument("--deg-s-at-100", type=float, default=500.0,
                    help="servo speed at firmware speed 100%% [deg/s] (illustrative; default 500)")
    ap.add_argument("--latency", type=float, default=0.0, metavar="MS",
                    help="delay every reply by MS milliseconds (a slow serial link; default 0)")
    ap.add_argument("--jitter", type=float, default=0.0, metavar="MS",
                    help="uniform random +-MS on that delay (default 0)")
    ap.add_argument("--powered", action="store_true", help="start with the servos powered")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not print every command")
    a = ap.parse_args(argv)
    if a.latency < 0 or a.jitter < 0:
        ap.error("--latency and --jitter must be >= 0")
    emu = MyCobotEmulator(a.angles, a.deg_s_at_100, powered=a.powered, verbose=not a.quiet,
                          latency_ms=a.latency, jitter_ms=a.jitter).start()
    print(f"myCobot emulator (pymycobot protocol) on {emu.port}"
          + (f", reply latency {a.latency:g} ms +- {a.jitter:g} ms" if a.latency or a.jitter else ""), flush=True)
    print("Use this path as the Port of the myCobot back-end; Ctrl-C to quit.", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        emu.close()
        print(f"frames: {dict(emu.counts)}; unknown codes: {dict(emu.unknown)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
