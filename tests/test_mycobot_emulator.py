"""The real myCobot stack without an arm: MyCobotDriver <-> pymycobot.MyCobot280 <-> firmware emulator on a pty.

``tools/mycobot_emulator.py`` answers pymycobot's serial frames on a pseudo-terminal, so these tests exercise the
real SDK (framing, encoding, retries, timeouts) and the real driver, run loop and Hardware tab.  Skipped when
pymycobot is not installed (it is optional: ``requirements-hardware.txt``) and on Windows (no pty).
No real serial device is opened.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if sys.platform == "win32":
    pytest.skip("the firmware emulator needs a POSIX pty", allow_module_level=True)
pytest.importorskip("pymycobot", reason="optional myCobot SDK not installed (pip install -r requirements-hardware.txt)")

import mycobot_dryrun as dry  # noqa: E402
from mycobot_emulator import MyCobotEmulator, frame, split_frames  # noqa: E402

from mrstudio.hardware.base import HardwareError  # noqa: E402
from mrstudio.hardware.mycobot import MyCobotCalibration, MyCobotDriver, sdk_version  # noqa: E402
from mrstudio.hardware.safety import SafetySupervisor  # noqa: E402
from mrstudio.hardware.session import HardwareRun, quintic_move  # noqa: E402

START_DEG = [10.0, -20.0, 30.0, 0.0, 5.0, -5.0]
CAL = MyCobotCalibration(sign=[1, -1, 1, 1, -1, 1], offset_deg=[30, 0, -10, 0, 5, 0])
RATE = 50.0


@pytest.fixture
def emu():
    try:
        e = MyCobotEmulator(angles=START_DEG).start()
    except PermissionError as err:      # e.g. a macOS sandbox that forbids pseudo-terminals: fail loudly, never skip
        pytest.fail(f"cannot open a pseudo-terminal for the emulator ({err}); run the tests outside the sandbox")
    yield e
    e.close()


def _connected(emu, cal=CAL) -> MyCobotDriver:
    d = MyCobotDriver(emu.port, 115200, calibration=cal)      # real pymycobot class (no sdk_factory)
    d.connect()
    return d


def _stream(run: HardwareRun, q_now, timeout: float = 15.0, on_tick=None) -> None:
    """Real-time 50 Hz loop like the Hardware tab's streaming worker."""
    run.start(q_now)
    t_next = time.monotonic()
    t_end = t_next + timeout
    while not run.finished and time.monotonic() < t_end:
        t_next += 1.0 / RATE
        time.sleep(max(0.0, t_next - time.monotonic()))
        if on_tick is not None:
            on_tick(run)
        run.safe_step()
    assert run.finished, "run did not finish in time"


# ------------------------------------------------------------------------------------------- protocol
def test_frame_codec_matches_pymycobot():
    from pymycobot import MyCobot280
    from pymycobot.common import ProtocolCode as PC
    sdk = MyCobot280.__new__(MyCobot280)                     # encoder only: no port is opened
    sdk.calibration_parameters = lambda **kw: None
    cmd, _, _ = super(MyCobot280, sdk)._mesg(PC.SEND_ANGLES, [1000, -2000, 0, 0, 0, 0], 20)
    buf = bytearray(b"\x00\x13" + bytes(cmd) + bytes(frame(PC.GET_ANGLES)))
    frames = split_frames(buf)
    assert [c for c, _ in frames] == [PC.SEND_ANGLES, PC.GET_ANGLES] and not buf
    assert len(frames[0][1]) == 13 and frames[0][1][-1] == 20


def test_dryrun_reports_all_sdk_methods(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MRSTUDIO_MYCOBOT_CALIBRATION", str(tmp_path / "cal.json"))
    CAL.save(tmp_path / "cal.json")
    assert dry.main([]) == 0
    out = capsys.readouterr().out
    n = len(dry.sdk_methods_called())
    assert n >= 5 and f"all {n} SDK methods present" in out
    for m in ("get_angles", "send_angles", "stop", "release_all_servos"):
        assert m in dry.sdk_methods_called()
    assert "FAIL" not in out.replace("Result: OK", "")
    (tmp_path / "cal.json").write_text("{")
    assert dry.main([]) == 1                                  # an unreadable calibration file fails the dry run


def test_probe_sends_read_commands_only(emu, tmp_path, monkeypatch, capsys):
    from pymycobot import MyCobot280
    monkeypatch.setenv("MRSTUDIO_MYCOBOT_CALIBRATION", str(tmp_path / "none.json"))
    rep = dry.Report()
    dry.probe(rep, MyCobot280, emu.port, 115200, n=10)
    assert not rep.fails and set(emu.counts) == {"IS_POWER_ON", "GET_ANGLES"}
    assert emu.counts["GET_ANGLES"] == 10 and emu.angles == START_DEG
    assert "10.0, -20.0, 30.0" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- driver
def test_connect_read_and_calibration_mapping(emu):
    d = _connected(emu)
    try:
        assert emu.counts["POWER_ON"] == 1 and emu.powered
        q = d.read_joints()
        assert d.last_read_fresh
        np.testing.assert_allclose(CAL.to_servo_deg(q), START_DEG, atol=1e-9)
        np.testing.assert_allclose(np.degrees(q), [-20, 20, 40, 0, 0, -5], atol=1e-9)   # s * (servo - o)
        assert d.describe()["sdk_version"] == sdk_version()
    finally:
        d.disconnect()
    assert not emu.unknown


def test_stream_2s_trajectory_then_latch(emu):
    d = _connected(emu)
    try:
        sup = SafetySupervisor(session="H1")
        q0 = d.read_joints()
        q1 = q0 + np.radians([20, -15, 10, 0, 10, 0])
        t, q = quintic_move(q0, q1, sup.speed_cap, dt=1 / RATE, T=2.0)
        assert not sup.trajectory_problems(t, q)
        run = HardwareRun(d, sup, t, q)
        _stream(run, q0)
        assert run.reason == "trajectory complete" and run.error is None, run.reason
        assert emu.counts["SEND_ANGLES"] >= 100                     # ~2.5 s at 50 Hz
        assert run.stalls == 0 and d.read_failures == 0
        err = np.degrees(np.abs(run.log.cmd - run.log.meas)).max()
        assert err < 2.0                                            # the emulated servos track the stream
        np.testing.assert_allclose(emu.reported_angles(), CAL.to_servo_deg(q1), atol=0.02)
        assert all(0 < s <= 100 for _, n, a in emu.log if n == "SEND_ANGLES" for s in [a[1]])
        # latched stop: halt goes out, nothing moves afterwards
        n_send = emu.counts["SEND_ANGLES"]
        assert d.stop() is True and d.halt_confirmed and emu.counts["STOP"] == 1
        assert d.send_joints(q0, 0.25) is False and emu.counts["SEND_ANGLES"] == n_send
        d.set_torque(False)
        assert not emu.torque and emu.counts["RELEASE_ALL_SERVOS"] == 1
        d.reset_latch()                                             # torque back on first
        assert emu.torque and emu.counts["FOCUS_SERVO"] == 1
    finally:
        d.disconnect()
    assert not emu.unknown


def test_stop_mid_run_halts_the_stream(emu):
    d = _connected(emu)
    try:
        sup = SafetySupervisor(session="H1")
        q0 = d.read_joints()
        t, q = quintic_move(q0, q0 + np.radians([30, 0, 0, 0, 0, 0]), sup.speed_cap, dt=1 / RATE, T=2.0)
        run = HardwareRun(d, sup, t, q, auto_stop_at=0.8)
        _stream(run, q0)
        assert run.t_stop == pytest.approx(0.8, abs=0.1) and run.stop_confirmed and d.latched
        names = [n for _, n, _ in emu.log]
        i_stop = names.index("STOP")
        assert "SEND_ANGLES" not in names[i_stop + 1:]              # no set-point after the halt
        held = emu.reported_angles()
        time.sleep(0.2)
        assert emu.reported_angles() == held
        assert 0 < held[0] - START_DEG[0] < 30                      # stopped part-way
    finally:
        d.disconnect()


def test_cable_pull_latches_within_one_tick(emu):
    d = _connected(emu)
    sup = SafetySupervisor(session="H1")
    q0 = d.read_joints()
    t, q = quintic_move(q0, q0 + np.radians([20, 10, -10, 0, 0, 0]), sup.speed_cap, dt=1 / RATE, T=2.0)
    run = HardwareRun(d, sup, t, q)
    pulled = {}

    def pull(r):
        if "t" not in pulled and r.t0 is not None and time.monotonic() - r.t0 > 0.5:
            pulled["t"] = time.monotonic() - r.t0
            emu.hangup()                                            # USB cable out

    _stream(run, q0, on_tick=pull)
    try:
        assert run.error and d.latched and run.t_stop is not None
        assert run.t_stop - pulled["t"] < 0.1                       # the next serial call fails at once
        assert run.stop_confirmed is False                          # the halt cannot reach the arm: say so
        assert any(e["kind"] == "error" for e in run.log.events)
    finally:
        d.disconnect()


def _fault_mid_run(emu, d, fault, after=0.3):
    """Stream a 2 s move and inject ``fault(emu)`` ``after`` s into it; returns (run, latch delay [s])."""
    sup = SafetySupervisor(session="H1")
    q0 = d.read_joints()
    t, q = quintic_move(q0, q0 + np.radians([10, 0, 0, 0, 0, 0]), sup.speed_cap, dt=1 / RATE, T=2.0)
    run = HardwareRun(d, sup, t, q)
    hit = {}

    def inject(r):
        if "t" not in hit and r.t0 is not None and time.monotonic() - r.t0 > after:
            hit["t"] = time.monotonic() - r.t0
            fault(emu)

    _stream(run, q0, timeout=30.0, on_tick=inject)
    assert run.t_stop is not None and "t" in hit
    return run, run.t_stop - hit["t"]


def test_silent_firmware_latches_within_one_second(emu):
    """A hung controller (port open, no answers).  pymycobot 4.0.7 would wait 3 x 0.5 s per call and return -1
    (the driver used to need ~9.4 s to latch); the driver's per-call deadline latches it in ~0.4 s."""
    d = _connected(emu)
    try:
        run, delay = _fault_mid_run(emu, d, lambda e: setattr(e, "silent", True))
        assert run.error and "no answer" in run.error and d.latched
        assert delay <= 1.0, delay
        assert run.stop_confirmed is False and d.halt_confirmed is False       # never claims a halt it can't see
        assert d.deadline_misses >= 1
        # the late halt still goes out once the SDK gives up on the hung call; no set-point follows it
        t_end = time.monotonic() + 6.0
        while "STOP" not in emu.counts and time.monotonic() < t_end:
            time.sleep(0.05)
        names = [n for _, n, _ in emu.log]
        assert "STOP" in names and "SEND_ANGLES" not in names[names.index("STOP"):]
    finally:
        d.disconnect()


def test_error_replies_latch_at_once_and_stop_is_not_confirmed(emu):
    """Firmware that answers with corrupted frames: pymycobot returns -1 (no exception); the driver treats it as a
    failure -- the set-point stream latches at once and the halt is reported NOT confirmed."""
    d = _connected(emu)
    try:
        run, delay = _fault_mid_run(emu, d, lambda e: setattr(e, "errors", True))
        assert run.error and "did not acknowledge" in run.error and "-1" in run.error
        assert delay < 0.2, delay
        assert run.stop_confirmed is False and emu.errored["SEND_ANGLES"] >= 1 and emu.errored["STOP"] >= 1
    finally:
        d.disconnect()


def test_stop_without_acknowledgement_is_not_confirmed(emu):
    d = _connected(emu)
    try:
        emu.errors = {"STOP"}
        assert d.stop() is False and d.halt_confirmed is False and d.latched
        assert emu.errored["STOP"] == 6 and emu.counts["STOP"] == 0      # 2 tries x 3 SDK attempts, none executed
        emu.errors = False
        assert d.stop() is True and d.halt_confirmed and emu.counts["STOP"] == 1   # STOP again retries
    finally:
        d.disconnect()


def test_connect_sets_and_verifies_fresh_mode(emu):
    assert emu.fresh_mode == 0
    d = _connected(emu)
    try:
        assert emu.fresh_mode == 1 and d.fresh_mode == 1 and d.describe()["fresh_mode"] == 1
        assert emu.counts["SET_FRESH_MODE"] == 1 and emu.counts["GET_FRESH_MODE"] == 1
        names = [n for _, n, _ in emu.log]
        assert names.index("SET_FRESH_MODE") < names.index("GET_ANGLES")    # before the first read / set-point
    finally:
        d.disconnect()


def test_connect_fails_when_fresh_mode_cannot_be_set(emu):
    emu.fresh_mode_supported = False                                    # an old firmware: no reply to 0x16
    d = MyCobotDriver(emu.port, 115200, calibration=CAL)
    with pytest.raises(HardwareError, match="set_fresh_mode"):
        d.connect()
    assert not d.connected and d.mc is None and emu.counts["SEND_ANGLES"] == 0


# ------------------------------------------------------------------------------------------- Hardware tab
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    yield app
    from mrstudio.gui.tabs.hardware_tab import HardwareTab
    for w in app.topLevelWidgets():
        if isinstance(w, HardwareTab):
            w.shutdown()
            w.close()
            w.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_hardware_tab_session_over_emulator(qapp, emu, monkeypatch, tmp_path):
    """Hardware tab, back-end "myCobot 280 (USB serial)", port = the emulator's pty: one real-time session on the
    streaming worker thread, log with sidecars, then close."""
    import json
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import _BACKENDS, HardwareTab
    cal_path = tmp_path / "mycobot_calibration.json"
    monkeypatch.setenv("MRSTUDIO_MYCOBOT_CALIBRATION", str(cal_path))
    CAL.save(cal_path)
    tab = HardwareTab(AppContext())
    assert tab.sdk_factory is None
    tab.backend_box.setCurrentText(_BACKENDS[1])
    assert "myCobot 280" in tab.backend_box.currentText()
    tab.port_box.setCurrentText(emu.port)
    assert tab.connect_backend(), tab.conn_lbl.text()
    assert isinstance(tab.driver, MyCobotDriver) and tab.driver.offset_calibrated
    for cb in tab.check_boxes.values():
        cb.setChecked(True)
    tab.source_box.setCurrentIndex(0)
    tab._set_target(tab.q_meas + np.radians([10, 0, 0, 0, 0, 0]))
    assert tab.start_run(), tab.status_lbl.text()
    assert tab.worker is not None                                 # real time: streaming worker thread
    t_end = time.monotonic() + 20.0
    while not tab.run.finished and time.monotonic() < t_end:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(100):
        qapp.processEvents()
        if tab.worker is None:
            break
        time.sleep(0.01)
    run = tab.run
    assert run.finished and run.reason == "trajectory complete", (run.reason, run.error)
    assert tab.worker is None and "Run complete" in tab.status_lbl.text()
    assert emu.counts["SEND_ANGLES"] > 20 and not emu.unknown
    assert np.median(np.diff(tab.last_log.t)) <= 0.05
    assert emu.reported_angles()[0] == pytest.approx(START_DEG[0] + 10, abs=0.05)   # J1 sign +1
    tab.save_log(str(tmp_path / "h.csv"), "361")
    meta = json.loads((tmp_path / "h.meta.json").read_text())
    assert meta["meta"]["port"] == emu.port and meta["meta"]["sdk_version"] == sdk_version()
    assert meta["meta"]["sign"] == [1, -1, 1, 1, -1, 1]
    drv = tab.driver
    tab.shutdown()
    assert drv.latched and not drv.connected and tab.driver is None
    assert emu.counts["STOP"] >= 1                                # closing halts the arm
