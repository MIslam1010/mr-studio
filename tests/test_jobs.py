"""JobRunner: stale results are dropped, cancel is honoured, the GUI thread never blocks on a worker."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump(app, cond, timeout=10.0):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.002)
        assert time.time() - t0 < timeout, "timed out"
    app.processEvents()


def _sleeper(value, seconds, polls=None):
    def fn(progress, cancelled):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if polls is not None and cancelled():
                polls.append("cancelled")
                return "cancelled"
            progress((time.time() - t0) / seconds)
            time.sleep(0.005)
        return value
    return fn


def test_result_and_progress_delivered(qapp):
    from mrstudio.gui.jobs import JobRunner
    r = JobRunner()
    got, prog = [], []
    r.run(_sleeper("A", 0.1), on_done=got.append, on_progress=prog.append, label="a")
    assert r.is_busy()
    r.wait(10)
    assert got == ["A"] and prog and all(0 <= f <= 1 for f in prog) and not r.is_busy()


def test_stale_result_and_progress_dropped(qapp):
    from mrstudio.gui.jobs import JobRunner
    r = JobRunner()
    got, prog_old = [], []
    r.run(_sleeper("old", 0.3), on_done=got.append, on_progress=prog_old.append)
    _pump(qapp, lambda: prog_old)                 # first job is running and reporting
    r.run(_sleeper("new", 0.05), on_done=got.append)
    n_old = len(prog_old)
    r.wait(10)
    time.sleep(0.4)                               # let the superseded job finish too
    _pump(qapp, lambda: True)
    assert got == ["new"]                         # the old result never reaches the caller
    assert len(prog_old) == n_old                 # nor does its progress after it was superseded


def test_cancel_sets_token_and_drops_result(qapp):
    from mrstudio.gui.jobs import JobRunner, live_workers
    r = JobRunner()
    got, errs, polls = [], [], []
    r.run(_sleeper("x", 5.0, polls), on_done=got.append, on_error=errs.append)
    time.sleep(0.05)
    r.cancel()
    assert not r.is_busy()
    _pump(qapp, lambda: polls)                    # the job saw cancelled() == True and returned early
    _pump(qapp, lambda: live_workers() == 0)
    assert got == [] and errs == []


def test_no_blocking_wait_in_gui_thread(qapp):
    """Superseding or cancelling a long job that ignores cancel() must return immediately."""
    from mrstudio.gui.jobs import JobRunner, live_workers
    r = JobRunner()
    got = []
    r.run(_sleeper("slow", 1.0), on_done=got.append)          # never polls cancelled()
    t0 = time.perf_counter()
    r.run(_sleeper("fast", 0.01), on_done=got.append)
    r.cancel()
    r.run(_sleeper("last", 0.01), on_done=got.append)
    assert time.perf_counter() - t0 < 0.1
    r.wait(10)
    assert got == ["last"]
    _pump(qapp, lambda: live_workers() == 0)      # retired workers were kept alive and finished cleanly
    assert got == ["last"]


def test_errors_reported_and_runner_deletable_while_running(qapp):
    from mrstudio.gui.jobs import JobRunner, live_workers

    def boom(progress, cancelled):
        raise ValueError("boom")

    r = JobRunner()
    errs = []
    r.run(boom, on_done=lambda _: None, on_error=errs.append)
    r.wait(10)
    assert errs and "ValueError: boom" in errs[0]
    # an exception in on_done goes to on_error too
    errs.clear()
    r.run(_sleeper(1, 0.0), on_done=lambda _: 1 / 0, on_error=errs.append)
    r.wait(10)
    assert errs and "ZeroDivisionError" in errs[0]
    # deleting the runner while its worker runs must not crash (worker is retired, kept alive)
    r2 = JobRunner()
    r2.run(_sleeper("z", 0.2), on_done=lambda _: None)
    r2.deleteLater()
    del r2
    _pump(qapp, lambda: live_workers() == 0)


def test_shutdown_is_bounded(qapp):
    from mrstudio.gui.jobs import JobRunner
    r = JobRunner()
    r.run(_sleeper("x", 3.0, []), on_done=lambda _: None)
    t0 = time.perf_counter()
    r.shutdown(500)
    assert time.perf_counter() - t0 < 1.0 and not r.is_busy()


def test_progress_info_forwarded(qapp):
    from mrstudio.gui.jobs import JobRunner
    r = JobRunner()
    seen = []

    def fn(progress, cancelled):
        progress(0.5, "step", "half way")
        progress(0.7, "result", 3, {"x": 1})
        return "ok"

    r.run(fn, on_done=lambda _: None, on_progress=lambda f, *info: seen.append((f, info)))
    r.wait(10)
    assert seen == [(0.5, ("step", "half way")), (0.7, ("result", 3, {"x": 1}))]


# ---------------------------------------------------------------- tabs built on JobRunner / Player
def _dispose(app, tab):
    """Quit-time path of a tab (shutdown hook), then delete it now rather than at interpreter exit."""
    from PySide6 import QtCore
    from mrstudio.gui.jobs import live_workers
    tab.shutdown()
    _pump(app, lambda: live_workers() == 0, timeout=30)
    tab.close()
    tab.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture(scope="module")
def ctx(qapp):
    from mrstudio.gui import theme
    from mrstudio.gui.base import AppContext
    theme.apply(qapp)
    return AppContext()


def test_player_does_not_autoplay_in_hidden_owner(qapp):
    import numpy as np
    from PySide6 import QtWidgets
    from mrstudio.gui.widgets import Player
    owner = QtWidgets.QWidget()
    p = Player(owner)
    p.set_times(np.linspace(0, 1, 11))
    assert not p.playing() and not p.timer.isActive()
    owner.show()
    p.set_times(np.linspace(0, 1, 11))
    assert p.playing() and p.timer.isActive()
    p.stop()
    owner.hide()


def test_chunked_simulation_stays_on_the_dt_grid():
    import numpy as np
    from mrstudio.dynamics import DynamicsEngine
    from mrstudio.gui.tabs.dynamics_tab import simulate_chunked
    from mrstudio.robots import get_robot
    r = get_robot("Planar 2R arm")
    eng = DynamicsEngine(r, "pinocchio")
    dt, T = 0.003, 1.0                                # T/dt = 333.3: not a multiple of the chunk count
    res = simulate_chunked(eng, r.home, np.zeros(2), lambda t, q, dq: np.zeros(2), T, dt, "rk4", 0.0,
                           lambda f: None, lambda: False, chunks=20)
    n = int(round(T / dt))
    np.testing.assert_allclose(res.t, np.arange(n + 1) * dt, atol=1e-12)
    assert len(res.theta) == n + 1


def test_dynamics_tab_jobs_superseded_without_blocking(qapp, ctx):
    from mrstudio.gui.tabs.dynamics_tab import DynamicsTab
    tab = DynamicsTab(ctx)
    _pump(qapp, lambda: tab.jobs.is_busy())             # the automatic RNEA run 50 ms after construction
    tab.wait_job(60)
    assert tab.anim[0] == "id" and not tab.player.playing()
    tab.Tsim.setValue(0.5)
    t0 = time.perf_counter()
    tab.run_forward()
    tab.run_forward()                                   # supersedes the first one: must not wait for it
    assert time.perf_counter() - t0 < 0.5
    tab.wait_job(60)
    assert tab.anim is not None and tab.anim[0] == "fd"
    assert not tab.player.playing()                     # hidden tab: no 30 ms playback timer
    # the integrator demo on the planar 2R arm while the UR5 is shown: plots, but leaves the 3-D animation alone
    anim = tab.anim
    tab.cmp_target.setCurrentIndex(1)
    tab.compare_integrators()
    tab.wait_job(60)
    assert "Planar 2R" in tab.cmp_lbl.text() and "10.0 ms" in tab.cmp_lbl.text()
    assert tab.anim is anim
    _dispose(qapp, tab)


def test_impedance_unstable_settings_need_confirmation(qapp, ctx):
    from mrstudio import task_control as tc
    from mrstudio.gui.tabs.advanced_control_tab import AdvancedControlTab
    tab = AdvancedControlTab(ctx)
    tab.mode_box.setCurrentIndex(tc.MODES.index("impedance"))
    p = tab.par["impedance"]
    p["m"].setValue(0.25)
    p["b"].setValue(500.0)
    assert "unstable at 1 kHz" in tab.imp_warn.text()
    tab.run()
    assert not tab.jobs.is_busy() and "unstable" in tab.status_lbl.text()
    tab.cb_unstable.setChecked(True)
    tab.run()
    assert tab.jobs.is_busy()
    tab.cancel_jobs()
    p["m"].setValue(2.0)
    p["b"].setValue(20.0)
    assert "unstable" not in tab.imp_warn.text()
    _dispose(qapp, tab)
