"""mrstudio:// launch links (Knowledge Map -> MR Studio), the Practice tab's Problems | Lessons | Assignment panels,
and the URL-scheme registration in the installer configs."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"
UR5 = "UR5 (6R industrial)"


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


@pytest.fixture(scope="module")
def win(qapp, tmp_path_factory):
    os.environ["MRSTUDIO_SETTINGS"] = str(tmp_path_factory.mktemp("settings") / "s.ini")
    from mrstudio.gui.app import MainWindow
    from mrstudio.gui.base import AppContext
    w = MainWindow(AppContext("Planar 2R arm"))
    w.persist = False
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    qapp.processEvents()


def _wait(qapp, s=0.1):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < s:
        qapp.processEvents()


def _current(win) -> str:
    return type(win.pages[win.tabs.currentIndex()]).__name__


def test_launch_url_opens_tab_and_robot(win, qapp):
    assert win.open_url("mrstudio://open?tab=MobileTab&robot=UR5%20(6R%20industrial)")
    _wait(qapp)
    assert _current(win) == "MobileTab"
    assert win.ctx.robot.name == UR5 and win.robot_box.currentText() == UR5
    assert not win.robot_box.isEnabled()          # the mobile tab has its own mechanism
    from mrstudio.kmap_links import TAB_ORDER
    assert win.open_url(f"mrstudio://open?tab={TAB_ORDER.index('GraspingTab')}")
    assert _current(win) == "GraspingTab"
    assert win.open_url("mrstudio://open?tab=FKTab&robot=Planar%202R%20arm")
    assert _current(win) == "FKTab" and win.ctx.robot.name == "Planar 2R arm"


def test_launch_url_presets(win, qapp):
    # tab with apply_preset (Grasping / Mobile): by name part
    from mrstudio import grasping as gr
    name = list(gr.PRESETS)[-1]
    assert win.open_url("mrstudio://open?tab=GraspingTab&preset=" + name.replace(" ", "%20"))
    assert win.pages[win.page_index("GraspingTab")].preset_box.currentText() == name
    # tabs without apply_preset: their preset_box (Haptics load_preset, Closed chains _preset)
    from mrstudio import haptics as hp
    key = "Virtual wall — hand pushes in"
    assert key in hp.PRESETS
    assert win.open_url("mrstudio://open?tab=HapticsTab&preset=" + key)
    assert win.pages[win.page_index("HapticsTab")].preset_box.currentText() == key
    assert win.open_url("mrstudio://open?tab=ClosedChainTab&preset=Crank-rocker")
    assert win.pages[win.page_index("ClosedChainTab")].preset_box.currentText() == "Crank-rocker"


def test_launch_url_bad_input_is_harmless(win, qapp):
    win.tabs.setCurrentIndex(0)
    assert not win.open_url("https://example.com/?tab=MobileTab")
    assert not win.open_url("mrstudio://delete?tab=1")
    assert not win.open_url("mrstudio://open?tab=NoSuchTab")
    assert _current(win) == "FKTab"
    win.open_url("mrstudio://open?tab=FKTab&robot=No%20such%20robot")
    assert win.ctx.robot.name in __import__("mrstudio.robots", fromlist=["ROBOTS"]).ROBOTS


def test_rejected_url_is_logged_truncated(win, qapp, monkeypatch):
    import mrstudio.gui.app as app_mod
    logged = []
    monkeypatch.setattr(app_mod, "log_text", logged.append)
    long_url = "mrstudio://delete?tab=1&x=" + "A" * 500 + "\n[forged] line"
    assert not win.open_url(long_url)
    assert len(logged) == 1 and "launch URL ignored" in logged[0]
    assert repr(long_url[:200]) in logged[0] and "A" * 300 not in logged[0] and "\n" not in logged[0]
    assert "more chars" in logged[0]
    logged.clear()
    f = app_mod.UrlOpenFilter()                       # the macOS link path ends in the same open_url
    f.set_window(win)
    from PySide6 import QtCore, QtGui
    assert f.eventFilter(None, QtGui.QFileOpenEvent(QtCore.QUrl("mrstudio://evil?tab=1"))) is True
    assert len(logged) == 1 and "'mrstudio://evil?tab=1'" in logged[0]
    assert app_mod.url_for_log("mrstudio://open?tab=0") == "'mrstudio://open?tab=0'"


def test_file_open_event_is_queued_then_delivered(win, qapp):
    from PySide6 import QtCore, QtGui
    from mrstudio.gui.app import UrlOpenFilter
    f = UrlOpenFilter()
    ev = QtGui.QFileOpenEvent(QtCore.QUrl("mrstudio://open?tab=MobileTab"))
    assert f.eventFilter(None, ev) is True          # consumed
    assert f.pending == ["mrstudio://open?tab=MobileTab"]
    win.tabs.setCurrentIndex(0)
    f.set_window(win)
    assert f.pending == [] and _current(win) == "MobileTab"
    other = QtGui.QFileOpenEvent("/tmp/some file.txt")
    assert f.eventFilter(None, other) is False      # not ours


def test_practice_subtabs_lessons_and_assignment(win, qapp):
    from mrstudio.gui.tabs.practice_host import PracticeHost
    i = win.page_index("PracticeTab")
    win.tabs.setCurrentIndex(i)
    _wait(qapp)
    tab = win.pages[i]
    assert isinstance(tab, PracticeHost)
    assert [tab.side.tabText(k) for k in range(tab.side.count())] == ["Problems", "Lessons", "Assignment"]
    assert tab.lessons_panel is None and tab.assignment_panel is None      # built on first use

    tab.side.setCurrentIndex(1)
    _wait(qapp)
    lp = tab.lessons_panel
    assert lp is not None and tab.problem is None and not tab.reveal_btn.isEnabled()
    assert "Step" in tab.statement.content()
    # "Go to tab" -> main window switches to the step's tab and robot
    t = lp.lesson.steps[lp.k].target
    lp._goto()
    _wait(qapp)
    assert _current(win) == t.tab
    if t.robot:
        assert win.ctx.robot.name == t.robot

    win.tabs.setCurrentIndex(i)
    tab.side.setCurrentIndex(2)
    _wait(qapp)
    assert tab.assignment_panel is not None
    from mrstudio import problems as P
    p = P.generate(P.TOPIC_KEYS[0], 7, 2)
    tab.show_problem(p, reveal_allowed=False)
    assert not tab.reveal_btn.isEnabled()
    tab.toggle_reveal(True)                                   # even if forced, nothing is revealed
    assert "Worked solution" in tab.solution.content() and "not shown" in tab.solution.content()

    tab.side.setCurrentIndex(0)                                 # back to Problems: Reveal works again
    _wait(qapp)
    assert tab.problem is not None and tab.reveal_btn.isEnabled()


def test_run_app_open_screenshot(tmp_path):
    """``run_app.py --open URL --screenshot DIR --only`` saves the requested tab."""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MRSTUDIO_SETTINGS=str(tmp_path / "s.ini"),
               MRSTUDIO_NO_EQ_CACHE="1")
    r = subprocess.run([sys.executable, str(ROOT / "run_app.py"), "--open",
                        "mrstudio://open?tab=MobileTab&robot=UR5%20(6R%20industrial)",
                        "--screenshot", str(tmp_path), "--only", "--delay", "300"],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    from mrstudio.kmap_links import TAB_ORDER
    assert (tmp_path / f"tab{TAB_ORDER.index('MobileTab')}_mobile.png").is_file(), r.stdout


def test_main_rejects_stray_argument():
    from mrstudio.gui.app import main
    with pytest.raises(SystemExit):
        main(["not-a-link"])


def test_url_scheme_registered_in_installers():
    spec = (PKG / "mrstudio.spec").read_text(encoding="utf-8")
    assert '"CFBundleURLSchemes": ["mrstudio"]' in spec
    iss = (PKG / "mrstudio.iss").read_text(encoding="utf-8")
    assert "[Registry]" in iss and r"Software\Classes\mrstudio\shell\open\command" in iss
    assert '"URL Protocol"' in iss and '""%1"""' in iss
    sys.path.insert(0, str(PKG))
    try:
        import bundle_config as cfg
    finally:
        sys.path.remove(str(PKG))
    files = [src for src, _dst in cfg.package_data("mrstudio", ["data/**/*"])]
    assert any(f.endswith(os.path.join("kmap", "mrstudio_targets.json")) for f in files)
