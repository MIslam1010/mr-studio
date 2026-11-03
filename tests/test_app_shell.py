"""App-shell tests: robot registry / engine cache, lazy tab updates, tab titles + shortcuts, crash log, theme."""
from __future__ import annotations

import os
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio import robots as R  # noqa: E402
from mrstudio.robots import BUILTIN_ROBOTS, ROBOTS, custom_robot, get_robot, register_robot, trans  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app, False, light=False)
    yield app
    theme.set_projector(app, False)


@pytest.fixture
def cleanup_robots():
    """Remove robots registered by a test from the global library afterwards."""
    before = set(ROBOTS)
    yield
    for nm in set(ROBOTS) - before:
        ROBOTS.pop(nm, None)
        R._cache.pop(nm, None)


def _rrr(name, L=0.4):
    return custom_robot(name, [[0, 0, 1]] * 3, [[0, 0, 0], [L, 0, 0], [2 * L, 0, 0]], trans([3 * L, 0, 0]))


# ----------------------------------------------------------------------------- robots / engines
def test_same_name_reregister_switches_robot_and_engine(qapp, cleanup_robots):
    from mrstudio.gui.base import AppContext
    ctx = AppContext()
    r1 = _rrr("Shell test robot", 0.4)
    assert register_robot(r1) == "Shell test robot"
    got = []
    ctx.robot_changed.connect(got.append)
    ctx.set_robot("Shell test robot")
    assert ctx.robot is r1 and got == [r1]
    e1 = ctx.engine()
    assert ctx.engine() is e1 and e1.robot is r1
    # re-create a different robot with the same name (Robot Builder "Create robot" again)
    r2 = _rrr("Shell test robot", 0.3)
    assert register_robot(r2) == "Shell test robot"
    assert get_robot("Shell test robot") is r2
    ctx.set_robot("Shell test robot")
    assert ctx.robot is r2 and got == [r1, r2]
    e2 = ctx.engine()
    assert e2 is not e1 and e2.robot is r2
    assert all(key[0] != id(r1) for key in ctx._engines), "engines of the replaced robot must be dropped"
    # selecting the same robot again is a no-op, force=True re-emits
    ctx.set_robot("Shell test robot")
    assert got == [r1, r2]
    ctx.set_robot("Shell test robot", force=True)
    assert got == [r1, r2, r2]


def test_builtin_names_cannot_be_overwritten(cleanup_robots):
    ur5 = get_robot("UR5 (6R industrial)")
    fake = _rrr("UR5 (6R industrial)")
    name = register_robot(fake)
    assert name == "UR5 (6R industrial) (2)" and fake.name == name
    assert get_robot("UR5 (6R industrial)") is ur5 and ROBOTS["UR5 (6R industrial)"] is R.ur5
    assert get_robot(name) is fake
    fake2 = _rrr("UR5 (6R industrial)")
    assert register_robot(fake2) == "UR5 (6R industrial) (3)"
    # re-registering the built-in object itself is harmless
    assert register_robot(ur5) == "UR5 (6R industrial)" and get_robot("UR5 (6R industrial)") is ur5
    assert "UR5 (6R industrial)" in BUILTIN_ROBOTS


@pytest.mark.parametrize("name", sorted(BUILTIN_ROBOTS))
def test_home_within_joint_limits(name):
    r = get_robot(name)
    lo, hi = r.joint_limits.T
    assert r.home.shape == (r.n,)
    assert np.all(r.home >= lo - 1e-12) and np.all(r.home <= hi + 1e-12), (r.home, r.joint_limits)


def test_get_robot_thread_safe_and_preload():
    import threading
    names = [n for n in ("Planar 2R arm", "Planar 3R arm", "UR5 (6R industrial)")]
    out = {}

    def work(k):
        out[k] = [get_robot(n) for n in names]

    th = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for k in range(4):
        assert all(a is b for a, b in zip(out[k], out[0]))
    assert R.preload(["Planar 2R arm", "no such robot"]) == []


# ----------------------------------------------------------------------------- lazy tab updates
def _counting_tab_class():
    from mrstudio.gui.base import BaseTab

    class CountingTab(BaseTab):
        title = "⓪ Test"

        def __init__(self, ctx):
            super().__init__(ctx)
            self.robot_calls = []
            self.backend_calls = []

        def on_robot_changed(self, robot):
            self.robot_calls.append(robot.name)

        def on_backend_changed(self, backend):
            self.backend_calls.append(backend)

    return CountingTab


def test_hidden_tab_does_not_rebuild_until_shown(qapp):
    from PySide6 import QtWidgets
    from mrstudio.gui.base import AppContext
    CountingTab = _counting_tab_class()
    ctx = AppContext("Planar 2R arm")
    tw = QtWidgets.QTabWidget()
    a, b = CountingTab(ctx), CountingTab(ctx)
    tw.addTab(a, "a")
    tw.addTab(b, "b")
    tw.resize(400, 300)
    tw.show()
    qapp.processEvents()
    assert a.isVisible() and not b.isVisible()
    ctx.set_robot("Planar 3R arm")
    ctx.set_backend("mr")
    assert a.robot_calls == ["Planar 3R arm"] and a.backend_calls == ["mr"]
    assert b.robot_calls == [] and b.backend_calls == [] and b.is_stale()
    ctx.set_robot("UR5 (6R industrial)")
    assert b.robot_calls == []
    tw.setCurrentIndex(1)               # showing the tab applies the latest state once
    qapp.processEvents()
    assert b.robot_calls == ["UR5 (6R industrial)"] and b.backend_calls == ["mr"] and not b.is_stale()
    b.refresh_if_stale()                # idempotent
    assert b.robot_calls == ["UR5 (6R industrial)"]
    tw.close()
    # a tab outside any window (scripts / tests) is updated immediately
    c = CountingTab(ctx)
    ctx.set_robot("Planar 2R arm")
    assert c.robot_calls == ["Planar 2R arm"]
    c.shutdown()                        # default hook exists and is a no-op
    assert not hasattr(c, "needs_dynamics")


# ----------------------------------------------------------------------------- tabs / main window
def test_tab_title_digit_matches_index():
    import importlib
    from mrstudio.gui.app import TAB_LONG_NAMES, TABS
    for i, (mod, cls) in enumerate(TABS):
        k = getattr(importlib.import_module(mod), cls)
        assert unicodedata.numeric(k.title[0]) == i + 1, (cls, k.title)
        assert cls in TAB_LONG_NAMES


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


def test_main_window_tabs_tooltips_shortcuts(win, qapp):
    from PySide6 import QtCore
    from mrstudio.gui.app import TAB_LONG_NAMES, TABS, ErrorTab
    assert win.tabs.count() == len(TABS)
    assert win.tabs.elideMode() == QtCore.Qt.ElideNone and win.tabs.usesScrollButtons()
    for i, (_mod, cls) in enumerate(TABS):
        assert win.tabs.tabToolTip(i).startswith(TAB_LONG_NAMES[cls])
        if not isinstance(win.pages[i], ErrorTab):   # a broken tab is labelled "⚠ module" instead
            assert unicodedata.numeric(win.tabs.tabText(i)[0]) == i + 1
    assert [a.shortcut().toString() for a in win.tab_acts] == [f"Ctrl+{i}" for i in range(1, 10)]
    win.tab_acts[2].trigger()
    assert win.tabs.currentIndex() == 2
    win.act_next_tab.trigger()
    assert win.tabs.currentIndex() == 3
    win.tabs.setCurrentIndex(0)
    win.act_prev_tab.trigger()
    assert win.tabs.currentIndex() == win.tabs.count() - 1
    assert "~3000" not in win.engine_box.toolTip()


def test_robot_box_disabled_on_own_mechanism_tabs(win, qapp):
    from mrstudio.gui.app import OWN_MECHANISM_TIP
    for i, page in enumerate(win.pages):
        win.tabs.setCurrentIndex(i)
        qapp.processEvents()
        uses = getattr(page, "uses_serial_robot", True)
        assert win.robot_box.isEnabled() == uses, type(page).__name__
        if not uses:
            assert win.robot_box.toolTip() == OWN_MECHANISM_TIP
    win.tabs.setCurrentIndex(0)
    _wait(qapp)


def test_robot_created_same_name_replaces_and_switches(win, qapp, cleanup_robots):
    win.tabs.setCurrentIndex(0)
    r1 = _rrr("My robot", 0.4)
    win._robot_created(r1)
    assert win.ctx.robot is r1 and win.robot_box.currentText() == "My robot"
    r2 = _rrr("My robot", 0.25)
    win._robot_created(r2)
    assert win.ctx.robot is r2 and win.robot_box.findText("My robot") >= 0
    assert [win.robot_box.itemText(i) for i in range(win.robot_box.count())].count("My robot") == 1
    # a built-in name is never overwritten: the robot is renamed and the user is told
    r3 = _rrr("Planar 2R arm")
    win._robot_created(r3)
    assert win.ctx.robot is r3 and r3.name == "Planar 2R arm (2)"
    assert "built-in" in win.statusBar().currentMessage()
    assert get_robot("Planar 2R arm") is not r3
    win.ctx.set_robot("Planar 2R arm")
    for nm in ("My robot", "Planar 2R arm (2)"):
        win.robot_box.removeItem(win.robot_box.findText(nm))
    _wait(qapp)


def test_error_tab_copy_details(qapp):
    from PySide6 import QtWidgets
    from mrstudio.gui.app import ErrorTab, load_tab
    from mrstudio.gui.base import AppContext
    page, label = load_tab("mrstudio.gui.theme", "NoSuchClass", AppContext())
    assert isinstance(page, ErrorTab) and label.startswith("⚠")
    heads = [w.text() for w in page.findChildren(QtWidgets.QLabel)]
    assert "This tab could not start" in heads
    page.copy_btn.click()
    assert "AttributeError" in QtWidgets.QApplication.clipboard().text()
    page.shutdown()


# ----------------------------------------------------------------------------- crash log
def test_log_path_is_writable(tmp_path, monkeypatch):
    from mrstudio.gui.app import log_path, rotate_log
    monkeypatch.delenv("MRSTUDIO_LOG", raising=False)
    p = log_path()
    assert isinstance(p, Path) and p.name == "mrstudio.log"
    assert p.parent.is_dir() and os.access(p.parent, os.W_OK)
    if sys.platform == "darwin":
        assert p.parent == Path.home() / "Library" / "Logs" / "MR Studio"
    # override + rotation at ~1 MB keeping one old copy
    monkeypatch.setenv("MRSTUDIO_LOG", str(tmp_path / "sub" / "x.log"))
    q = log_path()
    assert q == tmp_path / "sub" / "x.log" and q.parent.is_dir()
    q.write_bytes(b"x" * 2000)
    rotate_log(q, max_bytes=1000)
    assert not q.exists() and (tmp_path / "sub" / "x.log.1").stat().st_size == 2000


# ----------------------------------------------------------------------------- theme / theory / recorder
def test_series_colour_distinct_from_bad():
    from mrstudio.gui import theme
    assert theme.SERIES[3].lower() == "#d291ff"


def test_light_projector_palette_contrast(qapp):
    from mrstudio.gui import theme
    for c in (theme.LIGHT_TEXT, theme.LIGHT_MUTED, theme.LIGHT_LINK, theme.LIGHT_ACCENT, theme.LIGHT_WARN,
              theme.LIGHT_GOOD, theme.LIGHT_BAD):
        assert theme.contrast_ratio(c, theme.LIGHT_BG) >= 4.5, c
    try:
        theme.set_projector(qapp, True)
        theme.set_projector_light(qapp, True)
        assert theme.is_light() and theme.muted_color() == theme.LIGHT_MUTED
        assert theme.link_color() == theme.LIGHT_LINK and theme.text_color() == theme.LIGHT_TEXT
        assert theme.LIGHT_BASE in qapp.styleSheet()
    finally:
        theme.set_projector_light(qapp, False)
        theme.set_projector(qapp, False)
    assert not theme.is_light() and qapp.styleSheet() == theme.STYLESHEET


def test_theory_panel_renders_lazily_and_fits(qapp):
    from mrstudio.gui.theory import TheoryPanel
    p = TheoryPanel()
    long_eq = r"$$" + " + ".join(rf"\theta_{{{i}}}^2" for i in range(40)) + r"$$"
    p.set_content("<h2>T</h2>" + long_eq)
    assert p._rendered_key is None and p.needs_render()       # nothing rendered while hidden
    p.resize(320, 400)
    p.show()
    qapp.processEvents()
    assert p._rendered_key is not None and not p.needs_render()
    assert p._shrunk and p._widest > p._max_eq_width()       # too-wide equation scaled to the viewport
    p.close()


def test_recorder_follows_grab_callable_and_bounded_queue(qapp):
    pytest.importorskip("imageio_ffmpeg")
    from PySide6 import QtGui
    from mrstudio.gui import recorder as Rec

    sizes = [(200, 100)]

    def grab():
        img = QtGui.QImage(*sizes[0], QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor("#4cc9f0"))
        return img

    rec = Rec.Recorder()
    done = []
    rec.finished.connect(done.append)
    rec.start(grab, fps=25)
    assert rec._queue.maxsize == Rec.QUEUE_FRAMES
    deadline = time.monotonic() + 10
    while rec.frames < 6 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    sizes[0] = (90, 120)              # source changes (tab switch): frames are letter-boxed to the fixed size
    while rec.frames < 12 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    t0 = time.perf_counter()
    assert rec.stop(wait=False) is None
    assert time.perf_counter() - t0 < 0.2       # non-blocking
    while not done and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert done and done[0] and os.path.getsize(done[0]) > 0
    assert not rec.busy()
    os.remove(done[0])


def test_main_window_recording_follows_tab(win, qapp):
    pytest.importorskip("imageio_ffmpeg")
    grab = win.grab_function("view")
    assert callable(grab) and not grab().isNull()   # re-resolves the view (or the window) on every call
    win.start_recording("view", 25)
    assert win.recorder.recording
    deadline = time.monotonic() + 10
    while win.recorder.frames < 5 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    win.tabs.setCurrentIndex((win.tabs.currentIndex() + 1) % win.tabs.count())   # switch tab while recording
    while win.recorder.frames < 10 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    tmp = win.stop_recording(ask=False)
    assert tmp and os.path.getsize(tmp) > 0 and not win.act_record.isChecked()
    os.remove(tmp)


def test_shortcuts_dialog_lists_every_action_shortcut(win, qapp):
    """Help → Keyboard shortcuts is generated: every action with a shortcut is listed, with the same keys."""
    from PySide6 import QtWidgets
    from mrstudio.gui.app import ShortcutsDialog, shortcut_text
    rows = win.shortcut_rows()
    keys = {k for _g, _l, k in rows}
    for a in win.actions():
        if not a.shortcut().isEmpty():
            assert shortcut_text(a) in keys, a.text()
    labels = [lab for _g, lab, _k in rows]
    assert len(labels) == len(set(labels)), "an action is listed twice"
    assert all(lab and lab[0].isalnum() for lab in labels), labels   # no "＋ " / "● " / "…" noise
    groups = {g for g, _l, _k in rows}
    assert {"File", "View", "Help", "Tabs"} <= groups
    assert any("Hardware" in g for g in groups)            # Esc = Stop button inside the Hardware tab
    assert any(lab.startswith("Go to tab 1: Forward kinematics") for lab in labels)
    dlg = win._shortcuts()
    assert isinstance(dlg, ShortcutsDialog)
    n = sum(dlg.tree.topLevelItem(i).childCount() for i in range(dlg.tree.topLevelItemCount()))
    assert n == len(rows)
    dlg.close()
    assert win.act_keys in [a for m in win.menuBar().findChildren(QtWidgets.QMenu) for a in m.actions()]


def test_tooltips_show_platform_shortcut(win):
    """Tooltips get the shortcut appended by _act (⌘O on macOS, Ctrl+O elsewhere), not hand-written ⌘ glyphs."""
    from mrstudio.gui.app import shortcut_text
    for a in (win.act_urdf, win.act_shot, win.act_record, win.act_theory, win.act_projector, win.act_keys):
        k = shortcut_text(a)
        assert k and a.toolTip().endswith(f"({k})"), a.toolTip()
        assert a.toolTip().count("(") == 1 or "⌘" not in a.statusTip()
    assert " or " in shortcut_text(win.act_projector)                     # two shortcuts: Ctrl+Shift+P, F5
    html = win.about_html()
    assert shortcut_text(win.act_record) in html and "Keyboard shortcuts" in html


def test_error_box_has_details_and_does_not_block(qapp):
    from PySide6 import QtWidgets
    from mrstudio.gui.widgets import error_box, exception_details
    try:
        raise ValueError("bad URDF")
    except ValueError:
        details = exception_details()
    box = error_box(None, "Import URDF", "The robot could not be imported.", details=details, hint="Try again")
    assert isinstance(box, QtWidgets.QMessageBox) and box.isVisible()      # open(): returns at once
    assert "ValueError: bad URDF" in box.detailedText() and "Traceback" in box.detailedText()
    assert "Traceback" not in box.text() and box.informativeText() == "Try again"
    box.close()


def test_urdf_import_failure_shows_friendly_box(win, qapp, monkeypatch):
    import mrstudio.gui.urdf_dialog as ud

    def boom(parent=None):
        raise KeyError("link_7")
    monkeypatch.setattr(ud, "open_urdf_dialog", boom)
    win._import_urdf()
    box = win._error_box
    assert box.isVisible() and "could not be imported" in box.text()
    assert "KeyError" in box.detailedText() and "Traceback" not in box.text()
    box.close()
    qapp.processEvents()


def test_controls_column_fits_its_content(win, qapp):
    """The left control column widens (within limits) so its widgets are not cut off behind a scroll bar."""
    for i, page in enumerate(win.pages):
        sc = getattr(page, "_controls_scroll", None)
        if sc is None:
            continue
        win.tabs.setCurrentIndex(i)
        qapp.processEvents()
        need = sc.widget().minimumSizeHint().width()
        cap = int((page._controls_width + page.CONTROLS_GROW) * __import__("mrstudio.gui.theme").gui.theme.font_scale())
        if need + sc.verticalScrollBar().sizeHint().width() <= cap:
            assert sc.minimumWidth() >= need, type(page).__name__
    win.tabs.setCurrentIndex(0)
    _wait(qapp)


def test_screenshot_mode_ignores_saved_settings(tmp_path, monkeypatch):
    """--screenshot uses a throw-away settings file: the saved tab / robot / lesson state never leaks in."""
    import mrstudio.gui.app as A
    monkeypatch.delenv("MRSTUDIO_SETTINGS", raising=False)
    monkeypatch.setitem(A._SETTINGS, "ephemeral", None)
    path = A.use_ephemeral_settings()
    assert A.use_ephemeral_settings() == path and Path(A.app_settings().fileName()) == Path(path)  # Qt: '/' on Windows
    assert A.app_settings().value("robot") is None
    monkeypatch.setenv("MRSTUDIO_SETTINGS", str(tmp_path / "s.ini"))   # an explicit file still wins
    assert Path(A.app_settings().fileName()) == tmp_path / "s.ini"


def test_settings_restore_tab_robot_engine(qapp, tmp_path, monkeypatch):
    """Last tab, engine and theory-panel state come back; an unknown saved robot falls back to the default."""
    monkeypatch.setenv("MRSTUDIO_SETTINGS", str(tmp_path / "s.ini"))
    from mrstudio.gui.app import MainWindow, app_settings
    from mrstudio.gui.base import AppContext
    s = app_settings()
    s.setValue("tab", 3)
    s.setValue("engine", "mr")
    s.setValue("theory", False)
    s.sync()
    w = MainWindow(AppContext("Planar 2R arm"))
    w.persist = False
    try:
        w.restore_settings()
        assert w.tabs.currentIndex() == 3 and w.ctx.backend == "mr" and not w.act_theory.isChecked()
    finally:
        w.act_theory.setChecked(True)
        w.shutdown_pages()
        w.deleteLater()
        qapp.processEvents()
