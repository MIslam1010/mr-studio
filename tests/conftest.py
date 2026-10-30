"""Test-session hooks and shared fixtures.

* ``qapp`` -- one session-wide QApplication with an ordered teardown: timers stopped, then every window's / tab's
  ``shutdown`` hook (JobRunner worker threads, simulation timers, hardware ports), then pending events. Top-level
  widgets are *not* deleted explicitly: deleting a tab that holds pyqtgraph plots (e.g. ``PracticeTab``) from
  here segfaulted (exit code 139, in ``QGraphicsGridLayout``'s destructor); the interpreter's normal shutdown
  disposes of them cleanly.
* Exit status -- pytest's own status. The old workaround, ``os._exit(<pytest's status>)`` right after the
  terminal summary, is still available with ``MRSTUDIO_PYTEST_HARD_EXIT=1`` (default: off). It is *fail closed*:
  the status starts at 1 and is replaced by ``session.exitstatus`` only once pytest has set it, so a crash before
  that is never reported as success.
* PySide6 check -- a PySide6 build that drops references to None (PySide6 6.12.0's Linux wheel) makes Python 3.11
  abort part-way through the GUI tests; the run stops at once with an explanation instead (see
  ``_pyside_drops_none_refs``).
"""
import gc
import os
import sys

import pytest

_STATUS = {"code": 1}   # fail closed: 1 until pytest reports otherwise
_SESSION = {}


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    _SESSION["s"] = session


def pytest_sessionfinish(session, exitstatus):
    _STATUS["code"] = int(exitstatus)


def _hard_exit() -> bool:
    return os.environ.get("MRSTUDIO_PYTEST_HARD_EXIT", "0").strip().lower() not in ("0", "false", "no", "off", "")


def _pyside_drops_none_refs() -> bool:
    """Does this PySide6 lose a reference to None on each call of a void method?

    PySide6 6.12.0's Linux wheel (cp310-abi3-manylinux_2_34_x86_64) does: every ``QObject.setObjectName``,
    ``QWidget.update`` ... decrements None's reference count. On Python 3.11 None is an ordinary object (immortal
    only from 3.12, PEP 683), so after some 30,000 calls Python aborts with "Fatal Python error: none_dealloc:
    deallocating None" -- with pytest's output capture the message is lost and only "Aborted (core dumped)" is seen.
    The macOS wheel of 6.12.0 and PySide6 6.9-6.11 on Linux are fine.
    """
    if sys.version_info >= (3, 12):
        return False
    try:
        from PySide6 import QtCore
    except ImportError:
        return False
    obj = QtCore.QObject()
    obj.setObjectName("probe")
    before = sys.getrefcount(None)
    for _ in range(200):
        obj.setObjectName("probe")
    return sys.getrefcount(None) < before - 100


def pytest_collection_modifyitems(config, items):
    """MRSTUDIO_NO_GUI_TESTS=1 skips every test that needs a QApplication (the ``qapp`` fixture), e.g. for a quick
    numerical run on a machine without a display. CI runs the GUI tests on Linux (Xvfb), Windows and macOS.

    If GUI tests are collected and the installed PySide6 drops references to None, the session stops here with a
    non-zero status (it would abort later anyway, without a usable message).
    """
    gui = [item for item in items if "qapp" in getattr(item, "fixturenames", ())]
    if os.environ.get("MRSTUDIO_NO_GUI_TESTS", "").strip() in ("1", "true", "yes"):
        skip = pytest.mark.skip(reason="GUI test (MRSTUDIO_NO_GUI_TESTS=1)")
        for item in gui:
            item.add_marker(skip)
        return
    if gui and _pyside_drops_none_refs():
        import PySide6
        pytest.exit(f"PySide6 {PySide6.__version__} on this platform drops a reference to None on every void Qt call; "
                    f"Python {sys.version_info[0]}.{sys.version_info[1]} would abort part-way through the GUI tests "
                    "('Fatal Python error: none_dealloc'). Install another PySide6 (CI on Linux uses 6.11.3), use "
                    "Python >= 3.12, or set MRSTUDIO_NO_GUI_TESTS=1 to run only the non-GUI tests.", returncode=4)


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication for all GUI tests (a module-level fixture of the same name overrides it locally)."""
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    if _hard_exit():
        return   # the process ends with os._exit right after the terminal summary
    from shiboken6 import isValid
    widgets = list(app.topLevelWidgets())
    for obj in [app] + widgets:
        for t in obj.findChildren(QtCore.QTimer):
            t.stop()
    for w in widgets:   # the app's quit hooks: worker threads joined, timers stopped, hardware released
        fn = (getattr(w, "shutdown_pages", None) or getattr(w, "shutdown", None)) if isValid(w) else None
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    # No close()/deleteLater() of the top-level widgets here (see the module docstring).
    gc.collect()
    app.processEvents()


def pytest_unconfigure(config):
    """Runs after the terminal summary has been printed."""
    session = _SESSION.get("s")
    if session is not None:
        _STATUS["code"] = int(session.exitstatus)
    if not _hard_exit():
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_STATUS["code"])
