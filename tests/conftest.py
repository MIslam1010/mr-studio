"""Test-session hooks and shared fixtures.

* ``qapp`` -- one session-wide QApplication, torn down cleanly (timers stopped, widgets closed, deferred deletes
  processed, garbage collected) so that Qt/OpenGL objects are not destroyed during interpreter shutdown.
* Exit status -- on some systems Qt/OpenGL objects destroyed during Python's final garbage collection segfault
  (exit code 139) after a fully passing run (not fixed yet). As a guarded fallback the process exits with
  ``os._exit(<pytest's real status>)`` right after the terminal summary. The status is *fail closed*: it starts at 1
  and is replaced by ``session.exitstatus`` only once pytest has set it, so a crash before that is never
  reported as success. Set the environment variable ``MRSTUDIO_PYTEST_HARD_EXIT=0`` to disable the hard exit
  (default: on) and let the interpreter shut down normally, e.g. to debug the teardown segfault.
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
    return os.environ.get("MRSTUDIO_PYTEST_HARD_EXIT", "1").strip().lower() not in ("0", "false", "no", "off", "")


def pytest_collection_modifyitems(config, items):
    """MRSTUDIO_NO_GUI_TESTS=1 skips every test that needs a QApplication (the ``qapp`` fixture).

    Used by the Linux CI job: Mesa's software OpenGL under Xvfb aborts when a 3-D view redraws. The GUI tests run
    on the Windows and macOS build jobs instead, which are the platforms students use.
    """
    if os.environ.get("MRSTUDIO_NO_GUI_TESTS", "").strip() not in ("1", "true", "yes"):
        return
    skip = pytest.mark.skip(reason="GUI test (MRSTUDIO_NO_GUI_TESTS=1)")
    for item in items:
        if "qapp" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication for all GUI tests (a module-level fixture of the same name overrides it locally)."""
    from PySide6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    if _hard_exit():
        return   # the process ends with os._exit; tearing Qt/GL widgets down here segfaults on some systems
    widgets = list(app.topLevelWidgets())
    for obj in [app] + widgets:
        for t in obj.findChildren(QtCore.QTimer):
            t.stop()
    for w in widgets:
        try:
            w.close()
            w.deleteLater()
        except RuntimeError:   # already deleted on the C++ side
            pass
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
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
