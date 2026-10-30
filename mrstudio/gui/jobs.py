"""Background jobs for the GUI: one ``JobRunner`` per tab (or per independent computation).

Usage (GUI thread)::

    self.jobs = JobRunner(self)
    self.jobs.run(fn, on_done=self._done, on_error=self._failed, on_progress=bar_update, label="Simulating")

``fn(progress, cancelled)`` runs in a worker ``QThread``:

* ``progress(f, *info)`` -- report a fraction 0..1 (thread-safe, may be called often); optional extra positional
  ``info`` (e.g. a message, or a streamed partial result) is forwarded unchanged: ``on_progress(f, *info)``;
* ``cancelled()``  -- returns True once the job is cancelled or superseded; long loops should check it
  and return early (the return value of a cancelled job is discarded anyway).

Guarantees
----------
* ``on_done(result)`` / ``on_error(traceback_text)`` / ``on_progress(f)`` run in the GUI thread, and only for the
  *current* job: every job carries a generation number and results/progress of older generations are dropped
  inside the runner, so callers never see stale results.
* Starting a new job, ``cancel()`` and ``shutdown()`` never block the GUI thread for long: a superseded worker is
  told to stop (its cancel token is set) and *retired* -- it is kept alive (module-level registry) until its
  thread finishes, then deleted.  Only ``shutdown(timeout_ms)`` (used once at application quit) waits, briefly.
* An exception in ``on_done`` is reported through ``on_error`` like an exception in ``fn``.

Signals: ``busy_changed(bool)`` (a current job started / ended or was cancelled), ``progressed(float)``
(current job only), ``started(str)`` (label of a new job).

``AsyncMixin`` (below) wraps a ``JobRunner`` with a progress bar + Cancel button row for tabs
(``_init_async()``, ``run_async(label, fn, on_done)``, ``cancel_jobs()``, ``wait_job()``, ``busy_row``).
"""
from __future__ import annotations

import threading
import traceback

from PySide6 import QtCore

__all__ = ["JobRunner", "CancelToken", "AsyncMixin", "live_workers"]

class _Reaper(QtCore.QObject):
    """Lives in the GUI thread.  Parent (C++ owner) of every worker thread and keeper of its Python wrapper:

    * a retired worker stays alive until its thread has finished, even if the runner/tab that started it is gone;
    * the job function (a closure that often references a tab) and the result are released *here, in the GUI
      thread* -- if the last reference to a widget were dropped in the worker thread, PySide would delete the
      QWidget there, corrupting Qt's widget list (intermittent crashes at quit / in tests)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("mrstudio-job-reaper")
        self.live: set = set()          # Python wrappers of the workers (GUI thread only)

    def adopt(self, w):
        self.live.add(w)
        w.finished.connect(self.reap)   # queued: emitted by the worker thread, runs in the GUI thread

    @QtCore.Slot()
    def reap(self):
        w = self.sender()
        if w is None or w not in self.live:
            return
        w.fn = w.keep = None            # closure / result released in the GUI thread
        self.live.discard(w)
        w.deleteLater()                 # Qt-documented pattern: finished -> deleteLater


_REAPER: _Reaper | None = None


def _reaper() -> _Reaper:
    global _REAPER
    if _REAPER is None:
        _REAPER = _Reaper()
    return _REAPER


class CancelToken:
    """Thread-safe cancel flag handed to a job (``token()`` == ``token.cancelled``)."""
    __slots__ = ("_ev",)

    def __init__(self):
        self._ev = threading.Event()

    def cancel(self):
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def __call__(self) -> bool:
        return self._ev.is_set()


class _Worker(QtCore.QThread):
    progressed = QtCore.Signal(int, float, object)
    done = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, fn, gen: int, token: CancelToken):
        super().__init__(_reaper())     # not the runner: a retired worker may outlive its runner
        self.fn, self.gen, self.token = fn, gen, token
        self.keep = None

    def run(self):
        gen = self.gen
        try:
            res = self.fn(lambda f, *info: self.progressed.emit(gen, float(f), info), self.token)
        except Exception:
            if not self.token.cancelled:
                self.failed.emit(gen, traceback.format_exc())
            return
        self.keep = res                 # dropped by the reaper in the GUI thread, never here
        if not self.token.cancelled:
            self.done.emit(gen, res)
        del res


def live_workers() -> int:
    """Number of worker threads still running (current + retired) -- for tests/diagnostics."""
    if _REAPER is None:
        return 0
    return sum(1 for w in list(_REAPER.live) if w.isRunning())


class JobRunner(QtCore.QObject):
    """Runs one background job at a time (a new ``run`` supersedes the previous one).  See module docstring."""
    busy_changed = QtCore.Signal(bool)
    progressed = QtCore.Signal(float)
    started = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gen = 0
        self._busy = False
        self._worker: _Worker | None = None
        self._token: CancelToken | None = None
        self._cb: tuple = (None, None, None)
        self.label = ""

    # ------------------------------------------------------------------ API
    @property
    def generation(self) -> int:
        return self._gen

    def is_busy(self) -> bool:
        """True while the current job has neither delivered its result nor been cancelled."""
        return self._busy

    def run(self, fn, *, on_done, on_error=None, on_progress=None, label: str = "") -> int:
        """Start ``fn(progress, cancelled)`` in a worker thread, superseding any running job.  Returns the gen."""
        self._supersede()
        self._gen += 1
        gen = self._gen
        token = CancelToken()
        w = _Worker(fn, gen, token)
        w.progressed.connect(self._on_progress)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        _reaper().adopt(w)
        self._worker, self._token = w, token
        self._cb = (on_done, on_error, on_progress)
        self.label = label
        self._set_busy(True)
        self.started.emit(label)
        w.start()
        return gen

    def cancel(self):
        """Cancel the current job (non-blocking); its result, if any, is dropped."""
        self._supersede()
        self._gen += 1
        self._set_busy(False)

    def wait(self, timeout_s: float = 120.0):
        """Process events until the current job has finished (scripted tests only -- never in normal GUI code)."""
        import time
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        t0 = time.time()
        while self._busy:
            app.processEvents()
            time.sleep(0.005)
            if time.time() - t0 > timeout_s:
                raise TimeoutError("background job did not finish")
        app.processEvents()

    def shutdown(self, timeout_ms: int = 2000):
        """Cancel and wait (at most ``timeout_ms`` in total) for this runner's threads -- application quit only."""
        w = self._worker
        self.cancel()
        if w is not None and w.isRunning():
            w.wait(max(0, int(timeout_ms)))

    # ------------------------------------------------------------------ internals
    def _supersede(self):
        if self._token is not None:
            self._token.cancel()
        self._worker, self._token = None, None
        self._cb = (None, None, None)

    def _set_busy(self, on: bool):
        if on != self._busy:
            self._busy = on
            self.busy_changed.emit(on)

    @QtCore.Slot(int, float, object)
    def _on_progress(self, gen, f, info):
        if gen != self._gen:
            return
        cb = self._cb[2]
        if cb is not None:
            cb(f, *info)
        self.progressed.emit(f)

    @QtCore.Slot(int, object)
    def _on_done(self, gen, res):
        if gen != self._gen:
            return
        on_done, on_error, _ = self._cb
        self._finish()
        try:
            if on_done is not None:
                on_done(res)
        except Exception:
            self._report(on_error, traceback.format_exc())

    @QtCore.Slot(int, str)
    def _on_failed(self, gen, err):
        if gen != self._gen:
            return
        on_error = self._cb[1]
        self._finish()
        self._report(on_error, err)

    def _finish(self):
        self._worker, self._token = None, None
        self._cb = (None, None, None)
        self._set_busy(False)

    @staticmethod
    def _report(on_error, err):
        if on_error is not None:
            on_error(err)
        else:
            print(err)


class AsyncMixin:
    """One background job at a time per tab, with a progress bar and a Cancel button (``self.busy_row``).

    Call ``_init_async()`` in ``__init__``; ``run_async(label, fn, on_done)`` runs ``fn(progress, cancelled)``
    through ``self.jobs`` (a ``JobRunner``) and calls ``on_done(result)`` in the GUI thread -- only for the latest
    job.  ``cancel_jobs()`` drops the current job without blocking; ``shutdown()`` (quit) waits briefly.
    Errors are shown through ``self.status(text, color)``.
    """

    def _init_async(self):
        from PySide6 import QtWidgets
        from .widgets import button
        self.jobs = JobRunner(self)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(16)
        self.cancel_btn = button("Cancel", self._cancel_job)
        self.busy_row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(self.busy_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.progress, 1)
        lay.addWidget(self.cancel_btn)
        self.busy_row.setVisible(False)
        self.jobs.busy_changed.connect(self._busy_changed)

    @property
    def _job_gen(self) -> int:          # backwards compatibility (was a plain counter)
        return self.jobs.generation

    def run_async(self, label: str, fn, on_done):
        """fn(progress, cancelled) runs in a worker thread; on_done(result) runs in the GUI thread."""
        self.progress.setValue(0)
        self.progress.setFormat(label + "  %p%")
        self.jobs.run(fn, on_done=on_done, on_error=self._job_failed,
                      on_progress=lambda f, *_: self.progress.setValue(int(1000 * min(max(f, 0.0), 1.0))),
                      label=label)

    def wait_job(self, timeout_s: float = 120.0):
        """Block (processing events) until the current job finished -- used by scripted tests."""
        self.jobs.wait(timeout_s)

    def _busy_changed(self, on):
        self.busy_row.setVisible(on)

    def _job_failed(self, err):
        from . import theme
        print(err)
        self.status(f"⚠ computation failed: {err.strip().splitlines()[-1]}", theme.bad_color())

    def _cancel_job(self):
        """Cancel button: drop the running job (it stops at its next cancelled() check)."""
        if self.jobs.is_busy():
            label = self.jobs.label
            self.jobs.cancel()
            self.status(f"{label or 'computation'} cancelled")

    def cancel_jobs(self):
        self.jobs.cancel()

    def shutdown(self):
        """Application quit: cancel, wait briefly for the worker (BaseTab.shutdown hook)."""
        self.jobs.shutdown(1500)
        parent_shutdown = getattr(super(), "shutdown", None)
        if parent_shutdown is not None:
            parent_shutdown()

    def status(self, text, color=None):  # overridden
        pass
