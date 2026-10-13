"""Entry point of the frozen MR Studio app (PyInstaller) -- also works from source.

    MRStudio                      start the app (same options as run_app.py: --robot, --tab, --screenshot ...)
    MRStudio --self-test [--report FILE]
                                  headless smoke test of the *bundle*: every library imports, Pinocchio and
                                  Modern Robotics agree on the UR5 dynamics, the Corke DH models and the
                                  bundled URDFs load, equations render, ffmpeg is present, every tab imports.
                                  Exit code 0 = OK. Used by the build scripts and CI after freezing.
    MRStudio --validate [--report FILE]
                                  run the full cross-validation suite (the Validation tab) headless.

A windowed app on Windows has no console (sys.stdout is None); output then goes to the --report file
and to %TEMP%/mrstudio.log.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

FROZEN = bool(getattr(sys, "frozen", False))
HERE = os.path.dirname(os.path.abspath(__file__))
if not FROZEN:  # running from source: make `mrstudio` and `bundle_config` importable
    sys.path.insert(0, os.path.dirname(HERE))
    sys.path.insert(0, HERE)


def _ensure_streams() -> None:
    """GUI builds have no stdout/stderr; some libraries call sys.stderr.write directly."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    import tempfile
    try:
        log = open(os.path.join(tempfile.gettempdir(), "mrstudio.log"), "a", encoding="utf-8", buffering=1)
    except OSError:
        log = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


class _Report:
    def __init__(self, path: str | None):
        self.path, self.lines, self.failed = path, [], 0

    def __call__(self, ok: bool, name: str, detail: str = "") -> None:
        line = f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else "")
        self.failed += 0 if ok else 1
        self.lines.append(line)
        print(line, flush=True)

    def note(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def close(self) -> int:
        self.note(f"{'OK' if not self.failed else 'FAILED'}: {self.failed} failure(s)")
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
        return 1 if self.failed else 0


def _step(rep: _Report, name: str, fn) -> object:
    try:
        out = fn()
        rep(True, name, "" if out is None else str(out))
        return out
    except Exception as e:  # noqa: BLE001 -- the point is to report anything
        rep(False, name, f"{type(e).__name__}: {e}")
        rep.note(traceback.format_exc())
        return None


def self_test(report: str | None = None, tabs: bool = True) -> int:
    """Headless check that the bundle contains everything MR Studio needs. Returns an exit code."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import importlib
    import platform

    rep = _Report(report)
    rep.note(f"MR Studio self-test -- Python {platform.python_version()} on {platform.platform()}"
             f" ({'frozen' if FROZEN else 'source'})")
    try:
        import bundle_config as cfg
        mods = cfg.SELF_TEST_IMPORTS
        dists = cfg.METADATA_DISTS
        optional = set(cfg.METADATA_OPTIONAL)
    except ImportError:  # pragma: no cover
        mods, dists, optional = ["modern_robotics", "pinocchio", "roboticstoolbox", "PySide6.QtWidgets"], [], set()

    for m in mods:
        _step(rep, f"import {m}", lambda m=m: getattr(importlib.import_module(m), "__version__", None))

    def _metadata():
        from importlib import metadata
        missing, absent_ok = [], []
        for d in dists:
            try:
                metadata.version(d)
            except metadata.PackageNotFoundError:
                (absent_ok if d in optional else missing).append(d)
        if missing:
            raise RuntimeError(f"no metadata for {missing} (About box would show '-')")
        return f"{len(dists) - len(absent_ok)} distributions" + (
            f"; not pip-installed (conda-forge): {absent_ok}" if absent_ok else "")
    _step(rep, "package metadata (About box, report)", _metadata)

    def _dynamics():
        import numpy as np
        from mrstudio.dynamics import DynamicsEngine
        from mrstudio.robots import get_robot
        r = get_robot("UR5 (6R industrial)")
        rng = np.random.default_rng(0)
        th, dth, ddth = (rng.uniform(-1, 1, r.n) for _ in range(3))
        a = DynamicsEngine(r, backend="pinocchio").inverse(th, dth, ddth)
        b = DynamicsEngine(r, backend="mr").inverse(th, dth, ddth)
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        if not err < 1e-9:
            raise AssertionError(f"|tau_pin - tau_mr| = {err:.2e}")
        return f"UR5 RNEA Pinocchio vs MR: {err:.1e}"
    _step(rep, "Pinocchio == Modern Robotics (UR5 inverse dynamics)", _dynamics)

    def _dh_models():
        from mrstudio.robots import get_robot
        names = ["PUMA 560 (from Corke DH)", "Franka Panda (7R redundant)"]
        return ", ".join(f"{get_robot(n).n} joints" for n in names)
    _step(rep, "Corke DH models (rtb-data meshes bundled)", _dh_models)

    def _urdfs():
        from mrstudio import urdf_import as U
        ex = U.example_urdfs()
        if not any("bundled" in k for k in ex):
            raise FileNotFoundError(f"bundled iiwa7.urdf missing (DATA_DIR={U.DATA_DIR})")
        r = U.from_urdf(next(p for k, p in ex.items() if "bundled" in k))
        return f"{len(ex)} example URDFs; iiwa7 -> {r.n} joints"
    _step(rep, "URDF import (bundled + rtb-data examples)", _urdfs)

    def _mathtext():
        from mrstudio.gui.theory import validate_all
        bad = validate_all([r"$$T(\theta) = e^{[\mathcal{S}_1]\theta_1} \cdots e^{[\mathcal{S}_n]\theta_n} M$$"])
        if bad:
            raise RuntimeError(bad)
    _step(rep, "theory-panel equations render (matplotlib mathtext)", _mathtext)

    def _ffmpeg():
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not os.path.isfile(exe):
            raise FileNotFoundError(exe)
        return os.path.basename(exe)
    _step(rep, "ffmpeg binary for MP4 recording", _ffmpeg)

    if tabs:
        def _tabs():
            from PySide6 import QtWidgets
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
            from mrstudio.gui.app import TABS
            ok, skipped = [], []
            for mod, cls in TABS:
                try:
                    m = importlib.import_module(mod)
                except ModuleNotFoundError as e:
                    if e.name == mod:   # the app skips tabs that are not written yet, so do we
                        skipped.append(mod.rsplit(".", 1)[-1])
                        continue
                    raise
                getattr(m, cls)
                ok.append(cls)
            return f"{len(ok)} tabs import" + (f"; not present: {skipped}" if skipped else "")
        _step(rep, "every tab module imports", _tabs)
    return rep.close()


def validate(report: str | None = None) -> int:
    """Full cross-validation suite, headless."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from mrstudio import validation as V
    rep = _Report(report)
    t0 = time.perf_counter()
    res = V.run_all()
    for c in res:
        if V.is_skipped(c):
            rep.note(f"SKIP  {c.name}")
        else:
            rep(bool(c.passed), c.name, f"error {V.fmt_err(c.error)} / tol {V.fmt_err(c.tol)}")
    rep.note(f"{V.summary(res)}  ({time.perf_counter() - t0:.1f} s)")
    return rep.close()


def _pop_option(argv: list[str], name: str, has_value: bool = False):
    if name not in argv:
        return None
    i = argv.index(name)
    val = argv[i + 1] if has_value and i + 1 < len(argv) else True
    del argv[i:i + (2 if has_value else 1)]
    return val


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [a for a in argv if not a.startswith("-psn_")]   # old macOS Finder launch argument
    _ensure_streams()
    report = _pop_option(argv, "--report", has_value=True)
    report = report if isinstance(report, str) else None
    if _pop_option(argv, "--self-test"):
        return self_test(report)
    if _pop_option(argv, "--validate"):
        return validate(report)
    from mrstudio.gui.app import main
    return main(argv)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run())
