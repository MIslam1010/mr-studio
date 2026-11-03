"""Main window of MR Studio."""
from __future__ import annotations

import argparse
import faulthandler
import importlib
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..robots import ROBOTS, register_robot
from . import theme
from .base import AppContext

# (module, class) of every tab, in display order. Broken tabs show their traceback instead;
# a tab whose module does not exist yet is skipped silently.
TABS = [
    ("mrstudio.gui.tabs.fk_tab", "FKTab"),
    ("mrstudio.gui.tabs.velocity_tab", "VelocityTab"),
    ("mrstudio.gui.tabs.ik_tab", "IKTab"),
    ("mrstudio.gui.tabs.dynamics_tab", "DynamicsTab"),
    ("mrstudio.gui.tabs.control_tab", "ControlTab"),
    ("mrstudio.gui.tabs.closed_tab", "ClosedChainTab"),
    ("mrstudio.gui.tabs.stewart_tab", "StewartTab"),
    ("mrstudio.gui.tabs.advanced_control_tab", "AdvancedControlTab"),
    ("mrstudio.gui.tabs.parallel_tab", "ParallelRobotsTab"),
    ("mrstudio.gui.tabs.planning_tab", "PlanningTab"),
    ("mrstudio.gui.tabs.haptics_tab", "HapticsTab"),
    ("mrstudio.gui.tabs.grasping_tab", "GraspingTab"),
    ("mrstudio.gui.tabs.mobile_tab", "MobileTab"),
    ("mrstudio.gui.tabs.hardware_tab", "HardwareTab"),
    ("mrstudio.gui.tabs.practice_tab", "PracticeTab"),
    ("mrstudio.gui.tabs.validation_tab", "ValidationTab"),
]

#: tab class -> descriptive name (tab tooltip; the tab itself shows the short ``title``)
TAB_LONG_NAMES = {
    "FKTab": "Forward kinematics",
    "VelocityTab": "Velocity kinematics & statics",
    "IKTab": "Inverse kinematics",
    "DynamicsTab": "Dynamics",
    "ControlTab": "Trajectory & control",
    "ClosedChainTab": "Closed chains",
    "StewartTab": "Stewart–Gough platform",
    "AdvancedControlTab": "Task-space & force control",
    "ParallelRobotsTab": "3×RPR & Delta",
    "PlanningTab": "Motion planning",
    "HapticsTab": "Haptics & impedance",
    "GraspingTab": "Grasping & manipulation",
    "MobileTab": "Wheeled mobile robots",
    "HardwareTab": "Hardware bridge",
    "PracticeTab": "Practice problems",
    "ValidationTab": "Validation",
}

#: tabs for which the top bar's Engine selector matters (unless the tab sets ``uses_engine`` itself)
ENGINE_TABS = {"DynamicsTab", "ControlTab", "AdvancedControlTab", "ClosedChainTab"}
OWN_MECHANISM_TIP = "This tab uses its own mechanism"
ENGINE_TIP = ("Both engines implement the same equations and agree to ~1e-15 (see Validation tab). "
              "Pinocchio is orders of magnitude faster for simulation.")

DEFAULT_ROBOT = "UR5 (6R industrial)"

LIBRARIES = [
    ("modern-robotics", "modern_robotics", "Lynch & Park's official code: PoE, Jacobians, IK, dynamics, trajectories"),
    ("Pinocchio", "pin", "rigid-body dynamics (LAAS-CNRS), fast C++ RNEA/CRBA/ABA"),
    ("Robotics Toolbox", "roboticstoolbox-python", "Peter Corke: DH models, cross-checks"),
    ("TOPP-RA", "toppra", "time-optimal path parameterisation (Pham & Pham 2018)"),
    ("NumPy", "numpy", "arrays and linear algebra"),
    ("SciPy", "scipy", "ODE integration, root finding, optimisation"),
    ("SymPy", "sympy", "symbolic checks"),
    ("Matplotlib", "matplotlib", "mathtext rendering of the theory-panel equations"),
    ("PySide6 (Qt 6)", "PySide6", "user interface"),
    ("pyqtgraph", "pyqtgraph", "OpenGL 3-D views and plots"),
    ("PyOpenGL", "PyOpenGL", "OpenGL bindings"),
    ("imageio + imageio-ffmpeg", "imageio-ffmpeg", "MP4 (H.264) recording"),
    ("Pillow", "pillow", "GIF export"),
]


_SETTINGS = {"ephemeral": None}   # path of a throw-away settings file (--screenshot), see use_ephemeral_settings


def app_settings() -> QtCore.QSettings:
    """Persistent settings (``MRSTUDIO_SETTINGS=/path/file.ini`` overrides the location, e.g. for tests).

    After ``use_ephemeral_settings()`` (``--screenshot`` mode) every caller gets a fresh, empty file in the temp
    directory instead, so screenshots never depend on -- or change -- what the user saved (last tab, robot,
    lesson progress, ...)."""
    path = os.environ.get("MRSTUDIO_SETTINGS") or _SETTINGS["ephemeral"]
    if path:
        return QtCore.QSettings(path, QtCore.QSettings.IniFormat)
    return QtCore.QSettings("MRStudio", "MR Studio")


def use_ephemeral_settings() -> str:
    """Point ``app_settings()`` at an empty temporary file for the rest of this process (removed at exit)."""
    if _SETTINGS["ephemeral"] is None:
        import atexit
        import shutil
        d = tempfile.mkdtemp(prefix="mrstudio_settings_")
        atexit.register(shutil.rmtree, d, True)
        _SETTINGS["ephemeral"] = os.path.join(d, "settings.ini")
    return _SETTINGS["ephemeral"]


def _bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


# ---------------------------------------------------------------------------- crash log
LOG_MAX_BYTES = 1_000_000


def log_path() -> Path:
    """Per-user log file of MR Studio (uncaught exceptions, fatal-error tracebacks); its folder is created.

    macOS ``~/Library/Logs/MR Studio/mrstudio.log`` · Windows ``%LOCALAPPDATA%\\MR Studio\\mrstudio.log`` ·
    other ``~/.mrstudio/mrstudio.log``. ``MRSTUDIO_LOG=/path/file.log`` overrides the location. Falls back to
    the temp directory if that folder cannot be created or written.
    """
    env = os.environ.get("MRSTUDIO_LOG")
    if env:
        p = Path(env).expanduser()
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Logs" / "MR Studio" / "mrstudio.log"
    elif sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        p = Path(base) / "MR Studio" / "mrstudio.log"
    else:
        p = Path.home() / ".mrstudio" / "mrstudio.log"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(p.parent, os.W_OK):
            raise OSError(f"{p.parent} is not writable")
    except OSError:
        p = Path(tempfile.gettempdir()) / "MR Studio" / "mrstudio.log"
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def rotate_log(path: Path, max_bytes: int = LOG_MAX_BYTES) -> None:
    """Keep the log small: once it exceeds ``max_bytes`` it becomes ``<name>.1`` (one old copy is kept)."""
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


class _CrashBridge(QtCore.QObject):
    """Carries "something went wrong" notices from any thread to the GUI thread (queued signal)."""
    message = QtCore.Signal(str)


_crash = {"fh": None, "bridge": None, "last": 0.0, "prev_hook": None}
CRASH_NOTICE = "Something went wrong — details in the log (Help → Open log folder)."


def log_text(text: str) -> None:
    """Append ``text`` (with a timestamp) to the crash log, if crash logging is installed."""
    fh = _crash["fh"]
    if fh is None:
        return
    try:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text.rstrip()}\n")
        fh.flush()
    except Exception:
        pass


URL_LOG_MAX = 200


def url_for_log(url) -> str:
    """A received URL as it goes into the log: the raw text cut to ``URL_LOG_MAX`` characters, then ``repr``-quoted
    (control characters and newlines cannot forge log lines)."""
    text = str(url)
    cut = text[:URL_LOG_MAX]
    return repr(cut) + (f" (+{len(text) - URL_LOG_MAX} more chars)" if len(text) > URL_LOG_MAX else "")


def _log_exception(et, ev, tb, where: str = "") -> None:
    log_text(f"Uncaught exception{where}:\n" + "".join(traceback.format_exception(et, ev, tb)))
    now = time.monotonic()
    if _crash["bridge"] is not None and now - _crash["last"] > 10.0:  # rate-limited: no notice storm
        _crash["last"] = now
        try:
            _crash["bridge"].message.emit(CRASH_NOTICE)
        except RuntimeError:
            pass


def install_crash_logging() -> Path:
    """Log fatal errors (faulthandler) and uncaught exceptions (``sys.excepthook``, ``threading.excepthook``)
    to ``log_path()``; uncaught exceptions also show a rate-limited status-bar notice. Returns the log path."""
    path = log_path()
    if _crash["fh"] is not None:
        return path
    rotate_log(path)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    _crash["fh"] = fh
    fh.write(f"\n===== MR Studio started {time.strftime('%Y-%m-%d %H:%M:%S')} · Python "
             f"{'.'.join(map(str, sys.version_info[:3]))} · Qt {QtCore.qVersion()} · {sys.platform} =====\n")
    try:
        faulthandler.enable(file=fh, all_threads=True)
    except Exception:
        pass
    _crash["bridge"] = _CrashBridge()
    prev = _crash["prev_hook"] = sys.excepthook

    def hook(et, ev, tb):
        if not issubclass(et, KeyboardInterrupt):
            _log_exception(et, ev, tb)
        try:
            prev(et, ev, tb)  # still printed on stderr
        except Exception:
            pass

    def thread_hook(args):
        if args.exc_type is not SystemExit:
            name = getattr(args.thread, "name", "?")
            _log_exception(args.exc_type, args.exc_value, args.exc_traceback, f" in thread {name}")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = hook
    threading.excepthook = thread_hook
    return path


# ---------------------------------------------------------------------------- OpenGL
def software_gl_requested(flag: bool = False) -> bool:
    return flag or os.environ.get("MRSTUDIO_SOFTWARE_GL", "").strip().lower() in ("1", "true", "yes", "on")


def use_software_gl() -> None:
    """Select Qt's software OpenGL renderer; must run before the QApplication is created."""
    os.environ["QT_OPENGL"] = "software"
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)


def opengl_available() -> bool:
    """Can an OpenGL context be created and made current? (Headless platforms draw nothing: always True.)"""
    if QtGui.QGuiApplication.platformName() in ("offscreen", "minimal"):
        return True
    try:
        ctx = QtGui.QOpenGLContext()
        if not ctx.create():
            return False
        surf = QtGui.QOffscreenSurface()
        surf.setFormat(ctx.format())
        surf.create()
        ok = ctx.makeCurrent(surf)
        if ok:
            ctx.doneCurrent()
        surf.destroy()
        return bool(ok)
    except Exception:
        return False


class ErrorTab(QtWidgets.QWidget):
    """Shown instead of a tab whose module raised while loading: friendly heading + copyable traceback."""
    uses_serial_robot = False
    uses_engine = False

    def __init__(self, name, err):
        super().__init__()
        self.title = name.split(".")[-1]
        self.module = name
        self.details = err
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel("This tab could not start")
        head.setObjectName("title")
        lay.addWidget(head)
        msg = QtWidgets.QLabel(f"The module <b>{name}</b> raised an error while it was loading; the other tabs "
                               "are not affected. Please copy the details below when you report the problem.")
        msg.setWordWrap(True)
        msg.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(msg)
        self.copy_btn = QtWidgets.QPushButton("Copy details")
        self.copy_btn.clicked.connect(self.copy_details)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.copy_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.text = QtWidgets.QPlainTextEdit(err)
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)

    def copy_details(self):
        py = ".".join(map(str, sys.version_info[:3]))
        QtWidgets.QApplication.clipboard().setText(
            f"MR Studio — tab {self.module} could not start\nPython {py} · Qt {QtCore.qVersion()} · "
            f"{sys.platform}\n\n{self.details}")
        self.copy_btn.setText("Copied ✓")
        QtCore.QTimer.singleShot(1500, lambda: self.copy_btn.setText("Copy details"))

    def on_show(self):
        pass

    def on_hide(self):
        pass

    def shutdown(self):
        pass


class ReferencesDialog(QtWidgets.QDialog):
    """Help → References…: every book/paper cited in the theory panels, with an "Open PDF" button
    for each reference whose PDF exists on this machine (mrstudio.references.REFERENCES)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from ..references import REFERENCES, pdf_path
        self.setWindowTitle("References")
        self.resize(760, 640)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel("Books and papers cited in the theory panels (“Further reading”). "
                                 "Page numbers there are the <i>printed</i> page numbers of these editions.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        grid.setColumnStretch(0, 1)
        self.open_buttons: dict[str, QtWidgets.QPushButton] = {}
        for row, (key, ref) in enumerate(REFERENCES.items()):
            lab = QtWidgets.QLabel(f"<b>{ref['short']}</b><br>{ref['citation']}<br>"
                                   f"<span style='color:{theme.muted_color()}'>{ref['note']}</span>")
            lab.setTextFormat(QtCore.Qt.RichText)
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            grid.addWidget(lab, row, 0)
            path = pdf_path(key)
            if path:
                btn = QtWidgets.QPushButton("Open PDF")
                btn.setToolTip(path)
                btn.clicked.connect(lambda _=False, p=path: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(p)))
                self.open_buttons[key] = btn
                grid.addWidget(btn, row, 1, QtCore.Qt.AlignTop)
            else:
                na = QtWidgets.QLabel("PDF not found")
                na.setStyleSheet(f"color:{theme.muted_color()}")
                grid.addWidget(na, row, 1, QtCore.Qt.AlignTop)
        grid.setRowStretch(len(REFERENCES), 1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


def shortcut_text(obj) -> str:
    """The shortcut(s) of a QAction / button in the platform's notation (``⌘⇧S`` on macOS, ``Ctrl+Shift+S``
    elsewhere), several joined by " or "; '' if it has none."""
    seqs = obj.shortcuts() if hasattr(obj, "shortcuts") else [obj.shortcut()]
    return " or ".join(q.toString(QtGui.QKeySequence.NativeText) for q in seqs if not q.isEmpty())


def shortcut_label(action: QtGui.QAction) -> str:
    """Name of an action in the shortcut list: its ``shortcut_label`` property, else its menu text without
    mnemonics, leading symbols ("＋ ", "● ") and a trailing ellipsis."""
    lab = action.property("shortcut_label")
    return str(lab) if lab else clean_label(action.text())


def clean_label(text: str) -> str:
    """Menu / button text without mnemonics ("&"), leading symbols ("＋ ", "● ", "■ ") and an ellipsis."""
    t = text.replace("&&", "\0").replace("&", "").replace("\0", "&").replace("…", "").strip()
    while t and not t[0].isalnum():
        t = t[1:].lstrip()
    return t


class ShortcutsDialog(QtWidgets.QDialog):
    """Help → Keyboard shortcuts…: generated from the window's actions and the tabs' buttons (never stale)."""

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(560, 600)
        lay = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Action", "Shortcut"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for group, label, keys in rows:
            parent_item = groups.get(group)
            if parent_item is None:
                parent_item = groups[group] = QtWidgets.QTreeWidgetItem(self.tree, [group, ""])
                f = parent_item.font(0)
                f.setBold(True)
                parent_item.setFont(0, f)
                parent_item.setFirstColumnSpanned(True)
                parent_item.setFlags(QtCore.Qt.ItemIsEnabled)
            QtWidgets.QTreeWidgetItem(parent_item, [label, keys])
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.setColumnWidth(0, self.tree.columnWidth(0) + 24)
        lay.addWidget(self.tree, 1)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


def load_tab(mod: str, cls: str, ctx):
    """Instantiate a tab. Returns (page, label); None if the module itself does not exist (yet)."""
    try:
        m = importlib.import_module(mod)
        page = getattr(m, cls)(ctx)
        return page, page.title
    except ModuleNotFoundError as e:
        if e.name == mod:
            return None
        err = traceback.format_exc()
    except Exception:
        err = traceback.format_exc()
    print(err, file=sys.stderr)
    log_text(f"Tab {mod} could not start:\n{err}")
    return ErrorTab(mod, err), "⚠ " + mod.split(".")[-1]


def apply_page_preset(page, key: str) -> bool:
    """Select preset ``key`` on a tab (launch URLs): ``page.apply_preset(key)`` if the tab has it, else the tab's
    ``preset_box`` (exact name, 0-based index, or a case-insensitive part of the name). False if not found."""
    fn = getattr(page, "apply_preset", None)
    if callable(fn):
        return bool(fn(key))
    box = getattr(page, "preset_box", None)
    if not isinstance(box, QtWidgets.QComboBox) or not key:
        return False
    names = [box.itemText(i) for i in range(box.count())]
    k = str(key).strip()
    if k in names:
        i = names.index(k)
    elif k.isdecimal() and len(k) <= 6 and int(k) < len(names):   # isdigit() accepts "²", which int() rejects
        i = int(k)
    else:
        i = next((j for j, n in enumerate(names) if k.lower() in n.lower()), -1)
    if i < 0:
        return False
    if i == box.currentIndex():           # same entry: no signal, load it again explicitly
        load = getattr(page, "load_preset", None) or getattr(page, "_preset", None)
        if callable(load):
            load(names[i])
    else:
        box.setCurrentIndex(i)            # currentTextChanged -> the tab's own preset slot
    return True


_FILE_OPEN = QtCore.QEvent.FileOpen        # looked up once: the filter sees every event of the app


class UrlOpenFilter(QtCore.QObject):
    """Receives ``mrstudio://`` URLs that macOS delivers as ``QFileOpenEvent`` (the app is opened by a link).

    URLs that arrive before the main window exists are kept and handed over by ``set_window``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = None
        self.pending: list[str] = []

    def set_window(self, win):
        self.window = win
        pending, self.pending = self.pending, []
        for u in pending:
            win.open_url(u)

    def eventFilter(self, obj, ev):
        if ev.type() == _FILE_OPEN:
            url = ev.url().toString() or ev.file()
            if url.lower().startswith("mrstudio:"):
                if self.window is not None:
                    self.window.open_url(url)
                    self.window.raise_()
                    self.window.activateWindow()
                else:
                    self.pending.append(url)
                return True
        return super().eventFilter(obj, ev)


class _ModelWarmer(QtCore.QObject):
    done = QtCore.Signal(object)  # list of robot names that failed to load


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.persist = True  # save settings on close / toggle (off in --screenshot mode)
        self.setWindowTitle("MR Studio — Kinematics & Dynamics of Robotic Chains (Modern Robotics, Lynch & Park)")
        self.resize(1560, 940)

        from .recorder import Recorder
        self.recorder = Recorder(self)
        self.recorder.tick.connect(self._rec_tick)
        self.recorder.exported.connect(self._rec_exported)
        self.recorder.failed.connect(self._rec_failed)
        self.recorder.finished.connect(self._rec_finished)
        self.rec_source = "view"
        self.rec_fps = 30
        self._closing = False
        self._shut_down = False

        self._make_actions()
        self._make_menus()
        bar = self._make_top_bar()

        # ---- tabs -----------------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setUsesScrollButtons(True)
        self.pages = []
        for mod, cls in TABS:
            res = load_tab(mod, cls, ctx)
            if res is None:
                continue
            page, label = res
            self.pages.append(page)
            i = self.tabs.addTab(page, label)
            tip = TAB_LONG_NAMES.get(cls, label)
            if i < 9:
                key = QtGui.QKeySequence(f"Ctrl+{i + 1}").toString(QtGui.QKeySequence.NativeText)
                tip = f"{tip}  ({key})"
            self.tabs.setTabToolTip(i, tip)
        self._current = None
        self.tabs.currentChanged.connect(self._tab_changed)
        self._make_tab_shortcuts()

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(bar)
        v.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.default_status = ("Core: modern_robotics (Lynch & Park) · Pinocchio (LAAS-CNRS) · Robotics Toolbox "
                               "(P. Corke) · TOPP-RA · SciPy — all computations validated against each other "
                               "(Validation tab).")
        self.statusBar().showMessage(self.default_status)
        if _crash["bridge"] is not None:
            _crash["bridge"].message.connect(lambda m: self.statusBar().showMessage(m, 15000))
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown_pages)
        ctx.projector_changed.connect(self._projector_changed)
        self._projector_changed(theme.is_projector(), save=False)
        if theme.is_projector():
            QtCore.QTimer.singleShot(0, lambda: theme.style_plots_in(self))
        QtCore.QTimer.singleShot(0, lambda: self._tab_changed(self.tabs.currentIndex()))

    # ------------------------------------------------------------------ actions / menus / top bar
    def _act(self, text, slot=None, shortcut=None, checkable=False, tip=None, label=None):
        """A window action. Its tooltip gets the shortcut appended in the platform's notation (⌘O / Ctrl+O);
        ``label`` names it in Help → Keyboard shortcuts (default: ``text`` without symbols / ellipsis)."""
        a = QtGui.QAction(text, self)
        if shortcut:
            seqs = shortcut if isinstance(shortcut, (list, tuple)) else [shortcut]
            a.setShortcuts([QtGui.QKeySequence(s) for s in seqs])
            a.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        a.setCheckable(checkable)
        if label:
            a.setProperty("shortcut_label", label)
        if tip:
            keys = shortcut_text(a)
            a.setToolTip(f"{tip} ({keys})" if keys else tip)
            a.setStatusTip(tip)
        if slot is not None:
            (a.toggled if checkable else a.triggered).connect(slot)
        self.addAction(a)  # shortcuts work even without a visible menu bar
        return a

    def _make_actions(self):
        self.act_builder = self._act("＋ Robot Builder…", self._open_builder, "Ctrl+B",
                                     tip="Define your own robot by its screw axes (ω_i, q_i) and home pose M")
        self.act_urdf = self._act("Import URDF…", self._import_urdf, "Ctrl+O",
                                  tip="Load a serial robot from a URDF file")
        self.act_shot = self._act("Save screenshot (PNG)…", self._save_screenshot, "Ctrl+Shift+S",
                                  tip="Save the whole window as a PNG")
        self.act_shot_view = self._act("Save 3-D view image (PNG)…", self._save_view_screenshot, "Ctrl+Alt+S",
                                       tip="Save only the current tab's 3-D view (with labels) as a PNG")
        self.act_record = self._act("● Record", self._toggle_record, "Ctrl+R", checkable=True,
                                    tip="Start/stop recording a clip, saved as MP4 or GIF. "
                                        "Arrow: choose source / frame rate.", label="Start / stop recording")
        self.act_quit = self._act("Quit", self.close, QtGui.QKeySequence.Quit)
        self.act_quit.setMenuRole(QtGui.QAction.QuitRole)
        self.act_theory = self._act("Theory panel", self._toggle_theory, "F1", checkable=True,
                                    tip="Show/hide the equations panel")
        self.act_theory.setChecked(self.ctx.theory_visible)
        self.act_projector = self._act("Projector mode", self.ctx.set_projector, ["Ctrl+Shift+P", "F5"],
                                       checkable=True,
                                       tip="Large fonts, thick lines, high contrast for presenting")
        self.act_projector.setChecked(theme.is_projector())
        self.act_projector_light = self._act("Projector (light)", self._set_projector_light, checkable=True,
                                             tip="Light, high-contrast palette for projector mode "
                                                 "(for washed-out projectors and bright rooms)")
        self.act_projector_light.setChecked(theme.light_preference())
        self.act_log = self._act("Open log folder", self._open_log_folder,
                                 tip="Show the folder of mrstudio.log (error details for bug reports)")
        self.act_about = self._act("About MR Studio / libraries…", self._about)
        self.act_about.setMenuRole(QtGui.QAction.AboutRole)
        self.act_refs = self._act("References…", self._references,
                                  tip="Bibliography of the theory panels' “Further reading” (with local PDFs)")
        self.act_keys = self._act("Keyboard shortcuts…", self._shortcuts, "Ctrl+/",
                                  tip="List every keyboard shortcut of MR Studio")

        # recording options
        self.src_group = QtGui.QActionGroup(self)
        self.act_src_view = self._act("Record the current tab's 3-D view (incl. 3-D labels)", checkable=True)
        self.act_src_win = self._act("Record the whole window", checkable=True)
        for a, key in ((self.act_src_view, "view"), (self.act_src_win, "window")):
            self.src_group.addAction(a)
            a.triggered.connect(lambda _=False, k=key: self._set_rec_source(k))
        self.act_src_view.setChecked(True)
        self.fps_group = QtGui.QActionGroup(self)
        self.fps_acts = {}
        for f in (25, 30):
            a = self._act(f"{f} fps", checkable=True)
            self.fps_group.addAction(a)
            a.triggered.connect(lambda _=False, f=f: self._set_rec_fps(f))
            self.fps_acts[f] = a
        self.fps_acts[30].setChecked(True)

    def _make_tab_shortcuts(self):
        """Ctrl+1..9 (⌘1..9 on macOS) select tabs 1-9; Ctrl+Alt+Right/Left the next / previous tab."""
        self.tab_acts = []
        for i in range(min(9, self.tabs.count())):
            a = QtGui.QAction(f"Tab {i + 1}", self)
            name = TAB_LONG_NAMES.get(type(self.pages[i]).__name__, self.tabs.tabText(i))
            a.setProperty("shortcut_label", f"Go to tab {i + 1}: {name}")
            a.setShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"))
            a.triggered.connect(lambda _=False, k=i: self.tabs.setCurrentIndex(k))
            self.addAction(a)
            self.tab_acts.append(a)
        self.act_next_tab = QtGui.QAction("Next tab", self)
        self.act_next_tab.setShortcut(QtGui.QKeySequence("Ctrl+Alt+Right"))
        self.act_next_tab.triggered.connect(lambda: self.step_tab(1))
        self.act_prev_tab = QtGui.QAction("Previous tab", self)
        self.act_prev_tab.setShortcut(QtGui.QKeySequence("Ctrl+Alt+Left"))
        self.act_prev_tab.triggered.connect(lambda: self.step_tab(-1))
        for a in (self.act_next_tab, self.act_prev_tab):
            self.addAction(a)
        if hasattr(self, "view_menu"):
            self.view_menu.addSeparator()
            self.view_menu.addAction(self.act_next_tab)
            self.view_menu.addAction(self.act_prev_tab)

    def page_index(self, cls_name: str) -> int:
        """Index of the tab whose class is ``cls_name`` (-1 if it is not loaded)."""
        return next((i for i, p in enumerate(self.pages) if type(p).__name__ == cls_name), -1)

    def open_target(self, tab: str | int | None = None, robot: str | None = None, preset: str | None = None) -> bool:
        """Switch to ``tab`` (class name or index), select ``robot`` (a name in ROBOTS) and the tab's ``preset``.
        Used by launch URLs and the Lessons panel's "Go to tab". Returns False if the tab is unknown."""
        if robot:
            if robot in ROBOTS:
                self.robot_box.setCurrentText(robot)
            else:
                self.statusBar().showMessage(f"Unknown robot {robot!r}: keeping {self.ctx.robot.name}", 10000)
        if tab is None:
            return True
        i = tab if isinstance(tab, int) else self.page_index(tab)
        if not 0 <= i < self.tabs.count():
            self.statusBar().showMessage(f"No tab {tab!r} in this version of MR Studio", 10000)
            return False
        self.tabs.setCurrentIndex(i)
        if preset:
            try:
                ok = apply_page_preset(self.pages[i], preset)
            except Exception:
                traceback.print_exc()
                ok = False
            if not ok:
                self.statusBar().showMessage(f"Preset {preset!r} not found on this tab", 10000)
        return True

    def open_url(self, url: str) -> bool:
        """Handle ``mrstudio://open?tab=…&robot=…&preset=…`` (Knowledge Map links); False if it is not valid."""
        from ..kmap_links import parse_launch_url
        try:
            q = parse_launch_url(url)
        except ValueError as e:
            self.statusBar().showMessage(str(e), 10000)
            log_text(f"launch URL ignored: {e}; URL received: {url_for_log(url)}")
            return False
        ok = self.open_target(q["tab"], q["robot"], q["preset"])
        return ok and (q["tab"] is not None or q["robot"] is not None)

    def step_tab(self, d: int):
        n = self.tabs.count()
        if n:
            self.tabs.setCurrentIndex((self.tabs.currentIndex() + d) % n)

    def _set_rec_source(self, k):
        self.rec_source = k

    def _set_rec_fps(self, f):
        self.rec_fps = f

    def _make_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self.act_builder)
        m.addAction(self.act_urdf)
        m.addSeparator()
        m.addAction(self.act_shot)
        m.addAction(self.act_shot_view)
        self.rec_menu = QtWidgets.QMenu("Recording", self)
        self.rec_menu.addAction(self.act_record)
        self.rec_menu.addSeparator()
        self.rec_menu.addAction(self.act_src_view)
        self.rec_menu.addAction(self.act_src_win)
        self.rec_menu.addSeparator()
        for a in self.fps_acts.values():
            self.rec_menu.addAction(a)
        self.rec_menu.addSeparator()
        self.rec_menu.addAction(self.act_shot)
        self.rec_menu.addAction(self.act_shot_view)
        m.addMenu(self.rec_menu)
        m.addSeparator()
        m.addAction(self.act_quit)
        v = self.view_menu = mb.addMenu("&View")
        v.addAction(self.act_theory)
        v.addAction(self.act_projector)
        v.addAction(self.act_projector_light)
        h = mb.addMenu("&Help")
        h.addAction(self.act_refs)
        h.addAction(self.act_keys)
        h.addAction(self.act_log)
        h.addAction(self.act_about)

    def _tool(self, action, text=None, name=None):
        b = QtWidgets.QToolButton()
        b.setDefaultAction(action)
        b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        if text:
            action.setIconText(text)  # short label on the button; menus keep the full text
        if name:
            b.setObjectName(name)
        return b

    def _make_top_bar(self):
        bar = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(bar)
        hl.setContentsMargins(10, 6, 10, 2)
        title = QtWidgets.QLabel("MR Studio")
        title.setObjectName("title")
        hl.addWidget(title)
        self.subtitle = QtWidgets.QLabel("screw theory · PoE · Newton–Euler · constrained dynamics")
        self.subtitle.setObjectName("muted")
        hl.addWidget(self.subtitle)
        hl.addStretch(1)
        hl.addWidget(QtWidgets.QLabel("Robot:"))
        self.robot_box = QtWidgets.QComboBox()
        self.robot_box.addItems(list(ROBOTS))
        self.robot_box.setCurrentText(self.ctx.robot.name)
        self.robot_box.currentTextChanged.connect(self.ctx.set_robot)
        self.robot_box.setMinimumWidth(220)
        hl.addWidget(self.robot_box)
        hl.addWidget(self._tool(self.act_builder, "＋ Builder…"))
        hl.addWidget(self._tool(self.act_urdf))
        hl.addSpacing(12)
        hl.addWidget(QtWidgets.QLabel("Engine:"))
        self.engine_box = QtWidgets.QComboBox()
        self.engine_box.addItem("Pinocchio (C++, fast)", "pinocchio")
        self.engine_box.addItem("Modern Robotics (reference)", "mr")
        self.engine_box.currentIndexChanged.connect(lambda i: self.ctx.set_backend(self.engine_box.itemData(i)))
        self.engine_box.setToolTip(ENGINE_TIP)
        hl.addWidget(self.engine_box)
        hl.addSpacing(12)
        self.theory_btn = self._tool(self.act_theory, "Theory")
        hl.addWidget(self.theory_btn)
        self.projector_btn = self._tool(self.act_projector, "Projector")
        hl.addWidget(self.projector_btn)
        self.rec_btn = self._tool(self.act_record, name="rec")
        self.rec_btn.setMenu(self.rec_menu)
        self.rec_btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        hl.addWidget(self.rec_btn)
        self.rec_label = QtWidgets.QLabel("● REC 00:00")
        self.rec_label.setObjectName("rec")
        self.rec_label.setVisible(False)
        hl.addWidget(self.rec_label)
        shot = self._tool(self.act_shot, "PNG")
        hl.addWidget(shot)
        return bar

    # ------------------------------------------------------------------ robots
    def _open_builder(self):
        from .robot_builder import RobotBuilder
        dlg = RobotBuilder(self.ctx.robot, self)
        dlg.robot_created.connect(self._robot_created)
        dlg.exec()

    def _error(self, title, text, hint="", details=""):
        """Friendly error box (details folded away) + the same details in the log; returns the box."""
        from .widgets import error_box
        log_text(f"{title}: {text}\n{details}")
        self._error_box = error_box(self, title, text, details=details, hint=hint)
        return self._error_box

    def _import_urdf(self):
        from .widgets import exception_details
        try:
            from mrstudio.gui.urdf_dialog import open_urdf_dialog
        except Exception:
            self._error("Import URDF", "URDF import is not available in this installation.",
                        "A library it needs could not be loaded. The other features of MR Studio still work.",
                        exception_details())
            return
        try:
            robot = open_urdf_dialog(self)
        except Exception as e:
            traceback.print_exc()
            self._error("Import URDF", f"The robot could not be imported ({type(e).__name__}).",
                        "Check that the file is a URDF of a serial chain (one path from the base link to the "
                        "chosen tip link) with <inertial> data for every moving link.", exception_details(e))
            return
        if robot is not None:
            self._robot_created(robot)

    def _robot_created(self, robot):
        """A robot from the Robot Builder / URDF import: register it and switch to it -- also when a robot of
        the same name exists (it is replaced, unless it is a built-in robot: then the new one is renamed)."""
        requested = robot.name
        name = register_robot(robot)
        if self.robot_box.findText(name) < 0:
            self.robot_box.addItem(name)
        self.robot_box.blockSignals(True)
        self.robot_box.setCurrentText(name)
        self.robot_box.blockSignals(False)
        self.ctx.set_robot(name, force=True)
        if name != requested:
            self.statusBar().showMessage(f"“{requested}” is a built-in robot, so your robot was added as "
                                         f"“{name}”.", 12000)
        else:
            self.statusBar().showMessage(f"Robot “{name}” loaded.", 6000)

    def warm_robot_models(self):
        """Build the Robotics-Toolbox robots (PUMA, Panda, Stanford, SCARA) in a background thread, so that
        selecting one later does not freeze the window while ``roboticstoolbox`` is imported."""
        from .. import robots as R
        names = [n for n in R.RTB_ROBOTS if n in ROBOTS]
        if not names or getattr(self, "_warmer", None) is not None:
            return
        self._warmer = _ModelWarmer(self)
        self._warmer.done.connect(self._models_warm)
        self.statusBar().showMessage("Loading robot models…")
        threading.Thread(target=lambda: self._warmer.done.emit(R.preload(names)), name="mrstudio-preload",
                         daemon=True).start()

    def _models_warm(self, failed):
        if failed:
            log_text(f"Could not preload robot models: {', '.join(failed)}")
        if self.statusBar().currentMessage() == "Loading robot models…":
            self.statusBar().showMessage(self.default_status)

    # ------------------------------------------------------------------ view
    def _toggle_theory(self, on):
        self.ctx.theory_visible = on
        self.ctx.theory_visible_changed.emit(on)

    def _set_projector_light(self, on):
        theme.set_projector_light(QtWidgets.QApplication.instance(), on)
        if on and not theme.is_projector():
            self.ctx.set_projector(True)
        if self.persist:
            app_settings().setValue("projector_light", bool(on))

    def _open_log_folder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(log_path().parent)))

    def _projector_changed(self, on, save=True):
        if self.act_projector.isChecked() != on:
            self.act_projector.blockSignals(True)
            self.act_projector.setChecked(on)
            self.act_projector.blockSignals(False)
        self.subtitle.setVisible(not on)
        for p in self.pages:
            if not hasattr(p, "restyle_plots"):  # BaseTab pages restyle themselves
                theme.style_plots_in(p)
        if save and self.persist:
            app_settings().setValue("projector", on)

    def _tab_changed(self, idx):
        if idx < 0 or idx >= len(self.pages):
            return
        if self._current is not None:
            try:
                self._current.on_hide()
            except Exception:
                traceback.print_exc()
        self._current = page = self.pages[idx]
        self._update_top_bar(page)
        try:
            if hasattr(page, "refresh_if_stale"):
                page.refresh_if_stale()  # robot / engine changed while the tab was hidden
            page.on_show()
        except Exception:
            traceback.print_exc()
        if theme.is_projector():  # catch curves created since the last toggle
            QtCore.QTimer.singleShot(0, lambda p=self._current: theme.style_plots_in(p))

    def _update_top_bar(self, page):
        """Grey out the Robot (and Engine) selector on tabs that ignore it, instead of silently ignoring it."""
        uses_robot = bool(getattr(page, "uses_serial_robot", True))
        uses_engine = getattr(page, "uses_engine", None)
        if uses_engine is None:
            uses_engine = type(page).__name__ in ENGINE_TABS
        self.robot_box.setEnabled(uses_robot)
        self.robot_box.setToolTip("" if uses_robot else OWN_MECHANISM_TIP)
        self.engine_box.setEnabled(bool(uses_engine))
        self.engine_box.setToolTip(ENGINE_TIP if uses_engine else
                                   (OWN_MECHANISM_TIP if not uses_robot else
                                    "This tab does not use the dynamics engine"))

    def shortcut_rows(self) -> list[tuple[str, str, str]]:
        """(group, action, keys) for every keyboard shortcut: the menus' actions, the tab shortcuts, other window
        actions and buttons inside the tabs that have a shortcut (e.g. the Hardware tab's Esc = Stop)."""
        rows, seen = [], set()

        def add(group, a):
            keys = shortcut_text(a)
            if keys and id(a) not in seen:
                seen.add(id(a))
                rows.append((group, shortcut_label(a), keys))

        def walk(menu, group):
            for a in menu.actions():
                if a.menu() is not None:
                    walk(a.menu(), group)
                elif not a.isSeparator():
                    add(group, a)

        for top in self.menuBar().actions():
            if top.menu() is not None:
                walk(top.menu(), top.text().replace("&", ""))
        for a in self.tab_acts + [self.act_next_tab, self.act_prev_tab]:
            add("Tabs", a)
        for a in self.actions():
            add("Other", a)
        for i, page in enumerate(self.pages):
            group = TAB_LONG_NAMES.get(type(page).__name__, self.tabs.tabText(i)) + " tab"
            for b in page.findChildren(QtWidgets.QAbstractButton):
                if not b.shortcut().isEmpty() and id(b) not in seen:
                    seen.add(id(b))
                    rows.append((group, clean_label(b.text()) or b.toolTip(), shortcut_text(b)))
        return rows

    def _shortcuts(self):
        dlg = ShortcutsDialog(self.shortcut_rows(), self)
        dlg.show()
        self._keys_dialog = dlg  # keep a reference (non-modal)
        return dlg

    def about_html(self) -> str:
        from importlib import metadata
        rows = []
        for name, dist, what in LIBRARIES:
            try:
                ver = metadata.version(dist)
            except Exception:
                # e.g. conda-forge Pinocchio in the frozen Windows app is not a pip distribution called "pin"
                mod = {"pin": "pinocchio"}.get(dist, dist.replace("-", "_"))
                try:
                    ver = getattr(importlib.import_module(mod), "__version__", "—")
                except Exception:
                    ver = "—"
            rows.append(f"<tr><td><b>{name}</b></td><td>&nbsp;{ver}&nbsp;</td><td>{what}</td></tr>")
        py = ".".join(map(str, sys.version_info[:3]))
        main_rows = [(lab, k) for g, lab, k in self.shortcut_rows()
                     if g in ("File", "View", "Help") and lab != shortcut_label(self.act_keys)]
        first, last = self.tab_acts[:1], self.tab_acts[-1:]
        tabs = (f"{shortcut_text(first[0])}…{shortcut_text(last[0])} tabs 1–{len(self.tab_acts)}"
                if first else "")
        keys = " · ".join([f"{k} {lab}" for lab, k in main_rows] + ([tabs] if tabs else []))
        keys += f" — full list: Help → Keyboard shortcuts… ({shortcut_text(self.act_keys)})"
        return (
            "<h2>MR Studio</h2>"
            "<p>Interactive kinematics &amp; dynamics of open and closed robotic chains, following</p>"
            "<p><b>K. M. Lynch and F. C. Park, <i>Modern Robotics: Mechanics, Planning, and Control</i>, "
            "Cambridge University Press, 2017</b> — "
            "<a href='http://modernrobotics.org'>modernrobotics.org</a></p>"
            "<p>Conventions: twists V = (ω, v), wrenches F = (m, f), screw axes in {s}, product of "
            "exponentials, Newton–Euler inverse dynamics.</p>"
            f"<h3>Libraries</h3><table>{''.join(rows)}</table>"
            f"<p>Python {py} · Qt {QtCore.qVersion()}</p>"
            f"<h3>Shortcuts</h3><p>{keys}</p>"
            f"<p>Log file: {log_path()}</p>")

    def _about(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About MR Studio")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText(self.about_html())
        box.exec()

    def _references(self):
        dlg = ReferencesDialog(self)
        dlg.show()
        self._refs_dialog = dlg  # keep a reference (non-modal)
        return dlg

    # ------------------------------------------------------------------ screenshots / recording
    def current_view(self):
        """Largest visible SceneView in the current tab (None if the tab has no 3-D view)."""
        from .viewer import SceneView
        page = self.tabs.currentWidget()
        if page is None:
            return None
        views = [v for v in page.findChildren(SceneView) if v.isVisible() and v.width() > 10]
        return max(views, key=lambda v: v.width() * v.height()) if views else None

    def _default_dir(self):
        d = app_settings().value("last_dir", "")
        if d and os.path.isdir(d):
            return d
        for c in ("~/Movies", "~/Desktop", "~"):
            p = os.path.expanduser(c)
            if os.path.isdir(p):
                return p
        return os.getcwd()

    def _stem(self):
        t = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else "MRStudio"
        t = "".join(ch for ch in t if ch.isascii() and (ch.isalnum() or ch in " -_")).strip().replace(" ", "_")
        return f"MRStudio_{t or 'tab'}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    def _ask_path(self, title, ext, filters):
        d = self._default_dir()
        fn, flt = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(d, self._stem() + ext), filters)
        if not fn:
            return None
        if not os.path.splitext(fn)[1]:
            fn += ".gif" if "gif" in flt.lower() else ext
        app_settings().setValue("last_dir", os.path.dirname(fn))
        return fn

    def _save_screenshot(self):
        img = self.grab()
        fn = self._ask_path("Save screenshot", ".png", "PNG image (*.png)")
        if fn:
            img.save(fn)
            self.statusBar().showMessage(f"Saved {fn}", 6000)

    def _save_view_screenshot(self):
        view = self.current_view()
        if view is None:
            return self._save_screenshot()
        img = view.grabFramebuffer()
        fn = self._ask_path("Save 3-D view image", ".png", "PNG image (*.png)")
        if fn:
            img.save(fn)
            self.statusBar().showMessage(f"Saved {fn}", 6000)

    def grab_function(self, source=None):
        """Callable for the recorder. "view": re-resolves the current tab's 3-D view on every frame (so a tab
        switch while recording follows the new tab); falls back to the whole window if it has no 3-D view."""
        source = source or self.rec_source
        if source == "view":
            if self.current_view() is None:
                self.statusBar().showMessage("This tab has no 3-D view — recording the whole window.", 5000)

            def grab():
                view = self.current_view()
                if view is not None:
                    img = view.grabFramebuffer()
                    if not img.isNull():
                        return img
                return self.grab()  # no 3-D view on this tab (or no GL framebuffer)
            return grab
        return self.grab

    def _set_record_checked(self, on):
        if self.act_record.isChecked() != on:
            self.act_record.blockSignals(True)
            self.act_record.setChecked(on)
            self.act_record.blockSignals(False)
        self.act_record.setText("■ Stop" if on else "● Record")

    def start_recording(self, source=None, fps=None):
        if self.recorder.recording:
            return
        if self.recorder.busy():
            self._set_record_checked(False)
            self.statusBar().showMessage("Still finishing the previous recording — try again in a moment.", 5000)
            return
        try:
            self.recorder.start(self.grab_function(source), fps or self.rec_fps)
        except Exception as e:
            from .widgets import exception_details
            self._set_record_checked(False)
            hint = ("The video encoder is missing. From a source checkout, install it with "
                    "“pip install imageio imageio-ffmpeg pillow”."
                    if isinstance(e, ImportError) else
                    "Screenshots (File → Save screenshot) still work.")
            self._error("Record", "Recording could not start.", hint, exception_details(e))
            return
        self.rec_label.setText("● REC 00:00")
        self.rec_label.setVisible(True)
        self._set_record_checked(True)

    def stop_recording(self, ask=True):
        """Stop. ask=False: wait for the encoder and return the temp MP4 path; ask=True: return at once, the
        encoder finishes in its thread and ``_rec_finished`` then shows the save dialog and exports async."""
        if not self.recorder.recording:
            return None
        self.rec_label.setVisible(False)
        self._set_record_checked(False)
        if not ask:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                return self.recorder.stop(wait=True)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Finishing the recording…")
        self.recorder.stop(wait=False)
        return None

    def _save_recording(self, tmp, block=False):
        """Ask where to save the temp MP4 ``tmp`` and export it (MP4 move or GIF conversion)."""
        fn = self._ask_path("Save recording", ".mp4", "MP4 video (H.264) (*.mp4);;Animated GIF (*.gif)")
        if not fn:
            try:
                os.remove(tmp)
            except OSError:
                pass
            self.statusBar().showMessage("Recording discarded.", 4000)
            return
        self.statusBar().showMessage(f"Encoding {os.path.basename(fn)} …")
        self.recorder.export(tmp, fn, block=block)

    def _rec_finished(self, tmp):
        if self._closing or not tmp:
            return
        dropped = self.recorder.dropped
        if dropped:
            log_text(f"Recording: {dropped} frame(s) dropped (encoder too slow)")
        self._save_recording(tmp)

    def _toggle_record(self, on):
        if on:
            self.start_recording()
        else:
            self.stop_recording()

    def _rec_tick(self, t):
        s = int(t)
        dot = "●" if s % 2 == 0 else "○"
        self.rec_label.setText(f"{dot} REC {s // 60:02d}:{s % 60:02d}")

    def _rec_exported(self, path):
        try:
            mb = os.path.getsize(path) / 1e6
        except OSError:
            mb = 0
        self.statusBar().showMessage(f"Saved {path}  ({mb:.1f} MB)", 10000)

    def _rec_failed(self, msg):
        self.statusBar().showMessage(f"Recording failed: {msg}", 10000)
        log_text(f"Recording failed: {msg}")
        if self._closing:
            return
        from .widgets import error_box
        self._rec_failed_box = error_box(self, "Recording failed", "The recording could not be saved.",
                                         hint=f"{msg}\n\nCheck that the folder is writable and the disk is "
                                              "not full, then record again.")

    # ------------------------------------------------------------------ settings
    def restore_settings(self, restore_tab=True):
        s = app_settings()
        g = s.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        eng = s.value("engine")
        if eng:
            i = self.engine_box.findData(eng)
            if i >= 0:
                self.engine_box.setCurrentIndex(i)
        self.act_theory.setChecked(_bool(s.value("theory"), True))
        src = s.value("rec_source", "view")
        (self.act_src_win if src == "window" else self.act_src_view).trigger()
        try:
            fps = int(s.value("rec_fps", 30) or 30)
        except (TypeError, ValueError):
            fps = 30
        if fps in self.fps_acts:
            self.fps_acts[fps].trigger()
        if restore_tab:
            try:
                t = int(s.value("tab", 0) or 0)
            except (TypeError, ValueError):
                t = 0
            if 0 <= t < self.tabs.count():
                self.tabs.setCurrentIndex(t)

    def save_settings(self):
        s = app_settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("tab", self.tabs.currentIndex())
        s.setValue("robot", self.ctx.robot.name)
        s.setValue("engine", self.ctx.backend)
        s.setValue("theory", self.act_theory.isChecked())
        s.setValue("projector", theme.is_projector())
        s.setValue("projector_light", theme.light_preference())
        s.setValue("rec_source", self.rec_source)
        s.setValue("rec_fps", self.rec_fps)
        s.sync()

    def shutdown_pages(self):
        """Call ``shutdown()`` on every tab once (window close or application quit)."""
        if self._shut_down:
            return
        self._shut_down = True
        for p in self.pages:
            fn = getattr(p, "shutdown", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                traceback.print_exc()

    def closeEvent(self, e):
        if self.recorder.recording or self.recorder.busy():
            # quitting: finish synchronously, the event loop may not run again
            self.rec_label.setVisible(False)
            self._set_record_checked(False)
            self._closing = True
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                tmp = self.recorder.stop(wait=True)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            if tmp:
                self._save_recording(tmp, block=True)
        self._closing = True
        self.shutdown_pages()
        if self.persist:
            self.save_settings()
        super().closeEvent(e)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")  # before any pyplot import (roboticstoolbox, control.py): never open GUI figures

    parser = argparse.ArgumentParser(description="MR Studio")
    parser.add_argument("--robot", default=None, help=f"initial robot (default: last used, or {DEFAULT_ROBOT})")
    parser.add_argument("--tab", type=int, default=None,
                        help=f"initial tab index 0–{len(TABS) - 1} (default: last used)")
    parser.add_argument("--screenshot", default=None,
                        help="render every tab (or only --tab if given with --only) to PNG files in this directory and exit")
    parser.add_argument("--only", action="store_true", help="with --screenshot: only the --tab tab")
    parser.add_argument("--delay", type=int, default=1500, help="ms to wait before each screenshot")
    parser.add_argument("--projector", action="store_true", help="start in projector (presentation) mode")
    parser.add_argument("--no-projector", action="store_true", help="start in normal mode")
    parser.add_argument("--software-gl", action="store_true",
                        help="use Qt's software OpenGL renderer (no/broken graphics driver, remote desktop, VM); "
                             "same as MRSTUDIO_SOFTWARE_GL=1")
    parser.add_argument("--open", dest="open_url", default=None, metavar="URL",
                        help="open a link such as mrstudio://open?tab=MobileTab&robot=…&preset=… "
                             "(tab = class name or 0-based index); overrides --tab")
    parser.add_argument("url", nargs="?", default=None,
                        help="an mrstudio:// link (passed by the operating system when a link is clicked)")
    args = parser.parse_args(argv)
    launch = args.open_url or (args.url if args.url and args.url.lower().startswith("mrstudio:") else None)
    if args.url and not launch:
        parser.error(f"unexpected argument {args.url!r} (only mrstudio:// links are accepted)")
    launch_tab = None
    if launch:
        from ..kmap_links import parse_launch_url
        try:
            launch_tab = parse_launch_url(launch)["index"]
        except ValueError as e:
            print(f"ignoring launch URL {url_for_log(launch)}: {e}", file=sys.stderr)
            launch = None

    shot_mode = bool(args.screenshot)
    if shot_mode:
        use_ephemeral_settings()   # reproducible screenshots: ignore (and never touch) the user's saved settings
    try:
        install_crash_logging()
    except Exception:
        traceback.print_exc()
    software_gl = software_gl_requested(args.software_gl)
    if software_gl:
        use_software_gl()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("MR Studio")
    app.setOrganizationName("MRStudio")
    url_filter = UrlOpenFilter(app)          # macOS: links clicked while (or before) the app runs
    if sys.platform == "darwin":             # QFileOpenEvent is macOS-only; elsewhere links arrive as --open
        app.installEventFilter(url_filter)
    s = app_settings()
    if args.projector:
        projector = True
    elif args.no_projector or shot_mode:
        projector = False
    else:
        projector = _bool(s.value("projector"), False)
    theme.apply(app, projector, light=False if shot_mode else _bool(s.value("projector_light"), False))

    if not opengl_available():
        msg = ("MR Studio could not create an OpenGL context, so the 3-D views may stay blank or the app may "
               "crash.\n\n" + ("The software renderer (--software-gl) is already in use; please update the "
                                 "graphics driver." if software_gl else
                                 "Start MR Studio with --software-gl (or set the environment variable "
                                 "MRSTUDIO_SOFTWARE_GL=1) to use the software renderer, or update the graphics "
                                 "driver."))
        log_text("OpenGL probe failed" + (" (software GL)" if software_gl else ""))
        print(msg, file=sys.stderr)
        if not shot_mode:
            QtWidgets.QMessageBox.warning(None, "MR Studio — OpenGL not available", msg)

    robot = args.robot
    if robot is None:
        last = None if shot_mode else s.value("robot")
        robot = last if last in ROBOTS else DEFAULT_ROBOT
    ctx = AppContext(robot)
    win = MainWindow(ctx)
    win.persist = not shot_mode
    if not shot_mode:
        win.restore_settings(restore_tab=args.tab is None and launch_tab is None)
    win.show()
    if args.tab is not None:
        win.tabs.setCurrentIndex(args.tab)
    if launch:
        win.open_url(launch)
    url_filter.set_window(win)
    if not shot_mode:
        QtCore.QTimer.singleShot(400, win.warm_robot_models)  # after the window has painted

    if args.screenshot:
        os.makedirs(args.screenshot, exist_ok=True)
        tab0 = launch_tab if launch_tab is not None else (args.tab or 0)
        order = [tab0] if args.only else list(range(win.tabs.count()))
        state = {"k": 0}

        def snap():
            i = order[state["k"]]
            img = win.grab()
            words = ["".join(ch for ch in w if ch.isascii() and (ch.isalnum() or ch in "-_"))
                     for w in win.tabs.tabText(i).split()]  # "④ Dynamics" -> "dynamics"
            name = next((w for w in words if w), "x").lower()
            fn = os.path.join(args.screenshot, f"tab{i}_{name}{'_projector' if projector else ''}.png")
            img.save(fn)
            print("saved", fn, flush=True)
            state["k"] += 1
            if state["k"] < len(order):
                win.tabs.setCurrentIndex(order[state["k"]])
                QtCore.QTimer.singleShot(args.delay, snap)
            else:
                app.quit()

        win.tabs.setCurrentIndex(order[0])
        QtCore.QTimer.singleShot(args.delay + 800, snap)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
