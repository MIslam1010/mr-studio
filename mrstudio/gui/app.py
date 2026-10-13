"""Main window of MR Studio."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

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
    ("mrstudio.gui.tabs.hardware_tab", "HardwareTab"),
    ("mrstudio.gui.tabs.practice_tab", "PracticeTab"),
    ("mrstudio.gui.tabs.validation_tab", "ValidationTab"),
]

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


def app_settings() -> QtCore.QSettings:
    """Persistent settings (``MRSTUDIO_SETTINGS=/path/file.ini`` overrides the location, e.g. for tests)."""
    path = os.environ.get("MRSTUDIO_SETTINGS")
    if path:
        return QtCore.QSettings(path, QtCore.QSettings.IniFormat)
    return QtCore.QSettings("MRStudio", "MR Studio")


def _bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


class ErrorTab(QtWidgets.QWidget):
    def __init__(self, name, err):
        super().__init__()
        self.title = name.split(".")[-1]
        lay = QtWidgets.QVBoxLayout(self)
        t = QtWidgets.QPlainTextEdit(err)
        t.setReadOnly(True)
        lay.addWidget(t)

    def on_show(self):
        pass

    def on_hide(self):
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
    return ErrorTab(mod, err), "⚠ " + mod.split(".")[-1]


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
        self.rec_source = "view"
        self.rec_fps = 30

        self._make_actions()
        self._make_menus()
        bar = self._make_top_bar()

        # ---- tabs -----------------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.pages = []
        for mod, cls in TABS:
            res = load_tab(mod, cls, ctx)
            if res is None:
                continue
            page, label = res
            self.pages.append(page)
            self.tabs.addTab(page, label)
        self._current = None
        self.tabs.currentChanged.connect(self._tab_changed)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(bar)
        v.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Core: modern_robotics (Lynch & Park) · Pinocchio (LAAS-CNRS) · Robotics Toolbox (P. Corke) · "
            "TOPP-RA · SciPy — all computations validated against each other (Validation tab).")
        ctx.projector_changed.connect(self._projector_changed)
        self._projector_changed(theme.is_projector(), save=False)
        if theme.is_projector():
            QtCore.QTimer.singleShot(0, lambda: theme.style_plots_in(self))
        QtCore.QTimer.singleShot(0, lambda: self._tab_changed(self.tabs.currentIndex()))

    # ------------------------------------------------------------------ actions / menus / top bar
    def _act(self, text, slot=None, shortcut=None, checkable=False, tip=None):
        a = QtGui.QAction(text, self)
        if shortcut:
            seqs = shortcut if isinstance(shortcut, (list, tuple)) else [shortcut]
            a.setShortcuts([QtGui.QKeySequence(s) for s in seqs])
            a.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        if slot is not None:
            (a.toggled if checkable else a.triggered).connect(slot)
        self.addAction(a)  # shortcuts work even without a visible menu bar
        return a

    def _make_actions(self):
        self.act_builder = self._act("＋ Robot Builder…", self._open_builder, "Ctrl+B",
                                     tip="Define your own robot by its screw axes (ω_i, q_i) and home pose M")
        self.act_urdf = self._act("Import URDF…", self._import_urdf, "Ctrl+O",
                                  tip="Load a serial robot from a URDF file (⌘O)")
        self.act_shot = self._act("Save screenshot (PNG)…", self._save_screenshot, "Ctrl+Shift+S",
                                  tip="Save the whole window as a PNG (⌘⇧S)")
        self.act_shot_view = self._act("Save 3-D view image (PNG)…", self._save_view_screenshot, "Ctrl+Alt+S",
                                       tip="Save only the current tab's 3-D view (with labels) as a PNG (⌘⌥S)")
        self.act_record = self._act("● Record", self._toggle_record, "Ctrl+R", checkable=True,
                                    tip="Start/stop recording a clip (⌘R); saved as MP4 or GIF. "
                                        "Arrow: choose source / frame rate.")
        self.act_quit = self._act("Quit", self.close, QtGui.QKeySequence.Quit)
        self.act_quit.setMenuRole(QtGui.QAction.QuitRole)
        self.act_theory = self._act("Theory panel", self._toggle_theory, "F1", checkable=True,
                                    tip="Show/hide the equations panel (F1)")
        self.act_theory.setChecked(self.ctx.theory_visible)
        self.act_projector = self._act("Projector mode", self.ctx.set_projector, ["Ctrl+Shift+P", "F5"],
                                       checkable=True,
                                       tip="Large fonts, thick lines, high contrast for presenting (⌘⇧P or F5)")
        self.act_projector.setChecked(theme.is_projector())
        self.act_about = self._act("About MR Studio / libraries…", self._about)
        self.act_about.setMenuRole(QtGui.QAction.AboutRole)
        self.act_refs = self._act("References…", self._references,
                                  tip="Bibliography of the theory panels' “Further reading” (with local PDFs)")

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
        v = mb.addMenu("&View")
        v.addAction(self.act_theory)
        v.addAction(self.act_projector)
        h = mb.addMenu("&Help")
        h.addAction(self.act_refs)
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
        self.engine_box.setToolTip("Both engines implement the same equations and agree to ~1e-15 "
                                   "(see Validation tab). Pinocchio is ~3000x faster for simulation.")
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

    def _import_urdf(self):
        try:
            from mrstudio.gui.urdf_dialog import open_urdf_dialog
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import URDF",
                                          f"URDF import is not available:\n{type(e).__name__}: {e}")
            return
        try:
            robot = open_urdf_dialog(self)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Import URDF", f"Import failed:\n{type(e).__name__}: {e}")
            return
        if robot is not None:
            self._robot_created(robot)

    def _robot_created(self, robot):
        if robot.name not in ROBOTS:
            register_robot(robot)
        if self.robot_box.findText(robot.name) < 0:
            self.robot_box.addItem(robot.name)
        self.robot_box.setCurrentText(robot.name)
        self.ctx.set_robot(robot.name)

    # ------------------------------------------------------------------ view
    def _toggle_theory(self, on):
        self.ctx.theory_visible = on
        self.ctx.theory_visible_changed.emit(on)

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
        self._current = self.pages[idx]
        try:
            self._current.on_show()
        except Exception:
            traceback.print_exc()
        if theme.is_projector():  # catch curves created since the last toggle
            QtCore.QTimer.singleShot(0, lambda p=self._current: theme.style_plots_in(p))

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
            "<h3>Shortcuts</h3><p>⌘R record · ⌘⇧S screenshot · ⌘⌥S 3-D view PNG · ⌘⇧P / F5 projector mode · "
            "F1 theory panel · ⌘B robot builder · ⌘O import URDF</p>")

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
        source = source or self.rec_source
        if source == "view":
            view = self.current_view()
            if view is not None:
                return view.grabFramebuffer
            self.statusBar().showMessage("This tab has no 3-D view — recording the whole window.", 5000)
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
        try:
            self.recorder.start(self.grab_function(source), fps or self.rec_fps)
        except Exception as e:
            self._set_record_checked(False)
            QtWidgets.QMessageBox.warning(self, "Record", f"Cannot start recording:\n{e}\n\n"
                                          "Install: pip install imageio imageio-ffmpeg pillow")
            return
        self.rec_label.setText("● REC 00:00")
        self.rec_label.setVisible(True)
        self._set_record_checked(True)

    def stop_recording(self, ask=True):
        """Stop. ask=False: return the temp MP4 path; ask=True: show the save dialog and export async."""
        if not self.recorder.recording:
            return None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            tmp = self.recorder.stop()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.rec_label.setVisible(False)
        self._set_record_checked(False)
        if tmp is None or not ask:
            return tmp
        fn = self._ask_path("Save recording", ".mp4", "MP4 video (H.264) (*.mp4);;Animated GIF (*.gif)")
        if not fn:
            os.remove(tmp)
            self.statusBar().showMessage("Recording discarded.", 4000)
            return None
        self.statusBar().showMessage(f"Encoding {os.path.basename(fn)} …")
        self.recorder.export(tmp, fn)
        return None

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
        s.setValue("rec_source", self.rec_source)
        s.setValue("rec_fps", self.rec_fps)
        s.sync()

    def closeEvent(self, e):
        if self.recorder.recording:
            self.stop_recording()
        if self.persist:
            self.save_settings()
        super().closeEvent(e)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MR Studio")
    parser.add_argument("--robot", default=None, help=f"initial robot (default: last used, or {DEFAULT_ROBOT})")
    parser.add_argument("--tab", type=int, default=None, help="initial tab index (default: last used)")
    parser.add_argument("--screenshot", default=None,
                        help="render every tab (or only --tab if given with --only) to PNG files in this directory and exit")
    parser.add_argument("--only", action="store_true", help="with --screenshot: only the --tab tab")
    parser.add_argument("--delay", type=int, default=1500, help="ms to wait before each screenshot")
    parser.add_argument("--projector", action="store_true", help="start in projector (presentation) mode")
    parser.add_argument("--no-projector", action="store_true", help="start in normal mode")
    args = parser.parse_args(argv)

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("MR Studio")
    app.setOrganizationName("MRStudio")
    shot_mode = bool(args.screenshot)
    s = app_settings()
    if args.projector:
        projector = True
    elif args.no_projector or shot_mode:
        projector = False
    else:
        projector = _bool(s.value("projector"), False)
    theme.apply(app, projector)

    robot = args.robot
    if robot is None:
        last = None if shot_mode else s.value("robot")
        robot = last if last in ROBOTS else DEFAULT_ROBOT
    ctx = AppContext(robot)
    win = MainWindow(ctx)
    win.persist = not shot_mode
    if not shot_mode:
        win.restore_settings(restore_tab=args.tab is None)
    win.show()
    if args.tab is not None:
        win.tabs.setCurrentIndex(args.tab)

    if args.screenshot:
        os.makedirs(args.screenshot, exist_ok=True)
        tab0 = args.tab or 0
        order = [tab0] if args.only else list(range(win.tabs.count()))
        state = {"k": 0}

        def snap():
            i = order[state["k"]]
            img = win.grab()
            name = win.tabs.tabText(i).split()[0].strip("⚠").lower() or "x"
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
