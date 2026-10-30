"""Application context and the base class every tab derives from."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..dynamics import DynamicsEngine
from ..robots import ROBOTS, SerialRobot, get_robot, is_current
from . import theme
from .theory import TheoryPanel


class AppContext(QtCore.QObject):
    """Shared state: the currently selected serial robot and the dynamics back-end."""
    robot_changed = QtCore.Signal(object)
    backend_changed = QtCore.Signal(str)
    theory_visible_changed = QtCore.Signal(bool)
    #: projector / presentation mode toggled (fonts, line widths, contrast). TheoryPanels, SceneViews
    #: and BaseTab plots already react to it; tabs may override ``BaseTab.on_projector_changed``.
    projector_changed = QtCore.Signal(bool)

    def __init__(self, robot_name: str = "UR5 (6R industrial)"):
        super().__init__()
        self.robot: SerialRobot = get_robot(robot_name)
        self.backend = "pinocchio"
        self.theory_visible = True
        # (id(robot), backend) -> (robot, engine): keyed by object identity, so a re-created robot with
        # the same name (Robot Builder / URDF import) gets a fresh engine; the robot is kept alive in the
        # value so its id cannot be reused while the entry exists.
        self._engines: dict = {}
        theme.notifier().projector_changed.connect(self.projector_changed)

    @property
    def projector(self) -> bool:
        return theme.is_projector()

    def set_projector(self, on: bool):
        """Switch projector mode app-wide (emits ``projector_changed`` if it changed)."""
        theme.set_projector(QtWidgets.QApplication.instance(), on)

    def set_robot(self, name: str, force: bool = False):
        """Select the robot registered as ``name``. Emits ``robot_changed`` when that is a different
        object than the current robot (also when a robot was re-registered under the same name) or
        when ``force`` is set."""
        if name not in ROBOTS:
            return
        robot = get_robot(name)
        if robot is self.robot and not force:
            return
        old, self.robot = self.robot, robot
        if old is not robot and not is_current(old):  # replaced by a same-name robot: drop its engines
            self._drop_engines(old)
        self.robot_changed.emit(self.robot)

    def _drop_engines(self, robot: SerialRobot):
        for key in [k for k in self._engines if k[0] == id(robot)]:
            del self._engines[key]

    def set_backend(self, backend: str):
        if backend != self.backend:
            self.backend = backend
            self.backend_changed.emit(backend)

    def engine(self, robot: SerialRobot | None = None, backend: str | None = None) -> DynamicsEngine:
        robot = self.robot if robot is None else robot
        backend = self.backend if backend is None else backend
        key = (id(robot), backend)
        hit = self._engines.get(key)
        if hit is None or hit[0] is not robot:
            hit = self._engines[key] = (robot, DynamicsEngine(robot, backend))
        return hit[1]


class BaseTab(QtWidgets.QWidget):
    """A tab = left control column | centre (3-D view over plots) | right theory panel.

    Subclasses build their widgets in __init__ and call ``self.assemble(...)``.
    Override ``on_robot_changed(robot)`` (serial-robot tabs), ``on_backend_changed(backend)`` and
    ``on_show()/on_hide()`` to start/stop timers, and ``shutdown()`` to stop threads / hardware when the
    application quits.

    Robot / engine changes reach a *hidden* tab lazily: while the tab sits in a hidden page of the main
    window it is only marked stale, and ``on_robot_changed`` / ``on_backend_changed`` run when it is shown
    again (so hidden tabs never rebuild scenes or start jobs/animations). A tab that is not inside a
    window at all (e.g. in tests) is updated immediately.
    """
    title = "Tab"
    uses_serial_robot = True
    #: whether the top bar's Engine selector matters for this tab (None: decided by the main window)
    uses_engine: bool | None = None

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._stale_robot = False
        self._stale_backend = False
        self.theory = TheoryPanel()
        ctx.theory_visible_changed.connect(self.theory.setVisible)
        if self.uses_serial_robot:
            ctx.robot_changed.connect(self._robot_changed)
        ctx.backend_changed.connect(self._ctx_backend_changed)
        ctx.projector_changed.connect(self._projector_changed)

    def assemble(self, controls: QtWidgets.QWidget, center: QtWidgets.QWidget,
                 bottom: QtWidgets.QWidget | None = None, theory_html: str = "",
                 controls_width: int = 340, bottom_ratio=(3, 2)):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._controls_scroll, self._controls_width = scroll, controls_width
        self._apply_controls_width()
        if bottom is not None:
            vs = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            vs.addWidget(center)
            vs.addWidget(bottom)
            vs.setStretchFactor(0, bottom_ratio[0])
            vs.setStretchFactor(1, bottom_ratio[1])
            mid = vs
        else:
            mid = center
        hs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hs.addWidget(scroll)
        hs.addWidget(mid)
        hs.addWidget(self.theory)
        hs.setStretchFactor(0, 0)
        hs.setStretchFactor(1, 1)
        hs.setStretchFactor(2, 0)
        hs.setSizes([controls_width, 900, int(380 * theme.font_scale())])
        self._hsplit = hs
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(hs)
        self.theory.set_content(theory_html)
        self.theory.setVisible(self.ctx.theory_visible)

    # ------------------------------------------------------------------ lazy robot / engine updates
    def _defer(self) -> bool:
        """True while this tab is a hidden page of a window (update it when it is shown instead)."""
        return not self.isVisible() and self.parentWidget() is not None

    def _robot_changed(self, robot):
        if self._defer():
            self._stale_robot = True
            return
        self._stale_robot = False
        self.on_robot_changed(robot)
        if theme.is_projector():
            QtCore.QTimer.singleShot(0, self.restyle_plots)

    def _ctx_backend_changed(self, backend):
        if self._defer():
            self._stale_backend = True
            return
        self._stale_backend = False
        self.on_backend_changed(backend)

    def refresh_if_stale(self):
        """Apply robot / engine changes that happened while the tab was hidden (idempotent)."""
        if self._stale_robot:
            self._stale_robot = False
            self.on_robot_changed(self.ctx.robot)
            if theme.is_projector():
                QtCore.QTimer.singleShot(0, self.restyle_plots)
        if self._stale_backend:
            self._stale_backend = False
            self.on_backend_changed(self.ctx.backend)

    def is_stale(self) -> bool:
        return self._stale_robot or self._stale_backend

    def showEvent(self, e):
        super().showEvent(e)
        if self.is_stale():
            self.refresh_if_stale()

    def _apply_controls_width(self):
        sc = getattr(self, "_controls_scroll", None)
        if sc is None:
            return
        w = int(self._controls_width * theme.font_scale())
        sc.setMinimumWidth(w)
        sc.setMaximumWidth(w + int(120 * theme.font_scale()))

    def _projector_changed(self, on):
        self._apply_controls_width()
        self._apply_theory_width()
        self.restyle_plots()
        self.on_projector_changed(on)

    def _apply_theory_width(self):
        """Give the theory panel a width proportional to the font scale (equations grow in projector mode)."""
        hs = getattr(self, "_hsplit", None)
        if hs is None or hs.count() < 3:
            return
        sizes = hs.sizes()
        want = int(380 * theme.font_scale())
        if sum(sizes) <= 0 or self.theory.isHidden():
            return
        sizes[1] = max(200, sizes[1] + sizes[2] - want)
        sizes[2] = want
        hs.setSizes(sizes)

    def restyle_plots(self):
        """Apply the current (projector / normal) style to every pyqtgraph plot in this tab."""
        try:
            theme.style_plots_in(self)
        except RuntimeError:
            pass

    # hooks -------------------------------------------------------------------
    def on_robot_changed(self, robot: SerialRobot):
        pass

    def on_backend_changed(self, backend: str):
        """Optional hook: the dynamics engine changed (deferred while the tab is hidden)."""
        pass

    def on_projector_changed(self, on: bool):
        """Optional hook: projector mode toggled (plots/theory/3-D are restyled automatically)."""
        pass

    def on_show(self):
        pass

    def on_hide(self):
        pass

    def shutdown(self):
        """Optional hook: the application is quitting -- stop worker threads, timers, hardware."""
        pass
