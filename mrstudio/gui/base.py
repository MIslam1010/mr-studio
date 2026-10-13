"""Application context and the base class every tab derives from."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..dynamics import DynamicsEngine
from ..robots import ROBOTS, SerialRobot, get_robot
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
        self._engines: dict = {}
        theme.notifier().projector_changed.connect(self.projector_changed)

    @property
    def projector(self) -> bool:
        return theme.is_projector()

    def set_projector(self, on: bool):
        """Switch projector mode app-wide (emits ``projector_changed`` if it changed)."""
        theme.set_projector(QtWidgets.QApplication.instance(), on)

    def set_robot(self, name: str):
        if name in ROBOTS and name != self.robot.name:
            self.robot = get_robot(name)
            self.robot_changed.emit(self.robot)

    def set_backend(self, backend: str):
        if backend != self.backend:
            self.backend = backend
            self.backend_changed.emit(backend)

    def engine(self, robot: SerialRobot | None = None, backend: str | None = None) -> DynamicsEngine:
        robot = self.robot if robot is None else robot
        backend = self.backend if backend is None else backend
        key = (robot.name, backend)
        if key not in self._engines:
            self._engines[key] = DynamicsEngine(robot, backend)
        return self._engines[key]


class BaseTab(QtWidgets.QWidget):
    """A tab = left control column | centre (3-D view over plots) | right theory panel.

    Subclasses build their widgets in __init__ and call ``self.assemble(...)``.
    Override ``on_robot_changed(robot)`` (serial-robot tabs) and ``on_show()/on_hide()``
    to start/stop timers.  ``needs_dynamics = True`` greys the tab out for robots without inertias.
    """
    title = "Tab"
    uses_serial_robot = True
    needs_dynamics = False

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.theory = TheoryPanel()
        ctx.theory_visible_changed.connect(self.theory.setVisible)
        if self.uses_serial_robot:
            ctx.robot_changed.connect(self._robot_changed)
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
        hs.setSizes([controls_width, 900, 380])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(hs)
        self.theory.set_content(theory_html)
        self.theory.setVisible(self.ctx.theory_visible)

    def _robot_changed(self, robot):
        self.on_robot_changed(robot)
        if theme.is_projector():
            QtCore.QTimer.singleShot(0, self.restyle_plots)

    def _apply_controls_width(self):
        sc = getattr(self, "_controls_scroll", None)
        if sc is None:
            return
        w = int(self._controls_width * theme.font_scale())
        sc.setMinimumWidth(w)
        sc.setMaximumWidth(w + int(120 * theme.font_scale()))

    def _projector_changed(self, on):
        self._apply_controls_width()
        self.restyle_plots()
        self.on_projector_changed(on)

    def restyle_plots(self):
        """Apply the current (projector / normal) style to every pyqtgraph plot in this tab."""
        try:
            theme.style_plots_in(self)
        except RuntimeError:
            pass

    # hooks -------------------------------------------------------------------
    def on_robot_changed(self, robot: SerialRobot):
        pass

    def on_projector_changed(self, on: bool):
        """Optional hook: projector mode toggled (plots/theory/3-D are restyled automatically)."""
        pass

    def on_show(self):
        pass

    def on_hide(self):
        pass
