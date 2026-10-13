"""Import a serial robot from a URDF file (MR Sec. 4.2 and 8.8) -- see mrstudio/urdf_import.py."""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import urdf_import as U
from ..robots import ROBOTS, register_robot
from . import theme
from .viewer import RobotGraphic, SceneView
from .widgets import button, hbox

HELP = ("A URDF lists, for every joint, its <i>origin</i> (pose of the joint frame relative to the parent "
        "joint frame at θ = 0) and <i>axis</i>, and for every link its <i>inertial</i> element "
        "(CoM frame, mass, rotational inertia). The importer evaluates the chain at θ = 0 and extracts "
        "the screw axes S<sub>i</sub> in {s}, the home pose M of the chosen end-effector link and the "
        "Newton–Euler data M<sub>i-1,i</sub>, G<sub>i</sub> (MR Sec. 4.2, 8.8, Exercise 8.12). "
        "Fixed joints are merged; joints off the base→tip chain are locked at 0. "
        "The result is checked against Pinocchio loaded from the same file.")


class UrdfDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import URDF — serial chain → screw axes (Modern Robotics description)")
        self.resize(1150, 680)
        self.path: str | None = None
        self.robot = None

        self.examples = QtWidgets.QListWidget()
        self._examples = U.example_urdfs()
        self.examples.addItems(list(self._examples))
        self.examples.currentTextChanged.connect(lambda t: self._set_path(self._examples.get(t)))
        self.path_lbl = QtWidgets.QLabel("—")
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setObjectName("muted")
        self.tip = QtWidgets.QComboBox()
        self.tip.currentTextChanged.connect(self._convert)
        self.name = QtWidgets.QLineEdit()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.RichText)
        self.summary.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.summary)
        help_lbl = QtWidgets.QLabel(HELP)
        help_lbl.setWordWrap(True)
        help_lbl.setObjectName("muted")

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(QtWidgets.QLabel("Example URDFs available offline:"))
        ll.addWidget(self.examples, 1)
        ll.addLayout(hbox(button("Browse for a .urdf file…", self._browse), None))
        ll.addWidget(self.path_lbl)
        form = QtWidgets.QFormLayout()
        form.addRow("End-effector link", self.tip)
        form.addRow("Robot name", self.name)
        ll.addLayout(form)
        ll.addWidget(scroll, 2)
        ll.addWidget(help_lbl)
        self.ok = button("Import robot", self._accept, primary=True)
        self.ok.setEnabled(False)
        ll.addLayout(hbox(None, button("Cancel", self.reject), self.ok))

        self.view = SceneView()
        self.view.setMinimumWidth(420)
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.view)
        split.setSizes([560, 590])
        QtWidgets.QVBoxLayout(self).addWidget(split)
        self.graphic = None
        if self._examples:
            self.examples.setCurrentRow(0)

    # ------------------------------------------------------------------
    def _browse(self):
        start = os.path.dirname(self.path) if self.path else os.path.expanduser("~")
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open URDF", start, "URDF files (*.urdf *.xml);;All files (*)")
        if fn:
            self.examples.blockSignals(True)
            self.examples.clearSelection()
            self.examples.blockSignals(False)
            self._set_path(fn)

    def _set_path(self, path):
        if not path:
            return
        self.path = path
        self.path_lbl.setText(path)
        try:
            links = U.link_names(path)
            leaves = U.leaf_links(path)
        except Exception as e:
            self._error(f"Cannot parse the URDF: {e}")
            self.tip.clear()
            return
        self.tip.blockSignals(True)
        self.tip.clear()
        self.tip.addItems(links[1:] or links)
        if leaves:
            self.tip.setCurrentText(leaves[0])
        self.tip.blockSignals(False)
        self.name.setText("")
        self._convert()

    def _error(self, msg):
        self.robot = None
        self.ok.setEnabled(False)
        self.summary.setText(f"<span style='color:{theme.BAD}'>⚠ {msg}</span>")

    def _convert(self, *_):
        if not self.path or not self.tip.currentText():
            return
        try:
            r = U.from_urdf(self.path, tip=self.tip.currentText())
            err = U.compare_with_pinocchio(r, self.path, n_samples=4)
        except Exception as e:
            self._error(str(e))
            return
        if not self.name.text().strip():
            base = f"{r.name} (URDF)"
            nm, k = base, 2
            while nm in ROBOTS:
                nm, k = f"{base} {k}", k + 1
            self.name.setText(nm)
        self.robot = r
        worst = max(err.values())
        col = theme.GOOD if worst < 1e-9 else theme.BAD
        self.summary.setText(U.summary(r) + f"<p style='color:{col}'>{'✓' if worst < 1e-9 else '⚠'} vs Pinocchio "
                             f"(same URDF): FK {err['fk']:.1e}, RNEA {err['rnea']:.1e}, "
                             f"mass matrix {err['mass']:.1e}</p>")
        self.ok.setEnabled(True)
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("ud.")
        self.graphic = RobotGraphic(self.view, r, "ud.robot")
        self.graphic.show_axes = True
        self.graphic.show_labels = True
        self.graphic.update(r.home)
        self.view.set_triad("ud.s", np.eye(4), 0.2 * r.reach, "{s}")
        pts = np.vstack(r.skeleton)
        self.view.frame_scene(0.8 * r.reach, planar=False, center=(pts.min(0) + pts.max(0)) / 2)

    def _accept(self):
        if self.robot is None:
            return
        self.robot.name = self.name.text().strip() or self.robot.name
        self.accept()


def open_urdf_dialog(parent=None):
    """Modal URDF import dialog. Returns the imported (and registered) SerialRobot, or None."""
    dlg = UrdfDialog(parent)
    if dlg.exec() != QtWidgets.QDialog.Accepted or dlg.robot is None:
        return None
    register_robot(dlg.robot)
    return dlg.robot
