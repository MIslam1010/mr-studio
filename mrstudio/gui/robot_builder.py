"""Robot Builder dialog: define a robot with revolute (R) and prismatic (P) joints the Modern Robotics way
(w_i, q_i for R joints, v_i for P joints, and M)."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import kinematics as kin
from ..robots import custom_robot, screw_axis_line, trans
from . import theme
from .theory import TheoryPanel
from .viewer import RobotGraphic, SceneView
from .widgets import button, hbox, spin

PRESETS = {
    "SCARA-like RRR (planar)": dict(
        w=[[0, 0, 1], [0, 0, 1], [0, 0, 1]], q=[[0, 0, 0], [0.45, 0, 0], [0.85, 0, 0]], M=[1.05, 0, 0, 0, 0, 0]),
    "Anthropomorphic RRR (elbow arm)": dict(
        w=[[0, 0, 1], [0, 1, 0], [0, 1, 0]], q=[[0, 0, 0], [0, 0, 0.4], [0.45, 0, 0.4]], M=[0.85, 0, 0.4, 0, 0, 0]),
    "Wrist-partitioned 6R (spherical wrist)": dict(
        w=[[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]],
        q=[[0, 0, 0], [0, 0, 0.45], [0.5, 0, 0.45], [0.5, 0, 0.45], [0.9, 0, 0.45], [0.9, 0, 0.45]],
        M=[1.0, 0, 0.45, 0, 0, 0]),
    "RRRR spatial (non-standard axes)": dict(
        w=[[0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
        q=[[0, 0, 0], [0, 0, 0.35], [0.3, 0, 0.35], [0.6, 0.1, 0.35]], M=[0.8, 0.1, 0.35, 0, 0, 0]),
    "SCARA RRPR (prismatic quill)": dict(
        t="RRPR", w=[[0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1]],
        q=[[0, 0, 0], [0.4, 0, 0], [0.7, 0, 0], [0.7, 0, 0]], M=[0.7, 0, 0, 180, 0, 0]),
    "RRPRRR chain (MR Example 4.4, L1=0.4, L2=0.15)": dict(
        t="RRPRRR", w=[[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]],
        q=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0.4, 0], [0, 0, 0]],
        M=[0, 0.55, 0, 0, 0, 0]),
}

HELP = r"""
<h2>Describe a robot the Modern Robotics way</h2>
<p>For each joint give, <b>at the home configuration</b> and in the fixed frame {s}:</p>
<p>&bull; <b>R</b> (revolute): the unit rotation axis $\hat{\omega}_i$ and any point $q_i$ on that axis;
the screw axis is</p>
$$\mathcal{S}_i=(\hat{\omega}_i,\ -\hat{\omega}_i\times q_i)$$
<p>&bull; <b>P</b> (prismatic): the unit direction of translation $\hat{v}_i$ (entered in the ω columns;
$q_i$ is not needed); the screw axis is $\mathcal{S}_i=(0,\ \hat{v}_i)$ and $\theta_i$ is in metres
(MR Sec. 3.3.2.2, Example 4.4).</p>
<p>Finally give $M$, the pose of the end-effector frame {b} at home (position + roll/pitch/yaw).
That's all: forward kinematics is $T(\theta)=e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M$.</p>
<h3>Dynamics</h3>
<p>Mass properties are generated automatically: each link is modelled as thin rods along its
skeleton (density in kg/m) plus a motor (solid sphere) at its joint. The resulting spatial inertias
$\mathcal{G}_i$ and CoM frames $M_{i-1,i}$ are physically consistent, so every tab — dynamics,
control, validation — works on your robot.</p>
"""


class RobotBuilder(QtWidgets.QDialog):
    """Emits ``robot_created(robot)`` on "Create robot"; the receiver registers it (``robots.register_robot``),
    e.g. ``MainWindow._robot_created``, which also replaces an earlier user robot of the same name."""
    robot_created = QtCore.Signal(object)

    def __init__(self, current_robot=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robot Builder — screw axes (Modern Robotics description)")
        self.resize(1250, 720)
        self.current_robot = current_robot

        self.name = QtWidgets.QLineEdit("My robot")
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(1, 8)
        self.n_spin.setValue(3)
        self.n_spin.valueChanged.connect(self._resize_table)
        self.table = QtWidgets.QTableWidget(3, 7)
        self.table.setHorizontalHeaderLabels(["type R/P", "ωx (P: vx)", "ωy (P: vy)", "ωz (P: vz)",
                                              "qx [m]", "qy [m]", "qz [m]"])
        self.table.horizontalHeaderItem(0).setToolTip("R = revolute joint, P = prismatic joint")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.itemChanged.connect(self._preview)
        self.M_edits = [spin(-5, 5, 0, 3, 0.01) for _ in range(3)] + [spin(-180, 180, 0, 1, 5, "°") for _ in range(3)]
        for s in self.M_edits:
            s.valueChanged.connect(self._preview)
        self.density = spin(0.1, 50, 4.0, 1, 0.5, " kg/m")
        self.motor = spin(0.0, 20, 0.6, 2, 0.1, " kg")
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(["— load a preset —"] + list(PRESETS) + (["Copy current robot"] if current_robot else []))
        self.preset.currentTextChanged.connect(self._load_preset)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Preset", self.preset)
        form.addRow("Number of joints n", self.n_spin)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addLayout(form)
        ll.addWidget(QtWidgets.QLabel("Joint screw axes at home, in {s}:  R: ω̂ᵢ (axis) and qᵢ (point on axis)  ·  "
                                      "P: v̂ᵢ (slide direction)"))
        ll.addWidget(self.table, 1)
        ll.addWidget(QtWidgets.QLabel("Home pose M of the end-effector frame {b}:  x, y, z [m]  ·  roll, pitch, yaw"))
        ll.addLayout(hbox(*self.M_edits))
        ll.addLayout(hbox(QtWidgets.QLabel("Link density"), self.density, QtWidgets.QLabel("Motor mass"), self.motor, None))
        ll.addWidget(self.status)
        ok = button("Create robot", self._create, primary=True)
        ll.addLayout(hbox(None, button("Cancel", self.reject), ok))

        self.view = SceneView()
        self.view.setMinimumWidth(420)
        help_panel = TheoryPanel()
        help_panel.set_content(HELP)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.view)
        right.addWidget(help_panel)
        right.setSizes([420, 300])
        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([620, 600])
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(split)
        self.graphic = None
        self._load_preset("Anthropomorphic RRR (elbow arm)")

    # ------------------------------------------------------------------
    def _resize_table(self, n):
        old = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.setRowCount(n)
        for i in range(old, n):
            for j, v in enumerate(["R", 0, 0, 1, 0, 0, 0]):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{v:g}"))
        self.table.blockSignals(False)
        self._preview()

    def _load_preset(self, name):
        if name == "Copy current robot" and self.current_robot is not None:
            r = self.current_robot
            ws, qs = [], []
            for i in range(r.n):
                q, w, _ = screw_axis_line(r.Slist[:, i])
                ws.append(w)
                qs.append(q)
            rpy = np.degrees(kin.rot_to_rpy(r.M[:3, :3]))
            data = dict(t=r.joint_types, w=ws, q=qs, M=[*r.M[:3, 3], *rpy])
            self.name.setText(r.name + " (copy)")
        elif name in PRESETS:
            data = PRESETS[name]
            self.name.setText(name.split(" (")[0])
        else:
            return
        n = len(data["w"])
        self.table.blockSignals(True)
        self.n_spin.blockSignals(True)
        self.n_spin.setValue(n)
        self.table.setRowCount(n)
        types = data.get("t", "R" * n)
        for i in range(n):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(types[i]))
            for j, v in enumerate([*data["w"][i], *data["q"][i]]):
                self.table.setItem(i, j + 1, QtWidgets.QTableWidgetItem(f"{float(v):.4g}"))
        self.n_spin.blockSignals(False)
        self.table.blockSignals(False)
        for s, v in zip(self.M_edits, data["M"]):
            s.blockSignals(True)
            s.setValue(float(v))
            s.blockSignals(False)
        self._preview()

    def _read(self):
        n = self.table.rowCount()
        W, Q, T = [], [], ""
        for i in range(n):
            it = self.table.item(i, 0)
            t = (it.text().strip().upper()[:1] if it and it.text().strip() else "R")
            if t not in ("R", "P"):
                raise ValueError(f"joint {i + 1}: type must be R (revolute) or P (prismatic)")
            vals = []
            for j in range(1, 7):
                it = self.table.item(i, j)
                vals.append(float(it.text()) if it and it.text().strip() else 0.0)
            w = np.array(vals[:3])
            if np.linalg.norm(w) < 1e-9:
                raise ValueError(f"joint {i + 1}: {'v' if t == 'P' else 'ω'} must be non-zero")
            W.append(w / np.linalg.norm(w))
            Q.append(np.array(vals[3:]))
            T += t
        m = [s.value() for s in self.M_edits]
        M = trans(m[:3], kin.rpy_to_rot(np.radians(m[3:])))
        return W, Q, M, T

    def _build(self):
        W, Q, M, T = self._read()
        return custom_robot(self.name.text().strip() or "My robot", W, Q, M, self.density.value(), self.motor.value(),
                            joint_types=T)

    def _preview(self, *_):
        try:
            robot = self._build()
        except Exception as e:
            self.status.setText(f"<span style='color:{theme.BAD}'>⚠ {e}</span>")
            return
        if self.graphic is not None:
            self.graphic.remove()
        self.view.remove_prefix("rb.")
        self.graphic = RobotGraphic(self.view, robot, "rb.robot")
        self.graphic.show_axes = True
        self.graphic.show_labels = True
        self.graphic.update(robot.home)
        self.view.set_triad("rb.s", np.eye(4), 0.2 * robot.reach, "{s}")
        pts = np.vstack(robot.skeleton)
        self.view.frame_scene(0.8 * robot.reach, planar=robot.planar, center=(pts.min(0) + pts.max(0)) / 2)
        masses = ", ".join(f"{m:.2f}" for m in robot.masses)
        self.status.setText(f"<span style='color:{theme.GOOD}'>✓ {robot.n} joints ({robot.joint_types}) · "
                            f"{'planar (gravity −y)' if robot.planar else 'spatial (gravity −z)'} · "
                            f"reach {robot.reach:.2f} m · link masses [{masses}] kg</span>")
        self._robot = robot

    def _create(self):
        try:
            robot = self._build()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Robot Builder", str(e))
            return
        self.robot_created.emit(robot)
        self.accept()
