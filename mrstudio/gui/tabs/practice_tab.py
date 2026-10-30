"""Practice problems -- randomised exercises with worked solutions (MR Ch. 3-11), quiz export."""
from __future__ import annotations

import os
import random

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ... import problems as P
from ...references import further_reading
from .. import theme
from ..base import BaseTab
from ..theory import TheoryPanel
from ..viewer import GhostRobot, RobotGraphic, SceneView
from ..widgets import button, checkbox, combo, group, hbox, pen, plot, value_label

THEORY = r"""
<h2>Practice problems</h2>
<span class="ref">Modern Robotics, Ch. 3–11 · printed pages of the updated first edition (2019)</span>
<p>Each problem is generated from a <b>seed</b>: the same topic, difficulty and seed always give the same
numbers, so a seed is all a student needs to report.  Robots are drawn from small families (planar 2R/3R,
RRP SCARA-type, spatial 3R elbow, 6R UR-type) with round link lengths and angles in multiples of 15° or
30°.  Every answer is computed with Lynch &amp; Park's <b>modern_robotics</b> code and re-solved independently
(SciPy exponentials, finite differences, closed-form Lagrangian, …); see the Validation tab.</p>
<h3>Topics</h3>
<p><b>Ch. 3</b> Rodrigues' formula (3.51) p. 82; matrix log (3.58)–(3.61) pp. 85–86; screw exponential and
log (3.86)–(3.92) pp. 103–104:</p>
$$e^{[\hat{\omega}]\theta} = I + \sin\theta[\hat{\omega}] + (1-\cos\theta)[\hat{\omega}]^{2}$$
<p><b>Ch. 4</b> PoE, space (4.14) p. 140 and body (4.16) p. 147:</p>
$$T(\theta) = e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M = Me^{[\mathcal{B}_1]\theta_1}\cdots e^{[\mathcal{B}_n]\theta_n}$$
<p><b>Ch. 5</b> $J_s$ (5.11) p. 178, $J_b$ (5.18) p. 183, $J_b = [\mathrm{Ad}_{T_{bs}}]J_s$ (5.22) p. 185;
singularities Sec. 5.3 p. 189.</p>
<p><b>Ch. 6</b> Newton–Raphson with the body Jacobian, Sec. 6.2.2 p. 228 (Example 6.1, pp. 229–230):</p>
$$[\mathcal{V}_b] = \log\left(T_{sb}^{-1}(\theta^{i})T_{sd}\right),\quad \theta^{i+1} = \theta^{i} + J_b^{\dagger}(\theta^{i})\mathcal{V}_b$$
<p><b>Ch. 8</b> 2R Lagrangian dynamics (8.9)–(8.10) p. 273; Newton–Euler (8.50)–(8.55) pp. 292–293.</p>
<p><b>Ch. 9</b> cubic time scaling (9.9)–(9.13) pp. 331–332; quintic p. 332.</p>
<p><b>Ch. 11</b> second-order error dynamics (11.8) p. 412, overshoot and settling time pp. 414–415,
PD control (11.26) p. 425, computed torque (11.33)–(11.35) p. 431:</p>
$$\ddot{\theta}_e + 2\zeta\omega_n\dot{\theta}_e + \omega_n^{2}\theta_e = 0$$
<p>Settling-time answers give both MR's rule of thumb $t_s\approx 4/(\zeta\omega_n)$ (from the envelope
$e^{-\zeta\omega_n t}$) and the exact 2 % settling time; gain-design problems meet the exact value.</p>
<h3>Quizzes</h3>
<p><b>Export</b> writes N problems from the starting seed.  With <i>Mixed</i> topics question k is always on the
same topic, so versions A, B, … of a quiz have the same structure and different numbers.  The Word file
(via pandoc) contains native equations and the full worked solutions.</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 2 §3.2–3.3, pp. 39–50', 'twists, exponential coordinates, screws'),
    ('mls', 'Ch. 3 §4, pp. 115–129', 'spatial/body Jacobian, singularities'),
    ('corke', '§7.3, pp. 146–152; §8.4, pp. 187–188', 'closed-form and numerical IK'),
    ('corke', '§3.1, pp. 43–51; §7.4, pp. 152–158', 'quintic and LSPB scalings'),
    ('featherstone', '§5.3, pp. 92–97, Table 5.1 p. 96', 'recursive Newton–Euler algorithm'),
    ('astrom_murray', '§6.3, pp. 183–184', 'second-order systems: damping ratio, natural frequency'),
])

PLACEHOLDER = ("<h2>Worked solution</h2><p style='color:gray'>Try the problem first, then press "
               "<b>Reveal solution</b>.</p>")


class PracticeTab(BaseTab):
    title = "⑬ Practice"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.problem: P.Problem | None = None
        self._graphic = None
        self._ghost = None
        self._worker = None

        # ---------------------------------------------------------------- controls
        self.topic = combo(["Mixed (topic from seed)"] + [P.TOPICS[k]["title"] for k in P.TOPIC_KEYS])
        self.topic.setToolTip("Mixed: the topic is seed mod 7 (rotation, FK, Jacobian, IK, dynamics, trajectory, control)")
        self.diff = combo([f"{k} — {v}" for k, v in P.DIFFICULTIES.items()])
        self.diff.setCurrentIndex(1)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(1)
        self.seed.setKeyboardTracking(False)
        self.seed.setToolTip("Same topic + difficulty + seed → same problem (share the seed with students)")
        form = QtWidgets.QFormLayout()
        form.addRow("Topic", self.topic)
        form.addRow("Difficulty", self.diff)
        form.addRow("Seed", self.seed)
        self.gen_btn = button("Generate", self.generate, primary=True)
        self.rand_btn = button("🎲 New seed", self.new_seed, tooltip="Pick a random seed and generate")
        self.next_btn = button("Next ▸", lambda: self.seed.setValue(self.seed.value() + 1),
                               tooltip="Seed + 1")
        self.reveal_btn = button("Reveal solution", self.toggle_reveal, checkable=True)
        self.verify_lbl = value_label()
        self.info_lbl = value_label()
        self.info_lbl.setObjectName("muted")

        for w in (self.topic, self.diff):
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(12)
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(1, 60)
        self.n_spin.setValue(7)
        self.v_spin = QtWidgets.QSpinBox()
        self.v_spin.setRange(1, 8)
        self.v_spin.setValue(1)
        self.v_spin.setToolTip("Quiz versions A, B, … (same topics per question, different numbers)")
        ef = QtWidgets.QFormLayout()
        ef.addRow("Problems N", self.n_spin)
        ef.addRow("Versions", self.v_spin)
        self.exp_md_btn = button("Problem set (Markdown)…", self._export_md_dialog)
        self.exp_doc_btn = button("Problem set + solutions (Word)…", self._export_doc_dialog)
        self.exp_key_cb = checkbox("Also write a Markdown answer key", None, False)
        self.exp_lbl = value_label()
        self.exp_lbl.setObjectName("muted")
        theme.notifier().projector_changed.connect(self._retone)
        pandoc = P.find_pandoc()
        self.exp_doc_btn.setToolTip(f"pandoc: {pandoc}" if pandoc else
                                    "pandoc not found — a Markdown file with solutions is written instead")

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Problem", form, hbox(self.gen_btn, self.rand_btn), hbox(self.next_btn, None),
                           self.info_lbl))
        cl.addWidget(group("Solution", self.reveal_btn, self.verify_lbl))
        note = value_label("Uses the topic, difficulty and start seed above.")
        note.setObjectName("muted")
        cl.addWidget(group("Export problem set", note, ef, self.exp_md_btn, self.exp_key_cb, self.exp_doc_btn,
                           self.exp_lbl))
        cl.addStretch(1)

        # ---------------------------------------------------------------- centre
        self.statement = TheoryPanel()
        self.statement.setMinimumWidth(360)
        self.view = SceneView()
        self.plotw = plot("", "", "t [s]")
        self.visual = QtWidgets.QStackedWidget()
        self.visual.addWidget(self.view)
        self.visual.addWidget(self.plotw)
        self.visual.setMinimumWidth(260)
        center = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        center.addWidget(self.statement)
        center.addWidget(self.visual)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        self.solution = TheoryPanel()
        self.solution.set_content(PLACEHOLDER)
        self.assemble(controls, center, self.solution, THEORY, bottom_ratio=(3, 2))

        self.topic.currentIndexChanged.connect(lambda _i: self.generate())
        self.diff.currentIndexChanged.connect(lambda _i: self.generate())
        self.seed.valueChanged.connect(lambda _v: self.generate())
        self.generate()

    # ------------------------------------------------------------------ state

    def _tone(self, lbl, tone):
        """Colour a label with the current palette's good/warn/bad colour (None = default)."""
        lbl.setProperty("tone", tone)
        lbl.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")

    def _retone(self, *_):
        for lbl in (self.exp_lbl,):
            self._tone(lbl, lbl.property("tone"))

    def topic_key(self) -> str:
        i = self.topic.currentIndex()
        return P.MIXED if i == 0 else P.TOPIC_KEYS[i - 1]

    def difficulty(self) -> int:
        return self.diff.currentIndex() + 1

    def new_seed(self):
        self.seed.setValue(random.randint(0, 999_999))   # valueChanged -> generate

    def generate(self, *_):
        p = P.generate(self.topic_key(), self.seed.value(), self.difficulty())
        self.problem = p
        self.statement.set_content(p.statement_html())
        self.reveal_btn.blockSignals(True)
        self.reveal_btn.setChecked(False)
        self.reveal_btn.setText("Reveal solution")
        self.reveal_btn.blockSignals(False)
        self.solution.set_content(PLACEHOLDER)
        self.verify_lbl.setText("")
        self.info_lbl.setText(f"{P.TOPICS[p.topic]['title']}\nkey: {p.key}")
        self._draw_visual(p)
        return p

    def toggle_reveal(self, on):
        self.reveal_btn.setText("Hide solution" if on else "Reveal solution")
        if self.problem is None:
            return
        if on:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.solution.set_content(self.problem.solution_html())
                err = P.verify(self.problem)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            ok = err < 1e-6
            self.verify_lbl.setStyleSheet(f"color:{theme.GOOD if ok else theme.BAD}")
            self.verify_lbl.setText(("✓" if ok else "✗") + f" answer re-solved independently: max error {err:.1e}")
        else:
            self.solution.set_content(PLACEHOLDER)
            self.verify_lbl.setText("")

    # ------------------------------------------------------------------ visual
    def _clear_scene(self):
        if self._graphic is not None:
            self._graphic.remove()
            self._graphic = None
        if self._ghost is not None:
            self._ghost.remove()
            self._ghost = None
        self.view.remove_prefix("pp.")

    def _draw_visual(self, p: P.Problem):
        vis = p.visual or {}
        self._clear_scene()
        if vis.get("kind") == "plot":
            self.visual.setCurrentWidget(self.plotw)
            w = self.plotw
            w.clear()
            w.setTitle(vis["title"], color=theme.TEXT, size="10pt")
            w.setLabel("left", vis["ylabel"])
            for i, (name, y) in enumerate(vis["series"]):
                w.plot(vis["t"], y, pen=pen(i), name=name)
            if "marker" in vis:
                t1, ys = vis["marker"]
                w.plot([t1] * len(ys), list(ys), pen=None, symbol="o", symbolBrush=theme.WARN, symbolSize=9,
                       name=f"t = {t1:.3g} s")
            if "band" in vis:
                b = vis["band"]
                for yy in (b, -b):
                    w.addItem(pg.InfiniteLine(pos=yy, angle=0, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DashLine)))
            return
        self.visual.setCurrentWidget(self.view)
        if vis.get("kind") == "robot":
            spec = vis["spec"]
            robot = spec.serial_robot()
            prism = robot.is_prismatic
            if prism.any():
                robot.joint_limits[prism] = [0.0, 0.3]
            self._graphic = RobotGraphic(self.view, robot, "pp.robot")
            self._graphic.show_axes = True
            self._graphic.update(vis["theta"])
            if vis.get("ghost") is not None:
                self._ghost = GhostRobot(self.view, robot, "pp.ghost")
                self._ghost.update(vis["ghost"])
            reach = robot.reach
            self.view.set_triad("pp.s", np.eye(4), 0.25 * reach, "{s}")
            for name, T in vis.get("frames", {}).items():
                self.view.set_triad(f"pp.f.{name}", T, 0.2 * reach, "{" + name + "}")
            self.view.frame_scene(reach, planar=robot.planar,
                                  center=(0, 0, 0.25 * reach * (not robot.planar)))
        elif vis.get("kind") == "frames":
            self.view.set_triad("pp.s", np.eye(4), 0.6, "{s}")
            fr = vis.get("frames", {})
            if "R" in fr:
                T = np.eye(4)
                T[:3, :3] = fr["R"]
                self.view.set_triad("pp.R", T, 0.75, "R")
            if "T" in fr:
                self.view.set_triad("pp.T", fr["T"], 0.5, "T")
            w = np.asarray(vis["axis"], float)
            self.view.set_arrow("pp.axis", -0.9 * w, 1.8 * w, theme.WARN, 0.012)
            self.view.set_label("pp.axis.l", 0.95 * w, "ω̂", theme.WARN)
            self.view.frame_scene(1.4, planar=False)

    # ------------------------------------------------------------------ export
    def export_params(self) -> dict:
        """Snapshot of the export settings (read in the GUI thread; the worker only sees this dict)."""
        return dict(n=self.n_spin.value(), topic=self.topic_key(), difficulty=self.difficulty(),
                    seed=self.seed.value(), versions=self.v_spin.value(), key=self.exp_key_cb.isChecked())

    @staticmethod
    def _sets(prm: dict) -> list[tuple[int, list[P.Problem]]]:
        return [(v, P.problem_set(prm["n"], prm["topic"], prm["difficulty"], prm["seed"], v))
                for v in range(prm["versions"])]

    def export_problems(self, versions: int | None = None) -> list[tuple[int, list[P.Problem]]]:
        prm = self.export_params()
        if versions is not None:
            prm["versions"] = versions
        return self._sets(prm)

    @staticmethod
    def _versioned(path: str, v: int, nv: int, suffix: str = "") -> str:
        root, ext = os.path.splitext(path)
        tag = f"_{P.version_label(v)}" if nv > 1 else ""
        return f"{root}{tag}{suffix}{ext}"

    @classmethod
    def _write_markdown(cls, path: str, prm: dict) -> list[str]:
        sets = cls._sets(prm)
        out = []
        for v, probs in sets:
            vv = v if len(sets) > 1 else None
            out.append(P.export_markdown(cls._versioned(path, v, len(sets)), probs, False, version=vv))
            if prm["key"]:
                out.append(P.export_markdown(cls._versioned(path, v, len(sets), "_solutions"), probs, True,
                                             version=vv))
        return out

    @classmethod
    def _write_word(cls, path: str, prm: dict) -> list[str]:
        sets = cls._sets(prm)
        out = []
        have = P.find_pandoc() is not None
        for v, probs in sets:
            vv = v if len(sets) > 1 else None
            if have:
                out.append(P.export_docx(cls._versioned(path, v, len(sets)), probs, True, version=vv))
            else:
                md = os.path.splitext(cls._versioned(path, v, len(sets)))[0] + ".md"
                out.append(P.export_markdown(md, probs, True, version=vv))
        return out

    def export_markdown_to(self, path: str, key: bool | None = None) -> list[str]:
        prm = self.export_params()
        if key is not None:
            prm["key"] = key
        return self._write_markdown(path, prm)

    def export_word_to(self, path: str) -> list[str]:
        """Word files with solutions (pandoc); without pandoc, Markdown with solutions next to ``path``."""
        return self._write_word(path, self.export_params())

    def _ask(self, title, ext, filt):
        name = f"practice_{self.topic_key()}_d{self.difficulty()}_s{self.seed.value()}{ext}"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(os.path.expanduser("~"), name), filt)
        if fn and not fn.endswith(ext):
            fn += ext
        return fn

    def _report(self, files, note=""):
        self._tone(self.exp_lbl, "good")
        self.exp_lbl.setText(f"Wrote {len(files)} file(s){note}:\n" + "\n".join(os.path.basename(f) for f in files))

    def _export_md_dialog(self):
        fn = self._ask("Export problem set (Markdown)", ".md", "Markdown (*.md)")
        if fn:
            prm = self.export_params()
            self._run_export(lambda: self._write_markdown(fn, prm))

    def _export_doc_dialog(self):
        have = P.find_pandoc() is not None
        fn = self._ask("Export problem set + solutions (Word)", ".docx" if have else ".md",
                       "Word document (*.docx)" if have else "Markdown (*.md)")
        if fn:
            prm = self.export_params()
            self._run_export(lambda: self._write_word(fn, prm), "" if have else " (pandoc not found → Markdown)")

    def _run_export(self, fn, note=""):
        """Generate + write in a worker thread (N × versions problems and pandoc can take seconds)."""
        if self._worker is not None:
            return
        for b in (self.exp_md_btn, self.exp_doc_btn):
            b.setEnabled(False)
        self._tone(self.exp_lbl, None)
        self.exp_lbl.setText("Exporting…")
        w = _ExportWorker(fn, self)
        w.done.connect(lambda files: self._report(files, note))
        w.failed.connect(self._export_failed)
        w.finished.connect(self._export_finished)
        self._worker = w
        w.start()

    def _export_failed(self, msg):     # pandoc missing / failed / disk error -> show it, don't crash the app
        self._tone(self.exp_lbl, "bad")
        self.exp_lbl.setText(f"Export failed: {msg}")

    def _export_finished(self):
        for b in (self.exp_md_btn, self.exp_doc_btn):
            b.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None


class _ExportWorker(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:  # noqa: BLE001 -- reported in the tab
            self.failed.emit(str(e))
