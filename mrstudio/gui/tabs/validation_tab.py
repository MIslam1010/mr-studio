"""Tab 8 -- cross-validation suite: independent libraries and physical laws (MR Ch. 4-6, 8, 11)."""
from __future__ import annotations

import math
import os
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ...references import further_reading
from ... import validation as V
from .. import theme
from ..base import BaseTab
from ..jobs import JobRunner
from ..widgets import button, checkbox, combo, group, value_label

THEORY = r"""
<h2>Why can we trust the numbers?</h2>
<span class="ref">Cross-validation of Modern Robotics Ch. 4 (FK), 5 (Jacobians, statics), 6 (IK), 8 (dynamics)</span>
<p>Every quantity shown in the app is computed <b>at least twice, by code that shares nothing</b>,
and is also tested against <b>laws of physics</b> that any correct model must obey.
A bug would have to appear identically in several independent programs — or violate energy
conservation — to go unnoticed.</p>

<h3>1. Independent implementations</h3>
<p><b>Lynch &amp; Park's modern_robotics</b> (the textbook's own code: PoE, Newton–Euler),
<b>Pinocchio</b> (C++, LAAS-CNRS: Featherstone's RNEA, CRBA and Articulated-Body Algorithm, twists stored
as $(v,\omega)$ instead of $(\omega,v)$), <b>Peter Corke's Robotics Toolbox</b> (Denavit–Hartenberg tables
and its own <i>rne</i>) and a <b>symbolic Lagrangian</b> derived with SymPy.</p>
$$e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M \;=\; A_1(\theta_1)\cdots A_n(\theta_n)\quad\mathrm{(PoE = DH)}$$
$$J_b(\theta)=\left[\mathrm{Ad}_{T_{bs}}\right]J_s(\theta),\qquad [\mathcal{V}_s]=\dot{T}\,T^{-1}\quad(5.22),\ (3.74)$$
<p>The space Jacobian is also compared with a numerical derivative of the forward kinematics:</p>
$$J_{s,i}\approx\frac{\log\left(T(\theta+\delta e_i)T^{-1}\right)-\log\left(T(\theta-\delta e_i)T^{-1}\right)}{2\delta}$$

<h3>2. Two derivations of the dynamics</h3>
<p>Newton–Euler (forces and moments, link by link, MR 8.3) and Euler–Lagrange (energy, MR 8.1)
are different routes to the same equations; the app derives the Lagrangian of the planar arms
symbolically and checks that both give identical torques.</p>
$$\tau=\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{\theta}}-\frac{\partial \mathcal{L}}{\partial \theta}\;=\;M(\theta)\ddot{\theta}+c(\theta,\dot{\theta})+g(\theta)$$
$$M(\theta)=\sum_{i=1}^{n} J_i^{T}(\theta)\,\mathcal{G}_i\,J_i(\theta)\quad(8.57)$$

<h3>3. Laws of physics</h3>
<p><b>Kinetic energy is positive:</b> $M=M^{T}$ and $\dot{\theta}^{T}M\dot{\theta}>0$.</p>
<p><b>Passivity:</b> with $C$ from the Christoffel symbols, $\dot{M}-2C$ is skew-symmetric,
so $x^{T}(\dot{M}-2C)\,x=0$ for every $x$ (MR Prop. 8.1).</p>
<p><b>Energy:</b> without motors or friction the total energy is constant; with them,</p>
$$\frac{dE}{dt}=\tau^{T}\dot{\theta}-\dot{\theta}^{T}B\,\dot{\theta}$$
<p><b>Virtual work (statics):</b> $\tau=J_s^{T}(\theta)\,\mathcal{F}_s$ must equal Newton–Euler at rest.</p>

<h3>Reading the table</h3>
<p><b>max error</b> is the largest discrepancy found over all random samples (fixed seed, so every run is
identical). Double precision has $\epsilon\approx 2.2\times10^{-16}$: agreement at $10^{-15}$–$10^{-13}$ is pure
round-off. Finite-difference checks are limited to about $10^{-10}$. The row marked ⓘ shows the energy
drift of explicit Euler integration for contrast: it is an integrator artefact, not a modelling error.</p>
"""
THEORY += further_reading([
    ('featherstone', 'Tables 5.1, 6.2, 7.1, pp. 96, 107, 132', 're-implemented in mrstudio/featherstone.py and cross-checked'),
    ('featherstone', 'Ch. 10, from p. 195', 'accuracy of dynamics algorithms; sources of error (p. 196)'),
    ('mls', 'Ch. 4 Lemma 4.2, p. 171', 'structural properties tested here (M > 0, Ṁ − 2C skew)'),
    ('corke', '§9.1, pp. 191–200', 'the Robotics Toolbox rne used as a DH cross-check'),
])


COLUMNS = ["Category", "Check", "MR section", "max error", "tolerance", "✓/✗"]
CAT_COLORS = {
    V.KIN: theme.ACCENT, V.IKC: "#b392f0", V.STAT: "#5eead4", V.DYN: theme.JOINT, V.PHYS: "#ff6b9a",
}
_ROW_FAIL_BG = QtGui.QColor("#3a1d22")
_ROW_SKIP_FG = QtGui.QColor("#6b737d")


def _cat_color(cat: str) -> str:
    return CAT_COLORS.get(cat, theme.WARN)


def _suite_job(progress, cancelled):
    """Worker-thread body: run the whole suite, streaming (fraction, "step", message) and
    (fraction, "result", index, check) through the JobRunner's progress channel."""
    t0 = time.perf_counter()
    counter = [0]
    last = [0.0]

    def on_progress(f, m):
        last[0] = float(f)
        progress(float(f), "step", str(m))

    def on_result(c):
        progress(last[0], "result", counter[0], c)
        counter[0] += 1

    checks = V.run_all(progress=on_progress, on_result=on_result, cancelled=cancelled)
    return checks, time.perf_counter() - t0


class ValidationTab(BaseTab):
    title = "⑭ Validation"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.checks: list = []
        self.elapsed: float | None = None
        self.jobs = JobRunner(self)
        self._planned = False
        self._n_results = 0

        # ------------------------------------------------------------------ left column
        intro = QtWidgets.QLabel(
            "Every formula in the app is computed by at least two independent code bases and tested "
            "against laws of physics, for every robot. Fixed random seed → identical results on every run.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        self.run_btn = button("▶  Run all checks", self.run_checks, primary=True,
                              tooltip="Runs the whole cross-validation suite in a background thread (~6 s)")
        self.run_btn.setMinimumHeight(48)
        f = self.run_btn.font()
        f.setPointSizeF(f.pointSizeF() + 3)
        f.setBold(True)
        self.run_btn.setFont(f)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setFormat("%p %")
        self.step_lbl = QtWidgets.QLabel("Ready.")
        self.step_lbl.setObjectName("muted")
        self.step_lbl.setWordWrap(True)

        self.cat_combo = combo(["All categories"], lambda *_: self._apply_filter())
        self.fail_only = checkbox("Show failures only", lambda *_: self._apply_filter())
        self.cat_summary = QtWidgets.QLabel("—")
        self.cat_summary.setTextFormat(QtCore.Qt.RichText)

        self.export_btn = button("Export report (HTML)…", self.export_report,
                                 tooltip="Stand-alone HTML report with every check, its details and library versions")
        self.export_btn.setEnabled(False)
        self.saved_lbl = QtWidgets.QLabel("")
        self.saved_lbl.setWordWrap(True)
        self.saved_lbl.setOpenExternalLinks(True)
        self.saved_lbl.setObjectName("muted")

        self.sources_lbl = QtWidgets.QLabel()
        self.sources_lbl.setWordWrap(True)
        self.sources_lbl.setTextFormat(QtCore.Qt.RichText)
        self.sources_lbl.setText(
            f"<span style='color:{theme.text_color()}'>"
            "<b>modern_robotics</b> — Lynch &amp; Park's reference code<br>"
            "<b>Pinocchio</b> — C++, Featherstone algorithms<br>"
            "<b>Robotics Toolbox</b> — P. Corke, DH models + rne<br>"
            "<b>SymPy</b> — symbolic Euler–Lagrange<br>"
            "<b>SciPy</b> — adaptive RK45 (Dormand–Prince) integration</span>")

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Cross-validation suite", intro, self.run_btn, self.bar, self.step_lbl))
        cl.addWidget(group("Results by category", self.cat_summary))
        cl.addWidget(group("Filter", self.cat_combo, self.fail_only))
        cl.addWidget(group("Report", self.export_btn, self.saved_lbl))
        cl.addWidget(group("Independent references", self.sources_lbl))
        cl.addStretch(1)

        # ------------------------------------------------------------------ centre: banner + table
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(QtCore.Qt.RichText)
        self.banner.setMinimumHeight(58)
        self.banner.setContentsMargins(14, 8, 14, 8)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("QTableWidget { font-family: -apple-system, 'Helvetica Neue', 'Segoe UI', sans-serif; }"
                                 "QTableWidget::item { padding: 0 6px; }")
        hh = self.table.horizontalHeader()
        for col, mode in enumerate([QtWidgets.QHeaderView.ResizeToContents, QtWidgets.QHeaderView.Stretch,
                                    QtWidgets.QHeaderView.ResizeToContents, QtWidgets.QHeaderView.ResizeToContents,
                                    QtWidgets.QHeaderView.ResizeToContents, QtWidgets.QHeaderView.Fixed]):
            hh.setSectionResizeMode(col, mode)
        self.table.setColumnWidth(5, 46)
        self.table.currentCellChanged.connect(lambda r, *_: self._show_details(r))
        self._mono = QtGui.QFont("Menlo")
        self._mono.setStyleHint(QtGui.QFont.Monospace)

        center = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(center)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)
        vl.addWidget(self.banner)
        vl.addWidget(self.table, 1)

        # ------------------------------------------------------------------ bottom: details of selected row
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.details.setMinimumHeight(120)

        self.assemble(controls, center, self.details, THEORY, bottom_ratio=(5, 2))
        self._set_banner("idle")
        self._show_details(-1)

    # ====================================================================== table helpers
    def _item(self, text, color=None, mono=False, bold=False, align=None, tip=None):
        it = QtWidgets.QTableWidgetItem(text)
        it.setForeground(QtGui.QColor(color or theme.text_color()))
        if mono:
            it.setFont(self._mono)
        if bold:
            f = it.font()
            f.setBold(True)
            it.setFont(f)
        if align is not None:
            it.setTextAlignment(align)
        if tip:
            it.setToolTip(tip)
        return it

    def _set_row_pending(self, row, cat, name, chapter):
        right = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        self.table.setItem(row, 0, self._item(cat, _cat_color(cat)))
        self.table.setItem(row, 1, self._item(name, theme.muted_color()))
        self.table.setItem(row, 2, self._item(chapter, theme.muted_color()))
        self.table.setItem(row, 3, self._item("…", theme.muted_color(), mono=True, align=right))
        self.table.setItem(row, 4, self._item("", theme.muted_color(), mono=True, align=right))
        self.table.setItem(row, 5, self._item("·", theme.muted_color(), align=QtCore.Qt.AlignCenter))

    def _set_row(self, row, c):
        right = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        skipped = V.is_skipped(c)
        info = V.is_informational(c)
        ok = c.passed or info              # informational rows are never shown as failures
        if skipped:
            mark, mcol = "–", theme.muted_color()
        elif info:
            mark, mcol = "ⓘ", theme.muted_color()
        else:
            mark, mcol = ("✓", theme.good_color()) if ok else ("✗", theme.bad_color())
        name_col = theme.muted_color() if skipped else theme.text_color()
        items = [
            self._item(c.category, _cat_color(c.category)),
            self._item(c.name, name_col, tip=c.name),
            self._item(c.chapter, theme.muted_color()),
            self._item(V.fmt_err(c.error), theme.muted_color() if skipped else (theme.text_color() if ok else theme.bad_color()),
                       mono=True, align=right),
            self._item(V.fmt_err(c.tol), theme.muted_color(), mono=True, align=right),
            self._item(mark, mcol, bold=True, align=QtCore.Qt.AlignCenter),
        ]
        for col, it in enumerate(items):
            if not ok and not skipped:
                it.setBackground(_ROW_FAIL_BG)
            if skipped:
                it.setForeground(_ROW_SKIP_FG if col != 0 else QtGui.QColor(_cat_color(c.category)))
            self.table.setItem(row, col, it)

    def _populate_planned(self, force: bool = False):
        """Fill the table with the planned (not yet run) checks so the user sees what will be verified.
        Re-planned at every run: robots added with the Robot Builder are validated too."""
        if self._planned and not force:
            return
        self._planned = True
        try:
            planned = V.planned_checks()
        except Exception:
            planned = []
        if self.checks and not force:          # a run already finished meanwhile
            return
        self.table.setRowCount(len(planned) + len(V.OPTIONAL_MODULES))
        for r, (cat, name, chapter) in enumerate(planned):
            self._set_row_pending(r, cat, name, chapter)
        for j, (mod, cat) in enumerate(V.OPTIONAL_MODULES):
            self._set_row_pending(len(planned) + j, cat, f"{mod}.validation_checks()", "—")
        self.n_planned = len(planned)
        self._set_banner("idle")

    # ====================================================================== run
    def run_checks(self):
        if self.jobs.is_busy():
            return
        self.checks = []
        self._populate_planned(force=True)
        self._n_results = 0
        self.elapsed = None
        for r in range(self.table.rowCount()):           # reset rows to "pending"
            for col in (3, 4, 5):
                it = self.table.item(r, col)
                if it is not None:
                    it.setText("…" if col == 3 else ("·" if col == 5 else ""))
                    it.setForeground(QtGui.QColor(theme.muted_color()))
                    it.setBackground(QtGui.QBrush())
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…")
        self.export_btn.setEnabled(False)
        self.bar.setValue(0)
        self._set_banner("running")
        self.jobs.run(_suite_job, on_done=lambda res: self._on_done(*res), on_error=self._on_crash,
                      on_progress=self._on_job_progress, label="Validation suite")

    def _on_job_progress(self, frac, kind=None, *info):
        """Progress channel of the JobRunner (already filtered: only the current run gets here)."""
        if kind == "result":
            self._on_result(*info)
        elif kind == "step":
            self._on_progress(frac, *info)

    def _on_progress(self, frac, msg):
        self.bar.setValue(int(1000 * frac))
        self.step_lbl.setText(msg)
        if self.jobs.is_busy():
            self._set_banner("running", msg)

    def _on_result(self, idx, c):
        self._n_results = idx + 1
        if idx >= self.table.rowCount():
            self.table.setRowCount(idx + 1)
        self._set_row(idx, c)
        self.checks.append(c)
        self._update_category_summary()

    def _on_done(self, checks, elapsed):
        self.checks = list(checks)
        self.elapsed = elapsed
        self.table.setRowCount(len(self.checks))
        for r, c in enumerate(self.checks):
            self._set_row(r, c)
        cats = []
        for c in self.checks:
            if c.category not in cats:
                cats.append(c.category)
        cur = self.cat_combo.currentText()
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        self.cat_combo.addItems(["All categories"] + cats)
        self.cat_combo.setCurrentText(cur if cur in cats else "All categories")
        self.cat_combo.blockSignals(False)
        self._update_category_summary()
        self._apply_filter()
        self.bar.setValue(1000)
        s = V.summary(self.checks)
        self.step_lbl.setText(f"Finished {s['total']} checks in {elapsed:.1f} s.")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Run all checks again")
        self.export_btn.setEnabled(True)
        self._set_banner("done")
        first_fail = next((r for r, c in enumerate(self.checks)
                           if not c.passed and not V.is_skipped(c) and not V.is_informational(c)), 0)
        self.table.selectRow(first_fail)
        self._show_details(first_fail)

    def _on_crash(self, tb):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Run all checks")
        self.step_lbl.setText("The suite crashed — see details.")
        self.details.setPlainText(tb)
        self._set_banner("crash")

    def wait_job(self, timeout_s: float = 300.0):
        """Block (processing events) until the suite finished -- scripted tests only."""
        self.jobs.wait(timeout_s)

    def shutdown(self):
        """Application quit: stop the suite between two checks (run_all polls cancelled()); wait at most ~2 s."""
        self.jobs.shutdown(2000)
        parent = getattr(super(), "shutdown", None)
        if parent is not None:
            parent()

    # ====================================================================== presentation
    def _set_banner(self, state, msg=""):
        if state == "idle":
            n = self.table.rowCount()
            txt = (f"<span style='font-size:17px; font-weight:600'>Cross-validation suite</span><br>"
                   f"<span style='color:{theme.MUTED}'>{n if n else 'About 100'} checks ready (plus the checks "
                   f"of the closed-chain, parallel, trajectory and control modules). "
                   f"Press <b>▶ Run all checks</b> — it takes a few seconds.</span>")
            bg, border = "#1e2530", "#39424e"
        elif state == "running":
            done = self._n_results
            txt = (f"<span style='font-size:17px; font-weight:600'>Running… {done} checks done</span><br>"
                   f"<span style='color:{theme.MUTED}'>{msg}</span>")
            bg, border = "#1b2a36", theme.ACCENT
        elif state == "crash":
            txt = "<span style='font-size:17px; font-weight:600'>✗ The validation suite crashed</span>"
            bg, border = "#3a1d22", theme.BAD
        else:
            s = V.summary(self.checks)
            skipped = f" · {s['skipped']} optional module(s) not available" if s["skipped"] else ""
            if s.get("informational"):
                skipped += f" · {s['informational']} ⓘ informational row(s) not counted"
            if s["all_passed"]:
                head = (f"<span style='color:{theme.GOOD}'>✓</span> {s['passed']} / {s['total']} checks passed"
                        f" — max error {V.fmt_err(s['max_precise_error'])}")
                bg, border = "#15301f", theme.GOOD
            else:
                head = (f"<span style='color:{theme.BAD}'>✗</span> {s['failed']} of {s['total']} checks failed "
                        f"— {s['passed']} passed")
                bg, border = "#3a1d22", theme.BAD
            took = f" · run time {self.elapsed:.1f} s" if self.elapsed is not None else ""
            txt = (f"<span style='font-size:19px; font-weight:700'>{head}</span><br>"
                   f"<span style='color:{theme.MUTED}'>max error = largest error among the machine-precision "
                   f"comparisons (tolerance ≤ 1e-9); double precision ε ≈ 2.2e-16{took}{skipped}</span>")
        self.banner.setText(txt)
        self.banner.setStyleSheet(f"QLabel {{ background:{bg}; border:1px solid {border}; border-radius:7px; "
                                  f"color:{theme.TEXT}; }}")

    def _update_category_summary(self):
        cats: dict = {}
        for c in self.checks:
            d = cats.setdefault(c.category, [0, 0, 0])
            if V.is_informational(c):
                continue
            if V.is_skipped(c):
                d[2] += 1
            else:
                d[1] += 1
                d[0] += int(c.passed)
        rows = []
        for cat, (p, n, sk) in cats.items():
            ok = p == n
            status = (f"<span style='color:{theme.good_color()}'>✓</span>" if ok and n else
                      f"<span style='color:{theme.bad_color()}'>✗</span>" if n else
                      f"<span style='color:{theme.muted_color()}'>skipped</span>")
            count = f"{p} / {n}" if n else "—"
            rows.append(f"<tr><td><span style='color:{_cat_color(cat)}'>●</span>&nbsp;{cat}</td>"
                        f"<td align='right' style='font-family:Menlo'>&nbsp;&nbsp;{count}</td>"
                        f"<td align='center'>&nbsp;{status}</td></tr>")
        self.cat_summary.setText(f"<table width='100%' cellspacing='2'>{''.join(rows)}</table>")

    def _apply_filter(self):
        cat = self.cat_combo.currentText()
        fails = self.fail_only.isChecked()
        for r in range(self.table.rowCount()):
            c = self.checks[r] if r < len(self.checks) else None
            hide = False
            if c is not None:
                if cat != "All categories" and c.category != cat:
                    hide = True
                if fails and (c.passed or V.is_skipped(c) or V.is_informational(c)):
                    hide = True
            else:
                it = self.table.item(r, 0)
                if cat != "All categories" and it is not None and it.text() != cat:
                    hide = True
            self.table.setRowHidden(r, hide)

    def _show_details(self, row):
        css = (f"<style>body{{color:{theme.text_color()}; font-size:13px}} h3{{color:{theme.accent_color()}; margin:0 0 4px 0}}"
               f".m{{color:{theme.muted_color()}}} .k{{color:{theme.muted_color()}}} td{{padding-right:14px}}</style>")
        if row is None or row < 0 or row >= len(self.checks):
            self.details.setHtml(
                css + "<h3>Details</h3><p class='m'>Run the suite, then click any row to see exactly what was "
                      "compared with what, how many random samples were used and why it proves correctness.</p>"
                + "".join(f"<p><b style='color:{_cat_color(k)}'>{k}.</b> <span class='m'>{v}</span></p>"
                          for k, v in V.FAMILY_TEXT.items()))
            return
        c = self.checks[row]
        if V.is_skipped(c):
            verdict = f"<span style='color:{theme.muted_color()}'>skipped</span>"
        elif V.is_informational(c):
            verdict = (f"<span style='color:{theme.muted_color()}'><b>ⓘ informational</b> — shown for contrast, "
                       f"not counted in the pass total</span>")
        elif c.passed:
            margin = ""
            if c.error > 0 and math.isfinite(c.error) and c.tol > 0:
                margin = f" — {math.log10(c.tol / c.error):.1f} orders of magnitude below the tolerance"
            elif c.error == 0:
                margin = " — exact agreement"
            verdict = f"<span style='color:{theme.good_color()}'><b>✓ passed</b>{margin}</span>"
        else:
            verdict = f"<span style='color:{theme.bad_color()}'><b>✗ FAILED</b></span>"
        detail = (c.detail or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        fam = V.FAMILY_TEXT.get(c.category, "")
        self.details.setHtml(
            css + f"<h3>{c.name}</h3>"
            f"<table><tr><td class='k'>Result</td><td>{verdict}</td></tr>"
            f"<tr><td class='k'>Max error</td><td style='font-family:Menlo'>{V.fmt_err(c.error)}</td></tr>"
            f"<tr><td class='k'>Tolerance</td><td style='font-family:Menlo'>{V.fmt_err(c.tol)}</td></tr>"
            f"<tr><td class='k'>Textbook</td><td>{c.chapter} (Lynch &amp; Park, <i>Modern Robotics</i>)</td></tr>"
            f"<tr><td class='k'>Category</td><td style='color:{_cat_color(c.category)}'>{c.category}</td></tr></table>"
            f"<p><b>What was compared, and how:</b> {detail}</p>"
            + (f"<p class='m'><i>{fam}</i></p>" if fam else ""))

    # ====================================================================== export
    def export_report(self):
        if not self.checks:
            return
        default = os.path.join(os.path.expanduser("~"), "MR_Studio_validation_report.html")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export validation report", default,
                                                      "HTML report (*.html)")
        if not fn:
            return
        self.save_report(fn)

    def save_report(self, fn: str) -> str:
        if not fn.lower().endswith((".html", ".htm")):
            fn += ".html"
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write(V.report_html(self.checks, self.elapsed))
        url = QtCore.QUrl.fromLocalFile(fn).toString()
        self.saved_lbl.setText(f"Saved <a href='{url}' style='color:{theme.link_color()}'>{os.path.basename(fn)}</a> "
                               f"— open it in any browser or print it to PDF.")
        return fn

    # ====================================================================== hooks
    def on_show(self):
        if not self._planned:
            QtCore.QTimer.singleShot(30, self._populate_planned)
