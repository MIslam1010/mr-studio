"""Grasping tab -- planar contact kinematics, form closure and force closure (Modern Robotics, Ch. 12).

A planar body (polygon) with point contacts (p, n): contact labels R / Sl / Sr / B for a test twist, the feasible
centres of rotation, the form-closure LP (12.14), friction cones (12.16) and the force-closure test, the minimum
friction coefficient, the grasp metric (12.15) and quasistatic / dynamic balance (12.32)-(12.33).
Maths lives in mrstudio.grasping."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ... import grasping as gr
from ...references import further_reading
from .. import theme
from ..base import BaseTab
from ..widgets import MatrixView, SliderRow, button, checkbox, combo, group, hbox, no_si, plot, value_label

# Every "(12.n)" below is followed by the printed page on which that equation appears in the PDF of
# Modern Robotics (v2, Dec 2019 preprint; printed page = PDF page - 20 in Ch. 12), checked with pdftotext.
THEORY = r"""
<h2>Grasping: contact kinematics, form and force closure</h2>
<span class="ref">Modern Robotics, Ch. 12, pp. 463–500.  All twists and wrenches are in the space frame
(p. 467, footnote 1).  Planar twist $\mathcal{V}=(\omega_{z},v_{x},v_{y})$, planar wrench
$\mathcal{F}=(m_{z},f_{x},f_{y})$.</span>
<h3>First-order contact kinematics (Sec. 12.1.1–12.1.2)</h3>
<p>A distance function $d(q)$ is positive when the bodies are apart and zero in contact.  Using only the contact
normal (no curvature), the analysis stops at the first derivative:</p>
$$\dot{d}=\frac{\partial d}{\partial q}\dot{q}\quad(12.1)$$<span class="ref">p. 466</span>
<p>The unit normal $\hat{n}$ points <b>into body A</b>; a unit force along it is the wrench
$\mathcal{F}=([p_{A}]\hat{n},\hat{n})$, in the plane $m_{z}=p_{x}n_{y}-p_{y}n_{x}$ (p. 467).  Impenetrability:</p>
$$\mathcal{F}^{T}(\mathcal{V}_{A}-\mathcal{V}_{B})\geq 0\quad(12.5)$$<span class="ref">p. 467</span>
$$\mathcal{F}^{T}(\mathcal{V}_{A}-\mathcal{V}_{B})=0\quad(12.6)$$<span class="ref">p. 468 (active constraint)</span>
$$\dot{p}_{A}=v_{A}+[\omega_{A}]p_{A}=v_{B}+[\omega_{B}]p_{B}=\dot{p}_{B}\quad(12.8)$$<span class="ref">p. 468 (rolling)</span>
<p>Labels: <b>B</b> breaking ((12.5) strict), <b>S</b> sliding ((12.6) but not (12.8)), <b>R</b> rolling or
sticking ((12.8)).  For a stationary B the constraint is $\mathcal{F}^{T}\mathcal{V}_{A}\geq 0$ (12.7),
p. 468.  In the plane S splits into <b>Sl</b> / <b>Sr</b>: the movable body slips left / right relative to the
fixed constraint (p. 478); this tab reads left / right facing along $\hat{n}$.  A twist that violates (12.5)
is marked <b>X</b> (penetrating).</p>
<p><b>Example 12.2</b> (pp. 468–470): $p=(1,2)$, $\hat{n}=(0,1)$ give $(m_{z},f_{x},f_{y})=(1,0,1)$; with B
fixed the roll–slide condition becomes $\omega_{Az}+v_{Ay}=0$ and rolling adds $v_{Ax}-2\omega_{Az}=0$
(the planar forms of (12.9), p. 469, and of (12.10)–(12.11), p. 470).</p>
<h3>Feasible twists and centres of rotation (Sec. 12.1.3, 12.1.6)</h3>
<p>Stationary contacts give a polyhedral convex cone (p. 471):</p>
$$V=\{\mathcal{V}_{A}\ |\ \mathcal{F}_{i}^{T}\mathcal{V}_{A}\geq 0\ \mathrm{for\ all}\ i\}$$
<p>A planar twist is drawn as its centre of rotation $(-v_{y}/\omega_{z},\ v_{x}/\omega_{z})$ with the sign of
$\omega_{z}$ (p. 475).  The scene shades every feasible CoR: <b>blue</b> = counter-clockwise ('+'),
<b>orange</b> = clockwise ('−'); points left of an inward normal are '+', right are '−' (pp. 477–478).</p>
<h3>Form closure (Sec. 12.1.7)</h3>
<p>Form closure: the only feasible twist is zero, i.e. $\mathrm{pos}(\{\mathcal{F}_{1},\ldots,\mathcal{F}_{j}\})=\mathbb{R}^{3}$
(pp. 480–481).  Theorem 12.6 (p. 481): a planar body needs at least <b>four</b> point contacts, a spatial body seven.
With $F=[\mathcal{F}_{1}\cdots\mathcal{F}_{j}]$, the test is rank $F=n$ and the linear program</p>
$$\min_{k}\ \mathbf{1}^{T}k\quad\mathrm{such\ that}\quad Fk=0,\quad k_{i}\geq 1\quad(12.14)$$<span class="ref">p. 482</span>
<p>Example 12.7 (pp. 482–483): two fingers in a hole, LP solution $k=(2,1,2,1)$; with the right finger moved
the LP has no solution and the body slides down.  First-order form closure is form closure for any analysis;
roll–slide-only motions may still be blocked by curvature (pp. 481–482).  Disks are <b>exceptional</b>: no
contacts prevent spinning (p. 481).</p>
<p><b>Grasp metric</b> (Sec. 12.1.7.3): with contact force limits $f_{i,max}$ the contacts can apply</p>
$$CF=\left\{\sum_{i=1}^{j}f_{i}\mathcal{F}_{i}\ |\ f_{i}\in[0,f_{i,max}]\right\}\quad(12.15)$$<span class="ref">p. 484</span>
<p>and Qual is the radius of the largest wrench ball about the origin inside CF, moments divided by a
characteristic length $r$, origin near the centre of the body (p. 485).  Here $f_{i,max}=1$, $r$ = largest
centroid-to-vertex distance.  With friction the tab applies (12.15) to the two cone-edge forces of each
contact, $\frac{f_{max}}{2}(\hat{n}\pm\mu\hat{t})$, so that their normal components add up to $f_{max}$: a contact
pushes with at most $f_{max}$ along its normal, and a larger $\mu$ only adds tangential force, so Qual never
decreases as $\mu$ grows and equals the frictionless value at $\mu=0$ (an extension: the book states (12.15)
for the normals).</p>
<h3>Friction and force closure (Sec. 12.2)</h3>
<p>Coulomb friction (p. 486), normal along $\hat{z}$:</p>
$$\sqrt{f_{x}^{2}+f_{y}^{2}}\leq\mu f_{z},\quad f_{z}\geq 0\quad(12.16)$$<span class="ref">p. 486</span>
<p>with friction angle $\alpha=\tan^{-1}\mu$.  In the plane the cone is exactly the positive span of its two edges,
and each edge force $f$ at $p$ is the wrench $([p]f,f)$ (p. 487).  <b>Force closure</b>: the composite wrench
cone is the whole wrench space; the test is rank $F=n$ and the LP (12.14) with the cone edges as the columns
of F (exact for planar problems, pp. 491–492).  With $\mu=0$ force closure is first-order form closure.  Two
frictional contacts suffice in the plane; an equilateral triangle held on two edges needs
$\mu\geq\tan 30^{\circ}\approx 0.577$ (Fig. 12.22, p. 492).  The tab finds the minimum $\mu$ by bisection on
the LP (Sec. 12.2.3.2, p. 496).</p>
<p><b>Duality</b> (Sec. 12.2.4, p. 496): planar B, S and R contacts give 0, 1 and 2 motion constraints and the
same numbers of wrench freedoms.</p>
<h3>Quasistatic and dynamic balance (Sec. 12.3)</h3>
$$\mathcal{F}_{ext}+\sum k_{i}\mathcal{F}_{i}=0,\quad k_{i}\geq 0,\ \mathcal{F}_{i}\in WC_{i}\quad(12.32)$$<span class="ref">p. 498</span>
<p>Example 12.9: a block on a finger with $\mu=1$ and a frictionless finger cannot be held at rest (p. 498);
accelerating the fingers sideways (contact mode RR)</p>
$$k_{1}\mathcal{F}_{1}+k_{2}\mathcal{F}_{2}+k_{3}\mathcal{F}_{3}+(0,0,-mg)=(0,ma_{x},0)\quad(12.33)$$<span class="ref">p. 499</span>
<p>has $k_{i}\geq 0$ for $-5g\leq a_{x}\leq -g$ (p. 500): a <b>dynamic grasp</b>.</p>
<p><b>Using the tab.</b>  Edit the contacts in the table (inward normal angle in degrees), or drag the test-twist
sliders: the scene shows the CoR of the test twist and the label of every contact.  Spatial grasps with three
frictional contacts reduce to this planar test (Theorem 12.8, p. 493).</p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 5 §2, p. 214', 'grasp statics, contact models'),
    ('mls', 'Ch. 5 §3, p. 223', 'force-closure: definition and constructive tests'),
    ('mls', 'Ch. 5 §4.2, Thm. 5.6, p. 232', 'planar antipodal grasps'),
])

PLUS_RGB = (76, 201, 240)      # '+' CoRs (counter-clockwise), theme accent
MINUS_RGB = (242, 158, 76)     # '-' CoRs (clockwise), theme joint orange


class GraspingTab(BaseTab):
    title = "⑫ Grasping"
    uses_serial_robot = False
    uses_engine = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self._loading = False
        self.poly = np.zeros((0, 2))
        self.hole = None
        self.contacts: list[gr.Contact] = []
        self.gravity = False
        self._curve_key = None
        self._items: list = []

        # ---------------- scenario
        self.preset_box = combo(list(gr.PRESETS), self.load_preset)
        self.preset_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset_box.setMinimumContentsLength(24)
        self.preset_lbl = QtWidgets.QLabel()
        self.preset_lbl.setWordWrap(True)
        self.preset_lbl.setObjectName("muted")

        # ---------------- contacts table
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["x", "y", "n̂ angle [°]", "μ"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setMinimumHeight(150)
        self.table.setToolTip("Contact point (x, y) and the direction of the contact normal pointing INTO the body; "
                              "μ = Coulomb friction coefficient (0 = frictionless).")
        self.table.itemChanged.connect(self._table_edited)
        self.add_btn = button("+ contact", self._add_contact, tooltip="add a contact at the centroid's left")
        self.del_btn = button("− contact", self._del_contact, tooltip="remove the selected (or last) contact")
        self.mu_all = SliderRow("μ all", 0.0, 1.5, 0.0, decimals=2)
        self.mu_all.setToolTip("set the same friction coefficient at every contact")
        self.mu_all.valueChanged.connect(self._mu_all_changed)

        # ---------------- test twist
        self.w = SliderRow("ω_z", -2, 2, 0.0, decimals=2, unit=" rad/s")
        self.vx = SliderRow("v_x", -2, 2, 0.0, decimals=2)
        self.vy = SliderRow("v_y", -2, 2, 0.0, decimals=2)
        for s in (self.w, self.vx, self.vy):
            s.valueChanged.connect(self._twist_changed)
        self.twist_lbl = value_label()
        self.cb_cor = checkbox("Shade feasible centres of rotation", self._redraw, True)
        self.cb_cone = checkbox("Friction cones", self._redraw, True)

        # ---------------- results
        self.form_lbl, self.force_lbl, self.mu_lbl, self.qual_lbl, self.bal_lbl = (value_label() for _ in range(5))
        for lb in (self.form_lbl, self.force_lbl, self.mu_lbl, self.qual_lbl, self.bal_lbl):
            lb.setTextFormat(QtCore.Qt.RichText)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Scenario", self.preset_box, self.preset_lbl))
        cl.addWidget(group("Contacts (inward normals)", self.table, hbox(self.add_btn, self.del_btn), self.mu_all))
        cl.addWidget(group("Test twist 𝒱 = (ω_z, v_x, v_y)", self.w, self.vx, self.vy, self.twist_lbl,
                           self.cb_cor, self.cb_cone))
        cl.addWidget(group("Closure tests (LP 12.14)", self.form_lbl, self.force_lbl, self.mu_lbl, self.qual_lbl,
                           self.bal_lbl))
        cl.addStretch(1)

        # ---------------- scene (2-D, equal aspect) and bottom panels
        self.scene = plot("Body, contacts and feasible centres of rotation", "y", "x", legend=False)
        self.scene.setAspectLocked(True)
        self.scene.setMinimumHeight(260)
        self.cor_img = pg.ImageItem()
        self.cor_img.setZValue(-20)
        self.scene.addItem(self.cor_img)

        self.Fview = MatrixView(3, 1, decimals=2, row_labels=["m_z", "f_x", "f_y"])
        self.Fview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.Fview.setToolTip("Columns: the normal wrench of each contact (form closure) followed by the "
                              "friction-cone edge wrenches used for force closure")
        self.F_title = QtWidgets.QLabel()
        self.F_title.setObjectName("muted")
        self.F_title.setWordWrap(True)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.F_title)
        ll.addWidget(self.Fview)
        ll.addStretch(1)
        self.p_q = plot("Grasp metric (12.15) vs common μ", "Qual", "μ")
        no_si(self.p_q)
        self.c_q = self.p_q.plot(pen=pg.mkPen(theme.SERIES[0], width=2), symbol="o", symbolSize=5,
                                 symbolBrush=theme.SERIES[0], name="Qual (f_max = 1 per contact)")
        self.mu_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.WARN, width=1.5, style=QtCore.Qt.DashLine))
        self.p_q.addItem(self.mu_line)
        self.p_q.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.MUTED, width=1)))
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.addWidget(left)
        bottom.addWidget(self.p_q)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)

        self.assemble(controls, self.scene, bottom, THEORY, controls_width=370, bottom_ratio=(3, 2))

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.recompute)
        self.load_preset(self.preset_box.currentText())

    # ------------------------------------------------------------------ presets
    def apply_preset(self, key) -> bool:
        """Select a preset by index, full name, or a case-insensitive substring (for launch URLs)."""
        names = list(gr.PRESETS)
        name = None
        if isinstance(key, int) or (isinstance(key, str) and key.strip().isdecimal() and len(key.strip()) <= 6):
            i = int(key)
            name = names[i] if 0 <= i < len(names) else None
        elif key in gr.PRESETS:
            name = key
        else:
            k = str(key).lower().replace("_", " ").replace("-", " ")
            hits = [n for n in names if k in n.lower().replace("—", " ").replace("-", " ")]
            name = hits[0] if hits else None
        if name is None:
            return False
        if self.preset_box.currentText() == name:
            self.load_preset(name)
        else:
            self.preset_box.setCurrentText(name)
        return True

    def load_preset(self, name: str):
        pre = gr.PRESETS.get(name)
        if pre is None:
            return
        self._loading = True
        self.preset_lbl.setText(pre["text"])
        self.poly = np.asarray(pre["poly"], float)
        self.hole = None if pre.get("hole") is None else np.asarray(pre["hole"], float)
        self.contacts = [gr.Contact(c.p, c.n, c.mu, c.name) for c in pre["contacts"]]
        self.gravity = bool(pre.get("gravity", False))
        w, vx, vy = pre.get("twist", (0.0, 0.0, 0.0))
        self.w.set_value(w)
        self.vx.set_value(vx)
        self.vy.set_value(vy)
        mus = {round(c.mu, 6) for c in self.contacts}
        self.mu_all.set_value(mus.pop() if len(mus) == 1 else 0.0)
        self._fill_table()
        self._loading = False
        self._curve_key = None
        self.recompute()
        self._autorange()

    # ------------------------------------------------------------------ table
    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.contacts))
        for i, c in enumerate(self.contacts):
            vals = (c.p[0], c.p[1], math.degrees(c.angle), c.mu)
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(f"{v:.3g}" if j < 2 or j == 3 else f"{v:.1f}")
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, j, it)
        self.table.setVerticalHeaderLabels([c.name or str(i + 1) for i, c in enumerate(self.contacts)])
        self.table.blockSignals(False)

    def _table_edited(self, item):
        if self._loading:
            return
        i, j = item.row(), item.column()
        if i >= len(self.contacts):
            return
        c = self.contacts[i]
        try:
            v = float(item.text().replace("°", "").replace("−", "-").strip())
            if not math.isfinite(v):                   # "inf" / "nan" / "1e999": cos(inf) raises, NaN breaks the LP
                raise ValueError(item.text())
            if j == 0:
                c = gr.Contact((v, c.p[1]), c.n, c.mu, c.name)
            elif j == 1:
                c = gr.Contact((c.p[0], v), c.n, c.mu, c.name)
            elif j == 2:
                c = gr.Contact.at_angle(c.p[0], c.p[1], math.radians(v), c.mu, c.name)
            else:
                c = gr.Contact(c.p, c.n, max(0.0, v), c.name)
        except ValueError:
            self._fill_table()
            return
        self.contacts[i] = c
        self._debounce.start(40)

    def _add_contact(self):
        if len(self.contacts) >= 6:
            QtWidgets.QMessageBox.information(self, "Grasping", "At most six contacts (exact grasp-metric hull).")
            return
        ctr = gr.polygon_centroid(self.poly) if len(self.poly) else np.zeros(2)
        xmin = float(self.poly[:, 0].min()) if len(self.poly) else -1.0
        self.contacts.append(gr.Contact((xmin, ctr[1]), (1.0, 0.0), self.mu_all.value(), str(len(self.contacts) + 1)))
        self._fill_table()
        self.recompute()

    def _del_contact(self):
        if not self.contacts:
            return
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()}, reverse=True) or [len(self.contacts) - 1]
        for r in rows:
            if 0 <= r < len(self.contacts):
                del self.contacts[r]
        self._fill_table()
        self.recompute()

    def _mu_all_changed(self, v):
        if self._loading:
            return
        self.contacts = gr.with_mu(self.contacts, v)
        self._fill_table()
        self._debounce.start(40)

    def _twist_changed(self, *_):
        if not self._loading:
            self._update_twist()
            self._redraw()

    # ------------------------------------------------------------------ computation
    def twist(self) -> np.ndarray:
        return np.array([self.w.value(), self.vx.value(), self.vy.value()])

    def centroid(self) -> np.ndarray:
        return gr.polygon_centroid(self.poly) if len(self.poly) else np.zeros(2)

    def char_length(self) -> float:
        if not len(self.poly):
            return 1.0
        return float(np.max(np.linalg.norm(self.poly - self.centroid(), axis=1))) or 1.0

    def recompute(self):
        cs = self.contacts
        good, bad, warn = theme.good_color(), theme.bad_color(), theme.warn_color()
        self.form = gr.form_closure(cs)
        self.force = gr.force_closure(cs)

        def verdict(r, what):
            if r.closed:
                k = ", ".join(f"{v:.3g}" for v in r.k)
                return f"<span style='color:{good}'>✓ {what}</span> — rank F = {r.rank}, k = ({k})"
            return f"<span style='color:{bad}'>✗ no {what}</span> — {r.status}"
        self.form_lbl.setText(verdict(self.form, "form closure") + " <i>(frictionless normals)</i>")
        self.force_lbl.setText(verdict(self.force, "force closure") + " <i>(friction-cone edges, μ per contact)</i>")
        self.mu_min = gr.min_friction(cs, mu_max=5.0, tol=1e-4) if cs else math.inf
        if math.isinf(self.mu_min):
            mu_txt = "never (μ ≤ 5)"
        elif self.mu_min == 0.0:
            mu_txt = "<b>0</b> (form closure already)"
        elif self.mu_min <= 2e-4:
            mu_txt = "<b>any μ &gt; 0</b> (infimum 0: squeezing internal forces)"
        else:
            mu_txt = f"<b>{self.mu_min:.3f}</b> (α = tan⁻¹μ = {math.degrees(math.atan(self.mu_min)):.1f}°)"
        self.mu_lbl.setText("Minimum common μ for force closure: " + mu_txt)
        ctr, r = self.centroid(), self.char_length()
        self.qual = gr.grasp_quality(cs, ctr, r, 1.0, friction=True) if 0 < 2 * len(cs) <= 12 else float("nan")
        self.qual_lbl.setText(f"Grasp metric (12.15): Qual = {self.qual:.4f} "
                              f"<i>(moments about the centroid / r, r = {r:.2f}; normal force ≤ f_max = 1 per contact, split over its two cone edges)</i>")
        if self.gravity:
            Fm = gr.wrench_matrix(cs, friction=True)
            g = 9.81
            rest = gr.balance(Fm, np.array([0.0, 0.0, g]))
            dyn = gr.balance(Fm, np.array([0.0, -2 * g, g]))
            self.bal_lbl.setText(
                f"Gravity at the centroid (m = 1, g = 9.81): at rest (12.32) "
                + (f"<span style='color:{good}'>balanced</span>" if rest is not None else
                   f"<span style='color:{bad}'>not balanced</span> → the block moves") +
                "; fingers accelerating at a<sub>x</sub> = −2g, mode RR (12.33): "
                + (f"<span style='color:{good}'>k = ({', '.join(f'{v:.2f}' for v in dyn)}) ≥ 0</span>"
                   if dyn is not None else f"<span style='color:{bad}'>infeasible</span>"))
            self.bal_lbl.setVisible(True)
        else:
            self.bal_lbl.setVisible(False)
        # wrench matrix display
        Fn = gr.wrench_matrix(cs, friction=False)
        Fe = gr.wrench_matrix(cs, friction=True)
        cols = [f"F{c.name or i + 1}" for i, c in enumerate(cs)]
        ecols = []
        for i, c in enumerate(cs):
            nm = c.name or str(i + 1)
            ecols += [f"e{nm}"] if c.mu <= 0 else [f"e{nm}+", f"e{nm}−"]
        M = np.hstack([Fn, Fe]) if cs else np.zeros((3, 1))
        M[np.abs(M) < 5e-13] = 0.0                              # no "-0.00"
        self.Fview.set_matrix(M)
        if cs:
            self.Fview.set_labels(["m_z", "f_x", "f_y"], cols + ecols)
            self.Fview.setFixedHeight(self.Fview.height() + self.Fview.horizontalScrollBar().sizeHint().height())
        self.F_title.setText("Wrench matrix: normal wrenches F_i = ([p]n̂, n̂) for form closure, then the unit "
                             "friction-cone edge wrenches e for force closure.")
        self._update_curve()
        self._update_twist()
        self._redraw()

    def _update_curve(self):
        key = tuple((round(c.p[0], 9), round(c.p[1], 9), round(c.angle, 9)) for c in self.contacts)
        mu_hi = 1.5 if math.isinf(self.mu_min) else max(1.5, min(5.0, 1.3 * self.mu_min))
        key += (tuple(self.poly.ravel()), mu_hi)
        if key != self._curve_key:
            self._curve_key = key
            cs = self.contacts
            if 0 < 2 * len(cs) <= 12:
                ctr, r = self.centroid(), self.char_length()
                mus = np.linspace(0.0, mu_hi, 16)
                q = [gr.grasp_quality(gr.with_mu(cs, mu), ctr, r, 1.0) for mu in mus]
                self.c_q.setData(mus, q)
            else:
                self.c_q.setData([], [])
        self.mu_line.setValue(0.0 if math.isinf(self.mu_min) else self.mu_min)
        self.mu_line.setVisible(not math.isinf(self.mu_min))

    def _update_twist(self):
        V = self.twist()
        mode = gr.contact_mode(self.contacts, V) if self.contacts else ""
        self.mode = mode
        c = gr.cor(V)
        if not np.any(V):
            where = "zero twist (every contact R)"
        elif c is None:
            where = f"translation, direction ({V[1]:+.2f}, {V[2]:+.2f})"
        else:
            where = f"CoR = ({c[0]:+.2f}, {c[1]:+.2f}), {'+' if V[0] > 0 else '−'} (ω_z = {V[0]:+.2f})"
        labels = gr.contact_mode(self.contacts, V, tol=1e-9)
        ok = gr.PENETRATE not in labels
        col = theme.good_color() if ok else theme.bad_color()
        self.twist_lbl.setText(f"{where}<br>contact mode <b style='color:{col}'>{mode or '—'}</b> "
                               + ("(feasible)" if ok else "(X = penetrates: infeasible)"))
        self.twist_lbl.setTextFormat(QtCore.Qt.RichText)

    # ------------------------------------------------------------------ drawing
    def _clear(self):
        for it in self._items:
            self.scene.removeItem(it)
        self._items = []

    def _add(self, it):
        self.scene.addItem(it)
        self._items.append(it)
        return it

    def _poly_item(self, pts, fill, edge=None, width=1.5, z=0):
        qp = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in pts])
        it = QtWidgets.QGraphicsPolygonItem(qp)
        it.setBrush(pg.mkBrush(fill))
        it.setPen(pg.mkPen(edge, width=width) if edge is not None else QtGui.QPen(QtCore.Qt.NoPen))
        it.setZValue(z)
        return self._add(it)

    def _line(self, a, b, color, width=2.0, style=None, z=5):
        kw = dict(color=color, width=width)
        if style is not None:
            kw["style"] = style
        it = pg.PlotCurveItem([a[0], b[0]], [a[1], b[1]], pen=pg.mkPen(**kw))
        it.setZValue(z)
        return self._add(it)

    def _text(self, pos, txt, color, anchor=(0.5, 0.5), z=10):
        it = pg.TextItem(txt, color=color, anchor=anchor)
        it.setPos(float(pos[0]), float(pos[1]))
        it.setZValue(z)
        return self._add(it)

    def _extent(self):
        pts = [self.poly] + [np.array([c.p]) for c in self.contacts]
        P = np.vstack([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((1, 2))
        lo, hi = P.min(axis=0), P.max(axis=0)
        span = float(max(hi - lo)) or 2.0
        return lo, hi, span

    def _autorange(self):
        lo, hi, span = self._extent()
        m = 0.35 * span
        self.scene.setRange(xRange=(lo[0] - m, hi[0] + m), yRange=(lo[1] - m, hi[1] + m), padding=0)

    def _redraw(self, *_):
        self._clear()
        text, muted = theme.text_color(), theme.muted_color()
        lo, hi, span = self._extent()
        L = 0.22 * span                                         # arrow / cone length in scene units
        # feasible CoR region
        if self.cb_cor.isChecked() and self.contacts:
            m = 0.6 * span
            xs = np.linspace(lo[0] - m, hi[0] + m, 220)
            ys = np.linspace(lo[1] - m, hi[1] + m, 220)
            plus, minus = gr.cor_grid(self.contacts, xs, ys)
            img = np.zeros((len(xs), len(ys), 4), np.ubyte)    # ImageItem: axis 0 = x
            img[plus.T] = (*PLUS_RGB, 70)
            img[minus.T] = (*MINUS_RGB, 70)
            img[plus.T & minus.T] = (200, 200, 200, 90)
            self.cor_img.setImage(img, autoLevels=False)
            dx, dy = xs[1] - xs[0], ys[1] - ys[0]
            self.cor_img.setRect(QtCore.QRectF(xs[0] - dx / 2, ys[0] - dy / 2, dx * len(xs), dy * len(ys)))
            self.cor_img.setVisible(True)
        else:
            self.cor_img.setVisible(False)
        # body
        if len(self.poly):
            lc = QtGui.QColor(theme.LINK)
            lc.setAlpha(110)
            self._poly_item(self.poly, lc, theme.LINK, 2.0, z=1)
            if self.hole is not None:
                self._poly_item(self.hole, QtGui.QColor(theme.view_bg()), theme.LINK, 2.0, z=2)
            ctr = self.centroid()
            self._add(pg.ScatterPlotItem([ctr[0]], [ctr[1]], symbol="+", size=12, pen=pg.mkPen(muted, width=2)))
            if self.gravity:
                self._line(ctr, ctr + np.array([0, -1.2 * L]), theme.bad_color(), 2.5, z=6)
                self._add(_arrow_head(ctr + np.array([0, -1.2 * L]), -90.0, theme.bad_color()))
                self._text(ctr + np.array([0.15 * L, -1.2 * L]), "mg", theme.bad_color(), anchor=(0, 0.5))
        # contacts, normals, friction cones
        V = self.twist()
        mode = gr.contact_mode(self.contacts, V) if self.contacts else ""
        labels = [gr.contact_label(c, V) for c in self.contacts]
        for i, c in enumerate(self.contacts):
            col = theme.SERIES[i % len(theme.SERIES)]
            base = c.p - c.n * L                               # finger outside the body, pushing along n
            self._line(base, c.p, col, 3.0, z=7)
            self._add(_arrow_head(c.p, math.degrees(c.angle), col))
            if self.cb_cone.isChecked() and c.mu > 0:
                edges = gr.cone_edges(c)
                wedge = [c.p] + [c.p + e * 1.6 * L for e in edges]
                fc = QtGui.QColor(col)
                fc.setAlpha(60)
                self._poly_item(wedge, fc, col, 1.0, z=4)
            self._add(pg.ScatterPlotItem([c.p[0]], [c.p[1]], size=9, brush=pg.mkBrush(col), pen=pg.mkPen(text)))
            lab = labels[i]
            lcol = theme.bad_color() if lab == gr.PENETRATE else text
            self._text(base - c.n * 0.12 * L, f"{c.name or i + 1}: {lab}", lcol)
        # test twist: CoR marker or translation arrow
        Vn = float(np.linalg.norm(V))
        if Vn > 0:
            cr = gr.cor(V)
            ok = gr.PENETRATE not in mode
            col = theme.good_color() if ok else theme.bad_color()
            if cr is not None:
                self._add(pg.ScatterPlotItem([cr[0]], [cr[1]], symbol="o", size=14, brush=pg.mkBrush(None),
                                             pen=pg.mkPen(col, width=2.5)))
                self._text(cr + np.array([0.08 * L, 0.08 * L]), ("+ CoR" if V[0] > 0 else "− CoR"), col,
                           anchor=(0, 1))
            else:
                ctr = self.centroid()
                d = V[1:] / float(np.linalg.norm(V[1:]))
                tip = ctr + d * 1.3 * L
                self._line(ctr, tip, col, 2.5, QtCore.Qt.DashLine, z=8)
                self._add(_arrow_head(tip, math.degrees(math.atan2(d[1], d[0])), col))
                self._text(tip + d * 0.1 * L, "translation", col)
        self.scene.setTitle(f"Body, contacts and feasible CoRs — mode {mode or '—'}", color=text, size="10pt")


def _arrow_head(tip, angle_deg: float, color) -> pg.ArrowItem:
    """Arrow head at ``tip`` pointing along ``angle_deg`` (ArrowItem's angle is the direction it points from)."""
    a = pg.ArrowItem(angle=180.0 - angle_deg, tipAngle=35, headLen=13, tailLen=0, brush=pg.mkBrush(color),
                     pen=pg.mkPen(color))
    a.setPos(float(tip[0]), float(tip[1]))
    a.setZValue(9)
    return a
