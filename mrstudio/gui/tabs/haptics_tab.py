"""Haptics & impedance tab -- a Hapkit-type 1-DOF haptic paddle rendering sampled-data virtual walls and
springs (impedance and admittance rendering, MR Sec. 11.7), with a user-hand model, the passivity /
stability indicators and the energy-leak plot.  Physics lives in mrstudio.haptics (SBEG 461 Week 12)."""
from __future__ import annotations

import importlib.util
import math

import modern_robotics as mr
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...references import further_reading
from ... import haptics as hp
from .. import theme
from ..base import BaseTab
from ..viewer import SceneView
from ..widgets import SliderRow, button, checkbox, combo, group, hbox, pen, plot

THEORY = r"""
<h2>Haptic rendering: impedance, admittance and the sampled wall</h2>
<span class="ref">Modern Robotics, Sec. 11.7 (impedance control, pp. 443–447) with Sec. 11.6 (hybrid motion–force,
p. 439) and Sec. 11.8 (low-level torque control, p. 447)</span>
<p>Sec. 11.6 treats contact as an ideal constraint: infinite impedance in the motion directions, zero in the
force directions.  A haptic device instead <b>renders</b> a chosen impedance to the user's hand.  A
1-DOF device rendering a mass–spring–damper (p. 444):</p>
$$m\ddot{x}+b\dot{x}+kx=f\quad(11.62)$$
$$Z(s)=\frac{F(s)}{X(s)}=ms^{2}+bs+k,\qquad Y(s)=Z^{-1}(s)\quad(11.63)$$
<p>High impedance: small $\Delta x$ for a given $\Delta f$ (a good motion controller); low impedance: small
$\Delta f$ for a given $\Delta x$ (a good force controller).  The task-space goal (p. 445):</p>
$$M\ddot{x}+B\dot{x}+Kx=f_{ext}\quad(11.64)$$
<h3>Impedance rendering (11.7.1) — motion in, force out</h3>
$$\tau=J^{T}(\theta)\left(\tilde{\Lambda}(\theta)\ddot{x}+\tilde{\eta}(\theta,\dot{x})-(M\ddot{x}+B\dot{x}+Kx)\right)\quad(11.65)$$
<p>For the paddle $J=r_{h}$ and, as is common (p. 446), $M=0$ and no mass compensation: the user feels the
device's own mass $m_{h}$.  The wall is a unilateral spring–damper about $x_{w}$, computed every $T$ seconds
from the encoder position and a backward-difference velocity, then held (zero-order hold):</p>
$$F_{k}=-K(x_{k}-x_{w})-B\,\frac{x_{k}-x_{k-1}}{T}$$
<h3>Admittance rendering (11.7.2) — force in, motion out</h3>
$$\ddot{x}_{d}=M^{-1}(f_{ext}-B\dot{x}-Kx)\quad(11.66)$$
<p>A (simulated) force sensor measures $f_{ext}$; a virtual mass $M_{v}$ is integrated and the device follows it
with $F=m_{h}\ddot{x}_{d}+k_{g}x-f_{ext}+K_{p}(x_{d}-x)$.  The Hapkit has no force sensor: admittance suits
stiff, geared arms (casualty cradle); on a light device the inner loop $K_{p}$ is itself a sampled spring.</p>
<h3>The device (Sec. 11.8, Ch. 8)</h3>
<p>Handle lever $r_{h}$, sector/pulley ratio $N=r_{s}/r_{p}$.  From <b>mr.MassMatrix</b> and
<b>mr.GravityForces</b> of the paddle's screw model:</p>
$$m_{h}=\frac{\mathcal{I}_{pivot}+N^{2}J_{rotor}}{r_{h}^{2}},\qquad k_{g}=\frac{1}{r_{h}^{2}}\frac{\partial g}{\partial\theta}(0),\qquad i=\frac{F\,r_{h}}{N\,k_{t}}$$
<h3>Why stiff walls buzz — energy leak</h3>
<p>MR p. 446: a large $K$, "coupled with delays, sensor quantization, and sensor errors, can lead to
oscillatory behavior or instability".  A spring that is held for $T$ and delayed $T_{d}$ acts on an old
position; to first order it adds a <b>negative damper</b> $-K(T/2+T_{d})$:</p>
$$K_{max}\approx\frac{b+B}{T/2+T_{d}}$$
<p>The exact limit comes from the eigenvalues of the discrete-time closed loop (ZOH plant, backward-difference
velocity, $d$ samples of delay): stable iff $\rho(A)<1$.</p>
<p><b>Passivity (cited result).</b>  Colgate &amp; Schenkel (1997) showed that the sampled virtual wall is passive
— stable when coupled to <i>any</i> passive hand — if the device's physical damping satisfies</p>
$$b>\frac{KT}{2}+|B|$$
<p>(virtual damping computed from sampled positions does <i>not</i> help).  The energy plot measures the leak
$\int(F_{applied}-F_{ideal})\,\dot{x}\,dt$ — the work of the real held force beyond the ideal continuous
impedance on the same trajectory — against the device damping $\int b\dot{x}^{2}dt$.  Passive: leak − damping ≤ 0.
In admittance mode the ideal force is the one that turns the handle into the virtual mass,
$F_{ideal}=F_{env}+k_{g}x+(m_{h}-M_{v})\ddot{x}$; the inner tracking loop then also cancels the device damping
(leak ≈ $\int b\dot{x}^{2}dt$), so $b$ no longer protects passivity.  The stability verdict uses the exact
matrix of the whole admittance loop (virtual mass, force feedback, inner $K_{p}$).</p>
<p><b>Presets</b> follow the SBEG 461 project: S1 rehab virtual tunnel (HW1-2, $K_{n}$ = 300 N/m) and S6
casualty-handling soft contact (HW6, $K$ = 100 N/m, $B$ = 2 N·s/m).  Device numbers are illustrative.</p>
<p><span class="ref">J. E. Colgate and G. G. Schenkel, Passivity of a class of sampled-data systems: application to
haptic interfaces, J. Robotic Systems 14(1):37–47, 1997.</span></p>
"""
THEORY += further_reading([
    ('mls', 'Ch. 4 §6, pp. 200–206', 'constrained manipulators, hybrid force/position control'),
    ('astrom_murray', '§6.3, pp. 183–184', 'second-order systems: damping ratio, natural frequency'),
    ('astrom_murray', '§12.1–12.2, pp. 347–357', 'model uncertainty, delay and robust stability'),
])

COURSE_HAPKIT = "Hapkit (1-DOF haptic paddle)"
MM = 1e3
SPEEDS = {"real time": 1.0, "¼ speed": 0.25, "⅒ speed": 0.1}
KINDS = {"Gripping hand": "hand", "Soft surface (unilateral)": "surface", "Hand off (release)": "none"}
KINDS_INV = {v: k for k, v in KINDS.items()}
SHAPES = {"Unilateral wall": "wall", "Spring / tunnel (bilateral)": "spring"}
SHAPES_INV = {v: k for k, v in SHAPES.items()}


def _registry_note(dev: hp.HapticDevice) -> str:
    """Compare with the course robot if another module registered it (graceful fallback otherwise)."""
    try:
        from ...robots import ROBOTS, get_robot
        if COURSE_HAPKIT not in ROBOTS:
            return f"Course robot “{COURSE_HAPKIT}” not registered — using the built-in course numbers."
        r = get_robot(COURSE_HAPKIT)
        Ij = float(mr.MassMatrix(np.zeros(r.n), r.Mlist, r.Glist, r.Slist)[0, 0])
        return (f"Course robot “{COURSE_HAPKIT}”: mr.MassMatrix → I_pivot = {Ij * 1e4:.3f}e-4 kg·m² "
                f"(this tab: {dev.joint_inertia * 1e4:.3f}e-4).")
    except Exception as e:  # pragma: no cover - registry robot malformed
        return f"Course robot registry not usable ({type(e).__name__}) — using built-in numbers."


class HapticsTab(BaseTab):
    title = "⑪ Haptics"
    uses_serial_robot = False

    def __init__(self, ctx):
        super().__init__(ctx)
        self.dev = hp.default_device()
        self.sim: hp.SimResult | None = None
        self.view = SceneView()
        self._loading = False

        # ---------------- scenario
        self.preset_box = combo(list(hp.PRESETS), self.load_preset)
        self.preset_lbl = QtWidgets.QLabel()
        self.preset_lbl.setWordWrap(True)
        self.preset_lbl.setObjectName("muted")

        # ---------------- rendering
        self.mode_box = combo(["Impedance (11.65)", "Admittance (11.66)"], self._changed)
        self.shape_box = combo(list(SHAPES), self._changed)
        self.K = SliderRow("K", 0, 3000, 250, decimals=0, unit=" N/m")
        self.B = SliderRow("B", 0, 5, 0, decimals=2, unit=" N·s/m")
        self.T = SliderRow("T", 0.1, 10, 1.0, decimals=2, unit=" ms")
        self.delay = SliderRow("T_d", 0, 10, 0, decimals=0, unit=" samples")
        self.Mv = SliderRow("M_v", 0.02, 2.0, 0.3, decimals=2, unit=" kg")
        self.Kp = SliderRow("K_p", 0, 3000, 250, decimals=0, unit=" N/m")
        self.K.setToolTip("virtual stiffness (wall / spring)")
        self.B.setToolTip("virtual damping, from the backward-difference velocity")
        self.T.setToolTip("controller sample period (zero-order hold)")
        self.delay.setToolTip("extra loop delay, in samples of T")
        self.Mv.setToolTip("admittance: rendered virtual mass")
        self.Kp.setToolTip("admittance: stiffness of the inner position loop that tracks the virtual mass")
        self.cb_quant = checkbox("Encoder quantisation (Δx = 0.02 mm)", self._changed, True)
        self.cb_sat = checkbox("Force saturation (|F| ≤ 2 N)", self._changed, True)

        # ---------------- hand
        self.kind_box = combo(list(KINDS), self._changed)
        self.kH = SliderRow("k_H", 0, 3000, 150, decimals=0, unit=" N/m")
        self.bH = SliderRow("b_H", 0, 10, 1.0, decimals=2, unit=" N·s/m")
        self.mH = SliderRow("m_H", 0, 0.5, 0.05, decimals=3, unit=" kg")
        self.kH.setToolTip("hand grip stiffness (or surface stiffness)")
        for w in (self.K, self.B, self.T, self.delay, self.Mv, self.Kp, self.kH, self.bH, self.mH):
            w.valueChanged.connect(self._changed)

        # ---------------- indicators
        self.stab_lbl, self.pass_lbl, self.sim_lbl = (QtWidgets.QLabel() for _ in range(3))
        for lb in (self.stab_lbl, self.pass_lbl, self.sim_lbl):
            lb.setWordWrap(True)
            lb.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.dev_lbl = QtWidgets.QLabel(self.dev.summary() + "<br><i>" + _registry_note(self.dev) + "</i>")
        self.dev_lbl.setWordWrap(True)
        self.dev_lbl.setObjectName("muted")

        # ---------------- playback + hardware
        self.play_btn = button("❚❚ Pause", checkable=True)
        self.play_btn.setChecked(True)
        self.play_btn.toggled.connect(self._toggle_play)
        self.speed_box = combo(list(SPEEDS))
        self.speed_box.setCurrentText("¼ speed")
        self.hw_btn = button("Connect Hapkit (serial)…", self._connect_serial)
        self._serial = None
        if importlib.util.find_spec("serial") is None:
            self.hw_btn.setEnabled(False)
            self.hw_btn.setToolTip("pyserial is not installed (pip install pyserial). "
                                   "The simulation works without hardware.")
        else:
            self.hw_btn.setToolTip("Read-only: streams the handle position (one number per line, in mm) "
                                   "from the Hapkit firmware and shows it as a ghost handle. Sends nothing.")

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(group("Scenario", self.preset_box, self.preset_lbl))
        self.adm_rows = (self.Mv, self.Kp)
        cl.addWidget(group("Rendered environment (sampled)", self.mode_box, self.shape_box,
                           self.K, self.B, self.T, self.delay, self.Mv, self.Kp, self.cb_quant, self.cb_sat))
        cl.addWidget(group("User hand / contact", self.kind_box, self.kH, self.bH, self.mH))
        cl.addWidget(group("Stability && passivity", self.stab_lbl, self.pass_lbl, self.sim_lbl))
        cl.addWidget(group("Playback && hardware", hbox(self.play_btn, self.speed_box), self.hw_btn))
        cl.addWidget(group("Device (Hapkit-type, illustrative)", self.dev_lbl))
        cl.addStretch(1)

        # ---------------- plots
        self.p_x = plot("Handle position", "x [mm]")
        self.p_f = plot("Force on the handle", "F [N]")
        self.p_e = plot("Energy leak vs device damping", "")
        self.p_e.setLabel("left", "E", units="J")
        self.p_m = plot("Stability map (log–log)", "K [N/m]", "T [ms]")
        self.p_m.setLogMode(x=True, y=True)
        self.p_m.getAxis("bottom").setTicks([[(math.log10(v), f"{v:g}") for v in (0.1, 0.2, 0.5, 1, 2, 5, 10)], []])
        for p in (self.p_f, self.p_e):
            p.setXLink(self.p_x)
        self.c_x = self.p_x.plot(pen=pen(0), name="x")
        self.c_xr = self.p_x.plot(pen=pen(1, 1.5, "dash"), name="hand x_ref / surface")
        self.c_xs = self.p_x.plot(pen=pg.mkPen(theme.MUTED, width=1.5, style=QtCore.Qt.DotLine), name="wall / set point")
        self.c_xv = self.p_x.plot(pen=pen(4, 1.5, "dash"), name="virtual mass x_d")
        self.c_F = self.p_f.plot(pen=pen(0), name="F applied (motor)")
        self.c_fh = self.p_f.plot(pen=pen(1, 1.5), name="f hand / surface")
        for s in (+1, -1):
            self.p_f.addItem(pg.InfiniteLine(pos=s * self.dev.force_max, angle=0,
                                             pen=pg.mkPen(theme.BAD, width=1, style=QtCore.Qt.DashLine)))
        self.c_leak = self.p_e.plot(pen=pen(3), name="leak ∫(F−F_ideal)ẋ dt")
        self.c_D = self.p_e.plot(pen=pen(2), name="device damping ∫bẋ² dt")
        self.c_net = self.p_e.plot(pen=pg.mkPen(theme.WARN, width=2.5), name="net = leak − damping")
        self.p_e.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.MUTED, width=1)))
        self.c_mdev = self.p_m.plot(pen=pen(0), name="exact ρ(A)=1, device alone")
        self.c_mhand = self.p_m.plot(pen=pen(1, 2, "dash"), name="exact, with this hand")
        self.c_mpass = self.p_m.plot(pen=pg.mkPen(theme.GOOD, width=2), name="passive: b > KT/2 + |B|")
        self.c_mnow = self.p_m.plot(pen=None, symbol="o", symbolSize=10, symbolBrush=theme.WARN, name="current")
        self.cursors = []
        for p in (self.p_x, self.p_f, self.p_e):
            ln = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(theme.TEXT, width=1, style=QtCore.Qt.DashLine))
            p.addItem(ln)
            self.cursors.append(ln)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.p_x, 0, 0)
        grid.addWidget(self.p_f, 0, 1)
        grid.addWidget(self.p_e, 1, 0)
        grid.addWidget(self.p_m, 1, 1)
        bottom = QtWidgets.QWidget()
        bottom.setLayout(grid)

        self.assemble(controls, self.view, bottom, THEORY, controls_width=370, bottom_ratio=(1, 1))
        self.view.frame_scene(0.21, planar=True, center=(0, 0.035, 0))

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.recompute)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._anim_step)
        self.serial_timer = QtCore.QTimer(self)
        self.serial_timer.timeout.connect(self._read_serial)
        self.t_play = 0.0
        self._map_key = None
        self.load_preset(self.preset_box.currentText())

    # ------------------------------------------------------------------ parameters
    def load_preset(self, name: str):
        if name not in hp.PRESETS:
            return
        ren, hand, t_end, x0 = hp.preset(name)
        self._loading = True
        self.preset_lbl.setText(hp.PRESETS[name]["text"])
        self.mode_box.setCurrentIndex(0 if ren.mode == "impedance" else 1)
        self.shape_box.setCurrentText(SHAPES_INV[ren.shape])
        self.K.set_value(ren.K)
        self.B.set_value(ren.B)
        self.T.set_value(ren.T * 1e3)
        self.delay.set_value(ren.delay_steps)
        self.Mv.set_value(ren.M_v)
        self.Kp.set_value(ren.Kp)
        self.kind_box.setCurrentText(KINDS_INV[hand.kind])
        self.kH.set_value(hand.k)
        self.bH.set_value(hand.b)
        self.mH.set_value(hand.m)
        self._base_ren, self._base_hand, self.t_end, self.x0 = ren, hand, t_end, x0
        self._changed()
        self._loading = False
        self.recompute()

    def current(self) -> tuple[hp.Rendering, hp.Hand]:
        from dataclasses import replace
        ren = replace(self._base_ren,
                      mode="impedance" if self.mode_box.currentIndex() == 0 else "admittance",
                      shape=SHAPES[self.shape_box.currentText()], K=self.K.value(), B=self.B.value(),
                      T=self.T.value() * 1e-3, delay_steps=int(round(self.delay.value())),
                      M_v=self.Mv.value(), Kp=self.Kp.value(),
                      quantize=self.cb_quant.isChecked(), saturate=self.cb_sat.isChecked())
        kind = KINDS[self.kind_box.currentText()]
        hand = replace(self._base_hand, kind=kind, k=self.kH.value(), b=self.bH.value(), m=self.mH.value())
        if kind == "surface" and self._base_hand.kind != "surface":
            hand = replace(hand, x_start=ren.x0 + 0.010)        # pad 10 mm beyond the set point
        return ren, hand

    def _changed(self, *_):
        adm = self.mode_box.currentIndex() == 1
        for w in self.adm_rows:
            w.setVisible(adm)
        hand_on = KINDS.get(self.kind_box.currentText()) != "none"
        for w in (self.kH, self.bH):
            w.setEnabled(hand_on)
        self.mH.setEnabled(KINDS.get(self.kind_box.currentText()) == "hand")
        if not self._loading:
            self._debounce.start(60)

    # ------------------------------------------------------------------ computation
    def recompute(self):
        self._changed_visibility()
        ren, hand = self.current()
        self.ren, self.hand = ren, hand
        self.sim = hp.simulate(self.dev, ren, hand, self.t_end, x0=self.x0)
        self.rep = hp.analyse(self.dev, ren, hand)
        self._update_indicators()
        self._update_plots()
        self._update_map()
        self.t_play = 0.0
        self._draw(0)

    def _changed_visibility(self):
        adm = self.mode_box.currentIndex() == 1
        for w in self.adm_rows:
            w.setVisible(adm)

    def _update_indicators(self):
        r, ren, sim = self.rep, self.ren, self.sim
        good, bad = theme.GOOD, theme.BAD
        what = "inner loop K_p" if ren.mode == "admittance" else "K"
        self.stab_lbl.setText(
            f"<span style='color:{(theme.WARN if abs(r.rho - 1) < 1e-6 else good) if r.stable else bad}'>"
            f"{('~ marginally stable' if abs(r.rho - 1) < 1e-6 else '✓ stable') if r.stable else '✗ unstable'}</span>"
            f" — exact sampled loop ({what}{', hand in contact' if self.hand.kind == 'hand' else ''}):"
            f" ρ(A) = {r.rho:.4f}<br>"
            f"{'K<sub>max</sub>' if ren.mode == 'impedance' else 'K<sub>p,max</sub>'} exact: "
            + (f"{r.K_max_exact:.0f} N/m with this hand, " if self.hand.kind == "hand" else "")
            + f"{r.K_max_device:.0f} N/m device alone"
            + (f"; estimate (b+B)/(T/2+T<sub>d</sub>) = {r.K_estimate:.0f} N/m" if ren.mode == "impedance" else "")
            + ("<br><i>soft surface not included (contact is intermittent)</i>" if self.hand.kind == "surface" else ""))
        self.stab_lbl.setTextFormat(QtCore.Qt.RichText)
        if ren.delay_steps:
            ptxt = (f"<span style='color:{theme.WARN}'>? passivity condition stated for no added delay</span> — "
                    f"margin without delay b − (KT/2+|B|) = {r.margin:+.3f} N·s/m")
        else:
            ptxt = (f"<span style='color:{good if r.passive else bad}'>{'✓ passive' if r.passive else '✗ not guaranteed passive'}"
                    f"</span> (Colgate &amp; Schenkel 1997): b = {self.dev.b:.3f} "
                    f"{'&gt;' if r.margin > 0 else '≤'} KT/2 + |B| = {r.K_eff * ren.T / 2 + abs(r.B_eff):.3f} N·s/m"
                    f" → K<sub>passive,max</sub> = {r.K_passive:.0f} N/m")
        self.pass_lbl.setText(ptxt)
        self.pass_lbl.setTextFormat(QtCore.Qt.RichText)
        if np.all(np.isfinite(sim.x)):
            g = hp.oscillation_growth(sim)
            net = float(np.max(sim.net)) * 1e3
            nsat = int(np.sum(sim.saturated))
            self.sim_lbl.setText(
                f"Simulation: peak |F| = {np.max(np.abs(sim.F_app)):.2f} N, final F = {sim.F_app[-1]:+.2f} N, "
                f"oscillation growth {g:.2f} ({'decaying' if g < 1 else 'GROWING'}); "
                f"max net energy {net:+.3f} mJ; saturated {nsat} samples")
        else:
            self.sim_lbl.setText("Simulation diverged.")

    def _update_plots(self):
        s = self.sim
        t = s.t
        self.c_x.setData(t, s.x * MM, connect="finite")
        if self.hand.kind == "none":
            self.c_xr.setData([], [])
        else:
            self.c_xr.setData(t, s.x_ref * MM, connect="finite")
        self.c_xs.setData(t, s.x_set * MM)
        if self.ren.mode == "admittance":
            self.c_xv.setData(t, s.x_v * MM, connect="finite")
        else:
            self.c_xv.setData([], [])
        self.c_F.setData(t, s.F_app, connect="finite")
        self.c_fh.setData(t, s.f_hand, connect="finite")
        self.c_leak.setData(t, s.leak, connect="finite")
        self.c_D.setData(t, s.D_b, connect="finite")
        self.c_net.setData(t, s.net, connect="finite")
        for p in (self.p_x, self.p_f, self.p_e):
            p.enableAutoRange()

    def _update_map(self):
        ren, hand = self.ren, self.hand
        B = ren.B if ren.mode == "impedance" else ren.Kd
        adm = ren.mode == "admittance"
        key = (B, ren.delay_steps, hand.kind, hand.m, hand.b, hand.k) + ((ren.K, ren.B, ren.M_v) if adm else ())
        if key != self._map_key:
            self._map_key = key
            mp = hp.stability_map(self.dev, B, hand, ren.delay_steps, ren=ren)
            self.p_m.setLabel("left", "K_p [N/m]" if adm else "K [N/m]")

            def pos(a):
                a = np.asarray(a, float).copy()
                a[a <= 0] = np.nan
                return a
            T = mp["T"] * 1e3
            self.c_mdev.setData(T, pos(mp["exact_device"]), connect="finite")
            self.c_mpass.setData(T, pos(mp["passive"]), connect="finite")
            if "exact_hand" in mp:
                self.c_mhand.setData(T, pos(mp["exact_hand"]), connect="finite")
            else:
                self.c_mhand.setData([], [])
        Kc = ren.K if ren.mode == "impedance" else ren.Kp
        self.c_mnow.setData([ren.T * 1e3], [max(Kc, 1.0)])

    # ------------------------------------------------------------------ drawing
    def _draw(self, k: int):
        s, dev = self.sim, self.dev
        if s is None:
            return
        k = int(np.clip(k, 0, len(s.t) - 1))
        x = s.x[k]
        if not np.isfinite(x):
            return
        for ln in self.cursors:
            ln.setValue(s.t[k])
        v = self.view
        rh, rs, rp = dev.rh, dev.rs, dev.rp
        phi = x / rh

        def R(p, a=phi):          # rotate body point clockwise by a (x to the right is +)
            c, si = math.cos(a), math.sin(a)
            return np.array([c * p[0] + si * p[1], -si * p[0] + c * p[1], 0.0])

        def tip_x(xx):            # horizontal position of the grip for a handle coordinate xx
            return rh * math.sin(xx / rh)

        # base, sector, handle, pulley
        v.set_tube("hp.base", (0, -rs - 0.02, -0.004), (0, -rs - 0.02, 0.004), 0.03, theme.BASE)
        v.set_tube("hp.post", (0, -rs - 0.012, 0), (0, 0, 0), 0.004, theme.BASE)
        arc = [R((rs * math.sin(a), -rs * math.cos(a))) for a in np.linspace(-0.75, 0.75, 40)]
        v.set_polyline("hp.sector", [R((0, 0))] + arc + [R((0, 0))], theme.LINK, 3)
        v.set_tube("hp.handle", R((0, -0.005)), R((0, rh)), 0.0035, theme.LINK)
        v.set_sphere("hp.pivot", (0, 0, 0), 0.006, theme.JOINT)
        pc = np.array([0, -rs - rp, 0])
        v.set_disk("hp.pulley", pc, (0, 0, 1), rp, 0.004, theme.JOINT)
        ang = phi * dev.ratio
        v.set_polyline("hp.spoke", [pc, pc + rp * 1.6 * np.array([math.sin(ang), math.cos(ang), 0])], theme.TEXT, 2)
        grip = R((0, rh))
        v.set_sphere("hp.grip", grip, 0.007, theme.ACCENT)

        # rendered environment
        ren, hand = self.ren, self.hand
        y0, y1 = rh - 0.022, rh + 0.022
        xs = s.x_set[k]
        if ren.shape == "wall":
            xw = tip_x(xs)
            side = ren.side
            hatch = [(xw, y0, 0), (xw, y1, 0)]
            for yy in np.linspace(y0, y1, 9):
                hatch += [(xw, yy, 0), (xw + side * 0.008, yy - 0.005, 0)]
            v.set_polyline("hp.env", hatch, (0.9, 0.55, 0.3, 0.95), 2, mode="lines")
        else:
            xw = tip_x(xs)
            v.set_polyline("hp.env", [(xw, y0, 0), (xw, y1, 0), (xw - 0.004, y1, 0), (xw + 0.004, y1, 0)],
                           (0.9, 0.55, 0.3, 0.95), 2, mode="lines")
            v.set_polyline("hp.spring", _zigzag((xw, grip[1] + 0.012), (grip[0], grip[1] + 0.012)),
                           (0.9, 0.55, 0.3, 0.9), 2)
        if ren.shape == "wall":
            v.remove("hp.spring")
        v.set_label("hp.envlbl", (xw - 0.012, y1 + 0.006, 0),
                    "virtual wall" if ren.shape == "wall" else "set point", "#f0a060", 10)

        # hand / surface
        if hand.kind == "hand":
            xr = tip_x(s.x_ref[k])
            v.set_sphere("hp.hd.ref", (xr, grip[1] - 0.025, 0), 0.008, (1.0, 0.75, 0.6, 0.35))
            v.set_polyline("hp.hd.spring", _zigzag((xr, grip[1] - 0.025), (grip[0], grip[1] - 0.006)),
                           (1.0, 0.75, 0.6, 0.85), 2)
            v.set_label("hp.hd.lbl", (xr + 0.01, grip[1] - 0.03, 0), "hand", "#ffc39a", 10)
            v.remove("hp.surface")
            v.remove("hp.padlbl")
        elif hand.kind == "surface":
            xp = tip_x(hand.x_start)
            pts = []
            for yy in np.linspace(y0, y1, 7):
                pts += [(xp, yy, 0), (xp + 0.012, yy, 0)]
            pts += [(xp, y0, 0), (xp, y1, 0)]
            v.set_polyline("hp.surface", pts, (1.0, 0.6, 0.75, 0.9), 3, mode="lines")
            v.set_label("hp.padlbl", (xp + 0.004, y0 - 0.008, 0), "pad", "#ff9fc0", 10)
            v.remove_prefix("hp.hd.")
        else:
            v.remove_prefix("hp.hd.")
            v.remove("hp.surface")
            v.remove("hp.padlbl")
        if self.ren.mode == "admittance" and np.isfinite(s.x_v[k]):
            xv = tip_x(s.x_v[k])
            v.set_sphere("hp.virtual", (xv, grip[1] + 0.012, 0), 0.005, (0.7, 0.57, 0.94, 0.8))
        else:
            v.remove("hp.virtual")

        # force arrow (motor force on the handle)
        F = s.F_app[k]
        v.set_arrow("hp.F", grip + np.array([0, 0.012, 0]), np.array([F * 0.02, 0, 0]),
                    theme.BAD if s.saturated[k] else theme.WARN, 0.0018)
        v.set_label("hp.info", (-0.105, rh + 0.035, 0),
                    f"t = {s.t[k]:.2f} s   x = {x * MM:+.2f} mm   F = {F:+.2f} N"
                    + ("  (saturated)" if s.saturated[k] else ""), theme.TEXT, 11)

    def _anim_step(self):
        if self.sim is None:
            return
        self.t_play += 0.03 * SPEEDS.get(self.speed_box.currentText(), 0.25)
        if self.t_play > self.sim.t[-1]:
            self.t_play = 0.0
        self._draw(int(self.t_play / self.ren.T))

    def _toggle_play(self, on):
        self.play_btn.setText("❚❚ Pause" if on else "▶ Play")
        if on and self.isVisible():
            self.timer.start(30)
        else:
            self.timer.stop()

    def on_show(self):
        if self.play_btn.isChecked():
            self.timer.start(30)

    def on_hide(self):
        self.timer.stop()

    # ------------------------------------------------------------------ hardware (optional, read-only)
    def _connect_serial(self):
        if self._serial is not None:
            self.serial_timer.stop()
            try:
                self._serial.close()
            finally:
                self._serial = None
            self.hw_btn.setText("Connect Hapkit (serial)…")
            self.view.remove("hp.hw")
            return
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            return
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            QtWidgets.QMessageBox.information(self, "Hapkit", "No serial port found.")
            return
        port, ok = QtWidgets.QInputDialog.getItem(self, "Hapkit", "Serial port (115200 baud):", ports, 0, False)
        if not ok:
            return
        try:
            self._serial = serial.Serial(port, 115200, timeout=0)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hapkit", f"Could not open {port}: {e}")
            return
        self._buf = b""
        self.hw_btn.setText(f"Disconnect {port}")
        self.serial_timer.start(30)

    def _read_serial(self):
        try:
            self._buf += self._serial.read(4096)
        except Exception:
            return self._connect_serial()
        *lines, self._buf = self._buf.split(b"\n")
        for ln in reversed(lines):
            try:
                x = float(ln.strip().split(b",")[0]) / MM
            except ValueError:
                continue
            g = np.array([self.dev.rh * math.sin(x / self.dev.rh), self.dev.rh * math.cos(x / self.dev.rh), 0])
            self.view.set_sphere("hp.hw", g, 0.009, (0.3, 1.0, 0.6, 0.5))
            break


def _zigzag(a, b, n: int = 8, amp: float = 0.003) -> list:
    """Zig-zag spring polyline between 2-D points a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return [(*a, 0), (*b, 0)]
    u = d / L
    nrm = np.array([-u[1], u[0]])
    pts = [a, a + 0.15 * d]
    for i in range(n):
        f = 0.15 + 0.7 * (i + 0.5) / n
        pts.append(a + f * d + (amp if i % 2 == 0 else -amp) * nrm)
    pts += [a + 0.85 * d, b]
    return [(p[0], p[1], 0.0) for p in pts]
