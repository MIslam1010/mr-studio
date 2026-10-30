"""Scene drawing and readout updates for the closed-chains tab (mixin)."""
from __future__ import annotations

import numpy as np

from ... import closed_chains as cc
from .. import theme
from .closed_common import CURVE, LINK_COLS, _deg


class ClosedDrawMixin:
    """Methods mixed into ClosedChainTab (state lives on the tab)."""

    def _draw_ground(self):
        m = self.mech
        v = self.view
        for k, b in enumerate(m.branches):
            G = b.pivot
            h, w = 0.045, 0.03
            tri = np.array([G, G + [-w, -h, 0], G + [w, -h, 0], G]) + [0, 0, -0.02]
            v.set_polyline(f"cc.gnd{k}.tri", tri, (0.75, 0.78, 0.82, 1.0), 2.5)
            hatch = []
            for x in np.linspace(-1.4 * w, 1.4 * w, 7):
                hatch += [G + [x, -h, -0.02], G + [x - 0.012, -h - 0.016, -0.02]]
            hatch += [G + [-1.6 * w, -h, -0.02], G + [1.6 * w, -h, -0.02]]
            v.set_polyline(f"cc.gnd{k}.hatch", hatch, (0.6, 0.63, 0.68, 0.9), 1.5, mode="lines")
        O, D = m.branches[0].pivot, m.branches[1].pivot
        v.set_polyline("cc.gndline", [O + [0, 0, -0.02], D + [0, 0, -0.02]], (0.5, 0.55, 0.6, 0.6), 1.5)

    def _draw_mech(self, th, prefix="cc.m"):
        m, v = self.mech, self.view
        pts = m.points(th)
        r = 0.009
        j = 0
        for bi, P in enumerate(pts):
            for i in range(len(P) - 1):
                v.set_tube(f"{prefix}.link{j}", P[i], P[i + 1], r, LINK_COLS[j % len(LINK_COLS)])
                if i < len(P) - 1:
                    v.set_disk(f"{prefix}.j{bi}.{i}", P[i], [0, 0, 1], 0.016, 0.026, theme.JOINT)
                    v.set_disk(f"{prefix}.pin{bi}.{i}", P[i], [0, 0, 1], 0.006, 0.034, "#1b1e23")
                j += 1
        tip = pts[0][-1]
        v.set_disk(f"{prefix}.cut", tip, [0, 0, 1], 0.021, 0.024, theme.WARN)
        v.set_disk(f"{prefix}.cutpin", tip, [0, 0, 1], 0.008, 0.034, "#1b1e23")
        if self.four:
            A, B, C = pts[0][1], pts[0][2], m.coupler_point(th)
            v.set_tube(f"{prefix}.tri1", A, C, 0.005, LINK_COLS[1])
            v.set_tube(f"{prefix}.tri2", B, C, 0.005, LINK_COLS[1])
            v.set_sphere(f"{prefix}.C", C, 0.012, CURVE)
        return pts

    def _labels_toggled(self, on):
        self.view.set_visible_prefix("cc.lbl", on)
        self._redraw()

    def _redraw(self, *_):
        m, th, v = self.mech, self.theta, self.view
        if m is None or th is None:
            return
        pts = self._draw_mech(th)
        # labels
        if self.cb_labels.isChecked():
            names = (["O", "A", "B"], ["D", "B"]) if self.four else (["O", "A", "P"], ["D", "C", "P"])
            off = np.array([0.018, 0.02, 0.03])
            for bi, P in enumerate(pts):
                for i, nm in enumerate(names[bi]):
                    if bi == 1 and i == len(P) - 1:
                        continue
                    v.set_label(f"cc.lbl{bi}.{i}", P[i] + off, nm, theme.TEXT)
            if self.four:
                v.set_label("cc.lblC", m.coupler_point(th) + off, "C", "#ff8fe0")
            v.set_label("cc.lblcut", pts[0][-1] + np.array([0.025, -0.035, 0.03]), "cut joint", theme.WARN)
        # ghost of the other assembly mode
        other = m.fk_closed_form(th[m.actuated], -m.assembly_mode(th)) if self.cb_ghost.isChecked() else None
        if other is not None and np.max(np.abs(cc.wrap(other - th))) > 1e-6:
            for bi, P in enumerate(m.points(other)):
                v.set_polyline(f"cc.ghost{bi}", P + [0, 0, -0.012], (0.8, 0.85, 1.0, 0.35), 3)
        else:
            v.remove_prefix("cc.ghost")
        if self.four:
            self._redraw_four(th)
        else:
            self._redraw_five(th)
        # constraint Jacobian
        _, A, _ = m.constraint(th)
        self.A_view.set_matrix(A)
        # reaction-force arrow (dynamics)
        if self.lam_show is not None:
            lam, scale = self.lam_show
            v.set_arrow("cc.lam", pts[0][-1] + [0, 0, 0.03], np.r_[lam, 0] * scale, theme.ACCENT, 0.006)
        else:
            v.remove_prefix("cc.lam")

    def _redraw_four(self, th):
        m, v = self.mech, self.view
        if self.cb_curve.isChecked():
            for sg in (1, -1):
                col = CURVE if sg == self.sigma else (1.0, 0.35, 0.85, 0.28)
                for i, c in enumerate(self.curves[sg]):
                    v.set_polyline(f"cc.curve{sg}.{i}", c + [0, 0, -0.016], col, 2.2 if sg == self.sigma else 1.5)
        else:
            v.remove_prefix("cc.curve")
        if len(self.trace) > 1:
            v.set_polyline("cc.trace", np.array(self.trace) + [0, 0, -0.01], (1.0, 0.95, 0.4, 0.9), 2.5)
        else:
            v.remove("cc.trace")
        mu = m.transmission_angle(th)
        det = m.actuator_singularity(th)
        tog = abs(np.sin(th[1]))
        names = "θ₁, θ₂, θ₃"
        self.state_lbl.setText(f"{names} = ({_deg(th[0]):+.1f}°, {_deg(th[1]):+.1f}°, {_deg(th[2]):+.1f}°)\n"
                               f"‖φ(θ)‖ = {np.linalg.norm(m.constraint(th)[0]):.1e} m     σ = {self.sigma:+d}\n"
                               f"transmission angle μ = {_deg(mu):.1f}°   det Aₚ/(L₂L₃) = {det:+.3f}")
        if abs(det) < 0.05:
            txt, col = "ACTUATOR SINGULARITY (dead centre for the crank): coupler ∥ rocker, det Aₚ ≈ 0", theme.BAD
        elif _deg(mu) < 40 or _deg(mu) > 140:
            txt, col = f"Poor transmission: μ = {_deg(mu):.0f}° outside 40°–140°", theme.WARN
        else:
            txt, col = "Well-conditioned: μ within 40°–140°", theme.GOOD
        if tog < 0.03:
            txt += "<br>Toggle position: crank ∥ coupler → rocker at an extreme (dead centre if the rocker drove)"
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")
        x = _deg(th[0])
        for c in self.cursors:
            c.setValue(x)
        self.mu_dot.setData([x], [_deg(mu)])

    def _redraw_five(self, th):
        m, v = self.mech, self.view
        v.remove_prefix("cc.ws")
        if self.cb_ws.isChecked():
            for i, a in enumerate(self.ws_arcs):
                v.set_polyline(f"cc.ws{i}", a + [0, 0, -0.018], (1.0, 0.65, 0.2, 0.85), 2.0)
        if self.elbows not in self.t2:
            self._update_type2()
        if not self.cb_t2.isChecked():
            v.remove_prefix("cc.t2")
        elif not any(k.startswith("cc.t2") for k in v.items_by_key):
            self._update_type2()
        if len(self.trace) > 1:
            v.set_polyline("cc.trace", np.array(self.trace) + [0, 0, -0.01], (1.0, 0.95, 0.4, 0.9), 2.5)
        else:
            v.remove("cc.trace")
        s2, s4, d2 = m.singularity_measures(th)
        P = m.tip(th)
        self.state_lbl.setText(
            f"θ = ({_deg(th[0]):+.1f}°, {_deg(th[1]):+.1f}°, {_deg(th[2]):+.1f}°, {_deg(th[3]):+.1f}°)\n"
            f"P = ({P[0]:+.4f}, {P[1]:+.4f}) m    σ = {self.sigma:+d}\n"
            f"Type I:  sin θ₂ = {s2:+.3f},  sin θ₄ = {s4:+.3f}\nType II: det Aₚ/(L₂L₃) = {d2:+.3f}")
        msgs = []
        worst = 1.0
        if min(abs(s2), abs(s4)) < 0.05:
            msgs.append("TYPE I (serial / end-effector) singularity: a branch is stretched or folded")
        if abs(d2) < 0.05:
            msgs.append("TYPE II (parallel / actuator) singularity: distal links collinear, det Aₚ ≈ 0")
        worst = min(abs(s2), abs(s4), abs(d2))
        if msgs:
            txt, col = "<br>".join(msgs), theme.BAD
        elif worst < 0.2:
            txt, col = "Close to a singularity", theme.WARN
        else:
            txt, col = "Well-conditioned configuration", theme.GOOD
        self.sing_lbl.setText(f"<span style='color:{col}'>● {txt}</span>")
        self.p_dot.setData([P[0]], [P[1]])
        self.sing_hist.append((s2, s4, d2))
        self.sing_hist = self.sing_hist[-400:]
        h = np.array(self.sing_hist)
        x = np.arange(len(h))
        for c, y in zip(self.hist_curves, h.T):
            c.setData(x, y)
