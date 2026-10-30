"""Plot construction for the closed-chains tab (mixin)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from .. import theme
from ..widgets import pen
from .closed_common import _break_jumps


class ClosedPlotsMixin:
    """Methods mixed into ClosedChainTab (state lives on the tab)."""

    def _init_kin_plots_four(self):
        for p in (self.pk1, self.pk2):
            p.clear()
            p.setLogMode(False, False)
            p.setAspectLocked(False)
            p.enableAutoRange()
        m = self.mech
        self.pk1.setTitle("Transmission angle μ(θ₁)", color=theme.TEXT, size="10pt")
        self.pk1.setLabel("left", "μ [°]")
        self.pk1.setLabel("bottom", "crank angle θ₁ [°]")
        band = pg.LinearRegionItem((40, 140), orientation="horizontal", movable=False,
                                   brush=pg.mkBrush(87, 214, 141, 28), pen=pg.mkPen(None))
        self.pk1.addItem(band)
        th1 = np.linspace(-np.pi, np.pi, 721)
        AD2 = m.L1 ** 2 + m.d ** 2 - 2 * m.L1 * m.d * np.cos(th1)
        cmu = (m.L2 ** 2 + m.L3 ** 2 - AD2) / (2 * m.L2 * m.L3)
        mu = np.where(np.abs(cmu) <= 1, np.degrees(np.arccos(np.clip(cmu, -1, 1))), np.nan)
        self.pk1.plot(np.degrees(th1), mu, pen=pen(0, 2.5), name="μ (both σ)", connect="finite")
        self.pk1.addLine(y=90, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.DotLine))
        self.mu_dot = self.pk1.plot([], [], pen=None, symbol="o", symbolBrush=theme.WARN, symbolSize=9)
        self.pk1.setYRange(0, 180)
        self.pk2.setTitle("Passive joints from loop closure", color=theme.TEXT, size="10pt")
        self.pk2.setLabel("left", "angle [°]")
        self.pk2.setLabel("bottom", "crank angle θ₁ [°]")
        self.pass_curves = [self.pk2.plot([], [], pen=pen(1, 2.2), name="θ₂ (coupler)", connect="finite"),
                            self.pk2.plot([], [], pen=pen(2, 2.2), name="θ₃ (rocker)", connect="finite"),
                            self.pk2.plot([], [], pen=pen(1, 1.2, "dash"), connect="finite"),
                            self.pk2.plot([], [], pen=pen(2, 1.2, "dash"), connect="finite")]
        self.cursors = [p.addLine(x=0, pen=pg.mkPen(theme.WARN, width=1)) for p in (self.pk1, self.pk2)]
        self._update_kin_plots_four()

    def _update_kin_plots_four(self):
        if not self.four:
            return
        for k, sg in enumerate((self.sigma, -self.sigma)):
            th1, TH = self.sweeps[sg]
            x = np.degrees(th1)
            self.pass_curves[2 * k].setData(x, _break_jumps(np.degrees(TH[:, 1])))
            self.pass_curves[2 * k + 1].setData(x, _break_jumps(np.degrees(TH[:, 2])))

    def _init_kin_plots_five(self):
        for p in (self.pk1, self.pk2):
            p.clear()
            p.setLogMode(False, False)
            p.enableAutoRange()
        self.pk1.setTitle("Singularity measures (history)", color=theme.TEXT, size="10pt")
        self.pk1.setLabel("left", "")
        self.pk1.setLabel("bottom", "update #")
        self.hist_curves = [self.pk1.plot([], [], pen=pen(1, 2), name="sin θ₂ (Type I, left)"),
                            self.pk1.plot([], [], pen=pen(2, 2), name="sin θ₄ (Type I, right)"),
                            self.pk1.plot([], [], pen=pen(3, 2.5), name="det Aₚ/(L₂L₃) (Type II)")]
        self.pk1.addLine(y=0, pen=pg.mkPen(theme.BAD, style=QtCore.Qt.DashLine))
        self.pk1.setYRange(-1.05, 1.05)
        self.pk2.setTitle("|det Aₚ| over the workspace (this working mode; red = Type II)", color=theme.TEXT,
                          size="10pt")
        self.pk2.setLabel("left", "y [m]")
        self.pk2.setLabel("bottom", "x [m]")
        self.img = pg.ImageItem()
        cmap = pg.ColorMap([0.0, 0.12, 0.35, 1.0], [(255, 70, 70), (255, 190, 70), (60, 110, 170), (18, 40, 80)])
        self.img.setLookupTable(cmap.getLookupTable(0, 1, 256))
        self.pk2.addItem(self.img)
        self.img_lines = []
        self.p_dot = self.pk2.plot([], [], pen=None, symbol="+", symbolSize=16, symbolPen=pg.mkPen("w", width=2))
        self.pk2.setAspectLocked(True)
        for ax in ("left", "bottom"):
            self.pk2.getAxis(ax).enableAutoSIPrefix(False)
        self.cursors = []

    def _init_dyn_plots(self):
        self.E_curves = [self.pE.plot([], [], pen=pen(0, 2), name="kinetic K"),
                         self.pE.plot([], [], pen=pen(1, 2), name="potential P"),
                         self.pE.plot([], [], pen=pen(2, 2.5), name="K + P"),
                         self.pE.plot([], [], pen=pen(5, 2, "dash"), name="K + P − W_motor + W_damping")]
        self.phi_curve = self.pphi.plot([], [], pen=pen(3, 2))
        self.lam_curves = [self.plam.plot([], [], pen=pen(0, 1.8), name="λx"),
                           self.plam.plot([], [], pen=pen(1, 1.8), name="λy"),
                           self.plam.plot([], [], pen=pen(5, 2.2), name="|λ|")]
