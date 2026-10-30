"""Mass-matrix page of the Dynamics tab: heat-map of M(theta) with numeric cells, plus a readout of symmetry,
positive-definiteness, passivity (Mdot - 2C skew) and engine timings.  Functions take the owning tab as `self`."""
from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import theme
from ..widgets import value_label


def build_mass_page(self):
    page = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(page)
    h.setContentsMargins(0, 0, 0, 0)
    self.glw = pg.GraphicsLayoutWidget()
    self.mplot = self.glw.addPlot(title="M(θ) at the start configuration")
    self.mplot.invertY(True)
    self.mplot.setAspectLocked(True)
    self.mplot.hideButtons()
    self.mplot.setMouseEnabled(False, False)
    self.mimg = pg.ImageItem(axisOrder="row-major")
    self.mplot.addItem(self.mimg)
    try:
        cmap = pg.colormap.get("CET-D1")
    except Exception:
        cmap = pg.colormap.get("viridis")
    self.mbar = pg.ColorBarItem(values=(-1, 1), colorMap=cmap, width=14, interactive=False)
    self.mbar.setImageItem(self.mimg, insert_in=self.mplot)
    self.mtexts: list = []
    self.mass_lbl = value_label()
    self.mass_lbl.setTextFormat(QtCore.Qt.RichText)
    self.mass_lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    sc = QtWidgets.QScrollArea()
    sc.setWidget(self.mass_lbl)
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.NoFrame)
    h.addWidget(self.glw, 3)
    h.addWidget(sc, 2)
    return page

def update_mass_panel(self):
    robot = self.ctx.robot
    th = self.start_sl.values()
    if len(th) != robot.n or self.bottom.currentWidget() is not self.mass_page:
        return
    ep = self.ctx.engine(robot, "pinocchio")
    em = self.ctx.engine(robot, "mr")
    t0 = time.perf_counter()
    for _ in range(50):
        Mp = ep.mass_matrix(th)
    t_pin = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    Mm = em.mass_matrix(th)
    t_mr = time.perf_counter() - t0
    M = Mp if self.ctx.backend == "pinocchio" else Mm
    n = robot.n
    vmax = float(np.abs(M).max())
    self.mimg.setImage(M, autoLevels=False)
    self.mimg.setRect(QtCore.QRectF(0, 0, n, n))
    self.mbar.setLevels((-vmax, vmax))
    if len(self.mtexts) != n * n:
        for it in self.mtexts:
            self.mplot.removeItem(it)
        self.mtexts = []
        for i in range(n):
            for j in range(n):
                it = pg.TextItem("", anchor=(0.5, 0.5), color="#101010")
                it.setPos(j + 0.5, i + 0.5)
                self.mplot.addItem(it)
                self.mtexts.append(it)
        ticks = [[(i + 0.5, str(i + 1)) for i in range(n)]]
        self.mplot.getAxis("left").setTicks(ticks)
        self.mplot.getAxis("bottom").setTicks(ticks)
        self.mplot.setRange(xRange=(0, n), yRange=(0, n), padding=0.02)
    for i in range(n):
        for j in range(n):
            v = M[i, j] if abs(M[i, j]) >= 5e-4 else 0.0
            self.mtexts[i * n + j].setText(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
            self.mtexts[i * n + j].setColor("#101010" if abs(v) < 0.6 * vmax else "#f5f5f5")
    lam = np.linalg.eigvalsh(M)
    sym = float(np.abs(M - M.T).max())
    # passivity: (M_dot - 2C) skew-symmetric, M_dot by central differences along a random velocity
    rng = np.random.default_rng(0)
    dq = rng.uniform(-1, 1, n)
    h = 1e-6
    Md = (ep.mass_matrix(th + h * dq) - ep.mass_matrix(th - h * dq)) / (2 * h)
    Nm = Md - 2 * ep.coriolis_matrix(th, dq)
    skew = float(np.abs(Nm + Nm.T).max())
    ok, bad = f"<span style='color:{theme.good_color()}'>✓</span>", f"<span style='color:{theme.bad_color()}'>✗</span>"
    eng = "Pinocchio (CRBA)" if self.ctx.backend == "pinocchio" else "Modern Robotics (n RNEA calls)"
    self.mass_lbl.setText(
        f"<b>Engine shown:</b> {eng}<br>"
        f"<b>Eigenvalues λ(M)</b> [kg·m²]:<br>&nbsp;&nbsp;{', '.join(f'{x:.4g}' for x in lam)}<br>"
        f"symmetric M = Mᵀ: {ok if sym < 1e-9 else bad} ‖M−Mᵀ‖ = {sym:.1e}<br>"
        f"positive-definite: {ok if lam.min() > 0 else bad} λmin = {lam.min():.4g}<br>"
        f"condition number κ = λmax/λmin = {lam.max() / lam.min():.1f}<br>"
        f"Ṁ − 2C skew-symmetric: {ok if skew < 1e-5 else bad} ‖N+Nᵀ‖ = {skew:.1e}<br>"
        f"engines agree: {ok if np.abs(Mp - Mm).max() < 1e-9 else bad} ‖M_MR − M_Pin‖ = {np.abs(Mp - Mm).max():.1e}"
        f"<br><br><b>Computation time of M(θ)</b><br>"
        f"&nbsp;&nbsp;Modern Robotics (n = {n} RNEA calls): {t_mr * 1e3:.2f} ms<br>"
        f"&nbsp;&nbsp;Pinocchio CRBA (C++): {t_pin * 1e6:.1f} µs<br>"
        f"&nbsp;&nbsp;measured speed-up ×{t_mr / max(t_pin, 1e-9):.0f} (typically ~1000×; depends on n "
        f"and the machine)<br><br>"
        f"<span style='color:{theme.muted_color()}'>Diagonal mᵢᵢ = apparent inertia of joint i; off-diagonal terms "
        f"couple the joints (MR 8.1.3). Move the start sliders to watch M(θ) change.</span>")
