"""Shared constants and small helpers for the closed-chains tab: presets, colours, degree/plot helpers."""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from .. import theme

FOUR_PRESETS = {
    "Crank-rocker": (0.40, 0.12, 0.38, 0.28),
    "Double-crank (drag link)": (0.12, 0.30, 0.35, 0.28),
    "Double-rocker (Grashof)": (0.40, 0.30, 0.12, 0.35),
    "Change-point (parallelogram)": (0.40, 0.20, 0.40, 0.20),
    "Triple-rocker (non-Grashof)": (0.50, 0.25, 0.30, 0.28),
}
LINK_COLS = [theme.SERIES[0], theme.SERIES[1], theme.SERIES[2], theme.SERIES[4]]
CURVE = (1.0, 0.35, 0.85, 0.95)


def _deg(x):
    return np.degrees(x)


def _break_jumps(y, jump=180.0):
    y = np.array(y, float)
    d = np.abs(np.diff(y))
    y[1:][d > jump] = np.nan
    return y


def _wrapped(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    return lbl
