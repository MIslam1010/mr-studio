"""Draw the MR Studio application icon (1024x1024 PNG) with Qt -- no image editor needed.

    python packaging/make_icon.py [packaging/icon.png]

PyInstaller converts the PNG to .icns (macOS) / .ico (Windows) itself (it uses Pillow).
The motif: a planar 3R arm (theme colours) with the screw axis of its last joint.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui  # noqa: E402

BG_TOP, BG_BOTTOM = "#1f2a38", "#0d1117"
LINK, JOINT, ACCENT, GOOD, TEXT = "#8fa3b8", "#f29e4c", "#4cc9f0", "#57d68d", "#e6e8eb"


def draw(size: int = 1024) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    s = size / 1024.0
    # macOS-style rounded square (824 px body inside a 1024 canvas)
    body = QtCore.QRectF(100 * s, 100 * s, 824 * s, 824 * s)
    g = QtGui.QLinearGradient(body.topLeft(), body.bottomLeft())
    g.setColorAt(0, QtGui.QColor(BG_TOP))
    g.setColorAt(1, QtGui.QColor(BG_BOTTOM))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(g)
    p.drawRoundedRect(body, 185 * s, 185 * s)

    # base
    p.setBrush(QtGui.QColor("#4a5361"))
    p.drawRoundedRect(QtCore.QRectF(250 * s, 735 * s, 200 * s, 55 * s), 14 * s, 14 * s)
    # 3R arm
    pts = [QtCore.QPointF(350 * s, 720 * s), QtCore.QPointF(420 * s, 450 * s),
           QtCore.QPointF(660 * s, 330 * s), QtCore.QPointF(760 * s, 470 * s)]
    pen = QtGui.QPen(QtGui.QColor(LINK), 62 * s, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
    p.setPen(pen)
    for a, b in zip(pts[:-1], pts[1:]):
        p.drawLine(a, b)
    # dashed screw axis through the end effector
    pen = QtGui.QPen(QtGui.QColor(ACCENT), 14 * s, QtCore.Qt.DashLine, QtCore.Qt.RoundCap)
    p.setPen(pen)
    p.drawLine(QtCore.QPointF(615 * s, 615 * s), QtCore.QPointF(840 * s, 300 * s))
    # joints
    p.setPen(QtGui.QPen(QtGui.QColor(BG_BOTTOM), 10 * s))
    p.setBrush(QtGui.QColor(JOINT))
    for c in pts[:-1]:
        p.drawEllipse(c, 52 * s, 52 * s)
    p.setBrush(QtGui.QColor(GOOD))
    p.drawEllipse(pts[-1], 36 * s, 36 * s)
    # "MR"
    f = QtGui.QFont("Helvetica", 1)
    f.setPixelSize(int(170 * s))
    f.setBold(True)
    p.setFont(f)
    p.setPen(QtGui.QColor(TEXT))
    p.drawText(QtCore.QRectF(160 * s, 140 * s, 400 * s, 220 * s), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, "MR")
    p.end()
    return img


def main(argv: list[str]) -> int:
    from PySide6 import QtWidgets  # noqa: F401  (a QGuiApplication is needed for fonts)
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])  # noqa: F841
    out = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if not draw().save(out):
        print("could not write", out, file=sys.stderr)
        return 1
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
