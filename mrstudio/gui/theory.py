"""Theory panel: rich text with LaTeX equations rendered offline by matplotlib.mathtext.

Write content as HTML-ish text where display equations are wrapped in $$...$$
and inline math in $...$ (matplotlib mathtext subset: \\theta, \\dot{}, \\mathcal{}, \\left[ \\right],
^{}, _{}, \\sum, \\prod, \\in, \\mathbb{R} ... no matrices/environments).
"""
from __future__ import annotations

import io
import re

from matplotlib import mathtext
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

_cache: dict[tuple, QtGui.QImage] = {}
_SCALE = 2.0  # render at 2x for Retina sharpness


def render_math(tex: str, size: float = 14, color: str = theme.TEXT) -> QtGui.QImage:
    key = (tex, size, color)
    if key not in _cache:
        buf = io.BytesIO()
        fig = Figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontproperties=FontProperties(size=size * _SCALE, math_fontfamily="cm"),
                 color=color)
        fig.savefig(buf, dpi=72, format="png", bbox_inches="tight", pad_inches=0.03, transparent=True)
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        img.setDevicePixelRatio(_SCALE)  # crisp on Retina, no lossy down-scaling of thin strokes
        _cache[key] = img
    return _cache[key]


class TheoryPanel(QtWidgets.QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setMinimumWidth(300)
        self._n = 0
        self._text = ""
        self._rendered_scale = None
        theme.notifier().projector_changed.connect(self._on_projector)

    def _on_projector(self, _on: bool):
        # re-render now if visible, otherwise lazily on the next showEvent (rendering costs ~0.1 s/panel)
        try:
            if self.isVisible():
                self._render()
        except RuntimeError:  # deleted
            pass

    def showEvent(self, e):
        if self._rendered_scale != theme.font_scale() and self._text:
            self._render()
        super().showEvent(e)

    def _img(self, tex: str, size: float) -> str:
        img = render_math(tex, size, theme.text_color())
        self._n += 1
        url = QtCore.QUrl(f"eq://{self._n}")
        self.document().addResource(QtGui.QTextDocument.ImageResource, url, img)
        return f'<img src="{url.toString()}" style="vertical-align:middle">'

    def set_content(self, text: str):
        self._text = text
        self._render()

    def content(self) -> str:
        return self._text

    def _render(self):
        text = self._text
        sc = theme.font_scale()
        self._rendered_scale = sc
        bar = self.verticalScrollBar()
        frac = bar.value() / bar.maximum() if bar.maximum() > 0 else 0.0
        self.document().clear()
        self._n = 0

        def disp(m):
            return f'<p align="center" style="margin:6px 0 6px 0">{self._img(m.group(1).strip(), 15 * sc)}</p>'

        def inline(m):
            return self._img(m.group(1).strip(), 12.5 * sc)

        html = re.sub(r"\$\$(.+?)\$\$", disp, text, flags=re.S)
        html = re.sub(r"\$(.+?)\$", inline, html, flags=re.S)
        f = theme.px
        css = (f"<style>body{{color:{theme.text_color()}; font-size:{f(13)}px; line-height:140%;}}"
               f"h2{{color:{theme.ACCENT}; font-size:{f(16)}px; margin-bottom:2px}}"
               f"h3{{color:{theme.WARN}; font-size:{f(14)}px; margin-bottom:0px; margin-top:10px}}"
               f".ref{{color:{theme.muted_color()}; font-size:{f(12)}px}} b{{color:#ffffff}}</style>")
        self.setHtml(css + html)
        if frac:
            bar.setValue(int(frac * bar.maximum()))


def validate_all(texts: list[str]) -> list[str]:
    """Return a list of equations that fail to render (used by the test-suite)."""
    bad = []
    for t in texts:
        for m in re.finditer(r"\$\$(.+?)\$\$|\$(.+?)\$", t, flags=re.S):
            tex = (m.group(1) or m.group(2)).strip()
            try:
                buf = io.BytesIO()
                mathtext.math_to_image(f"${tex}$", buf, dpi=72, format="png")
            except Exception as e:  # pragma: no cover
                bad.append(f"{tex!r}: {e}")
    return bad
