"""Theory panel: rich text with LaTeX equations rendered offline by matplotlib.mathtext.

Write content as HTML-ish text where display equations are wrapped in $$...$$
and inline math in $...$ (matplotlib mathtext subset: \\theta, \\dot{}, \\mathcal{}, \\left[ \\right],
^{}, _{}, \\sum, \\prod, \\in, \\mathbb{R} ... no matrices/environments).

Panels render lazily: ``set_content`` only stores the text; the equations are rendered the first time the
panel is shown (and again after a projector-mode / palette change). Rendered equation images are cached in
memory and -- in the app -- on disk under the per-user cache directory, so later starts are fast.
"""
from __future__ import annotations

import hashlib
import io
import os
import re

import matplotlib
from matplotlib import mathtext
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

_cache: dict[tuple, QtGui.QImage] = {}
_SCALE = 2.0  # render at 2x for Retina sharpness
_disk_dir: str | None | bool = False  # False = not resolved yet, None = disk cache disabled


def _disk_cache_dir() -> str | None:
    """Directory of the on-disk equation cache (QStandardPaths CacheLocation/mathtext), or None.

    Only used inside MR Studio itself (application name "MR Studio"); ``MRSTUDIO_NO_EQ_CACHE=1`` disables it.
    """
    global _disk_dir
    if _disk_dir is False:
        app = QtCore.QCoreApplication.instance()
        if app is None:
            return None  # not resolved yet: ask again once the application exists
        _disk_dir = None
        if app.applicationName() == "MR Studio" and not os.environ.get("MRSTUDIO_NO_EQ_CACHE"):
            base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.CacheLocation)
            if base:
                d = os.path.join(base, "mathtext")
                try:
                    os.makedirs(d, exist_ok=True)
                    _disk_dir = d
                except OSError:
                    pass
    return _disk_dir


def _disk_key(tex: str, size: float, color: str) -> str:
    h = hashlib.sha1(f"{matplotlib.__version__}|{_SCALE}|{size:.3f}|{color}|{tex}".encode()).hexdigest()
    return h + ".png"


def render_math(tex: str, size: float = 14, color: str = theme.TEXT) -> QtGui.QImage:
    key = (tex, size, color)
    if key not in _cache:
        img = None
        d = _disk_cache_dir()
        path = os.path.join(d, _disk_key(tex, size, color)) if d else None
        if path and os.path.exists(path):
            img = QtGui.QImage(path)
            if img.isNull():
                img = None
        if img is None:
            buf = io.BytesIO()
            fig = Figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f"${tex}$", fontproperties=FontProperties(size=size * _SCALE, math_fontfamily="cm"),
                     color=color)
            fig.savefig(buf, dpi=72, format="png", bbox_inches="tight", pad_inches=0.03, transparent=True)
            data = buf.getvalue()
            img = QtGui.QImage.fromData(data, "PNG")
            if path:
                try:
                    tmp = path + f".{os.getpid()}.tmp"
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                    os.replace(tmp, path)
                except OSError:
                    pass
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
        self._rendered_key = None     # (font scale, text colour) of the current rendering
        self._dirty = False           # text changed since the last rendering
        self._widest = 0              # natural width of the widest equation (logical px)
        self._shrunk = False          # some equation was shrunk to fit the viewport
        self._render_width = 0
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._render_if_visible)
        theme.notifier().projector_changed.connect(self._on_projector)

    @staticmethod
    def _key():
        return theme.font_scale(), theme.text_color()

    def needs_render(self) -> bool:
        return bool(self._text) and (self._dirty or self._rendered_key != self._key())

    def _on_projector(self, _on: bool):
        # re-render now if visible, otherwise lazily on the next showEvent (rendering costs ~0.1 s/panel)
        try:
            if self.isVisible():
                self._render()
        except RuntimeError:  # deleted
            pass

    def _render_if_visible(self):
        try:
            if self.isVisible() and self._text:
                self._render()
        except RuntimeError:
            pass

    def showEvent(self, e):
        if self.needs_render():
            self._render()
        super().showEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._text or not self.isVisible() or self._rendered_key is None:
            return
        avail = self._max_eq_width()
        # re-render (debounced) if an equation now clips, or a shrunk one could be shown larger
        if self._widest > avail or (self._shrunk and avail > self._render_width + 16):
            self._resize_timer.start()

    def _max_eq_width(self) -> int:
        return max(80, self.viewport().width() - 2 * int(self.document().documentMargin()) - 12)

    def _img(self, tex: str, size: float) -> str:
        img = render_math(tex, size, theme.text_color())
        self._n += 1
        url = QtCore.QUrl(f"eq://{self._n}")
        self.document().addResource(QtGui.QTextDocument.ImageResource, url, img)
        dpr = max(img.devicePixelRatio(), 1.0)
        w, h = img.width() / dpr, img.height() / dpr
        self._widest = max(self._widest, int(w))
        maxw = self._max_eq_width()
        if w > maxw:  # too wide for the panel (projector mode / narrow panel): shrink instead of clipping
            self._shrunk = True
            return (f'<img src="{url.toString()}" width="{maxw}" height="{max(1, int(h * maxw / w))}" '
                    f'style="vertical-align:middle">')
        return f'<img src="{url.toString()}" style="vertical-align:middle">'

    def set_content(self, text: str):
        """Store the text; it is rendered now if the panel is visible, else on its next show."""
        self._text = text
        self._dirty = True
        if self.isVisible():
            self._render()

    def content(self) -> str:
        return self._text

    def _render(self):
        text = self._text
        sc = theme.font_scale()
        self._rendered_scale = sc
        self._rendered_key = self._key()
        self._dirty = False
        self._widest = 0
        self._shrunk = False
        self._render_width = self._max_eq_width()
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
        bold = theme.text_color() if theme.is_light() else "#ffffff"
        css = (f"<style>body{{color:{theme.text_color()}; font-size:{f(13)}px; line-height:140%;}}"
               f"h2{{color:{theme.accent_color()}; font-size:{f(16)}px; margin-bottom:2px}}"
               f"h3{{color:{theme.warn_color()}; font-size:{f(14)}px; margin-bottom:0px; margin-top:10px}}"
               f"a{{color:{theme.accent_color()}}}"
               f".ref{{color:{theme.muted_color()}; font-size:{f(12)}px}} b{{color:{bold}}}</style>")
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
