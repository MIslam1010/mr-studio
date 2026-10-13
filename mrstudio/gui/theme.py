"""Colours, the dark Qt palette shared by all tabs, and the projector / presentation mode.

Projector mode (``set_projector(app, True)``) is applied centrally: app font + stylesheet sizes are
enlarged, contrast is raised, and ``notifier().projector_changed`` is emitted so TheoryPanels
(re-render equations larger), SceneViews (thicker lines, bigger labels/links) and pyqtgraph plots
(``style_plot``) restyle themselves. Tabs need no code for it.
"""
from PySide6 import QtCore, QtGui, QtWidgets

VIEW_BG = "#14171c"
LINK = "#8fa3b8"
JOINT = "#f29e4c"
BASE = "#4a5361"
ACCENT = "#4cc9f0"
GOOD = "#57d68d"
BAD = "#ff6b6b"
WARN = "#ffd166"
TEXT = "#e6e8eb"
MUTED = "#9aa3ad"
# categorical series colours (joint 1..7) -- distinguishable on dark background
SERIES = ["#4cc9f0", "#f29e4c", "#57d68d", "#ff6b9a", "#b392f0", "#ffd166", "#5eead4"]

# ---------------------------------------------------------------- projector mode parameters
PROJ_FONT_SCALE = 1.45      # all font sizes
PROJ_LINE_SCALE = 2.0       # 3-D polylines / point sizes, plot curve widths
PROJ_LINK_SCALE = 1.25      # robot link radius in RobotGraphic
PROJ_VIEW_BG = "#07090c"    # darker 3-D background = more contrast on a projector
PROJ_TEXT = "#ffffff"
PROJ_MUTED = "#c8d0d8"

_projector = False


def is_projector() -> bool:
    return _projector


def font_scale() -> float:
    return PROJ_FONT_SCALE if _projector else 1.0


def line_scale() -> float:
    return PROJ_LINE_SCALE if _projector else 1.0


def link_scale() -> float:
    return PROJ_LINK_SCALE if _projector else 1.0


def view_bg() -> str:
    return PROJ_VIEW_BG if _projector else VIEW_BG


def text_color() -> str:
    return PROJ_TEXT if _projector else TEXT


def muted_color() -> str:
    return PROJ_MUTED if _projector else MUTED


def px(size: float) -> int:
    """Scale a pixel font size for the current mode."""
    return int(round(size * font_scale()))


class _Notifier(QtCore.QObject):
    projector_changed = QtCore.Signal(bool)


_notifier = None


def notifier() -> _Notifier:
    """Process-wide QObject whose ``projector_changed(bool)`` fires when projector mode toggles."""
    global _notifier
    if _notifier is None:
        _notifier = _Notifier()
    return _notifier


def make_stylesheet(projector: bool = False) -> str:
    s = PROJ_FONT_SCALE if projector else 1.0
    text = PROJ_TEXT if projector else TEXT
    muted = PROJ_MUTED if projector else MUTED
    border = "#55606e" if projector else "#333a44"

    def f(x):
        return int(round(x * s))

    return f"""
QWidget {{ font-size: {f(13)}px; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: {f(14)}px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; font-weight: 600; }}
QTabBar::tab {{ padding: {f(7)}px {f(14)}px; }}
QTabBar::tab:selected {{ color: {ACCENT}; font-weight: 600; }}
QPushButton, QToolButton {{ padding: {f(5)}px {f(10)}px; border-radius: 5px; background: #2b313a; border: 1px solid #3c4450; }}
QPushButton:hover, QToolButton:hover {{ background: #343c47; }}
QPushButton:pressed, QToolButton:pressed {{ background: #22272e; }}
QPushButton:checked, QToolButton:checked {{ background: #1f4e63; border-color: {ACCENT}; }}
QPushButton#primary {{ background: #1f6f8b; border-color: {ACCENT}; font-weight: 600; }}
QToolButton#rec {{ padding-right: {f(22)}px; }}
QToolButton#rec:checked {{ background: #6b1d24; border-color: {BAD}; color: #ffffff; }}
QTextBrowser {{ background: #181b20; border: 1px solid #2c323b; border-radius: 6px; }}
QTableWidget {{ background: #181b20; gridline-color: #2c323b; font-family: Menlo, monospace; }}
QHeaderView::section {{ background: #222730; color: {muted}; border: 0px; padding: 3px; }}
QLabel#value {{ font-family: Menlo, monospace; color: {text}; }}
QLabel#title {{ font-size: {f(16)}px; font-weight: 700; color: {text}; }}
QLabel#muted {{ color: {muted}; }}
QLabel#rec {{ color: #ffffff; background: #c62828; border-radius: 4px; padding: 2px 8px; font-weight: 700;
              font-family: Menlo, monospace; }}
QSplitter::handle {{ background: #22272e; }}
"""


STYLESHEET = make_stylesheet(False)  # kept for backward compatibility

_base_font_pt = None


def apply(app: QtWidgets.QApplication, projector: bool | None = None):
    """Install the Fusion style, dark palette and stylesheet (optionally switching projector mode)."""
    global _projector, _base_font_pt
    if projector is not None:
        _projector = bool(projector)
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    c = QtGui.QColor
    text = text_color()
    p.setColor(QtGui.QPalette.Window, c("#14171c" if _projector else "#1c2026"))
    p.setColor(QtGui.QPalette.WindowText, c(text))
    p.setColor(QtGui.QPalette.Base, c("#0f1115" if _projector else "#16191e"))
    p.setColor(QtGui.QPalette.AlternateBase, c("#1e232a"))
    p.setColor(QtGui.QPalette.ToolTipBase, c("#2b313a"))
    p.setColor(QtGui.QPalette.ToolTipText, c(text))
    p.setColor(QtGui.QPalette.Text, c(text))
    p.setColor(QtGui.QPalette.Button, c("#2b313a"))
    p.setColor(QtGui.QPalette.ButtonText, c(text))
    p.setColor(QtGui.QPalette.Highlight, c("#1f6f8b"))
    p.setColor(QtGui.QPalette.HighlightedText, c("#ffffff"))
    p.setColor(QtGui.QPalette.PlaceholderText, c(muted_color()))
    app.setPalette(p)
    if _base_font_pt is None:
        _base_font_pt = app.font().pointSizeF() if app.font().pointSizeF() > 0 else 13.0
    fnt = app.font()
    fnt.setPointSizeF(_base_font_pt * font_scale())
    app.setFont(fnt)
    app.setStyleSheet(make_stylesheet(_projector))
    install_plot_hooks()


def set_projector(app: QtWidgets.QApplication | None, on: bool):
    """Switch projector mode on/off app-wide and notify every listener."""
    global _projector
    on = bool(on)
    changed = on != _projector
    _projector = on
    app = app or QtWidgets.QApplication.instance()
    if app is not None:
        apply(app)
    if changed:
        notifier().projector_changed.emit(on)


# ---------------------------------------------------------------- pyqtgraph plots
def _pt(v, default):
    try:
        return float(str(v).replace("pt", ""))
    except (TypeError, ValueError):
        return default


def style_plot(pw) -> None:
    """Apply the current mode (fonts, axis pens, curve widths) to a pyqtgraph PlotWidget/PlotItem.

    Idempotent: original sizes are remembered on the items, so toggling back restores them.
    """
    try:
        import pyqtgraph as pg
    except Exception:  # pragma: no cover
        return
    pi = pw.getPlotItem() if hasattr(pw, "getPlotItem") else pw
    s, ls = font_scale(), line_scale()
    fg = text_color() if _projector else MUTED
    tick_font = None
    if _projector:
        tick_font = QtGui.QFont()
        tick_font.setPointSizeF(10 * s)
    for name in ("left", "bottom", "right", "top"):
        if name not in pi.axes:
            continue
        ax = pi.getAxis(name)
        try:
            ax.setStyle(tickFont=tick_font)
            ax.setTextPen(pg.mkPen(fg))
            ax.setPen(pg.mkPen(fg, width=1.5 if _projector else 1.0))
            ax.labelStyle = dict(ax.labelStyle or {}, **{"font-size": f"{9 * s:.0f}pt", "color": fg})
            ax._updateLabel()
            ax.label.setHtml(ax.labelString())
        except Exception:
            pass
    tl = getattr(pi, "titleLabel", None)
    if tl is not None and tl.text:
        if not hasattr(tl, "_mr_size"):
            tl._mr_size = tl.opts.get("size", "10pt")
        tl.setText(tl.text, size=f"{_pt(tl._mr_size, 10.0) * s:.0f}pt", color=text_color())
    leg = getattr(pi, "legend", None)
    if leg is not None:
        for _sample, label in getattr(leg, "items", []):
            if not hasattr(label, "_mr_size"):
                label._mr_size = label.opts.get("size", "8pt")
            label.setText(label.text, size=f"{_pt(label._mr_size, 8.0) * s:.0f}pt")
        try:
            leg.updateSize()
        except Exception:
            pass
    for item in pi.listDataItems():
        try:
            cur = item.opts.get("pen")
            if cur is None:
                continue
            if getattr(item, "_mr_pen_obj", None) is not cur:
                # pen was (re)set by the tab since we last styled it -> its width is the new base
                w = pg.mkPen(cur).widthF()
                item._mr_pen_w = w if w > 0 else 1.0
            pen = pg.mkPen(cur)
            pen.setWidthF(item._mr_pen_w * ls)
            pen.setCosmetic(True)
            item.setPen(pen)
            item._mr_pen_obj = item.opts.get("pen")
        except Exception:
            pass


def style_plots_in(widget) -> None:
    """style_plot() on every pyqtgraph PlotWidget below ``widget``."""
    try:
        import pyqtgraph as pg
    except Exception:  # pragma: no cover
        return
    for pw in widget.findChildren(pg.PlotWidget):
        style_plot(pw)


_patched = False


def install_plot_hooks() -> None:
    """Make plots/curves created *after* a projector toggle pick up the projector style automatically.

    Wraps ``pg.PlotItem.addItem`` / ``setTitle`` so that, in projector mode, the plot is restyled
    (coalesced, on the next event-loop turn). No effect in normal mode.
    """
    global _patched
    if _patched:
        return
    try:
        import pyqtgraph as pg
    except Exception:  # pragma: no cover
        return
    _patched = True

    def _schedule(pi):
        if not _projector or getattr(pi, "_mr_pending", False):
            return
        pi._mr_pending = True

        def run():
            pi._mr_pending = False
            try:
                style_plot(pi)
            except RuntimeError:
                pass
        QtCore.QTimer.singleShot(0, run)

    orig_add, orig_title = pg.PlotItem.addItem, pg.PlotItem.setTitle

    def addItem(self, *a, **k):
        r = orig_add(self, *a, **k)
        _schedule(self)
        return r

    def setTitle(self, *a, **k):
        r = orig_title(self, *a, **k)
        if hasattr(self.titleLabel, "_mr_size") and "size" in k:
            self.titleLabel._mr_size = k["size"]
        _schedule(self)
        return r

    pg.PlotItem.addItem = addItem
    pg.PlotItem.setTitle = setTitle
