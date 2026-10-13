"""Tests for app-level GUI features: projector mode, recording/export, settings, tab loading."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app, False)
    yield app
    theme.set_projector(app, False)


def test_tabs_list_and_missing_module_skipped(qapp):
    from mrstudio.gui.app import TABS, load_tab
    from mrstudio.gui.base import AppContext
    mods = [m for m, _ in TABS]
    assert mods.index("mrstudio.gui.tabs.advanced_control_tab") < mods.index("mrstudio.gui.tabs.validation_tab")
    assert mods.index("mrstudio.gui.tabs.parallel_tab") < mods.index("mrstudio.gui.tabs.validation_tab")
    assert load_tab("mrstudio.gui.tabs.__does_not_exist__", "X", AppContext()) is None
    # a module that exists but fails -> error tab, not skipped
    page, label = load_tab("mrstudio.gui.theme", "NoSuchClass", AppContext())
    assert label.startswith("⚠")


def test_projector_mode_roundtrip(qapp):
    from PySide6 import QtGui
    from mrstudio.gui import theme
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.theory import TheoryPanel
    ctx = AppContext()
    got = []
    ctx.projector_changed.connect(got.append)
    panel = TheoryPanel()
    panel.set_content(r"<h2>T</h2>$$T(\theta) = e^{[\mathcal{S}]\theta} M$$")
    panel.show()
    w0 = panel.document().size().width()
    base_pt = qapp.font().pointSizeF()
    ctx.set_projector(True)
    assert ctx.projector and got == [True]
    assert "font-size: 19px" in qapp.styleSheet()
    assert qapp.font().pointSizeF() == pytest.approx(base_pt * theme.PROJ_FONT_SCALE)
    assert panel._rendered_scale == theme.PROJ_FONT_SCALE
    ctx.set_projector(False)
    assert got == [True, False]
    assert qapp.styleSheet() == theme.STYLESHEET
    assert qapp.font().pointSizeF() == pytest.approx(base_pt)
    assert panel._rendered_scale == 1.0 and panel.content().startswith("<h2>")
    assert w0 > 0
    panel.close()


def test_plot_styling_restores(qapp):
    import pyqtgraph as pg
    from mrstudio.gui import theme
    pw = pg.PlotWidget()
    pw.setTitle("x", size="10pt")
    pw.addLegend()
    c = pw.plot([0, 1], [0, 1], pen=pg.mkPen("r", width=2), name="a")
    theme.set_projector(qapp, True)
    theme.style_plot(pw)
    assert c.opts["pen"].widthF() == pytest.approx(2 * theme.PROJ_LINE_SCALE)
    theme.set_projector(qapp, False)
    theme.style_plot(pw)
    assert c.opts["pen"].widthF() == pytest.approx(2)


def test_viewer_line_width_scales(qapp):
    from mrstudio.gui import theme
    from mrstudio.gui.viewer import SceneView
    v = SceneView()
    it = v.set_polyline("a", [[0, 0, 0], [1, 0, 0]], (1, 1, 1, 1), 2.0)
    theme.set_projector(qapp, True)
    assert it.width == pytest.approx(2.0 * theme.PROJ_LINE_SCALE)
    it2 = v.set_polyline("b", [[0, 0, 0], [1, 0, 0]], (1, 1, 1, 1), 1.5)
    assert it2.width == pytest.approx(1.5 * theme.PROJ_LINE_SCALE)
    theme.set_projector(qapp, False)
    assert it.width == pytest.approx(2.0) and it2.width == pytest.approx(1.5)


def test_recorder_mp4_and_gif(qapp, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as iio
    from PIL import Image
    from PySide6 import QtCore, QtGui
    from mrstudio.gui.recorder import Recorder

    k = {"i": 0}

    def grab():  # synthetic moving square, odd size to exercise even-cropping
        img = QtGui.QImage(321, 201, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor("#14171c"))
        p = QtGui.QPainter(img)
        p.fillRect(10 + 5 * k["i"], 50, 40, 40, QtGui.QColor("#4cc9f0"))
        p.end()
        k["i"] += 1
        return img

    rec = Recorder()
    rec.start(grab, fps=25)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.8:
        qapp.processEvents()
        time.sleep(0.005)
    tmp = rec.stop()
    assert tmp and os.path.getsize(tmp) > 0 and rec.frames >= 10
    import shutil
    tmp2 = str(tmp_path / "copy.mp4")
    shutil.copy(tmp, tmp2)
    mp4, gif = str(tmp_path / "a.mp4"), str(tmp_path / "a.gif")
    rec.export(tmp, mp4, block=True)
    rec.export(tmp2, gif, block=True)
    r = iio.get_reader(mp4)
    fr = r.get_data(0)
    assert fr.shape == (200, 320, 3)
    assert r.count_frames() == rec.frames
    im = Image.open(gif)
    assert im.n_frames >= 5 and im.size[0] == 320


def test_settings_location_override(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("MRSTUDIO_SETTINGS", str(tmp_path / "s.ini"))
    from mrstudio.gui.app import app_settings
    s = app_settings()
    s.setValue("projector", True)
    s.sync()
    assert (tmp_path / "s.ini").exists()
    assert app_settings().value("projector") in (True, "true")
