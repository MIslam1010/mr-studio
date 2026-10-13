"""Bibliography, theory-panel "Further reading" sections, References dialog, Featherstone dynamics."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TAB_MODULES = ["fk", "velocity", "ik", "dynamics", "control", "advanced_control", "closed", "parallel",
               "stewart", "validation"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_cite_and_references():
    from mrstudio.references import REFERENCES, cite, references_html
    assert {"featherstone", "astrom_murray", "gotor", "lavalle", "corke",
            "mls", "mr"} <= set(REFERENCES)
    assert "&amp;amp;reg;" not in REFERENCES["corke"]["path"] and "&amp;reg;" in REFERENCES["corke"]["path"]
    assert cite("featherstone", "Table 5.1, p. 96") == "<b>Featherstone</b>, Table 5.1, p. 96"
    assert cite("astrom_murray").startswith("<b>Åström &amp; Murray</b>")
    assert references_html().count("<p>") == len(REFERENCES)


def test_every_tab_has_further_reading():
    from mrstudio.gui.theory import validate_all
    from mrstudio.references import REFERENCES
    texts = []
    for m in TAB_MODULES:
        t = importlib.import_module(f"mrstudio.gui.tabs.{m}_tab").THEORY
        assert "<h3>Further reading</h3>" in t, m
        texts.append(t)
    assert validate_all(texts) == []
    used = " ".join(texts)
    for ref in REFERENCES.values():
        if ref["short"] not in ("Lynch & Park",):
            assert ref["short"].replace("&", "&amp;") in used, ref["short"]


def test_references_dialog(qapp):
    from mrstudio.gui.app import ReferencesDialog
    from mrstudio.references import REFERENCES, pdf_path
    dlg = ReferencesDialog()
    dlg.show()
    qapp.processEvents()
    expected = {k for k in REFERENCES if pdf_path(k)}
    assert set(dlg.open_buttons) == expected
    dlg.close()


def test_featherstone_matches_engines():
    from mrstudio.dynamics import DynamicsEngine
    from mrstudio.featherstone import FeatherstoneModel, validation_checks
    from mrstudio.robots import get_robot
    r = get_robot("Stanford arm (RRPRRR)")   # has a prismatic joint
    fm, eng = FeatherstoneModel(r), DynamicsEngine(r, "mr")
    rng = np.random.default_rng(0)
    q, qd, qdd = r.random_config(rng), rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
    tau = fm.rnea(q, qd, qdd)
    assert np.allclose(tau, eng.inverse(q, qd, qdd), atol=1e-10)
    assert np.allclose(fm.aba(q, qd, tau), qdd, atol=1e-10)
    cs = validation_checks()
    assert cs and all(c.passed for c in cs) and all(c.category == "Dynamics" for c in cs)
