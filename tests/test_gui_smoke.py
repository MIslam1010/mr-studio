"""GUI smoke tests: every tab module imports, its theory equations render, the Validation tab works headless."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere: `pytest tests`

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio.gui.app import TABS  # noqa: E402
from mrstudio.gui.theory import validate_all  # noqa: E402


def _import_tab(mod):
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name == mod:
            pytest.skip(f"{mod} is not written yet")
        raise


@pytest.mark.parametrize("mod,cls", TABS, ids=[m.split(".")[-1] for m, _ in TABS])
def test_tab_imports_and_theory_renders(mod, cls):
    m = _import_tab(mod)
    tab_cls = getattr(m, cls)
    assert isinstance(tab_cls.title, str) and tab_cls.title
    theories = [v for k, v in vars(m).items() if "THEORY" in k.upper() and isinstance(v, str)]
    assert validate_all(theories) == [], f"equations that do not render in {mod}"


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


def test_validation_tab_headless(qapp, tmp_path):
    from mrstudio import validation as V
    from mrstudio.checks import Check
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.validation_tab import ValidationTab

    tab = ValidationTab(AppContext())
    assert tab.title == "⑭ Validation" and tab.uses_serial_robot is False
    tab._populate_planned()
    assert tab.table.rowCount() >= len(V.planned_checks())
    # feed a small synthetic result set through the same slots the worker uses
    fake = [Check("a — UR5", "MR 4.1", 1e-16, 1e-10, "detail a", V.KIN),
            Check("b — UR5", "MR 8.3", 1.0, 1e-9, "detail b", V.DYN)]
    for i, c in enumerate(fake):
        tab._on_result(i, c)
    tab._on_done(fake, 0.5)
    assert tab.table.rowCount() == 2
    assert "1 of 2 checks failed" in tab.banner.text()
    tab.fail_only.setChecked(True)
    assert tab.table.isRowHidden(0) and not tab.table.isRowHidden(1)
    fn = tab.save_report(str(tmp_path / "r.html"))
    assert os.path.getsize(fn) > 1000
