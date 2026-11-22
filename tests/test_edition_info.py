"""Student / instructor edition switch (public repository = student; the private copy adds mrstudio/edition.py)."""
import sys
import types

import numpy as np

from mrstudio import edition_info


def test_default_is_student(monkeypatch):
    monkeypatch.delitem(sys.modules, "mrstudio.edition", raising=False)
    assert not edition_info.is_instructor()
    assert edition_info.edition_name() == "student"


def test_fake_edition_module_makes_instructor(instructor):
    assert edition_info.is_instructor() and edition_info.edition_name() == "instructor"


def test_broken_edition_module_falls_back_to_student(monkeypatch):
    class Boom(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("broken")
    monkeypatch.setitem(sys.modules, "mrstudio.edition", Boom("mrstudio.edition"))
    assert edition_info.edition_name() == "student"
    monkeypatch.setitem(sys.modules, "mrstudio.edition", None)      # import raises ImportError
    assert not edition_info.is_instructor()


def test_practice_export_group_student_vs_instructor(qapp, monkeypatch):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.practice_tab import PracticeTab, theory_html
    monkeypatch.delitem(sys.modules, "mrstudio.edition", raising=False)
    tab = PracticeTab(AppContext())
    assert tab.export_group is None
    assert "Quizzes" not in theory_html() and "answer key" not in theory_html()
    assert tab.reveal_btn is not None                                # solutions and lessons stay for students
    types_mod = types.ModuleType("mrstudio.edition")
    types_mod.EDITION = "instructor"
    monkeypatch.setitem(sys.modules, "mrstudio.edition", types_mod)
    tab2 = PracticeTab(AppContext())
    assert tab2.export_group is not None and "Quizzes" in theory_html()


def _hw_tab(monkeypatch, tmp_path):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.hardware_tab import HardwareTab
    monkeypatch.setenv("MRSTUDIO_MYCOBOT_CALIBRATION", str(tmp_path / "cal.json"))
    return HardwareTab(AppContext())


def test_calibration_read_only_for_students(qapp, monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "mrstudio.edition", raising=False)
    tab = _hw_tab(monkeypatch, tmp_path)
    widgets = [*tab.sign_boxes, *tab.offset_spins, tab.stream_pct]
    assert not any(w.isEnabled() for w in widgets)
    assert tab.calib_save_btn.isHidden()
    assert tab.save_calibration() is False
    assert "set by the instructor" in tab.calib_lbl.text()
    assert not (tmp_path / "cal.json").exists()
    tab.shutdown()


def test_existing_calibration_is_loaded_for_students(qapp, monkeypatch, tmp_path):
    from mrstudio.hardware.mycobot import MyCobotCalibration
    MyCobotCalibration(sign=[-1, 1, 1, 1, 1, 1], offset_deg=[10, 0, 0, 0, 0, 0]).save(str(tmp_path / "cal.json"))
    monkeypatch.delitem(sys.modules, "mrstudio.edition", raising=False)
    tab = _hw_tab(monkeypatch, tmp_path)
    assert tab.sign_boxes[0].isChecked() and np.isclose(tab.offset_spins[0].value(), 10.0)
    assert not tab.sign_boxes[0].isEnabled()
    tab.shutdown()


def test_calibration_editable_for_instructor(qapp, monkeypatch, tmp_path, instructor):
    tab = _hw_tab(monkeypatch, tmp_path)
    assert all(w.isEnabled() for w in [*tab.sign_boxes, *tab.offset_spins, tab.stream_pct])
    assert not tab.calib_save_btn.isHidden()
    assert tab.save_calibration() is True and (tmp_path / "cal.json").is_file()
    tab.shutdown()
