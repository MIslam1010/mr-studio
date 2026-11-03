"""Robustness of the Ch. 12 / Ch. 13 modules, lessons, assignments and launch links against odd input: one test
per defect found in the bug hunt of the feature/ideas-2-6 code (degenerate contacts, empty grasps, non-finite
table entries, answer parsing, tampered submissions, CSV cells, Windows URL-handler arguments)."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mrstudio import assignments as A  # noqa: E402
from mrstudio import grasping as gr  # noqa: E402
from mrstudio import lessons as LS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app)
    return app


# ------------------------------------------------------------------------------------------------ grasping (Ch. 12)

@pytest.mark.parametrize("p, n, mu", [((0, 0), (0, 0), 0.0), ((0, 0), (1e-320, 0), 0.0),
                                      ((math.nan, 0), (1, 0), 0.0), ((0, math.inf), (1, 0), 0.0),
                                      ((0, 0), (math.nan, 1), 0.0), ((0, 0), (1, 0), math.inf),
                                      ((0, 0), (1, 0), math.nan)])
def test_contact_rejects_degenerate_input(p, n, mu):
    """A zero / non-finite normal used to become a NaN normal silently (and a NaN point or an infinite mu then
    crashed the LP with LinAlgError); a NaN mu silently became 0."""
    with pytest.raises(ValueError):
        gr.Contact(p, n, mu)


def test_contact_normalises_a_valid_normal():
    c = gr.Contact((1, 2), (0, 3), 0.5)
    assert np.allclose(c.n, (0, 1)) and c.mu == 0.5
    assert gr.Contact((0, 0), (1, 0), -1.0).mu == 0.0          # negative mu still means frictionless


def test_empty_grasp_is_harmless():
    """No contacts: balance() raised ValueError (linprog with no variables) and grasp_quality() raised on an
    empty concatenate; both are reachable from the tab (Ex. 12.9 preset, delete every contact)."""
    F = gr.wrench_matrix([], friction=True)
    assert F.shape == (3, 0)
    assert gr.balance(F, np.array([0.0, 0.0, 9.81])) is None    # gravity cannot be balanced by nothing
    k = gr.balance(F, np.zeros(3))
    assert k is not None and k.shape == (0,)                     # the zero wrench needs no contact force
    assert gr.grasp_quality([], friction=True) == -math.inf
    assert gr.grasp_quality([], friction=False) == -math.inf
    assert not gr.form_closure([]).closed and not gr.force_closure([]).closed


def test_balance_matches_a_hand_solution():
    """Independent cross-check of balance(): one frictionless finger under the body pushing up (0, 1) at x = 0
    balances exactly the wrench (0, 0, f) with f >= 0 (k = f), and nothing with a sideways component."""
    F = gr.wrench_matrix([gr.Contact((0, 0), (0, 1))], friction=True)
    assert np.allclose(gr.balance(F, [0, 0, 2.5]), [2.5])
    assert gr.balance(F, [0, 0, -1.0]) is None and gr.balance(F, [0, 0.1, 1.0]) is None


def test_grasping_tab_survives_bad_table_entries_and_no_contacts(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.grasping_tab import GraspingTab
    tab = GraspingTab(AppContext())
    tab.apply_preset("Ex. 12.9")
    before = [(c.p.copy(), c.n.copy(), c.mu) for c in tab.contacts]
    # "inf" in the angle column raised ValueError (math.cos(inf)) inside the Qt slot; "nan" / "1e999" in x and
    # "inf" in mu reached the LP as NaN and raised LinAlgError in recompute()
    for col, txt in ((2, "inf"), (0, "nan"), (1, "1e999"), (3, "inf"), (3, "nan")):
        tab.table.item(0, col).setText(txt)
        tab.recompute()
        assert all(np.all(np.isfinite(c.p)) and np.all(np.isfinite(c.n)) and math.isfinite(c.mu)
                   for c in tab.contacts)
    after = [(c.p, c.n, c.mu) for c in tab.contacts]
    assert all(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2] for a, b in zip(before, after))
    tab.table.item(0, 0).setText("−2.5")                        # unicode minus, as the app prints numbers
    tab.recompute()
    assert tab.contacts[0].p[0] == -2.5
    # gravity preset with every contact deleted: recompute() raised in balance()
    while tab.contacts:
        tab._del_contact()
    tab.recompute()
    assert not tab.form.closed and not tab.force.closed and "not balanced" in tab.bal_lbl.text()


@pytest.mark.parametrize("key", ["²", "³", "½"])
def test_tab_presets_reject_non_decimal_digits(qapp, key):
    """``"²".isdigit()`` is True but ``int("²")`` raises: a launch URL ``preset=²`` raised ValueError inside
    apply_preset instead of returning False."""
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.grasping_tab import GraspingTab
    from mrstudio.gui.tabs.mobile_tab import MobileTab
    for cls in (GraspingTab, MobileTab):
        tab = cls(AppContext())
        assert tab.apply_preset(key) is False
        assert tab.apply_preset("1") is True                     # plain indices still work


# ------------------------------------------------------------------------------------------------ lessons

@pytest.mark.parametrize("text, value", [
    ("π/2", math.pi / 2), ("2π", None), ("π", math.pi), ("–0.5", -0.5),
    ("0.0603 kg·m²", 0.0603), ("4.5 N·m·s/rad", 4.5), ("67.5 °/s", 67.5),
    ("135°/s²", 135.0), ("16 1/s", 16.0), ("64 1/s²", 64.0), ("0.77 J", 0.77), ("0.12 m²", 0.12),
    ("12 N·m", 12.0), ("1,5 m", 1.5), ("1e-3", 1e-3), ("−0.5", -0.5), ("30°", 30.0), ("2 rad/s", 2.0),
])
def test_parse_answer_symbols_and_units(text, value):
    """The units the lessons print (kg·m², N·m·s/rad, °/s, 1/s², J, m²) and π / en dashes were not read."""
    if value is None:
        with pytest.raises(ValueError):
            LS.parse_answer(text)
    else:
        assert LS.parse_answer(text) == pytest.approx(value)


def test_every_lesson_unit_is_accepted_after_the_number():
    for L in LS.all_lessons():
        for s in L.steps:
            ref = s.check.reference
            typed = f"{ref:.6g} {s.check.unit}".strip()
            assert s.check.grade(typed).ok, (L.id, s.check.label, typed)


def test_parse_answer_big_integer_powers_are_fast():
    """Exponents were limited to 100 but bases were exact Python ints: '(((9**99)**99)**99)**99' built a
    95-million-digit integer (minutes of CPU, gigabytes) before failing."""
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        LS.parse_answer("(((9**99)**99)**99)**99")
    with pytest.raises(ValueError):
        LS.parse_answer("(10**99)**99*(10**99)**99")
    assert time.perf_counter() - t0 < 1.0
    assert LS.parse_answer("2**10") == 1024.0 and LS.parse_answer("7/2") == 3.5


# ------------------------------------------------------------------------------------------------ assignments

@pytest.fixture(scope="module")
def code():
    return A.make_code("hw9", "Homework 9", [("fk", 1), ("control", 2)], salt="robust-salt")


@pytest.mark.parametrize("given", [5, "R", [1, 2, 3], True])
def test_grade_survives_non_dict_answer_entries(code, given):
    """A submission whose answers entry is not a {quantity: value} object (edited file) raised TypeError in
    grade() and aborted tools/grade_assignments.py for the whole class."""
    sub = A.Submission("hw9", "1234", {"0": given, "1": {}}, code)
    rep = A.grade(code, sub)
    assert rep["score"] == 0 and rep["max_score"] == 2
    assert any("problem 1" in n for n in rep["notes"])


def test_grade_folder_skips_tampered_file(code, tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import grade_assignments as tool
    good = A.submit(code, "1234", A.answer_key(code, "1234"))
    A.save_submission(good, str(tmp_path))
    d = good.to_json()
    d["student_id"], d["answers"] = "5678", {"0": 5, "1": [1, 2]}
    (tmp_path / "5678_hw9.json").write_text(json.dumps(d))
    header, rows, details = tool.grade_folder(code, str(tmp_path))
    assert len(rows) == 2 and len(details) == 2
    assert {r[0] for r in rows} == {"1234", "5678"}


@pytest.mark.parametrize("points", ["NaN", "Infinity", "-Infinity"])
def test_code_with_non_finite_points_is_rejected(points):
    """json.loads accepts NaN / Infinity, and `points <= 0` is False for NaN: a code with "points": NaN was
    accepted and every score became NaN."""
    raw = ('{"v":1,"id":"x","title":"x","due":"","items":[{"topic":"fk","difficulty":1,"points":%s}],'
           '"salt":"s"}' % points)
    bad = A._b64e(raw.encode())
    with pytest.raises(A.AssignmentError):
        A.parse_code(bad)
    with pytest.raises(A.AssignmentError):
        A.make_code("x", "x", [{"topic": "fk", "difficulty": 1, "points": float(points.replace("Infinity", "inf"))}])


def test_parse_numbers_unicode_minus():
    assert A.parse_numbers("−0.5 –1e-3") == [-0.5, -1e-3]


def test_grade_csv_cells_cannot_start_a_formula(code, tmp_path):
    """The student writes the submission file: `created` / `app_version` (and the file name) went verbatim into
    the instructor's CSV, so '=HYPERLINK(...)' ran as a spreadsheet formula when the CSV was opened."""
    sys.path.insert(0, str(ROOT / "tools"))
    import grade_assignments as tool
    sub = A.submit(code, "1234", A.answer_key(code, "1234"), created="=HYPERLINK(\"http://x\",\"late?\")")
    sub.app_version = "@SUM(1+1)"
    A.save_submission(sub, str(tmp_path / "subs"))
    out = tmp_path / "g.csv"
    assert tool.main([code, str(tmp_path / "subs"), "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows and all(not str(v).startswith(("=", "+", "-", "@")) for r in rows for k, v in r.items()
                        if k not in ("score", "max_score", "percent") and not k.startswith("p"))
    assert rows[0]["percent"] == "100"


# ------------------------------------------------------------------------------------------------ launch links

def test_windows_link_launch_ignores_injected_arguments(monkeypatch):
    """Windows starts ``MRStudio.exe "%1"``: a link whose text breaks out of the quotes would add real options
    (``--self-test --report <any file>`` writes a file, ``--screenshot <dir>`` writes PNGs).  When the first
    argument is an mrstudio: link, the launcher passes only the link on."""
    sys.path.insert(0, str(ROOT / "packaging"))
    import launcher
    import mrstudio.gui.app as app_mod
    seen = []
    monkeypatch.setattr(app_mod, "main", lambda argv=None: seen.append(list(argv)) or 0)
    monkeypatch.setattr(launcher, "_cli_exit", lambda fn, report: pytest.fail("a headless mode was started"))
    url = "mrstudio://open?tab=MobileTab"
    assert launcher.run([url, "--self-test", "--report", "C:/Users/x/evil.txt", "--screenshot", "C:/tmp"]) == 0
    assert seen == [[url]]
    assert launcher.run(["MRSTUDIO:open?tab=1", "--validate"]) == 0
    assert seen[-1] == ["MRSTUDIO:open?tab=1"]
    assert launcher.run(["--tab", "3", "--open", url]) == 0      # ordinary command lines are unchanged
    assert seen[-1] == ["--tab", "3", "--open", url]
