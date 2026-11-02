"""Graded assignments: codes, per-student seeds, submissions, instructor-side grading, CLI tools and the panel."""
import csv
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

from mrstudio import assignments as A
from mrstudio import problems as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ITEMS = [("fk", 2), ("jacobian", 1), ("control", 2),
         {"topic": "dynamics", "difficulty": 2, "quantities": ["M", "tau"], "points": 2}]


@pytest.fixture(scope="module")
def code():
    return A.make_code("hw3", "Homework 3", ITEMS, due="2026-11-20", salt="fixed-salt")


def _wrong(key):
    return {k: {q: (np.asarray(v) * 1.1 + 0.05).tolist() for q, v in qs.items()} for k, qs in key.items()}


# ------------------------------------------------------------------------------------------------ codes and seeds

def test_code_round_trip(code):
    a = A.parse_code(code)
    assert (a.id, a.title, a.due, a.salt) == ("hw3", "Homework 3", "2026-11-20", "fixed-salt")
    assert [(it.topic, it.difficulty) for it in a.items] == [("fk", 2), ("jacobian", 1), ("control", 2),
                                                             ("dynamics", 2)]
    assert a.items[3].quantities == ["M", "tau"] and a.items[3].points == 2
    assert re.fullmatch(r"[A-Za-z0-9_-]+", code)                 # base64url, no padding: safe to paste anywhere
    assert A.parse_code("  " + code[:40] + "\n" + code[40:] + " ").id == "hw3"   # whitespace from e-mail tolerated


@pytest.mark.parametrize("bad", ["", "not a code", A._b64e(b'{"v": 99}'), A._b64e(b'{"v":1,"id":"x"}')])
def test_bad_codes_are_readable_errors(bad):
    with pytest.raises(A.AssignmentError):
        A.parse_code(bad)


def test_make_code_rejects_bad_items():
    for items in ([("mixed", 2)], [("nope", 2)], [("fk", 4)], [{"topic": "fk", "quantities": ["zzz"]}], []):
        with pytest.raises(A.AssignmentError):
            A.make_code("x", "x", items)
    with pytest.raises(A.AssignmentError):
        A.make_code("bad id!", "x", [("fk", 1)])
    with pytest.raises(A.AssignmentError):
        A.make_code("x", "x", [("fk", 1)], due="20 Nov")


def test_random_salt_by_default():
    assert A.parse_code(A.make_code("a", "a", [("fk", 1)])).salt != A.parse_code(A.make_code("a", "a", [("fk", 1)])).salt


def test_student_seeds(code):
    a = A.parse_code(code)
    s1 = [p.seed for p in A.build(code, "1234")]
    s2 = [p.seed for p in A.build(code, "5678")]
    assert s1 == [p.seed for p in A.build(code, "1234")]          # deterministic
    assert s1 != s2 and all(x != y for x, y in zip(s1, s2))       # two students -> different problems
    assert all(0 <= s < 10**6 for s in s1 + s2)
    import hashlib
    assert s1[0] == int(hashlib.sha256(f"{a.salt}|hw3|1234|0".encode()).hexdigest()[:8], 16) % 10**6
    for p, it in zip(A.build(code, "1234"), a.items):
        assert (p.topic, p.difficulty) == (it.topic, it.difficulty)
        assert p.statement == P.generate(it.topic, p.seed, it.difficulty).statement


def test_student_id_checked(code):
    for sid in ("", "a b", "x" * 41, "../etc"):
        with pytest.raises(A.AssignmentError):
            A.build(code, sid)


# ------------------------------------------------------------------------------------------------ grading

def test_round_trip_full_marks_and_zero(code):
    key = A.answer_key(code, "1234")
    rep = A.grade(code, A.submit(code, "1234", key))
    assert rep["score"] == rep["max_score"] == 5 and rep["percent"] == 100
    assert rep["fingerprint_ok"] and rep["code_ok"] and rep["notes"] == []
    rep0 = A.grade(code, A.submit(code, "1234", _wrong(key)))
    assert rep0["score"] == 0 and rep0["fingerprint_ok"]


def test_another_students_answers_score_nothing(code):
    rep = A.grade(code, A.submit(code, "5678", A.answer_key(code, "1234")))
    assert rep["score"] < 1.0


def test_partial_missing_and_malformed(code):
    key = A.answer_key(code, "1234")
    ans = json.loads(json.dumps(key))
    del ans["0"]                                    # problem 1 skipped
    ans["3"]["tau"] = "1, 2, 3"                     # wrong count
    ans["2"]["zeta"] = "pi/4"                       # not a number
    rep = A.grade(code, A.submit(code, "1234", ans))
    it = rep["items"]
    assert it[0]["score"] == 0 and it[0]["quantities"]["T"]["note"] == "no answer"
    assert it[1]["score"] == 1
    assert it[3]["score"] == 1 and "expected 2" in it[3]["quantities"]["tau"]["note"]
    assert not it[2]["quantities"]["zeta"]["ok"] and 0 < it[2]["score"] < 1
    assert 0 < rep["score"] < rep["max_score"]


def test_typed_text_formats_accepted(code):
    key = A.answer_key(code, "1234")
    ans = {}
    for k, qs in key.items():
        ans[k] = {}
        for q, v in qs.items():
            arr = np.atleast_2d(v) if np.ndim(v) == 2 else np.ravel(v)
            if np.ndim(v) == 2:                                   # "a b c; d e f" row by row
                ans[k][q] = "; ".join(" ".join(f"{x:.6g}" for x in r) for r in arr)
            else:
                ans[k][q] = "[" + ", ".join(f"{x:.6g}" for x in arr) + "]"
    assert A.grade(code, A.submit(code, "1234", ans))["percent"] == 100


@pytest.mark.parametrize("topic", P.TOPIC_KEYS)
def test_rounded_answers_pass_for_every_topic(topic):
    """Typing the numbers rounded to 3 significant figures or to 4 decimals earns full marks; 5 % off does not."""
    for d in (1, 2, 3):
        c = A.make_code("t", "t", [(topic, d)], salt="r")
        for sid in ("s1", "s2", "s3"):
            key = A.answer_key(c, sid)
            for rnd in (lambda x: float(f"{x:.3g}"), lambda x: round(x, 4)):
                ans = {k: {q: [rnd(x) for x in np.ravel(v)] for q, v in qs.items()} for k, qs in key.items()}
                assert A.grade(c, A.submit(c, sid, ans))["percent"] == 100, (topic, d, sid)
            off = {k: {q: (np.ravel(v) * 1.05 + np.where(np.ravel(v) == 0, 0.01, 0)).tolist()
                       for q, v in qs.items()} for k, qs in key.items()}
            res = A.grade(c, A.submit(c, sid, off))["items"][0]["quantities"]
            for q, r in res.items():
                ref = np.ravel(key["0"][q])
                if np.max(np.abs(ref)) > 0.01:                    # big enough for 5 % to exceed the tolerance
                    assert not r["ok"], (topic, d, q)


def test_fingerprint_and_wrong_assignment_flagged(code, tmp_path):
    key = A.answer_key(code, "1234")
    sub = A.submit(code, "1234", key)
    path = A.save_submission(sub, str(tmp_path))
    assert os.path.basename(path) == "1234_hw3.json"
    d = json.load(open(path))
    assert set(d) >= {"assignment_id", "student_id", "app_version", "created", "answers", "fingerprint"}
    assert d["fingerprint"] == A.fingerprint(code, "1234", d["answers"])
    d["answers"]["1"]["det"] = 123.0                               # edited after saving
    rep = A.grade(code, d)
    assert not rep["fingerprint_ok"] and any("fingerprint" in n for n in rep["notes"])
    other = A.make_code("hw4", "x", ITEMS, salt="fixed-salt")
    rep = A.grade(other, A.load_submission(path))
    assert not rep["code_ok"] and any("hw3" in n for n in rep["notes"])


def test_late_flag(code):
    key = A.answer_key(code, "1234")
    assert A.grade(code, A.submit(code, "1234", key, created="2026-11-20T23:00:00+00:00"))["late"] in (False, True)
    assert A.grade(code, A.submit(code, "1234", key, created="2026-11-19T10:00:00"))["late"] is False
    assert A.grade(code, A.submit(code, "1234", key, created="2026-11-22T10:00:00"))["late"] is True
    nodue = A.make_code("n", "n", [("fk", 1)])
    assert A.grade(nodue, A.submit(nodue, "1", {}))["late"] is None


def test_seed_dependent_and_unordered_quantities():
    c = A.make_code("c", "c", [{"topic": "control", "difficulty": 1, "quantities": ["roots"]}], salt="x")
    sids = [f"s{i}" for i in range(12)]
    kinds = {A.build(c, s)[0].answer["kind"] for s in sids}
    assert "overdamped" in kinds and len(kinds) > 1                 # some students have roots, some do not
    for sid in sids:
        p = A.build(c, sid)[0]
        qs = A.quantities(p, A.parse_code(c).items[0])
        assert qs and (("roots" in p.answer) == (list(qs) == ["roots"]))
        key = A.answer_key(c, sid)
        if "roots" in key["0"]:                                     # either order of the two roots is accepted
            key["0"]["roots"] = key["0"]["roots"][::-1]
        assert A.grade(c, A.submit(c, sid, key))["percent"] == 100


def test_parse_numbers():
    assert A.parse_numbers("0.5, -1  2e-3; 4") == [0.5, -1.0, 0.002, 4.0]
    assert A.parse_numbers("[[1, 0], [0, 1]]") == [1, 0, 0, 1]
    assert A.parse_numbers("-.5 +3.") == [-0.5, 3.0]
    for bad in ("", "pi", "1/2", "abc 3"):
        with pytest.raises(ValueError):
            A.parse_numbers(bad)


# ------------------------------------------------------------------------------------------------ CLI tools

def test_cli_new_show_key(tmp_path, capsys):
    import assignment as tool
    out = tmp_path / "code.txt"
    assert tool.main(["new", "--id", "lab2", "--title", "Lab 2", "--due", "2026-12-01", "--salt", "abc",
                      "--out", str(out), "fk:1", "jacobian:2:Jb,singular", "control"]) == 0
    code = capsys.readouterr().out.strip()
    assert out.read_text().strip() == code
    a = A.parse_code(code)
    assert a.items[1].quantities == ["Jb", "singular"] and a.items[2].difficulty == 2
    assert tool.main(["show", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "lab2" in shown and "Jb, singular" in shown
    assert tool.main(["key", code, "42", "--json"]) == 0
    k = json.loads(capsys.readouterr().out)
    assert k["seeds"] == [p.seed for p in A.build(code, "42")]
    assert tool.main(["new", "--id", "x", "mixed:2"]) == 2


def test_grade_assignments_csv(code, tmp_path):
    import grade_assignments as tool
    sub = tmp_path / "subs"
    A.save_submission(A.submit(code, "1234", A.answer_key(code, "1234")), str(sub))
    A.save_submission(A.submit(code, "5678", _wrong(A.answer_key(code, "5678"))), str(sub))
    (sub / "notes.json").write_text("[1, 2]")
    csv_path, det = tmp_path / "grades.csv", tmp_path / "details.json"
    assert tool.main([code, str(sub), "-o", str(csv_path), "--details", str(det)]) == 0
    rows = list(csv.DictReader(open(csv_path)))
    by = {r["file"]: r for r in rows}
    assert by["1234_hw3.json"]["percent"] == "100" and by["1234_hw3.json"]["p4"] == "2"
    assert by["5678_hw3.json"]["score"] == "0"
    assert by["notes.json"]["notes"].startswith("skipped")
    assert len(json.load(open(det))) == 2                          # strict JSON (no Infinity)
    assert tool.main([code, str(sub), "--min-percent", "50"]) == 1
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "grade_assignments.py"), code, str(sub)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("student_id,file,score")


# ------------------------------------------------------------------------------------------------ panel

class FakeHost:
    def __init__(self):
        self.shown, self.html, self.status = [], [], []
        self.ctx = None

    def show_problem(self, p, reveal_allowed=True):
        self.shown.append((p, reveal_allowed))

    def show_html(self, statement_html, solution_html=None):
        self.html.append((statement_html, solution_html))

    def set_status(self, text):
        self.status.append(text)


def test_panel_source_never_reaches_solutions():
    src = open(os.path.join(ROOT, "mrstudio", "gui", "tabs", "assignment_panel.py"), encoding="utf-8").read()
    assert "solution_html" not in src and "show_html" not in src
    assert {m.group(0) for m in re.finditer(r"reveal\w*\s*=\s*\w+", src)} == {"reveal_allowed=False"}


def test_panel_round_trip(qapp, code, tmp_path, monkeypatch):
    monkeypatch.setenv("MRSTUDIO_SETTINGS", str(tmp_path / "settings.ini"))
    from mrstudio.gui.tabs.assignment_panel import AssignmentPanel
    host = FakeHost()
    panel = AssignmentPanel(host)
    panel.code_edit.setPlainText(code)
    panel.sid_edit.setText("bad id")
    assert panel.load() is False and "student ID" in panel.status_lbl.text()
    panel.sid_edit.setText("1234")
    assert panel.load() is True
    assert panel.list.count() == 4 and host.shown and host.shown[-1][0].seed == A.build(code, "1234")[0].seed
    key = A.answer_key(code, "1234")
    for k in range(4):
        panel.list.setCurrentRow(k)
        assert set(panel.fields) == set(key[str(k)])
        for q, v in key[str(k)].items():
            panel.fields[q].setText(" ".join(f"{x:.6g}" for x in np.ravel(v)))
    panel.list.setCurrentRow(0)
    assert all(not ra for _p, ra in host.shown)                    # statements only, never the solution
    assert host.html == []
    path = panel.save(folder=str(tmp_path / "out"))
    assert path and os.path.exists(path) and "Saved 17/17" in panel.status_lbl.text()
    assert A.grade(code, A.load_submission(path))["percent"] == 100
    # the draft survives a restart of the panel
    p2 = AssignmentPanel(FakeHost())
    assert p2.sid_edit.text() == "1234" and p2.load()
    assert p2.answers["0"] == panel.answers["0"]
    # a malformed field is marked, not graded here
    p2.list.setCurrentRow(1)
    f = next(iter(p2.fields.values()))
    f.setText("1 2 x")
    assert "border" in f.styleSheet()
    panel.deleteLater()
    p2.deleteLater()
