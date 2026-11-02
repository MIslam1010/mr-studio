"""Assignment panel of the Practice tab: per-student problems from an instructor's code, typed answers, and a
submission file for the LMS.

The panel only ever shows problem *statements* (``host.show_problem(p, reveal_allowed=False)``); worked solutions
stay out of reach here.  Grading is done by the instructor with ``tools/grade_assignments.py`` (see
``docs/ASSIGNMENTS.md``), so nothing in this panel says whether an answer is right.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ... import assignments as A
from ... import problems as P
from ..widgets import button, group, hbox, value_label

if TYPE_CHECKING:                                   # written by the integration task (T3c)
    from .practice_host import PracticeHost         # noqa: F401


def _settings():
    from ..app import app_settings                  # lazy: app.py imports the tabs
    return app_settings()


class AssignmentPanel(QtWidgets.QWidget):
    """Paste code + student ID -> problem list -> answer fields -> "Save submission"."""

    #: emitted with the path of a saved submission file
    saved = QtCore.Signal(str)

    def __init__(self, host: "PracticeHost", parent=None):
        super().__init__(parent)
        self.host = host
        self.assignment: A.Assignment | None = None
        self.student_id = ""
        self.problems: list[P.Problem] = []
        self.answers: dict[str, dict[str, str]] = {}     # {"k": {quantity: typed text}}
        self.current = -1
        self.fields: dict[str, QtWidgets.QLineEdit] = {}

        self.code_edit = QtWidgets.QPlainTextEdit()
        self.code_edit.setPlaceholderText("Paste the assignment code from your instructor")
        self.code_edit.setFixedHeight(64)
        self.code_edit.setTabChangesFocus(True)
        self.sid_edit = QtWidgets.QLineEdit()
        self.sid_edit.setPlaceholderText("e.g. 20231234")
        self.sid_edit.setMaxLength(40)
        self.sid_edit.returnPressed.connect(self.load)
        self.load_btn = button("Load assignment", self.load, primary=True)
        self.info_lbl = value_label()
        self.info_lbl.setObjectName("muted")
        form = QtWidgets.QFormLayout()
        form.addRow("Student ID", self.sid_edit)

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(90)
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setTextElideMode(QtCore.Qt.ElideNone)
        self.list.currentRowChanged.connect(self.select)

        self.ans_box = QtWidgets.QWidget()
        self.ans_form = QtWidgets.QFormLayout(self.ans_box)
        self.ans_form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.ans_hint = value_label("Load an assignment, pick a problem and type its final numbers.")
        self.ans_hint.setObjectName("muted")
        self.ans_form.addRow(self.ans_hint)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.ans_box)
        scroll.setMinimumHeight(140)

        self.save_btn = button("Save submission", self.save, primary=True,
                               tooltip="Write the submission file to upload to your course page")
        self.folder_btn = button("Show folder", self.show_folder)
        self.save_btn.setEnabled(False)
        self.status_lbl = QtWidgets.QLabel()
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        units = value_label("SI units, angles in radians, 3–4 significant figures. Vectors and matrices: "
                            "numbers separated by commas or spaces, matrices row by row.")
        units.setObjectName("muted")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(group("Assignment", self.code_edit, form, self.load_btn, self.info_lbl))
        lay.addWidget(group("Problems", self.list))
        lay.addWidget(group("Your answers", units, scroll), 1)
        lay.addWidget(group("Submit", hbox(self.save_btn, self.folder_btn), self.status_lbl))

        self._restore()

    # ------------------------------------------------------------------ persistence (drafts survive restarts)
    def _restore(self):
        try:
            s = _settings()
            self.sid_edit.setText(str(s.value("assignments/student_id", "") or ""))
            self.code_edit.setPlainText(str(s.value("assignments/code", "") or ""))
        except Exception:  # noqa: BLE001 -- settings are a convenience only
            pass

    def _draft_key(self) -> str:
        return f"assignments/draft/{self.assignment.id}/{self.student_id}" if self.assignment else ""

    def _store_draft(self):
        if not self.assignment:
            return
        try:
            s = _settings()
            s.setValue(self._draft_key(), json.dumps(self.answers))
            s.setValue("assignments/student_id", self.student_id)
            s.setValue("assignments/code", self.assignment.code)
        except Exception:  # noqa: BLE001
            pass

    def _load_draft(self) -> dict:
        try:
            d = json.loads(str(_settings().value(self._draft_key(), "") or "{}"))
            return {str(k): {str(q): str(v) for q, v in qs.items()} for k, qs in d.items()}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------ actions
    def _status(self, text: str, tone: str | None = None):
        from .. import theme
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")
        try:
            self.host.set_status(text)
        except Exception:  # noqa: BLE001 -- the host's status line is optional
            pass

    def load(self, *_):
        """Decode the code, build this student's problems and fill the list."""
        self._commit_fields()
        try:
            asg = A.parse_code(self.code_edit.toPlainText())
            sid = A.check_student_id(self.sid_edit.text())
        except A.AssignmentError as e:
            self._status(str(e), "bad")
            return False
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            probs = A.build(asg, sid)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.assignment, self.student_id, self.problems = asg, sid, probs
        self.answers = self._load_draft()
        self.current = -1
        due = f" · due {asg.due}" if asg.due else ""
        self.info_lbl.setText(f"{asg.title} ({asg.id}){due} · {len(probs)} problem(s) for student {sid}")
        self.list.blockSignals(True)
        self.list.clear()
        for k in range(len(probs)):
            self.list.addItem(self._row_text(k))
            self.list.item(k).setToolTip(P.TOPICS[asg.items[k].topic]["title"])
        self.list.blockSignals(False)
        self.save_btn.setEnabled(True)
        self._store_draft()
        self._status(f"Loaded {asg.id}: your problems are numbered for student {sid}.", None)
        self.list.setCurrentRow(0)
        return True

    def _row_text(self, k: int) -> str:
        p, it = self.problems[k], self.assignment.items[k]
        n = len(A.quantities(p, it))
        done = sum(1 for v in (self.answers.get(str(k)) or {}).values() if str(v).strip())
        return f"{k + 1}. {P.TOPICS[it.topic]['title']} [{P.DIFFICULTIES[it.difficulty]}] — {done}/{n} answered"

    def select(self, row: int):
        if row < 0 or row >= len(self.problems):
            return
        self._commit_fields()
        self.current = row
        p, it = self.problems[row], self.assignment.items[row]
        self.host.show_problem(p, reveal_allowed=False)
        self._build_fields(p, it)

    def _build_fields(self, p: P.Problem, it: A.Item):
        while self.ans_form.rowCount():
            self.ans_form.removeRow(0)
        self.fields = {}
        saved = self.answers.get(str(self.current)) or {}
        head = value_label(f"Problem {self.current + 1} · seed {p.seed}")
        head.setObjectName("muted")
        self.ans_form.addRow(head)
        for name, ref in A.quantities(p, it).items():
            e = QtWidgets.QLineEdit()
            n = int(np.size(ref))
            e.setPlaceholderText("one number" if n == 1 else f"{n} numbers")
            e.setText(saved.get(name, ""))
            e.editingFinished.connect(self._commit_fields)
            e.textChanged.connect(lambda _t, w=e: self._mark(w))
            self.fields[name] = e
            self.ans_form.addRow(A.label(name, np.shape(ref)), e)
            self._mark(e)

    def _mark(self, e: QtWidgets.QLineEdit):
        """Red border while the text is not a list of the right number of numbers (format only, not correctness)."""
        name = next((k for k, w in self.fields.items() if w is e), None)
        if name is None:
            return
        text = e.text().strip()
        bad = False
        if text:
            try:
                n = len(A.parse_numbers(text))
                p, it = self.problems[self.current], self.assignment.items[self.current]
                bad = n != int(np.size(A.quantities(p, it)[name]))
            except (ValueError, KeyError):
                bad = True
        from .. import theme
        e.setStyleSheet(f"border:1px solid {theme.bad_color()}" if bad else "")
        e.setToolTip("Format: check the count of numbers and use only digits, '.', '-', 'e', commas or spaces"
                     if bad else "")

    def _commit_fields(self):
        if self.current < 0 or not self.fields:
            return
        self.answers[str(self.current)] = {q: e.text().strip() for q, e in self.fields.items()}
        item = self.list.item(self.current)
        if item is not None:
            item.setText(self._row_text(self.current))
        self._store_draft()

    def set_answer(self, k: int, name: str, text: str):
        """Type an answer programmatically (tests, demos)."""
        self.answers.setdefault(str(k), {})[name] = str(text)
        if k == self.current and name in self.fields:
            self.fields[name].setText(str(text))

    def submission(self) -> A.Submission:
        self._commit_fields()
        if not self.assignment:
            raise A.AssignmentError("load an assignment first")
        typed = {k: {q: v for q, v in qs.items() if str(v).strip()} for k, qs in self.answers.items()}
        return A.submit(self.assignment, self.student_id, typed)

    def save(self, *_, folder: str | None = None) -> str | None:
        try:
            sub = self.submission()
            path = A.save_submission(sub, folder)
        except (A.AssignmentError, OSError) as e:
            self._status(f"Not saved: {e}", "bad")
            return None
        total = sum(len(A.quantities(p, it)) for p, it in zip(self.problems, self.assignment.items))
        done = sum(1 for qs in sub.answers.values() for v in qs.values() if str(v).strip())
        self._status(f"Saved {done}/{total} answers to {path} — upload this file to your course page.", "good")
        self.saved.emit(path)
        return path

    def show_folder(self, *_):
        d = A.submissions_dir()
        os.makedirs(d, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))
