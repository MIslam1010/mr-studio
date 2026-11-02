"""Lessons panel of the Practice tab: guided step-by-step lessons with typed-answer checks (``mrstudio.lessons``).

``LessonsPanel(host)`` is a plain ``QWidget`` that talks to the Practice tab only through the ``PracticeHost``
interface (``show_html``, ``set_status``; see ``practice_host.py``).  "Go to tab" emits
``request_tab(tab_class_name, robot_name)`` (robot name "" = keep the current robot); the Practice tab connects
it to the main window.  Progress is stored in ``QSettings`` under ``lessons/<id>/step`` (number of steps done)
and ``lessons/current`` (last lesson opened).
"""
from __future__ import annotations

import html

from PySide6 import QtCore, QtWidgets

from ... import lessons as LS
from ..widgets import button, group, hbox, value_label

HINT_PLACEHOLDER = ("<h2>Hints</h2><p style='color:gray'>Work the step out first (by hand, or with the tab the "
                    "<b>Go to tab</b> button opens), then type your number and press <b>Check</b>.  "
                    "A hint appears here after your first try.</p>")
SHOW_AFTER = 3          # wrong tries before the expected value is shown


def _long_name(cls_name: str) -> str:
    try:
        from ..app import TAB_LONG_NAMES
        return TAB_LONG_NAMES.get(cls_name, cls_name)
    except Exception:       # pragma: no cover - app module unavailable
        return cls_name


class LessonsPanel(QtWidgets.QWidget):
    #: (tab class name, robot name or "") -- the Practice tab switches the main window to that tab
    request_tab = QtCore.Signal(str, str)
    #: (lesson id, steps done) after every change of progress
    progress_changed = QtCore.Signal(str, int)

    def __init__(self, host, settings: QtCore.QSettings | None = None, parent=None):
        super().__init__(parent)
        self.host = host
        self._settings = settings
        self.lessons = LS.all_lessons()
        self.lesson: LS.Lesson | None = None
        self.k = 0
        self._tries: dict[tuple[str, int], int] = {}

        self.lesson_box = QtWidgets.QComboBox()
        for L in self.lessons:
            self.lesson_box.addItem(f"{L.course} W{L.week:02d} · {L.title}", L.id)
            self.lesson_box.setItemData(self.lesson_box.count() - 1,
                                        f"{L.label} — {len(L.steps)} steps, about {L.minutes} min ({L.section})",
                                        QtCore.Qt.ToolTipRole)
        self.lesson_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.lesson_box.setMinimumContentsLength(14)
        self.progress_lbl = value_label()
        self.progress_lbl.setObjectName("muted")
        self.step_list = QtWidgets.QListWidget()
        self.step_list.setMinimumHeight(110)
        self.step_list.setMaximumHeight(150)
        self.goto_btn = button("Go to tab ▸", self._goto)
        self.answer_lbl = value_label()
        self.answer = QtWidgets.QLineEdit()
        self.answer.setPlaceholderText("number, e.g. 0.42 or 0.6*cos(pi/6)")
        self.answer.setClearButtonEnabled(True)
        self.check_btn = button("Check", self.check, primary=True)
        self.feedback = value_label()
        self.back_btn = button("◂ Back", self.back)
        self.next_btn = button("Next ▸", self.next)
        self.reset_btn = button("Restart lesson", self.restart, tooltip="Forget the progress of this lesson")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(group("Lesson", self.lesson_box, self.progress_lbl, self.step_list))
        lay.addWidget(group("This step", self.goto_btn, self.answer_lbl, hbox(self.answer, self.check_btn),
                            self.feedback, hbox(self.back_btn, self.next_btn)))
        lay.addWidget(self.reset_btn)
        lay.addStretch(1)

        self.lesson_box.currentIndexChanged.connect(self._lesson_selected)
        self.step_list.currentRowChanged.connect(self._row_selected)
        self.answer.returnPressed.connect(self.check)

        cur = self.settings().value("lessons/current", "")
        idx = self.lesson_box.findData(cur) if cur else -1
        self.lesson_box.blockSignals(True)
        self.lesson_box.setCurrentIndex(max(idx, 0))
        self.lesson_box.blockSignals(False)
        self._lesson_selected(self.lesson_box.currentIndex())

    # ------------------------------------------------------------------ progress
    def settings(self) -> QtCore.QSettings:
        if self._settings is None:
            from ..app import app_settings
            self._settings = app_settings()
        return self._settings

    def done(self, lesson_id: str | None = None) -> int:
        """Number of steps finished in a lesson (0 .. len(steps))."""
        lid = lesson_id or (self.lesson.id if self.lesson else "")
        try:
            return max(0, int(self.settings().value(f"lessons/{lid}/step", 0)))
        except (TypeError, ValueError):
            return 0

    def _set_done(self, n: int):
        self.settings().setValue(f"lessons/{self.lesson.id}/step", int(n))
        self.settings().sync()
        self.progress_changed.emit(self.lesson.id, int(n))

    # ------------------------------------------------------------------ navigation
    def select_lesson(self, lesson_id: str):
        i = self.lesson_box.findData(lesson_id)
        if i < 0:
            raise KeyError(lesson_id)
        self.lesson_box.setCurrentIndex(i)
        if i == self.lesson_box.currentIndex() and self.lesson is not None and self.lesson.id != lesson_id:
            self._lesson_selected(i)

    def _lesson_selected(self, i: int):
        if i < 0:
            return
        self.lesson = self.lessons[i]
        self.settings().setValue("lessons/current", self.lesson.id)
        n = len(self.lesson.steps)
        self._fill_steps()
        self.show_step(min(self.done(), n - 1))

    def _fill_steps(self):
        L, d = self.lesson, self.done()
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for k, s in enumerate(L.steps):
            mark = "✓" if k < d else ("▸" if k == d else "·")
            self.step_list.addItem(f"{mark} {k + 1}. {s.check.label}")
        self.step_list.blockSignals(False)
        total = len(L.steps)
        state = "finished ✓" if d >= total else f"{d} of {total} steps done"
        self.progress_lbl.setText(f"{L.course}, week {L.week} · about {L.minutes} min · {state}")

    def _row_selected(self, row: int):
        if row >= 0 and row != self.k:
            self.show_step(row)

    def show_step(self, k: int):
        L = self.lesson
        self.k = max(0, min(int(k), len(L.steps) - 1))
        s = L.steps[self.k]
        self.step_list.blockSignals(True)
        self.step_list.setCurrentRow(self.k)
        self.step_list.blockSignals(False)
        unit = f" [{s.check.unit}]" if s.check.unit else ""
        self.answer_lbl.setText(f"Step {self.k + 1}: {s.check.label}{unit}")
        self.answer.clear()
        rob = f" — {s.target.robot}" if s.target.robot else ""
        # "&" would become a keyboard mnemonic ("Trajectory & control")
        self.goto_btn.setText(f"Go to {_long_name(s.target.tab)}{rob} ▸".replace("&", "&&"))
        self.goto_btn.setToolTip("Open the tab where you can see this step in the app")
        solved = self.k < self.done()
        self.feedback.setText("Already solved — press Next or redo it." if solved else "")
        self._tone(None)
        self.back_btn.setEnabled(self.k > 0)
        self.next_btn.setEnabled(solved and self.k < len(L.steps) - 1)
        self.host.show_html(L.step_html(self.k), self._hint_html() if self._tries.get((L.id, self.k)) else
                            HINT_PLACEHOLDER)
        self.host.set_status(f"Lesson: {L.title} — step {self.k + 1} of {len(L.steps)}")

    def _hint_html(self) -> str:
        s = self.lesson.steps[self.k]
        return "<h2>Hint</h2>" + (s.hint_html or "<p>Re-read the step and check your units.</p>")

    def back(self):
        if self.k > 0:
            self.show_step(self.k - 1)

    def next(self):
        if self.k < len(self.lesson.steps) - 1:
            self.show_step(self.k + 1)

    def restart(self):
        self._set_done(0)
        for key in [k for k in self._tries if k[0] == self.lesson.id]:
            del self._tries[key]
        self._fill_steps()
        self.show_step(0)

    def _goto(self):
        t = self.lesson.steps[self.k].target
        self.request_tab.emit(t.tab, t.robot or "")

    # ------------------------------------------------------------------ checking
    def _tone(self, tone):
        from .. import theme
        self.feedback.setProperty("tone", tone)
        self.feedback.setStyleSheet(f"color:{getattr(theme, tone + '_color')()}" if tone else "")

    def check(self) -> LS.CheckResult:
        L, s = self.lesson, self.lesson.steps[self.k]
        res = s.check.grade(self.answer.text())
        if res.value is None:                         # unreadable input: not counted as a try
            self.feedback.setText(res.message)
            self._tone("warn")
            return res
        key = (L.id, self.k)
        if res.ok:
            self._tries.pop(key, None)
            if self.done() <= self.k:
                self._set_done(self.k + 1)
            last = self.k == len(L.steps) - 1
            self.feedback.setText(f"Correct ✓  (reference {s.check.reference_text()})"
                                  + ("  — lesson finished!" if last else ""))
            self._tone("good")
            self.next_btn.setEnabled(not last)
            self._fill_steps()
            self.step_list.blockSignals(True)
            self.step_list.setCurrentRow(self.k)
            self.step_list.blockSignals(False)
            self.host.set_status(f"Lesson: {L.title} — step {self.k + 1} correct"
                                 + (" · lesson finished" if last else ""))
        else:
            n = self._tries.get(key, 0) + 1
            self._tries[key] = n
            msg = res.message + " Look at the hint."
            if n >= SHOW_AFTER:
                msg += f"  Expected value: {s.check.reference_text()} — redo the step with it in mind."
            self.feedback.setText(msg)
            self._tone("bad")
            self.host.show_html(L.step_html(self.k), self._hint_html())
            self.host.set_status(f"Lesson: try {n} on step {self.k + 1}")
        return res

    # used by the Practice tab when it re-shows the panel
    def refresh(self):
        self.show_step(self.k)


def lesson_summary_html() -> str:
    """Short HTML list of all lessons (for docs / the Practice tab's theory box)."""
    rows = "".join(f"<li>{html.escape(L.label)} ({len(L.steps)} steps, ~{L.minutes} min)</li>"
                   for L in LS.all_lessons())
    return f"<ul>{rows}</ul>"
