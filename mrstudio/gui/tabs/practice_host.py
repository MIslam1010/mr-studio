"""The interface the Practice tab offers to its side panels (Lessons, Assignment).

``LessonsPanel`` and ``AssignmentPanel`` are plain ``QWidget``s that only talk to the Practice tab through this
protocol, so they can be tested with a small fake host.  ``PracticeTab`` implements it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ... import problems
    from ..base import AppContext


@runtime_checkable
class PracticeHost(Protocol):
    ctx: "AppContext"

    def show_problem(self, p: "problems.Problem", reveal_allowed: bool = True) -> None:
        """Show problem ``p`` (statement + visual). With ``reveal_allowed=False`` the worked solution is not
        reachable: the Reveal button is disabled and the solution pane shows a notice."""

    def show_html(self, statement_html: str, solution_html: str | None = None) -> None:
        """Show free HTML in the statement pane and (optionally) the solution pane; no problem is active."""

    def set_status(self, text: str) -> None:
        """Show a short message in the main window's status bar."""
