"""Which edition is running: student (the public repository) or instructor (the private copy).

The private instructor copy adds a module ``mrstudio/edition.py`` (it is not in this repository).  Everything
here asks for it through :func:`is_instructor` / :func:`edition_name`, which fall back to the student edition
when the module is missing or broken, so the same code runs in both.

This is a convenience, not secrecy: the student build hides instructor controls (problem-set export, myCobot
calibration editing) so students are not tempted by them; the generators and the calibration file format are
open source.
"""
from __future__ import annotations

import importlib


def _edition_module():
    """The private ``mrstudio.edition`` module, or None (public repository / import error)."""
    try:
        return importlib.import_module("mrstudio.edition")
    except Exception:  # noqa: BLE001 -- missing module or any error inside it: student edition
        return None


def edition_name() -> str:
    """'instructor' when the private edition module is present and says so, else 'student'."""
    mod = _edition_module()
    if mod is None:
        return "student"
    try:
        return "instructor" if str(getattr(mod, "EDITION", "")).lower() == "instructor" else "student"
    except Exception:  # noqa: BLE001
        return "student"


def is_instructor() -> bool:
    return edition_name() == "instructor"
