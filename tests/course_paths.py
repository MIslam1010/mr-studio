"""Where the SBEG 361/461 course materials live on the author's machine (the comparison tests skip without them).

The materials are kept in one or more folders named ``Robotics Track Course Materials*`` (a working copy and dated
revisions); for each file the copy modified most recently wins.
"""
from __future__ import annotations

from pathlib import Path

COURSE_ROOT = Path.home() / "Documents/Edu/Lectures/Robotics"


def newest_course_file(rel: str) -> Path:
    """The most recently modified ``Robotics Track Course Materials*/_build/<rel>``; a non-existent path if none."""
    found = [p for p in COURSE_ROOT.glob("Robotics Track Course Materials*/_build") if (p / rel).is_file()]
    if not found:
        return COURSE_ROOT / "Robotics Track Course Materials/_build" / rel
    return max((p / rel for p in found), key=lambda f: (f.stat().st_mtime, str(f)))
