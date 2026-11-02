"""Keep the three places that state package versions in agreement (standard library only).

    python packaging/sync_pins.py           rewrite the files that disagree, list what changed
    python packaging/sync_pins.py --check   change nothing; exit code 1 if a file would change

1. packaging/requirements-lock.txt  <-  requirements.txt (+ the exact pins of requirements-dev.txt)
   requirements.txt holds the validated pins and is the source of truth. Dependabot bumps both files, but not
   always alike: it resolves requirements.txt (e.g. it kept numpy where the lock's update proposed a minor
   upgrade), while the lock -- a `pip freeze` -- is bumped package by package. So a direct dependency's version
   is copied INTO the lock, never out of it. Packages released in lockstep with a direct dependency follow it
   (PySide6 -> shiboken6, PySide6_Addons, PySide6_Essentials; pin -> libpinocchio).
2. THIRD_PARTY_NOTICES.md (version column)  <-  the lock (what the installers bundle).

Idempotent: a second run changes nothing. tests/test_packaging.py checks the same agreements.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"
LOCK = ROOT / "packaging" / "requirements-lock.txt"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"

# distributions released with the same version number as a direct dependency (normalised names)
COMPANIONS = {
    "pyside6": ["shiboken6", "pyside6-addons", "pyside6-essentials"],
    "pin": ["libpinocchio"],
}

# THIRD_PARTY_NOTICES.md table row (first cell) -> distribution in the lock (normalised name)
NOTICE_ROWS = {
    "Qt for Python (PySide6, shiboken6) and the Qt libraries": "pyside6",
    "Pinocchio (`pin`)": "pin",
    "Coal (collision library used by Pinocchio)": "coal",
    "EigenPy": "eigenpy",
    "Robotics Toolbox for Python": "roboticstoolbox-python",
    "Spatial Maths for Python": "spatialmath-python",
    "modern_robotics (Lynch & Park)": "modern-robotics",
    "toppra": "toppra",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "SymPy": "sympy",
    "Matplotlib": "matplotlib",
    "pyqtgraph": "pyqtgraph",
    "PyOpenGL": "pyopengl",
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio-ffmpeg",
    "Pillow": "pillow",
    "pyserial": "pyserial",
}

_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<op>\s*==\s*)(?P<ver>[^\s#;]+)")


def norm(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def exact_pins(text: str) -> dict[str, str]:
    """{normalised name: version} of every `name==version` line (comments and ranges ignored)."""
    out = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            out[norm(m["name"])] = m["ver"]
    return out


def read(path: Path) -> str:
    """File text with its line endings kept (a rewrite must not turn CRLF into LF or back)."""
    if not path.is_file():
        return ""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def sync_lock(lock_text: str, wanted: dict[str, str]) -> tuple[str, list[str]]:
    """Set the version of each lock line whose package is in ``wanted``. Packages absent from the lock are not
    added (the lock is a freeze of the tested venv; a new dependency means re-freezing it)."""
    changes, lines = [], []
    for line in lock_text.splitlines(keepends=True):
        m = _PIN.match(line)
        if m and norm(m["name"]) in wanted and m["ver"] != wanted[norm(m["name"])]:
            new = wanted[norm(m["name"])]
            changes.append(f"{LOCK.relative_to(ROOT)}: {m['name']} {m['ver']} -> {new}")
            line = line[:m.start("ver")] + new + line[m.end("ver"):]
        lines.append(line)
    return "".join(lines), changes


def sync_notices(text: str, lock: dict[str, str]) -> tuple[str, list[str]]:
    """Set the version cell of each known row of the THIRD_PARTY_NOTICES.md table from the lock."""
    changes, lines = [], []
    for line in text.splitlines(keepends=True):
        cells = line.split("|")
        if len(cells) > 3 and cells[1].strip() in NOTICE_ROWS:
            row, pkg = cells[1].strip(), NOTICE_ROWS[cells[1].strip()]
            old, new = cells[2].strip(), lock.get(pkg)
            if new is None:
                raise SystemExit(f"{NOTICES.name}: row '{row}' maps to {pkg!r}, which is not in the lock")
            if old != new:
                changes.append(f"{NOTICES.name}: {row}: {old} -> {new}")
                cells[2] = cells[2].replace(old, new, 1) if old else f" {new} "
                line = "|".join(cells)
        lines.append(line)
    return "".join(lines), changes


def wanted_lock_pins() -> dict[str, str]:
    """Versions the lock must have: requirements.txt, the exact pins of requirements-dev.txt, companions."""
    wanted = {**exact_pins(read(REQUIREMENTS_DEV)), **exact_pins(read(REQUIREMENTS))}
    for pkg, companions in COMPANIONS.items():
        if pkg in wanted:
            for c in companions:
                wanted.setdefault(c, wanted[pkg])
    return wanted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if any file would change")
    args = ap.parse_args(argv)

    lock_old = read(LOCK)
    lock_new, changes = sync_lock(lock_old, wanted_lock_pins())
    notices_old = read(NOTICES)
    notices_new, more = sync_notices(notices_old, exact_pins(lock_new))
    changes += more

    for c in changes:
        print(("would change " if args.check else "changed ") + c)
    if args.check:
        if changes:
            print("Pins disagree: run `python packaging/sync_pins.py` and commit the result.")
            return 1
        print("pins agree: requirements.txt -> requirements-lock.txt -> THIRD_PARTY_NOTICES.md")
        return 0
    if lock_new != lock_old:
        write(LOCK, lock_new)
    if notices_new != notices_old:
        write(NOTICES, notices_new)
    if not changes:
        print("pins already agree; nothing changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
