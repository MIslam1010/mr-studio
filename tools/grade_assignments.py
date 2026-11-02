#!/usr/bin/env python3
"""Grade a folder of MR Studio submissions against an assignment code and write a CSV.

    python tools/grade_assignments.py <CODE or code-file> <folder> [-o grades.csv] [--details details.json]
                                      [--min-percent 0]

Every ``*.json`` file in the folder (recursively) is read as a submission.  Each student's problems are
regenerated from the code and their student ID, and every typed number is compared with the computed answer.
One CSV row per file: student, file, score, max, percent, per-problem scores, late flag (student's clock),
fingerprint/code checks and notes.  Files that are not submissions get a row with a note instead of stopping
the run.  Exit status: 0, or 1 when ``--min-percent`` is given and some submission scored below it (useful in
CI).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrstudio import assignments as A          # noqa: E402


def read_code(arg: str) -> str:
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            return f.read().strip()
    return arg.strip()


def find_submissions(folder: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(".json")]
    return sorted(out)


def grade_folder(code: str, folder: str) -> tuple[list[str], list[list], list[dict]]:
    asg = A.parse_code(code)
    n = len(asg.items)
    header = (["student_id", "file", "score", "max_score", "percent"] + [f"p{k + 1}" for k in range(n)]
              + ["late", "fingerprint_ok", "code_ok", "created", "app_version", "notes"])
    rows, details = [], []
    for path in find_submissions(folder):
        rel = os.path.relpath(path, folder)
        try:
            sub = A.load_submission(path)
            rep = A.grade(asg, sub)
        except A.AssignmentError as e:
            rows.append(["", rel, "", "", ""] + [""] * n + ["", "", "", "", "", f"skipped: {e}"])
            continue
        details.append({"file": rel, **rep})
        late = "" if rep["late"] is None else ("yes" if rep["late"] else "no")
        rows.append([rep["student_id"], rel, f"{rep['score']:g}", f"{rep['max_score']:g}", f"{rep['percent']:g}"]
                    + [f"{it['score']:g}" for it in rep["items"]]
                    + [late, rep["fingerprint_ok"], rep["code_ok"], rep["created"], sub.app_version,
                       "; ".join(rep["notes"])])
    return header, rows, details


def _finite(x):
    """JSON-safe copy: inf / nan (no answer, wrong length) become null."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("code", help="assignment code, or a file containing it")
    ap.add_argument("folder", help="folder with the students' submission .json files")
    ap.add_argument("-o", "--out", default=None, help="CSV file (default: print to stdout)")
    ap.add_argument("--details", default=None, help="also write per-quantity results to this JSON file")
    ap.add_argument("--min-percent", type=float, default=None, help="exit 1 if any submission scores below this")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.folder):
        print(f"error: {a.folder} is not a folder", file=sys.stderr)
        return 2
    try:
        header, rows, details = grade_folder(read_code(a.code), a.folder)
    except A.AssignmentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.out:
        f = open(a.out, "w", newline="", encoding="utf-8")
    else:
        f = sys.stdout
    try:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    finally:
        if a.out:
            f.close()
    if a.details:
        with open(a.details, "w", encoding="utf-8") as fd:
            json.dump(_finite(details), fd, indent=2)
    print(f"graded {len(details)} submission(s), {len(rows) - len(details)} skipped", file=sys.stderr)
    if a.min_percent is not None and any(d["percent"] < a.min_percent for d in details):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
