#!/usr/bin/env python3
"""Make and inspect MR Studio assignment codes (instructor tool).

Examples (run from the MR-Studio folder with the project's Python)::

    # list the topics and the quantities students answer for each difficulty
    python tools/assignment.py topics

    # a code with three problems, due 20 Nov 2026; give the printed code to the students
    python tools/assignment.py new --id hw3 --title "Homework 3" --due 2026-11-20 \\
        fk:2 jacobian:2:Jb,singular control:1

    # what is inside a code (no answers)
    python tools/assignment.py show <CODE>

    # one student's seeds and answers (instructor only -- do not publish)
    python tools/assignment.py key <CODE> 20231234

Item syntax: ``topic[:difficulty[:quantity,quantity…]]``; difficulty 1 basic, 2 standard (default), 3 challenge.
Keep the printed code (or the --out file): the grader needs exactly the same code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrstudio import assignments as A          # noqa: E402
from mrstudio import problems as P             # noqa: E402


def read_code(arg: str) -> str:
    """The code itself, or the contents of a file that holds it."""
    if os.path.isfile(arg):
        with open(arg, encoding="utf-8") as f:
            return f.read().strip()
    return arg.strip()


def _parse_item(text: str, rtol: float | None, atol: float | None) -> A.Item:
    parts = text.split(":")
    topic = parts[0]
    diff = int(parts[1]) if len(parts) > 1 and parts[1] else 2
    qs = [q for q in parts[2].split(",") if q] if len(parts) > 2 else None
    it = A.Item(topic, diff, qs or None)
    if rtol is not None:
        it.rtol = rtol
    if atol is not None:
        it.atol = atol
    return it


def cmd_new(a) -> int:
    items = [_parse_item(t, a.rtol, a.atol) for t in a.items]
    code = A.make_code(a.id, a.title or a.id, items, due=a.due or "", salt=a.salt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(code + "\n")
        print(f"wrote {a.out}", file=sys.stderr)
    print(code)
    return 0


def cmd_show(a) -> int:
    asg = A.parse_code(read_code(a.code))
    print(f"id:    {asg.id}\ntitle: {asg.title}\ndue:   {asg.due or '-'}\nitems:")
    for k, it in enumerate(asg.items, 1):
        p = P.generate(it.topic, 0, it.difficulty)
        qs = [q for q in A.quantities(p, it) if q not in A.SEED_DEPENDENT.get(it.topic, ())]
        extra = [q for q in A.SEED_DEPENDENT.get(it.topic, ()) if not it.quantities or q in it.quantities]
        more = f" (+ {' or '.join(extra)}, depending on the student's numbers)" if extra else ""
        print(f"  {k}. {it.topic} (difficulty {it.difficulty}, {it.points:g} pt, rtol {it.rtol:g}, atol {it.atol:g})"
              f": {', '.join(qs)}{more}")
    return 0


def cmd_key(a) -> int:
    asg = A.parse_code(read_code(a.code))
    probs = A.build(asg, a.student)
    key = A.answer_key(asg, a.student)
    if a.json:
        print(json.dumps({"seeds": [p.seed for p in probs], "answers": key}, indent=2))
        return 0
    for k, (it, p) in enumerate(zip(asg.items, probs)):
        print(f"Problem {k + 1}: {P.TOPICS[it.topic]['title']} -- difficulty {it.difficulty}, seed {p.seed}")
        for q, v in key[str(k)].items():
            print(f"  {q:16s} {json.dumps(v)}")
    return 0


def cmd_topics(_a) -> int:
    for t in P.TOPIC_KEYS:
        print(f"{t}: {P.TOPICS[t]['title']}")
        for d in (1, 2, 3):
            qs = [q for q in A.gradable(P.generate(t, 0, d)) if q not in A.SEED_DEPENDENT.get(t, ())]
            extra = A.SEED_DEPENDENT.get(t, ())
            more = f" (+ {' or '.join(extra)}, depending on the numbers)" if extra else ""
            print(f"  difficulty {d}: {', '.join(qs)}{more}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="make an assignment code")
    n.add_argument("--id", required=True, help="short id, e.g. hw3 (letters, digits, . _ -)")
    n.add_argument("--title", default="")
    n.add_argument("--due", default="", help="ISO date, e.g. 2026-11-20 (end of that day)")
    n.add_argument("--salt", default=None, help="fixed salt (default: random; keep the code!)")
    n.add_argument("--rtol", type=float, default=None, help=f"relative tolerance (default {A.DEFAULT_RTOL})")
    n.add_argument("--atol", type=float, default=None, help=f"absolute tolerance (default {A.DEFAULT_ATOL})")
    n.add_argument("--out", default=None, help="also write the code to this file")
    n.add_argument("items", nargs="+", help="topic[:difficulty[:q1,q2]]")
    n.set_defaults(fn=cmd_new)
    s = sub.add_parser("show", help="decode a code (no answers)")
    s.add_argument("code", help="the code or a file containing it")
    s.set_defaults(fn=cmd_show)
    k = sub.add_parser("key", help="one student's seeds and answers (instructor only)")
    k.add_argument("code")
    k.add_argument("student")
    k.add_argument("--json", action="store_true")
    k.set_defaults(fn=cmd_key)
    t = sub.add_parser("topics", help="topics and gradable quantities")
    t.set_defaults(fn=cmd_topics)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except A.AssignmentError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
