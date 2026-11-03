"""Graded assignments: per-student practice problems, student submissions and instructor-side grading.

How it fits together
--------------------
* The instructor writes an **assignment code** with ``tools/assignment.py new …``.  The code is the base64url
  encoding (no padding) of a small JSON object::

      {"v": 1, "id": "hw3", "title": "Homework 3", "due": "2026-11-20",
       "items": [{"topic": "fk", "difficulty": 2}, {"topic": "jacobian", "difficulty": 2, "quantities": ["Jb"]}],
       "salt": "k3J…"}

* Each student pastes the code and their student ID into the Practice tab's *Assignment* panel.  Problem ``k`` is
  ``problems.generate(topic_k, seed_k, difficulty_k)`` with the **per-student seed**
  ``int(sha256(f"{salt}|{id}|{student_id}|{k}")[:8], 16) % 10**6`` (:func:`student_seed`), so every student gets
  different numbers, and the instructor can recreate any student's problems from the code alone.
* The student types the final numbers and saves a :class:`Submission` JSON file
  (``~/Documents/MR Studio/submissions/<student>_<id>.json``) that is uploaded to the LMS.
* The instructor (or a CI job in the private course repo) runs :func:`grade` / ``tools/grade_assignments.py``,
  which regenerates each student's problems and compares the typed numbers with the computed answers.

What this is *not*: a way to keep answers secret.  The generators are open source, so a determined student can
recompute any answer (the Practice tab shows worked solutions for any seed).  The scheme only makes copying a
classmate's numbers useless and keeps grading on the instructor's side.  The fingerprint in a submission detects
accidental edits or a file saved for a different code / student; anyone who knows the code can recompute it,
so it is not a signature.

The answer of a problem is its ``Problem.answer`` dict.  Gradable quantities are its numeric entries (numbers,
booleans as 0/1, numeric arrays); text, lists of iteration rows and nested dicts are not graded.  Values are
compared in SI units with angles in **radians** (as in the worked solutions), tolerance per quantity
``max|a - r| <= rtol * max|r| + atol`` (max over the entries of a vector/matrix) with ``rtol = 1e-2`` and
``atol = 1e-4`` unless the item sets its own: about three significant figures, four decimals near zero.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import os
import re
import secrets
from dataclasses import dataclass, field

import numpy as np

from . import problems as P

CODE_VERSION = 1
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-4
STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
ASSIGNMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")

#: short labels (with units) for the answer quantities of the practice topics; anything else shows its key
LABELS = {
    "R": "R (rotation matrix)", "omega2": "ω̂ (unit axis)", "theta2": "θ (rad)", "S": "𝒮 = (ω, v)",
    "T": "T", "S4": "𝒮 (screw axis)", "theta4": "θ (rad)", "h4": "pitch h (m/rad)", "q4": "q (m)",
    "Slist": "space screw axes 𝒮ᵢ (columns)", "Blist": "body screw axes ℬᵢ (columns)",
    "Js": "J_s", "Jb": "J_b", "det": "det", "rank": "rank", "singular": "singular? (1 = yes, 0 = no)",
    "theta": "θ (rad)", "iterations": "iterations",
    "Ib": "𝓘_b (kg m²)", "Gb": "𝒢_b", "Ia": "𝓘_a (kg m²)", "Ga": "𝒢_a", "Ic": "𝓘_c (kg m²)", "KE": "kinetic energy (J)",
    "M": "mass matrix M(θ)", "c": "c(θ, θ̇) (N m)", "g": "g(θ) (N m)", "tau": "τ (N m)",
    "ddtheta": "θ̈ (rad/s²)", "theta_explicit": "θ after explicit Euler (rad)",
    "dtheta_explicit": "θ̇ after explicit Euler (rad/s)", "theta_semi": "θ after semi-implicit Euler (rad)",
    "dtheta_semi": "θ̇ after semi-implicit Euler (rad/s)",
    "J": "J(θ)", "Lam": "Λ(θ)", "m_u": "felt mass m_û (kg)", "Lam_eig": "eigenvalues of Λ",
    "M_rot": "M with rotor inertia", "Lam_rot": "Λ with rotor inertia", "m_u_rot": "felt mass with rotors (kg)",
    "A": "A(θ)", "Adot_dtheta": "Ȧθ̇", "lam": "λ", "tau_con": "constraint torque Aᵀλ (N m)",
    "power": "constraint power (W)", "ddtheta_free": "θ̈ without the constraint (rad/s²)",
    "coef": "polynomial coefficients a₀ … aₙ", "s_t1": "s(t₁)", "theta_t1": "θ(t₁) (rad)",
    "vmax": "max |θ̇| (rad/s)", "amax": "max |θ̈| (rad/s²)",
    "zeta": "ζ", "wn": "ωₙ (rad/s)", "wd": "ω_d (rad/s)", "overshoot": "overshoot (%)",
    "ts": "t_s ≈ 4/(ζωₙ) (s)", "ts_exact": "exact 2 % settling time (s)", "kd_critical": "K_d for ζ = 1",
    "kp": "K_p", "kd": "K_d",
    "Fs": "tip wrench 𝓕_s = (m_z, f_x, f_y)", "Js_p": "J_s, planar rows (ω_z, v_x, v_y)",
    "Jb_p": "J_b, planar rows (ω_z, v_x, v_y)", "Jv": "tip Jacobian J_v (m/rad)", "f_star": "tip force f* (N)",
    "Fb_star": "tool wrench 𝓕_b* (N m, N, N)", "ell_axes": "velocity-ellipse semi-axes √λᵢ",
    "force_axes": "force-ellipse semi-axes 1/√λᵢ", "mu1": "isotropy μ₁",
    "x_ss": "steady-state deflection (m)", "K_imp": "rendered stiffness K (N/m)", "B_imp": "damping B (N s/m)",
    "f_tip_P": "contact force, P control (N)", "fe_P": "force error, P control (N)", "wn_P": "ωₙ, P control (rad/s)",
    "zeta_P": "ζ, P control", "f_tip_PI": "contact force, PI control (N)",
    "char_poly": "characteristic polynomial coefficients (s³ … s⁰)", "kfi_max": "largest stable k_fi (1/s)",
    "stable": "stable? (1 = yes, 0 = no)",
    "F": "wrench matrix F (columns 𝓕ᵢ)", "closed": "closure? (1 = yes, 0 = no)", "mu_min": "minimum friction μ_min",
    "u_LR": "wheel speeds (u_L, u_R) (rad/s)", "qdot": "q̇ = (φ̇, ẋ, ẏ)", "v_enc": "v from the encoders (m/s)",
    "omega_enc": "ω from the encoders (rad/s)", "H0": "H(0)", "u": "wheel speeds u (rad/s)",
    "Vb_phi": "𝒱_b = (ω_bz, v_bx, v_by)", "u_phi": "wheel speeds for q̇ (rad/s)", "F_odo": "F = H†(0)",
    "Vb_odo": "odometry twist 𝒱_b",
}


#: answer quantities that only some seeds have (the control problems give ω_d for an underdamped and the two
#: real roots for an overdamped error dynamics); the panel always asks for what the student's own problem has
SEED_DEPENDENT = {"control": ("wd", "roots")}
LABELS.setdefault("roots", "real roots s₁, s₂ (1/s, any order)")
LABELS["Lam_eig"] = "eigenvalues of Λ (any order)"
LABELS["ell_axes"] += " (any order)"
LABELS["force_axes"] += " (any order)"
#: quantities whose entries may be typed in any order (roots, eigenvalues)
UNORDERED = {"roots", "Lam_eig", "ell_axes", "force_axes"}


class AssignmentError(ValueError):
    """Bad assignment code, student ID or submission file (message is shown to the user)."""


# ------------------------------------------------------------------------------------------------
# assignment codes
# ------------------------------------------------------------------------------------------------


@dataclass
class Item:
    topic: str
    difficulty: int = 2
    quantities: list[str] | None = None      # None = every numeric answer quantity
    rtol: float = DEFAULT_RTOL
    points: float = 1.0
    atol: float = DEFAULT_ATOL

    def to_json(self) -> dict:
        d = {"topic": self.topic, "difficulty": self.difficulty}
        if self.quantities:
            d["quantities"] = list(self.quantities)
        if self.rtol != DEFAULT_RTOL:
            d["rtol"] = self.rtol
        if self.points != 1.0:
            d["points"] = self.points
        if self.atol != DEFAULT_ATOL:
            d["atol"] = self.atol
        return d


@dataclass
class Assignment:
    id: str
    title: str
    items: list[Item]
    salt: str
    due: str = ""                             # ISO date "YYYY-MM-DD" (or full ISO datetime); "" = no due date
    code: str = field(default="", repr=False)

    def to_json(self) -> dict:
        return {"v": CODE_VERSION, "id": self.id, "title": self.title, "due": self.due,
                "items": [it.to_json() for it in self.items], "salt": self.salt}


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    text = re.sub(r"\s+", "", text)
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _check_item(it: Item, n: int) -> None:
    if it.topic not in P.TOPICS:
        raise AssignmentError(f"item {n}: unknown topic {it.topic!r} (choose from {', '.join(P.TOPIC_KEYS)}; "
                              "'mixed' is not allowed because every student must get the same kind of problem)")
    if it.difficulty not in P.DIFFICULTIES:
        raise AssignmentError(f"item {n}: difficulty must be 1, 2 or 3, not {it.difficulty!r}")
    if not (0 < it.rtol < 1):
        raise AssignmentError(f"item {n}: rtol must be between 0 and 1")
    if not (0 <= it.atol < 1):
        raise AssignmentError(f"item {n}: atol must be between 0 and 1")
    if not (it.points > 0 and math.isfinite(it.points)):      # json.loads accepts NaN / Infinity
        raise AssignmentError(f"item {n}: points must be a positive number")
    if it.quantities is not None:
        known = set(gradable(P.generate(it.topic, 0, it.difficulty))) | set(SEED_DEPENDENT.get(it.topic, ()))
        bad = [q for q in it.quantities if q not in known]
        if bad:
            raise AssignmentError(f"item {n}: {it.topic} (difficulty {it.difficulty}) has no gradable "
                                  f"quantity {', '.join(bad)}; choose from {', '.join(sorted(known))}")


def make_code(assignment_id: str, title: str, items: list[Item | dict | tuple], due: str = "",
              salt: str | None = None) -> str:
    """Assignment code for the given items (``Item``, ``{"topic":…, "difficulty":…}`` or ``(topic, difficulty)``)."""
    if not ASSIGNMENT_ID_RE.match(assignment_id or ""):
        raise AssignmentError("assignment id: 1-40 letters, digits, '.', '_' or '-'")
    its = [_item(x) for x in items]
    if not its:
        raise AssignmentError("an assignment needs at least one item")
    for n, it in enumerate(its, 1):
        _check_item(it, n)
    if due:
        _due_datetime(due)                    # raises on a bad date
    a = Assignment(assignment_id, title or assignment_id, its, salt or secrets.token_urlsafe(12), due)
    return _b64e(json.dumps(a.to_json(), separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _item(x) -> Item:
    if isinstance(x, Item):
        return x
    if isinstance(x, (tuple, list)):
        return Item(str(x[0]), int(x[1]) if len(x) > 1 else 2)
    if isinstance(x, dict):
        q = x.get("quantities")
        return Item(str(x.get("topic", "")), int(x.get("difficulty", 2)), list(q) if q else None,
                    float(x.get("rtol", DEFAULT_RTOL)), float(x.get("points", 1.0)),
                    float(x.get("atol", DEFAULT_ATOL)))
    raise AssignmentError(f"bad item {x!r}")


def parse_code(code: str) -> Assignment:
    """Decode and check an assignment code (raises :class:`AssignmentError` with a readable message)."""
    code = (code or "").strip()
    if not code:
        raise AssignmentError("paste the assignment code from your instructor")
    try:
        d = json.loads(_b64d(code).decode("utf-8"))
    except Exception:  # noqa: BLE001 -- any decoding failure means the same thing to the user
        raise AssignmentError("this is not a valid assignment code (copy the whole code, without spaces)") from None
    if not isinstance(d, dict) or d.get("v") != CODE_VERSION:
        raise AssignmentError("this assignment code was made by a different MR Studio version")
    try:
        items = [_item(x) for x in d["items"]]
        a = Assignment(str(d["id"]), str(d.get("title") or d["id"]), items, str(d["salt"]), str(d.get("due") or ""),
                       code)
    except (KeyError, TypeError, ValueError):
        raise AssignmentError("the assignment code is incomplete") from None
    if not ASSIGNMENT_ID_RE.match(a.id) or not a.items or not a.salt:
        raise AssignmentError("the assignment code is incomplete")
    for n, it in enumerate(a.items, 1):
        _check_item(it, n)
    return a


def check_student_id(student_id: str) -> str:
    sid = (student_id or "").strip()
    if not STUDENT_ID_RE.match(sid):
        raise AssignmentError("student ID: 1-40 letters, digits, '.', '_' or '-' (no spaces)")
    return sid


def student_seed(salt: str, assignment_id: str, student_id: str, k: int) -> int:
    """Per-student seed of problem ``k`` (0-based)."""
    h = hashlib.sha256(f"{salt}|{assignment_id}|{student_id}|{k}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10**6


def _as_assignment(code) -> Assignment:
    return code if isinstance(code, Assignment) else parse_code(code)


def build(code, student_id: str) -> list[P.Problem]:
    """The student's problems, in item order."""
    a = _as_assignment(code)
    sid = check_student_id(student_id)
    return [P.generate(it.topic, student_seed(a.salt, a.id, sid, k), it.difficulty) for k, it in enumerate(a.items)]


# ------------------------------------------------------------------------------------------------
# gradable quantities
# ------------------------------------------------------------------------------------------------


def _numeric(v):
    """The value as a float array, or None if it is not a gradable number/array."""
    if isinstance(v, (str, dict)) or v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return np.array(float(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.array(float(v))
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return v.astype(float)
    return None


def gradable(p: P.Problem) -> dict[str, np.ndarray]:
    """``{name: reference value}`` of every numeric answer quantity of ``p`` (answer order)."""
    out = {}
    for k, v in p.answer.items():
        a = _numeric(v)
        if a is not None:
            out[k] = a
    return out


def quantities(p: P.Problem, item: Item | None = None) -> dict[str, np.ndarray]:
    """The quantities the student answers for ``p`` (all gradable ones, or the item's selection).  When a
    selection names only seed-dependent quantities that this student's problem does not have, every gradable
    quantity is asked instead, so no problem is left without answers."""
    g = gradable(p)
    if item is not None and item.quantities:
        sel = {q: g[q] for q in item.quantities if q in g}
        return sel or g
    return g


def label(name: str, shape: tuple) -> str:
    """Field label: description plus the expected shape ("3×3, row by row")."""
    base = LABELS.get(name, name)
    if len(shape) == 0:
        return base
    if len(shape) == 1:
        return f"{base} — {shape[0]} numbers"
    return f"{base} — {shape[0]}×{shape[1]}, row by row"


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_numbers(text) -> list[float]:
    """Numbers from what a student typed: '0.5, -1  2e-3; 4' or '[[1, 0], [0, 1]]' -> flat list.

    Raises ValueError for anything that is not a list of numbers (letters, 'pi', fractions…)."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return [float(text)]
    if isinstance(text, (list, tuple)):
        return [float(x) for x in np.ravel(np.asarray(text, dtype=float))]
    s = str(text).strip().replace("−", "-").replace("–", "-")    # unicode minus / en dash typed or pasted
    if not s:
        raise ValueError("empty")
    junk = re.sub(r"[\s,;\[\]()]", "", _NUM.sub(" ", s))
    if junk:
        raise ValueError(f"only numbers please (found {junk[:12]!r})")
    return [float(x) for x in _NUM.findall(s)]


def tolerance(ref: np.ndarray, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> float:
    return rtol * float(np.max(np.abs(np.asarray(ref, dtype=float)))) + atol


def compare(ref: np.ndarray, given, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL,
            unordered: bool = False) -> tuple[bool, float, str]:
    """(correct, max absolute error, note); correct means ``max|a - r| <= tolerance(ref, rtol, atol)``.
    ``unordered`` compares sorted values (roots, eigenvalues)."""
    try:
        vals = np.asarray(parse_numbers(given), dtype=float)
    except (TypeError, ValueError) as e:
        return False, float("inf"), f"not a number list ({e})" if str(e) != "empty" else "no answer"
    ref = np.asarray(ref, dtype=float)
    if vals.size != ref.size:
        return False, float("inf"), f"expected {ref.size} number(s), got {vals.size}"
    if not np.all(np.isfinite(vals)):
        return False, float("inf"), "not finite"
    r = np.sort(ref.ravel()) if unordered else ref.ravel()
    err = float(np.max(np.abs((np.sort(vals) if unordered else vals) - r)))
    return err <= tolerance(ref, rtol, atol), err, ""


# ------------------------------------------------------------------------------------------------
# submissions
# ------------------------------------------------------------------------------------------------


def app_version() -> str:
    try:
        from importlib.metadata import version
        return version("mrstudio")
    except Exception:  # noqa: BLE001
        pass
    try:
        import tomllib
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
        with open(here, "rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    except Exception:  # noqa: BLE001
        return "unknown"


def _canonical_answers(answers: dict) -> str:
    return json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(code: str, student_id: str, answers: dict) -> str:
    """sha256 of ``code|student_id|answers`` (canonical JSON).  Integrity check, not a signature."""
    return hashlib.sha256(f"{code.strip()}|{student_id}|{_canonical_answers(answers)}".encode("utf-8")).hexdigest()


def _clean_answer(v):
    """Answers are stored as numbers / flat number lists when they parse, else as the typed text."""
    try:
        vals = parse_numbers(v)
    except (TypeError, ValueError):
        return str(v)
    return vals[0] if len(vals) == 1 else vals


@dataclass
class Submission:
    assignment_id: str
    student_id: str
    answers: dict                         # {"0": {"R": [..9 numbers..], "theta2": 1.047}, ...}
    code: str
    app_version: str = ""
    created: str = ""
    fingerprint: str = ""

    def to_json(self) -> dict:
        return {"assignment_id": self.assignment_id, "student_id": self.student_id, "app_version": self.app_version,
                "created": self.created, "answers": self.answers, "fingerprint": self.fingerprint,
                "code": self.code}

    @classmethod
    def from_json(cls, d: dict) -> "Submission":
        try:
            return cls(str(d["assignment_id"]), str(d["student_id"]), dict(d.get("answers") or {}),
                       str(d.get("code", "")), str(d.get("app_version", "")), str(d.get("created", "")),
                       str(d.get("fingerprint", "")))
        except (KeyError, TypeError, ValueError):
            raise AssignmentError("not an MR Studio submission file") from None


def submit(code, student_id: str, answers: dict, created: str | None = None) -> Submission:
    """A :class:`Submission` for ``answers`` = ``{k: {quantity: value}}`` (k int or str; value number, list or
    text).  The answers are normalised (numbers parsed) before the fingerprint is taken."""
    a = _as_assignment(code)
    sid = check_student_id(student_id)
    clean = {str(k): {str(q): _clean_answer(v) for q, v in (qs or {}).items()} for k, qs in (answers or {}).items()}
    when = created or _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return Submission(a.id, sid, clean, a.code, app_version(), when, fingerprint(a.code, sid, clean))


def submissions_dir() -> str:
    """``~/Documents/MR Studio/submissions`` (``MRSTUDIO_SUBMISSIONS_DIR`` overrides, e.g. for tests)."""
    return os.environ.get("MRSTUDIO_SUBMISSIONS_DIR") or os.path.join(
        os.path.expanduser("~"), "Documents", "MR Studio", "submissions")


def submission_filename(sub: Submission) -> str:
    return f"{sub.student_id}_{sub.assignment_id}.json"


def save_submission(sub: Submission, folder: str | None = None) -> str:
    folder = folder or submissions_dir()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, submission_filename(sub))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sub.to_json(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def load_submission(path: str) -> Submission:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise AssignmentError(f"cannot read {os.path.basename(path)}: {e}") from None
    if not isinstance(d, dict):
        raise AssignmentError("not an MR Studio submission file")
    return Submission.from_json(d)


# ------------------------------------------------------------------------------------------------
# grading (instructor side)
# ------------------------------------------------------------------------------------------------


def _due_datetime(due: str) -> _dt.datetime:
    try:
        if len(due) == 10:                     # a date: due at the end of that day, local time
            d = _dt.date.fromisoformat(due)
            return _dt.datetime(d.year, d.month, d.day, 23, 59, 59).astimezone()
        t = _dt.datetime.fromisoformat(due)
        return t if t.tzinfo else t.astimezone()
    except ValueError:
        raise AssignmentError(f"due date {due!r} is not an ISO date like 2026-11-20") from None


def is_late(due: str, created: str) -> bool | None:
    """True/False, or None when either time is missing or unreadable.  ``created`` is the student's clock: use
    the LMS upload time for anything that matters."""
    if not due or not created:
        return None
    try:
        c = _dt.datetime.fromisoformat(created)
        c = c if c.tzinfo else c.astimezone()
        return c > _due_datetime(due)
    except (ValueError, AssignmentError):
        return None


def grade(code, submission: Submission | dict, student_id: str | None = None) -> dict:
    """Regrade a submission: regenerate the student's problems and compare every quantity.

    Returns ``{assignment_id, student_id, score, max_score, percent, fingerprint_ok, code_ok, late, notes,
    items:[{k, topic, difficulty, seed, score, points, quantities:{name:{ok, error, tol, note}}}]}``.  Each item is
    worth its ``points`` (default 1), split evenly over its quantities."""
    a = _as_assignment(code)
    sub = submission if isinstance(submission, Submission) else Submission.from_json(submission)
    notes = []
    sid = check_student_id(student_id or sub.student_id)
    if student_id and student_id.strip() != sub.student_id:
        notes.append(f"file says student {sub.student_id!r}")
    code_ok = sub.assignment_id == a.id and (not sub.code or sub.code.strip() == a.code.strip())
    if sub.assignment_id != a.id:
        notes.append(f"submission is for assignment {sub.assignment_id!r}, not {a.id!r}")
    elif sub.code and sub.code.strip() != a.code.strip():
        notes.append("submission was made with a different code for this assignment id")
    fp_ok = sub.fingerprint == fingerprint(a.code, sub.student_id, sub.answers)
    if not fp_ok:
        notes.append("fingerprint mismatch (file edited after saving, or made for another code/student)")
    items, score, total = [], 0.0, 0.0
    for k, (it, p) in enumerate(zip(a.items, build(a, sid))):
        qs = quantities(p, it)
        given = sub.answers.get(str(k)) or {}
        if not isinstance(given, dict):          # edited file: not {quantity: value} -> graded as unanswered
            notes.append(f"problem {k + 1}: answers are not a list of quantities (ignored)")
            given = {}
        per = it.points / max(1, len(qs))
        res, s = {}, 0.0
        for name, ref in qs.items():
            if name in given:
                ok, err, note = compare(ref, given[name], it.rtol, it.atol, name in UNORDERED)
            else:
                ok, err, note = False, float("inf"), "no answer"
            res[name] = {"ok": bool(ok), "error": err, "tol": tolerance(ref, it.rtol, it.atol), "note": note}
            s += per if ok else 0.0
        items.append({"k": k, "topic": it.topic, "difficulty": it.difficulty, "seed": p.seed, "score": s,
                      "points": it.points, "quantities": res})
        score += s
        total += it.points
    return {"assignment_id": a.id, "student_id": sid, "score": round(score, 6), "max_score": total,
            "percent": round(100.0 * score / total, 2) if total else 0.0, "fingerprint_ok": fp_ok,
            "code_ok": code_ok, "late": is_late(a.due, sub.created), "created": sub.created, "notes": notes,
            "items": items}


def answer_key(code, student_id: str) -> dict:
    """``{k: {quantity: reference value (list/float)}}`` -- instructor tool only (``tools/assignment.py key``)."""
    a = _as_assignment(code)
    out = {}
    for k, (it, p) in enumerate(zip(a.items, build(a, student_id))):
        out[str(k)] = {q: (v.tolist() if v.ndim else float(v)) for q, v in quantities(p, it).items()}
    return out
