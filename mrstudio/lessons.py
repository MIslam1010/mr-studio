"""Guided lessons for students (SBEG 361 / SBEG 461): short step-by-step scripts with typed-answer checks.

A :class:`Lesson` is a list of :class:`Step` objects.  Each step tells the student, in plain language, what to
compute (by hand or with the app), which tab and robot to open to see it (:class:`Target`), and asks for one
number.  The :class:`Check` compares the typed number with a reference value that is **computed** here with
Lynch & Park's ``modern_robotics`` package and MR Studio's own models (``robots.get_robot``,
``kinematics``, ``dynamics.DynamicsEngine``, ``trajectory.time_scaling``) -- nothing is typed in by hand.

The numbers of each lesson (joint angles, gains, ...) are drawn deterministically from a seed made from the
lesson id and the scenario/step tag (:func:`seed_for`, SHA-256, so they do not change between runs or
machines; Python's own ``hash`` is salted per process and is not used).

Lessons (course, week as in the revised SBEG 361 / 461 schedules):

* SBEG 361 W06  product-of-exponentials forward kinematics (planar 3R arm)
* SBEG 361 W09  space and body Jacobians (planar 2R arm)
* SBEG 361 W10  singularities and statics (planar 2R arm)
* SBEG 361 W11  Newton-Raphson inverse kinematics (planar 2R arm)
* SBEG 461 W01  rigid-body inertia and the mass matrix (planar 2R arm)
* SBEG 461 W02  Newton-Euler (RNEA) joint torques (planar 2R arm)
* SBEG 461 W05  PD control of one joint: error dynamics
* SBEG 461 W06  computed-torque control (planar 2R arm)
* SBEG 461 W10  cubic and quintic time scaling (planar 2R arm)

Section references are to Modern Robotics (Lynch & Park); no equation numbers are cited here.
"""
from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import modern_robotics as mr
import numpy as np

DEG = math.pi / 180.0

R2_NAME = "Planar 2R arm"
R3_NAME = "Planar 3R arm"

# ------------------------------------------------------------------------------------------------
# data model
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """Where to look in the app: tab class name (as in ``gui.app.TABS``), robot name, optional preset key."""
    tab: str
    robot: str | None = None
    preset: str | None = None


@dataclass
class CheckResult:
    ok: bool
    value: float | None
    message: str


@dataclass
class Check:
    """Typed-answer check: ``|x - ref| <= max(abs_tol, rel_tol * |ref|)``; ``ref`` is computed lazily."""
    label: str                     # what to type, e.g. "x-coordinate of the tip"
    unit: str                      # unit the student types in ("m", "°", "N·m", "" ...)
    compute: Callable[[], float]
    rel_tol: float = 0.01
    abs_tol: float = 1e-4
    digits: int = 4                # significant digits when the reference is shown

    @cached_property
    def reference(self) -> float:
        v = float(self.compute())
        if not math.isfinite(v):
            raise ValueError(f"reference for {self.label!r} is not finite")
        return v

    @property
    def tolerance(self) -> float:
        return max(self.abs_tol, self.rel_tol * abs(self.reference))

    def accepts(self, value: float) -> bool:
        return math.isfinite(value) and abs(float(value) - self.reference) <= self.tolerance

    def reference_text(self) -> str:
        return f"{self.reference:.{self.digits}g}" + (f" {self.unit}" if self.unit else "")

    def grade(self, text: str) -> CheckResult:
        try:
            v = parse_answer(text)
        except ValueError as e:
            return CheckResult(False, None, str(e))
        if self.accepts(v):
            return CheckResult(True, v, "Correct.")
        hi = v > self.reference
        rel = abs(v - self.reference) / max(abs(self.reference), 1e-12)
        how = "a little" if rel < 0.1 else ("far" if rel > 1 else "")
        side = "too high" if hi else "too low"
        return CheckResult(False, v, f"Not yet: your value is {how + ' ' if how else ''}{side}.")


@dataclass
class Step:
    instruction_html: str
    target: Target
    check: Check
    hint_html: str = ""


@dataclass
class Lesson:
    id: str
    course: str          # "SBEG 361" / "SBEG 461"
    week: int
    title: str
    minutes: int
    steps: list[Step]
    intro_html: str = ""
    section: str = ""    # Modern Robotics section(s)
    #: the scenario numbers (SI units, radians) -- for tests and instructors; never shown as answers
    data: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.course} · W{self.week:02d} · {self.title}"

    def step_html(self, k: int) -> str:
        """Full HTML for step ``k`` (0-based): lesson heading, the step, then the set-up (intro) for reference."""
        s = self.steps[k]
        head = (f"<h2>{_esc(self.title)}</h2><span class='ref'>{_esc(self.course)}, week {self.week} · "
                f"{_esc(self.section)} · about {self.minutes} min</span>")
        unit = f" [{_esc(s.check.unit)}]" if s.check.unit else ""
        ask = f"<p><b>Type:</b> {_esc(s.check.label)}{unit}</p>"
        step = f"<h3>Step {k + 1} of {len(self.steps)}</h3>{s.instruction_html}{ask}"
        if k == 0:
            return f"{head}{self.intro_html}{step}"
        return f"{head}{step}<h3>Set-up</h3>{self.intro_html}"


def _esc(s: str) -> str:
    import html
    return html.escape(s, quote=False)


# ------------------------------------------------------------------------------------------------
# answers: numbers and simple arithmetic ("0.6*cos(30)" is not degrees -- trig works in radians)
# ------------------------------------------------------------------------------------------------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUN = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "atan2": math.atan2,
        "asin": math.asin, "acos": math.acos, "atan": math.atan, "exp": math.exp, "deg": math.degrees,
        "rad": math.radians}
_CONST = {"pi": math.pi, "e": math.e}


def parse_answer(text: str) -> float:
    """Parse a typed answer: a number (``-0.25``, ``1,5``, ``2.5e-3``, ``30°``) or a short arithmetic expression
    (``0.6*cos(pi/6)``, ``sqrt(2)/2``).  Raises ``ValueError`` with a student-readable message."""
    s = (text or "").strip()
    if not s:
        raise ValueError("Type a number first.")
    s = s.replace("−", "-").replace("×", "*").replace("·", "*").replace("^", "**")
    s = re.sub(r"\s*(°|deg|degrees|rad|m/s²|m/s|rad/s²|rad/s|N·m|Nm|N\*m|kg\*m\*\*2|kg·m²|kg|m|s|%|N)\s*$", "", s)
    if re.fullmatch(r"[+-]?\d+,\d+", s):          # decimal comma
        s = s.replace(",", ".")
    try:
        tree = ast.parse(s, mode="eval")
        v = float(_eval(tree.body))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError):
        raise ValueError(f"Could not read {text.strip()!r} as a number.") from None
    if not math.isfinite(v):
        raise ValueError("That value is not a finite number.")
    return v


def _eval(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100:
            raise ValueError
        return _BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
        return _UN[type(n.op)](_eval(n.operand))
    if isinstance(n, ast.Name) and n.id in _CONST:
        return _CONST[n.id]
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords
            and len(n.args) <= 2):
        return _FUN[n.func.id](*(_eval(a) for a in n.args))
    raise ValueError


# ------------------------------------------------------------------------------------------------
# deterministic numbers
# ------------------------------------------------------------------------------------------------


def seed_for(lesson_id: str, tag: str | int = "scenario") -> int:
    """Stable seed from (lesson id, step/scenario tag): first 8 hex digits of SHA-256."""
    return int(hashlib.sha256(f"{lesson_id}|{tag}".encode()).hexdigest()[:8], 16)


def rng_for(lesson_id: str, tag: str | int = "scenario") -> np.random.Generator:
    return np.random.default_rng(seed_for(lesson_id, tag))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _angles(rng, n: int, choices=(-75, -60, -45, -30, 30, 45, 60, 75, 90, 120)) -> np.ndarray:
    return np.array([_pick(rng, list(choices)) for _ in range(n)], float) * DEG


def _d(x) -> str:
    """Angle (rad) as a plain degree string."""
    return f"{np.degrees(x):g}°"


def _n(x, d: int = 3) -> str:
    r = round(float(x), d)
    return "0" if r == 0 else f"{r:.{d}f}".rstrip("0").rstrip(".")


def _robot(name: str):
    from .robots import get_robot
    return get_robot(name)


def _engine(name: str):
    from .dynamics import DynamicsEngine
    return DynamicsEngine(_robot(name), backend="mr")


# ------------------------------------------------------------------------------------------------
# SBEG 361
# ------------------------------------------------------------------------------------------------


def _lesson_poe() -> Lesson:
    lid = "361-w06-poe"
    rng = rng_for(lid)
    rob = _robot(R3_NAME)
    q2 = -float(rob.Slist[4, 1])              # S_i = (z, -z x q_i) -> v_y = -q_i,x
    q3 = -float(rob.Slist[4, 2])
    L3 = float(rob.M[0, 3] - q3)
    while True:
        th = _angles(rng, 3, (-60, -45, -30, 30, 45, 60, 90))
        T = mr.FKinSpace(rob.M, rob.Slist, th)
        phi = math.atan2(T[1, 0], T[0, 0])
        if min(abs(T[0, 3]), abs(T[1, 3])) > 0.08 and abs(phi) > 10 * DEG:
            break
    tgt = Target("FKTab", R3_NAME)
    th_txt = ", ".join(f"θ{i + 1} = {_d(t)}" for i, t in enumerate(th))
    intro = (f"<p>The <b>planar 3R arm</b> has three joints that all turn about the $\\hat{{z}}$ axis. Its links are "
             f"{_n(q2)} m, {_n(q3 - q2)} m and {_n(L3)} m long.  At the home position (all joints at 0) the arm lies "
             f"along the $x$ axis, so the tip frame is a pure shift: "
             f"$M$ has position $({_n(rob.M[0, 3])}, 0, 0)$ m.</p>"
             "<p>The product of exponentials (MR Sec. 4.1) builds the pose as one screw motion per joint:</p>"
             "$$T(\\theta) = e^{[\\mathcal{S}_1]\\theta_1}e^{[\\mathcal{S}_2]\\theta_2}"
             "e^{[\\mathcal{S}_3]\\theta_3}M$$")
    steps = [
        Step("<p>Each joint axis is a screw $\\mathcal{S}_i = (\\hat{\\omega}_i, v_i)$ with "
             "$v_i = -\\hat{\\omega}_i \\times q_i$, where $q_i$ is any point on the axis.  Joint 3 sits at "
             f"$q_3 = ({_n(q3)}, 0, 0)$ m and $\\hat{{\\omega}}_3 = (0, 0, 1)$.</p>"
             "<p>Work out the $y$-component of $v_3$.</p>",
             tgt, Check("y-component of v₃", "m", lambda: rob.Slist[4, 2], abs_tol=1e-3),
             "<p>$\\hat{z} \\times (a, 0, 0) = (0, a, 0)$, and $v_3$ is minus that.</p>"),
        Step(f"<p>Now turn the joints to <b>{th_txt}</b>.  Multiply the three screw exponentials and $M$, then read "
             "the tip position from the last column of $T$.</p><p>What is the $x$-coordinate of the tip?</p>"
             "<p>Check yourself: press <b>Go to tab</b>, set the same angles in <i>Joint coordinates θ</i> and look "
             "at <i>End-effector pose T_sb(θ)</i>.</p>",
             tgt, Check("x-coordinate of the tip", "m", lambda: mr.FKinSpace(rob.M, rob.Slist, th)[0, 3],
                        abs_tol=1e-3),
             "<p>For a planar chain the tip is "
             "$x = L_1\\cos\\theta_1 + L_2\\cos(\\theta_1+\\theta_2) + L_3\\cos(\\theta_1+\\theta_2+\\theta_3)$ — "
             "the PoE product gives the same number.</p>"),
        Step(f"<p>Same angles ({th_txt}).  What is the $y$-coordinate of the tip?</p>",
             tgt, Check("y-coordinate of the tip", "m", lambda: mr.FKinSpace(rob.M, rob.Slist, th)[1, 3],
                        abs_tol=1e-3),
             "<p>Replace every $\\cos$ by $\\sin$ in the $x$ formula.</p>"),
        Step("<p>The rotation part of $T$ turns the tip frame about $\\hat{z}$ by an angle $\\phi$: "
             "$R = \\mathrm{Rot}(\\hat{z}, \\phi)$.  Give $\\phi$ in degrees, between −180° and 180°.</p>",
             tgt, Check("tip orientation φ", "°", lambda: math.degrees(math.atan2(
                 mr.FKinSpace(rob.M, rob.Slist, th)[1, 0], mr.FKinSpace(rob.M, rob.Slist, th)[0, 0])),
                        abs_tol=0.2, digits=4),
             "<p>Rotations about the same axis add up.</p>"),
        Step("<p>The body form $T = Me^{[\\mathcal{B}_1]\\theta_1}e^{[\\mathcal{B}_2]\\theta_2}e^{[\\mathcal{B}_3]"
             "\\theta_3}$ must give the same pose.  The body screw axes are the joint axes seen from the tip frame "
             "at home.  Give the $y$-component of $v$ in $\\mathcal{B}_1$.</p>",
             tgt, Check("y-component of v in B₁", "m", lambda: rob.Blist[4, 0], abs_tol=1e-3),
             "<p>From the tip at home, joint 1 lies a distance $L_1 + L_2 + L_3$ behind you along $-x$: "
             "$v = -\\hat{z} \\times (-(L_1+L_2+L_3), 0, 0)$.</p>"),
    ]
    intro += f"<p><b>Numbers for this lesson:</b> {th_txt}.</p>"
    return Lesson(lid, "SBEG 361", 6, "Forward kinematics with the product of exponentials", 25, steps, intro,
                  "MR Sec. 4.1", dict(theta=th))


def _lesson_jacobian() -> Lesson:
    lid = "361-w09-jacobian"
    rng = rng_for(lid)
    rob = _robot(R2_NAME)
    L1 = float(rob.Slist[4, 1] * -1)
    while True:
        th = _angles(rng, 2, (-60, -45, -30, 30, 45, 60, 90, 120))
        dth = np.array([_pick(rng, [-1.0, -0.5, 0.5, 1.0, 1.5]), _pick(rng, [-1.0, -0.5, 0.5, 1.0, 2.0])])
        Js, Jb = mr.JacobianSpace(rob.Slist, th), mr.JacobianBody(rob.Blist, th)
        T = mr.FKinSpace(rob.M, rob.Slist, th)
        Vs = Js @ dth
        pdot = Vs[3:] + np.cross(Vs[:3], T[:3, 3])
        Vb = Jb @ dth
        if (abs(Js[3, 1]) > 0.1 and abs(Jb[4, 0]) > 0.1 and abs(pdot[0]) > 0.05 and abs(Vb[4]) > 0.05):
            break
    tgt = Target("VelocityTab", R2_NAME)
    th_txt = f"θ1 = {_d(th[0])}, θ2 = {_d(th[1])}"
    intro = (f"<p>The <b>planar 2R arm</b> has links $L_1 = {_n(L1)}$ m and $L_2 = {_n(rob.M[0, 3] - L1)}$ m. "
             "A Jacobian turns joint rates $\\dot{\\theta}$ into the twist of the tip.  The <b>space</b> Jacobian "
             "$J_s$ gives the twist in the fixed frame {s}; the <b>body</b> Jacobian $J_b$ gives it in the tip "
             "frame {b} (MR Sec. 5.1):</p>"
             "$$\\mathcal{V}_s = J_s(\\theta)\\dot{\\theta},\\quad \\mathcal{V}_b = J_b(\\theta)\\dot{\\theta},"
             "\\quad J_b = [\\mathrm{Ad}_{T_{bs}}]J_s$$"
             "<p>Rows are ordered $(\\omega_x, \\omega_y, \\omega_z, v_x, v_y, v_z)$.</p>")
    steps = [
        Step(f"<p>Put the arm at <b>{th_txt}</b>.  Column 2 of $J_s$ is joint 2's screw axis after joint 1 has "
             "moved it: the axis is still $\\hat{z}$ but now passes through "
             "$q = (L_1\\cos\\theta_1, L_1\\sin\\theta_1, 0)$.</p><p>What is the $v_x$ entry of column 2 of $J_s$?</p>",
             tgt, Check("v_x of column 2 of J_s", "m", lambda: mr.JacobianSpace(rob.Slist, th)[3, 1], abs_tol=1e-3),
             "<p>$v = -\\hat{z} \\times q = (q_y, -q_x, 0)$.</p>"),
        Step(f"<p>Same angles.  Column 1 of $J_b$ is joint 1's axis seen from the tip frame.  What is its "
             "$v_y$ entry?</p>",
             tgt, Check("v_y of column 1 of J_b", "m", lambda: mr.JacobianBody(rob.Blist, th)[4, 0], abs_tol=1e-3),
             "<p>Use $J_b = [\\mathrm{Ad}_{T_{bs}}]J_s$.  For this arm it works out to "
             "$L_1\\cos\\theta_2 + L_2$: turning joint 1 swings the tip sideways (along $\\hat{y}_b$) with "
             "that lever arm.</p>"),
        Step(f"<p>Now the joints move at $\\dot{{\\theta}}_1 = {_n(dth[0])}$ rad/s and "
             f"$\\dot{{\\theta}}_2 = {_n(dth[1])}$ rad/s (same angles).  The tip point moves with "
             "$\\dot{p} = v_s + \\omega_s \\times p$, where $(\\omega_s, v_s) = J_s\\dot{\\theta}$ and $p$ is the "
             "tip position.</p><p>What is the $x$-component of the tip velocity in {s}?</p>",
             tgt, Check("ẋ of the tip in {s}", "m/s",
                        lambda: (lambda Vs, p: Vs[3] + np.cross(Vs[:3], p)[0])(
                            mr.JacobianSpace(rob.Slist, th) @ dth, mr.FKinSpace(rob.M, rob.Slist, th)[:3, 3]),
                        abs_tol=1e-3),
             "<p>Enter the joint rates in <i>Joint rates θ̇ → twists</i> in the Velocity tab and compare.</p>"),
        Step("<p>Same motion.  In the tip frame the twist is $\\mathcal{V}_b = J_b\\dot{\\theta}$.  What is its "
             "$v_y$ component (the sideways speed of the tip seen from the tip)?</p>",
             tgt, Check("v_y of V_b", "m/s", lambda: (mr.JacobianBody(rob.Blist, th) @ dth)[4], abs_tol=1e-3),
             "<p>Multiply row 5 of $J_b$ by $\\dot{\\theta}$.</p>"),
    ]
    intro += (f"<p><b>Numbers for this lesson:</b> {th_txt}; joint rates "
              f"$\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s.</p>")
    return Lesson(lid, "SBEG 361", 9, "Space and body Jacobians", 25, steps, intro, "MR Sec. 5.1",
                  dict(theta=th, dtheta=dth))


def _lesson_singularities() -> Lesson:
    lid = "361-w10-singularities"
    rng = rng_for(lid)
    rob = _robot(R2_NAME)
    L1 = -float(rob.Slist[4, 1])
    L2 = float(rob.M[0, 3]) - L1
    from .kinematics import point_jacobian

    def J2(t):
        return point_jacobian(rob, t)[:2]
    th = np.array([_pick(rng, [-30, 30, 45, 60]), _pick(rng, [30, 45, 60, 120, 135])], float) * DEG
    F = np.array([0.0, -float(_pick(rng, [5, 10, 20]))])

    def second_singular_angle() -> float:
        from scipy.optimize import brentq
        f = lambda a: float(np.linalg.det(J2(np.array([th[0], a]))))
        return math.degrees(brentq(f, 90 * DEG, 270 * DEG, xtol=1e-12))

    tgt = Target("VelocityTab", R2_NAME)
    th_txt = f"θ1 = {_d(th[0])}, θ2 = {_d(th[1])}"
    intro = (f"<p>For the planar 2R arm ($L_1 = {_n(L1)}$ m, $L_2 = {_n(L2)}$ m) only the tip position "
             "$(x, y)$ matters, so we use the $2\\times 2$ Jacobian $J$ that maps $\\dot{\\theta}$ to "
             "$(\\dot{x}, \\dot{y})$.  The arm is <b>singular</b> when $J$ loses rank: the tip cannot move in some "
             "direction, however fast the joints turn (MR Sec. 5.3).  The same $J$ also gives the joint torques "
             "that hold a force at the tip: $\\tau = J^{T}F$ (MR Sec. 5.2).</p>")
    steps = [
        Step(f"<p>At <b>{th_txt}</b>, compute $\\det J$.  For this arm it simplifies to a product of the two link "
             "lengths and one sine.</p>",
             tgt, Check("det J", "m²", lambda: float(np.linalg.det(J2(th))), abs_tol=1e-4),
             "<p>$\\det J = L_1L_2\\sin\\theta_2$.</p>"),
        Step("<p>$\\det J = 0$ at $\\theta_2 = 0$ (arm stretched out).  Which other elbow angle between 90° and "
             "270° makes the arm singular?  Try it with the singularity button in the Velocity tab.</p>",
             tgt, Check("other singular elbow angle θ₂", "°", second_singular_angle, abs_tol=0.5),
             "<p>The arm is folded back on itself.</p>"),
        Step(f"<p>Back at {th_txt}.  The singular values $\\sigma_1 \\geq \\sigma_2$ of $J$ say how easily the tip "
             "moves in its best and worst directions.  What is the smallest one, $\\sigma_2$?</p>",
             tgt, Check("smallest singular value σ₂", "m",
                        lambda: float(np.linalg.svd(J2(th), compute_uv=False)[-1]),
                        abs_tol=1e-4),
             "<p>$\\sigma_1^{2}$ and $\\sigma_2^{2}$ are the eigenvalues of $JJ^{T}$; also "
             "$\\sigma_1\\sigma_2 = |\\det J|$.</p>"),
        Step("<p>The condition number $\\sigma_1/\\sigma_2$ grows without bound near a singularity.  "
             "What is it at the same angles?</p>",
             tgt, Check("condition number σ₁/σ₂", "", lambda: float(np.linalg.cond(J2(th))), abs_tol=1e-3),
             "<p>Divide the largest singular value by the smallest.</p>"),
        Step(f"<p>Statics: the tip must hold a weight, pushing with force $F = ({_n(F[0])}, {_n(F[1])})$ N in {{s}} "
             f"(same angles).  What torque must joint 1 give?  Use $\\tau = J^{{T}}F$.</p>",
             tgt, Check("joint-1 torque τ₁", "N·m", lambda: float((J2(th).T @ F)[0]), abs_tol=1e-3),
             "<p>$\\tau_1 = J_{11}F_x + J_{21}F_y$.  The <i>Statics</i> box of the Velocity tab computes the same "
             "thing.</p>"),
    ]
    intro += f"<p><b>Numbers for this lesson:</b> {th_txt}; tip force $F = ({_n(F[0])}, {_n(F[1])})$ N.</p>"
    return Lesson(lid, "SBEG 361", 10, "Singularities and statics", 25, steps, intro, "MR Secs. 5.2–5.3",
                  dict(theta=th, F=F, L1=L1, L2=L2))


def _nr_2r(rob, xd, th0, n_iter=1):
    """Position-only Newton-Raphson for the planar 2R arm (MR Sec. 6.2.1): θ ← θ + J⁻¹(x_d − f(θ))."""
    from .kinematics import point_jacobian
    th = np.array(th0, float)
    for _ in range(n_iter):
        e = xd - mr.FKinSpace(rob.M, rob.Slist, th)[:2, 3]
        th = th + np.linalg.solve(point_jacobian(rob, th)[:2], e)
    return th


def _lesson_ik() -> Lesson:
    lid = "361-w11-newton-raphson"
    rng = rng_for(lid)
    rob = _robot(R2_NAME)
    while True:
        thd = np.array([_pick(rng, [20, 30, 40, 50]), _pick(rng, [40, 50, 60, 70, 80])], float) * DEG
        th0 = thd + np.array([_pick(rng, [-15, -10, 10, 15]), _pick(rng, [-20, -15, 15, 20])]) * DEG
        xd = mr.FKinSpace(rob.M, rob.Slist, thd)[:2, 3]
        th1 = _nr_2r(rob, xd, th0, 1)
        e1 = np.linalg.norm(xd - mr.FKinSpace(rob.M, rob.Slist, th1)[:2, 3])
        thN = _nr_2r(rob, xd, th0, 30)
        if np.allclose(thN, thd, atol=1e-8) and e1 > 1e-4 and abs(np.degrees(th1[0])) > 1:
            break

    def err(th):
        return float(np.linalg.norm(xd - mr.FKinSpace(rob.M, rob.Slist, th)[:2, 3]))

    def iters() -> float:
        th = np.array(th0, float)
        for k in range(1, 50):
            th = _nr_2r(rob, xd, th, 1)
            if err(th) < 1e-6:
                return float(k)
        raise RuntimeError("no convergence")
    tgt = Target("IKTab", R2_NAME)
    intro = (f"<p>Inverse kinematics asks: which joint angles put the tip at a given point?  For the planar 2R arm "
             f"we want the tip at $x_d = ({_n(xd[0], 4)}, {_n(xd[1], 4)})$ m.  Newton–Raphson starts from a guess "
             "and repeats (MR Sec. 6.2):</p>"
             "$$\\theta^{i+1} = \\theta^{i} + J^{-1}(\\theta^{i})\\left(x_d - f(\\theta^{i})\\right)$$"
             f"<p>Start guess: $\\theta^{{0}} = ({np.degrees(th0[0]):g}^{{\\circ}}, "
             f"{np.degrees(th0[1]):g}^{{\\circ}})$.  "
             "Work in radians inside the formula; type angles in degrees.</p>")
    steps = [
        Step("<p>Compute the tip position $f(\\theta^{0})$ at the guess.  How far is it from the target?  "
             "Give the distance $\\|x_d - f(\\theta^{0})\\|$.</p>",
             tgt, Check("start error ‖x_d − f(θ⁰)‖", "m", lambda: err(th0), abs_tol=1e-4),
             "<p>Use $x = L_1\\cos\\theta_1 + L_2\\cos(\\theta_1+\\theta_2)$ and the matching $y$.</p>"),
        Step("<p>Do one Newton–Raphson step.  What is $\\theta_1^{1}$ (in degrees)?</p>",
             tgt, Check("θ₁ after one step", "°", lambda: math.degrees(_nr_2r(rob, xd, th0, 1)[0]), abs_tol=0.05),
             "<p>Build the $2\\times 2$ $J(\\theta^{0})$, solve $J\\,\\Delta\\theta = x_d - f(\\theta^{0})$, add "
             "$\\Delta\\theta$ (radians!) to $\\theta^{0}$, then convert to degrees.</p>"),
        Step("<p>And $\\theta_2^{1}$ (in degrees)?</p>",
             tgt, Check("θ₂ after one step", "°", lambda: math.degrees(_nr_2r(rob, xd, th0, 1)[1]), abs_tol=0.05),
             "<p>Second entry of the same update.</p>"),
        Step("<p>How far is the tip from the target after this one step?  Compare it with the start error: "
             "Newton–Raphson roughly squares the error each step once it is close.</p>",
             tgt, Check("error after one step", "m", lambda: err(_nr_2r(rob, xd, th0, 1)), rel_tol=0.02, abs_tol=1e-6),
             "<p>Evaluate $f(\\theta^{1})$ and take the distance to $x_d$.</p>"),
        Step("<p>Keep iterating until the error is below $10^{-6}$ m.  How many steps did it take (counting the "
             "first)?  The IK tab plots the error of every iteration.</p>",
             tgt, Check("number of iterations", "", iters, rel_tol=0.0, abs_tol=0.25),
             "<p>Expect only a handful: convergence is quadratic.</p>"),
    ]
    return Lesson(lid, "SBEG 361", 11, "Newton–Raphson inverse kinematics", 30, steps, intro, "MR Sec. 6.2",
                  dict(theta0=th0, theta_d=thd, x_d=xd))


# ------------------------------------------------------------------------------------------------
# SBEG 461
# ------------------------------------------------------------------------------------------------


def _r2_params():
    rob = _robot(R2_NAME)
    L1 = -float(rob.Slist[4, 1])
    L2 = float(rob.M[0, 3]) - L1
    m1, m2 = float(rob.Glist[0][3, 3]), float(rob.Glist[1][3, 3])
    return rob, L1, L2, m1, m2


def _lesson_mass_matrix() -> Lesson:
    lid = "461-w01-mass-matrix"
    rng = rng_for(lid)
    rob, L1, L2, m1, m2 = _r2_params()
    eng = _engine(R2_NAME)
    # elbow angles away from ~116°, where M12 nearly cancels and a 1 % check would be unfair
    th = np.array([_pick(rng, [0, 30, 45]), _pick(rng, [30, 45, 60, 150])], float) * DEG
    dth = np.array([_pick(rng, [0.5, 1.0, 1.5]), _pick(rng, [-1.0, 0.5, 1.0, 2.0])])
    r = 0.025
    tgt = Target("DynamicsTab", R2_NAME)
    intro = (f"<p>The planar 2R arm has two uniform rods: link 1 is {_n(L1)} m long with mass {_n(m1)} kg, link 2 is "
             f"{_n(L2)} m with mass {_n(m2)} kg.  Each rod is a thin cylinder of radius {r} m whose centre of mass "
             "is at its middle.  The mass matrix $M(\\theta)$ collects how hard the arm is to accelerate: "
             "the kinetic energy is $K = \\frac{1}{2}\\dot{\\theta}^{T}M(\\theta)\\dot{\\theta}$ "
             "(MR Secs. 8.1–8.2).</p>"
             "$$M(\\theta) = \\sum_{i} J_i^{T}\\mathcal{G}_iJ_i$$")
    steps = [
        Step(f"<p>The rotational inertia of a solid cylinder of mass $m$, length $L$ and radius $r$ about a "
             "transverse axis through its centre is $\\mathcal{I} = m(3r^{2} + L^{2})/12$.  Compute it for link 1.</p>",
             tgt, Check("I_zz of link 1 about its centre", "kg·m²", lambda: float(rob.Glist[0][2, 2]), abs_tol=1e-5),
             "<p>The $r^{2}$ term is small but counts at 1 % tolerance.</p>"),
        Step(f"<p>At <b>θ2 = {_d(th[1])}</b>, find $M_{{22}}$: the inertia joint 2 feels when it turns alone.  "
             "It is link 2's inertia about the joint-2 axis.</p>",
             tgt, Check("M₂₂", "kg·m²", lambda: float(eng.mass_matrix(th)[1, 1]), abs_tol=1e-5),
             "<p>Steiner: $\\mathcal{I}_2 + m_2(L_2/2)^{2}$; it does not depend on $\\theta$.</p>"),
        Step(f"<p>Still at θ2 = {_d(th[1])}: find the coupling term $M_{{12}}$.</p>",
             tgt, Check("M₁₂", "kg·m²", lambda: float(eng.mass_matrix(th)[0, 1]), abs_tol=1e-5),
             "<p>$M_{12} = \\mathcal{I}_2 + m_2\\left((L_2/2)^{2} + L_1(L_2/2)\\cos\\theta_2\\right)$.</p>"),
        Step(f"<p>And $M_{{11}}$ (θ2 = {_d(th[1])}).  The Dynamics tab shows $M(\\theta)$ for the current "
             "configuration — set the same angles and compare.</p>",
             tgt, Check("M₁₁", "kg·m²", lambda: float(eng.mass_matrix(th)[0, 0]), abs_tol=1e-5),
             "<p>$M_{11} = \\mathcal{I}_1 + m_1(L_1/2)^{2} + \\mathcal{I}_2 + m_2\\left(L_1^{2} + (L_2/2)^{2} + "
             "2L_1(L_2/2)\\cos\\theta_2\\right)$.</p>"),
        Step(f"<p>The joints move at $\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s.  What is the kinetic "
             "energy $K = \\frac{1}{2}\\dot{\\theta}^{T}M\\dot{\\theta}$?</p>",
             tgt, Check("kinetic energy K", "J", lambda: eng.kinetic_energy(th, dth), abs_tol=1e-5),
             "<p>$K = \\frac{1}{2}(M_{11}\\dot{\\theta}_1^{2} + 2M_{12}\\dot{\\theta}_1\\dot{\\theta}_2 + "
             "M_{22}\\dot{\\theta}_2^{2})$.</p>"),
    ]
    intro += (f"<p><b>Numbers for this lesson:</b> θ1 = {_d(th[0])}, θ2 = {_d(th[1])}; "
              f"$\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s.</p>")
    return Lesson(lid, "SBEG 461", 1, "Inertia and the mass matrix", 30, steps, intro, "MR Secs. 8.1–8.2",
                  dict(theta=th, dtheta=dth))


def _lesson_rnea() -> Lesson:
    lid = "461-w02-rnea"
    rng = rng_for(lid)
    rob, L1, L2, m1, m2 = _r2_params()
    eng = _engine(R2_NAME)
    while True:
        th = np.array([_pick(rng, [-30, 0, 30, 45]), _pick(rng, [30, 45, 60, 90])], float) * DEG
        dth = np.array([_pick(rng, [0.5, 1.0, 2.0]), _pick(rng, [-1.0, 1.0, 1.5])])
        ddth = np.array([_pick(rng, [-2.0, 1.0, 2.0]), _pick(rng, [-1.0, 1.0, 3.0])])
        g = eng.gravity_torque(th)
        tau = eng.inverse(th, dth, ddth)
        c = eng.coriolis(th, dth)
        if min(abs(g).min(), abs(tau).min(), abs(c[1])) > 0.05:
            break
    tgt = Target("DynamicsTab", R2_NAME)
    intro = (f"<p>The planar 2R arm (rods of {_n(L1)} m / {_n(m1)} kg and {_n(L2)} m / {_n(m2)} kg) moves in a "
             "<b>vertical</b> plane: gravity is $9.81$ m/s² along $-y$.  The recursive Newton–Euler algorithm "
             "(MR Sec. 8.3) gives the joint torques for a motion: a forward pass for link twists and accelerations, "
             "then a backward pass for the wrenches.  The result splits as</p>"
             "$$\\tau = M(\\theta)\\ddot{\\theta} + c(\\theta, \\dot{\\theta}) + g(\\theta)$$")
    steps = [
        Step("<p>The arm is held still ($\\dot{\\theta} = \\ddot{\\theta} = 0$).  What torque must joint 1 give to "
             "hold it against gravity?</p>",
             tgt, Check("gravity torque g₁", "N·m", lambda: float(eng.gravity_torque(th)[0]), abs_tol=1e-3),
             "<p>Each link's weight acts at its centre; $g_1$ is the moment of both weights about joint 1, "
             "e.g. $m_1 g (L_1/2)\\cos\\theta_1 + m_2 g (L_1\\cos\\theta_1 + (L_2/2)\\cos(\\theta_1+\\theta_2))$.</p>"),
        Step("<p>Still held still: the holding torque of joint 2?</p>",
             tgt, Check("gravity torque g₂", "N·m", lambda: float(eng.gravity_torque(th)[1]), abs_tol=1e-3),
             "<p>Only link 2's weight acts beyond joint 2.</p>"),
        Step(f"<p>Now the arm moves with $\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s.  The velocity "
             "terms $c(\\theta, \\dot{\\theta})$ (centripetal and Coriolis) need torque too.  What is $c_2$?</p>",
             tgt, Check("velocity torque c₂", "N·m", lambda: float(eng.coriolis(th, dth)[1]), abs_tol=1e-3),
             "<p>For the 2R arm $c_2 = m_2L_1(L_2/2)\\sin\\theta_2\\,\\dot{\\theta}_1^{2}$.</p>"),
        Step(f"<p>Add the accelerations $\\ddot{{\\theta}} = ({_n(ddth[0])}, {_n(ddth[1])})$ rad/s².  Run the "
             "Newton–Euler passes (or use $M\\ddot{\\theta} + c + g$).  What is the total torque $\\tau_1$?  "
             "The <i>Inverse dynamics — RNEA</i> box of the Dynamics tab computes it.</p>",
             tgt, Check("τ₁", "N·m", lambda: float(eng.inverse(th, dth, ddth)[0]), abs_tol=1e-3),
             "<p>Use $M(\\theta)$ from the mass-matrix lesson.</p>"),
        Step("<p>And $\\tau_2$ for the same motion?</p>",
             tgt, Check("τ₂", "N·m", lambda: float(eng.inverse(th, dth, ddth)[1]), abs_tol=1e-3),
             "<p>Second row of $M\\ddot{\\theta} + c + g$.</p>"),
    ]
    intro += ("<p><b>Numbers for this lesson:</b> "
              f"θ1 = {_d(th[0])}, θ2 = {_d(th[1])}; $\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s; "
              f"$\\ddot{{\\theta}} = ({_n(ddth[0])}, {_n(ddth[1])})$ rad/s².</p>")
    return Lesson(lid, "SBEG 461", 2, "Newton–Euler joint torques", 30, steps, intro, "MR Sec. 8.3",
                  dict(theta=th, dtheta=dth, ddtheta=ddth))


def _lesson_pd() -> Lesson:
    lid = "461-w05-pd"
    rng = rng_for(lid)
    Mj = float(_pick(rng, [0.5, 1.0, 2.0]))
    b = float(_pick(rng, [0.2, 0.5, 1.0]))
    Kp = float(_pick(rng, [20.0, 50.0, 80.0, 100.0]))
    Kd = float(_pick(rng, [2.0, 3.0, 4.0]))
    wn = math.sqrt(Kp / Mj)
    zeta = (b + Kd) / (2 * math.sqrt(Kp * Mj))
    assert 0.05 < zeta < 0.95
    tgt = Target("ControlTab", R2_NAME)
    intro = (f"<p>One joint with inertia $M = {_n(Mj)}$ kg·m² and viscous friction $b = {_n(b)}$ N·m·s/rad "
             "is driven "
             f"by a PD controller $\\tau = K_p\\theta_e + K_d\\dot{{\\theta}}_e$ with $K_p = {_n(Kp)}$ N·m/rad and "
             f"$K_d = {_n(Kd)}$ N·m·s/rad, holding a fixed setpoint (no gravity).  The error obeys a second-order "
             "equation (MR Secs. 11.2–11.3):</p>"
             "$$M\\ddot{\\theta}_e + (b + K_d)\\dot{\\theta}_e + K_p\\theta_e = 0$$"
             "$$\\omega_n = \\sqrt{K_p/M},\\quad \\zeta = \\frac{b + K_d}{2\\sqrt{K_pM}}$$")
    steps = [
        Step("<p>What is the natural frequency $\\omega_n$?</p>",
             tgt, Check("natural frequency ωₙ", "rad/s", lambda: wn, abs_tol=1e-3), ""),
        Step("<p>What is the damping ratio $\\zeta$?  Is the response underdamped ($\\zeta < 1$)?</p>",
             tgt, Check("damping ratio ζ", "", lambda: zeta, abs_tol=1e-4), ""),
        Step("<p>For an underdamped response the overshoot is $e^{-\\pi\\zeta/\\sqrt{1-\\zeta^{2}}}$.  "
             "Give it in percent.</p>",
             tgt, Check("overshoot", "%", lambda: 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)),
                        abs_tol=0.05),
             "<p>Multiply the fraction by 100.</p>"),
        Step("<p>MR's rule of thumb for the 2 % settling time is $t_s \\approx 4/(\\zeta\\omega_n)$.  "
             "What does it give?</p>",
             tgt, Check("settling time (rule of thumb)", "s", lambda: 4 / (zeta * wn), abs_tol=1e-3),
             "<p>Note that $\\zeta\\omega_n = (b + K_d)/(2M)$.</p>"),
        Step("<p>Which $K_d$ would make the response <b>critically damped</b> ($\\zeta = 1$), keeping $K_p$?</p>",
             tgt, Check("K_d for ζ = 1", "N·m·s/rad", lambda: 2 * math.sqrt(Kp * Mj) - b, abs_tol=1e-3),
             "<p>Solve $b + K_d = 2\\sqrt{K_pM}$.</p>"),
    ]
    return Lesson(lid, "SBEG 461", 5, "PD control of one joint", 20, steps, intro, "MR Secs. 11.2–11.3",
                  dict(M=Mj, b=b, Kp=Kp, Kd=Kd))


def _lesson_computed_torque() -> Lesson:
    lid = "461-w06-computed-torque"
    rng = rng_for(lid)
    rob, L1, L2, m1, m2 = _r2_params()
    wn = float(_pick(rng, [4.0, 5.0, 8.0, 10.0]))
    zeta = float(_pick(rng, [0.7, 0.8, 1.0]))
    kp, kd = wn**2, 2 * zeta * wn
    th = np.array([_pick(rng, [10, 20, 30]), _pick(rng, [40, 50, 60])], float) * DEG
    dth = np.array([_pick(rng, [0.2, 0.5]), _pick(rng, [-0.3, 0.4])])
    thd = th + np.array([_pick(rng, [5, 8, 10]), _pick(rng, [-6, 6, 12])]) * DEG
    dthd = np.array([_pick(rng, [0.4, 0.6]), _pick(rng, [0.1, 0.2])])
    ddthd = np.array([_pick(rng, [1.0, 2.0]), _pick(rng, [-1.0, 0.5])])
    g = rob.gravity

    def tau():
        return np.asarray(mr.ComputedTorque(th, dth, np.zeros(2), g, rob.Mlist, rob.Glist, rob.Slist,
                                            thd, dthd, ddthd, kp, 0.0, kd))

    def err_acc1():
        ddth = mr.ForwardDynamics(th, dth, tau(), g, np.zeros(6), rob.Mlist, rob.Glist, rob.Slist)
        return float(ddthd[0] - ddth[0])
    tgt = Target("ControlTab", R2_NAME)
    intro = ("<p>Computed-torque control uses the arm's own model to cancel its dynamics (MR Sec. 11.4):</p>"
             "$$\\tau = M(\\theta)\\left(\\ddot{\\theta}_d + K_p\\theta_e + K_d\\dot{\\theta}_e\\right) + "
             "c(\\theta, \\dot{\\theta}) + g(\\theta)$$"
             "<p>With a perfect model every joint error then follows "
             "$\\ddot{\\theta}_e + K_d\\dot{\\theta}_e + K_p\\theta_e = 0$, the same equation for each joint "
             "when $K_p = k_pI$ and $K_d = k_dI$.</p>")
    steps = [
        Step(f"<p>We want each joint error to have natural frequency $\\omega_n = {_n(wn)}$ rad/s.  "
             "Which $k_p$ gives that?</p>",
             tgt, Check("k_p", "1/s²", lambda: kp, abs_tol=1e-3), "<p>$\\omega_n = \\sqrt{k_p}$.</p>"),
        Step(f"<p>…and damping ratio $\\zeta = {_n(zeta)}$.  Which $k_d$?</p>",
             tgt, Check("k_d", "1/s", lambda: kd, abs_tol=1e-3), "<p>$2\\zeta\\omega_n = k_d$.</p>"),
        Step(f"<p>The planar 2R arm (gravity along $-y$) is at θ = ({_d(th[0])}, {_d(th[1])}) moving at "
             f"$\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s.  The reference is θ_d = ({_d(thd[0])}, "
             f"{_d(thd[1])}), $\\dot{{\\theta}}_d = ({_n(dthd[0])}, {_n(dthd[1])})$ rad/s, "
             f"$\\ddot{{\\theta}}_d = ({_n(ddthd[0])}, {_n(ddthd[1])})$ rad/s².  What torque does the controller "
             "command at joint 1?</p>",
             tgt, Check("commanded τ₁", "N·m", lambda: float(tau()[0]), abs_tol=1e-3),
             "<p>First form $\\ddot{\\theta}_d + k_p\\theta_e + k_d\\dot{\\theta}_e$ (errors in radians), multiply "
             "by $M(\\theta)$, then add $c + g$ from the Newton–Euler lesson.</p>"),
        Step("<p>And at joint 2?</p>",
             tgt, Check("commanded τ₂", "N·m", lambda: float(tau()[1]), abs_tol=1e-3),
             "<p>Second row of the same expression.</p>"),
        Step("<p>Apply that torque to the (perfectly modelled) arm.  What is the error acceleration "
             "$\\ddot{\\theta}_{e,1} = \\ddot{\\theta}_{d,1} - \\ddot{\\theta}_1$ of joint 1?  It must equal "
             "$-k_p\\theta_{e,1} - k_d\\dot{\\theta}_{e,1}$.</p>",
             tgt, Check("error acceleration of joint 1", "rad/s²", err_acc1, abs_tol=1e-3),
             "<p>You do not need forward dynamics: use the error equation directly.</p>"),
    ]
    intro += ("<p><b>Numbers for this lesson:</b> "
              f"θ = ({_d(th[0])}, {_d(th[1])}), $\\dot{{\\theta}} = ({_n(dth[0])}, {_n(dth[1])})$ rad/s; "
              f"θ_d = ({_d(thd[0])}, {_d(thd[1])}), $\\dot{{\\theta}}_d = ({_n(dthd[0])}, {_n(dthd[1])})$ rad/s, "
              f"$\\ddot{{\\theta}}_d = ({_n(ddthd[0])}, {_n(ddthd[1])})$ rad/s²; $\\omega_n = {_n(wn)}$ rad/s, "
              f"$\\zeta = {_n(zeta)}$.</p>")
    return Lesson(lid, "SBEG 461", 6, "Computed-torque control", 30, steps, intro, "MR Sec. 11.4",
                  dict(kp=kp, kd=kd, theta=th, dtheta=dth, theta_d=thd, dtheta_d=dthd, ddtheta_d=ddthd))


def _lesson_time_scaling() -> Lesson:
    lid = "461-w10-time-scaling"
    rng = rng_for(lid)
    from .trajectory import time_scaling
    d_deg = float(_pick(rng, [60, 90, 120]))
    th0_deg = float(_pick(rng, [-30, 0, 15]))
    T = float(_pick(rng, [1.5, 2.0, 3.0]))
    vmax = float(_pick(rng, [45, 60, 90]))          # deg/s limit for the last step
    tt = np.linspace(0, T, 20001)

    def peak(kind, k):
        return float(np.abs(time_scaling(kind, T, tt)[k]).max()) * d_deg
    tgt = Target("ControlTab", R2_NAME)
    intro = (f"<p>Joint 1 of the planar 2R arm must turn from {th0_deg:g}° to {th0_deg + d_deg:g}° "
             f"(a change of $\\Delta = {d_deg:g}^{{\\circ}}$) in $T = {T:g}$ s, starting and ending at rest.  "
             "A time scaling $s(t)$ goes from 0 to 1 and the joint follows $\\theta(t) = \\theta_0 + s(t)\\Delta$ "
             "(MR Sec. 9.2):</p>"
             "$$s_{cubic}(t) = 3\\left(\\frac{t}{T}\\right)^{2} - 2\\left(\\frac{t}{T}\\right)^{3}$$"
             "$$s_{quintic}(t) = 10\\left(\\frac{t}{T}\\right)^{3} - 15\\left(\\frac{t}{T}\\right)^{4} + "
             "6\\left(\\frac{t}{T}\\right)^{5}$$")
    steps = [
        Step("<p>With the <b>cubic</b> scaling, what is the largest joint speed $|\\dot{\\theta}|$ (in °/s)?</p>",
             tgt, Check("peak speed, cubic", "°/s", lambda: peak("cubic", 1), abs_tol=0.01),
             "<p>$\\dot{s}$ is largest at $t = T/2$: $\\dot{s}_{max} = 3/(2T)$.</p>"),
        Step("<p>Cubic again: the largest acceleration $|\\ddot{\\theta}|$ (in °/s²)?  Notice it jumps at the start "
             "— a cubic cannot start with zero acceleration.</p>",
             tgt, Check("peak acceleration, cubic", "°/s²", lambda: peak("cubic", 2), abs_tol=0.01),
             "<p>$\\ddot{s} = (6 - 12t/T)/T^{2}$, largest at $t = 0$ and $t = T$.</p>"),
        Step("<p>Now the <b>quintic</b> scaling.  Its peak speed (°/s)?  It is higher than the cubic one: smoothness "
             "has a price.</p>",
             tgt, Check("peak speed, quintic", "°/s", lambda: peak("quintic", 1), abs_tol=0.01),
             "<p>Also at $t = T/2$: $\\dot{s}_{max} = 15/(8T)$.</p>"),
        Step(f"<p>Quintic: where is joint 1 at $t = T/4 = {T / 4:g}$ s?  Give $\\theta(T/4)$ in degrees.</p>",
             tgt, Check("θ(T/4), quintic", "°",
                        lambda: th0_deg + d_deg * float(time_scaling("quintic", T, [T / 4])[0][0]), abs_tol=0.01),
             "<p>Put $t/T = 0.25$ into $s_{quintic}$.</p>"),
        Step(f"<p>The motor may not exceed {vmax:g}°/s.  What is the shortest $T$ for which the quintic move "
             "respects that limit?  Try it in the Control tab with a quintic time scaling.</p>",
             tgt, Check("shortest T, quintic", "s", lambda: peak("quintic", 1) * T / vmax, abs_tol=1e-3),
             "<p>The peak speed is proportional to $\\Delta/T$.</p>"),
    ]
    return Lesson(lid, "SBEG 461", 10, "Cubic and quintic time scaling", 20, steps, intro, "MR Sec. 9.2",
                  dict(delta_deg=d_deg, theta0_deg=th0_deg, T=T, vmax_deg=vmax))


# ------------------------------------------------------------------------------------------------
# registry
# ------------------------------------------------------------------------------------------------

_BUILDERS = [_lesson_poe, _lesson_jacobian, _lesson_singularities, _lesson_ik, _lesson_mass_matrix, _lesson_rnea,
             _lesson_pd, _lesson_computed_torque, _lesson_time_scaling]
_lessons: list[Lesson] | None = None


def all_lessons() -> list[Lesson]:
    """Every lesson, ordered by course then week (built once; references are computed on first use)."""
    global _lessons
    if _lessons is None:
        _lessons = sorted((b() for b in _BUILDERS), key=lambda L: (L.course, L.week))
    return _lessons


def get_lesson(lesson_id: str) -> Lesson:
    for L in all_lessons():
        if L.id == lesson_id:
            return L
    raise KeyError(lesson_id)
