"""Randomised practice problems with step-by-step worked solutions (Modern Robotics, Ch. 3-11).

Every generator is deterministic in ``(topic, seed, difficulty)`` and returns a :class:`Problem`:

* ``statement``  -- Markdown with LaTeX (pandoc-ready; the GUI renders the same blocks),
* ``data``       -- the numeric inputs (NumPy arrays / floats),
* ``steps``      -- the worked solution, one :class:`Step` per stage,
* ``answer``     -- the final numbers (dict) and ``final_answer`` (Markdown).

All answers are computed with Lynch & Park's ``modern_robotics`` package (``mr``).  Each topic also
has an *independent* re-solver (SciPy matrix exponentials, finite differences, closed-form
Lagrangian, boundary-value solve of the polynomial, eigenvalues of the error dynamics) used by
:func:`verify`, :func:`validation_checks` and the test-suite.

Topics (MR sections, printed pages of the Dec-2019 updated first edition):

========== ====================================================================================
rotation   Ch. 3  Rodrigues (3.51) p. 82, log of a rotation (3.58)-(3.61) pp. 85-86,
                  screw axis p. 110, exp/log on SE(3) (3.86)-(3.92) pp. 103-104
fk         Ch. 4  space / body PoE (4.14) p. 140, (4.16) p. 147
jacobian   Ch. 5  J_s (5.11) p. 178, J_b (5.18) p. 183, J_b = [Ad_Tbs] J_s (5.22) p. 185,
                  singularities Sec. 5.3 p. 189
ik         Ch. 6  Newton-Raphson with the body Jacobian, Sec. 6.2.2 p. 228, Example 6.1 pp. 229-230
dynamics   Ch. 8  2R Lagrangian (8.9)-(8.10) p. 273, Newton-Euler RNEA (8.50)-(8.55) pp. 292-293
trajectory Ch. 9  cubic (9.9)-(9.13) pp. 331-332, quintic p. 332 (Fig. 9.4 p. 333)
control    Ch. 11 second-order error dynamics (11.8) p. 412, overshoot / settling pp. 414-415,
                  PD (11.25)-(11.26) p. 425, computed torque (11.33)-(11.35) p. 431
========== ====================================================================================
"""
from __future__ import annotations

import html as _html
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np
from scipy.linalg import expm

from .checks import Check

DEG = np.pi / 180.0
G = 9.81

# ------------------------------------------------------------------------------------------------
# number formatting
# ------------------------------------------------------------------------------------------------


def num(x, d: int = 3) -> str:
    """Compact number: rounded to ``d`` decimals, trailing zeros stripped, no '-0'."""
    x = float(x)
    r = round(x, d)
    if r == 0:
        return "0"
    s = f"{r:.{d}f}".rstrip("0").rstrip(".")
    return s


def vec(v, d: int = 3) -> str:
    return "(" + ", ".join(num(x, d) for x in np.ravel(v)) + ")"


def degs(v, d: int = 2) -> str:
    """Angles (rad) as '(30°, 45°)' plain text."""
    return "(" + ", ".join(num(np.degrees(x), d) + "°" for x in np.ravel(v)) + ")"


def mdeg(x, d: int = 2) -> str:
    """One angle (rad) as LaTeX '30^{\\circ}'."""
    return num(np.degrees(x), d) + r"^{\circ}"


# ------------------------------------------------------------------------------------------------
# a tiny document model -> Markdown (export) and HTML (GUI, rendered by TheoryPanel)
# ------------------------------------------------------------------------------------------------


class Doc:
    """Sequence of blocks: paragraphs (Markdown inline text with $math$), display equations,
    matrices and tables.  Inline/display math is restricted to the matplotlib-mathtext subset so
    the GUI can render it; matrices become ``bmatrix`` in Markdown and HTML tables in the GUI."""

    def __init__(self):
        self.blocks: list[tuple] = []

    def p(self, text: str) -> "Doc":
        self.blocks.append(("p", text))
        return self

    def eq(self, tex: str) -> "Doc":
        self.blocks.append(("eq", tex))
        return self

    def mat(self, lhs: str, A, d: int = 3, rhs: str = "") -> "Doc":
        self.blocks.append(("mat", lhs, np.atleast_2d(np.asarray(A, float)), d, rhs))
        return self

    def table(self, headers: list[str], rows: list[list[str]]) -> "Doc":
        self.blocks.append(("table", list(headers), [list(map(str, r)) for r in rows]))
        return self

    def extend(self, other: "Doc") -> "Doc":
        self.blocks.extend(other.blocks)
        return self

    # -- Markdown -------------------------------------------------------------------------------
    def markdown(self) -> str:
        out = []
        for b in self.blocks:
            kind = b[0]
            if kind == "p":
                out.append(md_inline(b[1]))
            elif kind == "eq":
                out.append(f"$${b[1]}$$")
            elif kind == "mat":
                _, lhs, A, d, rhs = b
                rows = r" \\ ".join(" & ".join(num(x, d) for x in r) for r in A)
                body = r"\begin{bmatrix} " + rows + r" \end{bmatrix}"
                out.append(f"$${lhs} = {body}{(' ' + rhs) if rhs else ''}$$")
            elif kind == "table":
                _, hd, rows = b
                lines = ["| " + " | ".join(map(md_inline, hd)) + " |", "|" + "|".join(":-:" for _ in hd) + "|"]
                lines += ["| " + " | ".join(map(md_inline, r)) + " |" for r in rows]
                out.append("\n".join(lines))
        return "\n\n".join(out) + "\n"

    # -- HTML for the GUI -----------------------------------------------------------------------
    def html(self) -> str:
        out = []
        for b in self.blocks:
            kind = b[0]
            if kind == "p":
                out.append(f"<p>{inline_html(b[1])}</p>")
            elif kind == "eq":
                out.append(f"$${b[1]}$$")
            elif kind == "mat":
                _, lhs, A, d, rhs = b
                cells = "".join("<tr>" + "".join(f'<td align="right">&nbsp;{num(x, d)}&nbsp;</td>' for x in r)
                                + "</tr>" for r in A)
                m = (f'<table border="1" cellspacing="0" cellpadding="2" style="border-color:#4a5260; '
                     f'border-style:solid">{cells}</table>')
                out.append(f'<table align="center" cellspacing="4"><tr><td valign="middle">${lhs} =$</td>'
                           f'<td>{m}</td>{f"<td valign=middle>${rhs}$</td>" if rhs else ""}</tr></table>')
            elif kind == "table":
                _, hd, rows = b
                h = "".join(f"<th>&nbsp;{inline_html(x)}&nbsp;</th>" for x in hd)
                r = "".join("<tr>" + "".join(f'<td align="center">&nbsp;{inline_html(x)}&nbsp;</td>' for x in row)
                            + "</tr>" for row in rows)
                out.append(f'<table align="center" border="1" cellspacing="0" cellpadding="3" '
                           f'style="border-color:#4a5260; border-style:solid"><tr>{h}</tr>{r}</table>')
        return "\n".join(out)

    def math(self) -> list[str]:
        """Every math string the GUI will render (for mathtext validation in the tests)."""
        out = []
        for b in self.blocks:
            if b[0] == "p":
                out += re.findall(r"\$(.+?)\$", b[1])
            elif b[0] == "eq":
                out.append(b[1])
            elif b[0] == "mat":
                out.append(b[1] + " =")
                if b[4]:
                    out.append(b[4])
            elif b[0] == "table":
                for s in b[1] + [c for r in b[2] for c in r]:
                    out += re.findall(r"\$(.+?)\$", s)
        return out


def md_inline(text: str) -> str:
    """Pandoc-safe inline math: pandoc only reads ``$...$`` as TeX when there is no space just inside the
    dollars (``$\\theta = $`` would come out as literal text in Word), so trim it."""
    return re.sub(r"\$([^$]+)\$", lambda m: "$" + m.group(1).strip() + "$", text)


def inline_html(text: str) -> str:
    """Markdown inline text -> HTML (escape outside math; **bold**, *italic*)."""
    parts = re.split(r"(\$[^$]+\$)", text)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            out.append(p)
        else:
            s = _html.escape(p, quote=False)
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*", r"<i>\1</i>", s)
            out.append(s)
    return "".join(out)


@dataclass
class Step:
    title: str
    doc: Doc


@dataclass
class Problem:
    topic: str
    seed: int
    difficulty: int
    title: str
    chapter: str
    statement_doc: Doc
    data: dict
    steps: list[Step]
    answer: dict
    answer_doc: Doc
    visual: dict = field(default_factory=dict)

    # Markdown views ----------------------------------------------------------------------------
    @property
    def statement(self) -> str:
        return self.statement_doc.markdown()

    @property
    def final_answer(self) -> str:
        return self.answer_doc.markdown()

    @property
    def solution(self) -> str:
        """Full worked solution (Markdown)."""
        parts = [f"**Step {k}. {s.title}**\n\n{s.doc.markdown()}" for k, s in enumerate(self.steps, 1)]
        parts.append(f"**Final answer.**\n\n{self.final_answer}")
        return "\n".join(parts)

    @property
    def key(self) -> str:
        return f"{self.topic}-d{self.difficulty}-s{self.seed}"

    def markdown(self, number: int | None = None, with_solution: bool = False) -> str:
        head = f"Problem {number}. " if number is not None else ""
        md = (f"## {head}{self.title}\n\n*{self.chapter} — difficulty {DIFFICULTIES[self.difficulty]}, "
              f"seed {self.seed}*\n\n{self.statement}")
        if with_solution:
            md += f"\n### Solution\n\n{self.solution}"
        return md

    # HTML views (GUI) --------------------------------------------------------------------------
    def statement_html(self) -> str:
        return (f"<h2>{_html.escape(self.title)}</h2><span class='ref'>Modern Robotics, {_html.escape(self.chapter)}"
                f" · difficulty: {DIFFICULTIES[self.difficulty]} · seed {self.seed}</span>"
                + self.statement_doc.html())

    def solution_html(self) -> str:
        s = "".join(f"<h3>Step {k}. {_html.escape(st.title)}</h3>{st.doc.html()}" for k, st in enumerate(self.steps, 1))
        return "<h2>Worked solution</h2>" + s + "<h3>Final answer</h3>" + self.answer_doc.html()

    def all_math(self) -> list[str]:
        out = self.statement_doc.math() + self.answer_doc.math()
        for s in self.steps:
            out += s.doc.math()
        return out


DIFFICULTIES = {1: "basic", 2: "standard", 3: "challenge"}

# ------------------------------------------------------------------------------------------------
# robot families
# ------------------------------------------------------------------------------------------------


@dataclass
class RobotSpec:
    name: str
    family: str          # "2R", "3R", "RRP", "3Rs", "6R"
    joints: list[dict]   # dict(type="R"/"P", w=axis (R) | v=direction (P), q=point (R))
    M: np.ndarray
    planar: bool
    dims: dict
    sing_rows: list[int]  # rows of J_b used for the singularity test (square n x n)
    sing_note: str

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def Slist(self) -> np.ndarray:
        cols = []
        for j in self.joints:
            if j["type"] == "R":
                cols.append(mr.ScrewToAxis(np.asarray(j["q"], float), np.asarray(j["w"], float), 0.0))
            else:
                cols.append(np.r_[0, 0, 0, np.asarray(j["v"], float)])
        return np.array(cols).T

    @property
    def Blist(self) -> np.ndarray:
        return mr.Adjoint(mr.TransInv(self.M)) @ self.Slist

    def description_md(self) -> str:
        dims = ", ".join(f"${k} = {num(v)}$ m" for k, v in self.dims.items())
        return f"**Robot:** {self.name} ({dims})."

    def joint_table(self) -> tuple[list[str], list[list[str]]]:
        hd = ["joint", "type", r"axis $\hat{\omega}_i$ / direction", r"point $q_i$ on the axis [m]"]
        rows = []
        for i, j in enumerate(self.joints, 1):
            if j["type"] == "R":
                rows.append([str(i), "revolute", vec(j["w"]), vec(j["q"])])
            else:
                rows.append([str(i), "prismatic", vec(j["v"]), "—"])
        return hd, rows

    def serial_robot(self):
        """A :class:`mrstudio.robots.SerialRobot` (kinematics only) for drawing."""
        from .robots import SerialRobot
        return SerialRobot(self.name, self.M, self.Slist, planar=self.planar,
                           link_radius=0.025 * max(1.0, float(np.linalg.norm(self.M[:3, 3]))))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


_LEN = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]


def random_robot(rng, family: str) -> RobotSpec:
    z, y = [0, 0, 1], [0, 1, 0]
    if family == "2R":
        L1, L2 = _pick(rng, _LEN), _pick(rng, _LEN)
        return RobotSpec("planar 2R arm (joint axes along $\\hat{z}_s$)", "2R",
                         [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=z, q=[L1, 0, 0])],
                         mr.RpToTrans(np.eye(3), [L1 + L2, 0, 0]), True, {"L_1": L1, "L_2": L2},
                         [3, 4], r"rows $(v_x, v_y)$ of $J_b$ (tip linear velocity in {b})")
    if family == "3R":
        L1, L2, L3 = _pick(rng, _LEN), _pick(rng, _LEN), _pick(rng, [0.2, 0.3, 0.4, 0.5])
        return RobotSpec("planar 3R arm (MR Example 4.2 type)", "3R",
                         [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=z, q=[L1, 0, 0]),
                          dict(type="R", w=z, q=[L1 + L2, 0, 0])],
                         mr.RpToTrans(np.eye(3), [L1 + L2 + L3, 0, 0]), True, {"L_1": L1, "L_2": L2, "L_3": L3},
                         [2, 3, 4], r"rows $(\omega_z, v_x, v_y)$ of $J_b$ (planar twist)")
    if family == "RRP":
        L1, L2 = _pick(rng, [0.4, 0.5, 0.6]), _pick(rng, [0.3, 0.4, 0.5])
        H = _pick(rng, [0.3, 0.4, 0.5])
        return RobotSpec("RRP (SCARA-type) arm", "RRP",
                         [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=z, q=[L1, 0, 0]),
                          dict(type="P", v=[0, 0, -1])],
                         mr.RpToTrans(np.eye(3), [L1 + L2, 0, H]), False, {"L_1": L1, "L_2": L2, "H": H},
                         [3, 4, 5], r"rows $(v_x, v_y, v_z)$ of $J_b$ (tip linear velocity in {b})")
    if family == "3Rs":
        H, L1, L2 = _pick(rng, [0.2, 0.3, 0.4]), _pick(rng, [0.4, 0.5, 0.6]), _pick(rng, [0.3, 0.4, 0.5])
        return RobotSpec("spatial 3R elbow arm (base yaw + shoulder + elbow)", "3Rs",
                         [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=y, q=[0, 0, H]),
                          dict(type="R", w=y, q=[L1, 0, H])],
                         mr.RpToTrans(np.eye(3), [L1 + L2, 0, H]), False, {"H": H, "L_1": L1, "L_2": L2},
                         [3, 4, 5], r"rows $(v_x, v_y, v_z)$ of $J_b$ (tip linear velocity in {b})")
    if family == "6R":
        H1, H2 = _pick(rng, [0.1, 0.15, 0.2]), 0.1
        L1, L2 = _pick(rng, [0.4, 0.5]), _pick(rng, [0.4, 0.5])
        W1, W2 = _pick(rng, [0.1, 0.15]), 0.1
        a = L1 + L2
        M = np.array([[-1, 0, 0, a], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
        return RobotSpec("6R UR-type arm (MR Example 4.5 geometry)", "6R",
                         [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=y, q=[0, 0, H1]),
                          dict(type="R", w=y, q=[L1, 0, H1]), dict(type="R", w=y, q=[a, 0, H1]),
                          dict(type="R", w=[0, 0, -1], q=[a, W1, 0]), dict(type="R", w=y, q=[a, 0, H1 - H2])],
                         M, False, {"H_1": H1, "H_2": H2, "L_1": L1, "L_2": L2, "W_1": W1, "W_2": W2},
                         [0, 1, 2, 3, 4, 5], r"the full $6\times 6$ $J_b$")
    raise ValueError(family)


FAMILIES = {1: ["2R", "3R"], 2: ["RRP", "3Rs"], 3: ["6R"]}
#: joint whose value makes the configuration singular (for the Jacobian problems)
SINGULAR_JOINT = {"2R": (1, 0.0), "3R": (1, 0.0), "RRP": (1, 0.0), "3Rs": (2, 0.0), "6R": (4, 0.0)}


def random_theta(rng, spec: RobotSpec, difficulty: int) -> np.ndarray:
    step = 30 if difficulty == 1 else 15
    th = []
    for j in spec.joints:
        if j["type"] == "R":
            k = int(rng.integers(1, 90 // step + 1)) * step * _pick(rng, [-1, 1])
            th.append(k * DEG)
        else:
            th.append(float(_pick(rng, [0.05, 0.1, 0.15, 0.2])))
    return np.array(th)


def _theta_md(spec: RobotSpec, th) -> str:
    parts = []
    for i, (j, t) in enumerate(zip(spec.joints, th), 1):
        parts.append(f"\\theta_{{{i}}} = {mdeg(t) if j['type'] == 'R' else num(t) + ' m'}")
    return "$" + ",\\ ".join(parts) + "$"


def _theta_txt(spec: RobotSpec, th) -> str:
    return "(" + ", ".join(num(np.degrees(t), 2) + "°" if j["type"] == "R" else num(t, 3) + " m"
                           for j, t in zip(spec.joints, th)) + ")"


def _screw_rows(S: np.ndarray, label: str) -> list[list[str]]:
    return [[f"${label}_{{{i + 1}}}$", vec(S[:3, i]), vec(S[3:, i])] for i in range(S.shape[1])]


def _robot_visual(spec: RobotSpec, theta, ghost=None, frames=None) -> dict:
    return dict(kind="robot", spec=spec, theta=np.asarray(theta, float),
                ghost=None if ghost is None else np.asarray(ghost, float), frames=frames or {})


def _rng(topic: str, seed: int, difficulty: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, TOPIC_KEYS.index(topic), int(difficulty)])


# ------------------------------------------------------------------------------------------------
# Ch. 3 -- rotations and rigid motions: exponential and logarithm
# ------------------------------------------------------------------------------------------------

_AXES_EASY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
_AXES = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [0, 3, 4], [3, 0, 4], [4, 3, 0]]


def _rand_axis(rng, easy: bool) -> np.ndarray:
    a = np.array(_pick(rng, _AXES_EASY if easy else _AXES), float)
    a *= np.where(rng.random(3) < 0.3, -1, 1)
    return a / np.linalg.norm(a)


def _axis_md(w) -> str:
    """Unit vector with a nice normalisation factor if possible."""
    for k in (1, np.sqrt(2), np.sqrt(3), 3, 5):
        a = np.asarray(w) * k
        if np.allclose(a, np.round(a), atol=1e-9):
            inner = ", ".join(str(int(round(x))) for x in a)
            if k == 1:
                return f"({inner})"
            ks = {np.sqrt(2): r"\sqrt{2}", np.sqrt(3): r"\sqrt{3}", 3: "3", 5: "5"}[k]
            return f"({inner})/{ks}"
    return vec(w)


def gen_rotation(seed: int, difficulty: int) -> Problem:
    rng = _rng("rotation", seed, difficulty)
    w = _rand_axis(rng, difficulty == 1)
    th = int(rng.integers(1, 6)) * (30 if difficulty == 1 else 15) * DEG
    if difficulty > 1:
        th = int(rng.integers(1, 11)) * 15 * DEG
    w2 = _rand_axis(rng, False)
    th2 = int(rng.integers(2, 11)) * 15 * DEG       # 30..150 deg, away from 0 and pi
    while np.allclose(w2, w) and np.isclose(th2, th):
        th2 = int(rng.integers(2, 11)) * 15 * DEG
    R = mr.MatrixExp3(mr.VecToso3(w * th))
    R2 = mr.MatrixExp3(mr.VecToso3(w2 * th2))
    data = dict(omega_hat=w, theta=th, R2=R2)

    st = Doc()
    st.p(f"**(a)** Compute the rotation matrix $R = \\mathrm{{Rot}}(\\hat{{\\omega}}, \\theta)$ for "
         f"$\\hat{{\\omega}} = {_axis_md(w)}$ and $\\theta = {mdeg(th)}$ using Rodrigues' formula.")
    st.p("**(b)** Find the exponential coordinates $\\hat{\\omega}\\theta$ (unit axis and angle "
         "$\\theta \\in [0, \\pi]$) of the rotation")
    st.mat("R_2", R2, 4)
    steps = []
    s1 = Doc()
    W = mr.VecToso3(w)
    s1.p("The skew-symmetric matrix of the axis and its square:")
    s1.mat("[\\hat{\\omega}]", W, 4)
    s1.mat("[\\hat{\\omega}]^{2}", W @ W, 4)
    s1.p(f"With $\\sin\\theta = {num(np.sin(th), 4)}$ and $1-\\cos\\theta = {num(1 - np.cos(th), 4)}$, "
         "Rodrigues' formula (3.51) gives")
    s1.eq(r"R = I + \sin\theta\,[\hat{\omega}] + (1-\cos\theta)\,[\hat{\omega}]^{2}")
    s1.mat("R", R, 4)
    s1.p(f"Check: $R^{{T}}R = I$ and $\\det R = {num(np.linalg.det(R), 4)}$ (computed with `mr.MatrixExp3`).")
    steps.append(Step("(a) Rodrigues' formula", s1))

    s2 = Doc()
    tr = np.trace(R2)
    wv = mr.so3ToVec(mr.MatrixLog3(R2))
    w2h, th2c = mr.AxisAng3(wv)
    s2.p(f"$\\mathrm{{tr}}\\,R_2 = {num(tr, 4)}$, which is neither 3 ($R_2 = I$) nor $-1$ ($\\theta = \\pi$), "
         "so the general case of the matrix-log algorithm (3.58)–(3.61) applies:")
    s2.eq(r"\theta = \cos^{-1}\left(\frac{\mathrm{tr}\,R - 1}{2}\right) = " + mdeg(th2c, 2)
          + f" = {num(th2c, 4)}" + r"\ \mathrm{rad}")
    s2.eq(r"[\hat{\omega}] = \frac{1}{2\sin\theta}\left(R - R^{T}\right)")
    s2.mat("[\\hat{\\omega}]", mr.VecToso3(w2h), 4)
    s2.p(f"so $\\hat{{\\omega}} = {vec(w2h, 4)}$ and $\\hat{{\\omega}}\\theta = {vec(wv, 4)}$ "
         "(`mr.MatrixLog3`, `mr.AxisAng3`).")
    steps.append(Step("(b) Matrix logarithm of a rotation", s2))
    ans = dict(R=R, omega2=w2h, theta2=th2c)
    ad = Doc().mat("R", R, 4).p(f"(b) $\\hat{{\\omega}} = {vec(w2h, 4)}$, $\\theta = {mdeg(th2c)}$ "
                                 f"$= {num(th2c, 4)}$ rad.")
    frames = {"R": R}

    if difficulty >= 2:
        q = rng.integers(-2, 3, 3) * 0.5
        s_hat = _rand_axis(rng, difficulty == 2)
        h = float(_pick(rng, [0.0, 0.1, 0.2, -0.1, 0.5]))
        th3 = int(rng.integers(2, 9)) * 15 * DEG
        S = mr.ScrewToAxis(q, s_hat, h)
        T = mr.MatrixExp6(mr.VecTose3(S * th3))
        data.update(q=q, s_hat=s_hat, h=h, theta3=th3)
        st.p(f"**(c)** A screw axis passes through $q = {vec(q)}$ m with direction $\\hat{{s}} = {_axis_md(s_hat)}$ "
             f"and pitch $h = {num(h)}$ m/rad.  Write $\\mathcal{{S}} = (\\omega, v)$ and compute "
             f"$T = e^{{[\\mathcal{{S}}]\\theta}}$ for $\\theta = {mdeg(th3)}$.")
        s3 = Doc()
        s3.p("Screw axis from (q, ŝ, h) (MR Sec. 3.3.2.2, p. 110):")
        s3.eq(r"\mathcal{S} = \left(\hat{s},\ -\hat{s}\times q + h\hat{s}\right)")
        s3.p(f"$\\omega = {vec(S[:3], 4)}$, $v = {vec(S[3:], 4)}$ (`mr.ScrewToAxis`).  Then (3.88):")
        s3.eq(r"e^{[\mathcal{S}]\theta} = \left[e^{[\omega]\theta},\ G(\theta)\,v\right],\quad "
              r"G(\theta) = I\theta + (1-\cos\theta)[\omega] + (\theta-\sin\theta)[\omega]^{2}")
        Gm = np.eye(3) * th3 + (1 - np.cos(th3)) * mr.VecToso3(S[:3]) + (th3 - np.sin(th3)) * mr.VecToso3(S[:3]) @ mr.VecToso3(S[:3])
        s3.mat("G(\\theta)", Gm, 4)
        s3.p(f"$G(\\theta)v = {vec(Gm @ S[3:], 4)}$, and the rotation block is Rodrigues' formula as in (a):")
        s3.mat("T", T, 4)
        steps.append(Step("(c) Exponential of a screw motion", s3))
        ans.update(S=S, T=T)
        ad.p(f"(c) $\\mathcal{{S}} = {vec(S, 4)}$").mat("T", T, 4)
        frames["T"] = T
    if difficulty >= 3:
        q4 = rng.integers(-2, 3, 3) * 0.5
        s4 = _rand_axis(rng, False)
        h4 = float(_pick(rng, [0.1, 0.2, 0.25, -0.2]))
        th4 = int(rng.integers(2, 10)) * 15 * DEG
        while np.allclose(s4, w2) and np.isclose(th4, th2):     # (d) must not repeat the rotation of (b)
            s4, th4 = _rand_axis(rng, False), int(rng.integers(2, 10)) * 15 * DEG
        T4 = mr.MatrixExp6(mr.VecTose3(mr.ScrewToAxis(q4, s4, h4) * th4))
        data.update(T4=T4)
        st.p("**(d)** Find the screw axis $\\mathcal{S}$, the angle $\\theta \\in [0, \\pi]$, the pitch $h$ and the point "
             "$q$ on the axis closest to the origin for the displacement")
        st.mat("T_4", T4, 4)
        Sv = mr.se3ToVec(mr.MatrixLog6(T4))
        S4c, th4c = mr.AxisAng6(Sv)
        wh, vv = S4c[:3], S4c[3:]
        hc = float(wh @ vv)
        qc = np.cross(wh, vv)
        s5 = Doc()
        s5.p(f"Rotation part: $\\mathrm{{tr}}\\,R = {num(np.trace(T4[:3, :3]), 4)}$ gives $\\theta = {mdeg(th4c)}$ and "
             f"$\\hat{{\\omega}} = {vec(wh, 4)}$ as in (b).  Then (3.91)–(3.92):")
        s5.eq(r"v = G^{-1}(\theta)\,p,\quad G^{-1}(\theta) = \frac{1}{\theta}I - \frac{1}{2}[\omega] + "
              r"\left(\frac{1}{\theta} - \frac{1}{2}\cot\frac{\theta}{2}\right)[\omega]^{2}")
        s5.p(f"$p = {vec(T4[:3, 3], 4)}$ gives $v = {vec(vv, 4)}$ (`mr.MatrixLog6`, `mr.AxisAng6`).")
        s5.p(f"Pitch $h = \\hat{{\\omega}}^{{T}}v = {num(hc, 4)}$ m/rad; the point of the axis closest to the origin is "
             f"$q = \\hat{{\\omega}}\\times v = {vec(qc, 4)}$ m (because $v = -\\hat{{\\omega}}\\times q + h\\hat{{\\omega}}$ "
             "with $q \\perp \\hat{\\omega}$).")
        steps.append(Step("(d) Matrix logarithm of a rigid motion", s5))
        ans.update(S4=S4c, theta4=th4c, h4=hc, q4=qc)
        ad.p(f"(d) $\\mathcal{{S}} = {vec(S4c, 4)}$, $\\theta = {mdeg(th4c)}$, $h = {num(hc, 4)}$ m/rad, "
             f"$q = {vec(qc, 4)}$ m.")
    vis = dict(kind="frames", frames=frames, axis=w)
    return Problem("rotation", seed, difficulty, "Rotations and rigid motions: exp and log",
                   "Ch. 3 (eqs. 3.51, 3.58–3.61, 3.86–3.92)", st, data, steps, ans, ad, vis)


def _verify_rotation(p: Problem) -> float:
    from scipy.spatial.transform import Rotation
    d, a = p.data, p.answer
    e = np.abs(Rotation.from_rotvec(d["omega_hat"] * d["theta"]).as_matrix() - a["R"]).max()
    e = max(e, np.abs(expm(_skew(a["omega2"] * a["theta2"])) - d["R2"]).max())
    e = max(e, abs(np.linalg.norm(Rotation.from_matrix(d["R2"]).as_rotvec()) - a["theta2"]))
    if "T" in a:
        w = d["s_hat"]
        S = np.r_[w, -np.cross(w, d["q"]) + d["h"] * w]
        e = max(e, np.abs(expm(_se3(S * d["theta3"])) - a["T"]).max())
    if "S4" in a:
        e = max(e, np.abs(expm(_se3(a["S4"] * a["theta4"])) - d["T4"]).max())
        w, v = a["S4"][:3], a["S4"][3:]
        e = max(e, np.abs(-np.cross(w, a["q4"]) + a["h4"] * w - v).max())
    return float(e)


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


def _se3(V):
    out = np.zeros((4, 4))
    out[:3, :3] = _skew(V[:3])
    out[:3, 3] = V[3:]
    return out


# ------------------------------------------------------------------------------------------------
# Ch. 4 -- product of exponentials (space and body forms)
# ------------------------------------------------------------------------------------------------


def _robot_statement(st: Doc, spec: RobotSpec):
    st.p(spec.description_md() + " At the home configuration (all joint values zero) the end-effector frame {b} is")
    st.mat("M", spec.M, 3)
    hd, rows = spec.joint_table()
    st.p("The joint axes at home, expressed in the fixed frame {s}:")
    st.table(hd, rows)


def gen_fk(seed: int, difficulty: int) -> Problem:
    rng = _rng("fk", seed, difficulty)
    spec = random_robot(rng, _pick(rng, FAMILIES[difficulty]))
    th = random_theta(rng, spec, difficulty)
    S, B, M = spec.Slist, spec.Blist, spec.M
    T = mr.FKinSpace(M, S, th)
    Tb = mr.FKinBody(M, B, th)
    st = Doc()
    _robot_statement(st, spec)
    st.p(f"Joint values: {_theta_md(spec, th)}.")
    st.p("**(a)** Write the space screw axes $\\mathcal{S}_i$.  **(b)** Find the body screw axes "
         "$\\mathcal{B}_i = [\\mathrm{Ad}_{M^{-1}}]\\mathcal{S}_i$.  **(c)** Compute $T_{sb}(\\theta)$ with the space "
         "form of the PoE formula and check it with the body form.")
    steps = []
    s1 = Doc()
    s1.p("Revolute joint: $\\mathcal{S}_i = (\\hat{\\omega}_i,\\ -\\hat{\\omega}_i\\times q_i)$; prismatic joint: "
         "$\\mathcal{S}_i = (0,\\ \\hat{v}_i)$ (MR Sec. 4.1.1).")
    s1.table(["", r"$\omega_i$", r"$v_i$"], _screw_rows(S, r"\mathcal{S}"))
    steps.append(Step("(a) Space screw axes", s1))
    s2 = Doc()
    s2.p("$\\mathcal{B}_i$ is $\\mathcal{S}_i$ expressed in {b} at the home configuration:")
    s2.mat("[\\mathrm{Ad}_{M^{-1}}]", mr.Adjoint(mr.TransInv(M)), 3)
    s2.table(["", r"$\omega_i$", r"$v_i$"], _screw_rows(B, r"\mathcal{B}"))
    steps.append(Step("(b) Body screw axes", s2))
    s3 = Doc()
    s3.p("Each factor is a screw exponential (3.88), $e^{[\\mathcal{S}_i]\\theta_i}$:")
    for i in range(spec.n):
        s3.mat(f"e^{{[\\mathcal{{S}}_{{{i + 1}}}]\\theta_{{{i + 1}}}}}", mr.MatrixExp6(mr.VecTose3(S[:, i] * th[i])), 4)
    s3.p("Multiplying in order and post-multiplying by $M$, the space form (4.14) gives (`mr.FKinSpace`):")
    s3.eq(r"T_{sb}(\theta) = e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_n]\theta_n}M")
    s3.mat("T_{sb}(\\theta)", T, 4)
    steps.append(Step("(c) Space-form PoE", s3))
    s4 = Doc()
    s4.eq(r"T_{sb}(\theta) = M e^{[\mathcal{B}_1]\theta_1}\cdots e^{[\mathcal{B}_n]\theta_n}")
    s4.p(f"The body form (4.16) (`mr.FKinBody`) gives the same matrix: "
         f"$\\max|T_{{space}} - T_{{body}}| = {np.abs(T - Tb).max():.1e}$.")
    steps.append(Step("Check with the body form", s4))
    ad = Doc().mat("T_{sb}(\\theta)", T, 4).p(f"End-effector position $p = {vec(T[:3, 3], 4)}$ m.")
    return Problem("fk", seed, difficulty, "Forward kinematics with the PoE formula",
                   "Ch. 4 (eqs. 4.14, 4.16)", st, dict(M=M, joints=spec.joints, theta=th, Slist=S),
                   steps, dict(Slist=S, Blist=B, T=T), ad, _robot_visual(spec, th))


def _fk_expm(S, M, th):
    T = np.eye(4)
    for i in range(S.shape[1]):
        T = T @ expm(_se3(S[:, i] * th[i]))
    return T @ M


def _independent_S(joints) -> np.ndarray:
    cols = []
    for j in joints:
        if j["type"] == "R":
            w = np.asarray(j["w"], float)
            cols.append(np.r_[w, np.cross(np.asarray(j["q"], float), w)])   # -w x q = q x w
        else:
            cols.append(np.r_[0, 0, 0, j["v"]])
    return np.array(cols, float).T


def _verify_fk(p: Problem) -> float:
    d, a = p.data, p.answer
    S = _independent_S(d["joints"])
    e = np.abs(S - a["Slist"]).max()
    e = max(e, np.abs(_fk_expm(S, d["M"], d["theta"]) - a["T"]).max())
    Minv = np.linalg.inv(d["M"])
    # body form with an independently built adjoint (B_i = M^-1 [S_i] M)
    T = d["M"].copy()
    for i in range(S.shape[1]):
        T = T @ expm(Minv @ _se3(S[:, i]) @ d["M"] * d["theta"][i])
    return float(max(e, np.abs(T - a["T"]).max()))


# ------------------------------------------------------------------------------------------------
# Ch. 5 -- space and body Jacobians, singularities
# ------------------------------------------------------------------------------------------------


def gen_jacobian(seed: int, difficulty: int) -> Problem:
    rng = _rng("jacobian", seed, difficulty)
    spec = random_robot(rng, _pick(rng, FAMILIES[difficulty]))
    th = random_theta(rng, spec, difficulty)
    make_singular = rng.random() < 0.35
    if make_singular:
        k, v = SINGULAR_JOINT[spec.family]
        th[k] = v
    S, B, M = spec.Slist, spec.Blist, spec.M
    Js = mr.JacobianSpace(S, th)
    Jb = mr.JacobianBody(B, th)
    T = mr.FKinSpace(M, S, th)
    Jsq = Jb[spec.sing_rows, :]
    sv = np.linalg.svd(Jsq, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * max(1.0, sv[0])))
    det = float(np.linalg.det(Jsq))
    singular = rank < spec.n

    st = Doc()
    _robot_statement(st, spec)
    st.p(f"Joint values: {_theta_md(spec, th)}.")
    st.p("**(a)** Compute the space Jacobian $J_s(\\theta)$.  **(b)** Compute the body Jacobian $J_b(\\theta)$ and verify "
         "$J_b = [\\mathrm{Ad}_{T_{bs}}]J_s$.  **(c)** Using " + spec.sing_note + ", decide whether this "
         "configuration is singular.")
    steps = []
    s1 = Doc()
    s1.p("Column $i$ of the space Jacobian is $\\mathcal{S}_i$ moved by the joints before it (5.11):")
    s1.eq(r"J_{s1} = \mathcal{S}_1,\quad J_{si} = \left[\mathrm{Ad}_{e^{[\mathcal{S}_1]\theta_1}\cdots "
          r"e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}\right]\mathcal{S}_i")
    s1.table(["", r"$\omega_i$", r"$v_i$"], _screw_rows(S, r"\mathcal{S}"))
    s1.mat("J_s(\\theta)", Js, 4)
    s1.p("(`mr.JacobianSpace`).")
    steps.append(Step("(a) Space Jacobian", s1))
    s2 = Doc()
    s2.p("With $T_{sb}(\\theta)$ from the PoE formula,")
    s2.mat("T_{sb}(\\theta)", T, 4)
    s2.p("the body Jacobian is (5.22) $J_b = [\\mathrm{Ad}_{T_{bs}}]J_s$, or column by column (5.18) from the "
         "body screw axes $\\mathcal{B}_i$ (`mr.JacobianBody`):")
    s2.mat("J_b(\\theta)", Jb, 4)
    s2.p(f"Check: $\\max|J_b - [\\mathrm{{Ad}}_{{T_{{bs}}}}]J_s| = "
         f"{np.abs(Jb - mr.Adjoint(mr.TransInv(T)) @ Js).max():.1e}$.")
    steps.append(Step("(b) Body Jacobian", s2))
    s3 = Doc()
    s3.p(f"A configuration is singular when the Jacobian loses rank (MR Sec. 5.3).  Take {spec.sing_note}:")
    s3.mat("J", Jsq, 4)
    s3.p(f"$\\det J = {num(det, 5)}$, singular values $\\sigma = {vec(sv, 4)}$, rank $= {rank}$ "
         f"(of {spec.n}).")
    reason = {
        "2R": "the arm is fully stretched or folded ($\\sin\\theta_2 = 0$, $\\det = L_1L_2\\sin\\theta_2$)",
        "3R": "joints 1, 2 and 3 are collinear ($\\sin\\theta_2 = 0$)",
        "RRP": "the two revolute links are aligned ($\\sin\\theta_2 = 0$): the tip cannot move radially",
        "3Rs": "the elbow is fully stretched or folded ($\\sin\\theta_3 = 0$) or the tip lies on the base axis",
        "6R": "wrist singularity: with $\\theta_5 = 0$ the axes of joints 4 and 6 are parallel",
    }[spec.family]
    if singular and spec.family == "3Rs":      # name the case that actually occurs
        if abs(np.sin(th[2])) < 1e-9:
            reason = "the elbow is fully stretched or folded ($\\sin\\theta_3 = 0$)"
        else:
            reason = ("the tip lies on the base (joint-1) axis, so joint 1 cannot move it and the tip loses the "
                      "tangential direction")
    if singular:
        s3.p(f"**Singular**: {reason}.")
    else:
        s3.p(f"**Not singular** (rank is full).  The family's singularities occur when {reason}.")
    steps.append(Step("(c) Singularity check", s3))
    ad = Doc().mat("J_s", Js, 4).mat("J_b", Jb, 4).p(
        f"$\\det J = {num(det, 5)}$ → {'**singular**' if singular else '**not singular**'} (rank {rank}).")
    return Problem("jacobian", seed, difficulty, "Space and body Jacobians; singularity check",
                   "Ch. 5 (eqs. 5.11, 5.18, 5.22; Sec. 5.3)", st,
                   dict(M=M, joints=spec.joints, theta=th, sing_rows=spec.sing_rows), steps,
                   dict(Js=Js, Jb=Jb, det=det, rank=rank, singular=singular), ad, _robot_visual(spec, th))


def _verify_jacobian(p: Problem) -> float:
    d, a = p.data, p.answer
    S = _independent_S(d["joints"])
    th = d["theta"]
    T = _fk_expm(S, d["M"], th)
    Tinv = np.linalg.inv(T)
    h = 1e-6
    Js = np.zeros((6, len(th)))
    Jb = np.zeros((6, len(th)))
    for i in range(len(th)):
        dt = np.zeros(len(th))
        dt[i] = h
        dT = (_fk_expm(S, d["M"], th + dt) - _fk_expm(S, d["M"], th - dt)) / (2 * h)
        Vs, Vb = dT @ Tinv, Tinv @ dT
        Js[:, i] = [Vs[2, 1], Vs[0, 2], Vs[1, 0], *Vs[:3, 3]]
        Jb[:, i] = [Vb[2, 1], Vb[0, 2], Vb[1, 0], *Vb[:3, 3]]
    e = max(np.abs(Js - a["Js"]).max(), np.abs(Jb - a["Jb"]).max())
    sq = Jb[d["sing_rows"], :]
    sing = np.linalg.matrix_rank(sq, tol=1e-6) < len(th)
    return float(e + (0 if bool(sing) == bool(a["singular"]) else 1.0))


# ------------------------------------------------------------------------------------------------
# Ch. 6 -- Newton-Raphson numerical inverse kinematics (body Jacobian)
# ------------------------------------------------------------------------------------------------

EOMG, EV = 1e-3, 1e-4


def newton_raphson_table(Blist, M, Tsd, th0, eomg=EOMG, ev=EV, max_iter=30):
    """MR Sec. 6.2.2 iterations: returns rows dict(i, theta, p, Vb, nw, nv) and the final theta."""
    th = np.array(th0, float)
    rows = []
    for i in range(max_iter + 1):
        Tsb = mr.FKinBody(M, Blist, th)
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), Tsd)))
        nw, nv = np.linalg.norm(Vb[:3]), np.linalg.norm(Vb[3:])
        rows.append(dict(i=i, theta=th.copy(), p=Tsb[:3, 3].copy(), Vb=Vb, nw=nw, nv=nv))
        if nw <= eomg and nv <= ev:
            return rows, th, True
        th = th + np.linalg.pinv(mr.JacobianBody(Blist, th)) @ Vb
    return rows, th, False


def gen_ik(seed: int, difficulty: int) -> Problem:
    rng = _rng("ik", seed, difficulty)
    family = {1: "2R", 2: _pick(rng, ["3R", "3Rs"]), 3: _pick(rng, ["3Rs", "6R"])}[difficulty]
    spec = random_robot(rng, family)
    B, M = spec.Blist, spec.M
    for _ in range(200):
        thd = random_theta(rng, spec, 2)
        if family in ("2R", "3R", "3Rs"):
            k = SINGULAR_JOINT[family][0]
            if abs(np.sin(thd[k])) < 0.4:
                continue
        off = rng.integers(1, 3, spec.n) * 15 * DEG * rng.choice([-1, 1], spec.n)
        if family == "6R":
            off = rng.integers(0, 2, spec.n) * 15 * DEG * rng.choice([-1, 1], spec.n)
            if abs(np.sin(thd[4])) < 0.4:
                continue
        th0 = thd + off
        Tsd = mr.FKinBody(M, B, thd)
        rows, thf, ok = newton_raphson_table(B, M, Tsd, th0)
        if ok and 3 <= len(rows) <= 8 and np.abs(thf - thd).max() < 1e-2:
            break
    planar = spec.planar
    st = Doc()
    st.p(spec.description_md() + " Its forward kinematics in the end-effector frame uses")
    st.mat("M", M, 3)
    st.table(["", r"$\omega_i$", r"$v_i$"], _screw_rows(B, r"\mathcal{B}"))
    st.p("Find joint values that reach the goal configuration")
    st.mat("T_{sd}", Tsd, 4)
    st.p(f"with the body-Jacobian Newton–Raphson algorithm, starting from $\\theta^{{0}} = $ {_theta_txt(spec, th0)} "
         f"with tolerances $\\epsilon_{{\\omega}} = {EOMG}$ rad and $\\epsilon_v = 10^{{-4}}$ m.  Fill in an "
         "iteration table like MR Example 6.1.")
    steps = []
    r0 = rows[0]
    s1 = Doc()
    s1.p("Algorithm (MR Sec. 6.2.2): $[\\mathcal{V}_b] = \\log\\left(T_{sb}^{-1}(\\theta^{i})T_{sd}\\right)$; stop if "
         "$\\|\\omega_b\\| \\leq \\epsilon_{\\omega}$ and $\\|v_b\\| \\leq \\epsilon_v$; else "
         "$\\theta^{i+1} = \\theta^{i} + J_b^{\\dagger}(\\theta^{i})\\mathcal{V}_b$.")
    Tsb0 = mr.FKinBody(M, B, th0)
    s1.mat("T_{sb}(\\theta^{0})", Tsb0, 4)
    s1.mat("T_{bd}", mr.TransInv(Tsb0) @ Tsd, 4)
    s1.p(f"$\\mathcal{{V}}_b = {vec(r0['Vb'], 4)}$, $\\|\\omega_b\\| = {num(r0['nw'], 4)}$, "
         f"$\\|v_b\\| = {num(r0['nv'], 4)}$ → not converged.")
    J0 = mr.JacobianBody(B, th0)
    s1.mat("J_b(\\theta^{0})", J0, 4)
    dth = np.linalg.pinv(J0) @ r0["Vb"]
    s1.p(f"$\\Delta\\theta = J_b^{{\\dagger}}\\mathcal{{V}}_b = {vec(dth, 4)}$ rad, so $\\theta^{{1}} = $ "
         f"{_theta_txt(spec, th0 + dth)}.")
    steps.append(Step("First iteration in detail", s1))
    s2 = Doc()
    if planar:
        hd = ["i", "θⁱ", "(x, y) [m]", r"$\mathcal{V}_b = (\omega_{zb}, v_{xb}, v_{yb})$", r"$\|\omega_b\|$", r"$\|v_b\|$"]
        tab = [[str(r["i"]), _theta_txt(spec, r["theta"]), vec(r["p"][:2], 3), vec(r["Vb"][[2, 3, 4]], 3),
                num(r["nw"], 4), num(r["nv"], 4)] for r in rows]
        s2.p("Only $(\\omega_{zb}, v_{xb}, v_{yb})$ are shown since the motion is planar:")
    else:
        hd = ["i", "θⁱ", "p [m]", r"$\mathcal{V}_b$", r"$\|\omega_b\|$", r"$\|v_b\|$"]
        tab = [[str(r["i"]), _theta_txt(spec, r["theta"]), vec(r["p"], 3), vec(r["Vb"], 3),
                num(r["nw"], 4), num(r["nv"], 4)] for r in rows]
    s2.table(hd, tab)
    mr_th, mr_ok = mr.IKinBody(B, M, Tsd, th0, EOMG, EV)
    s2.p(f"Converged after {len(rows) - 1} iterations.  `mr.IKinBody` returns the same answer "
         f"(difference {np.abs(mr_th - thf).max():.1e} rad).")
    steps.append(Step("Iteration table", s2))
    ad = Doc().p(f"$\\theta = $ {_theta_txt(spec, thf)} after {len(rows) - 1} iterations "
                 f"(final $\\|\\omega_b\\| = {rows[-1]['nw']:.1e}$, $\\|v_b\\| = {rows[-1]['nv']:.1e}$).")
    return Problem("ik", seed, difficulty, "Newton–Raphson inverse kinematics",
                   "Ch. 6 (Sec. 6.2.2, Example 6.1)", st,
                   dict(M=M, Blist=B, Tsd=Tsd, theta0=th0, theta_goal=thd), steps,
                   dict(theta=thf, iterations=len(rows) - 1, rows=rows), ad,
                   _robot_visual(spec, thf, ghost=th0, frames={"goal": Tsd}))


def _verify_ik(p: Problem) -> float:
    """The final theta satisfies the stopping test and the tabulated twists are right (SciPy expm / logm)."""
    from scipy.linalg import logm
    d, a = p.data, p.answer
    B, M = d["Blist"], d["M"]

    def twist(th):
        Tb = M.copy()
        for i in range(B.shape[1]):
            Tb = Tb @ expm(_se3(B[:, i] * th[i]))
        L = np.real(logm(np.linalg.inv(Tb) @ d["Tsd"]))
        return np.r_[L[2, 1], L[0, 2], L[1, 0], L[:3, 3]]

    Vf = twist(a["theta"])
    e = max(0.0, np.linalg.norm(Vf[:3]) - EOMG, np.linalg.norm(Vf[3:]) - EV)
    for r in a["rows"]:
        e = max(e, np.abs(twist(r["theta"]) - r["Vb"]).max())
    return float(e)


# ------------------------------------------------------------------------------------------------
# Ch. 8 -- 2R dynamics: Lagrangian mass matrix, inverse dynamics, RNEA
# ------------------------------------------------------------------------------------------------


def point_mass_2r(L1, L2, m1, m2):
    """MR Fig. 8.1: point masses at the link tips. Returns (Mlist, Glist, Slist, g)."""
    Mlist = [mr.RpToTrans(np.eye(3), [L1, 0, 0]), mr.RpToTrans(np.eye(3), [L2, 0, 0]), np.eye(4)]
    Glist = [np.diag([0, 0, 0, m1, m1, m1]), np.diag([0, 0, 0, m2, m2, m2])]
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    return Mlist, Glist, Slist, np.array([0, -G, 0])


def lagrange_2r(L1, L2, m1, m2, th, dth, ddth, g=G):
    """MR (8.9)-(8.10), closed form -- independent of modern_robotics."""
    c1, c2, s2, c12 = np.cos(th[0]), np.cos(th[1]), np.sin(th[1]), np.cos(th[0] + th[1])
    Mm = np.array([[m1 * L1**2 + m2 * (L1**2 + 2 * L1 * L2 * c2 + L2**2), m2 * (L1 * L2 * c2 + L2**2)],
                   [m2 * (L1 * L2 * c2 + L2**2), m2 * L2**2]])
    c = np.array([-m2 * L1 * L2 * s2 * (2 * dth[0] * dth[1] + dth[1]**2), m2 * L1 * L2 * dth[0]**2 * s2])
    gv = np.array([(m1 + m2) * L1 * g * c1 + m2 * g * L2 * c12, m2 * g * L2 * c12])
    return Mm, c, gv, Mm @ ddth + c + gv


def gen_dynamics(seed: int, difficulty: int) -> Problem:
    rng = _rng("dynamics", seed, difficulty)
    L1, L2 = _pick(rng, [0.5, 0.6, 0.8, 1.0]), _pick(rng, [0.4, 0.5, 0.6, 0.8])
    m1, m2 = _pick(rng, [1.0, 1.5, 2.0, 3.0]), _pick(rng, [0.5, 1.0, 1.5, 2.0])
    th = np.array([int(rng.integers(-3, 4)) * 30, int(rng.integers(1, 6)) * 30 * _pick(rng, [-1, 1])], float) * DEG
    dth = np.array([_pick(rng, [-1.0, -0.5, 0.5, 1.0, 1.5]), _pick(rng, [-1.0, 0.5, 1.0, 2.0])])
    ddth = np.array([_pick(rng, [-2.0, -1.0, 0.5, 1.0, 2.0]), _pick(rng, [-1.0, 0.5, 1.0, 3.0])])
    Mlist, Glist, Slist, gvec = point_mass_2r(L1, L2, m1, m2)
    Mm = mr.MassMatrix(th, Mlist, Glist, Slist)
    c = mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist)
    gt = mr.GravityForces(th, gvec, Mlist, Glist, Slist)
    tau = mr.InverseDynamics(th, dth, ddth, gvec, np.zeros(6), Mlist, Glist, Slist)
    c1, c2, s2, c12 = np.cos(th[0]), np.cos(th[1]), np.sin(th[1]), np.cos(th.sum())

    st = Doc()
    st.p(f"A planar 2R arm moves in a vertical plane (MR Fig. 8.1): links of length $L_1 = {num(L1)}$ m and "
         f"$L_2 = {num(L2)}$ m with point masses $\\mathfrak{{m}}_1 = {num(m1)}$ kg and $\\mathfrak{{m}}_2 = {num(m2)}$ kg "
         f"at their tips, gravity $g = {G}$ m/s² along $-\\hat{{y}}$, zero joint friction.")
    st.p(f"State: $\\theta = $ {degs(th)}, $\\dot{{\\theta}} = {vec(dth)}$ rad/s"
         + (f", $\\ddot{{\\theta}} = {vec(ddth)}$ rad/s²." if difficulty > 1 else "."))
    if difficulty == 1:
        st.p("**(a)** Compute the mass matrix $M(\\theta)$ from the Lagrangian formulation.  **(b)** Compute the "
             "gravity torques $g(\\theta)$ and the velocity-product torques $c(\\theta, \\dot{\\theta})$.")
    elif difficulty == 2:
        st.p("Compute the joint torques $\\tau = M(\\theta)\\ddot{\\theta} + c(\\theta,\\dot{\\theta}) + g(\\theta)$ "
             "using the Lagrangian equations of motion (8.9).")
    else:
        st.p("Compute the joint torques with the recursive Newton–Euler algorithm (MR Sec. 8.3.2): list the link "
             "twists $\\mathcal{V}_i$, accelerations $\\dot{\\mathcal{V}}_i$ and wrenches $\\mathcal{F}_i$, then check "
             "with the Lagrangian form (8.9).")
    steps = []
    s1 = Doc()
    s1.p(f"$\\cos\\theta_1 = {num(c1, 4)}$, $\\cos\\theta_2 = {num(c2, 4)}$, $\\sin\\theta_2 = {num(s2, 4)}$, "
         f"$\\cos(\\theta_1+\\theta_2) = {num(c12, 4)}$.  From (8.10):")
    s1.eq(r"M_{11} = \mathfrak{m}_1L_1^{2} + \mathfrak{m}_2\left(L_1^{2} + 2L_1L_2\cos\theta_2 + L_2^{2}\right),\quad "
          r"M_{12} = M_{21} = \mathfrak{m}_2\left(L_1L_2\cos\theta_2 + L_2^{2}\right),\quad M_{22} = \mathfrak{m}_2L_2^{2}")
    s1.mat("M(\\theta)", Mm, 4)
    s1.p("(`mr.MassMatrix`; symmetric and positive definite, eigenvalues "
         f"{vec(np.linalg.eigvalsh(Mm), 3)}).")
    steps.append(Step("Mass matrix", s1))
    s2 = Doc()
    s2.eq(r"c = \left(-\mathfrak{m}_2L_1L_2\sin\theta_2\left(2\dot{\theta}_1\dot{\theta}_2 + \dot{\theta}_2^{2}\right),"
          r"\ \mathfrak{m}_2L_1L_2\dot{\theta}_1^{2}\sin\theta_2\right)")
    s2.eq(r"g(\theta) = \left((\mathfrak{m}_1+\mathfrak{m}_2)L_1g\cos\theta_1 + \mathfrak{m}_2gL_2\cos(\theta_1+\theta_2),"
          r"\ \mathfrak{m}_2gL_2\cos(\theta_1+\theta_2)\right)")
    s2.p(f"$c = {vec(c, 4)}$ N·m (`mr.VelQuadraticForces`), $g(\\theta) = {vec(gt, 4)}$ N·m (`mr.GravityForces`).")
    steps.append(Step("Velocity-product and gravity terms", s2))
    ans = dict(M=Mm, c=c, g=gt)
    ad = Doc().mat("M(\\theta)", Mm, 4).p(f"$c = {vec(c, 4)}$ N·m, $g(\\theta) = {vec(gt, 4)}$ N·m.")
    if difficulty >= 2:
        s3 = Doc()
        s3.p(f"$M\\ddot{{\\theta}} = {vec(Mm @ ddth, 4)}$, so")
        s3.eq(r"\tau = M(\theta)\ddot{\theta} + c(\theta,\dot{\theta}) + g(\theta) = " + vec(tau, 4)
              + r"\ \mathrm{N\,m}")
        s3.p("(`mr.InverseDynamics` — the Newton–Euler algorithm — gives the same vector.)")
        steps.append(Step("Joint torques", s3))
        ans["tau"] = tau
        ad = Doc().p(f"$\\tau = {vec(tau, 4)}$ N·m  ($M = $ {vec(Mm.ravel(), 4)} row-wise, $c = {vec(c, 4)}$, "
                     f"$g = {vec(gt, 4)}$).")
    if difficulty >= 3:
        s4 = Doc()
        s4.p("Link frames {1}, {2} at the point masses; {3} = {2}.  $\\mathcal{A}_i = [\\mathrm{Ad}_{M_i^{-1}}]\\mathcal{S}_i$; "
             "initial conditions $\\mathcal{V}_0 = 0$, $\\dot{\\mathcal{V}}_0 = (0, -\\mathfrak{g}) = (0,0,0,0,"
             + num(G) + ",0)$ (gravity trick), $\\mathcal{F}_3 = 0$.")
        s4.eq(r"T_{i,i-1} = e^{-[\mathcal{A}_i]\theta_i}M_{i,i-1},\quad \mathcal{V}_i = \mathrm{Ad}_{T_{i,i-1}}"
              r"(\mathcal{V}_{i-1}) + \mathcal{A}_i\dot{\theta}_i")
        s4.eq(r"\dot{\mathcal{V}}_i = \mathrm{Ad}_{T_{i,i-1}}(\dot{\mathcal{V}}_{i-1}) + \mathrm{ad}_{\mathcal{V}_i}"
              r"(\mathcal{A}_i)\dot{\theta}_i + \mathcal{A}_i\ddot{\theta}_i")
        s4.eq(r"\mathcal{F}_i = \mathrm{Ad}^{T}_{T_{i+1,i}}(\mathcal{F}_{i+1}) + \mathcal{G}_i\dot{\mathcal{V}}_i - "
              r"\mathrm{ad}^{T}_{\mathcal{V}_i}(\mathcal{G}_i\mathcal{V}_i),\quad \tau_i = \mathcal{F}_i^{T}\mathcal{A}_i")
        V, dV, F, A, Tm = rnea_trace(th, dth, ddth, gvec, Mlist, Glist, Slist)
        rows = [[str(i + 1), vec(A[i], 3), vec(V[i + 1], 3), vec(dV[i + 1], 3), vec(F[i], 3)] for i in range(2)]
        s4.table(["i", r"$\mathcal{A}_i$", r"$\mathcal{V}_i$", r"$\dot{\mathcal{V}}_i$", r"$\mathcal{F}_i$"], rows)
        tr = np.array([F[i] @ A[i] for i in range(2)])
        s4.p(f"$\\tau = (\\mathcal{{F}}_1^{{T}}\\mathcal{{A}}_1, \\mathcal{{F}}_2^{{T}}\\mathcal{{A}}_2) = {vec(tr, 4)}$ N·m, "
             f"equal to the Lagrangian result (difference {np.abs(tr - tau).max():.1e}).")
        steps.insert(0, Step("Recursive Newton–Euler (8.50)–(8.55)", s4))
        ans["rnea"] = dict(V=V, dV=dV, F=F)
    return Problem("dynamics", seed, difficulty, "Dynamics of a 2R arm: Lagrange and Newton–Euler",
                   "Ch. 8 (eqs. 8.9–8.10, 8.50–8.55)", st,
                   dict(L1=L1, L2=L2, m1=m1, m2=m2, theta=th, dtheta=dth, ddtheta=ddth), steps, ans, ad,
                   _robot_visual(_two_r_spec(L1, L2), th))


def _two_r_spec(L1, L2) -> RobotSpec:
    z = [0, 0, 1]
    return RobotSpec("planar 2R arm", "2R", [dict(type="R", w=z, q=[0, 0, 0]), dict(type="R", w=z, q=[L1, 0, 0])],
                     mr.RpToTrans(np.eye(3), [L1 + L2, 0, 0]), True, {"L_1": L1, "L_2": L2}, [3, 4], "")


def rnea_trace(th, dth, ddth, g, Mlist, Glist, Slist):
    """MR (8.50)-(8.55) step by step (same algorithm as mr.InverseDynamics, with the intermediates kept)."""
    n = len(th)
    Mi = np.eye(4)
    A, Tm = [], []
    V = [np.zeros(6)]
    dV = [np.r_[0, 0, 0, -np.asarray(g, float)]]
    for i in range(n):
        Mi = Mi @ Mlist[i]
        A.append(mr.Adjoint(mr.TransInv(Mi)) @ Slist[:, i])
        T = mr.MatrixExp6(mr.VecTose3(A[i] * -th[i])) @ mr.TransInv(Mlist[i])
        Tm.append(T)
        V.append(mr.Adjoint(T) @ V[i] + A[i] * dth[i])
        dV.append(mr.Adjoint(T) @ dV[i] + mr.ad(V[i + 1]) @ A[i] * dth[i] + A[i] * ddth[i])
    F = [None] * n
    Fn = np.zeros(6)
    Tn = mr.TransInv(Mlist[n])
    for i in range(n - 1, -1, -1):
        Fn = mr.Adjoint(Tn).T @ Fn + Glist[i] @ dV[i + 1] - mr.ad(V[i + 1]).T @ (Glist[i] @ V[i + 1])
        F[i] = Fn
        Tn = Tm[i]
    return V, dV, F, A, Tm


def _verify_dynamics(p: Problem) -> float:
    d, a = p.data, p.answer
    Mm, c, gv, tau = lagrange_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["theta"], d["dtheta"], d["ddtheta"])
    e = max(np.abs(Mm - a["M"]).max(), np.abs(c - a["c"]).max(), np.abs(gv - a["g"]).max())
    if "tau" in a:
        e = max(e, np.abs(tau - a["tau"]).max())
    return float(e)


# ------------------------------------------------------------------------------------------------
# Ch. 9 -- cubic / quintic time scaling
# ------------------------------------------------------------------------------------------------

_POLY = {3: dict(name="cubic", coeffs="a_2 = 3/T^{2},\\ a_3 = -2/T^{3}", vmax=1.5, amax=6.0,
                 v_tex=r"\dot{\theta}_{max} = \frac{3}{2T}|\Delta\theta|",
                 a_tex=r"\ddot{\theta}_{max} = \frac{6}{T^{2}}|\Delta\theta|"),
         5: dict(name="quintic", coeffs="a_3 = 10/T^{3},\\ a_4 = -15/T^{4},\\ a_5 = 6/T^{5}", vmax=15 / 8,
                 amax=10 / np.sqrt(3),
                 v_tex=r"\dot{\theta}_{max} = \frac{15}{8T}|\Delta\theta|",
                 a_tex=r"\ddot{\theta}_{max} = \frac{10}{\sqrt{3}\,T^{2}}|\Delta\theta|")}


def _scaling(method, T, t):
    return mr.CubicTimeScaling(T, t) if method == 3 else mr.QuinticTimeScaling(T, t)


def gen_trajectory(seed: int, difficulty: int) -> Problem:
    rng = _rng("trajectory", seed, difficulty)
    method = 3 if difficulty == 1 else (5 if difficulty == 2 else _pick(rng, [3, 5]))
    nj = 1 if difficulty == 1 else 2
    ths = np.array([int(rng.integers(-6, 7)) * 15 for _ in range(nj)], float)
    the = ths + np.array([int(rng.integers(2, 9)) * 15 * _pick(rng, [-1, 1]) for _ in range(nj)], float)
    T = float(_pick(rng, [1.0, 2.0, 3.0, 4.0]))
    t1 = T * _pick(rng, [0.25, 0.5, 0.75])
    P = _POLY[method]
    D = the - ths
    st = Doc()
    jt = "joint" if nj == 1 else "joints"
    st.p(f"A straight-line joint-space motion $\\theta(s) = \\theta_{{start}} + s(\\theta_{{end}} - \\theta_{{start}})$ "
         f"moves the {jt} from $\\theta_{{start}} = $ {vec(ths, 1)}° to $\\theta_{{end}} = $ {vec(the, 1)}° "
         f"with a {P['name']} time scaling $s(t)$ that starts and ends at rest"
         + (" with zero acceleration" if method == 5 else "") + ".")
    data = dict(theta_start=ths, theta_end=the, method=method)
    steps = []
    if difficulty < 3:
        st.p(f"The motion time is $T = {num(T)}$ s.  **(a)** Give the coefficients of $s(t)$.  **(b)** Evaluate "
             f"$\\theta(t)$ at $t = {num(t1)}$ s.  **(c)** Find the maximum joint speeds and accelerations.")
        data.update(T=T, t1=t1)
    else:
        vlim = float(_pick(rng, [30.0, 45.0, 60.0, 90.0]))
        alim = float(_pick(rng, [60.0, 90.0, 120.0, 180.0]))
        st.p(f"The joints are limited to $|\\dot{{\\theta}}| \\leq {num(vlim)}$ °/s and $|\\ddot{{\\theta}}| \\leq "
             f"{num(alim)}$ °/s².  **(a)** Find the minimum motion time $T$ that respects both limits.  **(b)** With that "
             "$T$, give the coefficients of $s(t)$ and the peak speed and acceleration of every joint.  "
             f"**(c)** Evaluate $\\theta(t)$ at $t = T/2$.")
        Tv = P["vmax"] * np.abs(D).max() / vlim
        Ta = np.sqrt(P["amax"] * np.abs(D).max() / alim)
        T = float(max(Tv, Ta))
        t1 = T / 2
        data.update(vlim=vlim, alim=alim)
        s0 = Doc()
        s0.p(f"Peak values scale with $|\\Delta\\theta|$; the largest is $|\\Delta\\theta|_{{max}} = {num(np.abs(D).max())}$°.")
        s0.eq(P["v_tex"] + r"\leq \dot{\theta}_{lim}\ \Rightarrow\ T \geq " + num(Tv, 4) + r"\ \mathrm{s}")
        s0.eq(P["a_tex"] + r"\leq \ddot{\theta}_{lim}\ \Rightarrow\ T \geq " + num(Ta, 4) + r"\ \mathrm{s}")
        s0.p(f"So $T_{{min}} = {num(T, 4)}$ s; the binding limit is the "
             f"{'velocity' if Tv >= Ta else 'acceleration'} limit.")
        steps.append(Step("Minimum time", s0))
    s1 = Doc()
    if method == 3:
        s1.p("Boundary conditions $s(0) = \\dot{s}(0) = 0$, $s(T) = 1$, $\\dot{s}(T) = 0$ in (9.9)–(9.10) give")
        s1.eq(r"s(t) = \frac{3}{T^{2}}t^{2} - \frac{2}{T^{3}}t^{3}")
        coef = np.array([0, 0, 3 / T**2, -2 / T**3])
    else:
        s1.p("Adding $\\ddot{s}(0) = \\ddot{s}(T) = 0$ gives the quintic (MR p. 332, Exercise 9.5)")
        s1.eq(r"s(t) = \frac{10}{T^{3}}t^{3} - \frac{15}{T^{4}}t^{4} + \frac{6}{T^{5}}t^{5}")
        coef = np.array([0, 0, 0, 10 / T**3, -15 / T**4, 6 / T**5])
    s1.p(f"With $T = {num(T, 4)}$ s: coefficients $a_0 \\ldots a_{method} = {vec(coef, 5)}$.")
    steps.append(Step("Time-scaling coefficients", s1))
    s = _scaling(method, T, t1)
    th1 = ths + s * D
    s2 = Doc()
    s2.p(f"$s({num(t1, 4)}) = {num(s, 5)}$ (`mr.{'Cubic' if method == 3 else 'Quintic'}TimeScaling`), so "
         f"$\\theta({num(t1, 4)}) = \\theta_{{start}} + s\\,\\Delta\\theta = $ {vec(th1, 3)}°.")
    steps.append(Step("Position at the given time", s2))
    vmax = P["vmax"] * np.abs(D) / T
    amax = P["amax"] * np.abs(D) / T**2
    s3 = Doc()
    s3.p("Peak speed at $t = T/2$; peak acceleration at " +
         ("$t = 0$ and $t = T$:" if method == 3 else "$t = T(3 \\pm \\sqrt{3})/6$:"))
    s3.eq(P["v_tex"] + r",\quad " + P["a_tex"])
    rows = [[str(j + 1), num(D[j], 2), num(vmax[j], 3), num(amax[j], 3)] for j in range(nj)]
    s3.table(["joint", "Δθ [°]", "max speed [°/s]", "max accel. [°/s²]"], rows)
    traj = mr.JointTrajectory(ths, the, T, 401, method)
    tt = np.linspace(0, T, 401)
    vnum = np.abs(np.gradient(traj, tt, axis=0)).max(axis=0)
    s3.p(f"Numerical check on `mr.JointTrajectory` (401 samples): max speeds {vec(vnum, 2)} °/s.")
    steps.append(Step("Peak speed and acceleration", s3))
    ans = dict(coef=coef, T=T, s_t1=s, theta_t1=th1, vmax=vmax, amax=amax)
    ad = Doc().p(f"$T = {num(T, 4)}$ s, coefficients {vec(coef, 5)}; $\\theta({num(t1, 4)}) = $ {vec(th1, 3)}°; "
                 f"max speeds {vec(vmax, 3)} °/s; max accelerations {vec(amax, 3)} °/s².")
    data.update(T=T, t1=t1)
    vis = dict(kind="plot", title="θ(t) — " + P["name"] + " time scaling", ylabel="θ [°]", t=tt,
               series=[(f"θ{j + 1}", traj[:, j]) for j in range(nj)], marker=(t1, th1))
    return Problem("trajectory", seed, difficulty, "Point-to-point time scaling",
                   "Ch. 9 (eqs. 9.9–9.13, quintic p. 332)", st, data, steps, ans, ad, vis)


def _verify_trajectory(p: Problem) -> float:
    d, a = p.data, p.answer
    T, m = a["T"], d["method"]
    # polynomial from the boundary conditions, solved as a linear system (independent of mr)
    rows, rhs = [], []
    for t, (pos, vel, acc) in ((0.0, (0, 0, 0)), (T, (1, 0, 0))):
        rows.append([t**k for k in range(m + 1)]); rhs.append(pos)
        rows.append([k * t**(k - 1) if k else 0 for k in range(m + 1)]); rhs.append(vel)
        if m == 5:
            rows.append([k * (k - 1) * t**(k - 2) if k > 1 else 0 for k in range(m + 1)]); rhs.append(acc)
    coef = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
    e = np.abs(coef - a["coef"]).max() * T**m
    tt = np.linspace(0, T, 20001)
    s = np.polyval(coef[::-1], tt)
    sd = np.polyval(np.polyder(coef[::-1]), tt)
    sdd = np.polyval(np.polyder(coef[::-1], 2), tt)
    D = np.abs(d["theta_end"] - d["theta_start"])
    e = max(e, np.abs(np.abs(sd).max() * D - a["vmax"]).max() / max(1, D.max()))
    e = max(e, np.abs(np.abs(sdd).max() * D - a["amax"]).max() / max(1, D.max()) * 1e-2)
    st = np.polyval(coef[::-1], d["t1"])
    e = max(e, np.abs(d["theta_start"] + st * (d["theta_end"] - d["theta_start"]) - a["theta_t1"]).max())
    if "vlim" in d:   # the minimum time is feasible and tight
        v, acc = np.abs(sd).max() * D.max(), np.abs(sdd).max() * D.max()
        e = max(e, max(v - d["vlim"], acc - d["alim"], 0) / d["vlim"],
                min(abs(v - d["vlim"]) / d["vlim"], abs(acc - d["alim"]) / d["alim"]) * 1e-2)
    return float(e + abs(s[-1] - 1) + abs(s[0]))


# ------------------------------------------------------------------------------------------------
# Ch. 11 -- PD / computed-torque error dynamics
# ------------------------------------------------------------------------------------------------


def second_order_response(zeta, wn, dt=None, t_end=None, e0=1.0):
    """Simulate e'' + 2 zeta wn e' + wn^2 e = 0, e(0)=e0, e'(0)=0 with mr.EulerStep (semi-implicit order:
    velocity first).  Returns t, e, measured overshoot (fraction) and 2% settling time."""
    ts_est = 4 / (zeta * wn) if zeta < 1 else 4 / (wn * (zeta - np.sqrt(zeta**2 - 1)))
    t_end = t_end or 1.6 * ts_est + 2 * np.pi / wn
    dt = dt or min(1e-3, t_end / 20000)
    n = int(t_end / dt) + 1
    e = np.empty(n)
    x, v = float(e0), 0.0
    for k in range(n):
        e[k] = x
        acc = -2 * zeta * wn * v - wn**2 * x
        v_new, _ = mr.EulerStep(np.array([v]), np.array([acc]), np.zeros(1), dt)  # v += acc*dt
        x_new, _ = mr.EulerStep(np.array([x]), v_new, np.zeros(1), dt)             # x += v_new*dt
        x, v = float(x_new[0]), float(v_new[0])
    t = np.arange(n) * dt
    overshoot = max(0.0, -e.min() / e0)
    outside = np.nonzero(np.abs(e) > 0.02 * abs(e0))[0]
    ts = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < n else (t[-1] if len(outside) else 0.0)
    return t, e, overshoot, ts


def _classify(zeta):
    if abs(zeta - 1) < 1e-9:
        return "critically damped"
    return "underdamped" if zeta < 1 else "overdamped"


def _response_facts(zeta, wn) -> tuple[dict, Doc]:
    d = Doc()
    out = dict(zeta=zeta, wn=wn, kind=_classify(zeta))
    if zeta < 1 - 1e-9:
        wd = wn * np.sqrt(1 - zeta**2)
        os_ = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
        ts = 4 / (zeta * wn)
        d.p(f"**Underdamped** ($\\zeta < 1$): $\\omega_d = \\omega_n\\sqrt{{1-\\zeta^{{2}}}} = {num(wd, 4)}$ rad/s, "
            f"overshoot $= e^{{-\\pi\\zeta/\\sqrt{{1-\\zeta^{{2}}}}}} = {num(100 * os_, 2)}$ %, 2% settling time "
            f"$t_s \\approx 4/(\\zeta\\omega_n) = {num(ts, 4)}$ s (MR pp. 414–415).")
        out.update(wd=wd, overshoot=os_, ts=ts)
    elif zeta > 1 + 1e-9:
        r = -wn * zeta + np.array([1, -1]) * wn * np.sqrt(zeta**2 - 1)
        ts = 4 / abs(r[0])
        d.p(f"**Overdamped** ($\\zeta > 1$): real roots $s_{{1,2}} = {vec(r, 4)}$; no overshoot; the slow root sets "
            f"the time constant $\\mathfrak{{t}} = 1/|s_1| = {num(1 / abs(r[0]), 4)}$ s, so $t_s \\approx 4\\mathfrak{{t}} = {num(ts, 4)}$ s.")
        out.update(overshoot=0.0, ts=ts, roots=r)
    else:
        ts = 4 / wn
        d.p(f"**Critically damped** ($\\zeta = 1$): double root $s = -\\omega_n$; no overshoot; "
            f"$t_s \\approx 4/\\omega_n = {num(ts, 4)}$ s.")
        out.update(overshoot=0.0, ts=ts)
    return out, d


def gen_control(seed: int, difficulty: int) -> Problem:
    rng = _rng("control", seed, difficulty)
    st = Doc()
    steps = []
    if difficulty == 1:
        Mj = float(_pick(rng, [0.5, 1.0, 2.0, 4.0]))
        b = float(_pick(rng, [0.0, 0.5, 1.0]))
        wn = float(_pick(rng, [2.0, 4.0, 5.0, 10.0]))
        zeta = float(_pick(rng, [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]))
        Kp = Mj * wn**2
        Kd = 2 * zeta * wn * Mj - b
        if Kd <= 0:          # friction alone would over-damp: drop it so the gains stay positive
            b, Kd = 0.0, 2 * zeta * wn * Mj
        st.p(f"A single joint with inertia $M = {num(Mj)}$ kg·m² and viscous friction $b = {num(b)}$ N·m·s/rad moves in a "
             f"horizontal plane: $\\tau = M\\ddot{{\\theta}} + b\\dot{{\\theta}}$.  A PD controller "
             f"$\\tau = K_p\\theta_e + K_d\\dot{{\\theta}}_e$ with $K_p = {num(Kp)}$ N·m/rad and $K_d = {num(Kd)}$ N·m·s/rad "
             "regulates it to a constant setpoint.  Find $\\omega_n$ and $\\zeta$, classify the error response, "
             "and give the overshoot and the 2% settling time.")
        s1 = Doc()
        s1.p("With $\\dot{\\theta}_d = \\ddot{\\theta}_d = 0$ the closed loop is (11.25)–(11.26):")
        s1.eq(r"M\ddot{\theta}_e + (b+K_d)\dot{\theta}_e + K_p\theta_e = 0,\quad \omega_n = \sqrt{\frac{K_p}{M}},"
              r"\quad \zeta = \frac{b+K_d}{2\sqrt{K_pM}}")
        zc, wc = (b + Kd) / (2 * np.sqrt(Kp * Mj)), np.sqrt(Kp / Mj)
        s1.p(f"$\\omega_n = \\sqrt{{{num(Kp)}/{num(Mj)}}} = {num(wc, 4)}$ rad/s, "
             f"$\\zeta = {num(b + Kd)}/(2\\sqrt{{{num(Kp * Mj)}}}) = {num(zc, 4)}$.")
        steps.append(Step("Natural frequency and damping ratio", s1))
        data = dict(M=Mj, b=b, Kp=Kp, Kd=Kd)
        title = "PD control of one joint: error dynamics"
        chap = "Ch. 11 (eqs. 11.8, 11.25–11.26)"
    else:
        Lp = _pick(rng, [0.5, 0.8, 1.0])
        if difficulty == 2:
            wn = float(_pick(rng, [2.0, 3.0, 4.0, 5.0, 10.0]))
            zeta = float(_pick(rng, [0.25, 0.5, 0.6, 0.8, 1.0, 1.25]))
            kp, kd = wn**2, 2 * zeta * wn
            st.p(f"A 2R arm (link lengths $L_1 = L_2 = {num(Lp)}$ m) is controlled with the computed-torque law (11.35) "
                 f"using a perfect model, $K_i = 0$ and diagonal gains $K_p = {num(kp)}\\,I$, $K_d = {num(kd)}\\,I$.  "
                 "**(a)** Write the error dynamics and find $\\omega_n$ and $\\zeta$ of each joint.  **(b)** Classify the "
                 "response; give the overshoot and 2% settling time.  **(c)** Which $K_d$ would give critical damping?")
            data = dict(kp=kp, kd=kd)
        else:
            ts_req = float(_pick(rng, [0.4, 0.5, 0.8, 1.0, 2.0]))
            zeta = float(_pick(rng, [1.0, 0.7, 0.5, 0.8]))
            wn = 4 / (zeta * ts_req)
            kp, kd = wn**2, 2 * zeta * wn
            st.p(f"A 2R arm (link lengths $L_1 = L_2 = {num(Lp)}$ m) is controlled with the computed-torque law (11.35) "
                 f"using a perfect model and $K_i = 0$.  Choose diagonal gains $K_p = k_pI$, $K_d = k_dI$ so that every "
                 f"joint error has damping ratio $\\zeta = {num(zeta)}$ and a 2% settling time of {num(ts_req)} s.  "
                 "Then predict the overshoot.")
            data = dict(ts=ts_req, zeta_req=zeta)
        s1 = Doc()
        s1.p("With a perfect model the computed-torque law (11.34)–(11.35) cancels the dynamics, leaving the decoupled "
             "linear error dynamics (11.33) with $K_i = 0$:")
        s1.eq(r"\ddot{\theta}_e + K_d\dot{\theta}_e + K_p\theta_e = 0\ \Rightarrow\ \omega_n = \sqrt{k_p},\quad "
              r"\zeta = \frac{k_d}{2\sqrt{k_p}}")
        if difficulty == 3:
            s1.p(f"Settling time $t_s \\approx 4/(\\zeta\\omega_n)$ gives $\\omega_n = 4/(\\zeta t_s) = {num(wn, 4)}$ rad/s, so "
                 f"$k_p = \\omega_n^{{2}} = {num(kp, 4)}$ and $k_d = 2\\zeta\\omega_n = {num(kd, 4)}$.")
        else:
            s1.p(f"$\\omega_n = \\sqrt{{{num(kp)}}} = {num(wn, 4)}$ rad/s, $\\zeta = {num(kd)}/(2\\cdot{num(wn, 4)}) = "
                 f"{num(zeta, 4)}$ for each joint.")
        steps.append(Step("Error dynamics", s1))
        data.update(kp=kp, kd=kd, L=Lp)
        title = "Computed-torque control: error dynamics" if difficulty == 2 else "Computed-torque gain design"
        chap = "Ch. 11 (eqs. 11.8, 11.33–11.35)"
        zc, wc = kd / (2 * np.sqrt(kp)), np.sqrt(kp)
    facts, fd = _response_facts(zc, wc)
    t, e, os_sim, ts_sim = second_order_response(zc, wc)
    fd.p(f"Simulation of the error equation with `mr.EulerStep` ($\\theta_e(0) = 1$, $\\dot{{\\theta}}_e(0) = 0$): "
         f"overshoot {num(100 * os_sim, 2)} %, 2% settling time {num(ts_sim, 3)} s "
         "(the $4/(\\zeta\\omega_n)$ rule is an envelope approximation).")
    steps.append(Step("Transient response", fd))
    ans = dict(**facts)
    ad = Doc().p(f"$\\omega_n = {num(wc, 4)}$ rad/s, $\\zeta = {num(zc, 4)}$ ({facts['kind']}); overshoot "
                 f"{num(100 * facts['overshoot'], 2)} %, $t_s \\approx {num(facts['ts'], 4)}$ s.")
    if difficulty == 2:
        kd_c = 2 * np.sqrt(kp)
        s3 = Doc().p(f"Critical damping needs $\\zeta = 1$: $k_d = 2\\sqrt{{k_p}} = {num(kd_c, 4)}$, "
                     f"giving $t_s \\approx 4/\\omega_n = {num(4 / wc, 4)}$ s.")
        steps.append(Step("(c) Critical damping", s3))
        ans["kd_critical"] = kd_c
        ad.p(f"(c) $k_d = {num(kd_c, 4)}$.")
    if difficulty == 3:
        ans.update(kp=kp, kd=kd)
        ad.p(f"Gains: $k_p = {num(kp, 4)}$, $k_d = {num(kd, 4)}$.")
    vis = dict(kind="plot", title="error response θₑ(t) (θₑ(0) = 1)", ylabel="θₑ", t=t[::max(1, len(t) // 1500)],
               series=[("θₑ", e[::max(1, len(t) // 1500)])], band=0.02)
    return Problem("control", seed, difficulty, title, chap, st, data, steps, ans, ad, vis)


def _verify_control(p: Problem) -> float:
    d, a = p.data, p.answer
    if "M" in d:
        A = np.array([[0, 1], [-d["Kp"] / d["M"], -(d["b"] + d["Kd"]) / d["M"]]])
    else:
        A = np.array([[0, 1], [-d["kp"], -d["kd"]]])
    lam = np.linalg.eigvals(A)
    wn = np.sqrt(np.real(np.prod(lam)))          # product of roots = wn^2
    zeta = -np.real(np.sum(lam)) / (2 * wn)      # sum of roots = -2 zeta wn
    e = max(abs(wn - a["wn"]), abs(zeta - a["zeta"]))
    if a["kind"] == "underdamped":
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda t, x: A @ x, (0, 3 * np.pi / np.imag(lam).max()), [1.0, 0.0], rtol=1e-10,
                        atol=1e-12, dense_output=True)
        tt = np.linspace(0, sol.t[-1], 20001)
        os_num = -sol.sol(tt)[0].min()
        e = max(e, abs(os_num - a["overshoot"]) * 1e-2)
    if "ts" in d:
        e = max(e, abs(4 / (a["zeta"] * a["wn"]) - d["ts"]))
    return float(e)


# ------------------------------------------------------------------------------------------------
# registry and public API
# ------------------------------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "rotation": dict(title="Rotations & rigid motions: exp / log (Ch. 3)", gen=gen_rotation, verify=_verify_rotation),
    "fk": dict(title="PoE forward kinematics, space & body (Ch. 4)", gen=gen_fk, verify=_verify_fk),
    "jacobian": dict(title="Space / body Jacobian & singularities (Ch. 5)", gen=gen_jacobian, verify=_verify_jacobian),
    "ik": dict(title="Newton–Raphson IK iteration table (Ch. 6)", gen=gen_ik, verify=_verify_ik),
    "dynamics": dict(title="2R dynamics: Lagrange mass matrix / RNEA torques (Ch. 8)", gen=gen_dynamics,
                     verify=_verify_dynamics),
    "trajectory": dict(title="Cubic / quintic time scaling (Ch. 9)", gen=gen_trajectory, verify=_verify_trajectory),
    "control": dict(title="PD / computed-torque error dynamics ζ, ωₙ (Ch. 11)", gen=gen_control,
                    verify=_verify_control),
}
TOPIC_KEYS = list(TOPICS)
MIXED = "mixed"


def generate(topic: str, seed: int, difficulty: int = 2) -> Problem:
    """Deterministic problem for (topic, seed, difficulty); topic "mixed" picks the topic from the seed."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be 1, 2 or 3, not {difficulty!r}")
    if topic == MIXED:
        topic = TOPIC_KEYS[int(seed) % len(TOPIC_KEYS)]
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}; choose from {TOPIC_KEYS + [MIXED]}")
    return TOPICS[topic]["gen"](int(seed), int(difficulty))


def verify(p: Problem) -> float:
    """Max discrepancy between the problem's answers and an independent re-solution."""
    return TOPICS[p.topic]["verify"](p)


def version_label(v: int) -> str:
    return chr(ord("A") + v) if v < 26 else f"V{v + 1}"


def problem_set(n: int, topic: str = MIXED, difficulty: int = 2, seed: int = 1, version: int = 0) -> list[Problem]:
    """``n`` problems.  With topic "mixed" problem k is on topic k mod 7, so every quiz version has the same
    topic per question; versions differ only in their numbers (seed offset)."""
    out = []
    for k in range(n):
        tp = TOPIC_KEYS[k % len(TOPIC_KEYS)] if topic == MIXED else topic
        s = int(seed) + 1000 * int(version) + (k // len(TOPIC_KEYS) if topic == MIXED else k)
        out.append(generate(tp, s, difficulty))
    return out


def problem_set_markdown(problems: list[Problem], title: str = "Practice problems", with_solutions: bool = False,
                         version: int | None = None) -> str:
    vtxt = f" — version {version_label(version)}" if version is not None else ""
    head = (f"# {title}{vtxt}\n\n*Modern Robotics (Lynch & Park, 2017). Generated by MR Studio; every answer is "
            "computed with the `modern_robotics` package and re-checked independently. Angles in degrees on the page, "
            "radians in formulas.*\n\n")
    body = "\n\n".join(p.markdown(i, with_solution=False) for i, p in enumerate(problems, 1))
    md = head + body
    if with_solutions:
        md += "\n\n# Solutions\n\n" + "\n\n".join(
            f"## Problem {i}. {p.title}\n\n{p.solution}" for i, p in enumerate(problems, 1))
    return md


def find_pandoc() -> str | None:
    p = shutil.which("pandoc")
    if p:
        return p
    for c in ("/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc", "/usr/bin/pandoc"):
        if os.path.exists(c):
            return c
    return None


def export_markdown(path: str, problems: list[Problem], with_solutions=False, title="Practice problems",
                    version=None) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(problem_set_markdown(problems, title, with_solutions, version))
    return path


def export_docx(path: str, problems: list[Problem], with_solutions=True, title="Practice problems",
                version=None, timeout=120) -> str:
    """Word document via pandoc (LaTeX -> native Word equations).  Raises RuntimeError without pandoc."""
    exe = find_pandoc()
    if exe is None:
        raise RuntimeError("pandoc was not found (install it, e.g. `brew install pandoc`)")
    md = problem_set_markdown(problems, title, with_solutions, version)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "set.md")
        with open(src, "w", encoding="utf-8") as f:
            f.write(md)
        r = subprocess.run([exe, src, "-f", "markdown", "-t", "docx", "-o", path],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"pandoc failed: {r.stderr.strip()[:500]}")
    return path


# ------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------


def validation_checks() -> list[Check]:
    """Each generator's answers are re-solved independently for a sample of seeds and difficulties."""
    out: list[Check] = []
    cat = "Practice problems"
    chap = {"rotation": "MR 3.2–3.3", "fk": "MR 4.1", "jacobian": "MR 5.1–5.3", "ik": "MR 6.2",
            "dynamics": "MR 8.1, 8.3", "trajectory": "MR 9.2", "control": "MR 11.2–11.4"}
    how = {"rotation": "SciPy Rotation / expm vs mr.MatrixExp3/Log3/Exp6/Log6",
           "fk": "screw axes rebuilt as q×ω, SciPy expm products (space and M⁻¹[S]M body form) vs mr.FKinSpace",
           "jacobian": "central finite differences of the PoE map (SciPy expm) vs mr.JacobianSpace/Body; rank decision",
           "ik": "final θ reaches T_sd (SciPy expm); tabulated V_b recomputed with SciPy logm",
           "dynamics": "closed-form Lagrangian (8.9)-(8.10) vs mr.MassMatrix / VelQuadraticForces / GravityForces / "
                       "InverseDynamics",
           "trajectory": "polynomial re-solved from its boundary conditions; peaks by dense sampling vs mr time scaling",
           "control": "ζ, ωn from the eigenvalues of the error-dynamics matrix; overshoot from solve_ivp"}
    for t in TOPIC_KEYS:
        err, n = 0.0, 0
        for d in (1, 2, 3):
            for s in (0, 1, 2):
                err = max(err, verify(generate(t, s, d)))
                n += 1
        out.append(Check(f"Practice problems: {t} answers re-solved independently ({n} problems)", chap[t],
                         err, 1e-6, how[t], cat))
    a, b = generate(MIXED, 12345, 2), generate(MIXED, 12345, 2)
    out.append(Check("Practice problems: same seed -> identical problem and solution", "—",
                     0.0 if (a.statement == b.statement and a.solution == b.solution) else 1.0, 0.0,
                     "two independent generate() calls compared as Markdown", cat))
    # computed torque with a perfect model produces exactly the commanded error dynamics (11.33)-(11.35)
    Mlist, Glist, Slist, g = point_mass_2r(0.8, 0.6, 2.0, 1.0)
    rng = np.random.default_rng(11)
    kp, kd = 25.0, 7.0
    th, dth = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    thd, dthd, ddthd = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    tau = mr.ComputedTorque(th, dth, np.zeros(2), g, Mlist, Glist, Slist, thd, dthd, ddthd, kp, 0.0, kd)
    ddth = mr.ForwardDynamics(th, dth, tau, g, np.zeros(6), Mlist, Glist, Slist)
    ee, ede, edde = thd - th, dthd - dth, ddthd - ddth
    out.append(Check("Practice problems: mr.ComputedTorque + mr.ForwardDynamics give θ̈e + Kd θ̇e + Kp θe = 0",
                     "MR 11.4.2.2", float(np.abs(edde + kd * ede + kp * ee).max()), 1e-9,
                     "2R point-mass arm, random state; basis of the Ch. 11 problems", cat))
    return out
