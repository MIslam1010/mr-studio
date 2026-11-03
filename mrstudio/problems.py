"""Randomised practice problems with step-by-step worked solutions (Modern Robotics, Ch. 3-13).

Every generator is deterministic in ``(topic, seed, difficulty)`` and returns a :class:`Problem`:

* ``statement``  -- Markdown with LaTeX (pandoc-ready; the GUI renders the same blocks),
* ``data``       -- the numeric inputs (NumPy arrays / floats),
* ``steps``      -- the worked solution, one :class:`Step` per stage,
* ``answer``     -- the final numbers (dict) and ``final_answer`` (Markdown).

All answers are computed with Lynch & Park's ``modern_robotics`` package (``mr``).  Each topic also
has an *independent* re-solver (SciPy matrix exponentials, finite differences, closed-form
Lagrangian, exact quadrature over a solid, KKT solves, boundary-value solve of the polynomial,
eigenvalues of the error dynamics) used by :func:`verify`, :func:`validation_checks` and the test-suite.

Topics (MR sections, printed pages of the Dec-2019 updated first edition):

========== ====================================================================================
rotation   Ch. 3  Rodrigues (3.51) p. 82, log of a rotation (3.58)-(3.61) pp. 85-86,
                  screw axis p. 110, exp/log on SE(3) (3.86)-(3.92) pp. 103-104
fk         Ch. 4  space / body PoE (4.14) p. 140, (4.16) p. 147
jacobian   Ch. 5  J_s (5.11) p. 178, J_b (5.18) p. 183, J_b = [Ad_Tbs] J_s (5.22) p. 185,
                  singularities Sec. 5.3 p. 189
statics    Ch. 5  tau = J^T F for a 2R / 3R arm, both directions (Sec. 5.2); manipulability and force
                  ellipses of J_v J_v^T, isotropy mu_1 (Sec. 5.4)
ik         Ch. 6  Newton-Raphson with the body Jacobian, Sec. 6.2.2 p. 228, Example 6.1 pp. 229-230
inertia    Ch. 8  box / cylinder principal inertias, Steiner, rotated inertia (Sec. 8.2.1), spatial
                  inertia G_b (8.32), G_a = [Ad_Tba]^T G_b [Ad_Tba] (8.42), kinetic energy
dynamics   Ch. 8  2R Lagrangian (8.9)-(8.10) p. 273, Newton-Euler RNEA (8.50)-(8.55) pp. 292-293
fwd_dynamics Ch. 8 2R forward dynamics M^-1 (tau - c - g) (Sec. 8.5), explicit / semi-implicit Euler step
task_space Ch. 8  Lambda = J^-T M J^-1 (8.90), felt mass along a direction, reflected rotor inertia
                  G^2 I_rotor (Sec. 8.9.2)
constrained Ch. 8 Pfaffian constraint A(theta) dtheta = 0, Lagrange multiplier, constraint power (Sec. 8.7)
trajectory Ch. 9  cubic (9.9)-(9.13) pp. 331-332, quintic p. 332 (Fig. 9.4 p. 333)
control    Ch. 11 second-order error dynamics (11.8) p. 412, overshoot / settling pp. 414-415,
                  PD (11.25)-(11.26) p. 425, computed torque (11.33)-(11.35) p. 431
force_control Ch. 11 1-DOF impedance M x'' + B x' + K x = f_ext: deflection f/K, zeta, omega_n, rendered
                  stiffness (Sec. 11.7); PI force control against a stiff surface, steady-state error and the
                  Routh-Hurwitz limit on k_fi (Sec. 11.5)
grasp_closure Ch. 12 wrench matrix of contact normals / friction-cone edges, first-order form closure
                  (Sec. 12.1.7), force closure and the minimum friction coefficient (Sec. 12.2.3); uses grasping.py
mobile_kin Ch. 13 diff-drive wheel speeds <-> (v, omega) (Sec. 13.3), H(0) of three omniwheels (13.9) and four
                  mecanum wheels (13.10), wheel speeds for a body twist, odometry with F = H^+(0) (Sec. 13.4);
                  uses mobile.py
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

from . import grasping as GR
from . import mobile as MB
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
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, _RNG_ID[topic], int(difficulty)])


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
    e = max(e, np.abs(np.abs(sdd).max() * D - a["amax"]).max() / max(1, D.max()))
    st = np.polyval(coef[::-1], d["t1"])
    e = max(e, np.abs(d["theta_start"] + st * (d["theta_end"] - d["theta_start"]) - a["theta_t1"]).max())
    if "vlim" in d:   # the minimum time is feasible and tight
        v, acc = np.abs(sd).max() * D.max(), np.abs(sdd).max() * D.max()
        e = max(e, max(v - d["vlim"], acc - d["alim"], 0) / d["vlim"],
                min(abs(v - d["vlim"]) / d["vlim"], abs(acc - d["alim"]) / d["alim"]))
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


def _step_error(zeta, wn, t):
    """Closed-form solution of e'' + 2 zeta wn e' + wn^2 e = 0, e(0) = 1, e'(0) = 0 (MR Sec. 11.2.2.1)."""
    t = np.asarray(t, float)
    if abs(zeta - 1) < 1e-9:
        return (1 + wn * t) * np.exp(-wn * t)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        return np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))
    r = np.sqrt(zeta**2 - 1)
    s1, s2 = -wn * (zeta - r), -wn * (zeta + r)
    return (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)


def exact_settling_time(zeta, wn, band: float = 0.02) -> float:
    """Exact 2 % settling time: the last time |e(t)| = band for the free response e(0) = 1, e'(0) = 0
    (MR's t_s ~ 4/(zeta wn) is a rule of thumb from the envelope e^{-zeta wn t})."""
    from scipy.optimize import brentq
    if zeta < 1 - 1e-9:      # beyond the envelope e^{-zeta wn t}/sqrt(1 - zeta^2) < band nothing can exceed it
        t_end = np.log(1 / (band * np.sqrt(1 - zeta**2))) / (zeta * wn) * 1.001
    else:
        t_end = 1.0 / wn
        while abs(float(_step_error(zeta, wn, t_end))) > band:
            t_end *= 2
    tt = np.linspace(0.0, t_end, 200001)
    out = np.nonzero(np.abs(_step_error(zeta, wn, tt)) > band)[0]
    if len(out) == 0:
        return 0.0
    k = out[-1]
    if k + 1 >= len(tt):
        return float(tt[-1])
    f = lambda x: abs(float(_step_error(zeta, wn, x))) - band
    return float(brentq(f, tt[k], tt[k + 1], xtol=1e-14, rtol=1e-14))


def _classify(zeta):
    if abs(zeta - 1) < 1e-9:
        return "critically damped"
    return "underdamped" if zeta < 1 else "overdamped"


def _response_facts(zeta, wn) -> tuple[dict, Doc]:
    """Regime, overshoot and settling time.  ``ts`` is the textbook rule of thumb (MR pp. 414-415:
    t_s ~ 4/(zeta wn); overdamped: 4 time constants of the slow root), ``ts_exact`` the exact 2 % time."""
    d = Doc()
    out = dict(zeta=zeta, wn=wn, kind=_classify(zeta))
    ts_ex = exact_settling_time(zeta, wn)
    if zeta < 1 - 1e-9:
        wd = wn * np.sqrt(1 - zeta**2)
        os_ = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
        ts = 4 / (zeta * wn)
        d.p(f"**Underdamped** ($\\zeta < 1$): $\\omega_d = \\omega_n\\sqrt{{1-\\zeta^{{2}}}} = {num(wd, 4)}$ rad/s, "
            f"overshoot $= e^{{-\\pi\\zeta/\\sqrt{{1-\\zeta^{{2}}}}}} = {num(100 * os_, 2)}$ %.  MR rule of thumb "
            f"$t_s \\approx 4/(\\zeta\\omega_n) = {num(ts, 4)}$ s (MR pp. 414–415); exact 2% settling time "
            f"(last time $|\\theta_e| = 0.02$) $= {num(ts_ex, 4)}$ s.")
        out.update(wd=wd, overshoot=os_, ts=ts)
    elif zeta > 1 + 1e-9:
        r = -wn * zeta + np.array([1, -1]) * wn * np.sqrt(zeta**2 - 1)
        ts = 4 / abs(r[0])
        d.p(f"**Overdamped** ($\\zeta > 1$): real roots $s_{{1,2}} = {vec(r, 4)}$; no overshoot; the slow root sets "
            f"the time constant $\\mathfrak{{t}} = 1/|s_1| = {num(1 / abs(r[0]), 4)}$ s, so the rule of thumb gives "
            f"$t_s \\approx 4\\mathfrak{{t}} = {num(ts, 4)}$ s; exact 2% settling time $= {num(ts_ex, 4)}$ s.")
        out.update(overshoot=0.0, ts=ts, roots=r)
    else:
        ts = 4 / wn
        d.p(f"**Critically damped** ($\\zeta = 1$): double root $s = -\\omega_n$; no overshoot.  MR rule of thumb "
            f"$t_s \\approx 4/(\\zeta\\omega_n) = {num(ts, 4)}$ s; exact 2% settling time $= {num(ts_ex, 4)}$ s "
            "(the factor $(1+\\omega_n t)$ makes the rule optimistic here).")
        out.update(overshoot=0.0, ts=ts)
    out["ts_exact"] = ts_ex
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
            # zeta = 1 avoided: there the 4/(zeta wn) rule is ~30 % off the exact settling time
            zeta = float(_pick(rng, [0.9, 0.7, 0.5, 0.8]))
            wn_rule = 4 / (zeta * ts_req)
            tau1 = exact_settling_time(zeta, 1.0)           # settling time scales as 1/wn
            wn = tau1 / ts_req                              # design to the exact 2 % settling time
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
            s1.p(f"MR rule of thumb $t_s \\approx 4/(\\zeta\\omega_n)$ gives a first estimate "
                 f"$\\omega_n \\approx 4/(\\zeta t_s) = {num(wn_rule, 4)}$ rad/s, but with it the exact 2% settling time "
                 f"would be {num(exact_settling_time(zeta, wn_rule), 4)} s.  The exact settling time scales as "
                 f"$1/\\omega_n$: for $\\zeta = {num(zeta)}$ it is $t_s = {num(tau1, 4)}/\\omega_n$, so the spec is met "
                 f"exactly with $\\omega_n = {num(tau1, 4)}/t_s = {num(wn, 4)}$ rad/s: "
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
    fd.p(f"Simulation of the error equation with `mr.EulerStep` ($\\theta_e(0) = 1$, $\\dot{{\\theta}}_e(0) = 0$, "
         f"1 ms steps): overshoot {num(100 * os_sim, 2)} %, 2% settling time {num(ts_sim, 3)} s — the exact value "
         f"{num(facts['ts_exact'], 3)} s up to the step size; the $4/(\\zeta\\omega_n)$ rule of thumb "
         f"({num(facts['ts'], 3)} s) is an envelope approximation.")
    steps.append(Step("Transient response", fd))
    ans = dict(**facts)
    ad = Doc().p(f"$\\omega_n = {num(wc, 4)}$ rad/s, $\\zeta = {num(zc, 4)}$ ({facts['kind']}); overshoot "
                 f"{num(100 * facts['overshoot'], 2)} %; settling time: MR rule of thumb $t_s \\approx "
                 f"{num(facts['ts'], 4)}$ s, exact 2% value {num(facts['ts_exact'], 4)} s.")
    if difficulty == 2:
        kd_c = 2 * np.sqrt(kp)
        s3 = Doc().p(f"Critical damping needs $\\zeta = 1$: $k_d = 2\\sqrt{{k_p}} = {num(kd_c, 4)}$, "
                     f"giving $t_s \\approx 4/\\omega_n = {num(4 / wc, 4)}$ s by the rule of thumb "
                     f"(exact 2% settling time {num(exact_settling_time(1.0, wc), 4)} s).")
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
        e = max(e, abs(os_num - a["overshoot"]))
    # damping-regime label from the eigenvalues (complex: under-, double: critically, distinct real: over-damped)
    disc = (np.trace(A))**2 - 4 * np.linalg.det(A)
    kind = "critically damped" if abs(disc) < 1e-9 * max(1.0, np.trace(A)**2) else (
        "underdamped" if disc < 0 else "overdamped")
    e = max(e, float(kind != a["kind"]))
    # rule-of-thumb settling time (MR pp. 414-415) and the exact 2 % settling time by direct integration
    if a["kind"] == "overdamped":
        e = max(e, abs(4 / np.abs(np.real(lam)).min() - a["ts"]))
    else:
        e = max(e, abs(4 / (zeta * wn) - a["ts"]))
    from scipy.integrate import solve_ivp
    t_hi = 3 * a["ts_exact"] + 1.0 / wn
    sol = solve_ivp(lambda t, x: A @ x, (0, t_hi), [1.0, 0.0], rtol=1e-12, atol=1e-14, dense_output=True)
    tt = np.linspace(0, t_hi, 400001)
    out = np.nonzero(np.abs(sol.sol(tt)[0]) > 0.02)[0]
    ts_num = tt[out[-1]] if len(out) else 0.0
    e = max(e, max(0.0, abs(ts_num - a["ts_exact"]) - (tt[1] - tt[0])))
    if "ts" in d:      # design problem: the gains meet the specified (exact) settling time and zeta
        e = max(e, abs(a["ts_exact"] - d["ts"]), abs(zeta - d["zeta_req"]))
    if "kd_critical" in a:
        e = max(e, abs(a["kd_critical"] / (2 * np.sqrt(d["kp"])) - 1.0))
    return float(e)


# ------------------------------------------------------------------------------------------------
# Ch. 8 -- rigid-body inertia, forward dynamics, task-space mass matrix, constrained dynamics
# ------------------------------------------------------------------------------------------------


def arm_2r(L1, L2, m1, m2, rods: bool = False):
    """2R arm in a vertical plane (gravity along -y_s).  ``rods=False``: point masses at the link tips (as
    :func:`point_mass_2r`); ``rods=True``: uniform slender rods (centre of mass mid-link, centroidal inertia
    mL²/12 about the axis parallel to the joints).  Returns (Mlist, Glist, Slist, g, params) where ``params``
    feeds the closed-form :func:`lagrange_2r_general`."""
    if not rods:
        Mlist, Glist, Slist, g = point_mass_2r(L1, L2, m1, m2)
        return Mlist, Glist, Slist, g, dict(L1=L1, lc1=L1, lc2=L2, m1=m1, m2=m2, I1=0.0, I2=0.0)
    I1, I2 = m1 * L1**2 / 12, m2 * L2**2 / 12
    Mlist = [mr.RpToTrans(np.eye(3), [L1 / 2, 0, 0]), mr.RpToTrans(np.eye(3), [(L1 + L2) / 2, 0, 0]),
             mr.RpToTrans(np.eye(3), [L2 / 2, 0, 0])]
    Glist = [np.diag([0, I1, I1, m1, m1, m1]), np.diag([0, I2, I2, m2, m2, m2])]
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    return (Mlist, Glist, Slist, np.array([0, -G, 0]),
            dict(L1=L1, lc1=L1 / 2, lc2=L2 / 2, m1=m1, m2=m2, I1=I1, I2=I2))


def lagrange_2r_general(L1, lc1, lc2, m1, m2, I1, I2, th, dth, g=G):
    """Closed-form 2R dynamics (centres of mass at lc1, lc2 from the joints, centroidal inertias I1, I2 about
    the joint-parallel axes, gravity along -y).  Independent of modern_robotics.  Returns (M, c, g(theta))."""
    c1, c2, s2, c12 = np.cos(th[0]), np.cos(th[1]), np.sin(th[1]), np.cos(th[0] + th[1])
    Mm = np.array([[m1 * lc1**2 + I1 + m2 * (L1**2 + lc2**2 + 2 * L1 * lc2 * c2) + I2,
                    m2 * (lc2**2 + L1 * lc2 * c2) + I2],
                   [m2 * (lc2**2 + L1 * lc2 * c2) + I2, m2 * lc2**2 + I2]])
    a = m2 * L1 * lc2 * s2
    c = np.array([-a * (2 * dth[0] * dth[1] + dth[1]**2), a * dth[0]**2])
    gv = np.array([(m1 * lc1 + m2 * L1) * g * c1 + m2 * lc2 * g * c12, m2 * lc2 * g * c12])
    return Mm, c, gv


def _tip_jacobian_mr(L1, L2, th) -> np.ndarray:
    """2x2 Jacobian of the tip's (x, y) velocity in {s}: rotate the linear rows of mr.JacobianBody."""
    spec = _two_r_spec(L1, L2)
    Tsb = mr.FKinBody(spec.M, spec.Blist, th)
    return (Tsb[:3, :3] @ mr.JacobianBody(spec.Blist, th)[3:, :])[:2]


def _tip_jacobian_closed(L1, L2, th) -> np.ndarray:
    s1, c1, s12, c12 = np.sin(th[0]), np.cos(th[0]), np.sin(th.sum()), np.cos(th.sum())
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def _arm_statement(L1, L2, m1, m2, rods: bool, gravity: bool = True) -> str:
    plane = "a vertical plane (gravity $g = 9.81$ m/s² along $-\\hat{y}_s$)" if gravity else "a horizontal plane"
    if rods:
        return (f"A planar 2R arm moves in {plane}.  Its links are uniform slender rods: $L_1 = {num(L1)}$ m, "
                f"$\\mathfrak{{m}}_1 = {num(m1)}$ kg and $L_2 = {num(L2)}$ m, $\\mathfrak{{m}}_2 = {num(m2)}$ kg "
                "(centre of mass mid-link, centroidal inertia $\\mathfrak{m}L^{2}/12$ about the joint-parallel axis); "
                "no joint friction.")
    return (f"A planar 2R arm moves in {plane}.  Links of length $L_1 = {num(L1)}$ m and $L_2 = {num(L2)}$ m carry "
            f"point masses $\\mathfrak{{m}}_1 = {num(m1)}$ kg and $\\mathfrak{{m}}_2 = {num(m2)}$ kg at their tips "
            "(massless links); no joint friction.")


def _mass_matrix_doc(doc: Doc, rods: bool, prm: dict, Mm):
    if rods:
        doc.p(f"Rod links: $l_{{c1}} = L_1/2 = {num(prm['lc1'])}$ m, $l_{{c2}} = L_2/2 = {num(prm['lc2'])}$ m, "
              f"$\\mathcal{{I}}_1 = {num(prm['I1'], 5)}$ kg·m², $\\mathcal{{I}}_2 = {num(prm['I2'], 5)}$ kg·m²; the "
              "Lagrangian gives")
        doc.eq(r"M_{11} = \mathfrak{m}_1l_{c1}^{2} + \mathcal{I}_1 + \mathfrak{m}_2\left(L_1^{2} + l_{c2}^{2} + "
               r"2L_1l_{c2}\cos\theta_2\right) + \mathcal{I}_2,\quad M_{12} = \mathfrak{m}_2\left(l_{c2}^{2} + "
               r"L_1l_{c2}\cos\theta_2\right) + \mathcal{I}_2,\quad M_{22} = \mathfrak{m}_2l_{c2}^{2} + \mathcal{I}_2")
    else:
        doc.p("Point masses at the link tips, from (8.10):")
        doc.eq(r"M_{11} = \mathfrak{m}_1L_1^{2} + \mathfrak{m}_2\left(L_1^{2} + 2L_1L_2\cos\theta_2 + L_2^{2}\right),"
               r"\quad M_{12} = \mathfrak{m}_2\left(L_1L_2\cos\theta_2 + L_2^{2}\right),\quad M_{22} = "
               r"\mathfrak{m}_2L_2^{2}")
    doc.mat("M(\\theta)", Mm, 4)
    doc.p("(`mr.MassMatrix`).")


def _random_2r_arm(rng):
    L1, L2 = _pick(rng, [0.5, 0.6, 0.8, 1.0]), _pick(rng, [0.4, 0.5, 0.6, 0.8])
    m1, m2 = _pick(rng, [1.0, 1.5, 2.0, 3.0]), _pick(rng, [0.5, 1.0, 1.5, 2.0])
    th = np.array([int(rng.integers(-3, 4)) * 30, int(rng.integers(1, 6)) * 30 * _pick(rng, [-1, 1])], float) * DEG
    return float(L1), float(L2), float(m1), float(m2), th


# -- rigid-body inertia ----------------------------------------------------------------------------

_SIDES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_OMEGAS = [(0, 0, 1), (0, 0, 2), (1, 0, 0), (0, 2, 0), (1, 1, 0), (0, 1, 1), (1, 0, 2), (2, 0, -1), (-1, 2, 0)]
_VELS = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 1), (0.2, 0.4, 0), (1, 0, 0.5), (0, -0.3, 0.4)]


def _shape_inertia(shape: str, m: float, dims) -> np.ndarray:
    if shape == "box":
        a, b, c = dims
        return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])
    r, h = dims
    ixx = m * (3 * r * r + h * h) / 12
    return np.diag([ixx, ixx, m * r * r / 2])


def _shape_quadrature(shape: str, m: float, dims):
    """Mass points and weights that integrate any polynomial of degree <= 2 (3 radially) over the solid exactly:
    tensor Gauss-Legendre for the box; Gauss-Legendre in r (with the r dr Jacobian), 8 equal angles and
    Gauss-Legendre in z for the cylinder.  Used by the independent inertia re-solve."""
    g2 = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    if shape == "box":
        a, b, c = dims
        pts = np.array([[a / 2 * x, b / 2 * y, c / 2 * z] for x in g2 for y in g2 for z in g2])
        return pts, np.full(len(pts), m / len(pts))
    R, h = dims
    rho = 1.0 / (np.pi * R * R * h)
    phis = 2 * np.pi * np.arange(8) / 8
    pts, w = [], []
    for r in R / 2 * (1 + g2):
        for ph in phis:
            for z in h / 2 * g2:
                pts.append([r * np.cos(ph), r * np.sin(ph), z])
                w.append(m * rho * (R / 2) * r * (2 * np.pi / 8) * (h / 2))
    return np.array(pts), np.array(w)


def _shape_md(shape: str, m: float, dims) -> str:
    if shape == "box":
        a, b, c = dims
        return (f"a solid rectangular block of mass $\\mathfrak{{m}} = {num(m)}$ kg with side lengths "
                f"$a = {num(a)}$ m, $b = {num(b)}$ m, $c = {num(c)}$ m along $\\hat{{x}}_b$, $\\hat{{y}}_b$, "
                "$\\hat{z}_b$")
    r, h = dims
    return (f"a solid circular cylinder of mass $\\mathfrak{{m}} = {num(m)}$ kg, radius $r = {num(r)}$ m and "
            f"height $h = {num(h)}$ m, its axis along $\\hat{{z}}_b$")


def _spatial(I, m) -> np.ndarray:
    Gm = np.zeros((6, 6))
    Gm[:3, :3] = I
    Gm[3:, 3:] = m * np.eye(3)
    return Gm


def gen_inertia(seed: int, difficulty: int) -> Problem:
    rng = _rng("inertia", seed, difficulty)
    shape = _pick(rng, ["box", "cylinder"])
    m = float(_pick(rng, [1.0, 2.0, 3.0, 4.0, 6.0]))
    if shape == "box":
        dims = tuple(float(_pick(rng, _SIDES)) for _ in range(3))
    else:
        dims = (float(_pick(rng, [0.05, 0.1, 0.15, 0.2])), float(_pick(rng, [0.2, 0.3, 0.4, 0.6])))
    w = np.array(_pick(rng, _OMEGAS), float)
    v = np.array(_pick(rng, _VELS), float)
    V = np.r_[w, v]
    R, p = np.eye(3), np.zeros(3)
    if difficulty >= 2:
        p = np.array([_pick(rng, [-0.3, -0.2, 0.2, 0.3, 0.4]), _pick(rng, [0.0, -0.1, 0.1, 0.2]),
                      _pick(rng, [0.0, 0.1, 0.2, -0.2])])
    ax, ang = None, 0.0
    if difficulty == 3:
        ax = _rand_axis(rng, bool(rng.random() < 0.5))
        ang = int(rng.integers(1, 6)) * 30 * DEG
        R = mr.MatrixExp3(mr.VecToso3(ax * ang))
    Ib = _shape_inertia(shape, m, dims)
    Gb = _spatial(Ib, m)

    st = Doc()
    st.p(f"A rigid body is {_shape_md(shape, m, dims)}.  Frame {{b}} sits at its centre of mass with axes along "
         "the symmetry axes.")
    if difficulty == 1:
        st.p("**(a)** Find the principal moments of inertia and the rotational inertia matrix $\\mathcal{I}_b$.  "
             "**(b)** Write the $6\\times 6$ spatial inertia matrix $\\mathcal{G}_b$.  **(c)** The body moves with "
             f"twist $\\mathcal{{V}}_b = (\\omega_b, v_b)$, $\\omega_b = {vec(w)}$ rad/s, $v_b = {vec(v)}$ m/s "
             "(expressed in {b}).  Compute its kinetic energy.")
    else:
        where = (f"has the same orientation as {{a}}" if difficulty == 2 else
                 f"is rotated relative to {{a}} by $R_{{ab}} = \\mathrm{{Rot}}(\\hat{{\\omega}}, \\theta)$ with "
                 f"$\\hat{{\\omega}} = {_axis_md(ax)}$, $\\theta = {mdeg(ang)}$")
        st.p(f"A second frame {{a}} is attached to the same body; frame {{b}} {where}, and the origin of {{b}} "
             f"(the centre of mass) is at $p = {vec(p)}$ m in {{a}} coordinates.")
        parts = ["**(a)** Find $\\mathcal{I}_b$ and $\\mathcal{G}_b$."]
        if difficulty == 3:
            parts.append("**(b)** Express the centroidal inertia in axes parallel to {a}: "
                         "$\\mathcal{I}_c = R_{ab}\\mathcal{I}_bR_{ab}^{T}$.")
        k = "c" if difficulty == 3 else "b"
        parts.append(f"**({k})** Use Steiner's theorem to find the rotational inertia $\\mathcal{{I}}_a$ about the "
                     "origin of {a} (axes of {a}).")
        k2 = "d" if difficulty == 3 else "c"
        parts.append(f"**({k2})** Find the spatial inertia $\\mathcal{{G}}_a$ and the kinetic energy for the twist "
                     f"$\\mathcal{{V}}_a = (\\omega_a, v_a)$, $\\omega_a = {vec(w)}$ rad/s, $v_a = {vec(v)}$ m/s "
                     "(expressed in {a}).")
        st.p("  ".join(parts))

    steps = []
    s1 = Doc()
    if shape == "box":
        s1.p("For a uniform rectangular block about its centre of mass the symmetry axes are principal axes and")
        s1.eq(r"\mathcal{I}_{xx} = \frac{\mathfrak{m}}{12}(b^{2}+c^{2}),\quad \mathcal{I}_{yy} = \frac{\mathfrak{m}}{12}"
              r"(a^{2}+c^{2}),\quad \mathcal{I}_{zz} = \frac{\mathfrak{m}}{12}(a^{2}+b^{2})")
    else:
        s1.p("For a uniform solid cylinder about its centre of mass the axis and any two perpendicular diameters are "
             "principal axes and")
        s1.eq(r"\mathcal{I}_{xx} = \mathcal{I}_{yy} = \frac{\mathfrak{m}}{12}(3r^{2}+h^{2}),\quad "
              r"\mathcal{I}_{zz} = \frac{\mathfrak{m}r^{2}}{2}")
    s1.p(f"Principal moments $({num(Ib[0, 0], 5)}, {num(Ib[1, 1], 5)}, {num(Ib[2, 2], 5)})$ kg·m², so "
         "$\\mathcal{I}_b$ is diagonal:")
    s1.mat("\\mathcal{I}_b", Ib, 5)
    steps.append(Step("Principal inertias at the centre of mass", s1))
    s2 = Doc()
    s2.p("In the centre-of-mass frame the spatial inertia (8.32) is block-diagonal: $\\mathcal{I}_b$ in the "
         "upper-left $3\\times 3$ block, $\\mathfrak{m}I$ in the lower-right one, zero coupling blocks.")
    s2.mat("\\mathcal{G}_b", Gb, 5)
    steps.append(Step("Spatial inertia in {b}", s2))
    ans = dict(Ib=Ib, Gb=Gb)
    if difficulty == 1:
        KE = 0.5 * V @ Gb @ V
        s3 = Doc()
        s3.eq(r"K = \frac{1}{2}\mathcal{V}_b^{T}\mathcal{G}_b\mathcal{V}_b = \frac{1}{2}\omega_b^{T}\mathcal{I}_b"
              r"\omega_b + \frac{1}{2}\mathfrak{m}v_b^{T}v_b")
        rot, tr = 0.5 * w @ Ib @ w, 0.5 * m * v @ v
        s3.p(f"Rotational part ${num(rot, 6)}$ J, translational part ${num(tr, 6)}$ J, so $K = {num(KE, 6)}$ J.  "
             "(The split is valid because {b} is at the centre of mass.)")
        steps.append(Step("Kinetic energy", s3))
        ans["KE"] = KE
        ad = Doc().p(f"Principal moments $({num(Ib[0, 0], 5)}, {num(Ib[1, 1], 5)}, {num(Ib[2, 2], 5)})$ kg·m², "
                     f"$\\mathcal{{G}}_b = \\mathrm{{diag}}({', '.join(num(x, 5) for x in np.diag(Gb))})$, "
                     f"$K = {num(KE, 6)}$ J.")
        chap = "Ch. 8 (Sec. 8.2, eq. 8.32)"
    else:
        Ic = R @ Ib @ R.T
        Ia = Ic + m * (p @ p * np.eye(3) - np.outer(p, p))
        Tab = mr.RpToTrans(R, p)
        Ad = mr.Adjoint(mr.TransInv(Tab))
        Ga = Ad.T @ Gb @ Ad
        KE = 0.5 * V @ Ga @ V
        Vb = Ad @ V
        if difficulty == 3:
            s3 = Doc()
            s3.p("Rotating the body-frame inertia into axes parallel to {a} (same centre of mass):")
            s3.mat("R_{ab}", R, 4)
            s3.eq(r"\mathcal{I}_c = R_{ab}\mathcal{I}_bR_{ab}^{T}")
            s3.mat("\\mathcal{I}_c", Ic, 5)
            s3.p("The trace (sum of principal moments) and the eigenvalues are unchanged by the rotation: "
                 f"$\\mathrm{{tr}}\\,\\mathcal{{I}}_c = {num(np.trace(Ic), 5)}$.")
            steps.append(Step("Rotated inertia", s3))
            ans["Ic"] = Ic
        s4 = Doc()
        base = "\\mathcal{I}_c" if difficulty == 3 else "\\mathcal{I}_b"
        s4.p(f"Steiner's theorem moves the reference point from the centre of mass by $p = {vec(p)}$ m "
             f"($p^{{T}}p = {num(p @ p, 4)}$ m²):")
        s4.eq(r"\mathcal{I}_a = " + base + r" + \mathfrak{m}\left(p^{T}p\,I - pp^{T}\right)")
        s4.mat("\\mathfrak{m}\\left(p^{T}p\\,I - pp^{T}\\right)", m * (p @ p * np.eye(3) - np.outer(p, p)), 5)
        s4.mat("\\mathcal{I}_a", Ia, 5)
        steps.append(Step("Steiner's (parallel-axis) theorem", s4))
        s5 = Doc()
        s5.p("With $T_{ab} = (R_{ab}, p)$ and $T_{ba} = T_{ab}^{-1}$, the spatial inertia transforms as (8.42):")
        s5.eq(r"\mathcal{G}_a = \left[\mathrm{Ad}_{T_{ba}}\right]^{T}\mathcal{G}_b\left[\mathrm{Ad}_{T_{ba}}\right]")
        s5.mat("\\mathcal{G}_a", Ga, 5)
        s5.p("Its upper-left $3\\times 3$ block is exactly $\\mathcal{I}_a$ from Steiner's theorem; the off-diagonal "
             "blocks $\\mathfrak{m}[p]$ appear because {a} is not at the centre of mass.")
        s5.eq(r"K = \frac{1}{2}\mathcal{V}_a^{T}\mathcal{G}_a\mathcal{V}_a = " + num(KE, 6) + r"\ \mathrm{J}")
        s5.p(f"Check: $\\mathcal{{V}}_b = [\\mathrm{{Ad}}_{{T_{{ba}}}}]\\mathcal{{V}}_a = {vec(Vb, 4)}$ gives the same "
             f"$\\frac{{1}}{{2}}\\mathcal{{V}}_b^{{T}}\\mathcal{{G}}_b\\mathcal{{V}}_b = {num(0.5 * Vb @ Gb @ Vb, 6)}$ J "
             "(`mr.Adjoint`, `mr.TransInv`).")
        steps.append(Step("Spatial inertia in {a} and kinetic energy", s5))
        ans.update(Ia=Ia, Ga=Ga, KE=KE)
        ad = Doc().p(f"Principal moments $({num(Ib[0, 0], 5)}, {num(Ib[1, 1], 5)}, {num(Ib[2, 2], 5)})$ kg·m²; "
                     f"$K = {num(KE, 6)}$ J.")
        ad.mat("\\mathcal{I}_a", Ia, 5)
        chap = "Ch. 8 (Sec. 8.2, eqs. 8.32, 8.42)"
    data = dict(shape=shape, m=m, dims=dims, R=R, p=p, V=V)
    vis = dict(kind="frames", frames={"R": R, "T": mr.RpToTrans(R, p)}, axis=w / np.linalg.norm(w))
    title = ("Inertia of a rigid body: principal moments and kinetic energy" if difficulty == 1 else
             "Inertia of a rigid body: Steiner shift and spatial inertia")
    return Problem("inertia", seed, difficulty, title, chap, st, data, steps, ans, ad, vis)


def _verify_inertia(p: Problem) -> float:
    """Re-solve by integrating over the solid with an exact quadrature rule (no inertia formulas, no Adjoint)."""
    d, a = p.data, p.answer
    pts, wts = _shape_quadrature(d["shape"], d["m"], d["dims"])
    ra = pts @ np.asarray(d["R"]).T + d["p"]

    def rot_inertia(r):
        return sum(wi * (ri @ ri * np.eye(3) - np.outer(ri, ri)) for ri, wi in zip(r, wts))

    def spatial(r):
        out = np.zeros((6, 6))
        for ri, wi in zip(r, wts):
            B = np.hstack([-_skew(ri), np.eye(3)])      # velocity of the mass point = v - [r] w
            out += wi * B.T @ B
        return out

    w, v = d["V"][:3], d["V"][3:]
    KE = 0.5 * sum(wi * np.sum((v + np.cross(w, ri))**2) for ri, wi in zip(ra, wts))
    e = max(np.abs(rot_inertia(pts) - a["Ib"]).max(), np.abs(spatial(pts) - a["Gb"]).max(), abs(KE - a["KE"]))
    if "Ia" in a:
        e = max(e, np.abs(rot_inertia(ra) - a["Ia"]).max(), np.abs(spatial(ra) - a["Ga"]).max())
    if "Ic" in a:
        e = max(e, np.abs(rot_inertia(pts @ np.asarray(d["R"]).T) - a["Ic"]).max())
    return float(e)


# -- forward dynamics and one integration step -----------------------------------------------------

def gen_fwd_dynamics(seed: int, difficulty: int) -> Problem:
    rng = _rng("fwd_dynamics", seed, difficulty)
    L1, L2, m1, m2, th = _random_2r_arm(rng)
    rods = difficulty == 3
    dth = np.array([_pick(rng, [-1.0, -0.5, 0.0, 0.5, 1.0]), _pick(rng, [-1.0, 0.5, 1.0, 2.0])])
    tau = np.array([_pick(rng, [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]), _pick(rng, [-4.0, -2.0, 0.0, 2.0, 5.0])])
    dt = float(_pick(rng, [0.01, 0.02, 0.05]))
    Mlist, Glist, Slist, gvec, prm = arm_2r(L1, L2, m1, m2, rods)
    Mm = mr.MassMatrix(th, Mlist, Glist, Slist)
    c = mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist)
    gt = mr.GravityForces(th, gvec, Mlist, Glist, Slist)
    ddth = mr.ForwardDynamics(th, dth, tau, gvec, np.zeros(6), Mlist, Glist, Slist)

    st = Doc()
    st.p(_arm_statement(L1, L2, m1, m2, rods))
    st.p(f"State: $\\theta = $ {degs(th)}, $\\dot{{\\theta}} = {vec(dth)}$ rad/s; the motors apply "
         f"$\\tau = {vec(tau)}$ N·m.")
    q = ("**(a)** Compute the mass matrix $M(\\theta)$, the velocity-product torques $c(\\theta,\\dot{\\theta})$ "
         "and the gravity torques $g(\\theta)$.  **(b)** Solve the forward dynamics for the joint accelerations "
         "$\\ddot{\\theta}$.")
    if difficulty >= 2:
        q += (f"  **(c)** Advance the state by one time step $\\Delta t = {num(dt)}$ s with (i) explicit Euler and "
              "(ii) semi-implicit (symplectic) Euler, which uses the *updated* velocity to update the angles.")
    st.p(q)

    steps = []
    s1 = Doc()
    c1, c2, s2v, c12 = np.cos(th[0]), np.cos(th[1]), np.sin(th[1]), np.cos(th.sum())
    s1.p(f"$\\cos\\theta_1 = {num(c1, 4)}$, $\\cos\\theta_2 = {num(c2, 4)}$, $\\sin\\theta_2 = {num(s2v, 4)}$, "
         f"$\\cos(\\theta_1+\\theta_2) = {num(c12, 4)}$.")
    _mass_matrix_doc(s1, rods, prm, Mm)
    steps.append(Step("Mass matrix", s1))
    s2 = Doc()
    if rods:
        s2.eq(r"c = \left(-\mathfrak{m}_2L_1l_{c2}\sin\theta_2\left(2\dot{\theta}_1\dot{\theta}_2 + \dot{\theta}_2^{2}"
              r"\right),\ \mathfrak{m}_2L_1l_{c2}\sin\theta_2\,\dot{\theta}_1^{2}\right)")
        s2.eq(r"g(\theta) = \left((\mathfrak{m}_1l_{c1} + \mathfrak{m}_2L_1)g\cos\theta_1 + \mathfrak{m}_2l_{c2}g"
              r"\cos(\theta_1+\theta_2),\ \mathfrak{m}_2l_{c2}g\cos(\theta_1+\theta_2)\right)")
    else:
        s2.eq(r"c = \left(-\mathfrak{m}_2L_1L_2\sin\theta_2\left(2\dot{\theta}_1\dot{\theta}_2 + \dot{\theta}_2^{2}"
              r"\right),\ \mathfrak{m}_2L_1L_2\dot{\theta}_1^{2}\sin\theta_2\right)")
        s2.eq(r"g(\theta) = \left((\mathfrak{m}_1+\mathfrak{m}_2)L_1g\cos\theta_1 + \mathfrak{m}_2gL_2\cos(\theta_1+"
              r"\theta_2),\ \mathfrak{m}_2gL_2\cos(\theta_1+\theta_2)\right)")
    s2.p(f"$c = {vec(c, 4)}$ N·m (`mr.VelQuadraticForces`), $g(\\theta) = {vec(gt, 4)}$ N·m (`mr.GravityForces`).")
    steps.append(Step("Velocity-product and gravity torques", s2))
    s3 = Doc()
    rhs = tau - c - gt
    s3.p("The equations of motion $M\\ddot{\\theta} + c + g = \\tau$ are solved for the accelerations:")
    s3.eq(r"\ddot{\theta} = M^{-1}(\theta)\left(\tau - c(\theta,\dot{\theta}) - g(\theta)\right)")
    s3.p(f"$\\tau - c - g = {vec(rhs, 4)}$ N·m and")
    s3.mat("M^{-1}", np.linalg.inv(Mm), 4)
    s3.p(f"so $\\ddot{{\\theta}} = {vec(ddth, 4)}$ rad/s² (`mr.ForwardDynamics`).")
    steps.append(Step("Forward dynamics", s3))
    ans = dict(M=Mm, c=c, g=gt, ddtheta=ddth)
    ad = Doc().p(f"$\\ddot{{\\theta}} = {vec(ddth, 4)}$ rad/s²  ($c = {vec(c, 4)}$, $g = {vec(gt, 4)}$ N·m).")
    if difficulty >= 2:
        th_e, dth_e = mr.EulerStep(th, dth, ddth, dt)
        dth_s = dth + dt * ddth
        th_s = th + dt * dth_s
        s4 = Doc()
        s4.p("Explicit Euler (`mr.EulerStep`) uses the old velocity for the angles:")
        s4.eq(r"\theta_{k+1} = \theta_k + \Delta t\,\dot{\theta}_k,\quad \dot{\theta}_{k+1} = "
              r"\dot{\theta}_k + \Delta t\,\ddot{\theta}_k")
        s4.p("Semi-implicit Euler updates the velocity first and uses the new one:")
        s4.eq(r"\dot{\theta}_{k+1} = \dot{\theta}_k + \Delta t\,\ddot{\theta}_k,\quad "
              r"\theta_{k+1} = \theta_k + \Delta t\,\dot{\theta}_{k+1}")
        s4.table(["method", r"$\theta_{k+1}$ [rad]", r"$\theta_{k+1}$ [°]", r"$\dot{\theta}_{k+1}$ [rad/s]"],
                 [["explicit Euler (`mr.EulerStep`)", vec(th_e, 5), vec(np.degrees(th_e), 3), vec(dth_e, 5)],
                  ["semi-implicit Euler", vec(th_s, 5), vec(np.degrees(th_s), 3), vec(dth_s, 5)]])
        s4.p(f"Both give the same velocity; the angles differ by $\\Delta t^{{2}}\\ddot{{\\theta}} = "
             f"{vec(th_s - th_e, 6)}$ rad.  Semi-implicit Euler is the usual choice for long simulations because it "
             "keeps the energy error bounded, while explicit Euler tends to add energy.")
        steps.append(Step("One Euler step", s4))
        ans.update(theta_explicit=th_e, dtheta_explicit=dth_e, theta_semi=th_s, dtheta_semi=dth_s)
        ad.p(f"Explicit Euler: $\\theta = {vec(th_e, 5)}$ rad, $\\dot{{\\theta}} = {vec(dth_e, 5)}$ rad/s; "
             f"semi-implicit: $\\theta = {vec(th_s, 5)}$ rad, $\\dot{{\\theta}} = {vec(dth_s, 5)}$ rad/s.")
    data = dict(L1=L1, L2=L2, m1=m1, m2=m2, rods=rods, theta=th, dtheta=dth, tau=tau, dt=dt)
    title = ("Forward dynamics of a 2R arm" if difficulty == 1 else
             "Forward dynamics of a 2R arm and one Euler step")
    return Problem("fwd_dynamics", seed, difficulty, title, "Ch. 8 (eqs. 8.9–8.10, Sec. 8.5)", st, data, steps,
                   ans, ad, _robot_visual(_two_r_spec(L1, L2), th))


def _verify_fwd_dynamics(p: Problem) -> float:
    d, a = p.data, p.answer
    _, _, _, gvec, prm = arm_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["rods"])
    Mm, c, gv = lagrange_2r_general(**prm, th=d["theta"], dth=d["dtheta"])
    ddth = np.linalg.solve(Mm, d["tau"] - c - gv)
    e = max(np.abs(Mm - a["M"]).max(), np.abs(c - a["c"]).max(), np.abs(gv - a["g"]).max(),
            np.abs(ddth - a["ddtheta"]).max())
    # the accelerations reproduce the applied torques through Newton-Euler inverse dynamics
    Mlist, Glist, Slist, gvec, _ = arm_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["rods"])
    tau = mr.InverseDynamics(d["theta"], d["dtheta"], a["ddtheta"], gvec, np.zeros(6), Mlist, Glist, Slist)
    e = max(e, np.abs(tau - d["tau"]).max())
    if "theta_explicit" in a:
        dt = d["dt"]
        e = max(e, np.abs(d["theta"] + dt * d["dtheta"] - a["theta_explicit"]).max(),
                np.abs(d["dtheta"] + dt * ddth - a["dtheta_explicit"]).max(),
                np.abs(d["dtheta"] + dt * ddth - a["dtheta_semi"]).max(),
                np.abs(d["theta"] + dt * (d["dtheta"] + dt * ddth) - a["theta_semi"]).max())
    return float(e)


# -- task-space mass matrix, felt mass, reflected rotor inertia ------------------------------------

_DIRS2 = [(1, 0), (0, 1), (1, 1), (1, -1), (3, 4), (4, -3), (-1, 1)]


def _sci_tex(k: int, e: int) -> str:
    return f"{k}\\times 10^{{{e}}}"


def gen_task_space(seed: int, difficulty: int) -> Problem:
    rng = _rng("task_space", seed, difficulty)
    L1, L2, m1, m2, th = _random_2r_arm(rng)
    rods = difficulty == 3
    Mlist, Glist, Slist, _, prm = arm_2r(L1, L2, m1, m2, rods)
    Mm = mr.MassMatrix(th, Mlist, Glist, Slist)
    J = _tip_jacobian_mr(L1, L2, th)
    Jinv = np.linalg.inv(J)
    Lam = Jinv.T @ Mm @ Jinv
    u = np.array(_pick(rng, _DIRS2), float)
    u /= np.linalg.norm(u)

    st = Doc()
    st.p(_arm_statement(L1, L2, m1, m2, rods, gravity=False))
    st.p(f"Configuration $\\theta = $ {degs(th)}.  Task coordinates are the tip position $x = (x, y)$ in {{s}}, with "
         "$\\dot{x} = J(\\theta)\\dot{\\theta}$.")
    q = ("**(a)** Find the tip Jacobian $J(\\theta)$ and the mass matrix $M(\\theta)$.  **(b)** Compute the task-space "
         "mass matrix $\\Lambda = J^{-T}MJ^{-1}$.")
    if difficulty >= 2:
        q += (f"  **(c)** A person pushes the tip along the unit direction $\\hat{{u}} = {_axis_md(u)}$.  What mass do "
              "they feel?  Give also the largest and smallest felt masses over all directions.")
    if difficulty == 3:
        k1, e1 = _pick(rng, [(1, -5), (2, -5), (5, -5), (1, -4)])
        Ir = k1 * 10.0**e1
        gr = np.array([float(_pick(rng, [20, 50, 80, 100])), float(_pick(rng, [20, 50, 80, 100]))])
        q += (f"  **(d)** Each joint is driven through a gearbox: rotor inertia $\\mathcal{{I}}_{{rot}} = "
              f"{_sci_tex(k1, e1)}$ kg·m², gear ratios $G_1 = {num(gr[0])}$, $G_2 = {num(gr[1])}$ (motor turns $G$ "
              "times faster than the joint).  Including the reflected rotor inertias in the joint-space mass matrix "
              "(neglect the rotor's gyroscopic coupling), how do $\\Lambda$ and the felt mass along $\\hat{u}$ change?")
    st.p(q)

    steps = []
    s1 = Doc()
    s1.p("Differentiating the tip position $(L_1\\cos\\theta_1 + L_2\\cos(\\theta_1+\\theta_2),\\ "
         "L_1\\sin\\theta_1 + L_2\\sin(\\theta_1+\\theta_2))$:")
    s1.eq(r"J_{11} = -L_1\sin\theta_1 - L_2\sin(\theta_1+\theta_2),\quad J_{12} = -L_2\sin(\theta_1+\theta_2)")
    s1.eq(r"J_{21} = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2),\quad J_{22} = L_2\cos(\theta_1+\theta_2)")
    s1.mat("J", J, 4)
    s1.p(f"(linear rows of `mr.JacobianBody` rotated into {{s}}).  $\\det J = L_1L_2\\sin\\theta_2 = "
         f"{num(np.linalg.det(J), 4)}$ ≠ 0, so the configuration is not singular and $J^{{-1}}$ exists.")
    _mass_matrix_doc(s1, rods, prm, Mm)
    steps.append(Step("Jacobian and joint-space mass matrix", s1))
    s2 = Doc()
    s2.p("Kinetic energy must be the same in both coordinates, $\\frac{1}{2}\\dot{\\theta}^{T}M\\dot{\\theta} = "
         "\\frac{1}{2}\\dot{x}^{T}\\Lambda\\dot{x}$, which gives (8.90):")
    s2.eq(r"\Lambda(\theta) = J^{-T}(\theta)M(\theta)J^{-1}(\theta)")
    s2.mat("J^{-1}", Jinv, 4)
    s2.mat("\\Lambda", Lam, 4, "\\mathrm{kg}")
    steps.append(Step("Task-space mass matrix", s2))
    ans = dict(J=J, M=Mm, Lam=Lam)
    ad = Doc().mat("\\Lambda", Lam, 4, "\\mathrm{kg}")
    if difficulty >= 2:
        Li = np.linalg.inv(Lam)
        mu = 1.0 / (u @ Li @ u)
        ev = np.linalg.eigvalsh(Lam)
        s3 = Doc()
        s3.p("A force $f$ applied at the tip produces the acceleration $\\ddot{x} = \\Lambda^{-1}f$ (from rest).  The "
             "component along $\\hat{u}$ of the response to $f = \\hat{u}$ gives the felt (effective) mass:")
        s3.eq(r"\mathfrak{m}_{\hat{u}} = \frac{1}{\hat{u}^{T}\Lambda^{-1}\hat{u}}")
        s3.mat("\\Lambda^{-1}", Li, 4)
        s3.p(f"$\\hat{{u}}^{{T}}\\Lambda^{{-1}}\\hat{{u}} = {num(u @ Li @ u, 5)}$, so $\\mathfrak{{m}}_{{\\hat{{u}}}} = "
             f"{num(mu, 4)}$ kg.  Over all directions the felt mass ranges between the eigenvalues of $\\Lambda$: "
             f"${num(ev[0], 4)}$ kg (lightest direction) to ${num(ev[1], 4)}$ kg (heaviest).")
        steps.append(Step("Felt mass along a direction", s3))
        ans.update(m_u=mu, Lam_eig=ev)
        ad.p(f"Felt mass along $\\hat{{u}}$: ${num(mu, 4)}$ kg; range ${num(ev[0], 4)}$ – ${num(ev[1], 4)}$ kg.")
    if difficulty == 3:
        refl = gr**2 * Ir
        Mr = Mm + np.diag(refl)
        Lam_r = Jinv.T @ Mr @ Jinv
        mu_r = 1.0 / (u @ np.linalg.inv(Lam_r) @ u)
        s4 = Doc()
        s4.p("A rotor spinning $G$ times faster than its joint stores kinetic energy "
             "$\\frac{1}{2}\\mathcal{I}_{rot}(G\\dot{\\theta}_i)^{2}$, so it appears at the joint as the reflected "
             "inertia $G^{2}\\mathcal{I}_{rot}$ (Sec. 8.9.2):")
        s4.p(f"$G_1^{{2}}\\mathcal{{I}}_{{rot}} = {num(refl[0], 5)}$ kg·m², $G_2^{{2}}\\mathcal{{I}}_{{rot}} = "
             f"{num(refl[1], 5)}$ kg·m², added to the diagonal of $M$:")
        s4.mat("M_{rot}", Mr, 4)
        s4.mat("\\Lambda_{rot} = J^{-T}M_{rot}J^{-1}", Lam_r, 4, "\\mathrm{kg}")
        s4.p(f"Felt mass along $\\hat{{u}}$: ${num(mu_r, 4)}$ kg instead of ${num(mu, 4)}$ kg "
             f"({num(100 * (mu_r / mu - 1), 2)}% more): with high gear ratios the motors, not the links, can dominate "
             "what a person feels at the tip.")
        steps.append(Step("Reflected rotor inertia", s4))
        ans.update(M_rot=Mr, Lam_rot=Lam_r, m_u_rot=mu_r)
        ad.p(f"With the rotors: felt mass ${num(mu_r, 4)}$ kg.")
    data = dict(L1=L1, L2=L2, m1=m1, m2=m2, rods=rods, theta=th, u=u)
    if difficulty == 3:
        data.update(I_rot=Ir, gear=gr)
    title = ("Task-space mass matrix of a 2R arm" if difficulty == 1 else
             "Task-space mass matrix and felt mass" if difficulty == 2 else
             "Task-space mass matrix, felt mass and reflected rotor inertia")
    spec = _two_r_spec(L1, L2)
    tip = mr.FKinBody(spec.M, spec.Blist, th)
    return Problem("task_space", seed, difficulty, title, "Ch. 8 (Sec. 8.6, eq. 8.90; Sec. 8.9.2)", st, data, steps,
                   ans, ad, _robot_visual(spec, th, frames={"tip": tip}))


def _verify_task_space(p: Problem) -> float:
    d, a = p.data, p.answer
    _, _, _, _, prm = arm_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["rods"])
    Mm, _, _ = lagrange_2r_general(**prm, th=d["theta"], dth=np.zeros(2))
    J = _tip_jacobian_closed(d["L1"], d["L2"], d["theta"])
    Lam = np.linalg.inv(J @ np.linalg.solve(Mm, J.T))           # (J M^-1 J^T)^-1, no explicit J^-1
    e = max(np.abs(J - a["J"]).max(), np.abs(Mm - a["M"]).max(), np.abs(Lam - a["Lam"]).max())

    def felt(Mq):
        """Least kinetic energy (x2) to give the tip unit speed along u: KKT of min th'^T M th' s.t. u^T J th' = 1."""
        Ju = d["u"] @ J
        K = np.block([[2 * Mq, -Ju[:, None]], [Ju[None, :], np.zeros((1, 1))]])
        sol = np.linalg.solve(K, np.r_[0.0, 0.0, 1.0])
        return float(sol[:2] @ Mq @ sol[:2])

    if "m_u" in a:
        e = max(e, abs(felt(Mm) - a["m_u"]), np.abs(np.sort(np.linalg.eigvals(Lam).real) - a["Lam_eig"]).max())
    if "m_u_rot" in a:
        Mr = Mm + np.diag(d["gear"]**2 * d["I_rot"])
        e = max(e, np.abs(Mr - a["M_rot"]).max(), abs(felt(Mr) - a["m_u_rot"]),
                np.abs(np.linalg.inv(J @ np.linalg.solve(Mr, J.T)) - a["Lam_rot"]).max())
    return float(e)


# -- constrained dynamics with a Pfaffian constraint -----------------------------------------------

def _tip_quadratic_accel(L1, L2, th, dth) -> np.ndarray:
    """J-dot(theta) theta-dot for the tip of a 2R arm (centripetal part of the tip acceleration)."""
    w1, w12 = dth[0], dth[0] + dth[1]
    return np.array([-L1 * np.cos(th[0]) * w1**2 - L2 * np.cos(th.sum()) * w12**2,
                     -L1 * np.sin(th[0]) * w1**2 - L2 * np.sin(th.sum()) * w12**2])


def gen_constrained(seed: int, difficulty: int) -> Problem:
    rng = _rng("constrained", seed, difficulty)
    L1, L2, m1, m2, th = _random_2r_arm(rng)
    rods = difficulty == 2
    tau = np.array([_pick(rng, [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]), _pick(rng, [-4.0, -2.0, 0.0, 2.0, 5.0])])
    Mlist, Glist, Slist, gvec, prm = arm_2r(L1, L2, m1, m2, rods)

    st = Doc()
    st.p(_arm_statement(L1, L2, m1, m2, rods))
    steps = []
    s1 = Doc()
    if difficulty < 3:
        kind = "orientation"
        A = np.array([[1.0, 1.0]])
        if difficulty == 1:
            dth = np.zeros(2)
        else:
            wv = float(_pick(rng, [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]))
            dth = np.array([wv, -wv])
        adq = np.zeros(1)
        st.p("A guide holds the tool (link 2) at a constant orientation $\\phi = \\theta_1 + \\theta_2$, i.e. the "
             "Pfaffian velocity constraint $A\\dot{\\theta} = \\dot{\\theta}_1 + \\dot{\\theta}_2 = 0$.")
        st.p(f"State: $\\theta = $ {degs(th)}, $\\dot{{\\theta}} = {vec(dth)}$ rad/s (consistent with the constraint); "
             f"motor torques $\\tau = {vec(tau)}$ N·m.")
        s1.p("$A = [1\\ \\ 1]$ is constant, so $\\dot{A} = 0$ and differentiating the constraint gives "
             "$A\\ddot{\\theta} = 0$.  The constraint torque $A^{T}\\lambda = (\\lambda, \\lambda)$ is a pure moment "
             "$\\lambda$ exerted by the guide on link 2.")
        s1.p(f"Check: $A\\dot{{\\theta}} = {num((A @ dth)[0], 6)}$.")
    else:
        kind = "line"
        dvec = np.array(_pick(rng, [(1, 0), (0, 1), (1, 1), (1, -1)]), float)
        dvec /= np.linalg.norm(dvec)
        nvec = np.array([-dvec[1], dvec[0]])
        vt = float(_pick(rng, [-0.5, -0.3, 0.2, 0.4, 0.5]))
        J = _tip_jacobian_mr(L1, L2, th)
        A = (nvec @ J)[None, :]
        dth = np.linalg.solve(J, vt * dvec)
        adq = np.array([nvec @ _tip_quadratic_accel(L1, L2, th, dth)])
        st.p(f"The tip slides without friction along a straight guide rail with unit direction $\\hat{{d}} = "
             f"{_axis_md(dvec)}$ (normal $\\hat{{n}} = {_axis_md(nvec)}$): the tip velocity has no component along "
             "$\\hat{n}$, i.e. $A(\\theta)\\dot{\\theta} = \\hat{n}^{T}J(\\theta)\\dot{\\theta} = 0$.")
        st.p(f"State: $\\theta = $ {degs(th)} with the tip moving along the rail at $\\dot{{x}} = {num(vt)}\\,\\hat{{d}}$ "
             f"m/s; motor torques $\\tau = {vec(tau)}$ N·m.")
        s1.mat("J", J, 4)
        s1.p(f"$A = \\hat{{n}}^{{T}}J = {vec(A[0], 4)}$ and the joint velocities that move the tip at "
             f"${num(vt)}\\,\\hat{{d}}$ m/s are $\\dot{{\\theta}} = J^{{-1}}({num(vt)}\\hat{{d}}) = {vec(dth, 4)}$ rad/s "
             f"(check $A\\dot{{\\theta}} = {num((A @ dth)[0], 6)}$).")
        s1.p("$A$ depends on $\\theta$, so the differentiated constraint is $A\\ddot{\\theta} + \\dot{A}\\dot{\\theta} = 0$ "
             "with $\\dot{A}\\dot{\\theta} = \\hat{n}^{T}\\dot{J}\\dot{\\theta}$, the normal component of the "
             "centripetal tip acceleration:")
        s1.eq(r"\dot{J}\dot{\theta} = -\left(L_1\cos\theta_1\dot{\theta}_1^{2} + L_2\cos(\theta_1+\theta_2)"
              r"\dot{\phi}^{2},\ L_1\sin\theta_1\dot{\theta}_1^{2} + L_2\sin(\theta_1+\theta_2)\dot{\phi}^{2}\right),"
              r"\quad \dot{\phi} = \dot{\theta}_1 + \dot{\theta}_2")
        s1.p(f"$\\dot{{J}}\\dot{{\\theta}} = {vec(_tip_quadratic_accel(L1, L2, th, dth), 4)}$ m/s², so "
             f"$\\dot{{A}}\\dot{{\\theta}} = {num(adq[0], 5)}$.  The constraint torque $A^{{T}}\\lambda = "
             "J^{T}(\\lambda\\hat{n})$ is the rail's normal force $\\lambda\\hat{n}$ (N) at the tip.")
    q = ("Write the equations of motion with the sign convention $M\\ddot{\\theta} + c + g = \\tau + A^{T}\\lambda$, "
         "where $A^{T}\\lambda$ are the joint torques produced by the constraint.  **(a)** Find the Lagrange "
         "multiplier $\\lambda$.  **(b)** Find the constrained accelerations $\\ddot{\\theta}$.  **(c)** Show that "
         "the constraint torques do no work.")
    if difficulty >= 2:
        q += "  **(d)** Compare with the accelerations the arm would have without the constraint."
    st.p(q)
    steps.append(Step("Constraint matrix", s1))

    Mm = mr.MassMatrix(th, Mlist, Glist, Slist)
    c = mr.VelQuadraticForces(th, dth, Mlist, Glist, Slist)
    gt = mr.GravityForces(th, gvec, Mlist, Glist, Slist)
    h = c + gt
    Minv = np.linalg.inv(Mm)
    s2 = Doc()
    _mass_matrix_doc(s2, rods, prm, Mm)
    s2.p(f"$h = c + g = {vec(c, 4)} + {vec(gt, 4)} = {vec(h, 4)}$ N·m (`mr.VelQuadraticForces`, "
         "`mr.GravityForces`).")
    steps.append(Step("Mass matrix and bias torques", s2))

    AMA = (A @ Minv @ A.T)[0, 0]
    lam = (-adq[0] - (A @ Minv @ (tau - h))[0]) / AMA
    ddth = Minv @ (tau - h + A[0] * lam)
    tcon = A[0] * lam
    power = float(tcon @ dth)
    ddth_free = Minv @ (tau - h)
    s3 = Doc()
    s3.p("Substitute $\\ddot{\\theta} = M^{-1}(\\tau - h + A^{T}\\lambda)$ into the differentiated constraint "
         "$A\\ddot{\\theta} = -\\dot{A}\\dot{\\theta}$ (Sec. 8.7) and solve for the multiplier:")
    s3.eq(r"\lambda = \left(AM^{-1}A^{T}\right)^{-1}\left(-\dot{A}\dot{\theta} - AM^{-1}(\tau - h)\right)")
    s3.p(f"$AM^{{-1}}A^{{T}} = {num(AMA, 5)}$, $AM^{{-1}}(\\tau - h) = {num((A @ Minv @ (tau - h))[0], 5)}$, so "
         f"$\\lambda = {num(lam, 4)}$ {'N·m' if kind == 'orientation' else 'N'}.")
    steps.append(Step("Lagrange multiplier", s3))
    s4 = Doc()
    s4.p(f"Constraint torques $A^{{T}}\\lambda = {vec(tcon, 4)}$ N·m and")
    s4.eq(r"\ddot{\theta} = M^{-1}\left(\tau - h + A^{T}\lambda\right) = " + vec(ddth, 4) + r"\ \mathrm{rad/s^{2}}")
    s4.p(f"Check: $A\\ddot{{\\theta}} + \\dot{{A}}\\dot{{\\theta}} = {num((A @ ddth)[0] + adq[0], 8)}$.  Power of the "
         f"constraint torques: $(A^{{T}}\\lambda)^{{T}}\\dot{{\\theta}} = \\lambda\\,A\\dot{{\\theta}} = {num(power, 8)}$ W "
         "— the constraint does no work because it only acts in the directions the arm is not allowed to move.")
    steps.append(Step("Constrained accelerations and power", s4))
    ans = dict(A=A, Adot_dtheta=adq, lam=lam, ddtheta=ddth, tau_con=tcon, power=power)
    unit = "N·m" if kind == "orientation" else "N"
    ad = Doc().p(f"$\\lambda = {num(lam, 4)}$ {unit}, $\\ddot{{\\theta}} = {vec(ddth, 4)}$ rad/s², constraint torques "
                 f"${vec(tcon, 4)}$ N·m, constraint power $0$.")
    if difficulty >= 2:
        s5 = Doc()
        s5.p(f"Without the constraint $\\ddot{{\\theta}}_{{free}} = M^{{-1}}(\\tau - h) = {vec(ddth_free, 4)}$ rad/s², "
             f"which violates it: $A\\ddot{{\\theta}}_{{free}} + \\dot{{A}}\\dot{{\\theta}} = "
             f"{num((A @ ddth_free)[0] + adq[0], 4)}$.")
        steps.append(Step("(d) Unconstrained comparison", s5))
        ans["ddtheta_free"] = ddth_free
        ad.p(f"(d) Unconstrained: $\\ddot{{\\theta}}_{{free}} = {vec(ddth_free, 4)}$ rad/s².")
    data = dict(L1=L1, L2=L2, m1=m1, m2=m2, rods=rods, theta=th, dtheta=dth, tau=tau, kind=kind)
    if kind == "line":
        data.update(d=dvec, n=nvec, v=vt)
    title = ("Constrained dynamics: constant tool orientation" if kind == "orientation" else
             "Constrained dynamics: tip on a guide rail")
    return Problem("constrained", seed, difficulty, title, "Ch. 8 (Sec. 8.7)", st, data, steps, ans, ad,
                   _robot_visual(_two_r_spec(L1, L2), th))


def _verify_constrained(p: Problem) -> float:
    """Closed-form Lagrangian, A(theta) from the closed-form Jacobian, A-dot theta-dot by central differences, and
    the joint KKT system [[M, -A^T], [A, 0]] (theta-dd, lambda) = (tau - h, -A-dot theta-dot) solved at once."""
    d, a = p.data, p.answer
    th, dth = d["theta"], d["dtheta"]
    _, _, _, _, prm = arm_2r(d["L1"], d["L2"], d["m1"], d["m2"], d["rods"])
    Mm, c, gv = lagrange_2r_general(**prm, th=th, dth=dth)

    def A_of(t):
        if d["kind"] == "orientation":
            return np.array([1.0, 1.0])
        return d["n"] @ _tip_jacobian_closed(d["L1"], d["L2"], t)

    A = A_of(th)
    eps = 1e-5
    adq = (A_of(th + eps * dth) - A_of(th - eps * dth)) / (2 * eps) @ dth
    K = np.block([[Mm, -A[:, None]], [A[None, :], np.zeros((1, 1))]])
    sol = np.linalg.solve(K, np.r_[d["tau"] - c - gv, -adq])
    ddth, lam = sol[:2], sol[2]
    tcon = A * lam
    e = max(np.abs(A - a["A"][0]).max(), abs(adq - a["Adot_dtheta"][0]), abs(lam - a["lam"]),
            np.abs(ddth - a["ddtheta"]).max(), np.abs(tcon - a["tau_con"]).max(),
            abs(tcon @ dth), abs(a["power"]), abs(A @ dth))
    if d["kind"] == "line":
        e = max(e, np.abs(_tip_jacobian_closed(d["L1"], d["L2"], th) @ dth - d["v"] * d["d"]).max())
    if "ddtheta_free" in a:
        e = max(e, np.abs(np.linalg.solve(Mm, d["tau"] - c - gv) - a["ddtheta_free"]).max())
    return float(e)


# ------------------------------------------------------------------------------------------------
# Ch. 5 statics, Ch. 11 impedance / force control, Ch. 12 grasping, Ch. 13 wheeled mobile robots
# ------------------------------------------------------------------------------------------------


def _cross2(a, b) -> float:
    """z-component of the planar cross product a x b (the moment of a force b applied at a)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _chain_points(L, th):
    """Planar nR chain in closed form: [joint 1, ..., joint n, tip] positions and the tip angle phi."""
    pts, ang = [np.zeros(2)], 0.0
    for Li, t in zip(L, th):
        ang += t
        pts.append(pts[-1] + Li * np.array([np.cos(ang), np.sin(ang)]))
    return pts, ang


def _fd_tip_jacobian(L, th, eps: float = 1e-6) -> np.ndarray:
    """2 x n Jacobian of the tip position by central differences of the closed-form chain."""
    th = np.asarray(th, float)
    cols = []
    for i in range(len(th)):
        dt = np.zeros(len(th))
        dt[i] = eps
        cols.append((_chain_points(L, th + dt)[0][-1] - _chain_points(L, th - dt)[0][-1]) / (2 * eps))
    return np.array(cols).T


_FORCES = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]


# -- Ch. 5: statics tau = J^T F and the manipulability ellipse --------------------------------------

def gen_statics(seed: int, difficulty: int) -> Problem:
    rng = _rng("statics", seed, difficulty)
    spec = random_robot(rng, "3R" if difficulty == 3 else "2R")
    n = spec.n
    th = random_theta(rng, spec, difficulty)
    if difficulty >= 2:      # keep J square and well away from the elbow singularity theta_2 = 0
        while abs(np.sin(th[1])) < 0.45:
            th[1] = int(rng.integers(2, 7)) * 15 * _pick(rng, [-1, 1]) * DEG
    L = [spec.dims[f"L_{i}"] for i in range(1, n + 1)]
    Tsb = mr.FKinSpace(spec.M, spec.Slist, th)
    p = Tsb[:2, 3]
    Jsp = mr.JacobianSpace(spec.Slist, th)[[2, 3, 4]]          # planar rows (omega_z, v_x, v_y)
    Jb = mr.JacobianBody(spec.Blist, th)
    Jbp = Jb[[2, 3, 4]]
    Jv = (Tsb[:3, :3] @ Jb[3:])[:2]                             # tip linear velocity in {s}
    q_pts = [np.array(j["q"][:2], float) for j in spec.joints]  # joint axes at theta = 0 ...
    q_now = [mr.FKinSpace(mr.RpToTrans(np.eye(3), np.r_[q, 0]), spec.Slist[:, :i], th[:i])[:2, 3]
             if i else q for i, q in enumerate(q_pts)]          # ... and where they are now

    st = Doc()
    st.p(spec.description_md() + "  The arm is at rest in a horizontal plane (no gravity, no friction).")
    st.p(f"Configuration $\\theta = $ {degs(th)}.  Planar twists are written $(\\omega_z, v_x, v_y)$ and planar "
         "wrenches $\\mathcal{F} = (m_z, f_x, f_y)$, the planar parts of MR's $(\\omega, v)$ and $(m, f)$.")
    steps, ans = [], {}
    data = dict(L=np.array(L), theta=th, family=spec.family)
    if difficulty <= 2:
        while True:
            f = np.array([_pick(rng, _FORCES), _pick(rng, _FORCES)])
            if np.any(f):
                break
        Fs = np.r_[_cross2(p, f), f]
        tau = Jsp.T @ Fs
        data["f"] = f
        q = (f"The tip pushes on the environment with the force $f = {vec(f)}$ N (in {{s}}) and no moment.  "
             "**(a)** Write the tip wrench $\\mathcal{F}_s = (m_z, f_x, f_y)$ in {s}.  **(b)** Find the planar rows of the "
             "space Jacobian $J_s(\\theta)$.  **(c)** Find the joint torques $\\tau$ that hold the arm at rest while it "
             "applies this force.")
        if difficulty == 2:
            ts_ = np.array([_pick(rng, [-10.0, -5.0, -2.0, 2.0, 5.0, 10.0]), _pick(rng, [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])])
            data["tau_star"] = ts_
            q += (f"  **(d)** Now the tip is pinned to a fixture (a point contact: any force, no moment) and the motors "
                  f"apply $\\tau^{{*}} = {vec(ts_)}$ N·m.  What force $f^{{*}}$ does the tip exert on the fixture?  (The "
                  "$2\\times 2$ Jacobian $J_v$ of the tip position is square here.)")
        st.p(q)
        s1 = Doc()
        s1.eq(r"p = \left(L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2),\ L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2)"
              r"\right) = " + vec(p, 4) + r"\ \mathrm{m}")
        s1.p(f"A force $f$ applied at $p$ has the moment $m_z = p_xf_y - p_yf_x = {num(Fs[0], 4)}$ N·m about the "
             f"{{s}} origin, so $\\mathcal{{F}}_s = (m_z, f_x, f_y) = {vec(Fs, 4)}$.")
        steps.append(Step("Tip position and tip wrench in {s}", s1))
        s2 = Doc()
        s2.p("Joint $i$ rotates about $\\hat{z}$ through the point $q_i$, so its space screw axis has "
             "$\\omega = \\hat{z}$ and $v = -\\omega\\times q_i = (q_{iy}, -q_{ix})$: the planar column is "
             "$(1, q_{iy}, -q_{ix})$.  " +
             ", ".join(f"$q_{{{i + 1}}} = {vec(qq, 4)}$" for i, qq in enumerate(q_now)) + ":")
        s2.mat("J_s", Jsp, 4)
        s2.p("(rows $\\omega_z, v_x, v_y$ of `mr.JacobianSpace`).")
        steps.append(Step("Space Jacobian", s2))
        s3 = Doc()
        s3.p("At rest the power the motors put in equals the power delivered at the tip, "
             "$\\tau^{T}\\dot{\\theta} = \\mathcal{F}_s^{T}\\mathcal{V}_s = \\mathcal{F}_s^{T}J_s\\dot{\\theta}$ for every "
             "$\\dot{\\theta}$, hence (Sec. 5.2)")
        s3.eq(r"\tau = J_s^{T}(\theta)\mathcal{F}_s = " + vec(tau, 4) + r"\ \mathrm{N\cdot m}")
        s3.p("Check: each $\\tau_i$ is the moment of $f$ about joint $i$, $\\tau_i = (p - q_i)\\times f$: " +
             ", ".join(f"$\\tau_{{{i + 1}}} = {num(_cross2(p - qq, f), 4)}$" for i, qq in enumerate(q_now)) + ".")
        steps.append(Step("Joint torques", s3))
        ans.update(Fs=Fs, Js_p=Jsp, tau=tau)
        ad = Doc().p(f"$\\mathcal{{F}}_s = {vec(Fs, 4)}$, $\\tau = J_s^{{T}}\\mathcal{{F}}_s = {vec(tau, 4)}$ N·m.")
        if difficulty == 2:
            fstar = np.linalg.solve(Jv.T, ts_)
            s4 = Doc()
            s4.p("For a pure force at the tip the moment rows drop out and $\\tau = J_v^{T}f$, with $J_v$ the Jacobian of "
                 "the tip position (linear rows of `mr.JacobianBody` rotated into {s}):")
            s4.mat("J_v", Jv, 4)
            s4.p(f"(check: $J_v^{{T}}f = {vec(Jv.T @ f, 4)}$ N·m, as in (c)).  $\\det J_v = L_1L_2\\sin\\theta_2 = "
                 f"{num(np.linalg.det(Jv), 4)}$ ≠ 0, so the map is invertible:")
            s4.eq(r"f^{*} = J_v^{-T}\tau^{*} = " + vec(fstar, 4) + r"\ \mathrm{N}")
            steps.append(Step("(d) Force from the joint torques", s4))
            ans.update(Jv=Jv, f_star=fstar)
            ad.p(f"(d) $f^{{*}} = J_v^{{-T}}\\tau^{{*}} = {vec(fstar, 4)}$ N.")
        title = ("Statics of a 2R arm: τ = JᵀF" if difficulty == 1 else "Statics of a 2R arm: torques ↔ tip force")
    else:
        Fb = np.array([_pick(rng, [-2.0, -1.0, 0.0, 1.0, 2.0]), _pick(rng, _FORCES), _pick(rng, _FORCES)])
        if not np.any(Fb):
            Fb[1] = 10.0
        ts_ = np.array([_pick(rng, [-10.0, -5.0, 5.0, 10.0]), _pick(rng, [-4.0, -2.0, 2.0, 4.0]),
                        _pick(rng, [-1.0, -0.5, 0.5, 1.0])])
        data.update(Fb=Fb, tau_star=ts_)
        st.p(f"A tool frame {{b}} sits at the tip, its $\\hat{{x}}_b$ axis along link 3.  **(a)** Find the planar rows of the "
             f"body Jacobian $J_b(\\theta)$.  **(b)** The tool applies the wrench $\\mathcal{{F}}_b = {vec(Fb)}$ "
             "(N·m, N, N; in {b}) to the environment: which joint torques hold it?  **(c)** With the tool clamped, the "
             f"motors apply $\\tau^{{*}} = {vec(ts_)}$ N·m: which wrench $\\mathcal{{F}}_b^{{*}}$ does the tool exert?  "
             "**(d)** Give the semi-axis lengths and directions of the tip's linear-velocity manipulability ellipse "
             "and of its force ellipse, and the isotropy $\\mu_1$.")
        tau = Jbp.T @ Fb
        Fbs = np.linalg.solve(Jbp.T, ts_)
        r_b = [Tsb[:2, :2].T @ (qq - p) for qq in q_now]
        s1 = Doc()
        s1.p(f"Tip $p = {vec(p, 4)}$ m, tool angle $\\phi = \\theta_1+\\theta_2+\\theta_3 = {mdeg(sum(th))}$.  In {{b}} "
             "joint $i$ sits at $r_i = R_{sb}^{T}(q_i - p)$, and its body screw axis has the planar column "
             "$(1, r_{iy}, -r_{ix})$: " + ", ".join(f"$r_{{{i + 1}}} = {vec(rr, 4)}$" for i, rr in enumerate(r_b)) + ".")
        s1.mat("J_b", Jbp, 4)
        s1.p(f"(rows $\\omega_z, v_x, v_y$ of `mr.JacobianBody`); $\\det J_b = L_1L_2\\sin\\theta_2 = "
             f"{num(np.linalg.det(Jbp), 4)}$.")
        steps.append(Step("Body Jacobian", s1))
        s2 = Doc()
        s2.p("Power balance in the body frame, $\\tau^{T}\\dot{\\theta} = \\mathcal{F}_b^{T}J_b\\dot{\\theta}$, gives "
             "(Sec. 5.2)")
        s2.eq(r"\tau = J_b^{T}(\theta)\mathcal{F}_b = " + vec(tau, 4) + r"\ \mathrm{N\cdot m}")
        steps.append(Step("(b) Joint torques", s2))
        s3 = Doc()
        s3.p("$J_b$ is $3\\times 3$ and non-singular, so the static relation can be inverted:")
        s3.eq(r"\mathcal{F}_b^{*} = J_b^{-T}\tau^{*} = " + vec(Fbs, 4))
        s3.p("(moment in N·m, force in N, both in {b}).")
        steps.append(Step("(c) Wrench from the joint torques", s3))
        ans.update(Jb_p=Jbp, tau=tau, Fb_star=Fbs)
        ad = Doc().p(f"(b) $\\tau = {vec(tau, 4)}$ N·m.  (c) $\\mathcal{{F}}_b^{{*}} = {vec(Fbs, 4)}$.")
        title = "Statics of a 3R arm: body wrench and manipulability"
    if difficulty == 3:
        A = Jv @ Jv.T
        lam, vecs = np.linalg.eigh(A)
        ax_v, ax_f = np.sqrt(lam), 1.0 / np.sqrt(lam)
        mu1 = float(np.sqrt(lam[1] / lam[0]))
        angs = [np.degrees(np.arctan2(v[1], v[0])) % 180.0 for v in vecs.T]
        s4 = Doc()
        s4.p("The tip-position Jacobian $J_v = R_{sb}J_{b,v}$ (the linear rows $J_{b,v}$ of $J_b$ rotated into {s}) maps unit joint "
             "speeds $\\|\\dot{\\theta}\\| = 1$ onto the manipulability ellipse $\\dot{p}^{T}A^{-1}\\dot{p} = 1$ with "
             "$A = J_vJ_v^{T}$ (Sec. 5.4):")
        s4.mat("J_v", Jv, 4)
        s4.mat("A = J_vJ_v^{T}", A, 4)
        s4.p(f"Eigenvalues $\\lambda = {vec(lam, 4)}$ with unit eigenvectors at {num(angs[0], 2)}° and {num(angs[1], 2)}° "
             f"from $\\hat{{x}}_s$.  Velocity-ellipse semi-axes $\\sqrt{{\\lambda_i}} = {vec(ax_v, 4)}$ m/s per rad/s; the "
             f"force ellipse ($f^{{T}}Af = 1$ for $\\|\\tau\\| = 1$) has the same axes with lengths $1/\\sqrt{{\\lambda_i}} = "
             f"{vec(ax_f, 4)}$ N per N·m — strongest where the arm is slowest.  Isotropy "
             f"$\\mu_1 = \\sqrt{{\\lambda_{{max}}/\\lambda_{{min}}}} = {num(mu1, 4)}$ (1 would be a circle).")
        steps.append(Step("(d) Manipulability and force ellipses", s4))
        ans.update(Jv=Jv, ell_axes=ax_v, force_axes=np.sort(ax_f), mu1=mu1)
        ad.p(f"(d) velocity ellipse semi-axes ${vec(ax_v, 4)}$ (at {num(angs[0], 2)}°, {num(angs[1], 2)}°), force "
             f"ellipse ${vec(ax_f, 4)}$, $\\mu_1 = {num(mu1, 4)}$.")
    chap = "Ch. 5 (Secs. 5.2, 5.4)" if difficulty == 3 else "Ch. 5 (Sec. 5.2)"
    return Problem("statics", seed, difficulty, title, chap, st, data, steps, ans, ad,
                   _robot_visual(spec, th, frames={"b": Tsb}))


def _verify_statics(p: Problem) -> float:
    """Closed-form joint positions: tau_i as the moment of f about joint i, J_s columns (1, q_y, -q_x); the tip
    Jacobian by central differences; least-squares inverses; ellipse axes from the singular values of J_v."""
    d, a = p.data, p.answer
    L, th = d["L"], d["theta"]
    pts, phi = _chain_points(L, th)
    tip, joints = pts[-1], pts[:-1]
    Jv = _fd_tip_jacobian(L, th)
    e = 0.0
    if "f" in d:
        f = d["f"]
        Jsp = np.array([[1.0, q[1], -q[0]] for q in joints]).T
        tau = np.array([_cross2(tip - q, f) for q in joints])
        e = max(abs(a["Fs"][0] - _cross2(tip, f)), np.abs(a["Fs"][1:] - f).max(), np.abs(Jsp - a["Js_p"]).max(),
                np.abs(tau - a["tau"]).max())
    if "f_star" in a:
        fs = np.linalg.lstsq(Jv.T, d["tau_star"], rcond=None)[0]
        e = max(e, np.abs(Jv - a["Jv"]).max(), np.abs(fs - a["f_star"]).max())
    if "Fb" in d:
        Jbp = np.vstack([np.ones(len(th)), _rot2(phi).T @ Jv])     # body twist (phi-dot, R^T p-dot)
        e = max(e, np.abs(Jbp - a["Jb_p"]).max(), np.abs(Jbp.T @ d["Fb"] - a["tau"]).max(),
                np.abs(np.linalg.lstsq(Jbp.T, d["tau_star"], rcond=None)[0] - a["Fb_star"]).max())
    if "ell_axes" in a:
        sv = np.sort(np.linalg.svd(Jv, compute_uv=False))
        e = max(e, np.abs(Jv - a["Jv"]).max(), np.abs(sv - a["ell_axes"]).max(),
                np.abs(np.sort(1 / sv) - a["force_axes"]).max(), abs(sv[-1] / sv[0] - a["mu1"]))
    return float(e)


# -- Ch. 11: impedance control and PI force control -------------------------------------------------

def _ts_rule(zeta, wn) -> float:
    """MR's 2 % rule of thumb: 4/(zeta wn), or four time constants of the slow root when overdamped."""
    if zeta > 1 + 1e-9:
        return 4.0 / (wn * (zeta - np.sqrt(zeta**2 - 1)))
    return 4.0 / (zeta * wn)


def _overshoot(zeta) -> float:
    return float(np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))) if zeta < 1 - 1e-9 else 0.0


def _response_doc(doc: Doc, zeta, wn, what: str):
    kind = _classify(zeta)
    os_, ts = _overshoot(zeta), _ts_rule(zeta, wn)
    if kind == "underdamped":
        doc.p(f"$\\zeta < 1$: **underdamped**.  {what} overshoots its final value by "
              f"$e^{{-\\pi\\zeta/\\sqrt{{1-\\zeta^{{2}}}}}} = {num(100 * os_, 2)}$ %; MR's rule of thumb gives the 2% "
              f"settling time $t_s \\approx 4/(\\zeta\\omega_n) = {num(ts, 4)}$ s.")
    elif kind == "overdamped":
        doc.p(f"$\\zeta > 1$: **overdamped**, no overshoot; the slow root $s_1 = -\\omega_n(\\zeta - "
              f"\\sqrt{{\\zeta^{{2}}-1}}) = {num(-4 / ts, 4)}$ 1/s sets the settling time "
              f"$t_s \\approx 4/|s_1| = {num(ts, 4)}$ s.")
    else:
        doc.p(f"$\\zeta = 1$: **critically damped**, no overshoot; $t_s \\approx 4/\\omega_n = {num(ts, 4)}$ s.")
    return os_, ts


def _lin_response(A, b, x0, t_end, n=1200):
    """Response of x' = A x + b from x0 on a uniform grid (exact: matrix exponential of the augmented system)."""
    k = len(x0)
    Aa = np.zeros((k + 1, k + 1))
    Aa[:k, :k], Aa[:k, k] = A, b
    dt = t_end / (n - 1)
    Ed = expm(Aa * dt)
    X = np.empty((n, k))
    z = np.r_[np.asarray(x0, float), 1.0]
    for i in range(n):
        X[i] = z[:k]
        z = Ed @ z
    return np.linspace(0, t_end, n), X


def gen_force_control(seed: int, difficulty: int) -> Problem:
    rng = _rng("force_control", seed, difficulty)
    st, steps = Doc(), []
    if difficulty <= 2:
        if difficulty == 1:
            M = float(_pick(rng, [1.0, 2.0, 4.0]))
            wn = float(_pick(rng, [5.0, 10.0, 20.0]))
            zeta = float(_pick(rng, [0.25, 0.5, 0.7, 1.0, 1.5]))
            K, B = M * wn**2, 2 * zeta * wn * M
            f = float(_pick(rng, [5.0, 10.0, 20.0, 40.0]))
            st.p("An impedance-controlled robot renders, along one task direction $x$, the virtual mass–spring–damper "
                 f"$M\\ddot{{x}} + B\\dot{{x}} + Kx = f_{{ext}}$ with $M = {num(M)}$ kg, $B = {num(B)}$ N·s/m and "
                 f"$K = {num(K)}$ N/m ($x$ is the end-effector's displacement from its rest position, $f_{{ext}}$ the "
                 f"force a user applies to it).  At rest at $x = 0$ the user suddenly pushes with a constant "
                 f"$f_{{ext}} = {num(f)}$ N.")
            st.p("**(a)** Find the steady-state deflection.  **(b)** Find $\\omega_n$ and $\\zeta$ and classify the "
                 "response.  **(c)** Give the overshoot beyond the steady state and the 2% settling time (MR's rule of "
                 "thumb).")
            s1 = Doc()
            s1.p("At steady state $\\ddot{x} = \\dot{x} = 0$, so only the virtual spring balances the push:")
            s1.eq(r"x_{ss} = \frac{f_{ext}}{K} = \frac{" + num(f) + "}{" + num(K) + "} = " + num(f / K, 5)
                  + r"\ \mathrm{m}")
            steps.append(Step("Steady-state deflection", s1))
            s2 = Doc()
            s2.p("Dividing by $M$ gives the standard second-order form "
                 "$\\ddot{x} + 2\\zeta\\omega_n\\dot{x} + \\omega_n^{2}x = f_{ext}/M$ (as in Sec. 11.2.2):")
            s2.eq(r"\omega_n = \sqrt{\frac{K}{M}} = " + num(wn, 4) + r"\ \mathrm{rad/s},\quad \zeta = "
                  r"\frac{B}{2\sqrt{KM}} = " + num(zeta, 4))
            steps.append(Step("Natural frequency and damping ratio", s2))
            data = dict(M=M, B=B, K=K, f=f)
            title = "Impedance control: deflection and transient"
        else:
            M = float(_pick(rng, [1.0, 2.0, 4.0]))
            f = float(_pick(rng, [10.0, 20.0, 30.0, 50.0]))
            delta = float(_pick(rng, [0.01, 0.02, 0.05]))
            zeta = float(_pick(rng, [0.5, 0.7, 1.0]))
            K = f / delta
            B = 2 * zeta * np.sqrt(K * M)
            wn = np.sqrt(K / M)
            st.p("You program an impedance controller to render $M\\ddot{x} + B\\dot{x} + Kx = f_{ext}$ along one "
                 f"direction with apparent mass $M = {num(M)}$ kg.  Specification: a steady push of {num(f)} N must "
                 f"deflect the end-effector by {num(100 * delta)} cm, and the response must have damping ratio "
                 f"$\\zeta = {num(zeta)}$.")
            st.p("**(a)** Choose the rendered stiffness $K$ and damping $B$.  **(b)** Find $\\omega_n$, the overshoot "
                 "and the 2% settling time (rule of thumb) of the deflection after a step push.")
            s1 = Doc()
            s1.p("At steady state $Kx_{ss} = f_{ext}$, so the stiffness follows from the allowed deflection; then "
                 "$\\zeta = B/(2\\sqrt{KM})$ fixes the damping:")
            s1.eq(r"K = \frac{f_{ext}}{x_{ss}} = \frac{" + num(f) + "}{" + num(delta) + "} = " + num(K, 4)
                  + r"\ \mathrm{N/m},\quad B = 2\zeta\sqrt{KM} = " + num(B, 4) + r"\ \mathrm{N\cdot s/m}")
            steps.append(Step("Stiffness and damping", s1))
            s2 = Doc()
            s2.eq(r"\omega_n = \sqrt{\frac{K}{M}} = " + num(wn, 4) + r"\ \mathrm{rad/s}")
            steps.append(Step("Natural frequency", s2))
            data = dict(M=M, f=f, delta=delta, zeta_req=zeta)
            title = "Impedance control: rendering a stiffness"
        s3 = Doc()
        s3.p("The deviation $x - x_{ss}$ obeys the free equation with initial value $-x_{ss}$, so the step response "
             "has the same overshoot and settling time as the error dynamics of Sec. 11.2.2.")
        os_, ts = _response_doc(s3, zeta, wn, "The deflection")
        steps.append(Step("Transient", s3))
        xss = f / K
        ans = dict(x_ss=xss, wn=wn, zeta=zeta, overshoot=os_, ts=ts) if difficulty == 1 else \
            dict(K_imp=K, B_imp=B, wn=wn, overshoot=os_, ts=ts)
        ad = Doc()
        ad.p((f"$K = {num(K, 4)}$ N/m, $B = {num(B, 4)}$ N·s/m; " if difficulty == 2 else "") +
             f"$x_{{ss}} = {num(xss, 5)}$ m, $\\omega_n = {num(wn, 4)}$ rad/s, $\\zeta = {num(zeta, 4)}$ "
             f"({_classify(zeta)}); overshoot {num(100 * os_, 2)} %, $t_s \\approx {num(ts, 4)}$ s.")
        t = np.linspace(0, 1.5 * ts + 2 * np.pi / wn, 1200)
        vis = dict(kind="plot", title="deflection x(t) after a step push", ylabel="x [m]", t=t,
                   series=[("x(t)", xss * (1 - _step_error(zeta, wn, t))), ("x_ss", np.full_like(t, xss))])
        return Problem("force_control", seed, difficulty, title, "Ch. 11 (Sec. 11.7)", st, data, steps, ans, ad, vis)

    # difficulty 3: PI force control of a 1-DOF robot against a stiff environment (Sec. 11.5)
    m = float(_pick(rng, [1.0, 2.0, 5.0]))
    ke = float(_pick(rng, [1000.0, 2000.0, 5000.0]))
    kfp = float(_pick(rng, [0.5, 1.0, 2.0, 4.0]))
    kd = float(_pick(rng, [20.0, 50.0, 100.0]))
    fd = float(_pick(rng, [5.0, 10.0, 20.0]))
    dist = float(_pick(rng, [-3.0, -2.0, 2.0, 3.0, 5.0]))
    kfi_max = kd * (1 + kfp) / m
    while True:
        kfi = float(_pick(rng, [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]))
        if not 0.8 < kfi / kfi_max < 1.25:
            break
    st.p(f"A 1-DOF robot of mass $\\mathfrak{{m}} = {num(m)}$ kg presses on a stiff surface modelled as a spring "
         f"$k_e = {num(ke)}$ N/m: with $x \\geq 0$ the penetration, the force the robot applies to the environment "
         "(measured by its force sensor) is $f_{tip} = k_ex$.  Gravity is compensated, but an unmodelled constant "
         f"force $d = {num(dist)}$ N also acts on the robot.  The force controller is")
    st.eq(r"f = f_d + k_{fp}f_e + k_{fi}\int f_e\,dt - k_{damp}\dot{x},\quad f_e = f_d - f_{tip},\quad "
          r"\mathfrak{m}\ddot{x} = f + d - k_ex")
    st.p(f"with $f_d = {num(fd)}$ N, $k_{{fp}} = {num(kfp)}$, $k_{{damp}} = {num(kd)}$ N·s/m.  **(a)** With "
         "$k_{fi} = 0$: find the steady-state contact force and force error, and $\\omega_n$, $\\zeta$ of the contact "
         f"dynamics.  **(b)** With $k_{{fi}} = {num(kfi)}$ 1/s: find the steady-state contact force and the "
         "closed-loop characteristic polynomial.  **(c)** For which $k_{fi}$ is the loop stable?  Is the given gain stable?")
    s1 = Doc()
    s1.p("With $k_{fi} = 0$ and $f_{tip} = k_ex$, setting $\\ddot{x} = \\dot{x} = 0$:")
    s1.eq(r"0 = f_d + k_{fp}(f_d - f_{tip}) + d - f_{tip}\ \Rightarrow\ f_{tip} = f_d + \frac{d}{1+k_{fp}}")
    fP = fd + dist / (1 + kfp)
    wnP = np.sqrt(ke * (1 + kfp) / m)
    zP = kd / (2 * np.sqrt(m * ke * (1 + kfp)))
    s1.p(f"$f_{{tip}} = {num(fP, 4)}$ N, force error $f_e = -d/(1+k_{{fp}}) = {num(fd - fP, 4)}$ N: the proportional "
         "gain only divides the disturbance.  The dynamics are "
         "$\\mathfrak{m}\\ddot{x} + k_{damp}\\dot{x} + (1+k_{fp})k_ex = (1+k_{fp})f_d + d$, so")
    s1.eq(r"\omega_n = \sqrt{\frac{(1+k_{fp})k_e}{\mathfrak{m}}} = " + num(wnP, 4) + r"\ \mathrm{rad/s},\quad \zeta = "
          r"\frac{k_{damp}}{2\sqrt{\mathfrak{m}(1+k_{fp})k_e}} = " + num(zP, 4))
    steps.append(Step("(a) Proportional force control", s1))
    s2 = Doc()
    s2.p("At a steady state the integrator input must vanish, $f_e = 0$, so $f_{tip} = f_d$ whatever $d$ is.  "
         "Differentiating the equation of motion once ($f_d$ and $d$ constant, $\\dot{f}_{tip} = k_e\\dot{x}$):")
    poly = np.array([m, kd, ke * (1 + kfp), kfi * ke])
    s2.eq(r"\mathfrak{m}s^{3} + k_{damp}s^{2} + (1+k_{fp})k_es + k_{fi}k_e = 0")
    s2.p(f"Coefficients $({num(poly[0])},\\ {num(poly[1])},\\ {num(poly[2])},\\ {num(poly[3])})$.")
    steps.append(Step("(b) Integral action", s2))
    s3 = Doc()
    stable = kfi < kfi_max
    s3.p("A cubic $a_3s^{3} + a_2s^{2} + a_1s + a_0$ with positive coefficients has all roots in the left half-plane "
         "iff $a_2a_1 > a_3a_0$ (Routh–Hurwitz):")
    s3.eq(r"k_{damp}(1+k_{fp})k_e > \mathfrak{m}k_{fi}k_e\ \Leftrightarrow\ k_{fi} < \frac{k_{damp}(1+k_{fp})}"
          r"{\mathfrak{m}} = " + num(kfi_max, 4) + r"\ \mathrm{1/s}")
    roots = np.roots(poly)
    s3.p(f"The given $k_{{fi}} = {num(kfi)}$ is {'below' if stable else 'above'} the limit: the loop is "
         f"**{'stable' if stable else 'unstable'}** (roots {', '.join(_root_txt(r) for r in roots)}).  Note the limit "
         "does not depend on $k_e$: a stiffer environment speeds the loop up but the integral gain is capped by the "
         "damping.")
    steps.append(Step("(c) Stability of the PI loop", s3))
    ans = dict(f_tip_P=fP, fe_P=fd - fP, wn_P=wnP, zeta_P=zP, f_tip_PI=fd, char_poly=poly, kfi_max=kfi_max,
               stable=bool(stable))
    ad = Doc().p(f"(a) $f_{{tip}} = {num(fP, 4)}$ N (error ${num(fd - fP, 4)}$ N), $\\omega_n = {num(wnP, 4)}$ rad/s, "
                 f"$\\zeta = {num(zP, 4)}$.  (b) $f_{{tip}} = f_d = {num(fd)}$ N; ${num(m)}s^{{3}} + {num(kd)}s^{{2}} + "
                 f"{num(poly[2])}s + {num(poly[3])} = 0$.  (c) $k_{{fi}} < {num(kfi_max, 4)}$ 1/s; the given gain is "
                 f"{'stable' if stable else 'unstable'}.")
    data = dict(m=m, ke=ke, kfp=kfp, kdamp=kd, fd=fd, d=dist, kfi=kfi)
    A3, b3 = _pi_force_system(data)
    sigma = float(np.max(roots.real))
    t_end = min(3.0, max(8 / (zP * wnP), 5 * (1 + kfp) / kfi)) if stable else min(3.0, np.log(50.0) / sigma)
    t, X = _lin_response(A3, b3, np.zeros(3), t_end)
    A2, b2 = _pi_force_system(data, integral=False)
    _, X2 = _lin_response(A2, b2, np.zeros(2), t_end)
    vis = dict(kind="plot", title="contact force f_tip(t) from first touch (linear model, contact kept)", ylabel="f_tip [N]", t=t,
               series=[("PI", ke * X[:, 0]), ("P only", ke * X2[:, 0]), ("f_d", np.full_like(t, fd))])
    return Problem("force_control", seed, difficulty, "PI force control against a stiff surface",
                   "Ch. 11 (Sec. 11.5)", st, data, steps, ans, ad, vis)


def _root_txt(r) -> str:
    r = complex(r)
    if abs(r.imag) < 1e-9:
        return num(r.real, 3)
    return f"{num(r.real, 3)} {'+' if r.imag > 0 else '−'} {num(abs(r.imag), 3)}j"


def _pi_force_system(d: dict, integral: bool = True, kfi=None):
    """State-space model of the 1-DOF force loop: states (x, x-dot[, integral of f_e])."""
    m, ke, kfp, kd, fd, dist = d["m"], d["ke"], d["kfp"], d["kdamp"], d["fd"], d["d"]
    kfi = d["kfi"] if kfi is None else kfi
    if not integral:
        return (np.array([[0.0, 1.0], [-(1 + kfp) * ke / m, -kd / m]]),
                np.array([0.0, ((1 + kfp) * fd + dist) / m]))
    return (np.array([[0.0, 1.0, 0.0], [-(1 + kfp) * ke / m, -kd / m, kfi / m], [-ke, 0.0, 0.0]]),
            np.array([0.0, ((1 + kfp) * fd + dist) / m, fd]))


def _verify_force_control(p: Problem) -> float:
    """Eigenvalues of the state matrix (omega_n, zeta, settling rule 4/min|Re s|), steady states by a linear solve,
    overshoot by solve_ivp; for the PI loop np.poly of the state matrix and the stability limit by root finding."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    d, a = p.data, p.answer
    if "kfi" in d:
        A2, b2 = _pi_force_system(d, integral=False)
        x2 = np.linalg.solve(A2, -b2)
        lam = np.linalg.eigvals(A2)
        wn = np.sqrt(np.real(np.prod(lam)))
        A3, b3 = _pi_force_system(d)
        x3 = np.linalg.solve(A3, -b3)
        e = max(abs(d["ke"] * x2[0] - a["f_tip_P"]), abs(d["fd"] - d["ke"] * x2[0] - a["fe_P"]),
                abs(wn - a["wn_P"]), abs(-np.real(lam.sum()) / (2 * wn) - a["zeta_P"]),
                abs(d["ke"] * x3[0] - a["f_tip_PI"]), np.abs(d["m"] * np.poly(A3) - a["char_poly"]).max())
        e = max(e, float((np.linalg.eigvals(A3).real.max() < 0) != a["stable"]))

        def growth(k):
            return float(np.linalg.eigvals(_pi_force_system(d, kfi=k)[0]).real.max())
        lim = brentq(growth, 0.5 * a["kfi_max"], 2 * a["kfi_max"], xtol=1e-12, rtol=1e-14)
        return float(max(e, abs(lim - a["kfi_max"]) / max(1.0, a["kfi_max"])))
    M = d["M"]
    K, B = (d["K"], d["B"]) if "B" in d else (a["K_imp"], a["B_imp"])   # design: the spec is checked below
    A = np.array([[0.0, 1.0], [-K / M, -B / M]])
    b = np.array([0.0, d["f"] / M])
    lam = np.linalg.eigvals(A)
    wn = np.sqrt(np.real(np.prod(lam)))
    zeta = -np.real(lam.sum()) / (2 * wn)
    xss = np.linalg.solve(A, -b)[0]
    e = max(abs(wn - a["wn"]), abs(4 / np.abs(lam.real).min() - a["ts"]))
    if "x_ss" in a:
        e = max(e, abs(xss - a["x_ss"]), abs(zeta - a["zeta"]))
    else:
        e = max(e, abs(xss - d["delta"]), abs(zeta - d["zeta_req"]))
    if zeta < 1 - 1e-6:                     # genuinely complex roots (a double root may come out with tiny j)
        sol = solve_ivp(lambda t, x: A @ x + b, (0, 3 * np.pi / np.abs(lam.imag).max()), [0.0, 0.0], rtol=1e-11,
                        atol=1e-13, dense_output=True)
        tt = np.linspace(0, sol.t[-1], 20001)
        e = max(e, abs(sol.sol(tt)[0].max() / xss - 1 - a["overshoot"]))
    else:
        e = max(e, abs(a["overshoot"]))
    return float(e)


# -- Ch. 12: wrench matrix, form and force closure, minimum friction --------------------------------

#: candidate planar contacts on the square body with corners (+-1, +-1): edge -> (point(s), inward normal)
_EDGES = {"left": (lambda s: (-1.0, s), (1.0, 0.0)), "right": (lambda s: (1.0, s), (-1.0, 0.0)),
          "bottom": (lambda s: (s, -1.0), (0.0, 1.0)), "top": (lambda s: (s, 1.0), (0.0, -1.0))}
_OFFS = [-0.5, 0.0, 0.5]
_MUS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5]


def _contact(edge: str, s: float, mu: float, name: str):
    pt, n = _EDGES[edge]
    return GR.Contact(pt(s), n, mu, name)


def _contacts_table(cs) -> tuple[list[str], list[list[str]]]:
    return (["contact", "point $p_i$", "inward normal $\\hat{n}_i$"],
            [[c.name, vec(c.p), vec(c.n)] for c in cs])


def _grasp_visual(cs, friction: bool) -> dict:
    lines = [dict(x=[-1, 1, 1, -1, -1], y=[-1, -1, 1, 1, -1], name="body", color=5, width=2)]
    for k, c in enumerate(cs):
        tail = c.p - 0.45 * c.n
        lines.append(dict(x=[tail[0], c.p[0]], y=[tail[1], c.p[1]], name=f"finger {c.name}", color=k, width=3))
        if friction and c.mu > 0:
            for edge in GR.cone_edges(c):
                tip = c.p + 0.7 * edge
                lines.append(dict(x=[c.p[0], tip[0]], y=[c.p[1], tip[1]], color=k, dash=True, width=1.5))
    return dict(kind="xy", title="contacts (solid) and friction-cone edges (dashed)", lines=lines)


def _alt_closure(F) -> bool:
    """Positive span of the columns = R^3 iff rank 3 and some strictly positive k has F k = 0: maximise the smallest
    weight t subject to F k = 0, sum k = 1, k_i >= t (a different LP from MR's 12.14 test in grasping.py)."""
    from scipy.optimize import linprog
    F = np.asarray(F, float)
    if np.linalg.matrix_rank(F, tol=1e-9) < 3:
        return False
    j = F.shape[1]
    c = np.r_[np.zeros(j), -1.0]
    A_eq = np.vstack([np.c_[F, np.zeros(3)], np.r_[np.ones(j), 0.0]])
    A_ub = np.c_[-np.eye(j), np.ones(j)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(j), A_eq=A_eq, b_eq=np.r_[np.zeros(3), 1.0],
                  bounds=[(0, None)] * j + [(None, 1.0)], method="highs")
    return res.status == 0 and -res.fun > 1e-9


def _two_finger_mu(p1, n1, p2, n2) -> float:
    """Two-finger force closure: the line between the contacts must lie inside both friction cones, so
    mu_min = max tan(angle between each inward normal and the direction to the other contact)."""
    d = (p2 - p1) / np.linalg.norm(p2 - p1)
    out = 0.0
    for n, u in ((n1, d), (n2, -d)):
        c = float(n @ u)
        out = max(out, np.sqrt(max(0.0, 1 - c * c)) / c)
    return out


def gen_grasp_closure(seed: int, difficulty: int) -> Problem:
    rng = _rng("grasp_closure", seed, difficulty)
    st, steps = Doc(), []
    st.p("A planar rigid body is the square with corners $(\\pm 1, \\pm 1)$ (lengths in any unit; closure does not "
         "depend on it).  Fingers touch it at stationary point contacts; each has a contact point $p_i$ and an inward "
         "unit normal $\\hat{n}_i$ (pointing into the body).  Planar wrenches are $\\mathcal{F} = (m_z, f_x, f_y)$.")
    if difficulty == 1:
        want = bool(rng.random() < 0.5)
        while True:
            if want or rng.random() < 0.4:
                edges = ["left", "right", "bottom", "top"]
            else:
                edges = [_pick(rng, list(_EDGES)) for _ in range(4)]
            cs = [_contact(e, float(_pick(rng, _OFFS)), 0.0, str(i + 1)) for i, e in enumerate(edges)]
            if len({(tuple(c.p), tuple(c.n)) for c in cs}) < 4:
                continue
            res = GR.form_closure(cs)
            if res.closed == want:
                break
        st.table(*_contacts_table(cs))
        st.p("The contacts are frictionless.  **(a)** Build the wrench matrix $F = [\\mathcal{F}_1\\ \\cdots\\ "
             "\\mathcal{F}_4]$ of the contact normals.  **(b)** What is its rank?  **(c)** Does the grasp have "
             "first-order form closure?  Justify with a positive combination $Fk = 0$ or with a twist that breaks "
             "the grasp.")
        F = GR.wrench_matrix(cs, friction=False)
        s1 = Doc()
        s1.p("A unit force $\\hat{n}_i$ at $p_i$ has the wrench $\\mathcal{F}_i = (p_{ix}n_{iy} - p_{iy}n_{ix},\\ "
             "n_{ix},\\ n_{iy})$:")
        s1.mat("F", F, 3)
        rank = int(np.linalg.matrix_rank(F))
        s1.p(f"$\\mathrm{{rank}}\\,F = {rank}$" + (" — less than 3, so some twist leaves every contact "
                                                    "inactive and the grasp cannot be in form closure." if rank < 3 else "."))
        steps.append(Step("Wrench matrix", s1))
        s2 = Doc()
        s2.p("First-order form closure (Sec. 12.1.7) holds iff the columns of $F$ positively span $\\mathbb{R}^{3}$: "
             "rank 3 and a solution of $Fk = 0$ with every $k_i \\geq 1$ (the linear program of Sec. 12.1.7).")
        if res.closed:
            k = res.k / res.k.min()
            s2.p(f"The LP returns $k = {vec(k, 3)}$ and $Fk = {vec(F @ k, 6)}$: the contact normals can balance each "
                 "other with strictly positive forces, so **form closure** holds (four contacts, the minimum for a "
                 "planar body).")
        else:
            V = GR.nonzero_twist(cs)
            V = V / np.abs(V).max()
            s2.p(f"No such $k$ exists (LP infeasible{', rank deficient' if rank < 3 else ''}).  For example the twist "
                 f"$\\mathcal{{V}} = (\\omega_z, v_x, v_y) = {vec(V, 3)}$ has $F^{{T}}\\mathcal{{V}} = {vec(F.T @ V, 3)}$ "
                 "≥ 0: no contact is penetrated, so the body can move — **no form closure**.")
        steps.append(Step("Form closure test", s2))
        ans = dict(F=F, rank=rank, closed=bool(res.closed))
        if res.closed:
            ans["k_lp"] = [float(x) for x in res.k]          # not unique: shown, not graded (a list)
        ad = Doc().p(f"rank $F = {rank}$; {'form closure' if res.closed else 'no form closure'}.")
        data = dict(contacts=[(c.p.copy(), c.n.copy()) for c in cs], mu=0.0)
        title = "Form closure with frictionless contacts"
        return Problem("grasp_closure", seed, difficulty, title, "Ch. 12 (Sec. 12.1.7)", st, data, steps, ans, ad,
                       _grasp_visual(cs, False))

    nc = 2 if difficulty == 2 else 3
    while True:
        edges = [str(x) for x in rng.permutation(list(_EDGES))[:nc]]
        cs = [_contact(e, float(_pick(rng, _OFFS)), 0.0, str(i + 1)) for i, e in enumerate(edges)]
        if nc == 2:
            if abs(cs[0].n @ cs[1].n + 1) < 1e-9 and abs(cs[0].tangent @ (cs[1].p - cs[0].p)) < 1e-9:
                continue                   # collinear antipodal fingers: any mu > 0 works, the minimum is not attained
            mu_min = GR.min_friction(cs, tol=1e-10)
        else:
            mu_min = GR.min_friction(cs, mu_max=10.0, tol=1e-10)
        if not np.isfinite(mu_min) or mu_min > 3.0 or mu_min < 0.05:
            continue
        want = bool(rng.random() < 0.5)                    # half the problems in closure, half not
        cands = [m for m in _MUS if (m > mu_min) == want and abs(m - mu_min) > 0.08 * max(1.0, mu_min)]
        if cands:
            mu = float(_pick(rng, cands))
            break
    cs = GR.with_mu(cs, mu)
    st.table(*_contacts_table(cs))
    st.p(f"Every contact has Coulomb friction $\\mu = {num(mu)}$.  **(a)** Write the unit edges of each friction cone and "
         f"the $3\\times {2 * nc}$ wrench matrix $F$ of the cone edges.  **(b)** Is the grasp in force closure?  "
         "**(c)** Find the smallest friction coefficient $\\mu_{min}$ (the same at every contact) that gives force "
         "closure.")
    F = GR.wrench_matrix(cs, friction=True)
    s1 = Doc()
    s1.p("A planar friction cone is spanned by its two unit edges $(\\hat{n}_i \\pm \\mu\\hat{t}_i)/\\sqrt{1+\\mu^{2}}$, "
         "$\\hat{t}_i$ the normal turned $+90^{\\circ}$; each edge force at $p_i$ gives one column $([p_i]f, f)$:")
    s1.table(["contact", "$\\hat{t}_i$", "edge forces"],
             [[c.name, vec(c.tangent), ",  ".join(vec(e, 4) for e in GR.cone_edges(c))] for c in cs])
    s1.mat("F", F, 4)
    steps.append(Step("Friction cones and wrench matrix", s1))
    res = GR.force_closure(cs)
    s2 = Doc()
    s2.p("Force closure (Sec. 12.2.3) holds iff the cone-edge wrenches positively span $\\mathbb{R}^{3}$, which is "
         "the same rank-and-LP test as form closure, applied to the edge columns (exact in the plane).")
    s2.p(f"$\\mathrm{{rank}}\\,F = {np.linalg.matrix_rank(F)}$; the LP is {'feasible' if res.closed else 'infeasible'}: "
         + ("**force closure**." if res.closed else "**no force closure** at this $\\mu$."))
    steps.append(Step("(b) Force closure", s2))
    s3 = Doc()
    if nc == 2:
        d12 = cs[1].p - cs[0].p
        angs = [np.degrees(np.arccos(cs[0].n @ d12 / np.linalg.norm(d12))),
                np.degrees(np.arccos(-cs[1].n @ d12 / np.linalg.norm(d12)))]
        s3.p("Two frictional fingers hold a planar body in force closure iff the segment joining the contacts lies "
             "inside both friction cones (each finger can push along the line towards the other).  The segment "
             f"$p_2 - p_1 = {vec(d12)}$ makes {num(angs[0], 2)}° with $\\hat{{n}}_1$ and {num(angs[1], 2)}° with "
             "$\\hat{n}_2$, and a cone of half-angle $\\arctan\\mu$ contains it when $\\mu \\geq \\tan$ of that angle:")
        s3.eq(r"\mu_{min} = \max\left(\tan " + mdeg(np.radians(angs[0])) + r",\ \tan " + mdeg(np.radians(angs[1]))
              + r"\right) = " + num(mu_min, 4))
    else:
        s3.p("The cones only grow with $\\mu$, so force closure is monotone in $\\mu$; bisection on the LP test "
             f"(`grasping.min_friction`) gives $\\mu_{{min}} = {num(mu_min, 4)}$.  Below it some twist still keeps "
             "every contact feasible; above it the cone edges positively span the plane's wrenches.")
    s3.p(f"With $\\mu = {num(mu)}$ the grasp is {'above' if mu > mu_min else 'below'} the limit, consistent with (b).")
    steps.append(Step("(c) Minimum friction coefficient", s3))
    ans = dict(F=F, closed=bool(res.closed), mu_min=float(mu_min))
    ad = Doc().p(f"{'Force closure' if res.closed else 'No force closure'} at $\\mu = {num(mu)}$; "
                 f"$\\mu_{{min}} = {num(mu_min, 4)}$.")
    data = dict(contacts=[(c.p.copy(), c.n.copy()) for c in cs], mu=mu)
    title = "Force closure with two frictional fingers" if nc == 2 else "Force closure with three frictional contacts"
    return Problem("grasp_closure", seed, difficulty, title, "Ch. 12 (Secs. 12.1.7, 12.2.3)", st, data, steps, ans, ad,
                   _grasp_visual(cs, True))


def _verify_grasp_closure(p: Problem) -> float:
    """Wrenches rebuilt from the geometry; closure by a max-min-weight LP (not 12.14) and, for frictionless contacts,
    by the dual twist test; two fingers: the angle rule; three: closure just above and not just below mu_min."""
    d, a = p.data, p.answer
    mu = d["mu"]
    cols = []
    for pt, n in d["contacts"]:
        if mu > 0:
            t = np.array([-n[1], n[0]])
            fs = [(n + mu * t) / np.sqrt(1 + mu * mu), (n - mu * t) / np.sqrt(1 + mu * mu)]
        else:
            fs = [n]
        cols += [np.r_[_cross2(pt, f), f] for f in fs]
    F = np.array(cols).T
    e = max(np.abs(F - a["F"]).max(), float(_alt_closure(F) != a["closed"]))
    if mu == 0:
        cs = [GR.Contact(pt, n) for pt, n in d["contacts"]]
        e = max(e, abs(np.linalg.matrix_rank(F) - a["rank"]), float((GR.nonzero_twist(cs) is None) != a["closed"]))
        if "k_lp" in a:
            k = np.asarray(a["k_lp"])
            e = max(e, np.abs(F @ k).max(), max(0.0, 1 - k.min()))
        return float(e)
    if len(d["contacts"]) == 2:
        (p1, n1), (p2, n2) = d["contacts"]
        return float(max(e, abs(_two_finger_mu(p1, n1, p2, n2) - a["mu_min"])))

    def closed_at(m):
        cs = [GR.Contact(pt, n, m) for pt, n in d["contacts"]]
        return _alt_closure(GR.wrench_matrix(cs, friction=True))
    m0 = a["mu_min"]
    return float(max(e, float(not closed_at(m0 * (1 + 1e-4))), float(closed_at(m0 * (1 - 1e-4)))))


# -- Ch. 13: wheeled mobile robots ------------------------------------------------------------------

def _wheel_speed(wh, Vb) -> float:
    """Driving speed of one wheel for a chassis twist V_b = (omega, v_x, v_y): velocity of the wheel centre
    v + omega x p in {b}, expressed along / across the driving direction beta; the rollers at gamma let the wheel
    slide across, so u = (v_drive + tan(gamma) v_across) / r."""
    w, vx, vy = Vb
    v = np.array([vx - w * wh.y, vy + w * wh.x])
    drive = np.array([np.cos(wh.beta), np.sin(wh.beta)])
    across = np.array([-np.sin(wh.beta), np.cos(wh.beta)])
    return float((drive @ v + np.tan(wh.gamma) * (across @ v)) / wh.r)


def _H_from_rigid_body(wheels) -> np.ndarray:
    return np.array([[_wheel_speed(wh, e) for e in np.eye(3)] for wh in wheels])


def _chassis_visual(kind: str, wheels, size, Vb) -> dict:
    if kind == "omni3":
        a = np.linspace(0, 2 * np.pi, 61)
        lines = [dict(x=list(size * np.cos(a)), y=list(size * np.sin(a)), name="chassis", color=5, width=2)]
    else:
        hx, hy = size
        lines = [dict(x=[-hx, hx, hx, -hx, -hx], y=[-hy, -hy, hy, hy, -hy], name="chassis", color=5, width=2)]
    for k, wh in enumerate(wheels):
        dvec = np.array([np.cos(wh.beta), np.sin(wh.beta)]) * 0.7 * wh.r
        lines.append(dict(x=[wh.x - dvec[0], wh.x + dvec[0]], y=[wh.y - dvec[1], wh.y + dvec[1]],
                          name=f"wheel {k + 1}", color=k, width=5))
    v = np.asarray(Vb[1:], float)
    if np.linalg.norm(v) > 1e-9:
        tip = 0.6 * (size if np.isscalar(size) else max(size)) * v / np.linalg.norm(v)
        lines.append(dict(x=[0, tip[0]], y=[0, tip[1]], name="(v_x, v_y)", color=6, width=2, dash=True))
    return dict(kind="xy", title="chassis frame {b}: wheels and driving directions", lines=lines)


def _wheel_table(wheels) -> tuple[list[str], list[list[str]]]:
    return (["wheel", "$(x_i, y_i)$ in {b} [m]", "$\\beta_i$", "$\\gamma_i$", "$r_i$ [m]"],
            [[str(k + 1), vec((wh.x, wh.y), 4), f"{num(np.degrees(wh.beta), 2)}°", f"{num(np.degrees(wh.gamma), 2)}°",
              num(wh.r)] for k, wh in enumerate(wheels)])


def gen_mobile_kin(seed: int, difficulty: int) -> Problem:
    rng = _rng("mobile_kin", seed, difficulty)
    st, steps = Doc(), []
    r = float(_pick(rng, [0.05, 0.1]))
    phi = int(rng.integers(-5, 7)) * 30 * DEG
    if difficulty == 1:
        dd = float(_pick(rng, [0.2, 0.25, 0.3]))
        v = float(_pick(rng, [-0.5, 0.2, 0.3, 0.5, 1.0]))
        w = float(_pick(rng, [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        uL2 = float(_pick(rng, [-4.0, 2.0, 4.0, 5.0, 6.0, 8.0]))
        uR2 = float(_pick(rng, [2.0, 4.0, 6.0, 8.0, 10.0]))
        if uR2 == uL2:
            uR2 += 2.0
        uL, uR = MB.diff_drive_wheel_speeds(v, w, r, dd)
        qd = MB.diff_drive_qdot(phi, uL, uR, r, dd)
        Vb2 = MB.F_diff_drive_book(r, dd) @ np.array([uL2, uR2])
        st.p(f"A differential-drive robot has two driven wheels of radius $r = {num(r)}$ m, a distance $2d = "
             f"{num(2 * dd)}$ m apart; the chassis frame {{b}} is midway between them with $\\hat{{x}}_b$ pointing "
             "forward, and the canonical controls are the forward speed $v$ and the yaw rate $\\omega$.")
        st.p(f"**(a)** Find the wheel speeds $(u_L, u_R)$ (rad/s) for $v = {num(v)}$ m/s, $\\omega = {num(w)}$ rad/s.  "
             f"**(b)** With the chassis at heading $\\phi = {mdeg(phi)}$, give $\\dot{{q}} = (\\dot{{\\phi}}, \\dot{{x}}, "
             f"\\dot{{y}})$.  **(c)** The encoders read $u_L = {num(uL2)}$, $u_R = {num(uR2)}$ rad/s: find $v$, $\\omega$ and "
             "the radius of the circle the robot drives.")
        s1 = Doc()
        s1.p("A wheel centre at lateral offset $\\pm d$ moves forward at $v \\mp \\omega d$ (left wheel at $+d$), and "
             "rolling means $r u = $ forward speed (Sec. 13.3):")
        s1.eq(r"u_L = \frac{v - \omega d}{r} = " + num(uL, 4) + r",\quad u_R = \frac{v + \omega d}{r} = " + num(uR, 4)
              + r"\ \mathrm{rad/s}")
        steps.append(Step("(a) Wheel speeds", s1))
        s2 = Doc()
        s2.p("The chassis twist is $\\mathcal{V}_b = (\\omega, v, 0)$ (no sideways slip); rotating into {s}:")
        s2.eq(r"\dot{q} = \left(\omega,\ v\cos\phi,\ v\sin\phi\right) = " + vec(qd, 4))
        steps.append(Step("(b) Chassis velocity", s2))
        s3 = Doc()
        s3.p("Inverting (a), $v = r(u_L + u_R)/2$ and $\\omega = r(u_R - u_L)/(2d)$:")
        s3.eq(r"v = " + num(Vb2[1], 4) + r"\ \mathrm{m/s},\quad \omega = " + num(Vb2[0], 4) + r"\ \mathrm{rad/s}")
        s3.p(f"Turning radius $|v/\\omega| = {num(abs(Vb2[1] / Vb2[0]), 4)}$ m, turning "
             f"{'left (counter-clockwise)' if Vb2[0] * Vb2[1] > 0 else 'right (clockwise)'} while driving "
             f"{'forward' if Vb2[1] > 0 else 'backward'}.")
        steps.append(Step("(c) Speeds from the encoders", s3))
        ans = dict(u_LR=np.array([uL, uR]), qdot=qd, v_enc=float(Vb2[1]), omega_enc=float(Vb2[0]))
        ad = Doc().p(f"(a) $(u_L, u_R) = {vec((uL, uR), 4)}$ rad/s.  (b) $\\dot{{q}} = {vec(qd, 4)}$.  (c) "
                     f"$v = {num(Vb2[1], 4)}$ m/s, $\\omega = {num(Vb2[0], 4)}$ rad/s.")
        data = dict(kind="diff", r=r, d=dd, v=v, omega=w, phi=phi, u_enc=np.array([uL2, uR2]))
        wheels = MB.diff_drive_wheels(r, dd)
        vis = _chassis_visual("diff", wheels, (0.6 * dd + r, 1.2 * dd), (w, v, 0.0))
        return Problem("mobile_kin", seed, difficulty, "Differential-drive kinematics", "Ch. 13 (Sec. 13.3)", st,
                       data, steps, ans, ad, vis)

    while True:
        Vb = np.array([_pick(rng, [-1.0, -0.5, 0.0, 0.5, 1.0]), _pick(rng, [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0]),
                       _pick(rng, [-0.5, -0.2, 0.0, 0.2, 0.5])])
        if np.count_nonzero(Vb) >= 2:
            break
    if difficulty == 2:
        dd = float(_pick(rng, [0.2, 0.25, 0.3]))
        wheels = MB.three_omni_wheels(r, dd)
        H0 = MB.H0_three_omni_book(r, dd)
        qd = np.array([_pick(rng, [-0.5, 0.0, 0.5, 1.0]), _pick(rng, [-0.4, 0.0, 0.3, 0.4]),
                       _pick(rng, [-0.3, 0.2, 0.4, 0.5])])
        st.p(f"A robot with three omniwheels (free rollers, $\\gamma_i = 0$) at distance $d = {num(dd)}$ m from the "
             "centre of its chassis frame {b}, as in MR Fig. 13.5:")
        st.table(*_wheel_table(wheels))
        st.p(f"**(a)** Build $H(0)$, the matrix with $u = H(0)\\mathcal{{V}}_b$.  **(b)** Find the wheel speeds for "
             f"$\\mathcal{{V}}_b = (\\omega_{{bz}}, v_{{bx}}, v_{{by}}) = {vec(Vb)}$.  **(c)** At heading $\\phi = "
             f"{mdeg(phi)}$ the chassis must move with $\\dot{{q}} = (\\dot{{\\phi}}, \\dot{{x}}, \\dot{{y}}) = {vec(qd)}$ "
             "(in {s}): find $\\mathcal{V}_b$ and the wheel speeds.")
        title, chap = "Three-omniwheel robot: H(0) and wheel speeds", "Ch. 13 (Sec. 13.2, eq. 13.9)"
    else:
        l, wd = float(_pick(rng, [0.2, 0.25, 0.3])), float(_pick(rng, [0.15, 0.2, 0.25]))
        wheels = MB.four_mecanum_wheels(r, l, wd)
        H0 = MB.H0_mecanum_book(r, l, wd)
        while True:
            V2 = np.array([_pick(rng, [-1.0, -0.5, 0.0, 0.5, 1.0]), _pick(rng, [-0.5, 0.0, 0.3, 0.5]),
                           _pick(rng, [-0.4, 0.0, 0.2, 0.4])])
            if np.any(V2):
                break
        slip = float(_pick(rng, [0.0, 0.0, 0.5, 1.0]))
        u_meas = H0 @ V2 + slip * np.array([1.0, 1.0, -1.0, -1.0])
        u_meas = np.round(u_meas, 3)
        st.p(f"A chassis with four mecanum wheels ($\\gamma_i = \\pm 45^{{\\circ}}$ rollers), wheel centres at "
             f"$(\\pm l, \\pm w)$ with $l = {num(l)}$ m, $w = {num(wd)}$ m, as in MR Fig. 13.5:")
        st.table(*_wheel_table(wheels))
        st.p(f"**(a)** Build $H(0)$ with $u = H(0)\\mathcal{{V}}_b$.  **(b)** Find the wheel speeds for $\\mathcal{{V}}_b = "
             f"{vec(Vb)}$.  **(c)** Odometry: the wheel encoders report the speeds $u = {vec(u_meas)}$ rad/s.  Find "
             "$F = H^{\\dagger}(0)$ and the chassis twist that best explains them; are the four readings consistent?")
        title, chap = "Mecanum-wheel robot: H(0), wheel speeds and odometry", "Ch. 13 (Secs. 13.2, 13.4; eq. 13.10)"
    s1 = Doc()
    s1.p("Each row is $h_i(0) = \\frac{1}{r_i\\cos\\gamma_i}\\left(x_i\\sin(\\beta_i+\\gamma_i) - "
         "y_i\\cos(\\beta_i+\\gamma_i),\\ \\cos(\\beta_i+\\gamma_i),\\ \\sin(\\beta_i+\\gamma_i)\\right)$: the velocity of "
         "the wheel centre, $v + \\omega\\times p_i$, resolved along the driving direction $\\beta_i$ and across it "
         "(the rollers at $\\gamma_i$ slide freely), divided by $r_i$:")
    s1.mat("H(0)", H0, 4)
    steps.append(Step("(a) Wheel-speed matrix H(0)", s1))
    u = H0 @ Vb
    s2 = Doc()
    s2.eq(r"u = H(0)\mathcal{V}_b = " + vec(u, 4) + r"\ \mathrm{rad/s}")
    steps.append(Step("(b) Wheel speeds", s2))
    ans = dict(H0=H0, u=u)
    ad = Doc().mat("H(0)", H0, 4).p(f"(b) $u = {vec(u, 4)}$ rad/s.")
    if difficulty == 2:
        Vbp = MB.body_twist(phi, qd)
        uphi = MB.H_matrix(wheels, phi) @ qd
        s3 = Doc()
        s3.p("The body twist is $\\dot{q}$ seen in {b}: $\\omega_{bz} = \\dot{\\phi}$, $(v_{bx}, v_{by}) = "
             "R(\\phi)^{T}(\\dot{x}, \\dot{y})$, and then $u = H(0)\\mathcal{V}_b = H(\\phi)\\dot{q}$:")
        s3.eq(r"\mathcal{V}_b = " + vec(Vbp, 4) + r",\quad u = " + vec(uphi, 4) + r"\ \mathrm{rad/s}")
        steps.append(Step("(c) From a velocity in {s}", s3))
        ans.update(Vb_phi=Vbp, u_phi=uphi)
        ad.p(f"(c) $\\mathcal{{V}}_b = {vec(Vbp, 4)}$, $u = {vec(uphi, 4)}$ rad/s.")
        data = dict(kind="omni3", r=r, d=dd, Vb=Vb, phi=phi, qdot=qd)
        vis = _chassis_visual("omni3", wheels, 1.25 * dd, Vb)
    else:
        Fm = MB.F_mecanum_book(r, l, wd)
        Vo = Fm @ u_meas
        res = u_meas - H0 @ Vo
        s3 = Doc()
        s3.p("Four wheel speeds over-determine the three twist components.  The least-squares twist uses the "
             "pseudoinverse $F = H^{\\dagger}(0) = (H^{T}H)^{-1}H^{T}$ (Sec. 13.4):")
        s3.mat("F", Fm, 5)
        s3.eq(r"\mathcal{V}_b = Fu = " + vec(Vo, 4))
        s3.p(f"Residual $u - H(0)\\mathcal{{V}}_b = {vec(res, 4)}$ rad/s: "
             + ("zero, so the readings are consistent with rolling without slip." if np.abs(res).max() < 1e-9 else
                "not zero, so the readings are inconsistent — some wheels slip (the residual is along "
                "$(1, 1, -1, -1)$, the one wheel-speed pattern no chassis twist can produce, which $F$ maps to zero)."))
        steps.append(Step("(c) Odometry with the pseudoinverse", s3))
        ans.update(F_odo=Fm, Vb_odo=Vo)
        ad.p(f"(c) $\\mathcal{{V}}_b = {vec(Vo, 4)}$ ({'consistent' if np.abs(res).max() < 1e-9 else 'wheels slip'}).")
        data = dict(kind="mecanum", r=r, l=l, w=wd, Vb=Vb, u_meas=u_meas)
        vis = _chassis_visual("mecanum", wheels, (l + 0.08, wd + 0.04), Vb)
    return Problem("mobile_kin", seed, difficulty, title, chap, st, data, steps, ans, ad, vis)


def _verify_mobile_kin(p: Problem) -> float:
    """Wheel speeds from the rigid-body velocity of each wheel centre (no H matrix formula), qdot / V_b by an explicit
    rotation, the encoder inverse by a linear solve and odometry by least squares (numpy lstsq / pinv)."""
    d, a = p.data, p.answer
    r = d["r"]
    if d["kind"] == "diff":
        wheels = [MB.Wheel(0.0, d["d"], 0.0, 0.0, r), MB.Wheel(0.0, -d["d"], 0.0, 0.0, r)]
        V = np.array([d["omega"], d["v"], 0.0])
        u = np.array([_wheel_speed(wh, V) for wh in wheels])
        qd = np.r_[d["omega"], _rot2(d["phi"]) @ np.array([d["v"], 0.0])]
        vw = np.linalg.solve(np.array([[1.0, -d["d"]], [1.0, d["d"]]]) / r, d["u_enc"])
        return float(max(np.abs(u - a["u_LR"]).max(), np.abs(qd - a["qdot"]).max(), abs(vw[0] - a["v_enc"]),
                         abs(vw[1] - a["omega_enc"])))
    if d["kind"] == "omni3":
        dd = d["d"]
        wheels = [MB.Wheel(dd * np.cos(ang), dd * np.sin(ang), beta, 0.0, r)
                  for beta, ang in ((0.0, np.pi / 2), (-2 * np.pi / 3, -np.pi / 6), (2 * np.pi / 3, 7 * np.pi / 6))]
    else:
        l, w = d["l"], d["w"]
        g = np.pi / 4
        wheels = [MB.Wheel(l, w, 0.0, -g, r), MB.Wheel(l, -w, 0.0, g, r), MB.Wheel(-l, -w, 0.0, -g, r),
                  MB.Wheel(-l, w, 0.0, g, r)]
    H = _H_from_rigid_body(wheels)
    e = max(np.abs(H - a["H0"]).max(), np.abs(np.array([_wheel_speed(wh, d["Vb"]) for wh in wheels]) - a["u"]).max())
    if "Vb_phi" in a:
        Vb = np.r_[d["qdot"][0], _rot2(d["phi"]).T @ d["qdot"][1:]]
        e = max(e, np.abs(Vb - a["Vb_phi"]).max(),
                np.abs(np.array([_wheel_speed(wh, Vb) for wh in wheels]) - a["u_phi"]).max())
    if "Vb_odo" in a:
        e = max(e, np.abs(np.linalg.pinv(H) - a["F_odo"]).max(),
                np.abs(np.linalg.lstsq(H, d["u_meas"], rcond=None)[0] - a["Vb_odo"]).max())
    return float(e)


# ------------------------------------------------------------------------------------------------
# registry and public API
# ------------------------------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "rotation": dict(title="Rotations & rigid motions: exp / log (Ch. 3)", gen=gen_rotation, verify=_verify_rotation),
    "fk": dict(title="PoE forward kinematics, space & body (Ch. 4)", gen=gen_fk, verify=_verify_fk),
    "jacobian": dict(title="Space / body Jacobian & singularities (Ch. 5)", gen=gen_jacobian, verify=_verify_jacobian),
    "statics": dict(title="Statics τ = JᵀF and manipulability (Ch. 5)", gen=gen_statics, verify=_verify_statics),
    "ik": dict(title="Newton–Raphson IK iteration table (Ch. 6)", gen=gen_ik, verify=_verify_ik),
    "inertia": dict(title="Rigid-body inertia: principal moments, Steiner, spatial inertia (Ch. 8)", gen=gen_inertia,
                    verify=_verify_inertia),
    "dynamics": dict(title="2R dynamics: Lagrange mass matrix / RNEA torques (Ch. 8)", gen=gen_dynamics,
                     verify=_verify_dynamics),
    "fwd_dynamics": dict(title="2R forward dynamics and an Euler step (Ch. 8)", gen=gen_fwd_dynamics,
                         verify=_verify_fwd_dynamics),
    "task_space": dict(title="Task-space mass matrix, felt mass, reflected inertia (Ch. 8)", gen=gen_task_space,
                       verify=_verify_task_space),
    "constrained": dict(title="Constrained dynamics: Pfaffian constraint, Lagrange multiplier (Ch. 8)",
                        gen=gen_constrained, verify=_verify_constrained),
    "trajectory": dict(title="Cubic / quintic time scaling (Ch. 9)", gen=gen_trajectory, verify=_verify_trajectory),
    "control": dict(title="PD / computed-torque error dynamics ζ, ωₙ (Ch. 11)", gen=gen_control,
                    verify=_verify_control),
    "force_control": dict(title="Impedance and PI force control: deflection, ζ, ωₙ, stability (Ch. 11)",
                          gen=gen_force_control, verify=_verify_force_control),
    "grasp_closure": dict(title="Grasping: wrench matrix, form / force closure, minimum μ (Ch. 12)",
                          gen=gen_grasp_closure, verify=_verify_grasp_closure),
    "mobile_kin": dict(title="Wheeled mobile robots: diff-drive, omniwheel / mecanum H(0) (Ch. 13)",
                       gen=gen_mobile_kin, verify=_verify_mobile_kin),
}
TOPIC_KEYS = list(TOPICS)
#: fixed random-stream id per topic (topics are listed by chapter, but a seed keeps giving the same problem
#: when topics are added or reordered)
_RNG_ID = {"rotation": 0, "fk": 1, "jacobian": 2, "ik": 3, "dynamics": 4, "trajectory": 5, "control": 6,
           "inertia": 7, "fwd_dynamics": 8, "task_space": 9, "constrained": 10, "statics": 11, "force_control": 12,
           "grasp_closure": 13, "mobile_kin": 14}
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
    """``n`` problems.  With topic "mixed" problem k is on topic k mod len(TOPICS), so every quiz version has the
    same topic per question; versions differ only in their numbers (seed offset)."""
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
            "dynamics": "MR 8.1, 8.3", "trajectory": "MR 9.2", "control": "MR 11.2–11.4", "inertia": "MR 8.2",
            "fwd_dynamics": "MR 8.1, 8.5", "task_space": "MR 8.6, 8.9.2", "constrained": "MR 8.7",
            "statics": "MR 5.2, 5.4", "force_control": "MR 11.5, 11.7", "grasp_closure": "MR 12.1.7, 12.2.3",
            "mobile_kin": "MR 13.2–13.4"}
    how = {"rotation": "SciPy Rotation / expm vs mr.MatrixExp3/Log3/Exp6/Log6",
           "fk": "screw axes rebuilt as q×ω, SciPy expm products (space and M⁻¹[S]M body form) vs mr.FKinSpace",
           "jacobian": "central finite differences of the PoE map (SciPy expm) vs mr.JacobianSpace/Body; rank decision",
           "ik": "final θ reaches T_sd (SciPy expm); tabulated V_b recomputed with SciPy logm",
           "dynamics": "closed-form Lagrangian (8.9)-(8.10) vs mr.MassMatrix / VelQuadraticForces / GravityForces / "
                       "InverseDynamics",
           "trajectory": "polynomial re-solved from its boundary conditions; peaks by dense sampling vs mr time scaling",
           "control": "ζ, ωn from the eigenvalues of the error-dynamics matrix; overshoot from solve_ivp",
           "inertia": "exact Gauss quadrature over the solid (inertia, G, kinetic energy of the mass points) vs "
                      "closed-form principal moments, Steiner and mr.Adjoint (8.42)",
           "fwd_dynamics": "closed-form Lagrangian solve vs mr.ForwardDynamics; θ̈ fed back through "
                           "mr.InverseDynamics; Euler steps recomputed",
           "task_space": "Λ = (J M⁻¹ Jᵀ)⁻¹ from the closed-form J and M; felt mass as a minimum-energy KKT solve",
           "constrained": "KKT system [[M, −Aᵀ], [A, 0]] with the closed-form Lagrangian and Ȧθ̇ by central "
                          "differences vs the multiplier formula",
           "statics": "τᵢ as the moment of f about joint i, J from finite differences of the closed-form chain, "
                      "lstsq inverse, ellipse axes from SVD vs mr.JacobianSpace/Body and eig(J Jᵀ)",
           "force_control": "eigenvalues and linear steady state of the state matrix, overshoot by solve_ivp, "
                            "np.poly of the PI loop and its stability limit by root finding vs Routh–Hurwitz",
           "grasp_closure": "wrenches rebuilt from the geometry; max-min-weight LP and the dual twist test vs the "
                            "LP of grasping.py; two-finger angle rule vs bisection for μ_min",
           "mobile_kin": "wheel speeds from the rigid-body velocity of each wheel centre, explicit rotations, "
                         "numpy lstsq / pinv vs mobile.py and the book matrices H(0), F"}
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
