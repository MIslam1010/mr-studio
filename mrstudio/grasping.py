"""Planar grasping and contact mechanics (Modern Robotics, Ch. 12).

Conventions (MR Ch. 12, "All twists and wrenches are expressed in a space frame in this chapter"):

* planar twist   ``V = (w_z, v_x, v_y)``  (the (omega, v) order of a spatial twist, reduced to the plane);
* planar wrench  ``F = (m_z, f_x, f_y)``  (the (m, f) order of a spatial wrench);
* a point contact is ``(p, n)``: contact point ``p`` and unit contact normal ``n`` pointing *into* the movable
  body A (MR Fig. 12.2); its normal wrench is ``F = ([p] n, n)``, in the plane ``m_z = p_x n_y - p_y n_x``;
* a planar friction cone is the positive span of its two unit edge forces ``(n +/- mu t)/sqrt(1 + mu^2)``
  (MR p. 487: "For planar problems, no approximation is necessary").

Contents: first-order contact classification R / Sl / Sr / B (Sec. 12.1.2, Sec. 12.1.6), feasible-twist cones
and centres of rotation (Sec. 12.1.3, 12.1.6), the form-closure linear program (12.14) (Sec. 12.1.7.2), the grasp
metric of (12.15) (Sec. 12.1.7.3), friction cones (12.16) and the force-closure test (Sec. 12.2.3), the minimum
friction coefficient for force closure (Sec. 12.2.3.2), quasistatic / dynamic balance (12.32)-(12.33) and
``validation_checks()`` for the Validation tab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .checks import Check

CATEGORY = "Grasping"
TOL = 1e-9

# contact labels (Sec. 12.1.2; Sl / Sr split for planar sliding, Sec. 12.1.6)
ROLL, SLIDE_L, SLIDE_R, BREAK, PENETRATE = "R", "Sl", "Sr", "B", "X"


# ============================================================================ contacts and wrenches
@dataclass
class Contact:
    """A planar point contact: point ``p``, inward unit normal ``n`` (into the movable body), friction ``mu``."""
    p: np.ndarray
    n: np.ndarray
    mu: float = 0.0
    name: str = ""

    def __post_init__(self):
        self.p = np.asarray(self.p, float).reshape(2)
        n = np.asarray(self.n, float).reshape(2)
        self.n = n / np.linalg.norm(n)
        self.mu = max(0.0, float(self.mu))

    @classmethod
    def at_angle(cls, x: float, y: float, angle: float, mu: float = 0.0, name: str = "") -> "Contact":
        """Contact at (x, y) whose inward normal makes ``angle`` (rad) with the x-axis."""
        return cls((x, y), (math.cos(angle), math.sin(angle)), mu, name)

    @property
    def angle(self) -> float:
        return math.atan2(self.n[1], self.n[0])

    @property
    def tangent(self) -> np.ndarray:
        """Unit tangent ``t = n`` rotated +90 deg (to the left when facing along the inward normal)."""
        return np.array([-self.n[1], self.n[0]])


def moment(p, f) -> float:
    """Planar moment ``m_z = p_x f_y - p_y f_x`` of a force ``f`` applied at ``p`` (``[p] f``, z-component)."""
    return float(p[0] * f[1] - p[1] * f[0])


def wrench(p, f) -> np.ndarray:
    """Planar wrench ``F = (m_z, f_x, f_y)`` of the force ``f`` acting at the point ``p``."""
    f = np.asarray(f, float)
    return np.array([moment(p, f), f[0], f[1]])


def normal_wrench(c: Contact) -> np.ndarray:
    """``F = ([p] n, n)``: the wrench of a unit force along the inward contact normal (MR p. 467)."""
    return wrench(c.p, c.n)


def cone_edges(c: Contact) -> list[np.ndarray]:
    """Unit edge forces of the planar friction cone: one (the normal) if ``mu = 0``, else two."""
    if c.mu <= 0:
        return [c.n.copy()]
    s = 1.0 / math.sqrt(1.0 + c.mu ** 2)
    return [(c.n + c.mu * c.tangent) * s, (c.n - c.mu * c.tangent) * s]


def wrench_matrix(contacts: list[Contact], friction: bool = True) -> np.ndarray:
    """``F = [F_1 ... F_j]`` (3 x j): normal wrenches (``friction=False``) or all friction-cone edge wrenches."""
    cols = []
    for c in contacts:
        if friction:
            cols += [wrench(c.p, f) for f in cone_edges(c)]
        else:
            cols.append(normal_wrench(c))
    return np.array(cols, float).T.reshape(3, len(cols))


def arrow_base(F) -> np.ndarray | None:
    """Base point of the arrow drawing a planar wrench (MR p. 489): ``(m_z f_y, -m_z f_x)/(f_x^2 + f_y^2)``.
    None for a pure moment."""
    m, fx, fy = (float(v) for v in F)
    d = fx * fx + fy * fy
    if d < 1e-18:
        return None
    return np.array([m * fy, -m * fx]) / d


# ============================================================================ twists and contact labels
def point_velocity(V, p) -> np.ndarray:
    """Velocity ``v + w x p`` of the body point at ``p`` for a planar twist ``V = (w, v_x, v_y)``."""
    w, vx, vy = (float(a) for a in V)
    return np.array([vx - w * p[1], vy + w * p[0]])


def cor(V) -> np.ndarray | None:
    """Centre of rotation ``(-v_y/w_z, v_x/w_z)`` of a planar twist (MR p. 475); None for a translation."""
    w, vx, vy = (float(a) for a in V)
    if abs(w) < 1e-12:
        return None
    return np.array([-vy / w, vx / w])


def twist_from_cor(c, w: float = 1.0) -> np.ndarray:
    """The planar twist rotating at ``w`` about the centre of rotation ``c``."""
    return np.array([w, w * c[1], -w * c[0]])


def contact_label(c: Contact, VA, VB=(0.0, 0.0, 0.0), tol: float = 1e-9) -> str:
    """First-order label of contact ``c`` for body twist ``VA`` against a body moving with ``VB``.

    ``X`` (penetrating) if ``F^T (V_A - V_B) < 0`` (12.5 violated); ``B`` (breaking) if ``> 0``; otherwise the
    active constraint (12.6) holds and the contact is ``R`` (rolling, (12.8) holds) or sliding.  For sliding,
    ``Sr`` / ``Sl`` = the movable body slips to the right / left relative to the constraint, as seen facing
    along the inward normal n (this tab's reading of MR Fig. 12.11)."""
    VA, VB = np.asarray(VA, float), np.asarray(VB, float)
    F = normal_wrench(c)
    scale = max(1.0, float(np.linalg.norm(VA - VB)))
    s = float(F @ (VA - VB))
    if s < -tol * scale:
        return PENETRATE
    if s > tol * scale:
        return BREAK
    dp = point_velocity(VA, c.p) - point_velocity(VB, c.p)
    if float(np.linalg.norm(dp)) <= tol * scale * (1 + float(np.linalg.norm(c.p))):
        return ROLL
    return SLIDE_R if float(dp @ (-c.tangent)) > 0 else SLIDE_L


def contact_mode(contacts: list[Contact], V, finger_twists=None, tol: float = 1e-9) -> str:
    """Concatenated contact labels of all contacts (MR p. 471), e.g. ``"RSrB"``."""
    out = []
    for i, c in enumerate(contacts):
        VB = (0.0, 0.0, 0.0) if finger_twists is None else finger_twists[i]
        out.append(contact_label(c, V, VB, tol))
    return "".join(out)


def feasible(contacts: list[Contact], V, tol: float = 1e-9) -> bool:
    """True if the twist satisfies every stationary impenetrability constraint ``F_i^T V >= 0``."""
    return all(contact_label(c, V, tol=tol) != PENETRATE for c in contacts)


def twist_cone_edges(contacts: list[Contact], tol: float = 1e-9) -> list[np.ndarray]:
    """Edges (unit rays) of the pointed feasible-twist cone ``{V | F_i^T V >= 0}`` for stationary contacts.

    Each edge has two independent active constraints: candidate rays ``+/- F_i x F_j`` (3-D cross product)
    that satisfy all constraints.  Returns [] if the cone is only the origin (form closure) and also if the
    cone contains a line (fewer than two independent constraints)."""
    Fs = [normal_wrench(c) for c in contacts]
    rays: list[np.ndarray] = []
    for i in range(len(Fs)):
        for j in range(i + 1, len(Fs)):
            r = np.cross(Fs[i], Fs[j])
            nr = float(np.linalg.norm(r))
            if nr < 1e-10:
                continue
            r = r / nr
            for s in (1.0, -1.0):
                v = s * r
                if all(float(F @ v) >= -tol for F in Fs) and not any(np.allclose(v, q, atol=1e-8) for q in rays):
                    rays.append(v)
    return rays


def cor_grid(contacts: list[Contact], xs: np.ndarray, ys: np.ndarray, tol: float = 1e-9):
    """Feasible centres of rotation on a grid (MR Sec. 12.1.6): boolean arrays (plus, minus) of shape
    (len(ys), len(xs)); ``plus[i, j]`` = a counter-clockwise rotation about (xs[j], ys[i]) is feasible."""
    X, Y = np.meshgrid(xs, ys)
    plus = np.ones_like(X, bool)
    minus = np.ones_like(X, bool)
    for c in contacts:
        m, fx, fy = normal_wrench(c)
        # F^T V for V = twist_from_cor((x, y), +1) = (1, y, -x)
        s = m + fx * Y - fy * X
        plus &= s >= -tol
        minus &= -s >= -tol
    return plus, minus


# ============================================================================ closure tests (LP 12.14)
@dataclass
class ClosureResult:
    closed: bool
    rank: int
    k: np.ndarray | None = None
    status: str = ""
    F: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))

    def summary(self) -> str:
        if self.closed:
            return "closure: k = (" + ", ".join(f"{v:.3g}" for v in self.k) + ")"
        return f"no closure ({self.status})"


def closure_lp(F: np.ndarray) -> ClosureResult:
    """First-order closure test (MR Sec. 12.1.7.2): rank F = n and the LP (12.14)
    ``min 1^T k  s.t.  F k = 0,  k_i >= 1`` has a solution.  Used for form closure (normal wrenches) and,
    with the friction-cone edges as columns, for force closure (Sec. 12.2.3)."""
    F = np.asarray(F, float)
    n, j = F.shape
    rank = int(np.linalg.matrix_rank(F, tol=1e-9)) if j else 0
    if rank < n:
        return ClosureResult(False, rank, None, f"rank F = {rank} < {n}", F)
    res = linprog(np.ones(j), A_eq=F, b_eq=np.zeros(n), bounds=[(1.0, None)] * j, method="highs")
    if res.status == 0:
        return ClosureResult(True, rank, np.asarray(res.x), "LP (12.14) feasible", F)
    return ClosureResult(False, rank, None, "LP (12.14) infeasible", F)


def form_closure(contacts: list[Contact]) -> ClosureResult:
    """First-order form closure of a planar body by frictionless stationary point contacts."""
    return closure_lp(wrench_matrix(contacts, friction=False))


def force_closure(contacts: list[Contact]) -> ClosureResult:
    """Force closure with Coulomb friction: the LP (12.14) on the friction-cone edge wrenches (exact in 2-D)."""
    return closure_lp(wrench_matrix(contacts, friction=True))


def nonzero_twist(contacts: list[Contact]) -> np.ndarray | None:
    """Dual test (independent of 12.14): a nonzero twist with all ``F_i^T V >= 0``, or None.

    First a twist in the null space of ``F^T`` (all constraints active), else the LP
    ``max sum_i F_i^T V  s.t.  0 <= F_i^T V <= 1``; a positive optimum is a twist that breaks some contact."""
    F = wrench_matrix(contacts, friction=False)
    _, s, vh = np.linalg.svd(F.T, full_matrices=True)
    r = int(np.sum(s > 1e-9))
    if r < 3:
        return vh[-1]
    res = linprog(-F.sum(axis=1), A_ub=np.vstack([-F.T, F.T]),
                  b_ub=np.concatenate([np.zeros(F.shape[1]), np.ones(F.shape[1])]),
                  bounds=[(None, None)] * 3, method="highs")
    if res.status == 0 and -res.fun > 1e-7:
        return np.asarray(res.x)
    return None


def with_mu(contacts: list[Contact], mu: float) -> list[Contact]:
    return [Contact(c.p, c.n, mu, c.name) for c in contacts]


def min_friction(contacts: list[Contact], mu_max: float = 10.0, tol: float = 1e-6) -> float:
    """Smallest common friction coefficient giving force closure (MR Sec. 12.2.3.2), by bisection on the
    LP; ``inf`` if not reached by ``mu_max``.  Force closure is monotone in mu (the cones only grow)."""
    if force_closure(with_mu(contacts, 0.0)).closed:
        return 0.0
    if not force_closure(with_mu(contacts, mu_max)).closed:
        return math.inf
    lo, hi = 0.0, mu_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if force_closure(with_mu(contacts, mid)).closed:
            hi = mid
        else:
            lo = mid
    return hi


def balance(F: np.ndarray, W) -> np.ndarray | None:
    """Nonnegative ``k`` with ``F k = W`` (a wrench W that the contacts can apply), or None."""
    F = np.asarray(F, float)
    res = linprog(np.ones(F.shape[1]), A_eq=F, b_eq=np.asarray(W, float),
                  bounds=[(0.0, None)] * F.shape[1], method="highs")
    return np.asarray(res.x) if res.status == 0 else None


def grasp_quality(contacts: list[Contact], center=(0.0, 0.0), r: float = 1.0, f_max: float = 1.0,
                  friction: bool = True) -> float:
    """Grasp metric of MR Sec. 12.1.7.3: the signed radius of the largest origin-centred ball inside
    ``CF = {sum f_i F_i | f_i in [0, f_max]}`` (12.15), with moments taken about ``center`` and divided by the
    characteristic length ``r``.  CF always contains the origin (all f_i = 0); without closure the origin lies on
    its boundary and the metric is 0 (no closure).  With ``friction`` the F_i are the friction-cone edge wrenches
    of each contact, scaled so that the *normal* components of its two edges sum to ``f_max``: each edge force is
    ``(f_max / 2)(n +/- mu t)`` (normal part f_max / 2, tangential part mu f_max / 2).  A contact can then push
    with any normal force up to ``f_max`` and, at normal force a, any tangential force up to
    ``mu min(a, f_max - a)``; that set only grows with mu, so Qual never decreases as mu grows, and at mu = 0 the
    two edges add up to the single normal of length ``f_max`` (continuous at mu = 0+).  MR states (12.15) for the
    contact normals; the split over the cone edges is the tab's extension to friction cones."""
    from scipy.spatial import ConvexHull, QhullError
    cs = [Contact(c.p - np.asarray(center, float), c.n, c.mu) for c in contacts]
    F = wrench_matrix(cs, friction=friction) * f_max
    if friction:      # unit edges (n +/- mu t)/sqrt(1 + mu^2) -> (n +/- mu t)/2: normal parts sum to f_max
        F = F * np.concatenate([[math.sqrt(1.0 + c.mu ** 2) / 2.0] * 2 if len(cone_edges(c)) == 2 else [1.0]
                                for c in cs])
    F[0] /= r
    j = F.shape[1]
    if j == 0:
        return -math.inf
    if j > 14:
        raise ValueError("too many cone edges for the exact Minkowski-sum hull")
    bits = ((np.arange(2 ** j)[:, None] >> np.arange(j)) & 1).astype(float)
    pts = bits @ F.T
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return 0.0
    # equations: a.x + b <= 0 inside, |a| = 1 -> signed distance of the origin to each facet = -b
    q = float(np.min(-hull.equations[:, -1]))
    return 0.0 if abs(q) < 1e-9 else q


def polygon_centroid(poly) -> np.ndarray:
    P = np.asarray(poly, float)
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    if abs(A) < 1e-12:
        return P.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * A)


# ============================================================================ planar rigid motion (finite displacement)
def planar_exp(V, t: float) -> np.ndarray:
    """SE(2) part of ``exp([V] t)`` computed with modern_robotics.MatrixExp6 on the spatial twist
    ``(0, 0, w, v_x, v_y, 0)``; returns the 3x3 homogeneous planar transform."""
    import modern_robotics as mr
    w, vx, vy = (float(a) for a in V)
    T = mr.MatrixExp6(mr.VecTose3(np.array([0, 0, w, vx, vy, 0.0]) * t))
    return np.array([[T[0, 0], T[0, 1], T[0, 3]], [T[1, 0], T[1, 1], T[1, 3]], [0, 0, 1.0]])


# ============================================================================ presets (tab and tests)
def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _triangle(side: float = 2.0):
    h = side * math.sqrt(3) / 2
    return [(-side / 2, -h / 3), (side / 2, -h / 3), (0.0, 2 * h / 3)]


def _edge_midpoint_contact(poly, i: int, mu: float, name: str) -> Contact:
    """Contact at the midpoint of edge i of a counter-clockwise polygon, inward normal."""
    a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
    e = (b - a) / np.linalg.norm(b - a)
    return Contact((a + b) / 2, (-e[1], e[0]), mu, name)


def _circle(r=1.0, n=48):
    return [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def _presets() -> dict[str, dict]:
    tri = _triangle(2.0)
    sq = _rect(-1, -1, 1, 1)
    up, dn, lf, rt = (0, 1), (0, -1), (-1, 0), (1, 0)
    return {
        "Ex. 12.7 — two fingers in a hole (form closure)": dict(
            poly=_rect(-1, -1, 3, 2), hole=_rect(0, 0, 2, 1),
            contacts=[Contact((0, 0), lf, 0, "1"), Contact((0, 0), dn, 0, "2"),
                      Contact((2, 1), rt, 0, "3"), Contact((2, 1), up, 0, "4")],
            twist=(0.0, 0.0, -1.0),
            text="Fig. 12.15: fingers in opposite corners of the hole push on four edges. F has rank 3 and the "
                 "LP (12.14) returns k = (2, 1, 2, 1) (MR p. 483): first-order form closure."),
        "Ex. 12.7 — right finger moved down (slides)": dict(
            poly=_rect(-1, -1, 3, 2), hole=_rect(0, 0, 2, 1),
            contacts=[Contact((0, 0), lf, 0, "1"), Contact((0, 0), dn, 0, "2"),
                      Contact((2, 0), rt, 0, "3"), Contact((2, 0), dn, 0, "4")],
            twist=(0.0, 0.0, -1.0),
            text="Right finger moved to the bottom-right corner: F is still full rank but the LP has no solution; "
                 "the body can slide downward (MR p. 483): the test twist v_y = -1 slides on two contacts, breaks two."),
        "Ex. 12.4 — three stationary contacts (twist cone)": dict(
            poly=_rect(-2.5, -1.5, 1, 1.5),
            contacts=[Contact((-2, -1.5), up, 0, "1"), Contact((1, 1), lf, 0, "2"),
                      Contact((-2.5, -1), rt, 0, "3")],
            twist=(1.0, 0.0, 2.0),
            text="F1 = (-2, 0, 1), F2 = (1, -1, 0), F3 = (1, 1, 0) (MR p. 473). Three contacts never give planar "
                 "form closure (Theorem 12.6); the feasible twists form a cone with three edges."),
        "Square — four frictionless fingers (form closure)": dict(
            poly=sq,
            contacts=[Contact((-1, 0.5), rt, 0, "1"), Contact((1, -0.5), lf, 0, "2"),
                      Contact((0.5, -1), up, 0, "3"), Contact((-0.5, 1), dn, 0, "4")],
            twist=(1.0, 0.0, 0.0),
            text="Four normals whose positive span is R^3 (Theorem 12.6: four is the minimum in the plane). "
                 "Every nonzero twist penetrates some finger."),
        "Square — three frictionless fingers (never closure)": dict(
            poly=sq,
            contacts=[Contact((-1, 0.5), rt, 0, "1"), Contact((1, -0.5), lf, 0, "2"),
                      Contact((0.5, -1), up, 0, "3")],
            twist=(0.0, 0.0, 1.0),
            text="Remove one finger: three vectors cannot positively span R^3 (fact 2, MR p. 464), so a "
                 "nonzero feasible twist always exists."),
        "Triangle — two fingers, μ = 0.25 (no force closure)": dict(
            poly=tri, contacts=[_edge_midpoint_contact(tri, 1, 0.25, "1"), _edge_midpoint_contact(tri, 2, 0.25, "2")],
            twist=(0.0, 0.0, -1.0),
            text="Fig. 12.22(a): an equilateral triangle held by two fingers on its edges needs μ ≥ tan 30° ≈ 0.577 "
                 "(MR p. 492). The line between the fingers is outside both friction cones."),
        "Triangle — two fingers, μ = 1 (force closure)": dict(
            poly=tri, contacts=[_edge_midpoint_contact(tri, 1, 1.0, "1"), _edge_midpoint_contact(tri, 2, 1.0, "2")],
            twist=(0.0, 0.0, -1.0),
            text="Fig. 12.22(b): with μ = 1 the two contacts \"see\" each other inside both friction cones: "
                 "force closure with only two frictional contacts."),
        "Ex. 12.9 — block on two fingers (μ = 1 and μ = 0)": dict(
            poly=[(-4, -1), (4, -1), (4, 0), (1, 0), (1, 2), (-1, 2), (-1, 0), (-4, 0)],
            contacts=[Contact((-3, -1), up, 1.0, "1"), Contact((1, 1), lf, 0.0, "2")],
            twist=(0.0, -1.0, 0.0), gravity=True,
            text="Fig. 12.25: lower finger μ = 1 at (-3, -1), upper finger frictionless at (1, 1); the centre of "
                 "mass is on x = 0 so gravity is F_ext = (0, 0, -mg). At rest the contacts cannot balance gravity "
                 "(MR p. 498); accelerating the fingers left at 2g they can (dynamic grasp, (12.33))."),
        "Disk — exceptional object": dict(
            poly=_circle(1.0),
            contacts=[Contact.at_angle(math.cos(a), math.sin(a), a + math.pi, 0.0, str(i + 1))
                      for i, a in enumerate((0.3, 1.9, 3.4, 4.9))],
            twist=(1.0, 0.0, 0.0),
            text="All normals of a disk pass through its centre: no number of frictionless contacts prevents "
                 "spinning about the centre (MR p. 481). Raise μ to see two antipodal fingers reach force closure."),
    }


PRESETS = _presets()


# ============================================================================ validation
def _random_contacts(rng, n: int) -> list[Contact]:
    return [Contact.at_angle(*rng.uniform(-1, 1, 2), rng.uniform(-math.pi, math.pi)) for _ in range(n)]


def validation_checks() -> list[Check]:
    checks: list[Check] = []
    g, m = 9.81, 1.0

    # 1. Example 12.2: wrench and the planar roll-slide / rolling constraints (12.9)-(12.11)
    c = Contact((1, 2), (0, 1))
    F = normal_wrench(c)
    err = float(np.max(np.abs(F - [1, 0, 1])))
    rng = np.random.default_rng(12)
    bad = 0
    for _ in range(400):
        w, vx, vy = rng.normal(size=3)
        kind = rng.integers(3)
        if kind == 0:
            vy = -w                       # roll-slide (12.9): w + v_y = 0
        elif kind == 1:
            vy, vx = -w, 2 * w            # rolling (12.10)-(12.11): v_x - 2w = 0, v_y + w = 0
        lab = contact_label(c, (w, vx, vy))
        s = w + vy
        book = (PENETRATE if s < -1e-9 else BREAK if s > 1e-9 else
                ROLL if abs(vx - 2 * w) < 1e-9 else "S")
        bad += (lab[0] if lab in (SLIDE_L, SLIDE_R) else lab) != book
    checks.append(Check("Example 12.2: F = (1, 0, 1) and R/S/B labels vs (12.9)-(12.11) on 400 twists",
                        "MR 12.1.2", err + bad, 1e-12,
                        f"|F - (1,0,1)| = {err:.1e}; label mismatches: {bad}", CATEGORY))

    # 2. first-order analysis vs finite displacement (vertex of A on a fixed face, d(t) by mr.MatrixExp6)
    worst1 = worst2 = 0.0
    for V in ((0.3, 1.0, 0.7), (-1.2, 0.4, 2.0), (2.0, -0.5, 0.1)):
        c = Contact((0.4, -0.3), (0.2, 1.0))
        dd = float(normal_wrench(c) @ np.asarray(V))          # first-order rate (12.1) = F^T V
        t = 1e-5
        P = planar_exp(V, t) @ np.array([*c.p, 1.0])
        d = float(c.n @ (P[:2] - c.p))
        worst1 = max(worst1, abs(d / t - dd))
    # roll-slide twist: d = O(t^2) (second-order term decides, MR p. 466)
    c = Contact((0.0, 0.0), (0.0, 1.0))
    V = twist_from_cor((0.0, 1.0), 1.0)                      # CoR on the normal: first-order sliding
    for t in (1e-2, 1e-3):
        P = planar_exp(V, t) @ np.array([0.0, 0.0, 1.0])
        worst2 = max(worst2, abs(abs(float(c.n @ P[:2])) / t ** 2 - 0.5))
    checks.append(Check("First-order rate F^T V vs finite displacement exp([V]t) (vertex on a fixed face)",
                        "MR 12.1.1", worst1 + abs(worst2), 1e-3,
                        f"max |d(t)/t - F^T V| = {worst1:.2e} (t = 1e-5); roll-slide twist about a CoR on the "
                        f"normal: d(t)/t^2 -> 1/2 (deviation {worst2:.1e}), i.e. only second order decides", CATEGORY))

    # 3. CoR formula (p. 475): the CoR is stationary; planar wrench arrow (p. 489) lies on the line of force
    e1 = e2 = 0.0
    for _ in range(50):
        V = rng.normal(size=3)
        e1 = max(e1, float(np.linalg.norm(point_velocity(V, cor(V)))))
        p, f = rng.normal(size=2), rng.normal(size=2)
        b = arrow_base(wrench(p, f))
        e2 = max(e2, abs(moment(b - p, f)), abs(float(b @ f)))
    checks.append(Check("CoR (-v_y/w, v_x/w) is stationary; wrench arrow base is on the line of force",
                        "MR 12.1.6, 12.2.2", e1 + e2, 1e-12,
                        f"max |v(CoR)| = {e1:.1e}; max moment of f about the arrow base relative to p = {e2:.1e}",
                        CATEGORY))

    # 4. Example 12.4: cone of feasible twists has edges (0,0,1), (1,1,2), (1,-1,2)
    pre = PRESETS["Ex. 12.4 — three stationary contacts (twist cone)"]
    Fm = wrench_matrix(pre["contacts"], friction=False)
    eF = float(np.max(np.abs(Fm - np.array([[-2, 1, 1], [0, -1, 1], [1, 0, 0]]))))
    edges = twist_cone_edges(pre["contacts"])
    ref = [np.array(v, float) / np.linalg.norm(v) for v in ((0, 0, 1), (1, 1, 2), (1, -1, 2))]
    eE = max(min(float(np.linalg.norm(e - r)) for e in edges) for r in ref) if len(edges) == 3 else 1.0
    checks.append(Check("Example 12.4: F1, F2, F3 from the contact geometry; twist-cone edges (0,0,1), (1,±1,2)",
                        "MR 12.1.3", eF + eE, 1e-9,
                        f"{len(edges)} edges; edges are pairwise intersections of the active constraints "
                        f"F_i^T V = 0 (values derived from the book's F_i)", CATEGORY))

    # 5. Example 12.7: LP (12.14) k = (2, 1, 2, 1); moved finger -> infeasible
    r = form_closure(PRESETS["Ex. 12.7 — two fingers in a hole (form closure)"]["contacts"])
    r2 = form_closure(PRESETS["Ex. 12.7 — right finger moved down (slides)"]["contacts"])
    Fb = np.array([[0, 0, -1, 2], [-1, 0, 1, 0], [0, -1, 0, 1]], float)
    Fb2 = np.array([[0, 0, 0, -2], [-1, 0, 1, 0], [0, -1, 0, -1]], float)
    eF = float(np.max(np.abs(r.F - Fb))) + float(np.max(np.abs(r2.F - Fb2)))
    ek = float(np.max(np.abs(r.k - [2, 1, 2, 1]))) if r.closed else 1.0
    checks.append(Check("Example 12.7: F matrices; LP (12.14) gives k = (2, 1, 2, 1); moved finger: no solution",
                        "MR 12.1.7", eF + ek + (0 if (r2.rank == 3 and not r2.closed) else 1), 1e-7,
                        f"scipy linprog (HiGHS): {r.summary()}; moved finger: rank {r2.rank}, {r2.status}", CATEGORY))

    # 6. Theorem 12.6: three planar point contacts never give form closure; 4 random sometimes do
    n3 = sum(form_closure(_random_contacts(rng, 3)).closed for _ in range(300))
    n4 = sum(form_closure(_random_contacts(rng, 4)).closed for _ in range(300))
    checks.append(Check("Theorem 12.6: no form closure with 3 planar contacts (300 random sets)", "MR 12.1.7",
                        float(n3) + (1.0 if n4 == 0 else 0.0), 0.0,
                        f"3 contacts: {n3} closed; 4 contacts: {n4}/300 closed (4 is enough)", CATEGORY))

    # 7. duality: LP (12.14) feasible  <=>  no nonzero feasible twist (Gordan / Farkas)
    mism = 0
    for n in (3, 4, 5, 6) * 60:
        cs = _random_contacts(rng, n)
        closed = form_closure(cs).closed
        V = nonzero_twist(cs)
        mism += closed == (V is not None)
        if V is not None:
            mism += not feasible(cs, V, tol=1e-7)
    checks.append(Check("Duality: LP (12.14) closure <=> feasible twist cone V = {0} (240 random grasps)",
                        "MR 12.1.3, 12.1.7", float(mism), 0.0,
                        "form closure from (12.14) vs an independent twist-space LP max sum F_i^T V, "
                        "0 <= F_i^T V <= 1; mismatches counted", CATEGORY))

    # 8. form closure => force closure; mu = 0 force closure == form closure
    viol = eq = tested = 0
    for _ in range(200):
        cs = _random_contacts(rng, 4)
        fc = form_closure(cs).closed
        eq += fc != force_closure(with_mu(cs, 0.0)).closed
        if fc:
            tested += 1
            viol += not force_closure(with_mu(cs, float(rng.uniform(0.05, 1.0)))).closed
    checks.append(Check("Form closure => force closure for any μ; μ = 0 force closure == form closure",
                        "MR 12.2.3", float(viol + eq), 0.0,
                        f"{tested} form-closure grasps re-tested with friction; mismatches {viol} + {eq}", CATEGORY))

    # 9. Fig. 12.22: equilateral triangle, two fingers on two edges: mu_min = tan 30 deg
    tri = PRESETS["Triangle — two fingers, μ = 1 (force closure)"]["contacts"]
    mu = min_friction(tri, tol=1e-7)
    checks.append(Check("Fig. 12.22: two fingers on an equilateral triangle need μ ≥ tan 30° (bisection on the LP)",
                        "MR 12.2.3", abs(mu - math.tan(math.radians(30))), 1e-6,
                        f"μ_min = {mu:.7f}, tan 30° = {math.tan(math.radians(30)):.7f}", CATEGORY))

    # 10. exceptional object (disk): no form closure with any number of contacts; mu > 0 antipodal = closure
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    disk = [Contact.at_angle(math.cos(a), math.sin(a), a + math.pi) for a in ang]
    rd = form_closure(disk)
    anti = [Contact.at_angle(1, 0, math.pi, 0.2), Contact.at_angle(-1, 0, 0.0, 0.2)]
    checks.append(Check("Disk (exceptional object): 8 normal contacts, no form closure; 2 antipodal, μ = 0.2: force closure",
                        "MR 12.1.7, 12.2.3", float(rd.closed) + float(not force_closure(anti).closed), 0.0,
                        f"8 contacts: rank F = {rd.rank} (all moments zero); antipodal pair: "
                        f"{force_closure(anti).summary()}", CATEGORY))

    # 11. grasp metric (12.15) sign agrees with LP closure
    mism = 0
    for _ in range(80):
        cs = _random_contacts(rng, 4)
        q = grasp_quality(cs, friction=False)
        mism += (q > 1e-9) != form_closure(cs).closed
    checks.append(Check("Grasp metric (12.15): Qual > 0 exactly for form-closure grasps (80 random sets)",
                        "MR 12.1.7", float(mism), 0.0,
                        "radius of the largest origin-centred ball in the convex hull of the 2^j vertex sums",
                        CATEGORY))
    # 11b. with friction: each contact's cone edges carry normal parts summing to f_max -> Qual continuous at
    # mu = 0+ and non-decreasing in mu (the contact force sets are nested)
    worst = 0.0
    for _ in range(20):
        cs = _random_contacts(rng, 3)
        qs = [grasp_quality(with_mu(cs, m)) if m > 0 else grasp_quality(cs, friction=False)
              for m in (0.0, 1e-6, 0.1, 0.25, 0.5, 1.0, 2.0)]
        worst = max(worst, abs(qs[1] - qs[0]), max(0.0, -min(np.diff(qs[1:]))))
    checks.append(Check("Grasp metric with friction: continuous at μ = 0+ and non-decreasing in μ (20 random 3-finger grasps)",
                        "MR 12.1.7, 12.2", worst, 1e-6,
                        "edge forces (f_max/2)(n ± μt): normal parts of a contact's two edges sum to f_max",
                        CATEGORY))

    # 12. Example 12.9: dynamic grasp k1..k3 vs the book's closed form; feasible range -5g <= a_x <= -g
    pre = PRESETS["Ex. 12.9 — block on two fingers (μ = 1 and μ = 0)"]
    Fm = wrench_matrix(pre["contacts"], friction=True)
    Fb = np.array([[-4, -1, 1], [-2, 1, 1]], float).T / math.sqrt(2)
    Fb = np.column_stack([Fb, [1, -1, 0]])
    eF = float(np.max(np.abs(Fm - Fb)))
    ek = 0.0
    for ax in (-6 * g, -3 * g, -2 * g, 0.0, 2 * g):
        k = np.linalg.solve(Fm, np.array([0.0, m * ax, 0.0]) - np.array([0.0, 0.0, -m * g]))   # (12.33)
        kb = np.array([-(ax + g) * m / (2 * math.sqrt(2)), (ax + 5 * g) * m / (2 * math.sqrt(2)),
                       -(ax - 3 * g) * m / 2])
        ek = max(ek, float(np.max(np.abs(k - kb))))
    def ok(ax):
        return balance(Fm, np.array([0.0, m * ax, m * g])) is not None

    def edge(a_in, a_out):              # bisection between a feasible and an infeasible acceleration
        for _ in range(40):
            mid = 0.5 * (a_in + a_out)
            a_in, a_out = (mid, a_out) if ok(mid) else (a_in, mid)
        return a_in
    inside = ok(-3 * g) and not ok(-8 * g) and not ok(2 * g)
    lo, hi = edge(-3 * g, -8 * g), edge(-3 * g, 2 * g)
    erange = (abs(lo / g + 5) + abs(hi / g + 1)) if inside else 1.0
    rest = balance(Fm, np.array([0.0, 0.0, m * g]))
    checks.append(Check("Example 12.9: wrenches, k(a_x) from (12.33) vs book; RR feasible for -5g ≤ a_x ≤ -g",
                        "MR 12.3", eF + ek + erange + (rest is not None), 1e-6,
                        f"max |k - k_book| = {ek:.1e}; LP-feasible a_x in [{lo / g:.2f} g, {hi / g:.2f} g] "
                        f"(bisection on the LP); at rest: {'balanced' if rest is not None else 'not balanced (as in the book)'}",
                        CATEGORY))

    # 13. duality of force and motion freedoms (Sec. 12.2.4): planar B, S, R -> 0, 1, 2
    c = Contact((0.3, -0.2), (0.6, 0.8), 0.5)
    J_roll = np.array([[-c.p[1], 1, 0], [c.p[0], 0, 1]], float)      # point velocity v + w x p = 0
    motion = {BREAK: 0, SLIDE_R: int(np.linalg.matrix_rank(normal_wrench(c)[None, :])),
              ROLL: int(np.linalg.matrix_rank(J_roll))}
    E = wrench_matrix([c])
    wrench_free = {BREAK: 0, SLIDE_R: int(np.linalg.matrix_rank(E[:, :1])), ROLL: int(np.linalg.matrix_rank(E))}
    err = sum(abs(motion[k] - wrench_free[k]) for k in motion) + abs(motion[ROLL] - 2) + abs(motion[SLIDE_R] - 1)
    checks.append(Check("Duality of force and motion freedoms: planar B, S, R give 0, 1, 2 of each",
                        "MR 12.2.4", float(err), 0.0,
                        f"motion constraints {motion}; wrench freedoms {wrench_free}", CATEGORY))
    return checks
