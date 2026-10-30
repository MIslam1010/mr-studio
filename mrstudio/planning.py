"""Motion planning for planar revolute arms, following *Modern Robotics* Chapter 10.

* Configuration-space obstacles of a planar 2R / 3R arm among circular and polygonal workspace obstacles
  (MR Sec. 10.2.1, Fig. 10.2): the links are thick segments (capsules of radius ``r``), a configuration
  q is in C_obs iff some link capsule touches some obstacle (MR Sec. 10.2.2: collision detection).
* The C-space of a revolute joint without limits is a circle, so the 2R C-space is a **torus**
  T^2 = S^1 x S^1 (MR Sec. 2.3): angle differences, distances, edges and grids wrap around at +-pi.
* Grid methods (MR Sec. 10.4): an occupancy grid of the C-space, then graph search with
  Dijkstra or A* (MR Sec. 10.2.4, Algorithm 10.1) with the optimistic (admissible) Euclidean heuristic.
* Sampling methods (MR Sec. 10.5): RRT (Algorithm 10.3) with goal bias, RRT* (Sec. 10.5.1.4),
  PRM (Algorithm 10.4) + A* on the roadmap.  Straight C-space edges (local planner, Sec. 10.5.1.3) are
  checked with a *certified* test: samples + a bound on how far the arm can sweep between them, with
  bisection where the bound is not conclusive -- so no path tunnels through a thin obstacle or a corner.
  Grid edges between neighbouring free cells are checked the same way.
* Smoothing (MR Sec. 10.8): random "shortcutting" -- replace a stretch of the path by a straight
  collision-free segment.
* Time scaling (MR Sec. 9.2.2): the planned geometric path theta(s) is turned into a trajectory with the
  polynomial / trapezoidal time scalings of :mod:`mrstudio.trajectory` (``mr.QuinticTimeScaling``).

Everything is vectorised with NumPy so a full 2R C-space (tens of thousands of configurations) takes
milliseconds; the planners report every node they expand / add so the GUI can show the search.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .checks import Check
from .robots import ROBOTS, SerialRobot, get_robot

CAT = "Motion planning"
TWO_PI = 2.0 * np.pi
PLANNERS = ("A* (grid)", "Dijkstra (grid)", "PRM", "RRT", "RRT*")


# ============================================================================ obstacles (workspace)
@dataclass
class Circle:
    c: np.ndarray            # centre (2,)
    r: float                 # radius

    def __post_init__(self):
        self.c = np.asarray(self.c, float).reshape(2)
        self.r = float(self.r)


@dataclass
class Polygon:
    pts: np.ndarray          # (k, 2) vertices (convex or not, either orientation)

    def __post_init__(self):
        self.pts = np.asarray(self.pts, float).reshape(-1, 2)


def box(x0, y0, x1, y1) -> Polygon:
    return Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


# ============================================================================ vectorised 2-D geometry
def _point_segment_dist(P, A, B):
    """Distance from points P (..., 2) to segments A-B (..., 2) (broadcast)."""
    abx, aby = B[..., 0] - A[..., 0], B[..., 1] - A[..., 1]
    apx, apy = P[..., 0] - A[..., 0], P[..., 1] - A[..., 1]
    L2 = np.maximum(abx * abx + aby * aby, 1e-300)
    t = np.clip((apx * abx + apy * aby) / L2, 0.0, 1.0)
    dx, dy = t * abx - apx, t * aby - apy
    return np.sqrt(dx * dx + dy * dy)


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def _segments_intersect(A, B, C, D):
    """Proper (crossing) intersection of segments A-B and C-D (broadcast).  Touching / collinear contacts are
    left to the point-segment distances in :func:`_segment_segment_dist` (a non-strict test would report
    collinear but disjoint segments as intersecting)."""
    dcx, dcy = D[..., 0] - C[..., 0], D[..., 1] - C[..., 1]
    bax, bay = B[..., 0] - A[..., 0], B[..., 1] - A[..., 1]
    d1 = _cross2(dcx, dcy, A[..., 0] - C[..., 0], A[..., 1] - C[..., 1])
    d2 = _cross2(dcx, dcy, B[..., 0] - C[..., 0], B[..., 1] - C[..., 1])
    d3 = _cross2(bax, bay, C[..., 0] - A[..., 0], C[..., 1] - A[..., 1])
    d4 = _cross2(bax, bay, D[..., 0] - A[..., 0], D[..., 1] - A[..., 1])
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _segment_segment_dist(A, B, C, D):
    d = np.minimum(np.minimum(_point_segment_dist(A, C, D), _point_segment_dist(B, C, D)),
                   np.minimum(_point_segment_dist(C, A, B), _point_segment_dist(D, A, B)))
    return np.where(_segments_intersect(A, B, C, D), 0.0, d)


def _point_in_polygon(P, poly):
    """Even-odd rule, P (..., 2), poly (k, 2) -> bool (...)."""
    x, y = P[..., 0], P[..., 1]
    inside = np.zeros(x.shape, bool)
    k = len(poly)
    for i in range(k):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % k]
        if y1 == y2:
            continue
        cond = (y1 > y) != (y2 > y)
        xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xin)
    return inside


def segment_obstacle_dist(A, B, ob) -> np.ndarray:
    """Exact distance between segments A-B (..., 2) and an obstacle (0 when they overlap)."""
    if isinstance(ob, Circle):
        return np.maximum(_point_segment_dist(ob.c, A, B) - ob.r, 0.0)
    P = ob.pts
    C, D = P, np.roll(P, -1, axis=0)                              # polygon edges (k, 2)
    d = _segment_segment_dist(A[..., None, :], B[..., None, :], C, D).min(axis=-1)
    return np.where(_point_in_polygon(A, P), 0.0, d)   # segment entirely inside the polygon


# ============================================================================ the arm
@dataclass
class PlanarArm:
    """A planar chain of revolute joints whose axes are all parallel to z (MR Fig. 10.2).

    Built from the PoE data of a :class:`SerialRobot`: joint i passes through the point q_i of its screw
    axis S_i = (w_i, v_i) (q_i = w_i x v_i) and turns by s_i theta_i with s_i = w_iz = +-1.  In the plane,
    the vector from joint i to joint i+1 is the home vector d_i rotated by phi_i = sum_{j<=i} s_j theta_j,
    which is exactly what the product of exponentials e^[S1]th1 ... e^[Sn]thn gives (checked in
    :func:`validation_checks`)."""
    name: str
    base: np.ndarray          # q_1 (2,)
    vecs: np.ndarray          # (n, 2) home vectors joint i -> joint i+1 (last: joint n -> end-effector)
    signs: np.ndarray         # (n,) +-1
    radius: float             # link half-thickness [m]
    limits: np.ndarray        # (n, 2) joint limits [rad]

    @property
    def n(self) -> int:
        return len(self.vecs)

    @property
    def reach(self) -> float:
        return float(np.linalg.norm(self.base) + np.sum(np.linalg.norm(self.vecs, axis=1)))

    @property
    def link_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.vecs, axis=1)

    @classmethod
    def from_robot(cls, robot: SerialRobot) -> "PlanarArm":
        if not is_planar_revolute(robot):
            raise ValueError(f"{robot.name} is not a planar revolute arm (all joint axes along z)")
        S = robot.Slist
        q = []
        for i in range(robot.n):
            w, v = S[:3, i], S[3:, i]
            q.append(np.cross(w, v)[:2])          # point on the axis (v = -w x q  =>  q_perp = w x v)
        pts = np.vstack(q + [robot.M[:2, 3]])
        return cls(robot.name, pts[0].copy(), np.diff(pts, axis=0), np.sign(S[2]).astype(float),
                   float(robot.link_radius), np.asarray(robot.joint_limits, float).copy())

    def points(self, Q) -> np.ndarray:
        """Joint positions + end-effector for configurations Q (N, n) -> (N, n+1, 2)."""
        Q = np.atleast_2d(np.asarray(Q, float))
        phi = np.cumsum(Q * self.signs, axis=1)                         # (N, n)
        c, s = np.cos(phi), np.sin(phi)
        dx = c * self.vecs[:, 0] - s * self.vecs[:, 1]
        dy = s * self.vecs[:, 0] + c * self.vecs[:, 1]
        P = np.zeros((Q.shape[0], self.n + 1, 2))
        P[:, 0] = self.base
        P[:, 1:, 0] = self.base[0] + np.cumsum(dx, axis=1)
        P[:, 1:, 1] = self.base[1] + np.cumsum(dy, axis=1)
        return P


def is_planar_revolute(robot: SerialRobot) -> bool:
    S = np.asarray(robot.Slist, float)
    w = S[:3]
    return bool(robot.n >= 1 and np.all(np.abs(w[:2]) < 1e-9) and np.allclose(np.abs(w[2]), 1.0, atol=1e-9))


def planning_robot_names(max_dof: int = 3) -> list[str]:
    """Registered robots usable by the planner: planar revolute 2R / 3R arms (course robots included)."""
    out = []
    for name in ROBOTS:
        try:
            r = get_robot(name)
        except Exception:   # noqa: BLE001 -- a broken robot factory must not break the planner
            continue
        if 2 <= r.n <= max_dof and is_planar_revolute(r):
            out.append(name)
    return out


# ============================================================================ C-space topology
@dataclass
class Space:
    """Box / torus C-space: joints whose range covers a full turn wrap around (S^1)."""
    lo: np.ndarray
    hi: np.ndarray
    wrap: np.ndarray          # bool (n,)

    @classmethod
    def for_arm(cls, arm: PlanarArm, torus: bool = True) -> "Space":
        lim = arm.limits
        wrap = (lim[:, 1] - lim[:, 0] >= TWO_PI - 1e-6) & torus
        lo = np.where(wrap, -np.pi, np.maximum(lim[:, 0], -np.pi))
        hi = np.where(wrap, np.pi, np.minimum(lim[:, 1], np.pi))
        return cls(lo, hi, wrap)

    @property
    def n(self) -> int:
        return len(self.lo)

    def diff(self, a, b):
        """Shortest displacement from a to b (b - a, wrapped to (-pi, pi] on wrapping joints)."""
        d = np.asarray(b, float) - np.asarray(a, float)
        return np.where(self.wrap, (d + np.pi) % TWO_PI - np.pi, d)

    def dist(self, a, b):
        return np.linalg.norm(self.diff(a, b), axis=-1)

    def canon(self, q):
        """Representative inside [lo, hi] (wrapping joints mapped into [-pi, pi))."""
        q = np.asarray(q, float)
        wrap = np.asarray(self.wrap)[: q.shape[-1]]        # also accepts projected (fewer-column) arrays, e.g. 3R -> 2D
        return np.where(wrap, (q + np.pi) % TWO_PI - np.pi, q)

    def inside(self, Q):
        Q = self.canon(np.atleast_2d(Q))
        return np.all((Q >= self.lo - 1e-12) & (Q <= self.hi + 1e-12), axis=-1)

    def sample(self, rng, m: int) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=(m, self.n))


# ============================================================================ the world
class World:
    """Arm + obstacles + C-space topology; all collision queries are vectorised over configurations."""

    def __init__(self, arm: PlanarArm, obstacles=(), torus: bool = True, edge_res: float = np.radians(4.0)):
        self.arm = arm
        self.obstacles = list(obstacles)
        self.space = Space.for_arm(arm, torus)
        self.edge_res = float(edge_res)
        self.n_checks = 0                        # number of configurations collision-checked

    # -- point queries ----------------------------------------------------------
    def link_clearance(self, Q) -> np.ndarray:
        """(N,) min over links & obstacles of (segment-obstacle distance) - link radius (<0: collision)."""
        Q = np.atleast_2d(np.asarray(Q, float))
        P = self.arm.points(Q)
        A, B = P[:, :-1], P[:, 1:]                       # (N, n, 2) link segments
        d = np.full(Q.shape[0], np.inf)
        for ob in self.obstacles:
            d = np.minimum(d, segment_obstacle_dist(A, B, ob).min(axis=1))
        return d - self.arm.radius

    def collides(self, Q, chunk: int = 200_000) -> np.ndarray:
        """Bool (N,) -- configuration in C_obs (links hit an obstacle) or outside the joint limits."""
        c, ins = self._clearance_inside(Q, chunk)
        return (c <= 0.0) | ~ins

    def free(self, q) -> bool:
        return not bool(self.collides(q)[0])

    def _clearance_inside(self, Q, chunk: int = 200_000):
        """(clearance (N,), inside-limits (N,)) -- clearance = +inf without obstacles."""
        Q = np.atleast_2d(np.asarray(Q, float))
        c = np.full(Q.shape[0], np.inf)
        if self.obstacles:
            for k in range(0, Q.shape[0], chunk):
                c[k:k + chunk] = self.link_clearance(Q[k:k + chunk])
        self.n_checks += Q.shape[0]
        return c, self.space.inside(Q)

    @property
    def sweep_coeffs(self) -> np.ndarray:
        """rho_j = length of the chain from joint j to the tip.  When theta moves by dtheta, no point of the
        arm moves more than  sum_j |dtheta_j| rho_j  (|p_dot| <= sum_j |theta_dot_j| |p - p_j|, p_j = joint j)."""
        return np.cumsum(self.arm.link_lengths[::-1])[::-1]

    # -- edges (local planner, MR Sec. 10.5.1.3): straight line in C-space -------------------------
    def _edge_points(self, a, b):
        d = self.space.diff(a, b)
        m = max(int(np.ceil(np.linalg.norm(d) / self.edge_res)), 1)
        t = np.linspace(0.0, 1.0, m + 1)[:, None]
        return a + t * d

    def edge_free(self, a, b) -> bool:
        return bool(self.edges_free(np.asarray(a, float)[None], np.asarray(b, float)[None])[0])

    def edges_free(self, A, B, max_depth: int = 40) -> np.ndarray:
        """Certified collision test of the straight C-space edges A[e] -> B[e] ((E, n) arrays).

        The edge is sampled every ``edge_res`` rad; a sub-interval of C-space length vector h between two samples
        with clearances c_a, c_b > 0 is *certified* free when c_a + c_b > beta(h) = sum_j |h_j| rho_j (no arm point
        moves more than beta(h) along it, so the clearance can drop by at most beta/2 in between).  Intervals that
        cannot be certified are bisected (vectorised) until certified or a collision is found -- so a free edge
        really is free, not just free at the samples (no tunnelling through thin obstacles or corners)."""
        A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
        E = len(A)
        if E == 0:
            return np.zeros(0, bool)
        D = self.space.diff(A, B)
        m = np.maximum(np.ceil(np.linalg.norm(D, axis=1) / self.edge_res).astype(int), 1)
        idx = np.repeat(np.arange(E), m + 1)
        first = np.repeat(np.cumsum(m + 1) - (m + 1), m + 1)
        t = (np.arange(len(idx)) - first) / m[idx]
        pts = A[idx] + t[:, None] * D[idx]
        c, ins = self._clearance_inside(pts)
        bad = np.zeros(E, bool)
        np.logical_or.at(bad, idx, (c <= 0.0) | ~ins)
        if not self.obstacles:
            return ~bad
        rho = self.sweep_coeffs
        # sub-intervals k -> k+1 inside each edge
        lo = np.flatnonzero(np.r_[idx[1:] == idx[:-1], False])
        e = idx[lo]
        H = D[e] / m[e][:, None]                                   # C-space step of each sub-interval
        beta = np.abs(H) @ rho
        ca, cb = c[lo], c[lo + 1]
        u = ~bad[e] & ~(ca + cb > beta)                            # not yet certified
        Qa, H, ca, cb, e = pts[lo][u], H[u], ca[u], cb[u], e[u]
        for _ in range(max_depth):
            if len(e) == 0:
                break
            keep = ~bad[e]
            Qa, H, ca, cb, e = Qa[keep], H[keep], ca[keep], cb[keep], e[keep]
            if len(e) == 0:
                break
            H = 0.5 * H
            Qm = Qa + H
            cm, insm = self._clearance_inside(Qm)
            hit = (cm <= 0.0) | ~insm
            bad[e[hit]] = True
            beta = np.abs(H) @ rho
            ok = ~hit
            # two halves: [a, m] and [m, b]
            Qa2 = np.vstack([Qa[ok], Qm[ok]])
            ca2, cb2 = np.r_[ca[ok], cm[ok]], np.r_[cm[ok], cb[ok]]
            H2, e2, b2 = np.vstack([H[ok], H[ok]]), np.r_[e[ok], e[ok]], np.r_[beta[ok], beta[ok]]
            u = ~(ca2 + cb2 > b2)
            Qa, H, ca, cb, e = Qa2[u], H2[u], ca2[u], cb2[u], e2[u]
        else:
            bad[e] = True                                          # grazing contact: conservatively blocked
        return ~bad

    def path_free(self, path) -> bool:
        path = np.asarray(path, float)
        if len(path) == 1:
            return self.free(path[0])
        return bool(np.all(self.edges_free(path[:-1], path[1:])))

    def path_length(self, path) -> float:
        path = np.asarray(path, float)
        return float(np.sum(self.space.dist(path[:-1], path[1:]))) if len(path) > 1 else 0.0

    def fingerprint(self) -> tuple:
        """Hashable description of everything a plan or a C-space grid depends on: the arm (name, geometry,
        link radius, joint limits), the obstacles, the topology (torus / box) and the edge resolution.
        Two worlds with equal fingerprints give identical collision answers."""
        return world_fingerprint(self)


def _rounded(a) -> tuple:
    return tuple(np.round(np.asarray(a, float).ravel(), 12).tolist())


def world_fingerprint(world: "World") -> tuple:
    arm = world.arm
    obs = []
    for ob in world.obstacles:
        if isinstance(ob, Circle):
            obs.append(("circle", _rounded(ob.c), round(float(ob.r), 12)))
        else:
            obs.append(("polygon", _rounded(ob.pts)))
    return (arm.name, _rounded(arm.base), _rounded(arm.vecs), _rounded(arm.signs), round(float(arm.radius), 12),
            _rounded(arm.limits), tuple(obs), tuple(bool(w) for w in world.space.wrap),
            round(world.edge_res, 12))


def unwrap_path(space: Space, pts) -> np.ndarray:
    """Turn a list of configurations into a continuous path (consecutive steps = shortest displacement)."""
    pts = np.asarray(pts, float)
    out = [pts[0].copy()]
    for p in pts[1:]:
        out.append(out[-1] + space.diff(out[-1], p))
    return np.array(out)


# ============================================================================ C-space grid (MR 10.4)
@dataclass
class CGrid:
    axes: list                    # cell-centre coordinates per joint
    occ: np.ndarray               # bool occupancy, shape (k1, ..., kn)
    lo: np.ndarray
    h: np.ndarray                 # cell sizes
    edge_ok: np.ndarray | None = None   # (cells, 3^n - 1) bool: straight edge to neighbour GRID_OFFSETS[j] is free
    fingerprint: tuple | None = None    # World.fingerprint() of the world the grid was computed for

    @property
    def shape(self):
        return self.occ.shape

    def configs(self, flat_idx) -> np.ndarray:
        sub = np.unravel_index(np.asarray(flat_idx), self.occ.shape)
        return np.stack([self.lo[i] + (sub[i] + 0.5) * self.h[i] for i in range(len(sub))], axis=-1)

    def cell_of(self, space: Space, q) -> int:
        q = space.canon(q)
        sub = np.clip(np.floor((q - self.lo) / self.h).astype(int), 0, np.array(self.occ.shape) - 1)
        return int(np.ravel_multi_index(tuple(sub), self.occ.shape))


def grid_offsets(n: int) -> np.ndarray:
    """All neighbour offsets {-1, 0, 1}^n \\ 0 (3^n - 1 of them, fixed order)."""
    return np.array([o for o in itertools.product((-1, 0, 1), repeat=n) if any(o)])


def cspace_grid(world: World, k, progress=None, cancelled=None, edges: bool = True) -> CGrid:
    """Occupancy grid: a cell is in C_obs iff the configuration at its centre collides (MR Sec. 10.4).

    With ``edges`` the straight segment between the centres of every pair of neighbouring free cells is also
    checked (certified :meth:`World.edges_free`, vectorised): a move is allowed only along a free segment, so
    diagonal steps cannot cut the corners of C-obstacles and the grid path is really collision free."""
    sp = world.space
    ks = np.broadcast_to(np.asarray(k, int), (sp.n,))
    h = (sp.hi - sp.lo) / ks
    axes = [sp.lo[i] + (np.arange(ks[i]) + 0.5) * h[i] for i in range(sp.n)]
    f_occ = 0.3 if edges else 1.0
    if sp.n == 2:
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
        occ = world.collides(G).reshape(ks)
    else:   # slab by slab along the first joint (progress + cancel)
        occ = np.zeros(ks, bool)
        rest = np.stack(np.meshgrid(*axes[1:], indexing="ij"), -1).reshape(-1, sp.n - 1)
        for i, a in enumerate(axes[0]):
            if cancelled is not None and cancelled():
                raise InterruptedError
            Q = np.column_stack([np.full(len(rest), a), rest])
            occ[i] = world.collides(Q).reshape(ks[1:])
            if progress is not None:
                progress(f_occ * (i + 1) / ks[0])
    grid = CGrid(axes, occ, sp.lo.copy(), h, fingerprint=world.fingerprint())
    if not edges:
        return grid
    offs = grid_offsets(sp.n)
    lookup = {tuple(o): j for j, o in enumerate(offs.tolist())}
    occf = occ.ravel()
    edge_ok = np.zeros((occ.size, len(offs)), bool)
    free = np.flatnonzero(~occf)
    sub = np.array(np.unravel_index(free, occ.shape)).T
    shape = np.array(occ.shape)
    half = [j for j, o in enumerate(offs.tolist()) if tuple(o) > (0,) * sp.n]   # one of each +-o pair
    for c, j in enumerate(half):
        if cancelled is not None and cancelled():
            raise InterruptedError
        S = sub + offs[j]
        ok = np.ones(len(S), bool)
        for i in range(sp.n):
            if sp.wrap[i]:
                S[:, i] %= shape[i]
            else:
                ok &= (S[:, i] >= 0) & (S[:, i] < shape[i])
        S[~ok] = 0
        v = np.ravel_multi_index(tuple(S.T), occ.shape)
        m = ok & ~occf[v]
        u, v = free[m], v[m]
        res = np.zeros(len(u), bool)
        for c0 in range(0, len(u), 40_000):
            if cancelled is not None and cancelled():
                raise InterruptedError
            res[c0:c0 + 40_000] = world.edges_free(grid.configs(u[c0:c0 + 40_000]), grid.configs(v[c0:c0 + 40_000]))
        edge_ok[u, j] = res
        edge_ok[v, lookup[tuple(-offs[j])]] = res
        if progress is not None:
            progress(f_occ + (1 - f_occ) * (c + 1) / len(half))
    grid.edge_ok = edge_ok
    return grid


def cspace_image(world: World, k: int = 160, fixed=None):
    """(theta1, theta2) picture of the C-space over the full square [-pi, pi)^2 (MR Fig. 10.2, right).

    Returns (axis, obstacle, outside) with ``axis`` the k cell centres [rad] and two (k, k) bool arrays indexed
    [i1, i2]: link-obstacle collision, and outside the joint limits.  For a 3R arm the remaining joints are
    held at ``fixed`` (a slice of the 3-D C-space)."""
    ax = -np.pi + (np.arange(k) + 0.5) * TWO_PI / k
    G = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    if world.space.n > 2:
        G = np.column_stack([G, np.tile(np.asarray(fixed, float)[2:], (len(G), 1))])
    obst = (world.link_clearance(G) <= 0.0) if world.obstacles else np.zeros(len(G), bool)
    outside = ~world.space.inside(G)
    world.n_checks += len(G)
    return ax, obst.reshape(k, k), outside.reshape(k, k)


# ============================================================================ results
@dataclass
class PlanResult:
    planner: str
    success: bool
    path: np.ndarray                         # (m, n) continuous (unwrapped) configurations
    cost: float = np.inf                     # C-space length of the path [rad]
    expanded: np.ndarray | None = None       # configurations in the order they were expanded / added
    edges: np.ndarray | None = None          # (E, 2, n) tree / roadmap edges for display
    n_nodes: int = 0
    n_checks: int = 0                        # collision checks (configurations)
    seconds: float = 0.0
    message: str = ""
    raw_cost: float = np.inf                 # before smoothing
    extra: dict = field(default_factory=dict)
    fingerprint: tuple | None = None         # World.fingerprint() of the world the plan was made for


# ============================================================================ graph search (MR 10.2.4)
def astar(neighbors, start, goal, heuristic=None, cancelled=None):
    """A* (MR Algorithm 10.1); ``heuristic=None`` -> Dijkstra.  neighbors(u) yields (v, cost).

    Returns (path list or None, cost, expanded list in order of expansion (CLOSED))."""
    h = (lambda u: 0.0) if heuristic is None else heuristic
    past = {start: 0.0}
    parent = {start: None}
    closed = set()
    expanded = []
    tie = itertools.count()
    OPEN = [(h(start), next(tie), start)]
    while OPEN:
        _, _, u = heapq.heappop(OPEN)
        if u in closed:
            continue
        closed.add(u)
        expanded.append(u)
        if cancelled is not None and len(expanded) % 2000 == 0 and cancelled():
            raise InterruptedError
        if u == goal:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1], past[u], expanded
        pu = past[u]
        for v, c in neighbors(u):
            if v in closed:
                continue
            tentative = pu + c
            if tentative < past.get(v, np.inf):
                past[v] = tentative
                parent[v] = u
                heapq.heappush(OPEN, (tentative + h(v), next(tie), v))
    return None, np.inf, expanded


def _grid_neighbors(grid: CGrid, space: Space, connect: str = "8"):
    """Neighbour function for graph search on the grid: '8' = all 3^n - 1 neighbours (diagonals), '4' = axis
    moves only (2n neighbours).  Edge cost = Euclidean length of the step.  If the grid carries ``edge_ok``
    (see :func:`cspace_grid`), only moves along a collision-free straight segment are offered."""
    n = grid.occ.ndim
    offs = grid_offsets(n)
    sel = np.ones(len(offs), bool) if connect != "4" else (np.abs(offs).sum(axis=1) == 1)
    costs = np.linalg.norm(offs * grid.h, axis=1)
    shape = np.array(grid.occ.shape)
    occ = grid.occ.ravel()

    def nb(u):
        sub = np.array(np.unravel_index(u, grid.occ.shape))
        S = sub + offs
        ok = sel.copy()
        for i in range(n):
            if space.wrap[i]:
                S[:, i] %= shape[i]
            else:
                ok &= (S[:, i] >= 0) & (S[:, i] < shape[i])
        if grid.edge_ok is not None:
            ok &= grid.edge_ok[u]
        S[~ok] = 0
        idx = np.ravel_multi_index(tuple(S.T), grid.occ.shape)
        ok &= ~occ[idx]
        yield from zip(idx[ok].tolist(), costs[ok].tolist())
    return nb


def _free_cell(world: World, grid: CGrid, q):
    """Grid cell used for configuration q: the nearest free cell (its own, or one up to two cells away on a
    coarse grid) whose centre is reachable from q by a collision-free straight edge."""
    sp = world.space
    c = grid.cell_of(sp, q)
    sub = np.array(np.unravel_index(c, grid.occ.shape))
    offs = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=grid.occ.ndim)))
    S = sub + offs
    shape = np.array(grid.occ.shape)
    ok = np.ones(len(S), bool)
    for i in range(grid.occ.ndim):
        if sp.wrap[i]:
            S[:, i] %= shape[i]
        else:
            ok &= (S[:, i] >= 0) & (S[:, i] < shape[i])
    idx = np.unique(np.ravel_multi_index(tuple(S[ok].T), grid.occ.shape))
    idx = idx[~grid.occ.ravel()[idx]]
    if len(idx) == 0:
        return None
    C = grid.configs(idx)
    order = np.argsort(sp.dist(C, q))
    free = world.edges_free(np.tile(np.asarray(q, float), (len(order), 1)), C[order])
    return int(idx[order[np.argmax(free)]]) if free.any() else None


def grid_search(world: World, grid: CGrid, start, goal, method: str = "astar", connect: str = "8",
                cancelled=None) -> PlanResult:
    """Dijkstra / A* on the occupancy grid (MR Sec. 10.4).  Heuristic: C-space (torus) Euclidean distance
    to the goal cell -- optimistic because every edge cost is the Euclidean length of its step."""
    t0 = time.time()
    sp = world.space
    name = "A* (grid)" if method == "astar" else "Dijkstra (grid)"
    occ = grid.occ.ravel()
    s, g = _free_cell(world, grid, start), _free_cell(world, grid, goal)
    if s is None or g is None:
        return PlanResult(name, False, np.atleast_2d(start), message="start or goal cell is in C_obs",
                          seconds=time.time() - t0)
    gq = grid.configs(g)
    heur = None if method == "dijkstra" else (lambda u: float(sp.dist(grid.configs(u), gq)))
    cells, cost, expanded = astar(_grid_neighbors(grid, sp, connect), s, g, heur, cancelled)
    exp_q = grid.configs(np.array(expanded, int)) if expanded else None
    if cells is None:
        return PlanResult(name, False, np.atleast_2d(start), expanded=exp_q, n_nodes=len(expanded),
                          message="no path on this grid (try a finer grid)", seconds=time.time() - t0)
    pts = [np.asarray(start, float)] + list(grid.configs(np.array(cells, int))) + [np.asarray(goal, float)]
    path = unwrap_path(sp, pts)
    res = PlanResult(name, True, path, world.path_length(path), exp_q, None, len(expanded),
                     seconds=time.time() - t0, extra=dict(grid_cost=cost, cells=cells))
    res.message = f"{len(expanded)} cells expanded, grid path cost {cost:.4f} rad"
    return res


# ============================================================================ PRM (MR 10.5.2)
def prm(world: World, start, goal, n_samples: int = 300, k: int = 10, seed: int = 0,
        progress=None, cancelled=None) -> PlanResult:
    """Probabilistic roadmap (MR Algorithm 10.4): N free samples, each connected to its k nearest
    neighbours by collision-free straight edges; then A* on the roadmap from start to goal."""
    t0 = time.time()
    sp, rng = world.space, np.random.default_rng(seed)
    nodes = [np.asarray(start, float), np.asarray(goal, float)]
    tries = 0
    free_s = []
    while sum(len(f) for f in free_s) < n_samples and tries < 200:
        cand = sp.sample(rng, max(n_samples, 64))
        free_s.append(cand[~world.collides(cand)])
        tries += 1
    S = np.vstack(free_s)[:n_samples] if free_s else np.zeros((0, sp.n))
    X = np.vstack([np.array(nodes), S])
    N = len(X)
    if progress:
        progress(0.2)
    D = np.empty((N, N))
    for r0 in range(0, N, 256):                        # row chunks: no (N, N, n) temporary
        D[r0:r0 + 256] = np.linalg.norm(sp.diff(X[r0:r0 + 256, None, :], X[None, :, :]), axis=-1)
    np.fill_diagonal(D, np.inf)
    kk = min(k, N - 1)
    nbr = np.argsort(D, axis=1)[:, :kk]
    pairs = {tuple(sorted((i, int(j)))) for i in range(N) for j in nbr[i]}
    pairs = np.array(sorted(pairs))
    ok = np.zeros(len(pairs), bool)
    chunk = 200
    for c in range(0, len(pairs), chunk):
        if cancelled is not None and cancelled():
            raise InterruptedError
        p = pairs[c:c + chunk]
        ok[c:c + chunk] = world.edges_free(X[p[:, 0]], X[p[:, 1]])
        if progress:
            progress(0.2 + 0.75 * min(1.0, (c + chunk) / len(pairs)))
    E = pairs[ok]
    adj: dict[int, list] = {i: [] for i in range(N)}
    for i, j in E:
        adj[i].append((int(j), D[i, j]))
        adj[j].append((int(i), D[i, j]))
    path_idx, cost, expanded = astar(lambda u: adj[u], 0, 1, lambda u: float(D[u, 1]) if u != 1 else 0.0)
    edges = np.stack([X[E[:, 0]], X[E[:, 0]] + sp.diff(X[E[:, 0]], X[E[:, 1]])], axis=1) if len(E) else None
    res = PlanResult("PRM", path_idx is not None, np.atleast_2d(start), expanded=X[expanded] if expanded else None,
                     edges=edges, n_nodes=N, seconds=time.time() - t0, extra=dict(roadmap=X, n_edges=len(E)))
    if path_idx is None:
        res.message = f"start and goal not connected by the roadmap ({N} nodes, {len(E)} edges) — add samples"
        return res
    res.path = unwrap_path(sp, X[path_idx])
    res.cost = world.path_length(res.path)
    res.message = f"roadmap {N} nodes / {len(E)} edges, A* expanded {len(expanded)}"
    return res


# ============================================================================ RRT / RRT* (MR 10.5.1)
def rrt(world: World, start, goal, max_iter: int = 3000, step: float = np.radians(15), goal_bias: float = 0.1,
        seed: int = 0, star: bool = False, radius: float | None = None, progress=None,
        cancelled=None) -> PlanResult:
    """RRT (MR Algorithm 10.3) with goal-biased sampling; ``star=True`` gives RRT* (MR Sec. 10.5.1.4):
    each new node picks the cheapest parent among its neighbours within ``radius`` and the neighbours are
    rewired through it when that shortens their path.  RRT stops at the first connection to the goal,
    RRT* uses all ``max_iter`` samples (an anytime algorithm whose cost decreases with the samples)."""
    t0 = time.time()
    sp, rng = world.space, np.random.default_rng(seed)
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    name = "RRT*" if star else "RRT"
    radius = 2.0 * step if radius is None else radius
    cap = max_iter + 2
    X = np.zeros((cap, sp.n))
    X[0] = start
    parent = np.full(cap, -1)
    cost = np.zeros(cap)
    children: list[list[int]] = [[]]
    n = 1
    goal_cands: dict[int, float] = {}     # node -> edge length to the goal (edge already checked free)
    found_at = None
    for it in range(max_iter):
        if cancelled is not None and it % 100 == 0 and cancelled():
            raise InterruptedError
        if progress is not None and it % 100 == 0:
            progress(it / max_iter)
        q_s = goal if rng.random() < goal_bias else sp.sample(rng, 1)[0]
        d_all = sp.dist(X[:n], q_s)
        near = int(np.argmin(d_all))
        d = d_all[near]
        if d < 1e-12:
            continue
        q_new = X[near] + sp.diff(X[near], q_s) * min(1.0, step / d)
        q_new = sp.canon(q_new)
        if not world.edge_free(X[near], q_new):      # local planner (also checks q_new itself)
            continue
        best = near
        if star:                                      # cheapest parent among the neighbours within radius
            dn = sp.dist(X[:n], q_new)
            nb = np.flatnonzero(dn <= radius)
            nb = nb[nb != near]
            free_nb = nb[world.edges_free(X[nb], np.tile(q_new, (len(nb), 1)))] if len(nb) else nb
            cand = np.r_[near, free_nb].astype(int)
            best = int(cand[np.argmin(cost[cand] + dn[cand])])
            nb = free_nb
        i = n
        X[i], parent[i] = q_new, best
        cost[i] = cost[best] + float(sp.dist(X[best], q_new))
        children.append([])
        children[best].append(i)
        n += 1
        if star:                           # rewire the neighbourhood through the new node
            for j in np.r_[nb, near].astype(int):        # edges new<->nb were checked above
                if j == best:
                    continue
                c_new = cost[i] + float(dn[j])
                if c_new + 1e-12 < cost[j]:
                    children[parent[j]].remove(j)
                    parent[j] = i
                    children[i].append(j)
                    delta = c_new - cost[j]
                    stack = [j]
                    while stack:          # propagate the cost decrease to the subtree
                        u = stack.pop()
                        cost[u] += delta
                        stack.extend(children[u])
        dg = float(sp.dist(q_new, goal))
        if dg <= (radius if star else step) and world.edge_free(q_new, goal):
            goal_cands[i] = dg
            if found_at is None:
                found_at = it + 1
            if not star:
                break
    edges = np.stack([X[parent[1:n]], X[parent[1:n]] + sp.diff(X[parent[1:n]], X[1:n])], axis=1) if n > 1 else None
    res = PlanResult(name, bool(goal_cands), np.atleast_2d(start), expanded=X[:n].copy(), edges=edges, n_nodes=n,
                     seconds=time.time() - t0, extra=dict(found_at=found_at))
    if not goal_cands:
        res.message = f"goal not reached after {max_iter} iterations ({n} nodes)"
        return res
    last = min(goal_cands, key=lambda j: cost[j] + goal_cands[j])
    chain = [last]
    while parent[chain[-1]] >= 0:
        chain.append(int(parent[chain[-1]]))
    pts = list(X[chain[::-1]]) + [goal]
    res.path = unwrap_path(sp, pts)
    res.cost = world.path_length(res.path)
    res.message = f"{n} tree nodes, goal first reached at iteration {found_at}"
    return res


# ============================================================================ smoothing (MR 10.8)
def shortcut(world: World, path, iters: int = 200, seed: int = 0) -> np.ndarray:
    """Random shortcutting: pick two points on the path; if the straight C-space segment between them is
    collision free, replace the stretch in between.  The straight segment is the shortest connection (also on
    the torus), so the length can never increase."""
    sp, rng = world.space, np.random.default_rng(seed)
    P = np.asarray(path, float).copy()
    for _ in range(iters):
        if len(P) < 3:
            break
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        L = np.concatenate([[0.0], np.cumsum(seg)])
        if L[-1] < 1e-12:
            break
        a, b = np.sort(rng.uniform(0, L[-1], 2))
        ia, ib = (np.searchsorted(L, [a, b], side="right") - 1).clip(0, len(P) - 2)
        if ia == ib:
            continue
        pa = P[ia] + (a - L[ia]) / max(seg[ia], 1e-15) * (P[ia + 1] - P[ia])
        pb = P[ib] + (b - L[ib]) / max(seg[ib], 1e-15) * (P[ib + 1] - P[ib])
        old = (L[ib + 1] - L[ia + 1]) + (L[ia + 1] - a) - (L[ib + 1] - b)   # path length from a to b
        d = sp.diff(pa, pb)
        if np.linalg.norm(d) >= old - 1e-12 or not world.edge_free(pa, pb):
            continue
        shift = (pa + d) - pb                            # multiple of 2 pi on wrapping joints
        P = np.vstack([P[:ia + 1], pa, pa + d, P[ib + 1:] + shift])
    # deterministic pass ("string pulling"): from each vertex jump to the farthest later vertex visible along a
    # straight free segment -- removes the short zig-zags left by grid / tree paths; by the triangle inequality the
    # length cannot increase.  Only segments whose shortest (torus) displacement equals the unwrapped one are used.
    out = [P[0]]
    i = 0
    while i < len(P) - 1:
        J = np.arange(len(P) - 1, i, -1)
        same = np.all(np.abs(sp.diff(P[i], P[J]) - (P[J] - P[i])) < 1e-9, axis=1)
        J = J[same]
        ok = world.edges_free(np.tile(P[i], (len(J), 1)), P[J]) if len(J) else np.zeros(0, bool)
        j = int(J[np.argmax(ok)]) if ok.any() else i + 1
        out.append(P[j])
        i = j
    P = np.array(out)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-12]
    return P[keep]


# ============================================================================ time scaling (MR 9.2)
def time_scale(path, Tf: float = 4.0, kind: str = "quintic", dt: float = 0.02, rest_at_vias: bool = False):
    """Trajectory theta(t) along the piecewise-linear C-space path.

    * whole path:  theta(t) = theta(s(t)), s = normalised arc length, s(t) from :func:`trajectory.time_scaling`
      (cubic / quintic = ``mr.CubicTimeScaling`` / ``mr.QuinticTimeScaling``, or trapezoidal) -- MR Sec. 9.2.
      The velocity direction jumps at the corners of the path.
    * rest_at_vias: each straight segment gets its own time scaling (zero velocity at every via point,
      segment durations proportional to sqrt(length), so peak accelerations are equalised)."""
    from .trajectory import time_scaling
    P = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.arange(0.0, Tf + 1e-9, dt)
    if len(P) < 2 or seg.sum() < 1e-12:
        Q = np.repeat(P[:1], len(t), axis=0)
        return t, Q, np.zeros_like(Q), np.zeros_like(Q)
    if not rest_at_vias:
        L = np.concatenate([[0.0], np.cumsum(seg)])
        s, sd, sdd = time_scaling(kind, Tf, t)
        arc = s * L[-1]
        k = (np.searchsorted(L, arc, side="right") - 1).clip(0, len(seg) - 1)
        u = (arc - L[k]) / np.maximum(seg[k], 1e-15)
        dirs = np.diff(P, axis=0) / np.maximum(seg, 1e-15)[:, None]       # d theta / d arc
        Q = P[k] + u[:, None] * (P[k + 1] - P[k])
        dQ = dirs[k] * (L[-1] * sd)[:, None]
        ddQ = dirs[k] * (L[-1] * sdd)[:, None]
        return t, Q, dQ, ddQ
    w = np.sqrt(np.maximum(seg, 0.0))
    Ti = Tf * w / w.sum()
    T0 = np.concatenate([[0.0], np.cumsum(Ti)])
    k = (np.searchsorted(T0, t, side="right") - 1).clip(0, len(seg) - 1)
    Q, dQ, ddQ = (np.zeros((len(t), P.shape[1])) for _ in range(3))
    for j in range(len(seg)):
        m = k == j
        if not m.any() or Ti[j] <= 0:
            continue
        s, sd, sdd = time_scaling(kind, Ti[j], t[m] - T0[j])
        D = P[j + 1] - P[j]
        Q[m] = P[j] + s[:, None] * D
        dQ[m] = sd[:, None] * D
        ddQ[m] = sdd[:, None] * D
    return t, Q, dQ, ddQ


# ============================================================================ preset scenes
def _scale(obs, R):
    out = []
    for o in obs:
        out.append(Circle(o.c * R, o.r * R) if isinstance(o, Circle) else Polygon(o.pts * R))
    return out


def _scene_mr():
    # in the spirit of MR Fig. 10.2: one obstacle within reach of link 1 (a full band of theta1 is blocked)
    # and two that only link 2 can hit (islands in the C-space)
    return [Circle([0.05, 0.80], 0.12), Polygon([[-0.62, -0.05], [-0.30, -0.12], [-0.42, -0.42]]),
            Circle([0.72, -0.42], 0.11)]


def _scene_patient():
    # top view of a bedside rehabilitation robot: the patient's torso beside the base, the head, the bed
    # rail behind the robot and an IV pole -- the arm must reach around the patient
    return [box(0.22, 0.30, 0.55, 0.62),               # torso
            Circle([0.86, 0.44], 0.10),                  # head
            box(-0.95, -0.80, 0.95, -0.72),              # bed rail
            Circle([-0.72, 0.40], 0.05)]                 # IV pole


def _scene_eod():
    # side view of an EOD manipulator on a small UGV: chassis (deck + tracks) just below the shoulder, camera
    # mast with its head behind, a low wall in front and debris on the ground -- reach over the wall
    return [box(-0.90, -0.45, 0.30, -0.10),             # UGV chassis / tracks
            box(-0.82, -0.10, -0.75, 0.42),              # camera mast
            Circle([-0.785, 0.50], 0.08),                # camera head
            box(0.68, -0.80, 0.78, 0.18),                # low wall
            Circle([0.58, -0.62], 0.08)]                 # debris


def _scene_narrow():
    # two blocks leave a narrow gap that link 2 has to thread; a post near the base closes the other way round
    return [box(0.68, 0.10, 1.05, 0.60), box(0.68, -0.60, 1.05, -0.06), Circle([-0.42, 0.0], 0.08)]


#: name -> (obstacles in units of the arm reach, start [deg], goal [deg])  (3R: last joint appended)
SCENES = {
    "MR Fig. 10.2-style (three obstacles)": (_scene_mr, [-60, 70, 20], [135, -60, -30]),
    "Patient-side obstacles (rehab, top view)": (_scene_patient, [-40, 40, 30], [120, 40, 60]),
    "EOD vehicle clutter (UGV, side view)": (_scene_eod, [172, -165, -10], [30, -75, -40]),
    "Narrow passage": (_scene_narrow, [70, 20, 0], [-70, -20, 0]),
    "Empty workspace": (lambda: [], [-90, 45, 0], [90, -45, 0]),
}


def _components(world: World, k: int):
    """Connected components of the free cells of a coarse grid (wrap-around joints glued)."""
    from scipy import ndimage
    G = cspace_grid(world, k, edges=False)
    lab, nlab = ndimage.label(~G.occ, structure=np.ones((3,) * G.occ.ndim))
    par = list(range(nlab + 1))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i in np.flatnonzero(world.space.wrap):
        a, b = np.take(lab, 0, axis=i), np.take(lab, -1, axis=i)
        for x, y in zip(a.ravel(), b.ravel()):
            if x and y:
                par[find(x)] = find(y)
    lab = np.vectorize(lambda v: find(v) if v else 0)(lab) if nlab else lab
    return G, lab


def scene(name: str, arm: PlanarArm):
    """(obstacles, start, goal) scaled to the arm's reach.  The preset angles are clipped to the joint
    limits and moved to the nearest free configuration; if start and goal then lie in different
    connected components of C_free (e.g. a course robot with other link lengths / limits), both are moved
    into the largest component so the preset always has a solution."""
    fn, s, g = SCENES[name]
    obs = _scale(fn(), arm.reach)
    w = World(arm, obs)
    sp = w.space

    def prep(q):
        q = np.radians(np.array(list(q) + [0] * 5, float)[:arm.n])
        q = np.where(sp.wrap, sp.canon(q), np.clip(q, sp.lo, sp.hi))
        return nearest_free(w, q)
    s, g = prep(s), prep(g)
    G, lab = _components(w, 48 if arm.n == 2 else 18)
    ls, lg = lab.ravel()[G.cell_of(sp, s)], lab.ravel()[G.cell_of(sp, g)]
    if ls and ls == lg:
        return obs, s, g
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        return obs, s, g
    cells = np.flatnonzero(lab.ravel() == int(np.argmax(sizes)))
    C = G.configs(cells)
    roomy = w.link_clearance(C) > arm.radius            # keep the poses off the C-obstacle boundary
    if roomy.any():
        C = C[roomy]
    s2 = C[np.argmin(sp.dist(C, s))]
    g2 = C[np.argmin(sp.dist(C, g))]
    if sp.dist(s2, g2) < np.radians(40):
        g2 = C[np.argmax(sp.dist(C, s2))]
    return obs, s2, g2


def nearest_free(world: World, q, seed: int = 0, margin: float | None = None) -> np.ndarray:
    """q itself if its link clearance exceeds ``margin`` (default: the link radius), else the closest such
    configuration among growing random perturbations (falls back to merely free, then to q).  A margin keeps
    preset start / goal poses away from the C-obstacle boundary (not wedged into a pocket)."""
    q = np.asarray(q, float)
    margin = world.arm.radius if margin is None else margin
    sp = world.space

    def good(C, mg):
        c, ins = world._clearance_inside(C)
        return ins & (c > mg)
    if good(q[None], margin)[0]:
        return q
    rng = np.random.default_rng(seed)
    for mg in (margin, 0.0):
        for sigma in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
            C = sp.canon(q + rng.normal(0, sigma, (400, sp.n)))
            C = np.where(sp.wrap, C, np.clip(C, sp.lo, sp.hi))
            ok = good(C, mg)
            if ok.any():
                C = C[ok]
                return C[np.argmin(sp.dist(C, q))]
    return q


def plan(world: World, start, goal, planner: str = "A* (grid)", *, grid_k: int = 64, connect: str = "8",
         grid: CGrid | None = None, n_samples: int = 300, k_nn: int = 10, max_iter: int = 3000,
         step: float = np.radians(15), goal_bias: float = 0.1, radius: float | None = None, seed: int = 0,
         smooth_iters: int = 0, progress=None, cancelled=None) -> PlanResult:
    """One entry point for the GUI: run the chosen planner, then (optionally) shortcut smoothing."""
    world.n_checks = 0
    t_total = time.time()
    fp = world.fingerprint()
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if world.collides(start)[0] or world.collides(goal)[0]:
        return PlanResult(planner, False, np.atleast_2d(start), message="start or goal configuration is in collision",
                          fingerprint=fp)
    if planner in ("A* (grid)", "Dijkstra (grid)"):
        # a grid is reused only if it was computed for this very world (same arm, obstacles, topology) and k
        if (grid is None or grid.fingerprint != fp
                or grid.occ.shape != tuple(np.broadcast_to(grid_k, (world.space.n,)))):
            grid = cspace_grid(world, grid_k, progress, cancelled)
        res = grid_search(world, grid, start, goal, "astar" if planner.startswith("A*") else "dijkstra",
                          connect, cancelled)
        res.extra["grid"] = grid
    elif planner == "PRM":
        res = prm(world, start, goal, n_samples, k_nn, seed, progress, cancelled)
    elif planner in ("RRT", "RRT*"):
        res = rrt(world, start, goal, max_iter, step, goal_bias, seed, planner == "RRT*", radius, progress,
                  cancelled)
    else:
        raise ValueError(planner)
    res.raw_cost = res.cost
    res.extra["raw_path"] = res.path.copy()
    res.fingerprint = fp
    if res.success and smooth_iters > 0:
        t0 = time.time()
        res.path = shortcut(world, res.path, smooth_iters, seed)
        res.cost = world.path_length(res.path)
        res.seconds += time.time() - t0
    if res.success and not world.path_free(res.path):       # final certification of the returned path
        res.success = False
        res.message = "the returned path failed the final collision re-check (" + (res.message or res.planner) + ")"
    res.n_checks = world.n_checks
    res.seconds = time.time() - t_total
    return res


# ============================================================================ validation (MR Ch. 10)
def _reference_collision(world: World, Q, m: int = 400):
    """Independent check: m points along every link, point-in-obstacle tests only (no segment formulas).
    Returns (collides, spacing) with spacing = largest gap between sampled points."""
    P = world.arm.points(Q)
    t = np.linspace(0, 1, m)
    hit = np.zeros(len(Q), bool)
    r = world.arm.radius
    for i in range(world.arm.n):
        A, B = P[:, i], P[:, i + 1]
        X = A[:, None, :] + t[None, :, None] * (B - A)[:, None, :]          # (N, m, 2)
        for ob in world.obstacles:
            if isinstance(ob, Circle):
                hit |= np.any(np.linalg.norm(X - ob.c, axis=-1) <= ob.r + r, axis=1)
            else:
                k = len(ob.pts)
                d = np.min(np.stack([_point_segment_dist(X, ob.pts[j], ob.pts[(j + 1) % k]) for j in range(k)]),
                           axis=0)
                hit |= np.any((d <= r) | _point_in_polygon(X, ob.pts), axis=1)
    return hit, float(np.max(world.arm.link_lengths)) / (m - 1)


def dense_min_clearance(world: World, path, res: float = np.radians(0.05)) -> float:
    """Independent re-check of a path: plain sampling every ``res`` rad (no sweep bound), min link clearance;
    -inf if a sample leaves the joint limits."""
    path = np.asarray(path, float)
    pts = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        d = world.space.diff(a, b)
        m = max(int(np.ceil(np.linalg.norm(d) / res)), 1)
        pts.append(a + np.linspace(0.0, 1.0, m + 1)[1:, None] * d)
    Q = np.vstack(pts)
    if not world.space.inside(Q).all():
        return -np.inf
    return float(world.link_clearance(Q).min()) if world.obstacles else np.inf


def _demo_world(robot_name: str, scene_name: str):
    arm = PlanarArm.from_robot(get_robot(robot_name))
    obs, s, g = scene(scene_name, arm)
    return World(arm, obs), s, g


def validation_checks() -> list[Check]:
    import modern_robotics as mr
    out: list[Check] = []
    rng = np.random.default_rng(7)

    # 1. planar chain formula == product of exponentials (joint points and end-effector)
    for name in ("Planar 2R arm", "Planar 3R arm"):
        rob = get_robot(name)
        arm = PlanarArm.from_robot(rob)
        err = 0.0
        for _ in range(20):
            th = rng.uniform(-np.pi, np.pi, rob.n)
            P = arm.points(th)[0]
            T = mr.FKinSpace(rob.M, rob.Slist, th)
            Ts = rob.link_transforms(th)
            q = [np.r_[arm.base, 0.0]] + [np.r_[arm.base + np.sum(arm.vecs[:i], 0), 0.0] for i in range(1, rob.n)]
            jp = np.array([(Ts[i] @ np.r_[q[i], 1.0])[:2] for i in range(rob.n)])
            err = max(err, np.abs(P[-1] - T[:2, 3]).max(), np.abs(P[:-1] - jp).max())
        out.append(Check(f"Planar arm points = PoE e^[S1]θ1…e^[Sn]θn M — {name}", "MR 4.1 / 10.2.1", err, 1e-12,
                         "Joint and end-effector positions used for collision checking vs mr.FKinSpace and the "
                         "PoE link transforms, 20 random θ.", CAT))

    # 2. C-obstacles: exact capsule test vs independent dense point sampling
    for scn in ("MR Fig. 10.2-style (three obstacles)", "Patient-side obstacles (rehab, top view)",
                "EOD vehicle clutter (UGV, side view)"):
        w, _, _ = _demo_world("Planar 2R arm", scn)
        Q = w.space.sample(rng, 4000)
        exact = w.collides(Q)
        ref, hsp = _reference_collision(w, Q)
        clear = w.link_clearance(Q)
        bad = (exact != ref) & (np.abs(clear) > hsp)      # disagreement not explained by the point spacing
        out.append(Check(f"C-obstacle membership: segment test vs sampled points — 2R, {scn.split(' (')[0]}",
                         "MR 10.2.1–10.2.2", float(bad.sum()), 0.0,
                         f"4000 random θ; {int((exact != ref).sum())} boundary disagreements, all within the "
                         f"sampling gap {hsp * 1e3:.2f} mm of the obstacle surface.", CAT))

    # 3. grid indexing: occupancy of random cells == direct collision check at the cell centre
    w, s, g = _demo_world("Planar 2R arm", "MR Fig. 10.2-style (three obstacles)")
    grid = cspace_grid(w, 64, edges=False)
    cells = rng.integers(0, grid.occ.size, 500)
    Q = grid.configs(cells)
    ref, hsp = _reference_collision(w, Q)
    bad = (grid.occ.ravel()[cells] != ref) & (np.abs(w.link_clearance(Q)) > hsp)
    out.append(Check("C-space grid cell ↔ configuration consistent (64×64)", "MR 10.4", float(bad.sum()), 0.0,
                     "500 random grid cells: stored occupancy vs independent sampled check at the cell centre.", CAT))

    # 4. A* (admissible heuristic) is optimal: same cost as Dijkstra, fewer expansions
    for rname, scn, k in (("Planar 2R arm", "MR Fig. 10.2-style (three obstacles)", 64),
                          ("Planar 2R arm", "Narrow passage", 64),
                          ("Planar 3R arm", "Patient-side obstacles (rehab, top view)", 16)):
        w, s, g = _demo_world(rname, scn)
        grid = cspace_grid(w, k)
        a = grid_search(w, grid, s, g, "astar")
        d = grid_search(w, grid, s, g, "dijkstra")
        ok = a.success and d.success
        err = abs(a.extra["grid_cost"] - d.extra["grid_cost"]) if ok else np.inf
        out.append(Check(f"A* cost = Dijkstra cost — {rname.split(' arm')[0]}, {scn.split(' (')[0]}, grid {k}^{w.space.n}",
                         "MR 10.2.4 (Alg. 10.1)", err, 1e-9,
                         f"A* expanded {a.n_nodes} cells vs Dijkstra {d.n_nodes} (heuristic = torus Euclidean "
                         "distance, optimistic).", CAT))
        out.append(Check(f"A* expands no more cells than Dijkstra — {rname.split(' arm')[0]}, {scn.split(' (')[0]}",
                         "MR 10.2.4", float(max(0, a.n_nodes - d.n_nodes)), 0.0,
                         f"{a.n_nodes} ≤ {d.n_nodes}", CAT))

    # 4b. grid shortest path == SciPy's Dijkstra on the same graph (independent implementation)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra
    w, s, g = _demo_world("Planar 2R arm", "Narrow passage")
    grid = cspace_grid(w, 48)
    nb = _grid_neighbors(grid, w.space, "8")
    rows, cols, vals = [], [], []
    for u in np.flatnonzero(~grid.occ.ravel()).tolist():
        for v, c in nb(u):
            rows.append(u), cols.append(v), vals.append(c)
    Gs = csr_matrix((vals, (rows, cols)), shape=(grid.occ.size,) * 2)
    a = grid_search(w, grid, s, g, "astar")
    ref = sp_dijkstra(Gs, indices=a.extra["cells"][0])[a.extra["cells"][-1]] if a.success else np.inf
    out.append(Check("Grid A* cost = scipy.sparse.csgraph.dijkstra — 2R, Narrow passage, 48²", "MR 10.2.4 / 10.4",
                     abs(a.extra["grid_cost"] - ref) if a.success else np.inf, 1e-9,
                     f"{len(vals)} directed edges (corner-cutting moves removed); cost {ref:.4f} rad.", CAT))

    # 5. every planner returns a path that joins start and goal and is collision free -- re-checked by plain
    #    sampling every 0.05° (independent of the certified edge test used while planning)
    for rname, scn in (("Planar 2R arm", "MR Fig. 10.2-style (three obstacles)"),
                       ("Planar 2R arm", "Patient-side obstacles (rehab, top view)"),
                       ("Planar 3R arm", "EOD vehicle clutter (UGV, side view)")):
        w, s, g = _demo_world(rname, scn)
        kg = 64 if w.space.n == 2 else 16
        grid = cspace_grid(w, kg)                     # shared by A* and Dijkstra
        for pl in PLANNERS:
            r = plan(w, s, g, pl, grid_k=kg, grid=grid, max_iter=800 if pl == "RRT*" else 4000,
                     n_samples=400, seed=3, smooth_iters=100)
            short = f"{rname.split(' arm')[0]}, {scn.split(' (')[0]}"
            if not r.success:
                out.append(Check(f"{pl} finds a path — {short}", "MR 10.5", 1.0, 0.0, r.message, CAT))
                continue
            clr = min(dense_min_clearance(w, r.path), dense_min_clearance(w, r.extra["raw_path"]))
            ends = max(float(w.space.dist(r.path[0], s)), float(w.space.dist(r.path[-1], g)))
            out.append(Check(f"{pl} path (raw & smoothed) collision-free & joins start→goal — {short}",
                             "MR 10.4–10.5, 10.8", (0.0 if clr > 0 else 1.0) + ends, 1e-9,
                             f"{len(r.path)} vertices, length {r.cost:.3f} rad; min link clearance "
                             f"{clr * 1e3:.2f} mm sampled every 0.05°.", CAT))

    # 5b. certified edge test is never optimistic: random edges declared free are free under dense sampling
    w, s, g = _demo_world("Planar 2R arm", "MR Fig. 10.2-style (three obstacles)")
    A = w.space.sample(rng, 600)
    B = A + rng.normal(0.0, 0.35, A.shape)
    ok = w.edges_free(A, B)
    wrong = sum(dense_min_clearance(w, np.vstack([a, b]), np.radians(0.02)) <= 0 for a, b in zip(A[ok], B[ok]))
    out.append(Check("Certified edge test (sweep bound + bisection) never misses a collision", "MR 10.5.1.3",
                     float(wrong), 0.0, f"{int(ok.sum())} of 600 random edges declared free; each re-sampled every "
                     "0.02°.", CAT))

    # 5c. regression: collinear but disjoint link / polygon edge is not a collision
    arm = PlanarArm.from_robot(get_robot("Planar 2R arm"))
    R = arm.reach
    wc = World(arm, [box(1.5 * R, 0.0, 2.0 * R, 0.5 * R)])
    out.append(Check("Collinear, disjoint link and obstacle edge: no false collision", "MR 10.2.2",
                     float(wc.collides(np.zeros(2))[0]), 0.0,
                     "Straight arm along +x, box edge on the same line beyond the reach.", CAT))

    # 6. smoothing never lengthens a path
    w, s, g = _demo_world("Planar 2R arm", "MR Fig. 10.2-style (three obstacles)")
    worst = 0.0
    for seed in range(4):
        r = plan(w, s, g, "RRT", max_iter=4000, seed=seed)
        if not r.success:
            worst = np.inf
            break
        P2 = shortcut(w, r.path, 150, seed)
        worst = max(worst, w.path_length(P2) - r.cost)
        if dense_min_clearance(w, P2) <= 0:
            worst = np.inf
    out.append(Check("Shortcut smoothing never increases length & stays collision-free (4 RRT paths)",
                     "MR 10.8", max(worst, 0.0), 1e-12, "max(length after − length before) over 4 seeds.", CAT))

    # 7. time scaling reproduces the path end points with zero end velocities (quintic, MR 9.2.2.1)
    r = plan(w, s, g, "A* (grid)", smooth_iters=100)
    t, Q, dQ, ddQ = time_scale(r.path, 3.0, "quintic", 0.01)
    e = max(np.abs(Q[0] - r.path[0]).max(), np.abs(Q[-1] - r.path[-1]).max(), np.abs(dQ[0]).max(),
            np.abs(dQ[-1]).max())
    from scipy.integrate import cumulative_trapezoid
    e2 = np.abs(Q[0] + cumulative_trapezoid(dQ, t, axis=0, initial=0) - Q).max() / np.abs(np.diff(r.path, axis=0)).sum()
    out.append(Check("Time-scaled path: θ(0), θ(T) = path ends, θ̇(0) = θ̇(T) = 0", "MR 9.2.2.1", e, 1e-9,
                     "Quintic s(t) (mr.QuinticTimeScaling) along the arc-length parameterised C-space path.", CAT))
    out.append(Check("Time-scaled path: ∫θ̇ dt = θ(t) − θ(0) with θ̇ = (dθ/ds)·ṡ (relative)", "MR 9.1",
                     float(e2), 1e-3, "Trapezoidal integration of the analytic velocity, dt = 10 ms.", CAT))
    return out
