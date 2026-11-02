"""Wheeled mobile robots: kinematics, odometry and feedback control (Modern Robotics, Ch. 13).

Conventions follow Lynch & Park, Ch. 13 strictly (equation numbers below are those of the MR PDF used
by the app; printed page = PDF page - 20 in Ch. 13):

* chassis configuration ``q = (phi, x, y)`` of {b} in {s}; velocity ``qdot = (phi_dot, x_dot, y_dot)``;
* planar body twist ``V_b = (omega_bz, v_bx, v_by)`` expressed in {b}: Eq. (13.1) / (13.2), p. 515;
* wheel i of an omnidirectional robot: centre (x_i, y_i) and driving direction beta_i in {b}, radius r_i,
  free-sliding angle gamma_i (omniwheel 0, mecanum +-pi/4): ``u_i = h_i(phi) qdot``, Eq. (13.5)-(13.6),
  p. 519; stacked ``u = H(phi) qdot`` (13.7) and ``u = H(0) V_b`` (13.8), p. 520; the three-omniwheel and
  four-mecanum models (13.9)-(13.10), p. 521;
* nonholonomic robots: unicycle (13.13), diff-drive (13.15), car (13.16)-(13.18) and the canonical model
  ``qdot = G(q) u`` with ``u = (v, omega)``, pp. 525-528; reference point P (13.19)-(13.21), pp. 528-529;
* odometry: ``V_b = F dtheta`` with ``F = H^+(0)`` (13.32)-(13.34), p. 549, and the exact integration of a
  constant planar twist (13.35)-(13.36), p. 550;
* feedback: omnidirectional feedforward + PI (13.11), p. 522, and the "linear spring" (13.22), p. 530;
  nonholonomic reference-point control (13.29), p. 545, error coordinates (13.30) and the nonlinear
  feedforward + feedback tracking law (13.31), p. 546;
* Lie bracket of vector fields (13.28), p. 534, Example 13.5 p. 535; se(2) commutator (13.38), p. 560.

Angles are radians inside; the GUI shows degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import modern_robotics as mr
import numpy as np

from .checks import Check

CATEGORY = "Mobile robots"
TWO_PI = 2.0 * math.pi


def wrap(a):
    """Angle(s) wrapped to [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % TWO_PI - math.pi


def _rot(phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]])


# ============================================================================ chassis twist (13.1), (13.2)
def body_twist(phi: float, qdot) -> np.ndarray:
    """V_b = (omega_bz, v_bx, v_by) from qdot = (phi_dot, x_dot, y_dot), MR Eq. (13.1)."""
    c, s = math.cos(phi), math.sin(phi)
    p, xd, yd = np.asarray(qdot, float)
    return np.array([p, c * xd + s * yd, -s * xd + c * yd])


def qdot_from_twist(phi: float, Vb) -> np.ndarray:
    """qdot from the body twist, MR Eq. (13.2)."""
    c, s = math.cos(phi), math.sin(phi)
    w, vx, vy = np.asarray(Vb, float)
    return np.array([w, c * vx - s * vy, s * vx + c * vy])


# ============================================================================ omnidirectional wheels
@dataclass(frozen=True)
class Wheel:
    """Wheel i in {b}: centre (x, y), driving direction beta, sliding angle gamma, radius r (Fig. 13.4)."""
    x: float
    y: float
    beta: float
    gamma: float
    r: float


def h_row(w: Wheel, phi: float = 0.0) -> np.ndarray:
    """h_i(phi) in closed form, MR Eq. (13.6): u_i = h_i(phi) qdot."""
    bg = w.beta + w.gamma
    return np.array([w.x * math.sin(bg) - w.y * math.cos(bg),
                     math.cos(bg + phi), math.sin(bg + phi)]) / (w.r * math.cos(w.gamma))


def h_row_product(w: Wheel, phi: float = 0.0) -> np.ndarray:
    """h_i(phi) as the product of the four transformations of MR Eq. (13.5) (read right to left)."""
    a = np.array([1.0 / w.r, math.tan(w.gamma) / w.r])                       # (13.4)
    to_wheel = np.array([[math.cos(w.beta), math.sin(w.beta)],
                         [-math.sin(w.beta), math.cos(w.beta)]])
    vel_at_wheel = np.array([[-w.y, 1.0, 0.0], [w.x, 0.0, 1.0]])
    c, s = math.cos(phi), math.sin(phi)
    to_body = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])          # (13.1)
    return a @ to_wheel @ vel_at_wheel @ to_body


def H_matrix(wheels, phi: float = 0.0) -> np.ndarray:
    """H(phi) in R^{m x 3}, rows h_i(phi): u = H(phi) qdot (13.7); with phi = 0, u = H(0) V_b (13.8)."""
    return np.array([h_row(w, phi) for w in wheels])


def three_omni_wheels(r: float, d: float) -> list[Wheel]:
    """The three-omniwheel robot of MR Fig. 13.5 (wheels at distance d from the centre of {b})."""
    out = []
    for beta, ang in ((0.0, math.pi / 2), (-2 * math.pi / 3, -math.pi / 6), (2 * math.pi / 3, 7 * math.pi / 6)):
        out.append(Wheel(d * math.cos(ang), d * math.sin(ang), beta, 0.0, r))
    return out


def four_mecanum_wheels(r: float, l: float, w: float) -> list[Wheel]:
    """The four-mecanum robot of MR Fig. 13.5: beta_i = 0, gamma = (-pi/4, pi/4, -pi/4, pi/4); wheels 1..4 at
    (l, w), (l, -w), (-l, -w), (-l, w) in {b} (front-left, front-right, rear-right, rear-left)."""
    g = math.pi / 4
    return [Wheel(l, w, 0.0, -g, r), Wheel(l, -w, 0.0, g, r),
            Wheel(-l, -w, 0.0, -g, r), Wheel(-l, w, 0.0, g, r)]


def diff_drive_wheels(r: float, d: float) -> list[Wheel]:
    """Left and right driven wheels at (0, +d) and (0, -d) as conventional wheels (gamma = 0, no sliding is
    actually allowed: the second row of (13.3) is the nonholonomic constraint)."""
    return [Wheel(0.0, d, 0.0, 0.0, r), Wheel(0.0, -d, 0.0, 0.0, r)]


def H0_three_omni_book(r: float, d: float) -> np.ndarray:
    """MR Eq. (13.9)."""
    s = math.sin(math.pi / 3)
    return np.array([[-d, 1.0, 0.0], [-d, -0.5, -s], [-d, -0.5, s]]) / r


def H0_mecanum_book(r: float, l: float, w: float) -> np.ndarray:
    """MR Eq. (13.10)."""
    a = l + w
    return np.array([[-a, 1, -1], [a, 1, 1], [a, 1, -1], [-a, 1, 1]], float) / r


def F_three_omni_book(r: float, d: float) -> np.ndarray:
    """MR Eq. (13.32): V_b = F dtheta for the three-omniwheel robot."""
    s = math.sin(math.pi / 3)
    return r * np.array([[-1 / (3 * d)] * 3, [2 / 3, -1 / 3, -1 / 3], [0.0, -1 / (2 * s), 1 / (2 * s)]])


def F_mecanum_book(r: float, l: float, w: float) -> np.ndarray:
    """MR Eq. (13.33)."""
    a = 1.0 / (l + w)
    return r / 4 * np.array([[-a, a, a, -a], [1, 1, 1, 1], [-1, 1, -1, 1]], float)


def F_diff_drive_book(r: float, d: float) -> np.ndarray:
    """MR Eq. (13.34): (dtheta_L, dtheta_R) -> V_b."""
    return r * np.array([[-1 / (2 * d), 1 / (2 * d)], [0.5, 0.5], [0.0, 0.0]])


# ============================================================================ nonholonomic models
def unicycle_qdot(phi: float, u1: float, u2: float, r: float) -> np.ndarray:
    """Simplified unicycle (13.13): u1 wheel driving speed, u2 heading turning speed."""
    return np.array([u2, r * math.cos(phi) * u1, r * math.sin(phi) * u1])


def diff_drive_qdot(phi: float, uL: float, uR: float, r: float, d: float) -> np.ndarray:
    """Simplified diff-drive (13.15): wheel angular speeds uL, uR; wheels 2d apart."""
    return np.array([r / (2 * d) * (uR - uL), r / 2 * math.cos(phi) * (uL + uR), r / 2 * math.sin(phi) * (uL + uR)])


def canonical_qdot(phi: float, v: float, omega: float) -> np.ndarray:
    """Canonical simplified nonholonomic model (13.18): qdot = G(q) (v, omega)."""
    return np.array([omega, math.cos(phi) * v, math.sin(phi) * v])


def G_canonical(q) -> np.ndarray:
    phi = float(q[0])
    return np.array([[0.0, 1.0], [math.cos(phi), 0.0], [math.sin(phi), 0.0]])


def unicycle_controls(v: float, omega: float, r: float) -> tuple[float, float]:
    """Control transformation to the unicycle (p. 528): u1 = v / r, u2 = omega."""
    return v / r, omega


def diff_drive_wheel_speeds(v: float, omega: float, r: float, d: float) -> tuple[float, float]:
    """Control transformation to the diff-drive (p. 528): uL = (v - omega d)/r, uR = (v + omega d)/r."""
    return (v - omega * d) / r, (v + omega * d) / r


def car_steering_angle(v: float, omega: float, ell: float) -> float:
    """(13.17): steering angle psi = atan(ell omega / v) of the car with wheelbase ell."""
    if abs(v) < 1e-12:
        return math.copysign(math.pi / 2, omega) if omega else 0.0
    return math.atan(ell * omega / v)


def pfaffian_A(q) -> np.ndarray:
    """A(q) of the canonical model's constraint A(q) qdot = x_dot sin(phi) - y_dot cos(phi) = 0 (p. 528)."""
    return np.array([0.0, math.sin(q[0]), -math.cos(q[0])])


# ---------------------------------------------------------------- reference point P (13.19)-(13.21)
def point_P(q, xr: float, yr: float) -> np.ndarray:
    """World coordinates of the chassis point P at (xr, yr) in {b}, Eq. (13.19)."""
    return np.asarray(q[1:3], float) + _rot(float(q[0])) @ np.array([xr, yr])


def point_P_velocity(phi: float, v: float, omega: float, xr: float, yr: float) -> np.ndarray:
    """(x_P_dot, y_P_dot) from (v, omega): Eq. (13.20) with phi_dot = omega, (x_dot, y_dot) = v (cos, sin)."""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([v * c, v * s]) + omega * np.array([[-s, -c], [c, -s]]) @ np.array([xr, yr])


def controls_for_point(phi: float, pdot, xr: float, yr: float) -> tuple[float, float]:
    """(v, omega) giving the world-frame velocity pdot of P, Eq. (13.21) (singular for xr = 0)."""
    c, s = math.cos(phi), math.sin(phi)
    Jinv = np.array([[xr * c - yr * s, xr * s + yr * c], [-s, c]]) / xr
    v, w = Jinv @ np.asarray(pdot, float)
    return float(v), float(w)


# ---------------------------------------------------------------- Lie bracket (13.28)
def jacobian(g, q, h: float = 1e-6) -> np.ndarray:
    q = np.asarray(q, float)
    cols = []
    for k in range(len(q)):
        e = np.zeros_like(q)
        e[k] = h
        cols.append((np.asarray(g(q + e)) - np.asarray(g(q - e))) / (2 * h))
    return np.array(cols).T


def lie_bracket(gi, gj, q, h: float = 1e-6) -> np.ndarray:
    """[g_i, g_j](q) = (dg_j/dq) g_i - (dg_i/dq) g_j, Eq. (13.28) (central-difference Jacobians)."""
    q = np.asarray(q, float)
    return jacobian(gj, q, h) @ np.asarray(gi(q)) - jacobian(gi, q, h) @ np.asarray(gj(q))


def flow(g, q, t: float, steps: int = 200) -> np.ndarray:
    """F_t^g(q): follow the vector field g for time t (RK4)."""
    q = np.asarray(q, float).copy()
    hstep = t / steps
    for _ in range(steps):
        k1 = np.asarray(g(q))
        k2 = np.asarray(g(q + 0.5 * hstep * k1))
        k3 = np.asarray(g(q + 0.5 * hstep * k2))
        k4 = np.asarray(g(q + hstep * k3))
        q = q + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return q


def se2(Vb) -> np.ndarray:
    """3x3 matrix [V_b] in se(2) of the planar twist (omega, v_x, v_y)."""
    w, vx, vy = Vb
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]])


def se2_vec(A) -> np.ndarray:
    return np.array([A[1, 0], A[0, 2], A[1, 2]])


# ============================================================================ odometry (13.35), (13.36)
def delta_qb(Vb) -> np.ndarray:
    """Displacement (dphi_b, dx_b, dy_b) in {b} of the constant twist V_b followed for unit time, Eq. (13.35)."""
    w, vx, vy = (float(v) for v in Vb)
    if abs(w) < 1e-12:
        return np.array([0.0, vx, vy])
    return np.array([w, (vx * math.sin(w) + vy * (math.cos(w) - 1.0)) / w,
                     (vy * math.sin(w) + vx * (1.0 - math.cos(w))) / w])


def delta_q(Vb, phi_k: float) -> np.ndarray:
    """dq in {s}, Eq. (13.36): rotate dq_b by the chassis angle phi_k."""
    dqb = delta_qb(Vb)
    c, s = math.cos(phi_k), math.sin(phi_k)
    return np.array([dqb[0], c * dqb[1] - s * dqb[2], s * dqb[1] + c * dqb[2]])


def delta_q_expm(Vb, phi_k: float) -> np.ndarray:
    """The same displacement from T_bb' = exp([V_b6]) with modern_robotics.MatrixExp6 (validation)."""
    w, vx, vy = (float(v) for v in Vb)
    T = mr.MatrixExp6(mr.VecTose3([0.0, 0.0, w, vx, vy, 0.0]))
    dphi = math.atan2(T[1, 0], T[0, 0])
    if abs(w) > math.pi:                     # keep the full rotation angle, not its wrapped value
        dphi = w
    p = _rot(phi_k) @ T[:2, 3]
    return np.array([dphi, p[0], p[1]])


def odometry_update(q, F: np.ndarray, dtheta) -> np.ndarray:
    """q_{k+1} = q_k + dq with V_b = F dtheta (13.32)-(13.34) and dq from (13.35)-(13.36)."""
    q = np.asarray(q, float)
    return q + delta_q(F @ np.asarray(dtheta, float), float(q[0]))


# ============================================================================ feedback laws
def error_coords(q, qd) -> np.ndarray:
    """q_e = (phi_e, x_e, y_e), Eq. (13.30): (x - x_d, y - y_d) expressed in the reference frame {d}."""
    phd = float(qd[0])
    c, s = math.cos(phd), math.sin(phd)
    dx, dy = float(q[1] - qd[1]), float(q[2] - qd[2])
    return np.array([float(wrap(q[0] - phd)), c * dx + s * dy, -s * dx + c * dy])


PHI_E_MAX = 1.4   # |phi_e| used in (13.31) is clipped below pi/2 (MR p. 546: the law assumes |phi_e| < pi/2)


def tracking_control(q, qd, vd: float, wd: float, k1: float, k2: float, k3: float) -> tuple[float, float]:
    """Nonlinear feedforward + feedback law, Eq. (13.31)."""
    pe, xe, ye = error_coords(q, qd)
    pe = float(np.clip(pe, -PHI_E_MAX, PHI_E_MAX))
    c, t = math.cos(pe), math.tan(pe)
    v = (vd - k1 * abs(vd) * (xe + ye * t)) / c
    w = wd - (k2 * vd * ye + k3 * abs(vd) * t) * c * c
    return v, w


def omni_control(q, qd, qd_dot, e_int, kp: float, ki: float) -> np.ndarray:
    """Feedforward + PI on the chassis configuration, Eq. (13.11): returns the commanded qdot."""
    e = np.asarray(qd, float) - np.asarray(q, float)
    e[0] = float(wrap(e[0]))
    return np.asarray(qd_dot, float) + kp * e + ki * np.asarray(e_int, float)


def point_control(P, Pd, kp: float, Pd_dot=None) -> np.ndarray:
    """Proportional reference-point law, Eq. (13.29) (+ optional feedforward of the desired P velocity)."""
    out = kp * (np.asarray(Pd, float) - np.asarray(P, float))
    return out if Pd_dot is None else out + np.asarray(Pd_dot, float)


# ============================================================================ robot models
KINDS = {
    "diff": "Differential drive",
    "unicycle": "Unicycle",
    "car": "Car (reverse gear, steering limit)",
    "omni3": "Three omniwheels",
    "mecanum": "Four mecanum wheels (youBot-like)",
}
HOLONOMIC = {"omni3", "mecanum"}


@dataclass
class MobileRobot:
    """Geometry + limits. ``radius_error[i]`` = fractional error of the TRUE radius of wheel i (the
    controller and the odometry use the nominal r: this is how odometry drift is produced)."""
    kind: str = "diff"
    r: float = 0.05            # wheel radius (m)
    d: float = 0.15            # diff/car: half track; omni3: centre-to-wheel distance (m)
    l: float = 0.235           # mecanum: half wheelbase (m)
    w: float = 0.15            # mecanum: half track (m)
    ell: float = 0.30          # car: wheelbase (m)
    u_max: float = 15.0        # wheel driving-speed limit (rad/s)
    omega_max: float = 2.0     # unicycle heading-rate limit (rad/s)
    psi_max: float = math.radians(35.0)   # car steering limit
    radius_error: tuple = ()

    @property
    def holonomic(self) -> bool:
        return self.kind in HOLONOMIC

    def wheels(self, true: bool = False) -> list[Wheel]:
        if self.kind == "omni3":
            ws = three_omni_wheels(self.r, self.d)
        elif self.kind == "mecanum":
            ws = four_mecanum_wheels(self.r, self.l, self.w)
        elif self.kind in ("diff", "car"):
            ws = diff_drive_wheels(self.r, self.d)
        else:                                   # unicycle: one wheel at the origin
            ws = [Wheel(0.0, 0.0, 0.0, 0.0, self.r)]
        if true and self.radius_error:
            ws = [replace(wh, r=wh.r * (1.0 + (self.radius_error[i] if i < len(self.radius_error) else 0.0)))
                  for i, wh in enumerate(ws)]
        return ws

    @property
    def m(self) -> int:
        return {"omni3": 3, "mecanum": 4}.get(self.kind, 2)

    def wheel_names(self) -> list[str]:
        if self.kind in ("diff", "car"):
            return ["u_L", "u_R"]
        if self.kind == "unicycle":
            return ["u_1 (drive)", "u_2 (turn)"]
        return [f"u_{i + 1}" for i in range(self.m)]

    def H0(self, true: bool = False) -> np.ndarray:
        """H(0) (13.8); for the unicycle the 2x3 map V_b -> (u1, u2) = (v_bx / r, omega_bz)."""
        if self.kind == "unicycle":
            r = self.wheels(true)[0].r
            return np.array([[0.0, 1.0 / r, 0.0], [1.0, 0.0, 0.0]])
        return H_matrix(self.wheels(true), 0.0)

    def F(self, true: bool = False) -> np.ndarray:
        """F = H^+(0): wheel increments -> body twist (13.32)-(13.34)."""
        return np.linalg.pinv(self.H0(true))

    def wheel_speeds(self, Vb) -> np.ndarray:
        return self.H0() @ np.asarray(Vb, float)

    def saturate(self, Vb_cmd) -> tuple[np.ndarray, np.ndarray, bool]:
        """Commanded body twist -> (wheel speeds u, executed nominal twist, saturated?) under the limits."""
        Vb = np.asarray(Vb_cmd, float).copy()
        sat = False
        if self.kind == "car":
            vmax = self.r * self.u_max
            v = float(np.clip(Vb[1], -vmax, vmax))
            wlim = abs(v) * math.tan(self.psi_max) / self.ell
            w = float(np.clip(Vb[0], -wlim, wlim))
            sat = (v != Vb[1]) or (abs(w - Vb[0]) > 1e-12)
            Vb = np.array([w, v, 0.0])
        elif self.kind == "unicycle":
            vmax = self.r * self.u_max
            v = float(np.clip(Vb[1], -vmax, vmax))
            w = float(np.clip(Vb[0], -self.omega_max, self.omega_max))
            sat = (v != Vb[1]) or (w != Vb[0])
            Vb = np.array([w, v, 0.0])
        elif self.kind == "diff":
            Vb[2] = 0.0
        u = self.wheel_speeds(Vb)
        if self.kind in ("diff", "omni3", "mecanum"):
            peak = float(np.max(np.abs(u)))
            if peak > self.u_max:               # scale all wheels: keeps the direction of V_b (p. 522)
                u = u * (self.u_max / peak)
                sat = True
        return u, self.F() @ u, sat

    def max_speed(self) -> float:
        return self.r * self.u_max

    def control_set(self) -> np.ndarray:
        """Boundary of the control set (Fig. 13.11) as a closed polyline: (v, omega) for nonholonomic
        robots, the omega_bz = 0 slice (v_bx, v_by) of the feasible twist polytope (Fig. 13.6) otherwise."""
        vm = self.r * self.u_max
        if self.kind == "unicycle":
            W = self.omega_max
            return np.array([[vm, W], [-vm, W], [-vm, -W], [vm, -W], [vm, W]])
        if self.kind == "diff":
            W = vm / self.d
            return np.array([[vm, 0], [0, W], [-vm, 0], [0, -W], [vm, 0]])
        if self.kind == "car":
            k = vm * math.tan(self.psi_max) / self.ell
            return np.array([[0, 0], [vm, k], [vm, -k], [0, 0], [-vm, k], [-vm, -k], [0, 0]])
        return twist_slice_polygon(self.H0(), self.u_max)


def default_robot(kind: str) -> MobileRobot:
    """Illustrative geometry per robot type (mecanum: youBot-like r = 47.5 mm, l = 235 mm, w = 150 mm)."""
    rob = MobileRobot(kind=kind)
    if kind == "mecanum":
        rob.r = 0.0475
    elif kind == "omni3":
        rob.d = 0.2
    elif kind == "car":
        rob.d = 0.12
    return rob


def twist_slice_polygon(H0: np.ndarray, u_max: float) -> np.ndarray:
    """omega_bz = 0 slice of {V_b : |H(0) V_b| <= u_max}: polygon in (v_bx, v_by) (Fig. 13.6, bottom)."""
    A = H0[:, 1:]
    lines = [(a, u_max) for a in A] + [(-a, u_max) for a in A]
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            M = np.array([lines[i][0], lines[j][0]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            p = np.linalg.solve(M, [lines[i][1], lines[j][1]])
            if np.all(np.abs(A @ p) <= u_max * (1 + 1e-9)):
                pts.append(p)
    if not pts:
        return np.zeros((0, 2))
    P = np.unique(np.round(np.array(pts), 12), axis=0)
    P = P[np.argsort(np.arctan2(P[:, 1], P[:, 0]))]
    return np.vstack([P, P[:1]])


# ============================================================================ reference trajectories
@dataclass
class Reference:
    """Desired chassis trajectory q_d(t) with qd_dot and the nominal canonical controls (v_d, omega_d)."""
    name: str
    T: float
    kind: str                      # "tangent": heading along the path (feasible for every robot); "omni"
    path: object = None            # t -> (x, y, xd, yd, xdd, ydd) for tangent / fixed-heading paths
    pose: object = None            # t -> (q_d, qd_dot) for omni-only trajectories
    fixed_heading: float | None = None

    def sample(self, t: float):
        if self.pose is not None:
            qd, qdd = self.pose(t)
            vb = body_twist(qd[0], qdd)
            return np.asarray(qd, float), np.asarray(qdd, float), float(vb[1]), float(qdd[0])
        x, y, xd, yd, xdd, ydd = self.path(t)
        sp2 = xd * xd + yd * yd
        v = math.sqrt(sp2)
        if self.fixed_heading is not None:
            ph = self.fixed_heading
            return np.array([ph, x, y]), np.array([0.0, xd, yd]), v, 0.0
        ph = math.atan2(yd, xd)
        w = (xd * ydd - yd * xdd) / sp2 if sp2 > 1e-18 else 0.0
        return np.array([ph, x, y]), np.array([w, xd, yd]), v, w

    @property
    def omni_only(self) -> bool:
        return self.kind == "omni" or self.fixed_heading is not None


def _circle(R=1.0, v=0.3):
    W = v / R

    def path(t):
        s, c = math.sin(W * t), math.cos(W * t)
        return R * s, R * (1 - c), R * W * c, R * W * s, -R * W * W * s, R * W * W * c
    return path


def _figure8(A=1.2, W=0.25):
    def path(t):
        s1, c1, s2, c2 = math.sin(W * t), math.cos(W * t), math.sin(2 * W * t), math.cos(2 * W * t)
        return (A * s1, 0.5 * A * s2, A * W * c1, A * W * c2, -A * W * W * s1, -2 * A * W * W * s2)
    return path


def _line(v=0.3):
    def path(t):
        return v * t, 0.0, v, 0.0, 0.0, 0.0
    return path


def _slalom(v=0.3, a=0.4, L=2.5):
    k = TWO_PI / L

    def path(t):
        x = v * t
        return (x, a * math.sin(k * x), v, a * k * v * math.cos(k * x), 0.0, -a * k * k * v * v * math.sin(k * x))
    return path


def _square(side=1.0, v=0.25):
    seg = side / v

    def pose(t):
        k = int(min(3, max(0, t // seg)))
        tau = t - k * seg
        corners = [(0, 0), (side, 0), (side, side), (0, side), (0, 0)]
        (x0, y0), (x1, y1) = corners[k], corners[k + 1]
        dx, dy = (x1 - x0) / seg, (y1 - y0) / seg
        return np.array([0.0, x0 + dx * tau, y0 + dy * tau]), np.array([0.0, dx, dy])
    return pose


def _ex1313(t):
    return np.array([t, 0.0, t]), np.array([1.0, 0.0, 1.0])


def _hold(t):
    return np.zeros(3), np.zeros(3)


REFERENCES = {
    "Circle (R = 1 m, 0.3 m/s)": lambda: Reference("circle", 2 * math.pi / 0.3, "tangent", path=_circle()),
    "Figure-eight": lambda: Reference("figure8", 2 * math.pi / 0.25, "tangent", path=_figure8()),
    "Straight line": lambda: Reference("line", 12.0, "tangent", path=_line()),
    "Slalom": lambda: Reference("slalom", 20.0, "tangent", path=_slalom()),
    "Square, constant heading (omni)": lambda: Reference("square", 16.0, "omni", pose=_square()),
    "Ex. 13.13: q_d = (t, 0, t) (omni)": lambda: Reference("ex1313", math.pi, "omni", pose=_ex1313),
    "Hold at the origin, (13.22) (omni)": lambda: Reference("hold", 6.0, "omni", pose=_hold),
}


def make_reference(name: str, fixed_heading: float | None = None) -> Reference:
    ref = REFERENCES[name]()
    if fixed_heading is not None and ref.kind == "tangent":
        ref.fixed_heading = fixed_heading
    return ref


# ============================================================================ closed-loop simulation
CONTROLLERS_NH = {
    "Nonlinear tracking (13.31)": "track",
    "Reference point P (13.29)": "point",
    "Open loop (feedforward only)": "open",
}
CONTROLLERS_OMNI = {
    "Feedforward + PI (13.11)": "pi",
    "Open loop (feedforward only)": "open",
}


@dataclass
class Gains:
    k1: float = 1.0
    k2: float = 4.0
    k3: float = 2.0
    kp: float = 1.5
    ki: float = 0.0
    xr: float = 0.25           # reference point P in {b} (13.19)
    yr: float = 0.0
    feedforward_P: bool = False


@dataclass
class MobileSim:
    t: np.ndarray
    q: np.ndarray              # true chassis configuration (N, 3)
    q_d: np.ndarray            # reference (N, 3)
    q_odo: np.ndarray          # odometry estimate (N, 3)
    u: np.ndarray              # wheel speeds (N, m) commanded = encoder rates
    Vb: np.ndarray             # executed nominal body twist (N, 3)
    theta: np.ndarray          # wheel angles (N, m)
    P: np.ndarray              # reference point P of the true chassis (N, 2)
    P_d: np.ndarray
    err: np.ndarray            # (phi_e, x_e, y_e) of the true chassis, (13.30)
    saturated: np.ndarray
    robot: MobileRobot
    controller: str
    feedback: str
    ref_name: str = ""

    @property
    def track_err(self) -> np.ndarray:
        return np.hypot(self.err[:, 1], self.err[:, 2])

    @property
    def odo_err(self) -> np.ndarray:
        return np.hypot(*(self.q_odo[:, 1:] - self.q[:, 1:]).T)


def simulate(robot: MobileRobot, ref: Reference, controller: str = "track", q0=None,
             gains: Gains | None = None, dt: float = 0.01, T: float | None = None,
             feedback: str = "true") -> MobileSim:
    """Closed-loop kinematic simulation with zero-order-hold wheel speeds.

    Every step: controller (on the true pose or on the odometry estimate) -> commanded twist -> wheel speeds
    (nominal geometry, limits) -> true twist V_b = F_true u -> exact constant-twist step (13.35)-(13.36);
    odometry integrates F_nominal dtheta with the same formulas from the encoder increments dtheta = u dt."""
    g = gains or Gains()
    T = ref.T if T is None else T
    N = int(round(T / dt)) + 1
    t = np.arange(N) * dt
    qd0 = ref.sample(0.0)[0]
    q = np.array(qd0 if q0 is None else q0, float)
    qo = q.copy()
    F_true, F_nom = robot.F(true=True), robot.F()
    m = robot.m
    out = dict(q=np.zeros((N, 3)), qd=np.zeros((N, 3)), qo=np.zeros((N, 3)), u=np.zeros((N, m)),
               Vb=np.zeros((N, 3)), th=np.zeros((N, m)), P=np.zeros((N, 2)), Pd=np.zeros((N, 2)),
               err=np.zeros((N, 3)), sat=np.zeros(N, bool))
    e_int = np.zeros(3)
    theta = np.zeros(m)
    for k in range(N):
        tk = t[k]
        qd, qd_dot, vd, wd = ref.sample(tk)
        qf = q if feedback == "true" else qo
        if robot.holonomic:
            if controller == "pi":
                qdot = omni_control(qf, qd, qd_dot, e_int, g.kp, g.ki)
            else:
                qdot = qd_dot
            Vcmd = body_twist(float(qf[0]), qdot)
        else:
            if controller == "track":
                v, w = tracking_control(qf, qd, vd, wd, g.k1, g.k2, g.k3)
            elif controller == "point":
                P_f = point_P(qf, g.xr, g.yr)
                Pd = point_P(qd, g.xr, g.yr)
                Pd_dot = None
                if g.feedforward_P:
                    Pd_dot = point_P_velocity(float(qd[0]), vd, wd, g.xr, g.yr)
                v, w = controls_for_point(float(qf[0]), point_control(P_f, Pd, g.kp, Pd_dot), g.xr, g.yr)
            else:
                v, w = vd, wd
            Vcmd = np.array([w, v, 0.0])
        u, Vnom, sat = robot.saturate(Vcmd)
        out["q"][k], out["qd"][k], out["qo"][k] = q, qd, qo
        out["u"][k], out["Vb"][k], out["th"][k], out["sat"][k] = u, Vnom, theta, sat
        out["P"][k] = point_P(q, g.xr, g.yr)
        out["Pd"][k] = point_P(qd, g.xr, g.yr)
        out["err"][k] = error_coords(q, qd)
        if k == N - 1:
            break
        # integrate one ZOH step
        e = np.asarray(qd, float) - qf
        e[0] = float(wrap(e[0]))
        e_int = e_int + e * dt
        dth = u * dt
        q = q + delta_q(F_true @ dth, float(q[0]))
        qo = qo + delta_q(F_nom @ dth, float(qo[0]))
        theta = theta + dth
    return MobileSim(t, out["q"], out["qd"], out["qo"], out["u"], out["Vb"], out["th"], out["P"], out["Pd"],
                     out["err"], out["sat"], robot, controller, feedback, ref.name)


# ============================================================================ presets
PRESETS = {
    "Diff-drive: circle, nonlinear tracking (13.31)": dict(
        kind="diff", ref="Circle (R = 1 m, 0.3 m/s)", ctrl="Nonlinear tracking (13.31)",
        q0=(math.radians(25), -0.3, 0.25), gains=dict(k1=1.0, k2=4.0, k3=2.0),
        text="Start 0.39 m and 25° off a 1 m circle; (13.31) pulls x_e, y_e and φ_e to zero while the wheels "
             "stay inside the diamond control set."),
    "Diff-drive: figure-eight, reference point P (13.29)": dict(
        kind="diff", ref="Figure-eight", ctrl="Reference point P (13.29)",
        q0=(0.0, -0.2, 0.2), gains=dict(kp=1.5, xr=0.25, feedforward_P=True),
        text="Only the point P (x_r = 0.25 m ahead of the axle) is controlled, through (13.21), here with the desired "
             "velocity of P fed forward; the chassis lines up with the path because P keeps moving. Untick the "
             "feed-forward for the pure proportional law (13.29): P then lags its target."),
    "Diff-drive: P control makes a cusp (Fig. 13.19)": dict(
        kind="diff", ref="Circle (R = 1 m, 0.3 m/s)", ctrl="Reference point P (13.29)",
        q0=(math.radians(180), 0.4, 0.15), gains=dict(kp=1.5, xr=0.25),
        text="Start facing against the path: the pure proportional law (13.29) first drives the robot backwards "
             "(v < 0) for several seconds, then the chassis turns round in an unintended cusp, as in MR Fig. 13.19 "
             "(right) — nothing in (13.29) fixes the orientation; (13.31) does. Pure P also leaves a lag "
             "|P − P_d| ≈ v_d / k_p."),
    "Car: slalom with steering limit (13.31)": dict(
        kind="car", ref="Slalom", ctrl="Nonlinear tracking (13.31)",
        q0=(0.0, 0.0, -0.3), gains=dict(k1=1.0, k2=4.0, k3=2.0),
        text="Car-like robot (wheelbase 0.30 m, |ψ| ≤ 35°): the bowtie control set of Fig. 13.11 limits ω to "
             "|v| tan ψ_max / ℓ, so corrections need forward speed."),
    "Three omniwheels: Exercise 13.13": dict(
        kind="omni3", ref="Ex. 13.13: q_d = (t, 0, t) (omni)", ctrl="Feedforward + PI (13.11)",
        q0=(-math.pi / 4, 0.5, -0.5), gains=dict(kp=2.0, ki=0.0), u_max=45.0,
        text="MR Exercise 13.13: track q_d(t) = (t, 0, t), t ∈ [0, π], from q(0) = (−π/4, 0.5, −0.5) with "
             "feedforward + proportional control; the error decays as e^(−k_p t) (u = H(φ) q̇, (13.7)). "
             "The fast reference needs wheel speeds up to ~40 rad/s: lower u_max to see saturation."),
    "Mecanum: square at constant heading (13.11)": dict(
        kind="mecanum", ref="Square, constant heading (omni)", ctrl="Feedforward + PI (13.11)",
        q0=(math.radians(20), -0.15, 0.1), gains=dict(kp=1.5, ki=0.3),
        text="Four mecanum wheels drive a square without turning: sideways legs need wheels 1, 3 backwards and "
             "2, 4 forwards (p. 521)."),
    "Omni: stabilise to the origin (13.22)": dict(
        kind="omni3", ref="Hold at the origin, (13.22) (omni)", ctrl="Feedforward + PI (13.11)",
        q0=(math.radians(60), 0.6, -0.4), gains=dict(kp=1.2, ki=0.0),
        text="With q_d = 0 the law (13.11) reduces to the linear spring q̇ = −K q of (13.22): an omnidirectional "
             "robot is linearly controllable — a nonholonomic one is not (Theorem 13.1)."),
    "Odometry drift: diff-drive, 2 % radius error": dict(
        kind="diff", ref="Circle (R = 1 m, 0.3 m/s)", ctrl="Nonlinear tracking (13.31)",
        q0=(0.0, 0.0, 0.0), gains=dict(k1=1.0, k2=4.0, k3=2.0), feedback="odometry", radius_error=(0.02, 0.0),
        text="The left wheel is really 2 % larger than the controller believes. Feedback uses the odometry estimate "
             "(13.34)-(13.36): the dashed estimate stays on the reference while the true path drifts away."),
}


def preset(name: str):
    """(robot, reference, controller code, q0, gains, feedback) of a preset."""
    p = PRESETS[name]
    robot = default_robot(p["kind"])
    robot.radius_error = tuple(p.get("radius_error", ()))
    robot.u_max = float(p.get("u_max", robot.u_max))
    ref = make_reference(p["ref"])
    ctrl = {**CONTROLLERS_NH, **CONTROLLERS_OMNI}[p["ctrl"]]
    return robot, ref, ctrl, np.array(p["q0"], float), Gains(**p.get("gains", {})), p.get("feedback", "true")


# ============================================================================ validation
def validation_checks() -> list[Check]:
    cs: list[Check] = []
    rng = np.random.default_rng(13)

    # 1. h_i(phi): closed form (13.6) vs the product of transformations (13.5)
    err = 0.0
    for _ in range(50):
        w = Wheel(*rng.uniform(-0.4, 0.4, 2), rng.uniform(-math.pi, math.pi), rng.uniform(-1.2, 1.2),
                  rng.uniform(0.02, 0.1))
        ph = rng.uniform(-math.pi, math.pi)
        err = max(err, float(np.max(np.abs(h_row(w, ph) - h_row_product(w, ph)))))
    cs.append(Check("Omni wheel: h_i(φ) closed form (13.6) vs product of transformations (13.5)", "MR 13.2.1",
                    err, 1e-12, "50 random wheels (x_i, y_i, β_i, γ_i, r_i) and chassis angles", CATEGORY))

    # 2-3. book matrices (13.9), (13.10) from the general wheel model
    r, d, l, wd = 0.05, 0.2, 0.235, 0.15
    e9 = float(np.max(np.abs(H_matrix(three_omni_wheels(r, d)) - H0_three_omni_book(r, d))))
    cs.append(Check("Three omniwheels: H(0) from (13.6) with Fig. 13.5 geometry = Eq. (13.9)", "MR 13.2.1", e9,
                    1e-12, "wheels at distance d, β = (0, −2π/3, 2π/3), γ = 0", CATEGORY))
    e10 = float(np.max(np.abs(H_matrix(four_mecanum_wheels(0.0475, l, wd)) - H0_mecanum_book(0.0475, l, wd))))
    cs.append(Check("Four mecanum wheels: H(0) from (13.6) = Eq. (13.10)", "MR 13.2.1", e10, 1e-12,
                    "wheels at (±ℓ, ±w), β = 0, γ = (−π/4, π/4, −π/4, π/4)", CATEGORY))

    # 4. F = H^+(0): (13.32), (13.33), (13.34)
    eF = max(float(np.max(np.abs(np.linalg.pinv(H0_three_omni_book(r, d)) - F_three_omni_book(r, d)))),
             float(np.max(np.abs(np.linalg.pinv(H0_mecanum_book(r, l, wd)) - F_mecanum_book(r, l, wd)))),
             float(np.max(np.abs(np.linalg.pinv(H_matrix(diff_drive_wheels(r, 0.15))) - F_diff_drive_book(r, 0.15)))))
    cs.append(Check("Odometry matrices: pinv(H(0)) = F of (13.32), (13.33), (13.34)", "MR 13.4", eF, 1e-12,
                    "three omniwheels, four mecanum wheels, diff-drive (wheels as γ = 0 rows of (13.6))", CATEGORY))

    # 5. rank conditions (p. 520)
    ranks = (np.linalg.matrix_rank(H0_three_omni_book(r, d)), np.linalg.matrix_rank(H0_mecanum_book(r, l, wd)))
    aligned = [Wheel(0.0, 0.2, 0.0, 0.0, r), Wheel(0.17, -0.1, 0.0, 0.0, r), Wheel(-0.17, -0.1, 0.0, 0.0, r)]
    r_al = np.linalg.matrix_rank(H_matrix(aligned))
    cs.append(Check("rank H(0) = 3 for omni/mecanum; = 2 when all driving and sliding directions are aligned",
                    "MR 13.2.1", float(abs(ranks[0] - 3) + abs(ranks[1] - 3) + abs(r_al - 2)), 0.0,
                    f"ranks {ranks[0]}, {ranks[1]}; aligned omniwheels {r_al} (p. 520)", CATEGORY))

    # 6. pure sideways motion: mecanum pattern (p. 521) vs three omniwheels
    um = H0_mecanum_book(0.0475, l, wd) @ np.array([0.0, 0.0, 0.3])
    pat = um / np.max(np.abs(um))
    uo = H0_three_omni_book(r, d) @ np.array([0.0, 0.0, 0.3])
    vmax_x = 0.0475 * 10.0 / float(np.max(np.abs(H0_mecanum_book(0.0475, l, wd)[:, 1])))
    vmax_y = 0.0475 * 10.0 / float(np.max(np.abs(H0_mecanum_book(0.0475, l, wd)[:, 2])))
    e6 = float(np.max(np.abs(pat - np.array([-1, 1, -1, 1])))) + abs(uo[0]) + abs(uo[1] + uo[2]) \
        + abs(vmax_x - vmax_y) / 0.0475
    cs.append(Check("Pure sideways motion: mecanum wheels (−,+,−,+) at equal speed; omni wheel 1 idle, 2/3 opposite",
                    "MR 13.2.1", e6, 1e-12,
                    f"mecanum u = {np.round(um, 3).tolist()} rad/s, omni u = {np.round(uo, 3).tolist()} rad/s; "
                    "mecanum forward and sideways speed limits equal (p. 521)", CATEGORY))

    # 7. diff-drive = unicycle = canonical model (13.13), (13.15), (13.18) and = H(phi) of its two wheels
    e7 = 0.0
    rr, dd = 0.05, 0.15
    for _ in range(30):
        ph, v, om = rng.uniform(-math.pi, math.pi), rng.uniform(-1, 1), rng.uniform(-2, 2)
        uL, uR = diff_drive_wheel_speeds(v, om, rr, dd)
        u1, u2 = unicycle_controls(v, om, rr)
        qc = canonical_qdot(ph, v, om)
        e7 = max(e7, float(np.max(np.abs(diff_drive_qdot(ph, uL, uR, rr, dd) - qc))),
                 float(np.max(np.abs(unicycle_qdot(ph, u1, u2, rr) - qc))),
                 float(np.max(np.abs(H_matrix(diff_drive_wheels(rr, dd), ph) @ qc - [uL, uR]))),
                 abs(float(pfaffian_A([ph, 0, 0]) @ qc)))
    cs.append(Check("Diff-drive (13.15) = unicycle (13.13) = canonical (13.18) under the p. 528 control maps",
                    "MR 13.3.1", e7, 1e-12, "30 random (φ, v, ω); also u = H(φ) q̇ for the two wheels and A(q) q̇ = 0",
                    CATEGORY))

    # 8. (13.1)/(13.2) inverse pair, and V_b = vee(T^-1 T_dot) with modern_robotics
    e8 = 0.0
    for _ in range(20):
        q = rng.uniform(-2, 2, 3)
        qd = rng.uniform(-1, 1, 3)
        Vb = body_twist(q[0], qd)
        h = 1e-6

        def T_of(qq):
            return mr.RpToTrans(mr.MatrixExp3(mr.VecToso3([0, 0, qq[0]])), [qq[1], qq[2], 0.0])
        Tdot = (T_of(q + h * qd) - T_of(q - h * qd)) / (2 * h)
        V6 = mr.se3ToVec(mr.TransInv(T_of(q)) @ Tdot)
        e8 = max(e8, float(np.max(np.abs(qdot_from_twist(q[0], Vb) - qd))),
                 float(np.max(np.abs(np.array([V6[2], V6[3], V6[4]]) - Vb))))
    cs.append(Check("Chassis twist: (13.1)∘(13.2) = identity and V_b = T⁻¹Ṫ from modern_robotics", "MR 13.1",
                    e8, 1e-8, "20 random q, q̇; central difference of T_sb(q) in SE(3)", CATEGORY))

    # 9. odometry step (13.35)-(13.36) vs MatrixExp6 and vs RK4 integration of (13.2)
    e9a, e9b = 0.0, 0.0
    for Vb in ([0.0, 0.3, -0.2], [0.8, 0.5, 0.1], [-2.5, -0.4, 0.7], [1e-14, 0.2, 0.0]):
        ph = float(rng.uniform(-math.pi, math.pi))
        dq = delta_q(Vb, ph)
        e9a = max(e9a, float(np.max(np.abs(dq - delta_q_expm(Vb, ph)))))
        qT = flow(lambda qq, Vb=Vb: qdot_from_twist(qq[0], Vb), [ph, 0.0, 0.0], 1.0, 400)
        e9b = max(e9b, float(np.max(np.abs(dq - (qT - [ph, 0, 0])))))
    cs.append(Check("Odometry: exact constant-twist step (13.35)-(13.36) vs mr.MatrixExp6 of V_b6", "MR 13.4",
                    e9a, 1e-12, "ω_bz = 0 and ω_bz ≠ 0 branches", CATEGORY))
    cs.append(Check("Odometry: (13.35)-(13.36) vs RK4 integration of q̇ = (13.2) with constant V_b", "MR 13.4",
                    e9b, 1e-10, "400 RK4 steps over the unit interval", CATEGORY))

    # 10. odometry from large encoder increments reproduces an exact circle; Euler integration does not
    rr, dd = 0.05, 0.15
    uL, uR = 4.0, 6.0
    om = rr * (uR - uL) / (2 * dd)
    R = dd * (uR + uL) / (uR - uL)
    q = np.zeros(3)
    qe = np.zeros(3)
    F = F_diff_drive_book(rr, dd)
    dtk, n = 1.0, 10
    for _ in range(n):
        dth = np.array([uL, uR]) * dtk
        q = odometry_update(q, F, dth)
        Vb = F @ dth
        qe = qe + qdot_from_twist(qe[0], Vb)          # naive Euler: dq = (13.2) at the old angle
    tt = n * dtk
    exact = np.array([om * tt, R * math.sin(om * tt), R * (1 - math.cos(om * tt))])
    cs.append(Check("Odometry from 1 s encoder increments on a circle (diff-drive, (13.34)-(13.36)) = exact pose",
                    "MR 13.4", float(np.max(np.abs(q - exact))), 1e-12,
                    f"R = {R:.3f} m, 10 increments; naive Euler on the same increments errs by "
                    f"{np.max(np.abs(qe - exact)) * 100:.1f} cm", CATEGORY))

    # 11. Lie bracket (13.28): Example 13.5, the parallel-parking loop, and the se(2) commutator (13.38)
    g1 = lambda qq: np.array([0.0, math.cos(qq[0]), math.sin(qq[0])])   # noqa: E731
    g2 = lambda qq: np.array([1.0, 0.0, 0.0])                            # noqa: E731
    e11 = 0.0
    for _ in range(10):
        q = rng.uniform(-2, 2, 3)
        e11 = max(e11, float(np.max(np.abs(lie_bracket(g1, g2, q) - [0.0, math.sin(q[0]), -math.cos(q[0])]))))
    cs.append(Check("Lie bracket [g₁, g₂] (13.28) = (0, sin φ, −cos φ) of Example 13.5", "MR 13.3.2", e11, 1e-8,
                    "central-difference ∂g/∂q at 10 random q", CATEGORY))
    q0 = np.array([0.4, 0.1, -0.2])
    eps = 1e-2
    qa = flow(g1, q0, eps, 20)
    qa = flow(g2, qa, eps, 20)
    qa = flow(lambda qq: -g1(qq), qa, eps, 20)
    qa = flow(lambda qq: -g2(qq), qa, eps, 20)
    rel = float(np.linalg.norm((qa - q0) / eps ** 2 - lie_bracket(g1, g2, q0)))
    cs.append(Check("Parallel parking: g₁, g₂, −g₁, −g₂ for ε each moves ε²[g₁, g₂] + O(ε³)", "MR 13.3.2",
                    rel, 3 * eps, f"ε = {eps}: |Δq/ε² − [g₁, g₂]| = {rel:.2e} (order ε)", CATEGORY))
    A1, A2 = se2([0.0, 1.0, 0.0]), se2([1.0, 0.0, 0.0])
    A3 = se2_vec(A1 @ A2 - A2 @ A1)
    e38 = 0.0
    for ph in np.linspace(-3, 3, 7):
        e38 = max(e38, float(np.max(np.abs(qdot_from_twist(ph, A3) - lie_bracket(g1, g2, [ph, 0.0, 0.0])))))
    cs.append(Check("Left-invariant fields: [A₁, A₂] = A₁A₂ − A₂A₁ (13.38) gives the same field as (13.28)",
                    "MR 13.3.2", e38, 1e-8, f"A₃ = {np.round(A3, 3).tolist()} (body twist), extended by (13.2)",
                    CATEGORY))

    # 12. reference point (13.21) inverts (13.20); singular for x_r = 0
    e12 = 0.0
    for _ in range(20):
        ph, v, om = rng.uniform(-3, 3), rng.uniform(-1, 1), rng.uniform(-1, 1)
        xr, yr = rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3)
        pdot = point_P_velocity(ph, v, om, xr, yr)
        e12 = max(e12, abs(controls_for_point(ph, pdot, xr, yr)[0] - v), abs(controls_for_point(ph, pdot, xr, yr)[1] - om))
    J0 = np.column_stack([point_P_velocity(0.3, 1, 0, 0.0, 0.2), point_P_velocity(0.3, 0, 1, 0.0, 0.2)])
    cs.append(Check("Reference point: (13.21) inverts (13.20); J(q) singular on x_r = 0", "MR 13.3.1", e12
                    + abs(float(np.linalg.det(J0))), 1e-12, f"det J at x_r = 0: {np.linalg.det(J0):.1e}", CATEGORY))

    # 13. controllers
    rob = MobileRobot(kind="omni3", d=0.2, u_max=1e3)
    sim = simulate(rob, make_reference("Ex. 13.13: q_d = (t, 0, t) (omni)"), "pi",
                   q0=[-math.pi / 4, 0.5, -0.5], gains=Gains(kp=2.0, ki=0.0), dt=0.001)
    e = sim.q_d - sim.q
    e[:, 0] = wrap(e[:, 0])
    en = np.linalg.norm(e, axis=1)
    dev = float(np.max(np.abs(en - en[0] * np.exp(-2.0 * sim.t)))) / en[0]
    cs.append(Check("Omni feedforward + P (13.11), Exercise 13.13: error follows |e(0)| e^(−k_p t)", "MR 13.2.3",
                    dev, 2e-3, f"max over t ∈ [0, π] of | |e(t)| − |e(0)| e^(−2t) | / |e(0)|; |e(π)| = {en[-1]:.5f} "
                    "(zero-order hold at 1 ms: deviation is O(Δt))", CATEGORY))
    rob = MobileRobot(kind="diff")
    ref = make_reference("Circle (R = 1 m, 0.3 m/s)")
    sim = simulate(rob, ref, "track", q0=[math.radians(25), -0.3, 0.25], gains=Gains(k1=1.0, k2=4.0, k3=2.0),
                   T=2 * ref.T)
    err_end = float(np.max(np.abs(sim.err[-1])))
    cs.append(Check("Diff-drive nonlinear tracking (13.31): error coordinates (13.30) → 0 on a circle", "MR 13.3.4",
                    err_end, 1e-4, f"start |(x_e, y_e)| = {sim.track_err[0]:.3f} m, φ_e = 25°; after two laps "
                    f"max|q_e| = {err_end:.1e}; |A(q) q̇| ≤ "
                    f"{max(abs(float(pfaffian_A(qq) @ canonical_qdot(qq[0], V[1], V[0]))) for qq, V in zip(sim.q, sim.Vb)):.0e}",
                    CATEGORY))
    sim = simulate(rob, make_reference("Figure-eight"), "point", q0=[0.0, -0.2, 0.2],
                   gains=Gains(kp=1.5, xr=0.25, feedforward_P=True), dt=0.001, T=6.0)
    eP = np.linalg.norm(sim.P - sim.P_d, axis=1)
    dev = float(np.max(np.abs(eP - eP[0] * np.exp(-1.5 * sim.t)))) / eP[0]
    cs.append(Check("Reference-point control (13.29) + feed-forward through (13.21): |P − P_d| follows e^(−k_p t)",
                    "MR 13.3.4", dev, 2e-3,
                    f"|P − P_d|: {eP[0]:.3f} m → {eP[-1]:.2e} m in 6 s; max deviation from the exponential "
                    "relative to |P(0) − P_d(0)| (ZOH at 1 ms)", CATEGORY))
    return cs
