"""Haptic rendering on a 1-DOF haptic paddle (Hapkit type): sampled-data virtual walls and springs,
impedance and admittance rendering, a user-hand model and the passivity / stability indicators.

Modern Robotics (MR) Sec. 11.7 (impedance control, pp. 443-447): rendered impedance
``m x'' + b x' + k x = f`` (MR Eq. 11.62), ``Z(s) = ms^2 + bs + k`` (Eq. 11.63), the impedance-control
law (Eq. 11.65, here with M = 0: the device's own mass is felt) and the admittance-control law
``x''_d = M^-1 (f_ext - B x' - K x)`` (Eq. 11.66).  MR p. 446 warns that a large K "coupled with delays,
sensor quantization, and sensor errors, can lead to oscillatory behavior or instability"; this module
makes that quantitative.

Device (illustrative numbers of the SBEG 461 course device, shared ``course_robots.hapkit`` and the
Week 12 Lab 11 ``device.json``): handle lever r_h = 0.090 m, sector r_s = 0.073 m, motor pulley
r_p = 0.0042 m (ratio ~17.4), 0.10 kg sector with its CoM 10 mm below the pivot, I_zz = 1.5e-4 kg m^2,
rotor 1.0e-6 kg m^2, handle damping b = 0.15 N s/m, position resolution 0.02 mm, force limit 2 N,
range +-40 mm, 1 kHz loop.  The handle mass m_h and the gravity stiffness k_g are computed with
``modern_robotics`` (MassMatrix, GravityForces) from the screw model of the paddle.

Sign conventions: x = r_h theta is the handle displacement (m); ``F`` is the force the motor applies
to the handle; the hand / environment applies ``f_hand`` to the handle.  Plant (continuous):
``(m_h + m_H) x'' + b x' + k_g x = F + f_hand`` with the hand mass m_H lumped with the handle.
Controller every T: quantise x, backward-difference velocity, compute F, saturate, hold (ZOH),
optionally delayed by d samples.

Passivity: the sampled-data condition ``b > K T / 2 + |B|`` is quoted from J. E. Colgate and
G. G. Schenkel, "Passivity of a class of sampled-data systems: application to haptic interfaces",
Journal of Robotic Systems 14(1):37-47, 1997 (it is not derived here).  The exact stability of the
linear loop is from the eigenvalues of the discrete-time closed-loop matrix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import modern_robotics as mr
import numpy as np
from scipy.linalg import expm

from .checks import Check

G0 = 9.81
CATEGORY = "Haptics & impedance"

COLGATE_CITATION = ("J. E. Colgate and G. G. Schenkel, “Passivity of a class of sampled-data systems: "
                    "application to haptic interfaces”, J. Robotic Systems 14(1):37–47, 1997")


def _trans(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


# ==========================================================================================
# device
# ==========================================================================================
@dataclass
class HapticDevice:
    """1-DOF haptic paddle.  Geometry/inertia -> handle mass and gravity stiffness via modern_robotics."""
    name: str = "Hapkit-type 1-DOF haptic paddle"
    rh: float = 0.090            # handle lever (pivot -> grip), m
    rs: float = 0.073            # sector radius, m
    rp: float = 0.0042           # motor pulley radius, m
    m_sector: float = 0.10       # moving mass (sector + handle), kg
    r_c: float = 0.010           # its CoM below the pivot, m
    I_c: tuple = (2.0e-4, 1.0e-4, 1.5e-4)   # principal inertia about the CoM, kg m^2
    rotor_inertia: float = 1.0e-6           # motor rotor, kg m^2
    b: float = 0.15              # physical (viscous) damping at the handle, N s/m
    x_res: float = 2.0e-5        # handle position resolution (encoder / MR sensor), m
    force_max: float = 2.0       # derated handle force limit, N
    x_range: float = 0.040       # +- handle range, m
    k_t: float = 0.010           # motor torque constant, N m / A (illustrative)
    i_max: float = 1.5           # motor current limit, A
    T: float = 0.001             # nominal loop period, s
    source: str = ("SBEG 461 course device (course_robots.hapkit; Week 12 Lab 11 device.json) -- "
                   "illustrative values of the open-source Hapkit design class")
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---------------------------------------------------------------- screw model (MR conventions)
    @property
    def ratio(self) -> float:
        """Sector/pulley transmission ratio r_s / r_p."""
        return self.rs / self.rp

    def screw_model(self):
        """(Slist, M, Mlist, Glist, g): one revolute joint about z_s through the pivot, {b} at the grip
        (r_h above the pivot), link CoM r_c below the pivot, gravity -y."""
        Slist = np.array([[0, 0, 1, 0, 0, 0]], float).T
        M = _trans([0, self.rh, 0])
        M01 = _trans([0, -self.r_c, 0])
        M12 = mr.TransInv(M01) @ M
        G1 = np.diag([*self.I_c, self.m_sector, self.m_sector, self.m_sector])
        return Slist, M, [M01, M12], [G1], np.array([0.0, -G0, 0.0])

    @property
    def joint_inertia(self) -> float:
        """Link inertia about the pivot, from mr.MassMatrix at theta = 0 (kg m^2)."""
        if "Ij" not in self._cache:
            S, _, Ml, Gl, _ = self.screw_model()
            self._cache["Ij"] = float(mr.MassMatrix([0.0], Ml, Gl, S)[0, 0])
        return self._cache["Ij"]

    @property
    def m(self) -> float:
        """Effective translational mass at the handle: (I_pivot + J_rotor ratio^2) / r_h^2 (kg)."""
        return (self.joint_inertia + self.rotor_inertia * self.ratio ** 2) / self.rh ** 2

    @property
    def k_g(self) -> float:
        """Gravity stiffness at the handle: (d g(theta)/d theta at 0) / r_h^2 with g = mr.GravityForces (N/m)."""
        if "kg" not in self._cache:
            S, _, Ml, Gl, g = self.screw_model()
            h = 1e-6
            dg = (mr.GravityForces([h], g, Ml, Gl, S)[0] - mr.GravityForces([-h], g, Ml, Gl, S)[0]) / (2 * h)
            self._cache["kg"] = float(dg) / self.rh ** 2
        return self._cache["kg"]

    # ---------------------------------------------------------------- transmission / sensing
    @property
    def tau_max(self) -> float:
        """Joint (pivot) torque limit equivalent to the handle force limit, N m."""
        return self.force_max * self.rh

    @property
    def counts_per_handle_rev(self) -> float:
        return 2 * math.pi * self.rh / self.x_res

    @property
    def counts_per_motor_rev(self) -> float:
        return self.counts_per_handle_rev / self.ratio

    def force_to_current(self, F: float) -> float:
        """Handle force -> motor current (MR Sec. 11.8): tau_joint = F r_h, tau_motor = tau_joint / ratio,
        I = tau_motor / k_t."""
        return F * self.rh / self.ratio / self.k_t

    def summary(self) -> str:
        return (f"m_h = {self.m * 1e3:.2f} g (rotor {100 * self.rotor_inertia * self.ratio**2 / self.rh**2 / self.m:.0f} %), "
                f"b = {self.b:.2f} N·s/m, k_g = {self.k_g:.3f} N/m, ratio r_s/r_p = {self.ratio:.2f}, "
                f"Δx = {self.x_res * 1e3:.3f} mm ({self.counts_per_motor_rev:.0f} counts/motor rev), "
                f"|F| ≤ {self.force_max:.1f} N (τ ≤ {self.tau_max * 1e3:.0f} mN·m, "
                f"I ≤ {self.force_to_current(self.force_max):.2f} A), range ±{self.x_range * 1e3:.0f} mm")


def default_device() -> HapticDevice:
    return HapticDevice()


# ==========================================================================================
# rendered environment + hand
# ==========================================================================================
@dataclass
class Rendering:
    """What the device renders.

    shape  "wall":   unilateral wall occupying side*(x - x0) > 0 (it can only push the hand out);
           "spring": bilateral spring-damper about the set point x0 (e.g. a rehab tunnel cross-section).
    The set point may move: x_set(t) = x0 + v_set * clip(t - t_set0, 0, t_set).
    mode   "impedance": F = -K e - B (v_est - v_set)                (MR Eq. 11.65 with M = 0)
           "admittance": a virtual mass M_v obeys M_v a_v = f_sensor + F_env(x_v, v_v)   (MR Eq. 11.66),
                         and the device follows it with the 1-DOF form of MR's admittance torque,
                         F = m_h a_v + k_g x - f_sensor + Kp (x_v - x) + Kd (v_v - v_est)."""
    mode: str = "impedance"
    shape: str = "wall"
    K: float = 200.0
    B: float = 0.0
    T: float = 0.001
    delay_steps: int = 0
    x0: float = 0.0
    side: int = +1
    v_set: float = 0.0
    t_set0: float = 0.0
    t_set: float = 0.0
    M_v: float = 0.2
    Kp: float = 250.0
    Kd: float = 0.0
    quantize: bool = True
    saturate: bool = True

    def x_set(self, t: float) -> float:
        return self.x0 + self.v_set * min(max(t - self.t_set0, 0.0), self.t_set)

    def v_set_at(self, t: float) -> float:
        return self.v_set if (self.t_set0 <= t < self.t_set0 + self.t_set) else 0.0

    def env_force(self, t: float, x: float, v: float) -> float:
        """Force of the ideal (continuous) virtual environment on a body at (x, v)."""
        c, vc = self.x_set(t), self.v_set_at(t)
        if self.shape == "wall":
            if self.side * (x - c) <= 0:
                return 0.0
            F = -self.K * (x - c) - self.B * (v - vc)
            return min(F, 0.0) if self.side > 0 else max(F, 0.0)
        return -self.K * (x - c) - self.B * (v - vc)

    def env_energy(self, t: float, x: float) -> float:
        c = self.x_set(t)
        if self.shape == "wall" and self.side * (x - c) <= 0:
            return 0.0
        return 0.5 * self.K * (x - c) ** 2


@dataclass
class Hand:
    """The user's hand (or a passive surface) coupled to the handle.

    kind "hand":    gripping hand, mass m lumped with the handle, spring k and damper b towards the hand's
                    intended position x_ref(t) (a quintic move from x_start to x_end, mr.QuinticTimeScaling),
                    plus an optional force pulse (e.g. a spasm) of amplitude f_pulse on [t_on, t_off);
    kind "surface": massless unilateral contact (foam pad, limb, tissue) occupying x > x_start:
                    f = -k pen - b v while pen > 0, it can only push;
    kind "none":    hand off (release test)."""
    kind: str = "hand"
    m: float = 0.05
    b: float = 1.0
    k: float = 150.0
    x_start: float = 0.0
    x_end: float = 0.015
    t_move0: float = 0.2
    t_move: float = 1.0
    f_pulse: float = 0.0
    t_on: float = 0.0
    t_off: float = 0.0

    @property
    def mass(self) -> float:
        return self.m if self.kind == "hand" else 0.0

    def ref(self, t: float) -> tuple[float, float]:
        if self.kind != "hand" or self.t_move <= 0:
            return self.x_start, 0.0
        tau = min(max(t - self.t_move0, 0.0), self.t_move)
        s = mr.QuinticTimeScaling(self.t_move, tau)
        h = 1e-6 * self.t_move   # ds/dt by central difference of mr.QuinticTimeScaling
        ds = 0.0 if tau <= 0 or tau >= self.t_move else (
            mr.QuinticTimeScaling(self.t_move, tau + h) - mr.QuinticTimeScaling(self.t_move, tau - h)) / (2 * h)
        d = self.x_end - self.x_start
        return self.x_start + s * d, ds * d

    def force(self, t: float, x: float, v: float, xr: float = None, vr: float = None) -> float:
        """Force applied by the hand / surface on the handle (excluding the hand's inertia)."""
        if self.kind == "none":
            return 0.0
        if self.kind == "surface":
            pen = x - self.x_start
            if pen <= 0:
                return 0.0
            return min(-self.k * pen - self.b * v, 0.0)
        if xr is None:
            xr, vr = self.ref(t)
        f = self.k * (xr - x) + self.b * (vr - v)
        if self.t_on <= t < self.t_off:
            f += self.f_pulse
        return f

    def stiffness(self) -> float:
        return self.k if self.kind == "hand" else 0.0

    def damping(self) -> float:
        return self.b if self.kind == "hand" else 0.0


# ==========================================================================================
# sampled-data simulation
# ==========================================================================================
@dataclass
class SimResult:
    t: np.ndarray            # sample instants t_k
    x: np.ndarray            # true handle position at t_k
    v: np.ndarray
    x_meas: np.ndarray       # quantised position
    v_est: np.ndarray        # backward-difference velocity
    F_cmd: np.ndarray        # commanded force (after saturation)
    F_app: np.ndarray        # force applied on [t_k, t_k+1) (delayed command)
    f_hand: np.ndarray       # force of the hand / surface on the handle at t_k
    x_ref: np.ndarray        # hand's intended position (or surface position)
    x_set: np.ndarray        # wall position / spring set point
    x_v: np.ndarray          # admittance: virtual-mass position (= nan in impedance mode)
    saturated: np.ndarray
    W_motor: np.ndarray      # int F_app v dt  (energy delivered by the motor to the handle)
    W_ideal: np.ndarray      # int F_ideal v dt: impedance F_ideal = F_env(x, v, t); admittance F_ideal =
    #                          F_env + k_g x + (m_h - M_v) x'' (the force that makes the handle a perfect M_v)
    D_b: np.ndarray          # int b v^2 dt  (dissipated by the device's physical damping)
    E_stored: np.ndarray     # device KE + hand KE/spring + gravity spring + ideal virtual-environment energy

    @property
    def leak(self) -> np.ndarray:
        """Energy generated by sampling / hold / quantisation / delay: work of the real force in excess of
        the ideal continuous impedance on the same trajectory (> 0 = the virtual environment is active)."""
        return self.W_motor - self.W_ideal

    @property
    def net(self) -> np.ndarray:
        """leak - device damping dissipation: passivity of device + sampled renderer needs net <= ~0."""
        return self.leak - self.D_b


def simulate(dev: HapticDevice, ren: Rendering, hand: Hand, t_end: float = 2.0, x0: float = 0.0,
             v0: float = 0.0, dt_max: float = 2e-4) -> SimResult:
    """Simulate the sampled-data loop.  Between samples the plant
    ``(m_h + m_H) x'' + b x' + k_g x = F_app + f_hand(t, x, v)`` is integrated with RK4 (sub-steps
    <= dt_max); the energies W_motor, W_ideal, D_b are integrated alongside.  At each sample the
    controller quantises x, estimates v by backward difference, computes F (impedance or admittance),
    saturates it and holds it (ZOH), delayed by ren.delay_steps samples."""
    T = float(ren.T)
    n = int(round(t_end / T))
    nsub = max(1, int(math.ceil(T / dt_max)))
    h = T / nsub
    m_tot = dev.m + hand.mass
    b, kg, Fmax, xres = dev.b, dev.k_g, dev.force_max, dev.x_res
    d = int(ren.delay_steps)
    imp = ren.mode == "impedance"

    t = np.arange(n + 1) * T
    out = {k: np.zeros(n + 1) for k in ("x", "v", "xm", "ve", "Fc", "Fa", "fh", "xr", "xs", "xv",
                                         "W", "Wi", "D", "E")}
    sat = np.zeros(n + 1, bool)
    buf = [0.0] * d
    x, v, W, Wi, D = float(x0), float(v0), 0.0, 0.0, 0.0
    xm_prev = None
    xv, vv = float(x0), float(v0)          # admittance virtual mass state
    Fa_prev = 0.0
    env, hforce, href = ren.env_force, hand.force, hand.ref

    dm = dev.m - ren.M_v       # admittance: ideal motor force reshapes the felt mass m_h -> M_v

    def deriv(tt, x, v, Fa):
        xr, vr = href(tt)
        fh = hforce(tt, x, v, xr, vr)
        a = (Fa + fh - b * v - kg * x) / m_tot
        Fi = env(tt, x, v) if imp else env(tt, x, v) + kg * x + dm * a
        return v, a, Fa * v, Fi * v, b * v * v

    for k in range(n + 1):
        tk = t[k]
        xm = round(x / xres) * xres if ren.quantize else x
        ve = 0.0 if xm_prev is None else (xm - xm_prev) / T
        xm_prev = xm
        xr, vr = href(tk)
        fh = hforce(tk, x, v, xr, vr)
        if imp:
            c, vc = ren.x_set(tk), ren.v_set_at(tk)
            if ren.shape == "wall":
                if ren.side * (xm - c) <= 0:
                    F = 0.0
                else:
                    F = -ren.K * (xm - c) - ren.B * (ve - vc)
                    F = min(F, 0.0) if ren.side > 0 else max(F, 0.0)
            else:
                F = -ren.K * (xm - c) - ren.B * (ve - vc)
            out["xv"][k] = np.nan
        else:
            # simulated force sensor between hand and handle: hand force minus the hand's inertial load
            a_now = (Fa_prev + fh - b * v - kg * x) / m_tot
            f_s = fh - hand.mass * a_now
            # virtual mass (MR Eq. 11.66), semi-implicit Euler at the controller rate
            av = (f_s + env(tk, xv, vv)) / ren.M_v
            vv += av * T
            xv += vv * T
            # MR's admittance law: tau = J^T (Lambda x''_d + eta - f_ext) + inner PD on the tracking error
            F = dev.m * av + kg * xm - f_s + ren.Kp * (xv - xm) + ren.Kd * (vv - ve)
            out["xv"][k] = xv
        sat[k] = ren.saturate and abs(F) > Fmax
        if ren.saturate:
            F = max(-Fmax, min(Fmax, F))
        buf.append(F)
        Fa = buf.pop(0)
        Fa_prev = Fa
        out["x"][k], out["v"][k], out["xm"][k], out["ve"][k] = x, v, xm, ve
        out["Fc"][k], out["Fa"][k], out["fh"][k] = F, Fa, fh
        out["xr"][k] = xr if hand.kind == "hand" else hand.x_start
        out["xs"][k] = ren.x_set(tk)
        out["W"][k], out["Wi"][k], out["D"][k] = W, Wi, D
        Eh = 0.5 * hand.stiffness() * (x - xr) ** 2 if hand.kind == "hand" else (
            0.5 * hand.k * max(x - hand.x_start, 0.0) ** 2 if hand.kind == "surface" else 0.0)
        out["E"][k] = 0.5 * m_tot * v * v + 0.5 * kg * x * x + Eh + ren.env_energy(tk, x)
        if k == n:
            break
        # ---- RK4 over the hold interval (plain floats: fast)
        tt = tk
        for _ in range(nsub):
            k1 = deriv(tt, x, v, Fa)
            k2 = deriv(tt + h / 2, x + h / 2 * k1[0], v + h / 2 * k1[1], Fa)
            k3 = deriv(tt + h / 2, x + h / 2 * k2[0], v + h / 2 * k2[1], Fa)
            k4 = deriv(tt + h, x + h * k3[0], v + h * k3[1], Fa)
            x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            W += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            Wi += h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            D += h / 6 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
            tt += h
        if not (abs(x) < 1e3 and abs(v) < 1e6):     # diverged: freeze the rest
            for key in out:
                out[key][k + 1:] = np.nan
            break
    return SimResult(t, out["x"], out["v"], out["xm"], out["ve"], out["Fc"], out["Fa"], out["fh"],
                     out["xr"], out["xs"], out["xv"], sat, out["W"], out["Wi"], out["D"], out["E"])


def oscillation_growth(sim: SimResult, t_from: float = 0.2) -> float:
    """Peak |x - x_end| in the last quarter of the run / peak in the first quarter after t_from:
    < 1 decaying, > 1 growing (a quick stability verdict from a simulation)."""
    t = sim.t
    x = sim.x - np.nanmean(sim.x[-max(5, len(t) // 20):]) if np.all(np.isfinite(sim.x)) else sim.x
    if not np.all(np.isfinite(x)):
        return float("inf")
    q = (t[-1] - t_from) / 4
    a = np.abs(x[(t >= t_from) & (t < t_from + q)]).max(initial=0.0)
    z = np.abs(x[t >= t[-1] - q]).max(initial=0.0)
    return float(z / a) if a > 0 else 0.0


# ==========================================================================================
# linear analysis of the sampled loop
# ==========================================================================================
def closed_loop_matrix(m: float, b: float, k: float, K: float, B: float, T: float,
                       delay_steps: int = 0) -> np.ndarray:
    """Exact discrete-time closed-loop matrix of the linear sampled loop (no quantisation/saturation):
    plant m x'' + b x' + k x = F discretised with a zero-order hold (augmented-matrix exponential),
    controller c_k = -K x_k - B (x_k - x_{k-1}) / T, applied force F_k = c_{k-d}.
    State z_k = [x_k, v_k, x_{k-1}, c_{k-1}, ..., c_{k-d}] (dimension 3 + d)."""
    Ac = np.array([[0.0, 1.0], [-k / m, -b / m]])
    Bc = np.array([0.0, 1.0 / m])
    Aug = np.zeros((3, 3))
    Aug[:2, :2], Aug[:2, 2] = Ac, Bc
    E = expm(Aug * T)
    Ad, Bd = E[:2, :2], E[:2, 2]
    nz = 3 + delay_steps
    crow = np.zeros(nz)
    crow[0], crow[2] = -K - B / T, B / T
    A = np.zeros((nz, nz))
    if delay_steps == 0:
        A[:2, :] = np.c_[Ad, np.zeros(2)] + np.outer(Bd, crow)
    else:
        A[:2, :2] = Ad
        A[:2, 2 + delay_steps] = Bd
        A[3, :] = crow
        for j in range(1, delay_steps):
            A[3 + j, 2 + j] = 1.0
    A[2, 0] = 1.0
    return A


def _zoh(m: float, b: float, k: float, T: float):
    Aug = np.zeros((3, 3))
    Aug[:2, :2] = [[0.0, 1.0], [-k / m, -b / m]]
    Aug[1, 2] = 1.0 / m
    E = expm(Aug * T)
    return E[:2, :2], E[:2, 2]


def admittance_loop_matrix(dev: "HapticDevice", ren: "Rendering", hand: "Hand", Kp: float | None = None,
                           T: float | None = None) -> np.ndarray:
    """Exact discrete-time closed-loop matrix of the admittance renderer of ``simulate`` (linear case:
    spring environment, no quantisation/saturation, hand reference at rest, hand in contact).
    State z_k = [x, v, x_{k-1}, x_v, v_v, F_{k-1}, c_{k-1}, ..., c_{k-d}] (dimension 6 + d).
    Built by composing the controller's linear operations as row vectors over z."""
    Kp = ren.Kp if Kp is None else Kp
    T = ren.T if T is None else T
    d = int(ren.delay_steps)
    mH = hand.mass
    kH, bH = (hand.k, hand.b) if hand.kind == "hand" else (0.0, 0.0)
    mt = dev.m + mH
    n = 6 + d
    e = np.eye(n)
    X, V, XP, XV, VV, FP = (e[i] for i in range(6))
    fh = -kH * X - bH * V
    a_now = (FP + fh - dev.b * V - dev.k_g * X) / mt
    f_s = fh - mH * a_now
    av = (f_s - ren.K * XV - ren.B * VV) / ren.M_v
    vv1 = VV + av * T
    xv1 = XV + vv1 * T
    ve = (X - XP) / T
    c = dev.m * av + dev.k_g * X - f_s + Kp * (xv1 - X) + ren.Kd * (vv1 - ve)
    Fa = c if d == 0 else e[6 + d - 1]
    Ad, Bd = _zoh(mt, dev.b + bH, dev.k_g + kH, T)
    A = np.zeros((n, n))
    A[0] = Ad[0, 0] * X + Ad[0, 1] * V + Bd[0] * Fa
    A[1] = Ad[1, 0] * X + Ad[1, 1] * V + Bd[1] * Fa
    A[2], A[3], A[4], A[5] = X, xv1, vv1, Fa
    if d:
        A[6] = c
        for j in range(1, d):
            A[6 + j] = e[6 + j - 1]
    return A


def loop_matrix(dev: "HapticDevice", ren: "Rendering", hand: "Hand", gain: float | None = None,
                T: float | None = None) -> np.ndarray:
    """Exact closed-loop matrix of the current rendering (impedance: K, B; admittance: the whole
    admittance controller with inner gain K_p), hand in contact.  ``gain`` overrides K (or K_p)."""
    T = ren.T if T is None else T
    if ren.mode == "impedance":
        K = ren.K if gain is None else gain
        return closed_loop_matrix(dev.m + hand.mass, dev.b + hand.damping(), dev.k_g + hand.stiffness(),
                                  K, ren.B, T, ren.delay_steps)
    return admittance_loop_matrix(dev, ren, hand, gain, T)


def max_stable_gain(dev: "HapticDevice", ren: "Rendering", hand: "Hand", T: float | None = None,
                    K_hi: float = 2e5, tol: float = 0.5) -> float:
    """Largest K (impedance) or K_p (admittance) with spectral radius < 1, by bisection from a stable
    low gain (admittance: K_p -> 0 leaves the handle drifting, so the search starts at 1 N/m)."""
    def ok(g):
        return spectral_radius(loop_matrix(dev, ren, hand, g, T)) < 1 + RHO_TOL
    lo = 0.0 if ren.mode == "impedance" else 1.0
    if not ok(lo):
        # admittance: the stable K_p set may not start at ~0 -> find its first stable gain on a log grid
        grid = np.geomspace(max(lo, 1.0), K_hi, 80) if ren.mode == "admittance" else []
        first = next((g for g in grid if ok(g)), None)
        if first is None:
            return 0.0
        lo = float(first)
        nxt = next((g for g in grid if g > lo and not ok(g)), None)
        if nxt is None:
            return K_hi
        hi = float(nxt)
        while hi - lo > tol * max(1.0, lo * 1e-3):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo
    hi = K_hi
    if ok(hi):
        return hi
    while hi - lo > tol * max(1.0, lo * 1e-3):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


RHO_TOL = 1e-9   # |lambda| = 1 to round-off counts as (marginally) stable, e.g. an undamped virtual mass


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def max_stable_K(m: float, b: float, k: float, B: float, T: float, delay_steps: int = 0,
                 K_hi: float = 2e5, tol: float = 0.01) -> float:
    """Largest K with spectral radius < 1 (bisection; the loop is assumed stable at K = 0)."""
    lo, hi = 0.0, K_hi
    if spectral_radius(closed_loop_matrix(m, b, k, hi, B, T, delay_steps)) < 1:
        return hi
    if spectral_radius(closed_loop_matrix(m, b, k, 0.0, B, T, delay_steps)) >= 1:
        return 0.0
    while hi - lo > tol * max(1.0, lo * 1e-3):
        mid = 0.5 * (lo + hi)
        if spectral_radius(closed_loop_matrix(m, b, k, mid, B, T, delay_steps)) < 1:
            lo = mid
        else:
            hi = mid
    return lo


def first_order_K_limit(b: float, B: float, T: float, delay_steps: int = 0) -> float:
    """Taylor estimate: a spring late by T/2 + dT acts as negative damping K (T/2 + dT), so
    K_max ~ (b + B) / (T/2 + d T)  (SBEG 461 Week 12, Sec. 6.2)."""
    return (b + B) / (T / 2 + delay_steps * T)


def colgate_margin(b: float, K: float, B: float, T: float) -> float:
    """b - (K T / 2 + |B|): the sampled-data passivity condition b > K T/2 + |B| of Colgate & Schenkel
    (1997) holds iff the margin is > 0.  Quoted result (no loop delay, backward-difference velocity)."""
    return b - (K * T / 2 + abs(B))


def colgate_K_max(b: float, B: float, T: float) -> float:
    """Largest stiffness that keeps the Colgate-Schenkel condition: 2 (b - |B|) / T (0 if |B| >= b)."""
    return max(0.0, 2 * (b - abs(B)) / T)


@dataclass
class StabilityReport:
    rho: float                 # spectral radius of the linear sampled loop (with the hand, in contact)
    stable: bool
    K_max_exact: float         # with this hand
    K_max_device: float        # device alone (hand off)
    K_estimate: float          # first-order estimate (device alone)
    margin: float              # Colgate margin b - (KT/2 + |B|)
    passive: bool
    K_passive: float
    K_eff: float               # stiffness/damping actually analysed (Kp, Kd in admittance mode)
    B_eff: float


def analyse(dev: HapticDevice, ren: Rendering, hand: Hand) -> StabilityReport:
    """Linear stability (exact, eigenvalues) and passivity (Colgate-Schenkel) of the current setting.
    Impedance mode: the sampled spring-damper K, B.  Admittance mode: the exact matrix of the whole
    admittance controller (virtual mass, force feedback, inner loop K_p, K_d); the Colgate condition is
    then applied to the inner sampled spring K_p, K_d (an indication only).  The "hand" is taken in
    contact (bilateral); a soft surface is not included."""
    K, B = (ren.K, ren.B) if ren.mode == "impedance" else (ren.Kp, ren.Kd)
    off = replace(hand, kind="none")
    rho = spectral_radius(loop_matrix(dev, ren, hand))
    Kx = max_stable_gain(dev, ren, hand)
    Kd = max_stable_gain(dev, ren, off)
    est = first_order_K_limit(dev.b, B, ren.T, ren.delay_steps) if ren.mode == "impedance" else float("nan")
    mg = colgate_margin(dev.b, K, B, ren.T)
    return StabilityReport(rho, rho < 1 + RHO_TOL, Kx, Kd, est,
                           mg, mg > 0 and ren.delay_steps == 0, colgate_K_max(dev.b, B, ren.T), K, B)


def stability_map(dev: HapticDevice, B: float, hand: Hand | None = None, delay_steps: int = 0,
                  T_values=None, ren: Rendering | None = None) -> dict:
    """K_max(T) curves: exact (device alone and with the hand), Colgate-Schenkel passivity bound.
    With an admittance ``ren`` the exact curves are the largest stable inner gain K_p(T)."""
    T_values = np.geomspace(1e-4, 1e-2, 24) if T_values is None else np.asarray(T_values)
    if ren is None:
        ren = Rendering(shape="spring", B=B, delay_steps=delay_steps)
    else:
        B = ren.B if ren.mode == "impedance" else ren.Kd
        delay_steps = ren.delay_steps
    off = Hand(kind="none")
    dev_K = np.array([max_stable_gain(dev, ren, off, T) for T in T_values])
    out = dict(T=T_values, exact_device=dev_K,
               passive=np.array([colgate_K_max(dev.b, B, T) for T in T_values]))
    if hand is not None and hand.kind == "hand":
        out["exact_hand"] = np.array([max_stable_gain(dev, ren, hand, T) for T in T_values])
    return out


def impedance_magnitude(m: float, b: float, k: float, w) -> np.ndarray:
    """|Z(j w)| of MR Eq. 11.63, Z(s) = m s^2 + b s + k."""
    w = np.asarray(w, float)
    return np.abs(k - m * w ** 2 + 1j * b * w)


def static_push_stiffness(dev: HapticDevice, K: float, B: float, F_load: float = 0.5, T: float = 0.001,
                          t_end: float = 2.0) -> float:
    """Rendered stiffness F_load / x_ss of a spring about 0 under a constant push (no quantisation)."""
    ren = Rendering(shape="spring", K=K, B=B, T=T, quantize=False, saturate=False)
    hand = Hand(kind="hand", m=0.0, b=0.0, k=0.0, f_pulse=F_load, t_on=0.0, t_off=1e9)
    sim = simulate(dev, ren, hand, t_end)
    x_ss = float(np.mean(sim.x[sim.t >= t_end - 0.2]))
    return F_load / x_ss


# ==========================================================================================
# presets
# ==========================================================================================
PRESETS: dict[str, dict] = {
    "Virtual wall — hand pushes in": dict(
        text="A unilateral wall at x = 5 mm. The relaxed hand (50 g, 1 N·s/m, 150 N/m grip) moves its "
             "intended position from 0 to 15 mm. Raise K above 2b/T = 300 N/m or add delay to see the energy leak.",
        ren=dict(mode="impedance", shape="wall", K=250.0, B=0.0, T=0.001, delay_steps=0, x0=0.005),
        hand=dict(kind="hand", m=0.05, b=1.0, k=150.0, x_start=0.0, x_end=0.015, t_move0=0.2, t_move=1.0),
        t_end=2.0, x0=0.0),
    "Release test (hand off)": dict(
        text="Handle released 5 mm from the set point of a virtual spring at x = 0, no hand (SBEG 461 Week 12 Sec. 6.2): with "
             "B = 1 N·s/m and 5 ms delay, K = 100 N/m decays but 300 N/m grows.",
        ren=dict(mode="impedance", shape="spring", K=300.0, B=1.0, T=0.001, delay_steps=5, x0=0.0),
        hand=dict(kind="none"),
        t_end=1.5, x0=0.005),
    "Rehab virtual tunnel (S1)": dict(
        text="Cross-section of the S1 therapy tunnel rendered on the haptic device (HW1-2: K_n = 300 N/m, "
             "B_n = 1 N·s/m about the centre x_c = 0). The patient's relaxed hand receives a 1 N lateral "
             "spasm for 0.3 s; the tunnel pulls it back.",
        ren=dict(mode="impedance", shape="spring", K=300.0, B=1.0, T=0.001, delay_steps=0, x0=0.0),
        hand=dict(kind="hand", m=0.10, b=0.5, k=30.0, x_start=0.0, x_end=0.0, t_move=0.0,
                  f_pulse=1.0, t_on=0.3, t_off=0.6),
        t_end=1.5, x0=0.0),
    "Casualty-handling soft contact (S6)": dict(
        text="S6 hardware task: the cradle impedance K = 100 N/m, B = 2 N·s/m tracks a set point moving at "
             "20 mm/s for 1 s towards a foam pad 15 mm away (2 kN/m, 5 N·s/m). Pass: peak ≤ 1.5 N, final ≤ 0.8 N.",
        ren=dict(mode="impedance", shape="spring", K=100.0, B=2.0, T=0.001, delay_steps=0, x0=0.0,
                 v_set=0.020, t_set0=0.2, t_set=1.0, M_v=0.3, Kp=250.0),
        hand=dict(kind="surface", k=2000.0, b=5.0, x_start=0.015),
        t_end=2.0, x0=0.0),
}


def preset(name: str) -> tuple[Rendering, Hand, float, float]:
    p = PRESETS[name]
    return Rendering(**p["ren"]), Hand(**p["hand"]), float(p["t_end"]), float(p["x0"])


# ==========================================================================================
# validation
# ==========================================================================================
def validation_checks() -> list[Check]:
    checks: list[Check] = []
    dev = default_device()
    ch = "MR 11.7"

    # 1. handle mass from mr.MassMatrix vs hand calculation (SBEG 461 W12 WE7: 0.0570 kg)
    I_hand = dev.I_c[2] + dev.m_sector * dev.r_c ** 2
    m_hand = (I_hand + dev.rotor_inertia * dev.ratio ** 2) / dev.rh ** 2
    checks.append(Check("Haptic paddle: handle mass from mr.MassMatrix vs (I_c + m r_c² + J_r N²)/r_h²",
                        "MR 8.1, 11.8", abs(dev.m - m_hand), 1e-12,
                        f"m_h = {dev.m:.6f} kg (course WE7: 0.0570 kg); rotor share "
                        f"{100 * dev.rotor_inertia * dev.ratio**2 / dev.rh**2 / dev.m:.0f} %", CATEGORY))
    # 2. gravity stiffness from mr.GravityForces vs m g r_c / r_h^2 (1.21 N/m)
    kg_hand = dev.m_sector * G0 * dev.r_c / dev.rh ** 2
    checks.append(Check("Haptic paddle: gravity stiffness from mr.GravityForces vs m g r_c / r_h²",
                        "MR 8.1", abs(dev.k_g - kg_hand), 1e-6,
                        f"k_g = {dev.k_g:.5f} N/m (central difference of g(θ) at 0)", CATEGORY))
    # 3. force -> current chain vs WE7 printed value (1.5 N -> 0.777 A)
    checks.append(Check("Haptic paddle: 1.5 N at the handle -> motor current (WE7: 0.777 A)", "MR 11.8",
                        abs(dev.force_to_current(1.5) - 0.777), 1e-3,
                        f"I = F r_h / (r_s/r_p) / k_t = {dev.force_to_current(1.5):.4f} A", CATEGORY))
    # 4. exact sampled-data stability limit vs the course check values (WE8 table)
    m, b, kg = 0.0570, 0.15, 1.21
    for B, dsteps, ref in ((0.0, 0, 300.1), (1.0, 5, 200.8)):
        Kx = max_stable_K(m, b, kg, B, 1e-3, dsteps)
        checks.append(Check(f"Sampled wall: exact K_max (B = {B:g}, delay {dsteps} ms) vs course WE8 table",
                            ch, abs(Kx - ref), 0.6,
                            f"eigenvalues of the ZOH closed-loop matrix, bisection: {Kx:.2f} N/m (table {ref})",
                            CATEGORY))
    # 5. Colgate bound coincides with the exact limit for B = 0, no delay (device alone)
    Kx = max_stable_K(dev.m, dev.b, dev.k_g, 0.0, dev.T)
    Kc = colgate_K_max(dev.b, 0.0, dev.T)
    checks.append(Check("Sampled wall: exact stability limit vs Colgate–Schenkel bound 2b/T (B = 0)", ch,
                        abs(Kx - Kc) / Kc, 0.01,
                        f"exact {Kx:.2f} N/m, passivity bound {Kc:.2f} N/m (Colgate & Schenkel 1997)", CATEGORY))
    # 6. closed-loop matrix vs the nonlinear-capable simulator (linear case: spring, no quantisation)
    ren = Rendering(shape="spring", K=200.0, B=0.5, T=1e-3, delay_steps=2, quantize=False, saturate=False)
    sim = simulate(dev, ren, Hand(kind="none"), 0.3, x0=0.004)
    A = closed_loop_matrix(dev.m, dev.b, dev.k_g, ren.K, ren.B, ren.T, ren.delay_steps)
    z = np.zeros(3 + ren.delay_steps)
    z[0], z[2] = 0.004, 0.004
    xs = []
    for _ in range(len(sim.t)):
        xs.append(z[0])
        z = A @ z
    err = float(np.max(np.abs(np.array(xs) - sim.x)))
    checks.append(Check("Sampled wall: RK4 simulation vs exact discrete-time matrix (2-sample delay)", ch,
                        err, 1e-9, "max |x_sim - x_matrix| over 0.3 s (m)", CATEGORY))
    # 6b. admittance controller: exact matrix vs simulator (hand in contact, 2-sample delay)
    ren = Rendering(mode="admittance", shape="spring", K=120.0, B=0.8, T=1e-3, delay_steps=2, M_v=0.25,
                    Kp=300.0, Kd=0.3, quantize=False, saturate=False)
    hand_c = Hand(kind="hand", m=0.05, b=1.0, k=150.0, x_end=0.0, t_move=0.0)
    sim = simulate(dev, ren, hand_c, 0.3, x0=0.004)
    A = admittance_loop_matrix(dev, ren, hand_c)
    z = np.zeros(6 + ren.delay_steps)
    z[0] = z[2] = z[3] = 0.004
    xs = []
    for _ in range(len(sim.t)):
        xs.append(z[0])
        z = A @ z
    err = float(np.max(np.abs(np.array(xs) - sim.x)))
    checks.append(Check("Admittance renderer: RK4 simulation vs exact discrete-time matrix (hand, 2-sample delay)",
                        "MR 11.7.2", err, 1e-9, "max |x_sim - x_matrix| over 0.3 s (m)", CATEGORY))
    # 7. static push: rendered stiffness = K + k_g (MR Eq. 11.63 at s = 0)
    Kr = static_push_stiffness(dev, 300.0, 1.0)
    checks.append(Check("Rendered impedance at DC: F/x_ss = K + k_g (Z(0) of MR Eq. 11.63)", "MR 11.7",
                        abs(Kr - (300.0 + dev.k_g)) / 300.0, 1e-6, f"F/x_ss = {Kr:.4f} N/m", CATEGORY))
    # 8. energy: passive setting -> no net energy generation; active setting -> leak and growth
    ren_p = Rendering(shape="spring", K=150.0, B=0.0, T=1e-3, quantize=False, saturate=False)
    sp = simulate(dev, ren_p, Hand(kind="none"), 1.0, x0=0.005)
    checks.append(Check("Passive setting (b > KT/2 + |B|): leak − damping dissipation ≤ 0", ch,
                        max(0.0, float(np.max(sp.net))), 1e-9,
                        f"K = 150 N/m, T = 1 ms; max(net) = {np.max(sp.net):.2e} J", CATEGORY))
    ren_a = Rendering(shape="spring", K=450.0, B=0.0, T=1e-3, quantize=False, saturate=False)
    sa = simulate(dev, ren_a, Hand(kind="none"), 1.0, x0=0.002)
    g = oscillation_growth(sa)
    checks.append(Check("Non-passive setting (K = 450 N/m > 2b/T): energy leak and growing oscillation", ch,
                        0.0 if (np.nanmax(sa.net) > 0 and g > 1) else 1.0, 0.5,
                        f"growth ratio {g:.2f}, max(net) = {np.nanmax(sa.net):.2e} J", CATEGORY))
    # 9. admittance (MR Eq. 11.66): constant push -> x_v settles at f / K_virtual
    ren_ad = Rendering(mode="admittance", shape="spring", K=50.0, B=2.0, T=1e-3, M_v=0.2, Kp=200.0, Kd=0.0,
                       quantize=False, saturate=False)
    hand = Hand(kind="hand", m=0.0, b=0.0, k=0.0, f_pulse=0.5, t_on=0.0, t_off=1e9)
    sad = simulate(dev, ren_ad, hand, 3.0)
    xv_ss = float(np.mean(sad.x_v[sad.t > 2.6]))
    checks.append(Check("Admittance rendering: virtual mass settles at f/K (MR Eq. 11.66)", "MR 11.7.2",
                        abs(xv_ss - 0.5 / 50.0) / 0.01, 0.01,
                        f"x_v = {xv_ss * 1e3:.3f} mm (ideal 10 mm); handle x = {np.mean(sad.x[sad.t > 2.6]) * 1e3:.3f} mm",
                        CATEGORY))
    return checks
