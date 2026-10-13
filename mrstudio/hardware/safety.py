"""The course-arm hardware safety checklist (both courses), enforced in software.

Single source: ``_build/shared/course_robots.md`` "Hardware safety checklist for the course arm"
(= ``course_robots.py`` ``COURSE_ARM_SAFETY``), quoted by the SBEG 361 brief Sec. 4.1, the SBEG 461 brief
Sec. 5.2 and the Week 8 lab handouts:

1. software joint limits >= 10 deg inside the hardware limits            -> ``soft_limits`` + ``clamp``
2. taped keep-out zone, radius = reach + 0.30 m                           -> ``keepout_radius`` (reminder)
3. joint speed <= 25 % of rated on the first run of a new motion and in
   SBEG 361 session H1, <= 50 % otherwise                                 -> ``speed_fraction`` + preflight
4. payload <= 50 % of rated = 0.125 kg                                    -> preflight
5. induction completed by everyone who touches the arm                    -> confirmation checkbox
6. nobody inside the keep-out zone while the arm can move; a hand enters
   only in latched stop with torque disabled                              -> checkbox + driver latch
7. e-stop tested before every session; supervisor present                 -> checkboxes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import COURSE_ARM, DEG, ArmSpec

#: = course_robots.py COURSE_ARM_SAFETY
COURSE_ARM_SAFETY = dict(
    soft_limit_margin_deg=10.0,
    keepout_extra_m=0.30,
    speed_fraction_first_run=0.25,
    speed_fraction=0.50,
    payload_max_kg=0.125,
)

SESSIONS = ("H1", "H2", "lab")
SESSION_LABELS = {"H1": "H1 — 361 first hardware session (≤ 25 % speed)",
                  "H2": "H2 — later project session (≤ 50 %)",
                  "lab": "Lab (hands-on lab, ≤ 50 %)"}

#: Items the people at the bench must confirm (rules 2, 5, 6, 7).  Rules 1, 3, 4 are checked by software.
CHECKLIST: list[tuple[str, str]] = [
    ("induction", "Everyone who will touch the arm has completed the hardware safety induction (rule 5)"),
    ("keepout", "Keep-out zone taped around the base, radius {keepout:.2f} m = reach + 0.30 m (rule 2)"),
    ("clear", "Nobody is inside the keep-out zone while the arm can move (rule 6)"),
    ("estop", "Emergency stop tested this session (rule 7)"),
    ("supervisor", "Supervisor present for the whole run (rule 7)"),
    ("payload", "Payload on the tool weighed and entered below (rule 4)"),
]


def allowed_speed_fraction(session: str = "H2", first_run: bool = False) -> float:
    """Rule 3: 25 % of rated in session H1 and on the first run of any new motion, else 50 %."""
    s = str(session).upper()
    if s not in ("H1", "H2", "LAB"):
        raise ValueError(f"unknown session {session!r}; use H1, H2 or lab")
    if s == "H1" or first_run:
        return COURSE_ARM_SAFETY["speed_fraction_first_run"]
    return COURSE_ARM_SAFETY["speed_fraction"]


def soft_limits(hw_limits, margin_deg: float = COURSE_ARM_SAFETY["soft_limit_margin_deg"]) -> np.ndarray:
    """Rule 1: software limits ``margin_deg`` inside the hardware limits (rad, n x 2)."""
    if margin_deg < COURSE_ARM_SAFETY["soft_limit_margin_deg"]:
        raise ValueError(f"soft-limit margin {margin_deg}° < the required "
                         f"{COURSE_ARM_SAFETY['soft_limit_margin_deg']}°")
    hw = np.asarray(hw_limits, float)
    return hw + np.array([margin_deg, -margin_deg]) * DEG


def keepout_radius(reach: float) -> float:
    """Rule 2: radius of the taped keep-out zone [m]."""
    return float(reach + COURSE_ARM_SAFETY["keepout_extra_m"])


def peak_speeds(t, q) -> np.ndarray:
    """Peak commanded |dq/dt| per joint of a sampled trajectory.

    The run loop interpolates the samples *linearly*, so the speed actually commanded on
    [t_k, t_k+1] is the segment slope |q_k+1 - q_k| / (t_k+1 - t_k).  That is what rule 3 must bound.
    (The central differences of ``np.gradient`` -- used by hwcompare HW-5 on the *measured* log --
    average two neighbouring segments and can under-estimate a step in the command by up to 2x.)"""
    t = np.asarray(t, float)
    q = np.asarray(q, float).reshape(len(t), -1)
    if len(t) < 2:
        return np.zeros(q.shape[1])
    dt = np.diff(t)[:, None]
    dq = np.abs(np.diff(q, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dq == 0, 0.0, dq / np.where(dt > 0, dt, 0.0))   # a jump at a repeated time = inf
    return v.max(axis=0)


def retime_to_speed(t, q, vmax):
    """Stretch time uniformly so the peak speed of every joint is <= vmax (returns new t, factor)."""
    t = np.asarray(t, float)
    k = float(np.max(peak_speeds(t, q) / np.asarray(vmax, float))) if len(t) > 1 else 0.0
    if not np.isfinite(k):
        raise ValueError("trajectory jumps at a repeated time stamp; it cannot be slowed down")
    if k <= 1.0:
        return t.copy(), 1.0
    return t[0] + (t - t[0]) * k, k


@dataclass
class SafetySupervisor:
    """Holds one session's safety settings and checks trajectories and the checklist against them."""
    spec: ArmSpec = field(default_factory=lambda: COURSE_ARM)
    session: str = "H2"
    first_run: bool = True
    payload_kg: float = 0.0
    margin_deg: float = COURSE_ARM_SAFETY["soft_limit_margin_deg"]

    @property
    def soft_limits(self) -> np.ndarray:
        return soft_limits(self.spec.hw_limits, self.margin_deg)

    @property
    def speed_fraction(self) -> float:
        return allowed_speed_fraction(self.session, self.first_run)

    @property
    def speed_cap(self) -> np.ndarray:
        """Allowed joint speeds [rad/s]."""
        return self.speed_fraction * self.spec.rated_speed

    @property
    def keepout_radius(self) -> float:
        return keepout_radius(self.spec.reach)

    def checklist_items(self) -> list[tuple[str, str]]:
        return [(k, txt.format(keepout=self.keepout_radius)) for k, txt in CHECKLIST]

    def missing_items(self, ticked: dict) -> list[str]:
        return [txt for k, txt in self.checklist_items() if not ticked.get(k, False)]

    def static_problems(self) -> list[str]:
        out = []
        if self.payload_kg > COURSE_ARM_SAFETY["payload_max_kg"] + 1e-12:
            out.append(f"payload {self.payload_kg:.3f} kg > {COURSE_ARM_SAFETY['payload_max_kg']:.3f} kg "
                       "(50 % of rated, rule 4)")
        if self.payload_kg < 0:
            out.append("payload must be ≥ 0")
        return out

    def within_soft_limits(self, q, tol: float = 1e-9) -> np.ndarray:
        lo, hi = self.soft_limits.T
        q = np.atleast_2d(q)
        return np.all((q >= lo - tol) & (q <= hi + tol), axis=1)

    def clamp(self, q) -> np.ndarray:
        lo, hi = self.soft_limits.T
        return np.clip(np.asarray(q, float), lo, hi)

    def trajectory_problems(self, t, q) -> list[str]:
        """Rules 1 and 3 for a commanded trajectory q(t) [rad]."""
        out = []
        q = np.asarray(q, float).reshape(len(t), -1)
        if q.shape[1] != self.spec.n:
            return [f"trajectory has {q.shape[1]} joints, the arm has {self.spec.n}"]
        lo, hi = self.soft_limits.T
        bad = np.nonzero(np.any(q < lo - 1e-9, axis=0) | np.any(q > hi + 1e-9, axis=0))[0]
        for j in bad:
            out.append(f"J{j + 1} leaves the software limits [{np.degrees(lo[j]):.0f}°, {np.degrees(hi[j]):.0f}°] "
                       f"(range {np.degrees(q[:, j].min()):.1f}° … {np.degrees(q[:, j].max()):.1f}°, rule 1)")
        ratio = peak_speeds(t, q) / self.speed_cap
        for j in np.nonzero(ratio > 1 + 1e-6)[0]:
            out.append(f"J{j + 1} peak speed {np.degrees(ratio[j] * self.speed_cap[j]):.1f}°/s > "
                       f"{np.degrees(self.speed_cap[j]):.0f}°/s ({100 * self.speed_fraction:.0f} % of rated, rule 3)")
        return out

    def preflight(self, ticked: dict, t=None, q=None) -> tuple[bool, list[str]]:
        """Everything that must hold before Run: checklist ticked, payload, trajectory limits/speeds."""
        probs = ["not confirmed: " + m for m in self.missing_items(ticked)]
        probs += self.static_problems()
        if t is not None and q is not None:
            probs += self.trajectory_problems(t, q)
        return (not probs), probs
