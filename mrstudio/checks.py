"""Common record type for the cross-validation suite (shown in the Validation tab and run by pytest).

Every core module may expose ``validation_checks() -> list[Check]``; the Validation tab collects them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    name: str            # what is being verified, e.g. "RNEA: Modern Robotics vs Pinocchio (UR5)"
    chapter: str         # textbook section, e.g. "MR 8.3"
    error: float         # measured discrepancy (max abs error, residual, drift ...)
    tol: float           # pass threshold
    detail: str = ""     # how it was computed / what was compared against what
    category: str = ""   # "Kinematics", "Dynamics", "Closed chains", "Trajectory & control", ...

    @property
    def passed(self) -> bool:
        return bool(self.error <= self.tol)
