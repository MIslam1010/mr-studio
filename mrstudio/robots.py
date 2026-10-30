"""Robot models described the *Modern Robotics* way (Lynch & Park, 2017).

Every serial robot is described exclusively by
  * M      -- the home configuration of the end-effector frame {b} in {s}   (Sec. 4.1)
  * Slist  -- the joint screw axes S_i = (w_i, v_i) expressed in {s}        (Sec. 4.1.1)
  * Mlist  -- relative home configurations of the link centre-of-mass frames (Sec. 8.3.2)
  * Glist  -- 6x6 spatial inertia matrices G_i expressed in the CoM frames   (Sec. 8.2.2)

No Denavit-Hartenberg parameters are used internally.  Robots that only exist as
DH tables (Peter Corke's Robotics Toolbox models) are converted automatically to
the PoE description by :func:`from_rtb_dh`; the validation suite proves that the
conversion reproduces Corke's DH forward kinematics and dynamics exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import modern_robotics as mr
import numpy as np


# ----------------------------------------------------------------------------- helpers
def screw_revolute(w, q, h: float = 0.0) -> np.ndarray:
    """Screw axis of a joint rotating about unit axis w through point q with pitch h (MR Sec. 3.3.2)."""
    w = np.asarray(w, float)
    q = np.asarray(q, float)
    return np.r_[w, -np.cross(w, q) + h * w]


def screw_prismatic(v) -> np.ndarray:
    """Screw axis of a prismatic joint translating along unit direction v: S = (0, v)  (MR Sec. 3.3.2.2)."""
    v = np.asarray(v, float)
    return np.r_[np.zeros(3), v / np.linalg.norm(v)]


def trans(p, R=None) -> np.ndarray:
    return mr.RpToTrans(np.eye(3) if R is None else np.asarray(R, float), np.asarray(p, float))


def rod_inertia(m: float, L: float, r: float = 0.025) -> np.ndarray:
    """Spatial inertia G (MR Eq. 8.32) of a uniform solid cylinder of length L lying along its local x axis."""
    Ixx = 0.5 * m * r**2
    Iyy = Izz = m * (3 * r**2 + L**2) / 12.0
    return np.diag([Ixx, Iyy, Izz, m, m, m])


def screw_axis_line(S: np.ndarray):
    """Return (point, direction, is_revolute) of the line of a screw axis."""
    w, v = S[:3], S[3:]
    if np.linalg.norm(w) > 1e-9:
        w = w / np.linalg.norm(w)
        return np.cross(w, v), w, True  # closest point of the axis to the origin
    return np.zeros(3), v / np.linalg.norm(v), False


def _common_normal(p1, d1, p2, d2, ref):
    """Feet (a on line 1, b on line 2) of the common normal of two lines.
    For parallel lines the normal through the projection of `ref` on line 1 is used."""
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-10:  # parallel
        foot1 = p1 + d1 * ((ref - p1) @ d1)
        foot2 = p2 + d2 * ((foot1 - p2) @ d2)
        return foot1, foot2
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return p1 + s * d1, p2 + t * d2


# ----------------------------------------------------------------------------- model
PRISMATIC_DEFAULT_LIMITS = (-0.5, 0.5)   # metres, used when a prismatic joint has no limits


@dataclass
class SerialRobot:
    name: str
    M: np.ndarray
    Slist: np.ndarray                       # 6 x n
    Mlist: list | None = None               # n+1 transforms M_{i-1,i}
    Glist: list | None = None               # n spatial inertias
    joint_limits: np.ndarray | None = None  # n x 2
    planar: bool = False
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    home: np.ndarray | None = None
    tau_max: np.ndarray | None = None
    description: str = ""
    source: str = ""
    link_radius: float = 0.03

    def __post_init__(self):
        self.M = np.asarray(self.M, float)
        self.Slist = np.asarray(self.Slist, float).reshape(6, -1)
        if self.joint_limits is None:
            self.joint_limits = np.array([PRISMATIC_DEFAULT_LIMITS if p else (-np.pi, np.pi)
                                          for p in self.is_prismatic], float).reshape(self.n, 2)
        self.joint_limits = np.asarray(self.joint_limits, float)
        if self.home is None:
            self.home = np.zeros(self.n)
        if self.tau_max is None:
            self.tau_max = np.full(self.n, 100.0)
        self.gravity = np.asarray(self.gravity, float)

    # -- basic properties ------------------------------------------------------
    @property
    def n(self) -> int:
        return self.Slist.shape[1]

    @property
    def is_prismatic(self) -> np.ndarray:
        """True for prismatic joints, i.e. screw axes S_i = (0, v_i) with zero angular part (MR Sec. 3.3.2)."""
        return np.linalg.norm(self.Slist[:3], axis=0) < 1e-9

    @property
    def joint_types(self) -> str:
        """Joint-type string such as "RRPRRR" (R = revolute, P = prismatic)."""
        return "".join("P" if p else "R" for p in self.is_prismatic)

    def joint_unit(self, i: int) -> str:
        return "m" if self.is_prismatic[i] else "rad"

    def display_unit(self, i: int) -> str:
        """Unit shown on screen: degrees for revolute joints, metres for prismatic joints."""
        return "m" if self.is_prismatic[i] else "°"

    def to_display(self, theta) -> np.ndarray:
        """Joint values (rad / m) -> display units (deg / m); works on (..., n) arrays."""
        th = np.asarray(theta, float)
        return np.where(self.is_prismatic, th, np.degrees(th))

    @property
    def has_dynamics(self) -> bool:
        return self.Mlist is not None and self.Glist is not None

    @cached_property
    def Blist(self) -> np.ndarray:
        """Body-frame screw axes B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16)."""
        return mr.Adjoint(mr.TransInv(self.M)) @ self.Slist

    @cached_property
    def reach(self) -> float:
        pts = np.vstack([p for poly in self.skeleton for p in poly])
        r = float(np.max(np.linalg.norm(pts, axis=1)))
        # prismatic joints can push the chain further than the home skeleton shows
        for i in np.flatnonzero(self.is_prismatic):
            r += float(min(np.max(np.abs(self.joint_limits[i])), 2.0))
        return r + 1e-9

    @cached_property
    def com_home(self) -> list[np.ndarray]:
        """Home configurations M_{0,i} of the link CoM frames (i = 1..n)."""
        out, T = [], np.eye(4)
        for i in range(self.n):
            T = T @ self.Mlist[i]
            out.append(T.copy())
        return out

    @property
    def masses(self) -> np.ndarray:
        return np.array([G[3, 3] for G in self.Glist])

    # -- kinematic chain geometry (pure PoE) -----------------------------------
    def link_transforms(self, theta) -> list[np.ndarray]:
        """T_i(theta) = e^[S1]th1 ... e^[Si]thi for i = 0..n (T_0 = I). A link-i point x_home maps to T_i x_home."""
        out = [np.eye(4)]
        T = np.eye(4)
        for i in range(self.n):
            T = T @ mr.MatrixExp6(mr.VecTose3(self.Slist[:, i] * theta[i]))
            out.append(T)
        return out

    def fk(self, theta) -> np.ndarray:
        return mr.FKinSpace(self.M, self.Slist, np.asarray(theta, float))

    @cached_property
    def skeleton(self) -> list[np.ndarray]:
        """Home-configuration polyline for every link (index 0 = base) built only from
        the screw axes: segments run along joint axes and along common normals of
        consecutive axes -- i.e. the DH skeleton recovered from the PoE description."""
        lines = [screw_axis_line(self.Slist[:, i]) for i in range(self.n)]
        ee = self.M[:3, 3]
        polys = []
        # foot of axis 1 nearest to the origin of {s}
        p, d, _ = lines[0]
        b = p + d * ((np.zeros(3) - p) @ d)
        polys.append(np.array([np.zeros(3), b]))
        for i in range(self.n):
            p, d, rev = lines[i]
            if not rev:  # prismatic: line through current foot
                p = b
            if i < self.n - 1:
                p2, d2, rev2 = lines[i + 1]
                if not rev2:
                    p2 = None
                if p2 is None:
                    a, b_next = b, b
                else:
                    a, b_next = _common_normal(p, d, p2, d2, b)
            else:
                a = p + d * ((ee - p) @ d)
                b_next = ee
            polys.append(np.array([b, a, b_next]))
            b = b_next
        return polys

    @cached_property
    def joint_points_home(self) -> np.ndarray:
        """Point on each joint axis where the previous link attaches (for drawing joints)."""
        return np.array([self.skeleton[i + 1][0] for i in range(self.n)])

    def geometry(self, theta):
        """Everything needed to draw the robot at configuration theta.
        Returns dict(polys, joint_pos, joint_dir, T_links, T_ee)."""
        Ts = self.link_transforms(theta)
        polys = [(Ts[i][:3, :3] @ P.T).T + Ts[i][:3, 3] for i, P in enumerate(self.skeleton)]
        jp, jd = [], []
        prism = self.is_prismatic
        for i in range(self.n):
            # a prismatic joint is drawn as a sleeve fixed on link i-1; link i slides through it,
            # so its polyline starts with the (variable-length) rod sleeve -> sliding part
            T = Ts[i] if prism[i] else Ts[i + 1]
            jp.append(T[:3, :3] @ self.joint_points_home[i] + T[:3, 3])
            if prism[i]:
                polys[i + 1] = np.vstack([jp[-1], polys[i + 1]])
            _, d, _ = screw_axis_line(self.Slist[:, i])
            jd.append(T[:3, :3] @ d)
        return dict(polys=polys, joint_pos=np.array(jp), joint_dir=np.array(jd),
                    T_links=Ts, T_ee=Ts[-1] @ self.M)

    def com_positions(self, theta) -> np.ndarray:
        Ts = self.link_transforms(theta)
        return np.array([(Ts[i + 1] @ self.com_home[i])[:3, 3] for i in range(self.n)])

    def random_config(self, rng=None) -> np.ndarray:
        """Uniform sample inside the joint limits (revolute: clipped to [-pi, pi] rad;
        prismatic: clipped to +-1 m)."""
        rng = np.random.default_rng() if rng is None else rng
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        cap = np.where(self.is_prismatic, 1.0, np.pi)
        return rng.uniform(np.maximum(lo, -cap), np.minimum(hi, cap))

    # -- presentation ----------------------------------------------------------
    def screw_table(self, frame: str = "space") -> list[list[str]]:
        L = self.Slist if frame == "space" else self.Blist
        rows = []
        for i in range(self.n):
            w, v = L[:3, i], L[3:, i]
            rows.append([f"{i + 1}", *(f"{x:+.4f}" for x in w), *(f"{x:+.4f}" for x in v)])
        return rows


# ----------------------------------------------------------------------------- library of robots
def planar_2r(L1=0.6, L2=0.4, m1=2.0, m2=1.0) -> SerialRobot:
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, 1], [L1, 0, 0])]
    M = trans([L1 + L2, 0, 0])
    Mlist = [trans([L1 / 2, 0, 0]), trans([(L1 + L2) / 2, 0, 0]), trans([L2 / 2, 0, 0])]
    Glist = [rod_inertia(m1, L1), rod_inertia(m2, L2)]
    return SerialRobot(
        "Planar 2R arm", M, S, Mlist, Glist, planar=True, gravity=np.array([0, -9.81, 0]),
        home=np.array([0.6, 0.9]), tau_max=np.array([40.0, 15.0]), link_radius=0.03,
        description=f"Two revolute joints in the x-y plane, L1={L1} m, L2={L2} m, uniform rods "
                    f"m1={m1} kg, m2={m2} kg. Gravity acts along -y (vertical plane). "
                    "Dynamics as in MR Sec. 8.1.1 (Fig. 8.1), but with rod links.",
        source="Modern Robotics, Ch. 4 & 8 (uniform-rod links)")


def planar_3r(L=(0.5, 0.4, 0.25), m=(2.0, 1.5, 0.5)) -> SerialRobot:
    L1, L2, L3 = L
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]),
              screw_revolute([0, 0, 1], [L1, 0, 0]),
              screw_revolute([0, 0, 1], [L1 + L2, 0, 0])]
    M = trans([L1 + L2 + L3, 0, 0])
    Mlist = [trans([L1 / 2, 0, 0]), trans([(L1 + L2) / 2, 0, 0]),
             trans([(L2 + L3) / 2, 0, 0]), trans([L3 / 2, 0, 0])]
    Glist = [rod_inertia(m[i], L[i]) for i in range(3)]
    return SerialRobot(
        "Planar 3R arm", M, S, Mlist, Glist, planar=True, gravity=np.array([0, -9.81, 0]),
        home=np.array([0.4, 0.8, 0.6]), tau_max=np.array([50.0, 25.0, 8.0]), link_radius=0.03,
        description=f"Three revolute joints in the x-y plane (MR Example 4.2), lengths {L} m, "
                    f"masses {m} kg. Redundant for position, exactly determined for planar pose.",
        source="Modern Robotics, Example 4.2")


def ur5() -> SerialRobot:
    """Universal Robots UR5 exactly as in Modern Robotics Example 4.5 and the MR Ch. 8 software exercises."""
    W1, W2, L1, L2, H1, H2 = 0.10915, 0.0823, 0.425, 0.39225, 0.089159, 0.09465
    Slist = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 1],
                      [1, 0, 0, 0, -1, 0],
                      [0, -H1, -H1, -H1, -W1, H2 - H1],
                      [0, 0, 0, 0, L1 + L2, 0],
                      [0, 0, L1, L1 + L2, 0, L1 + L2]], float)
    Mlist = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]),
             np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]),
             np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])]
    Glist = [np.diag([0.010267495893, 0.010267495893, 0.00666, 3.7, 3.7, 3.7]),
             np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393]),
             np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219]),
             np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])]
    M = np.eye(4)
    for Mi in Mlist:
        M = M @ Mi
    return SerialRobot(
        "UR5 (6R industrial)", M, Slist, Mlist, Glist,
        joint_limits=np.tile([-2 * np.pi, 2 * np.pi], (6, 1)),
        home=np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]),
        tau_max=np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]), link_radius=0.035,
        description="Universal Robots UR5 collaborative arm: W1=109 mm, W2=82 mm, L1=425 mm, "
                    "L2=392 mm, H1=89 mm, H2=95 mm. Screw axes and inertias are the ones used in "
                    "Modern Robotics Example 4.5 and the Ch. 8 programming assignment.",
        source="Modern Robotics, Example 4.5 + Course 3 UR5 dynamic parameters")


def from_rtb_dh(rtb_robot, name: str, description: str = "", tau_max=None, home=None,
                inertias: str = "rtb", source: str | None = None) -> SerialRobot:
    """Convert a Denavit-Hartenberg model from Peter Corke's Robotics Toolbox into the
    Modern Robotics (PoE + spatial inertia) description.

    * S_i is read from the geometric Jacobian at the zero configuration:
      revolute  J_i = [w_i x (p_ee - q_i); w_i]  =>  v_i = -w_i x q_i = J_v,i - w_i x p_ee;
      prismatic J_i = [v_i; 0]                   =>  S_i = (0, v_i)   (MR Sec. 3.3.2).
    * Each CoM frame {i} is aligned with DH link frame i and placed at the CoM r_i.
    * inertias="auto" ignores the toolbox mass data and uses :func:`auto_inertias` instead
      (for models without, or with implausible, dynamic parameters).
    """
    n = rtb_robot.n
    q0 = np.zeros(n)
    T0 = np.asarray(rtb_robot.fkine(q0).A)
    J0 = np.asarray(rtb_robot.jacob0(q0))
    p = T0[:3, 3]
    S = np.zeros((6, n))
    for i in range(n):
        v, w = J0[:3, i], J0[3:, i]
        S[:, i] = np.r_[w, v - np.cross(w, p)] if np.linalg.norm(w) > 1e-9 else np.r_[0, 0, 0, v]
    Mlist = Glist = None
    qlim = getattr(rtb_robot, "qlim", None)
    limits = np.asarray(qlim, float).T if qlim is not None else None
    if inertias == "auto":
        Mlist, Glist = auto_inertias(S, T0, joint_limits=limits)
    elif all(l.m is not None for l in rtb_robot.links) and any(l.m > 0 for l in rtb_robot.links):
        frames = rtb_robot.fkine_all(q0)
        Mc = [np.eye(4)]
        Glist = []
        for i, link in enumerate(rtb_robot.links):
            Ti = np.asarray(frames[i + 1].A)
            Mc.append(Ti @ trans(np.asarray(link.r, float)))
            G = np.zeros((6, 6))
            G[:3, :3] = np.asarray(link.I, float)
            G[3:, 3:] = link.m * np.eye(3)
            Glist.append(G)
        Mlist = [mr.TransInv(Mc[i]) @ Mc[i + 1] for i in range(n)] + [mr.TransInv(Mc[-1]) @ T0]
    return SerialRobot(name, T0, S, Mlist, Glist, joint_limits=limits,
                       home=home if home is not None else np.zeros(n), tau_max=tau_max,
                       description=description, source=source or "Converted automatically from P. Corke's "
                       "Robotics Toolbox DH model (DH -> PoE)", link_radius=0.04)


def puma560() -> SerialRobot:
    import roboticstoolbox as rtb
    r = rtb.models.DH.Puma560()
    return from_rtb_dh(r, "PUMA 560 (from Corke DH)",
                       "Classic Unimation PUMA 560 (6R, spherical wrist). Kinematic and inertial "
                       "parameters (Armstrong/Corke) taken from Peter Corke's Robotics Toolbox DH "
                       "model and converted to screw axes + spatial inertias.",
                       tau_max=np.array([97.6, 186.4, 89.4, 24.2, 20.1, 21.3]),
                       home=np.array([0.0, 0.6, -2.3, 0.0, -0.6, 0.0]))


def panda() -> SerialRobot:
    import roboticstoolbox as rtb
    r = rtb.models.DH.Panda()
    return from_rtb_dh(r, "Franka Panda (7R redundant)",
                       "Franka Emika Panda 7-DOF collaborative arm (modified DH in Corke's toolbox, "
                       "dynamic parameters from Gaz et al. 2019). One degree of redundancy: the "
                       "elbow can move while the gripper stays still (self-motion).",
                       tau_max=np.array([87.0, 87, 87, 87, 12, 12, 12]),
                       home=np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.78]))


def stanford_arm() -> SerialRobot:
    """Scheinman's Stanford arm (RRPRRR): the classic arm with a prismatic third joint."""
    import roboticstoolbox as rtb
    r = rtb.models.DH.Stanford()
    return from_rtb_dh(r, "Stanford arm (RRPRRR)",
                       "Victor Scheinman's Stanford arm (1969): two revolute shoulder joints, a prismatic "
                       "boom (joint 3, screw axis S_3 = (0, v_3), travel 0.30-1.27 m) and a spherical wrist. "
                       "Kinematics from Corke's DH model converted to screw axes; the toolbox mass data "
                       "(Paul 1981) have CoM offsets of several metres, so physically consistent inertias "
                       "(rods + motors, MR Eq. 8.32) are generated instead.",
                       tau_max=np.array([120.0, 120.0, 600.0, 20.0, 20.0, 20.0]),
                       home=np.array([0.0, 1.2, 0.6, 0.0, 0.6, 0.0]), inertias="auto",
                       source="Corke's Robotics Toolbox DH model rtb.models.DH.Stanford (DH -> PoE); "
                              "inertias from robots.auto_inertias")


def scara() -> SerialRobot:
    """Omron/Adept Cobra 600 SCARA (RRPR) -- kinematics from Corke's DH model."""
    import roboticstoolbox as rtb
    r = rtb.models.DH.Cobra600()
    return from_rtb_dh(r, "SCARA Cobra 600 (RRPR)",
                       "Selective-compliance assembly robot arm (Omron/Adept Cobra 600): two parallel vertical "
                       "revolute axes (325 mm + 275 mm links), a vertical prismatic quill (joint 3, 0-210 mm, "
                       "moving down) and a tool roll. Kinematics from Corke's DH model converted to screw "
                       "axes; the toolbox has no mass data, so inertias are generated (rods + motors).",
                       tau_max=np.array([60.0, 30.0, 300.0, 5.0]),
                       home=np.array([0.5, -1.0, 0.1, 0.0]), inertias="auto",
                       source="Corke's Robotics Toolbox DH model rtb.models.DH.Cobra600 (DH -> PoE); "
                              "inertias from robots.auto_inertias")


def iiwa7() -> SerialRobot:
    """KUKA LBR iiwa 7 R800, imported from its URDF (MR Sec. 4.2 / 8.8) -- see urdf_import.py."""
    from .urdf_import import IIWA7_URDF, from_urdf
    r = from_urdf(IIWA7_URDF, name="KUKA iiwa 7 (7R, URDF)", tip="iiwa_link_ee")
    r.home = np.array([0.0, 0.5, 0.0, -1.4, 0.0, 1.2, 0.0])
    r.tau_max = np.array([176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0])   # KUKA datasheet (URDF: 300 N m)
    r.description = ("KUKA LBR iiwa 7 R800 lightweight 7-DOF arm (one degree of redundancy), imported "
                     "from its URDF: screw axes, M, CoM frames, spatial inertias, joint and effort limits "
                     "are all computed from the URDF joint and inertial elements (MR Sec. 4.2, 8.8). "
                     "Torque limits: KUKA datasheet (176/176/110/110/110/40/40 N m).")
    r.source = ("URDF iiwa7.urdf (KUKA iiwa stack, as shipped in facebookresearch/differentiable-robot-model,"
                " MIT) converted by mrstudio.urdf_import.from_urdf")
    return r


ROBOTS = {
    "Planar 2R arm": planar_2r,
    "Planar 3R arm": planar_3r,
    "UR5 (6R industrial)": ur5,
    "PUMA 560 (from Corke DH)": puma560,
    "Franka Panda (7R redundant)": panda,
    "KUKA iiwa 7 (7R, URDF)": iiwa7,
    "Stanford arm (RRPRRR)": stanford_arm,
    "SCARA Cobra 600 (RRPR)": scara,
}

# SBEG 361 / 461 course robots (numbers vendored from the course materials' shared course_robots.py)
from .course_models import COURSE_ROBOTS  # noqa: E402  (needs SerialRobot, screw_revolute, trans above)

ROBOTS.update(COURSE_ROBOTS)

_cache: dict[str, SerialRobot] = {}


def get_robot(name: str) -> SerialRobot:
    if name not in _cache:
        _cache[name] = ROBOTS[name]()
    return _cache[name]


# ----------------------------------------------------------------------------- user-defined robots
def auto_inertias(Slist: np.ndarray, M: np.ndarray, density: float = 4.0, motor_mass: float = 0.6,
                  motor_radius: float = 0.04, joint_limits=None):
    """Physically consistent Mlist/Glist for a robot given only by its screw axes:
    every link = thin rods along its PoE skeleton segments (density kg/m) + a motor (solid sphere)
    at its joint.  The sliding member of a prismatic joint additionally gets a rod (boom) of the
    joint's travel length lying along the slide direction, behind its tip.
    CoM frames are aligned with {s} at home.  Returns (Mlist, Glist)."""
    tmp = SerialRobot("tmp", M, Slist, joint_limits=joint_limits)
    n = tmp.n
    Mc, Glist = [np.eye(4)], []
    for i in range(n):
        P = tmp.skeleton[i + 1]
        parts = [(motor_mass, P[0], 0.4 * motor_mass * motor_radius**2 * np.eye(3))]
        if tmp.is_prismatic[i]:
            d = Slist[3:, i] / np.linalg.norm(Slist[3:, i])
            L = float(max(np.ptp(tmp.joint_limits[i]), 0.1))
            m = density * L
            parts.append((m, P[0] - d * L / 2, m * L**2 / 12 * (np.eye(3) - np.outer(d, d))))
        for a, b in zip(P[:-1], P[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            m = density * L
            d = (b - a) / L
            parts.append((m, (a + b) / 2, m * L**2 / 12 * (np.eye(3) - np.outer(d, d))))
        mt = sum(m for m, _, _ in parts)
        c = sum(m * p for m, p, _ in parts) / mt
        Ic = np.zeros((3, 3))
        for m, p, I in parts:
            r = p - c
            Ic += I + m * (r @ r * np.eye(3) - np.outer(r, r))
        G = np.zeros((6, 6))
        G[:3, :3] = Ic + 1e-6 * np.eye(3)
        G[3:, 3:] = mt * np.eye(3)
        Glist.append(G)
        Mc.append(trans(c))
    Mlist = [mr.TransInv(Mc[i]) @ Mc[i + 1] for i in range(n)] + [mr.TransInv(Mc[-1]) @ M]
    return Mlist, Glist


def custom_robot(name: str, omegas, points, M, density: float = 4.0, motor_mass: float = 0.6,
                 description: str = "", joint_types: str | None = None, joint_limits=None) -> SerialRobot:
    """Build a robot from MR data at the home configuration, in {s}:
    revolute joint ('R'):  unit axis w_i and a point q_i on it -> S_i = (w_i, -w_i x q_i);
    prismatic joint ('P'): unit direction v_i (given in `omegas`; q_i unused) -> S_i = (0, v_i)  (MR Sec. 3.3.2);
    plus the end-effector home pose M.  joint_types e.g. "RRPR" (default: all revolute)."""
    n = len(omegas)
    joint_types = (joint_types or "R" * n).upper()
    if len(joint_types) != n or set(joint_types) - {"R", "P"}:
        raise ValueError(f"joint_types must be {n} letters R/P, got {joint_types!r}")
    omegas = [np.asarray(w, float) / np.linalg.norm(w) for w in omegas]
    S = np.column_stack([screw_prismatic(w) if t == "P" else screw_revolute(w, q)
                         for w, q, t in zip(omegas, points, joint_types)])
    planar = all((abs(w[2]) < 1e-9) if t == "P" else (abs(abs(w[2]) - 1) < 1e-9 and abs(q[2]) < 1e-9)
                 for w, q, t in zip(omegas, points, joint_types)) and abs(M[2, 3]) < 1e-9 \
        and "R" in joint_types
    tmp = SerialRobot(name, M, S, joint_limits=joint_limits)
    Mlist, Glist = auto_inertias(S, np.asarray(M, float), density, motor_mass, joint_limits=tmp.joint_limits)
    home = np.where(tmp.is_prismatic, np.clip(0.4 * tmp.joint_limits[:, 1], *tmp.joint_limits.T), 0.0)
    r = SerialRobot(name, M, S, Mlist, Glist, planar=planar, joint_limits=tmp.joint_limits, home=home,
                    gravity=np.array([0, -9.81, 0]) if planar else np.array([0, 0, -9.81]),
                    tau_max=np.where(tmp.is_prismatic, 400.0, 80.0),
                    link_radius=max(0.015, 0.035 * tmp.reach),
                    description=description or "User-defined robot (screw axes entered in the Robot Builder; "
                    "inertias generated automatically from rods + motors).",
                    source="Robot Builder (Modern Robotics description: w_i, q_i, M)")
    return r


def register_robot(robot: SerialRobot):
    """Add a robot to the global library (shows up in the robot selector)."""
    ROBOTS[robot.name] = lambda r=robot: r
    _cache[robot.name] = robot
