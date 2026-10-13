"""The SBEG 361 / SBEG 461 course robots as MR Studio :class:`~mrstudio.robots.SerialRobot` models.

Source of truth: ``Robotics Track Course Materials/_build/shared/course_robots.py`` (and
``course_robots.md``).  MR Studio must stay self-contained, so the *numbers* of that file are vendored
here (same geometry, masses, inertias, limits, rated torques, home poses, gravity) and rebuilt with the
same Modern Robotics formulas:

  * screw axes  S_i = (w_i, -w_i x q_i)                         (MR Sec. 3.3.2, Eq. 4.14)
  * home pose   M of {b} in {s}                                  (MR Sec. 4.1)
  * CoM frames  Mlist = [M_01, M_12, ..., M_n,n+1]               (MR Sec. 8.3.2)
  * inertias    G_i = diag(I_b, m I3) in the CoM frame           (MR Eq. 8.32)

``tests/test_course_robots_app.py`` compares every model with the shared file whenever that file exists.

Robots (names as they appear in the robot selector):

======================================  ===============  =========================================
MR Studio name                           shared key       course use
======================================  ===============  =========================================
Course arm (myCobot-280 class 6R)        course_arm       every hardware lab, 361/461 projects
EOD arm (6R on UGV)                      eod_6r_tracked   461 project S5 (6R EOD arm with dynamics)
EOD arm (3R planar)                      eod_3r           461 Weeks 1-3, w13 practical B
Arm–exoskeleton 2R                       arm_exo_2r       461 Weeks 1-2
Rehab robot 2R (planar)                  rehab_2r         461 project S1
Hapkit (1-DOF haptic paddle)             hapkit           461 haptics weeks (Tier 1 haptic device)
======================================  ===============  =========================================

All inertial data, rated torques and speeds are ILLUSTRATIVE teaching values (as in the shared file).
The 361 project S4 ``eod_6r_ugv`` is kinematic only (no Mlist/Glist); MR Studio uses the 461 S5
model ``eod_6r_tracked``, which has dynamics, for "EOD arm (6R on UGV)".
"""
from __future__ import annotations

import time

import modern_robotics as mr
import numpy as np

from .checks import Check

G0 = 9.81
DEG = np.pi / 180.0
CAT = "Course robots"

# shared-file key of every MR Studio course robot (used by the comparison test)
SHARED_KEYS = {
    "Course arm (myCobot-280 class 6R)": "course_arm",
    "EOD arm (6R on UGV)": "eod_6r_tracked",
    "EOD arm (3R planar)": "eod_3r",
    "Arm–exoskeleton 2R": "arm_exo_2r",
    "Rehab robot 2R (planar)": "rehab_2r",
    "Hapkit (1-DOF haptic paddle)": "hapkit",
}


# ----------------------------------------------------------------------------- inertia helpers (= shared file)
def body_inertia(m: float, Ib) -> np.ndarray:
    """G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32). Ib: principal moments (3-vector) or 3x3."""
    Ib = np.diag(np.asarray(Ib, float)) if np.ndim(Ib) == 1 else np.asarray(Ib, float)
    G = np.zeros((6, 6))
    G[:3, :3] = Ib
    G[3:, 3:] = m * np.eye(3)
    return G


def rod_inertia_x(m: float, L: float, r: float = 0.03) -> np.ndarray:
    """Uniform solid cylinder (radius r) of length L along the CoM-frame x axis."""
    return body_inertia(m, [0.5 * m * r**2, m * (3 * r**2 + L**2) / 12.0, m * (3 * r**2 + L**2) / 12.0])


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Solid box with edges a, b, c along the CoM-frame x, y, z axes."""
    return body_inertia(m, [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12])


def mlist_from_absolute(com_frames, M_ee) -> list:
    """MR's Mlist from the absolute home CoM frames M_{s,i} and the end-effector home frame M."""
    frames = [np.eye(4)] + list(com_frames) + [np.asarray(M_ee, float)]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# ----------------------------------------------------------------------------- vendored numbers
# Course arm: desktop UR-type 6R (myCobot-280 class), MR Example 4.5 layout (screw table p. 145)
COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)
COURSE_ARM_LIMITS_DEG = [(-165, 165)] * 5 + [(-175, 175)]
COURSE_ARM_TAU_RATED = [1.8, 3.0, 1.5, 0.6, 0.6, 0.4]            # N m, illustrative smart-servo ratings
COURSE_ARM_HOME = [0.0, -0.6, 1.4, -2.4, -1.57, 0.0]
COURSE_ARM_RATED_SPEED_DEG_S = 160.0
COURSE_ARM_PAYLOAD_MAX = 0.25                                     # kg
# link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
COURSE_ARM_LINKS = [
    (0.20, (0.000, 0.010, 0.105), (0.050, 0.060, 0.060)),   # 1 shoulder housing (turns with J1)
    (0.16, (0.055, 0.030, 0.1312), (0.130, 0.040, 0.040)),  # 2 upper arm
    (0.12, (0.158, 0.005, 0.1312), (0.120, 0.035, 0.035)),  # 3 forearm
    (0.08, (0.206, 0.040, 0.1312), (0.035, 0.050, 0.035)),  # 4 wrist 1 (J5 servo)
    (0.08, (0.206, 0.0634, 0.090), (0.035, 0.035, 0.050)),  # 5 wrist 2 (J6 servo)
    (0.03, (0.206, 0.100, 0.0561), (0.040, 0.015, 0.040)),  # 6 flange
]

# EOD 3R (vertical plane): planar chain L, m, CoM distance r, Izz (461 w01/w02 data), rated torques (w13)
EOD_3R = dict(L=(0.60, 0.50, 0.15), m=(8.0, 5.0, 3.5), r=(0.30, 0.25, 0.10), Izz=(0.24, 0.10, 0.008),
              Ixx_factor=0.2, limits=((-1.0, 2.0), (-2.6, 2.6), (-2.0, 2.0)), tau_max=(155.0, 70.0, 15.0))
EOD_3R_GRAVITY_AT_ZERO = (106.44, 32.86, 3.43)                    # N m, quoted in course_robots.md

# Arm + exoskeleton 2R (sagittal plane), 461 Weeks 1-2
ARM_EXO_2R = dict(L=(0.30, 0.35), m=(2.6, 2.0), r=(0.13, 0.17), Izz=(0.025, 0.020), Ixx_factor=0.2,
                  limits=((-0.5, 2.6), (0.0, 2.4)), tau_max=(30.0, 15.0))

# Rehabilitation robot 2R (horizontal plane), 461 project S1
REHAB_2R = dict(L=(0.35, 0.35), m=(2.4, 1.6), limits=((0.0, np.pi), (0.2, 2.8)), tau_max=(20.0, 12.0),
                qdot_max=(3.0, 3.0), home=(0.9, 1.6))

# Hapkit-type 1-DOF haptic paddle (illustrative)
HAPKIT = dict(rh=0.090, rs=0.073, rp=0.0042, m=0.10, Ib=(2.0e-4, 1.0e-4, 1.5e-4), com_y=-0.010,
              force_max=2.0, x_range=0.040, rotor_inertia=1.0e-6, qdot_max=20.0)

# EOD 6R on a tracked UGV deck with dynamics, 461 project S5
EOD_6R = dict(z0=0.25, xs=0.0, xe=0.60, xr=0.70, xw=1.15, xg=1.35,
              coms=((0, 0, 0.12), (0.30, 0, 0.25), (0.65, 0, 0.25), (0.93, 0, 0.25), (1.20, 0, 0.25),
                    (1.30, 0, 0.25)),
              limits=((-np.pi, np.pi), (-1.9, 1.2), (-2.8, 2.8), (-np.pi, np.pi), (-2.0, 2.0), (-np.pi, np.pi)),
              tau_max=(120.0, 250.0, 120.0, 30.0, 40.0, 15.0), qdot_max=(1.0, 1.0, 1.0, 1.5, 1.5, 2.0),
              home=(0.0, -0.35, -2.5, 0.0, 1.2, 0.0), payload_mass=5.0)


# ----------------------------------------------------------------------------- builders
def _ur_type_kinematics(W1, W2, L1, L2, H1, H2):
    """UR-type 6R exactly as in MR Example 4.5 (screw table p. 145): Slist, M."""
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    return S, M


def course_arm_dh(W1, W2, L1, L2, H1, H2) -> dict:
    """Hand-derived standard DH table of the course arm (same zero pose and joint signs as the PoE
    model; base = I, tool = Rot(z, pi)), vendored from the shared file for the DH-vs-PoE check."""
    return dict(d=[H1, 0, 0, W1, H2, W2], a=[0, L1, L2, 0, 0, 0],
                alpha=[-np.pi / 2, 0, 0, -np.pi / 2, np.pi / 2, 0], offset=[0.0] * 6,
                base=np.eye(4), tool=trans((0, 0, 0), _rotz(np.pi)))


def dh_fk(dh: dict, q) -> np.ndarray:
    """Standard DH forward kinematics T = base * prod Rot(z,th) Trans(z,d) Trans(x,a) Rot(x,alpha) * tool."""
    T = np.array(dh["base"], float)
    for i, qi in enumerate(q):
        th = dh["offset"][i] + qi
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(dh["alpha"][i]), np.sin(dh["alpha"][i])
        T = T @ np.array([[ct, -st * ca, st * sa, dh["a"][i] * ct],
                          [st, ct * ca, -ct * sa, dh["a"][i] * st],
                          [0, sa, ca, dh["d"][i]],
                          [0, 0, 0, 1.0]])
    return T @ np.asarray(dh["tool"], float)


def course_arm() -> SerialRobot:
    g_ = COURSE_ARM_GEOMETRY
    S, M = _ur_type_kinematics(**g_)
    coms = [trans(c) for _, c, _ in COURSE_ARM_LINKS]
    G = [box_inertia(m, *e) for m, _, e in COURSE_ARM_LINKS]
    r = SerialRobot(
        "Course arm (myCobot-280 class 6R)", M, S, mlist_from_absolute(coms, M), G,
        joint_limits=np.array(COURSE_ARM_LIMITS_DEG, float) * DEG,
        gravity=np.array([0, 0, -G0]), home=np.array(COURSE_ARM_HOME, float),
        tau_max=np.array(COURSE_ARM_TAU_RATED, float), link_radius=0.010,
        description="SBEG 361/461 course arm: low-cost desktop UR-type 6R (myCobot-280 class), MR Example 4.5 "
                    "layout with W1 = 63.4 mm, W2 = 45.6 mm, L1 = 110 mm, L2 = 96 mm, H1 = 131.2 mm, "
                    "H2 = 75.1 mm (nominal reach ≈ 0.28 m; {b} is 0.24 m from {s} at the PoE zero pose). Limits ±165° (J6 ±175°). Link masses (0.67 kg total), "
                    "box inertias and rated torques (1.8/3.0/1.5/0.6/0.6/0.4 N m) are ILLUSTRATIVE values. "
                    "The PoE zero pose (arm stretched along +x) is not the servo zero of a commercial arm.",
        source="Course Materials _build/shared/course_robots.py course_arm (kinematics = SBEG 361 project "
               "desktop_6r / scenario sheet S3, MR Example 4.5 form); vendored in mrstudio/course_models.py")
    r.qdot_max = np.full(6, COURSE_ARM_RATED_SPEED_DEG_S * DEG)
    r.dh = course_arm_dh(**g_)
    r.course_key = "course_arm"
    return r


def _planar_chain(name, key, L, m, r, Izz, Ixx_factor, limits, tau_max, description, source,
                  home=None, link_radius=0.03) -> SerialRobot:
    """Planar n-R chain in the vertical x-y plane (gravity -y), joints along +z, zero pose along +x;
    CoM r_i from joint i; G_i = diag(Ixx_factor Izz, Izz, Izz, m, m, m) (= shared planar_chain)."""
    n = len(L)
    jx = np.r_[0.0, np.cumsum(L)]
    coms = [trans([jx[i] + r[i], 0, 0]) for i in range(n)]
    M = trans([jx[n], 0, 0])
    S = np.column_stack([screw_revolute([0, 0, 1], [jx[i], 0, 0]) for i in range(n)])
    G = [np.diag([Ixx_factor * Izz[i], Izz[i], Izz[i], m[i], m[i], m[i]]) for i in range(n)]
    rob = SerialRobot(name, M, S, mlist_from_absolute(coms, M), G, joint_limits=np.array(limits, float),
                      planar=True, gravity=np.array([0, -G0, 0]),
                      home=np.zeros(n) if home is None else np.asarray(home, float),
                      tau_max=np.array(tau_max, float), link_radius=link_radius,
                      description=description, source=source)
    rob.course_key = key
    return rob


def eod_3r() -> SerialRobot:
    p = EOD_3R
    return _planar_chain(
        "EOD arm (3R planar)", "eod_3r", p["L"], p["m"], p["r"], p["Izz"], p["Ixx_factor"], p["limits"],
        p["tau_max"],
        "Vertical-plane shoulder / elbow / wrist-pitch arm of a tracked EOD robot (SBEG 461 Weeks 1–3): "
        "L = (0.60, 0.50, 0.15) m, m = (8.0, 5.0, 3.5) kg (link 3 = wrist + gripper + 2 kg suspect package), "
        "CoM r = (0.30, 0.25, 0.10) m, Izz = (0.24, 0.10, 0.008) kg m². Rated torques (155, 70, 15) N m cover "
        "the worst static gravity load g(0) = (106.44, 32.86, 3.43) N m. Gravity −y. Illustrative values.",
        "Course Materials _build/shared/course_robots.py eod_3r (461 w01/w02 arms.json / robots.json; "
        "τ_max from 461 w13 practical_B.json)", link_radius=0.035)


def arm_exo_2r() -> SerialRobot:
    p = ARM_EXO_2R
    return _planar_chain(
        "Arm–exoskeleton 2R", "arm_exo_2r", p["L"], p["m"], p["r"], p["Izz"], p["Ixx_factor"], p["limits"],
        p["tau_max"],
        "Sagittal-plane arm + exoskeleton (SBEG 461 Weeks 1–2): joint 1 = shoulder flexion, joint 2 = elbow "
        "flexion. Upper arm + cuff L = 0.30 m, 2.6 kg (CoM 0.13 m, Izz 0.025 kg m²); forearm + hand + cuff "
        "L = 0.35 m, 2.0 kg (CoM 0.17 m, Izz 0.020 kg m²). Vertical plane, gravity −y. Illustrative values.",
        "Course Materials _build/shared/course_robots.py arm_exo_2r (461 w01 arms.json, w02 robots.json)",
        link_radius=0.028)


def rehab_2r() -> SerialRobot:
    p = REHAB_2R
    L1, L2 = p["L"]
    m1, m2 = p["m"]
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, 1], [L1, 0, 0])]
    M = trans([L1 + L2, 0, 0])
    coms = [trans([L1 / 2, 0, 0]), trans([L1 + L2 / 2, 0, 0])]
    r = SerialRobot(
        "Rehab robot 2R (planar)", M, S, mlist_from_absolute(coms, M),
        [rod_inertia_x(m1, L1), rod_inertia_x(m2, L2)], joint_limits=np.array(p["limits"], float),
        planar=True, gravity=np.array([0, 0, -G0]), home=np.array(p["home"], float),
        tau_max=np.array(p["tau_max"], float), link_radius=0.025,
        description="Planar end-effector upper-limb rehabilitation robot (SBEG 461 project S1) moving in a "
                    "HORIZONTAL table plane: gravity is along −z, perpendicular to the plane, so it produces no "
                    "joint torque. L = (0.35, 0.35) m, uniform rods m = (2.4, 1.6) kg (radius 0.03 m), limits "
                    "θ1 ∈ [0, 180°], θ2 ∈ [11°, 160°], rated torques (20, 12) N m. Illustrative values.",
        source="Course Materials _build/shared/course_robots.py rehab_2r (461 project models.scenario_1_rehab)")
    r.qdot_max = np.array(p["qdot_max"], float)
    r.course_key = "rehab_2r"
    return r


def hapkit() -> SerialRobot:
    p = HAPKIT
    S = screw_revolute([0, 0, 1], [0, 0, 0]).reshape(6, 1)
    M = trans([0, p["rh"], 0])
    th_max = p["x_range"] / p["rh"]
    r = SerialRobot(
        "Hapkit (1-DOF haptic paddle)", M, S, mlist_from_absolute([trans([0, p["com_y"], 0])], M),
        [body_inertia(p["m"], p["Ib"])], joint_limits=np.array([[-th_max, th_max]]),
        planar=True, gravity=np.array([0, -G0, 0]), home=np.zeros(1),
        tau_max=np.array([p["force_max"] * p["rh"]]), link_radius=0.006,
        description="Hapkit-type 1-DOF haptic paddle (SBEG 461, Tier 1 haptic device): the handle pivots about "
                    "z_s; {b} at the grip r_h = 90 mm above the pivot; sector radius r_s = 73 mm, motor pulley "
                    "r_p = 4.2 mm (friction-drive ratio ≈ 17.4). Sector + handle 0.10 kg with its CoM 10 mm "
                    "below the pivot. Derated limits: handle range ±40 mm (θ = ±25.5°), handle force ≤ 2 N "
                    "(τ_max = 0.18 N m). Handle arc displacement x ≈ r_h θ (for θ > 0 the grip moves "
                    "towards −x_s). Illustrative values.",
        source="Course Materials _build/shared/course_robots.py hapkit (course requirements C.4; 461 "
               "scenario_sheets.md derated limits)")
    r.qdot_max = np.array([p["qdot_max"]])
    r.params = dict(rh=p["rh"], rs=p["rs"], rp=p["rp"], ratio=p["rs"] / p["rp"],
                    rotor_inertia=p["rotor_inertia"],
                    reflected_rotor_inertia=p["rotor_inertia"] * (p["rs"] / p["rp"]) ** 2,
                    force_max=p["force_max"], x_range=p["x_range"], loop_rate_hz=1000.0)
    r.course_key = "hapkit"
    return r


def eod_6r_ugv() -> SerialRobot:
    p = EOD_6R
    z0 = p["z0"]
    pts = [(0, 0, 0), (p["xs"], 0, z0), (p["xe"], 0, z0), (p["xr"], 0, z0), (p["xw"], 0, z0), (p["xw"], 0, z0)]
    ws = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    S = np.column_stack([screw_revolute(w, q) for w, q in zip(ws, pts)])
    M = trans([p["xg"], 0, z0])
    G = [box_inertia(6.0, 0.25, 0.25, 0.25), rod_inertia_x(5.0, 0.60, 0.05), rod_inertia_x(1.5, 0.15, 0.05),
         rod_inertia_x(2.5, 0.45, 0.04), box_inertia(1.0, 0.10, 0.08, 0.08), box_inertia(1.5, 0.20, 0.12, 0.06)]
    r = SerialRobot(
        "EOD arm (6R on UGV)", M, S, mlist_from_absolute([trans(c) for c in p["coms"]], M), G,
        joint_limits=np.array(p["limits"], float), gravity=np.array([0, 0, -G0]),
        home=np.array(p["home"], float), tau_max=np.array(p["tau_max"], float), link_radius=0.04,
        description="6-DOF EOD manipulator on a tracked UGV deck (SBEG 461 project S5): turret yaw, shoulder "
                    "pitch, elbow pitch, forearm roll, wrist pitch, gripper roll (spherical wrist at 1.15 m). "
                    "{s} on the turret base, shoulder 0.25 m above it, gripper point at 1.35 m. Links: turret "
                    "6 kg, boom 5 kg, elbow 1.5 kg, forearm 2.5 kg, wrist 1 kg, gripper 1.5 kg; rated torques "
                    "(120, 250, 120, 30, 40, 15) N m. Illustrative values (protective / humanitarian use).",
        source="Course Materials _build/shared/course_robots.py eod_6r_tracked (461 project "
               "models.scenario_5_eod); the 361 S4 eod_6r_ugv variant is kinematic only")
    r.qdot_max = np.array(p["qdot_max"], float)
    r.course_key = "eod_6r_tracked"
    return r


#: short labels (two course robots start with "EOD arm", so validation.short() alone is ambiguous)
SHORT = {"Course arm (myCobot-280 class 6R)": "Course arm", "EOD arm (6R on UGV)": "EOD 6R",
         "EOD arm (3R planar)": "EOD 3R", "Arm–exoskeleton 2R": "Arm–exo 2R",
         "Rehab robot 2R (planar)": "Rehab 2R", "Hapkit (1-DOF haptic paddle)": "Hapkit"}

#: name -> builder, registered into robots.ROBOTS by robots.py
COURSE_ROBOTS = {
    "Course arm (myCobot-280 class 6R)": course_arm,
    "EOD arm (6R on UGV)": eod_6r_ugv,
    "EOD arm (3R planar)": eod_3r,
    "Arm–exoskeleton 2R": arm_exo_2r,
    "Rehab robot 2R (planar)": rehab_2r,
    "Hapkit (1-DOF haptic paddle)": hapkit,
}


# Imported last on purpose: robots.py imports COURSE_ROBOTS from this module while it is being initialised,
# so COURSE_ROBOTS must exist before robots.py runs (the builders only use these names when called).
from .robots import SerialRobot, screw_revolute, trans  # noqa: E402


# ----------------------------------------------------------------------------- validation
def validation_checks() -> list[Check]:
    """Course-robot checks that do not depend on the shared file (independent derivations)."""
    from .dynamics import DynamicsEngine
    from .robots import get_robot

    out: list[Check] = []
    rng = np.random.default_rng(361)

    # 1. course arm: PoE FK (MR Eq. 4.14) = hand-derived standard DH table (MR App. C)
    t0 = time.perf_counter()
    r = get_robot("Course arm (myCobot-280 class 6R)")
    err = 0.0
    for _ in range(50):
        q = r.random_config(rng)
        err = max(err, np.abs(r.fk(q) - dh_fk(r.dh, q)).max())
    out.append(Check("Course arm: PoE FK = standard DH table", "MR 4.1, App. C", float(err), 1e-12,
                     "T(θ) = e^[S1]θ1⋯e^[S6]θ6 M vs base·ΠRot(z,θ)Trans(z,d)Trans(x,a)Rot(x,α)·tool with the "
                     f"hand-derived DH table of the course notes; 50 random poses, max |ΔT| "
                     f"[{(time.perf_counter() - t0) * 1e3:.0f} ms]", CAT))

    # 2. EOD 3R: gravity torque at q = 0 equals the value quoted in course_robots.md (2-decimal rounding)
    r = get_robot("EOD arm (3R planar)")
    g0 = DynamicsEngine(r, "mr").gravity_torque(np.zeros(3))
    out.append(Check("EOD 3R: g(0) = (106.44, 32.86, 3.43) N m", "MR 8.1, 8.3", float(np.abs(
        g0 - EOD_3R_GRAVITY_AT_ZERO).max()), 5e-3,
        f"RNEA gravity torque at the stretched pose = {np.round(g0, 3)} N m vs course notes; hand formula "
        "g_i = Σ_{k≥i} m_k g (x_ck − x_i)", CAT))

    # 3. rated torques cover the worst static gravity load of every course robot
    worst = []
    for name in COURSE_ROBOTS:
        r = get_robot(name)
        eng = DynamicsEngine(r, "mr")
        gmax = np.zeros(r.n)
        for _ in range(200):
            gmax = np.maximum(gmax, np.abs(eng.gravity_torque(r.random_config(rng))))
        worst.append(float(np.max(gmax / r.tau_max)))
    out.append(Check("Course robots: τ_max ≥ |g(θ)| (static holding)", "MR 8.1, 8.3", max(worst), 1.0,
                     "max_i |g_i(θ)|/τ_max,i over 200 random poses inside the joint limits, per robot: "
                     + ", ".join(f"{SHORT[n]} {w:.2f}" for n, w in zip(COURSE_ROBOTS, worst))
                     + " (must be ≤ 1)", CAT))

    # 4. rehab robot in a horizontal plane: gravity does no work, g(θ) ≡ 0
    r = get_robot("Rehab robot 2R (planar)")
    eng = DynamicsEngine(r, "mr")
    e = max(np.abs(eng.gravity_torque(r.random_config(rng))).max() for _ in range(20))
    out.append(Check("Rehab 2R (horizontal plane): g(θ) = 0", "MR 8.1", float(e), 1e-12,
                     "gravity −z is perpendicular to the plane of motion, so RNEA gives zero gravity torque", CAT))

    # 5. Hapkit 1-DOF: M(θ) = I_zz + m|c|² (parallel-axis theorem) and pendulum gravity torque
    r = get_robot("Hapkit (1-DOF haptic paddle)")
    eng = DynamicsEngine(r, "mr")
    p = HAPKIT
    Ipiv = p["Ib"][2] + p["m"] * p["com_y"] ** 2
    e = 0.0
    for _ in range(20):
        th = r.random_config(rng)
        g_hand = p["m"] * G0 * (-p["com_y"]) * np.sin(th[0])     # CoM below the pivot: restoring torque
        e = max(e, abs(eng.mass_matrix(th)[0, 0] - Ipiv), abs(eng.gravity_torque(th)[0] - g_hand))
    out.append(Check("Hapkit: M = I_zz + m d², g = m g d sinθ", "MR 8.1, 8.2", float(e), 1e-12,
                     f"1-DOF paddle: pivot inertia {Ipiv:.3e} kg m² (Steiner), gravity torque of the CoM "
                     f"{-p['com_y'] * 1e3:.0f} mm below the pivot, vs MR MassMatrix / GravityForces", CAT))
    return out
