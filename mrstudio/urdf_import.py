"""URDF -> Modern Robotics description (MR Sec. 4.2 and Sec. 8.8, Exercise 8.12).

A URDF describes every joint i by an <origin> (pose of joint frame i in the frame of joint i-1 at
theta = 0) and an <axis>, and every link by its <inertial> element (CoM frame origin, mass, rotational
inertia).  MR Sec. 8.8 leaves the conversion to the quantities used by Newton-Euler (Slist, Mlist,
Glist) as Exercise 8.12; this module performs it:

* the XML is parsed by Pinocchio (``pin.buildModelFromUrdf`` / ``buildModelFromXML``) -- no
  hand-written parser.  Pinocchio merges links attached by fixed joints into their parent
  (mass, CoM and inertia are lumped exactly) and keeps them as frames;
* at the zero configuration theta = 0 the space Jacobian columns are the screw axes
  S_i = (w_i, v_i) in {s} (MR Sec. 5.1.1: J_s(0) = [S_1 ... S_n]); revolute and continuous joints give
  w_i != 0, prismatic joints S_i = (0, v_i) (MR Sec. 3.3.2.2);
* M = pose of the chosen end-effector link at theta = 0 (MR Sec. 4.1);
* CoM frame {i} = joint frame i translated to the link CoM; G_i = diag(I_b, m I) about the CoM
  (MR Eq. 8.32), and Mlist = [M_{i-1,i}] relative CoM-frame poses at home (MR Sec. 8.3.2);
* joint position limits and <limit effort> (-> tau_max) are copied.

Kinematic trees: if the URDF has several leaves the user must pick the end-effector link; the chain
from the root to it is kept and every joint off that chain is locked at 0 (its bodies become rigidly
attached, ``pin.buildReducedModel``), so the result is an open serial chain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import modern_robotics as mr
import numpy as np

from .checks import Check
from .robots import PRISMATIC_DEFAULT_LIMITS, SerialRobot

try:
    import pinocchio as pin
except Exception:  # pragma: no cover
    pin = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "urdf")
IIWA7_URDF = os.path.join(DATA_DIR, "iiwa7.urdf")
CAT = "URDF import"

_REVOLUTE = ("RX", "RY", "RZ", "RevoluteUnaligned", "RUBX", "RUBY", "RUBZ", "RevoluteUnboundedUnaligned")
_PRISMATIC = ("PX", "PY", "PZ", "PrismaticUnaligned")

#: small test arm used by the validation checks: revolute + continuous + prismatic + fixed joints,
#: non-trivial joint origins (rpy), axes not along x/y/z, off-diagonal inertias
TEST_URDF = """<?xml version="1.0"?>
<robot name="rcpr_test_arm">
  <link name="base"><inertial><origin xyz="0 0 0.05"/><mass value="4"/>
    <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.03"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.2" rpy="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-2.5" upper="2.5" effort="80" velocity="2"/></joint>
  <link name="l1"><inertial><origin xyz="0.02 0.01 0.15" rpy="0.1 0.2 0.3"/><mass value="2.5"/>
    <inertia ixx="0.03" ixy="0.001" ixz="-0.002" iyy="0.025" iyz="0.0015" izz="0.01"/></inertial></link>
  <joint name="j2" type="continuous"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.05 0 0.3" rpy="1.5707963 0 0"/><axis xyz="0 0.6 0.8"/>
    <limit effort="40" velocity="2"/></joint>
  <link name="l2"><inertial><origin xyz="0.2 0 0.01"/><mass value="1.8"/>
    <inertia ixx="0.004" ixy="0.0005" ixz="0" iyy="0.03" iyz="0" izz="0.031"/></inertial></link>
  <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.4 0 0" rpy="0 0.2 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.25" effort="300" velocity="0.5"/></joint>
  <link name="l3"><inertial><origin xyz="0.1 0 0"/><mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/></inertial></link>
  <joint name="mount" type="fixed"><parent link="l3"/><child link="wrist_mount"/>
    <origin xyz="0.2 0 0" rpy="0 1.5707963 0"/></joint>
  <link name="wrist_mount"><inertial><origin xyz="0 0 0.02"/><mass value="0.3"/>
    <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0001"/></inertial></link>
  <joint name="j4" type="revolute"><parent link="wrist_mount"/><child link="l4"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 -1"/>
    <limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
  <link name="l4"><inertial><origin xyz="0 0 0.04"/><mass value="0.5"/>
    <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0002"/></inertial></link>
  <joint name="tool" type="fixed"><parent link="l4"/><child link="tool0"/>
    <origin xyz="0 0 0.1" rpy="0 0 0.7"/></joint>
  <link name="tool0"/>
</robot>
"""


@dataclass
class UrdfInfo:
    """Book-keeping attached to an imported robot as ``robot.urdf``."""
    source: str                      # file path or "<string>"
    robot_name: str
    tip: str                         # end-effector link
    joint_names: list[str]
    joint_kinds: list[str]           # "revolute" | "continuous" | "prismatic"
    locked: list[str] = field(default_factory=list)   # joints off the chain, locked at 0
    n_fixed_merged: int = 0
    model: object = None             # the (reduced) pinocchio model the conversion was made from
    nq_idx: list[int] = field(default_factory=list)

    def q_pin(self, theta) -> np.ndarray:
        """Pinocchio configuration vector for MR joint values theta (continuous joints: (cos, sin))."""
        q = pin.neutral(self.model)
        for i, (k, iq) in enumerate(zip(self.joint_kinds, self.nq_idx)):
            if k == "continuous":
                q[iq:iq + 2] = np.cos(theta[i]), np.sin(theta[i])
            else:
                q[iq] = theta[i]
        return q


# ============================================================================ loading
def _is_xml(s: str) -> bool:
    return s.lstrip().startswith("<")


def load_pinocchio_model(urdf):
    """Pinocchio model straight from a URDF path or URDF XML string (fixed base)."""
    if pin is None:
        raise RuntimeError("pinocchio is not installed")
    if _is_xml(urdf):
        return pin.buildModelFromXML(urdf)
    if not os.path.isfile(urdf):
        raise FileNotFoundError(urdf)
    return pin.buildModelFromUrdf(urdf)


def _children(model):
    ch = {j: [] for j in range(model.njoints)}
    for j in range(1, model.njoints):
        ch[model.parents[j]].append(j)
    return ch


def link_names(urdf) -> list[str]:
    """All link names of a URDF (Pinocchio BODY frames), root first."""
    model = load_pinocchio_model(urdf)
    return [f.name for f in model.frames if f.type == pin.FrameType.BODY]


def leaf_links(urdf_or_model) -> list[str]:
    """Candidate end-effector links: the deepest link frame attached to every leaf joint."""
    model = urdf_or_model if pin is not None and isinstance(urdf_or_model, pin.Model) \
        else load_pinocchio_model(urdf_or_model)
    ch = _children(model)
    leaves = [j for j in range(1, model.njoints) if not ch[j]] or [0]
    out = []
    for j in leaves:
        bodies = [f.name for f in model.frames if f.parentJoint == j and f.type == pin.FrameType.BODY]
        if bodies:
            out.append(bodies[-1])
    return out


def _kind(jm) -> str:
    short = jm.shortname().replace("JointModel", "")
    if short in _PRISMATIC:
        return "prismatic"
    if short in _REVOLUTE:
        return "continuous" if jm.nq == 2 else "revolute"
    raise ValueError(f"joint type '{short}' is not a 1-dof revolute/continuous/prismatic joint "
                     "(floating, planar and spherical joints are not supported)")


# ============================================================================ conversion
def from_urdf(urdf: str, tip: str | None = None, name: str | None = None,
              gravity=(0.0, 0.0, -9.81)) -> SerialRobot:
    """Convert a URDF (file path or XML string) into a Modern Robotics SerialRobot.

    tip   -- end-effector link name (default: the last link of the chain; required if the URDF is a
             branched tree, then the chain root -> tip is kept and off-chain joints are locked at 0).
    name  -- robot name (default: the URDF <robot name>).
    The result carries ``robot.urdf`` (:class:`UrdfInfo`)."""
    full = load_pinocchio_model(urdf)
    src = "<string>" if _is_xml(urdf) else os.path.abspath(urdf)
    if full.njoints <= 1:
        raise ValueError("the URDF has no movable joint")
    if tip is None:
        leaves = leaf_links(full)
        if len(leaves) != 1:
            raise ValueError("the URDF is a branched kinematic tree (" + ", ".join(leaves) + " are all chain "
                             "ends); choose the end-effector link to import the serial chain base -> tip")
        tip = leaves[0]
    if not full.existFrame(tip):
        raise ValueError(f"link '{tip}' not found in the URDF")
    fid = full.getFrameId(tip)
    chain, j = [], full.frames[fid].parentJoint
    while j > 0:
        chain.append(j)
        j = full.parents[j]
    chain = chain[::-1]
    if not chain:
        raise ValueError(f"link '{tip}' is rigidly attached to the base (no joint between them)")
    locked = [j for j in range(1, full.njoints) if j not in chain]
    if locked:
        model = pin.buildReducedModel(full, locked, pin.neutral(full))
    else:
        model = full
    locked_names = [full.names[j] for j in locked]
    n = model.njoints - 1
    kinds = [_kind(model.joints[j]) for j in range(1, n + 1)]
    data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    J = pin.computeJointJacobians(model, data, q0)        # world frame, rows (v, w): = J_s(0)
    S = np.vstack([J[3:], J[:3]])                         # MR order (w, v)
    for i in range(n):
        w = S[:3, i]
        nw = np.linalg.norm(w)
        S[:, i] /= nw if nw > 1e-9 else np.linalg.norm(S[3:, i])
        if (kinds[i] == "prismatic") != (nw < 1e-9):
            raise ValueError(f"joint {model.names[i + 1]}: inconsistent axis")
    M = data.oMf[model.getFrameId(tip)].homogeneous.copy()

    # ---- inertias: CoM frame {i} = joint frame i shifted to the CoM (orientation of joint frame)
    masses = np.array([model.inertias[j].mass for j in range(1, n + 1)])
    Mlist = Glist = None
    if np.all(masses > 0):
        Mc, Glist = [np.eye(4)], []
        for j in range(1, n + 1):
            I = model.inertias[j]
            Mc.append(data.oMi[j].homogeneous @ mr.RpToTrans(np.eye(3), np.asarray(I.lever)))
            G = np.zeros((6, 6))
            G[:3, :3] = np.asarray(I.inertia)
            G[3:, 3:] = I.mass * np.eye(3)
            Glist.append(G)
        Mlist = [mr.TransInv(Mc[i]) @ Mc[i + 1] for i in range(n)] + [mr.TransInv(Mc[-1]) @ M]

    # ---- limits
    lim, tau = np.zeros((n, 2)), np.zeros(n)
    for i in range(n):
        jm = model.joints[i + 1]
        if kinds[i] == "continuous":
            lim[i] = (-2 * np.pi, 2 * np.pi)
        else:
            lo, hi = model.lowerPositionLimit[jm.idx_q], model.upperPositionLimit[jm.idx_q]
            ok = np.isfinite(lo) and np.isfinite(hi) and hi > lo and max(abs(lo), abs(hi)) < 1e6
            lim[i] = (lo, hi) if ok else (PRISMATIC_DEFAULT_LIMITS if kinds[i] == "prismatic" else (-np.pi, np.pi))
        e = model.effortLimit[jm.idx_v]
        tau[i] = e if (np.isfinite(e) and 0 < e < 1e6) else (400.0 if kinds[i] == "prismatic" else 100.0)

    auto = Glist is None
    if auto:   # no inertial data: physically consistent stand-in (rods + motors) so every tab works
        from .robots import auto_inertias
        Mlist, Glist = auto_inertias(S, M, joint_limits=lim)

    n_fixed = sum(1 for f in full.frames if f.type == pin.FrameType.FIXED_JOINT and f.name != "universe")
    rname = name or full.name or "URDF robot"
    jt = "".join("P" if k == "prismatic" else "R" for k in kinds)
    desc = (f"Imported from URDF '{os.path.basename(src) if src != '<string>' else full.name}': {n} joints ({jt}), "
            f"end-effector link '{tip}'"
            + (f", {n_fixed} fixed joint(s) merged" if n_fixed else "")
            + (f", {len(locked)} off-chain joint(s) locked at 0" if locked else "")
            + ("." if not auto else "; the URDF has links without inertial (mass) data, so inertias were generated "
               "automatically (rods + motors) -- dynamics are illustrative only.")
            + " Joint damping/friction and rotor inertias in the URDF are ignored.")
    robot = SerialRobot(rname, M, S, Mlist, Glist, joint_limits=lim, gravity=np.asarray(gravity, float),
                        home=np.clip(np.zeros(n), lim[:, 0], lim[:, 1]), tau_max=tau, description=desc,
                        source=f"URDF {src} (MR Sec. 4.2, 8.8) via mrstudio.urdf_import.from_urdf")
    robot.link_radius = float(np.clip(0.035 * robot.reach, 0.012, 0.045))
    robot.inertia_source = "auto" if auto else "urdf"
    robot.urdf = UrdfInfo(src, full.name, tip, [model.names[j] for j in range(1, n + 1)], kinds,
                          locked_names, n_fixed, model, [model.joints[j].idx_q for j in range(1, n + 1)])
    return robot


def summary(robot: SerialRobot) -> str:
    """Short HTML summary of an imported robot (used by the URDF dialog)."""
    u = getattr(robot, "urdf", None)
    rows = []
    for i in range(robot.n):
        lo, hi = robot.joint_limits[i]
        unit, k = ("m", 1.0) if robot.is_prismatic[i] else ("°", 180 / np.pi)
        kind = u.joint_kinds[i] if u else robot.joint_types[i]
        jn = u.joint_names[i] if u else f"joint {i + 1}"
        rows.append(f"<tr><td>{i + 1}</td><td>{jn}</td><td>{kind}</td>"
                    f"<td>{lo * k:.3g} … {hi * k:.3g} {unit}</td><td>{robot.tau_max[i]:.4g}</td></tr>")
    mass = f"{robot.masses.sum():.2f} kg (moving links)"
    if getattr(robot, "inertia_source", "") == "auto":
        mass += " <i>(generated — URDF has no inertial data)</i>"
    return (f"<b>{robot.name}</b> — {robot.n} joints <b>{robot.joint_types}</b>, reach ≈ {robot.reach:.2f} m, "
            f"mass {mass}<br>{robot.description}"
            "<table cellspacing=6><tr><th>#</th><th>joint</th><th>type</th><th>limits</th>"
            "<th>effort</th></tr>" + "".join(rows) + "</table>")


# ============================================================================ examples
def example_urdfs() -> dict[str, str]:
    """Bundled / locally available example URDF files: {label: path}. Works offline."""
    out = {}
    if os.path.isfile(IIWA7_URDF):
        out["KUKA LBR iiwa 7 R800 (bundled)"] = IIWA7_URDF
    try:
        import rtbdata
        base = os.path.join(os.path.dirname(rtbdata.__file__), "xacro")
        for label, rel in (("KUKA KR 16-2 (rtb-data)", "kuka_description/kuka_kr16/urdf/kr16_2.urdf"),
                           ("KUKA KR 120 R2500 pro (rtb-data)", "kuka_description/kuka_kr120/urdf/kr120r2500pro.urdf"),
                           ("KUKA KR 210 L150 (rtb-data)", "kuka_description/kuka_kr210/urdf/kr210l150.urdf"),
                           ("Lynxmotion AL5D (rtb-data)", "lynxmotion_al5d_description/urdf/al5d_robot.urdf"),
                           ("ABB IRB 140, kinematics only (rtb-data)", "abb_irb140_description/urdf/irb140.urdf"),
                           ("PUMA 560, kinematics only (rtb-data)",
                            "unimation_puma560_description/urdf/puma560_robot.urdf")):
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                out[label] = p
    except Exception:
        pass
    # robot_descriptions caches (only if already downloaded -- never triggers a download)
    cache = os.path.expanduser("~/.cache/robot_descriptions")
    for label, rel in (("Franka Panda (robot_descriptions cache)",
                        "example-robot-data/robots/panda_description/urdf/panda.urdf"),
                       ("UR5 (robot_descriptions cache)", "example-robot-data/robots/ur_description/urdf/ur5_robot.urdf")):
        p = os.path.join(cache, rel)
        if os.path.isfile(p):
            out[label] = p
    return out


# ============================================================================ validation
def compare_with_pinocchio(robot: SerialRobot, urdf, n_samples: int = 8, seed: int = 0) -> dict:
    """Max errors of the converted robot (MR code: FKinSpace, InverseDynamics, MassMatrix) against the
    Pinocchio model loaded directly from the same URDF (frame placement, RNEA, CRBA)."""
    u = robot.urdf
    ref = load_pinocchio_model(urdf)
    if u.locked:
        ids = [ref.getJointId(nm) for nm in u.locked]
        ref = pin.buildReducedModel(ref, ids, pin.neutral(ref))
    data = ref.createData()
    fid = ref.getFrameId(u.tip)
    rng = np.random.default_rng(seed)
    e_fk = e_id = e_mm = 0.0
    info = UrdfInfo(u.source, u.robot_name, u.tip, u.joint_names, u.joint_kinds, model=ref,
                    nq_idx=[ref.joints[ref.getJointId(nm)].idx_q for nm in u.joint_names])
    for _ in range(n_samples):
        th = robot.random_config(rng)
        q = info.q_pin(th)
        pin.framesForwardKinematics(ref, data, q)
        e_fk = max(e_fk, np.abs(robot.fk(th) - data.oMf[fid].homogeneous).max())
        if robot.has_dynamics and getattr(robot, "inertia_source", "urdf") == "urdf":
            dth, ddth = rng.uniform(-1, 1, robot.n), rng.uniform(-1, 1, robot.n)
            tau_mr = mr.InverseDynamics(th, dth, ddth, robot.gravity, np.zeros(6), robot.Mlist, robot.Glist,
                                        robot.Slist)
            tau_pin = pin.rnea(ref, data, q, dth, ddth)
            e_id = max(e_id, np.abs(np.asarray(tau_mr) - tau_pin).max() / max(1.0, np.abs(tau_pin).max()))
            Mmr = np.asarray(mr.MassMatrix(th, robot.Mlist, robot.Glist, robot.Slist))
            Mp = pin.crba(ref, data, q)
            Mp = np.triu(Mp) + np.triu(Mp, 1).T
            e_mm = max(e_mm, np.abs(Mmr - Mp).max() / max(1.0, np.abs(Mp).max()))
    return dict(fk=e_fk, rnea=e_id, mass=e_mm)


def validation_checks() -> list[Check]:
    out: list[Check] = []
    cases = [("test arm R-C-P-R + fixed joints (embedded URDF)", TEST_URDF, "tool0")]
    if os.path.isfile(IIWA7_URDF):
        cases.append(("KUKA iiwa 7 (bundled URDF)", IIWA7_URDF, "iiwa_link_ee"))
    ex = example_urdfs()
    if "KUKA KR 16-2 (rtb-data)" in ex:
        cases.append(("KUKA KR 16-2 (rtb-data URDF)", ex["KUKA KR 16-2 (rtb-data)"], None))
    for label, urdf, tip in cases:
        try:
            robot = from_urdf(urdf, tip=tip)
            e = compare_with_pinocchio(robot, urdf)
        except Exception as exc:  # pragma: no cover
            out.append(Check(f"URDF import — {label}", "MR 4.2, 8.8", float("inf"), 0.0, repr(exc), CAT))
            continue
        kinds = ", ".join(sorted(set(robot.urdf.joint_kinds)))
        out.append(Check(f"URDF → PoE: FK vs Pinocchio from the same URDF — {label}", "MR 4.1, 4.2", e["fk"], 1e-12,
                         f"T(θ) = e^[S1]θ1…e^[Sn]θn M (mr.FKinSpace) with S_i, M extracted at θ = 0, versus "
                         f"Pinocchio's placement of link '{robot.urdf.tip}' computed from the URDF joint origins; "
                         f"8 random configurations; joint kinds: {kinds}.", CAT))
        if robot.inertia_source == "urdf":
            out.append(Check(f"URDF → Mlist, Glist: RNEA vs Pinocchio — {label}", "MR 8.3, 8.8", e["rnea"], 1e-12,
                             "τ = mr.InverseDynamics(θ, θ̇, θ̈) with Mlist/Glist built from the URDF <inertial> "
                             "elements (MR Exercise 8.12) versus pin.rnea on the model parsed directly; relative "
                             "to max(1, |τ|).", CAT))
            out.append(Check(f"URDF → Mlist, Glist: mass matrix vs Pinocchio CRBA — {label}", "MR 8.4, 8.8",
                             e["mass"], 1e-12, "mr.MassMatrix vs pin.crba, relative to max(1, |M|).", CAT))
    # branched trees are rejected with a clear message unless a tip is given
    branched = TEST_URDF.replace("</robot>", """  <joint name="side" type="revolute"><parent link="l1"/><child link="side_link"/>
    <origin xyz="0 0.1 0.1"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="5" velocity="1"/></joint>
  <link name="side_link"><inertial><origin xyz="0 0.05 0"/><mass value="0.4"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
</robot>""")
    try:
        from_urdf(branched)
        rejected = False
    except ValueError as exc:
        rejected = "branched" in str(exc)
    r = from_urdf(branched, tip="tool0")
    e = compare_with_pinocchio(r, branched)
    err = max(e.values()) if rejected else float("inf")
    out.append(Check("URDF branched tree: rejected without a tip; chain to the tip (off-chain joint locked) "
                     "vs Pinocchio", "MR 4.2", err, 1e-12,
                     "test arm + a side branch on link 1: from_urdf(urdf) raises ValueError ('branched'); "
                     "from_urdf(urdf, tip='tool0') locks the side joint at 0 (pin.buildReducedModel); "
                     "error = max of FK / RNEA / mass-matrix discrepancies (inf if not rejected).", CAT))
    return out
