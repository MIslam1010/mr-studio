"""Pure maths / helpers for the inverse-kinematics tab (no Qt): task classification, task error and Jacobian,
formatting of joint vectors, and the textual cross-checks against the Modern Robotics library and closed-form IK."""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

from ... import kinematics as kin
from ...robots import trans


def _deg(th, d=1) -> str:
    return "(" + ", ".join(f"{np.degrees(x):+.{d}f}" for x in th) + ")"


def _fmt_theta(robot, th) -> str:
    """Joint vector with units: revolute in degrees, prismatic in metres."""
    return "(" + ", ".join(f"{x:+.3f} m" if p else f"{np.degrees(x):+.2f}°"
                           for x, p in zip(th, robot.is_prismatic)) + ")"


def _rotz(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _task(robot) -> str:
    """IK task of a robot: 'pose' (6-D), 'planar' (x, y, φ), 'position' (3-D point) or 'point2d' (x, y)."""
    if robot.planar:
        return "planar" if robot.n >= 3 else "point2d"
    return "pose" if robot.n >= 6 else "position"


def _redundancy(robot):
    """Task used for the self-motion demo, or None if the arm is not redundant for any sensible task."""
    if robot.planar:
        return "point2d" if robot.n > 2 else None
    if robot.n > 6:
        return "pose"
    return "position" if 3 < robot.n < 6 else None


def _textbook_planar(robot) -> bool:
    """Planar 2R/3R arm with all joint axes along z through points of the x axis (closed form applies)."""
    S = robot.Slist
    return bool(robot.planar and robot.n in (2, 3) and np.allclose(S[:3], np.array([[0], [0], [1]]))
                and np.allclose(S[3], 0) and np.allclose(S[5], 0) and np.allclose(robot.M[:3, :3], np.eye(3))
                and abs(robot.M[1, 3]) < 1e-12 and abs(robot.M[2, 3]) < 1e-12)


_TASK_DIM = {"pose": 6, "planar": 3, "position": 3, "point2d": 2}


def _task_error(robot, th, T_d, task):
    """Error vector e and Jacobian J of the task (Newton step: Δθ = J⁺ e)."""
    if task in ("pose", "planar"):
        T = kin.fk_body(robot, th)
        Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_d))
        return Vb, kin.jacobian_body(robot, th)
    T = kin.fk_space(robot, th)
    e = T_d[:3, 3] - T[:3, 3]
    J = kin.point_jacobian(robot, th)
    if task == "point2d":
        return e[:2], J[:2]
    return e, J


def cross_check_lines(robot, T_target, res, th0, args):
    """Lines comparing one IK run with mr.IKinBody / mr.IKinSpace (re-run on the same data) and analytic IK."""
    lines = []
    if not args["position_only"]:
        space = args["frame"] == "space"
        name = "mr.IKinSpace" if space else "mr.IKinBody"
        fn, lst = (mr.IKinSpace, robot.Slist) if space else (mr.IKinBody, robot.Blist)
        th_mr, ok_mr = fn(lst, robot.M, T_target, th0, args["eomg"], args["ev"])
        th_mr = np.asarray(th_mr, float)
        if args["damping"] > 0:
            d = kin.joint_distance(robot, th_mr, res.theta)
            if not ok_mr:
                same = f"{name} did not converge from this θ⁰" + (" — DLS did ✓" if res.success else "")
            elif not res.success:
                same = "DLS not converged (increase iterations or lower λ)"
            elif d < 1e-4:
                same = "same solution ✓"
            else:
                same = "both converged, to different IK branches ✓"
            lines.append(f"DLS vs {name} (pure Newton, same θ⁰):\n  ‖Δθ‖ = {d:.1e} → {same}")
        else:
            dk = [np.linalg.norm(th_mr - h) for h in res.history]
            k = int(np.argmin(dk))
            if dk[k] < 1e-9:
                lines.append(f"✓ matches {name}: its output = our iterate θ^{k},  ‖Δθ‖ = {dk[k]:.1e}" +
                             ("" if ok_mr else f"\n   ({name} stops after 20 iterations, not converged)"))
            else:
                lines.append(f"≈ {name}: ‖Δθ‖ = {np.linalg.norm(th_mr - res.theta):.1e}"
                             f" ({'converged' if ok_mr else 'not converged in its 20 iterations'})")
    if _textbook_planar(robot):
        T = T_target
        phi = np.arctan2(T[1, 0], T[0, 0]) if robot.n == 3 else None
        an = kin.ik_analytic_planar(robot, T[0, 3], T[1, 3], phi)
        if an:
            d = [np.linalg.norm(kin.wrap_angles(a - res.theta)) for a in an]
            k = int(np.argmin(d))
            lines.append(f"{'✓' if d[k] < 1e-5 else '≈'} analytic IK (law of cosines, "
                         f"{'elbow-down' if k == 0 else 'elbow-up'}): ‖Δθ‖ = {d[k]:.1e}")
        else:
            lines.append("analytic IK: target outside the annular workspace (no solution)")
    elif args["position_only"]:
        lines.append("position-only task: mr.IKinBody (full pose) does not apply")
    return lines


def analytic_vs_numeric_lines(robot, T, sols):
    """Closed-form planar IK (both elbow branches) compared with the numerically found solutions."""
    phi = np.arctan2(T[1, 0], T[0, 0]) if robot.n == 3 else None
    an = kin.ik_analytic_planar(robot, T[0, 3], T[1, 3], phi)
    lines = ["Analytic IK (MR §6.1) vs numeric:"]
    for a, lab in zip(an, ["elbow-down θ₂>0", "elbow-up   θ₂<0"]):
        d = min((kin.joint_distance(robot, a, s) for s in sols), default=np.nan)
        lines.append(f"{lab}: {_deg(a)}°\n   ↔ numeric ‖Δθ‖ = {d:.1e} {'✓' if d < 1e-5 else '✗'}")
    if not an:
        lines.append("target unreachable (|c₂| > 1)")
    return lines


def spread_samples(robot, sols, k=8):
    """k well-spread members of a solution set (farthest-point sampling in joint distance)."""
    chosen = [0]
    D = np.array([[kin.joint_distance(robot, a, b) for b in sols] for a in sols])
    while len(chosen) < k:
        chosen.append(int(np.argmax(D[:, chosen].min(axis=1))))
    return [sols[i] for i in chosen]


def redundancy_text(robot, red) -> str:
    """Explanatory text for the self-motion panel (`red` = result of _redundancy(robot))."""
    if red is None:
        return (f"{robot.name}: n = {robot.n} joints = task dimension → not redundant "
                "(only isolated solutions).  Select the Franka Panda (7R) or the planar 3R arm.")
    what = {"pose": "full 6-D pose", "point2d": "position (x, y) only",
            "position": "position only"}[red]
    dim = {"pose": 6, "point2d": 2, "position": 3}[red]
    return (f"task = {what}: n − m = {robot.n} − {dim} = {robot.n - dim} → "
            f"{robot.n - dim}-D null space of J.  Press ▶ to swing the arm while the "
            "end-effector stays put.")


def robust_ik(robot, T, theta, args):
    """Warm-started IK for real-time use: Newton–Raphson; if it fails (target near a singularity or
    outside the workspace) retry with DLS and keep the best iterate of both runs (never worse than θ)."""
    res = kin.ik_newton_raphson(robot, T, theta, **args)
    if res.success:
        return res
    res2 = kin.ik_newton_raphson(robot, T, theta,
                                 **dict(args, damping=max(args["damping"], 0.05), max_iter=25))
    best = None
    for r in (res, res2):
        e = np.asarray(r.err_w) + np.asarray(r.err_v)
        k = int(np.argmin(e))
        if best is None or e[k] < best[0]:
            best = (e[k], r, k)
    _, r, k = best
    r.theta, r.iterations = r.history[k].copy(), k
    return r


def circle_frame(robot, T0, r):
    """Centre and in-plane axes (e1, e2) of the real-time-IK circle of radius r through the current end-effector.
    The circle bulges towards the base (stays inside the workspace): planar arms -> in the x-y plane;
    spatial arms -> vertical plane through the base axis."""
    p0 = T0[:3, 3]
    radial = np.array([p0[0], p0[1], 0.0])
    e1 = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > 1e-6 else np.array([1.0, 0, 0])
    e2 = np.cross([0, 0, 1.0], e1) if robot.planar else np.array([0, 0, 1.0])
    return p0 - r * e1, e1, e2


def target_label_text(robot, theta, T_target) -> str:
    """Target position / orientation and current task error of the IK target label."""
    task = _task(robot)
    p = T_target[:3, 3]
    T = kin.fk_space(robot, theta)
    if task in ("pose", "planar"):
        Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_target))
        rpy = np.degrees(kin.rot_to_rpy(T_target[:3, :3]))
        orient = (f"rpy = ({rpy[0]:+.1f}°, {rpy[1]:+.1f}°, {rpy[2]:+.1f}°)" if task == "pose" else
                  f"φ = {np.degrees(np.arctan2(T_target[1, 0], T_target[0, 0])):+.1f}°")
        return (f"p_d = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m\n{orient}\n"
                f"current error: ‖ω_b‖ = {np.linalg.norm(Vb[:3]):.2e}, "
                f"‖v_b‖ = {np.linalg.norm(Vb[3:]):.2e}")
    else:
        e = np.linalg.norm(p - T[:3, 3])
        return (f"p_d = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m\n"
                f"position-only task ({'x, y' if robot.planar else 'x, y, z'}); "
                f"current error ‖p_d − p‖ = {e:.2e} m")


def target_row_spec(robot):
    """Slider rows (name, lo, hi, decimals, unit, scale) of the target pose for this robot's IK task."""
    task = _task(robot)
    R = 1.1 * robot.reach
    spec = [("x", -R, R, 3, " m", 1.0), ("y", -R, R, 3, " m", 1.0)]
    if not robot.planar:
        spec.append(("z", -R, R, 3, " m", 1.0))
    a = (-np.pi, np.pi, 1, "°", 180 / np.pi)
    if task == "pose":
        spec += [("roll", *a), ("pitch", *a), ("yaw", *a)]
    elif task == "planar":
        spec += [("φ", *a)]
    return spec


def iterate_rows(robot, res):
    """Text matrix of the Newton iterates: index, joint values (deg / m), error norms."""
    rows = []
    for i, th in enumerate(res.history):
        rows.append([str(i)] + [f"{x:+.3f}" if p else f"{np.degrees(x):+.2f}" for x, p in zip(th, robot.is_prismatic)]
                    + [f"{res.err_w[i]:.2e}", f"{res.err_v[i]:.2e}"])
    return np.array(rows, dtype=object)


def self_motion_step(robot, th, sm):
    """One null-space step of the self-motion demo, corrected back onto the task pose T0 (state kept in `sm`).
    Returns the new joint vector, or None when no step was taken (null space vanished / joint limit reached)."""
    task, T0 = sm["task"], sm["T0"]
    _, J = _task_error(robot, th, T0, task)
    if sm["dir"] is None:                                       # null vector of J from the SVD
        sm["dir"] = np.linalg.svd(J)[2][-1]
    if task == "pose":
        dth = kin.null_space_step(robot, th, sm["dir"])        # (I − J_b⁺ J_b) z
    else:
        dth = (np.eye(robot.n) - np.linalg.pinv(J) @ J) @ sm["dir"]
    nrm = np.linalg.norm(dth)
    if nrm < 1e-9:
        sm["dir"] = None
        return None
    dth *= np.radians(0.8) / nrm                               # 0.8° of joint motion per frame
    th1 = th + dth
    e1, _ = _task_error(robot, th1, T0, task)
    sm["drift1"] = float(np.linalg.norm(e1))                   # drift of the pure null-space step
    for _ in range(3):                                         # Newton correction back onto T0
        e, Jc = _task_error(robot, th1, T0, task)
        if np.linalg.norm(e) < 1e-14:
            break
        th1 = th1 + np.linalg.pinv(Jc) @ e
    lo, hi = robot.joint_limits[:, 0], robot.joint_limits[:, 1]
    inside = (th >= lo) & (th <= hi)                           # only guard joints that start feasible
    if np.any(inside & ((th1 < lo + 1e-3) | (th1 > hi - 1e-3))):   # joint limit → reverse
        sm["dir"] = -dth / np.linalg.norm(dth)
        return None
    sm["dir"] = dth / np.linalg.norm(dth)
    sm["steps"] += 1
    return th1


def target_row_values(T) -> dict:
    """Slider values (x, y, z, roll, pitch, yaw, φ) describing the pose T."""
    p = T[:3, 3]
    R = T[:3, :3]
    vals = dict(x=p[0], y=p[1], z=p[2])
    rpy = kin.rot_to_rpy(R)
    vals.update(roll=rpy[0], pitch=rpy[1], yaw=rpy[2], **{"φ": np.arctan2(R[1, 0], R[0, 0])})
    return vals


def pose_from_rows(v: dict, T_target):
    """Target pose from slider values; orientation is kept from T_target for position-only tasks."""
    p = [v["x"], v["y"], v.get("z", 0.0)]
    if "roll" in v:
        R = kin.rpy_to_rot([v["roll"], v["pitch"], v["yaw"]])
    elif "φ" in v:
        R = _rotz(v["φ"])
    else:
        R = T_target[:3, :3]
    return trans(p, R)
