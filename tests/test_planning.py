"""Tests for mrstudio.planning (MR Ch. 10) and the Motion Planning tab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from mrstudio import planning as P  # noqa: E402
from mrstudio.robots import get_robot  # noqa: E402

MR_SCENE = "MR Fig. 10.2-style (three obstacles)"


def world(robot="Planar 2R arm", scene=MR_SCENE, **kw):
    arm = P.PlanarArm.from_robot(get_robot(robot))
    obs, s, g = P.scene(scene, arm)
    return P.World(arm, obs, **kw), s, g


def test_geometry_primitives():
    A, B = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert P._point_segment_dist(np.array([0.5, 2.0]), A, B) == pytest.approx(2.0)
    assert P._point_segment_dist(np.array([3.0, 0.0]), A, B) == pytest.approx(2.0)
    assert P._segment_segment_dist(A, B, np.array([0.5, -1.0]), np.array([0.5, 1.0])) == 0.0
    sq = P.box(0, 0, 1, 1)
    assert P._point_in_polygon(np.array([[0.5, 0.5], [1.5, 0.5]]), sq.pts).tolist() == [True, False]
    # segment fully inside a polygon has distance 0
    assert P.segment_obstacle_dist(np.array([[0.2, 0.2]]), np.array([[0.8, 0.8]]), sq)[0] == 0.0
    assert P.segment_obstacle_dist(np.array([[2.0, 0.0]]), np.array([[2.0, 1.0]]), P.Circle([0, 0.5], 0.5))[0] \
        == pytest.approx(1.5)


def test_planar_arm_matches_poe():
    import modern_robotics as mr
    rob = get_robot("Planar 3R arm")
    arm = P.PlanarArm.from_robot(rob)
    th = np.array([0.3, -1.1, 2.0])
    assert np.allclose(arm.points(th)[0, -1], mr.FKinSpace(rob.M, rob.Slist, th)[:2, 3], atol=1e-12)
    with pytest.raises(ValueError):
        P.PlanarArm.from_robot(get_robot("UR5 (6R industrial)"))


def test_torus_metric():
    w, _, _ = world()
    assert w.space.wrap.all()
    a, b = np.radians([170, 0]), np.radians([-170, 0])
    assert w.space.dist(a, b) == pytest.approx(np.radians(20))


def test_collision_against_reference_sampler():
    w, _, _ = world(scene="EOD vehicle clutter (UGV, side view)")
    Q = w.space.sample(np.random.default_rng(0), 2000)
    ref, h = P._reference_collision(w, Q)
    bad = (w.collides(Q) != ref) & (np.abs(w.link_clearance(Q)) > h)
    assert not bad.any()


@pytest.mark.parametrize("scene", list(P.SCENES))
@pytest.mark.parametrize("robot", ["Planar 2R arm", "Planar 3R arm"])
def test_presets_free_and_solvable(robot, scene):
    w, s, g = world(robot, scene)
    assert w.free(s) and w.free(g)
    r = P.plan(w, s, g, "A* (grid)", grid_k=64 if w.space.n == 2 else 20)
    assert r.success, r.message


def test_astar_equals_dijkstra_and_expands_less():
    w, s, g = world(scene="Narrow passage")
    grid = P.cspace_grid(w, 48)
    a = P.grid_search(w, grid, s, g, "astar")
    d = P.grid_search(w, grid, s, g, "dijkstra")
    assert a.success and d.success
    assert a.extra["grid_cost"] == pytest.approx(d.extra["grid_cost"], abs=1e-9)
    assert a.n_nodes <= d.n_nodes
    a4 = P.grid_search(w, grid, s, g, "astar", connect="4")
    assert a4.extra["grid_cost"] >= a.extra["grid_cost"] - 1e-12   # fewer moves cannot be cheaper


@pytest.mark.parametrize("planner", ["RRT", "RRT*", "PRM"])
def test_sampling_planners_collision_free(planner):
    w, s, g = world()
    r = P.plan(w, s, g, planner, max_iter=1500 if planner == "RRT*" else 4000, seed=1, smooth_iters=100)
    assert r.success, r.message
    assert np.allclose(r.path[0], s) and w.space.dist(r.path[-1], g) < 1e-9
    assert P.dense_min_clearance(w, r.path) > 0          # independent re-check every 0.05°
    assert P.dense_min_clearance(w, r.extra["raw_path"]) > 0
    assert r.cost <= r.raw_cost + 1e-12
    assert r.edges is not None and r.n_nodes > 2


def test_collinear_disjoint_segments_not_a_collision():
    # regression: a non-strict orientation test reported collinear but disjoint segments as intersecting
    A, B = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert P._segment_segment_dist(A, B, np.array([2.0, 0.0]), np.array([3.0, 0.0])) == pytest.approx(1.0)
    assert P._segment_segment_dist(A, B, np.array([1.0, 0.0]), np.array([3.0, 0.0])) == 0.0   # touching
    arm = P.PlanarArm.from_robot(get_robot("Planar 2R arm"))
    w = P.World(arm, [P.box(1.5, 0.0, 2.0, 0.5)])
    assert w.free(np.zeros(2))


def test_certified_edges_never_optimistic():
    w, _, _ = world(scene="Narrow passage")
    rng = np.random.default_rng(5)
    A = w.space.sample(rng, 300)
    B = A + rng.normal(0.0, 0.5, A.shape)
    ok = w.edges_free(A, B)
    assert 0 < ok.sum() < len(A)
    for a, b in zip(A[ok], B[ok]):
        assert P.dense_min_clearance(w, np.vstack([a, b]), np.radians(0.02)) > 0
    # thin wall: an edge whose end samples are free but which crosses the wall is blocked
    arm = P.PlanarArm.from_robot(get_robot("Planar 2R arm"))
    wall = P.World(arm, [P.box(0.70, -0.004, 1.2, 0.004)], edge_res=np.radians(30))
    a, b = np.radians([-10.0, 0.0]), np.radians([10.0, 0.0])
    assert wall.free(a) and wall.free(b) and not wall.edge_free(a, b)


@pytest.mark.parametrize("robot", ["Planar 2R arm", "Planar 3R arm"])
def test_grid_paths_do_not_cut_corners(robot):
    # regression: grid paths used to go through C-obstacle corners (unchecked diagonal / end moves)
    for scene in P.SCENES:
        w, s, g = world(robot, scene)
        r = P.plan(w, s, g, "A* (grid)", grid_k=64 if w.space.n == 2 else 16)
        assert r.success, (scene, r.message)
        assert P.dense_min_clearance(w, r.path) > 0, scene


def test_grid_cost_matches_scipy():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    w, s, g = world(scene="Patient-side obstacles (rehab, top view)")
    grid = P.cspace_grid(w, 40)
    nb = P._grid_neighbors(grid, w.space, "8")
    rows, cols, vals = [], [], []
    for u in np.flatnonzero(~grid.occ.ravel()).tolist():
        for v, c in nb(u):
            rows.append(u), cols.append(v), vals.append(c)
    G = csr_matrix((vals, (rows, cols)), shape=(grid.occ.size,) * 2)
    for method in ("astar", "dijkstra"):
        r = P.grid_search(w, grid, s, g, method)
        cells = r.extra["cells"]
        assert r.extra["grid_cost"] == pytest.approx(dijkstra(G, indices=cells[0])[cells[-1]], abs=1e-9)


def test_all_planners_on_all_planar_robots_and_presets():
    for name in P.planning_robot_names():
        arm = P.PlanarArm.from_robot(get_robot(name))
        for scene in P.SCENES:
            obs, s, g = P.scene(scene, arm)
            w = P.World(arm, obs)
            for pl in ("A* (grid)", "PRM", "RRT"):
                r = P.plan(w, s, g, pl, grid_k=48 if arm.n == 2 else 14, seed=0)
                assert r.success, (name, scene, pl, r.message)


def test_rrt_star_not_worse_than_rrt_on_average():
    w, s, g = world()
    c_rrt = [P.plan(w, s, g, "RRT", seed=k).cost for k in range(3)]
    c_star = [P.plan(w, s, g, "RRT*", max_iter=1500, seed=k).cost for k in range(3)]
    assert np.mean(c_star) <= np.mean(c_rrt)


def test_shortcut_monotone():
    w, s, g = world()
    r = P.plan(w, s, g, "RRT", seed=4)
    L = [w.path_length(r.path)]
    p = r.path
    for k in range(5):
        p = P.shortcut(w, p, 20, seed=k)
        L.append(w.path_length(p))
        assert P.dense_min_clearance(w, p) > 0
    assert all(b <= a + 1e-12 for a, b in zip(L, L[1:]))


def test_start_in_collision_reported():
    w, s, g = world()
    bad = np.radians([0.0, 0.0])
    while w.free(bad):     # find a colliding configuration
        bad = w.space.sample(np.random.default_rng(3), 1)[0]
    r = P.plan(w, bad, g, "RRT")
    assert not r.success and "collision" in r.message


@pytest.mark.parametrize("rest", [False, True])
def test_time_scaling(rest):
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    t, Q, dQ, ddQ = P.time_scale(path, 2.0, "quintic", 0.01, rest)
    assert np.allclose(Q[0], path[0]) and np.allclose(Q[-1], path[-1])
    assert np.allclose(dQ[0], 0) and np.allclose(dQ[-1], 0)
    if rest:   # stops at the via point
        k = np.argmin(np.linalg.norm(Q - path[1], axis=1))
        assert np.linalg.norm(dQ[k]) < 0.05


def test_course_robots_if_registered():
    for name in ("Arm–exoskeleton 2R", "Rehab robot 2R (planar)", "EOD arm (3R planar)"):
        try:
            rob = get_robot(name)
        except KeyError:
            continue
        assert name in P.planning_robot_names()
        arm = P.PlanarArm.from_robot(rob)
        for scn in ("Patient-side obstacles (rehab, top view)", "EOD vehicle clutter (UGV, side view)"):
            obs, s, g = P.scene(scn, arm)
            w = P.World(arm, obs)
            assert w.free(s) and w.free(g)
            assert w.space.inside(s) and w.space.inside(g)


def test_validation_checks_pass():
    checks = P.validation_checks()
    assert len(checks) >= 15
    failed = [c.name for c in checks if not c.passed]
    assert not failed, failed


# ---------------------------------------------------------------------------- GUI
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from mrstudio.gui import theme
    theme.apply(app, False)
    return app


def test_planning_tab_headless(qapp):
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.planning_tab import THEORY, PlanningTab
    from mrstudio.gui.theory import validate_all
    assert validate_all([THEORY]) == []
    ctx = AppContext()                      # UR5 selected -> the tab falls back to a planar arm
    tab = PlanningTab(ctx)
    assert tab.arm is not None and tab.arm.n in (2, 3)
    for planner in P.PLANNERS:
        tab.planner_box.setCurrentText(planner)
        if planner == "RRT*":
            tab.iter_spin.setValue(800)
        tab.plan()
        tab.wait_job()
        assert tab.result is not None and tab.result.success, (planner, tab.result.message)
        assert tab.traj is not None and tab.anim_btn.isEnabled()
    assert tab.hist.rowCount() == len(P.PLANNERS)
    # animation step
    tab.anim_btn.setChecked(True)
    tab._anim_step()
    tab.anim_btn.setChecked(False)
    # edit the obstacles -> plan is invalidated
    n = len(tab.rois)
    tab._add_obstacle("circle")
    assert len(tab.rois) == n + 1 and tab.result is None
    tab._remove_last()
    # global robot -> planar 3R is adopted
    ctx.set_robot("Planar 3R arm")
    assert tab.arm.n == 3
    tab.planner_box.setCurrentText("A* (grid)")
    tab.plan()
    tab.wait_job()
    assert tab.result.success
    # every preset scene loads for the 3R
    for i in range(tab.scene_box.count()):
        tab.scene_box.setCurrentIndex(i)
        assert len(tab.start_sl.values()) == 3
    tab.on_hide()
