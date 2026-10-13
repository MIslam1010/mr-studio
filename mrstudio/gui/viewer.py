"""OpenGL 3-D scene (pyqtgraph.opengl) with a small keyed scene-graph API.

Every drawable is addressed by a string key, so tabs can call e.g.
``view.set_tube("leg3", a, b, 0.01, color)`` every animation frame: the first call
creates the item, later calls only update its transform / colour (cheap).
"""
from __future__ import annotations

import weakref

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui

from ..robots import SerialRobot
from . import theme

_CYL = gl.MeshData.cylinder(rows=1, cols=24, radius=[1.0, 1.0], length=1.0)
_CONE = gl.MeshData.cylinder(rows=1, cols=24, radius=[1.0, 0.0], length=1.0)
_SPHERE = gl.MeshData.sphere(rows=14, cols=20, radius=1.0)
_DISK = gl.MeshData.cylinder(rows=1, cols=32, radius=[1.0, 1.0], length=1.0)


def _rot_z_to(d: np.ndarray) -> np.ndarray:
    """Rotation matrix taking the z axis onto the unit vector d."""
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(z @ d)
    if np.linalg.norm(v) < 1e-10 or c < -1.0 + 1e-9:  # (anti)parallel to z: avoid 1/(1+c) blow-up
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _qmat(T: np.ndarray) -> QtGui.QMatrix4x4:
    return QtGui.QMatrix4x4(*np.asarray(T, float).ravel().tolist())


def _rgba(c):
    if isinstance(c, str):
        q = QtGui.QColor(c)
        return (q.redF(), q.greenF(), q.blueF(), q.alphaF())
    c = tuple(float(x) for x in c)
    return c if len(c) == 4 else (*c, 1.0)


class SceneView(gl.GLViewWidget):
    """3-D view with keyed primitives."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(theme.VIEW_BG)
        self.items_by_key: dict[str, list] = {}
        self.grid = gl.GLGridItem()
        self.grid.setColor((255, 255, 255, 28))
        self.addItem(self.grid)
        self.setMinimumSize(320, 260)
        self._planar = False
        self._graphics: dict[str, weakref.ref] = {}  # key -> RobotGraphic (re-drawn on projector toggle)
        self.frame_scene(1.0)
        theme.notifier().projector_changed.connect(self._on_projector)
        if theme.is_projector():
            self._on_projector(True)

    # ------------------------------------------------------------------ projector mode
    def _on_projector(self, on: bool):
        """Restyle existing items for projector mode (thicker lines, larger labels/points, links)."""
        try:
            self.setBackgroundColor(theme.view_bg())
        except RuntimeError:  # widget already deleted
            return
        self.grid.setColor((255, 255, 255, 55 if on else 28))
        ls, fs = theme.line_scale(), theme.font_scale()
        for items in self.items_by_key.values():
            for it in items:
                if isinstance(it, gl.GLLinePlotItem) and hasattr(it, "_mr_w"):
                    it.setData(width=it._mr_w * ls)
                elif isinstance(it, gl.GLScatterPlotItem) and hasattr(it, "_mr_size"):
                    it.setData(size=it._mr_size * ls)
                elif isinstance(it, gl.GLTextItem) and hasattr(it, "_mr_font"):
                    it.setData(font=QtGui.QFont("Helvetica", int(round(it._mr_font * fs))))
        for ref in list(self._graphics.values()):
            g = ref()
            if g is not None and g.view is self and g.last_theta is not None:
                try:
                    g.refresh()
                except Exception:  # never break the toggle because of one drawable
                    pass
        self.update()

    def grab_frame(self) -> QtGui.QImage:
        """Rendered 3-D view incl. text labels (device pixels)."""
        return self.grabFramebuffer()

    # ------------------------------------------------------------------ camera
    def frame_scene(self, extent: float, planar: bool = False, center=(0, 0, 0)):
        self._planar = planar
        ext = max(extent, 0.05)
        s = 10 ** np.floor(np.log10(ext / 2))
        self.grid.setSize(4 * ext, 4 * ext, 1)
        self.grid.setSpacing(s, s, s)
        self.opts["center"] = pg.Vector(*center)
        if planar:
            self.setCameraPosition(distance=2.5 * ext, elevation=90, azimuth=-90)
        else:
            self.setCameraPosition(distance=2.7 * ext, elevation=22, azimuth=-55)

    # ------------------------------------------------------------------ bookkeeping
    def _get(self, key):
        return self.items_by_key.get(key)

    def _put(self, key, items):
        self.items_by_key[key] = items
        for it in items:
            self.addItem(it)

    def remove(self, key: str):
        for it in self.items_by_key.pop(key, []):
            self.removeItem(it)

    def remove_prefix(self, prefix: str):
        for k in [k for k in self.items_by_key if k.startswith(prefix)]:
            self.remove(k)

    def set_visible(self, key: str, visible: bool):
        for it in self.items_by_key.get(key, []):
            it.setVisible(visible)

    def set_visible_prefix(self, prefix: str, visible: bool):
        for k, items in self.items_by_key.items():
            if k.startswith(prefix):
                for it in items:
                    it.setVisible(visible)

    def has(self, key: str) -> bool:
        return key in self.items_by_key

    # ------------------------------------------------------------------ primitives
    def _mesh(self, key, md, T, color, translucent=False):
        items = self._get(key)
        rgba = _rgba(color)
        if items is None:
            it = gl.GLMeshItem(meshdata=md, smooth=True, shader="shaded", color=rgba,
                               glOptions="translucent" if (translucent or rgba[3] < 1) else "opaque")
            self._put(key, [it])
        else:
            it = items[0]
            it.setColor(rgba)
            it.setVisible(True)
        it.setTransform(_qmat(T))
        return it

    def set_tube(self, key, a, b, radius, color, translucent=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        T = np.eye(4)
        T[:3, :3] = _rot_z_to(d) @ np.diag([radius, radius, max(L, 1e-6)])
        T[:3, 3] = a
        return self._mesh(key, _CYL, T, color, translucent)

    def set_sphere(self, key, c, radius, color, translucent=False):
        T = np.diag([radius, radius, radius, 1.0])
        T[:3, 3] = c
        return self._mesh(key, _SPHERE, T, color, translucent)

    def set_disk(self, key, c, normal, radius, thickness, color):
        T = np.eye(4)
        T[:3, :3] = _rot_z_to(np.asarray(normal, float)) @ np.diag([radius, radius, thickness])
        T[:3, 3] = np.asarray(c) - T[:3, :3] @ np.array([0, 0, 0.5])
        return self._mesh(key, _DISK, T, color)

    def set_ellipsoid(self, key, center, axes, radii, color):
        T = np.eye(4)
        T[:3, :3] = np.asarray(axes) @ np.diag(np.maximum(np.asarray(radii, float), 1e-4))
        T[:3, 3] = center
        return self._mesh(key, _SPHERE, T, color, translucent=True)

    def set_arrow(self, key, start, vec, color, radius=0.01, head=0.25):
        start, vec = np.asarray(start, float), np.asarray(vec, float)
        L = float(np.linalg.norm(vec))
        if L < 1e-9:
            self.set_visible(key + ".shaft", False)
            self.set_visible(key + ".head", False)
            return
        hl = min(head * L, 6 * radius) if L > 6 * radius else 0.4 * L
        tip_base = start + vec * (1 - hl / L)
        self.set_tube(key + ".shaft", start, tip_base, radius, color)
        R = _rot_z_to(vec)
        T = np.eye(4)
        T[:3, :3] = R @ np.diag([2.2 * radius, 2.2 * radius, hl])
        T[:3, 3] = tip_base
        self._mesh(key + ".head", _CONE, T, color)

    def set_polyline(self, key, pts, color=(1, 1, 1, 1), width=2.0, mode="line_strip"):
        pts = np.asarray(pts, float).reshape(-1, 3)
        items = self._get(key)
        col = color if (isinstance(color, np.ndarray) and color.ndim == 2) else _rgba(color)
        w = width * theme.line_scale()
        if items is None:
            it = gl.GLLinePlotItem(pos=pts, color=col, width=w, antialias=True, mode=mode)
            it.setGLOptions("translucent")
            self._put(key, [it])
        else:
            it = items[0]
            it.setData(pos=pts, color=col, width=w, mode=mode)
            it.setVisible(True)
        it._mr_w = width
        return it

    def set_points(self, key, pts, color=(1, 1, 1, 1), size=6.0):
        pts = np.asarray(pts, float).reshape(-1, 3)
        items = self._get(key)
        col = color if (isinstance(color, np.ndarray) and color.ndim == 2) else _rgba(color)
        sz = size * theme.line_scale() if np.isscalar(size) else np.asarray(size) * theme.line_scale()
        if items is None:
            it = gl.GLScatterPlotItem(pos=pts, color=col, size=sz, pxMode=True)
            it.setGLOptions("translucent")
            self._put(key, [it])
        else:
            it = items[0]
            it.setData(pos=pts, color=col, size=sz)
            it.setVisible(True)
        it._mr_size = size
        return it

    def set_label(self, key, pos, text, color="#e8e8e8", size=11):
        items = self._get(key)
        if items is None:
            font = QtGui.QFont("Helvetica", int(round(size * theme.font_scale())))
            it = gl.GLTextItem(pos=np.asarray(pos, float), text=text, color=QtGui.QColor(color), font=font)
            it._mr_font = size
            self._put(key, [it])
        else:
            it = items[0]
            if getattr(it, "_mr_font", size) != size:
                it._mr_font = size
                it.setData(font=QtGui.QFont("Helvetica", int(round(size * theme.font_scale()))))
            it.setData(pos=np.asarray(pos, float), text=text, color=QtGui.QColor(color))
            it.setVisible(True)
        return it

    def set_triad(self, key, T, size=0.1, label: str | None = None, width=3.0, alpha=1.0):
        T = np.asarray(T, float)
        o = T[:3, 3]
        pts = np.array([o, o + size * T[:3, 0], o, o + size * T[:3, 1], o, o + size * T[:3, 2]])
        cols = np.array([[1, .3, .3, alpha]] * 2 + [[.3, 1, .3, alpha]] * 2 + [[.35, .55, 1, alpha]] * 2)
        self.set_polyline(key, pts, cols, width=width, mode="lines")
        r = size * 0.06
        self.set_tube(key + ".x", o, o + size * T[:3, 0], r, (1, .3, .3, alpha))
        self.set_tube(key + ".y", o, o + size * T[:3, 1], r, (.3, 1, .3, alpha))
        self.set_tube(key + ".z", o, o + size * T[:3, 2], r, (.35, .55, 1, alpha))
        if label:
            self.set_label(key + ".lbl", o + size * 0.15 * (T[:3, 0] + T[:3, 1] + T[:3, 2]) - size * 0.25 * T[:3, 2],
                           label, "#f0f0f0")

    def remove_triad(self, key):
        self.remove_prefix(key)


class RobotGraphic:
    """Draws a SerialRobot (links = tubes along the PoE skeleton, joints = cylinders along screw axes)."""

    def __init__(self, view: SceneView, robot: SerialRobot, key: str = "robot", alpha: float = 1.0,
                 link_color=theme.LINK, joint_color=theme.JOINT, show_ee_frame=True, scale=1.0):
        self.view, self.robot, self.key, self.alpha = view, robot, key, alpha
        self.link_color = (*_rgba(link_color)[:3], alpha)
        self.joint_color = (*_rgba(joint_color)[:3], alpha)
        self.r = robot.link_radius * scale
        self.show_ee_frame = show_ee_frame
        self.show_axes = False
        self.show_labels = False
        self.geom = None
        self.last_theta = None
        try:
            view._graphics[key] = weakref.ref(self)
        except (AttributeError, TypeError):
            pass

    def refresh(self):
        """Redraw at the last configuration (used when projector mode changes the link radius)."""
        if self.last_theta is not None:
            self.view.remove(f"{self.key}.base")
            self.update(self.last_theta)

    def update(self, theta):
        v, r, k = self.view, self.r * theme.link_scale(), self.key
        self.last_theta = np.array(theta, float)
        g = self.robot.geometry(np.asarray(theta, float))
        self.geom = g
        tr = self.alpha < 1
        for i, P in enumerate(g["polys"]):
            for j in range(len(P) - 1):
                if np.linalg.norm(P[j + 1] - P[j]) > 1e-6:
                    v.set_tube(f"{k}.l{i}.{j}", P[j], P[j + 1], r if i else 1.25 * r,
                               self.link_color if i else (*_rgba(theme.BASE)[:3], self.alpha), tr)
                else:
                    v.set_visible(f"{k}.l{i}.{j}", False)
                v.set_sphere(f"{k}.s{i}.{j}", P[j + 1], r, self.link_color, tr)
        if not v.has(f"{k}.base") and self.alpha >= 1:
            base_n = np.array([0, 0, 1.0])
            v.set_disk(f"{k}.base", [0, 0, 0], base_n, 3.2 * r, 0.6 * r, theme.BASE)
        for i in range(self.robot.n):
            p, d = g["joint_pos"][i], g["joint_dir"][i]
            h = 1.5 * r
            v.set_tube(f"{k}.j{i}", p - h * d, p + h * d, 1.55 * r, self.joint_color, tr)
            if self.show_labels:
                v.set_label(f"{k}.jl{i}", p + 2.5 * r * np.array([0, 0, 1.0]), f"{i + 1}", theme.JOINT)
            else:
                v.set_visible(f"{k}.jl{i}", False)
            if self.show_axes:
                ext = 0.6 * self.robot.reach
                v.set_polyline(f"{k}.ax{i}", [p - ext * d, p + ext * d], (1.0, 0.85, 0.2, 0.55), 1.5)
            else:
                v.set_visible(f"{k}.ax{i}", False)
        if self.show_ee_frame:
            v.set_triad(f"{k}.ee", g["T_ee"], 0.18 * self.robot.reach, "{b}" if self.alpha >= 1 else None,
                        alpha=self.alpha)
        return g

    def set_visible(self, vis: bool):
        self.view.set_visible_prefix(self.key + ".", vis)

    def remove(self):
        self.view.remove_prefix(self.key + ".")


class GhostRobot:
    """Light-weight translucent stick figure (for IK iterates, multiple solutions, desired motion)."""

    def __init__(self, view: SceneView, robot: SerialRobot, key: str, color=(0.6, 0.8, 1.0, 0.45), width=3.0):
        self.view, self.robot, self.key, self.color, self.width = view, robot, key, color, width

    def update(self, theta):
        g = self.robot.geometry(np.asarray(theta, float))
        pts = np.vstack(g["polys"])
        self.view.set_polyline(self.key + ".line", pts, self.color, self.width)
        self.view.set_points(self.key + ".pts", g["joint_pos"], self.color, 7)
        return g

    def remove(self):
        self.view.remove_prefix(self.key + ".")
