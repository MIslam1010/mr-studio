"""Links between MR Studio and the Robotics Knowledge Map.

Two directions:

* **MR Studio -> map.** Every tab's theory panel ends with a "Knowledge Map" footer (``theory_footer``) that links
  up to eight map nodes about the tab's topic. ``map_url(node_id)`` builds the deep link
  ``<map>#map/<node id>/<map tab>``; the map itself is the local student edition in the folder named by the
  environment variable ``MRSTUDIO_KMAP_DIR`` if that is set and holds one, else the published artifact
  (``MRSTUDIO_KMAP_URL`` overrides both).  There is no built-in local path.
* **Map -> MR Studio.** The map can open MR Studio on the right tab with
  ``mrstudio://open?tab=<class or 0-based index>&robot=<name>&preset=<key>``; ``parse_launch_url`` reads such a
  URL and ``launch_url`` writes one.

Which tab belongs to which map node is derived, not hand-listed: ``tools/make_kmap_links.py`` reads the map's
node data (``kmap-data/a_*.js``), takes each node's Modern Robotics reference (``§x.y``), looks the section up in
``SECTION_TABS`` (longest prefix wins) and writes ``mrstudio/data/kmap/mrstudio_targets.json``. That JSON is read
here (footer) and by the map build (the "Open in MR Studio" button). Contract: ``docs/KMAP-LINKING.md``.
"""
from __future__ import annotations

import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit

#: tab classes in display order (0-based index = MR Studio's ``--tab`` index). Must match ``gui.app.TABS``
#: (tests/test_kmap_links.py checks it); GraspingTab / MobileTab are the Ch. 12 / Ch. 13 tabs after Haptics.
TAB_ORDER = [
    "FKTab", "VelocityTab", "IKTab", "DynamicsTab", "ControlTab", "ClosedChainTab", "StewartTab",
    "AdvancedControlTab", "ParallelRobotsTab", "PlanningTab", "HapticsTab", "GraspingTab", "MobileTab",
    "HardwareTab", "PracticeTab", "ValidationTab",
]

#: long names for the tabs that ``gui.app.TAB_LONG_NAMES`` may not list yet (used only when app.py lacks them)
_EXTRA_LONG_NAMES = {"GraspingTab": "Grasping & manipulation", "MobileTab": "Wheeled mobile robots"}

UR5 = "UR5 (6R industrial)"
P2R = "Planar 2R arm"
P3R = "Planar 3R arm"
COURSE_ARM = "Course arm (myCobot-280 class 6R)"

#: Modern Robotics section prefix -> (tab class, default robot or None, preset key or None).
#: The longest matching prefix wins ("11.4.3" before "11.4" before "11"). Robots are names in ``robots.ROBOTS``
#: (only for tabs that use the serial-robot selector); presets are keys the tab's own preset box knows.
SECTION_TABS: dict[str, tuple[str, str | None, str | None]] = {
    # Ch. 2 configuration space
    "2": ("FKTab", P3R, None),
    "2.2": ("ClosedChainTab", None, "Crank-rocker"),             # Grübler's formula, four-bar example
    "2.4": ("ClosedChainTab", None, "Crank-rocker"),             # loop-closure (holonomic) constraints
    # Ch. 3 rigid-body motions
    "3": ("FKTab", UR5, None),
    "3.1": ("FKTab", P3R, None),                                  # rigid bodies in the plane
    "3.4": ("VelocityTab", UR5, None),                            # wrenches
    # Ch. 4 forward kinematics
    "4": ("FKTab", UR5, None),
    "4.2": ("FKTab", "KUKA iiwa 7 (7R, URDF)", None),             # URDF
    # Ch. 5 velocity kinematics and statics
    "5": ("VelocityTab", UR5, None),
    # Ch. 6 inverse kinematics
    "6": ("IKTab", UR5, None),
    "6.1": ("IKTab", P2R, None),                                  # analytic IK (planar arms in the tab)
    # Ch. 7 closed chains
    "7": ("ClosedChainTab", None, "Crank-rocker"),
    "7.1": ("ParallelRobotsTab", None, None),                     # 3×RPR, Stewart–Gough, Delta FK/IK
    "7.1.2": ("StewartTab", None, None),
    "7.2": ("StewartTab", None, None),                            # differential kinematics (Stewart–Gough)
    "7.3": ("ParallelRobotsTab", None, None),                     # singularities
    # Ch. 8 dynamics
    "8": ("DynamicsTab", UR5, None),
    "8.1": ("DynamicsTab", P2R, None),                            # Lagrangian formulation, 2R example
    "8.6": ("AdvancedControlTab", UR5, None),                     # dynamics in the task space
    "8.7": ("ClosedChainTab", None, "Crank-rocker"),              # constrained dynamics
    "8.9": ("HardwareTab", None, None),                           # actuation, gearing and friction
    # Ch. 9 trajectory generation
    "9": ("ControlTab", UR5, None),
    # Ch. 10 motion planning
    "10": ("PlanningTab", P2R, None),
    # Ch. 11 robot control
    "11": ("ControlTab", UR5, None),
    "11.4.3": ("AdvancedControlTab", UR5, None),                  # task-space motion control
    "11.5": ("AdvancedControlTab", UR5, None),                    # force control
    "11.6": ("AdvancedControlTab", UR5, None),                    # hybrid motion-force control
    "11.7": ("HapticsTab", None, "Virtual wall — hand pushes in"),  # impedance / admittance control
    "11.8": ("HardwareTab", None, None),                          # low-level joint force/torque control
    # Ch. 12 grasping and manipulation, Ch. 13 wheeled mobile robots
    "12": ("GraspingTab", None, None),
    "13": ("MobileTab", None, None),
}

#: the map's own tab opened by a deep link (Learn | Practise | Context | Sources)
MAP_TAB = "learn"
ARTIFACT_URL = "https://claude.ai/artifact/Qd3y25kCzuRo4ChtS8gsW5"
KMAP_DIR_ENV = "MRSTUDIO_KMAP_DIR"     # folder of a local Knowledge Map copy (no default: unset = artifact)
TARGETS_JSON = Path(__file__).resolve().parent / "data" / "kmap" / "mrstudio_targets.json"
SCHEME = "mrstudio"
SCHEME_TEMPLATE = "mrstudio://open?tab={tab}&robot={robot}&preset={preset}"
FOOTER_MAX = 8
NO_LINKS_ENV = "MRSTUDIO_NO_KMAP_LINKS"


# ----------------------------------------------------------------------------- MR section -> tab
def tab_for_section(section: str) -> tuple[str, str | None, str | None] | None:
    """(tab class, robot, preset) for an MR section such as ``"11.4.3"`` (longest prefix of SECTION_TABS)."""
    parts = section.split(".")
    for k in range(len(parts), 0, -1):
        hit = SECTION_TABS.get(".".join(parts[:k]))
        if hit:
            return hit
    return None


_SEC_RE = re.compile(r"§\s*(\d+(?:\.\d+)*)")
_CH_RE = re.compile(r"\b(?:Ch\.|Chapter|Example|Exercise|Fig\.|Figs\.|Eq\.|Eqs\.)\s*\(?(\d+)[.\s,)]")


def mr_section(where: str) -> str | None:
    """MR section of a reference's ``where`` text: ``"§4.1, Eq. (4.14), p. 140"`` -> ``"4.1"``.

    Without a ``§``, a chapter is taken from "Ch. 12 introduction", "Example 12.2", "Eq. (8.9)" and the like
    (-> ``"12"``, ``"8"``); appendices and front matter give None."""
    m = _SEC_RE.search(where)
    if m:
        return m.group(1)
    if re.match(r"\s*(App\.|Appendix|Foreword|Preface)", where):
        return None
    m = _CH_RE.search(where)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------- map URLs
def kmap_dir() -> Path | None:
    """Folder of the local Knowledge Map (``MRSTUDIO_KMAP_DIR``); None when the variable is unset or empty."""
    env = os.environ.get(KMAP_DIR_ENV, "").strip()
    return Path(env).expanduser() if env else None


def map_base_url() -> str:
    """URL of the map page: ``MRSTUDIO_KMAP_URL``, else the local student edition, else the published artifact."""
    env = os.environ.get("MRSTUDIO_KMAP_URL", "").strip()
    if env:
        return env.split("#", 1)[0]
    d = kmap_dir()
    local = None if d is None else d / "Robotics Knowledge Map - Student Edition" / "index.html"
    if local is not None and local.is_file():
        return local.resolve().as_uri()
    return ARTIFACT_URL


def map_url(node_id: str, tab: str = MAP_TAB) -> str:
    """Deep link that opens the map on ``node_id`` (e.g. ``"04.01.1"``) and its ``tab``."""
    return f"{map_base_url()}#map/{quote(node_id, safe='.')}/{tab}"


# ----------------------------------------------------------------------------- mrstudio:// launch URLs
def launch_url(tab: str | int, robot: str | None = None, preset: str | None = None) -> str:
    """``mrstudio://open?tab=…&robot=…&preset=…`` (empty parts are left out)."""
    q = {"tab": str(tab)}
    if robot:
        q["robot"] = robot
    if preset:
        q["preset"] = preset
    return f"{SCHEME}://open?" + urlencode(q, quote_via=quote)


def parse_launch_url(url: str) -> dict:
    """Read ``mrstudio://open?tab=<class or 0-based index>&robot=<name>&preset=<key>``.

    Returns ``{"action": "open", "tab": class name or None, "index": 0-based index or None, "robot": str or None,
    "preset": str or None}``. An unknown tab class or an index out of range gives ``tab = index = None`` (the
    caller then keeps the current tab); robot and preset are passed through for the caller to check.
    Raises ``ValueError`` for another scheme or action."""
    s = urlsplit(url.strip())
    if s.scheme.lower() != SCHEME:
        raise ValueError(f"not an {SCHEME}:// URL: {url!r}")
    action = (s.netloc or s.path.strip("/").split("/", 1)[0] or "open").lower()
    if action != "open":
        raise ValueError(f"unknown {SCHEME}:// action {action!r}")
    q = {k: v[0].strip() for k, v in parse_qs(s.query, keep_blank_values=True).items() if v}
    tab, index = None, None
    t = q.get("tab", "")
    if re.fullmatch(r"\d+", t):
        i = int(t)
        if i < len(TAB_ORDER):
            tab, index = TAB_ORDER[i], i
    elif t in TAB_ORDER:
        tab, index = t, TAB_ORDER.index(t)
    return {"action": "open", "tab": tab, "index": index,
            "robot": q.get("robot") or None, "preset": q.get("preset") or None}


# ----------------------------------------------------------------------------- generated targets + footer
@lru_cache(maxsize=1)
def load_targets(path: str | None = None) -> dict:
    """The generated ``mrstudio_targets.json`` ({} if missing or unreadable)."""
    try:
        with open(path or TARGETS_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _sec_key(sec: str) -> tuple:
    return tuple(int(p) for p in sec.split(".") if p.isdecimal())


def nodes_for_tab(tab_cls: str, targets: dict | None = None) -> list[tuple[str, dict]]:
    """All mapped (node id, entry) of a tab, in MR section order."""
    nodes = (load_targets() if targets is None else targets).get("nodes", {})
    hits = [(nid, e) for nid, e in nodes.items() if e.get("tab") == tab_cls]
    return sorted(hits, key=lambda x: (_sec_key(x[1].get("mr", "") or "99"), x[0]))


def footer_nodes(tab_cls: str, n: int = FOOTER_MAX, targets: dict | None = None) -> list[tuple[str, dict]]:
    """Up to ``n`` nodes of the tab, spread over its MR sections (first node of each section, round robin)."""
    by_sec: dict[str, list] = {}
    for nid, e in nodes_for_tab(tab_cls, targets):
        by_sec.setdefault(e.get("mr", ""), []).append((nid, e))
    queues = list(by_sec.values())
    out: list = []
    while queues and len(out) < n:
        for q in list(queues):
            if len(out) >= n:
                break
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return sorted(out, key=lambda x: (_sec_key(x[1].get("mr", "") or "99"), x[0]))


def theory_footer(tab_cls: str, targets: dict | None = None) -> str:
    """HTML appended to a tab's theory panel: a "Knowledge Map" heading and up to 8 node links ('' if disabled
    with ``MRSTUDIO_NO_KMAP_LINKS`` or nothing is mapped and no map is known)."""
    if os.environ.get(NO_LINKS_ENV, "").strip() not in ("", "0"):
        return ""
    t = load_targets() if targets is None else targets
    picks = footer_nodes(tab_cls, targets=t)
    total = len(nodes_for_tab(tab_cls, t))
    head = "<h3>Knowledge Map</h3>"
    if not picks:
        if not t:
            return ""
        return (head + f'<p>Browse the <a href="{html.escape(map_base_url(), quote=True)}">Robotics Knowledge Map</a>'
                ".</p>")
    items = []
    for nid, e in picks:
        label = html.escape(e.get("label") or nid)
        mr = html.escape(e.get("mr", ""))
        cite = f" <small>(MR §{mr})</small>" if mr and mr[0].isdigit() else ""
        items.append(f'<li><a href="{html.escape(map_url(nid), quote=True)}">{label}</a>{cite}</li>')
    more = f" {total} map nodes link to this tab; a few of them:" if total > len(picks) else ""
    return (head + f"<p>Related nodes in the Robotics Knowledge Map (opens in the browser).{more}</p>"
            + "<ul>" + "".join(items) + "</ul>")
