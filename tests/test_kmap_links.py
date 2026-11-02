"""Knowledge Map links: section -> tab table, generated targets JSON, mrstudio:// URLs, theory-panel footer."""
import importlib
import json
import os

import pytest

from mrstudio import kmap_links as kl
from mrstudio.gui.theory import validate_all

NEW_TABS = {"GraspingTab", "MobileTab"}   # Ch. 12 / Ch. 13 tabs (registered in gui/app.py by the integration step)


def _app_tabs():
    from mrstudio.gui.app import TABS
    return [c for _m, c in TABS]


# ----------------------------------------------------------------------------- tab order and section table
def test_tab_order_matches_app():
    app = _app_tabs()
    assert set(app) <= set(kl.TAB_ORDER) and set(kl.TAB_ORDER) <= set(app) | NEW_TABS
    assert [c for c in kl.TAB_ORDER if c in app] == app      # same relative order
    if NEW_TABS <= set(app):                                 # once registered: identical, Ch. 12/13 after Haptics
        assert app == kl.TAB_ORDER
    assert kl.TAB_ORDER.index("GraspingTab") == kl.TAB_ORDER.index("HapticsTab") + 1
    assert kl.TAB_ORDER.index("MobileTab") == kl.TAB_ORDER.index("GraspingTab") + 1


def test_section_table_targets_exist():
    from mrstudio.robots import ROBOTS
    for sec, (tab, robot, preset) in kl.SECTION_TABS.items():
        assert tab in kl.TAB_ORDER, sec
        assert robot is None or robot in ROBOTS, (sec, robot)
    from mrstudio import haptics
    from mrstudio.gui.tabs.closed_common import FOUR_PRESETS
    known = {"HapticsTab": set(haptics.PRESETS), "ClosedChainTab": set(FOUR_PRESETS)}
    for sec, (tab, _robot, preset) in kl.SECTION_TABS.items():
        if preset is not None:
            assert preset in known.get(tab, set()), (sec, tab, preset)
    # robots only for tabs that use the serial-robot selector
    for sec, (tab, robot, _p) in kl.SECTION_TABS.items():
        if robot is None or tab in NEW_TABS:
            continue
        mod = dict((c, m) for m, c in importlib.import_module("mrstudio.gui.app").TABS)[tab]
        assert getattr(importlib.import_module(mod), tab).uses_serial_robot, (sec, tab)


@pytest.mark.parametrize("sec,tab", [("4.1", "FKTab"), ("5.1.2", "VelocityTab"), ("6.1", "IKTab"),
                                     ("11.4.3", "AdvancedControlTab"), ("11.4.1", "ControlTab"),
                                     ("11.7.2", "HapticsTab"), ("12.1.7", "GraspingTab"), ("13.3.3", "MobileTab"),
                                     ("7.1.2", "StewartTab"), ("7.1.1", "ParallelRobotsTab"), ("8.9.2", "HardwareTab")])
def test_longest_prefix(sec, tab):
    assert kl.tab_for_section(sec)[0] == tab


def test_tab_for_unknown_section():
    assert kl.tab_for_section("1.2") is None and kl.tab_for_section("14") is None


@pytest.mark.parametrize("where,sec", [("§4.1, Eq. (4.14), p. 140", "4.1"), ("§ 13.3.3.1, p. 540", "13.3.3.1"),
                                       ("Ch. 12 introduction, p. 463", "12"), ("Example 12.2, Fig. 12.3", "12"),
                                       ("Eq. (8.9), p. 273", "8"), ("App. B.1, Eq. (B.1), p. 577", None),
                                       ("Foreword by R. Brockett, p. ix", None)])
def test_mr_section(where, sec):
    assert kl.mr_section(where) == sec


# ----------------------------------------------------------------------------- generated JSON
def test_targets_json():
    assert kl.TARGETS_JSON.is_file(), "run tools/make_kmap_links.py"
    d = json.loads(kl.TARGETS_JSON.read_text(encoding="utf-8"))
    assert d["version"] >= 1 and d["scheme"].startswith("mrstudio://open?") and d["generated"]
    assert list(d["tabs"]) == kl.TAB_ORDER
    assert [v["index"] for v in d["tabs"].values()] == list(range(len(kl.TAB_ORDER)))
    nodes = d["nodes"]
    assert len(nodes) >= 300
    allowed = set(_app_tabs()) | NEW_TABS
    assert {e["tab"] for e in nodes.values()} <= allowed
    assert {"GraspingTab", "MobileTab", "FKTab", "DynamicsTab"} <= {e["tab"] for e in nodes.values()}
    for nid, e in nodes.items():
        assert set(e) >= {"tab", "robot", "preset", "label"}, nid
        assert e["label"], nid
        assert kl.tab_for_section(e["mr"])[0] == e["tab"], nid
        p = kl.parse_launch_url(e["url"])
        assert (p["tab"], p["robot"], p["preset"]) == (e["tab"], e["robot"], e["preset"]), nid


# ----------------------------------------------------------------------------- mrstudio:// URLs
@pytest.mark.parametrize("tab,robot,preset", [("FKTab", "UR5 (6R industrial)", None),
                                              ("HapticsTab", None, "Virtual wall — hand pushes in"),
                                              ("MobileTab", None, None),
                                              ("ClosedChainTab", None, "Double-crank (drag link)")])
def test_launch_url_round_trip_class(tab, robot, preset):
    url = kl.launch_url(tab, robot, preset)
    assert url.startswith("mrstudio://open?tab=") and " " not in url
    p = kl.parse_launch_url(url)
    assert p == {"action": "open", "tab": tab, "index": kl.TAB_ORDER.index(tab), "robot": robot, "preset": preset}


def test_launch_url_round_trip_index():
    for i, cls in enumerate(kl.TAB_ORDER):
        p = kl.parse_launch_url(kl.launch_url(i, "Planar 2R arm"))
        assert (p["tab"], p["index"], p["robot"]) == (cls, i, "Planar 2R arm")


def test_parse_launch_url_variants():
    p = kl.parse_launch_url("mrstudio://open?tab=MobileTab&robot=UR5%20(6R%20industrial)")
    assert p["tab"] == "MobileTab" and p["robot"] == "UR5 (6R industrial)"
    p = kl.parse_launch_url("MRSTUDIO://open?tab=2&robot=UR5+%286R+industrial%29")
    assert p["tab"] == "IKTab" and p["index"] == 2 and p["robot"] == "UR5 (6R industrial)"
    assert kl.parse_launch_url("mrstudio:open?tab=FKTab")["tab"] == "FKTab"
    assert kl.parse_launch_url("mrstudio://open/?tab=0")["index"] == 0
    for bad in ("mrstudio://open?tab=99", "mrstudio://open?tab=EvilTab", "mrstudio://open", "mrstudio://open?tab="):
        p = kl.parse_launch_url(bad)
        assert p["tab"] is None and p["index"] is None and p["robot"] is None
    for bad in ("https://example.com/?tab=1", "mrstudio://delete?tab=1", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            kl.parse_launch_url(bad)


# ----------------------------------------------------------------------------- map URLs
def test_map_url_env_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("MRSTUDIO_KMAP_URL", "https://example.org/map/index.html#old")
    assert kl.map_url("04.01.1") == "https://example.org/map/index.html#map/04.01.1/learn"
    monkeypatch.delenv("MRSTUDIO_KMAP_URL")
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(tmp_path))            # no local edition there
    assert kl.map_url("04.01.1") == kl.ARTIFACT_URL + "#map/04.01.1/learn"
    se = tmp_path / "Robotics Knowledge Map - Student Edition"
    se.mkdir()
    (se / "index.html").write_text("<html></html>")
    u = kl.map_url("04.01.1", "practise")
    assert u.startswith("file://") and u.endswith("/index.html#map/04.01.1/practise") and " " not in u


# ----------------------------------------------------------------------------- footer
def test_theory_footer(monkeypatch):
    monkeypatch.delenv(kl.NO_LINKS_ENV, raising=False)
    for cls in kl.TAB_ORDER:
        html = kl.theory_footer(cls)
        assert "<h3>Knowledge Map</h3>" in html, cls
        assert html.count("<li>") <= kl.FOOTER_MAX
        if kl.nodes_for_tab(cls):
            assert 'href="' in html and "#map/" in html and html.count("<li>") >= 1, cls
    assert validate_all([kl.theory_footer(c) for c in kl.TAB_ORDER]) == []
    assert "$" not in "".join(kl.theory_footer(c) for c in kl.TAB_ORDER)   # no stray mathtext from labels
    monkeypatch.setenv(kl.NO_LINKS_ENV, "1")
    assert kl.theory_footer("FKTab") == ""


def test_footer_spreads_over_sections():
    picks = kl.footer_nodes("FKTab")
    assert len(picks) == kl.FOOTER_MAX
    assert len({e["mr"] for _n, e in picks}) == kl.FOOTER_MAX


def test_theory_panel_has_map_links(qapp, monkeypatch):
    monkeypatch.delenv(kl.NO_LINKS_ENV, raising=False)
    from mrstudio.gui.base import AppContext
    from mrstudio.gui.tabs.fk_tab import FKTab
    tab = FKTab(AppContext("Planar 2R arm"))
    try:
        html = tab.theory.content()
        assert "<h3>Knowledge Map</h3>" in html and 'href="' in html and "#map/" in html
    finally:
        tab.shutdown()
    monkeypatch.setenv(kl.NO_LINKS_ENV, "1")
    tab = FKTab(AppContext("Planar 2R arm"))
    try:
        assert "#map/" not in tab.theory.content()
    finally:
        tab.shutdown()
