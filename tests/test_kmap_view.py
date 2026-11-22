"""The optional Knowledge Map tab (gui/kmap_view.py): content lookup, the kmap: scheme handler, link routing, the main
window without / with the tab, and one real Qt WebEngine run in a subprocess.

No real map content is used or committed: the tests build a tiny fake map folder in ``tmp_path``. The real-content run
happens only when ``MRSTUDIO_TEST_KMAP_DIR`` names a built map folder (local machines). Everything that needs Qt
WebEngine runs in a subprocess (it must be set up before the QApplication) and is skipped where Qt WebEngine cannot
be imported (e.g. pip PySide6 on Linux CI without Chromium's system libraries).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402

from mrstudio import kmap_links as KL  # noqa: E402
from mrstudio.gui import kmap_view as K  # noqa: E402

FAKE_TITLE = "Robotics Knowledge Map"


def make_fake_map(root: Path, edition: str | None = "student") -> Path:
    """A tiny stand-in for a built map folder (NOT the real map): index.html, a script, a pyodide-like loader."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(textwrap.dedent(f"""\
        <!doctype html><html><head><meta charset="utf-8"><title>{FAKE_TITLE}</title>
        <script src="data/fake.js"></script></head>
        <body style="margin:0;background:#1e5aa0;color:#fff;font:40px sans-serif">
        <h1 id="h">Fake map</h1><div style="height:300px;background:#e0a030"></div>
        <a id="mrs" href="mrstudio://open?tab=FKTab&amp;robot=Planar%203R%20arm">Open in MR Studio</a>
        <a id="web" href="https://example.org/page">web</a></body></html>
        """), encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fake.js").write_text("window.FAKE_LOADED = true;\n", encoding="utf-8")
    (root / "pyodide").mkdir(exist_ok=True)
    (root / "pyodide" / "pyodide.js").write_text("// fake loader\n", encoding="utf-8")
    (root / "fake.wasm").write_bytes(b"\0asm\1\0\0\0")
    if edition is not None:
        (root / "build-info.json").write_text(json.dumps({"product": "fake", "edition": edition}), encoding="utf-8")
    return root


@pytest.fixture
def fake_map(tmp_path):
    return make_fake_map(tmp_path / "fake_map")


@pytest.fixture
def no_kmap_env(monkeypatch):
    for v in ("MRSTUDIO_KMAP_DIR", "MRSTUDIO_KMAP_URL", "MRSTUDIO_KMAP_EMBED"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(KL, "_EMBED", {"on": False})


# ----------------------------------------------------------------------------- scheme handler
@pytest.mark.parametrize("path", ["../secret.txt", "/../secret.txt", "data/../../secret.txt", "../fake_map/index.html"])
def test_resolve_refuses_paths_outside_the_map(fake_map, path):
    (fake_map.parent / "secret.txt").write_text("no", encoding="utf-8")
    status, p = K.resolve(fake_map, b"GET", "map", path)
    if path == "../fake_map/index.html":      # leaves and re-enters the folder: still inside, fine
        assert status == "ok" and p == (fake_map / "index.html").resolve()
    else:
        assert (status, p) == ("denied", None)


def test_resolve_absolute_and_encoded_paths(fake_map):
    outside = fake_map.parent / "secret.txt"
    outside.write_text("no", encoding="utf-8")
    assert K.resolve(fake_map, "GET", "map", str(outside))[0] in ("denied", "notfound")   # never served
    assert K.resolve(fake_map, "GET", "map", "/" + str(outside))[0] in ("denied", "notfound")
    # %2e%2e arrives percent-decoded from QUrl (the handler asks for FullyDecoded)
    for raw in ("kmap://map/%2e%2e/secret.txt", "kmap://map/data/%2E%2E/%2e%2e/secret.txt",
                "kmap://map/..%2Fsecret.txt"):
        job = K.FakeJob(raw)
        assert K.serve(job, fake_map, K.FAKE_ERRORS) == "denied", raw
        assert job.error == "RequestDenied" and job.data is None
    assert K.resolve(fake_map, "GET", "map", "data/x\x00.js") == ("denied", None)


@pytest.mark.skipif(not hasattr(os, "symlink") or sys.platform == "win32", reason="symlinks")
def test_resolve_refuses_symlink_out_of_the_map(fake_map):
    secret = fake_map.parent / "secret.txt"
    secret.write_text("no", encoding="utf-8")
    (fake_map / "link.txt").symlink_to(secret)
    assert K.resolve(fake_map, "GET", "map", "link.txt") == ("denied", None)


def test_resolve_method_host_and_missing(fake_map):
    assert K.resolve(fake_map, b"POST", "map", "index.html") == ("denied", None)
    assert K.resolve(fake_map, QtCore.QByteArray(b"PUT"), "map", "index.html") == ("denied", None)
    assert K.resolve(fake_map, b"GET", "other", "index.html") == ("denied", None)
    assert K.resolve(fake_map, b"GET", "map", "nope.js") == ("notfound", None)
    assert K.resolve(fake_map, b"GET", "map", "data") == ("notfound", None)          # a folder is not a file
    assert K.resolve(fake_map, b"GET", "map", "")[0] == "ok"                          # / -> index.html
    job = K.FakeJob("kmap://map/index.html", method=b"POST")
    assert K.serve(job, fake_map, K.FAKE_ERRORS) == "denied" and job.error == "RequestDenied"
    job = K.FakeJob("kmap://map/missing.css")
    assert K.serve(job, fake_map, K.FAKE_ERRORS) == "notfound" and job.error == "UrlNotFound"


def test_serve_replies_with_data_and_mime(fake_map):
    for rel, mime in [("index.html", "text/html"), ("data/fake.js", "text/javascript"),
                      ("pyodide/pyodide.js", "text/javascript"), ("fake.wasm", "application/wasm"),
                      ("build-info.json", "application/json")]:
        job = K.serve_check(fake_map, rel)
        assert job.error is None and job.mime == mime and job.data == (fake_map / rel).read_bytes(), rel


@pytest.mark.parametrize("name,mime", [
    ("a.wasm", "application/wasm"), ("a.js", "text/javascript"), ("a.mjs", "text/javascript"),
    ("a.json", "application/json"), ("a.css", "text/css"), ("a.svg", "image/svg+xml"), ("a.woff2", "font/woff2"),
    ("a.html", "text/html"), ("a.png", "image/png"), ("a.jpg", "image/jpeg"), ("a.webp", "image/webp"),
    ("a.gif", "image/gif"), ("a.data", "application/octet-stream"), ("a.zip", "application/octet-stream"),
    ("a.whl", "application/octet-stream"), ("a.tar", "application/octet-stream"), ("A.WASM", "application/wasm"),
])
def test_mime_types(name, mime):
    assert K.mime_for(name) == mime


# ----------------------------------------------------------------------------- navigation policy
def test_route_navigation():
    launched, external = [], []
    route = (lambda u: K.route_navigation(QtCore.QUrl(u), launched.append, external.append))
    assert route("mrstudio://open?tab=FKTab&robot=Planar 3R arm") is False
    assert launched == ["mrstudio://open?tab=FKTab&robot=Planar%203R%20arm"] and not external
    assert KL.parse_launch_url(launched[0])["robot"] == "Planar 3R arm"
    assert route("https://example.org/a?b=1") is False and route("http://x.org") is False
    assert route("mailto:someone@example.org") is False
    assert [u.toString() for u in external] == ["https://example.org/a?b=1", "http://x.org",
                                                "mailto:someone@example.org"]
    assert route("kmap://map/index.html#map/04.01.1/learn") is True
    assert route("data:text/html,hi") is True and route("about:blank") is True
    assert route("file:///etc/passwd") is False and route("javascript:alert(1)") is False
    assert len(launched) == 1 and len(external) == 3


def test_inject_source_and_hash():
    src = K.inject_source("student", "darwin")
    info = json.loads(src.split("Object.freeze(", 1)[1].split(");", 1)[0])
    assert info == {"desktop": True, "platform": "darwin", "edition": "student", "embedded": True}
    assert "window.KMAP_PYODIDE_BASE='pyodide/'" in src
    assert K.node_hash("04.01.1") == "map/04.01.1/learn" and K.node_hash("x", "") == "map/x"


def test_edition_defaults_to_student():
    if (ROOT / "mrstudio" / "edition.py").exists():
        pytest.skip("private instructor copy")
    assert KL.app_edition() == "student"


# ----------------------------------------------------------------------------- content lookup
def test_lookup_without_content(no_kmap_env, tmp_path, monkeypatch):
    assert KL.embedded_content_dir() is None
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(tmp_path))                 # set, but no map there
    assert KL.embedded_content_dir() is None
    assert KL.bundled_kmap_dir() is None                                   # running from source


def test_lookup_map_folder_and_student_edition_layout(no_kmap_env, tmp_path, monkeypatch):
    built = make_fake_map(tmp_path / "content" / "student")
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(built))
    assert KL.embedded_content_dir() == built.resolve()
    km = tmp_path / "Knowledge Map"                                        # the old MRSTUDIO_KMAP_DIR layout
    se = make_fake_map(km / KL.STUDENT_EDITION_FOLDER, edition=None)
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(km))
    assert KL.embedded_content_dir() == se.resolve()
    assert KL.map_base_url() == (se / "index.html").resolve().as_uri()     # browser links unchanged
    monkeypatch.setenv("MRSTUDIO_KMAP_EMBED", "0")
    assert KL.embedded_content_dir() is None


def test_lookup_bundled_map_in_frozen_app(no_kmap_env, tmp_path, monkeypatch):
    make_fake_map(tmp_path / "bundle" / KL.BUNDLED_KMAP)
    other = make_fake_map(tmp_path / "other")
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(other))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    assert KL.bundled_kmap_dir() == (tmp_path / "bundle" / "kmap").resolve()
    assert KL.embedded_content_dir() == (tmp_path / "bundle" / "kmap").resolve()     # bundled map first


@pytest.mark.parametrize("plant", ["kmap-x", "instructor-dir", "instructor-edition"])
def test_student_app_never_embeds_instructor_content(no_kmap_env, tmp_path, monkeypatch, plant):
    root = make_fake_map(tmp_path / "m", edition="instructor" if plant == "instructor-edition" else "student")
    if plant == "kmap-x":
        (root / "data" / "x_test.js").write_text("// planted\n", encoding="utf-8")
    elif plant == "instructor-dir":
        (root / "instructor").mkdir()
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(root))
    assert KL.content_problems(root, "student")
    if KL.app_edition() == "student":
        assert KL.embedded_content_dir() is None


def test_content_problems(tmp_path):
    root = make_fake_map(tmp_path / "ok")
    assert KL.content_problems(root, "student") == []
    assert KL.content_problems(root, "student", require_build_info=True) == []
    assert any("edition" in p for p in KL.content_problems(root, "instructor"))
    bare = make_fake_map(tmp_path / "bare", edition=None)
    assert KL.content_problems(bare, "student") == []
    assert any("build-info" in p for p in KL.content_problems(bare, "student", require_build_info=True))
    assert any("index.html" in p for p in KL.content_problems(tmp_path / "empty", "student"))


# ----------------------------------------------------------------------------- main window
@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MRSTUDIO_SETTINGS", str(tmp_path / "s.ini"))


def test_no_prepared_webengine_means_no_tab(qapp, no_kmap_env, settings_file, fake_map, monkeypatch):
    """Map content alone does not add a tab: only ``prepare`` (before the QApplication) does. 16 tabs, as before."""
    from mrstudio.gui.app import TABS, MainWindow
    from mrstudio.gui.base import AppContext
    monkeypatch.setenv("MRSTUDIO_KMAP_DIR", str(fake_map))
    assert K.prepared_root() is None
    w = MainWindow(AppContext("Planar 2R arm"))
    w.persist = False
    try:
        assert w.tabs.count() == len(TABS) == len(KL.TAB_ORDER) == 16
        assert w.kmap_tab is None and not KL.embedded()
        assert not w.open_map_url(QtCore.QUrl(KL.EMBED_URL + "#map/x/learn"))
        assert KL.map_url("04.01.1").startswith(KL.ARTIFACT_URL)    # fake map has no Student Edition sub-folder
    finally:
        w.close()


def test_main_window_with_map_tab(qapp, no_kmap_env, settings_file, fake_map, monkeypatch):
    """With Qt WebEngine prepared (simulated here), the map is a 17th, last tab outside TABS / TAB_ORDER; theory
    footers and other map links open in it. The web view itself is not created here (``ensure_view`` stubbed)."""
    from PySide6 import QtGui, QtWidgets
    from mrstudio.gui.app import TABS, MainWindow
    from mrstudio.gui.base import AppContext
    monkeypatch.setitem(K._STATE, "prepared", True)
    monkeypatch.setitem(K._STATE, "root", fake_map)
    shown = []
    monkeypatch.setattr(K.KmapTab, "ensure_view", lambda self: True)
    monkeypatch.setattr(K.KmapTab, "show_hash", lambda self, frag: shown.append(frag) or True)
    w = MainWindow(AppContext("Planar 2R arm"))
    w.persist = False
    try:
        assert KL.embedded() and w.tabs.count() == len(TABS) + 1 == 17
        assert w.tabs.widget(16) is w.kmap_tab and isinstance(w.kmap_tab, K.KmapTab)
        assert w.tabs.tabText(16) == K.TAB_TITLE and w.page_index("KmapTab") == 16
        assert [type(p).__name__ for p in w.pages[:16]] == KL.TAB_ORDER
        # theory footers point at the embedded map ...
        fk = w.pages[0]
        assert KL.EMBED_URL + "#map/" in fk.theory._text and "Knowledge Map tab" in fk.theory._text
        # ... and a click on such a link (QTextBrowser -> QDesktopServices) switches to the tab
        w.tabs.setCurrentIndex(0)
        assert QtGui.QDesktopServices.openUrl(QtCore.QUrl(KL.map_url("04.01.1")))
        qapp.processEvents()
        assert w.tabs.currentIndex() == 16 and shown == ["map/04.01.1/learn"]
        assert not w.robot_box.isEnabled() and "Knowledge Map" in w.robot_box.toolTip()
        # mrstudio:// links from the page go to open_url like OS-delivered ones
        assert w.open_url("mrstudio://open?tab=MobileTab")
        assert type(w.pages[w.tabs.currentIndex()]).__name__ == "MobileTab"
        assert isinstance(w.kmap_tab, QtWidgets.QWidget)
    finally:
        w.close()
        KL.set_embedded(False)
    assert w._kmap_sink is None      # URL handler removed on shutdown


# ----------------------------------------------------------------------------- Qt WebEngine (subprocess)
def _webengine_importable() -> bool:
    code = "import PySide6.QtWebEngineCore, PySide6.QtWebEngineWidgets, PySide6.QtQuick"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    return r.returncode == 0


_WE = {}


def need_webengine():
    if "ok" not in _WE:
        _WE["ok"] = _webengine_importable()
    if not _WE["ok"]:
        pytest.skip("Qt WebEngine cannot be imported here")


def test_prepare_selects_opengl_for_qt_quick_source():
    """Static guard: without OpenGL for Qt Quick the map renders black next to the 3-D views (macOS)."""
    src = (ROOT / "mrstudio" / "gui" / "kmap_view.py").read_text(encoding="utf-8")
    body = src.split("def prepare(", 1)[1].split("\ndef ", 1)[0]
    assert "QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)" in body
    assert "register_schemes(" in body and "AA_ShareOpenGLContexts" in body
    app_src = (ROOT / "mrstudio" / "gui" / "app.py").read_text(encoding="utf-8")
    main = app_src.split("\ndef main(", 1)[1]
    assert main.index("prepare_kmap()") < main.index("(sys.argv[:1])")      # before the QApplication is created


PREPARE_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from PySide6 import QtWidgets
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
from mrstudio.gui import kmap_view as K
before = QQuickWindow.graphicsApi()
ok = K.prepare(sys.argv[2])
app = QtWidgets.QApplication([])
from PySide6.QtWebEngineCore import QWebEngineUrlScheme
print(json.dumps({"ok": ok, "err": K.last_error(),
                  "api": QQuickWindow.graphicsApi() == QSGRendererInterface.GraphicsApi.OpenGL,
                  "api_before": before == QSGRendererInterface.GraphicsApi.OpenGL,
                  "kmap": bytes(QWebEngineUrlScheme.schemeByName(b"kmap").name()).decode(),
                  "mrstudio": bytes(QWebEngineUrlScheme.schemeByName(b"mrstudio").name()).decode(),
                  "late": K.prepare(sys.argv[2])}))
"""


def test_prepare_sets_opengl_and_schemes(tmp_path, fake_map):
    need_webengine()
    script = tmp_path / "probe.py"
    script.write_text(PREPARE_PROBE, encoding="utf-8")
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(fake_map)], capture_output=True, text=True,
                       timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ok"] and out["api"] and not out["api_before"], out
    assert out["kmap"] == "kmap" and out["mrstudio"] == "mrstudio"
    assert out["late"] is False      # after the QApplication exists: refused, never half-initialised


LOAD_PROBE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
root, kind = sys.argv[2], sys.argv[3]
from PySide6 import QtCore, QtGui, QtWidgets
from mrstudio.gui import kmap_view as K
res = {"prepared": K.prepare(root), "err": K.last_error()}
app = QtWidgets.QApplication([])

class Sink(QtCore.QObject):
    @QtCore.Slot(QtCore.QUrl)
    def handle(self, url):
        res.setdefault("external", []).append(url.toString())
sink = Sink()
for s in ("http", "https"):                       # never start a real browser from a test
    QtGui.QDesktopServices.setUrlHandler(s, sink, "handle")
launched = res.setdefault("launched", [])
tab = K.KmapTab(root, launched.append, on_log=lambda t: res.setdefault("log", []).append(t))
win = QtWidgets.QMainWindow(); win.setCentralWidget(tab); win.resize(900, 700); win.show()
t0 = time.perf_counter()

def finish(*_):
    res["seconds"] = round(time.perf_counter() - t0, 2)
    print("RESULT " + json.dumps(res), flush=True)
    tab.shutdown()
    app.quit()

def js(code, then):
    tab.page.runJavaScript(code, 0, lambda r: QtCore.QTimer.singleShot(0, lambda: then(r)))

steps = []
def nxt(*_):
    if steps:
        steps.pop(0)()
    else:
        finish()

def s_info():
    js("JSON.stringify({title: document.title, app: window.kmapApp, py: window.KMAP_PYODIDE_BASE,"
       " frozen: Object.isFrozen(window.kmapApp), fake: !!window.FAKE_LOADED, origin: location.origin})",
       lambda r: (res.update(info=json.loads(r)), nxt()))
def s_hash():
    tab.show_hash(K.node_hash("01.03.5"))
    QtCore.QTimer.singleShot(400, lambda: js("location.hash", lambda r: (res.update(hash=r), nxt())))
def s_launch():
    js("location.href = 'mrstudio://open?tab=FKTab&robot=Planar%203R%20arm'", lambda r: QtCore.QTimer.singleShot(800, nxt))
def s_click():
    js("document.getElementById('mrs') && document.getElementById('mrs').click()",
       lambda r: QtCore.QTimer.singleShot(800, nxt))
def s_web():
    js("location.href = 'https://example.org/page'", lambda r: QtCore.QTimer.singleShot(800, nxt))
def s_fetch():
    js("fetch('pyodide/pyodide.js').then(r => r.status).catch(e => 'ERR')", lambda r: QtCore.QTimer.singleShot(0, nxt))
def s_after():
    js("JSON.stringify({href: location.href, alive: document.title})", lambda r: (res.update(after=json.loads(r)), nxt()))
def s_pixels():
    img = tab.view.grab().toImage()
    cols = set()
    for y in range(0, img.height(), max(1, img.height() // 40)):
        for x in range(0, img.width(), max(1, img.width() // 40)):
            cols.add(img.pixel(x, y))
    res["colours"] = len(cols)
    nxt()

steps[:] = [s_info, s_hash, s_launch, s_click, s_web, s_fetch, s_after, s_pixels]
def loaded(ok):
    res.setdefault("loads", []).append(ok)
    if ok and "started" not in res:
        res["started"] = True
        QtCore.QTimer.singleShot(1500 if kind == "real" else 300, nxt)
tab.ensure_view()
tab.page.loadFinished.connect(loaded)
QtCore.QTimer.singleShot(90_000, lambda: (res.update(timeout=True), finish()))
app.exec()
"""


def _run_load_probe(tmp_path, root, kind):
    need_webengine()
    script = tmp_path / "load_probe.py"
    script.write_text(LOAD_PROBE, encoding="utf-8")
    env = dict(os.environ, MRSTUDIO_SETTINGS=str(tmp_path / "s.ini"), MRSTUDIO_LOG=str(tmp_path / "m.log"))
    r = subprocess.run([sys.executable, str(script), str(ROOT), str(root), kind], capture_output=True, text=True,
                       timeout=240, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        pytest.fail(f"no result (exit {r.returncode}): {r.stderr[-3000:]}")
    res = json.loads(lines[-1][len("RESULT "):])
    if not res.get("prepared"):
        pytest.skip(f"Qt WebEngine not usable here: {res.get('err')}")
    if res.get("timeout") and not res.get("loads"):
        pytest.skip(f"Qt WebEngine never finished loading on platform {os.environ.get('QT_QPA_PLATFORM')!r}")
    assert (tmp_path / "kmap-web").is_dir()          # persistent profile next to MRSTUDIO_SETTINGS, not ~
    return res


def test_real_webengine_load_and_routing(tmp_path, fake_map):
    res = _run_load_probe(tmp_path, fake_map, "fake")
    assert not res.get("timeout"), res
    assert res["loads"][0] is True
    info = res["info"]
    assert info["title"] == FAKE_TITLE and info["fake"] and info["origin"] == "kmap://map"
    assert info["app"] == {"desktop": True, "platform": sys.platform, "edition": KL.app_edition(), "embedded": True}
    assert info["py"] == "pyodide/" and info["frozen"]
    assert res["hash"] == "#map/01.03.5/learn"
    # location.href and a script click on an mrstudio:// link both reach the launch callback (once each)
    assert res["launched"] == ["mrstudio://open?tab=FKTab&robot=Planar%203R%20arm"] * 2, res
    assert res["external"] == ["https://example.org/page"], res
    assert res["after"]["href"].startswith("kmap://map/index.html") and res["after"]["alive"] == FAKE_TITLE
    assert not [t for t in res.get("log", []) if "did not load" in t], res.get("log")


@pytest.mark.skipif(not os.environ.get("MRSTUDIO_TEST_KMAP_DIR"), reason="set MRSTUDIO_TEST_KMAP_DIR to a built map")
def test_real_map_content_loads(tmp_path):
    root = KL._map_folder(Path(os.environ["MRSTUDIO_TEST_KMAP_DIR"]))
    assert root is not None
    res = _run_load_probe(tmp_path, root, "real")
    assert not res.get("timeout"), res
    assert res["loads"][0] is True and res["info"]["title"] == "Robotics Knowledge Map"
    assert res["hash"] == "#map/01.03.5/learn"
    assert res["launched"] and all(u.startswith("mrstudio://open?tab=FKTab") for u in res["launched"])


def test_app_with_map_routes_file_open_events_without_an_app_wide_filter(qapp):
    """With Qt WebEngine, mrstudio:// links from macOS reach the window through MRStudioApplication.event (an
    application-wide Python event filter crashed PySide6 while the map page loaded)."""
    import types

    from PySide6 import QtGui
    from mrstudio.gui.app import MRStudioApplication, UrlOpenFilter
    f = UrlOpenFilter()
    fake_app = types.SimpleNamespace(url_filter=f)
    ev = QtGui.QFileOpenEvent(QtCore.QUrl("mrstudio://open?tab=MobileTab"))
    assert MRStudioApplication.event(fake_app, ev) is True
    assert f.pending == ["mrstudio://open?tab=MobileTab"]
    src = (ROOT / "mrstudio" / "gui" / "app.py").read_text(encoding="utf-8")
    assert "app.url_filter = url_filter" in src and "MRStudioApplication if kmap else" in src
