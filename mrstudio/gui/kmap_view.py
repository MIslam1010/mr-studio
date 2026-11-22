"""The Robotics Knowledge Map inside MR Studio: an optional last tab, "⑰ Knowledge Map" (Qt WebEngine).

The tab exists only when a built map is found (``kmap_links.embedded_content_dir``: the map bundled with a frozen
app, or ``MRSTUDIO_KMAP_DIR``) *and* Qt WebEngine can be loaded; otherwise MR Studio is exactly as before and the
theory-panel links open the map in the browser. The public repository and its CI never contain or bundle map
content. Contract and environment variables: ``docs/KMAP-LINKING.md``.

How it works
------------
* ``prepare(root)`` runs in ``app.main`` *before* the QApplication exists: it imports Qt WebEngine (lazily -- the
  import fails harmlessly on systems without Chromium's libraries), registers the URL schemes ``kmap`` and
  ``mrstudio`` and selects OpenGL for Qt Quick. Without ``QQuickWindow.setGraphicsApi(OpenGL)`` the web view
  renders solid black next to the QOpenGLWidget 3-D views on macOS (Metal by default) -- silently.
* The page is served from ``kmap://map/<path>`` by ``MapSchemeHandler`` (GET only, files inside the map folder
  only, see ``resolve``); nothing is loaded from the network (``Interceptor``).
* ``KmapTab`` creates the web view lazily, the first time the tab is shown or a map link is opened. A
  ``QWebEngineScript`` sets ``window.kmapApp = {desktop, platform, edition, embedded: true}`` and
  ``window.KMAP_PYODIDE_BASE = 'pyodide/'`` before the page's own scripts run.
* Navigation (``route_navigation``): ``mrstudio://open?...`` ("Open in MR Studio") goes to
  ``MainWindow.open_url``; ``http(s)``/``mailto`` links open in the system browser; ``kmap:`` stays in the view.
* Map links elsewhere in MR Studio (``kmap://map/index.html#map/<node>/learn``, theory footers) reach
  ``MainWindow.open_map_url`` through ``QDesktopServices.setUrlHandler("kmap", ...)``.
* Web storage (progress, settings of the map) is kept next to MR Studio's data (``profile_dir``); in
  ``--screenshot`` mode the profile is off the record.
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import types
from functools import lru_cache
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

SCHEME = "kmap"
HOST = "map"
START_URL = f"{SCHEME}://{HOST}/index.html"
LAUNCH_SCHEME = "mrstudio"
TAB_TITLE = "⑰ Knowledge Map"
TAB_LONG_NAME = "Robotics Knowledge Map"
PROFILE_NAME = "kmap"
PROFILE_FOLDER = "kmap-web"
#: schemes the page may load from (everything else is blocked: the map is fully offline)
ALLOWED_SCHEMES = frozenset({SCHEME, "data", "blob", "about"})
#: schemes opened in the system browser / mail program when the page navigates to them
EXTERNAL_SCHEMES = frozenset({"http", "https", "mailto"})
MIME = {
    ".html": "text/html", ".htm": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
    ".json": "application/json", ".map": "application/json", ".webmanifest": "application/manifest+json",
    ".css": "text/css", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
    ".otf": "font/otf", ".wasm": "application/wasm", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
    ".txt": "text/plain", ".pdf": "application/pdf", ".data": "application/octet-stream",
    ".zip": "application/octet-stream", ".whl": "application/octet-stream", ".tar": "application/octet-stream",
}
#: a navigation refused by ``route_navigation`` makes Qt report a failed load; ignore those for this long
REJECT_GRACE_S = 3.0
#: a web link opens the system browser only if the user pressed a mouse button / key this recently: Qt reports every
#: renderer-initiated navigation (a timer's ``location.href = ...`` too) as "link clicked", so the type alone proves nothing
INPUT_GRACE_S = 2.0
#: a missing or refused file with one of these suffixes is answered with ``fail()`` (a <script>/<link>/<img>/font
#: then fires ``onerror``). Anything else (data files fetched with fetch / XHR) gets an empty, finished body instead:
#: ``fail()`` leaves the body of a fetch() response unfinished, so ``fetch(url).then(r => r.text())`` never settles.
FAIL_SUFFIXES = frozenset({".js", ".mjs", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".woff",
                           ".woff2", ".ttf", ".otf"})

_STATE: dict = {"prepared": False, "root": None, "error": ""}


# ----------------------------------------------------------------------------- set-up (before the QApplication)
def register_schemes(QWebEngineUrlScheme) -> None:
    """Register ``kmap`` (host syntax, secure, CORS, fetch) and ``mrstudio`` (path syntax). Before the
    QApplication; a registered ``mrstudio`` scheme makes page-initiated navigations (``location.href = …``,
    script clicks) reach ``acceptNavigationRequest`` too, not only real mouse clicks."""
    if QWebEngineUrlScheme.schemeByName(SCHEME.encode()).name():
        return
    k = QWebEngineUrlScheme(SCHEME.encode())
    k.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    k.setFlags(QWebEngineUrlScheme.Flag.SecureScheme | QWebEngineUrlScheme.Flag.CorsEnabled
               | QWebEngineUrlScheme.Flag.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(k)
    m = QWebEngineUrlScheme(LAUNCH_SCHEME.encode())
    m.setSyntax(QWebEngineUrlScheme.Syntax.Path)
    m.setFlags(QWebEngineUrlScheme.Flag.SecureScheme)
    QWebEngineUrlScheme.registerScheme(m)


def prepare(root) -> bool:
    """Get Qt WebEngine ready for the map in ``root``; must run before the QApplication is created.

    Returns False (and the app runs without the tab) if a QApplication already exists or Qt WebEngine cannot be
    imported (``last_error()`` says why)."""
    if QtCore.QCoreApplication.instance() is not None:
        _STATE["error"] = "a QApplication already exists (prepare() must run before it is created)"
        return False
    try:
        from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
        from PySide6.QtWebEngineCore import QWebEngineUrlScheme
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401  (loads Chromium's libraries now)
    except Exception as e:  # noqa: BLE001 - ImportError, or OSError from a missing system library
        _STATE["error"] = f"Qt WebEngine is not available: {type(e).__name__}: {e}"
        return False
    register_schemes(QWebEngineUrlScheme)
    # Metal/D3D (the Qt Quick default) leaves the web view black next to the QOpenGLWidget views: use OpenGL.
    QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    _STATE.update(prepared=True, root=Path(root), error="")
    return True


def prepared_root() -> Path | None:
    """The map folder if ``prepare`` succeeded in this process (then the main window adds the tab)."""
    return _STATE["root"] if _STATE["prepared"] else None


def last_error() -> str:
    return _STATE["error"]


# ----------------------------------------------------------------------------- pure helpers (no WebEngine)
def mime_for(path) -> str:
    p = Path(path)
    return MIME.get(p.suffix.lower()) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"


def resolve(root, method, host: str, path: str) -> tuple[str, Path | None]:
    """Map a request to a file: ``("ok", file)``, ``("notfound", None)`` or ``("denied", None)``.

    Only GET requests for host ``map`` are served, and only files inside ``root`` (after resolving ``..`` and
    symbolic links); ``path`` is the URL path, already percent-decoded."""
    if isinstance(method, (bytes, bytearray, QtCore.QByteArray)):
        method = bytes(method).decode("latin-1")
    if str(method).upper() != "GET" or (host or "").lower() != HOST:
        return "denied", None
    rel = (path or "").lstrip("/\\") or "index.html"
    if "\x00" in rel:
        return "denied", None
    try:
        base = Path(root).resolve()
        p = (base / rel).resolve()
    except (OSError, ValueError, RuntimeError):
        return "denied", None
    if not p.is_relative_to(base):
        return "denied", None
    if not p.is_file():
        return "notfound", None
    return "ok", p


def serve(job, root, errors) -> str:
    """Answer a ``QWebEngineUrlRequestJob`` (or anything with the same methods) from the map folder ``root``.
    ``errors`` is ``QWebEngineUrlRequestJob.Error`` (any object with UrlNotFound / RequestDenied / RequestFailed).
    Returns the outcome (``"ok"``, ``"notfound"``, ``"denied"``, ``"failed"``). A refused or missing file whose
    suffix is not in ``FAIL_SUFFIXES`` gets an empty, finished body instead of ``fail()`` (see there)."""
    url = job.requestUrl()
    status, p = resolve(root, job.requestMethod(), url.host(),
                        url.path(QtCore.QUrl.ComponentFormattingOption.FullyDecoded))
    fetch_like = Path(url.path()).suffix.lower() not in FAIL_SUFFIXES
    if status != "ok":
        if fetch_like:
            _reply_bytes(job, b"", "text/plain")
        else:
            job.fail(errors.UrlNotFound if status == "notfound" else errors.RequestDenied)
        return status
    try:
        data = p.read_bytes()
    except OSError:
        if fetch_like:
            _reply_bytes(job, b"", "text/plain")
        else:
            job.fail(errors.RequestFailed)
        return "failed"
    _reply_bytes(job, data, mime_for(p))
    return "ok"


def _reply_bytes(job, data: bytes, mime: str) -> None:
    buf = QtCore.QBuffer(job)            # owned by the job: freed with it
    buf.setData(QtCore.QByteArray(data))
    buf.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    job.reply(mime.encode(), buf)


def classify_navigation(url: QtCore.QUrl) -> str:
    """``"mrstudio"`` (a launch link), ``"external"`` (http/https/mailto), ``"allow"`` (kmap, data, blob, about) or
    ``"block"`` (anything else, e.g. file:)."""
    s = url.scheme().lower()
    if s == LAUNCH_SCHEME:
        return "mrstudio"
    if s in EXTERNAL_SCHEMES:
        return "external"
    return "allow" if s in ALLOWED_SCHEMES else "block"


def launch_text(url: QtCore.QUrl) -> str:
    """A launch URL as text, fully percent-encoded (spaces in robot names stay ``%20``)."""
    try:
        return url.toString(QtCore.QUrl.ComponentFormattingOption.FullyEncoded)
    except TypeError:   # older bindings: only the UrlFormattingOption overload
        return bytes(url.toEncoded()).decode("ascii", "replace")


def route_navigation(url: QtCore.QUrl, on_launch, on_external, *, main_frame: bool = True, clicked: bool = True,
                     scripted: bool = False) -> bool:
    """The page's navigation policy (``QWebEnginePage.acceptNavigationRequest``): True = let the view load it.

    * Sub-frames (iframes) never reach a callback: only ``kmap:``/``data:``/``blob:``/``about:`` load in them.
    * ``on_external(QUrl)`` (system browser / mail) only for a real link click (``clicked``): a script that does
      ``location.href = "https://..."`` goes nowhere.
    * ``on_launch(text)`` for ``mrstudio://`` on a click, or ``scripted`` (the map's own "Open in MR Studio" button
      sets ``location.href``; it only switches a tab of this app, and ``parse_launch_url`` checks the link)."""
    kind = classify_navigation(url)
    if not main_frame:
        return kind == "allow"
    if kind == "mrstudio":
        if clicked or scripted:
            on_launch(launch_text(url))
        return False
    if kind == "external":
        if clicked:
            on_external(QtCore.QUrl(url))
        return False
    return kind == "allow"


def choose_download_path(request_name: str, chooser, default_dir: str | None = None):
    """Where to save a download: ``chooser(default_path)`` returns the chosen path ('' = cancelled). Returns
    ``(directory, file name)`` or None. Only the base name of the page-supplied file name is used."""
    name = os.path.basename((request_name or "").replace("\\", "/")) or "download"
    folder = default_dir or default_download_dir()
    chosen = chooser(os.path.join(folder, name))
    if not chosen:
        return None
    chosen = os.path.abspath(os.path.expanduser(str(chosen)))
    return os.path.dirname(chosen), os.path.basename(chosen)


def default_download_dir() -> str:
    loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DownloadLocation)
    return loc or str(Path.home() / "Downloads")


def qt_file_chooser(parent=None):
    """The default chooser: a native "Save as" dialog."""
    def choose(default_path: str) -> str:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(parent, "Save", default_path)
        return path
    return choose


def handle_download(request, chooser) -> bool:
    """``QWebEngineProfile.downloadRequested``: ask where to save (``chooser``), then accept or cancel.
    The map's Export JSON / coverage CSV / SCORM zip are blob downloads."""
    got = choose_download_path(request.downloadFileName(), chooser)
    if got is None:
        request.cancel()
        return False
    folder, name = got
    request.setDownloadDirectory(folder)
    request.setDownloadFileName(name)
    request.accept()
    return True


def permission_decision(permission_type_name: str, origin: QtCore.QUrl) -> bool:
    """Grant only the clipboard, and only to the map's own origin ``kmap://map``; everything else is denied."""
    return (permission_type_name == "ClipboardReadWrite" and origin.scheme().lower() == SCHEME
            and origin.host().lower() == HOST)


def node_hash(node_id: str, tab: str = "learn") -> str:
    """Route of a node in the map page: ``map/<id>/<tab>`` (the page also knows ``book/<edition>/<id>`` and
    ``<id>``)."""
    return f"map/{node_id}/{tab}" if tab else f"map/{node_id}"


def inject_source(edition: str, platform: str | None = None) -> str:
    """JavaScript run at document creation in the main frame: what the map page needs to know about its host."""
    info = {"desktop": True, "platform": platform or sys.platform, "edition": edition, "embedded": True}
    return ("(function(){window.kmapApp=Object.freeze(" + json.dumps(info) + ");"
            "window.KMAP_PYODIDE_BASE='pyodide/';})();")


def profile_dir() -> Path | None:
    """Where the map's web storage lives (None = off the record).

    ``--screenshot`` mode (ephemeral settings): None. ``MRSTUDIO_SETTINGS=/dir/file.ini`` (tests, portable use):
    ``/dir/kmap-web``. Otherwise MR Studio's per-user data folder (``~/Library/Application Support/MRStudio/MR
    Studio`` on macOS, ``%LOCALAPPDATA%\\MRStudio\\MR Studio`` on Windows) + ``kmap-web``."""
    from . import app as _app
    if _app._SETTINGS["ephemeral"]:
        return None
    env = os.environ.get("MRSTUDIO_SETTINGS", "").strip()
    if env:
        return Path(env).expanduser().resolve().parent / PROFILE_FOLDER
    loc = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppLocalDataLocation)
    return (Path(loc) if loc else Path.home() / ".mrstudio") / PROFILE_FOLDER


class FakeJob(QtCore.QObject):
    """Stand-in for ``QWebEngineUrlRequestJob`` (``--self-test`` and tests): records what ``serve`` answers."""

    def __init__(self, url: str, method: bytes = b"GET"):
        super().__init__()
        self._url, self._method = QtCore.QUrl(url), method
        self.error = None
        self.mime = None
        self.data = None

    def requestUrl(self):
        return self._url

    def requestMethod(self):
        return QtCore.QByteArray(self._method)

    def fail(self, error):
        self.error = error

    def reply(self, mime, device):
        self.mime = bytes(mime).decode()
        self.data = bytes(device.readAll())


FAKE_ERRORS = types.SimpleNamespace(UrlNotFound="UrlNotFound", RequestDenied="RequestDenied",
                                    RequestFailed="RequestFailed")


def serve_check(root, rel: str) -> FakeJob:
    """Serve ``kmap://map/<rel>`` from ``root`` through ``serve`` (no browser engine needed)."""
    job = FakeJob(f"{SCHEME}://{HOST}/{rel}")
    serve(job, root, FAKE_ERRORS)
    return job


class InputWatcher(QtCore.QObject):
    """Event filter that calls ``callback()`` for real user input (mouse buttons, keys, touch) on the objects it is
    installed on; it never consumes an event."""
    TYPES = frozenset({QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease,
                       QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.TouchBegin, QtCore.QEvent.Type.TouchEnd})

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback
        self._seen: set = set()

    def eventFilter(self, obj, event):
        if event.type() in self.TYPES:
            self._callback()
        return False

    def watch(self, widget) -> None:
        """Install on ``widget``, its focus proxy and its child widgets (Qt WebEngine receives input there)."""
        for w in [widget, widget.focusProxy(), *widget.findChildren(QtWidgets.QWidget)]:
            if w is not None and id(w) not in self._seen:
                self._seen.add(id(w))
                w.installEventFilter(self)


class UrlSink(QtCore.QObject):
    """Receiver for ``QDesktopServices.setUrlHandler(SCHEME, sink, "handle")``: map links clicked anywhere in MR
    Studio (theory footers) go to ``callback(QUrl)`` instead of the operating system."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    @QtCore.Slot(QtCore.QUrl)
    def handle(self, url):
        self._callback(QtCore.QUrl(url))


# ----------------------------------------------------------------------------- WebEngine classes (lazy)
@lru_cache(maxsize=1)
def _classes():
    """The QtWebEngine subclasses, defined on first use (importing QtWebEngine is optional and slow)."""
    from PySide6.QtWebEngineCore import (QWebEnginePage, QWebEngineUrlRequestInterceptor, QWebEngineUrlRequestJob,
                                         QWebEngineUrlSchemeHandler)

    class MapSchemeHandler(QWebEngineUrlSchemeHandler):
        """Serves ``kmap://map/<path>`` from the map folder."""

        def __init__(self, root, parent=None):
            super().__init__(parent)
            self.root = Path(root).resolve()

        def requestStarted(self, job):
            serve(job, self.root, QWebEngineUrlRequestJob.Error)

    class LaunchSchemeHandler(QWebEngineUrlSchemeHandler):
        """``mrstudio:`` requests that are not navigations (fetch, subframes): refused, never handed to the OS."""

        def requestStarted(self, job):
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)

    class Interceptor(QWebEngineUrlRequestInterceptor):
        """The map is offline: block every request that is not kmap:/data:/blob:/about: (CDN fallbacks included)."""

        def interceptRequest(self, info):
            if info.requestUrl().scheme().lower() not in ALLOWED_SCHEMES:
                info.block(True)

    class MapPage(QWebEnginePage):
        """The map page: navigation policy, permissions, new windows, printing."""
        def __init__(self, profile, parent, on_launch, on_external, on_print=None):
            super().__init__(profile, parent)
            self._on_launch, self._on_external, self._on_print = on_launch, on_external, on_print
            self._input_at = -1e9
            self.permissionRequested.connect(self._permission)
            self.newWindowRequested.connect(self._new_window)
            self.printRequested.connect(self._print)

        def note_input(self):
            """The user pressed a mouse button / key in the view (``InputWatcher``)."""
            self._input_at = time.monotonic()

        def acceptNavigationRequest(self, url, nav_type, is_main_frame):
            return route_navigation(
                url, self._on_launch, self._on_external, main_frame=is_main_frame,
                clicked=(nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked
                         and time.monotonic() - self._input_at < INPUT_GRACE_S),
                scripted=nav_type in (QWebEnginePage.NavigationType.NavigationTypeLinkClicked,
                                      QWebEnginePage.NavigationType.NavigationTypeOther,
                                      QWebEnginePage.NavigationType.NavigationTypeTyped))

        def _permission(self, permission):
            """Clipboard for the map, nothing else (camera, microphone, location, notifications ...)."""
            if permission_decision(permission.permissionType().name, permission.origin()):
                permission.grant()
            else:
                permission.deny()

        def _new_window(self, request):
            """``target=_blank`` / ``window.open``: never a window. Only a user-initiated request is routed (a map
            page opens in this tab, mrstudio:// and web links as for a click); a script's popup is dropped."""
            if not request.isUserInitiated():
                return
            url = request.requestedUrl()
            if classify_navigation(url) == "allow" and url.scheme().lower() not in ("about", "data", "blob"):
                self.setUrl(url)
            else:
                route_navigation(url, self._on_launch, self._on_external, clicked=True)

        def _print(self):
            if self._on_print is not None:
                self._on_print()

    return types.SimpleNamespace(MapSchemeHandler=MapSchemeHandler, LaunchSchemeHandler=LaunchSchemeHandler,
                                 Interceptor=Interceptor, MapPage=MapPage)


def make_profile(root, edition: str, parent=None, download_chooser=None):
    """The map's QWebEngineProfile: named and persistent under ``profile_dir()`` (off the record without one),
    with the kmap/mrstudio scheme handlers, the offline interceptor, the ``window.kmapApp`` script and a "Save as"
    prompt for downloads (``download_chooser(default_path) -> path or ''``; tests inject one)."""
    from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineProfileBuilder, QWebEngineScript
    cls = _classes()
    pdir = profile_dir()
    if pdir is None:
        profile = QWebEngineProfile(parent)                     # off the record (--screenshot)
    else:
        b = QWebEngineProfileBuilder()
        b.setPersistentStoragePath(str(pdir / "storage"))
        b.setCachePath(str(pdir / "cache"))
        b.setPersistentPermissionsPolicy(QWebEngineProfile.PersistentPermissionsPolicy.AskEveryTime)
        profile = b.createProfile(PROFILE_NAME, parent)
    chooser = download_chooser or qt_file_chooser(parent if isinstance(parent, QtWidgets.QWidget) else None)
    profile.downloadRequested.connect(lambda request: handle_download(request, chooser))
    profile.installUrlSchemeHandler(SCHEME.encode(), cls.MapSchemeHandler(root, profile))
    profile.installUrlSchemeHandler(LAUNCH_SCHEME.encode(), cls.LaunchSchemeHandler(profile))
    profile.setUrlRequestInterceptor(cls.Interceptor(profile))
    scr = QWebEngineScript()
    scr.setName("kmapApp")
    scr.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    scr.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    scr.setRunsOnSubFrames(False)
    scr.setSourceCode(inject_source(edition))
    profile.scripts().insert(scr)
    return profile


# ----------------------------------------------------------------------------- the tab
class KmapTab(QtWidgets.QWidget):
    """Last tab of the main window: the Knowledge Map page. Not in ``app.TABS`` / ``kmap_links.TAB_ORDER`` (tab
    indexes, launch URLs and the screenshot count of the standard build stay as they are)."""
    title = TAB_TITLE
    uses_serial_robot = False
    uses_engine = False
    robot_box_tip = "The Knowledge Map does not use the robot selector"

    def __init__(self, root, on_launch, on_log=None, edition: str | None = None, parent=None,
                 download_chooser=None, pdf_chooser=None):
        super().__init__(parent)
        from ..kmap_links import app_edition
        self.root = Path(root)
        self.edition = edition or app_edition()
        self._on_launch_cb = on_launch
        self._download_chooser, self._pdf_chooser = download_chooser, pdf_chooser
        self._log = on_log or (lambda _text: None)
        self.view = self.page = self.profile = None
        self._watcher = None
        self.loaded = False
        self.pending_hash: str | None = None
        self._rejected_at = -1e9
        self._stack = QtWidgets.QStackedLayout(self)
        self.placeholder = QtWidgets.QLabel("The Robotics Knowledge Map opens here.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self._stack.addWidget(self.placeholder)

    # -- BaseTab-like hooks used by MainWindow
    def on_show(self):
        self.ensure_view()

    def on_hide(self):
        pass

    def ensure_view(self) -> bool:
        """Create the web view, profile and page (once) and start loading the map."""
        if self.view is not None:
            return True
        if prepared_root() is None:
            self.placeholder.setText("The Knowledge Map needs Qt WebEngine, which could not be started.\n"
                                     + last_error())
            return False
        try:
            self._build()
        except Exception as e:  # noqa: BLE001 - an optional tab must never break the app
            import traceback
            self._log(f"Knowledge Map tab could not start:\n{traceback.format_exc()}")
            self.placeholder.setText(f"The Knowledge Map could not start ({type(e).__name__}: {e}).")
            self.view = None
            return False
        return True

    def _build(self):
        from PySide6.QtWebEngineWidgets import QWebEngineView
        cls = _classes()
        view = QWebEngineView(self)                 # created before the profile: Qt deletes children in order,
        self.profile = make_profile(self.root, self.edition, self, self._download_chooser)    # so the page goes before its profile
        page = cls.MapPage(self.profile, view, self._launch, self._external, self._print_page)
        page.setBackgroundColor(self.palette().color(QtGui.QPalette.ColorRole.Window))
        page.loadFinished.connect(self._load_finished)
        page.renderProcessTerminated.connect(self._render_died)
        view.setPage(page)
        self._watcher = InputWatcher(page.note_input, view)
        self._watcher.watch(view)
        self.view, self.page = view, page
        self._stack.addWidget(view)
        self._stack.setCurrentWidget(view)
        view.load(QtCore.QUrl(START_URL))

    # -- navigation
    def _launch(self, text: str):
        self._rejected_at = time.monotonic()
        QtCore.QTimer.singleShot(0, lambda: self._on_launch_cb(text))   # not from inside the navigation callback

    def _external(self, url: QtCore.QUrl):
        self._rejected_at = time.monotonic()
        QtCore.QTimer.singleShot(0, lambda: QtGui.QDesktopServices.openUrl(url))

    def _print_page(self):
        """``window.print()`` / the map's print buttons: Qt WebEngine has no print preview, so save a PDF."""
        if self.page is None:
            return
        title = (self.page.title() or "Knowledge Map").strip()
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in title)[:80].strip() or "Knowledge Map"
        chooser = self._pdf_chooser or qt_file_chooser(self)
        got = choose_download_path(safe + ".pdf", chooser)
        if got is None:
            return
        self.page.printToPdf(os.path.join(*got))
        self._log(f"Knowledge Map: saving a PDF to {os.path.join(*got)}")

    def show_hash(self, fragment: str) -> bool:
        """Show a route of the map (``map/<id>/learn`` …): at once if the page is loaded, else after loading."""
        frag = (fragment or "").lstrip("#")
        if not self.ensure_view():
            return False
        if self.loaded:
            self.page.runJavaScript(f"location.hash = {json.dumps('#' + frag)};")
        else:
            self.pending_hash = frag
        return True

    def _load_finished(self, ok: bool):
        if self.view is not None and self._watcher is not None:
            self._watcher.watch(self.view)          # the web content widget may appear only after the first load
        if ok:
            self.loaded = True
            if self.pending_hash is not None:
                frag, self.pending_hash = self.pending_hash, None
                self.page.runJavaScript(f"location.hash = {json.dumps('#' + frag)};")
            return
        if time.monotonic() - self._rejected_at < REJECT_GRACE_S:
            return              # the "failed load" of a link we handed to MR Studio / the browser
        if not self.loaded:
            self._log(f"Knowledge Map: {START_URL} did not load (map folder {self.root})")

    def _render_died(self, status, code):
        self._log(f"Knowledge Map: the page's render process ended ({status}, exit code {code}); reloading")
        self.loaded = False
        QtCore.QTimer.singleShot(500, lambda: self.view is not None and self.view.reload())

    def shutdown(self):
        """Delete the page before its profile (Qt warns, and may crash, otherwise)."""
        view, profile = self.view, self.profile
        self.view = self.page = self.profile = None
        self._watcher = None
        self.loaded = False
        if view is None:
            return
        from shiboken6 import delete, isValid
        try:
            if isValid(view):
                self._stack.removeWidget(view)
                delete(view)            # deletes the page too (its child)
            if profile is not None and isValid(profile):
                delete(profile)
        except RuntimeError:
            pass
