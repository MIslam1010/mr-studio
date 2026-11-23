"""What goes into the MR Studio installers -- plain data, no PyInstaller import.

``mrstudio.spec`` reads this file, and ``tests/test_packaging.py`` checks it *statically* against the
current venv (every hidden import resolves, every data file exists, every distribution has metadata),
so a typo is caught by ``pytest`` before anybody spends 10 minutes on a PyInstaller build.
"""
from __future__ import annotations

APP_NAME = "MR Studio"            # window / bundle display name
EXE_NAME = "MRStudio"             # executable and dist folder name (no space: simpler on Windows)
BUNDLE_ID = "org.mrstudio.app"    # macOS CFBundleIdentifier
VERSION = "2026.12.0"

# Packages whose *every* submodule is bundled. MR Studio imports its tabs by name at run time
# (gui/app.py: importlib.import_module), so PyInstaller's static analysis cannot see them.
COLLECT_ALL_SUBMODULES = [
    "mrstudio",
    "roboticstoolbox.models.DH",   # Puma560, Panda, Stanford, Cobra600 (imported via rtb.models.DH.<name>)
]

# Modules imported dynamically or lazily (inside functions / try blocks) by MR Studio or its libraries.
HIDDEN_IMPORTS = [
    # Lynch & Park, dynamics, toolboxes
    "modern_robotics",
    "pinocchio",
    "eigenpy",
    "coal",                         # pinocchio imports it when built WITH_COLLISION
    "roboticstoolbox",
    "spatialmath",
    "spatialgeometry",
    "rtbdata",                      # example URDFs (urdf_import.example_urdfs)
    "toppra",
    "toppra.algorithm",
    "toppra.constraint",
    "toppra.solverwrapper.cy_seidel_solverwrapper",
    "toppra._CythonUtils",
    # numerics
    "scipy.integrate",
    "scipy.optimize",
    "scipy.interpolate",
    "sympy",
    "contourpy",
    # plotting / GUI
    "matplotlib.backends.backend_agg",   # control.py: plt.switch_backend("agg"); mathtext rendering
    "matplotlib.mathtext",
    "pyqtgraph",
    "pyqtgraph.opengl",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",           # pyqtgraph.opengl.GLViewWidget
    "PySide6.QtSvg",
    "PySide6.QtPrintSupport",
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.ctypesarrays",
    "OpenGL.arrays.lists",
    "OpenGL.arrays.numbers",
    "OpenGL.arrays.strings",
    "OpenGL.platform.darwin",            # selected at run time by sys.platform
    "OpenGL.platform.win32",
    # recording (Record button)
    "imageio",
    "imageio.v2",
    "imageio.plugins.ffmpeg",
    "imageio.plugins.pillow",
    "imageio_ffmpeg",
    "PIL.Image",
]

# Optional modules: bundled only when installed in the build environment (PyInstaller cannot bundle what is
# not there). pyserial is in requirements-build.txt so that the installed app can list serial ports for the
# Hardware / Haptics tabs; pymycobot (myCobot SDK) is bundled if the builder installed it.
OPTIONAL_HIDDEN_IMPORTS = [
    "serial", "serial.tools.list_ports",
    "pymycobot",
]

# Pinocchio has NO Windows wheel on PyPI (the `pin` cmeel wheels exist for macOS and Linux only), so on
# Windows it comes from conda-forge. Map: pip distribution in requirements.txt -> conda-forge spec.
# PySide6 also comes from conda-forge on Windows: pip's PySide6 wheels clash with the Qt/ICU/MSVC runtime DLLs that
# conda puts on PATH ("DLL load failed while importing QtCore: The specified procedure could not be found").
CONDA_ON_WINDOWS = {"pin": "pinocchio=4.1.0", "PySide6": "pyside6=6.11.*"}   # conda-forge has no 6.12 yet (Oct 2026); macOS uses PySide6 6.12.0 from PyPI

# Package data. (package, include-globs relative to the package dir). Python sources are never listed.
DATA_PACKAGES = [
    ("mrstudio", ["data/**/*"]),                       # bundled URDF (KUKA iiwa 7), data/kmap/*.json (map links)
    # rtb-data is 146 MB; bundle only what MR Studio touches:
    #  * the 9 example URDFs of File -> Import URDF (~64 kB);
    #  * the STL folders of the DH models we instantiate. rtb.models.DH.Puma560() / .Panda() look their
    #    `meshdir` up with rtb_path_to_datafile() and raise ValueError if it is missing (~10 MB).
    ("rtbdata", ["xacro/**/*.urdf", "meshes/UNIMATE/puma560/*", "meshes/FRANKA-EMIKA/Panda/*"]),
]

# Single files copied to the bundle root (relative to the repository root): licence notices (LGPL/GPL components).
EXTRA_FILES = ["LICENSE", "THIRD_PARTY_NOTICES.md"]

# Distributions whose metadata is read at run time (About box, HTML validation report:
# importlib.metadata.version(...)). Without it the About box shows "—".
METADATA_DISTS = [
    "modern-robotics", "pin", "roboticstoolbox-python", "toppra", "numpy", "scipy", "sympy",
    "matplotlib", "PySide6", "pyqtgraph", "PyOpenGL", "imageio", "imageio-ffmpeg", "pillow",
    "spatialmath-python", "pyserial",
]

# Metadata that may legitimately be absent: conda-forge's pinocchio is not a pip distribution called "pin"
# (the About box then shows "—" for Pinocchio's version; everything else works).
METADATA_OPTIONAL = list(CONDA_ON_WINDOWS)

# Never bundle these (size, or a second Qt binding that would make PyInstaller abort).
EXCLUDES = [
    "tkinter", "_tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "jupyter", "notebook",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends._backend_tk",
    # Qt modules MR Studio never uses (PySide6-WebEngine alone is ~300 MB)
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtWebView",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets", "PySide6.QtQuickControls2",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DExtras", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtGraphs", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtLocation",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSerialBus", "PySide6.QtNfc",
    "PySide6.QtNetworkAuth", "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSql",
    "PySide6.QtTextToSpeech", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtSpatialAudio",
    "PySide6.QtHttpServer",
    # optional backends of the toolboxes that are not installed / not used
    "roboticstoolbox.backends.swift", "roboticstoolbox.blocks", "pinocchio.visualize",
    "robot_descriptions",
]

# Libraries that need a smoke test after freezing (packaging/launcher.py --self-test imports them).
SELF_TEST_IMPORTS = [
    "modern_robotics", "pinocchio", "roboticstoolbox", "toppra", "scipy.integrate", "sympy",
    "matplotlib.mathtext", "PySide6.QtWidgets", "PySide6.QtOpenGLWidgets", "pyqtgraph.opengl",
    "OpenGL.GL", "imageio_ffmpeg", "PIL.Image", "mrstudio.video",
]


# ----------------------------------------------------------------------------- optional: the Knowledge Map
# Set MRSTUDIO_BUNDLE_KMAP=<built map folder> (e.g. the map app's content/student) to put the Robotics Knowledge Map
# into the installer as the "Knowledge Map" tab (mrstudio/gui/kmap_view.py). Local builds only: the public repository
# and its CI never contain or bundle map content. Unset (the default) the build is exactly the standard one.
KMAP_BUNDLE_ENV = "MRSTUDIO_BUNDLE_KMAP"
KMAP_EDITION_ENV = "MRSTUDIO_KMAP_EDITION"   # edition the map must be (default student; checked in build-info.json)
KMAP_LEAK_CHECK_ENV = "MRSTUDIO_KMAP_LEAK_CHECK"   # path of the map pipeline's tests/leak_check.py ("skip": tests only)
KMAP_DEST = "kmap"                           # folder inside the bundle (kmap_links.BUNDLED_KMAP)
# Qt WebEngine and what it needs; in EXCLUDES for the standard build, bundled with the map
WEBENGINE_MODULES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtQuickWidgets", "PySide6.QtPositioning",
]
KMAP_SELF_TEST_IMPORTS = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick"]
KMAP_SKIP_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}


LEAK_CHECK_RESULT = "not run"      # set by leak_check_problems; printed by kmap_bundle (build logs show the gate ran)


def leak_check_problems(root: str, edition: str, environ=None, run=None) -> list[str]:
    """Run the map pipeline's content leak test (``leak_check.py``, found through ``MRSTUDIO_KMAP_LEAK_CHECK``) on a
    *student* map folder; [] when it passes. ``content_problems`` only looks at file and folder names; this one reads
    the data files (answer keys, withheld solutions, exam text). The control folder is the ``instructor`` sibling of
    ``root`` (``content/instructor`` next to ``content/student``) when it exists. Other editions: nothing to check.
    Unset for a student map: a problem (the build stops). ``MRSTUDIO_KMAP_LEAK_CHECK=skip``: no check, honoured only inside pytest
    (``PYTEST_CURRENT_TEST``); a real build refuses it. The outcome is kept in ``LEAK_CHECK_RESULT``."""
    import os
    import subprocess
    import sys
    global LEAK_CHECK_RESULT
    if edition != "student":
        LEAK_CHECK_RESULT = f"not needed ({edition} edition)"
        return []
    env = os.environ if environ is None else environ
    raw = (env.get(KMAP_LEAK_CHECK_ENV) or "").strip()
    if raw.lower() == "skip":
        if "PYTEST_CURRENT_TEST" not in os.environ:      # a real build can never skip the content test
            return [f"{KMAP_LEAK_CHECK_ENV}=skip is only allowed inside the test suite"]
        LEAK_CHECK_RESULT = "SKIPPED (test suite)"
        return []
    if not raw:
        return [f"{KMAP_LEAK_CHECK_ENV} is not set: a student map must pass the pipeline's tests/leak_check.py "
                "before it goes into an installer (set it to the path of that file)"]
    script = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isfile(script):
        return [f"{KMAP_LEAK_CHECK_ENV}={raw!r}: no such file"]
    cmd = [sys.executable, script, root, "--quiet"]
    control = os.path.join(os.path.dirname(os.path.abspath(root)), "instructor")
    if os.path.isdir(control):
        cmd += ["--control", control]
    try:
        r = (run or subprocess.run)(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as e:
        return [f"leak check could not run ({type(e).__name__}: {e})"]
    if r.returncode != 0:
        tail = [ln for ln in ((r.stdout or "") + (r.stderr or "")).splitlines() if ln.strip()][-15:]
        return [f"leak check failed (exit {r.returncode}): " + " | ".join(tail)]
    LEAK_CHECK_RESULT = f"passed ({script}, control {'yes' if '--control' in cmd else 'no'})"
    return []


def kmap_bundle(environ=None) -> dict | None:
    """The map to bundle: None when MRSTUDIO_BUNDLE_KMAP is unset or empty (standard build), else
    ``{"dir": <absolute folder>, "edition": <edition>}``. Refuses (SystemExit, the build stops) a folder without
    index.html, with any x_* file, an "instructor" folder, a symbolic link, a build-info.json whose edition is not
    MRSTUDIO_KMAP_EDITION (default student), or (student edition) a map that fails the pipeline's leak_check.py
    (``leak_check_problems``; the path comes from MRSTUDIO_KMAP_LEAK_CHECK)."""
    import os
    import sys
    env = os.environ if environ is None else environ
    raw = (env.get(KMAP_BUNDLE_ENV) or "").strip()
    if not raw:
        return None
    edition = (env.get(KMAP_EDITION_ENV) or "").strip() or "student"
    root = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(root):
        raise SystemExit(f"{KMAP_BUNDLE_ENV}={raw!r}: no such folder")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from mrstudio.kmap_links import content_problems
    problems = content_problems(root, edition, require_build_info=True)
    real = os.path.realpath(root)
    if real == repo or real.startswith(repo + os.sep):
        problems.append(f"{root} is inside the MR Studio repository (map content must never be committed)")
    if not problems:      # the content test reads every data file: only for a student map that passed the cheap checks
        problems += leak_check_problems(root, edition, env)
    if problems:
        raise SystemExit(f"{KMAP_BUNDLE_ENV}={raw!r}: refusing to bundle this Knowledge Map ({edition} build):\n  - "
                         + "\n  - ".join(problems))
    print(f"Knowledge Map bundle: {root} ({edition}); leak check: {LEAK_CHECK_RESULT}", file=sys.stderr)
    return {"dir": root, "edition": edition}


def excludes_for(kmap: dict | None) -> list[str]:
    """EXCLUDES, without the Qt WebEngine modules when the map is bundled."""
    return list(EXCLUDES) if not kmap else [m for m in EXCLUDES if m not in WEBENGINE_MODULES]


def kmap_hidden_imports(kmap: dict | None) -> list[str]:
    return list(WEBENGINE_MODULES) if kmap else []


def kmap_datas(kmap: dict | None) -> list[tuple[str, str]]:
    """PyInstaller ``datas`` for every file of the map folder, under ``kmap/`` in the bundle ([] without a map)."""
    import os
    if not kmap:
        return []
    root = kmap["dir"]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        dest = KMAP_DEST if rel == "." else os.path.join(KMAP_DEST, rel)
        for f in sorted(filenames):
            if f not in KMAP_SKIP_FILES:
                out.append((os.path.join(dirpath, f), dest))
    return out


# ----------------------------------------------------------------------------- helpers (used by the spec + tests)
def package_dir(pkg: str) -> str:
    """Directory of an installed package, located WITHOUT importing it."""
    import importlib.util
    import os
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"package {pkg!r} is not installed in this Python")
    return os.path.abspath(list(spec.submodule_search_locations)[0])


def package_data(pkg: str, patterns: list[str]) -> list[tuple[str, str]]:
    """PyInstaller `datas` entries (source file, destination folder inside the bundle) for the files
    of ``pkg`` matching ``patterns`` (recursive globs relative to the package directory)."""
    import glob
    import os
    root = package_dir(pkg)
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
            if os.path.isfile(f) and not f.endswith((".py", ".pyc")) and "__pycache__" not in f:
                rel = os.path.relpath(os.path.dirname(f), root)
                dest = pkg if rel == "." else os.path.join(pkg, rel)
                out.append((f, dest))
    return out


def all_package_data() -> list[tuple[str, str]]:
    return [entry for pkg, pats in DATA_PACKAGES for entry in package_data(pkg, pats)]


def cmeel_windows_dlls() -> list[str]:
    """Windows only: the DLLs of the cmeel wheels (pinocchio, eigenpy, coal, boost, urdfdom ...).

    They live in <site-packages>/cmeel.prefix/bin (or /Lib), not next to the .pyd extension modules,
    and pinocchio finds them through a path *relative to its own folder* -- which does not exist
    in a frozen app. The spec therefore copies them to the bundle root, and the runtime hook points
    PINOCCHIO/EIGENPY/COAL_WINDOWS_DLL_PATH there."""
    import glob
    import os
    d = package_dir("pinocchio")
    while os.path.basename(d) != "cmeel.prefix":
        parent = os.path.dirname(d)
        if parent == d:
            return []
        d = parent
    # skip DLLs inside cmeel.prefix/Lib/site-packages (the Python packages themselves). Test the path
    # RELATIVE to cmeel.prefix: cmeel.prefix itself sits in <venv>/Lib/site-packages, so the absolute
    # path of every DLL contains "site-packages".
    return sorted(f for f in glob.glob(os.path.join(d, "**", "*.dll"), recursive=True)
                  if "site-packages" not in os.path.relpath(f, d).replace("\\", "/").lower().split("/"))


def optional_hidden_imports() -> list[str]:
    """The OPTIONAL_HIDDEN_IMPORTS that are installed in this Python."""
    import importlib.util
    out = []
    for m in OPTIONAL_HIDDEN_IMPORTS:
        try:
            if importlib.util.find_spec(m) is not None:
                out.append(m)
        except (ModuleNotFoundError, ValueError):
            pass
    return out


def pip_requirements_without(src: str, dst: str, drop=None) -> list[str]:
    """Copy the requirements file ``src`` to ``dst`` without the distributions in ``drop`` (default: those
    installed from conda-forge on Windows). Returns the dropped lines."""
    import os
    import re
    drop = {d.lower() for d in (CONDA_ON_WINDOWS if drop is None else drop)}
    kept, dropped = [], []
    with open(src, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)", line.split("#")[0])
            name = m.group(1).lower().replace("_", "-") if m else None
            (dropped if name in drop else kept).append(line)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(kept)
    return dropped


if __name__ == "__main__":   # used by build_windows.ps1 and the CI workflow
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "conda-packages":
        print(" ".join(CONDA_ON_WINDOWS.values()))
    elif len(sys.argv) == 4 and sys.argv[1] == "windows-requirements":
        for line in pip_requirements_without(sys.argv[2], sys.argv[3]):
            print("installed from conda-forge instead of pip:", line.strip())
    elif len(sys.argv) == 2 and sys.argv[1] == "version":
        print(VERSION)
    else:
        sys.exit("usage: bundle_config.py conda-packages | version | windows-requirements SRC DST")
