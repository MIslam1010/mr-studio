# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for MR Studio -- one-folder build:
#   macOS   -> packaging/dist/MR Studio.app   (build_macos.sh wraps it in a .dmg)
#   Windows -> packaging/dist/MRStudio/MRStudio.exe   (build_windows.ps1 zips it + Inno Setup installer)
#
#   pyinstaller --noconfirm --clean --distpath packaging/dist --workpath packaging/build packaging/mrstudio.spec
#
# What to bundle is listed in bundle_config.py (plain Python, checked by tests/test_packaging.py).
import os
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

SPEC_DIR = os.path.abspath(SPECPATH)  # noqa: F821  (SPECPATH is defined by PyInstaller)
ROOT = os.path.dirname(SPEC_DIR)
for p in (ROOT, SPEC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bundle_config as cfg  # noqa: E402

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

# ---------------------------------------------------------------- hidden imports
hiddenimports = list(cfg.HIDDEN_IMPORTS) + cfg.optional_hidden_imports() + ["bundle_config"]
for pkg in cfg.COLLECT_ALL_SUBMODULES:
    hiddenimports += collect_submodules(pkg, on_error="warn once")
if not IS_WIN:
    hiddenimports.remove("OpenGL.platform.win32")
if not IS_MAC:
    hiddenimports.remove("OpenGL.platform.darwin")

# ---------------------------------------------------------------- data files
datas = cfg.all_package_data()
datas += [(os.path.join(ROOT, f), ".") for f in cfg.EXTRA_FILES]   # THIRD_PARTY_NOTICES.md
for dist in cfg.METADATA_DISTS:
    try:
        datas += copy_metadata(dist)
    except Exception:   # PackageNotFoundError: e.g. Pinocchio from conda-forge on Windows is not "pin"
        if dist not in cfg.METADATA_OPTIONAL:
            raise
        print(f"mrstudio.spec: no metadata for {dist!r} (About box will show '-' for it)")

# ---------------------------------------------------------------- native libraries
binaries = []
if IS_WIN:
    # Pinocchio/eigenpy/coal/boost DLLs of the cmeel wheels -> bundle root (see rthook_mrstudio.py)
    # pip/cmeel layout (if a Windows `pin` wheel ever appears): copy the DLLs of cmeel.prefix to the bundle root.
    # conda-forge layout (what build_windows.ps1 and CI use): the DLLs are in <env>\Library\bin, which must be
    # on PATH so that PyInstaller's binary analysis finds the dependencies of pinocchio's .pyd files.
    dlls = cfg.cmeel_windows_dlls()
    binaries += [(f, ".") for f in dlls]
    dll_dirs = sorted({os.path.dirname(f) for f in dlls})
    conda_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(conda_bin):
        dll_dirs.append(conda_bin)
    if not dll_dirs:
        raise SystemExit("mrstudio.spec: neither cmeel DLLs nor a conda Library\\bin folder found -- "
                         "the frozen app could not import pinocchio (see packaging/README.md, option C)")
    os.environ["PATH"] = os.pathsep.join(dll_dirs + [os.environ.get("PATH", "")])

ICON_PNG = os.path.join(SPEC_DIR, "icon.png")
icon = ICON_PNG if os.path.isfile(ICON_PNG) else None   # PyInstaller converts PNG -> .icns/.ico (Pillow)

a = Analysis(  # noqa: F821
    [os.path.join(SPEC_DIR, "launcher.py")],
    pathex=[ROOT, SPEC_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[os.path.join(SPEC_DIR, "rthook_mrstudio.py")],
    excludes=cfg.EXCLUDES,
    noarchive=False,
    optimize=0,          # keep docstrings/asserts: modern_robotics and sympy rely on them
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=cfg.EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX corrupts Qt / OpenGL libraries
    console=False,           # GUI app: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,        # native arch of the build machine (arm64 on Apple-Silicon runners)
    codesign_identity=os.environ.get("CODESIGN_IDENTITY") or None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=cfg.EXE_NAME,
)

if IS_MAC:
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{cfg.APP_NAME}.app",
        icon=icon,
        bundle_identifier=cfg.BUNDLE_ID,
        version=cfg.VERSION,
        info_plist={
            "CFBundleDisplayName": cfg.APP_NAME,
            "CFBundleShortVersionString": cfg.VERSION,
            "CFBundleVersion": cfg.VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSApplicationCategoryType": "public.app-category.education",
            "LSMinimumSystemVersion": "14.0",   # PySide6 6.12 / NumPy 2.4 wheels need macOS 14
        },
    )
