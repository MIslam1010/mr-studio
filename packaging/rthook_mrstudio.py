# PyInstaller runtime hook: runs inside the frozen app before launcher.py.
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")           # mathtext / SimulateControl only, never a plot window
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")  # do not probe for PyQt5/PyQt6

if sys.platform == "win32":
    _base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    # pinocchio / eigenpy / coal look for their DLLs relative to their own folder, unless told otherwise
    for _var in ("PINOCCHIO_WINDOWS_DLL_PATH", "EIGENPY_WINDOWS_DLL_PATH", "COAL_WINDOWS_DLL_PATH"):
        os.environ.setdefault(_var, _base)
    try:
        os.add_dll_directory(_base)
    except (OSError, AttributeError):
        pass
