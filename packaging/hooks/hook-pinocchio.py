# PyInstaller hook for Pinocchio installed from the `pin` (cmeel) wheel.
# The C++ extension registers its submodules itself (pinocchio.rpy, .cholesky, ...), and imports
# eigenpy and coal from C++, so static analysis cannot see them.
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

hiddenimports = ["numpy", "eigenpy", "coal"] + collect_submodules(
    "pinocchio", filter=lambda name: not name.startswith("pinocchio.visualize"))
binaries = collect_dynamic_libs("pinocchio")
# macOS/Linux: the shared libraries in cmeel.prefix/lib are found through the extension's
# @loader_path rpath by PyInstaller's binary analysis. Windows: see mrstudio.spec (cmeel DLLs).
excludedimports = ["pinocchio.visualize", "meshcat", "panda3d_viewer", "rerun"]
