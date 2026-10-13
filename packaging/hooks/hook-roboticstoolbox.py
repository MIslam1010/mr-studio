# PyInstaller hook for Peter Corke's Robotics Toolbox. MR Studio only needs the DH models and the
# compiled RNE / ETS extensions; the optional graphical back-ends (Swift, bdsim blocks) are not bundled.
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("roboticstoolbox.models.DH") + [
    "roboticstoolbox._frne_c", "roboticstoolbox._fknm_c", "rtbdata", "spatialmath", "spatialgeometry",
]
excludedimports = ["swift", "bdsim", "roboticstoolbox.backends.swift", "roboticstoolbox.blocks"]
