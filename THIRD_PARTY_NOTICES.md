# Third-party notices

MR Studio's own source code is released under the MIT License (see `LICENSE`).

MR Studio is built on, and the installers (`.dmg`, Windows `setup.exe` and `.zip`) bundle, the open-source components
below. Licences were read from the installed package metadata (`pip show -v`, version in brackets). Each
component remains under its own licence; the full licence texts ship inside the installed package folders
(`*.dist-info/licenses`) of the bundle and at the project links.

| Component | Version | Licence (from package metadata) | Project |
|---|---|---|---|
| Qt for Python (PySide6, shiboken6) and the Qt libraries | 6.12.0 | LGPL-3.0-only (also offered as GPL-2.0-only / GPL-3.0-only) | https://www.qt.io , https://doc.qt.io/qtforpython-6/ |
| Pinocchio (`pin`) | 4.1.0 | BSD-3-Clause | https://github.com/stack-of-tasks/pinocchio |
| Coal (collision library used by Pinocchio) | 3.0.3 | BSD-3-Clause | https://github.com/coal-library/coal |
| EigenPy | 3.13.0 | BSD-2-Clause | https://github.com/stack-of-tasks/eigenpy |
| Boost (cmeel-boost) and other cmeel wheels (assimp, urdfdom, octomap, qhull, tinyxml2, zlib, console_bridge) | see wheels | cmeel-boost: BSL-1.0; others under their own permissive licences, stated in each wheel's `dist-info` | https://github.com/cmake-wheel |
| Robotics Toolbox for Python | 1.4.4 | MIT | https://github.com/petercorke/robotics-toolbox-python |
| Spatial Maths for Python | 1.1.18 | MIT | https://github.com/bdaiinstitute/spatialmath-python |
| rtb-data (example URDFs and meshes, subset) | see wheel | as stated in the wheel | https://github.com/petercorke/rtb-data |
| modern_robotics (Lynch & Park) | 1.1.1 | MIT | https://github.com/NxRLab/ModernRobotics |
| toppra | 0.6.3 | MIT | https://github.com/hungpham2511/toppra |
| NumPy | 2.4.6 | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0-1.0 parts) | https://numpy.org |
| SciPy | 1.17.1 | BSD-3-Clause (classifier: BSD License) | https://scipy.org |
| SymPy | 1.14.0 | BSD | https://www.sympy.org |
| Matplotlib | 3.11.2 | Matplotlib licence (PSF-based; classifier: Python Software Foundation License) | https://matplotlib.org |
| pyqtgraph | 0.14.0 | MIT | https://www.pyqtgraph.org |
| PyOpenGL | 3.1.10 | BSD (classifier: BSD License) | https://mcfletch.github.io/pyopengl/ |
| imageio | 2.38.1 | BSD-2-Clause | https://github.com/imageio/imageio |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause (the wrapper; see FFmpeg below) | https://github.com/imageio/imageio-ffmpeg |
| FFmpeg executable (shipped inside imageio-ffmpeg) | 7.1 | see below | https://ffmpeg.org |
| Pillow | 12.3.0 | MIT-CMU (HPND) | https://python-pillow.github.io |
| pyserial | 3.5 | BSD | https://github.com/pyserial/pyserial |
| Python (interpreter, PyInstaller bootloader) | 3.11 | PSF-2.0; PyInstaller is GPL-2.0-or-later with a special exception that permits bundling any program | https://www.python.org , https://pyinstaller.org |

## Qt / PySide6 (LGPL-3.0)

The app uses Qt for Python under the LGPL-3.0. Qt and PySide6 are **dynamically linked**: they are separate
shared libraries (`PySide6/`, `Qt*.dylib` / `*.dll`) in the application folder, not compiled into the MR Studio
executable, so you may replace them with your own build of the same version. On Windows the folder is the
install directory (or the unzipped portable folder); on macOS, `MR Studio.app/Contents/Frameworks` (and
`Contents/Resources`). The corresponding source code of Qt and PySide6 is available from
https://download.qt.io/official_releases/QtForPython/ and https://code.qt.io/cgit/pyside/pyside-setup.git .
The LGPL-3.0 text is at https://www.gnu.org/licenses/lgpl-3.0.html .

## FFmpeg (GPL build)

The Record button writes MP4 files with the `ffmpeg` command-line program that the `imageio-ffmpeg` wheel
ships as a **separate executable** (`imageio_ffmpeg/binaries/ffmpeg-*`); MR Studio starts it as a child
process and does not link against it. The macOS arm64 build inside the wheel (FFmpeg 7.1) reports `--enable-gpl`
and links libx264/libx265, i.e. it is licensed under the **GNU GPL (v2 or later)**. The Windows x64 binary
(`ffmpeg-win-x86_64-v7.1.exe`) was checked the same way and is also built with `--enable-gpl` and libx264. FFmpeg source code: https://ffmpeg.org/download.html
(release 7.1, https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz); the build recipe of the wheel's binary is
documented at https://github.com/imageio/imageio-binaries and https://github.com/imageio/imageio-ffmpeg .
Anyone who redistributes MR Studio installers redistributes this GPL executable and must keep this notice and
make the source available.

## MR Studio itself

Licence: MIT (see `LICENSE`).
