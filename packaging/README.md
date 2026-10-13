# Building the MR Studio installers

This folder turns MR Studio into installers that need no Python:

| Platform | Result (in `packaging/dist/`) | What students do |
|---|---|---|
| macOS 14 or later (Apple Silicon) | `MR-Studio-1.0.0-macOS-arm64.dmg` | open the .dmg and drag **MR Studio** onto **Applications** |
| Windows 10/11 (64-bit) | `MR-Studio-1.0.0-Windows-x64-setup.exe` | double-click it: Next, Next, Finish (no administrator rights needed) |
| Windows, no install | `MR-Studio-1.0.0-Windows-x64.zip` | unzip it and double-click `MRStudio\MRStudio.exe` |

PyInstaller freezes Python 3.11 and every library (Pinocchio, Robotics Toolbox, TOPP-RA, PySide6, …)
into one folder. You can only build an installer **on the platform it is for**, so you cannot make the
Windows installer on a Mac. Use GitHub Actions (option A) to get both without owning a PC.

---

## Option A: let GitHub build both installers (recommended)

1. Put the project in a GitHub repository (the folder is not a git repository yet):
   ```bash
   cd ~/Documents/Edu/Lectures/Robotics/MR-Studio
   git init && git add . && git commit -m "MR Studio"
   gh repo create mr-studio --private --source . --push   # or create the repo on github.com and `git push`
   ```
   The project's `.gitignore` keeps `.venv/` (2 GB), `__pycache__/` and the build output out of the repository.
   Check with `git status` before the first commit that `.venv` is not listed.
2. Each push runs `.github/workflows/build.yml` on a Windows and a macOS machine. It installs
   `requirements.txt` (on Windows, Pinocchio comes from conda-forge, see option C), runs the whole test suite, builds the app, runs the frozen app's
   `--self-test`, and makes the installers.
3. Open the repository on github.com, then **Actions**, then the latest run. Download **MR-Studio-Windows** and
   **MR-Studio-macOS** from the *Artifacts* box. Each is a zip that contains the installer and the self-test report.
4. Optional: `git tag v1.0.0 && git push --tags` also attaches the installers to a GitHub *Release*
   page that you can give to students.

A run takes about 20–30 minutes. You can start one by hand with **Actions**, then **build installers**, then **Run workflow**.

## Option B: build the Mac installer on your Mac

From the project root (the `.venv` already has every requirement):

```bash
cd ~/Documents/Edu/Lectures/Robotics/MR-Studio
./packaging/build_macos.sh
```

The script:
1. installs PyInstaller into `.venv` (`pip install -r packaging/requirements-build.txt`);
2. runs `pytest` (add `--skip-tests` to save time);
3. runs `pyinstaller packaging/mrstudio.spec` and makes `packaging/dist/MR Studio.app`;
4. runs `MR Studio.app/Contents/MacOS/MRStudio --self-test` and stops if anything is missing;
5. wraps the app in `packaging/dist/MR-Studio-<version>-macOS-arm64.dmg`.

Try the app before you share it: `open "packaging/dist/MR Studio.app"`.

## Option C: build the Windows installer on a Windows PC

1. Install **Miniforge** (https://conda-forge.org/download/, the Windows x86_64 installer, "just me").
   Pinocchio is not on PyPI for Windows (the `pin` wheels exist only for macOS and Linux), so the Windows build
   gets Pinocchio from conda-forge and everything else from pip. Miniforge brings its own Python; python.org
   Python is not needed.
2. Optional, for the `setup.exe`: install **Inno Setup 6** (https://jrsoftware.org/isdl.php). Without it you
   only get the portable zip.
3. Copy the project folder to the PC (not the `.venv` folder, because a Mac venv does not work on Windows). Then,
   in a **Miniforge Prompt** (Start menu):
   ```powershell
   cd C:\path\to\MR-Studio
   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
   ```
   The first run creates the conda environment `.conda-env\` (Python 3.11 + `pinocchio=4.1.0` from conda-forge)
   and pip-installs the rest of `requirements.txt` (TOPP-RA is compiled from source there, which needs the
   "Desktop development with C++" workload of the free Visual Studio Build Tools). After that it does the
   same steps as the Mac script (tests, PyInstaller, self-test, zip, Inno Setup).

Options of both scripts: `--skip-tests` / `-SkipTests`, `--no-install` / `-NoInstall` (PyInstaller is
already installed), and `--locked` / `-Locked` (install the exact package versions of the tested Mac venv
from `requirements-lock.txt`). The Mac script also reads `PYTHON=/path/to/python`; the Windows script takes `-Python`.

---

## Checking a built app

```bash
"packaging/dist/MR Studio.app/Contents/MacOS/MRStudio" --self-test     # about 3 s, prints PASS/FAIL per item
"packaging/dist/MR Studio.app/Contents/MacOS/MRStudio" --validate      # the full Validation-tab suite
```
On Windows, use `packaging\dist\MRStudio\MRStudio.exe --self-test --report st.txt`. The exe has no
console, so read `st.txt`. Normal options also work: `--robot`, `--tab`, `--projector`, `--screenshot`.

`--self-test` checks the things a frozen app is most likely to lose:
* every library imports (Pinocchio and its C++ libraries, eigenpy, coal, the Robotics Toolbox, TOPP-RA, PySide6/OpenGL);
* Pinocchio and Modern Robotics give the same UR5 inverse dynamics;
* the Corke DH models load, because they need the rtb-data mesh folders;
* the bundled and example URDFs import;
* the theory-panel equations render;
* the ffmpeg binary for the Record button is present;
* every tab module imports.

## Files

| File | Purpose |
|---|---|
| `mrstudio.spec` | PyInstaller recipe: one-folder build and a macOS `.app` bundle with its icon and Info.plist |
| `bundle_config.py` | What to bundle: hidden imports, data files, package metadata and excluded modules. `tests/test_packaging.py` checks it against the venv |
| `launcher.py` | Entry point of the frozen app. Adds `--self-test` and `--validate`, and starts `mrstudio.gui.app.main` otherwise |
| `hooks/hook-pinocchio.py`, `hooks/hook-roboticstoolbox.py` | PyInstaller hooks for Pinocchio (cmeel wheel on macOS, conda-forge on Windows) and the Robotics Toolbox |
| `rthook_mrstudio.py` | Runtime hook: Agg matplotlib backend, PySide6 for pyqtgraph, DLL paths for Pinocchio on Windows |
| `mrstudio.iss` | Inno Setup script for the Windows `setup.exe` (installs per user, with Start-menu and desktop shortcuts and an uninstaller) |
| `build_macos.sh`, `build_windows.ps1` | One-command builds |
| `requirements-build.txt` | PyInstaller (only needed to build) and pyserial (bundled for the Hardware / Haptics serial ports) |
| `requirements-lock.txt` | `pip freeze` of the tested macOS venv (constraints for reproducible builds) |
| `icon.png`, `make_icon.py` | App icon (1024×1024, drawn with Qt). PyInstaller converts it to `.icns` and `.ico` |

What gets bundled:
* **Data files:** the KUKA iiwa URDF in `mrstudio/data`. From rtb-data (146 MB), only the 9 example URDFs and the
  PUMA 560 and Panda STL folders (about 10 MB), because `rtb.models.DH.Puma560()` and `.Panda()` fail without them.
* **Excluded:** Qt WebEngine (about 300 MB), Qt Quick, Qt 3D, Qt Multimedia and the other Qt modules MR Studio
  never uses, plus tkinter and the optional Swift and bdsim back-ends of the toolbox.
* **Expected size:** about 450–600 MB unpacked and 200–250 MB as a `.dmg` or `setup.exe`. Most of it is Qt, SciPy and SymPy.

## Known limitations

* **Unsigned apps.** The first time a student opens the app:
  * On macOS: right-click **MR Studio**, choose **Open**, then **Open** again. Or run `xattr -dr com.apple.quarantine "/Applications/MR Studio.app"`.
  * On Windows SmartScreen: **More info**, then **Run anyway**.

  To sign the Mac app, set `CODESIGN_IDENTITY="Developer ID Application: …"` before you run `build_macos.sh`.
  Notarisation (`xcrun notarytool submit … --wait`, then `xcrun stapler staple`) is not automated.
* **macOS 14 (Sonoma) or later.** The PySide6 6.12 wheel is built for macOS 14, so the app does not start on older macOS.
* **Windows needs conda to build** (not to run): see option C. In the Windows app the About box may show "—" as
  Pinocchio's version, because conda-forge's Pinocchio is not a pip distribution called `pin`.
* **Hardware.** pyserial is bundled; the myCobot SDK (`pymycobot`) only if it was installed when the app was built.
  The installed app cannot `pip install` anything afterwards.
* **Apple Silicon only.** `macos-latest` is an arm64 machine. For Intel Macs, add an Intel runner label (for
  example `macos-15-intel`) to the matrix in `build.yml`. The pinned wheels must exist for x86_64.
* **First launch is slower** (10–20 s) while matplotlib builds its font cache. Later launches take a few seconds.
* **Local PDFs.** The PDF links in *Help → References* point to files on the author's Mac, so they are hidden
  elsewhere. The citations are still shown.
* **Not tested yet.** The installers were not built while these files were written: the spec, hooks and scripts were
  only checked statically (`tests/test_packaging.py`). The first real build is the GitHub Actions run, or
  `./packaging/build_macos.sh`, and its `--self-test` report shows anything still missing.
