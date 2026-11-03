"""Static checks of the installer recipe (packaging/): nothing is downloaded and PyInstaller is not run.

They catch what usually breaks a frozen app: a misspelt hidden import, a data file that moved, a missing
distribution metadata, an excluded module that the app actually imports, a broken CI file.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import bundle_config as cfg  # noqa: E402


def _exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


# ----------------------------------------------------------------------------- spec / hooks parse
@pytest.mark.parametrize("path", ["mrstudio.spec", "bundle_config.py", "launcher.py", "rthook_mrstudio.py",
                                  "make_icon.py", "hooks/hook-pinocchio.py", "hooks/hook-roboticstoolbox.py"])
def test_python_files_parse(path):
    ast.parse((PKG / path).read_text(encoding="utf-8"), filename=path)


def test_spec_uses_config_launcher_hooks():
    spec = (PKG / "mrstudio.spec").read_text(encoding="utf-8")
    tree = ast.parse(spec)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"Analysis", "PYZ", "EXE", "COLLECT", "BUNDLE"} <= calls
    for needle in ("launcher.py", "bundle_config", "rthook_mrstudio.py", '"hooks"', "copy_metadata",
                   "cmeel_windows_dlls", "upx=False", "console=False"):
        assert needle in spec, needle
    for f in ("launcher.py", "rthook_mrstudio.py", "icon.png", "hooks"):
        assert (PKG / f).exists(), f


# ----------------------------------------------------------------------------- bundle_config vs the venv
@pytest.mark.parametrize("mod", cfg.HIDDEN_IMPORTS)
def test_hidden_import_exists(mod):
    assert _exists(mod), f"hidden import {mod!r} is not installed in this venv"


@pytest.mark.parametrize("mod", cfg.COLLECT_ALL_SUBMODULES + cfg.SELF_TEST_IMPORTS)
def test_collected_and_self_test_modules_exist(mod):
    assert _exists(mod), mod


def test_hook_hidden_imports_exist():
    for hook in (PKG / "hooks").glob("hook-*.py"):
        tree = ast.parse(hook.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(
                    r"(roboticstoolbox|pinocchio|eigenpy|coal|rtbdata|spatial\w+|numpy)(\.\w+)*", node.value):
                assert _exists(node.value), f"{hook.name}: {node.value}"


@pytest.mark.parametrize("dist", cfg.METADATA_DISTS)
def test_metadata_dist_installed(dist):
    assert metadata.version(dist)


def test_app_library_list_has_metadata_bundled():
    """Every distribution shown in Help -> About must have its metadata copied into the bundle."""
    from mrstudio.gui.app import LIBRARIES
    norm = {d.lower().replace("_", "-") for d in cfg.METADATA_DISTS}
    missing = [dist for _, dist, _ in LIBRARIES if dist.lower().replace("_", "-") not in norm]
    assert not missing, f"add to METADATA_DISTS: {missing}"


def test_data_files_exist_and_are_small():
    data = cfg.all_package_data()
    assert data
    for src, dest in data:
        assert os.path.isfile(src), src
        assert not os.path.isabs(dest) and dest.split(os.sep)[0] in {p for p, _ in cfg.DATA_PACKAGES}
    total = sum(os.path.getsize(s) for s, _ in data)
    assert total < 30e6, f"data files are {total / 1e6:.0f} MB -- did a glob pick up all of rtb-data?"


def test_data_covers_what_the_app_loads():
    """The bundled URDF, the rtb-data example URDFs and the DH-model mesh folders are all included."""
    from mrstudio import urdf_import as U
    dests = {os.path.join(d, os.path.basename(s)) for s, d in cfg.all_package_data()}
    rel = os.path.relpath(U.IIWA7_URDF, str(ROOT))
    assert rel in dests
    rtb_root = cfg.package_dir("rtbdata")
    for label, path in U.example_urdfs().items():
        if path.startswith(rtb_root):
            assert os.path.join("rtbdata", os.path.relpath(path, rtb_root)) in dests, label
    import roboticstoolbox as rtb
    for name in ("Puma560", "Panda", "Stanford", "Cobra600", "UR5"):
        meshdir = getattr(getattr(rtb.models.DH, name)(), "meshdir", None) if hasattr(rtb.models.DH, name) else None
        if meshdir is not None:
            rel = os.path.relpath(str(meshdir), rtb_root)
            assert any(d.startswith(os.path.join("rtbdata", rel)) for d in dests), f"{name}: {rel}"


def test_excludes_do_not_hit_modules_the_app_imports():
    excluded = set(cfg.EXCLUDES)

    def hit(mod):
        parts = mod.split(".")
        return any(".".join(parts[:i]) in excluded for i in range(1, len(parts) + 1))

    for mod in cfg.HIDDEN_IMPORTS + cfg.SELF_TEST_IMPORTS:
        assert not hit(mod), mod
    for py in (ROOT / "mrstudio").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                assert not hit(n), f"{py.relative_to(ROOT)} imports excluded module {n}"


def test_cmeel_dll_helper_runs():
    dlls = cfg.cmeel_windows_dlls()
    assert isinstance(dlls, list)
    if sys.platform != "win32":
        assert dlls == []


# ----------------------------------------------------------------------------- launcher
def test_launcher_options():
    import launcher
    argv = ["--robot", "UR5", "--report", "x.txt", "--self-test"]
    assert launcher._pop_option(argv, "--report", has_value=True) == "x.txt"
    assert launcher._pop_option(argv, "--self-test") is True
    assert launcher._pop_option(argv, "--validate") is None
    assert argv == ["--robot", "UR5"]


def test_launcher_self_test_passes(tmp_path):
    import launcher
    report = tmp_path / "st.txt"
    assert launcher.self_test(str(report)) == 0
    text = report.read_text(encoding="utf-8")
    assert "FAIL" not in text and "Pinocchio == Modern Robotics" in text
    assert "30 frames @ 30 fps" in text      # the MP4 round trip with the recorder's writer settings


def test_recorder_uses_the_shared_writer_settings():
    from mrstudio import video
    assert video.WRITER_KWARGS["codec"] == "libx264" and video.WRITER_KWARGS["pixelformat"] == "yuv420p"
    src = (ROOT / "mrstudio/gui/recorder.py").read_text(encoding="utf-8")
    assert "open_writer(" in src and "libx264" not in src     # one definition, in the Qt-free module
    assert "PySide6" not in (ROOT / "mrstudio/video.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- scripts / CI / docs
@pytest.mark.skipif(sys.platform == "win32", reason="macOS build script; Windows' bash is WSL and cannot read D:\\ paths")
def test_build_macos_script():
    sh = PKG / "build_macos.sh"
    assert os.access(sh, os.X_OK)
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(sh)], check=True)
    text = sh.read_text(encoding="utf-8")
    for needle in ("mrstudio.spec", "--self-test", "hdiutil create", "requirements-build.txt", "set -euo pipefail"):
        assert needle in text, needle


def test_build_windows_script():
    text = (PKG / "build_windows.ps1").read_text(encoding="utf-8")
    for needle in ("mrstudio.spec", "--self-test", "Compress-Archive", "7z", "-tzip", "mrstudio.iss", "-Wait",
                   "$LASTEXITCODE"):
        assert needle in text, needle
    assert text.count("{") == text.count("}") and text.count("(") == text.count(")")
    if shutil.which("pwsh"):  # real parser when PowerShell is available
        cmd = ("$e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
               f"'{PKG / 'build_windows.ps1'}', [ref]$null, [ref]$e) | Out-Null; exit $e.Count")
        subprocess.run(["pwsh", "-NoProfile", "-Command", cmd], check=True)


def test_inno_setup_script_matches_build():
    iss = (PKG / "mrstudio.iss").read_text(encoding="utf-8")
    assert f'#define MyAppExe "{cfg.EXE_NAME}.exe"' in iss
    assert f"dist\\{cfg.EXE_NAME}\\*" in iss
    assert "PrivilegesRequired=lowest" in iss
    out = re.search(r"OutputBaseFilename=(.+)", iss).group(1).replace("{#MyAppVersion}", cfg.VERSION)
    assert out == f"MR-Studio-{cfg.VERSION}-Windows-x64-setup"
    assert "Compression=lzma2/{#LzmaLevel}" in iss and '#define LzmaLevel "max"' in iss
    assert "SolidCompression=yes" in iss and "LZMANumBlockThreads={#LzmaThreads}" in iss


def test_workflow_screenshot_count_matches_tabs():
    """The frozen-app screenshot step counts PNGs: one per tab, so the number must follow the tab list."""
    from mrstudio.kmap_links import TAB_ORDER

    text = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    counts = {int(n) for n in re.findall(r"\$png\.Count -ne (\d+)", text)}
    assert counts == {len(TAB_ORDER)}, f"build.yml expects {counts} screenshots, the app has {len(TAB_ORDER)} tabs"
    assert f"frozen app ({len(TAB_ORDER)} tabs)" in text


def test_workflow_yaml():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))   # PyYAML reads the key `on` as True
    assert {"push", "pull_request", "workflow_dispatch"} <= set(on)
    job = wf["jobs"]["build"]
    assert set(job["strategy"]["matrix"]["os"]) == {"windows-2022", "macos-15"}   # GitHub retired macos-14 on 2026-11-02
    assert job["needs"] == "test" and "pull_request" in job["if"]      # installers: not on every PR
    test_job = wf["jobs"]["test"]
    assert test_job["runs-on"] == "ubuntu-24.04"   # fixed image, not the moving ubuntu-latest
    test_runs = " ".join(s.get("run", "") for s in test_job["steps"])
    assert "pytest" in test_runs and "ruff check" in test_runs and "libegl1" in test_runs
    assert "requirements-dev.txt" in test_runs
    # the GUI tests run on Linux too (Xvfb + Mesa); PySide6 is swapped there (6.12.0's Linux wheel leaks None refs)
    assert "MRSTUDIO_NO_GUI_TESTS" not in str(test_job.get("env", {})) + str([s.get("env") for s in test_job["steps"]])
    assert "xvfb-run" in test_runs and "s/^PySide6==.*/PySide6==" in test_runs
    steps = job["steps"]
    py = [s for s in steps if s.get("uses", "").startswith("actions/setup-python")]
    assert py and str(py[0]["with"]["python-version"]) == "3.11"
    runs = " ".join(s.get("run", "") for s in steps)
    assert "build_windows.ps1" in runs and "build_macos.sh" in runs
    assert "--validate" in runs and "--screenshot" in runs and "validate-" in runs
    up = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
    assert up and "packaging/dist/*.dmg" in up[0]["with"]["path"] and "validate-*.txt" in up[0]["with"]["path"]
    assert "packaging/dist/BUILD-INFO-*.txt" in up[0]["with"]["path"]
    info = [s for s in steps if s.get("name", "").startswith("Build record (BUILD-INFO)")]
    assert info and info[0]["shell"] == "bash" and steps.index(info[0]) < steps.index(up[0])
    assert "BUILD-INFO-${RUNNER_OS}.txt" in info[0]["run"] and "shasum -a 256" in info[0]["run"]
    assert "actions/runs/${{ github.run_id }}" in info[0]["env"]["RUN_URL"]
    # the build job skips only the source-tree runs of the cross-validation suite (the frozen --validate repeats
    # them); the Linux test job runs everything
    build_tests = [s for s in steps if s.get("name") == "Run tests"][0]["run"]
    assert "--deselect tests/test_validation.py::test_all_checks_pass" in build_tests
    assert "--deselect" not in test_runs and "-k " not in test_runs
    release = wf["jobs"]["release"]
    rel = [s for s in release["steps"] if s.get("uses", "").startswith("softprops/action-gh-release")][0]
    assert rel["with"]["generate_release_notes"] is True and rel["with"]["fail_on_unmatched_files"] is True
    assert "SHA256SUMS.txt" in rel["with"]["files"] and "installers/BUILD-INFO-*.txt" in rel["with"]["files"]
    # every action is pinned to a full commit SHA (a tag can be moved; the "# vX.Y.Z" comment says which release)
    for j in wf["jobs"].values():
        for st in j["steps"]:
            if "uses" in st:
                ref = st["uses"].split("@", 1)[1]
                assert re.fullmatch(r"[0-9a-f]{40}", ref), st["uses"]


def test_dependabot_covers_actions_and_pip():
    yaml = pytest.importorskip("yaml")
    cfg_ = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    assert {u["package-ecosystem"] for u in cfg_["updates"]} == {"github-actions", "pip"}
    # ONE pip entry and ONE group for requirements.txt (/) and the lock (/packaging): a bump changes both files in
    # the same pull request (separate entries gave separate pull requests that each failed the lock test)
    pip = [u for u in cfg_["updates"] if u["package-ecosystem"] == "pip"]
    assert len(pip) == 1 and "directory" not in pip[0]
    assert set(pip[0]["directories"]) == {"/", "/packaging"} and len(pip[0]["groups"]) == 1
    ignored = {i["dependency-name"] for i in pip[0]["ignore"]}
    assert {"*", "toppra", "mpmath", "numpy", "scipy", "contourpy"} <= ignored


def test_dependabot_sync_workflow_is_safe():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github/workflows/dependabot-sync.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert set(on) == {"pull_request"}            # never pull_request_target (it would run with a write token)
    assert wf["permissions"] == {}
    check, push = wf["jobs"]["check"], wf["jobs"]["push"]
    assert "github.actor == 'dependabot[bot]'" in check["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in check["if"]
    assert check["permissions"] == {"contents": "read"}
    assert push["needs"] == "check" and push["permissions"] == {"contents": "write", "actions": "write"}
    check_runs = " ".join(s.get("run", "") for s in check["steps"])
    assert "sync_pins.py" in check_runs and "tests/test_packaging.py" in check_runs
    assert all(s.get("with", {}).get("persist-credentials") is False
               for s in check["steps"] if s.get("uses", "").startswith("actions/checkout"))
    push_runs = " ".join(s.get("run", "") for s in push["steps"])
    assert "pip" not in push_runs and "pytest" not in push_runs   # the job with write access runs nothing from the PR
    assert "gh workflow run build.yml" in push_runs and "git apply" in push_runs
    for j in wf["jobs"].values():
        for st in j["steps"]:
            if "uses" in st:
                assert re.fullmatch(r"[0-9a-f]{40}", st["uses"].split("@", 1)[1]), st["uses"]


def test_sync_pins_check_passes():
    """requirements.txt, the lock and THIRD_PARTY_NOTICES.md agree (`python packaging/sync_pins.py` fixes them)."""
    r = subprocess.run([sys.executable, str(PKG / "sync_pins.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_sync_pins_rewrites_and_is_idempotent():
    import sync_pins as sp
    lock = "# comment\nnumpy==2.4.6\r\nPySide6==6.12.0\nshiboken6==6.12.0\nPySide6_Addons==6.12.0\nsix==1.17.0\n"
    wanted = {"numpy": "2.4.7", "pyside6": "6.12.1", "shiboken6": "6.12.1", "pyside6-addons": "6.12.1"}
    new, changes = sp.sync_lock(lock, wanted)
    assert new == ("# comment\nnumpy==2.4.7\r\nPySide6==6.12.1\nshiboken6==6.12.1\nPySide6_Addons==6.12.1\n"
                   "six==1.17.0\n") and len(changes) == 4
    assert sp.sync_lock(new, wanted) == (new, [])
    notices = "| NumPy | 2.4.6 | BSD-3-Clause | https://numpy.org |\n| other | 1.0 | MIT | x |\n"
    out, ch = sp.sync_notices(notices, {"numpy": "2.4.7"})
    assert out == notices.replace("2.4.6", "2.4.7") and len(ch) == 1
    assert sp.sync_notices(out, {"numpy": "2.4.7"}) == (out, [])
    assert set(sp.NOTICE_ROWS.values()) <= set(sp.exact_pins(sp.read(sp.LOCK)))


def test_build_inputs_are_pinned():
    build = (PKG / "requirements-build.txt").read_text(encoding="utf-8")
    for name in ("pyinstaller", "pyinstaller-hooks-contrib"):
        assert re.search(rf"^{name}==\d", build, re.M), f"{name} must be pinned exactly"
    assert cfg.CONDA_ON_WINDOWS["PySide6"] == "pyside6=6.11.*"


def test_version_is_consistent():
    import tomllib
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert proj["version"] == cfg.VERSION
    assert f'version: "{cfg.VERSION}"' in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert subprocess.run([sys.executable, str(PKG / "bundle_config.py"), "version"], capture_output=True,
                          text=True, check=True).stdout.strip() == cfg.VERSION


def test_third_party_notices_are_bundled():
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    assert "THIRD_PARTY_NOTICES.md" in cfg.EXTRA_FILES
    assert "THIRD_PARTY_NOTICES.md" in (PKG / "mrstudio.spec").read_text(encoding="utf-8")
    iss = (PKG / "mrstudio.iss").read_text(encoding="utf-8")
    assert "THIRD_PARTY_NOTICES.md" in iss and "PrivilegesRequiredOverridesAllowed" not in iss
    assert "CloseApplications=yes" in iss and "LicenseFile" not in iss


def test_launcher_log_location_and_faulthandler():
    import launcher
    path = launcher._log_file()
    assert path.endswith("mrstudio.log")
    if sys.platform in ("darwin", "win32"):   # Linux uses ~/.mrstudio (or $XDG_STATE_HOME)
        assert "MR Studio" in path
    assert "faulthandler" in (PKG / "launcher.py").read_text(encoding="utf-8")


def test_lock_file_agrees_with_requirements():
    def pins(path):
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==")
                out[name.strip().lower().replace("_", "-")] = ver.strip()
        return out
    req, lock = pins(ROOT / "requirements.txt"), pins(PKG / "requirements-lock.txt")
    for name, ver in req.items():
        assert lock.get(name) == ver, f"{name}: requirements.txt {ver} vs lock {lock.get(name)}"


def test_readme_has_install_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Installing MR Studio" in readme and "packaging/README.md" in readme
    assert (PKG / "README.md").is_file()


# ----------------------------------------------------------------------------- review additions
def test_cmeel_dll_filter_on_a_windows_layout(tmp_path, monkeypatch):
    """cmeel.prefix sits INSIDE <venv>/Lib/site-packages, so the filter must look at the path relative to it."""
    prefix = tmp_path / "venv" / "Lib" / "site-packages" / "cmeel.prefix"
    pkg = prefix / "Lib" / "site-packages" / "pinocchio"
    pkg.mkdir(parents=True)
    (prefix / "bin").mkdir()
    for f in (prefix / "bin" / "pinocchio_default.dll", prefix / "bin" / "boost_python311.dll",
              pkg / "something.dll"):
        f.write_bytes(b"")
    monkeypatch.setattr(cfg, "package_dir", lambda name: str(pkg))
    got = sorted(os.path.basename(f) for f in cfg.cmeel_windows_dlls())
    assert got == ["boost_python311.dll", "pinocchio_default.dll"]


def test_windows_requirements_drop_conda_packages(tmp_path):
    dst = tmp_path / "req-win.txt"
    dropped = cfg.pip_requirements_without(str(ROOT / "requirements.txt"), str(dst))
    assert sorted(d.split("==")[0].strip() for d in dropped) == ["PySide6", "pin"]
    kept = dst.read_text(encoding="utf-8")
    assert "pin==" not in kept and "modern-robotics==" in kept and "pyside6==" not in kept.lower()
    assert "pinocchio=4.1.0" in cfg.CONDA_ON_WINDOWS.values() and "pyside6=6.11.*" in cfg.CONDA_ON_WINDOWS.values()
    pin_req = re.search(r"^pin==(\S+)", (ROOT / "requirements.txt").read_text(encoding="utf-8"), re.M).group(1)
    assert cfg.CONDA_ON_WINDOWS["pin"] == f"pinocchio={pin_req}", "conda-forge Pinocchio must match requirements.txt"


def test_windows_script_and_ci_use_conda_for_pinocchio():
    ps1 = (PKG / "build_windows.ps1").read_text(encoding="utf-8")
    for spec in cfg.CONDA_ON_WINDOWS.values():
        assert f'"{spec}"' in ps1, spec
    assert "windows-requirements" in ps1 and '"-r", "requirements.txt"' not in ps1
    wf = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    assert "conda-incubator/setup-miniconda" in wf and "windows-requirements" in wf
    out = subprocess.run([sys.executable, str(PKG / "bundle_config.py"), "conda-packages"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == list(cfg.CONDA_ON_WINDOWS.values())


def test_gitignore_keeps_venv_and_build_out_of_git():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for pat in (".venv/", "__pycache__/", "packaging/build/", "packaging/dist/", ".conda-env/"):
        assert pat in gi, pat


def test_optional_hidden_imports_are_only_the_installed_ones():
    got = cfg.optional_hidden_imports()
    assert set(got) <= set(cfg.OPTIONAL_HIDDEN_IMPORTS)
    assert all(_exists(m) for m in got)
    assert "pyserial" in (PKG / "requirements-build.txt").read_text(encoding="utf-8")


def test_macos_minimum_version_matches_pyside6_wheel():
    spec = (PKG / "mrstudio.spec").read_text(encoding="utf-8")
    assert '"LSMinimumSystemVersion": "14.0"' in spec


def test_third_party_notice_versions_match_lock():
    """THIRD_PARTY_NOTICES.md ships inside the installers: its version column must match what is bundled."""
    rows = {"Qt for Python (PySide6, shiboken6) and the Qt libraries": "pyside6", "Pinocchio (`pin`)": "pin",
            "Coal (collision library used by Pinocchio)": "coal", "EigenPy": "eigenpy",
            "Robotics Toolbox for Python": "roboticstoolbox-python", "Spatial Maths for Python": "spatialmath-python",
            "modern_robotics (Lynch & Park)": "modern-robotics", "toppra": "toppra", "NumPy": "numpy",
            "SciPy": "scipy", "SymPy": "sympy", "Matplotlib": "matplotlib", "pyqtgraph": "pyqtgraph",
            "PyOpenGL": "pyopengl", "imageio": "imageio", "imageio-ffmpeg": "imageio-ffmpeg", "Pillow": "pillow",
            "pyserial": "pyserial"}
    lock = {}
    for line in (PKG / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, ver = line.split("==")
            lock[name.strip().lower().replace("_", "-")] = ver.strip()
    table = {}
    for line in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1] in rows:
            table[cells[1]] = cells[2]
    assert set(table) == set(rows), f"notice rows missing: {set(rows) - set(table)}"
    for row, pkg in rows.items():
        assert table[row] == lock[pkg], f"THIRD_PARTY_NOTICES '{row}' says {table[row]}, the lock bundles {lock[pkg]}"
