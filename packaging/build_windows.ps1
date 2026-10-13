<#
Build the Windows installer of MR Studio:
    packaging\dist\MR-Studio-<version>-Windows-x64-setup.exe   (Inno Setup, if installed)
    packaging\dist\MR-Studio-<version>-Windows-x64.zip         (portable folder, always)

    powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

Pinocchio has no Windows wheel on PyPI, so on Windows MR Studio runs in a conda environment:
Pinocchio comes from conda-forge, everything else from pip. Needs Miniforge (or Miniconda) once:
https://conda-forge.org/download/  -- the first run creates the environment in .conda-env\ by itself.

Options:
    -Python <path>   python.exe of an environment that already has MR Studio's requirements
                     (default: .conda-env\python.exe, created with conda if it does not exist)
    -NoInstall       do not pip-install PyInstaller (it is already in the Python used)
    -SkipTests       do not run the pytest suite before building
    -Locked          install requirements with the exact versions of packaging\requirements-lock.txt
#>
param(
    [string]$Python = "",
    [switch]$NoInstall,
    [switch]$SkipTests,
    [switch]$Locked
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"                     # tab titles contain characters such as (1), cp1252 cannot print them

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

$WinReq = Join-Path $Root "packaging\build\requirements-windows.txt"   # requirements.txt without `pin`
$EnvDir = Join-Path $Root ".conda-env"
if (-not $Python) { $Python = Join-Path $EnvDir "python.exe" }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    # CONDA_EXE is set in a Miniforge Prompt / after `conda init`; `conda` itself may be a PowerShell function
    $Conda = $env:CONDA_EXE
    if (-not $Conda) { $Conda = (Get-Command conda.exe, conda.bat -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1).Source }
    if (-not $Conda) {
        throw "conda not found. Install Miniforge (https://conda-forge.org/download/), open a new 'Miniforge Prompt' or PowerShell, and run this script again. (Pinocchio has no Windows wheel on PyPI.)"
    }
    Write-Host "==> Creating the conda environment $EnvDir (Python 3.11 + Pinocchio from conda-forge)"
    $CondaPkgs = @("pinocchio=4.1.0")   # = bundle_config.CONDA_ON_WINDOWS (checked by tests/test_packaging.py)
    Invoke-Checked $Conda (@("create", "-y", "-p", $EnvDir, "-c", "conda-forge", "--override-channels", "python=3.11") + $CondaPkgs)
    $Python = Join-Path $EnvDir "python.exe"
    $NoInstall = $false   # a fresh environment needs the pip requirements
}
Invoke-Checked $Python @("-c", "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 'MR Studio is built and tested with Python 3.11')")

# conda keeps Pinocchio's C++ DLLs in <env>\Library\bin: put it on PATH (tests and PyInstaller's analysis)
$Prefix = (& $Python -c "import sys; print(sys.prefix)").Trim()
foreach ($d in @((Join-Path $Prefix "Library\bin"), $Prefix, (Join-Path $Prefix "Scripts"))) {
    if (Test-Path $d) { $env:PATH = "$d;$env:PATH" }
}

if (-not $NoInstall) {
    Write-Host "==> Installing MR Studio's requirements (pip) and the build tools (PyInstaller)"
    Invoke-Checked $Python @("packaging\bundle_config.py", "windows-requirements", "requirements.txt", $WinReq)
    $Constraints = @()
    if ($Locked) { $Constraints = @("-c", "packaging\requirements-lock.txt") }
    Invoke-Checked $Python (@("-m", "pip", "install", "-r", $WinReq, "-r", "packaging\requirements-build.txt") + $Constraints)
}

if (-not $SkipTests) {
    Write-Host "==> Running the test suite"
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Checked $Python @("-m", "pytest", "-q", "tests")
    Remove-Item Env:\QT_QPA_PLATFORM
}

$Version = (& $Python "packaging\bundle_config.py" version).Trim()
$Dist = Join-Path $Root "packaging\dist"
$Build = Join-Path $Root "packaging\build"
$AppDir = Join-Path $Dist "MRStudio"
$Exe = Join-Path $AppDir "MRStudio.exe"

Write-Host "==> PyInstaller (this takes a few minutes)"
Invoke-Checked $Python @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", $Dist, "--workpath", $Build, "packaging\mrstudio.spec")

Write-Host "==> Self-test of the frozen app"
$Report = Join-Path $Dist "self-test-Windows.txt"
# a windowed exe returns immediately when started from a console: wait for it explicitly
$p = Start-Process -FilePath $Exe -ArgumentList @("--self-test", "--report", "`"$Report`"") -Wait -PassThru -NoNewWindow
if (Test-Path $Report) { Get-Content $Report }
if ($p.ExitCode -ne 0) { throw "self-test of the frozen app failed (exit code $($p.ExitCode)), see $Report" }

Write-Host "==> Portable zip"
$Zip = Join-Path $Dist "MR-Studio-$Version-Windows-x64.zip"
if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path $AppDir -DestinationPath $Zip

Write-Host "==> Installer (Inno Setup)"
$Iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $Iscc) {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $c) { $Iscc = $c; break }
    }
}
if ($Iscc) {
    New-Item -ItemType Directory -Force -Path $Build | Out-Null
    Invoke-Checked $Python @("-c", "from PIL import Image; Image.open(r'packaging\icon.png').save(r'packaging\build\icon.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])")
    Invoke-Checked $Iscc @("/Qp", "/DMyAppVersion=$Version", "packaging\mrstudio.iss")
    $Setup = Join-Path $Dist "MR-Studio-$Version-Windows-x64-setup.exe"
} else {
    Write-Warning "Inno Setup 6 not found (https://jrsoftware.org/isdl.php): only the portable zip was made."
    $Setup = "(not built)"
}

Write-Host ""
Write-Host "Done:"
Write-Host "  app        $Exe"
Write-Host "  installer  $Setup"
Write-Host "  portable   $Zip"
Write-Host "  self-test  $Report"
