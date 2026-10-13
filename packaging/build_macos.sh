#!/usr/bin/env bash
# Build the macOS installer of MR Studio:  packaging/dist/MR-Studio-<version>-macOS-<arch>.dmg
#
#   ./packaging/build_macos.sh               # from the project root (or from anywhere)
#
# Options:
#   --no-install   do not pip-install PyInstaller (it is already in the Python used)
#   --skip-tests   do not run the pytest suite before building
#   --locked       install requirements with the exact versions of packaging/requirements-lock.txt
# Environment:
#   PYTHON=/path/to/python     Python 3.11 with MR Studio's requirements (default: .venv/bin/python;
#                              the venv is created with ./setup.sh if it does not exist)
#   CODESIGN_IDENTITY="Developer ID Application: ..."   sign with a real certificate (default: ad-hoc)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NO_INSTALL=0; SKIP_TESTS=0; LOCKED=0
for arg in "$@"; do
  case "$arg" in
    --no-install) NO_INSTALL=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --locked) LOCKED=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo "==> $PY not found: creating the virtual environment with ./setup.sh"
  ./setup.sh
  PY="$ROOT/.venv/bin/python"
fi
"$PY" - <<'EOF'
import sys
if sys.version_info[:2] != (3, 11):
    sys.exit(f"MR Studio is built and tested with Python 3.11, this is {sys.version.split()[0]}")
EOF

if [ "$NO_INSTALL" = 0 ]; then
  echo "==> Installing build tools (PyInstaller)"
  if [ "$LOCKED" = 1 ]; then
    "$PY" -m pip install -r requirements.txt -c packaging/requirements-lock.txt
  fi
  "$PY" -m pip install -r packaging/requirements-build.txt
fi

if [ "$SKIP_TESTS" = 0 ]; then
  echo "==> Running the test suite"
  QT_QPA_PLATFORM=offscreen "$PY" -m pytest -q tests
fi

VERSION="$("$PY" -c 'import sys; sys.path.insert(0, "packaging"); import bundle_config as c; print(c.VERSION)')"
ARCH="$(uname -m)"
DIST="$ROOT/packaging/dist"
APP="$DIST/MR Studio.app"

echo "==> PyInstaller (this takes a few minutes)"
"$PY" -m PyInstaller --noconfirm --clean \
  --distpath "$DIST" --workpath "$ROOT/packaging/build" packaging/mrstudio.spec

echo "==> Self-test of the frozen app"
"$APP/Contents/MacOS/MRStudio" --self-test --report "$DIST/self-test-macOS.txt"

if [ -n "${CODESIGN_IDENTITY:-}" ]; then
  echo "==> Signing with $CODESIGN_IDENTITY"
  codesign --force --deep --options runtime --timestamp --sign "$CODESIGN_IDENTITY" "$APP"
fi
if codesign --verify --deep --strict "$APP"; then echo "    signature OK"; else echo "    warning: codesign --verify failed (the app may still run after right-click -> Open)"; fi

echo "==> Disk image"
DMG="$DIST/MR-Studio-$VERSION-macOS-$ARCH.dmg"
STAGE="$ROOT/packaging/build/dmg"
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "MR Studio" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"

echo
echo "Done:"
echo "  app        $APP"
echo "  installer  $DMG"
echo "  self-test  $DIST/self-test-macOS.txt"
