#!/bin/bash
# One-time installation: creates an isolated Python environment in .venv and installs the libraries.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3)
echo "Using $PY"
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Start the app with ./MR_Studio.command (or: .venv/bin/python run_app.py)"
