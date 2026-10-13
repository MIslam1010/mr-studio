#!/bin/bash
# Double-click in Finder to start MR Studio.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then ./setup.sh; fi
exec .venv/bin/python run_app.py "$@"
