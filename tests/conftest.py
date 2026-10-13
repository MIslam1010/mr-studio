"""Test-session hooks."""
import os
import sys


_STATUS = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """Exit without running interpreter teardown.

    Runs after the terminal summary has been printed. On some systems Qt/OpenGL objects destroyed during Python's
    final garbage collection segfault (exit code 139) after a fully passing run; skipping teardown keeps the real
    pytest exit status.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_STATUS["code"])
