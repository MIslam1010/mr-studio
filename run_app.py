"""Launch MR Studio:  .venv/bin/python run_app.py"""
import matplotlib

matplotlib.use("Agg")  # before any pyplot import: equations/figures are rendered offscreen, never in a window

from mrstudio.gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
