"""Application entry point.
First-run setup, then the GUI (imported only after setup).
"""
import sys

from . import APP_NAME, bootstrap


def main():
    bootstrap.enable_dpi_awareness()
    if not bootstrap.ensure_ready(APP_NAME): ## setup before importing GUI
        sys.exit(1)
    from .gui import run_gui
    run_gui()


if __name__ == "__main__":
    main()
