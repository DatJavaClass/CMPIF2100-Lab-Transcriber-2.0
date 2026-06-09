"""Application entry point.

Runs first-run setup (dependency install, TLS, CUDA DLL wiring), then opens
the GUI. Importing the GUI is deferred until after setup so the packages it
needs are guaranteed present.
"""
import os
import sys

from . import APP_NAME
from . import bootstrap


def _enable_hidpi():
    # Tell Windows we handle scaling ourselves so the window is crisp on a
    # high-DPI display instead of bitmap-stretched. The newer per-monitor call
    # is preferred; fall back to the older system-wide one on older Windows.
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    _enable_hidpi()
    # Do setup before importing the GUI so its dependencies are guaranteed to
    # be present. A False return means a required component could not install.
    if not bootstrap.ensure_ready(APP_NAME):
        sys.exit(1)
    from .gui import run_gui
    run_gui()


if __name__ == "__main__":
    main()
