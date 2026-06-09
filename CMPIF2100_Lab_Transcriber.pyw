"""Double-click launcher for CMPIF2100 Lab Transcriber 2.0.

Keeps the package importable when run directly, then hands off to the app
entry point (which does first-run setup before opening the GUI).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcriber.app import main

if __name__ == "__main__":
    main()
