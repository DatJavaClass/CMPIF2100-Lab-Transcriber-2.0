"""Double-click launcher. Puts the package on sys.path, then runs the app."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcriber.app import main

if __name__ == "__main__":
    main()
