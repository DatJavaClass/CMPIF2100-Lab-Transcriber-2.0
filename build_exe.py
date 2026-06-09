"""Build the portable Windows exe with PyInstaller.

Hybrid bundle: the app and the CPU transcription stack go inside the exe, but
the ~1.3 GB NVIDIA CUDA wheels are deliberately excluded. On a machine with an
NVIDIA card the app fetches those at first run (see transcriber/bootstrap.py),
which keeps this exe small while GPU acceleration still works.

Usage:
    python build_exe.py            # one-file portable exe (default)
    python build_exe.py --onedir   # one-folder build (more robust, easier to debug)
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "CMPIF2100_Lab_Transcriber.pyw"
APP_NAME = "CMPIF2100 Lab Transcriber 2.0"

# Packages whose data files / submodules PyInstaller won't find on its own.
COLLECT_ALL = [
    "faster_whisper",     # includes the bundled Silero VAD asset (vad_filter=True)
    "ctranslate2",
    "soundcard",
    "cffi",               # soundcard's WASAPI backend is cffi-based
    "comtypes",           # soundcard uses COM on Windows
    "soundfile",
    "sv_ttk",
    "darkdetect",
    "truststore",
    "tokenizers",
    "huggingface_hub",
    "onnxruntime",
]

HIDDEN_IMPORTS = [
    "transcriber",
    "transcriber.app",
    "transcriber.gui",
    "transcriber.engine",
    "transcriber.audio",
    "transcriber.bootstrap",
]

# Kept out on purpose: GPU wheels (fetched at runtime) and heavy unused libs.
EXCLUDES = [
    "nvidia",
    "torch",
    "matplotlib",
    "scipy",
    "pandas",
    "PIL",
    "tkinter.test",
]


def main():
    onedir = "--onedir" in sys.argv

    # Start from a clean slate so stale artifacts can't leak into the build.
    for d in ("build", "dist"):
        p = ROOT / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    # --windowed: no console window. collect-all pulls each tricky package's
    # data files and submodules; the excludes keep the GPU wheels and other
    # unused heavyweights out of the bundle.
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", APP_NAME,
        "--onefile" if not onedir else "--onedir",
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for mod in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", mod]
    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    cmd.append(str(ENTRY))

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("\nBuild failed.", file=sys.stderr)
        sys.exit(rc)

    out = ROOT / "dist" / (APP_NAME + (".exe" if not onedir else ""))
    print(f"\nBuild complete. Output under: {ROOT / 'dist'}")
    if out.exists():
        print(f"Portable exe: {out}")


if __name__ == "__main__":
    main()
