"""First-run setup and runtime wiring.

Two jobs:

1. Make sure the packages the app needs are importable. When running as a
   plain script we pip-install anything missing into the user's site-packages.
   When running as a frozen exe the core packages are already bundled, so we
   only ever fetch the big NVIDIA CUDA wheels, and only if a GPU is present.

2. Point the Windows loader at the cuBLAS/cuDNN DLLs so ctranslate2 can find
   them at runtime. CUDA is not on PATH on this kind of machine; the libraries
   arrive as pip wheels and have to be registered with os.add_dll_directory.
"""
import importlib.util
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SUBPROC_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# Core packages, needed on every machine. (import_name, pip_name)
CORE_PACKAGES = [
    ("truststore", "truststore"),
    ("numpy", "numpy"),
    ("soundcard", "soundcard"),
    ("soundfile", "soundfile"),
    ("darkdetect", "darkdetect"),
    ("sv_ttk", "sv-ttk"),
    ("faster_whisper", "faster-whisper"),
]

# GPU-only wheels. Big (~1.3 GB), only fetched when an NVIDIA card is present.
# Versions are pinned to the majors that the bundled ctranslate2 (4.x) is built
# against: cuBLAS 12.x and cuDNN 9.x. An unpinned install grabs latest, which
# can be the wrong cuDNN major and fails to load at transcription time.
# (subdir, pip_name)
GPU_PACKAGES = [
    ("cublas", "nvidia-cublas-cu12>=12,<13"),
    ("cudnn", "nvidia-cudnn-cu12>=9,<10"),
]


def is_frozen() -> bool:
    """True when running from a PyInstaller-built exe."""
    return getattr(sys, "frozen", False)


def gpu_runtime_dir() -> Path:
    """Per-user folder the GPU wheels get installed into for a frozen app.

    A frozen exe can't pip-install into its own bundle, so the CUDA wheels go
    here and this folder is added to the DLL search path at startup.
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "CMPIF2100Transcriber" / "gpu_runtime"


def has_nvidia_gpu() -> bool:
    """True if nvidia-smi is on PATH and exits 0."""
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
            creationflags=SUBPROC_NO_WINDOW,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _spec_exists(import_name: str, extra_paths=None) -> bool:
    # Can this module be imported? extra_paths lets us check a location that
    # isn't on sys.path yet (the per-user GPU runtime) by adding it just for
    # the lookup and restoring sys.path afterward.
    if extra_paths:
        saved = list(sys.path)
        sys.path.extend(str(p) for p in extra_paths)
        try:
            return importlib.util.find_spec(import_name) is not None
        except (ImportError, ValueError):
            return False
        finally:
            sys.path[:] = saved
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _cuda_dlls_present(subdir):
    """True if the cuBLAS/cuDNN bin folder actually has DLLs on disk.

    Checking for files, not just an importable nvidia package, so a half-done
    install (interrupted download, AV quarantine) is treated as missing and
    re-fetched rather than silently failing at transcription time.
    """
    roots = []
    try:
        import nvidia  # type: ignore
        roots.append(Path(nvidia.__path__[0]))
    except ImportError:
        pass
    rt = gpu_runtime_dir() / "nvidia"
    if rt.is_dir():
        roots.append(rt)
    for root in roots:
        bind = root / subdir / "bin"
        if bind.is_dir() and any(bind.glob("*.dll")):
            return True
    return False


def _can_install_gpu():
    """A frozen exe needs an external python to install the GPU wheels.

    Without one there's no way to fetch them, so we stay on CPU rather than
    misfiring the install (running the exe itself as if it were pip).
    """
    if is_frozen():
        return _find_external_python() is not None
    return True


def missing_packages():
    """List of (label, pip_name) that need installing on this machine.

    Core packages are skipped when frozen (they're bundled). GPU packages are
    only considered when an NVIDIA card is present, their DLLs aren't already
    on disk, and we actually have a way to install them.
    """
    needed = []
    if not is_frozen():
        needed += [p for p in CORE_PACKAGES if not _spec_exists(p[0])]

    if os.name == "nt" and has_nvidia_gpu() and _can_install_gpu():
        for subdir, pip_name in GPU_PACKAGES:
            if not _cuda_dlls_present(subdir):
                needed.append((subdir, pip_name))
    return needed


def _pip_cmd(pip_name: str, force=False):
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    is_gpu = pip_name.startswith("nvidia-")
    if is_frozen():
        # The exe has no usable pip; shell out to a real python on PATH.
        py = _find_external_python()
        if py:
            cmd[0] = py
    if force:
        cmd.append("--force-reinstall")
    if is_gpu and is_frozen():
        runtime = gpu_runtime_dir()
        runtime.mkdir(parents=True, exist_ok=True)
        cmd += ["--target", str(runtime)]
    elif sys.prefix == sys.base_prefix:
        # --user keeps us out of an all-users install; pip rejects it in a venv.
        cmd.append("--user")
    cmd.append(pip_name)
    return cmd


def _find_external_python():
    """Locate a system python for the frozen-app install path."""
    for name in ("python", "python3", "py"):
        try:
            r = subprocess.run([name, "--version"], capture_output=True,
                               timeout=5, creationflags=SUBPROC_NO_WINDOW)
            if r.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def register_cuda_dlls() -> bool:
    """Add the cuBLAS/cuDNN DLL folders to the Windows loader search path.

    Looks in both the importable nvidia package and the per-user GPU runtime
    dir (where a frozen app stows the wheels). No-op off Windows or when the
    wheels aren't present.
    """
    if not hasattr(os, "add_dll_directory"):
        return False

    roots = []
    try:
        import nvidia  # type: ignore
        roots.append(Path(nvidia.__path__[0]))
    except ImportError:
        pass
    runtime_nvidia = gpu_runtime_dir() / "nvidia"
    if runtime_nvidia.is_dir():
        roots.append(runtime_nvidia)

    found = False
    for root in roots:
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = root / sub / "bin"
            if d.is_dir():
                try:
                    # add_dll_directory is what the loader actually uses;
                    # prepending PATH as well covers tools that still read it.
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                    found = True
                except OSError:
                    pass
    return found


def inject_truststore():
    """Use the OS cert store for TLS so HuggingFace model downloads work.

    huggingface_hub uses httpx, which ignores SSL_CERT_FILE/certifi. Without
    this, the first model download fails with CERTIFICATE_VERIFY_FAILED.
    """
    try:
        import truststore  # type: ignore
        truststore.inject_into_ssl()
    except Exception:
        pass


def add_runtime_to_path():
    """Make the per-user GPU runtime importable (for the nvidia wheels)."""
    runtime = gpu_runtime_dir()
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))


def install_packages(to_install, status_cb=None, progress_cb=None, force=False):
    """pip-install each (label, pip_name), no UI of its own.

    Calls status_cb(message) and progress_cb(0..100) as it goes. Returns a list
    of (pip_name, error_text) for anything that failed. Safe to run on a worker
    thread. With force=True, reinstalls even if already present.
    """
    import tempfile
    failures = []
    total = max(1, len(to_install))
    for idx, (_label, pip_name) in enumerate(to_install):
        slice_start = (idx / total) * 100
        slice_end = ((idx + 1) / total) * 100
        if status_cb:
            status_cb(f"Installing {pip_name}  ({idx + 1} of {len(to_install)})...")
        log_path = None
        try:
            # pip output goes to a logfile, not a PIPE: a chatty install (the big
            # CUDA wheels) can fill an unread pipe buffer and hang the process.
            fd, log_path = tempfile.mkstemp(suffix=".log", prefix="cmpif_pip_")
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as logf:
                proc = subprocess.Popen(
                    _pip_cmd(pip_name, force=force), stdout=logf,
                    stderr=subprocess.STDOUT, text=True,
                    creationflags=SUBPROC_NO_WINDOW,
                )
                t_start = time.time()
                while proc.poll() is None:
                    if progress_cb:
                        elapsed = time.time() - t_start
                        eased = 1 - math.exp(-elapsed / 25)
                        progress_cb(slice_start + (slice_end - slice_start) * eased * 0.9)
                    time.sleep(0.25)
                rc = proc.wait()
            if rc != 0:
                out = ""
                try:
                    out = Path(log_path).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
                failures.append((pip_name, (out or "").strip()[-600:]
                                 or f"pip exited with code {rc}"))
        except Exception as e:
            failures.append((pip_name, f"{type(e).__name__}: {e}"))
        finally:
            if log_path:
                try:
                    os.unlink(log_path)
                except OSError:
                    pass
        if progress_cb:
            progress_cb(slice_end)
    return failures


def run_install_splash(to_install, app_name="CMPIF2100 Lab Transcriber"):
    """Standalone modal splash (own Tk root) that installs the missing packages.

    Used at startup, before the main GUI exists. Returns the failure list.
    """
    import tkinter as tk
    from tkinter import ttk

    win = tk.Tk()
    win.title(f"{app_name}: First-time setup")
    W, H = 520, 200
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{W}x{H}+{(sw - W) // 2}+{(sh - H) // 2}")
    win.resizable(False, False)
    BG = "#F5F5F7"
    win.configure(bg=BG)

    pad = tk.Frame(win, bg=BG, padx=24, pady=22)
    pad.pack(fill=tk.BOTH, expand=True)
    tk.Label(pad, text="Setting things up...", font=("Segoe UI", 12, "bold"),
             bg=BG, fg="#1C1C1E").pack(anchor="w")
    tk.Label(pad, text="One-time install. The GPU libraries are large and may "
             "take a few minutes.", font=("Segoe UI", 9), bg=BG, fg="#6E6E73",
             wraplength=W - 60, justify="left").pack(anchor="w", pady=(2, 14))

    status_var = tk.StringVar(value="Preparing...")
    tk.Label(pad, textvariable=status_var, font=("Segoe UI", 9), bg=BG,
             fg="#1C1C1E", anchor="w").pack(fill=tk.X)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Install.Horizontal.TProgressbar", background="#4A90E2",
                    troughcolor="#E5E5EA", bordercolor=BG, lightcolor="#4A90E2",
                    darkcolor="#4A90E2", thickness=10)
    bar = ttk.Progressbar(pad, style="Install.Horizontal.TProgressbar",
                          mode="determinate", maximum=100, length=W - 70)
    bar.pack(fill=tk.X, pady=(8, 0))

    failures = []

    def worker():
        failures.extend(install_packages(
            to_install,
            status_cb=lambda m: win.after(0, status_var.set, m),
            progress_cb=lambda v: win.after(0, bar.configure, {"value": v}),
        ))
        win.after(0, win.destroy)

    threading.Thread(target=worker, daemon=True).start()
    win.protocol("WM_DELETE_WINDOW", lambda: None)
    win.mainloop()
    return failures


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def model_cache_dirs():
    """faster-whisper model folders in the HuggingFace cache."""
    bases = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        bases.append(Path(hf_home) / "hub")
    bases.append(Path.home() / ".cache" / "huggingface" / "hub")
    dirs = []
    for b in bases:
        if b.is_dir():
            dirs += [p for p in b.glob("models--*aster-whisper*") if p.is_dir()]
    return dirs


def removable_targets():
    """The large, re-fetchable pieces: GPU runtime + cached Whisper models."""
    targets = []
    rt = gpu_runtime_dir()
    if rt.exists():
        targets.append(rt)
    targets += model_cache_dirs()
    return targets


def removable_size():
    return sum(_dir_size(t) for t in removable_targets())


def remove_dependencies():
    """Delete the GPU runtime and cached models. Returns (removed, freed, locked).

    locked is a list of paths that could not be fully removed because a file
    was in use (a model/DLL loaded by the running app); those clear on restart.
    """
    import shutil
    removed, locked, freed = [], [], 0
    for t in removable_targets():
        size = _dir_size(t)
        shutil.rmtree(t, ignore_errors=True)
        if t.exists():
            locked.append(t)
        else:
            removed.append(t)
            freed += size
    return removed, freed, locked


def reinstall_targets():
    """Packages to (re)install for a repair: wipe the GPU runtime first so the
    CUDA wheels come down clean, then whatever is now missing."""
    import shutil
    shutil.rmtree(gpu_runtime_dir(), ignore_errors=True)
    return missing_packages()


def ensure_ready(app_name="CMPIF2100 Lab Transcriber"):
    """Full startup sequence. Returns True if the app can proceed.

    Installs anything missing (showing a splash if so), then wires up TLS and
    the CUDA DLL search path. On a fatal install failure, shows an error and
    returns False.
    """
    add_runtime_to_path()
    to_install = missing_packages()
    if to_install:
        failures = run_install_splash(to_install, app_name)
        if failures:
            # A failed GPU install is non-fatal: the app still runs on CPU.
            fatal = [(n, e) for n, e in failures if not n.startswith("nvidia-")]
            if fatal:
                try:
                    import tkinter as tk
                    from tkinter import messagebox
                    r = tk.Tk()
                    r.withdraw()
                    messagebox.showerror(
                        f"{app_name}: Setup failed",
                        "Could not install required components:\n\n"
                        + "\n\n".join(f"* {n}\n{e}" for n, e in fatal))
                    r.destroy()
                except Exception:
                    pass
                return False
        add_runtime_to_path()

    inject_truststore()
    register_cuda_dlls()
    return True
