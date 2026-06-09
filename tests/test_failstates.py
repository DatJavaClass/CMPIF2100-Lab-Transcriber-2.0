"""Programmatic fail-state tests for the transcriber engine.

Run directly: python tests/test_failstates.py
These do not need a microphone with sound; loopback silence is enough to
drive the lifecycle. The GPU path needs the medium.en model cached.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcriber import audio, engine
from transcriber.engine import Session, safe_basename

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class Collector:
    def __init__(self):
        self.finished = threading.Event()
        self.finished_count = 0
        self.errors = []
        self.finals = []
        self.partials = 0
        self.lock = threading.Lock()

    def on_partial(self, t):
        with self.lock:
            self.partials += 1

    def on_status(self, m):
        pass

    def on_error(self, m):
        with self.lock:
            self.errors.append(m)

    def on_final(self, p, t):
        with self.lock:
            self.finals.append(Path(p))

    def on_finished(self):
        with self.lock:
            self.finished_count += 1
        self.finished.set()

    def session(self, dest, name, device="__default__"):
        dev = audio.default_loopback_device() if device == "__default__" else device
        return Session(dest, name, device=dev,
                       on_partial=self.on_partial, on_status=self.on_status,
                       on_error=self.on_error, on_final=self.on_final,
                       on_finished=self.on_finished)


def test_safe_basename():
    print("test_safe_basename")
    check("strips illegal chars", safe_basename('Lec 3/4: Q&A?') == "Lec 34 Q&A")
    check("empty falls back", safe_basename("   ") == "Lab Recording")
    check("reserved name guarded", safe_basename("CON").upper().startswith("CON_"))
    check("path separators removed", "\\" not in safe_basename("a\\b") and "/" not in safe_basename("a/b"))
    check("length capped", len(safe_basename("x" * 500)) <= 180)


def test_uniquify_no_overwrite():
    print("test_uniquify_no_overwrite")
    dest = Path(tempfile.mkdtemp())
    (dest / "Lab.wav").write_text("existing")
    (dest / "Lab.txt").write_text("existing")
    s = Session(dest, "Lab")
    s._prepare_destination()
    check("renames to avoid overwrite", s.basename == "Lab (1)", f"got {s.basename!r}")
    check("original wav untouched", (dest / "Lab.wav").read_text() == "existing")


def test_prepare_dest_not_writable():
    print("test_prepare_dest_not_writable")
    # Point dest at an existing FILE so mkdir fails: simulates an unusable path.
    tmp = Path(tempfile.mkdtemp()) / "afile"
    tmp.write_text("x")
    s = Session(tmp, "Lab")
    raised = False
    try:
        s._prepare_destination()
    except Exception:
        raised = True
    check("unusable dest raises", raised)


def test_no_device():
    print("test_no_device")
    saved = audio.default_loopback_device
    audio.default_loopback_device = lambda: None
    try:
        dest = Path(tempfile.mkdtemp())
        c = Collector()
        s = c.session(dest, "NoDevice", device=None)
        s.start()
        ok = c.finished.wait(20)
        check("finished fired", ok)
        check("exactly one finished", c.finished_count == 1, f"count={c.finished_count}")
        check("error reported", len(c.errors) >= 1, str(c.errors[:1]))
        check("not running", s.is_running is False)
    finally:
        audio.default_loopback_device = saved


def test_happy_short():
    print("test_happy_short (records ~12s of loopback silence)")
    dest = Path(tempfile.mkdtemp())
    c = Collector()
    s = c.session(dest, "Short Session")
    s.start()
    time.sleep(12)
    label = s.device_label
    s.stop()
    ok = c.finished.wait(120)
    check("finished fired", ok)
    check("exactly one finished", c.finished_count == 1, f"count={c.finished_count}")
    check("device label set", label in ("GPU", "CPU"), str(label))
    check("wav written", (dest / "Short Session.wav").exists())
    check("txt written", (dest / "Short Session.txt").exists())
    check("no errors", not c.errors, str(c.errors[:1]))
    if (dest / "Short Session.txt").exists():
        body = (dest / "Short Session.txt").read_text(encoding="utf-8")
        check("notice appended", "University of Pittsburgh" in body)


def test_stop_during_load():
    print("test_stop_during_load (stop before model finishes loading)")
    dest = Path(tempfile.mkdtemp())
    c = Collector()
    s = c.session(dest, "Stop During Load")
    s.start()
    time.sleep(1.0)            # well before a cold model load completes
    s.stop()
    ok = c.finished.wait(120)
    check("finished fired", ok)
    check("exactly one finished", c.finished_count == 1, f"count={c.finished_count}")
    check("no crash error", not c.errors or "No audio" in (c.errors[0] if c.errors else ""),
          str(c.errors[:1]))


def test_double_stop_and_start():
    print("test_double_stop_and_start")
    dest = Path(tempfile.mkdtemp())
    c = Collector()
    s = c.session(dest, "Double Ops")
    s.start()
    s.start()                  # second start must be a no-op
    time.sleep(11)
    s.stop()
    s.stop()                   # second stop must be a no-op
    ok = c.finished.wait(120)
    check("finished fired", ok)
    check("exactly one finished despite double stop", c.finished_count == 1,
          f"count={c.finished_count}")


def main():
    test_safe_basename()
    test_uniquify_no_overwrite()
    test_prepare_dest_not_writable()
    test_no_device()
    test_stop_during_load()
    test_double_stop_and_start()
    test_happy_short()
    print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
