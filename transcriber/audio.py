"""System (loopback) audio capture.

Records what's playing on the PC (Zoom, Teams, a recorded lecture) using
WASAPI loopback via the soundcard library. soundcard resamples to whatever
rate we ask for, so we ask for 16 kHz directly (Whisper's native rate) and
only have to downmix to mono. Each captured block is streamed straight to the
output .wav and also pushed to a queue for live transcription, so a long
recording never has to be held in RAM.
"""
import queue
import threading
from pathlib import Path

import numpy as np

# soundcard is imported lazily inside the functions below, never at module
# load. Importing it calls CoInitializeEx and forces the calling thread into a
# multithreaded COM apartment; if that happened on the Tk UI thread it would
# break the native folder-picker dialog (which needs a single-threaded
# apartment). All soundcard use therefore happens on worker threads.

TARGET_SR = 16000          # Whisper's native sample rate; soundcard resamples to it.
BLOCK_SECONDS = 0.5        # How often we pull audio from the device.


class Device:
    """A capturable audio source. id is the soundcard device id."""

    def __init__(self, id, name, is_loopback):
        self.id = id
        self.name = name
        self.is_loopback = is_loopback

    def __repr__(self):
        tag = "loopback" if self.is_loopback else "input"
        return f"<Device {self.name!r} ({tag})>"


def list_loopback_devices():
    """Loopback (system-audio) sources, default speaker first.

    Call this from a worker thread, not the UI thread (see the module note).
    """
    import soundcard as sc
    try:
        default_spk = sc.default_speaker()
    except Exception:
        default_spk = None
    try:
        mics = sc.all_microphones(include_loopback=True)
    except Exception:
        mics = []

    loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
    # Put the loopback that matches the current default speaker first so the
    # dropdown defaults to whatever the user is actually hearing, then add the
    # rest. seen guards against listing the same device twice.
    seen = set()
    ordered = []
    if default_spk is not None:
        for m in loopbacks:
            if m.name == default_spk.name and m.id not in seen:
                ordered.append(m)
                seen.add(m.id)
    for m in loopbacks:
        if m.id not in seen:
            ordered.append(m)
            seen.add(m.id)
    return [Device(m.id, m.name, True) for m in ordered]


def default_loopback_device():
    devs = list_loopback_devices()
    return devs[0] if devs else None


def _to_mono(block):
    """Downmix a (frames, channels) float32 block to mono float32."""
    if block.ndim == 2 and block.shape[1] > 1:
        mono = block.mean(axis=1)
    else:
        mono = block.reshape(-1)
    return mono.astype(np.float32, copy=False)


class LoopbackRecorder:
    """Captures a loopback device on a background thread.

    Mono/16k blocks are streamed to `wav_path` as they arrive and also put on
    `chunk_queue` for the live engine. If a write fails (folder removed, disk
    full) the capture stops and the reason is left in `self.error`.
    """

    def __init__(self, device=None, wav_path=None):
        self.device = device or default_loopback_device()
        if self.device is None:
            raise RuntimeError(
                "No system-audio (loopback) device was found. Make sure an "
                "output device is enabled in Windows sound settings.")
        self.wav_path = Path(wav_path) if wav_path else None
        self.chunk_queue = queue.Queue()
        self.frames_written = 0
        self.error = None
        self._writer = None
        self._writer_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        # Reset state so a recorder instance could in principle be reused.
        self._stop.clear()
        self.error = None
        self.frames_written = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # Runs on its own thread. soundcard is imported here, not at module
        # level, so the COM-apartment switch it triggers lands on this thread
        # rather than the UI thread.
        try:
            import soundcard as sc
            mic = sc.get_microphone(self.device.id, include_loopback=True)
            frames_per_block = max(1, int(TARGET_SR * BLOCK_SECONDS))
            with mic.recorder(samplerate=TARGET_SR, channels=None) as rec:
                while not self._stop.is_set():
                    # record() blocks until it has a full block, which paces
                    # the loop without us needing to sleep.
                    data = rec.record(numframes=frames_per_block)
                    block = _to_mono(data)
                    if block.size:
                        self._write(block)
                        self.chunk_queue.put(block)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            # Always close the file and signal the consumer, even on error, so
            # the live side never blocks waiting on a stream that has ended.
            self._close_writer()
            self.chunk_queue.put(None)  # sentinel: stream ended

    def _write(self, block):
        if self.wav_path is None:
            self.frames_written += block.size
            return
        with self._writer_lock:
            if self._writer is None:
                import soundfile as sf
                # Opened lazily so a recording that captured nothing leaves no
                # file behind.
                self._writer = sf.SoundFile(
                    str(self.wav_path), mode="w", samplerate=TARGET_SR,
                    channels=1, subtype="PCM_16")
            self._writer.write(block)
            self.frames_written += block.size

    def _close_writer(self):
        # Locked so a wedged capture thread can't write while we close the file
        # the final pass is about to read.
        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:
                    pass
                self._writer = None

    def stop(self, timeout=5):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._close_writer()
        return self.frames_written
