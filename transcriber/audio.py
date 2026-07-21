"""System (loopback) audio capture.
Streams WASAPI loopback to a 16 kHz mono .wav and a live queue, never RAM.
"""
import queue
import threading
from pathlib import Path

import numpy as np

## soundcard is imported lazily, on worker threads only: its CoInitializeEx
## forces an MTA apartment that would break Tk's folder-picker dialog.

TARGET_SR = 16000 ## Whisper's native rate; soundcard resamples to it
BLOCK_SECONDS = 0.5 ## how often we pull audio from the device


class Device:
    """A capturable audio source; id is the soundcard device id."""

    def __init__(self, id, name, is_loopback):
        self.id, self.name, self.is_loopback = id, name, is_loopback

    def __repr__(self):
        tag = "loopback" if self.is_loopback else "input"
        return f"<Device {self.name!r} ({tag})>"


def list_loopback_devices():
    """Loopback (system-audio) sources, default speaker first. Worker thread only."""
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
    ## Default-speaker loopback first so the dropdown matches playback; seen dedups.
    seen, ordered = set(), []
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

    Mono/16k blocks stream to wav_path and onto chunk_queue as they arrive. A
    write failure stops capture and leaves the reason in self.error.
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
        self.error = self._writer = self._thread = None
        self._writer_lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        ## Reset state so a recorder instance could be reused.
        self._stop.clear()
        self.error = None
        self.frames_written = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        ## soundcard imported here so its COM switch avoids the UI thread.
        try:
            import soundcard as sc
            mic = sc.get_microphone(self.device.id, include_loopback=True)
            frames_per_block = max(1, int(TARGET_SR * BLOCK_SECONDS))
            with mic.recorder(samplerate=TARGET_SR, channels=None) as rec:
                while not self._stop.is_set():
                    ## record() blocks for a full block, pacing the loop.
                    data = rec.record(numframes=frames_per_block)
                    block = _to_mono(data)
                    if block.size:
                        self._write(block)
                        self.chunk_queue.put(block)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            ## Close the file and signal the consumer even on error.
            self._close_writer()
            self.chunk_queue.put(None) ## sentinel: stream ended

    def _write(self, block):
        if self.wav_path is None:
            self.frames_written += block.size
            return
        with self._writer_lock:
            if self._writer is None:
                import soundfile as sf
                ## Opened lazily so a silent recording leaves no file behind.
                self._writer = sf.SoundFile(
                    str(self.wav_path), mode="w", samplerate=TARGET_SR,
                    channels=1, subtype="PCM_16")
            self._writer.write(block)
            self.frames_written += block.size

    def _close_writer(self):
        ## Locked so a wedged capture thread can't write mid-close.
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
