"""Live record-and-transcribe session.

A Session ties the loopback recorder to a faster-whisper model. While
recording it transcribes a rolling buffer every few seconds and pushes the
growing text out through a callback, so the GUI can show words as they land.
The recorder streams the audio straight to a .wav. On stop the session runs
one clean full-file pass with the larger model and saves that as the .txt
(with the Pitt notice). The live text is a preview; the saved transcript is
the accurate one.

How the GUI drives a session:

    from transcriber.engine import Session
    from transcriber import audio

    audio.list_loopback_devices() -> [Device]      # for a device dropdown
    audio.default_loopback_device() -> Device | None

    s = Session(dest_dir, basename, device=None,
                on_partial=fn(text:str),
                on_status=fn(msg:str),
                on_error=fn(msg:str),
                on_final=fn(txt_path:Path, clean_text:str),
                on_finished=fn())            # always fires when a session ends
    s.start()
    s.stop()             # returns immediately; final pass runs in background
    s.is_running         # bool
    s.device_label       # "GPU" or "CPU" once started

All callbacks are invoked from worker threads. A Tk GUI should marshal them
onto the UI thread with root.after.
"""
import queue
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import audio

COPYRIGHT_NOTICE = (
    "The contents of this transcript are the exclusive intellectual property "
    "of the University of Pittsburgh and intended for personal use only. "
    "They are not to be distributed, shared, sold, or otherwise transmitted "
    "without the express permission of the University."
)

# Live-preview pacing. The buffer is transcribed in small chunks that are
# committed once each (never re-transcribed), so the work per second of audio
# stays flat and the preview can't fall behind. On a GPU we also refresh a
# cheap partial of the not-yet-committed tail for a snappier feel.
LIVE_COMMIT_SECONDS = 5.0     # commit (lock in) a chunk this big
LIVE_PARTIAL_STEP = 1.0       # GPU only: refresh the uncommitted tail this often
LIVE_LOAD_CAP = 20.0          # cap the buffer while the model is still loading

_SR = audio.TARGET_SR
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _noop(*_a, **_k):
    pass


def safe_basename(name, fallback="Lab Recording"):
    """A filename safe on Windows: illegal chars stripped, trailing dots/spaces
    trimmed, reserved device names avoided. Empty input falls back."""
    name = _ILLEGAL.sub("", (name or "")).strip().rstrip(". ")
    if not name:
        return fallback
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} \
        | {f"LPT{i}" for i in range(1, 10)}
    if name.upper() in reserved:
        name = name + "_"
    return name[:180]


def pick_models():
    """Choose (live_model, final_model, device, compute) for this machine.

    GPU: medium.en live and final (it keeps up easily).
    CPU: small.en for the responsive live preview, medium.en for the final
    clean pass.
    """
    from .bootstrap import register_cuda_dlls
    if register_cuda_dlls():
        return ("medium.en", "medium.en", "cuda", "float16")
    return ("small.en", "medium.en", "cpu", "int8")


class Session:
    def __init__(self, dest_dir, basename, device=None,
                 on_partial=None, on_status=None, on_error=None,
                 on_final=None, on_finished=None):
        self.dest_dir = Path(dest_dir)
        self.basename = safe_basename(basename)
        self.device = device
        self.on_partial = on_partial or _noop
        self.on_status = on_status or _noop
        self.on_error = on_error or _noop
        self.on_final = on_final or _noop
        self.on_finished = on_finished or _noop

        self.is_running = False
        self.device_label = None
        self._recorder = None
        self._live_model = None
        self._final_model = None
        self._live_name = None
        self._final_name = None
        self._whisper_device = None
        self._compute = None
        self._consumer = None
        self._committed = ""        # text already locked in for the live view
        self._fast = False          # GPU: cheap enough to show partial previews
        self._stopping = False

        # A session juggles three threads (start worker, capture/consumer,
        # stop worker), so a few primitives keep them honest:
        #   _model_ready  set once the live model is loaded and usable
        #   _model_lock   serialises model loads so two threads never load at once
        #   _finalize     runs at most once, whether reached by Stop or a failure
        #   _finished     fires the on_finished callback exactly once
        self._model_ready = threading.Event()
        self._model_lock = threading.Lock()
        self._finalize_lock = threading.Lock()
        self._finalized = False
        self._finish_lock = threading.Lock()
        self._finished_fired = False

    # Starting, stopping, and the worker that loads the model.

    def start(self):
        # Guarded so a double-click on Record can't launch two sessions.
        if self.is_running:
            return
        self.is_running = True
        self._stopping = False
        self._committed = ""
        self._finalized = False
        self._finished_fired = False
        self._model_ready.clear()
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        try:
            self.on_status("Preparing the save folder...")
            self._prepare_destination()

            self.on_status("Preparing audio device...")
            wav_path = self.dest_dir / (self.basename + ".wav")
            self._recorder = audio.LoopbackRecorder(self.device, wav_path=wav_path)

            # Start capture first so no audio is lost while the model loads
            # (the model download/load can take longer than a short clip).
            self._recorder.start()
            time.sleep(0.2)
            if self._recorder.error:
                raise RuntimeError(self._recorder.error)

            # The consumer buffers audio right away and waits for the model
            # before it starts producing live text.
            self._consumer = threading.Thread(target=self._consume, daemon=True)
            self._consumer.start()

            self.on_status("Loading transcription model...")
            self._resolve_models()
            self._live_model = self._load_model(self._live_name,
                                                cpu_fallback="small.en")
            self._fast = (self._whisper_device == "cuda")
            self._model_ready.set()
            self.on_status(f"Recording and transcribing on {self.device_label}...")
        except Exception as e:
            # No recording happened; report and bail without a final pass.
            self._stopping = True
            self._finalized = True
            if self._recorder is not None:
                self._recorder.stop()
            self.is_running = False
            self.on_error(f"{type(e).__name__}: {e}")
            self._signal_finished()

    def _prepare_destination(self):
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        probe = self.dest_dir / ".cmpif_write_test"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            raise RuntimeError(f"Cannot write to the save folder: {e}")
        self._uniquify()

    def _uniquify(self):
        """Avoid silently overwriting an existing recording with the same name."""
        base, i, cand = self.basename, 1, self.basename
        while ((self.dest_dir / (cand + ".wav")).exists()
               or (self.dest_dir / (cand + ".txt")).exists()):
            cand = f"{base} ({i})"
            i += 1
        self.basename = cand

    def _resolve_models(self):
        if self._final_name is None:
            (self._live_name, self._final_name,
             self._whisper_device, self._compute) = pick_models()
            self.device_label = "GPU" if self._whisper_device == "cuda" else "CPU"

    def _load_model(self, name, cpu_fallback=None):
        from faster_whisper import WhisperModel  # type: ignore
        with self._model_lock:
            try:
                return WhisperModel(name, device=self._whisper_device,
                                    compute_type=self._compute)
            except Exception:
                # GPU can fail even when nvidia-smi works (driver/runtime skew).
                if self._whisper_device == "cuda":
                    self.on_status("GPU unavailable, using CPU...")
                    self._whisper_device = "cpu"
                    self._compute = "int8"
                    self.device_label = "CPU"
                    return WhisperModel(cpu_fallback or name, device="cpu",
                                        compute_type="int8")
                raise

    def stop(self):
        # _stopping marks this as a deliberate stop so the consumer thread does
        # not also treat the stream ending as a capture failure.
        if not self.is_running or self._stopping:
            return
        self._stopping = True
        self.on_status("Finishing up...")
        threading.Thread(target=self._stop_and_finalize, daemon=True).start()

    def _stop_and_finalize(self):
        # Stopping the recorder closes the .wav and unblocks the consumer; then
        # we run the clean pass over the finished file.
        if self._recorder is not None:
            self._recorder.stop()
        self._finalize(None)

    # The live preview: turning captured audio into text as it arrives.

    def _consume(self):
        """Stream captured blocks into the live preview.

        Each ~LIVE_COMMIT_SECONDS chunk is transcribed once and locked into the
        committed text, then dropped, so per-second work stays flat. On a GPU
        the uncommitted tail is also previewed between commits.
        """
        pending = np.empty(0, dtype=np.float32)
        last_partial = time.time()
        q = self._recorder.chunk_queue

        while True:
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                block = _EMPTY
            if block is None:
                break
            if block is not _EMPTY and block.size:
                pending = np.concatenate([pending, block])

            # While the model is still loading, keep capturing but cap the live
            # buffer. The recorder streams every frame to the .wav regardless.
            if not self._model_ready.is_set():
                cap = int(LIVE_LOAD_CAP * _SR)
                if pending.size > cap:
                    pending = pending[-cap:]
                last_partial = time.time()
                continue

            pending_secs = pending.size / _SR
            if pending_secs >= LIVE_COMMIT_SECONDS:
                self._commit(pending)
                pending = np.empty(0, dtype=np.float32)
                last_partial = time.time()
            elif (self._fast and pending_secs >= 1.0
                  and (time.time() - last_partial) >= LIVE_PARTIAL_STEP):
                self._emit_live(self._transcribe_buffer(pending))
                last_partial = time.time()

        if self._model_ready.is_set() and pending.size >= int(0.3 * _SR):
            self._commit(pending)

        # The recorder ending on its own (device unplugged, driver error) while
        # the user hasn't pressed Stop is a capture failure: salvage and report.
        if (self._recorder is not None and self._recorder.error
                and not self._stopping):
            threading.Thread(target=self._finalize,
                             args=(self._recorder.error,), daemon=True).start()

    def _commit(self, chunk):
        # Lock this chunk's text into the running transcript and clear the
        # window. Passing "" to _emit_live shows the committed text on its own.
        text = self._transcribe_buffer(chunk)
        if text:
            self._committed = (self._committed + " " + text).strip()
        self._emit_live("")

    def _transcribe_buffer(self, buf):
        if self._live_model is None:
            return ""
        try:
            segments, _info = self._live_model.transcribe(
                buf, beam_size=1, language="en", vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )
            return " ".join(s.text.strip() for s in segments
                            if s.text and s.text.strip())
        except Exception:
            return ""

    def _emit_live(self, window_text):
        # The preview is always the locked-in text plus the current window, so
        # the GUI can just replace its pane with this string each time.
        self.on_partial((self._committed + " " + window_text).strip())

    # Stopping cleanly and writing the accurate final transcript.

    def _finalize(self, capture_error):
        # The once-only guard: Stop and a mid-recording capture failure can both
        # land here, but only the first gets to run the save and final pass.
        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True

        self.is_running = False
        try:
            if self._recorder is not None:
                self._recorder.stop()
            if (self._consumer is not None
                    and self._consumer is not threading.current_thread()):
                self._consumer.join(timeout=15)

            frames = self._recorder.frames_written if self._recorder else 0
            wav_path = self._recorder.wav_path if self._recorder else None
            if frames == 0 or wav_path is None or not wav_path.exists():
                note = (f"Recording stopped: {capture_error} "
                        if capture_error else "")
                self.on_error(note + "No audio was captured. Nothing to transcribe.")
                return

            if capture_error:
                self.on_status("Recording ended early. Saving what was captured...")
            self.on_status("Producing the final transcript...")
            txt_path, clean_text = self._final_pass(wav_path)
            self.on_final(txt_path, clean_text)
            if capture_error:
                self.on_status(f"Recording ended early ({capture_error}), but "
                               f"{wav_path.name} and {txt_path.name} were saved.")
            else:
                self.on_status(f"Done. Saved {wav_path.name} and {txt_path.name}.")
        except Exception as e:
            self.on_error(f"{type(e).__name__}: {e}")
        finally:
            self.is_running = False
            self._signal_finished()

    def _final_pass(self, wav_path):
        # Resolve models here too: if the user stopped before the live model
        # finished loading, this is the first place the model gets set up.
        self._resolve_models()
        if self._final_model is None:
            # Reuse the live model when it is the same one, otherwise load the
            # larger accuracy model for the saved transcript.
            if self._final_name == self._live_name and self._live_model is not None:
                self._final_model = self._live_model
            else:
                self.on_status("Loading the accuracy model for the final pass...")
                self._final_model = self._load_model(
                    self._final_name, cpu_fallback=self._final_name)

        lines = self._run_final(self._final_model, wav_path)
        body = "\n".join(lines)
        return self._write_transcript(body), body

    def _run_final(self, model, wav_path):
        def go(m):
            segments, _info = m.transcribe(
                str(wav_path), beam_size=5, language="en", vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            return [s.text.strip() for s in segments if s.text and s.text.strip()]
        try:
            return go(model)
        except Exception:
            # A long recording can exhaust GPU memory; finish it on the CPU.
            if self._whisper_device == "cuda":
                self.on_status("GPU ran out of memory; finishing on CPU...")
                from faster_whisper import WhisperModel  # type: ignore
                return go(WhisperModel(self._final_name, device="cpu",
                                       compute_type="int8"))
            raise

    def _write_transcript(self, body):
        content = f"{body}\n\n{'-' * 78}\n{COPYRIGHT_NOTICE}\n"
        txt_path = self.dest_dir / (self.basename + ".txt")
        try:
            txt_path.write_text(content, encoding="utf-8")
            return txt_path
        except OSError:
            # Save folder went away or filled up: keep the transcript anyway.
            fb = Path(tempfile.gettempdir()) / (self.basename + ".txt")
            fb.write_text(content, encoding="utf-8")
            self.on_status(f"Could not write to the chosen folder; saved the "
                           f"transcript to {fb}.")
            return fb

    def _signal_finished(self):
        # Fire on_finished exactly once so the GUI resets its controls a single
        # time, no matter how the session wound down.
        with self._finish_lock:
            if self._finished_fired:
                return
            self._finished_fired = True
        self.on_finished()


# Shared empty array, reused as the "queue timed out, nothing new" marker so we
# are not allocating a throwaway array on every idle tick.
_EMPTY = np.empty(0, dtype=np.float32)
