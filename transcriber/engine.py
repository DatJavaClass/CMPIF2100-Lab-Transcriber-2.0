"""Live record-and-transcribe session.

Live text is a rolling preview; the saved .txt is one clean full-file
pass on stop. Callbacks fire on worker threads; marshal before Tk use.
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

## Chunks commit once, never re-transcribed; per-second work stays flat.
LIVE_COMMIT_SECONDS = 5.0 ## lock in a chunk this big
LIVE_PARTIAL_STEP = 1.0 ## GPU only: uncommitted tail refresh rate
LIVE_LOAD_CAP = 20.0 ## buffer cap while the model loads

_SR = audio.TARGET_SR
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _noop(*_a, **_k):
    pass


def safe_basename(name, fallback="Lab Recording"):
    """Windows-safe filename; empty input falls back."""
    name = _ILLEGAL.sub("", (name or "")).strip().rstrip(". ")
    if not name:
        return fallback
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} \
        | {f"LPT{i}" for i in range(1, 10)} ## reserved device names, thanks DOS
    if name.upper() in reserved:
        name = name + "_"
    return name[:180]


def pick_models():
    """(live, final, device, compute) for this machine."""
    from .bootstrap import register_cuda_dlls
    if register_cuda_dlls():
        return ("medium.en", "medium.en", "cuda", "float16") ## GPU keeps up live
    return ("small.en", "medium.en", "cpu", "int8") ## small live, medium final


class Session:
    def __init__(self, dest_dir, basename, device=None,
                 on_partial=None, on_status=None, on_error=None,
                 on_final=None, on_finished=None):
        self.dest_dir, self.basename = Path(dest_dir), safe_basename(basename)
        self.device = device
        self.on_partial, self.on_status = on_partial or _noop, on_status or _noop
        self.on_error, self.on_final = on_error or _noop, on_final or _noop
        self.on_finished = on_finished or _noop

        self.is_running, self.device_label = False, None
        self._recorder = self._consumer = None
        self._live_model = self._final_model = None
        self._live_name = self._final_name = None
        self._whisper_device = self._compute = None
        self._committed = "" ## text locked into the live view
        self._fast = False ## GPU: partial previews are cheap
        self._stopping = False

        ## Three threads share a session; keep them honest.
        self._model_ready = threading.Event() ## live model usable
        self._model_lock = threading.Lock() ## one model load at a time
        self._finalize_lock, self._finalized = threading.Lock(), False
        self._finish_lock, self._finished_fired = threading.Lock(), False

    ## Lifecycle: start, stop, model load.

    def start(self):
        if self.is_running:
            return ## double-click guard
        self.is_running, self._stopping = True, False
        self._committed = ""
        self._finalized = self._finished_fired = False
        self._model_ready.clear()
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        try:
            self.on_status("Preparing the save folder...")
            self._prepare_destination()

            self.on_status("Preparing audio device...")
            wav_path = self.dest_dir / (self.basename + ".wav")
            self._recorder = audio.LoopbackRecorder(self.device, wav_path=wav_path)

            ## Capture first; a model load must not cost audio.
            self._recorder.start()
            time.sleep(0.2)
            if self._recorder.error:
                raise RuntimeError(self._recorder.error)

            ## Consumer buffers now, produces text once the model lands.
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
            ## Nothing recorded; report and bail, no final pass.
            self._stopping = self._finalized = True
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
        ## No silent overwrites; append (1), (2), etc.
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
                if self._whisper_device != "cuda":
                    raise
                ## GPU can fail even when nvidia-smi works.
                self.on_status("GPU unavailable, using CPU...")
                self._whisper_device, self._compute = "cpu", "int8"
                self.device_label = "CPU"
                return WhisperModel(cpu_fallback or name, device="cpu",
                                    compute_type="int8")

    def stop(self):
        if not self.is_running or self._stopping:
            return
        self._stopping = True ## deliberate stop, not a capture failure
        self.on_status("Finishing up...")
        threading.Thread(target=self._stop_and_finalize, daemon=True).start()

    def _stop_and_finalize(self):
        ## Recorder stop closes the .wav and unblocks the consumer.
        if self._recorder is not None:
            self._recorder.stop()
        self._finalize(None)

    ## Live preview: captured audio to text.

    def _consume(self):
        pending = np.empty(0, dtype=np.float32)
        last_partial = time.time()
        q = self._recorder.chunk_queue

        while True:
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                block = _EMPTY
            if block is None:
                break ## sentinel: stream ended
            if block is not _EMPTY and block.size:
                pending = np.concatenate([pending, block])

            ## Model still loading: keep capturing, cap the live buffer.
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

        ## Stream died without Stop: capture failure, salvage and report.
        if (self._recorder is not None and self._recorder.error
                and not self._stopping):
            threading.Thread(target=self._finalize,
                             args=(self._recorder.error,), daemon=True).start()

    def _commit(self, chunk):
        ## Lock this chunk's text in; "" clears the window.
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
        ## Committed text plus window; the GUI just swaps its pane.
        self.on_partial((self._committed + " " + window_text).strip())

    ## Shutdown and the accurate final transcript.

    def _finalize(self, capture_error):
        ## Stop and capture failure both land here; first one wins.
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
        self._resolve_models() ## stopped before load: first setup is here
        if self._final_model is None:
            if self._final_name == self._live_name and self._live_model is not None:
                self._final_model = self._live_model ## same model, reuse it
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
            if self._whisper_device != "cuda":
                raise
            ## Long recordings can exhaust VRAM; finish on CPU.
            self.on_status("GPU ran out of memory; finishing on CPU...")
            from faster_whisper import WhisperModel  # type: ignore
            return go(WhisperModel(self._final_name, device="cpu",
                                   compute_type="int8"))

    def _write_transcript(self, body):
        content = f"{body}\n\n{'-' * 78}\n{COPYRIGHT_NOTICE}\n"
        txt_path = self.dest_dir / (self.basename + ".txt")
        try:
            txt_path.write_text(content, encoding="utf-8")
            return txt_path
        except OSError:
            ## Folder vanished or filled; temp keeps the transcript alive.
            fb = Path(tempfile.gettempdir()) / (self.basename + ".txt")
            fb.write_text(content, encoding="utf-8")
            self.on_status(f"Could not write to the chosen folder; saved the "
                           f"transcript to {fb}.")
            return fb

    def _signal_finished(self):
        ## on_finished fires once, however the session wound down.
        with self._finish_lock:
            if self._finished_fired:
                return
            self._finished_fired = True
        self.on_finished()


_EMPTY = np.empty(0, dtype=np.float32) ## idle-tick marker, no throwaway allocs
