"""Tkinter GUI: Sun Valley theme, follows the Windows light/dark setting.

Session callbacks arrive on worker threads; _post marshals them onto the
Tk thread before any widget is touched.
"""
import os
import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import darkdetect
import sv_ttk

from transcriber.engine import Session
from transcriber import audio, bootstrap

WINDOW_TITLE = "CMPIF2100 Lab Transcriber"
VERSION = "2.0"
SLACK_URL = "https://pitt-mds.slack.com/archives/C07HW9Y3DBR"
DEFAULT_SIZE = "820x620"
MIN_SIZE = (640, 480)

ABOUT_README = (
    "Records your computer's audio (a Zoom or Teams call, a recorded lecture, "
    "anything playing through your speakers) and transcribes it to text.\n\n"
    "How to use it:\n"
    "1. Pick your system audio source from the dropdown.\n"
    "2. Choose a save folder and a file name.\n"
    "3. Click Record and play the audio you want captured.\n"
    "4. Watch the transcript appear live. Click Stop when you are done.\n"
    "5. A recording (.wav) and a clean transcript (.txt) are saved to your "
    "folder. Use Open Transcript to read it.\n\n"
    "Transcription runs locally on your machine, on the GPU when one is "
    "available, otherwise on the CPU."
)

MIT_LICENSE = (
    "MIT License\n\n"
    "Copyright (c) 2026 Victor S.\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a "
    "copy of this software and associated documentation files (the "
    "\"Software\"), to deal in the Software without restriction, including "
    "without limitation the rights to use, copy, modify, merge, publish, "
    "distribute, sublicense, and/or sell copies of the Software, and to permit "
    "persons to whom the Software is furnished to do so, subject to the "
    "following conditions:\n\n"
    "The above copyright notice and this permission notice shall be included "
    "in all copies or substantial portions of the Software.\n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS "
    "OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF "
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN "
    "NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, "
    "DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR "
    "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE "
    "USE OR OTHER DEALINGS IN THE SOFTWARE."
)

## Transcript pane colors per theme; ttk has no Text widget.
TEXT_COLORS = {
    "dark": {"bg": "#1c1c1c", "fg": "#e8e8e8", "insert": "#e8e8e8"},
    "light": {"bg": "#ffffff", "fg": "#1c1c1c", "insert": "#1c1c1c"},
}


def _current_theme():
    return sv_ttk.get_theme() if sv_ttk.get_theme() in ("dark", "light") else "light"


def _default_dest_dir():
    desktop = Path.home() / "Desktop"
    return desktop if desktop.is_dir() else Path.home()


def _default_basename():
    from datetime import datetime ## lazy; module import stays side-effect free
    return "Lab Recording " + datetime.now().strftime("%Y-%m-%d %H%M")


class TranscriberWindow:
    def __init__(self, root):
        self.root = root
        self.session, self.devices = None, []
        self.dest_dir, self.last_transcript = _default_dest_dir(), None
        self._busy = False ## a deps operation is running
        self._closed = False

        root.title(WINDOW_TITLE + " " + VERSION)
        root.geometry(DEFAULT_SIZE)
        root.minsize(*MIN_SIZE)

        ## Menu bar, then each row, top to bottom.
        self._build_menubar()
        self._build_header()
        self._build_device_row()
        self._build_save_row()
        self._build_record_row()
        self._build_transcript()
        self._build_status()

        self._refresh_devices() ## first scan, runs on a worker
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    ## Widget construction.

    def _build_menubar(self):
        menubar = tk.Menu(self.root)

        menu = tk.Menu(menubar, tearoff=0)
        menu.add_command(label="Re-import Dependencies", command=self._reimport_deps)
        menu.add_command(label="Remove Dependencies", command=self._remove_deps)
        menu.add_separator()
        menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="Menu", menu=menu)

        about = tk.Menu(menubar, tearoff=0)
        about.add_command(label="About " + WINDOW_TITLE, command=self._show_about)
        menubar.add_cascade(label="About", menu=about)

        self.root.configure(menu=menubar)

    def _build_header(self):
        header = ttk.Frame(self.root, padding=(16, 14, 16, 6))
        header.pack(fill="x")
        title = ttk.Label(header, text=WINDOW_TITLE,
                          font=("Segoe UI Semibold", 18))
        title.pack(side="left")
        badge = ttk.Label(header, text="2.0", font=("Segoe UI Semibold", 10),
                         padding=(8, 2))
        badge.pack(side="left", padx=(10, 0), pady=(8, 0))

    def _build_device_row(self):
        row = ttk.Frame(self.root, padding=(16, 6))
        row.pack(fill="x")
        ttk.Label(row, text="System audio source").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(row, textvariable=self.device_var,
                                         state="readonly", width=44)
        self.device_combo.pack(side="left", padx=(10, 8))
        self.refresh_btn = ttk.Button(row, text="Refresh",
                                      command=self._refresh_devices)
        self.refresh_btn.pack(side="left")

    def _build_save_row(self):
        row = ttk.Frame(self.root, padding=(16, 6))
        row.pack(fill="x")
        ttk.Label(row, text="Save to").pack(side="left")
        self.dest_var = tk.StringVar(value=str(self.dest_dir))
        self.dest_entry = ttk.Entry(row, textvariable=self.dest_var,
                                    state="readonly")
        self.dest_entry.pack(side="left", fill="x", expand=True, padx=(10, 8))
        self.change_btn = ttk.Button(row, text="Change...",
                                     command=self._choose_dir)
        self.change_btn.pack(side="left")

        name_row = ttk.Frame(self.root, padding=(16, 0, 16, 6))
        name_row.pack(fill="x")
        ttk.Label(name_row, text="File name").pack(side="left")
        self.name_var = tk.StringVar(value=_default_basename())
        self.name_entry = ttk.Entry(name_row, textvariable=self.name_var)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(10, 8))
        self.open_btn = ttk.Button(name_row, text="Open Transcript",
                                   command=self._open_transcript)
        self.open_btn.pack(side="left")
        self.open_btn.state(["disabled"]) ## enabled once a transcript exists

    def _build_record_row(self):
        row = ttk.Frame(self.root, padding=(16, 8))
        row.pack(fill="x")
        self.record_btn = ttk.Button(row, text="Record", style="Accent.TButton",
                                     width=16, command=self._toggle_record)
        self.record_btn.pack(side="left")

        self.indicator = ttk.Label(row, text="", font=("Segoe UI Semibold", 11))
        self.indicator.pack(side="left", padx=(14, 0))

    def _build_transcript(self):
        ## Plain tk.Text; _apply_text_colors themes it by hand.
        wrap = ttk.Frame(self.root, padding=(16, 6))
        wrap.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(wrap, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.transcript = tk.Text(
            wrap, wrap="word", relief="flat", borderwidth=0,
            font=("Segoe UI", 11), padx=12, pady=10,
            yscrollcommand=scroll.set, state="disabled",
            highlightthickness=0,
        )
        self.transcript.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self.transcript.yview)
        self.transcript.tag_configure("marker", foreground="#7aa2f7",
                                      font=("Segoe UI Italic", 10))
        self._apply_text_colors()

    def _build_status(self):
        bar = ttk.Frame(self.root, padding=(16, 6, 16, 12))
        bar.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

    def _apply_text_colors(self):
        colors = TEXT_COLORS.get(_current_theme(), TEXT_COLORS["light"])
        self.transcript.configure(
            background=colors["bg"], foreground=colors["fg"],
            insertbackground=colors["insert"],
            selectbackground="#3a6ea5", selectforeground="#ffffff",
        )

    ## Device list and save location.

    def _refresh_devices(self):
        ## Worker thread: soundcard's COM init breaks the folder picker.
        self.refresh_btn.state(["disabled"])
        self.status_var.set("Looking for audio devices...")
        threading.Thread(target=self._enumerate_devices, daemon=True).start()

    def _enumerate_devices(self):
        try:
            devs, err = audio.list_loopback_devices(), None
        except Exception as e:
            devs, err = [], str(e)
        self._post(self._apply_devices, devs, err)

    def _apply_devices(self, devices, err):
        ## Back on the UI thread; Refresh stays off while recording.
        self.devices = devices
        running = bool(self.session and self.session.is_running)
        if not running:
            self.refresh_btn.state(["!disabled"])
        if err:
            self.status_var.set(f"Could not list devices: {err}")

        names = [d.name for d in self.devices]
        self.device_combo.configure(values=names)
        if names:
            if self.device_var.get() not in names:
                self.device_combo.current(0)
            self.record_btn.state(["!disabled"])
            if not running and not err:
                self.status_var.set("Ready.")
        else:
            self.device_var.set("")
            self.record_btn.state(["disabled"])
            if not err:
                self.status_var.set(
                    "No system-audio device found. Enable an output device in "
                    "Windows sound settings, then Refresh.")

    def _selected_device(self):
        name = self.device_var.get()
        for d in self.devices:
            if d.name == name:
                return d
        return None

    def _choose_dir(self):
        chosen = filedialog.askdirectory(
            initialdir=str(self.dest_dir), title="Choose a save folder")
        if chosen:
            self.dest_dir = Path(chosen)
            self.dest_var.set(str(self.dest_dir))

    ## Recording.

    def _toggle_record(self):
        ## One button, both jobs.
        if self.session and self.session.is_running:
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self):
        device = self._selected_device()
        if device is None:
            messagebox.showerror(WINDOW_TITLE, "Select a system-audio source first.")
            return
        if not self.dest_dir.is_dir():
            messagebox.showerror(
                WINDOW_TITLE, "The save folder does not exist. Pick another one.")
            return
        basename = self.name_var.get().strip() or _default_basename()

        self._set_transcript("")
        self.session = Session(
            self.dest_dir, basename, device=device,
            on_partial=self._cb(self._set_transcript),
            on_status=self._cb(self.status_var.set),
            on_error=self._cb(self._handle_error),
            on_final=self._cb(self._handle_final),
            on_finished=self._cb(self._handle_finished),
        )
        self.session.start()

        ## Recording mode; _handle_finished flips it back.
        self.record_btn.configure(text="Stop")
        self._set_inputs_enabled(False)
        self.indicator.configure(text="REC", foreground="#e64545")
        self.progress.start(12)

    def _stop_session(self):
        if not self.session:
            return
        self.session.stop()
        self.record_btn.configure(text="Finishing...")
        self.record_btn.state(["disabled"])

    def _set_inputs_enabled(self, enabled):
        self.device_combo.configure(state="readonly" if enabled else "disabled")
        for widget in (self.refresh_btn, self.name_entry, self.change_btn):
            widget.state(["!disabled"] if enabled else ["disabled"])

    ## Transcript pane: kept read-only, flipped writable just long enough.

    def _set_transcript(self, text):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        if text:
            self.transcript.insert("1.0", text)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _append_marker(self, text):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "\n\n" + text, "marker")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    ## Callbacks land off-thread; _post hops to Tk.

    def _post(self, fn, *args):
        if self._closed:
            return ## window already torn down
        try:
            self.root.after(0, fn, *args)
        except tk.TclError:
            pass

    def _cb(self, fn):
        ## Marshal a handler onto the Tk thread.
        return lambda *a: self._post(fn, *a)

    def _handle_error(self, msg):
        self.status_var.set(msg)
        messagebox.showerror(WINDOW_TITLE, msg)

    def _handle_final(self, txt_path, clean_text):
        ## Show the name actually used (sanitized, maybe (1)-suffixed).
        self.last_transcript = Path(txt_path)
        self.open_btn.state(["!disabled"])
        self.name_var.set(Path(txt_path).stem)
        self._append_marker(f"[Clean transcript saved: {Path(txt_path).name}]")

    def _open_transcript(self):
        if not self.last_transcript or not self.last_transcript.exists():
            messagebox.showinfo(WINDOW_TITLE, "No transcript has been saved yet.")
            self.open_btn.state(["disabled"])
            return
        try:
            os.startfile(str(self.last_transcript)) ## default text viewer
        except Exception as e:
            messagebox.showerror(WINDOW_TITLE, f"Could not open the transcript:\n{e}")

    def _handle_finished(self):
        ## Fires once per session end; put the controls back.
        self.progress.stop()
        self.indicator.configure(text="")
        self.record_btn.state(["!disabled"])
        self.record_btn.configure(text="Record")
        self._set_inputs_enabled(True)
        if not self.devices:
            self.record_btn.state(["disabled"]) ## devices vanished meanwhile

    ## Menu: re-importing and removing the heavy dependencies.

    def _busy_or_recording(self):
        if self.session and self.session.is_running:
            messagebox.showinfo(WINDOW_TITLE, "Stop the current recording first.")
            return True
        return self._busy

    def _reimport_deps(self):
        if self._busy_or_recording():
            return
        if not messagebox.askyesno(
                WINDOW_TITLE,
                "Re-import the transcription dependencies?\n\n"
                "This reinstalls them as a repair. The GPU libraries are large, "
                "so it can take a few minutes. The app stays open."):
            return
        self._run_deps_job("Re-importing dependencies", self._reimport_worker)

    def _reimport_worker(self, status, progress):
        targets = bootstrap.reinstall_targets()
        if not targets:
            return [], "Dependencies already look complete; nothing to reinstall."
        failures = bootstrap.install_packages(
            targets, status_cb=status, progress_cb=progress, force=True)
        if failures:
            return failures, None
        bootstrap.register_cuda_dlls()
        return [], ("Dependencies reinstalled. The transcription model will "
                    "re-download on next use if needed. A restart is "
                    "recommended so the changes take full effect.")

    def _remove_deps(self):
        if self._busy_or_recording():
            return
        gb = bootstrap.removable_size() / (1024 ** 3)
        if not messagebox.askyesno(
                WINDOW_TITLE,
                f"Remove all downloaded dependencies (about {gb:.1f} GB)?\n\n"
                "This deletes the GPU libraries and the cached transcription "
                "model. They are re-imported automatically the next time you "
                "use the app, which needs an internet connection.\n\nContinue?"):
            return
        self._run_deps_job("Removing dependencies", self._remove_worker,
                           determinate=False)

    def _remove_worker(self, status, progress):
        status("Deleting downloaded files...")
        _removed, freed, locked = bootstrap.remove_dependencies()
        gb = freed / (1024 ** 3)
        msg = f"Removed about {gb:.1f} GB. They will be re-imported on next use."
        if locked:
            msg += ("\n\nSome files are in use by the running app and were not "
                    "deleted. Close and reopen the app, then run Remove "
                    "Dependencies again to finish clearing them.")
        return [], msg

    def _run_deps_job(self, title, work, determinate=True):
        ## Modal progress dialog around work(status, progress) on a worker.
        ## work returns (failures, message); failures is empty on success.
        self._busy = True
        top = tk.Toplevel(self.root)
        top.title(title)
        top.transient(self.root)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", lambda: None)
        frm = ttk.Frame(top, padding=20)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=title + "...",
                  font=("Segoe UI Semibold", 11)).pack(anchor="w")
        sv = tk.StringVar(value="Working...")
        ttk.Label(frm, textvariable=sv, width=54).pack(anchor="w", pady=(6, 10))
        bar = ttk.Progressbar(frm, length=400, maximum=100,
                              mode="determinate" if determinate else "indeterminate")
        bar.pack(fill="x")
        if not determinate:
            bar.start(12)
        self._center_on_parent(top)

        def status(m):
            self._post(sv.set, m)

        def progress(v):
            self._post(bar.configure, {"value": v})

        def run():
            try:
                failures, msg = work(status, progress)
            except Exception as e:
                failures, msg = [("operation", f"{type(e).__name__}: {e}")], None
            self._post(self._finish_deps_job, top, failures, msg)

        threading.Thread(target=run, daemon=True).start()

    def _finish_deps_job(self, top, failures, msg):
        self._busy = False
        try:
            top.grab_release()
            top.destroy()
        except tk.TclError:
            pass
        if failures:
            messagebox.showerror(
                WINDOW_TITLE, "Some steps did not complete:\n\n"
                + "\n\n".join(f"* {n}\n{e}" for n, e in failures))
        elif msg:
            messagebox.showinfo(WINDOW_TITLE, msg)

    ## Dialogs: About, license, shared centering.

    def _center_on_parent(self, top):
        ## Modal grab, then center over the main window.
        top.grab_set()
        top.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - top.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - top.winfo_height()) // 2
        top.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _show_about(self):
        top = tk.Toplevel(self.root)
        top.title("About " + WINDOW_TITLE)
        top.transient(self.root)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=20)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=f"{WINDOW_TITLE} {VERSION}",
                  font=("Segoe UI Semibold", 15)).pack(anchor="w")
        ttk.Label(frm, text=ABOUT_README, wraplength=460, justify="left"
                  ).pack(anchor="w", pady=(8, 12))
        ttk.Separator(frm, orient="horizontal").pack(fill="x")

        ttk.Label(frm, text="Licensed under the MIT License. "
                  "Copyright (c) 2026 Victor S.", wraplength=460, justify="left"
                  ).pack(anchor="w", pady=(12, 2))
        ttk.Button(frm, text="View full license",
                   command=self._show_license).pack(anchor="w", pady=(0, 12))

        ## Dressed as a hyperlink; opens the class Slack channel.
        link = ttk.Label(frm, text="Give Vic Mad Props In Slack?",
                         foreground="#3a7bdb", cursor="hand2",
                         font=("Segoe UI", 10, "underline"))
        link.pack(anchor="w")
        link.bind("<Button-1>", lambda _e: webbrowser.open(SLACK_URL))

        ttk.Button(frm, text="Close", command=top.destroy).pack(anchor="e",
                                                                pady=(16, 0))
        self._center_on_parent(top)

    def _show_license(self):
        top = tk.Toplevel(self.root)
        top.title("MIT License")
        top.transient(self.root)
        frm = ttk.Frame(top, padding=16)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, wrap="word", width=72, height=22, relief="flat",
                      font=("Segoe UI", 10), padx=10, pady=10)
        colors = TEXT_COLORS.get(_current_theme(), TEXT_COLORS["light"])
        txt.configure(background=colors["bg"], foreground=colors["fg"])
        txt.insert("1.0", MIT_LICENSE)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        ttk.Button(frm, text="Close", command=top.destroy).pack(anchor="e",
                                                                pady=(12, 0))
        top.grab_set()

    ## Closing the window.

    def _on_close(self):
        ## Live recording: confirm, ask it to stop, then tear down.
        if self.session and self.session.is_running:
            if not messagebox.askyesno(
                    WINDOW_TITLE,
                    "A recording is in progress. Stop it and close?"):
                return
            try:
                self.session.stop()
            except Exception:
                pass
        self._closed = True ## marshaller drops late callbacks
        self.root.destroy()


def run_gui():
    ## Called after bootstrap has installed and wired everything.
    bootstrap.enable_dpi_awareness()
    root = tk.Tk()
    sv_ttk.set_theme(darkdetect.theme() or "light") ## match Windows theme
    TranscriberWindow(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
