<p align="center"><img width="651" height="701" alt="ChatGPT Image Jun 11, 2026, 12_49_40 PM" src="https://github.com/user-attachments/assets/99da5453-5993-4993-8aac-51c0a4b6510c" /></p>

# CMPIF2100 Lab Transcriber 2.0

Record your computer's audio and turn it into a text transcript, live, on your
own machine. Point it at a Zoom or Teams lecture, a recorded class, or anything
playing through your speakers, press Record, and watch the words appear as they
are spoken. When you stop, it saves both the recording and a clean transcript.

**Windows only.** It captures system audio through WASAPI loopback, which is a
Windows feature.

## This is version 2.0

Version 1 only recorded the audio. To get a transcript out of it, you then
handed that recording to a separate, third-party program to do the
transcription. The old workflow was two tools: one to record, another to
transcribe.

Version 2.0 does the whole job in one place. It records the audio and
transcribes it itself, with the text showing up live while it records. No
second program, no exporting and re-importing files.

## What it does

1. You pick which system-audio source to capture (your speakers or headset).
2. You choose a save folder and a file name.
3. You press Record. Capture starts immediately and the transcript fills in
   live as the audio plays.
4. You press Stop. The app saves the recording as a `.wav`, then makes one
   clean, full-quality pass over the whole recording and saves that as a
   `.txt`. The live text is a fast preview; the saved `.txt` is the accurate
   version.

The University of Pittsburgh copyright notice is appended to the bottom of
every transcript.

## Features

- Records system/loopback audio: Zoom, Teams, recorded lectures, anything
  coming out of the speakers.
- Live transcription shown in real time while recording.
- Saves both the `.wav` recording and a clean `.txt` transcript.
- Runs on your NVIDIA GPU when one is available, otherwise on the CPU. The
  saved transcript quality is the same either way; the GPU is a speed bonus.
- A modern window that follows your Windows light or dark theme.
- An Open Transcript button to read the result the moment it is saved.
- A Menu for managing the downloaded components (see Requirements below).
- An About window with this guide, the license, and a link to reach me.

## Requirements

- Windows 10 or 11, 64-bit.
- A working audio output device (any speaker or headset shows up as a capture
  source).
- For the portable `.exe`: nothing to install in advance. See below.
- To run from source instead: Python 3.10 or newer.
- Optional: an NVIDIA GPU with a current driver, for faster transcription.

### It downloads what it needs, and can remove it again

You do not install the heavy pieces by hand. The first time it needs them, the
app downloads them for you:

- the transcription engine and a speech model (about 1.5 GB), and
- on machines with an NVIDIA card, the GPU acceleration libraries (about
  1.3 GB).

This download happens once. Later launches go straight to the main window.

If you want that space back, open **Menu > Remove Dependencies**. It deletes the
downloaded GPU libraries and the cached model (a few gigabytes) after warning
you, and the app simply re-downloads whatever it needs the next time you use it.
If something ever seems broken, **Menu > Re-import Dependencies** reinstalls
them as a repair.

## Getting it

### Option A: the portable exe (no Python needed)

Download `CMPIF2100 Lab Transcriber 2.0.exe` from the
[Releases](../../releases) page, put it anywhere you like, and double-click it.
The first launch downloads what it needs (see Requirements), then opens the
main window. Future launches open straight away.

### Option B: run from source

1. Install Python 3.10+ from python.org (check "Add python.exe to PATH" in the
   installer).
2. Download or clone this repository.
3. Double-click `CMPIF2100_Lab_Transcriber.pyw`, or run
   `python CMPIF2100_Lab_Transcriber.pyw` from a terminal in the folder. The
   first run installs the Python packages it needs into your user account (no
   admin rights required).

## Building the exe yourself

```
python build_exe.py            # one-file portable exe (default)
python build_exe.py --onedir   # one-folder build, easier to debug
```

The build leaves the large NVIDIA CUDA libraries out of the exe on purpose to
keep it small; the app fetches those on first run instead. If you are building
inside a Dropbox or OneDrive folder and the build fails with a "file is being
used by another process" error, pause syncing and run it again.

## Reading your transcript

The `.txt` is plain text, one segment per line, with the Pitt copyright notice
at the bottom. Open it in Notepad or any editor, or use the Open Transcript
button in the app.

## Troubleshooting

**"No system-audio device found."** Enable an output device in Windows sound
settings, then click Refresh.

**The first launch sits on "First-time setup" for a while.** It is downloading
the components described under Requirements. This happens once.

**It says it is using the CPU but I have an NVIDIA card.** Your driver may be
too old, or the GPU libraries did not finish downloading. Try
**Menu > Re-import Dependencies**. The CPU result is identical in quality.

**The live text is blank but the saved file is fine.** Whisper transcribes
speech. Silence, music, or very quiet audio produces little live text; the
final saved transcript still captures whatever speech was present.

## License

Released under the MIT License. See [LICENSE](LICENSE). Copyright (c) 2026
Victor S.

## Questions

The best place to reach me with questions is on Slack, in the class channel:
[pitt-mds.slack.com](https://pitt-mds.slack.com/archives/C07HW9Y3DBR). If the
tool saved you some time, I accept Mad Props as thanks.
