"""CMPIF2100 Lab Transcriber 2.0.

Records system (loopback) audio, transcribes it live with a local
faster-whisper model, shows the text as it goes, and on stop saves both the
recorded .wav and a clean .txt transcript with the Pitt copyright notice.
"""

APP_NAME = "CMPIF2100 Lab Transcriber"
VERSION = "2.0"

__all__ = ["APP_NAME", "VERSION"]
