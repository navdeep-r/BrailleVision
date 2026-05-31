"""
inference/tts_engine.py

Text-to-speech engine with offline/online fallback.

Tries pyttsx3 (offline, no internet required) first.
Falls back to gTTS (requires internet) if pyttsx3 is unavailable.
"""

import os
import sys
import tempfile
import platform
from typing import Optional


class TTSEngine:
    """
    Text-to-speech with offline/online fallback.

    Usage:
        tts = TTSEngine()
        tts.speak("Hello, this is BrailleVision speaking.")
    """

    def __init__(
        self,
        prefer_offline: bool = True,
        rate: int = 150,
        volume: float = 0.9,
    ):
        self._mode: str = "none"
        self._engine = None
        self._rate   = rate
        self._volume = volume

        if prefer_offline:
            self._try_init_offline()

        if self._mode == "none":
            self._try_init_online()

        print(f"[TTS] Mode: {self._mode}")

    def _try_init_offline(self) -> None:
        """Try to initialise pyttsx3 (offline TTS)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            self._engine = engine
            self._mode   = "offline"
        except Exception as e:
            print(f"[TTS] pyttsx3 unavailable: {e}")

    def _try_init_online(self) -> None:
        """Try to confirm gTTS is importable (online TTS)."""
        try:
            import gtts  # noqa: F401
            self._mode = "online"
        except Exception as e:
            print(f"[TTS] gTTS unavailable: {e}")
            print("[TTS] No TTS engine available.")

    def speak(self, text: str) -> None:
        """Speak the given text using the available engine."""
        if not text or not text.strip():
            return

        if self._mode == "offline":
            self._speak_offline(text)
        elif self._mode == "online":
            self._speak_online(text)
        else:
            print(f"[TTS] (no engine) Text: {text[:80]}")

    def _speak_offline(self, text: str) -> None:
        """Speak using pyttsx3."""
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Offline playback error: {e}")

    def _speak_online(self, text: str) -> None:
        """Speak using gTTS → temp MP3 → system audio player."""
        try:
            from gtts import gTTS

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            tts = gTTS(text, lang="en", slow=False)
            tts.save(tmp_path)

            system = platform.system()
            if system == "Windows":
                os.startfile(tmp_path)
            elif system == "Darwin":
                os.system(f"afplay '{tmp_path}'")
            else:
                # Linux — try mpg123, then ffplay, then aplay
                for player in ["mpg123", "ffplay -nodisp -autoexit", "cvlc --play-and-exit"]:
                    if os.system(f"{player} '{tmp_path}' > /dev/null 2>&1") == 0:
                        break

        except Exception as e:
            print(f"[TTS] Online playback error: {e}")
