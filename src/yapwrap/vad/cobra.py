"""Picovoice Cobra VAD implementation for yapwrap."""

import os
from typing import Any, Optional

from yapwrap.core.base import BaseVAD


class CobraVAD(BaseVAD):
    """Picovoice Cobra highly accurate voice activity detector."""

    def __init__(
        self,
        access_key: Optional[str] = None,
        threshold: float = 0.5,
        onset_frames: int = 2,
        min_silence_frames: int = 15,
    ) -> None:
        self.access_key = access_key or os.getenv("PICOVOICE_ACCESS_KEY", "")
        self.threshold = threshold
        self.onset_frames = onset_frames
        self.min_silence_frames = min_silence_frames

        self._cobra: Optional[Any] = None
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

    def _ensure_loaded(self) -> None:
        if self._cobra is not None:
            return
        try:
            import pvcobra
        except ImportError as exc:
            raise ImportError("CobraVAD requires `pip install pvcobra`.") from exc

        if not self.access_key:
            raise ValueError("CobraVAD requires an access_key or PICOVOICE_ACCESS_KEY env var.")
        self._cobra = pvcobra.create(access_key=self.access_key)

    def process(self, pcm: bytes) -> Optional[str]:
        """Process 16 kHz PCM frame and return 'start', 'stop', or None."""
        self._ensure_loaded()
        if not pcm:
            return None

        import struct
        samples = [struct.unpack_from("<h", pcm, i)[0] for i in range(0, len(pcm), 2)]
        prob = self._cobra.process(samples)

        if prob >= self.threshold:
            self._consecutive_silence = 0
            self._consecutive_speech += 1
            if not self._is_speaking and self._consecutive_speech >= self.onset_frames:
                self._is_speaking = True
                return "start"
        else:
            self._consecutive_speech = 0
            if self._is_speaking:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self.min_silence_frames:
                    self._is_speaking = False
                    return "stop"
        return None

    def reset(self) -> None:
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

