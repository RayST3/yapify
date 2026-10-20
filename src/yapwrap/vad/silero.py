"""Silero VAD implementation for yapwrap."""

from collections import deque
from typing import Any, Optional

from yapwrap.core.base import BaseVAD
from yapwrap.core.audio import AudioReslicer


class SileroVAD(BaseVAD):
    """Silero VAD using ONNX runtime for ultra-fast, local neural turn detection.

    Expects 16 kHz 16-bit mono PCM frames.
    """

    sample_rate = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ) -> None:
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._iterator: Optional[Any] = None
        # Silero requires exactly 512 samples, independently of transport framing.
        self._reslicer = AudioReslicer(frame_bytes=512 * 2)
        self._events = deque()

    def _ensure_loaded(self) -> None:
        if self._iterator is not None:
            return
        try:
            import numpy as np
            import torch
            from silero_vad import VADIterator, load_silero_vad

            model = load_silero_vad(onnx=True)
            self._iterator = VADIterator(
                model,
                threshold=self.threshold,
                sampling_rate=16000,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )
            self._np = np
            self._torch = torch
            return
        except ImportError as exc:
            raise ImportError("SileroVAD requires `pip install 'yapwrap[silero]'`.") from exc

    def process(self, pcm: bytes) -> Optional[str]:
        """Process 16 kHz PCM audio and return 'start', 'stop', or None."""
        if not pcm:
            return self._events.popleft() if self._events else None
        self._ensure_loaded()
        for window in self._reslicer.process(pcm):
            samples = self._np.frombuffer(window, dtype="<i2").astype(self._np.float32) / 32768.0
            event = self._iterator(self._torch.from_numpy(samples))
            if event:
                if "start" in event:
                    self._events.append("start")
                if "end" in event:
                    self._events.append("stop")
        return self._events.popleft() if self._events else None

    def reset(self) -> None:
        """Reset Silero internal state."""
        if self._iterator is not None:
            self._iterator.reset_states()
        self._reslicer.clear()
        self._events.clear()
