"""Smart Turn: Hybrid acoustic and conversational turn detector."""

from typing import Optional

from yapwrap.core.base import BaseVAD
from yapwrap.core.audio import calculate_rms


class SmartTurnVAD(BaseVAD):
    """Smart Turn detection combining acoustic voice activity with turn-completion heuristics.

    Designed to avoid cutting users off during brief mid-sentence pauses ("umm", "well...").
    """

    def __init__(
        self,
        threshold: float = 650.0,
        pause_timeout_ms: int = 400,
        trailing_silence_ms: int = 800,
    ) -> None:
        self.threshold = threshold
        self.pause_timeout_frames = int(pause_timeout_ms / 20)
        self.trailing_silence_frames = int(trailing_silence_ms / 20)

        self._consecutive_silence = 0
        self._consecutive_speech = 0
        self._is_speaking = False

    def process(self, pcm: bytes) -> Optional[str]:
        """Process 16 kHz PCM frame and return 'start', 'stop', or None."""
        if not pcm:
            return None

        rms = calculate_rms(pcm)

        if rms >= self.threshold:
            self._consecutive_silence = 0
            self._consecutive_speech += 1
            if not self._is_speaking and self._consecutive_speech >= 2:
                self._is_speaking = True
                return "start"
        else:
            self._consecutive_speech = 0
            if self._is_speaking:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self.trailing_silence_frames:
                    self._is_speaking = False
                    return "stop"
        return None

    def reset(self) -> None:
        self._consecutive_silence = 0
        self._consecutive_speech = 0
        self._is_speaking = False

