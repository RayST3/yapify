"""Zero-dependency energy/RMS-based Voice Activity Detector."""

from typing import Optional

from yapify.core.audio import calculate_rms


class EnergyVAD:
    """Zero-dependency Energy/RMS based Voice Activity Detector.

    Useful as a lightweight fallback when neural VAD packages are not installed.
    """

    def __init__(
        self,
        threshold: float = 600.0,
        onset_frames: int = 2,
        min_silence_frames: int = 15,
    ) -> None:
        self.threshold = threshold
        self.onset_frames = onset_frames
        self.min_silence_frames = min_silence_frames

        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

    def process(self, pcm: bytes) -> Optional[str]:
        """Process PCM frame and return 'start', 'stop', or None."""
        rms = calculate_rms(pcm)

        if rms >= self.threshold:
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
        """Reset VAD internal state."""
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

