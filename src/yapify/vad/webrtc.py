"""Google WebRTC VAD implementation for yapify."""

from typing import Any, Optional


class WebRTCVAD:
    """Fast, C-based WebRTC Voice Activity Detector.

    Requires frames of 10, 20, or 30 ms at 16 kHz (e.g. 640 bytes for 20ms).
    """

    def __init__(
        self,
        mode: int = 3,
        sample_rate: int = 16000,
        onset_frames: int = 2,
        min_silence_frames: int = 15,
    ) -> None:
        self.mode = mode
        self.sample_rate = sample_rate
        self.onset_frames = onset_frames
        self.min_silence_frames = min_silence_frames

        self._vad: Optional[Any] = None
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

    def _ensure_loaded(self) -> None:
        if self._vad is not None:
            return
        try:
            import webrtcvad
        except ImportError as exc:
            raise ImportError(
                "WebRTCVAD requires `pip install 'yapify[webrtc-vad]' webrtcvad`."
            ) from exc
        self._vad = webrtcvad.Vad(self.mode)

    def process(self, pcm: bytes) -> Optional[str]:
        """Process PCM frame and return 'start', 'stop', or None."""
        self._ensure_loaded()
        if not pcm:
            return None

        is_speech = self._vad.is_speech(pcm, self.sample_rate)

        if is_speech:
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
        """Reset internal state."""
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

