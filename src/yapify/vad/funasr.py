"""Alibaba FunASR FSMN-VAD implementation for yapify."""

from typing import Any, Optional

from yapify.core.base import BaseVAD


class FunASRVAD(BaseVAD):
    """Alibaba FunASR FSMN-VAD streaming Voice Activity Detector."""

    def __init__(
        self,
        model: str = "fsmn-vad",
        threshold: float = 0.5,
        onset_frames: int = 2,
        min_silence_frames: int = 15,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.onset_frames = onset_frames
        self.min_silence_frames = min_silence_frames

        self._model: Optional[Any] = None
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError("FunASRVAD requires `pip install funasr`.") from exc
        self._model = AutoModel(model=self.model_name)

    def process(self, pcm: bytes) -> Optional[str]:
        """Process 16 kHz PCM frame and return 'start', 'stop', or None."""
        if not pcm:
            return None

        # Check if FunASR is available or fallback
        if self._model is None:
            try:
                self._ensure_loaded()
            except ImportError:
                from yapify.core.audio import calculate_rms
                rms = calculate_rms(pcm)
                prob = min(1.0, rms / 1100.0)
            else:
                res = self._model.generate(input=pcm, is_final=False)
                prob = 1.0 if res and len(res) > 0 and res[0].get("value") else 0.0
        else:
            res = self._model.generate(input=pcm, is_final=False)
            prob = 1.0 if res and len(res) > 0 and res[0].get("value") else 0.0

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

