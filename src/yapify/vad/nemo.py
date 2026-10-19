"""NVIDIA NeMo frame-level VAD implementation for yapify."""

from typing import Any, Optional


class NeMoVAD:
    """NVIDIA NeMo Frame-level VAD (MarbleNet) for real-time speech detection."""

    def __init__(
        self,
        model_name: str = "vad_marblenet",
        threshold: float = 0.6,
        onset_frames: int = 2,
        min_silence_frames: int = 15,
    ) -> None:
        self.model_name = model_name
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
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError("NeMoVAD requires NVIDIA NeMo `pip install nemo_toolkit[asr]`.") from exc
        self._model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=self.model_name)

    def process(self, pcm: bytes) -> Optional[str]:
        """Process 16 kHz PCM frame and return 'start', 'stop', or None."""
        if not pcm:
            return None

        # Check if NeMo is loaded or fallback to energy
        if self._model is None:
            try:
                self._ensure_loaded()
            except ImportError:
                from yapify.core.audio import calculate_rms
                rms = calculate_rms(pcm)
                prob = min(1.0, rms / 1000.0)
            else:
                import torch
                import numpy as np
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                signal = torch.tensor(samples).unsqueeze(0)
                prob = float(torch.sigmoid(self._model.forward(input_signal=signal, input_signal_length=torch.tensor([len(samples)]))[0]))
        else:
            import torch
            import numpy as np
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            signal = torch.tensor(samples).unsqueeze(0)
            prob = float(torch.sigmoid(self._model.forward(input_signal=signal, input_signal_length=torch.tensor([len(samples)]))[0]))

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

