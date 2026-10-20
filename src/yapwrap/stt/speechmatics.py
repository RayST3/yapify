"""Speechmatics streaming recognition with local utterance boundaries."""

import json
import os
from typing import Optional

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class SpeechmaticsSTT(WebSocketSTT):
    """ForceEndOfUtterance drains results without closing the recognition session."""

    reuse_connection = True

    def __init__(
        self, api_key: Optional[str] = None, language: str = "en",
        sample_rate: int = 16000, base_url: str = "wss://eu.rt.speechmatics.com/v2", *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY", "")
        self.language, self.base_url = language, base_url
        self._audio_bytes = 0

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("SpeechmaticsSTT requires api_key or SPEECHMATICS_API_KEY.")
        return self.base_url, {"Authorization": f"Bearer {self.api_key}"}

    async def _configure_connection(self, connection) -> None:
        self._audio_bytes = 0
        await connection.send(json.dumps({
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "transcription_config": {"language": self.language, "enable_partials": True},
        }))
        data = json.loads(await connection.recv())
        if data.get("message") != "RecognitionStarted":
            raise RuntimeError(f"Speechmatics STT initialization failed: {data}")

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        await self._connection.send(chunk)
        self._audio_bytes += len(chunk)

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({
            "message": "ForceEndOfUtterance",
            "timestamp": self._audio_bytes / (self.sample_rate * 2),
        }))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        kind = data.get("message")
        if kind == "Error":
            raise RuntimeError(f"Speechmatics STT: {data.get('reason') or data}")
        if kind in ("AddPartialTranscript", "AddTranscript"):
            return [TranscriptEvent(data.get("metadata", {}).get("transcript", ""),
                                    is_final=kind == "AddTranscript")], False
        if kind == "EndOfTranscript":
            raise RuntimeError("Speechmatics STT ended the recognition session unexpectedly")
        return [], kind == "EndOfUtterance"

