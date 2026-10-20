"""Deepgram Nova streaming transcription with an explicit final drain."""

import json
import os
from typing import List, Optional
from urllib.parse import urlencode

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class DeepgramSTT(WebSocketSTT):
    """Drain CloseStream before ending an utterance; final segments aren't EOF."""

    completion_on_close = True
    keepalive_interval = 4.0
    keepalive_message = '{"type":"KeepAlive"}'

    def __init__(
        self, api_key: Optional[str] = None, model: str = "nova-3",
        language: str = "en", smart_format: bool = True, sample_rate: int = 16000,
        endpointing: Optional[int] = None, keyterms: Optional[List[str]] = None, *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.model, self.language, self.smart_format = model, language, smart_format
        self.endpointing, self.keyterms = endpointing, keyterms

    def _build_url(self) -> str:
        if self.model.startswith("flux"):
            raise ValueError("DeepgramSTT uses the Nova v1 protocol; Flux requires a separate v2 adapter.")
        params = [
            ("model", self.model), ("language", self.language), ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)), ("channels", "1"),
            ("interim_results", "true"), ("smart_format", str(self.smart_format).lower()),
        ]
        if self.endpointing is not None:
            params.append(("endpointing", str(self.endpointing)))
        for term in self.keyterms or []:
            params.append(("keyterm", term))
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("DeepgramSTT requires api_key or DEEPGRAM_API_KEY.")
        return self._build_url(), {"Authorization": f"Token {self.api_key}"}

    async def _finish_audio(self) -> None:
        # Finalize alone does not guarantee an acknowledgement for silent audio.
        await self._connection.send(json.dumps({"type": "CloseStream"}))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        if data.get("type") == "Error" or data.get("error"):
            raise RuntimeError(f"Deepgram STT: {data.get('description') or data.get('error') or data}")
        alternatives = data.get("channel", {}).get("alternatives", [])
        events = []
        if alternatives:
            best = alternatives[0]
            events.append(TranscriptEvent(
                best.get("transcript", ""), is_final=data.get("is_final", False),
                confidence=best.get("confidence", 1.0),
            ))
        return events, False

