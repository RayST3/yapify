"""Sarvam Bulbul WebSocket text-to-speech with explicit completion events."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.tts import WebSocketTTS


class SarvamTTS(WebSocketTTS):
    """Stream Indian-language speech and retain the socket between replies."""

    def __init__(
        self, api_key: Optional[str] = None, language_code: str = "hi-IN",
        speaker: str = "anushka", model: str = "bulbul:v2", sample_rate: int = 16000,
        base_url: str = "wss://api.sarvam.ai/text-to-speech/ws", *,
        context_idle_timeout: float = 0.12, connect_timeout: float = 15.0,
        response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.language_code, self.speaker, self.model = language_code, speaker, model
        self.base_url = base_url

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("SarvamTTS requires an api_key or SARVAM_API_KEY env var.")
        if not self.base_url.startswith(("ws://", "wss://")):
            raise ValueError("SarvamTTS base_url must be a streaming WebSocket endpoint")
        params = urlencode({"model": self.model, "send_completion_event": "true"})
        separator = "&" if "?" in self.base_url else "?"
        return self.base_url + separator + params, {"api-subscription-key": self.api_key}

    async def _configure_connection(self, connection) -> None:
        await connection.send(json.dumps({
            "type": "config", "data": {
                "language_code": self.language_code, "speaker": self.speaker,
                "speech_sample_rate": self.sample_rate, "output_audio_codec": "linear16",
                "min_buffer_size": 50, "max_chunk_length": 200,
            },
        }))

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        async for delta in text:
            await self._connection.send(json.dumps({"type": "text", "data": {"text": delta}}))
        await self._connection.send(json.dumps({"type": "flush"}))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        payload = data.get("data") or {}
        if data.get("type") == "error":
            raise RuntimeError(f"Sarvam TTS: {payload}")
        pcm = base64.b64decode(payload.get("audio") or "", validate=True) if data.get("type") == "audio" else b""
        return pcm, data.get("type") == "event" and payload.get("event_type") == "final"

