"""Cartesia's protocol over the shared realtime TTS engine."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.tts import WebSocketTTS


class CartesiaTTS(WebSocketTTS):
    """Stream Sonic text and audio, reusing the connection across replies."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "sonic-3.6",
        voice: str = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4", sample_rate: int = 16000,
        base_url: str = "wss://api.cartesia.ai/tts/websocket", *,
        api_version: str = "2026-08-14", language: str = "en",
        max_buffer_delay_ms: int = 80, context_idle_timeout: float = 0.75,
        connect_timeout: float = 15.0, response_timeout: float = 30.0,
    ) -> None:
        if not 0 < context_idle_timeout < 1:
            raise ValueError("context_idle_timeout must be between zero and one second")
        if not 0 <= max_buffer_delay_ms <= 5000:
            raise ValueError("max_buffer_delay_ms must be between 0 and 5000")
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY", "")
        self.model, self.voice, self.base_url = model, voice, base_url
        self.api_version, self.language = api_version, language
        self.max_buffer_delay_ms = max_buffer_delay_ms

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("CartesiaTTS requires an api_key or CARTESIA_API_KEY env var.")
        separator = "&" if "?" in self.base_url else "?"
        return (self.base_url + separator + urlencode({"cartesia_version": self.api_version}),
                {"X-API-Key": self.api_key})

    def _request(self, context_id: str, text: str, continuation: bool) -> dict:
        return {
            "model_id": self.model, "transcript": text, "voice": self.voice,
            "language": self.language,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "context_id": context_id, "continue": continuation,
            "max_buffer_delay_ms": self.max_buffer_delay_ms,
        }

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        async for delta in text:
            await self._connection.send(json.dumps(self._request(context_id, delta, True)))
        await self._connection.send(json.dumps(self._request(context_id, "", False)))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        if data.get("context_id") not in (None, context_id):
            return b"", False
        if data.get("type") == "error" or data.get("error"):
            raise RuntimeError(f"Cartesia TTS: {data.get('message') or data.get('error') or 'provider error'}")
        return (base64.b64decode(data.get("data") or "", validate=True),
                data.get("type") == "done" or bool(data.get("done")))

    async def _cancel_context(self, context_id: str) -> None:
        await self._connection.send(json.dumps({"context_id": context_id, "cancel": True}))
