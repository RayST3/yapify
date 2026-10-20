"""ElevenLabs multi-context streaming text-to-speech."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.tts import WebSocketTTS


class ElevenLabsTTS(WebSocketTTS):
    """Stream text deltas and keep one socket across tool pauses and replies."""

    def __init__(
        self, api_key: Optional[str] = None, voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2_5", sample_rate: int = 16000, *,
        context_idle_timeout: float = 0.12, connect_timeout: float = 15.0,
        response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id, self.model_id = voice_id, model_id
        self._closing_context: Optional[str] = None

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("ElevenLabsTTS requires api_key or ELEVENLABS_API_KEY env var.")
        params = urlencode({"model_id": self.model_id, "output_format": f"pcm_{self.sample_rate}"})
        return (f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input?{params}",
                {"xi-api-key": self.api_key})

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        await self._connection.send(json.dumps({
            "context_id": context_id, "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }))
        async for delta in text:
            await self._connection.send(json.dumps({"context_id": context_id, "text": delta}))
        await self._connection.send(json.dumps({"context_id": context_id, "flush": True}))
        await self._connection.send(json.dumps({"context_id": context_id, "close_context": True}))
        self._closing_context = context_id

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        if data.get("context_id", data.get("contextId")) not in (None, context_id):
            return b"", False
        if data.get("error"):
            raise RuntimeError(f"ElevenLabs TTS: {data.get('message') or data['error']}")
        return base64.b64decode(data.get("audio") or "", validate=True), bool(data.get("is_final"))

    async def _cancel_context(self, context_id: str) -> None:
        if self._closing_context != context_id:
            await self._connection.send(json.dumps({"context_id": context_id, "close_context": True}))

