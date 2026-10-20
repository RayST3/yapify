"""Hume Octave streaming TTS, replacing the unrelated EVI chat endpoint."""

import base64
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.audio import PCM16Stream
from yapwrap.core.tts import WebSocketTTS


class HumeTTS(WebSocketTTS):
    """Stream Octave snippets, converting its 48 kHz PCM to the session rate.

    Hume acknowledges snippets rather than a complete flushed input batch. End
    each context with close=true and drain until the server closes normally.
    """

    completion_on_close = True

    def __init__(
        self, api_key: Optional[str] = None, voice_id: str = "Ava Song",
        sample_rate: int = 16000, base_url: str = "wss://api.hume.ai/v0/tts/stream/input", *,
        context_idle_timeout: float = 0.12, connect_timeout: float = 15.0,
        response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("HUME_API_KEY", "")
        self.voice_id, self.base_url = voice_id, base_url

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("HumeTTS requires an api_key or HUME_API_KEY env var.")
        params = urlencode({
            "api_key": self.api_key, "format_type": "pcm", "no_binary": "true",
            "strip_headers": "true", "instant_mode": "true",
        })
        separator = "&" if "?" in self.base_url else "?"
        return self.base_url + separator + params, {}

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        from uuid import UUID
        try:
            UUID(self.voice_id)
            voice = {"id": self.voice_id}
        except ValueError:
            voice = {"name": self.voice_id, "provider": "HUME_AI"}
        async for delta in text:
            await self._connection.send(json.dumps({"text": delta, "voice": voice}))
        await self._connection.send(json.dumps({"close": True}))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        if data.get("type") == "error" or data.get("error"):
            raise RuntimeError(f"Hume TTS: {data.get('message') or data.get('error') or data}")
        # is_last_chunk ends a snippet, not necessarily all submitted text.
        pcm = base64.b64decode(data.get("audio") or "", validate=True) if data.get("type") == "audio" else b""
        return pcm, False

    async def _synthesize_context(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        converter = PCM16Stream(48000, self.sample_rate)
        async with aclosing(super()._synthesize_context(text)) as audio:
            async for chunk in audio:
                pcm = converter.process(chunk)
                if pcm:
                    yield pcm
        tail = converter.finish()
        if tail:
            yield tail

