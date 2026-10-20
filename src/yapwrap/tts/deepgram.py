"""Deepgram Aura bidirectional streaming text-to-speech."""

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.tts import WebSocketTTS


class DeepgramAuraTTS(WebSocketTTS):
    """Use Speak/Flush on a persistent socket instead of one HTTP call per phrase."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "aura-asteria-en",
        sample_rate: int = 16000, *, context_idle_timeout: float = 0.12,
        connect_timeout: float = 15.0, response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.model = model

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("DeepgramAuraTTS requires api_key or DEEPGRAM_API_KEY env var.")
        params = urlencode({"model": self.model, "encoding": "linear16", "sample_rate": self.sample_rate})
        return f"wss://api.deepgram.com/v1/speak?{params}", {"Authorization": f"Token {self.api_key}"}

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        async for delta in text:
            await self._connection.send(json.dumps({"type": "Speak", "text": delta}))
        await self._connection.send(json.dumps({"type": "Flush"}))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        if isinstance(message, bytes):
            return message, False
        data = json.loads(message)
        kind = data.get("type")
        detail = data.get("description") or data.get("message") or str(data)
        if kind == "Error" or (kind == "Warning" and "flush" in str(data).lower()):
            # In particular, a flush-limit warning has no Flushed acknowledgement.
            # Surface it immediately instead of leaving the caller waiting silently.
            raise RuntimeError(f"Deepgram TTS: {detail}")
        if kind == "Warning":
            logging.getLogger(__name__).warning("Deepgram TTS: %s", detail)
        return b"", data.get("type") == "Flushed"

