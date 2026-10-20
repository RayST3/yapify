"""Rime JSON WebSocket speech with explicit synthesis boundaries."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.tts import WebSocketTTS


class RimeTTS(WebSocketTTS):
    """Stream tokens to Mist and reuse the socket across synthesis batches."""

    max_context_chars = 150

    def __init__(
        self, api_key: Optional[str] = None, speaker: str = "cove",
        model_id: str = "mistv3", sample_rate: int = 16000,
        base_url: str = "wss://users-ws.rime.ai/ws3", *,
        context_idle_timeout: float = 0.12, connect_timeout: float = 15.0,
        response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("RIME_API_KEY", "")
        self.speaker, self.model_id, self.base_url = speaker, model_id, base_url

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("RimeTTS requires an api_key or RIME_API_KEY env var.")
        if not self.base_url.startswith(("ws://", "wss://")):
            raise ValueError("RimeTTS base_url must be a JSON WebSocket endpoint")
        params = urlencode({
            "speaker": self.speaker, "modelId": self.model_id,
            "samplingRate": self.sample_rate, "audioFormat": "pcm", "segment": "never",
        })
        separator = "&" if "?" in self.base_url else "?"
        return self.base_url + separator + params, {"Authorization": f"Bearer {self.api_key}"}

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        # With segment=never, one explicit flush corresponds to one done event.
        # The shared engine also ends batches at punctuation/150 characters.
        async for delta in text:
            await self._connection.send(json.dumps({"text": delta, "contextId": context_id}))
        await self._connection.send(json.dumps({"operation": "flush"}))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        if data.get("contextId") not in (None, context_id):
            return b"", False
        if data.get("type") == "error":
            raise RuntimeError(f"Rime TTS: {data.get('message') or data.get('error') or data}")
        pcm = base64.b64decode(data.get("data") or "", validate=True) if data.get("type") == "chunk" else b""
        return pcm, data.get("type") == "done"

