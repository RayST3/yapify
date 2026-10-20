"""Inworld bidirectional TTS with independent contexts on one connection."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Optional

from yapwrap.core.tts import WebSocketTTS


class InworldTTS(WebSocketTTS):
    """Stream incremental text using Inworld's context protocol and raw PCM."""

    def __init__(
        self, api_key: Optional[str] = None, voice_id: str = "Ashley",
        sample_rate: int = 16000,
        base_url: str = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional", *,
        model: str = "inworld-tts-2", context_idle_timeout: float = 0.12,
        connect_timeout: float = 15.0, response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout,
                         connect_timeout=connect_timeout, response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("INWORLD_API_KEY", "")
        self.voice_id, self.base_url, self.model = voice_id, base_url, model

    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("InworldTTS requires an api_key or INWORLD_API_KEY env var.")
        if not self.base_url.startswith(("ws://", "wss://")):
            raise ValueError("InworldTTS base_url must be a bidirectional WebSocket endpoint")
        return self.base_url, {"Authorization": f"Basic {self.api_key}"}

    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        await self._connection.send(json.dumps({
            "contextId": context_id,
            "create": {
                "voiceId": self.voice_id, "modelId": self.model,
                "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": self.sample_rate},
                "autoMode": False, "maxBufferDelayMs": 80,
            },
        }))
        async for delta in text:
            await self._connection.send(json.dumps({"contextId": context_id, "sendText": {"text": delta}}))
        await self._connection.send(json.dumps({"contextId": context_id, "closeContext": {}}))

    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        data = json.loads(message)
        result = data.get("result", data)
        if result.get("contextId") not in (None, context_id):
            return b"", False
        audio = result.get("audioChunk", {})
        status = result.get("status") or audio.get("status") or {}
        if data.get("error") or status.get("code", 0):
            raise RuntimeError(f"Inworld TTS: {data.get('error') or status.get('message')}")
        return base64.b64decode(audio.get("audioContent") or "", validate=True), "contextClosed" in result

