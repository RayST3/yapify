"""Gladia Live v2: HTTP session initialization followed by WebSocket audio."""

import json
import os
from typing import Optional

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class GladiaSTT(WebSocketSTT):
    """Drain final live segments before the aggregate completion message."""

    def __init__(
        self, api_key: Optional[str] = None, language: Optional[str] = None,
        sample_rate: int = 16000, base_url: str = "https://api.gladia.io/v2/live", *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("GLADIA_API_KEY", "")
        self.language, self.base_url = language, base_url

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        import httpx

        if not self.api_key:
            raise ValueError("GladiaSTT requires api_key or GLADIA_API_KEY.")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("GladiaSTT base_url must be the HTTP live-session initialization endpoint.")
        config = {
            "encoding": "wav/pcm", "sample_rate": self.sample_rate, "bit_depth": 16, "channels": 1,
            "messages_config": {
                "receive_partial_transcripts": True, "receive_final_transcripts": True,
                "receive_post_processing_events": True, "receive_errors": True,
            },
        }
        if self.language:
            config["language_config"] = {"languages": [self.language]}
        async with httpx.AsyncClient(timeout=self.connect_timeout) as client:
            response = await client.post(self.base_url, headers={"x-gladia-key": self.api_key}, json=config)
            response.raise_for_status()
            return response.json()["url"], {}

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({"type": "stop_recording"}))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        if data.get("error") or data.get("type") == "error":
            raise RuntimeError(f"Gladia STT: {data.get('error') or data}")
        if data.get("type") == "transcript":
            result = data.get("data") or {}
            utterance = result.get("utterance") or {}
            return [TranscriptEvent(utterance.get("text", ""), is_final=result.get("is_final", False),
                                    confidence=utterance.get("confidence", 1.0))], False
        # The aggregate repeats live segments; don't append it to the transcript.
        return [], data.get("type") == "post_final_transcript"

