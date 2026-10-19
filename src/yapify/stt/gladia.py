"""Gladia Live v2 streaming transcription."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapify.core.base import BaseSTT

logger = logging.getLogger("yapify.stt.gladia")


class GladiaSTT(BaseSTT):
    """Stream audio to Gladia Live Speech-to-Text v2 WebSocket API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: int = 16000,
        base_url: str = "wss://api.gladia.io/v2/live",
    ) -> None:
        self.api_key = api_key or os.getenv("GLADIA_API_KEY", "")
        self.language = language
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to Gladia and yield transcripts."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("GladiaSTT requires `pip install 'yapify[websockets]'`.") from exc

        if not self.api_key:
            raise ValueError("GladiaSTT requires an api_key or GLADIA_API_KEY env var.")

        headers = {"x-gladia-key": self.api_key}

        async with connect(self.base_url, additional_headers=headers) as ws:
            # Send initialization configuration
            config = {
                "encoding": "wav/pcm",
                "sample_rate": self.sample_rate,
                "bit_depth": 16,
                "channels": 1,
            }
            if self.language:
                config["language_config"] = {"languages": [self.language]}

            await ws.send(json.dumps(config))

            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "stop_recording"}))
                except Exception as exc:
                    logger.debug("Gladia sender finished: %s", exc)

            sender = asyncio.create_task(send_audio())
            try:
                async for message in ws:
                    data = json.loads(message)
                    event_type = data.get("type")
                    if event_type in ("transcript", "final"):
                        text = data.get("data", {}).get("utterance", {}).get("text", "")
                        if text:
                            yield text
                    if event_type == "post_final_transcript":
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

