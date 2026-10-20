"""Soniox real-time streaming transcription."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, List, Optional

from yapwrap.core.base import BaseSTT

logger = logging.getLogger("yapwrap.stt.soniox")


class SonioxSTT(BaseSTT):
    """Stream audio to Soniox Real-Time WebSocket API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "stt-rt-v5",
        language_hints: Optional[List[str]] = None,
        sample_rate: int = 16000,
        base_url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
    ) -> None:
        self.api_key = api_key or os.getenv("SONIOX_API_KEY", "")
        self.model = model
        self.language_hints = language_hints
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to Soniox and yield transcripts."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("SonioxSTT requires `pip install 'yapwrap[websockets]'`.") from exc

        if not self.api_key:
            raise ValueError("SonioxSTT requires an api_key or SONIOX_API_KEY env var.")

        async with connect(self.base_url) as ws:
            # Send config message
            config = {
                "api_key": self.api_key,
                "model": self.model,
                "audio_format": "pcm_s16le",
                "sample_rate": self.sample_rate,
                "num_channels": 1,
            }
            if self.language_hints:
                config["language_hints"] = self.language_hints

            await ws.send(json.dumps(config))

            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                    await ws.send(b"")  # empty frame signals end of audio
                except Exception as exc:
                    logger.debug("Soniox sender finished: %s", exc)

            sender = asyncio.create_task(send_audio())
            current_text = ""
            try:
                async for message in ws:
                    data = json.loads(message)
                    words = data.get("words", [])
                    if words:
                        for w in words:
                            current_text += w.get("text", "")
                        if current_text:
                            yield current_text
                    if data.get("is_final"):
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

