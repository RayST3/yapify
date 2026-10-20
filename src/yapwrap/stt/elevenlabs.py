"""ElevenLabs Scribe Realtime streaming transcription."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapwrap.core.base import BaseSTT

logger = logging.getLogger("yapwrap.stt.elevenlabs")


class ElevenLabsSTT(BaseSTT):
    """Stream audio to ElevenLabs Scribe Realtime Speech-to-Text WebSocket."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "scribe_v1",
        language: str = "en",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to ElevenLabs Scribe and yield transcripts."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("ElevenLabsSTT requires `pip install 'yapwrap[elevenlabs]'`.") from exc

        if not self.api_key:
            raise ValueError("ElevenLabsSTT requires an api_key or ELEVENLABS_API_KEY env var.")

        url = f"wss://api.elevenlabs.io/v1/speech-to-text/streaming?model_id={self.model}"
        headers = {"xi-api-key": self.api_key}

        async with connect(url, additional_headers=headers) as ws:
            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                except Exception as exc:
                    logger.debug("ElevenLabs STT sender finished: %s", exc)

            sender = asyncio.create_task(send_audio())
            try:
                async for message in ws:
                    data = json.loads(message)
                    text = data.get("text", "")
                    if text:
                        yield text
                    if data.get("is_final"):
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

