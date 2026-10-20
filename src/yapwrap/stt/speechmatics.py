"""Speechmatics real-time streaming transcription."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapwrap.core.base import BaseSTT

logger = logging.getLogger("yapwrap.stt.speechmatics")


class SpeechmaticsSTT(BaseSTT):
    """Stream audio to Speechmatics Real-Time WebSocket API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 16000,
        base_url: str = "wss://eu2.rt.speechmatics.com/v2",
    ) -> None:
        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY", "")
        self.language = language
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to Speechmatics and yield transcripts."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("SpeechmaticsSTT requires `pip install 'yapwrap[websockets]'`.") from exc

        if not self.api_key:
            raise ValueError("SpeechmaticsSTT requires an api_key or SPEECHMATICS_API_KEY env var.")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with connect(self.base_url, additional_headers=headers) as ws:
            # Send StartRecognition
            config = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                "transcription_config": {
                    "language": self.language,
                    "enable_partials": True,
                },
            }
            await ws.send(json.dumps(config))

            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))
                except Exception as exc:
                    logger.debug("Speechmatics sender finished: %s", exc)

            sender = asyncio.create_task(send_audio())
            try:
                async for message in ws:
                    data = json.loads(message)
                    msg = data.get("message")
                    if msg in ("AddPartialTranscript", "AddTranscript"):
                        metadata = data.get("metadata", {})
                        transcript = metadata.get("transcript", "")
                        if transcript:
                            yield transcript
                    if msg == "EndOfTranscript":
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

