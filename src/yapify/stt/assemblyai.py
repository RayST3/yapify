"""AssemblyAI streaming transcription over WebSocket."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from yapify.core.base import BaseSTT

logger = logging.getLogger("yapify.stt.assemblyai")


class AssemblyAISTT(BaseSTT):
    """Stream audio to AssemblyAI Streaming STT v2."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        word_boost: Optional[list] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        self.sample_rate = sample_rate
        self.word_boost = word_boost or []

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to AssemblyAI and yield transcripts."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("AssemblyAISTT requires `pip install 'yapify[websockets]'`.") from exc

        if not self.api_key:
            raise ValueError("AssemblyAISTT requires an api_key or ASSEMBLYAI_API_KEY env var.")

        params = {"sample_rate": str(self.sample_rate)}
        if self.word_boost:
            params["word_boost"] = json.dumps(self.word_boost)

        url = f"wss://api.assemblyai.com/v2/realtime/ws?{urlencode(params)}"
        headers = {"Authorization": self.api_key}

        async with connect(url, additional_headers=headers) as ws:
            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"terminate_session": True}))
                except Exception as exc:
                    logger.debug("AssemblyAI sender finished: %s", exc)

            sender = asyncio.create_task(send_audio())
            try:
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("message_type")
                    text = data.get("text", "")
                    if text:
                        yield text
                    if msg_type == "FinalTranscript":
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

