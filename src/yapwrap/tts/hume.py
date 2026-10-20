"""Hume AI Empathic Voice Interface (EVI) streaming text-to-speech."""

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapwrap.core.base import BaseTTS

logger = logging.getLogger("yapwrap.tts.hume")


class HumeTTS(BaseTTS):
    """Stream speech from Hume AI Empathic Voice Interface (EVI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "ITO",
        sample_rate: int = 16000,
        base_url: str = "wss://api.hume.ai/v0/evi/chat",
    ) -> None:
        self.api_key = api_key or os.getenv("HUME_API_KEY", "")
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("HumeTTS requires `pip install 'yapwrap[websockets]'`.") from exc

        if not self.api_key:
            raise ValueError("HumeTTS requires an api_key or HUME_API_KEY env var.")

        url = f"{self.base_url}?api_key={self.api_key}"

        async with connect(url) as ws:
            # Send text input message
            msg = {
                "type": "assistant_input",
                "text": text,
            }
            await ws.send(json.dumps(msg))

            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "audio_output":
                    b64_audio = data.get("data")
                    if b64_audio:
                        yield base64.b64decode(b64_audio)
                elif msg_type == "assistant_end":
                    break

