"""Cartesia Sonic streaming text-to-speech."""

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Optional
import uuid

from yapwrap.core.base import BaseTTS

logger = logging.getLogger("yapwrap.tts.cartesia")


class CartesiaTTS(BaseTTS):
    """Stream speech from Cartesia Sonic over WebSocket."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonic-3.6",
        voice: str = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        sample_rate: int = 16000,
        base_url: str = "wss://api.cartesia.ai/tts/websocket",
    ) -> None:
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY", "")
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("CartesiaTTS requires `pip install 'yapwrap[cartesia]'`.") from exc

        if not self.api_key:
            raise ValueError("CartesiaTTS requires an api_key or CARTESIA_API_KEY env var.")

        url = f"{self.base_url}?cartesia_version=2024-06-10"
        headers = {"X-API-Key": self.api_key}
        context_id = str(uuid.uuid4())

        async with connect(url, additional_headers=headers) as ws:
            request = {
                "model_id": self.model,
                "transcript": text,
                "voice": {"mode": "id", "id": self.voice},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                "context_id": context_id,
                "continue": False,
            }
            await ws.send(json.dumps(request))

            async for message in ws:
                data = json.loads(message)
                if data.get("type") == "error" or data.get("error"):
                    raise RuntimeError(f"Cartesia TTS: {data.get('message') or data.get('error') or 'provider error'}")
                if data.get("context_id") != context_id:
                    continue
                audio_b64 = data.get("data")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
                if data.get("done"):
                    return
            raise RuntimeError("Cartesia closed before synthesis completed")

