"""Inworld AI character streaming text-to-speech."""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from yapify.core.base import BaseTTS

logger = logging.getLogger("yapify.tts.inworld")


class InworldTTS(BaseTTS):
    """Stream speech from Inworld AI Character Voice."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "default",
        sample_rate: int = 16000,
        base_url: str = "https://api.inworld.ai/v1/tts",
    ) -> None:
        self.api_key = api_key or os.getenv("INWORLD_API_KEY", "")
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            import httpx
        except ImportError as exc:
            raise ImportError("InworldTTS requires `pip install httpx`.") from exc

        if not self.api_key:
            raise ValueError("InworldTTS requires an api_key or INWORLD_API_KEY env var.")

        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "sample_rate": self.sample_rate,
            "format": "pcm16",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk

