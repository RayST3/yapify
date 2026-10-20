"""Deepgram Aura streaming text-to-speech."""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from yapwrap.core.base import BaseTTS

logger = logging.getLogger("yapwrap.tts.deepgram")


class DeepgramAuraTTS(BaseTTS):
    """Stream speech from Deepgram Aura TTS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aura-asteria-en",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.model = model
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio chunks."""
        if not text.strip():
            return

        try:
            import httpx
        except ImportError as exc:
            raise ImportError("DeepgramAuraTTS requires `pip install httpx`.") from exc

        if not self.api_key:
            raise ValueError("DeepgramAuraTTS requires api_key or DEEPGRAM_API_KEY env var.")

        params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
        }
        url = f"https://api.deepgram.com/v1/speak?{urlencode(params)}"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            async with client.stream("POST", url, headers=headers, json={"text": text}) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk

