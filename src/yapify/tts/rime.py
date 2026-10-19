"""Rime Labs neural streaming text-to-speech."""

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapify.core.base import BaseTTS

logger = logging.getLogger("yapify.tts.rime")


class RimeTTS(BaseTTS):
    """Stream speech from Rime Labs ultra-low-latency neural TTS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        speaker: str = "young_male",
        model_id: str = "v1",
        sample_rate: int = 16000,
        base_url: str = "https://users.rime.ai/v1/rime-tts",
    ) -> None:
        self.api_key = api_key or os.getenv("RIME_API_KEY", "")
        self.speaker = speaker
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            import httpx
        except ImportError as exc:
            raise ImportError("RimeTTS requires `pip install httpx`.") from exc

        if not self.api_key:
            raise ValueError("RimeTTS requires an api_key or RIME_API_KEY env var.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload = {
            "text": text,
            "speaker": self.speaker,
            "modelId": self.model_id,
            "samplingRate": self.sample_rate,
            "audioFormat": "pcm",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk

