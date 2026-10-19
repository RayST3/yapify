"""Sarvam AI Bulbul streaming text-to-speech for Indian languages."""

import asyncio
import base64
import logging
import os
from typing import AsyncIterator, Optional

from yapify.core.base import BaseTTS

logger = logging.getLogger("yapify.tts.sarvam")


class SarvamTTS(BaseTTS):
    """Stream speech from Sarvam AI Bulbul TTS (optimized for Indian languages)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "hi-IN",
        speaker: str = "meera",
        model: str = "bulbul:v1",
        sample_rate: int = 16000,
        base_url: str = "https://api.sarvam.ai/text-to-speech",
    ) -> None:
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.language_code = language_code
        self.speaker = speaker
        self.model = model
        self.sample_rate = sample_rate
        self.base_url = base_url

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            import httpx
        except ImportError as exc:
            raise ImportError("SarvamTTS requires `pip install httpx`.") from exc

        if not self.api_key:
            raise ValueError("SarvamTTS requires an api_key or SARVAM_API_KEY env var.")

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": [text],
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            audios = data.get("audios", [])
            if audios:
                audio_bytes = base64.b64decode(audios[0])
                # Skip 44-byte WAV header if present to yield raw PCM
                if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
                    audio_bytes = audio_bytes[44:]

                chunk_size = 1024
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i : i + chunk_size]

