"""OpenAI streaming text-to-speech."""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

logger = logging.getLogger("yapify.tts.openai")


class OpenAITTS:
    """Stream speech from OpenAI TTS-1."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy",  # alloy, echo, fable, onyx, nova, shimmer
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into PCM audio chunks using OpenAI Speech API."""
        if not text.strip():
            return

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("OpenAITTS requires `pip install 'yapify[openai]'`.") from exc

        if not self.api_key:
            raise ValueError("OpenAITTS requires an api_key or OPENAI_API_KEY env var.")

        client = AsyncOpenAI(api_key=self.api_key)
        try:
            # response_format="pcm" returns raw 24kHz 16-bit linear PCM
            response = await client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm",
            )
            from yapify.core.audio import resample_pcm

            async for chunk in response.aiter_bytes(chunk_size=1024):
                if chunk:
                    # Resample 24kHz PCM to target sample_rate (e.g. 16kHz)
                    if self.sample_rate != 24000:
                        yield resample_pcm(chunk, from_rate=24000, to_rate=self.sample_rate)
                    else:
                        yield chunk
        finally:
            await client.close()

