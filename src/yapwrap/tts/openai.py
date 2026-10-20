"""OpenAI speech with streamed response bodies and a reusable SDK client."""

import os
from collections.abc import AsyncGenerator
from typing import Optional

from yapwrap.core.audio import PCM16Stream
from yapwrap.core.tts import ManagedTTS


class OpenAITTS(ManagedTTS):
    """Stream PCM for bounded phrases; the Speech API accepts complete text."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "tts-1",
        voice: str = "alloy", sample_rate: int = 16000,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model, self.voice = model, voice

    async def _open_connection(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("OpenAITTS requires pip install 'yapwrap[openai]'.") from exc
        if not self.api_key:
            raise ValueError("OpenAITTS requires an api_key or OPENAI_API_KEY env var.")
        return AsyncOpenAI(api_key=self.api_key, timeout=30.0, max_retries=0)

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return
        async with self._client() as client:
            converter = PCM16Stream(24000, self.sample_rate)
            async with client.audio.speech.with_streaming_response.create(
                model=self.model, voice=self.voice, input=text, response_format="pcm",
            ) as response:
                async for chunk in response.iter_bytes():
                    pcm = converter.process(chunk)
                    if pcm:
                        yield pcm
                tail = converter.finish()
                if tail:
                    yield tail

