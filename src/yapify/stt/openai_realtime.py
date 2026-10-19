"""OpenAI Realtime API streaming transcription (Whisper / gpt-live-transcribe)."""

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapify.core.base import BaseSTT

logger = logging.getLogger("yapify.stt.openai")


class OpenAIRealtimeSTT(BaseSTT):
    """Stream audio to OpenAI Realtime API for ultra-low latency transcription."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-realtime-preview",
        language: Optional[str] = "en",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to OpenAI Realtime API and yield transcribed text."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("OpenAIRealtimeSTT requires `pip install 'yapify[openai]'`.") from exc

        if not self.api_key:
            raise ValueError("OpenAIRealtimeSTT requires an api_key or OPENAI_API_KEY env var.")

        client = AsyncOpenAI(api_key=self.api_key)

        try:
            async with client.realtime.connect(model=self.model) as connection:
                # Configure session for transcription
                await connection.session.update(
                    session={
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1",
                            "language": self.language,
                        },
                        "turn_detection": None,  # Yapify handles VAD turn detection locally
                    }
                )

                async def send_audio() -> None:
                    try:
                        async for chunk in audio:
                            b64_audio = base64.b64encode(chunk).decode("ascii")
                            await connection.input_audio_buffer.append(audio=b64_audio)
                        await connection.input_audio_buffer.commit()
                    except Exception as exc:
                        logger.debug("OpenAI Realtime audio sender finished: %s", exc)

                sender = asyncio.create_task(send_audio())

                try:
                    async for event in connection:
                        event_type = getattr(event, "type", "")
                        if event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = getattr(event, "transcript", "")
                            if transcript:
                                yield transcript
                                break
                        elif event_type == "error":
                            error = getattr(event, "error", None)
                            logger.error("OpenAI Realtime error: %s", error)
                            break
                finally:
                    if not sender.done():
                        sender.cancel()
                        await asyncio.gather(sender, return_exceptions=True)
        finally:
            await client.close()

