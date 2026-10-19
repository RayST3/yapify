"""Google Cloud / Gemini Live text-to-speech."""

import asyncio
import logging
from typing import AsyncIterator, Optional

from yapify.core.base import BaseTTS

logger = logging.getLogger("yapify.tts.gemini")


class GoogleGeminiTTS(BaseTTS):
    """Stream speech from Google Cloud Text-to-Speech / Gemini."""

    def __init__(
        self,
        voice_name: str = "en-US-Journey-F",
        language_code: str = "en-US",
        sample_rate: int = 16000,
    ) -> None:
        self.voice_name = voice_name
        self.language_code = language_code
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            from google.cloud import texttospeech
        except ImportError as exc:
            raise ImportError("GoogleGeminiTTS requires `pip install google-cloud-texttospeech`.") from exc

        client = texttospeech.TextToSpeechAsyncClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
        )

        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        # Skip 44-byte WAV header if present to yield raw PCM
        audio_content = response.audio_content
        if audio_content.startswith(b"RIFF") and len(audio_content) > 44:
            audio_content = audio_content[44:]

        # Yield in chunks
        chunk_size = 1024
        for i in range(0, len(audio_content), chunk_size):
            yield audio_content[i : i + chunk_size]

