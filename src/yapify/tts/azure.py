"""Azure Cognitive Services Speech text-to-speech."""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

logger = logging.getLogger("yapify.tts.azure")


class AzureSpeechTTS:
    """Stream speech from Azure Cognitive Services Speech Synthesizer."""

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: Optional[str] = None,
        voice_name: str = "en-US-JennyNeural",
        sample_rate: int = 16000,
    ) -> None:
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY", "")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.voice_name = voice_name
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio bytes."""
        if not text.strip():
            return

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError("AzureSpeechTTS requires `pip install azure-cognitiveservices-speech`.") from exc

        if not self.subscription_key:
            raise ValueError("AzureSpeechTTS requires subscription_key or AZURE_SPEECH_KEY env var.")

        speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: synthesizer.speak_text_async(text).get())

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i : i + chunk_size]
        else:
            logger.error("Azure speech synthesis canceled or failed: %s", result.reason)

