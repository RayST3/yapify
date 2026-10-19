"""Azure Cognitive Services Speech streaming transcription."""

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

logger = logging.getLogger("yapify.stt.azure")


class AzureSpeechSTT:
    """Stream audio to Azure Cognitive Services Speech SDK."""

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: Optional[str] = None,
        language: str = "en-US",
        sample_rate: int = 16000,
    ) -> None:
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY", "")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.language = language
        self.sample_rate = sample_rate

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to Azure Speech and yield transcripts."""
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError("AzureSpeechSTT requires `pip install azure-cognitiveservices-speech`.") from exc

        if not self.subscription_key:
            raise ValueError("AzureSpeechSTT requires subscription_key or AZURE_SPEECH_KEY env var.")

        speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_config.speech_recognition_language = self.language

        audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=self.sample_rate, bits_per_sample=16, channels=1)
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_recognizing(evt):
            if evt.result.text:
                loop.call_soon_threadsafe(queue.put_nowait, evt.result.text)

        def on_recognized(evt):
            if evt.result.text:
                loop.call_soon_threadsafe(queue.put_nowait, evt.result.text)

        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.start_continuous_recognition()

        async def feed_audio():
            try:
                async for chunk in audio:
                    push_stream.write(chunk)
                push_stream.close()
            except Exception as exc:
                logger.debug("Azure audio feeder error: %s", exc)

        feeder = asyncio.create_task(feed_audio())

        try:
            while True:
                text = await asyncio.wait_for(queue.get(), timeout=1.5)
                if text is None:
                    break
                yield text
        except asyncio.TimeoutError:
            pass
        finally:
            recognizer.stop_continuous_recognition()
            if not feeder.done():
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)

