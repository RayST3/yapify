"""Google Cloud Speech-to-Text v2 / Chirp 3 streaming transcription."""

import asyncio
import logging
from typing import AsyncIterator, Optional

from yapwrap.core.base import BaseSTT

logger = logging.getLogger("yapwrap.stt.google_chirp")


class GoogleChirpSTT(BaseSTT):
    """Stream audio to Google Cloud Speech-to-Text v2 / Chirp 3."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        model: str = "chirp_2",
        language_code: str = "en-US",
        sample_rate: int = 16000,
    ) -> None:
        self.project_id = project_id
        self.model = model
        self.language_code = language_code
        self.sample_rate = sample_rate

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to Google Cloud Speech and yield transcripts."""
        try:
            from google.cloud import speech_v2
        except ImportError as exc:
            raise ImportError("GoogleChirpSTT requires `pip install google-cloud-speech`.") from exc

        client = speech_v2.SpeechAsyncClient()

        async def request_generator():
            # First request: StreamingRecognitionConfig
            config = speech_v2.RecognitionConfig(
                explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                    encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=[self.language_code],
                model=self.model,
            )
            streaming_config = speech_v2.StreamingRecognitionConfig(
                config=config,
                streaming_features=speech_v2.StreamingRecognitionFeatures(interim_results=True),
            )
            yield speech_v2.StreamingRecognizeRequest(streaming_config=streaming_config)

            async for chunk in audio:
                yield speech_v2.StreamingRecognizeRequest(audio=chunk)

        try:
            stream = await client.streaming_recognize(requests=request_generator())
            async for response in stream:
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        if transcript:
                            yield transcript
        except Exception as exc:
            logger.debug("Google Chirp streaming finished: %s", exc)

