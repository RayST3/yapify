"""AWS Transcribe streaming transcription."""

import asyncio
import logging
from typing import AsyncIterator, Optional

from yapify.core.base import BaseSTT

logger = logging.getLogger("yapify.stt.aws")


class AWSTranscribeSTT(BaseSTT):
    """Stream audio to AWS Transcribe Streaming."""

    def __init__(
        self,
        region: str = "us-east-1",
        language_code: str = "en-US",
        sample_rate: int = 16000,
    ) -> None:
        self.region = region
        self.language_code = language_code
        self.sample_rate = sample_rate

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio frames to AWS Transcribe and yield transcripts."""
        try:
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
        except ImportError as exc:
            raise ImportError("AWSTranscribeSTT requires `pip install amazon-transcribe`.") from exc

        client = TranscribeStreamingClient(region=self.region)
        stream = await client.start_stream_transcription(
            language_code=self.language_code,
            media_sample_rate_hz=self.sample_rate,
            media_encoding="pcm",
        )

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        class Handler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                results = transcript_event.transcript.results
                for res in results:
                    for alt in res.alternatives:
                        if alt.transcript:
                            await queue.put(alt.transcript)

        handler = Handler(stream.output_stream)
        handler_task = asyncio.create_task(handler.handle_events())

        async def send_audio():
            try:
                async for chunk in audio:
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                await stream.input_stream.end_stream()
            except Exception as exc:
                logger.debug("AWS Transcribe audio sender finished: %s", exc)

        sender_task = asyncio.create_task(send_audio())

        try:
            while True:
                text = await asyncio.wait_for(queue.get(), timeout=1.5)
                if text is None:
                    break
                yield text
        except asyncio.TimeoutError:
            pass
        finally:
            for t in (handler_task, sender_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(handler_task, sender_task, return_exceptions=True)

