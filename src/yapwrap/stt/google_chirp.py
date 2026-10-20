"""Google Speech v2 streaming recognition with a reusable regional SDK client."""

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Optional

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import ManagedSTT


class GoogleChirpSTT(ManagedSTT):
    """Open one client per call and one streaming recognition RPC per utterance."""

    def __init__(
        self, project_id: Optional[str] = None, model: str = "chirp_3",
        language_code: str = "en-US", sample_rate: int = 16000, *,
        location: str = "us", response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.model, self.language_code, self.location = model, language_code, location
        self.response_timeout = response_timeout

    async def _open_connection(self):
        try:
            from google.cloud import speech_v2
        except ImportError as exc:
            raise ImportError("GoogleChirpSTT requires pip install google-cloud-speech.") from exc
        if not self.project_id:
            raise ValueError("GoogleChirpSTT requires project_id or GOOGLE_CLOUD_PROJECT.")
        self._sdk = speech_v2
        endpoint = "speech.googleapis.com" if self.location == "global" else f"{self.location}-speech.googleapis.com"
        return speech_v2.SpeechAsyncClient(client_options={"api_endpoint": endpoint})

    async def _close_connection(self) -> None:
        client, self._connection = self._connection, None
        if client is not None:
            await client.transport.close()

    async def _transcribe_stream(self, audio: AsyncIterator[bytes]) -> AsyncGenerator[TranscriptEvent, None]:
        sdk = self._sdk
        loop = asyncio.get_running_loop()
        failure = loop.create_future()
        input_done = asyncio.Event()

        async def requests():
            try:
                yield sdk.StreamingRecognizeRequest(
                    recognizer=f"projects/{self.project_id}/locations/{self.location}/recognizers/_",
                    streaming_config=sdk.StreamingRecognitionConfig(
                        config=sdk.RecognitionConfig(
                            explicit_decoding_config=sdk.ExplicitDecodingConfig(
                                encoding=sdk.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                sample_rate_hertz=self.sample_rate, audio_channel_count=1,
                            ), language_codes=[self.language_code], model=self.model,
                        ), streaming_features=sdk.StreamingRecognitionFeatures(interim_results=True),
                    ),
                )
                async for chunk in audio:
                    yield sdk.StreamingRecognizeRequest(audio=chunk)
                input_done.set()
            except BaseException as exc:
                if not failure.done():
                    failure.set_result(exc)
                raise

        stream, receiver = None, None
        eof = asyncio.create_task(input_done.wait())
        async with aclosing(requests()) as source:
            opening = asyncio.create_task(self._connection.streaming_recognize(
                requests=source, retry=None, timeout=300.0,
            ))
            try:
                deadline = None
                while not opening.done():
                    waiting = {opening, failure}
                    if deadline is None:
                        waiting.add(eof)
                    done, _ = await asyncio.wait(
                        waiting, timeout=None if deadline is None else max(0, deadline - loop.time()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done or (deadline is not None and loop.time() >= deadline):
                        raise TimeoutError("Google STT did not finalize within response_timeout")
                    if failure in done:
                        raise failure.result()
                    if eof in done:
                        deadline = loop.time() + self.response_timeout
                stream = opening.result()
                iterator = stream.__aiter__()
                receiver = asyncio.create_task(anext(iterator))
                while True:
                    waiting = {receiver, failure}
                    if deadline is None:
                        waiting.add(eof)
                    done, _ = await asyncio.wait(
                        waiting, timeout=None if deadline is None else max(0, deadline - loop.time()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done or (deadline is not None and loop.time() >= deadline):
                        raise TimeoutError("Google STT did not finalize within response_timeout")
                    if failure in done:
                        raise failure.result()
                    if eof in done:
                        deadline = loop.time() + self.response_timeout
                    if receiver not in done:
                        continue
                    try:
                        response = receiver.result()
                    except StopAsyncIteration:
                        if not input_done.is_set():
                            raise RuntimeError("Google STT closed before input audio ended") from None
                        return
                    for result in response.results:
                        if result.alternatives:
                            best = result.alternatives[0]
                            before = loop.time()
                            yield TranscriptEvent(best.transcript, is_final=result.is_final,
                                                  confidence=getattr(best, "confidence", 1.0))
                            if deadline is not None:
                                deadline += loop.time() - before
                    receiver = asyncio.create_task(anext(iterator))
            finally:
                tasks = [opening, eof] + ([receiver] if receiver is not None else [])
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if stream is None and not opening.cancelled() and opening.exception() is None:
                    stream = opening.result()
                if stream is not None:
                    cancel = getattr(stream, "cancel", None)
                    if cancel is not None:
                        cancel()
                        code = getattr(stream, "code", None)
                        if code is not None:
                            await code()
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
                failure.cancel()

