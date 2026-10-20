"""Azure Speech SDK text streaming with audio delivered through SDK callbacks."""

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Optional

from yapwrap.core.audio import PCM16Stream
from yapwrap.core.tts import RealtimeTTS


class AzureSpeechTTS(RealtimeTTS):
    """Reuse a synthesizer and stream both text and audio on the v2 endpoint."""

    def __init__(
        self, subscription_key: Optional[str] = None, region: Optional[str] = None,
        voice_name: str = "en-US-JennyNeural", sample_rate: int = 16000, *,
        context_idle_timeout: float = 0.12, response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout)
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY", "")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.voice_name, self.response_timeout = voice_name, response_timeout
        self._sdk_connection = None

    async def _open_connection(self):
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError("AzureSpeechTTS requires azure-cognitiveservices-speech>=1.37.") from exc
        if not self.subscription_key:
            raise ValueError("AzureSpeechTTS requires subscription_key or AZURE_SPEECH_KEY env var.")
        if not hasattr(speechsdk, "SpeechSynthesisRequest"):
            raise ImportError("AzureSpeechTTS text streaming requires azure-cognitiveservices-speech>=1.37.")
        self._sdk = speechsdk
        config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            endpoint=f"wss://{self.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2",
        )
        config.speech_synthesis_voice_name = self.voice_name
        config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        connection = speechsdk.Connection.from_speech_synthesizer(synthesizer)
        self._sdk_connection = connection
        try:
            await asyncio.to_thread(connection.open, True)
        except BaseException:
            await asyncio.to_thread(connection.close)
            self._sdk_connection = None
            raise
        return synthesizer

    async def _close_connection(self) -> None:
        self._connection = None
        connection, self._sdk_connection = self._sdk_connection, None
        if connection is not None:
            await asyncio.to_thread(connection.close)

    async def _synthesize_context(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        sdk, synthesizer = self._sdk, self._connection
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue = asyncio.Queue()
        accepting = True

        def on_audio(event):
            pcm = bytes(event.result.audio_data)
            if accepting and pcm:
                loop.call_soon_threadsafe(chunks.put_nowait, pcm)

        request = sdk.SpeechSynthesisRequest(input_type=sdk.SpeechSynthesisRequestInputType.TextStream)

        async def send_text():
            try:
                async for delta in text:
                    request.input_stream.write(delta)
            finally:
                request.input_stream.close()

        async def finish(result_future):
            result = await asyncio.to_thread(result_future.get)
            if result.reason != sdk.ResultReason.SynthesizingAudioCompleted:
                details = getattr(result, "cancellation_details", None)
                raise RuntimeError(f"Azure TTS: {getattr(details, 'error_details', None) or result.reason}")
            chunks.put_nowait(None)

        synthesizer.synthesizing.connect(on_audio)
        tasks = []
        completed = False
        converter = PCM16Stream(24000, self.sample_rate)
        try:
            result_future = synthesizer.speak_async(request)
            sender = asyncio.create_task(send_text())
            finisher = asyncio.create_task(finish(result_future))
            receiver = asyncio.create_task(chunks.get())
            tasks = [sender, finisher, receiver]
            running = {sender, finisher}
            while True:
                done, _ = await asyncio.wait(
                    {*running, receiver}, timeout=self.response_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError("Azure TTS did not produce audio within response_timeout")
                for task in done & running:
                    running.remove(task)
                    task.result()
                if receiver not in done:
                    continue
                chunk = receiver.result()
                if chunk is None:
                    await sender
                    tail = converter.finish()
                    if tail:
                        yield tail
                    completed = True
                    return
                pcm = converter.process(chunk)
                if pcm:
                    yield pcm
                receiver = asyncio.create_task(chunks.get())
                tasks[-1] = receiver
        finally:
            accepting = False
            synthesizer.synthesizing.disconnect(on_audio)
            for task in tasks:
                if not task.done():
                    task.cancel()
            if not completed:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(lambda: synthesizer.stop_speaking_async().get()), timeout=2,
                    )
                except Exception:
                    with suppress(Exception):
                        await self._close_connection()
            await asyncio.gather(*tasks, return_exceptions=True)

