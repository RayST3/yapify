"""Google Cloud TTS: native streaming for Chirp 3 HD, phrase fallback otherwise."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing

from yapwrap.core.audio import PCM16Stream, wav_to_pcm16
from yapwrap.core.tts import ManagedTTS, RealtimeTTS


class GoogleGeminiTTS(RealtimeTTS):
    """Reuse a Cloud TTS client; select a Chirp3-HD voice for streaming text."""

    def __init__(
        self, voice_name: str = "en-US-Journey-F", language_code: str = "en-US",
        sample_rate: int = 16000, *, context_idle_timeout: float = 0.12,
        response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout)
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        self.voice_name, self.language_code = voice_name, language_code
        self.response_timeout = response_timeout

    async def _open_connection(self):
        try:
            from google.cloud import texttospeech
        except ImportError as exc:
            raise ImportError("GoogleGeminiTTS requires pip install google-cloud-texttospeech.") from exc
        self._sdk = texttospeech
        return texttospeech.TextToSpeechAsyncClient()

    async def _close_connection(self) -> None:
        client, self._connection = self._connection, None
        if client is not None:
            await client.transport.close()

    def _voice(self):
        return self._sdk.VoiceSelectionParams(language_code=self.language_code, name=self.voice_name)

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return
        if "Chirp3-HD" in self.voice_name:
            async with aclosing(super().synthesize(text)) as audio:
                async for chunk in audio:
                    yield chunk
        else:
            async with self._client() as client:
                response = await client.synthesize_speech(
                    input=self._sdk.SynthesisInput(text=text), voice=self._voice(),
                    audio_config=self._sdk.AudioConfig(
                        audio_encoding=self._sdk.AudioEncoding.LINEAR16,
                        sample_rate_hertz=self.sample_rate,
                    ), retry=None, timeout=self.response_timeout,
                )
                pcm = wav_to_pcm16(response.audio_content, self.sample_rate)
                for offset in range(0, len(pcm), 1024):
                    yield pcm[offset:offset + 1024]

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        stream = (super().synthesize_stream(text) if "Chirp3-HD" in self.voice_name
                  else ManagedTTS.synthesize_stream(self, text))
        async with aclosing(stream) as audio:
            async for chunk in audio:
                yield chunk

    async def _synthesize_context(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        sdk = self._sdk

        async def requests():
            yield sdk.StreamingSynthesizeRequest(streaming_config=sdk.StreamingSynthesizeConfig(
                voice=self._voice(), streaming_audio_config=sdk.StreamingAudioConfig(
                    audio_encoding=sdk.AudioEncoding.PCM, sample_rate_hertz=self.sample_rate,
                ),
            ))
            async for delta in text:
                yield sdk.StreamingSynthesizeRequest(input=sdk.StreamingSynthesisInput(text=delta))

        converter = PCM16Stream(self.sample_rate, self.sample_rate)
        async with aclosing(requests()) as source:
            stream = await self._connection.streaming_synthesize(
                requests=source, retry=None, timeout=self.response_timeout,
            )
            try:
                async for response in stream:
                    pcm = converter.process(response.audio_content)
                    if pcm:
                        yield pcm
                tail = converter.finish()
                if tail:
                    yield tail
            finally:
                cancel = getattr(stream, "cancel", None)
                if cancel is not None:
                    cancel()
                    code = getattr(stream, "code", None)
                    if code is not None:
                        await code()
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

