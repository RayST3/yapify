"""Base classes for extending YapWrap.

Subclass the component you want to replace and implement its abstract methods.
Streaming methods return async iterators; implementations normally use
``async def`` with ``yield``. Constructors and provider SDKs belong to subclasses.

All audio is signed little-endian PCM16, mono, at ``sample_rate``. Components in
one session must use the same rate. Stream implementations should release their
resources in ``finally`` or ``async with`` when closed or cancelled.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager

from yapwrap.core.audio import DEFAULT_SAMPLE_RATE
from yapwrap.core.events import AgentEvent, SessionEvent, TranscriptEvent


class BaseAgent(ABC):
    """Turn a user's text into a stream of agent events."""

    @abstractmethod
    def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvent.text(delta) for each new piece of the spoken reply.

        Text events contain only new text, not the accumulated reply. Tool
        events may also be yielded. Return when the response is complete.
        """
        raise NotImplementedError


class BaseSTT(ABC):
    """Turn one utterance's incoming audio into recognized text.

    Stateful adapters can extend ManagedSTT or WebSocketSTT in core.stt.
    Custom subclasses still need only implement transcribe().
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["BaseSTT", None]:
        """Own provider resources for a voice call; stateless adapters need no setup."""
        yield self

    @abstractmethod
    def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str | TranscriptEvent]:
        """Consume PCM until input ends, flush recognition, then finish.

        Yield TranscriptEvent(text, is_final=True) for a committed segment;
        interim events replace the current uncommitted text. Plain strings are
        also supported and replace the current utterance transcript.
        """
        raise NotImplementedError


class BaseTTS(ABC):
    """Public TTS contract, with a phrase fallback for complete-text APIs.

    Stateful adapters can extend ManagedTTS, RealtimeTTS, or WebSocketTTS in
    yapwrap.core.tts to share lifecycle and streaming behavior. Existing custom
    subclasses need only implement synthesize().
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks in playback order, then finish.

        Decode or convert provider output to PCM16 mono at sample_rate before
        yielding it. Do not yield MP3, base64 text, or WAV headers.
        """
        raise NotImplementedError

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["BaseTTS", None]:
        """Own provider resources for one voice call; stateless providers need no setup."""
        yield self

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Speak text deltas while the agent is still generating.

        Providers with streaming text input should override this method. The
        default keeps existing synthesize(str) providers working with bounded
        phrases. YapSesh opens session() once around the entire call.
        """
        from yapwrap.core.phrases import stream_phrases

        async with aclosing(stream_phrases(text)) as phrases:
            async for phrase in phrases:
                stream = self.synthesize(phrase)
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()


class BaseVAD(ABC):
    """Detect when the user starts and stops speaking."""

    sample_rate: int = DEFAULT_SAMPLE_RATE

    @abstractmethod
    def process(self, pcm: bytes) -> str | None:
        """Return 'start', 'stop', or None for a PCM frame.

        The session normally supplies 20 ms frames. Buffer internally if your
        model requires a different frame size.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear speech state after a session. Stateless detectors need no override."""
        pass


class SessionEventSink(ABC):
    """An output that displays or forwards conversation events."""

    @abstractmethod
    async def send_event(self, event: SessionEvent) -> None:
        """Handle a transcript, agent reply, or speech boundary event."""
        raise NotImplementedError


class BaseTransport(SessionEventSink):
    """Connect the pipeline to incoming audio and outgoing playback."""

    sample_rate: int = DEFAULT_SAMPLE_RATE

    @abstractmethod
    def audio_in(self) -> AsyncIterator[bytes]:
        """Yield incoming PCM chunks; end the iterator when the client disconnects."""
        raise NotImplementedError

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Send or queue synthesized PCM for playback, preserving chunk order."""
        raise NotImplementedError

    @abstractmethod
    async def stop_playback(self) -> None:
        """Clear queued playback when the user interrupts the agent."""
        raise NotImplementedError

    async def send_event(self, event: SessionEvent) -> None:
        """Optionally display conversation text. Audio-only transports can omit this."""
        pass
