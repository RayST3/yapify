"""Core runtime protocols for yapify.

Every component in yapify adheres to one of these five protocols:
- Transport: Ingests microphone audio and sends synthesized playback.
- VAD: Detects speech boundaries ('start' / 'stop').
- STT: Transcribes audio streams into text.
- Agent: Processes user text and streams agent events.
- TTS: Synthesizes text into PCM audio.
"""

from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yapify.core.events import AgentEvent, SessionEvent, TranscriptEvent


@runtime_checkable
class Transport(Protocol):
    """Transport protocol handling incoming and outgoing audio."""

    def audio_in(self) -> AsyncIterator[bytes]:
        """Yield signed little-endian PCM16 mono at the session's sample rate."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Send synthesized audio chunk to the client/playback device."""
        ...

    async def stop_playback(self) -> None:
        """Immediately cancel/stop playback on the client (barge-in interruption)."""
        ...


@runtime_checkable
class SessionEventSink(Protocol):
    """Optional transport capability for displaying conversation events.

    Audio-only transports remain valid without this method.
    """

    async def send_event(self, event: "SessionEvent") -> None:
        """Display or forward a session event in order with other output."""
        ...


@runtime_checkable
class VAD(Protocol):
    """Voice Activity Detection protocol for turn detection."""

    def process(self, pcm: bytes) -> str | None:
        """Process a PCM audio chunk. Return 'start', 'stop', or None."""
        ...


@runtime_checkable
class STT(Protocol):
    """Speech-to-Text streaming transcription protocol."""

    def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator["str | TranscriptEvent"]:
        """Yield transcript snapshots or events until input EOF is finalized.

        Strings replace the current utterance transcript. TranscriptEvent finals
        append committed segments; interim events replace the uncommitted tail.
        """
        ...


@runtime_checkable
class Agent(Protocol):
    """Agent execution protocol."""

    def respond(self, text: str) -> AsyncIterator["AgentEvent"]:
        """Respond to user input, streaming agent events and text tokens."""
        ...


@runtime_checkable
class TTS(Protocol):
    """Text-to-Speech synthesis protocol."""

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into PCM audio chunks."""
        ...

