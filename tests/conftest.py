"""Shared test fixtures and mocks for yapwrap tests."""

import asyncio
from typing import AsyncIterator, List
import pytest

from yapwrap import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport
from yapwrap.core.events import AgentEvent


class MockTransport(BaseTransport):
    """Mock transport yielding pre-defined audio and capturing output."""

    def __init__(self, incoming_frames: List[bytes] = None) -> None:
        self.incoming_frames = incoming_frames or []
        self.sent_audio: List[bytes] = []
        self.stopped_playback_count = 0

    async def audio_in(self) -> AsyncIterator[bytes]:
        for frame in self.incoming_frames:
            yield frame
            await asyncio.sleep(0.001)

    async def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    async def stop_playback(self) -> None:
        self.stopped_playback_count += 1


class MockVAD(BaseVAD):
    """Mock VAD yielding pre-programmed sequence of events."""

    def __init__(self, events: List[str | None] = None) -> None:
        self.events = list(events or [])
        self.idx = 0

    def process(self, pcm: bytes) -> str | None:
        if self.idx < len(self.events):
            event = self.events[self.idx]
            self.idx += 1
            return event
        return None


class MockSTT(BaseSTT):
    """Mock STT returning predetermined transcripts."""

    def __init__(self, transcript: str = "hello world") -> None:
        self.transcript = transcript

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[str]:
        # Consume incoming audio
        async for _ in audio:
            pass
        yield self.transcript


class MockAgent(BaseAgent):
    """Mock Agent returning tokens."""

    def __init__(self, reply: str = "Hello, I am YapWrap.") -> None:
        self.reply = reply
        self.received_prompts: List[str] = []

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        self.received_prompts.append(text)
        for word in self.reply.split():
            yield AgentEvent.text(word + " ")


class MockTTS(BaseTTS):
    """Mock TTS generating synthetic PCM audio bytes."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        # Return 320 samples of mock audio (640 bytes)
        yield b"\x00\x01" * 320

