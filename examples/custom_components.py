"""Write your own components by subclassing YapWrap's base classes.

Run: python -m examples.custom_components

This offline walkthrough uses demo recognition and silence so no API keys or
audio devices are needed. Supply your own async SDK functions to BufferedSTT
and BufferedTTS to use real speech. These adapters handle a whole utterance or
phrase at a time; streaming SDKs can yield results/chunks as they arrive.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from yapwrap import AgentEvent, BaseAgent, BaseSTT, BaseTTS, TranscriptEvent, YapWrap
from yapwrap.vad import EnergyVAD


class EchoAgent(BaseAgent):
    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        yield AgentEvent.text("You said: ")
        yield AgentEvent.text(text)


class BufferedSTT(BaseSTT):
    """Adapt an async function that recognizes PCM16 mono at 16 kHz."""

    def __init__(self, recognize: Callable[[bytes], Awaitable[str]]) -> None:
        self.recognize = recognize

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        pcm = bytearray()
        async for chunk in audio:
            pcm.extend(chunk)
        if pcm:
            text = await self.recognize(bytes(pcm))
            yield TranscriptEvent(text, is_final=True)


class BufferedTTS(BaseTTS):
    """Adapt an async function that returns PCM16 mono at 16 kHz."""

    def __init__(self, speak: Callable[[str], Awaitable[bytes]]) -> None:
        self.speak = speak

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if text.strip():
            yield await self.speak(text)


async def main() -> None:
    # Replace these demo functions with calls to your speech provider.
    async def recognize(pcm: bytes) -> str:
        return "Hello from my own STT."

    async def speak(text: str) -> bytes:
        return b"\x00\x00" * 320  # Demo output: 20 ms of silence, not speech.

    async def microphone() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 320

    yapper = YapWrap(
        agent=EchoAgent(),
        stt=BufferedSTT(recognize),
        tts=BufferedTTS(speak),
        vad=EnergyVAD(),
    )

    # Exercise the configured components with demo audio, without opening devices.
    async for transcript in yapper.stt.transcribe(microphone()):
        print(f"You: {transcript.text}")
        reply = "".join([event.content async for event in yapper.agent.respond(transcript.text)])
        print(f"Agent: {reply}")
        audio = b"".join([chunk async for chunk in yapper.tts.synthesize(reply)])
        print(f"Demo PCM output: {len(audio)} bytes")

    # With real recognize/speak functions and the local extra installed:
    # await yapper.startlocal()


if __name__ == "__main__":
    asyncio.run(main())
