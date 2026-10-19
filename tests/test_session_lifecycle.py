import asyncio

import pytest

from tests.conftest import MockAgent, MockSTT, MockTTS, MockTransport, MockVAD
from yapify import BaseSTT, BaseTTS, TranscriptEvent, VoiceSession


class LiveTransport(MockTransport):
    def __init__(self):
        super().__init__()
        self.input = asyncio.Queue()
        self.opened = asyncio.Event()
        self.closed = asyncio.Event()
        self.output = asyncio.Event()
        self.stopped = asyncio.Event()
        self.events = []

    async def send_event(self, event):
        self.events.append(event)

    async def audio_in(self):
        self.opened.set()
        try:
            while True:
                chunk = await self.input.get()
                if chunk is None:
                    return
                yield chunk
        finally:
            self.closed.set()

    async def send_audio(self, pcm):
        await super().send_audio(pcm)
        self.output.set()

    async def stop_playback(self):
        await super().stop_playback()
        self.stopped.set()

    def feed(self, *values):
        self.input.put_nowait(b"".join(bytes([value, 0]) * 320 for value in values))


async def start(events, **components):
    transport = LiveTransport()
    session = VoiceSession(
        transport=transport, vad=MockVAD(events),
        stt=components.get("stt", MockSTT()),
        agent=components.get("agent", MockAgent()),
        tts=components.get("tts", MockTTS()),
    )
    task = asyncio.create_task(session.run())
    await asyncio.wait_for(transport.opened.wait(), 1)
    return session, transport, task


@pytest.mark.asyncio
async def test_complete_frames_and_transcript_segments():
    frames = []

    class STT(BaseSTT):
        async def transcribe(self, audio):
            async for frame in audio:
                frames.append(frame)
            yield TranscriptEvent("hello", is_final=True)
            yield TranscriptEvent("wor")
            yield TranscriptEvent("world", is_final=True)

    session, transport, task = await start([None, "start", None, "stop"], stt=STT())
    try:
        transport.feed(1, 2, 3, 4)
        await asyncio.wait_for(transport.output.wait(), 1)
        await asyncio.wait_for(session._active_reply_task, 1)
        assert [frame[0] for frame in frames] == [1, 2, 3, 4]
        assert session.agent.received_prompts == ["hello world"]
        assert len(session.memory) == 1
        transcripts = [event.data for event in transport.events if event.type == "transcript"]
        assert [(data["text"], data["is_final"]) for data in transcripts] == [
            ("hello", False), ("hello wor", False), ("hello world", False), ("hello world", True),
        ]
        deltas = [event.data["text"] for event in transport.events if event.type == "agent_delta"]
        complete = transport.events[-1]
        assert complete.type == "agent_complete"
        assert complete.data == {"turn_id": 1, "text": "".join(deltas), "status": "completed"}
    finally:
        session.stop()
        await asyncio.wait_for(task, 1)
    assert transport.closed.is_set()
    assert not session._tasks


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_transcription():
    entered, closed = asyncio.Event(), asyncio.Event()

    class STT(BaseSTT):
        async def transcribe(self, audio):
            entered.set()
            try:
                async for _ in audio:
                    pass
                yield "late reply"
            finally:
                closed.set()

    session, transport, task = await start(["start"], stt=STT())
    transport.feed(1)
    await asyncio.wait_for(entered.wait(), 1)
    transport.input.put_nowait(None)
    await asyncio.wait_for(task, 1)
    assert closed.is_set()
    assert session.agent.received_prompts == []
    assert not session._tasks


@pytest.mark.asyncio
async def test_new_utterance_cancels_stale_transcription():
    first_finished, first_cancelled = asyncio.Event(), asyncio.Event()

    class STT(BaseSTT):
        calls = 0

        async def transcribe(self, audio):
            self.calls += 1
            number = self.calls
            async for _ in audio:
                pass
            if number == 1:
                first_finished.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    first_cancelled.set()
            yield f"turn {number}"

    session, transport, task = await start(["start", "stop", "start", "stop"], stt=STT())
    try:
        transport.feed(1, 2)
        await asyncio.wait_for(first_finished.wait(), 1)
        transport.feed(3, 4)
        await asyncio.wait_for(transport.output.wait(), 1)
        assert first_cancelled.is_set()
        assert session.agent.received_prompts == ["turn 2"]
    finally:
        session.stop()
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_barge_in_closes_tts_and_records_interruption():
    closed = asyncio.Event()

    class TTS(BaseTTS):
        async def synthesize(self, text):
            try:
                yield b"\x01\x00" * 320
                await asyncio.Event().wait()
            finally:
                closed.set()

    session, transport, task = await start(["start", "stop", "start"], tts=TTS())
    try:
        transport.feed(1, 2)
        await asyncio.wait_for(transport.output.wait(), 1)
        transport.stopped.clear()
        transport.feed(3)
        await asyncio.wait_for(transport.stopped.wait(), 1)
        await asyncio.wait_for(closed.wait(), 1)
        await asyncio.gather(session._active_reply_task, return_exceptions=True)
        assert len(transport.sent_audio) == 1
        assert session.memory.turns[0].was_interrupted
        assert session.memory.turns[0].user_heard == ""
        completion = next(event for event in transport.events if event.type == "agent_complete")
        assert completion.data["turn_id"] == 1
        assert completion.data["status"] == "interrupted"
        starts = [event.data["turn_id"] for event in transport.events if event.type == "speech_started"]
        assert starts == [1, 2]
    finally:
        session.stop()
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["stt", "agent", "tts"])
async def test_provider_failures_propagate_and_close_input(component):
    class Broken:
        async def transcribe(self, audio):
            raise RuntimeError("provider failed")
            yield

        async def respond(self, text):
            raise RuntimeError("provider failed")
            yield

        async def synthesize(self, text):
            raise RuntimeError("provider failed")
            yield

    session, transport, task = await start(["start", "stop"], **{component: Broken()})
    transport.feed(1, 2)
    with pytest.raises(RuntimeError, match="provider failed"):
        await asyncio.wait_for(task, 1)
    assert transport.closed.is_set()
    assert not session._tasks
    if component != "stt":
        completion = next(event for event in transport.events if event.type == "agent_complete")
        assert completion.data["status"] == "error"


@pytest.mark.asyncio
async def test_stop_wakes_idle_input():
    session, transport, task = await start([])
    session.stop()
    await asyncio.wait_for(task, 1)
    assert transport.closed.is_set()


@pytest.mark.asyncio
async def test_external_cancellation_is_preserved():
    session, transport, task = await start([])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed.is_set()


def test_session_rejects_mismatched_pcm_rates():
    with pytest.raises(ValueError, match="tts sample_rate"):
        VoiceSession(MockTransport(), MockVAD(), MockSTT(), MockAgent(), MockTTS(24000))
