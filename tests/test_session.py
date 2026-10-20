"""Test YapSesh orchestration and barge-in interruption."""

import asyncio
import pytest

from tests.conftest import MockTransport, MockVAD, MockSTT, MockAgent, MockTTS
from yapwrap.core.memory import InMemoryHistory
from yapwrap.core.session import YapSesh


@pytest.mark.asyncio
async def test_voice_session_turn():
    # 10 audio frames of 640 bytes each
    frames = [b"\x00\x05" * 320 for _ in range(10)]
    transport = MockTransport(incoming_frames=frames)

    # VAD events: None, "start", None, None, "stop", None...
    vad_events = [None, "start", None, None, "stop", None, None, None, None, None]
    vad = MockVAD(events=vad_events)

    stt = MockSTT(transcript="what is your name")
    agent = MockAgent(reply="I am YapWrap.")
    tts = MockTTS()
    memory = InMemoryHistory()

    session = YapSesh(
        transport=transport,
        vad=vad,
        stt=stt,
        agent=agent,
        tts=tts,
        memory=memory,
        preroll_frames=2,
    )

    await session.run()

    # Allow background agent reply tasks to complete
    await asyncio.sleep(0.05)

    # Verify agent was invoked
    assert "what is your name" in agent.received_prompts

    # Verify TTS audio was sent to transport
    assert len(transport.sent_audio) > 0

    # Verify memory recorded the turn
    assert len(memory) == 1
    assert memory.turns[0].user_transcript == "what is your name"
    assert "YapWrap" in memory.turns[0].agent_response


@pytest.mark.asyncio
async def test_voice_session_barge_in():
    frames = [b"\x00\x05" * 320 for _ in range(10)]
    transport = MockTransport(incoming_frames=frames)

    # VAD triggers "start", cutting off any playback
    vad_events = ["start", None, "stop", "start", None, "stop"]
    vad = MockVAD(events=vad_events)

    stt = MockSTT(transcript="stop talking")
    agent = MockAgent(reply="I am speaking a long answer here.")
    tts = MockTTS()
    memory = InMemoryHistory()

    session = YapSesh(
        transport=transport,
        vad=vad,
        stt=stt,
        agent=agent,
        tts=tts,
        memory=memory,
    )

    await session.run()

    # Stop playback should have been called at least once when user started speaking
    assert transport.stopped_playback_count >= 1

