"""Test YapWrap unified class."""

import pytest
from yapify import YapWrap, Yapper
from yapify.agents.callable import CallableAgent
from yapify.vad.energy import EnergyVAD
from tests.conftest import MockSTT, MockTTS


def test_yapwrap_init():
    agent = CallableAgent(lambda text: "echo " + text)
    vad = EnergyVAD()
    stt = MockSTT()
    tts = MockTTS()

    yapper = YapWrap(
        agent=agent,
        vad=vad,
        stt=stt,
        tts=tts,
    )

    assert yapper.agent is agent
    assert yapper.vad is vad
    assert yapper.stt is stt
    assert yapper.tts is tts
    assert hasattr(yapper, "startlocal")
    assert hasattr(yapper, "server_ws")
    assert hasattr(yapper, "servewebrtc")

    # Verify alias Yapper is YapWrap
    assert Yapper is YapWrap


@pytest.mark.asyncio
async def test_generic_transport_and_concurrent_reuse_guard():
    import asyncio
    from tests.test_session_lifecycle import LiveTransport

    wrapper = YapWrap(CallableAgent(lambda text: text), MockTTS(), MockSTT(), EnergyVAD())
    transport = LiveTransport()
    task = asyncio.create_task(wrapper.run(transport))
    await asyncio.wait_for(transport.opened.wait(), 1)
    try:
        with pytest.raises(RuntimeError, match="separate YapWrap"):
            await wrapper.run(LiveTransport())
    finally:
        transport.input.put_nowait(None)
        await asyncio.wait_for(task, 1)
    assert not wrapper._running

