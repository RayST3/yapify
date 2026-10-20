"""Test protocol conformance for yapwrap components."""

import pytest

from yapwrap.core.protocols import Transport, VAD, STT, Agent, TTS
from yapwrap.transports.websocket import WebSocketTransport
from yapwrap.transports.local import LocalTransport
from yapwrap.transports.webrtc import WebRTCTransport
from yapwrap.vad.energy import EnergyVAD
from yapwrap.vad.silero import SileroVAD
from yapwrap.vad.webrtc import WebRTCVAD
from yapwrap.vad.smart_turn import SmartTurnVAD
from yapwrap.stt.deepgram import DeepgramSTT
from yapwrap.stt.openai_realtime import OpenAIRealtimeSTT
from yapwrap.stt.assemblyai import AssemblyAISTT
from yapwrap.tts.cartesia import CartesiaTTS
from yapwrap.tts.elevenlabs import ElevenLabsTTS
from yapwrap.tts.deepgram import DeepgramAuraTTS
from yapwrap.tts.openai import OpenAITTS
from yapwrap.agents.agno import AgnoAgent
from yapwrap.agents.langgraph import LangGraphAgent
from yapwrap.agents.pydantic_ai import PydanticAIAgent
from yapwrap.agents.openai_agents import OpenAIAgentsSDKAgent
from yapwrap.agents.callable import CallableAgent


def test_transport_protocols():
    ws = WebSocketTransport(ws=None)
    local = LocalTransport()
    webrtc = WebRTCTransport()

    assert isinstance(ws, Transport)
    assert isinstance(local, Transport)
    assert isinstance(webrtc, Transport)


def test_vad_protocols():
    energy = EnergyVAD()
    silero = SileroVAD()
    webrtc_vad = WebRTCVAD()
    smart_turn = SmartTurnVAD()

    assert isinstance(energy, VAD)
    assert isinstance(silero, VAD)
    assert isinstance(webrtc_vad, VAD)
    assert isinstance(smart_turn, VAD)

    # Protocol conformance must not require downloading/loading a neural model.
    silero.reset()


def test_stt_protocols():
    deepgram = DeepgramSTT(api_key="test")
    openai_stt = OpenAIRealtimeSTT(api_key="test")
    assemblyai = AssemblyAISTT(api_key="test")

    assert isinstance(deepgram, STT)
    assert isinstance(openai_stt, STT)
    assert isinstance(assemblyai, STT)


def test_tts_protocols():
    cartesia = CartesiaTTS(api_key="test")
    elevenlabs = ElevenLabsTTS(api_key="test")
    deepgram_tts = DeepgramAuraTTS(api_key="test")
    openai_tts = OpenAITTS(api_key="test")

    assert isinstance(cartesia, TTS)
    assert isinstance(elevenlabs, TTS)
    assert isinstance(deepgram_tts, TTS)
    assert isinstance(openai_tts, TTS)


def test_agent_protocols():
    agno = AgnoAgent(agent=None)
    langgraph = LangGraphAgent(graph=None)
    pydantic_ai = PydanticAIAgent(agent=None)
    openai_agent = OpenAIAgentsSDKAgent(agent=None)
    callable_agent = CallableAgent(lambda text: "echo")

    assert isinstance(agno, Agent)
    assert isinstance(langgraph, Agent)
    assert isinstance(pydantic_ai, Agent)
    assert isinstance(openai_agent, Agent)
    assert isinstance(callable_agent, Agent)

