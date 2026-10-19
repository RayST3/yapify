"""Test protocol conformance for yapify components."""

import pytest

from yapify.core.protocols import Transport, VAD, STT, Agent, TTS
from yapify.transports.websocket import WebSocketTransport
from yapify.transports.local import LocalTransport
from yapify.transports.webrtc import WebRTCTransport
from yapify.vad.energy import EnergyVAD
from yapify.vad.silero import SileroVAD
from yapify.vad.webrtc import WebRTCVAD
from yapify.vad.smart_turn import SmartTurnVAD
from yapify.stt.deepgram import DeepgramSTT
from yapify.stt.openai_realtime import OpenAIRealtimeSTT
from yapify.stt.assemblyai import AssemblyAISTT
from yapify.tts.cartesia import CartesiaTTS
from yapify.tts.elevenlabs import ElevenLabsTTS
from yapify.tts.deepgram import DeepgramAuraTTS
from yapify.tts.openai import OpenAITTS
from yapify.agents.agno import AgnoAgent
from yapify.agents.langgraph import LangGraphAgent
from yapify.agents.pydantic_ai import PydanticAIAgent
from yapify.agents.openai_agents import OpenAIAgentsSDKAgent
from yapify.agents.callable import CallableAgent


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

