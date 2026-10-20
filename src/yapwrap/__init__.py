"""YapWrap: Framework-agnostic, ultra-low-latency streaming voice pipeline for AI agents."""

from yapwrap.core.base import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport
from yapwrap.core.yapwrap import YapWrap
from yapwrap.core.protocols import Transport, VAD, STT, Agent, TTS, SessionEventSink
from yapwrap.core.events import AgentEvent, SessionEvent, TranscriptEvent
from yapwrap.core.session import VoiceSession
from yapwrap.core.memory import InMemoryHistory, ConversationTurn

# Yapper is an alias for YapWrap
Yapper = YapWrap

__all__ = [
    "BaseAgent",
    "BaseSTT",
    "BaseTTS",
    "BaseVAD",
    "BaseTransport",
    "YapWrap",
    "Yapper",
    "Transport",
    "VAD",
    "STT",
    "Agent",
    "TTS",
    "AgentEvent",
    "SessionEvent",
    "SessionEventSink",
    "TranscriptEvent",
    "VoiceSession",
    "InMemoryHistory",
    "ConversationTurn",
]

__version__ = "0.0.1a1"

