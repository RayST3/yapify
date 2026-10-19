"""Yapify: Framework-agnostic, ultra-low-latency streaming voice pipeline for AI agents."""

from yapify.core.yapwrap import YapWrap
from yapify.core.protocols import Transport, VAD, STT, Agent, TTS, SessionEventSink
from yapify.core.events import AgentEvent, SessionEvent, TranscriptEvent
from yapify.core.session import VoiceSession
from yapify.core.memory import InMemoryHistory, ConversationTurn

# Yapper is an alias for YapWrap
Yapper = YapWrap

__all__ = [
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

__version__ = "0.1.0"

