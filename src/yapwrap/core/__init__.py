"""YapWrap core architecture: base classes, events, audio processing, memory, and session pipeline."""

from yapwrap.core.base import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport
from yapwrap.core.protocols import Transport, VAD, STT, Agent, TTS
from yapwrap.core.events import AgentEvent, SessionEvent, TranscriptEvent
from yapwrap.core.memory import InMemoryHistory, ConversationTurn
from yapwrap.core.session import VoiceSession

__all__ = [
    "BaseAgent",
    "BaseSTT",
    "BaseTTS",
    "BaseVAD",
    "BaseTransport",
    "Transport",
    "VAD",
    "STT",
    "Agent",
    "TTS",
    "AgentEvent",
    "SessionEvent",
    "TranscriptEvent",
    "InMemoryHistory",
    "ConversationTurn",
    "VoiceSession",
]

