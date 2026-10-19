"""Compatibility imports for the former protocol names.

New extensions should subclass BaseAgent, BaseSTT, BaseTTS, BaseVAD, or
BaseTransport from yapify. These old names are aliases for the same abstract
base classes; they no longer perform structural Protocol checks.
"""

from yapify.core.base import (
    BaseAgent as Agent,
    BaseSTT as STT,
    BaseTTS as TTS,
    BaseVAD as VAD,
    BaseTransport as Transport,
    SessionEventSink,
)

__all__ = ["Agent", "STT", "TTS", "VAD", "Transport", "SessionEventSink"]

