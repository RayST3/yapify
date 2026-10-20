"""Event models for yapwrap streaming sessions."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass
class AgentEvent:
    """An event emitted by an agent during execution."""

    type: str  # "text_delta", "tool_start", "tool_end", "text_complete", "thought"
    content: str = ""
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None

    @classmethod
    def text(cls, delta: str) -> "AgentEvent":
        """Convenience constructor for a streamed text delta."""
        return cls(type="text_delta", content=delta)

    @classmethod
    def tool_start(cls, name: str, args: Optional[Dict[str, Any]] = None) -> "AgentEvent":
        """Convenience constructor for tool invocation start."""
        return cls(type="tool_start", name=name, args=args or {})

    @classmethod
    def tool_end(cls, name: str, result: Optional[Any] = None) -> "AgentEvent":
        """Convenience constructor for tool invocation end."""
        return cls(type="tool_end", name=name, result=result)


@dataclass
class TranscriptEvent:
    """Incremental or final speech recognition transcript."""

    text: str
    is_final: bool = False
    turn_id: int = 0
    confidence: float = 1.0


@dataclass
class SessionEvent:
    """Session lifecycle or conversation event.

    Conversation data includes turn_id. Transcript text is a cumulative snapshot
    with is_final; agent_delta text is appended; agent_complete contains the full
    generated text and a status (completed, interrupted, or error). These events
    describe recognized/generated text, not acknowledged audio playback.
    """

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

