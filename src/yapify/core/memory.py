"""In-memory conversation history adapter for yapify."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional


@dataclass
class ConversationTurn:
    """Represents a single conversational turn between the user and the agent."""

    user_transcript: str
    agent_response: str
    was_interrupted: bool = False
    user_heard: str = ""
    timestamp: float = field(default_factory=time.time)


class InMemoryHistory:
    """Simple in-memory history storage for user transcriptions and agent responses."""

    def __init__(self, max_turns: int = 50) -> None:
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add(
        self,
        user_transcript: str,
        agent_response: str,
        was_interrupted: bool = False,
        user_heard: str = "",
    ) -> ConversationTurn:
        """Record a completed turn."""
        turn = ConversationTurn(
            user_transcript=user_transcript,
            agent_response=agent_response,
            was_interrupted=was_interrupted,
            user_heard=user_heard,
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        return turn

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add an existing ConversationTurn instance."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_messages(self) -> List[Dict[str, str]]:
        """Return history as standard LLM message dictionaries."""
        messages: List[Dict[str, str]] = []
        for turn in self.turns:
            if turn.user_transcript:
                messages.append({"role": "user", "content": turn.user_transcript})
            if turn.agent_response:
                # If interrupted and we know what was heard, use that for realistic context
                content = turn.user_heard if turn.was_interrupted and turn.user_heard else turn.agent_response
                messages.append({"role": "assistant", "content": content})
        return messages

    def last_turn(self) -> Optional[ConversationTurn]:
        """Return the most recent conversation turn, if any."""
        return self.turns[-1] if self.turns else None

    def clear(self) -> None:
        """Clear all stored turns."""
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

