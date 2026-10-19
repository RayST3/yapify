"""Microsoft Agent Framework adapter."""

from typing import Any, AsyncIterator
from yapify.core.base import BaseAgent
from yapify.core.events import AgentEvent


class MicrosoftAgentFrameworkAgent(BaseAgent):
    """Adapt an Agent's ResponseStream, closing it when speech is interrupted."""

    def __init__(self, agent: Any, **run_kwargs: Any) -> None:
        self.agent = agent
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        async with self.agent.run(text, **{**self.run_kwargs, "stream": True}) as stream:
            async for update in stream:
                if update.text:
                    yield AgentEvent.text(update.text)

