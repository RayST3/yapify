"""Pydantic AI text agent adapter."""

from typing import Any, AsyncIterator
from yapify.core.base import BaseAgent
from yapify.core.events import AgentEvent


class PydanticAIAgent(BaseAgent):
    """Adapt a text-output Agent; pass dependencies/history as run options."""

    def __init__(self, agent: Any, **run_kwargs: Any) -> None:
        self.agent = agent
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        if hasattr(self.agent, "run_stream"):
            async with self.agent.run_stream(text, **self.run_kwargs) as result:
                async for delta in result.stream_text(delta=True):
                    if delta:
                        yield AgentEvent.text(delta)
        else:
            result = await self.agent.run(text, **self.run_kwargs)
            if not isinstance(result.output, str):
                raise TypeError("PydanticAIAgent requires a text-output agent")
            if result.output:
                yield AgentEvent.text(result.output)

