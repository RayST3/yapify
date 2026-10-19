"""CrewAI Crew adapter."""

import asyncio
from typing import Any, AsyncIterator
from yapify.core.base import BaseAgent
from yapify.core.events import AgentEvent


class CrewAIAgent(BaseAgent):
    """Speak a Crew's final output. Tasks receive the utterance as {input}."""

    def __init__(self, crew_or_agent: Any) -> None:
        self.crew = crew_or_agent

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        inputs = {"input": text}
        if hasattr(self.crew, "akickoff"):
            result = await self.crew.akickoff(inputs=inputs)
        elif hasattr(self.crew, "kickoff_async"):
            result = await self.crew.kickoff_async(inputs=inputs)
        else:
            result = await asyncio.to_thread(self.crew.kickoff, inputs=inputs)
        if result.raw:
            yield AgentEvent.text(result.raw)

