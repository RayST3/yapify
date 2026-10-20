"""LlamaIndex FunctionAgent / AgentWorkflow adapter."""

from typing import Any, AsyncIterator
from yapwrap.core.base import BaseAgent
from yapwrap.core.events import AgentEvent


class LlamaIndexAgent(BaseAgent):
    """Adapt a workflow; pass ctx=Context(agent) to retain conversation state."""

    def __init__(self, agent: Any, **run_kwargs: Any) -> None:
        self.agent = agent
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        handler = self.agent.run(user_msg=text, **self.run_kwargs)
        finished = False
        try:
            async for event in handler.stream_events():
                if type(event).__name__ == "AgentStream" and event.delta:
                    yield AgentEvent.text(event.delta)
            await handler  # Surface failures without repeating the final text.
            finished = True
        finally:
            if not finished:
                await handler.cancel_run()

