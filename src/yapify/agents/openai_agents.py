"""OpenAI Agents SDK adapter; the SDK is an optional dependency."""

from contextlib import aclosing
from typing import Any, AsyncIterator
from yapify.core.events import AgentEvent


class OpenAIAgentsSDKAgent:
    """Adapt an SDK Agent, with optional Runner and run options (e.g. session)."""

    def __init__(self, agent: Any, runner: Any = None, **run_kwargs: Any) -> None:
        self.agent = agent
        self.runner = runner
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        runner = self.runner
        if runner is None:
            from agents import Runner
            runner = Runner
        result = runner.run_streamed(self.agent, input=text, **self.run_kwargs)
        finished = False
        try:
            async with aclosing(result.stream_events()) as stream:
                async for event in stream:
                    if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
                        if event.data.delta:
                            yield AgentEvent.text(event.data.delta)
            finished = True
        finally:
            if not finished:
                result.cancel()

