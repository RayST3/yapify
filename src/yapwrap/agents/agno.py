"""Agno Agent adapter for yapwrap."""

import asyncio
import inspect
from typing import Any, AsyncIterator

from yapwrap.core.base import BaseAgent
from yapwrap.core.events import AgentEvent


class AgnoAgent(BaseAgent):
    """Adapts an Agno Agent into a yapwrap Agent."""

    def __init__(self, agent: Any, **run_kwargs: Any) -> None:
        self.agent = agent
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        """Stream events from an Agno agent run."""
        # Agno agents support async streaming with stream=True
        if hasattr(self.agent, "arun"):
            stream = self.agent.arun(text, **{**self.run_kwargs, "stream": True, "stream_events": True})
            if inspect.isawaitable(stream):
                stream = await stream
            try:
                async for event in stream:
                    kind = getattr(event, "event", "")
                    content = getattr(event, "content", None)

                    if kind == "RunContent" and isinstance(content, str):
                        yield AgentEvent.text(content)
                    elif kind in ("ToolCallStarted", "ToolCallCompleted"):
                        tool = getattr(event, "tool", None)
                        name = getattr(tool, "tool_name", "tool")
                        if kind == "ToolCallStarted":
                            yield AgentEvent.tool_start(name, getattr(tool, "tool_args", None))
                        else:
                            yield AgentEvent.tool_end(name, getattr(tool, "result", None))
                    elif kind in ("RunError", "RunCancelled"):
                        raise RuntimeError(f"Agno {kind}: {content or ''}")
                    elif isinstance(content, str) and content and not kind:
                        yield AgentEvent.text(content)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        elif hasattr(self.agent, "run"):
            # Synchronous fallback
            resp = await asyncio.to_thread(self.agent.run, text, **self.run_kwargs)
            content = getattr(resp, "content", str(resp))
            yield AgentEvent.text(content)
        else:
            raise TypeError("AgnoAgent expects an object with arun() or run()")

