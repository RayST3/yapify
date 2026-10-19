"""Claude Agent SDK client adapter."""

from contextlib import aclosing
from typing import Any, AsyncIterator
from yapify.core.base import BaseAgent
from yapify.core.events import AgentEvent


class ClaudeAgentSDKAgent(BaseAgent):
    """Adapt a connected ClaudeSDKClient owned by an async context manager."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        await self.agent.query(text)
        finished = False
        partial = False
        try:
            async with aclosing(self.agent.receive_response()) as stream:
                async for message in stream:
                    kind = type(message).__name__
                    if kind == "StreamEvent":
                        event = message.event
                        if event.get("type") == "message_start":
                            partial = False
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            partial = True
                            if delta.get("text"):
                                yield AgentEvent.text(delta["text"])
                    elif kind == "AssistantMessage":
                        if not partial:
                            for block in message.content:
                                if type(block).__name__ == "TextBlock" and block.text:
                                    yield AgentEvent.text(block.text)
                        partial = False
                    elif kind == "ResultMessage":
                        finished = True
                        if message.is_error:
                            raise RuntimeError(message.result or "Claude agent run failed")
            finished = True
        finally:
            if not finished:
                await self.agent.interrupt()
                # Drain the interrupted run before the next query.
                async for _ in self.agent.receive_response():
                    pass

