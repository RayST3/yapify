"""Hugging Face smolagents adapter."""

import asyncio
from typing import Any, AsyncIterator
from yapwrap.core.base import BaseAgent
from yapwrap.core.events import AgentEvent


class SmolAgentsAgent(BaseAgent):
    """Speak final answers without reading intermediate reasoning/tool logs.

    Synchronous agent work runs in a worker thread. A running thread cannot be
    forcibly cancelled; the next turn waits for it to finish before reusing it.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._pending: Any = None

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._pending is not None:
            try:
                await asyncio.shield(self._pending)
            finally:
                if self._pending.done():
                    self._pending = None
        self._pending = asyncio.create_task(asyncio.to_thread(self.agent.run, text))
        # Retrieve exceptions even if the session ends during the worker.
        self._pending.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:
            result = await asyncio.shield(self._pending)
            if result is not None:
                yield AgentEvent.text(str(result))
        finally:
            if self._pending.done():
                self._pending = None

