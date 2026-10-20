"""Adapt plain Python callables without importing an agent framework."""

import asyncio
import inspect
from collections.abc import AsyncIterable, Iterable
from typing import Any, AsyncIterator, Callable

from yapwrap.core.base import BaseAgent
from yapwrap.core.events import AgentEvent


class CallableAgent(BaseAgent):
    """Accept text, events, and synchronous or asynchronous streams of either.

    Async callables are preferred for cancellable network work. Synchronous
    callables run in a worker thread; Python cannot interrupt that thread.
    """

    def __init__(self, func: Callable[[str], Any]) -> None:
        self.func = func

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        if inspect.iscoroutinefunction(self.func) or inspect.isasyncgenfunction(self.func):
            result = self.func(text)
        else:
            result = await asyncio.to_thread(self.func, text)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, (str, AgentEvent)):
            yield self._event(result)
        elif isinstance(result, AsyncIterable):
            stream = result.__aiter__()
            try:
                async for item in stream:
                    yield self._event(item)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        elif isinstance(result, Iterable):
            iterator = iter(result)
            end = object()
            while True:
                item = await asyncio.to_thread(next, iterator, end)
                if item is end:
                    break
                yield self._event(item)
        else:
            raise TypeError("CallableAgent must return text, AgentEvent, or an iterable of either")

    @staticmethod
    def _event(item: Any) -> AgentEvent:
        if isinstance(item, AgentEvent):
            return item
        if isinstance(item, str):
            return AgentEvent.text(item)
        raise TypeError("Agent stream items must be strings or AgentEvent objects")
