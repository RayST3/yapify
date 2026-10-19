"""LangGraph and LangChain create_agent adapter."""

from contextlib import aclosing
from typing import Any, AsyncIterator
from yapify.core.events import AgentEvent


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


class LangGraphAgent:
    """Adapt a compiled graph, optionally passing config (e.g. thread_id)."""

    def __init__(self, graph: Any, **run_kwargs: Any) -> None:
        self.graph = graph
        self.run_kwargs = run_kwargs

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        inputs = {"messages": [{"role": "user", "content": text}]}
        if hasattr(self.graph, "astream_events"):
            stream = self.graph.astream_events(inputs, **{**self.run_kwargs, "version": "v2"})
            async with aclosing(stream):
                async for event in stream:
                    kind, data = event.get("event"), event.get("data", {})
                    if kind == "on_chat_model_stream":
                        content = _text(getattr(data.get("chunk"), "content", ""))
                        if content:
                            yield AgentEvent.text(content)
                    elif kind == "on_tool_start":
                        yield AgentEvent.tool_start(event.get("name", "tool"), data.get("input"))
                    elif kind == "on_tool_end":
                        yield AgentEvent.tool_end(event.get("name", "tool"), data.get("output"))
            return  # Never execute tools again as a fallback after an error.

        async with aclosing(self.graph.astream(inputs, **self.run_kwargs)) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    yield AgentEvent.text(chunk)
                    continue
                states = [chunk] if "messages" in chunk else chunk.values()
                for state in states:
                    if not isinstance(state, dict):
                        continue
                    for message in state.get("messages", []):
                        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
                        if role in ("assistant", "ai"):
                            content = _text(message.get("content") if isinstance(message, dict) else message.content)
                            if content:
                                yield AgentEvent.text(content)
