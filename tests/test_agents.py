"""Test agent adapters."""

import pytest

from yapify.agents.callable import CallableAgent
from yapify.agents.agno import AgnoAgent
from yapify.agents.langgraph import LangGraphAgent
from yapify.agents.pydantic_ai import PydanticAIAgent
from yapify.core.events import AgentEvent


@pytest.mark.asyncio
async def test_callable_agent_generator():
    async def sample_generator(text: str):
        yield "Echo: "
        yield text

    agent = CallableAgent(sample_generator)
    events = []
    async for evt in agent.respond("hello"):
        events.append(evt)

    assert len(events) == 2
    assert events[0].content == "Echo: "
    assert events[1].content == "hello"


@pytest.mark.asyncio
async def test_callable_agent_coroutine():
    async def sample_coroutine(text: str):
        return f"Hello {text}"

    agent = CallableAgent(sample_coroutine)
    events = []
    async for evt in agent.respond("world"):
        events.append(evt)

    assert len(events) == 1
    assert events[0].content == "Hello world"


@pytest.mark.asyncio
async def test_agno_agent_adapter():
    class DummyAgnoEvent:
        def __init__(self, event, content):
            self.event = event
            self.content = content

    class DummyAgnoAgent:
        def arun(self, text, stream=True, stream_events=True):
            async def generator():
                yield DummyAgnoEvent("RunContent", "Hello from ")
                yield DummyAgnoEvent("RunContent", "Agno!")
            return generator()

    agent = AgnoAgent(DummyAgnoAgent())
    events = []
    async for evt in agent.respond("hi"):
        events.append(evt)

    assert len(events) == 2
    assert events[0].content == "Hello from "
    assert events[1].content == "Agno!"


@pytest.mark.asyncio
async def test_langgraph_adapter():
    class DummyLangGraph:
        async def astream(self, inputs):
            yield {"messages": [{"role": "assistant", "content": "LangGraph reply"}]}

    agent = LangGraphAgent(DummyLangGraph())
    events = []
    async for evt in agent.respond("hi"):
        events.append(evt)

    assert len(events) == 1
    assert events[0].content == "LangGraph reply"


@pytest.mark.asyncio
async def test_callable_sync_generator_and_worker_thread():
    import threading

    main_thread = threading.get_ident()
    def reply(text):
        assert threading.get_ident() != main_thread
        yield "Hello "
        yield text

    assert [event.content async for event in CallableAgent(reply).respond("there")] == ["Hello ", "there"]


@pytest.mark.asyncio
async def test_callable_coroutine_returning_async_iterator():
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "finished", False):
                raise StopAsyncIteration
            self.finished = True
            return "hello"

    async def reply(text):
        return Stream()

    assert [event.content async for event in CallableAgent(reply).respond("hi")] == ["hello"]


@pytest.mark.asyncio
async def test_agno_awaitable_stream_and_session_options():
    from types import SimpleNamespace

    class Agent:
        async def arun(self, text, **kwargs):
            assert kwargs == {"session_id": "session-1", "stream": True, "stream_events": True}
            async def stream():
                yield SimpleNamespace(event="RunContent", content="Hello")
                yield SimpleNamespace(event="RunCompleted", content="Hello")
                yield SimpleNamespace(event="ToolCallStarted", tool=SimpleNamespace(tool_name="search", tool_args={"q": "test"}))
            return stream()

    events = [event async for event in AgnoAgent(Agent(), session_id="session-1").respond("hi")]
    assert len(events) == 2  # Final aggregate content must not be spoken twice.
    assert events[0].content == "Hello"
    assert events[1].args == {"q": "test"}

