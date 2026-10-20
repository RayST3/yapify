"""SDK event contracts, without model requests or credentials."""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest

from yapwrap.agents import (
    ClaudeAgentSDKAgent, CrewAIAgent, GoogleADKAgent, LangGraphAgent,
    LlamaIndexAgent, MicrosoftAgentFrameworkAgent, OpenAIAgentsSDKAgent,
    PydanticAIAgent, SmolAgentsAgent,
)


async def collect(agent):
    return [event.content async for event in agent.respond("hi")]


async def test_langgraph_filters_blocks_and_never_retries_tools():
    class Graph:
        async def astream_events(self, inputs, **kwargs):
            assert kwargs == {"version": "v2", "config": {"configurable": {"thread_id": "one"}}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": NS(content=[
                {"type": "reasoning", "text": "private"}, {"type": "text", "text": "hello"},
            ])}}
            raise RuntimeError("tool failed")

        async def astream(self, inputs):
            pytest.fail("An execution failure must not repeat the graph/tools")
            yield

    stream = LangGraphAgent(Graph(), config={"configurable": {"thread_id": "one"}}).respond("hi")
    assert (await anext(stream)).content == "hello"
    with pytest.raises(RuntimeError, match="tool failed"):
        await anext(stream)


async def test_pydantic_current_output_and_stream_cleanup():
    closed = []

    class Agent:
        @asynccontextmanager
        async def run_stream(self, prompt):
            async def tokens(*, delta):
                assert delta is True
                yield "hello"
                yield " there"
            try:
                yield NS(stream_text=tokens)
            finally:
                closed.append(True)

    stream = PydanticAIAgent(Agent()).respond("hi")
    assert (await anext(stream)).content == "hello"
    await stream.aclose()
    assert closed == [True]

    async def run(prompt):
        return NS(output="current output", data="obsolete output")
    assert await collect(PydanticAIAgent(NS(run=run))) == ["current output"]


async def test_openai_streamed_events_and_cancel():
    class Result:
        cancelled = False

        async def stream_events(self):
            yield NS(type="raw_response_event", data=NS(type="response.reasoning_text.delta", delta="private"))
            yield NS(type="raw_response_event", data=NS(type="response.output_text.delta", delta="hello"))
            yield NS(type="run_item_stream_event", item=NS(text="hello"))

        def cancel(self):
            self.cancelled = True

    result = Result()

    def run_streamed(agent, *, input, session):
        assert input == "hi" and session == "conversation"
        return result

    adapter = OpenAIAgentsSDKAgent(object(), NS(run_streamed=run_streamed), session="conversation")
    assert await collect(adapter) == ["hello"]
    assert not result.cancelled
    stream = adapter.respond("hi")
    await anext(stream)
    await stream.aclose()
    assert result.cancelled


async def test_adk_partial_and_aggregate_per_model_response():
    pytest.importorskip("google.genai")
    closed = []

    async def run_async(**kwargs):
        assert kwargs["session_id"] == "one"
        assert kwargs["user_id"] == "local"
        assert kwargs["new_message"].parts[0].text == "hi"
        try:
            for text, partial in [("hello", True), ("hello", False), (" again", False)]:
                yield NS(partial=partial, content=NS(role="model", parts=[
                    NS(text="private", thought=True), NS(text=text, thought=False),
                ]))
        finally:
            closed.append(True)

    adapter = GoogleADKAgent(NS(run_async=run_async), user_id="local", session_id="one")
    assert await collect(adapter) == ["hello", " again"]
    assert closed == [True]


def sdk_event(name, **kwargs):
    return type(name, (), kwargs)()


async def test_claude_does_not_repeat_partial_text_or_read_thoughts():
    class Client:
        async def query(self, text):
            assert text == "hi"

        async def receive_response(self):
            yield sdk_event("StreamEvent", event={"type": "message_start"})
            yield sdk_event("StreamEvent", event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}})
            yield sdk_event("AssistantMessage", content=[sdk_event("TextBlock", text="hello")])
            yield sdk_event("AssistantMessage", content=[sdk_event("ThinkingBlock", thinking="private"), sdk_event("TextBlock", text=" again")])
            yield sdk_event("ResultMessage", is_error=False, result="hello again")

        async def interrupt(self):
            pytest.fail("Completed responses should not be interrupted")

    assert await collect(ClaudeAgentSDKAgent(Client())) == ["hello", " again"]


async def test_claude_interrupt_drains_before_next_query():
    operations = []

    class Client:
        async def query(self, text):
            operations.append("query")

        async def receive_response(self):
            if "interrupt" in operations:
                operations.append("drain")
                yield sdk_event("ResultMessage", is_error=False)
            else:
                yield sdk_event("AssistantMessage", content=[sdk_event("TextBlock", text="hello")])

        async def interrupt(self):
            operations.append("interrupt")

    stream = ClaudeAgentSDKAgent(Client()).respond("hi")
    await anext(stream)
    await stream.aclose()
    assert operations == ["query", "interrupt", "drain"]


async def test_crewai_uses_native_async_final_output():
    async def akickoff(*, inputs):
        assert inputs == {"input": "hi"}
        return NS(raw="final answer")
    assert await collect(CrewAIAgent(NS(akickoff=akickoff))) == ["final answer"]


async def test_llamaindex_stream_and_cancellation():
    class Handler:
        cancelled = False

        async def stream_events(self):
            yield sdk_event("AgentStream", delta="hello")
            yield sdk_event("AgentOutput", response="hello")

        def __await__(self):
            async def finish():
                return "hello"
            return finish().__await__()

        async def cancel_run(self):
            self.cancelled = True

    handler = Handler()
    def run(*, user_msg, ctx):
        assert user_msg == "hi" and ctx == "context"
        return handler
    adapter = LlamaIndexAgent(NS(run=run), ctx="context")
    assert await collect(adapter) == ["hello"]
    assert not handler.cancelled
    stream = adapter.respond("hi")
    await anext(stream)
    await stream.aclose()
    assert handler.cancelled


async def test_microsoft_stream_context_closes_on_interruption():
    closed = []
    @asynccontextmanager
    async def run(text, *, stream, session):
        assert stream is True and session == "one"
        async def updates():
            yield NS(text="hello")
            yield NS(text=" there")
        try:
            yield updates()
        finally:
            closed.append(True)
    stream = MicrosoftAgentFrameworkAgent(NS(run=run), session="one").respond("hi")
    assert (await anext(stream)).content == "hello"
    await stream.aclose()
    assert closed == [True]


async def test_smolagents_worker_finishes_before_reusing_agent():
    started, finish = threading.Event(), threading.Event()
    calls = []
    loop_thread = threading.get_ident()

    def run(text):
        assert threading.get_ident() != loop_thread
        calls.append(text)
        started.set()
        assert finish.wait(5)
        return text

    agent = SmolAgentsAgent(NS(run=run))
    first = asyncio.create_task(anext(agent.respond("first")))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(collect(agent))
        await asyncio.sleep(0)
        assert calls == ["first"]
        finish.set()
        assert await asyncio.wait_for(second, 5) == ["hi"]
        assert calls == ["first", "hi"]
    finally:
        finish.set()


async def test_current_langchain_create_agent_without_network():
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    graph = create_agent(model=FakeListChatModel(responses=["Hello."]), tools=[])
    assert "".join(await collect(LangGraphAgent(graph))) == "Hello."


async def test_current_pydantic_ai_without_network():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text="Hello."))
    assert "".join(await collect(PydanticAIAgent(agent))) == "Hello."
