"""Google ADK Runner adapter."""

from contextlib import aclosing
from typing import Any, AsyncIterator
from yapify.core.events import AgentEvent


class GoogleADKAgent:
    """Adapt a Runner and an existing ADK session, owned by the application."""

    def __init__(self, runner: Any, *, user_id: str, session_id: str, run_config: Any = None) -> None:
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.run_config = run_config

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        from google.genai import types

        message = types.Content(role="user", parts=[types.Part(text=text)])
        stream = self.runner.run_async(
            user_id=self.user_id, session_id=self.session_id,
            new_message=message, run_config=self.run_config,
        )
        partial = False
        async with aclosing(stream):
            async for event in stream:
                if getattr(event, "error_code", None):
                    raise RuntimeError(event.error_message or event.error_code)
                content = getattr(event, "content", None)
                if content is None or content.role == "user":
                    continue
                output = "".join(
                    part.text for part in (content.parts or [])
                    if part.text and not getattr(part, "thought", False)
                )
                if event.partial:
                    partial = True
                    if output:
                        yield AgentEvent.text(output)
                else:
                    # ADK emits an aggregate after partial tokens of each response.
                    if output and not partial:
                        yield AgentEvent.text(output)
                    partial = False

