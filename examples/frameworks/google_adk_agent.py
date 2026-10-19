"""Google ADK Runner + Gemini, with a per-call in-memory session.
Use the isolated install in examples/README.md to keep ADK dependencies separate.
Module: examples.frameworks.google_adk_agent
"""

import os

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import GoogleADKAgent


async def main():
    from google.adk.agents import Agent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    os.environ["GOOGLE_API_KEY"] = gemini_key()
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="yapify", user_id="local")
    runner = Runner(
        agent=Agent(name="voice_assistant", model=gemini_model(), instruction=INSTRUCTIONS),
        app_name="yapify",
        session_service=sessions,
    )
    try:
        agent = GoogleADKAgent(
            runner, user_id="local", session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
        await make_yapper(agent).startlocal()
    finally:
        await runner.close()


if __name__ == "__main__":
    run(main)
