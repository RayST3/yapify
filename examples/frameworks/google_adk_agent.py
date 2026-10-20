"""Google ADK Runner + Gemini, with a per-call in-memory session.
Use the isolated install in examples/README.md to keep ADK dependencies separate.
Module: examples.frameworks.google_adk_agent
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import GoogleADKAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from google.adk.agents import Agent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name="yapwrap", user_id="local")
    agent = Agent(
        name="voice_assistant",
        model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        instruction="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
    )
    runner = Runner(agent=agent, app_name="yapwrap", session_service=sessions)
    try:
        voice_agent = GoogleADKAgent(
            runner, user_id="local", session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
        yapper = YapWrap(
            agent=voice_agent,
            stt=DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"]),
            tts=CartesiaTTS(api_key=os.environ["CARTESIA_API_KEY"]),
            vad=SileroVAD(),
        )
        await yapper.startlocal()
    finally:
        await runner.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
