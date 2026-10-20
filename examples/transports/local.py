"""Local microphone/speakers with Agno + Gemini Flash Lite.
Run: uv run python -m examples.transports.local
Install and key setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import AgnoAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from agno.agent import Agent
    from agno.models.google import Gemini

    agent = Agent(
        model=Gemini(
            id=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        ),
        instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        markdown=False,
    )
    yapper = YapWrap(
        agent=AgnoAgent(agent),
        stt=DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"]),
        tts=CartesiaTTS(api_key=os.environ["CARTESIA_API_KEY"]),
        vad=SileroVAD(),
    )
    await yapper.startlocal()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
