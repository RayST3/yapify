"""Pydantic AI + its current GoogleModel/GoogleProvider integration.
Run: uv run --with "pydantic-ai-slim[google]" python -m examples.frameworks.pydantic_ai_agent
Install and key setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import PydanticAIAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    model = GoogleModel(
        os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        provider=GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]),
    )
    agent = Agent(model, instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.")
    yapper = YapWrap(
        agent=PydanticAIAgent(agent),
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
