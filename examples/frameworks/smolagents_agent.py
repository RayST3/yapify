"""smolagents ToolCallingAgent + Gemini's OpenAI-compatible endpoint.
Run: uv run --with "smolagents[openai]" python -m examples.frameworks.smolagents_agent
Install and key setup: examples/README.md. Speaks final answers, not reasoning.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import SmolAgentsAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from smolagents import OpenAIServerModel, ToolCallingAgent

    model = OpenAIServerModel(
        model_id=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        verbosity_level=0,
    )
    yapper = YapWrap(
        agent=SmolAgentsAgent(agent),
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
