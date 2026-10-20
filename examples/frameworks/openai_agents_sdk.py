"""OpenAI Agents SDK. Set OPENAI_API_KEY and OPENAI_MODEL.
Run: uv run --with openai-agents python -m examples.frameworks.openai_agents_sdk
Install and voice key setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import OpenAIAgentsSDKAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from agents import Agent, SQLiteSession

    agent = Agent(
        name="Voice assistant",
        instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        model=os.environ["OPENAI_MODEL"],
    )
    yapper = YapWrap(
        agent=OpenAIAgentsSDKAgent(agent, session=SQLiteSession("local-voice")),
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
