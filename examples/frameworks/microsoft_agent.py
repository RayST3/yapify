"""Microsoft Agent Framework. Set OPENAI_API_KEY and OPENAI_MODEL.
Run: uv run --with agent-framework-openai python -m examples.frameworks.microsoft_agent
Install and voice key setup: examples/README.md.
"""

import asyncio
import os

from yapify import YapWrap
from yapify.agents import MicrosoftAgentFrameworkAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD


async def main():
    from agent_framework_openai import OpenAIChatClient

    agent = OpenAIChatClient(model=os.environ["OPENAI_MODEL"]).as_agent(
        name="VoiceAssistant",
        instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
    )
    yapper = YapWrap(
        agent=MicrosoftAgentFrameworkAgent(agent, session=agent.create_session()),
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
