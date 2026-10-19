"""LlamaIndex FunctionAgent workflow with Gemini.
Run: uv run --with llama-index-core --with llama-index-llms-google-genai python -m examples.frameworks.llamaindex_agent
Install and voice key setup: examples/README.md.
"""

import asyncio
import os

from yapify import YapWrap
from yapify.agents import LlamaIndexAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD


async def main():
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.llms.google_genai import GoogleGenAI

    agent = FunctionAgent(
        llm=GoogleGenAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        ),
        tools=[],
        system_prompt="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
    )
    yapper = YapWrap(
        agent=LlamaIndexAgent(agent, ctx=Context(agent)),
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
