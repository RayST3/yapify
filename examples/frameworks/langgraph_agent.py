"""LangChain create_agent (built on LangGraph) + Gemini.
Run: uv run --with langchain --with langchain-google-genai python -m examples.frameworks.langgraph_agent
Install and key setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import LangGraphAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.memory import InMemorySaver

    graph = create_agent(
        model=ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        ),
        tools=[],
        system_prompt="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        checkpointer=InMemorySaver(),
    )
    yapper = YapWrap(
        agent=LangGraphAgent(graph, config={"configurable": {"thread_id": "local-voice"}}),
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
