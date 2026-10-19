"""Standalone WebSocket server with Agno + Gemini Flash Lite.
Run: uv run python -m examples.transports.websocket
Browser client and installation: examples/README.md.
"""

import asyncio
import os

from yapify import YapWrap
from yapify.agents import AgnoAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD


def create_yapper() -> YapWrap:
    """Create a separate agent and voice components for each connection."""
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
    return YapWrap(
        agent=AgnoAgent(agent),
        stt=DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"]),
        tts=CartesiaTTS(api_key=os.environ["CARTESIA_API_KEY"]),
        vad=SileroVAD(),
    )


async def main():
    yapper = create_yapper()
    print("Voice WebSocket listening on ws://127.0.0.1:8765")
    await yapper.server_ws(host="127.0.0.1", port=8765, session_factory=create_yapper)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
