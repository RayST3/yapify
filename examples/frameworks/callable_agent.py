"""Plain Python echo: no model API or agent framework required.
Run: uv run python -m examples.frameworks.callable_agent
Set DEEPGRAM_API_KEY and CARTESIA_API_KEY. Setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import CallableAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def echo(text: str):
    yield f"You said: {text}"


async def main():
    yapper = YapWrap(
        agent=CallableAgent(echo),
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
