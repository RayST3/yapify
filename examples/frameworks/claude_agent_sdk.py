"""Claude Agent SDK. Authenticate Claude Code or set ANTHROPIC_API_KEY.
Run: uv run --with claude-agent-sdk python -m examples.frameworks.claude_agent_sdk
Install and voice key setup: examples/README.md.
"""

import asyncio
import os

from yapwrap import YapWrap
from yapwrap.agents import ClaudeAgentSDKAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD


async def main():
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        model=os.getenv("CLAUDE_MODEL", "sonnet"),
        system_prompt="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        tools=[],
        include_partial_messages=True,
    )
    async with ClaudeSDKClient(options=options) as client:
        yapper = YapWrap(
            agent=ClaudeAgentSDKAgent(client),
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
