"""Claude Agent SDK. Authenticate Claude Code or set ANTHROPIC_API_KEY.
Run: uv run --with claude-agent-sdk python -m examples.frameworks.claude_agent_sdk
Install and voice key setup: examples/README.md.
"""

import os

from examples._shared import INSTRUCTIONS, make_yapper, run
from yapify.agents import ClaudeAgentSDKAgent


async def main():
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        model=os.getenv("CLAUDE_MODEL", "sonnet"),
        system_prompt=INSTRUCTIONS,
        tools=[],
        include_partial_messages=True,
    )
    async with ClaudeSDKClient(options=options) as client:
        await make_yapper(ClaudeAgentSDKAgent(client)).startlocal()


if __name__ == "__main__":
    run(main)
