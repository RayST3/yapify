"""OpenAI Agents SDK. Set OPENAI_API_KEY and OPENAI_MODEL.
Run: uv run --with openai-agents python -m examples.frameworks.openai_agents_sdk
Install and voice key setup: examples/README.md.
"""

import os

from examples._shared import INSTRUCTIONS, make_yapper, run
from yapify.agents import OpenAIAgentsSDKAgent


def create_agent() -> OpenAIAgentsSDKAgent:
    from agents import Agent, SQLiteSession

    agent = Agent(name="Voice assistant", instructions=INSTRUCTIONS, model=os.environ["OPENAI_MODEL"])
    return OpenAIAgentsSDKAgent(agent, session=SQLiteSession("local-voice"))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
