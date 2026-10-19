"""Microsoft Agent Framework. Set OPENAI_API_KEY and OPENAI_MODEL.
Run: uv run --with agent-framework-openai python -m examples.frameworks.microsoft_agent
Install and voice key setup: examples/README.md.
"""

import os

from examples._shared import INSTRUCTIONS, make_yapper, run
from yapify.agents import MicrosoftAgentFrameworkAgent


def create_agent() -> MicrosoftAgentFrameworkAgent:
    from agent_framework_openai import OpenAIChatClient

    agent = OpenAIChatClient(model=os.environ["OPENAI_MODEL"]).as_agent(
        name="VoiceAssistant", instructions=INSTRUCTIONS,
    )
    return MicrosoftAgentFrameworkAgent(agent, session=agent.create_session())


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
