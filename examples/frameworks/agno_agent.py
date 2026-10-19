"""Agno + Gemini. Run: uv run python -m examples.frameworks.agno_agent
Install and key setup: examples/README.md.
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import AgnoAgent


def create_agent() -> AgnoAgent:
    from agno.agent import Agent
    from agno.models.google import Gemini

    return AgnoAgent(Agent(
        model=Gemini(id=gemini_model(), api_key=gemini_key()),
        instructions=INSTRUCTIONS,
        markdown=False,
    ))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
