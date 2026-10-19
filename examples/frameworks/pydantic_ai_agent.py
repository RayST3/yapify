"""Pydantic AI + its current GoogleModel/GoogleProvider integration.
Run: uv run --with "pydantic-ai-slim[google]" python -m examples.frameworks.pydantic_ai_agent
Install and key setup: examples/README.md.
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import PydanticAIAgent


def create_agent() -> PydanticAIAgent:
    from pydantic_ai import Agent
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    model = GoogleModel(gemini_model(), provider=GoogleProvider(api_key=gemini_key()))
    return PydanticAIAgent(Agent(model, instructions=INSTRUCTIONS))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
