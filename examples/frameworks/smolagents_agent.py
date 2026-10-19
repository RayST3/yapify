"""smolagents ToolCallingAgent + Gemini's OpenAI-compatible endpoint.
Run: uv run --with "smolagents[openai]" python -m examples.frameworks.smolagents_agent
Install and key setup: examples/README.md. Speaks final answers, not reasoning.
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import SmolAgentsAgent


def create_agent() -> SmolAgentsAgent:
    from smolagents import OpenAIServerModel, ToolCallingAgent

    model = OpenAIServerModel(
        model_id=gemini_model(), api_key=gemini_key(),
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return SmolAgentsAgent(ToolCallingAgent(
        tools=[], model=model, instructions=INSTRUCTIONS, verbosity_level=0,
    ))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
