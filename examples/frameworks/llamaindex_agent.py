"""LlamaIndex FunctionAgent workflow with Gemini.
Run: uv run --with llama-index-core --with llama-index-llms-google-genai python -m examples.frameworks.llamaindex_agent
Install and voice key setup: examples/README.md.
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import LlamaIndexAgent


def create_agent() -> LlamaIndexAgent:
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from llama_index.llms.google_genai import GoogleGenAI

    agent = FunctionAgent(
        llm=GoogleGenAI(model=gemini_model(), api_key=gemini_key()),
        tools=[], system_prompt=INSTRUCTIONS,
    )
    return LlamaIndexAgent(agent, ctx=Context(agent))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
