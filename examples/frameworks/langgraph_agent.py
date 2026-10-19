"""LangChain create_agent (built on LangGraph) + Gemini.
Run: uv run --with langchain --with langchain-google-genai python -m examples.frameworks.langgraph_agent
Install and key setup: examples/README.md.
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import LangGraphAgent


def create_agent() -> LangGraphAgent:
    from langchain.agents import create_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.memory import InMemorySaver

    graph = create_agent(
        model=ChatGoogleGenerativeAI(model=gemini_model(), api_key=gemini_key()),
        tools=[],
        system_prompt=INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )
    return LangGraphAgent(graph, config={"configurable": {"thread_id": "local-voice"}})


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
