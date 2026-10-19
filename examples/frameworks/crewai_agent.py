"""CrewAI + Gemini. Speaks the completed crew answer.
Requires Python 3.10-3.13; use the isolated install in examples/README.md.
Module: examples.frameworks.crewai_agent
"""

from examples._shared import INSTRUCTIONS, gemini_key, gemini_model, make_yapper, run
from yapify.agents import CrewAIAgent


def create_agent() -> CrewAIAgent:
    from crewai import Agent, Crew, LLM, Task

    agent = Agent(
        role="Voice assistant", goal=INSTRUCTIONS,
        backstory="You help people through short spoken conversations.",
        llm=LLM(model="gemini/" + gemini_model(), api_key=gemini_key()),
        verbose=False, allow_delegation=False,
    )
    task = Task(
        description="Reply to the user: {input}",
        expected_output="A brief, conversational answer without markdown.",
        agent=agent,
    )
    return CrewAIAgent(Crew(agents=[agent], tasks=[task], verbose=False))


async def main():
    await make_yapper(create_agent()).startlocal()


if __name__ == "__main__":
    run(main)
