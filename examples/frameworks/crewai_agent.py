"""CrewAI + Gemini. Speaks the completed crew answer.
Requires Python 3.10-3.13; use the isolated install in examples/README.md.
Module: examples.frameworks.crewai_agent
"""

import asyncio
import os

from yapify import YapWrap
from yapify.agents import CrewAIAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD


async def main():
    from crewai import Agent, Crew, LLM, Task

    agent = Agent(
        role="Voice assistant",
        goal="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
        backstory="You help people through short spoken conversations.",
        llm=LLM(
            model="gemini/" + os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
        ),
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description="Reply to the user: {input}",
        expected_output="A brief, conversational answer without markdown.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    yapper = YapWrap(
        agent=CrewAIAgent(crew),
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
