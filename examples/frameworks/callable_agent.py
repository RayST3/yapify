"""Plain Python echo: no model API or agent framework required.
Run: uv run python -m examples.frameworks.callable_agent
Set DEEPGRAM_API_KEY and CARTESIA_API_KEY. Setup: examples/README.md.
"""

from examples._shared import make_yapper, run
from yapify.agents import CallableAgent


async def echo(text: str):
    yield f"You said: {text}"


async def main():
    await make_yapper(CallableAgent(echo)).startlocal()


if __name__ == "__main__":
    run(main)
