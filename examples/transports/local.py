"""Local microphone/speakers with Agno + Gemini Flash Lite.
Run: uv run python -m examples.transports.local
Install and key setup: examples/README.md.
"""

from examples._shared import run
from examples.transports._shared import create_yapper


async def main():
    await create_yapper().startlocal()


if __name__ == "__main__":
    run(main)
