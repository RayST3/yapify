"""Standalone WebSocket server with Agno + Gemini Flash Lite.
Run: uv run python -m examples.transports.websocket
Browser client and installation: examples/README.md.
"""

from examples._shared import run
from examples.transports._shared import create_yapper


async def main():
    yapper = create_yapper()
    print("Voice WebSocket listening on ws://127.0.0.1:8765")
    await yapper.server_ws(host="127.0.0.1", port=8765, session_factory=create_yapper)


if __name__ == "__main__":
    run(main)
