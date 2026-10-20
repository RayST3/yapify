"""Browser voice example powered by an Agno agent.

Install: uv sync --group examples --extra web --extra silero --extra deepgram --extra cartesia
Set GOOGLE_API_KEY (or GEMINI_API_KEY), DEEPGRAM_API_KEY, and CARTESIA_API_KEY.
GEMINI_MODEL optionally overrides the Agno agent's Gemini model.
Run: uv run python -m examples.transports.fastapi_websocket
Or:  uv run uvicorn examples.transports.fastapi_websocket:app --reload
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState

from yapwrap import YapWrap
from yapwrap.agents import AgnoAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD

app = FastAPI(title="YapWrap Voice Server")
logger = logging.getLogger("yapwrap.example")


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    missing = [name for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY") if not os.getenv(name)]
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    if missing:
        await websocket.send_json({"type": "error", "message": "Set server environment variables: " + ", ".join(missing)})
        await websocket.close(code=1011)
        return
    try:
        from agno.agent import Agent
        from agno.models.google import Gemini

        agent = Agent(
            model=Gemini(
                id=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
                api_key=os.getenv("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"],
            ),
            instructions="You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown.",
            markdown=False,
        )
        yapper = YapWrap(
            agent=AgnoAgent(agent),
            stt=DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"]),
            tts=CartesiaTTS(api_key=os.environ["CARTESIA_API_KEY"]),
            vad=SileroVAD(),
        )
        await yapper.server_ws(ws=websocket)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("Voice session failed")
        try:
            await websocket.send_json({"type": "error", "message": "Voice session failed. See the server log for details."})
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return Path(__file__).with_name("voice-client.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
