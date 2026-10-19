"""Browser WebRTC with Agno + Gemini Flash Lite and WebSocket signaling.
Run: uv run python -m examples.transports.webrtc
Open http://localhost:8000. Install and key setup: examples/README.md.
"""

import asyncio
import logging
import os
from pathlib import Path

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from yapify import YapWrap
from yapify.agents import AgnoAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD
from yapify.transports import WebRTCTransport

app = FastAPI(title="Yapify WebRTC Voice")
logger = logging.getLogger("yapify.example.webrtc")


def create_yapper() -> YapWrap:
    """Create a separate agent and voice components for each connection."""
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
    return YapWrap(
        agent=AgnoAgent(agent),
        stt=DeepgramSTT(api_key=os.environ["DEEPGRAM_API_KEY"]),
        tts=CartesiaTTS(api_key=os.environ["CARTESIA_API_KEY"]),
        vad=SileroVAD(),
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path(__file__).with_name("voice-client.html").read_text(encoding="utf-8")
    return html.replace("<body>", '<body data-transport="webrtc">')


@app.websocket("/ws/offer")
async def offer(websocket: WebSocket):
    await websocket.accept()
    missing = [name for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY") if not os.getenv(name)]
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    if missing:
        await websocket.send_json({"type": "error", "message": "Set environment variables: " + ", ".join(missing)})
        await websocket.close(code=1011)
        return

    # Host candidates suffice for localhost. Deployments configure STUN/TURN.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    transport = WebRTCTransport(pc=pc)
    tasks = []
    ended = asyncio.Event()

    @pc.on("datachannel")
    def on_channel(channel):
        if channel.label == "events":
            transport.data_channel = channel

    @pc.on("connectionstatechange")
    def on_state():
        if pc.connectionState in ("failed", "closed"):
            ended.set()

    try:
        description = await asyncio.wait_for(websocket.receive_json(), timeout=30)
        if description.get("type") != "offer":
            raise ValueError("Expected a WebRTC offer")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=description["sdp"], type="offer"))
        if not any(r.track and r.track.kind == "audio" for r in pc.getReceivers()):
            raise ValueError("The offer must include microphone audio")
        pc.addTrack(transport.create_output_track())
        await pc.setLocalDescription(await pc.createAnswer())
        await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
        voice = asyncio.create_task(create_yapper().run(transport))
        tasks = [voice, asyncio.create_task(websocket.receive()), asyncio.create_task(ended.wait())]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if voice in done:
            await voice
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebRTC session failed")
        try:
            await websocket.send_json({"type": "error", "message": "Voice session failed. See the server log."})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await transport.close()
        await pc.close()
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
