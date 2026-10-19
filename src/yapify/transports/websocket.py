"""Binary PCM and JSON controls over an accepted WebSocket."""

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncIterator

from yapify.core.audio import DEFAULT_SAMPLE_RATE
from yapify.core.events import SessionEvent


class WebSocketTransport:
    """Support FastAPI/Starlette and the websockets asyncio library.

    Binary messages are signed little-endian PCM16 mono at sample_rate.
    The application accepts/authenticates the socket before handing it over.
    """

    def __init__(self, ws: Any, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.ws = ws
        self.sample_rate = sample_rate
        self._lock = asyncio.Lock()
        self._is_open = True

    async def audio_in(self) -> AsyncIterator[bytes]:
        try:
            while self._is_open:
                if hasattr(self.ws, "receive"):
                    message = await self.ws.receive()
                    if message.get("type") in ("websocket.disconnect", "websocket.close"):
                        break
                    if message.get("bytes"):
                        yield message["bytes"]
                    elif message.get("text"):
                        self._handle_text_control(message["text"])
                elif hasattr(self.ws, "recv"):
                    try:
                        message = await self.ws.recv()
                    except Exception as exc:
                        from websockets.exceptions import ConnectionClosed
                        if isinstance(exc, ConnectionClosed):
                            break
                        raise
                    if isinstance(message, bytes):
                        yield message
                    elif isinstance(message, str):
                        self._handle_text_control(message)
                else:
                    raise TypeError("Expected a Starlette or websockets WebSocket")
        finally:
            self._is_open = False

    def _handle_text_control(self, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict) and data.get("type") == "stop":
            self._is_open = False

    async def send_audio(self, pcm: bytes) -> None:
        if not self._is_open or not pcm:
            return
        async with self._lock:
            if hasattr(self.ws, "send_bytes"):
                await self.ws.send_bytes(pcm)
            else:
                await self.ws.send(pcm)

    async def stop_playback(self) -> None:
        await self._send_json({"type": "stop_playback"})

    async def send_event(self, event: SessionEvent) -> None:
        """Forward text and turn events beside the binary PCM messages."""
        await self._send_json(asdict(event))

    async def _send_json(self, payload: dict) -> None:
        if not self._is_open:
            return
        async with self._lock:
            message = json.dumps(payload)
            if hasattr(self.ws, "send_text"):
                await self.ws.send_text(message)
            else:
                await self.ws.send(message)
