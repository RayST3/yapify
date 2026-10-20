"""Convenience entry points over the framework-independent VoiceSession."""

import asyncio
from typing import Any, Callable, Optional

from yapwrap.core.audio import DEFAULT_SAMPLE_RATE
from yapwrap.core.memory import InMemoryHistory
from yapwrap.core.base import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport
from yapwrap.core.session import VoiceSession


class YapWrap:
    """A voice conversation with pluggable providers and transport.

    Create a separate wrapper and stateful components for each concurrent client.
    Agent frameworks and web servers are optional integrations.
    """

    def __init__(
        self,
        agent: BaseAgent,
        tts: BaseTTS,
        stt: BaseSTT,
        vad: BaseVAD,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        memory: Optional[InMemoryHistory] = None,
    ) -> None:
        self.agent = agent
        self.tts = tts
        self.stt = stt
        self.vad = vad
        self.sample_rate = sample_rate
        self.memory = memory if memory is not None else InMemoryHistory()
        self._running = False

    async def run(self, transport: BaseTransport) -> None:
        """Run with a BaseTransport implementation."""
        if self._running:
            raise RuntimeError("Create a separate YapWrap for each concurrent connection")
        self._running = True
        try:
            session = VoiceSession(
                transport=transport,
                vad=self.vad,
                stt=self.stt,
                agent=self.agent,
                tts=self.tts,
                memory=self.memory,
                sample_rate=self.sample_rate,
            )
            await session.run()
        finally:
            self._running = False

    async def startlocal(self, *, show_transcripts: bool = True) -> None:
        """Run microphone/speakers and print the conversation; requires yapwrap[local]."""
        from yapwrap.transports.local import LocalTransport

        transport = LocalTransport(sample_rate=self.sample_rate, show_transcripts=show_transcripts)
        try:
            await self.run(transport)
        finally:
            transport.stop()

    async def server_ws(
        self,
        ws: Any = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        session_factory: Optional[Callable[[], "YapWrap"]] = None,
    ) -> None:
        """Attach to an accepted socket, or serve standalone.

        For concurrent standalone clients, provide session_factory returning a
        new YapWrap per connection. Without it this wrapper serves one at a time.
        """
        from yapwrap.transports.websocket import WebSocketTransport

        if ws is not None:
            await self.run(WebSocketTransport(ws=ws, sample_rate=self.sample_rate))
            return
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise ImportError("Standalone server_ws requires `pip install 'yapwrap[websockets]'`.") from exc

        async def handler(websocket: Any) -> None:
            wrapper = session_factory() if session_factory is not None else self
            if wrapper._running:
                await websocket.close(code=1013, reason="Voice session is busy")
                return
            await wrapper.server_ws(ws=websocket)

        async with serve(handler, host, port):
            await asyncio.Future()

    async def servewebrtc(self, pc: Any = None) -> None:
        """Run an already-negotiated aiortc peer connection.

        For offer/answer setup, create WebRTCTransport and add its output track
        before negotiation, then pass that transport to run().
        """
        from yapwrap.transports.webrtc import WebRTCTransport

        if pc is None:
            raise ValueError("Pass a negotiated RTCPeerConnection, or use run(WebRTCTransport(...))")
        transport = WebRTCTransport(pc=pc, sample_rate=self.sample_rate)
        output = transport.create_output_track()
        pc.addTrack(output)
        try:
            await self.run(transport)
        finally:
            output.stop()
