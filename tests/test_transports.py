"""Test transports: WebSocket, WebRTC, and Local."""

import asyncio
import json
import pytest

from yapwrap import SessionEvent, SessionEventSink

from yapwrap.transports.websocket import WebSocketTransport
from yapwrap.transports.webrtc import WebRTCTransport
from yapwrap.transports.local import LocalTransport


class DummyWebSocket:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent_bytes = []
        self.sent_text = []

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    async def send_text(self, text: str):
        self.sent_text.append(text)


@pytest.mark.asyncio
async def test_websocket_transport():
    dummy_ws = DummyWebSocket(
        incoming=[
            {"type": "websocket.receive", "bytes": b"\x00\x01" * 160},
            {"type": "websocket.receive", "text": '{"type": "ping"}'},
            {"type": "websocket.receive", "bytes": b"\x00\x02" * 160},
        ]
    )

    transport = WebSocketTransport(ws=dummy_ws)

    # Send audio while open
    await transport.send_audio(b"\x00\x03" * 160)
    assert len(dummy_ws.sent_bytes) == 1

    # Stop playback sends control event while open
    await transport.stop_playback()
    assert len(dummy_ws.sent_text) == 1
    assert "stop_playback" in dummy_ws.sent_text[0]

    # Receive audio frames until disconnect
    frames = []
    async for frame in transport.audio_in():
        frames.append(frame)

    assert len(frames) == 2
    assert transport._is_open is False


@pytest.mark.asyncio
async def test_webrtc_transport_queue():
    pytest.importorskip("aiortc")
    from fractions import Fraction
    transport = WebRTCTransport()
    output = transport.create_output_track()
    await transport.send_audio(b"\x00\x01" * 160)
    await transport.send_audio(b"\x00\x02" * 160)
    frame = await output.recv()
    assert frame.to_ndarray().tobytes() == b"\x00\x01" * 160 + b"\x00\x02" * 160
    assert frame.sample_rate == 16000
    assert frame.time_base == Fraction(1, 16000)
    assert frame.pts == 0
    await transport.send_audio(b"\x00\x03" * 800)
    await transport.stop_playback()
    silence = await output.recv()
    assert silence.to_ndarray().tobytes() == b"\x00" * 640
    assert silence.pts == 320
    await transport.close()


@pytest.mark.asyncio
async def test_local_transport_queue():
    transport = LocalTransport()

    # Send audio and stop playback
    await transport.send_audio(b"\x00\x01" * 160)
    await transport.stop_playback()

    transport.stop()
    assert transport._is_open is False


@pytest.mark.asyncio
async def test_local_pcm_chunks_play_in_order_and_can_be_cleared():
    transport = LocalTransport()
    await transport.send_audio(b"\x01\x00" * 3)
    await transport.send_audio(b"\x02\x00" * 3)
    assert transport._read_playback(8) == b"\x01\x00" * 3 + b"\x02\x00"
    assert transport._read_playback(6) == b"\x02\x00" * 2 + b"\x00\x00"
    await transport.send_audio(b"\x03\x00" * 3)
    await transport.stop_playback()
    assert transport._read_playback(6) == b"\x00" * 6


@pytest.mark.asyncio
async def test_websocket_cancel_is_not_swallowed():
    class Socket:
        async def receive(self):
            raise asyncio.CancelledError

    transport = WebSocketTransport(Socket())
    with pytest.raises(asyncio.CancelledError):
        await anext(transport.audio_in())


@pytest.mark.asyncio
async def test_websocket_send_failure_is_not_swallowed():
    class Socket:
        async def send_bytes(self, data):
            raise RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await WebSocketTransport(Socket()).send_audio(b"\x00\x00")


@pytest.mark.asyncio
@pytest.mark.parametrize("standard_socket", [False, True])
async def test_websocket_forwards_conversation_events(standard_socket):
    class StandardSocket:
        def __init__(self):
            self.sent_text = []

        async def send(self, message):
            self.sent_text.append(message)

    socket = StandardSocket() if standard_socket else DummyWebSocket()
    transport = WebSocketTransport(socket)
    assert isinstance(transport, SessionEventSink)
    await transport.send_event(SessionEvent("transcript", {"turn_id": 3, "text": "Hello नमस्ते", "is_final": True}, timestamp=123.0))
    assert json.loads(socket.sent_text[0]) == {
        "type": "transcript",
        "data": {"turn_id": 3, "text": "Hello नमस्ते", "is_final": True},
        "timestamp": 123.0,
    }
    transport._is_open = False
    await transport.send_event(SessionEvent("agent_delta", {"turn_id": 3, "text": "hi"}))
    assert len(socket.sent_text) == 1


@pytest.mark.asyncio
async def test_terminal_conversation_streams_without_duplicate_text(capsys):
    transport = LocalTransport()
    assert isinstance(transport, SessionEventSink)
    await transport.send_event(SessionEvent("transcript", {"turn_id": 1, "text": "hel", "is_final": False}))
    assert capsys.readouterr().out == ""
    await transport.send_event(SessionEvent("transcript", {"turn_id": 1, "text": "hello", "is_final": True}))
    await transport.send_event(SessionEvent("agent_delta", {"turn_id": 1, "text": "Hi "}))
    assert capsys.readouterr().out == "You: hello\nAgent: Hi "
    await transport.send_event(SessionEvent("agent_delta", {"turn_id": 1, "text": "there!"}))
    await transport.send_event(SessionEvent("agent_complete", {"turn_id": 1, "text": "Hi there!", "status": "completed"}))
    assert capsys.readouterr().out == "there!\n"
    await transport.send_event(SessionEvent("transcript", {"turn_id": 2, "text": "tell me more", "is_final": True}))
    await transport.send_event(SessionEvent("agent_delta", {"turn_id": 2, "text": "Sure…"}))
    await transport.send_event(SessionEvent("agent_complete", {"turn_id": 2, "text": "Sure…", "status": "interrupted"}))
    assert capsys.readouterr().out == "You: tell me more\nAgent: Sure… [interrupted]\n"


@pytest.mark.asyncio
async def test_terminal_conversation_can_be_disabled(capsys):
    transport = LocalTransport(show_transcripts=False)
    await transport.send_event(SessionEvent("transcript", {"turn_id": 1, "text": "hello", "is_final": True}))
    await transport.send_event(SessionEvent("agent_delta", {"turn_id": 1, "text": "hi"}))
    await transport.send_event(SessionEvent("agent_complete", {"turn_id": 1, "text": "hi", "status": "completed"}))
    assert capsys.readouterr().out == ""
