"""Speech continues across tool pauses without reconnecting or losing tokens."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosedError
from websockets.protocol import State

from yapwrap.tts.cartesia import CartesiaTTS


class StreamingSocket:
    def __init__(self, *, respond=True):
        self.state = State.OPEN
        self.sent = []
        self.messages = asyncio.Queue()
        self.finalized = asyncio.Queue()
        self.respond = respond

    def audio(self, context_id, pcm):
        self.messages.put_nowait({
            "type": "chunk", "context_id": context_id,
            "data": base64.b64encode(pcm).decode(),
        })

    async def send(self, message):
        if self.state is not State.OPEN:
            raise ConnectionClosedError(None, None)
        data = json.loads(message)
        self.sent.append(data)
        if data.get("cancel") or not self.respond:
            return
        if data["transcript"]:
            self.audio(data["context_id"], data["transcript"].encode("utf-16le"))
        if not data["continue"]:
            self.messages.put_nowait({"type": "done", "context_id": data["context_id"]})
            self.finalized.put_nowait(data["context_id"])

    async def recv(self):
        message = await self.messages.get()
        if isinstance(message, Exception):
            raise message
        return json.dumps(message)

    async def close(self):
        self.state = State.CLOSED
        self.messages.put_nowait(ConnectionClosedError(None, None))


@pytest.fixture
def sockets(monkeypatch):
    connections = []

    async def connect(*args, **kwargs):
        socket = StreamingSocket()
        connections.append(socket)
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    return connections


async def collect(tts, source, first_audio=None):
    result = []
    async for pcm in tts.synthesize_stream(source):
        result.append(pcm)
        if first_audio is not None:
            first_audio.set()
    return b"".join(result)


@pytest.mark.asyncio
async def test_speaks_before_tool_finishes_and_reuses_socket_across_contexts_and_replies(sockets):
    release_tool, first_audio, source_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def text():
        try:
            yield "Let me check"  # No punctuation: must not wait for a phrase.
            await release_tool.wait()
            yield "Found it."
        finally:
            source_closed.set()

    tts = CartesiaTTS(api_key="test", context_idle_timeout=0.025)
    async with tts.session():
        task = asyncio.create_task(collect(tts, text(), first_audio))
        try:
            await asyncio.wait_for(first_audio.wait(), 1)
            first_context = await asyncio.wait_for(sockets[0].finalized.get(), 1)
            assert not task.done()
            assert not source_closed.is_set()
            assert not release_tool.is_set()
            release_tool.set()
            assert await asyncio.wait_for(task, 1) == "Let me checkFound it.".encode("utf-16le")
            assert b"".join([chunk async for chunk in tts.synthesize("Next reply.")]) == "Next reply.".encode("utf-16le")
            requests = sockets[0].sent
            assert len(sockets) == 1
            assert [r["transcript"] for r in requests] == ["Let me check", "", "Found it.", "", "Next reply.", ""]
            assert len({r["context_id"] for r in requests}) == 3
            assert requests[0]["context_id"] == requests[1]["context_id"] == first_context
            assert [r["continue"] for r in requests] == [True, False] * 3
            assert all(r["max_buffer_delay_ms"] == 80 for r in requests)
        finally:
            release_tool.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert sockets[0].state is State.CLOSED


@pytest.mark.asyncio
async def test_reconnects_when_socket_dies_during_tool_pause(sockets):
    release = asyncio.Event()

    async def text():
        yield "Before."
        await release.wait()
        yield "After."

    tts = CartesiaTTS(api_key="test", context_idle_timeout=0.025)
    async with tts.session():
        task = asyncio.create_task(collect(tts, text()))
        try:
            await asyncio.wait_for(sockets[0].finalized.get(), 1)
            # Finalization is queued before the receiver consumes its final done.
            # Closing the socket leaves that done available, as on a real socket.
            await sockets[0].close()
            release.set()
            assert await asyncio.wait_for(task, 1) == "Before.After.".encode("utf-16le")
            assert len(sockets) == 2
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert all(socket.state is State.CLOSED for socket in sockets)


@pytest.mark.asyncio
async def test_cancel_context_discards_late_audio_and_preserves_connection(sockets):
    first_audio, source_closed = asyncio.Event(), asyncio.Event()

    async def text():
        try:
            yield "Old reply"
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    tts = CartesiaTTS(api_key="test")
    async with tts.session():
        task = asyncio.create_task(collect(tts, text(), first_audio))
        await asyncio.wait_for(first_audio.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert source_closed.is_set()
        socket = sockets[0]
        old_context = socket.sent[0]["context_id"]
        assert socket.sent[-1] == {"context_id": old_context, "cancel": True}
        socket.audio(old_context, b"\xff\xff")
        socket.messages.put_nowait({"type": "error", "context_id": old_context, "message": "cancelled"})
        pcm = b"".join([chunk async for chunk in tts.synthesize("New reply.")])
        assert pcm == "New reply.".encode("utf-16le")
        assert len(sockets) == 1


@pytest.mark.asyncio
async def test_provider_timeout_cancels_context_and_standalone_call_closes_socket(monkeypatch):
    socket = StreamingSocket(respond=False)

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    tts = CartesiaTTS(api_key="test", response_timeout=0.025)
    with pytest.raises(TimeoutError, match="response_timeout"):
        await asyncio.wait_for(collect(tts, one_token("Hello")), 1)
    assert socket.sent[-1].get("cancel") is True
    assert socket.state is State.CLOSED


async def one_token(text):
    yield text


@pytest.mark.asyncio
async def test_empty_text_does_not_open_connection(sockets):
    tts = CartesiaTTS(api_key="test")
    assert await collect(tts, one_token("   ")) == b""
    assert sockets == []


@pytest.mark.asyncio
async def test_split_pcm_samples_are_reassembled(sockets):
    tts = CartesiaTTS(api_key="test")
    async with tts.session():
        socket = sockets[0]
        original_send = socket.send

        async def send(message):
            data = json.loads(message)
            if data.get("transcript"):
                socket.sent.append(data)
                socket.audio(data["context_id"], b"\x01")
                socket.audio(data["context_id"], b"\x02\x03")
                socket.audio(data["context_id"], b"\x04")
            else:
                await original_send(message)

        socket.send = send
        chunks = [chunk async for chunk in tts.synthesize("Hello")]
        assert chunks == [b"\x01\x02", b"\x03\x04"]


@pytest.mark.asyncio
async def test_connection_failure_closes_the_input_stream(monkeypatch):
    source_closed = asyncio.Event()

    async def text():
        try:
            yield "Hello"
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    async def connect(*args, **kwargs):
        raise OSError("connection failed")

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    with pytest.raises(OSError, match="connection failed"):
        await collect(CartesiaTTS(api_key="test"), text())
    assert source_closed.is_set()
