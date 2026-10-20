"""Exercise the shared engine through each provider's documented wire protocol."""

import asyncio
import base64
import json

import pytest

pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close
from websockets.protocol import State

from yapwrap.tts import (
    CartesiaTTS, DeepgramAuraTTS, ElevenLabsTTS, HumeTTS,
    InworldTTS, RimeTTS, SarvamTTS,
)


PROVIDERS = [CartesiaTTS, DeepgramAuraTTS, ElevenLabsTTS, HumeTTS, InworldTTS, RimeTTS, SarvamTTS]


class ProtocolSocket:
    def __init__(self, provider, *, hold_done=False):
        self.provider = provider
        self.hold_done = hold_done
        self.state = State.OPEN
        self.sent = []
        self.messages = asyncio.Queue()
        self.contexts = {}
        self.current = None

    def emit(self, message):
        self.messages.put_nowait(json.dumps(message) if isinstance(message, dict) else message)

    def finish(self, message):
        if not self.hold_done:
            self.emit(message)

    def audio(self, text, context=None):
        pcm = text.encode("utf-16le")
        encoded = base64.b64encode(pcm).decode()
        if self.provider is CartesiaTTS:
            self.emit({"type": "chunk", "context_id": context, "data": encoded})
        elif self.provider is ElevenLabsTTS:
            self.emit({"context_id": context, "audio": encoded, "is_final": False})
        elif self.provider is DeepgramAuraTTS:
            self.emit(pcm)
        elif self.provider is RimeTTS:
            self.emit({"type": "chunk", "contextId": context, "data": encoded})
        elif self.provider is InworldTTS:
            self.emit({"result": {"contextId": context, "audioChunk": {"audioContent": encoded}}})
        elif self.provider is SarvamTTS:
            self.emit({"type": "audio", "data": {"audio": encoded, "content_type": "audio/pcm"}})
        elif self.provider is HumeTTS:
            # Three samples at 48 kHz become one sample at the session's 16 kHz.
            pcm = b"".join(pcm[i:i + 2] * 3 for i in range(0, len(pcm), 2))
            self.emit({"type": "audio", "audio": base64.b64encode(pcm).decode(),
                       "is_last_chunk": True, "snippet_id": str(len(self.sent))})

    async def send(self, message):
        if self.state is not State.OPEN:
            raise ConnectionClosedError(None, None)
        data = json.loads(message)
        self.sent.append(data)
        context = data.get("context_id", data.get("contextId"))
        if self.provider is CartesiaTTS:
            if data.get("cancel"):
                return
            if data.get("transcript"):
                self.audio(data["transcript"], context)
            if not data.get("continue"):
                self.finish({"type": "done", "context_id": context})
        elif self.provider is ElevenLabsTTS:
            if "voice_settings" in data:
                self.contexts[context] = ""
            elif data.get("close_context"):
                self.contexts.pop(context, None)
            elif data.get("flush"):
                self.audio(self.contexts[context], context)
                self.finish({"context_id": context, "is_final": True})
            else:
                self.contexts[context] += data["text"]
        elif self.provider is DeepgramAuraTTS:
            if data["type"] == "Speak":
                self.contexts[None] = self.contexts.get(None, "") + data["text"]
            elif data["type"] == "Flush":
                self.audio(self.contexts.pop(None, ""))
                self.finish({"type": "Flushed", "sequence_id": 0})
        elif self.provider is RimeTTS:
            if "text" in data:
                self.current = context
                self.contexts[context] = self.contexts.get(context, "") + data["text"]
            elif data["operation"] == "flush":
                self.audio(self.contexts.pop(self.current, ""), self.current)
                self.finish({"type": "done", "contextId": self.current})
        elif self.provider is InworldTTS:
            if "create" in data:
                assert data["create"]["audioConfig"] == {"audioEncoding": "PCM", "sampleRateHertz": 16000}
            elif "sendText" in data:
                self.audio(data["sendText"]["text"], context)
                # Intermediate batch completion must not terminate the context.
                self.emit({"result": {"contextId": context, "flushCompleted": {}}})
            elif "closeContext" in data:
                self.finish({"result": {"contextId": context, "contextClosed": {}}})
        elif self.provider is SarvamTTS:
            if data["type"] == "config":
                assert data["data"]["speech_sample_rate"] == 16000
                assert data["data"]["output_audio_codec"] == "linear16"
            elif data["type"] == "text":
                self.contexts[None] = self.contexts.get(None, "") + data["data"]["text"]
            elif data["type"] == "flush":
                self.audio(self.contexts.pop(None, ""))
                self.finish({"type": "event", "data": {"event_type": "final"}})
        elif self.provider is HumeTTS:
            if data.get("close"):
                if not self.hold_done:
                    self.state = State.CLOSED
                    self.emit(ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True))
            elif data.get("text"):
                self.audio(data["text"])

    async def recv(self):
        item = await self.messages.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        self.state = State.CLOSED
        self.emit(ConnectionClosedError(None, None))


@pytest.fixture(params=PROVIDERS, ids=lambda p: p.__name__)
def provider(request, monkeypatch):
    sockets = []
    settings = []

    async def connect(url, **kwargs):
        socket = ProtocolSocket(request.param)
        sockets.append(socket)
        settings.append((url, kwargs))
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    return request.param(api_key="test", context_idle_timeout=0.02), sockets, settings


async def collect(stream, started=None):
    output = []
    async for pcm in stream:
        assert len(pcm) % 2 == 0
        output.append(pcm)
        if started is not None:
            started.set()
    return b"".join(output)


async def tokens(*parts):
    for part in parts:
        yield part


async def test_tool_pause_flushes_before_tool_finishes_then_resumes(provider):
    tts, sockets, _ = provider
    first_audio, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def text():
        try:
            yield "Checking"
            await release.wait()
            yield "Found"
        finally:
            closed.set()

    async with tts.session():
        task = asyncio.create_task(collect(tts.synthesize_stream(text()), first_audio))
        try:
            await asyncio.wait_for(first_audio.wait(), 1)
            assert not release.is_set()
            assert not closed.is_set()
            await asyncio.sleep(0.04)  # Tool pause outlasts the context flush deadline.
            release.set()
            assert await asyncio.wait_for(task, 1) == "CheckingFound".encode("utf-16le")
            assert await collect(tts.synthesize("Next")) == "Next".encode("utf-16le")
            assert len(sockets) == (3 if isinstance(tts, HumeTTS) else 1)
            if isinstance(tts, ElevenLabsTTS):
                assert not sockets[0].contexts  # All completed contexts are released.
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set()
    assert all(s.state is State.CLOSED for s in sockets)


async def test_interruption_closes_source_and_prevents_old_audio_in_next_reply(provider):
    tts, sockets, _ = provider
    started, closed = asyncio.Event(), asyncio.Event()

    async def text():
        try:
            yield "Old"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async with tts.session():
        sockets[0].hold_done = True
        task = asyncio.create_task(collect(tts.synthesize_stream(text()), started))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert closed.is_set()
        old = sockets[0]
        old.hold_done = False
        context = next((d.get("context_id") for d in old.sent if d.get("context_id")), None)
        old.audio("STALE", context)
        assert await asyncio.wait_for(collect(tts.synthesize("New")), 1) == "New".encode("utf-16le")
        if isinstance(tts, (CartesiaTTS, ElevenLabsTTS)):
            assert len(sockets) == 1
        else:
            assert len(sockets) == 2


async def test_upstream_error_wakes_receiver_and_releases_resources(provider):
    tts, sockets, _ = provider

    async def broken():
        yield "Before"
        raise RuntimeError("agent source failed")

    with pytest.raises(RuntimeError, match="agent source failed"):
        await asyncio.wait_for(collect(tts.synthesize_stream(broken())), 1)
    assert all(s.state is State.CLOSED for s in sockets)


async def test_empty_input_has_no_connection_and_session_cannot_be_shared(provider):
    tts, sockets, _ = provider
    assert await collect(tts.synthesize_stream(tokens("", "   "))) == b""
    assert not sockets
    async with tts.session():
        with pytest.raises(RuntimeError, match="concurrent"):
            async with tts.session():
                pass
    assert all(s.state is State.CLOSED for s in sockets)


async def test_closed_idle_socket_is_reopened_before_next_reply(provider):
    tts, sockets, _ = provider
    async with tts.session():
        assert await collect(tts.synthesize("One")) == "One".encode("utf-16le")
        await sockets[0].close()
        assert await collect(tts.synthesize("Two")) == "Two".encode("utf-16le")
        assert len(sockets) == 2


async def test_missing_completion_times_out_and_next_reply_recovers(provider):
    tts, sockets, _ = provider
    tts.response_timeout = 0.03
    async with tts.session():
        sockets[0].hold_done = True
        with pytest.raises(TimeoutError, match="response_timeout"):
            await asyncio.wait_for(collect(tts.synthesize("Timeout")), 1)
        sockets[0].hold_done = False
        assert await asyncio.wait_for(collect(tts.synthesize("Recovered")), 1) == "Recovered".encode("utf-16le")


async def test_provider_disconnect_during_synthesis_raises_and_recovers(provider):
    tts, sockets, _ = provider
    async with tts.session():
        sockets[0].emit(ConnectionClosedError(None, None))
        with pytest.raises(RuntimeError, match="closed before synthesis completed"):
            await asyncio.wait_for(collect(tts.synthesize("Hello")), 1)
        # Simulate the actual transport transitioning to CLOSED on disconnect.
        sockets[0].state = State.CLOSED
        assert await asyncio.wait_for(collect(tts.synthesize("Recovered")), 1) == "Recovered".encode("utf-16le")


async def test_deepgram_flush_quota_error_does_not_wait_for_missing_ack(monkeypatch):
    sockets = []

    async def connect(*args, **kwargs):
        socket = ProtocolSocket(DeepgramAuraTTS)
        sockets.append(socket)
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    tts = DeepgramAuraTTS(api_key="test")
    async with tts.session():
        sockets[0].emit({"type": "Warning", "description": "Too many flush messages"})
        with pytest.raises(RuntimeError, match="Too many flush"):
            await asyncio.wait_for(collect(tts.synthesize("Hello")), 1)
        assert await collect(tts.synthesize("Recovered")) == "Recovered".encode("utf-16le")


async def test_failed_open_releases_session_and_input(monkeypatch):
    calls, closed = 0, asyncio.Event()
    socket = ProtocolSocket(DeepgramAuraTTS)

    async def connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("connect failed")
        return socket

    async def text():
        try:
            yield "Hello"
        finally:
            closed.set()

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    tts = DeepgramAuraTTS(api_key="test")
    with pytest.raises(OSError, match="connect failed"):
        await collect(tts.synthesize_stream(text()))
    assert closed.is_set()
    async with tts.session():
        assert await collect(tts.synthesize("Recovered")) == "Recovered".encode("utf-16le")


async def test_hume_drains_multiple_snippets_instead_of_stopping_at_first_last_chunk(provider):
    tts, _, _ = provider
    assert await collect(tts.synthesize_stream(tokens("One", "Two", "Three"))) == "OneTwoThree".encode("utf-16le")


async def test_rime_flushes_continuous_text_before_input_eof(monkeypatch):
    socket = ProtocolSocket(RimeTTS)

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    first_audio = asyncio.Event()
    tts = RimeTTS(api_key="test")

    async def text():
        yield "A complete sentence."
        # The next input waits on playback, without a tool/EOF to flush it.
        await asyncio.wait_for(first_audio.wait(), 1)
        yield "Another sentence."

    pcm = await asyncio.wait_for(collect(tts.synthesize_stream(text()), first_audio), 2)
    assert pcm == "A complete sentence.Another sentence.".encode("utf-16le")
