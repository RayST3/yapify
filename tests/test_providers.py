import asyncio
import base64
import json

import pytest

from yapify.stt.deepgram import DeepgramSTT
from yapify.tts.cartesia import CartesiaTTS


class Socket:
    def __init__(self, messages, fail_send=False):
        self.messages = messages
        self.sent = []
        self.ready = asyncio.Event()
        self.fail_send = fail_send

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, message):
        if self.fail_send:
            raise RuntimeError("upload failed")
        self.sent.append(message)
        if isinstance(message, str):
            self.ready.set()

    async def close(self):
        self.ready.set()

    async def __aiter__(self):
        await self.ready.wait()
        for message in self.messages:
            if self.sent and isinstance(self.sent[-1], str):
                context_id = json.loads(self.sent[-1]).get("context_id")
                if context_id:
                    message = {**message, "context_id": context_id}
            yield json.dumps(message)


def connect_fake(monkeypatch, socket):
    client = pytest.importorskip("websockets.asyncio.client")
    monkeypatch.setattr(client, "connect", lambda *args, **kwargs: socket)


@pytest.mark.asyncio
async def test_deepgram_flushes_all_segments_after_input_eof(monkeypatch):
    def result(text, final):
        return {"type": "Results", "is_final": final, "channel": {"alternatives": [{"transcript": text}]}}

    socket = Socket([result("hello", True), {"type": "UtteranceEnd"}, result("wor", False), result("world", True)])
    connect_fake(monkeypatch, socket)

    async def audio():
        yield b"\x01\x00" * 320

    events = [event async for event in DeepgramSTT(api_key="test").transcribe(audio())]
    assert [(event.text, event.is_final) for event in events] == [("hello", True), ("wor", False), ("world", True)]
    assert json.loads(socket.sent[-1]) == {"type": "CloseStream"}


@pytest.mark.asyncio
async def test_deepgram_sender_failure_wakes_receiver(monkeypatch):
    socket = Socket([], fail_send=True)
    connect_fake(monkeypatch, socket)

    async def audio():
        yield b"\x00\x00"

    async def consume():
        return [event async for event in DeepgramSTT(api_key="test").transcribe(audio())]

    with pytest.raises(RuntimeError, match="upload failed"):
        await asyncio.wait_for(consume(), 1)


@pytest.mark.asyncio
async def test_cartesia_streams_pcm_and_finishes(monkeypatch):
    pcm = b"\x01\x00" * 320
    socket = Socket([{"data": base64.b64encode(pcm).decode()}, {"done": True}])
    connect_fake(monkeypatch, socket)
    assert [chunk async for chunk in CartesiaTTS(api_key="test").synthesize("hello")] == [pcm]
    request = json.loads(socket.sent[0])
    assert request["output_format"] == {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}


@pytest.mark.asyncio
async def test_cartesia_provider_error_is_actionable(monkeypatch):
    connect_fake(monkeypatch, Socket([{"type": "error", "message": "Invalid model", "done": True}]))
    with pytest.raises(RuntimeError, match="Invalid model"):
        _ = [chunk async for chunk in CartesiaTTS(api_key="test").synthesize("hello")]


@pytest.mark.asyncio
async def test_cartesia_premature_close_is_an_error(monkeypatch):
    connect_fake(monkeypatch, Socket([]))
    with pytest.raises(RuntimeError, match="before synthesis completed"):
        _ = [chunk async for chunk in CartesiaTTS(api_key="test").synthesize("hello")]
