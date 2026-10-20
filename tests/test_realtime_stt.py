"""Protocol and lifecycle tests for every WebSocket STT adapter, without API calls."""

import asyncio
import base64
import json
from contextlib import aclosing

import pytest

pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close
from websockets.protocol import State

from yapwrap.stt import (
    AssemblyAISTT, DeepgramSTT, ElevenLabsSTT, GladiaSTT,
    OpenAIRealtimeSTT, SonioxSTT, SpeechmaticsSTT,
)


PROVIDERS = [AssemblyAISTT, DeepgramSTT, ElevenLabsSTT, GladiaSTT,
             OpenAIRealtimeSTT, SonioxSTT, SpeechmaticsSTT]


class Socket:
    def __init__(self, provider):
        self.provider = provider
        self.state = State.OPEN
        self.messages = asyncio.Queue()
        self.sent, self.audio = [], []
        self.hold_final = False
        self.fail_upload = False
        self.first = True
        self.turn = 0
        if provider is AssemblyAISTT:
            self.emit({"type": "Begin", "id": "test"})
        elif provider is ElevenLabsSTT:
            self.emit({"message_type": "session_started"})

    def emit(self, data):
        self.messages.put_nowait(json.dumps(data) if isinstance(data, dict) else data)

    def transcript(self, text, final=False):
        if self.provider is DeepgramSTT:
            self.emit({"type": "Results", "channel": {"alternatives": [{"transcript": text}]},
                       "is_final": final})
        elif self.provider is AssemblyAISTT:
            self.emit({"type": "Turn", "turn_order": self.turn, "transcript": text, "end_of_turn": final})
            if final:
                self.turn += 1
        elif self.provider is SpeechmaticsSTT:
            self.emit({"message": "AddTranscript" if final else "AddPartialTranscript",
                       "metadata": {"transcript": text}})
        elif self.provider is GladiaSTT:
            self.emit({"type": "transcript", "data": {"is_final": final, "utterance": {"text": text}}})
        elif self.provider is ElevenLabsSTT:
            self.emit({"message_type": "committed_transcript" if final else "partial_transcript", "text": text})
        elif self.provider is SonioxSTT:
            self.emit({"tokens": [{"text": text, "is_final": final}]})
        elif self.provider is OpenAIRealtimeSTT:
            self.emit({"type": "conversation.item.input_audio_transcription." + ("completed" if final else "delta"),
                       "transcript" if final else "delta": text, "item_id": f"item-{self.turn}"})

    async def send(self, message):
        if self.state is not State.OPEN:
            raise ConnectionClosedError(None, None)
        self.sent.append(message)
        data = {} if isinstance(message, bytes) else json.loads(message)
        kind = data.get("type", data.get("message_type", data.get("message")))
        if kind == "StartRecognition":
            self.emit({"message": "RecognitionStarted"})
        elif kind == "session.update":
            assert data["session"]["type"] == "transcription"
            assert data["session"]["audio"]["input"]["format"]["rate"] == 24000
            assert data["session"]["audio"]["input"]["turn_detection"] is None
            self.emit({"type": "session.updated"})
        pcm = message if isinstance(message, bytes) else None
        if kind == "input_audio_chunk" and not data.get("commit"):
            pcm = base64.b64decode(data["audio_base_64"])
        elif kind == "input_audio_buffer.append":
            pcm = base64.b64decode(data["audio"])
        if pcm:
            if self.fail_upload:
                raise RuntimeError("upload failed")
            self.audio.append(pcm)
            if self.first:
                self.first = False
                self.transcript("hel")
                if self.provider in (DeepgramSTT, AssemblyAISTT, SpeechmaticsSTT, GladiaSTT):
                    self.transcript("hello", final=True)
        if kind in ("CloseStream", "Terminate", "ForceEndOfUtterance", "finalize",
                    "stop_recording", "input_audio_buffer.commit") or data.get("commit"):
            if not self.hold_final:
                self.finish()

    def finish(self):
        if self.provider in (DeepgramSTT, AssemblyAISTT, SpeechmaticsSTT, GladiaSTT):
            self.transcript("world", final=True)
        elif self.provider is OpenAIRealtimeSTT:
            self.emit({"type": "input_audio_buffer.committed", "item_id": f"item-{self.turn}"})
            self.transcript("hello world", final=True)
            self.turn += 1
        else:
            self.transcript("hello world", final=True)
        if self.provider is DeepgramSTT:
            self.emit({"type": "Metadata"})
            self.state = State.CLOSED
            self.emit(ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True))
        elif self.provider is AssemblyAISTT:
            self.emit({"type": "Termination"})
        elif self.provider is SpeechmaticsSTT:
            self.emit({"message": "EndOfUtterance"})
        elif self.provider is SonioxSTT:
            self.emit({"tokens": [{"text": "<fin>", "is_final": True}]})
        elif self.provider is GladiaSTT:
            self.emit({"type": "stop_recording", "acknowledged": True})
            self.emit({"type": "post_final_transcript", "data": {"transcription": {"full_transcript": "hello world"}}})
        self.first = True

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
    sockets, settings, http_calls = [], [], []

    async def connect(url, **kwargs):
        socket = Socket(request.param)
        sockets.append(socket)
        settings.append((url, kwargs))
        return socket

    class HTTP:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            http_calls.append((url, kwargs))
            class Response:
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"url": "wss://signed.gladia.test/stream"}
            return Response()

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    monkeypatch.setattr("httpx.AsyncClient", HTTP)
    return request.param(api_key="test"), sockets, settings, http_calls


async def audio():
    yield b"\x01"  # Preserve samples split across incoming network chunks.
    yield b"\x00" + b"\x01\x00" * 1599


def transcript(events):
    committed, interim = [], ""
    for event in events:
        if event.is_final:
            if event.text.strip():
                committed.append(event.text.strip())
            interim = ""
        else:
            interim = event.text
    return " ".join([*committed, interim.strip()]).strip()


async def collect(stream):
    return [event async for event in stream]


async def test_every_adapter_streams_before_eof_and_drains_all_final_segments(provider):
    stt, sockets, _, _ = provider
    release, closed = asyncio.Event(), asyncio.Event()

    async def source():
        try:
            async for pcm in audio():
                yield pcm
            await release.wait()
            yield b"\x02\x00" * 1600
        finally:
            closed.set()

    async with stt.session():
        async with aclosing(stt.transcribe(source())) as stream:
            first = await asyncio.wait_for(anext(stream), 1)
            assert first.text == "hel"
            assert not closed.is_set()
            # A final provider segment is not permission to stop reading the mic.
            release.set()
            events = [first, *await asyncio.wait_for(collect(stream), 1)]
        assert transcript(events) == "hello world"
        assert closed.is_set()
        assert sum(len(pcm) for pcm in sockets[0].audio) == (9600 if isinstance(stt, OpenAIRealtimeSTT) else 6400)
        assert all(len(pcm) % 2 == 0 for pcm in sockets[0].audio)
    assert all(s.state is State.CLOSED for s in sockets)


async def test_reuses_only_protocols_with_an_utterance_completion_boundary(provider):
    stt, sockets, _, _ = provider
    async with stt.session():
        for _ in range(2):
            assert transcript(await asyncio.wait_for(collect(stt.transcribe(audio())), 1)) == "hello world"
        assert len(sockets) == (1 if stt.reuse_connection else 2)
    assert all(s.state is State.CLOSED for s in sockets)
    assert stt._keepalive_task is None


async def test_interruption_discards_connection_and_closes_source(provider):
    stt, sockets, _, _ = provider
    closed = asyncio.Event()

    async def source():
        try:
            async for pcm in audio():
                yield pcm
            await asyncio.Event().wait()
        finally:
            closed.set()

    async with stt.session():
        async with aclosing(stt.transcribe(source())) as stream:
            assert (await asyncio.wait_for(anext(stream), 1)).text
        assert closed.is_set() and sockets[0].state is State.CLOSED
        sockets[0].transcript("STALE", final=True)
        assert transcript(await asyncio.wait_for(collect(stt.transcribe(audio())), 1)) == "hello world"
        assert len(sockets) == 2


@pytest.mark.parametrize("failure", ["source", "upload", "disconnect", "provider"])
async def test_errors_propagate_and_release_resources(provider, failure):
    stt, sockets, _, _ = provider

    async def source():
        async for pcm in audio():
            yield pcm
        if failure == "source":
            raise RuntimeError("source failed")

    async with stt.session():
        if failure == "upload":
            sockets[0].fail_upload = True
        elif failure == "disconnect":
            sockets[0].emit(ConnectionClosedError(None, None))
        elif failure == "provider":
            p = type(stt)
            data = (
                {"message": "Error", "reason": "provider failed"} if p is SpeechmaticsSTT else
                {"error_code": 401, "error_message": "provider failed"} if p is SonioxSTT else
                {"type": "error", "error": "provider failed"}
            )
            sockets[0].emit(data)
        with pytest.raises(RuntimeError, match="failed|closed before"):
            await asyncio.wait_for(collect(stt.transcribe(source())), 1)
        assert sockets[0].state is State.CLOSED


async def test_finalization_timeout_does_not_limit_live_audio_duration(provider):
    stt, sockets, _, _ = provider
    stt.response_timeout = 0.025
    async with stt.session():
        async def slow():
            async for pcm in audio():
                yield pcm
            await asyncio.sleep(0.06)
        assert transcript(await asyncio.wait_for(collect(stt.transcribe(slow())), 1)) == "hello world"
        # A missing final response after EOF must not hang forever.
        await stt._ensure_open()
        sockets[-1].hold_final = True
        with pytest.raises(TimeoutError, match="finalize"):
            await asyncio.wait_for(collect(stt.transcribe(audio())), 1)
        assert sockets[-1].state is State.CLOSED


async def test_empty_audio_skips_connection_and_bad_pcm_is_rejected(provider):
    stt, sockets, _, _ = provider
    async def empty():
        yield b""
    assert await collect(stt.transcribe(empty())) == []
    assert not sockets
    async def odd():
        yield b"\x01"
    with pytest.raises(ValueError, match="incomplete PCM16"):
        await collect(stt.transcribe(odd()))
    assert not sockets


async def test_idle_closed_socket_reopens_and_sessions_cannot_overlap(provider):
    stt, sockets, _, _ = provider
    async with stt.session():
        with pytest.raises(RuntimeError, match="concurrent"):
            async with stt.session():
                pass
        await sockets[0].close()
        assert transcript(await collect(stt.transcribe(audio()))) == "hello world"
        assert len(sockets) == 2


def test_soniox_replaces_provisional_tokens_without_spaces_or_duplicates():
    stt = SonioxSTT(api_key="test")
    stt._reset_utterance()
    messages = [
        {"tokens": [{"text": "hel", "is_final": False}]},
        {"tokens": [{"text": "hel", "is_final": True}, {"text": "p", "is_final": False}]},
        {"tokens": [{"text": "lo", "is_final": True}, {"text": " world", "is_final": False}]},
        {"tokens": [{"text": " world", "is_final": True}, {"text": "<fin>", "is_final": True}]},
    ]
    events = []
    for message in messages:
        batch, _ = stt._decode_message(json.dumps(message))
        events.extend(batch)
    assert [e.text for e in events] == ["hel", "help", "hello world", "hello world"]
    assert transcript(events) == "hello world"


def test_assemblyai_ignores_repeated_final_turns():
    stt = AssemblyAISTT(api_key="test")
    stt._reset_utterance()
    message = json.dumps({"type": "Turn", "turn_order": 0, "transcript": "hello", "end_of_turn": True})
    assert stt._decode_message(message)[0][0].text == "hello"
    assert stt._decode_message(message) == ([], False)
