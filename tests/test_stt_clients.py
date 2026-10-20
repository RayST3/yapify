"""Google streaming RPC and shared STT ownership checks."""

import asyncio
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace as NS

import pytest

from yapwrap import YapSesh
from yapwrap.stt import GoogleChirpSTT, OpenAIRealtimeSTT
from tests.test_tts_clients import install_module
from tests.test_session_lifecycle import LiveTransport, start
from tests.conftest import MockSTT, MockTTS, MockAgent, MockVAD


async def collect(stream):
    return [event async for event in stream]


@pytest.fixture
def google_sdk(monkeypatch):
    clients = []

    class RPC:
        def __init__(self, requests, client):
            self.queue = asyncio.Queue()
            self.client = client
            self.cancelled = False
            self.task = asyncio.create_task(self.consume(requests))

        async def consume(self, requests):
            try:
                async for request in requests:
                    self.client.requests.append(request)
                    if hasattr(request, "audio"):
                        self.queue.put_nowait(NS(results=[NS(
                            alternatives=[NS(transcript="hello")], is_final=False,
                        )]))
                if not self.client.hold_final:
                    self.queue.put_nowait(NS(results=[
                        NS(alternatives=[NS(transcript="hello")], is_final=True),
                        NS(alternatives=[NS(transcript="world")], is_final=True),
                    ]))
                    self.queue.put_nowait(None)
            except Exception:
                # Some SDK request consumers swallow producer exceptions.
                # The adapter must still surface the original source error.
                pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            value = await self.queue.get()
            if value is None:
                raise StopAsyncIteration
            return value

        def cancel(self):
            self.cancelled = True
            self.task.cancel()

        async def code(self):
            await asyncio.gather(self.task, return_exceptions=True)

    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            self.requests, self.rpcs = [], []
            self.hold_final = False
            self.defer_open = False
            self.transport = NS(close=self.close)
            clients.append(self)

        async def close(self):
            self.closed = True

        async def streaming_recognize(self, requests, **kwargs):
            rpc = RPC(requests, self)
            self.rpcs.append(rpc)
            if self.defer_open:
                try:
                    await asyncio.Event().wait()
                except BaseException:
                    rpc.cancel()
                    await rpc.code()
                    raise
            return rpc

    class Decoding(NS):
        AudioEncoding = NS(LINEAR16="linear16")

    sdk = NS(
        SpeechAsyncClient=Client, StreamingRecognizeRequest=NS,
        StreamingRecognitionConfig=NS, RecognitionConfig=NS,
        ExplicitDecodingConfig=Decoding, StreamingRecognitionFeatures=NS,
    )
    install_module(monkeypatch, "google.cloud.speech_v2", sdk)
    return clients


async def audio():
    yield b"\x01\x00" * 160


async def test_google_streams_before_eof_reuses_client_and_drains_finals(google_sdk):
    stt = GoogleChirpSTT(project_id="test-project")
    release, closed = asyncio.Event(), asyncio.Event()

    async def source():
        try:
            async for chunk in audio():
                yield chunk
            await release.wait()
        finally:
            closed.set()

    async with stt.session():
        async with aclosing(stt.transcribe(source())) as stream:
            first = await asyncio.wait_for(anext(stream), 1)
            assert first.text == "hello" and not first.is_final
            assert not closed.is_set()
            release.set()
            events = await asyncio.wait_for(collect(stream), 1)
            assert [(e.text, e.is_final) for e in events] == [("hello", True), ("world", True)]
        assert len(await collect(stt.transcribe(audio()))) == 3
        assert len(google_sdk) == 1
        assert all(r.cancelled for r in google_sdk[0].rpcs)
    client = google_sdk[0]
    assert client.closed and closed.is_set()
    assert client.options["client_options"]["api_endpoint"] == "us-speech.googleapis.com"
    assert client.requests[0].recognizer == "projects/test-project/locations/us/recognizers/_"


@pytest.mark.parametrize("failure", ["source", "timeout", "opening_timeout", "cancel"])
async def test_google_cleanup_on_source_error_missing_final_and_interrupt(google_sdk, failure):
    stt = GoogleChirpSTT(project_id="test", response_timeout=0.03)
    closed = asyncio.Event()

    async def source():
        try:
            async for pcm in audio():
                yield pcm
            if failure == "source":
                raise RuntimeError("microphone failed")
            if failure == "cancel":
                await asyncio.Event().wait()
        finally:
            closed.set()

    async with stt.session():
        client = google_sdk[0]
        client.hold_final = failure == "timeout"
        client.defer_open = failure == "opening_timeout"
        if failure == "cancel":
            async with aclosing(stt.transcribe(source())) as stream:
                await asyncio.wait_for(anext(stream), 1)
        else:
            with pytest.raises(RuntimeError if failure == "source" else TimeoutError):
                await asyncio.wait_for(collect(stt.transcribe(source())), 1)
        assert closed.is_set() and client.rpcs[0].cancelled
        assert client.closed


async def test_stt_resources_close_after_transcription_cleanup():
    lifecycle = []
    entered = asyncio.Event()

    class STT(MockSTT):
        @asynccontextmanager
        async def session(self):
            lifecycle.append("stt open")
            try:
                yield self
            finally:
                lifecycle.append("stt close")

        async def transcribe(self, audio):
            try:
                async for _ in audio:
                    entered.set()
                await asyncio.Event().wait()
                yield
            finally:
                lifecycle.append("transcription cleanup")

    session, transport, task = await start(["start"], stt=STT())
    transport.feed(1)
    await asyncio.wait_for(entered.wait(), 1)
    session.stop()
    await asyncio.wait_for(task, 1)
    assert lifecycle == ["stt open", "transcription cleanup", "stt close"]


async def test_tts_startup_failure_closes_stt_and_session_can_retry():
    lifecycle = []

    class STT(MockSTT):
        @asynccontextmanager
        async def session(self):
            lifecycle.append("open")
            try:
                yield self
            finally:
                lifecycle.append("close")

    class TTS(MockTTS):
        @asynccontextmanager
        async def session(self):
            raise RuntimeError("TTS startup failed")
            yield self

    session = YapSesh(LiveTransport(), MockVAD(), STT(), MockAgent(), TTS())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="startup failed"):
            await session.run()
        assert not session._is_running
    assert lifecycle == ["open", "close", "open", "close"]


async def test_waiting_for_component_lock_still_closes_input_once():
    class Input:
        def __init__(self):
            self.closed = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            await asyncio.Event().wait()
        async def aclose(self):
            self.closed += 1

    stt, source = OpenAIRealtimeSTT(api_key="test"), Input()
    async with stt._lock:
        task = asyncio.create_task(collect(stt.transcribe(source)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert source.closed == 1
