"""SDK adapters must stream actual output and release reused clients on shutdown."""

import asyncio
import io
import sys
import threading
import wave
from contextlib import aclosing
from types import ModuleType, SimpleNamespace as NS

import pytest

from yapwrap.tts import AzureSpeechTTS, GoogleGeminiTTS, OpenAITTS


async def collect(stream):
    return b"".join([pcm async for pcm in stream])


async def tokens(*parts):
    for part in parts:
        yield part


def install_module(monkeypatch, path, module):
    pieces = path.split(".")
    for index in range(1, len(pieces)):
        name = ".".join(pieces[:index])
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, path, module)
    parent = sys.modules[".".join(pieces[:-1])] if len(pieces) > 1 else None
    if parent is not None:
        monkeypatch.setattr(parent, pieces[-1], module, raising=False)


class SpeechResponse:
    def __init__(self, release=None):
        self.closed = False
        self.release = release

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def iter_bytes(self):
        yield b"\x01"  # Split a PCM16 sample at the network boundary.
        yield b"\x00" + b"\x01\x00" * 239
        if self.release is not None:
            await self.release.wait()
        yield b"\x02\x00" * 240


@pytest.fixture
def openai_sdk(monkeypatch):
    clients, responses, requests = [], [], []

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            self.release = None
            self.audio = NS(speech=NS(with_streaming_response=NS(create=self.create)))
            clients.append(self)

        def create(self, **kwargs):
            requests.append(kwargs)
            response = SpeechResponse(self.release)
            responses.append(response)
            return response

        async def close(self):
            self.closed = True

    install_module(monkeypatch, "openai", NS(AsyncOpenAI=Client))
    return clients, responses, requests


async def test_openai_reuses_one_client_for_phrases_and_replies(openai_sdk):
    clients, responses, requests = openai_sdk
    tts = OpenAITTS(api_key="test")
    async with tts.session():
        await collect(tts.synthesize_stream(tokens("First.", "Second.")))
        await collect(tts.synthesize("Third."))
        assert len(clients) == 1 and not clients[0].closed
    assert clients[0].closed
    assert all(r.closed for r in responses)
    assert [r["input"] for r in requests] == ["First.", "Second.", "Third."]
    assert all(r["response_format"] == "pcm" for r in requests)


async def test_openai_yields_before_response_finishes_and_closes_on_interrupt(openai_sdk):
    clients, responses, _ = openai_sdk
    tts = OpenAITTS(api_key="test")
    async with tts.session():
        clients[0].release = asyncio.Event()
        async with aclosing(tts.synthesize("Hello")) as audio:
            first = await asyncio.wait_for(anext(audio), 1)
            assert first and len(first) % 2 == 0
            assert not clients[0].release.is_set()
        assert responses[0].closed
    assert clients[0].closed


async def test_standalone_phrase_stream_reuses_and_closes_client_and_source(openai_sdk):
    clients, responses, _ = openai_sdk
    closed = asyncio.Event()

    async def text():
        try:
            yield "First."
            yield "Second."
        finally:
            closed.set()

    assert await collect(OpenAITTS(api_key="test").synthesize_stream(text()))
    assert len(clients) == 1 and clients[0].closed
    assert len(responses) == 2 and all(r.closed for r in responses)
    assert closed.is_set()


@pytest.fixture
def google_sdk(monkeypatch):
    clients = []

    class Client:
        def __init__(self):
            self.closed = False
            self.requests = []
            self.transport = NS(close=self.close)
            self.stream_closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

        async def synthesize_speech(self, **kwargs):
            self.requests.append(kwargs)
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x01\x00" * 160)
            return NS(audio_content=output.getvalue())

        async def streaming_synthesize(self, requests, **kwargs):
            async def responses():
                try:
                    async for request in requests:
                        self.requests.append(request)
                        if hasattr(request, "input"):
                            yield NS(audio_content=request.input.text.encode("utf-16le"))
                finally:
                    self.stream_closed = True
            return responses()

    sdk = NS(TextToSpeechAsyncClient=Client, VoiceSelectionParams=NS, SynthesisInput=NS,
             AudioConfig=NS, AudioEncoding=NS(LINEAR16="wav", PCM="pcm"),
             StreamingSynthesizeRequest=NS, StreamingSynthesizeConfig=NS,
             StreamingAudioConfig=NS, StreamingSynthesisInput=NS)
    install_module(monkeypatch, "google.cloud.texttospeech", sdk)
    return clients


async def test_google_journey_phrase_fallback_reuses_client_and_decodes_wav(google_sdk):
    tts = GoogleGeminiTTS()
    assert await collect(tts.synthesize_stream(tokens("One.", "Two."))) == b"\x01\x00" * 320
    assert len(google_sdk) == 1 and google_sdk[0].closed
    assert len(google_sdk[0].requests) == 2


async def test_google_chirp_streams_before_input_finishes_and_cleans_rpc(google_sdk):
    tts = GoogleGeminiTTS(voice_name="en-US-Chirp3-HD-Charon")
    closed = asyncio.Event()

    async def text():
        try:
            yield "Before"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async with tts.session():
        async with aclosing(tts.synthesize_stream(text())) as audio:
            assert await asyncio.wait_for(anext(audio), 1) == "Before".encode("utf-16le")
        assert google_sdk[0].stream_closed and closed.is_set()
        assert await collect(tts.synthesize("Next")) == "Next".encode("utf-16le")
    assert len(google_sdk) == 1 and google_sdk[0].closed
    assert google_sdk[0].requests[0].streaming_config.streaming_audio_config.audio_encoding == "pcm"


@pytest.fixture
def azure_sdk(monkeypatch):
    synthesizers, connections = [], []

    class Signal:
        def __init__(self):
            self.handlers = []

        def connect(self, handler):
            self.handlers.append(handler)

        def disconnect(self, handler):
            self.handlers.remove(handler)

        def emit(self, pcm):
            for handler in list(self.handlers):
                handler(NS(result=NS(audio_data=pcm)))

    class Request:
        def __init__(self, **kwargs):
            self.finished = threading.Event()
            self.text = []
            self.synthesizer = None
            self.input_stream = NS(write=self.write, close=self.finished.set)

        def write(self, text):
            self.text.append(text)
            self.synthesizer.synthesizing.emit(b"\x01\x00" * 240)

    class Synthesizer:
        def __init__(self, **kwargs):
            self.synthesizing = Signal()
            self.requests = []
            self.stops = 0
            synthesizers.append(self)

        def speak_async(self, request):
            request.synthesizer = self
            self.requests.append(request)

            def get():
                assert request.finished.wait(2), "SDK worker was not stopped"
                return NS(reason="completed")
            return NS(get=get)

        def stop_speaking_async(self):
            self.stops += 1
            for request in self.requests:
                request.finished.set()
            return NS(get=lambda: None)

    class Connection:
        def __init__(self):
            self.opened, self.closed = False, False
            connections.append(self)

        def open(self, keep_open):
            self.opened = keep_open

        def close(self):
            self.closed = True

    sdk = NS(
        SpeechConfig=lambda **kwargs: NS(set_speech_synthesis_output_format=lambda value: None),
        SpeechSynthesisOutputFormat=NS(Raw24Khz16BitMonoPcm="pcm"), SpeechSynthesizer=Synthesizer,
        Connection=NS(from_speech_synthesizer=lambda synth: Connection()),
        SpeechSynthesisRequest=Request, SpeechSynthesisRequestInputType=NS(TextStream="text-stream"),
        ResultReason=NS(SynthesizingAudioCompleted="completed"),
    )
    install_module(monkeypatch, "azure.cognitiveservices.speech", sdk)
    return synthesizers, connections


async def test_azure_streams_audio_callbacks_before_text_eof_and_stops_on_interrupt(azure_sdk):
    synthesizers, connections = azure_sdk
    closed = asyncio.Event()

    async def text():
        try:
            yield "Before"
            await asyncio.Event().wait()
        finally:
            closed.set()

    tts = AzureSpeechTTS(subscription_key="test")
    async with tts.session():
        async with aclosing(tts.synthesize_stream(text())) as audio:
            pcm = await asyncio.wait_for(anext(audio), 1)
            assert pcm and not closed.is_set()
        assert synthesizers[0].stops == 1
        assert closed.is_set()
        assert not synthesizers[0].synthesizing.handlers
        assert await collect(tts.synthesize("Next")) == b"\x01\x00" * 160
    assert len(synthesizers) == 1
    assert connections[0].opened and connections[0].closed
