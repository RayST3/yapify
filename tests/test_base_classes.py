"""Public inheritance contracts for built-in and third-party components."""

import importlib
import inspect

import pytest

import yapify
from yapify import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport, SessionEvent
from yapify.core import protocols


@pytest.mark.parametrize("base,required", [
    (BaseAgent, {"respond"}),
    (BaseSTT, {"transcribe"}),
    (BaseTTS, {"synthesize"}),
    (BaseVAD, {"process"}),
    (BaseTransport, {"audio_in", "send_audio", "stop_playback"}),
])
def test_incomplete_extensions_cannot_be_instantiated(base, required):
    class Incomplete(base):
        pass

    assert Incomplete.__abstractmethods__ == required
    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


@pytest.mark.parametrize("package,base", [
    ("agents", BaseAgent), ("stt", BaseSTT), ("tts", BaseTTS),
    ("vad", BaseVAD), ("transports", BaseTransport),
])
def test_every_exported_component_inherits_its_interface(package, base):
    module = importlib.import_module(f"yapify.{package}")
    assert getattr(module, base.__name__) is base
    providers = [getattr(module, name) for name in module.__all__ if name != base.__name__]
    assert providers
    for provider in providers:
        assert issubclass(provider, base), provider.__name__
        assert not inspect.isabstract(provider), provider.__name__


def test_legacy_interface_names_are_aliases():
    for name, base in [("Agent", BaseAgent), ("STT", BaseSTT), ("TTS", BaseTTS), ("VAD", BaseVAD), ("Transport", BaseTransport)]:
        assert getattr(yapify, name) is base
        assert getattr(protocols, name) is base


async def test_extensions_inherit_optional_hooks():
    class Detector(BaseVAD):
        def process(self, pcm):
            return None

    class AudioOnlyTransport(BaseTransport):
        async def audio_in(self):
            yield b"\x00\x00"

        async def send_audio(self, pcm):
            pass

        async def stop_playback(self):
            pass

    detector, transport = Detector(), AudioOnlyTransport()
    assert detector.sample_rate == transport.sample_rate == 16000
    detector.reset()
    await transport.send_event(SessionEvent("speech_started", {"turn_id": 1}))


async def test_custom_speech_components_exchange_real_input_and_output():
    from examples.custom_components import BufferedSTT, BufferedTTS, EchoAgent

    async def recognize(pcm):
        assert pcm == b"\x01\x00\x02\x00"
        return "hello"

    async def speak(text):
        assert text == "You said: hello"
        return b"\x03\x00"

    async def audio():
        yield b"\x01\x00"
        yield b"\x02\x00"

    stt, agent, tts = BufferedSTT(recognize), EchoAgent(), BufferedTTS(speak)
    transcript, = [item async for item in stt.transcribe(audio())]
    assert transcript.is_final
    reply = "".join([event.content async for event in agent.respond(transcript.text)])
    assert [chunk async for chunk in tts.synthesize(reply)] == [b"\x03\x00"]
