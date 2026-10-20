import importlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

example = importlib.import_module("examples.transports.fastapi_websocket")


def test_webrtc_browser_and_invalid_offer_cleanup(monkeypatch):
    pytest.importorskip("aiortc")
    rtc = importlib.import_module("examples.transports.webrtc")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with TestClient(rtc.app) as client:
        assert 'data-transport="webrtc"' in client.get("/").text
        with client.websocket_connect("/ws/offer") as ws:
            assert "DEEPGRAM_API_KEY" in ws.receive_json()["message"]
            assert ws.receive()["code"] == 1011
        for key in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.setenv(key, "test-no-network")
        with client.websocket_connect("/ws/offer") as ws:
            ws.send_json({"type": "invalid"})
            assert ws.receive_json()["type"] == "error"
            assert ws.receive()["type"] == "websocket.close"


def test_browser_page_and_missing_credentials(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(example.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "AudioWorkletNode" in response.text
        with client.websocket_connect("/ws/voice") as ws:
            message = ws.receive_json()
            assert message["type"] == "error"
            assert "DEEPGRAM_API_KEY" in message["message"]
            assert "GOOGLE_API_KEY" in message["message"]


def test_browser_silence_and_reconnect_with_real_silero(monkeypatch):
    pytest.importorskip("silero_vad")
    pytest.importorskip("agno")
    pytest.importorskip("google.genai")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-no-network")
    monkeypatch.setenv("CARTESIA_API_KEY", "test-no-network")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-no-network")
    wrappers = []
    wrapper_class = example.YapWrap
    sockets = []
    connection_urls = []

    class IdleSpeechSocket:
        closed = False

        async def close(self):
            self.closed = True

    async def connect(url, **kwargs):
        # STT and TTS connect at call startup, including calls containing only
        # silence. Exercise both lifecycles without contacting live providers.
        connection_urls.append(url)
        socket = IdleSpeechSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)

    def record_wrapper(**kwargs):
        wrapper = wrapper_class(**kwargs)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(example, "YapWrap", record_wrapper)
    with TestClient(example.app) as client:
        for _ in range(2):
            with client.websocket_connect("/ws/voice") as ws:
                for _ in range(50):
                    ws.send_bytes(b"\x00" * 640)
                ws.send_json({"type": "stop"})
                assert ws.receive() == {"type": "websocket.close", "code": 1000, "reason": ""}
    assert wrappers[0].vad is not wrappers[1].vad
    assert wrappers[0].memory is not wrappers[1].memory
    assert wrappers[0].agent.agent is not wrappers[1].agent.agent
    assert len(sockets) == 4
    assert sum(url.startswith("wss://api.deepgram.com/") for url in connection_urls) == 2
    assert sum(url.startswith("wss://api.cartesia.ai/") for url in connection_urls) == 2
    assert all(socket.closed for socket in sockets)


def test_browser_receives_conversation_text_and_audio(monkeypatch):
    from tests.conftest import MockAgent, MockSTT, MockTTS, MockVAD

    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-no-network")
    monkeypatch.setenv("CARTESIA_API_KEY", "test-no-network")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-no-network")
    monkeypatch.setitem(sys.modules, "agno.agent", SimpleNamespace(Agent=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "agno.models.google", SimpleNamespace(Gemini=SimpleNamespace))
    monkeypatch.setattr(example, "AgnoAgent", lambda agent: MockAgent("Hello there."))
    monkeypatch.setattr(example, "DeepgramSTT", lambda **kwargs: MockSTT("hello"))
    monkeypatch.setattr(example, "CartesiaTTS", lambda **kwargs: MockTTS())
    monkeypatch.setattr(example, "SileroVAD", lambda: MockVAD(["start", "stop"]))
    import json
    with TestClient(example.app) as client:
        with client.websocket_connect("/ws/voice") as ws:
            ws.send_bytes(b"\x01\x00" * 640)
            events, audio = [], []
            while True:
                message = ws.receive()
                if message.get("bytes"):
                    audio.append(message["bytes"])
                elif message.get("text"):
                    event = json.loads(message["text"])
                    events.append(event)
                    if event["type"] == "agent_complete":
                        break
            final_transcript = next(event for event in events if event["type"] == "transcript" and event["data"]["is_final"])
            assert final_transcript["data"] == {"turn_id": 1, "text": "hello", "is_final": True}
            assert "".join(event["data"]["text"] for event in events if event["type"] == "agent_delta") == "Hello there. "
            assert audio
            assert events[-1]["data"]["status"] == "completed"
            ws.send_json({"type": "stop"})
            assert ws.receive()["code"] == 1000
