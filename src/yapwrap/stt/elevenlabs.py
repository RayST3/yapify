"""ElevenLabs Scribe Realtime with explicit audio commits."""

import base64
import json
import os
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class ElevenLabsSTT(WebSocketSTT):
    """Reuse a manual-commit Scribe connection across utterances."""

    reuse_connection = True

    def __init__(
        self, api_key: Optional[str] = None, model: str = "scribe_v2_realtime",
        language: str = "en", sample_rate: int = 16000, *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.model, self.language = model, language

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("ElevenLabsSTT requires api_key or ELEVENLABS_API_KEY.")
        params = {
            "model_id": self.model, "audio_format": f"pcm_{self.sample_rate}",
            "commit_strategy": "manual", "include_timestamps": "false",
        }
        if self.language:
            params["language_code"] = self.language
        return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urlencode(params)}", {"xi-api-key": self.api_key}

    async def _configure_connection(self, connection) -> None:
        while True:
            data = json.loads(await connection.recv())
            if data.get("message_type") == "session_started":
                return
            if data.get("error"):
                raise RuntimeError(f"ElevenLabs STT: {data['error']}")

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        await self._connection.send(json.dumps({
            "message_type": "input_audio_chunk", "audio_base_64": base64.b64encode(chunk).decode(),
            "sample_rate": self.sample_rate,
        }))

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({
            "message_type": "input_audio_chunk", "audio_base_64": "", "commit": True,
            "sample_rate": self.sample_rate,
        }))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        if data.get("error") or data.get("message_type") == "error":
            raise RuntimeError(f"ElevenLabs STT: {data.get('error') or data}")
        kind = data.get("message_type")
        if kind in ("partial_transcript", "committed_transcript"):
            final = kind == "committed_transcript"
            return [TranscriptEvent(data.get("text", ""), is_final=final)], final
        return [], False

