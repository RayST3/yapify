"""Soniox realtime tokens with explicit finalization and connection keepalive."""

import json
import os
from typing import List, Optional

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class SonioxSTT(WebSocketSTT):
    """Preserve token spacing and drain through <fin> before starting a new turn."""

    reuse_connection = True
    keepalive_interval = 5.0
    keepalive_message = '{"type":"keepalive"}'

    def __init__(
        self, api_key: Optional[str] = None, model: str = "stt-rt-v5",
        language_hints: Optional[List[str]] = None, sample_rate: int = 16000,
        base_url: str = "wss://stt-rt.soniox.com/transcribe-websocket", *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("SONIOX_API_KEY", "")
        self.model, self.language_hints, self.base_url = model, language_hints, base_url
        self._final_text = ""

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("SonioxSTT requires api_key or SONIOX_API_KEY.")
        return self.base_url, {}

    async def _configure_connection(self, connection) -> None:
        config = {
            "api_key": self.api_key, "model": self.model, "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate, "num_channels": 1, "enable_endpoint_detection": False,
        }
        if self.language_hints:
            config["language_hints"] = self.language_hints
        await connection.send(json.dumps(config))

    def _reset_utterance(self) -> None:
        self._final_text = ""

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({"type": "finalize"}))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        if data.get("error_code") or data.get("error_message"):
            raise RuntimeError(f"Soniox STT: {data.get('error_message') or data}")
        if data.get("finished"):
            raise RuntimeError("Soniox STT ended before manual finalization")
        partial, finished = "", False
        tokens = data.get("tokens", [])
        for token in tokens:
            word = token.get("text", "")
            if word == "<fin>" and token.get("is_final"):
                finished = True
            elif word not in ("<end>", "<fin>"):
                if token.get("is_final"):
                    self._final_text += word
                else:
                    partial += word
        if tokens:
            # Confirmed subwords cannot be emitted as separate final segments:
            # the session joins segments with spaces, corrupting e.g. "hel"+"lo".
            return [TranscriptEvent(self._final_text + partial, is_final=finished)], finished
        return [], False

