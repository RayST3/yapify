"""OpenAI transcription-only Realtime sessions with manual input commits."""

import base64
import json
import os
from typing import Optional

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class OpenAIRealtimeSTT(WebSocketSTT):
    """Resample session PCM to 24 kHz and reuse a transcription connection."""

    reuse_connection = True
    input_sample_rate = 24000
    min_audio_ms = 100

    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-live-transcribe",
        language: Optional[str] = "en", sample_rate: int = 16000, *,
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model, self.language = model, language
        self._text = ""
        self._item_id: Optional[str] = None

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("OpenAIRealtimeSTT requires api_key or OPENAI_API_KEY.")
        return "wss://api.openai.com/v1/realtime?intent=transcription", {"Authorization": f"Bearer {self.api_key}"}

    async def _configure_connection(self, connection) -> None:
        transcription = {"model": self.model}
        if self.language:
            if self.model.startswith("gpt-live-transcribe"):
                transcription["languages"] = [self.language]
            else:
                transcription["language"] = self.language
        await connection.send(json.dumps({
            "type": "session.update",
            "session": {"type": "transcription", "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": transcription, "turn_detection": None,
            }}},
        }))
        while True:
            data = json.loads(await connection.recv())
            if data.get("type") == "error":
                raise RuntimeError(f"OpenAI STT: {data.get('error')}")
            if data.get("type") == "session.updated":
                return

    def _reset_utterance(self) -> None:
        self._text, self._item_id = "", None

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        await self._connection.send(json.dumps({
            "type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode(),
        }))

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        kind = data.get("type", "")
        if kind == "error" or kind == "conversation.item.input_audio_transcription.failed":
            raise RuntimeError(f"OpenAI STT: {data.get('error') or data}")
        if kind == "input_audio_buffer.committed":
            self._item_id = data["item_id"]
        if kind.startswith("conversation.item.input_audio_transcription."):
            item = data.get("item_id")
            if self._item_id is not None and item != self._item_id:
                return [], False
            if item is not None:
                self._item_id = item
            if kind.endswith(".delta"):
                self._text += data.get("delta", "")
                return [TranscriptEvent(self._text)], False
            if kind.endswith(".completed"):
                return [TranscriptEvent(data.get("transcript", ""), is_final=True)], True
        return [], False

