"""AssemblyAI Streaming v3 transcription."""

import json
import os
from typing import Optional
from urllib.parse import urlencode

from yapwrap.core.events import TranscriptEvent
from yapwrap.core.stt import WebSocketSTT


class AssemblyAISTT(WebSocketSTT):
    """Read all turns through Termination, including finals produced at EOF."""

    audio_chunk_ms = 50

    def __init__(
        self, api_key: Optional[str] = None, sample_rate: int = 16000,
        word_boost: Optional[list] = None, *, model: str = "universal-streaming-english",
        connect_timeout: float = 15.0, response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, connect_timeout=connect_timeout,
                         response_timeout=response_timeout)
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        self.word_boost, self.model = word_boost or [], model
        self._finalized_turns: set[int] = set()

    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        if not self.api_key:
            raise ValueError("AssemblyAISTT requires api_key or ASSEMBLYAI_API_KEY.")
        params = {"sample_rate": self.sample_rate, "speech_model": self.model, "format_turns": "false"}
        if self.word_boost:
            params["keyterms_prompt"] = json.dumps(self.word_boost)
        return f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}", {"Authorization": self.api_key}

    async def _configure_connection(self, connection) -> None:
        data = json.loads(await connection.recv())
        if data.get("type") != "Begin":
            raise RuntimeError(f"AssemblyAI STT initialization failed: {data}")

    def _reset_utterance(self) -> None:
        self._finalized_turns.clear()

    async def _finish_audio(self) -> None:
        await self._connection.send(json.dumps({"type": "Terminate"}))

    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        data = json.loads(message)
        if data.get("type") == "Error" or data.get("error"):
            raise RuntimeError(f"AssemblyAI STT: {data.get('error') or data}")
        if data.get("type") == "Turn":
            turn = data["turn_order"]
            if turn not in self._finalized_turns:
                final = bool(data.get("end_of_turn"))
                if final:
                    self._finalized_turns.add(turn)
                return [TranscriptEvent(data.get("transcript", ""), is_final=final)], False
        return [], data.get("type") == "Termination"

