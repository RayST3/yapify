"""ElevenLabs streaming text-to-speech."""

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator, Optional

from yapwrap.core.base import BaseTTS

logger = logging.getLogger("yapwrap.tts.elevenlabs")


class ElevenLabsTTS(BaseTTS):
    """Stream speech from ElevenLabs Turbo / Multilingual v2 over WebSocket."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id: str = "eleven_turbo_v2_5",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text into raw PCM16 audio chunks."""
        if not text.strip():
            return

        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("ElevenLabsTTS requires `pip install 'yapwrap[elevenlabs]'`.") from exc

        if not self.api_key:
            raise ValueError("ElevenLabsTTS requires api_key or ELEVENLABS_API_KEY env var.")

        output_format = f"pcm_{self.sample_rate}"
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}&output_format={output_format}"
        headers = {"xi-api-key": self.api_key}

        async with connect(url, additional_headers=headers) as ws:
            # 1. Send BOS (beginning of stream) message with voice settings
            bos = {
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
            }
            await ws.send(json.dumps(bos))

            # 2. Send actual text
            await ws.send(json.dumps({"text": text}))

            # 3. Send EOS (end of stream)
            await ws.send(json.dumps({"text": ""}))

            async for message in ws:
                data = json.loads(message)
                audio_b64 = data.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
                if data.get("isFinal"):
                    break

