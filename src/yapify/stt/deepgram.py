"""Deepgram Nova-3 & Flux live streaming transcription."""

import asyncio
import json
import logging
import os
from typing import AsyncIterator, List, Optional
from urllib.parse import urlencode

from yapify.core.events import TranscriptEvent

logger = logging.getLogger("yapify.stt.deepgram")


class DeepgramSTT:
    """Stream audio to Deepgram Nova-3 / Flux over WebSocket."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-3",
        language: str = "en",
        smart_format: bool = True,
        sample_rate: int = 16000,
        endpointing: Optional[int] = None,
        keyterms: Optional[List[str]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.sample_rate = sample_rate
        self.endpointing = endpointing
        self.keyterms = keyterms

    def _build_url(self) -> str:
        params = [
            ("model", self.model),
            ("language", self.language),
            ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)),
            ("channels", "1"),
            ("interim_results", "true"),
            ("smart_format", "true" if self.smart_format else "false"),
        ]
        if self.endpointing is not None:
            params.append(("endpointing", str(self.endpointing)))
        if self.keyterms:
            for term in self.keyterms:
                params.append(("keyterm", term))
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        """Yield interim and committed segments until CloseStream is finalized."""
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError("DeepgramSTT requires `pip install 'yapify[deepgram]'`.") from exc

        if not self.api_key:
            raise ValueError("DeepgramSTT requires an api_key or DEEPGRAM_API_KEY environment variable.")

        url = self._build_url()
        headers = {"Authorization": f"Token {self.api_key}"}

        async with connect(url, additional_headers=headers) as ws:
            async def send_audio() -> None:
                try:
                    async for chunk in audio:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    # Wake the receiver, then propagate the sender error below.
                    await ws.close()
                    raise

            sender = asyncio.create_task(send_audio())
            try:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "Error":
                        raise RuntimeError(f"Deepgram STT: {data.get('description', 'provider error')}")
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        if transcript:
                            yield TranscriptEvent(text=transcript, is_final=data.get("is_final", False))
                if sender.done():
                    await sender
                else:
                    raise RuntimeError("Deepgram closed before input audio was finalized")
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

