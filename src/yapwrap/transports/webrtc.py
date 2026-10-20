"""WebRTC audio transport with PCM normalization and paced output."""

import asyncio
import json
from dataclasses import asdict
from fractions import Fraction
from typing import Any, AsyncIterator

from yapwrap.core.base import BaseTransport
from yapwrap.core.audio import DEFAULT_SAMPLE_RATE
from yapwrap.core.events import SessionEvent


class WebRTCTransport(BaseTransport):
    """Adapt aiortc audio to PCM16 mono; the application owns peer signaling."""

    def __init__(
        self, pc: Any = None, track: Any = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE, data_channel: Any = None,
    ) -> None:
        self.pc = pc
        self.track = track
        self.sample_rate = sample_rate
        self.data_channel = data_channel
        self._is_open = True
        self._buffer = bytearray()
        self._condition = asyncio.Condition()
        self._max_buffer = sample_rate * 2  # One second of PCM16 backpressure.
        self._output_track: Any = None

    async def audio_in(self) -> AsyncIterator[bytes]:
        from aiortc.mediastreams import MediaStreamError
        from av import AudioResampler

        if self.track is None and self.pc is not None:
            self.track = next(
                (r.track for r in self.pc.getReceivers() if r.track and r.track.kind == "audio"),
                None,
            )
        if self.track is None:
            raise ValueError("Attach an incoming audio track before running WebRTCTransport")
        resampler = AudioResampler(format="s16", layout="mono", rate=self.sample_rate)
        try:
            while self._is_open:
                try:
                    frame = await self.track.recv()
                except MediaStreamError:
                    break
                # Browser Opus normally arrives as 48 kHz stereo, not pipeline PCM.
                for normalized in resampler.resample(frame):
                    yield normalized.to_ndarray().astype("<i2", copy=False).tobytes()
        finally:
            await self.close()

    async def send_audio(self, pcm: bytes) -> None:
        offset = 0
        async with self._condition:
            while self._is_open and offset < len(pcm):
                await self._condition.wait_for(
                    lambda: not self._is_open or len(self._buffer) < self._max_buffer
                )
                if not self._is_open:
                    return
                size = min(len(pcm) - offset, self._max_buffer - len(self._buffer))
                self._buffer.extend(pcm[offset:offset + size])
                offset += size

    async def send_event(self, event: SessionEvent) -> None:
        channel = self.data_channel
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(asdict(event)))

    async def stop_playback(self) -> None:
        async with self._condition:
            self._buffer.clear()
            self._condition.notify_all()

    async def close(self) -> None:
        self._is_open = False
        await self.stop_playback()
        if self._output_track is not None:
            self._output_track.stop()

    def create_output_track(self) -> Any:
        """Create a single output track; add it before negotiating the answer."""
        if self._output_track is not None:
            return self._output_track
        try:
            from aiortc import MediaStreamTrack
            from aiortc.mediastreams import MediaStreamError
            from av import AudioFrame
        except ImportError as exc:
            raise ImportError("WebRTCTransport requires pip install 'yapwrap[webrtc]'") from exc

        transport = self

        class OutputTrack(MediaStreamTrack):
            kind = "audio"

            def __init__(self):
                super().__init__()
                self.pts = 0
                self.deadline = None

            async def recv(self):
                if self.readyState != "live" or not transport._is_open:
                    raise MediaStreamError
                loop = asyncio.get_running_loop()
                if self.deadline is None:
                    self.deadline = loop.time()
                await asyncio.sleep(max(0, self.deadline - loop.time()))
                if self.readyState != "live" or not transport._is_open:
                    raise MediaStreamError
                samples = transport.sample_rate // 50
                self.deadline = max(self.deadline + samples / transport.sample_rate, loop.time())
                async with transport._condition:
                    size = min(samples * 2, len(transport._buffer) // 2 * 2)
                    data = bytes(transport._buffer[:size])
                    del transport._buffer[:size]
                    transport._condition.notify_all()
                # Send silence between replies so RTP timing remains continuous.
                frame = AudioFrame(format="s16", layout="mono", samples=samples)
                frame.planes[0].update(data.ljust(samples * 2, b"\x00"))
                frame.sample_rate = transport.sample_rate
                frame.time_base = Fraction(1, transport.sample_rate)
                frame.pts = self.pts
                self.pts += samples
                return frame

        self._output_track = OutputTrack()
        return self._output_track

