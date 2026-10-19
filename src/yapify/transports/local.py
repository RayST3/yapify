"""Local microphone input and continuous, interruptible speaker playback."""

import asyncio
import logging
import threading
from typing import AsyncIterator, Optional

from yapify.core.audio import DEFAULT_SAMPLE_RATE
from yapify.core.events import SessionEvent

logger = logging.getLogger("yapify.transport.local")


class LocalTransport:
    """Use one sounddevice input stream and one output stream per session."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, frame_bytes: Optional[int] = None, *, show_transcripts: bool = True) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = frame_bytes if frame_bytes is not None else int(sample_rate * 0.02) * 2
        self._is_open = True
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._playback = bytearray()
        self._playback_lock = threading.Lock()
        self._space = asyncio.Event()
        self._max_playback_bytes = sample_rate * 2  # At most one second queued.
        self.show_transcripts = show_transcripts
        self._agent_turn: Optional[int] = None

    async def send_event(self, event: SessionEvent) -> None:
        """Print final user transcripts and stream the agent's generated text."""
        if not self.show_transcripts:
            return
        data = event.data
        turn_id = data.get("turn_id")
        # Keep model/transcript control characters from changing terminal state.
        text = "".join(char for char in data.get("text", "") if char.isprintable() or char in "\n\t")
        if event.type == "transcript" and data.get("is_final") and text:
            if self._agent_turn is not None:
                print(flush=True)
                self._agent_turn = None
            print(f"You: {text}", flush=True)
        elif event.type == "agent_delta" and text:
            if self._agent_turn != turn_id:
                if self._agent_turn is not None:
                    print(flush=True)
                print("Agent: ", end="", flush=True)
                self._agent_turn = turn_id
            print(text, end="", flush=True)
        elif event.type == "agent_complete":
            if self._agent_turn != turn_id and not text:
                return
            if self._agent_turn != turn_id:
                print(f"Agent: {text}", end="", flush=True)
            label = {"interrupted": " [interrupted]", "error": " [error]"}.get(data.get("status"), "")
            print(label, flush=True)
            self._agent_turn = None

    async def audio_in(self) -> AsyncIterator[bytes]:
        """Open devices and stream microphone frames until stopped."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise ImportError("LocalTransport requires `pip install 'yapify[local]'`.") from exc

        loop = asyncio.get_running_loop()

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.debug("Microphone status: %s", status)
            loop.call_soon_threadsafe(self._input_queue.put_nowait, bytes(indata))

        def speaker_callback(outdata, frames, time_info, status):
            if status:
                logger.debug("Speaker status: %s", status)
            outdata[:] = self._read_playback(frames * 2)
            loop.call_soon_threadsafe(self._space.set)

        options = dict(samplerate=self.sample_rate, blocksize=self.frame_bytes // 2, dtype="int16", channels=1)
        try:
            with sd.RawInputStream(callback=mic_callback, **options), sd.RawOutputStream(callback=speaker_callback, **options):
                print("Session online. Start speaking (Ctrl+C to stop).", flush=True)
                while self._is_open:
                    chunk = await self._input_queue.get()
                    if chunk is None:
                        break
                    yield chunk
        finally:
            self.stop()

    def _read_playback(self, size: int) -> bytes:
        with self._playback_lock:
            data = bytes(self._playback[:size])
            del self._playback[:size]
        return data.ljust(size, b"\x00")

    async def send_audio(self, pcm: bytes) -> None:
        """Queue chunks in order, applying backpressure when playback is behind."""
        offset = 0
        while self._is_open and offset < len(pcm):
            self._space.clear()
            with self._playback_lock:
                count = min(len(pcm) - offset, self._max_playback_bytes - len(self._playback))
                self._playback.extend(pcm[offset:offset + count])
            offset += count
            if offset < len(pcm):
                await self._space.wait()

    async def stop_playback(self) -> None:
        """Clear queued PCM; the next device callback outputs silence."""
        with self._playback_lock:
            self._playback.clear()
        self._space.set()

    def stop(self) -> None:
        """Wake input/output waiters and close the device contexts."""
        self._is_open = False
        self._input_queue.put_nowait(None)
        with self._playback_lock:
            self._playback.clear()
        self._space.set()
