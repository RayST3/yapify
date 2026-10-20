"""Shared STT resource lifetime and duplex WebSocket recognition."""

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, Optional

from yapwrap.core.audio import PCM16Stream
from yapwrap.core.base import BaseSTT
from yapwrap.core.connections import ManagedConnection
from yapwrap.core.events import TranscriptEvent


class ManagedSTT(ManagedConnection, BaseSTT):
    """Serialize utterances, normalize PCM, and reuse a client within session()."""

    reuse_connection = True
    input_sample_rate: Optional[int] = None
    audio_chunk_ms = 0
    min_audio_ms = 0

    def _reset_utterance(self) -> None:
        """Reset adapter transcript state before consuming this utterance."""

    async def _audio_chunks(self, audio: AsyncIterator[bytes]) -> AsyncGenerator[bytes, None]:
        iterator = audio.__aiter__()
        rate = self.input_sample_rate or self.sample_rate
        converter = PCM16Stream(self.sample_rate, rate)
        block = rate * self.audio_chunk_ms // 1000 * 2
        pending = bytearray()
        total = 0
        async for chunk in iterator:
            pcm = converter.process(chunk)
            total += len(pcm)
            pending.extend(pcm)
            while pending and (not block or len(pending) >= block):
                size = block or min(len(pending), 20000)
                yield bytes(pending[:size])
                del pending[:size]
        tail = converter.finish()
        total += len(tail)
        pending.extend(tail)
        if total:
            pending.extend(b"\x00" * max(0, rate * self.min_audio_ms // 1000 * 2 - total))
        if block and pending:
            pending.extend(b"\x00" * (-len(pending) % block))
        while pending:
            size = block or min(len(pending), 20000)
            yield bytes(pending[:size])
            del pending[:size]

    async def transcribe(self, audio: AsyncIterator[bytes]) -> AsyncGenerator[TranscriptEvent, None]:
        iterator = audio.__aiter__()
        try:
            async with aclosing(self._audio_chunks(iterator)) as chunks:
                async with self._lock:
                    completed = False
                    try:
                        first = await anext(chunks, None)
                        if first is None:
                            completed = True
                            return
                        await self._ensure_open()
                        self._reset_utterance()

                        async def remaining() -> AsyncGenerator[bytes, None]:
                            yield first
                            async for chunk in chunks:
                                yield chunk

                        async with aclosing(remaining()) as source:
                            async with aclosing(self._transcribe_stream(source)) as stream:
                                async for event in stream:
                                    yield event
                        completed = True
                    finally:
                        if not completed or not self._session_active or not self.reuse_connection:
                            await self._close_connection()
        finally:
            # Own the upstream iterator even if cancelled while waiting for the lock.
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    @abstractmethod
    def _transcribe_stream(self, audio: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        """Consume all audio, flush, and drain every committed segment."""
        raise NotImplementedError


class WebSocketSTT(ManagedSTT):
    """Concurrent audio upload/transcript reception with explicit end-of-input."""

    reuse_connection = False
    completion_on_close = False
    keepalive_interval = 0.0
    keepalive_message: Optional[str] = None

    def __init__(
        self, *, sample_rate: int = 16000, connect_timeout: float = 15.0,
        response_timeout: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        if connect_timeout <= 0 or response_timeout <= 0:
            raise ValueError("Provider timeouts must be positive")
        self.connect_timeout, self.response_timeout = connect_timeout, response_timeout
        self._keepalive_task: Optional[asyncio.Task] = None
        self._input_ended = False

    @abstractmethod
    async def _connection_settings(self) -> tuple[str, dict[str, str]]:
        """Return a URL and headers; HTTP session initialization can happen here."""
        raise NotImplementedError

    async def _open_connection(self) -> Any:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError(f"{type(self).__name__} requires pip install 'yapwrap[websockets]'.") from exc
        url, headers = await self._connection_settings()
        connection = await connect(
            url, additional_headers=headers, open_timeout=self.connect_timeout,
            close_timeout=2, max_size=2**22, max_queue=16,
        )
        try:
            await asyncio.wait_for(self._configure_connection(connection), self.connect_timeout)
        except BaseException:
            await connection.close()
            raise
        if self.keepalive_interval and self.keepalive_message is not None:
            self._keepalive_task = asyncio.create_task(self._keepalive(connection))
        return connection

    async def _configure_connection(self, connection: Any) -> None:
        """Optionally configure the socket and wait for a provider ready message."""

    async def _keepalive(self, connection: Any) -> None:
        while True:
            await asyncio.sleep(self.keepalive_interval)
            await asyncio.wait_for(connection.send(self.keepalive_message), self.response_timeout)

    async def _close_connection(self) -> None:
        task, self._keepalive_task = self._keepalive_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await super()._close_connection()

    async def _ensure_open(self) -> None:
        if self._connection is not None:
            from websockets.protocol import State
            if self._connection.state is State.OPEN:
                if self._keepalive_task is not None and self._keepalive_task.done():
                    self._keepalive_task.result()
                return
            await self._close_connection()
        await super()._ensure_open()

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        await self._connection.send(chunk)

    @abstractmethod
    async def _finish_audio(self) -> None:
        """Commit/flush this utterance only after all its audio has been sent."""
        raise NotImplementedError

    @abstractmethod
    def _decode_message(self, message: str | bytes) -> tuple[list[TranscriptEvent], bool]:
        """Return transcript events and whether the entire input is finalized."""
        raise NotImplementedError

    async def _transcribe_stream(self, audio: AsyncIterator[bytes]) -> AsyncGenerator[TranscriptEvent, None]:
        from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

        self._input_ended = False

        async def send_audio() -> None:
            async for chunk in audio:
                await asyncio.wait_for(self._send_audio_chunk(chunk), self.response_timeout)
            self._input_ended = True
            await asyncio.wait_for(self._finish_audio(), self.response_timeout)

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(self._connection.recv())
        sending = True
        deadline = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                waiting = {receiver}
                if sending:
                    waiting.add(sender)
                if self._keepalive_task is not None:
                    waiting.add(self._keepalive_task)
                done, _ = await asyncio.wait(
                    waiting, timeout=None if deadline is None else max(0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done or (deadline is not None and loop.time() >= deadline):
                    raise TimeoutError(f"{type(self).__name__} did not finalize within response_timeout")
                if self._keepalive_task in done:
                    self._keepalive_task.result()
                if sender in done:
                    sender.result()
                    sending = False
                    deadline = loop.time() + self.response_timeout
                if receiver not in done:
                    continue
                try:
                    message = receiver.result()
                except ConnectionClosed as exc:
                    if self.completion_on_close and isinstance(exc, ConnectionClosedOK) and self._input_ended:
                        await asyncio.wait_for(sender, self.response_timeout)
                        return
                    raise RuntimeError(f"{type(self).__name__} closed before transcription completed") from exc
                events, finished = self._decode_message(message)
                for event in events:
                    before = loop.time()
                    yield event
                    if deadline is not None:
                        deadline += loop.time() - before
                if finished:
                    if not self._input_ended:
                        raise RuntimeError(f"{type(self).__name__} completed before input audio ended")
                    await asyncio.wait_for(sender, self.response_timeout)
                    return
                receiver = asyncio.create_task(self._connection.recv())
        finally:
            for task in (sender, receiver):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
