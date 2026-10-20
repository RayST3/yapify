"""Reusable TTS lifetimes and streaming engines; adapters implement wire formats.

ManagedTTS owns a client for a call. RealtimeTTS accepts incremental text and
ends synthesis contexts across tool pauses. WebSocketTTS adds duplex I/O, PCM
framing, and cancellation for providers using a WebSocket protocol.
"""

import asyncio
import re
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, suppress
from typing import Any, Optional
from uuid import uuid4

from yapwrap.core.base import BaseTTS
from yapwrap.core.connections import ManagedConnection


class ManagedTTS(ManagedConnection, BaseTTS):
    """Reuse provider resources across phrases and replies, with exclusive access."""

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        # The phrase fallback must reuse its client even outside YapSesh.
        try:
            if not self._session_active:
                async with self.session(), aclosing(super().synthesize_stream(text)) as audio:
                    async for chunk in audio:
                        yield chunk
            else:
                async with aclosing(super().synthesize_stream(text)) as audio:
                    async for chunk in audio:
                        yield chunk
        finally:
            close = getattr(text, "aclose", None)
            if close is not None:
                await close()


class RealtimeTTS(ManagedTTS):
    """Stream text directly, flushing each context when input pauses or ends."""

    max_context_chars: int = 0

    def __init__(self, *, sample_rate: int = 16000, context_idle_timeout: float = 0.12) -> None:
        super().__init__(sample_rate=sample_rate)
        if context_idle_timeout <= 0:
            raise ValueError("context_idle_timeout must be positive")
        self.context_idle_timeout = context_idle_timeout

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return

        async def tokens() -> AsyncGenerator[str, None]:
            yield text

        async with aclosing(tokens()) as source, aclosing(self.synthesize_stream(source)) as audio:
            async for chunk in audio:
                yield chunk

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        tokens = TextContexts(text, self.context_idle_timeout, self.max_context_chars)
        try:
            async with self._lock:
                try:
                    while True:
                        first = await tokens.next()
                        if first is None:
                            return
                        if not first.strip():
                            continue
                        await self._ensure_open()
                        async with aclosing(tokens.segment(first)) as segment:
                            async with aclosing(self._synthesize_context(segment)) as audio:
                                async for chunk in audio:
                                    yield chunk
                finally:
                    if not self._session_active:
                        await self._close_connection()
        finally:
            await tokens.close()

    @abstractmethod
    def _synthesize_context(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Consume one continuous stretch of text, flush, and yield all its audio."""
        raise NotImplementedError


class WebSocketTTS(RealtimeTTS):
    """Shared WebSocket engine. Provider hooks only encode/decode their protocol."""

    completion_on_close = False

    def __init__(
        self, *, sample_rate: int = 16000, context_idle_timeout: float = 0.12,
        connect_timeout: float = 15.0, response_timeout: float = 30.0,
    ) -> None:
        super().__init__(sample_rate=sample_rate, context_idle_timeout=context_idle_timeout)
        if connect_timeout <= 0 or response_timeout <= 0:
            raise ValueError("Provider timeouts must be positive")
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout

    @abstractmethod
    def _connection_settings(self) -> tuple[str, dict[str, str]]:
        """Validate configuration and return the URL and authentication headers."""
        raise NotImplementedError

    async def _open_connection(self) -> Any:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ImportError(f"{type(self).__name__} requires `pip install 'yapwrap[websockets]'`.") from exc
        url, headers = self._connection_settings()
        connection = await connect(
            url, additional_headers=headers, open_timeout=self.connect_timeout,
            close_timeout=2, max_size=2**22, max_queue=16,
        )
        try:
            await self._configure_connection(connection)
        except BaseException:
            await connection.close()
            raise
        return connection

    async def _configure_connection(self, connection: Any) -> None:
        """Optionally send configuration once, immediately after connecting."""

    async def _ensure_open(self) -> None:
        if self._connection is not None:
            from websockets.protocol import State
            if self._connection.state is State.OPEN:
                return
            await self._close_connection()
        await super()._ensure_open()

    @abstractmethod
    async def _send_text(self, text: AsyncIterator[str], context_id: str) -> None:
        """Send incoming deltas followed by the provider's end/flush message."""
        raise NotImplementedError

    @abstractmethod
    def _decode_message(self, message: str | bytes, context_id: str) -> tuple[bytes, bool]:
        """Return raw PCM16 and completion; filter other contexts and raise errors."""
        raise NotImplementedError

    async def _cancel_context(self, context_id: str) -> None:
        # Without correlated contexts, abandoning the socket prevents late audio
        # from the cancelled request contaminating the next reply.
        await self._close_connection()

    async def _synthesize_context(self, text: AsyncIterator[str]) -> AsyncGenerator[bytes, None]:
        from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

        connection = self._connection
        context_id = str(uuid4())
        sender = asyncio.create_task(self._send_text(text, context_id))
        receiver = asyncio.create_task(connection.recv())
        sending, completed = True, False
        pending_byte = b""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.response_timeout
        try:
            while True:
                waiting = {sender, receiver} if sending else {receiver}
                done, _ = await asyncio.wait(
                    waiting, timeout=max(0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done or loop.time() >= deadline:
                    raise TimeoutError(f"{type(self).__name__} did not produce audio within response_timeout")
                if sender in done:
                    sending = False
                    sender.result()
                if receiver not in done:
                    continue
                pcm, finished = self._decode_message(receiver.result(), context_id)
                if pcm:
                    deadline = loop.time() + self.response_timeout
                    pcm = pending_byte + pcm
                    whole = len(pcm) // 2 * 2
                    pending_byte = pcm[whole:]
                    if whole:
                        yield pcm[:whole]
                        # Downstream playback backpressure isn't a provider timeout.
                        deadline = loop.time() + self.response_timeout
                if finished:
                    await asyncio.wait_for(sender, timeout=self.response_timeout)
                    if pending_byte:
                        raise RuntimeError(f"{type(self).__name__} returned an incomplete PCM16 sample")
                    completed = True
                    return
                receiver = asyncio.create_task(connection.recv())
        except ConnectionClosed as exc:
            if self.completion_on_close and isinstance(exc, ConnectionClosedOK) and sender.done():
                sender.result()
                if pending_byte:
                    raise RuntimeError(f"{type(self).__name__} returned an incomplete PCM16 sample") from exc
                completed = True
                return
            raise RuntimeError(f"{type(self).__name__} closed before synthesis completed") from exc
        finally:
            for task in (sender, receiver):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
            if not completed:
                try:
                    await asyncio.wait_for(self._cancel_context(context_id), timeout=2)
                except Exception:
                    with suppress(Exception):
                        await self._close_connection()


class TextContexts:
    """Keep an outstanding token read alive when a pause ends a speech context."""

    def __init__(self, text: AsyncIterator[str], idle_timeout: float, max_chars: int = 0) -> None:
        self._text = text.__aiter__()
        self._idle_timeout = idle_timeout
        self._max_chars = max_chars
        self._pending: Optional[asyncio.Future] = None
        self._exhausted = False

    async def next(self, timeout: Optional[float] = None) -> Optional[str]:
        while not self._exhausted:
            if self._pending is None:
                self._pending = asyncio.ensure_future(self._text.__anext__())
            done, _ = await asyncio.wait({self._pending}, timeout=timeout)
            if not done:
                raise asyncio.TimeoutError
            try:
                delta = self._pending.result()
            except StopAsyncIteration:
                self._exhausted = True
                return None
            finally:
                self._pending = None
            if delta:
                return delta
        return None

    async def segment(self, first: str) -> AsyncGenerator[str, None]:
        yield first
        size, last = len(first), first
        while True:
            if self._max_chars and (size >= self._max_chars or re.search(r"[.!?。！？।;\n](?:\s|$)", last)):
                return
            try:
                delta = await self.next(timeout=self._idle_timeout)
            except asyncio.TimeoutError:
                return
            if delta is None:
                return
            yield delta
            size += len(delta)
            last = delta

    async def close(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            await asyncio.gather(self._pending, return_exceptions=True)
        close = getattr(self._text, "aclose", None)
        if close is not None:
            await close()
