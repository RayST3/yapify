"""Internal resource lifetime shared by speech recognition and synthesis."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any


class ManagedConnection(ABC):
    """Own one provider resource per call, with serialized access and cleanup."""

    def __init__(self, *, sample_rate: int = 16000) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self._connection: Any = None
        self._session_active = False
        self._lock = asyncio.Lock()

    @abstractmethod
    async def _open_connection(self) -> Any:
        """Return an opened client or connection. Clean up if setup fails."""
        raise NotImplementedError

    async def _ensure_open(self) -> None:
        if self._connection is None:
            self._connection = await self._open_connection()

    async def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            close = getattr(connection, "aclose", None) or connection.close
            result = close()
            if inspect.isawaitable(result):
                await result

    @asynccontextmanager
    async def session(self) -> AsyncGenerator["ManagedConnection", None]:
        if self._session_active:
            raise RuntimeError(f"Create a separate {type(self).__name__} for each concurrent voice call")
        self._session_active = True
        try:
            async with self._lock:
                await self._ensure_open()
            yield self
        finally:
            try:
                async with self._lock:
                    await self._close_connection()
            finally:
                self._session_active = False

    @asynccontextmanager
    async def _client(self) -> AsyncGenerator[Any, None]:
        """Serialize a request; standalone requests also close their resources."""
        async with self._lock:
            try:
                await self._ensure_open()
                yield self._connection
            finally:
                if not self._session_active:
                    await self._close_connection()
