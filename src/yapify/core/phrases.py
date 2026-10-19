"""Phrase boundary chunker for streaming token-to-speech synthesis."""

import asyncio
import re
from typing import AsyncIterator, Optional


def find_phrase_boundary(text: str, max_chars: int = 150, timed_out: bool = False) -> int:
    """Find the character index to split text at a natural spoken boundary."""
    # Match sentence endings including multilingual punctuation
    match = re.search(r"[.!?。！？।;\n](?:\s|$)", text)
    if match and match.end() <= max_chars:
        return match.end()

    # If too long, split at the last space before max_chars
    if len(text) >= max_chars:
        split = text.rfind(" ", 0, max_chars + 1)
        return split + 1 if split > 0 else max_chars

    # If timed out waiting for more tokens, split at last space
    if timed_out and text.strip():
        split = text.rfind(" ")
        return split + 1 if split > 0 else len(text)

    return 0


async def stream_phrases(
    text_stream: AsyncIterator[str],
    max_chars: int = 150,
    max_delay_ms: float = 120.0,
) -> AsyncIterator[str]:
    """Yield complete phrases from an asynchronous token stream."""
    if max_chars <= 0 or max_delay_ms <= 0:
        raise ValueError("max_chars and max_delay_ms must be positive")
    iterator = text_stream.__aiter__()
    pending: Optional[asyncio.Future] = None
    buffer = ""
    deadline: Optional[float] = None
    exhausted = False
    loop = asyncio.get_running_loop()
    max_delay = max_delay_ms / 1000.0

    try:
        while not exhausted or buffer:
            if exhausted and not buffer.strip():
                break
            cut = find_phrase_boundary(buffer, max_chars=max_chars)
            if cut or (exhausted and buffer.strip()):
                cut = cut or len(buffer)
                phrase = buffer[:cut].strip()
                buffer = buffer[cut:]
                deadline = loop.time() + max_delay if buffer else None
                if phrase:
                    yield phrase
                continue

            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())

            timeout = max(0.0, deadline - loop.time()) if deadline is not None else None
            done, _ = await asyncio.wait({pending}, timeout=timeout)

            if not done:
                # Timeout waiting for tokens; try to flush partial phrase
                cut = find_phrase_boundary(buffer, max_chars=max_chars, timed_out=True)
                if cut:
                    phrase = buffer[:cut].strip()
                    buffer = buffer[cut:]
                    deadline = loop.time() + max_delay if buffer else None
                    if phrase:
                        yield phrase
                continue

            try:
                token = pending.result()
            except StopAsyncIteration:
                exhausted = True
            else:
                buffer += token
                if buffer and deadline is None:
                    deadline = loop.time() + max_delay
            pending = None

    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

