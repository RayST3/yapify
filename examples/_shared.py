"""Shared voice settings; framework SDK imports stay in their own examples."""

import asyncio
import os

from yapify import YapWrap
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD

INSTRUCTIONS = "You are a friendly voice assistant. Give natural spoken answers in at most two sentences. Avoid markdown."


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")


def gemini_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return key


def missing_keys(*, gemini: bool = False) -> list[str]:
    missing = [name for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY") if not os.getenv(name)]
    if gemini and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    return missing


def make_yapper(agent) -> YapWrap:
    missing = missing_keys()
    if missing:
        raise ValueError("Set environment variables: " + ", ".join(missing))
    return YapWrap(agent=agent, stt=DeepgramSTT(), tts=CartesiaTTS(), vad=SileroVAD())


def run(main) -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended.")
