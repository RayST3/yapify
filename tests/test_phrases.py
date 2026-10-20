"""Test phrase boundary chunker for streaming TTS."""

import asyncio
import pytest
from yapwrap.core.phrases import find_phrase_boundary, stream_phrases


def test_find_phrase_boundary():
    # Punctuation boundary
    text = "Hello world! How are you?"
    cut = find_phrase_boundary(text, max_chars=150)
    assert cut == 13  # after "Hello world! "
    assert text[:cut].strip() == "Hello world!"

    # Chinese/Japanese full-width punctuation
    cjk_text = "你好世界。你好吗？"
    cut = find_phrase_boundary(cjk_text, max_chars=150)
    assert cut > 0


@pytest.mark.asyncio
async def test_stream_phrases():
    tokens = ["Hello", " there!", " This", " is", " a", " test.", " Are", " you", " ready?"]

    async def token_generator():
        for tok in tokens:
            yield tok

    phrases = []
    async for phrase in stream_phrases(token_generator()):
        phrases.append(phrase)

    assert len(phrases) >= 3
    assert phrases[0] == "Hello there!"
    assert phrases[1] == "This is a test."
    assert phrases[2] == "Are you ready?"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [["   "], ["Hello.", "  \n"]])
async def test_trailing_whitespace_terminates(tokens):
    async def source():
        for token in tokens:
            yield token

    async def consume():
        return [phrase async for phrase in stream_phrases(source())]

    phrases = await asyncio.wait_for(consume(), 1)
    assert phrases == ([] if tokens == ["   "] else ["Hello."])


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [" ", "Checking.  ", "Checking.\n\n\n"])
async def test_whitespace_during_tool_pause_waits_without_busy_polling(monkeypatch, prefix):
    import yapwrap.core.phrases as module

    checks = 0
    boundary = module.find_phrase_boundary

    def count_checks(*args, **kwargs):
        nonlocal checks
        checks += 1
        return boundary(*args, **kwargs)

    monkeypatch.setattr(module, "find_phrase_boundary", count_checks)

    async def source():
        yield prefix
        await asyncio.sleep(0.05)
        yield "Found it."

    result = [phrase async for phrase in stream_phrases(source(), max_delay_ms=5)]
    assert result == (["Found it."] if not prefix.strip() else ["Checking.", "Found it."])
    # An idle stream should sleep, even after the phrase deadline passes.
    # The regression did thousands of boundary checks during this pause.
    assert checks < 20

