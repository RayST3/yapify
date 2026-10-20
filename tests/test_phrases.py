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

